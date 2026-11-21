# -*- coding: utf-8 -*-
"""模型清单与切换（供 ``/model`` 用）。

## 为什么单独一个模块

``/model`` 要做三件事，且都要能单测（不打网络、不依赖真实引擎）：

1. **列**——候选从哪来；
2. **切**——运行时把引擎的模型换掉（下次调用立即生效，不用重启）；
3. **存**——写回 ``llm_config.json``，否则下次启动又变回去。

## 候选来源

优先问网关要（``GET {api_base}/models``，OpenAI 兼容接口都有；本机反代能返回
全部可用模型）；问不到就退回配置文件里能看到的模型名（``model`` /
``fallback_models`` / ``routing_rules`` / ``provider_models``）。
**在线清单失败不是错误**：离线/网关没起来时，配置里的名字仍然可用。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

#: 在线查询超时（秒）。网关没起来时不希望卡住输入框。
PROBE_TIMEOUT = 4.0


@dataclass(frozen=True)
class ModelInfo:
    """一个候选模型。"""

    name: str
    #: 来源：``在线`` / ``配置``
    source: str = "配置"
    #: 附加说明（上下文长度、是否当前）
    note: str = ""


def models_from_config_data(data: dict[str, Any]) -> list[str]:
    """从配置字典里挖出所有出现过的模型名（保持出现顺序、去重）。"""
    found: list[str] = []

    def _add(value: Any) -> None:
        name = str(value or "").strip()
        if name and name not in found:
            found.append(name)

    _add(data.get("model"))
    for name in data.get("fallback_models") or []:
        _add(name)
    routing = data.get("routing_rules")
    if isinstance(routing, dict):
        for names in routing.values():
            if isinstance(names, list):
                for name in names:
                    _add(name)
    provider_models = data.get("provider_models")
    if isinstance(provider_models, dict):
        for names in provider_models.values():
            if isinstance(names, list):
                for name in names:
                    _add(name)
    return found


def load_config_data(path: str | Path | None) -> dict[str, Any]:
    """读配置文件（读不到就返回空字典——调用方按"只有当前模型"处理）。"""
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    try:
        with open(target, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # pragma: no cover - 坏配置不该让 /model 崩
        logger.warning("读取模型配置失败: %s", exc)
        return {}


def probe_online_models(api_base: str, api_key: str) -> list[str]:
    """问网关要模型清单；拿不到就返回空表（不抛）。"""
    if not api_base:
        return []
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key or "none", base_url=api_base, timeout=PROBE_TIMEOUT, max_retries=0)
        page = client.models.list()
        names: list[str] = []
        for item in getattr(page, "data", []) or []:
            name = str(getattr(item, "id", "") or "").strip()
            if name and name not in names:
                names.append(name)
        return names
    except Exception as exc:
        logger.debug("在线模型清单不可用（改用配置里的名字）: %s", exc)
        return []


def available_models(
    *,
    config_data: dict[str, Any],
    api_base: str = "",
    api_key: str = "",
    probe: bool = True,
) -> list[ModelInfo]:
    """汇总候选模型：在线清单优先，配置里的名字兜底。"""
    current = str(config_data.get("model") or "").strip()
    online = probe_online_models(api_base, api_key) if probe else []
    configured = models_from_config_data(config_data)

    items: list[ModelInfo] = []
    seen: set[str] = set()
    for name in online:
        if name in seen:
            continue
        seen.add(name)
        items.append(ModelInfo(name=name, source="在线", note=_note(name, current)))
    for name in configured:
        if name in seen:
            continue
        seen.add(name)
        items.append(ModelInfo(name=name, source="配置", note=_note(name, current)))
    return items


def _note(name: str, current: str) -> str:
    return "当前使用" if name == current else ""


def apply_model(config: Any, name: str) -> None:
    """把运行时配置切到 ``name``（下次 LLM 调用立即生效）。

    ``EngineLLMConfig.models_for()`` 的顺序是"显式指定 → routing_rules →
    model → fallback"，**routing 档位排在 model 前面**。所以只改 ``model``
    是不够的：所有档位必须一起指过去，否则任务仍会走旧模型（这个坑实测过）。
    """
    target = (name or "").strip()
    if not target:
        return
    config.model = target
    rules = getattr(config, "routing_rules", None)
    if isinstance(rules, dict):
        for key, models in list(rules.items()):
            if isinstance(models, list) and models:
                rules[key] = [target]
    # 备用模型若与新主模型不同，保留它作为兜底；同名的去掉避免重复尝试
    fallbacks = [m for m in (getattr(config, "fallback_models", []) or []) if m != target]
    config.fallback_models = fallbacks


def persist_model(config_path: str | Path | None, name: str) -> tuple[bool, str]:
    """把模型写回配置文件。返回 ``(是否成功, 说明)``。"""
    if not config_path:
        return False, "没有配置文件路径"
    target = Path(config_path)
    if not target.exists():
        return False, f"配置文件不存在：{target}"
    data = load_config_data(target)
    if not data:
        return False, f"配置文件读不出内容：{target}"
    data["model"] = name
    routing = data.get("routing_rules")
    if isinstance(routing, dict):
        for key, models in list(routing.items()):
            if isinstance(models, list) and models:
                routing[key] = [name]
    if data.get("fallback_models"):
        data["fallback_models"] = [m for m in data["fallback_models"] if m != name]
    try:
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        return False, f"写入失败：{exc}"
    return True, f"已写入 {target.name}"


def resolve_config_path(workspace: str | Path, explicit: str | Path | None = None) -> Path | None:
    """定位实际生效的 llm_config.json（与引擎的解析顺序保持一致）。"""
    if explicit:
        return Path(explicit)
    from ..engine.llm_client import default_config_path

    return default_config_path(workspace)


def format_model_list(items: Sequence[ModelInfo], current: str) -> str:
    """给 ``/model`` 无参数时打印的清单。"""
    if not items:
        return "（没有可用的模型清单：配置里没写、网关也没返回）"
    lines: list[str] = []
    width = max(len(item.name) for item in items)
    for item in items:
        mark = "●" if item.name == current else " "
        note = f"  {item.note}" if item.note else ""
        lines.append(f"  {mark} {item.name:<{width}}  [{item.source}]{note}")
    return "\n".join(lines)


__all__ = [
    "ModelInfo",
    "apply_model",
    "available_models",
    "format_model_list",
    "load_config_data",
    "models_from_config_data",
    "persist_model",
    "probe_online_models",
    "resolve_config_path",
]

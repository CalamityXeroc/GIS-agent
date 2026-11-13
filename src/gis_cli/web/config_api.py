# -*- coding: utf-8 -*-
"""配置 API：读取（脱敏）+ 白名单写入（原子写 + 备份 + 乐观并发 + 运行中锁）。

安全边界（见 ``docs/p2-console-plan.md`` §6 / §9）：

- **读**：任何层级的 ``api_key`` 出站前都变成 ``sk-…后4位``；
- **写**：只允许改 ``routing_rules.*`` / ``fallback_models`` / ``engine`` 的 5 个
  预算标量 / ``temperature`` / ``max_tokens`` / ``provider`` / ``model`` /
  ``api_base`` / ``timeout``。改 ``model`` 时**联动**：patch 没显式给的角色的
  ``routing_rules`` 一并指到新模型（路由在引擎里优先于 ``model``，不联动就换不动）。
  ``api_key`` 只能走独立的 ``api_key`` 字段（留空 =
  不改，且日志只记“是否改动”，不记值）。``provider`` 是**元信息**：只用于
  ``provider_models`` 分组与前端展示，引擎不读它（真正生效的是 ``api_base`` +
  ``model`` + ``api_key``）；
- **测试连接**：``POST /api/config/test`` 现场探 ``GET {api_base}/models``，失败也
  返回 200 + ``ok:false``，错误串先抹掉 key / URL 查询串再出站；
- **并发**：``GET`` 返回文件原文的 ``sha256``，写回必须带着它；对不上就是 409；
- **运行中**：``JobManager`` 非空闲（排队/运行/等计划卡）时 409，配置只对下一个
  run 生效；
- **落盘**：先备份到 ``config/backups/llm_config-<ts>.json``，再写同目录临时文件
  并用 ``os.replace`` 原子替换；保留原文件的换行风格（Windows 上常见 CRLF）。
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .security import mask_secret

logger = logging.getLogger(__name__)

#: 配置文件名（相对 workspace）。
CONFIG_RELATIVE = Path("config") / "llm_config.json"

#: 必须与引擎 ``routing_rules`` 的 10 个角色一致（计划 §2.5 写的是 9，实际配置有 10 个）。
ROLE_NAMES: tuple[str, ...] = (
    "agent",
    "agent_strong",
    "planning",
    "planning_expert",
    "intent",
    "task_refine",
    "recovery",
    "recovery_expert",
    "summarize",
    "verify",
)

#: ``engine`` 段里允许页面修改的标量（最小值, 最大值）。其余键只读。
ENGINE_INT_LIMITS: dict[str, tuple[int, int]] = {
    "max_turns": (1, 10_000),
    "max_repairs": (0, 1_000),
    "context_soft_limit_tokens": (1_000, 10_000_000),
    "kernel_idle_timeout_seconds": (30, 604_800),
    "kernel_exec_timeout_seconds": (30, 604_800),
}

#: patch 顶层允许出现的键。
PATCH_KEYS = frozenset(
    {
        "routing_rules",
        "fallback_models",
        "engine",
        "temperature",
        "max_tokens",
        "provider",
        "model",
        "api_base",
        "timeout",
    }
)

#: 供应商预设（静态表，不联网、零外链）。``custom`` 给反代/自建网关用。
PROVIDER_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "id": "openai",
        "name": "OpenAI",
        "default_base": "https://api.openai.com/v1",
        "key_required": True,
        "hint": "官方地址；走反代请选 custom",
    },
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "default_base": "https://api.deepseek.com/v1",
        "key_required": True,
        "hint": "官方 OpenAI 兼容地址，以 /v1 结尾",
    },
    {
        "id": "anthropic",
        "name": "Anthropic Claude",
        "default_base": "https://api.anthropic.com",
        "key_required": True,
        "hint": "官方地址 https://api.anthropic.com（Messages API，引擎已适配）",
    },
    {
        "id": "dashscope",
        "name": "通义千问 DashScope",
        "default_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_required": True,
        "hint": "用「OpenAI 兼容」模式地址",
    },
    {
        "id": "moonshot",
        "name": "Moonshot Kimi",
        "default_base": "https://api.moonshot.cn/v1",
        "key_required": True,
        "hint": "模型名形如 kimi-k2-0905-preview",
    },
    {
        "id": "zhipu",
        "name": "智谱 GLM",
        "default_base": "https://open.bigmodel.cn/api/paas/v4",
        "key_required": True,
        "hint": "地址以 /v4 结尾（不是 /v1）",
    },
    {
        "id": "ollama",
        "name": "Ollama（本机）",
        "default_base": "http://127.0.0.1:11434/v1",
        "key_required": False,
        "hint": "先 ollama serve；本机模型不需要 key",
    },
    {
        "id": "custom",
        "name": "自定义 / 反代",
        "default_base": "",
        "key_required": False,
        "hint": "填反代地址，通常以 /v1 结尾",
    },
)

#: 合法 ``provider``：预设 id（含 ``custom``）。
PROVIDER_IDS = frozenset(item["id"] for item in PROVIDER_PRESETS)

#: ``POST /api/config/test`` 的单次探测超时（秒）与候选上限。
#: 必须小于外层 :data:`TEST_DEADLINE`，否则慢网关永远报不出真实的 httpx 超时原因。
TEST_TIMEOUT = 7.0
#: 连接阶段单独收紧：坏地址（拒绝连接/黑洞）3s 内就能判定，不必等满 7s。
TEST_CONNECT_TIMEOUT = 3.0
#: 整次探测的硬截止（秒）。httpx 的 timeout 是**每段**超时，慢网关逐个吐模型时
#: 总耗时能超过 8s（DAIR-257 实测 9912ms），所以外层再用线程 join 兜一条硬线。
TEST_DEADLINE = 7.5
TEST_MODEL_LIMIT = 20

#: 错误串里要抹掉的 URL / 疑似 key。
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")
_SECRET_RE = re.compile(r"sk-[A-Za-z0-9._-]{6,}")

#: 备份文件名：``llm_config-<ts>.json``；同时挡住 ``..`` / 分隔符。
BACKUP_RE = re.compile(r"^llm_config-[0-9A-Za-z_.-]+\.json$")
BACKUP_PREFIX = "llm_config-"

#: 进程内写锁：单进程服务里保证「读 hash → 校验 → 备份 → 替换」不被交叉。
_WRITE_LOCK = threading.RLock()

#: 忙碌判定包含的状态：排队中的 run 也会读当前配置，同样不能改。
_BUSY_STATUSES = "queued,running,awaiting_plan"


class ConfigValidationError(ValueError):
    """白名单 / 取值校验失败（映射成 400）。"""


class ConfigPatchBody(BaseModel):
    """``PUT /api/config`` 请求体。

    ``base_sha256`` 是 GET 拿到的乐观并发令牌。``patch`` 只放白名单键；
    ``api_key`` 是独立字段（留空 = 不改），不允许出现在 ``patch`` 里。

    未知字段直接 422：拼错字段名时宁可报错，也不要“看起来保存成功”其实什么都没写。
    """

    model_config = ConfigDict(extra="forbid")

    base_sha256: str | None = None
    patch: dict[str, Any] = Field(default_factory=dict)
    api_key: str | None = None


class ConfigTestBody(BaseModel):
    """``POST /api/config/test`` 请求体。

    ``api_key`` 缺省 = 沿用已保存的真 key（不是 GET 返回的脱敏值）；
    ``api_base`` 为空直接报“先填地址”，不猜。
    """

    model_config = ConfigDict(extra="forbid")

    provider: str | None = None
    api_base: str = ""
    model: str | None = None
    api_key: str | None = None


def resolve_config_path(
    workspace: str | Path, explicit: str | Path | None = None
) -> Path:
    """与引擎一致的查找顺序：workspace → 环境变量 → 仓库自带 config。"""
    if explicit:
        return Path(explicit)
    candidate = Path(workspace) / CONFIG_RELATIVE
    if candidate.is_file():
        return candidate
    env_path = os.environ.get("GIS_LLM_CONFIG")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[3] / CONFIG_RELATIVE


def _is_secret_key(key: str) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return normalized == "apikey" or normalized.endswith("api_key")


def mask_config(value: Any) -> Any:
    """递归脱敏：任何层级的 ``api_key`` 都只留后 4 位。"""
    if isinstance(value, dict):
        masked: dict[str, Any] = {}
        for key, item in value.items():
            if _is_secret_key(key):
                masked[key] = mask_secret(item if isinstance(item, str) else str(item))
            else:
                masked[key] = mask_config(item)
        return masked
    if isinstance(value, list):
        return [mask_config(item) for item in value]
    return value


def read_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise HTTPException(status_code=404, detail="config not found")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"config unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="config must be a JSON object")
    return mask_config(data)


# ---------------------------------------------------------------- 序列化 / 落盘


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def serialize_config(data: dict[str, Any], original: bytes) -> bytes:
    """按原文件的缩进与换行风格回写，避免一次保存把整个文件重排。"""
    newline = "\r\n" if b"\r\n" in original else "\n"
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    return text.replace("\n", newline).encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    """同目录临时文件 + ``os.replace``：断电/崩溃时不会留下半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 0o600
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        pass
    fd, tmp_name = tempfile.mkstemp(prefix=".llm_config-", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(tmp, mode)
        except OSError:  # pragma: no cover - Windows 上 chmod 只影响只读位
            pass
        os.replace(tmp, path)
    finally:
        if tmp.exists():  # pragma: no cover - 只有替换失败才会走到
            try:
                tmp.unlink()
            except OSError:
                pass


def backup_dir_for(config_path: Path) -> Path:
    return Path(config_path).parent / "backups"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S_%f")[:-3]


def create_backup(config_path: Path, payload: bytes) -> str | None:
    """把当前文件复制进 ``config/backups/``，返回备份文件名。"""
    directory = backup_dir_for(config_path)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{BACKUP_PREFIX}{_timestamp()}"
    name = f"{stem}.json"
    index = 0
    while (directory / name).exists():
        index += 1
        name = f"{stem}-{index}.json"
    shutil.copyfile(config_path, directory / name)
    return name


def list_backups(config_path: Path) -> list[dict[str, Any]]:
    directory = backup_dir_for(config_path)
    if not directory.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in directory.glob(f"{BACKUP_PREFIX}*.json"):
        if not BACKUP_RE.match(path.name):
            continue
        try:
            stat = path.stat()
            raw = path.read_bytes()
        except OSError:  # pragma: no cover - 备份被并发删除
            continue
        created = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        items.append(
            {
                "name": path.name,
                "size_bytes": stat.st_size,
                "created_at": created.isoformat().replace("+00:00", "Z"),
                "sha256": _sha256(raw),
            }
        )
    items.sort(key=lambda item: item["created_at"], reverse=True)
    return items


def _display_path(workspace: str | Path, path: Path) -> str:
    """只回相对 workspace 的路径；配置在别处时只给文件名，不泄漏绝对路径。"""
    try:
        return Path(path).resolve().relative_to(Path(workspace).resolve()).as_posix()
    except (ValueError, OSError):
        return Path(path).name


# ---------------------------------------------------------------- 校验 / diff


def _require_model_list(value: Any, path: str, *, allow_empty: bool) -> list[str]:
    if not isinstance(value, list):
        raise ConfigValidationError(f"{path} 必须是字符串数组")
    models: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigValidationError(f"{path} 里只能是非空字符串")
        models.append(item.strip())
    if not models and not allow_empty:
        raise ConfigValidationError(f"{path} 至少要有一个模型")
    return models


def _require_int(value: Any, path: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(f"{path} 必须是整数")
    if not low <= value <= high:
        raise ConfigValidationError(f"{path} 超出范围（{low}..{high}）")
    return value


def _require_number(value: Any, path: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(f"{path} 必须是数字")
    number = float(value)
    if not low <= number <= high:
        raise ConfigValidationError(f"{path} 超出范围（{low}..{high}）")
    return number


def _require_model_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigValidationError("model 必须是字符串")
    text = value.strip()
    if not text:
        raise ConfigValidationError("model 不能为空")
    if len(text) > 200:
        raise ConfigValidationError("model 太长（≤200 字符）")
    return text


def _require_api_base(value: Any) -> str:
    """空串 = 用官方默认地址；否则必须是干净的 http(s) 地址。

    ``#片段`` / 控制字符（含 NUL）/ ``user:pass@`` 都会被引擎或 SDK 静默截断、
    丢弃，或把密码写进日志，用户只看到「连接测试莫名失败」，所以落盘前就拒掉。
    """
    if not isinstance(value, str):
        raise ConfigValidationError("api_base 必须是字符串")
    text = value.strip()
    if not text:
        return ""
    if len(text) > 500:
        raise ConfigValidationError("api_base 太长（≤500 字符）")
    if not text.lower().startswith(("http://", "https://")):
        raise ConfigValidationError("api_base 必须以 http:// 或 https:// 开头")
    if any(ch.isspace() for ch in text):
        raise ConfigValidationError("api_base 不能包含空白字符")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in text):
        raise ConfigValidationError("api_base 不能包含控制字符")
    if "?" in text:
        raise ConfigValidationError("api_base 不能带查询串（?...）")
    if "#" in text:
        raise ConfigValidationError("api_base 不能带片段（#...），SDK 会把它丢弃")
    authority = text.split("://", 1)[1].split("/", 1)[0]
    if not authority:
        raise ConfigValidationError("api_base 缺少主机名")
    if "@" in authority:
        raise ConfigValidationError("api_base 不能包含账号密码（user:pass@）")
    return text


def _require_provider(value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigValidationError("provider 必须是字符串")
    text = value.strip()
    if text not in PROVIDER_IDS:
        raise ConfigValidationError(
            f"未知供应商：{text or '(空)'}（可选：{', '.join(sorted(PROVIDER_IDS))}）"
        )
    return text


def _require_timeout(value: Any) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError("timeout 必须是数字")
    if not 5 <= float(value) <= 3600:
        raise ConfigValidationError("timeout 超出范围（5..3600 秒）")
    return value


def apply_patch(config: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """把白名单 patch 合进配置副本；任何越界键都抛 ``ConfigValidationError``。"""
    if not isinstance(patch, dict):
        raise ConfigValidationError("patch 必须是 JSON 对象")
    unknown = sorted(str(key) for key in patch if key not in PATCH_KEYS)
    if unknown:
        raise ConfigValidationError(f"不允许修改的键：{', '.join(unknown)}")

    updated = copy.deepcopy(config)

    if "routing_rules" in patch:
        rules = patch["routing_rules"]
        if not isinstance(rules, dict):
            raise ConfigValidationError("routing_rules 必须是对象")
        target = updated.get("routing_rules")
        if not isinstance(target, dict):
            target = {}
            updated["routing_rules"] = target
        for role, models in rules.items():
            role_name = str(role)
            if role_name not in ROLE_NAMES:
                raise ConfigValidationError(f"未知角色：{role_name}")
            target[role_name] = _require_model_list(
                models, f"routing_rules.{role_name}", allow_empty=False
            )

    if "fallback_models" in patch:
        updated["fallback_models"] = _require_model_list(
            patch["fallback_models"], "fallback_models", allow_empty=True
        )

    if "engine" in patch:
        engine = patch["engine"]
        if not isinstance(engine, dict):
            raise ConfigValidationError("engine 必须是对象")
        target_engine = updated.get("engine")
        if not isinstance(target_engine, dict):
            target_engine = {}
            updated["engine"] = target_engine
        for key, value in engine.items():
            key_name = str(key)
            if key_name not in ENGINE_INT_LIMITS:
                raise ConfigValidationError(f"不允许修改的键：engine.{key_name}")
            low, high = ENGINE_INT_LIMITS[key_name]
            target_engine[key_name] = _require_int(value, f"engine.{key_name}", low, high)

    if "temperature" in patch:
        updated["temperature"] = _require_number(patch["temperature"], "temperature", 0.0, 2.0)

    if "max_tokens" in patch:
        updated["max_tokens"] = _require_int(patch["max_tokens"], "max_tokens", 1, 1_000_000)

    # provider 只如实写：引擎不读它（真正生效的是 api_base/model/api_key），
    # 保留它是给 provider_models 分组和前端“当前供应商”展示用。
    if "provider" in patch:
        updated["provider"] = _require_provider(patch["provider"])

    if "model" in patch:
        updated["model"] = _require_model_name(patch["model"])

    if "api_base" in patch:
        updated["api_base"] = _require_api_base(patch["api_base"])

    if "timeout" in patch:
        updated["timeout"] = _require_timeout(patch["timeout"])

    # 引擎里 ``routing_rules[role]`` 排在 ``model`` 前面（``EngineLLMConfig.models_for``）：
    # 只改 ``model`` 而不动路由，10 个角色仍会先试旧模型，新 run 看着换了模型其实没换
    # （DAIR-265 实测：6 次 llm_result 全是旧模型）。与 CLI 的
    # ``model_catalog.apply_model()`` 对齐：改主模型名时把没被这次特意指定的角色
    # 一起指过去；patch 里显式给了某个角色的路由则以用户为准。
    if "model" in patch and updated.get("model") != config.get("model"):
        rules = updated.get("routing_rules")
        explicit = patch.get("routing_rules")
        explicit = explicit if isinstance(explicit, dict) else {}
        if isinstance(rules, dict):
            for role, models in list(rules.items()):
                if role in explicit:
                    continue
                if isinstance(models, list) and models:
                    rules[role] = [updated["model"]]

    return updated


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(item, path))
        return flat
    # 数组整体当一个叶子：前端 diff 里显示“候选顺序”整行，而不是按索引拆碎。
    return {prefix: value} if prefix else {}


def _secret_path(path: str) -> bool:
    return any(_is_secret_key(part) for part in str(path).split("."))


def diff_configs(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """脱敏后的 before/after 差异（键值对）；secret 只显示脱敏值。"""
    flat_before = _flatten(mask_config(before))
    flat_after = _flatten(mask_config(after))
    rows: list[dict[str, Any]] = []
    for path in sorted(set(flat_before) | set(flat_after)):
        old = flat_before.get(path)
        new = flat_after.get(path)
        if json.dumps(old, sort_keys=True, ensure_ascii=False) == json.dumps(
            new, sort_keys=True, ensure_ascii=False
        ):
            continue
        rows.append(
            {"path": path, "before": old, "after": new, "secret": _secret_path(path)}
        )
    return rows


def _looks_masked(secret: str) -> bool:
    """``sk-…abcd`` / ``***`` 是 GET 的脱敏值；写回它等于把真 key 换成掩码。"""
    return "…" in secret or secret.startswith("***")


def _busy_run_id(manager: Any) -> str | None:
    if manager is None:
        return None
    try:
        jobs = manager.list_jobs(status=_BUSY_STATUSES, limit=5)
    except Exception:  # pragma: no cover - 管理器异常不该把配置页打成 500
        return None
    if not jobs:
        return None
    return str(getattr(jobs[0], "id", "") or "unknown")


def _read_raw(path: Path, *, missing_ok: bool = False) -> tuple[bytes, dict[str, Any]]:
    if not path.is_file():
        if missing_ok:
            return b"", {}
        raise HTTPException(status_code=404, detail="config not found")
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"config unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="config must be a JSON object")
    return raw, data


def _active_summary(data: dict[str, Any], *, config_exists: bool) -> dict[str, Any]:
    """前端「当前生效」卡片 / 首次引导用的摘要。"""
    key = str(data.get("api_key") or "")
    return {
        "provider": str(data.get("provider") or ""),
        "model": str(data.get("model") or ""),
        "api_base": str(data.get("api_base") or ""),
        # 只给后 4 位（短 key 不给任何尾字符），和 GET 的脱敏口径一致。
        "key_tail": key[-4:] if len(key) > 8 else "",
        "config_exists": config_exists,
    }


def _strip_url_query(url: str) -> str:
    """错误串里的 URL 去掉查询串/片段（key 常藏在 ``?api_key=`` 里）。"""
    for sep in ("?", "#"):
        url = url.split(sep, 1)[0]
    return url


def _sanitize_error(text: Any, secret: str = "") -> str:
    """对外错误串：抹掉真 key、URL 查询串、疑似 sk- 片段，并截断。"""
    value = str(text or "").strip()
    if not value:
        return "连接失败（上游没有返回错误详情）"
    if secret:
        value = value.replace(secret, "***")
    value = _URL_RE.sub(lambda match: _strip_url_query(match.group(0)), value)
    value = _SECRET_RE.sub("sk-***", value)
    return value[:400]


def _error_detail(exc: Exception) -> str:
    """openai 的连接类异常 ``str()`` 只有 "Connection error."，真正的系统错误在 ``__cause__``。"""
    text = str(exc or "").strip()
    cause = exc.__cause__ or exc.__context__
    detail = str(cause).strip() if cause is not None else ""
    if detail and detail not in text:
        text = f"{text} {detail}".strip()
    return text


def _is_loopback(api_base: str) -> bool:
    """地址是不是本机回环（localhost / 127.0.0.0-8 / ::1）。"""
    host = (urlparse(str(api_base or "")).hostname or "").strip("[]").lower()
    return host == "localhost" or host == "::1" or host.startswith("127.")


def _probe_models(api_base: str, api_key: str) -> list[str]:
    """真实探测 ``GET {api_base}/models``（失败向上抛，由调用方转成 ok:false）。"""
    import httpx
    from openai import OpenAI

    # 本机地址不走系统代理：httpx 会从 Windows 注册表读代理设置（实测本机反代
    # 127.0.0.1:8789 被代理拖到 5s+、不存在的 127.0.0.1:9 被代理回一个假 502）。
    # 本机地址连不上就该是一次快速的真连接失败，而不是慢半天再报一个代理错误。
    http_kwargs: dict[str, Any] = {}
    if _is_loopback(api_base):
        http_kwargs["http_client"] = httpx.Client(trust_env=False)
    # with：连同传入的 http_client 一起关掉（每点一次「测试连接」不留未关闭客户端）
    with OpenAI(
        api_key=api_key or "none",
        base_url=api_base,
        timeout=httpx.Timeout(TEST_TIMEOUT, connect=TEST_CONNECT_TIMEOUT),
        max_retries=0,
        **http_kwargs,
    ) as client:
        page = client.models.list()
        names: list[str] = []
        for item in getattr(page, "data", []) or []:
            name = str(getattr(item, "id", "") or "").strip()
            if name and name not in names:
                names.append(name)
    return names


def _probe_models_within(api_base: str, api_key: str) -> list[str]:
    """探测但**硬限**总耗时：超过 :data:`TEST_DEADLINE` 直接放弃。

    httpx 的 timeout 是每段（connect/read/write/pool）的超时，不是整次请求的总时长，
    所以“慢但一直有响应”的网关能超过设计界 8s。这里把阻塞调用丢到 daemon 线程，
    超时就把它抛在后面（它自己还会按 httpx 超时退出），请求线程问 deadline 内必须返回。
    """
    outcome: dict[str, Any] = {}
    done = threading.Event()

    def _run() -> None:
        try:
            outcome["names"] = _probe_models(api_base, api_key)
        except BaseException as exc:  # noqa: BLE001 - 原样传给调用方
            outcome["error"] = exc
        finally:
            done.set()

    threading.Thread(target=_run, name="config-probe", daemon=True).start()
    if not done.wait(TEST_DEADLINE):
        raise TimeoutError(f"探测超过 {TEST_DEADLINE:g}s 未返回，上游太慢或地址不可达")
    error = outcome.get("error")
    if error is not None:
        raise error
    return list(outcome.get("names") or [])


def _save_result(
    *,
    path: Path,
    raw: bytes,
    before: dict[str, Any],
    after: dict[str, Any],
    backup: str | None,
    workspace: str | Path,
    exists: bool,
) -> dict[str, Any]:
    return {
        "ok": True,
        # 与 GET 同口径：文件不存在时令牌是 ""，不能是 sha256(b"")。
        "sha256": _sha256(raw) if exists else "",
        "backup": backup,
        "diff": diff_configs(before, after),
        "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "path": _display_path(workspace, path),
    }


def build_router(
    workspace: str | Path,
    config_path: str | Path | None = None,
    manager: Any = None,
) -> APIRouter:
    router = APIRouter()

    def _path() -> Path:
        return resolve_config_path(workspace, config_path)

    @router.get("/api/config")
    def get_config() -> dict[str, Any]:
        path = _path()
        # 首次配置：文件不存在时返回 {} + sha256:""，前端据此走引导流程。
        exists = path.is_file()
        raw, data = _read_raw(path, missing_ok=True)
        busy_run = _busy_run_id(manager)
        body: dict[str, Any] = mask_config(data)
        body["sha256"] = _sha256(raw) if exists else ""
        body["_meta"] = {
            # 控制台是本机自用（token 保护），新建运行页要只读展示当前工作区。
            "workspace": str(Path(workspace).resolve()),
            "path": _display_path(workspace, path),
            "backup_dir": _display_path(workspace, backup_dir_for(path)),
            "size_bytes": len(raw),
            "config_exists": exists,
            "busy": busy_run is not None,
            "busy_run": busy_run,
            "writable": busy_run is None,
            "active": _active_summary(data, config_exists=exists),
        }
        return body

    @router.get("/api/config/providers")
    def get_providers() -> dict[str, Any]:
        items = [dict(item) for item in PROVIDER_PRESETS]
        return {"items": items, "count": len(items)}

    @router.post("/api/config/test")
    def test_config(body: ConfigTestBody) -> dict[str, Any]:
        """现场测连接：成功给候选模型清单，失败也 200 + ok:false（带脱敏原因）。"""
        provider = str(body.provider or "").strip()
        base = str(body.api_base or "").strip()
        model = str(body.model or "").strip()
        result: dict[str, Any] = {
            "ok": False,
            "latency_ms": 0,
            "models": [],
            "error": "",
            "provider": provider,
            "api_base": base,
            "model": model,
            "key_source": "none",
        }

        if provider and provider not in PROVIDER_IDS:
            result["error"] = f"未知供应商：{provider}"
            return result
        if not base:
            result["error"] = "先填地址（api_base 不能为空）"
            return result
        try:
            base = _require_api_base(base)
        except ConfigValidationError as exc:
            result["error"] = str(exc)
            return result
        result["api_base"] = base

        try:
            _, data = _read_raw(_path(), missing_ok=True)
        except HTTPException as exc:
            # 配置文件损坏 / 不是 JSON 对象：仍走「失败也 200 + ok:false」契约。
            result["error"] = _sanitize_error(exc.detail)
            return result
        configured_key = str(data.get("api_key") or "")
        if body.api_key is not None and body.api_key.strip():
            secret = body.api_key.strip()
            if _looks_masked(secret):
                result["error"] = (
                    "检测到脱敏值，请填写真实 API Key，或留空表示沿用已保存的 key"
                )
                return result
            result["key_source"] = "request"
        else:
            secret = configured_key
            result["key_source"] = "config" if secret else "none"

        result["has_key"] = bool(secret)
        started = time.perf_counter()
        try:
            names = _probe_models_within(base, secret)
        except Exception as exc:
            result["latency_ms"] = int((time.perf_counter() - started) * 1000)
            result["error"] = _sanitize_error(_error_detail(exc), secret)
            logger.info(
                "config test failed: base=%s provider=%s key_source=%s",
                base,
                provider or "-",
                result["key_source"],
            )
            return result

        result["latency_ms"] = int((time.perf_counter() - started) * 1000)
        result["models"] = names[:TEST_MODEL_LIMIT]
        result["ok"] = True
        logger.info(
            "config test ok: base=%s provider=%s models=%d latency_ms=%d",
            base,
            provider or "-",
            len(names),
            result["latency_ms"],
        )
        return result

    @router.put("/api/config")
    def put_config(body: ConfigPatchBody) -> dict[str, Any]:
        path = _path()
        # None = 没带令牌（428）；"" = 空配置（文件不存在）的合法令牌，见 GET。
        if body.base_sha256 is None:
            raise HTTPException(
                status_code=428,
                detail="缺少 base_sha256：请先 GET /api/config 再保存（乐观并发）",
            )
        with _WRITE_LOCK:
            exists = path.is_file()
            raw, before = _read_raw(path, missing_ok=True)
            current_sha = _sha256(raw) if exists else ""
            # 先校验 patch：非法键在任何情况下都不该落盘，也不该被并发/忙碌掩盖。
            try:
                after = apply_patch(before, body.patch)
            except ConfigValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            key_changed = False
            if body.api_key is not None:
                secret = body.api_key.strip()
                if secret:
                    if _looks_masked(secret):
                        raise HTTPException(
                            status_code=400,
                            detail="检测到脱敏值，请填写真实 API Key，或留空表示不修改",
                        )
                    after["api_key"] = secret
                    key_changed = True

            busy_run = _busy_run_id(manager)
            if busy_run is not None:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"有 run 正在运行（{busy_run}）：配置只对下一个 run 生效，"
                        "请等当前 run 结束后再保存"
                    ),
                )

            if body.base_sha256 != current_sha:
                raise HTTPException(
                    status_code=409,
                    detail="配置文件已被改动（sha256 不匹配），请刷新后再保存",
                )

            rows = diff_configs(before, after)
            if not rows:
                logger.info("config save skipped: no changes (path=%s)", path.name)
                return _save_result(
                    path=path,
                    raw=raw,
                    before=before,
                    after=after,
                    backup=None,
                    workspace=workspace,
                    exists=exists,
                )

            payload = serialize_config(after, raw)
            # 空配置第一次保存：没有原文可备份，直接建文件。
            backup = create_backup(path, raw) if exists else None
            atomic_write(path, payload)
            logger.info(
                "config saved: changes=%d backup=%s api_key_changed=%s path=%s",
                len(rows),
                backup,
                key_changed,
                path.name,
            )
            return _save_result(
                path=path,
                raw=payload,
                before=before,
                after=after,
                backup=backup,
                workspace=workspace,
                exists=True,
            )

    @router.get("/api/config/backups")
    def get_backups() -> dict[str, Any]:
        path = _path()
        items = list_backups(path)
        return {
            "items": items,
            "count": len(items),
            "dir": _display_path(workspace, backup_dir_for(path)),
        }

    @router.post("/api/config/revert/{backup_name}")
    def revert_config(backup_name: str) -> dict[str, Any]:
        if not BACKUP_RE.match(backup_name or ""):
            raise HTTPException(status_code=400, detail="备份文件名不合法")
        path = _path()
        directory = backup_dir_for(path)
        backup_path = directory / backup_name
        try:
            resolved = backup_path.resolve()
        except OSError as exc:  # pragma: no cover - 极端路径
            raise HTTPException(status_code=400, detail="备份文件名不合法") from exc
        if resolved.parent != directory.resolve() or not resolved.is_file():
            raise HTTPException(status_code=404, detail="备份不存在")

        with _WRITE_LOCK:
            busy_run = _busy_run_id(manager)
            if busy_run is not None:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"有 run 正在运行（{busy_run}）：配置只对下一个 run 生效，"
                        "请等当前 run 结束后再还原"
                    ),
                )

            raw, before = _read_raw(path)
            try:
                backup_raw = resolved.read_bytes()
                restored = json.loads(backup_raw.decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"备份文件不是合法配置：{exc}") from exc
            if not isinstance(restored, dict):
                raise HTTPException(status_code=400, detail="备份文件不是 JSON 对象")

            rows = diff_configs(before, restored)
            if not rows:
                return _save_result(
                    path=path,
                    raw=raw,
                    before=before,
                    after=restored,
                    backup=None,
                    workspace=workspace,
                    exists=True,
                )

            # 原样写回备份字节（不重新序列化）：还原必须逐字节等于当初的原文。
            payload = backup_raw
            snapshot = create_backup(path, raw)  # 还原前也留一份，还原本身可撤销
            atomic_write(path, payload)
            logger.info(
                "config reverted: from=%s changes=%d pre_revert_backup=%s",
                backup_name,
                len(rows),
                snapshot,
            )
            result = _save_result(
                path=path,
                raw=payload,
                before=before,
                after=restored,
                backup=snapshot,
                workspace=workspace,
                exists=True,
            )
            result["restored_from"] = backup_name
            return result

    return router

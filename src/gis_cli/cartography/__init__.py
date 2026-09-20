# -*- coding: utf-8 -*-
"""制图能力包：把"数据事实"变成"专业版式"的完整管线。

    facts（采集事实） → design（规则引擎出 LayoutSpec） → apply（渲染到 ArcGIS）
                        ↑                                        ↓
                        └──── qc（版面体检） ←── 不合格则按建议微调重渲 ────┘

对外只暴露两个入口：

- :func:`design_only`：只做设计，返回人类可读的设计说明 + spec（给 Agent 用，先"读设计"再出图）；
- :func:`create_map`：一条龙出图（含版面自修复闭环），配方与工具都调它。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from . import facts as _facts_mod
from .design import design_layout, normalize_intent, suggest_fixes
from .profiles import DEFAULT_PROFILE_NAME, load_profile, profile_names
from .styles import load_catalog

logger = logging.getLogger(__name__)

MAX_REPAIR_ROUNDS = 2


def collect_facts(layers: list[Any], *, field: str = "", code_runner: Any = None) -> dict[str, Any]:
    return _facts_mod.collect_map_facts(layers, field=field, code_runner=code_runner)


def design_only(
    layers: list[Any],
    intent: dict[str, Any] | None = None,
    *,
    code_runner: Any = None,
    facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """只做设计：返回 ``{"spec", "facts", "summary"}``（不落盘、不渲染）。"""
    data_facts = facts if facts is not None else collect_facts(
        layers, field=normalize_intent(intent).get("field", ""), code_runner=code_runner)
    spec = design_layout(data_facts, intent)
    return {"spec": spec, "facts": data_facts, "summary": describe_spec(spec)}


def create_map(
    layers: list[Any],
    intent: dict[str, Any] | None = None,
    *,
    output_path: str,
    aprx_path: str = "",
    field: str = "",
    code_runner: Any = None,
    facts: dict[str, Any] | None = None,
    max_rounds: int = MAX_REPAIR_ROUNDS,
) -> dict[str, Any]:
    """出图（含版面自修复）：facts → design → render → qc →（不合格）微调重渲。"""
    from . import apply as _apply
    from . import qc as _qc

    normalized = normalize_intent(intent)
    if field and not normalized.get("field"):
        normalized["field"] = field
    data_facts = facts if facts is not None else collect_facts(
        layers, field=normalized.get("field", "") or field, code_runner=code_runner)
    profile = load_profile(normalized.get("style_profile"))
    catalog = load_catalog()

    hints: dict[str, Any] = {}
    repair_log: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    # 每轮渲染到**临时路径**，最后一轮结束后再拷贝到目标：
    # 否则第二轮会因第一轮的 .aprx 仍被 ArcGIS 占用而报 PermissionError。
    out_path = Path(output_path).expanduser()
    import tempfile as _tempfile

    work_dir = Path(_tempfile.mkdtemp(prefix="cartography_"))   # 每次调用独立临时目录，避免上一步的 ArcGIS 句柄冲突
    work_dir.mkdir(parents=True, exist_ok=True)
    rounds = max(1, max_rounds + 1)
    final_output = ""
    final_aprx = ""
    for round_index in range(rounds):
        round_out = str(work_dir / f"round{round_index}{out_path.suffix or '.jpg'}")
        round_aprx = str(work_dir / f"round{round_index}.aprx")
        spec = design_layout(data_facts, normalized, profile=profile, hints=hints)
        result = _apply.render(spec, round_out, aprx_path=round_aprx, catalog=catalog)
        final_output, final_aprx = result["output"], result["aprx"]
        verdict = _qc.check_layout(final_aprx, spec=spec, image_path=final_output)
        result["qc_ok"] = bool(verdict.get("ok"))
        result["qc"] = verdict
        result["qc_summary"] = _qc.summarize(verdict)
        result["repair_rounds"] = round_index
        if verdict.get("ok") or round_index >= max_rounds:
            break
        new_hints = suggest_fixes(verdict.get("checks") or [])
        repair_log.append({"round": round_index, "failed": _failed_names(verdict), "hints": new_hints})
        if not new_hints:
            break
        hints.update(new_hints)

    # 把获胜轮次的成果拷贝到目标路径（此时已不再持有临时文件句柄）
    result["output"] = _publish(final_output, str(out_path))
    aprx_target = str(Path(aprx_path).expanduser()) if aprx_path else str(out_path.with_suffix(".aprx"))
    result["aprx"] = _publish(final_aprx, aprx_target)
    try:
        result["size"] = Path(result["output"]).stat().st_size
        result["aprx_size"] = Path(result["aprx"]).stat().st_size
        result["aprx_saved"] = True
    except Exception:
        pass
    import shutil as _shutil

    _shutil.rmtree(work_dir, ignore_errors=True)
    result["repair_log"] = repair_log
    result["design_note"] = result.get("design_note") or ""
    result["summary_note"] = describe_spec(result.get("spec") or {})
    return result


def _publish(source: str, target: str) -> str:
    """把临时产物拷贝到目标路径（目标被 ArcGIS Pro 占用时给出明确提示）。

    批量出图时 ArcGIS 进程可能还持有上一步的文件句柄，因此加几次短重试。
    """
    import shutil as _shutil
    import time as _time

    if not source:
        return target
    if os.path.abspath(source) == os.path.abspath(target):
        return target
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            _shutil.copy2(source, target)
            return target
        except PermissionError as exc:
            last_error = exc
            _time.sleep(0.6 * (attempt + 1))
        except OSError as exc:
            last_error = exc
            _time.sleep(0.4)
    raise RuntimeError(
        f"无法写入 {target}（{last_error}）：该文件可能正在 ArcGIS Pro 中打开，请先关闭后重试"
    )


def _failed_names(verdict: dict[str, Any]) -> list[str]:
    return [str(c.get("name")) for c in (verdict.get("checks") or []) if not c.get("ok")]


def describe_spec(spec: dict[str, Any]) -> str:
    """把 spec 变成一段人能读的"设计说明"（Agent 可据此向用户解释）。"""
    if not spec:
        return ""
    page = spec.get("page") or {}
    title = spec.get("title") or {}
    legend = spec.get("legend") or {}
    scale_bar = spec.get("scale_bar") or {}
    north = spec.get("north_arrow") or {}
    renderer = spec.get("renderer") or {}
    lines = [
        f"纸张：{page.get('paper')} {page.get('orientation')}（{page.get('width_mm')}×{page.get('height_mm')} mm）",
        f"图名：{title.get('text')}（{title.get('height_pt')}pt，{title.get('lines')} 行，顶部居中）",
        (
            f"图例：{'放' + str(legend.get('corner')) if legend.get('needed') else '不放'}"
            + (f"，标题「{legend.get('title')}」" if legend.get('title') else "，无标题")
            + f"，标签模式 {renderer.get('labels_mode')}"
        ),
        f"比例尺：{scale_bar.get('length_m'):g} m / {scale_bar.get('divisions')} 段，单位 {scale_bar.get('unit_label')}"
        f"（地图比例尺约 1:{scale_bar.get('map_scale'):,.0f}）",
        f"指北针：{'罗盘玫瑰' if north.get('needed') else '不放'}，{north.get('size_mm'):g} mm 右上角",
        f"配色：{renderer.get('color_ramp') or renderer.get('category_colors') or '默认'}，"
        f"{renderer.get('class_count')} 类（{renderer.get('classification_method')}）",
    ]
    reasons = [f"- {d.get('key')}：{d.get('value')}（{d.get('why')}）" for d in (spec.get("decisions") or [])]
    if reasons:
        lines.append("决策理由：")
        lines.extend(reasons)
    return "\n".join(lines)


__all__ = [
    "collect_facts",
    "design_only",
    "create_map",
    "describe_spec",
    "design_layout",
    "normalize_intent",
    "suggest_fixes",
    "load_profile",
    "profile_names",
    "DEFAULT_PROFILE_NAME",
]
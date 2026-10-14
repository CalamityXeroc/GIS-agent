# -*- coding: utf-8 -*-
"""把 UI 各层接到引擎上（一次性任务与交互会话共用）。

- :func:`make_event_renderer`：引擎事件 → 屏幕（并累积状态行、缓存可折叠输出）
- :func:`render_finish`：一轮结束 → 结果卡片 + 状态行 + 追踪路径
- :func:`engine_meta`：从 bundle 取模型/数据目录/配方数等横幅信息
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from . import events as ui_events
from . import render as ui_render


def engine_meta(bundle: Any) -> dict[str, Any]:
    """收集横幅要用的信息（缺什么都不报错）。"""
    meta: dict[str, Any] = {"model": "", "api_base": "", "workspace": "", "catalog": "空", "recipes": 0, "kernel": "-", "session": ""}
    config = getattr(getattr(bundle, "llm", None), "config", None)
    if config is not None:
        meta["model"] = str(getattr(config, "model", "") or "")
        meta["api_base"] = str(getattr(config, "api_base", "") or "")
    workspace = getattr(bundle, "workspace", None)
    if workspace is not None:
        meta["workspace"] = str(workspace)
    try:
        counts = (bundle.catalog.counts() or {}) if getattr(bundle, "catalog", None) is not None else {}
        meta["catalog"] = "、".join(f"{kind} {n}" for kind, n in counts.items()) or "空"
    except Exception:
        meta["catalog"] = "空"
    try:
        meta["recipes"] = len(bundle.recipes.all())
    except Exception:
        meta["recipes"] = 0
    if getattr(bundle, "code_runner", None) is not None:
        meta["kernel"] = "ready"
    loop_state = getattr(getattr(bundle, "loop", None), "state", None)
    meta["session"] = str(getattr(loop_state, "run_id", "") or "")
    return meta


def make_event_renderer(
    console: Any,
    tracker: ui_events.StatusTracker,
    store: ui_render.ToolOutputStore,
    *,
    show_turn_headers: bool = True,
) -> Callable[[str, dict], None]:
    """生成引擎事件回调：更新状态行并渲染到终端。"""

    def _render(name: str, payload: dict) -> None:
        tracker.apply(name, payload)
        unit = ui_events.to_unit(name, payload)
        if unit is None:
            return
        if unit.kind == "final":
            # 结果卡片由 render_finish 统一输出，避免两处重复
            return
        if unit.kind == "turn" and not show_turn_headers:
            return
        if name == "tool_result":
            summary = str((payload or {}).get("summary", ""))
            if len(summary) > 200:
                store.add(str((payload or {}).get("tool", "")), summary)
        elapsed = tracker.action_elapsed if name == "tool_result" else None
        console.print(ui_render.render_unit(unit, elapsed=elapsed))

    return _render


def state_payload(state: Any) -> dict[str, Any]:
    """引擎 state → dict（兼容 dataclass 与 dict）。"""
    if hasattr(state, "to_dict"):
        return state.to_dict()
    return dict(state or {})


def render_finish(
    console: Any,
    tracker: ui_events.StatusTracker,
    state: Any,
    *,
    trace_path: str = "",
    show_trace: bool = True,
) -> dict[str, Any]:
    """打印结果卡片 + 追踪路径 + 状态行，并同步状态行数据。"""
    payload = state_payload(state)
    tracker.turn = int(payload.get("turn") or tracker.turn or 0)
    tracker.status = str(payload.get("status") or tracker.status or "stopped")
    tracker.current_action = ""
    if not tracker.finished_at:
        import time as _time

        tracker.finished_at = _time.monotonic()

    console.print()
    console.print(ui_render.result_card(payload))
    if show_trace and trace_path:
        from rich.text import Text

        console.print(Text(f"追踪 {trace_path}", style="gis.dim"))
    if tracker.status == "awaiting_user":
        from rich.text import Text

        console.print(Text("引擎在等待你的回答：直接输入补充信息即可。", style="gis.warn"))
    console.print(ui_render.footer_text(tracker, width=getattr(console, "width", 0) or 0))
    return payload


def banner_for(console: Any, bundle: Any, *, version: str = "") -> None:
    """打印启动横幅。"""
    meta = engine_meta(bundle)
    console.print(
        ui_render.banner(
            model=meta["model"],
            api_base=meta["api_base"],
            workspace=meta["workspace"],
            catalog=meta["catalog"],
            recipes=meta["recipes"],
            session_id=meta["session"],
            kernel=meta["kernel"],
            version=version,
        )
    )


__all__ = [
    "banner_for",
    "engine_meta",
    "make_event_renderer",
    "render_finish",
    "state_payload",
]

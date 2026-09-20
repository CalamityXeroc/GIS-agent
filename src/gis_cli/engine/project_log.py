# -*- coding: utf-8 -*-
"""项目日志：把每次运行的事实结论写成可续接的简短记录。

竞赛/生产里的 GIS 任务常是链式的（上午三步、下午六步），后一步要接着前一步
的产物与口径继续做。日志只记事实（产物路径、关键数字、口径、踩过的坑），
不记过程流水账，目的有两个：

1. 下一次运行开始时，把最近几条读回上下文，避免重复劳动或口径漂移；
2. 给用户留一份人能读的进展记录。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LOG_NAME = "PROJECT_LOG.md"
_HEADER = (
    "# 项目日志\n\n"
    "> 由引擎在每次运行结束时自动追加。只记事实：产物、关键数字、口径、遗留问题。\n"
)

_STATUS_LABEL = {
    "completed": "已完成",
    "max_turns": "未完成（轮次耗尽）",
    "stopped": "已停止",
    "awaiting_user": "等待用户",
    "llm_unavailable": "中断（模型不可用）",
}


def log_path(workspace: str | Path) -> Path:
    return Path(workspace) / ".gis_agent" / LOG_NAME


def read_recent(workspace: str | Path, max_chars: int = 2500) -> str:
    """读取日志尾部（最近若干条），用于注入下一次运行的上下文。"""
    path = log_path(workspace)
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover
        logger.warning("read project log failed: %s", exc)
        return ""
    entries = _split_entries(text)
    picked: list[str] = []
    total = 0
    for entry in reversed(entries):
        if total + len(entry) > max_chars and picked:
            break
        picked.append(entry)
        total += len(entry)
    return "\n".join(reversed(picked)).strip()


def _split_entries(text: str) -> list[str]:
    blocks: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith("## "):
            blocks.append([line])
        elif blocks:
            blocks[-1].append(line)
    return ["\n".join(block).strip() for block in blocks if len(block) > 1]


def _fmt_numbers(text: str, limit: int = 1200) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit] + ("…" if len(text) > limit else "")


def build_entry(
    *,
    goal: str,
    status: str,
    turns: int,
    artifacts: list[str],
    workspace: str | Path,
    summary: str = "",
    hints: list[str] | None = None,
    design_notes: list[str] | None = None,
    run_id: str = "",
) -> str:
    """生成一条日志条目（确定性，不调用模型）。"""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    goal_line = " ".join(str(goal or "").split())[:200]
    label = _STATUS_LABEL.get(status, status)
    lines = [f"## {stamp} · {run_id or '-'} · {label}", f"- 目标: {goal_line}"]
    lines.append(f"- 轮次: {turns} · 产物: {len(artifacts)} 个")
    if summary:
        lines.append(f"- 结论: {_fmt_numbers(summary)}")
    if artifacts:
        shown = [_rel(path, workspace) for path in artifacts[:10]]
        more = f" 等 {len(artifacts)} 个" if len(artifacts) > 10 else ""
        lines.append(f"- 关键产出: {', '.join(shown)}{more}")
    if hints:
        lines.append("- 踩过的坑: " + "；".join(hint.split("] ", 1)[-1][:160] for hint in hints[:4]))
    if design_notes:
        lines.append("- 制图设计: " + "；".join(str(note)[:200] for note in design_notes[:3]))
    lines.append("")
    return "\n".join(lines)


def _rel(path: str, workspace: str | Path) -> str:
    try:
        return str(Path(path).relative_to(Path(workspace)))
    except Exception:
        return str(path)


def append_entry(workspace: str | Path, entry: str) -> Path:
    """追加一条日志条目（文件不存在时先写表头）。"""
    path = log_path(workspace)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(_HEADER + "\n", encoding="utf-8")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(entry.rstrip() + "\n\n")
    except Exception as exc:  # pragma: no cover
        logger.warning("append project log failed: %s", exc)
    return path


def record_run(
    workspace: str | Path,
    *,
    goal: str,
    status: str,
    turns: int,
    artifacts: list[str] | None = None,
    summary: str = "",
    hints: list[str] | None = None,
    design_notes: list[str] | None = None,
    run_id: str = "",
) -> str:
    """组装并追加一条运行日志，返回条目文本。"""
    entry = build_entry(
        goal=goal,
        status=status,
        turns=turns,
        artifacts=list(artifacts or []),
        workspace=workspace,
        summary=summary,
        hints=hints,
        design_notes=design_notes,
        run_id=run_id,
    )
    append_entry(workspace, entry)
    return entry


def digest(workspace: str | Path, max_chars: int = 2500) -> str:
    """注入提示词用的日志摘要（无日志时返回空串）。"""
    recent = read_recent(workspace, max_chars=max_chars)
    if not recent:
        return ""
    return (
        "## 项目日志（本工作区此前的运行结论，用于续接；如与当前任务冲突以当前任务为准）\n" + recent
    )


__all__: list[Any] = ["log_path", "read_recent", "append_entry", "record_run", "build_entry", "digest"]
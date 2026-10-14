# -*- coding: utf-8 -*-
"""会话持久化与续聊。

落盘 ``workspace/.gis_agent/sessions/<session_id>.jsonl``（一行一条记录）：
提问、模型/工具摘要、卡点与产出。用途：

- ``/export`` 导出本次转录，``/session`` 查看用量；
- ``--continue`` / ``/resume`` 续聊：把上一次会话压成一段"上下文前言"，
  拼在新任务前面（引擎的对话状态不做反序列化，这样最稳、也不侵入引擎）。

记录格式（每行一个 JSON）::

    {"ts": "2026-10-10T15:00:00", "role": "user", "text": "把点位做核密度"}
    {"ts": "...", "role": "tool", "text": "run_recipe 完成", "meta": {"tool": "run_recipe"}}
    {"ts": "...", "role": "assistant", "text": "已交付 4 张图", "meta": {"status": "completed"}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

MAX_TEXT = 4000


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _clip(text: Any, limit: int = MAX_TEXT) -> str:
    value = "" if text is None else str(text)
    return value if len(value) <= limit else value[: limit - 1] + "…"


@dataclass
class SessionRecord:
    ts: str
    role: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"ts": self.ts, "role": self.role, "text": self.text, "meta": self.meta},
            ensure_ascii=False,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionRecord":
        return cls(
            ts=str(data.get("ts") or ""),
            role=str(data.get("role") or "note"),
            text=str(data.get("text") or ""),
            meta=dict(data.get("meta") or {}),
        )


@dataclass
class SessionInfo:
    session_id: str
    path: Path
    mtime: float
    records: int
    first_user_text: str

    @property
    def title(self) -> str:
        text = " ".join(self.first_user_text.split())
        return text[:60] or "(空会话)"


class SessionStore:
    """会话目录的读写。"""

    def __init__(self, workspace: str | Path, *, root: Path | None = None) -> None:
        self.workspace = Path(workspace)
        self.root = Path(root) if root is not None else self.workspace / ".gis_agent" / "sessions"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # -- 基础 ---------------------------------------------------------

    def new_id(self) -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def path_for(self, session_id: str) -> Path:
        safe = "".join(ch for ch in str(session_id) if ch.isalnum() or ch in "-_")
        return self.root / f"{safe or 'session'}.jsonl"

    def append(self, session_id: str, role: str, text: str, **meta: Any) -> None:
        record = SessionRecord(ts=_now(), role=role, text=_clip(text), meta={k: v for k, v in meta.items() if v is not None})
        try:
            with self.path_for(session_id).open("a", encoding="utf-8") as fh:
                fh.write(record.to_json() + "\n")
        except Exception:
            pass

    # -- 读取 ---------------------------------------------------------

    def load(self, session_id: str) -> list[SessionRecord]:
        path = self.path_for(session_id)
        if not path.exists():
            return []
        out: list[SessionRecord] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(SessionRecord.from_dict(json.loads(line)))
                except Exception:
                    continue
        except Exception:
            return []
        return out

    def list_sessions(self, limit: int = 20) -> list[SessionInfo]:
        items: list[SessionInfo] = []
        try:
            paths = sorted(self.root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            return items
        for path in paths[: max(1, limit)]:
            records = self.load(path.stem)
            first_user = next((r.text for r in records if r.role == "user"), "")
            try:
                mtime = path.stat().st_mtime
            except Exception:
                mtime = 0.0
            items.append(
                SessionInfo(session_id=path.stem, path=path, mtime=mtime, records=len(records), first_user_text=first_user)
            )
        return items

    def latest_id(self) -> str:
        items = self.list_sessions(limit=1)
        return items[0].session_id if items else ""

    # -- 续聊 ---------------------------------------------------------

    def context_prefix(self, session_id: str, *, max_records: int = 14, max_chars: int = 1600) -> str:
        """把上一次会话压成"上下文前言"，拼在新任务前面用。"""
        records = self.load(session_id)
        if not records:
            return ""
        picked: list[SessionRecord] = []
        for record in reversed(records):
            if record.role in ("user", "assistant", "tool", "note"):
                picked.append(record)
            if len(picked) >= max_records:
                break
        picked.reverse()

        lines = [f"[接续会话 {session_id} 的上下文（仅供参考，若与本条指令冲突以本条为准）]"]
        budget = max_chars
        for record in picked:
            label = {"user": "用户", "assistant": "助手", "tool": "工具", "note": "记录"}.get(record.role, record.role)
            text = " ".join(record.text.split())
            if not text:
                continue
            entry = f"- {label}：{text[:300]}"
            if budget - len(entry) <= 0:
                break
            budget -= len(entry)
            lines.append(entry)
        if len(lines) == 1:
            return ""
        lines.append("[上下文结束]")
        return "\n".join(lines)

    def export(self, session_id: str, target: str | Path | None = None) -> Path | None:
        """导出为 Markdown（``/export``）。"""
        records = self.load(session_id)
        if not records:
            return None
        target_path = Path(target) if target is not None else self.root / f"{session_id}.md"
        lines = [f"# GIS Agent 会话 {session_id}", ""]
        for record in records:
            label = {"user": "你", "assistant": "Agent", "tool": "工具", "note": "记录"}.get(record.role, record.role)
            lines.append(f"**{record.ts} · {label}**")
            lines.append("")
            lines.append(record.text)
            lines.append("")
        try:
            target_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            return None
        return target_path


def summarize_records(records: Iterable[SessionRecord], *, limit: int = 8) -> str:
    """给 ``/session`` 用的一行行摘要。"""
    lines: list[str] = []
    for record in list(records)[-limit:]:
        label = {"user": "你", "assistant": "Agent", "tool": "工具", "note": "记录"}.get(record.role, record.role)
        lines.append(f"{record.ts}  {label:<6} {' '.join(record.text.split())[:70]}")
    return "\n".join(lines)


__all__ = ["SessionInfo", "SessionRecord", "SessionStore", "summarize_records"]

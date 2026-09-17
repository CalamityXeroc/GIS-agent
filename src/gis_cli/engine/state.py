# -*- coding: utf-8 -*-
"""Task state for the engine loop: task list, requirements, artifacts, history."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


VALID_STATUS = {"pending", "running", "done", "failed", "blocked", "skipped"}


@dataclass
class TaskItem:
    """One item on the visible task list."""

    id: str
    title: str
    status: str = "pending"
    evidence: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "evidence": self.evidence,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskItem":
        status = str(data.get("status", "pending")).lower()
        return cls(
            id=str(data.get("id", "") or f"t{id(data) % 10000}"),
            title=str(data.get("title", "") or ""),
            status=status if status in VALID_STATUS else "pending",
            evidence=str(data.get("evidence", "") or ""),
            detail=str(data.get("detail", "") or ""),
        )


@dataclass
class Requirement:
    """An extracted acceptance requirement."""

    id: str
    text: str
    acceptance: str = ""
    source: str = ""
    status: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "acceptance": self.acceptance,
            "source": self.source,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Requirement":
        return cls(
            id=str(data.get("id", "") or ""),
            text=str(data.get("text", "") or ""),
            acceptance=str(data.get("acceptance", "") or ""),
            source=str(data.get("source", "") or ""),
            status=str(data.get("status", "unknown") or "unknown"),
        )


@dataclass
class Observation:
    """Normalized result of one action, fed back to the model."""

    ok: bool
    summary: str
    data: Any = None
    artifacts: list[str] = field(default_factory=list)
    error: dict[str, Any] | None = None
    hint: str = ""
    images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary,
            "data": self.data,
            "artifacts": self.artifacts,
            "error": self.error,
            "hint": self.hint,
            "images": self.images,
        }

    def to_message(self, max_chars: int = 12000) -> str:
        """Render for the model conversation, truncating bulky payloads."""
        payload = self.to_dict()
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) > max_chars:
            payload["data"] = _truncate(payload.get("data"), 6000)
            payload["images"] = []
            text = json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) > max_chars:
            text = text[:max_chars] + f"...[truncated {len(text) - max_chars} chars]"
        return text


def _truncate(value: Any, limit: int) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return value
    return text[:limit] + f"...[truncated {len(text) - limit} chars]"


@dataclass
class TaskState:
    """Mutable state of one agent run."""

    run_id: str
    goal: str
    workspace: str
    requirements: list[Requirement] = field(default_factory=list)
    tasklist: list[TaskItem] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    status: str = "running"
    error: str = ""
    turn: int = 0
    methodology: str = ""
    final_summary: str = ""
    created_at: str = field(default_factory=_now)

    # ------------------------------------------------------------- task list
    def apply_tasklist(self, items: list[dict[str, Any]]) -> None:
        """Merge a task-list update from the model."""
        by_id = {item.id: item for item in self.tasklist}
        order: list[str] = [item.id for item in self.tasklist]
        for raw in items:
            if not isinstance(raw, dict):
                continue
            item = TaskItem.from_dict(raw)
            if not item.id or not item.title:
                continue
            if item.id in by_id:
                existing = by_id[item.id]
                existing.title = item.title or existing.title
                existing.status = item.status
                if item.evidence:
                    existing.evidence = item.evidence
                if item.detail:
                    existing.detail = item.detail
            else:
                by_id[item.id] = item
                order.append(item.id)
        self.tasklist = [by_id[i] for i in order if i in by_id]

    def add_artifact(self, path: str) -> None:
        if path and path not in self.artifacts:
            self.artifacts.append(path)

    def add_requirements(self, items: list[dict[str, Any]]) -> None:
        existing = {r.id for r in self.requirements}
        for raw in items:
            if not isinstance(raw, dict):
                continue
            req = Requirement.from_dict(raw)
            if req.id and req.id not in existing:
                self.requirements.append(req)
                existing.add(req.id)

    def tasklist_digest(self) -> str:
        if not self.tasklist:
            return "（尚未建立任务清单）"
        marks = {
            "pending": "[ ]",
            "running": "[~]",
            "done": "[x]",
            "failed": "[!]",
            "blocked": "[#]",
            "skipped": "[-]",
        }
        lines = []
        for item in self.tasklist:
            mark = marks.get(item.status, "[ ]")
            line = f"{mark} {item.id} {item.title}"
            if item.evidence:
                line += f"  ({item.evidence})"
            lines.append(line)
        return "\n".join(lines)

    def requirements_digest(self) -> str:
        if not self.requirements:
            return ""
        lines = []
        for req in self.requirements:
            line = f"- {req.id}: {req.text}"
            if req.acceptance:
                line += f"  [验收: {req.acceptance}]"
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "workspace": self.workspace,
            "status": self.status,
            "turn": self.turn,
            "requirements": [r.to_dict() for r in self.requirements],
            "tasklist": [t.to_dict() for t in self.tasklist],
            "artifacts": list(self.artifacts),
            "methodology": self.methodology,
            "final_summary": self.final_summary,
            "error": self.error,
            "created_at": self.created_at,
        }

    def save(self, path: str) -> None:
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

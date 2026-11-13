"""Workflow state manager with LangGraph-compatible abstraction.

This module provides durable thread/workflow state tracking. It works with a
JSON store by default and exposes compatibility metadata for future LangGraph
checkpoint integration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class WorkflowState:
    """Durable per-thread workflow state."""

    thread_id: str
    session_id: str = ""
    status: str = "initialized"
    current_plan_id: str = ""
    current_goal: str = ""
    last_result_success: bool | None = None
    last_error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "session_id": self.session_id,
            "status": self.status,
            "current_plan_id": self.current_plan_id,
            "current_goal": self.current_goal,
            "last_result_success": self.last_result_success,
            "last_error": self.last_error,
            "metadata": self.metadata,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowState":
        updated = data.get("updated_at")
        updated_at = datetime.fromisoformat(updated) if updated else datetime.now(timezone.utc)
        return cls(
            thread_id=str(data.get("thread_id") or ""),
            session_id=str(data.get("session_id") or ""),
            status=str(data.get("status") or "initialized"),
            current_plan_id=str(data.get("current_plan_id") or ""),
            current_goal=str(data.get("current_goal") or ""),
            last_result_success=data.get("last_result_success"),
            last_error=str(data.get("last_error") or ""),
            metadata=dict(data.get("metadata") or {}),
            updated_at=updated_at,
        )


class LangGraphStateManager:
    """Durable state manager with optional LangGraph awareness."""

    def __init__(self, store_path: Path | str | None = None):
        self.store_path = Path(store_path) if store_path else None
        self._states: dict[str, WorkflowState] = {}
        self._langgraph_available = False

        try:
            import langgraph  # noqa: F401

            self._langgraph_available = True
        except Exception:
            self._langgraph_available = False

        self._load()

    @property
    def backend(self) -> str:
        return "langgraph-compatible" if self._langgraph_available else "json"

    def get_state(self, thread_id: str) -> WorkflowState | None:
        return self._states.get(thread_id)

    def list_threads(self) -> list[str]:
        return sorted(self._states.keys())

    def upsert_state(self, thread_id: str, **updates: Any) -> WorkflowState:
        state = self._states.get(thread_id)
        if state is None:
            state = WorkflowState(thread_id=thread_id)
            self._states[thread_id] = state

        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
            else:
                state.metadata[key] = value
        state.updated_at = datetime.now(timezone.utc)
        self._save()
        return state

    def record_event(self, thread_id: str, event: str, payload: dict[str, Any] | None = None) -> WorkflowState:
        payload = payload or {}
        state = self.upsert_state(thread_id)
        events = state.metadata.get("events")
        if not isinstance(events, list):
            events = []
        events.append(
            {
                "event": event,
                "payload": payload,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        state.metadata["events"] = events[-50:]
        state.updated_at = datetime.now(timezone.utc)
        self._states[thread_id] = state
        self._save()
        return state

    def get_summary(self, thread_id: str) -> dict[str, Any]:
        state = self._states.get(thread_id)
        if state is None:
            return {
                "backend": self.backend,
                "thread_id": thread_id,
                "exists": False,
            }
        events = state.metadata.get("events") if isinstance(state.metadata, dict) else []
        if not isinstance(events, list):
            events = []
        return {
            "backend": self.backend,
            "thread_id": thread_id,
            "exists": True,
            "status": state.status,
            "current_plan_id": state.current_plan_id,
            "current_goal": state.current_goal,
            "last_result_success": state.last_result_success,
            "last_error": state.last_error,
            "updated_at": state.updated_at.isoformat(),
            "events_count": len(events),
            "last_event": events[-1] if events else None,
        }

    def _load(self) -> None:
        if self.store_path is None or not self.store_path.exists():
            return
        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            raw_states = payload.get("states")
            if not isinstance(raw_states, dict):
                return
            for thread_id, state_data in raw_states.items():
                if not isinstance(state_data, dict):
                    continue
                state = WorkflowState.from_dict(state_data)
                if not state.thread_id:
                    state.thread_id = str(thread_id)
                self._states[state.thread_id] = state
        except Exception:
            return

    def _save(self) -> None:
        if self.store_path is None:
            return
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "backend": self.backend,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "states": {k: v.to_dict() for k, v in self._states.items()},
            }
            self.store_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            return

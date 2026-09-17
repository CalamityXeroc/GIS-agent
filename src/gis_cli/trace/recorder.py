# -*- coding: utf-8 -*-
"""Structured run tracing for the engine.

Writes one JSONL file per run plus a human-readable Markdown report.
Everything is append-only and crash-safe: a partially written run is still
inspectable.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any, max_str: int = 4000) -> Any:
    """Best-effort JSON-safe conversion with truncation."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_str else value[:max_str] + f"...[truncated {len(value) - max_str} chars]"
    if isinstance(value, dict):
        return {str(k): _jsonable(v, max_str) for k, v in list(value.items())[:100]}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v, max_str) for v in list(value)[:100]]
    return _jsonable(str(value), max_str)


@dataclass
class TraceEvent:
    """A single trace record."""

    seq: int
    ts: str
    event: str
    payload: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = {"seq": self.seq, "ts": self.ts, "event": self.event}
        if self.duration_ms:
            data["duration_ms"] = self.duration_ms
        data["payload"] = self.payload
        return data


class RunRecorder:
    """Append-only recorder for one agent run."""

    def __init__(self, run_id: str, trace_dir: str | Path):
        self.run_id = run_id
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.trace_dir / f"{run_id}.jsonl"
        self.report_path = self.trace_dir / f"{run_id}.md"
        self._seq = 0
        self._started_at = time.time()
        self._fh = open(self.jsonl_path, "a", encoding="utf-8")
        self._summary: dict[str, Any] = {
            "run_id": run_id,
            "started_at": _now_iso(),
            "turns": 0,
            "tool_calls": 0,
            "repairs": 0,
            "errors": 0,
            "total_tokens": 0,
            "artifacts": [],
        }

    # ------------------------------------------------------------------ write
    def record(self, event: str, payload: dict[str, Any] | None = None, duration_ms: int = 0) -> TraceEvent:
        self._seq += 1
        evt = TraceEvent(
            seq=self._seq,
            ts=_now_iso(),
            event=event,
            payload=_jsonable(payload or {}),
            duration_ms=duration_ms,
        )
        self._fh.write(json.dumps(evt.to_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()
        self._track(event, evt.payload)
        return evt

    def _track(self, event: str, payload: dict[str, Any]) -> None:
        if event == "turn":
            self._summary["turns"] += 1
        elif event == "tool_call":
            self._summary["tool_calls"] += 1
        elif event == "repair_attempt":
            self._summary["repairs"] += 1
        elif event in {"tool_error", "llm_error", "run_error"}:
            self._summary["errors"] += 1
        elif event == "llm_result":
            usage = payload.get("usage") or {}
            self._summary["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        elif event == "artifact":
            path = payload.get("path")
            if path and path not in self._summary["artifacts"]:
                self._summary["artifacts"].append(path)

    # ------------------------------------------------------------------ close
    def finish(self, status: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        self._summary["status"] = status
        self._summary["finished_at"] = _now_iso()
        self._summary["duration_seconds"] = round(time.time() - self._started_at, 2)
        if extra:
            self._summary.update(_jsonable(extra))
        self.record("run_end", self._summary)
        try:
            self._fh.close()
        except Exception:
            pass
        self._write_report()
        return self._summary

    def _write_report(self) -> None:
        s = self._summary
        lines = [
            f"# Run {s.get('run_id')}",
            "",
            f"- status: **{s.get('status')}**",
            f"- started: {s.get('started_at')}",
            f"- duration: {s.get('duration_seconds')}s",
            f"- turns: {s.get('turns')}",
            f"- tool calls: {s.get('tool_calls')}",
            f"- repairs: {s.get('repairs')}",
            f"- errors: {s.get('errors')}",
            f"- total tokens: {s.get('total_tokens')}",
            "",
            "## Artifacts",
            "",
        ]
        artifacts = s.get("artifacts") or []
        if artifacts:
            lines.extend(f"- `{p}`" for p in artifacts)
        else:
            lines.append("_none_")
        lines += ["", "## Events", "", "```"]
        try:
            with open(self.jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        evt = json.loads(line)
                    except Exception:
                        continue
                    summary = _event_summary(evt)
                    if summary:
                        lines.append(f"[{evt.get('seq'):>3}] {evt.get('event'):<18} {summary}")
        except Exception:
            pass
        lines += ["```", ""]
        try:
            self.report_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass


def _event_summary(evt: dict[str, Any]) -> str:
    payload = evt.get("payload") or {}
    if evt.get("event") == "tool_call":
        return f"{payload.get('tool')}({_brief(payload.get('args'))})"
    if evt.get("event") == "tool_result":
        return f"ok={payload.get('ok')} {_brief(payload.get('summary'))}"
    if evt.get("event") == "llm_result":
        return f"model={payload.get('model')} tools={len(payload.get('tool_calls') or [])} tokens={payload.get('usage', {}).get('total_tokens')}"
    if evt.get("event") == "repair_attempt":
        return f"attempt={payload.get('attempt')} {_brief(payload.get('error'))}"
    if evt.get("event") == "verification":
        return f"pass={payload.get('pass')} missing={_brief(payload.get('missing'))}"
    if evt.get("event") in {"run_start", "run_end", "run_error", "llm_error", "tool_error"}:
        return _brief(payload)
    if evt.get("event") == "action":
        return _brief(payload)
    return ""


def _brief(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

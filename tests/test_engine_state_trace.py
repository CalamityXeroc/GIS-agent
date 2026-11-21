# -*- coding: utf-8 -*-
"""Unit tests for engine state and trace (no ArcGIS required)."""
from __future__ import annotations

import json
from pathlib import Path

from gis_cli.engine.state import Observation, Requirement, TaskState
from gis_cli.trace.recorder import RunRecorder


def test_tasklist_merge_and_digest():
    state = TaskState(run_id="r1", goal="g", workspace=".")
    state.apply_tasklist(
        [
            {"id": "t1", "title": "扫描数据", "status": "running"},
            {"id": "t2", "title": "生成代码", "status": "pending"},
        ]
    )
    assert [t.id for t in state.tasklist] == ["t1", "t2"]
    state.apply_tasklist([{"id": "t1", "title": "扫描数据", "status": "done", "evidence": "6 个图层"}])
    assert state.tasklist[0].status == "done"
    digest = state.tasklist_digest()
    assert "[x] t1" in digest and "[ ] t2" in digest


def test_artifacts_and_requirements_dedupe():
    state = TaskState(run_id="r1", goal="g", workspace=".")
    state.add_artifact("a.shp")
    state.add_artifact("a.shp")
    assert state.artifacts == ["a.shp"]
    state.add_requirements([{"id": "r1", "text": "投影到 4508"}])
    state.add_requirements([{"id": "r1", "text": "duplicate"}])
    assert len(state.requirements) == 1


def test_observation_message_truncates_bulk():
    obs = Observation(ok=True, summary="s", data={"blob": "x" * 50000})
    text = obs.to_message(max_chars=2000)
    assert len(text) <= 2200
    assert "truncated" in text


def test_recorder_writes_jsonl_and_report(tmp_path: Path):
    recorder = RunRecorder("run_test", tmp_path)
    recorder.record("turn", {"turn": 1})
    recorder.record("tool_call", {"tool": "execute_code", "args": {"code": "print(1)"}})
    recorder.record("llm_result", {"model": "m", "usage": {"total_tokens": 42}})
    summary = recorder.finish("completed")
    assert summary["turns"] == 1
    assert summary["tool_calls"] == 1
    assert summary["total_tokens"] == 42
    lines = recorder.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 4
    assert json.loads(lines[0])["event"] == "turn"
    assert recorder.report_path.exists()

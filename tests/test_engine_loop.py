# -*- coding: utf-8 -*-
"""Loop test with mock LLM/verifier (no ArcGIS required)."""
from __future__ import annotations

import json
from pathlib import Path

from gis_cli.engine.llm_client import LLMResponse, ToolCall
from gis_cli.engine.loop import AgentLoop, LoopConfig
from gis_cli.engine.tools import EngineContext, build_default_registry
from gis_cli.trace.recorder import RunRecorder


class ScriptedLLM:
    class _Cfg:
        model = "mock"

    config = _Cfg()

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(content=json.dumps({"tool": "finish", "args": {"summary": "done"}}), model="mock")


class StubVerifier:
    def __init__(self, verdict=None):
        self.verdict = verdict or {"pass": True, "summary": "验收通过"}

    def verify(self, state):
        return self.verdict


def _tc(name, args):
    return LLMResponse(
        tool_calls=[ToolCall(id=f"c_{name}", name=name, arguments=args, raw_arguments=json.dumps(args))],
        model="mock",
    )


def _make_loop(tmp_path: Path, responses, verdict=None):
    llm = ScriptedLLM(responses)
    registry = build_default_registry()
    context = EngineContext(workspace=tmp_path)
    recorder = RunRecorder("run_unit", tmp_path / "traces")
    loop = AgentLoop(
        workspace=tmp_path,
        llm_client=llm,
        registry=registry,
        context=context,
        verifier=StubVerifier(verdict),
        recorder=recorder,
        config=LoopConfig(max_turns=10),
        auto_extract_requirements=False,
    )
    return loop, llm


def test_loop_completes_after_finish(tmp_path: Path):
    loop, llm = _make_loop(
        tmp_path,
        [
            _tc("update_tasklist", {"items": [{"id": "t1", "title": "做事情", "status": "running"}]}),
            _tc("finish", {"summary": "完成", "methodology": "测试"}),
        ],
    )
    state = loop.run("测试任务")
    assert state.status == "completed"
    assert state.tasklist[0].status == "running"
    assert state.final_summary == "完成"


def test_loop_rejects_failed_verification_then_completes(tmp_path: Path):
    loop, llm = _make_loop(
        tmp_path,
        [
            _tc("finish", {"summary": "第一次"}),
            _tc("finish", {"summary": "第二次"}),
        ],
        verdict={"pass": False, "summary": "缺文件", "repair_hint": "补文件"},
    )
    state = loop.run("测试任务")
    # Stub verifier always fails -> loop hits rejection cap and stops as failed
    assert state.status == "failed"
    assert "验收" in state.error


def test_loop_ask_user_pauses(tmp_path: Path):
    loop, llm = _make_loop(
        tmp_path,
        [_tc("ask_user", {"question": "数据在哪？", "options": ["input", "其他"]})],
    )
    state = loop.run("测试任务")
    assert state.status == "awaiting_user"


def test_loop_stops_gracefully_when_llm_unavailable(tmp_path: Path):
    """Gateway outage must not crash the runner; status becomes llm_unavailable."""
    class DownLLM:
        class _Cfg:
            model = "mock"
        config = _Cfg()

        def chat(self, messages, **kwargs):
            raise RuntimeError("Connection error: gateway down")

    loop, _ = _make_loop(tmp_path, [])
    loop.llm = DownLLM()
    loop.codec.client = DownLLM()
    state = loop.run("测试任务")
    assert state.status == "llm_unavailable"
    assert "不可用" in state.error

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


def test_loop_deadline_disabled_when_nonpositive(tmp_path: Path):
    """run_deadline_seconds <= 0 视为不启用，正常跑完。"""
    loop, _ = _make_loop(tmp_path, [_tc("finish", {"summary": "done"})])
    loop.config.run_deadline_seconds = 0.0
    state = loop.run("测试任务")
    assert state.status == "completed"


def test_loop_deadline_exceeded_with_past_deadline(tmp_path: Path):
    """run_deadline_seconds 很小（如 1e-6 秒）时，第二轮循环顶应触发 deadline_exceeded。"""
    class SlowLLM:
        class _Cfg:
            model = "mock"
        config = _Cfg()

        def chat(self, messages, **kwargs):
            import time as _t

            _t.sleep(0.05)
            return _tc("catalog_query", {"query": "x"})

    import time as _t

    loop, _ = _make_loop(tmp_path, [])
    loop.llm = SlowLLM()
    loop.codec.client = SlowLLM()
    loop.config.run_deadline_seconds = 0.05  # 第一轮 LLM 调用后即过期
    state = loop.run("测试任务")
    assert state.status == "deadline_exceeded"
    assert "截止" in state.error


def test_sanitize_messages_converts_orphan_tool_messages(tmp_path: Path):
    """孤立 tool 消息必须降级为 user——严格端点（DeepSeek）否则直接 400。

    实测触发场景：一轮里多个工具调用时，插入的图片 user 消息会把 tool 结果拆开。
    """
    loop, _ = _make_loop(tmp_path, [])
    loop._messages = [
        {"role": "system", "content": "s"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "结果1"},
        {"role": "tool", "tool_call_id": "c9", "content": "孤立结果"},
    ]
    loop._sanitize_messages()
    assert [m["role"] for m in loop._messages] == ["system", "assistant", "tool", "user"]
    assert loop._messages[-1]["content"] == "孤立结果"

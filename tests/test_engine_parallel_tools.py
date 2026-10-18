# -*- coding: utf-8 -*-
"""只读工具并发执行（P1-4）。

模型一轮里经常同时发多个只读查询（一次查三个数据集）。串行时每个都要等
一次完整的读/查，并发能省下真实等待。但**顺序绝不能乱**：tool_call_id 与
结果的对应关系一旦错配，模型会看到"参数是 A、结果却是 B"。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from gis_cli.engine.llm_client import LLMResponse, ToolCall
from gis_cli.engine.loop import AgentLoop, LoopConfig
from gis_cli.engine.state import Observation
from gis_cli.engine.tools import EngineContext, EngineTool, EngineToolRegistry


def _tc(name, args, cid=None):
    return LLMResponse(
        tool_calls=[
            ToolCall(id=cid or f"c_{name}_{abs(hash(json.dumps(args, sort_keys=True))) % 9999}",
                     name=name, arguments=args, raw_arguments=json.dumps(args))
        ],
        model="mock",
    )


class ScriptedLLM:
    class _Cfg:
        model = "mock"

    config = _Cfg()

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen: list[list[dict]] = []

    def chat(self, messages, **kwargs):
        self.seen.append(list(messages))
        if self.responses:
            return self.responses.pop(0)
        return _tc("finish", {"summary": "done"})


class StubVerifier:
    def verify(self, state):
        return {"pass": True, "summary": "ok"}


def _registry(slow: float = 0.25, readonly: bool = True):
    """造一个带可测工具注册表：slow_read（只读）/ slow_write（写）。"""
    registry = EngineToolRegistry()

    def _read(args, ctx):
        time.sleep(slow)
        return Observation(ok=True, summary=f"read:{args.get('tag')}", data={"tag": args.get("tag")})

    def _write(args, ctx):
        time.sleep(slow)
        return Observation(ok=True, summary=f"write:{args.get('tag')}", data={"tag": args.get("tag")})

    registry.register(
        EngineTool(
            name="slow_read",
            description="只读",
            parameters={"type": "object", "properties": {"tag": {"type": "string"}}},
            handler=_read,
            readonly=readonly,
        )
    )
    registry.register(
        EngineTool(
            name="slow_write",
            description="写",
            parameters={"type": "object", "properties": {"tag": {"type": "string"}}},
            handler=_write,
            readonly=False,
        )
    )
    def _finish(args, ctx):
        # 真实 finish 会把 summary 写回 state（_handle_finish 靠它做 final_summary）
        if ctx.state is not None:
            ctx.state.final_summary = str(args.get("summary") or "")
        return Observation(ok=True, summary=f"准备交付：{args.get('summary')}")

    registry.register(
        EngineTool(
            name="finish",
            description="结束",
            parameters={"type": "object", "properties": {"summary": {"type": "string"}}},
            handler=_finish,
            readonly=False,
        )
    )
    return registry


def _make(tmp_path: Path, responses, *, registry=None, parallel=3):
    llm = ScriptedLLM(responses)
    ctx = EngineContext(workspace=tmp_path)
    loop = AgentLoop(
        workspace=tmp_path,
        llm_client=llm,
        registry=registry or _registry(),
        context=ctx,
        verifier=StubVerifier(),
        config=LoopConfig(max_turns=10, max_parallel_readonly=parallel),
        auto_extract_requirements=False,
    )
    return loop, llm


def _merge(*responses):
    """把多个单工具响应合并成"一轮里多个 tool_call"。"""
    calls = []
    for response in responses:
        calls.extend(response.tool_calls)
    return LLMResponse(tool_calls=calls, model="mock")


# ------------------------------------------------------------------ 核心行为
def test_readonly_tools_run_concurrently(tmp_path: Path):
    """3 个只读工具并发：总耗时应接近 1 个，而不是 3 个。"""
    registry = _registry(slow=0.25)
    loop, _ = _make(
        tmp_path,
        [_merge(_tc("slow_read", {"tag": "a"}), _tc("slow_read", {"tag": "b"}), _tc("slow_read", {"tag": "c"}))],
        registry=registry,
    )
    started = time.monotonic()
    loop._run_actions([
        __import__("gis_cli.engine.tool_codec", fromlist=["Action"]).Action(
            kind="tool", tool="slow_read", args={"tag": tag}
        )
        for tag in ("a", "b", "c")
    ])
    elapsed = time.monotonic() - started
    assert elapsed < 0.25 * 2.5, f"看起来没并发：{elapsed:.2f}s（串行应为 ~0.75s）"


def test_results_keep_original_order(tmp_path: Path):
    """**关键**：结果必须按 actions 原顺序回填（否则 tool_call_id 会错配）。"""
    registry = _registry(slow=0.05)
    loop, _ = _make(tmp_path, [], registry=registry)
    from gis_cli.engine.tool_codec import Action

    actions = [
        Action(kind="tool", tool="slow_read", args={"tag": tag}) for tag in ("a", "b", "c", "d")
    ]
    pairs = loop._run_actions(actions)
    assert [a.args["tag"] for a, _ in pairs] == ["a", "b", "c", "d"]
    assert [o.data["tag"] for _, o in pairs] == ["a", "b", "c", "d"]


def test_write_tool_breaks_the_parallel_batch(tmp_path: Path):
    """写操作不能被并进只读批次（它可能依赖前面的结果，且顺序不能动）。"""
    registry = _registry(slow=0.05)
    loop, _ = _make(tmp_path, [], registry=registry)
    from gis_cli.engine.tool_codec import Action

    actions = [
        Action(kind="tool", tool="slow_read", args={"tag": "a"}),
        Action(kind="tool", tool="slow_write", args={"tag": "w"}),
        Action(kind="tool", tool="slow_read", args={"tag": "b"}),
    ]
    pairs = loop._run_actions(actions)
    assert [a.tool for a, _ in pairs] == ["slow_read", "slow_write", "slow_read"]
    assert [o.summary for _, o in pairs] == ["read:a", "write:w", "read:b"]


def test_parallel_disabled_when_limit_is_one(tmp_path: Path):
    """max_parallel_readonly=1 时退回完全串行（保留旧行为的开关）。"""
    registry = _registry(slow=0.2)
    loop, _ = _make(tmp_path, [], registry=registry, parallel=1)
    from gis_cli.engine.tool_codec import Action

    actions = [Action(kind="tool", tool="slow_read", args={"tag": t}) for t in ("a", "b", "c")]
    started = time.monotonic()
    pairs = loop._run_actions(actions)
    elapsed = time.monotonic() - started
    assert elapsed >= 0.2 * 3 * 0.9, f"应当串行执行，实际 {elapsed:.2f}s"
    assert [o.data["tag"] for _, o in pairs] == ["a", "b", "c"]


def test_non_readonly_tool_not_parallelized(tmp_path: Path):
    """readonly=False 的工具即使连续也不并发。"""
    registry = _registry(slow=0.2, readonly=False)
    loop, _ = _make(tmp_path, [], registry=registry)
    from gis_cli.engine.tool_codec import Action

    actions = [Action(kind="tool", tool="slow_read", args={"tag": t}) for t in ("a", "b")]
    started = time.monotonic()
    loop._run_actions(actions)
    assert time.monotonic() - started >= 0.2 * 2 * 0.9


def test_parse_error_action_yields_failed_observation_and_skips(tmp_path: Path):
    """参数解析失败的动作不执行，但要有失败观察（模型才知道要重发）。"""
    registry = _registry(slow=0.01)
    loop, _ = _make(tmp_path, [], registry=registry)
    from gis_cli.engine.tool_codec import Action

    actions = [
        Action(kind="tool", tool="slow_read", args={}, parse_error="bad json"),
        Action(kind="tool", tool="slow_read", args={"tag": "a"}),
    ]
    pairs = loop._run_actions(actions)
    assert len(pairs) == 2
    assert pairs[0][1].ok is False
    assert "解析失败" in pairs[0][1].summary
    assert pairs[1][1].ok is True


def test_loop_completes_with_parallel_readonly_turn(tmp_path: Path):
    """端到端：一轮里并发两个只读 + 然后 finish，整条链路要正常走完。"""
    loop, _ = _make(
        tmp_path,
        [
            _merge(_tc("slow_read", {"tag": "a"}), _tc("slow_read", {"tag": "b"})),
            _tc("finish", {"summary": "完成"}),
        ],
    )
    state = loop.run("并发只读测试")
    assert state.status == "completed"
    assert state.final_summary == "完成"

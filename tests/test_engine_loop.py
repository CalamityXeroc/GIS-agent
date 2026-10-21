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


def test_loop_resume_after_ask_user_does_not_crash(tmp_path: Path):
    """ask_user 暂停后用 resume() 续跑，不能崩。

    实测 bug：``_continue()`` 里引用了 ``turn_observations``，但那个变量只在
    ``run()`` 的作用域里定义 —— 任何走 resume 的路径都会 NameError。
    而 ask_user 是模型"关键信息缺失时才问"的合法出口，用户答完必然走 resume。
    """
    loop, llm = _make_loop(
        tmp_path,
        [
            _tc("ask_user", {"question": "数据在哪？"}),
            _tc("finish", {"summary": "续跑完成"}),
        ],
    )
    state = loop.run("测试任务")
    assert state.status == "awaiting_user"

    state = loop.resume("数据在 input/ 下")
    assert state.status == "completed", f"续跑失败: {state.status} / {state.error}"
    assert state.final_summary == "续跑完成"


def test_loop_resume_appends_user_answer_and_keeps_messages_valid(tmp_path: Path):
    """续跑必须把用户回答作为 user 消息接进对话，且消息序列保持合法。"""
    loop, llm = _make_loop(
        tmp_path,
        [
            _tc("ask_user", {"question": "坐标系？"}),
            _tc("finish", {"summary": "好"}),
        ],
    )
    loop.run("测试任务")
    loop.resume("CGCS2000")

    roles = [m.get("role") for m in loop._messages]
    assert "user" in roles
    # 用户回答必须真的进了对话内容（否则模型看不到答案）
    assert any("CGCS2000" in str(m.get("content", "")) for m in loop._messages)
    # 消息序列合法：role=tool 不能孤立（严格端点会 400）
    pending: set[str] = set()
    for m in loop._messages:
        if m.get("role") == "assistant":
            pending = {c.get("id") for c in (m.get("tool_calls") or []) if c.get("id")}
        elif m.get("role") == "tool":
            assert m.get("tool_call_id") in pending, "出现了孤立的 tool 消息"


def test_loop_resume_records_observations_of_resumed_turns(tmp_path: Path):
    """续跑轮的工具观察也要进对话（曾经因为变量未定义而整段丢失）。"""
    loop, llm = _make_loop(
        tmp_path,
        [
            _tc("ask_user", {"question": "继续？"}),
            _tc("update_tasklist", {"items": [{"id": "t1", "title": "续跑步骤", "status": "done"}]}),
            _tc("finish", {"summary": "完成"}),
        ],
    )
    loop.run("测试任务")
    state = loop.resume("继续")
    assert state.status == "completed"
    assert any("续跑步骤" in str(m.get("content", "")) for m in loop._messages), \
        "续跑轮的工具结果没进对话"


# --------------------------------------------------------- 断点续跑
def _write_checkpoint(workspace: Path, *, status="llm_unavailable", turn=7, with_messages=True):
    runs = workspace / ".gis_agent" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    state = {
        "run_id": "run_broken",
        "goal": "把气象站清洗并插值",
        "workspace": str(workspace),
        "status": status,
        "turn": turn,
        "requirements": [],
        "tasklist": [{"id": "t1", "title": "解析 CSV", "status": "completed"}],
        "artifacts": [str(workspace / "output" / "result_data.gdb")],
        "methodology": "",
        "final_summary": "",
        "error": "LLM 不可用",
        "created_at": "2026-01-01T00:00:00",
    }
    (runs / "run_broken.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    if with_messages:
        messages = [
            {"role": "system", "content": "你是 GIS 助手"},
            {"role": "user", "content": "把气象站清洗并插值"},
            {"role": "assistant", "content": "先解析 CSV"},
        ]
        (runs / "run_broken.messages.json").write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")


def test_find_checkpoint_requires_messages(tmp_path: Path):
    """没有消息历史的旧断点不能续（续了也没有上下文）。"""
    loop, _ = _make_loop(tmp_path, [])
    _write_checkpoint(tmp_path, with_messages=False)
    assert loop.find_checkpoint() is None

    _write_checkpoint(tmp_path, with_messages=True)
    found = loop.find_checkpoint()
    assert found is not None and found[0].endswith("run_broken.json")


def test_find_checkpoint_skips_completed(tmp_path: Path):
    loop, _ = _make_loop(tmp_path, [])
    _write_checkpoint(tmp_path, status="completed")
    assert loop.find_checkpoint() is None, "已完成的运行不该被当成断点"


def test_resume_from_checkpoint_continues_and_finishes(tmp_path: Path):
    """续跑要把断点里的消息历史接上，而不是从头重来。"""
    _write_checkpoint(tmp_path)
    loop, llm = _make_loop(tmp_path, [_tc("finish", {"summary": "续跑完成"})])

    state = loop.resume_from_checkpoint()

    assert state.status == "completed", state.status
    # 断点里的 turn=7 应当被继承（不是从 0 重新数）
    assert state.turn >= 7, state.turn
    # 旧消息必须还在（模型靠它知道前面做过什么）
    assert any("先解析 CSV" in str(m.get("content")) for m in loop._messages)
    # 续跑会追加一条系统提示，告诉模型别重做
    assert any("从断点恢复" in str(m.get("content")) for m in loop._messages)


def test_output_residue_is_warned_mid_run(tmp_path: Path):
    """output/ 下的下划线临时件要在运行中提醒，不能等最终验收。

    实测 14 届任务二a：模型把 _mp_fix.shp/.dbf 落在 output/ 里，一直到交付
    卫生断言才被抓住——那时 60 轮已用尽，想清也来不及。
    """
    from gis_cli.engine.state import Observation

    loop, _ = _make_loop(tmp_path, [])
    out = tmp_path / "output"
    (out / "成果图").mkdir(parents=True)
    (out / "_mp_fix.shp").write_text("x", encoding="utf-8")
    (out / "成果图" / "_m0.jpg").write_text("x", encoding="utf-8")
    (out / "正常交付.jpg").write_text("x", encoding="utf-8")
    # GDB 内部不该被扫（里面成千上万个文件）
    (out / "result_data.gdb").mkdir()
    (out / "result_data.gdb" / "_a0001.gdbtable").write_text("x", encoding="utf-8")

    obs = Observation(ok=True, summary="执行成功")
    loop._check_workspace_hygiene(obs)

    hint = obs.hint or ""
    assert "_mp_fix.shp" in hint, hint
    assert "成果图/_m0.jpg" in hint, hint
    assert "正常交付.jpg" not in hint, hint
    assert "gdbtable" not in hint, "GDB 内部不该被当残留"

    # 同一批不重复提醒
    obs2 = Observation(ok=True, summary="执行成功")
    loop._check_workspace_hygiene(obs2)
    assert not (obs2.hint or ""), obs2.hint


def test_resume_extends_turn_budget_when_exhausted(tmp_path: Path):
    """撞 max_turns 的断点续跑时必须追加预算，否则续跑等于空转。

    实测 14 届任务二a：断点 turn=60 / max_turns=60，续跑后 while 条件
    直接不成立，只把旧总结又打印了一遍。
    """
    _write_checkpoint(tmp_path, status="max_turns", turn=60)
    loop, _ = _make_loop(tmp_path, [_tc("finish", {"summary": "续跑真的跑了"})])
    loop.config.max_turns = 60

    state = loop.resume_from_checkpoint()

    assert loop.config.max_turns > 60, loop.config.max_turns
    assert state.turn > 60, f"续跑没真正推进：turn={state.turn}"
    assert state.status == "completed", state.status

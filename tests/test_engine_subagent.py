# -*- coding: utf-8 -*-
"""子代理（delegate）用例：不依赖 ArcGIS，全部用 stub LLM + stub runner。

覆盖点：上下文隔离、工具表减配、内核不被重置、子运行不落盘、子代理失败不拖垮父、
派发上限、事件转发、回执截断。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gis_cli.engine import subagent as subagent_mod
from gis_cli.engine.llm_client import LLMResponse, ToolCall
from gis_cli.engine.loop import AgentLoop, LoopConfig
from gis_cli.engine.subagent import ROLE_PREAMBLE, SubAgentRunner, SubtaskResult
from gis_cli.engine.tools import EngineContext, build_default_registry
from gis_cli.runtime.kernel import ExecResult
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

    def usage_summary(self):
        return {"prompt_tokens": 100 * self.calls, "completion_tokens": 10 * self.calls}


class StubVerifier:
    def verify(self, state):
        return {"pass": True, "summary": "验收通过"}


class StubRunner:
    """记录 reset 次数：子代理不能清父代理的内核。"""

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def run(self, code, timeout=None, workspace=None):  # pragma: no cover - 本组用例不跑代码
        raise AssertionError("子代理不该执行代码")


def _tc(name, args):
    return LLMResponse(
        tool_calls=[ToolCall(id=f"c_{name}", name=name, arguments=args, raw_arguments=json.dumps(args))],
        model="mock",
    )


def _parent(tmp_path: Path, responses, *, runner=None, events=None, **kw):
    llm = ScriptedLLM(responses)
    registry = build_default_registry()
    code_runner = kw.pop("code_runner", None)
    context = EngineContext(workspace=tmp_path, code_runner=code_runner)
    loop = AgentLoop(
        workspace=tmp_path,
        llm_client=llm,
        registry=registry,
        context=context,
        code_runner=code_runner,
        verifier=StubVerifier(),
        recorder=RunRecorder("parent_run", tmp_path / "traces"),
        config=LoopConfig(max_turns=10),
        on_event=(lambda k, p: events.append((k, p))) if events is not None else None,
        auto_extract_requirements=False,
        **kw,
    )
    runner = runner or SubAgentRunner(loop)
    context.extra["subagent_factory"] = runner.run
    return loop, llm, runner


def test_delegate_runs_child_and_merges_artifacts(tmp_path: Path):
    """父派子任务 → 子代理在自己的上下文里跑完 → 产物并入父的交付清单。"""
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "sub.md").write_text("子产物", encoding="utf-8")

    loop, llm, _ = _parent(
        tmp_path,
        [
            _tc("delegate", {"task": "把 input/a.csv 按地类汇总并写 output/sub.md", "acceptance": "output/sub.md 存在"}),
            _tc("finish", {"summary": "子代理结论：5 个地类", "methodology": "groupby", "artifacts": ["output/sub.md"]}),
            _tc("finish", {"summary": "父完成"}),
        ],
    )
    state = loop.run("父任务")

    assert state.status == "completed"
    assert state.final_summary == "父完成"
    assert any("sub.md" in a for a in state.artifacts)
    # 子代理的 trace 挂在 .gis_agent/traces 下、名字带父 run 前缀，父自己的消息里不含子的过程
    assert list((tmp_path / ".gis_agent" / "traces").glob("*__sa1.jsonl"))
    assert len(loop._messages) < 10


def test_child_registry_and_config_are_narrowed(tmp_path: Path, monkeypatch):
    captured: dict = {}

    class CapturingLoop(AgentLoop):
        def __init__(self, **kwargs):
            captured["registry"] = kwargs["registry"]
            captured["config"] = kwargs["config"]
            captured["should_stop"] = kwargs.get("should_stop")
            super().__init__(**kwargs)

    monkeypatch.setattr(subagent_mod, "AgentLoop", CapturingLoop)
    loop, _, _ = _parent(
        tmp_path,
        [
            _tc("delegate", {"task": "一个足够长的自包含子任务描述", "max_turns": 12}),
            _tc("finish", {"summary": "子完成"}),
            _tc("finish", {"summary": "父完成"}),
        ],
    )
    loop.run("父任务")

    names = captured["registry"].names()
    assert "delegate" not in names, "子代理不能再派子代理"
    assert "ask_user" not in names, "子代理不能停下来等人"
    assert "execute_code" in names
    cfg = captured["config"]
    assert cfg.role_preamble == ROLE_PREAMBLE
    assert cfg.max_turns == 12
    assert cfg.distill_after_success == "off"
    assert callable(captured["should_stop"])


def test_child_does_not_reset_kernel(tmp_path: Path):
    runner = StubRunner()
    loop, _, _ = _parent(
        tmp_path,
        [
            _tc("delegate", {"task": "一个足够长的自包含子任务描述"}),
            _tc("finish", {"summary": "子完成"}),
            _tc("finish", {"summary": "父完成"}),
        ],
        code_runner=runner,
    )
    loop.run("父任务")
    assert runner.resets == 1, "只有父 run 该 reset 内核，子代理不能清父的变量"


def test_child_does_not_write_run_state(tmp_path: Path):
    loop, _, _ = _parent(
        tmp_path,
        [
            _tc("delegate", {"task": "一个足够长的自包含子任务描述"}),
            _tc("finish", {"summary": "子完成"}),
            _tc("finish", {"summary": "父完成"}),
        ],
    )
    loop.run("父任务")
    runs = sorted(
        p.name
        for p in (tmp_path / ".gis_agent" / "runs").glob("*.json")
        if not p.name.endswith(".messages.json")
    )
    assert len(runs) == 1, f"子运行不该落 runs/：{runs}"
    assert "__sa" not in runs[0]


def test_child_hitting_max_turns_does_not_fail_parent(tmp_path: Path):
    responses = [_tc("delegate", {"task": "一个足够长的自包含子任务描述", "max_turns": 5})]
    # 子代理连做 5 轮工具调用，始终不 finish → 撞 max_turns
    responses += [_tc("update_tasklist", {"items": [{"id": "t1", "title": "x", "status": "running"}]})] * 5
    responses += [_tc("finish", {"summary": "父完成"})]
    loop, _, _ = _parent(tmp_path, responses)
    state = loop.run("父任务")

    assert state.status == "completed"
    obs = [m for m in loop._messages if "子代理未完成" in json.dumps(m, ensure_ascii=False)]
    assert obs, "父应该看到子代理未完成的回执"


def test_subagent_limit(tmp_path: Path):
    loop, llm, runner = _parent(tmp_path, [], runner=None)
    runner = SubAgentRunner(loop, max_per_run=1)
    first = runner.run(task="第一个足够长的子任务描述")
    second = runner.run(task="第二个足够长的子任务描述")
    assert first.status != "refused"
    assert second.status == "refused"
    assert "上限" in second.summary


def test_delegate_reports_disabled_without_factory(tmp_path: Path):
    registry = build_default_registry()
    ctx = EngineContext(workspace=tmp_path)
    obs = registry.execute("delegate", {"task": "一个足够长的自包含子任务描述"}, ctx)
    assert obs.ok is False
    assert "子代理不可用" in obs.summary


def test_should_stop_stops_child(tmp_path: Path):
    llm = ScriptedLLM([_tc("update_tasklist", {"items": []})])
    loop = AgentLoop(
        workspace=tmp_path,
        llm_client=llm,
        registry=build_default_registry(),
        context=EngineContext(workspace=tmp_path),
        verifier=StubVerifier(),
        recorder=RunRecorder("child_stop", tmp_path / "traces"),
        config=LoopConfig(max_turns=5),
        auto_extract_requirements=False,
        should_stop=lambda: True,
    )
    state = loop.run("子任务", reset_runner=False, persist_state=False)
    assert state.status == "stopped"
    assert "父任务已停止" in state.error
    assert llm.calls == 0, "父已停止时子代理不该再调模型"


def test_result_observation_is_truncated():
    result = SubtaskResult(subtask_id="sa1", status="completed", summary="长" * 5000)
    obs = result.to_observation()
    assert obs.ok is True
    assert len(json.dumps(obs.data, ensure_ascii=False)) <= 4000


def test_events_are_forwarded(tmp_path: Path):
    events: list[tuple[str, dict]] = []
    loop, _, _ = _parent(
        tmp_path,
        [
            _tc("delegate", {"task": "一个足够长的自包含子任务描述"}),
            _tc("update_tasklist", {"items": [{"id": "t1", "title": "x", "status": "running"}]}),
            _tc("finish", {"summary": "子完成"}),
            _tc("finish", {"summary": "父完成"}),
        ],
        events=events,
    )
    loop.run("父任务")
    kinds = [k for k, _ in events]
    assert "subagent_start" in kinds
    assert "subagent_tool" in kinds
    assert "subagent_end" in kinds
    end = next(p for k, p in events if k == "subagent_end")
    assert end["status"] == "completed" and end["subtask_id"] == "sa1"
    assert isinstance(end["artifacts"], int)


# ------------------------------------------------------- 补充：隔离/预算/UI
def test_parent_context_state_is_not_repointed(tmp_path: Path):
    """子代理跑完，父的 context.state 必须还是父状态。

    共享 context 的话，子 ``run()`` 会把 ``context.state`` 覆盖成子状态，
    父之后的 execute_code 就会把产物记到子任务上（产物、验收全串）。
    """
    loop, _, _ = _parent(
        tmp_path,
        [
            _tc("delegate", {"task": "一个足够长的自包含子任务描述"}),
            _tc("finish", {"summary": "子完成"}),
            _tc("finish", {"summary": "父完成"}),
        ],
    )
    state = loop.run("父任务")

    assert loop.context.state is state
    assert state.final_summary == "父完成"


def test_child_deadline_is_clamped_by_parent_remaining(tmp_path: Path):
    import time as _time

    loop, _, _ = _parent(tmp_path, [])
    runner = SubAgentRunner(loop, timeout_seconds=900)

    # 父还没设 deadline：用子代理自己的上限
    assert runner._deadline_budget() == 900.0
    # 父只剩 200s：子预算被夹到 ~170s（留 30s 收尾），不能按 900s 跑
    loop._deadline = _time.monotonic() + 200
    clamped = runner._deadline_budget()
    assert 100 < clamped <= 171
    # 父只剩 40s：直接拒派，而不是让子代理把父拖过时限
    loop._deadline = _time.monotonic() + 40
    result = runner.run(task="一个足够长的自包含子任务描述")
    assert result.status == "refused"
    assert "剩余时间不足" in result.summary


def test_registry_subset_keeps_readonly_flags():
    registry = build_default_registry()
    sub = registry.subset(["catalog_query", "finish"])
    assert set(sub.names()) == {"catalog_query", "finish"}
    assert sub.is_readonly("catalog_query") is True
    assert sub.is_readonly("finish") is False
    # 原表不受影响
    assert "execute_code" in registry.names()


def test_subagent_events_render_to_units():
    from gis_cli.ui.events import StatusTracker, to_unit

    start = to_unit("subagent_start", {"subtask_id": "sa1", "task": "出 4 张密度图", "max_turns": 30, "timeout_s": 900})
    assert start is not None and start.kind == "note" and "sa1" in start.text

    step = to_unit("subagent_tool", {"subtask_id": "sa1", "tool": "execute_code", "args": {"description": "算核密度"}})
    assert step is not None and step.kind == "step" and "execute_code" in step.text

    end = to_unit("subagent_end", {"subtask_id": "sa1", "status": "completed", "turns": 17, "elapsed_s": 223, "artifacts": 4})
    assert end is not None and end.kind == "verify" and end.ok is True

    bad = to_unit("subagent_end", {"subtask_id": "sa2", "status": "max_turns", "turns": 30, "elapsed_s": 600, "artifacts": 0})
    assert bad is not None and bad.kind == "warn" and bad.ok is False

    tracker = StatusTracker()
    tracker.apply("turn", {"turn": 7})
    tracker.apply("subagent_turn", {"subtask_id": "sa1", "turn": 3})
    assert "sa1" in tracker.current_action
    assert tracker.turn == 7, "子代理轮次不能覆盖父轮次"
    tracker.apply("subagent_end", {"subtask_id": "sa1", "status": "completed"})
    assert tracker.current_action == ""


def test_narrowed_tools_always_keep_finish(tmp_path: Path, monkeypatch):
    """收窄 tools 时 finish 不能被过滤掉 —— 否则子代理永远交不了差。"""
    captured: dict = {}

    class CapturingLoop(AgentLoop):
        def __init__(self, **kwargs):
            captured["registry"] = kwargs["registry"]
            super().__init__(**kwargs)

    monkeypatch.setattr(subagent_mod, "AgentLoop", CapturingLoop)
    loop, _, _ = _parent(tmp_path, [])
    runner = SubAgentRunner(loop)
    runner.run(task="只做调研，不许执行代码", tools=["catalog_query"])

    names = captured["registry"].names()
    assert "finish" in names, names
    assert "catalog_query" in names
    assert "execute_code" not in names, "tools 收窄应生效"


# ------------------------------------------- 子代理的卡死半径与内核重启信号
def test_child_exec_timeout_and_skills_are_trimmed(tmp_path: Path, monkeypatch):
    """子代理的 execute_code 超时要短于父（600→300），技能目录只给少数几条。"""
    captured: dict = {}

    class CapturingLoop(AgentLoop):
        def __init__(self, **kwargs):
            captured["config"] = kwargs["config"]
            captured["context"] = kwargs["context"]
            super().__init__(**kwargs)

    monkeypatch.setattr(subagent_mod, "AgentLoop", CapturingLoop)
    loop, _, _ = _parent(
        tmp_path,
        [
            _tc("delegate", {"task": "一个足够长的自包含子任务描述"}),
            _tc("finish", {"summary": "子完成"}),
            _tc("finish", {"summary": "父完成"}),
        ],
    )
    loop.config.exec_timeout = 600.0
    runner = SubAgentRunner(loop, exec_timeout_seconds=300)
    loop.context.extra["subagent_factory"] = runner.run
    loop.run("父任务")

    cfg = captured["config"]
    assert cfg.exec_timeout == 300.0, "子代理不能沿用父的 600s：卡死一次要等 10 分钟"
    assert cfg.skills_max == 12, "111 个技能全列占 13k 字符，子代理用不到"
    assert captured["context"].exec_timeout == 300.0, "execute_code 读的是 ctx.exec_timeout"
    # 父自己的配置不被改动
    assert loop.config.exec_timeout == 600.0


def test_kernel_restart_is_reported_back(tmp_path: Path):
    """子代理跑的时候内核被看门狗硬杀重建，父必须收到信号（它的内核变量已丢）。"""

    class SwappingRunner:
        def __init__(self):
            self.resets = 0
            self._kernel = object()

        def reset(self):
            self.resets += 1

        def prewarm(self):
            return False

        def run(self, code, timeout=None, workspace=None):
            self._kernel = object()  # 模拟硬杀后 CodeRunner 重建内核
            return ExecResult(ok=True, result={"ok": True})

        def shutdown(self):
            pass

    loop, _, _ = _parent(
        tmp_path,
        [
            _tc("delegate", {"task": "一个足够长的自包含子任务描述"}),
            _tc("execute_code", {"code": "print(1)", "description": "随便跑一行"}),
            _tc("finish", {"summary": "子完成"}),
            _tc("finish", {"summary": "父完成"}),
        ],
        code_runner=SwappingRunner(),
    )
    state = loop.run("父任务")

    assert state.status == "completed"
    observations = []
    for message in loop._messages:
        if message.get("role") != "tool":
            continue
        try:
            observations.append(json.loads(message.get("content") or "{}"))
        except Exception:
            continue
    restart = [o for o in observations if (o.get("data") or {}).get("kernel_restarted")]
    assert restart, "父必须收到内核被重启的信号"
    assert "内核被看门狗重启过" in (restart[0].get("hint") or "")


def test_model_cannot_raise_exec_timeout_past_child_cap(tmp_path: Path):
    """模型自己传 timeout_seconds=900 时，子代理的 300s 上限仍然生效。"""
    from gis_cli.engine.tools import build_default_registry

    seen: dict = {}

    class TimeoutRunner:
        def __init__(self):
            self._kernel = object()

        def reset(self):
            pass

        def prewarm(self):
            return False

        def run(self, code, timeout=None, workspace=None):
            seen["timeout"] = timeout
            return ExecResult(ok=True, result={})

        def shutdown(self):
            pass

    registry = build_default_registry()
    ctx = EngineContext(workspace=tmp_path, code_runner=TimeoutRunner(), exec_timeout=600.0, max_exec_timeout=300.0)
    obs = registry.execute("execute_code", {"code": "print(1)", "timeout_seconds": 900}, ctx)

    assert obs.ok is True
    assert seen["timeout"] == 300.0, "子代理的 exec 上限不能被模型的 timeout_seconds 顶掉"
    # 父代理没设上限时保持原样（模型要 900 就给 900）
    ctx2 = EngineContext(workspace=tmp_path, code_runner=TimeoutRunner(), exec_timeout=600.0)
    registry.execute("execute_code", {"code": "print(1)", "timeout_seconds": 900}, ctx2)
    assert seen["timeout"] == 900.0

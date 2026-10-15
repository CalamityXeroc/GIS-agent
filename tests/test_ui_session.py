# -*- coding: utf-8 -*-
"""交互会话测试：工作线程、运行中纠偏、Esc 中断、排队与状态同步。

用假 loop 驱动（不需要 LLM/ArcGIS）：假 loop 支持 inject_user_message /
request_stop，并在每一轮之间检查状态，模拟引擎的真实行为。
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from gis_cli.ui.session_runner import EngineSession


class FakeState:
    def __init__(self) -> None:
        self.status = "running"
        self.turn = 0
        self.artifacts: list[str] = []


class FakeLoop:
    """最小可用的假引擎：每 10ms 推进一轮，直到完成/被请求停止。"""

    def __init__(self, *, turns: int = 6, wait_forever: bool = False) -> None:
        self.state = FakeState()
        self.turns = turns
        self.wait_forever = wait_forever
        self.injected: list[str] = []
        self.stop_calls: list[str] = []
        self.started = threading.Event()
        self.finished = threading.Event()

    # 与真引擎同名的方法
    def inject_user_message(self, text: str) -> bool:
        if self.state.status not in ("running", "awaiting_user"):
            return False
        self.injected.append(text)
        return True

    def request_stop(self, reason: str = "") -> bool:
        if self.state.status != "running":
            return False
        self.stop_calls.append(reason)
        self.state.status = "stopped"
        return True

    def run(self, task: str) -> FakeState:
        self.started.set()
        while self.state.status == "running":
            self.state.turn += 1
            if self.wait_forever:
                time.sleep(0.01)
                continue
            if self.state.turn >= self.turns:
                self.state.status = "completed"
        self.finished.set()
        return self.state


class FakeBundle:
    def __init__(self, loop: FakeLoop, workspace: str = "workspace") -> None:
        self.loop = loop
        self.workspace = workspace
        self.llm = type("L", (), {"config": type("C", (), {"model": "m", "api_base": "https://x/v1"})()})()

    def close(self) -> None:
        return None


def _session(loop: FakeLoop) -> EngineSession:
    return EngineSession(FakeBundle(loop))


def test_start_runs_in_worker_thread_and_completes():
    loop = FakeLoop(turns=3)
    session = _session(loop)

    assert session.start("任务") is True
    assert loop.started.wait(2.0)

    state = session.wait(timeout=5.0)
    assert state is loop.state
    assert state.status == "completed"
    assert session.busy is False


def test_busy_is_true_while_engine_runs():
    """运行中 busy 必须为真（输入框据此切提示与快捷键）。"""
    loop = FakeLoop(wait_forever=True)
    session = _session(loop)
    assert session.start("任务") is True
    assert loop.started.wait(2.0)
    try:
        assert session.busy is True
    finally:
        session.interrupt()
        session.wait(timeout=5.0)


def test_start_returns_false_while_busy():
    loop = FakeLoop(wait_forever=True)
    session = _session(loop)
    assert session.start("a") is True
    assert loop.started.wait(2.0)
    try:
        assert session.start("b") is False  # 运行中不接受新任务
    finally:
        session.interrupt()


def test_guide_queues_while_running_without_touching_current_task():
    """运行中的追加输入**排队**，不再静默注入当前任务。

    旧行为是 ``loop.inject_user_message``：消息看起来消失了（没有可见反馈），
    用户会以为没提交成功而重复输入（实测）。改为排队后：立刻有提示，
    本轮结束后合并成同一个任务继续。
    """
    loop = FakeLoop(wait_forever=True)
    session = _session(loop)
    session.start("任务")
    assert loop.started.wait(2.0)
    try:
        assert session.guide("换个方法：先用 Select") is False, "排队返回 False"
        assert loop.injected == [], "不应再注入当前任务"
        assert session.queued_count == 1
        assert "排队" in session.queue_hint()
    finally:
        session.interrupt()
        session.wait(timeout=5.0)


def test_guide_queues_when_not_running():
    loop = FakeLoop(turns=2)
    session = _session(loop)
    assert session.guide("先做投影") is False
    assert session.take_queued() == ["先做投影"]
    assert session.take_queued() == []  # 取走即清空


def test_interrupt_requests_stop_and_ends_run():
    loop = FakeLoop(wait_forever=True)
    session = _session(loop)
    session.start("长任务")
    assert loop.started.wait(2.0)

    assert session.interrupt() is True
    state = session.wait(timeout=5.0)
    assert state.status == "stopped"
    assert loop.stop_calls and "Esc" in loop.stop_calls[0]
    assert session.busy is False


def test_interrupt_without_running_returns_false():
    session = _session(FakeLoop(turns=1))
    assert session.interrupt() is False


def test_events_flow_into_tracker_and_renderer():
    loop = FakeLoop(turns=2)
    session = _session(loop)
    seen: list[tuple[str, dict]] = []
    session._renderer = lambda name, payload: seen.append((name, payload))

    session.drain_events("run_start", {"goal": "g"})
    session.drain_events("tool_call", {"tool": "run_recipe", "args": {}})
    session.drain_events("tool_result", {"tool": "run_recipe", "ok": True, "summary": "ok"})
    session.drain_events("llm_result", {"model": "m", "actions": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    assert session.tracker.tool_count == 1
    assert session.tracker.total_tokens == 15
    assert [name for name, _ in seen][:3] == ["run_start", "tool_call", "tool_result"]


def test_sync_status_from_state_after_run():
    loop = FakeLoop(turns=2)
    session = _session(loop)
    session.start("任务")
    session.wait(timeout=5.0)

    assert session.tracker.status == "running"  # 没有 run_end 事件时不会自己变
    status = session.reset_after_run()
    assert status == "completed"
    assert session.tracker.turn == 2


def test_worker_exception_is_reraised_by_wait():
    class Boom(FakeLoop):
        def run(self, task: str) -> FakeState:  # type: ignore[override]
            raise RuntimeError("engine exploded")

    session = _session(Boom(turns=1))
    session.start("任务")
    with pytest.raises(RuntimeError, match="engine exploded"):
        session.wait(timeout=5.0)


def test_tracker_metadata_is_seeded_from_bundle():
    session = _session(FakeLoop(turns=1))
    assert session.tracker.model == "m"
    assert session.tracker.provider == "https://x/v1"
    assert session.tracker.workspace == "workspace"

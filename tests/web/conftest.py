# -*- coding: utf-8 -*-
"""Web 层公共 fixture：真服务（S1 只读 API）+ 假引擎（S3 运行态）。"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from gis_cli.web import jobs
from gis_cli.web.app import create_app
from gis_cli.web.event_bus import EventBus
from gis_cli.web.security import load_or_create_token


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def token(workspace: Path) -> str:
    return load_or_create_token(workspace)


@pytest.fixture()
def client(workspace: Path, token: str) -> TestClient:
    return TestClient(create_app(workspace, token=token))


@pytest.fixture()
def auth(token: str) -> dict[str, str]:
    return {"X-Web-Token": token}


def wait_until(predicate: Callable[[], bool], timeout: float = 5.0, interval: float = 0.01) -> bool:
    """轮询等待条件成立（测试内部用；不是给生产代码用的 sleep）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class FakeCard:
    """假计划卡：只提供引擎/管理器用到的 to_dict()/render()。"""

    def __init__(self, text: str = "做什么：假任务") -> None:
        self.text = text

    def to_dict(self) -> dict[str, Any]:
        return {"rows": [{"label": "做什么", "value": self.text}], "text": self.text, "goal": self.text}

    def render(self, *, interactive: bool = False) -> str:
        return self.text if not interactive else f"{self.text}\n[s]/Enter 开始  [e] 修改  [q] 取消"


class FakeHandle:
    """假引擎句柄：按脚本发事件，行为可注入（步数/失败/计划卡/阻塞）。"""

    def __init__(
        self,
        runner: "FakeRunner",
        job: jobs.Job,
        emit: Callable[[str, dict[str, Any]], None],
        gate: Any,
    ) -> None:
        self.runner = runner
        self.job = job
        self.emit = emit
        self.gate = gate
        self.stop_flag = threading.Event()
        self.closed = False
        self.decision: Any = None
        self.injected: list[str] = []

    # ---------------------------------------------------------------- 引擎侧
    def start(self) -> dict[str, Any]:
        opts = self.runner.options
        self.runner.enter(self.job.id)
        try:
            self.emit("run_start", {"goal": self.job.goal, "run_id": f"engine_{self.job.id}"})
            self.emit("turn", {"turn": 1, "tasklist": "- [ ] 假步骤"})

            if opts.get("plan"):
                self.emit("plan_card", {"text": "计划：先做 A 再做 B", "round": 1})
                # 与真引擎一致：没装闸门（plan_gate=False）时只发事件，不等决策。
                if self.gate is not None:
                    self.decision = self.gate.ask(FakeCard("计划：先做 A 再做 B"))
                self.emit("plan_decided", {})

            if opts.get("block"):
                self.runner.blocked.wait(10.0)

            for index in range(int(opts.get("steps", 2))):
                if self.stop_flag.is_set():
                    self.emit("stop_requested", {"reason": "用户中断"})
                    return {"status": "stopped", "turn": index}
                tool = f"step_{index}"
                self.emit("tool_call", {"tool": tool, "args": {"index": index}})
                time.sleep(float(opts.get("step_delay", 0.0)))
                self.emit("tool_result", {"tool": tool, "ok": True, "summary": f"{tool} ok"})

            if opts.get("fail"):
                self.emit("run_end", {"status": "failed", "error": "假失败", "turns": 1, "total_tokens": 3})
                return {"status": "failed", "error": "假失败", "turn": 1}

            self.emit("llm_result", {"model": "fake-model", "usage": {"total_tokens": 11}})
            self.emit("verification", {"pass": True, "problems": []})
            self.emit("deliverables", {"artifacts": ["output/fake.txt"]})
            self.emit("run_end", {"status": "completed", "turns": 1, "total_tokens": 11})
            return {"status": "completed", "turn": 1, "total_tokens": 11}
        finally:
            self.runner.leave(self.job.id)

    def stop(self) -> bool:
        self.stop_flag.set()
        return True

    def inject(self, text: str) -> bool:
        self.injected.append(text)
        self.emit("user_injection", {"text": text})
        return True

    def close(self) -> None:
        self.closed = True


class FakeRunner:
    """可断言的假引擎工厂：并发数、执行顺序、每个 job 的句柄都留痕。"""

    def __init__(self, **options: Any) -> None:
        self.options = options
        self.handles: dict[str, FakeHandle] = {}
        self.order: list[str] = []
        self.active = 0
        self.max_concurrent = 0
        self.blocked = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, job: jobs.Job, emit: Any, gate: Any) -> FakeHandle:
        handle = FakeHandle(self, job, emit, gate)
        with self._lock:
            self.handles[job.id] = handle
        return handle

    def enter(self, job_id: str) -> None:
        with self._lock:
            self.order.append(job_id)
            self.active += 1
            self.max_concurrent = max(self.max_concurrent, self.active)

    def leave(self, job_id: str) -> None:
        with self._lock:
            self.active -= 1


@pytest.fixture()
def bus() -> EventBus:
    return EventBus(delta_window_ms=20)


@pytest.fixture()
def fake_runner() -> FakeRunner:
    return FakeRunner(steps=2)


@pytest.fixture()
def manager(workspace: Path, bus: EventBus, fake_runner: FakeRunner):
    manager = jobs.JobManager(workspace, bus=bus, runner=fake_runner, plan_timeout=5.0)
    yield manager
    manager.close()


@pytest.fixture()
def run_client(workspace: Path, token: str, bus: EventBus, manager: jobs.JobManager) -> TestClient:
    """带假引擎的管理器装配出的完整应用（SSE 用）。"""
    app = create_app(workspace, token=token, bus=bus, manager=manager, heartbeat=0.2)
    with TestClient(app) as test_client:
        yield test_client

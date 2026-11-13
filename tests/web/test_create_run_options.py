# -*- coding: utf-8 -*-
"""``POST /api/runs`` 的每-run 选项：max_turns / plan_gate（新建运行页用）。

页面契约（DAIR-247）：
- ``max_turns`` 留空 = 用 ``engine.max_turns``；< 1 或非整数由服务端 422 挡住；
- ``plan_gate=false`` = 计划卡照发但不等确认，直接开始。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gis_cli.web import jobs
from gis_cli.web.event_bus import EventBus

from conftest import FakeRunner


def test_submit_rejects_invalid_max_turns(workspace: Path, bus: EventBus) -> None:
    manager = jobs.JobManager(workspace, bus=bus, runner=FakeRunner(steps=1))
    try:
        with pytest.raises(ValueError):
            manager.submit("目标", max_turns=0)
        with pytest.raises(ValueError):
            manager.submit("目标", max_turns=-3)
        with pytest.raises(ValueError):
            manager.submit("目标", max_turns="三")  # type: ignore[arg-type]
        assert manager.queue_length == 0, "非法 max_turns 不该入队"
    finally:
        manager.close()


def test_submit_carries_max_turns_to_engine_overrides(workspace: Path, bus: EventBus) -> None:
    runner = FakeRunner(steps=1)
    manager = jobs.JobManager(workspace, bus=bus, runner=runner)
    try:
        job = manager.submit("限 3 轮", max_turns=3)
        assert manager.wait_for_idle(10.0)
        assert job.status == jobs.SUCCEEDED
        assert runner.handles[job.id].job.max_turns == 3
        assert jobs.engine_overrides_for(job) == {"max_turns": 3}
        # 留空 = 不覆盖配置
        assert jobs.engine_overrides_for(jobs.Job(id="run_x", goal="g")) is None
    finally:
        manager.close()


def test_plan_gate_off_starts_without_waiting(workspace: Path, bus: EventBus) -> None:
    runner = FakeRunner(steps=1, plan=True)
    manager = jobs.JobManager(workspace, bus=bus, runner=runner, plan_timeout=30.0)
    try:
        job = manager.submit("不等确认", plan_gate=False)
        # 闸门 30s 才超时；这里能跑完说明根本没挂起
        assert manager.wait_for_idle(10.0)
        assert job.status == jobs.SUCCEEDED
        assert job.gate is None
        assert runner.handles[job.id].decision is None
    finally:
        manager.close()


def test_create_run_route_accepts_options(
    run_client: TestClient, auth: dict[str, str]
) -> None:
    response = run_client.post(
        "/api/runs",
        json={"goal": "页面新建", "max_turns": 2, "plan_gate": False},
        headers=auth,
    )
    assert response.status_code == 202, response.text
    run = response.json()["run"]
    assert run["goal"] == "页面新建"
    assert run["status"] == jobs.QUEUED


def test_create_run_route_rejects_zero_max_turns(
    run_client: TestClient, auth: dict[str, str]
) -> None:
    response = run_client.post(
        "/api/runs", json={"goal": "页面新建", "max_turns": 0}, headers=auth
    )
    assert response.status_code == 422
    assert "max_turns" in response.json()["detail"]

# -*- coding: utf-8 -*-
"""JobManager 契约：单 worker 串行、状态机、stop 语义、jobs.json 重启恢复。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gis_cli.web import jobs
from gis_cli.web.event_bus import EventBus

from conftest import FakeRunner, wait_until


def _events(bus: EventBus, job_id: str) -> list:
    return bus.snapshot(job_id)


def test_queue_is_strictly_serial(workspace: Path, bus: EventBus) -> None:
    runner = FakeRunner(steps=2, step_delay=0.02)
    manager = jobs.JobManager(workspace, bus=bus, runner=runner)
    try:
        submitted = [manager.submit(f"任务 {index}") for index in range(3)]
        assert manager.wait_for_idle(15.0), "队列没有在超时内跑完"
        assert runner.order == [job.id for job in submitted], "执行顺序与入队顺序不一致"
        assert runner.max_concurrent == 1, f"出现并发执行: {runner.max_concurrent}"
        assert [job.status for job in submitted] == [jobs.SUCCEEDED] * 3
        assert submitted[0].turns == 1 and submitted[0].tokens == 11
    finally:
        runner.blocked.set()
        manager.close()


def test_state_machine_rejects_illegal_transition() -> None:
    job = jobs.Job(id="run_x", goal="g")
    assert job.status == jobs.QUEUED
    job.transition(jobs.RUNNING)
    job.transition(jobs.AWAITING_PLAN)
    job.transition(jobs.RUNNING)
    job.transition(jobs.SUCCEEDED)
    with pytest.raises(jobs.IllegalTransition):
        job.transition(jobs.RUNNING)
    with pytest.raises(jobs.IllegalTransition):
        jobs.Job(id="run_y", goal="g").transition(jobs.SUCCEEDED)
    with pytest.raises(jobs.IllegalTransition):
        jobs.Job(id="run_z", goal="g").transition(jobs.AWAITING_PLAN)


def test_state_machine_allows_stopping_then_stopped() -> None:
    """running → stopping → stopped 合法；stopping 不是终态，也不能回到 running。"""
    job = jobs.Job(id="run_stop", goal="g")
    job.transition(jobs.RUNNING)
    job.transition(jobs.STOPPING)
    assert job.status == jobs.STOPPING
    assert job.terminal is False
    job.transition(jobs.STOPPED)
    assert job.terminal is True and job.ended_at
    with pytest.raises(jobs.IllegalTransition):
        job.transition(jobs.RUNNING)

    # 计划卡上点停止：awaiting_plan → stopping 也要合法
    plan_job = jobs.Job(id="run_plan_stop", goal="g")
    plan_job.transition(jobs.RUNNING)
    plan_job.transition(jobs.AWAITING_PLAN)
    plan_job.transition(jobs.STOPPING)
    plan_job.transition(jobs.FAILED)


def test_stop_enters_stopping_immediately_and_is_idempotent(
    workspace: Path, bus: EventBus
) -> None:
    """stop 不等引擎返回：立即 stopping + 立即推 stop_requested；重复 stop 幂等。"""
    # 0.2s 一步：stop 返回时当前步还没结束，job 一定还在 stopping 上，重复 stop 可验。
    runner = FakeRunner(steps=200, step_delay=0.2)
    manager = jobs.JobManager(workspace, bus=bus, runner=runner)
    try:
        job = manager.submit("点停止后立刻进 stopping")
        assert wait_until(lambda: job.status == jobs.RUNNING and bool(job.current_tool), 5.0)

        manager.stop(job.id, "用户中断")
        assert job.status == jobs.STOPPING, "stop 没有立即把状态迁到 stopping"
        ring = _events(bus, job.id)
        assert "stop_requested" in [item.event for item in ring], "stop_requested 没有立刻推送"

        # 重复 stop：不抛异常、不重复推事件
        pushed = len([item for item in ring if item.event == "stop_requested"])
        again = manager.stop(job.id, "用户中断")
        assert again.status == jobs.STOPPING
        assert len([i for i in _events(bus, job.id) if i.event == "stop_requested"]) == pushed

        assert wait_until(lambda: job.status in jobs.TERMINAL, 5.0)
        assert job.status == jobs.STOPPED
        assert job.duration_s is not None and job.ended_at
        ring = _events(bus, job.id)
        stop_seq = next(item.seq for item in ring if item.event == "stop_requested")
        assert not [i for i in ring if i.seq > stop_seq and i.event == "tool_call"], (
            "stop 之后还在执行下一步"
        )
    finally:
        manager.close()


def test_stop_prevents_next_step(workspace: Path, bus: EventBus) -> None:
    runner = FakeRunner(steps=200, step_delay=0.01)
    manager = jobs.JobManager(workspace, bus=bus, runner=runner)
    try:
        job = manager.submit("跑到一半停")
        assert wait_until(lambda: len(_events(bus, job.id)) >= 4, 5.0)
        assert job.status == jobs.RUNNING

        manager.stop(job.id, "测试停止")
        assert wait_until(lambda: job.status in jobs.TERMINAL, 5.0)
        assert job.status == jobs.STOPPED
        assert job.stop_requested is True

        ring = _events(bus, job.id)
        evicted = [item.event for item in ring]
        assert "stop_requested" in evicted
        stop_seq = next(item.seq for item in ring if item.event == "stop_requested")
        after = [item for item in ring if item.seq > stop_seq]
        assert not [item for item in after if item.event == "tool_call"], "stop 之后还在执行下一步"
        # 200 步的脚本没有被跑完（停在前面某一步）
        assert len([item for item in ring if item.event == "tool_call"]) < 200
    finally:
        manager.close()


def test_stop_queued_job_is_never_executed(workspace: Path, bus: EventBus) -> None:
    runner = FakeRunner(steps=1, block=True)
    manager = jobs.JobManager(workspace, bus=bus, runner=runner)
    try:
        first = manager.submit("占住 worker")
        assert wait_until(lambda: first.status == jobs.RUNNING, 5.0)
        second = manager.submit("排队中被取消")
        assert second.status == jobs.QUEUED

        manager.stop(second.id)
        assert second.status == jobs.STOPPED

        runner.blocked.set()
        assert manager.wait_for_idle(10.0)
        assert second.id not in runner.order, "排队中被 stop 的 run 仍被执行了"
        assert [item.event for item in _events(bus, second.id)] == ["stop_requested", "run_end"]
    finally:
        runner.blocked.set()
        manager.close()


def test_restart_marks_live_jobs_interrupted(workspace: Path) -> None:
    state_dir = workspace / ".gis_agent"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "jobs.json").write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": [
                    {"id": "run_live", "goal": "跑到一半服务挂了", "status": "running"},
                    {"id": "run_wait", "goal": "还在排队", "status": "queued"},
                    {"id": "run_plan", "goal": "等计划卡", "status": "awaiting_plan"},
                    {"id": "run_done", "goal": "已完成", "status": "succeeded", "turns": 3},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manager = jobs.JobManager(workspace, bus=EventBus(), runner=FakeRunner())
    try:
        assert manager.get("run_live").status == jobs.INTERRUPTED
        assert manager.get("run_wait").status == jobs.INTERRUPTED
        assert manager.get("run_plan").status == jobs.INTERRUPTED
        assert manager.get("run_done").status == jobs.SUCCEEDED
        assert "服务重启" in manager.get("run_live").error

        on_disk = json.loads((state_dir / "jobs.json").read_text(encoding="utf-8"))
        statuses = {item["id"]: item["status"] for item in on_disk["jobs"]}
        assert statuses["run_live"] == jobs.INTERRUPTED
        assert statuses["run_done"] == jobs.SUCCEEDED
    finally:
        manager.close()


def test_shutdown_marks_running_job_interrupted(workspace: Path, bus: EventBus) -> None:
    runner = FakeRunner(steps=1, block=True)
    manager = jobs.JobManager(workspace, bus=bus, runner=runner)
    job = manager.submit("被服务关闭打断")
    assert wait_until(lambda: job.status == jobs.RUNNING, 5.0)
    manager.close()
    assert job.status == jobs.INTERRUPTED
    on_disk = json.loads((workspace / ".gis_agent" / "jobs.json").read_text(encoding="utf-8"))
    assert {item["id"]: item["status"] for item in on_disk["jobs"]}[job.id] == jobs.INTERRUPTED
    runner.blocked.set()


def test_plan_gate_blocks_until_decision(workspace: Path, bus: EventBus) -> None:
    runner = FakeRunner(steps=1, plan=True)
    manager = jobs.JobManager(workspace, bus=bus, runner=runner, plan_timeout=10.0)
    try:
        job = manager.submit("需要确认的计划")
        assert wait_until(lambda: job.status == jobs.AWAITING_PLAN, 5.0)
        assert job.plan is not None and "计划" in str(job.plan.get("text"))
        assert job.plan.get("deadline"), "计划卡没有超时时刻，前端无法提示"

        manager.decide_plan(job.id, "revise", "只做 A，不做 B")
        assert manager.wait_for_idle(10.0)
        assert job.status == jobs.SUCCEEDED
        handle = runner.handles[job.id]
        assert handle.decision is not None
        assert handle.decision.action == "revise"
        assert handle.decision.text == "只做 A，不做 B"
        assert job.plan is None
    finally:
        manager.close()


def test_plan_gate_times_out_and_starts(workspace: Path, bus: EventBus) -> None:
    runner = FakeRunner(steps=1, plan=True)
    manager = jobs.JobManager(workspace, bus=bus, runner=runner, plan_timeout=0.2)
    try:
        job = manager.submit("没人决策的计划")
        assert manager.wait_for_idle(10.0)
        assert job.status == jobs.SUCCEEDED
        assert runner.handles[job.id].decision.action == "start"
        assert "plan_gate_timeout" in [item.event for item in _events(bus, job.id)]
    finally:
        manager.close()


def test_inject_reaches_running_engine(workspace: Path, bus: EventBus) -> None:
    runner = FakeRunner(steps=30, step_delay=0.02)
    manager = jobs.JobManager(workspace, bus=bus, runner=runner)
    try:
        job = manager.submit("插话测试")
        assert wait_until(lambda: job.id in runner.handles and runner.handles[job.id], 5.0)
        assert wait_until(lambda: job.turns >= 1, 5.0)
        manager.inject(job.id, "顺便把坐标系换成 4326")
        assert runner.handles[job.id].injected == ["顺便把坐标系换成 4326"]
        assert "user_injection" in [item.event for item in _events(bus, job.id)]
        manager.stop(job.id)
    finally:
        manager.close()


def test_conflicts_after_finish(workspace: Path, bus: EventBus) -> None:
    runner = FakeRunner(steps=1)
    manager = jobs.JobManager(workspace, bus=bus, runner=runner)
    try:
        job = manager.submit("正常结束")
        assert manager.wait_for_idle(10.0)
        with pytest.raises(jobs.JobConflict):
            manager.stop(job.id)
        with pytest.raises(jobs.JobConflict):
            manager.inject(job.id, "太晚了")
        with pytest.raises(jobs.JobConflict):
            manager.decide_plan(job.id, "start")
        with pytest.raises(jobs.JobConflict):
            manager.resume(job.id)  # 成功的 run 不需要续跑
        with pytest.raises(jobs.JobNotFound):
            manager.get("run_missing")
    finally:
        manager.close()


def test_resume_creates_new_run(workspace: Path, bus: EventBus) -> None:
    """失败的 run 可以续跑：新 job 带 resume_from，原 run 的历史保留。"""
    runner = FakeRunner(steps=1, fail=True)
    manager = jobs.JobManager(workspace, bus=bus, runner=runner)
    try:
        job = manager.submit("会失败的假任务")
        assert manager.wait_for_idle(10.0)
        assert job.status == jobs.FAILED

        # 模拟"网关恢复"：第一次失败，续跑这一次成功
        runner.options["fail"] = False
        resumed = manager.resume(job.id)
        assert resumed.id != job.id
        assert resumed.resume_from == job.id
        assert resumed.status == jobs.QUEUED
        assert manager.wait_for_idle(10.0)
        assert resumed.status == jobs.SUCCEEDED
        assert job.status == jobs.FAILED, "续跑不该改动原 run 的状态"
    finally:
        manager.close()


# ------------------------------------------- 历史 run 只读详情（DAIR-270）


def _write_trace(workspace: Path, run_id: str, events: list[dict]) -> Path:
    trace_dir = workspace / ".gis_agent" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{run_id}.jsonl"
    path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )
    return path


def test_run_detail_falls_back_to_trace(run_client, workspace: Path, auth) -> None:
    """服务重启后 jobs.json 里没有的 run：详情按 trace 回放，标 readonly。"""
    run_id = "run_20260101_101010_12345"
    _write_trace(
        workspace,
        run_id,
        [
            {"seq": 1, "ts": "2026-01-01T02:10:10+00:00", "event": "run_start",
             "payload": {"goal": "统计 input 目录的文件数"}},
            {"seq": 2, "ts": "2026-01-01T02:10:12+00:00", "event": "turn",
             "payload": {"turn": 1}},
            {"seq": 3, "ts": "2026-01-01T02:10:15+00:00", "event": "llm_result",
             "payload": {"model": "fake-model", "usage": {"total_tokens": 1234}}},
            {"seq": 4, "ts": "2026-01-01T02:10:20+00:00", "event": "verification",
             "payload": {"pass": True, "problems": []}},
            {"seq": 5, "ts": "2026-01-01T02:10:22+00:00", "event": "run_end",
             "payload": {"status": "completed", "started_at": "2026-01-01T02:10:10+00:00",
                         "finished_at": "2026-01-01T02:10:22.5+00:00", "turns": 2,
                         "total_tokens": 1234, "duration_seconds": 12.5}},
        ],
    )

    response = run_client.get(f"/api/runs/{run_id}", headers=auth)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["readonly"] is True
    assert body["source"] == "trace"
    assert body["status"] == "succeeded"
    assert body["goal"] == "统计 input 目录的文件数"
    assert (body["turns"], body["tokens"]) == (2, 1234)
    assert body["model"] == "fake-model"
    assert body["verdict"] == "pass"
    assert body["duration_s"] == 12.5
    assert body["trace_path"].endswith(f"{run_id}.jsonl")
    # 只读详情不能带可操作的字段
    assert body["plan"] is None and body["stop_requested"] is False


def test_run_detail_trace_fallback_maps_bad_config_failure(run_client, workspace: Path, auth) -> None:
    """坏配置的 run：llm_unavailable → failed，并把 llm_error 的文案带出来。"""
    run_id = "run_20260101_110000_99999"
    _write_trace(
        workspace,
        run_id,
        [
            {"seq": 1, "ts": "2026-01-01T03:00:00+00:00", "event": "run_start",
             "payload": {"goal": "随便做点事"}},
            {"seq": 2, "ts": "2026-01-01T03:00:03+00:00", "event": "llm_error",
             "payload": {"error": "All models failed. Last error: Error code: 502"}},
            {"seq": 3, "ts": "2026-01-01T03:00:03+00:00", "event": "run_end",
             "payload": {"status": "llm_unavailable", "turns": 1, "total_tokens": 0}},
        ],
    )

    body = run_client.get(f"/api/runs/{run_id}", headers=auth).json()
    assert body["status"] == "failed"
    assert "502" in body["error"]
    assert body["readonly"] is True


def test_run_detail_unknown_is_still_404(run_client, auth) -> None:
    response = run_client.get("/api/runs/run_does_not_exist", headers=auth)
    assert response.status_code == 404


@pytest.mark.parametrize(
    ("engine_status", "expected"),
    [
        ("completed", "succeeded"),
        ("cancelled", "stopped"),
        ("stopped", "stopped"),
        ("failed", "failed"),
        ("deadline_exceeded", "failed"),
        ("max_turns", "failed"),
        ("llm_unavailable", "failed"),
        ("awaiting_user", "interrupted"),
        ("error", "interrupted"),
        ("incomplete", "interrupted"),
    ],
)
def test_trace_fallback_uses_live_status_map(
    run_client, workspace: Path, auth, engine_status: str, expected: str
) -> None:
    """重启后（trace 回退）的徽章必须与 live 详情同一口径（DAIR-271 P2）。

    以前 `_TRACE_STATUS` 是一张独立小表：`cancelled` 显示成「已中断」、`deadline_exceeded`
    /`max_turns` 显示成中断而不是失败，同一个 run 重启前后徽章会变。
    """
    run_id = f"run_status_{engine_status}"
    _write_trace(
        workspace,
        run_id,
        [
            {"seq": 1, "ts": "2026-01-01T03:00:00+00:00", "event": "run_start",
             "payload": {"goal": "状态映射"}},
            {"seq": 2, "ts": "2026-01-01T03:00:05+00:00", "event": "run_end",
             "payload": {"status": engine_status, "turns": 1, "total_tokens": 10}},
        ],
    )
    body = run_client.get(f"/api/runs/{run_id}", headers=auth).json()
    assert body["status"] == expected
    assert body["readonly"] is True


def test_trace_fallback_error_comes_from_run_end(run_client, workspace: Path, auth) -> None:
    """只有 run_end 带 error 的 run（如「多次验收未通过」）详情页不能是空错误。"""
    run_id = "run_only_run_end_error"
    _write_trace(
        workspace,
        run_id,
        [
            {"seq": 1, "ts": "2026-01-01T03:00:00+00:00", "event": "run_start",
             "payload": {"goal": "多轮验收"}},
            {"seq": 2, "ts": "2026-01-01T03:00:30+00:00", "event": "run_end",
             "payload": {"status": "failed", "turns": 4, "total_tokens": 100,
                         "error": "多次验收未通过：缺少 dair270.md"}},
        ],
    )
    body = run_client.get(f"/api/runs/{run_id}", headers=auth).json()
    assert body["status"] == "failed"
    assert "多次验收未通过" in body["error"]


def test_trace_fallback_prefers_run_end_error(run_client, workspace: Path, auth) -> None:
    """run_end 的 error 是收尾结论：优先于中途的 llm_error 文案。"""
    run_id = "run_run_end_over_llm_error"
    _write_trace(
        workspace,
        run_id,
        [
            {"seq": 1, "ts": "2026-01-01T03:00:00+00:00", "event": "run_start",
             "payload": {"goal": "g"}},
            {"seq": 2, "ts": "2026-01-01T03:00:01+00:00", "event": "llm_error",
             "payload": {"error": "中途一次调用失败"}},
            {"seq": 3, "ts": "2026-01-01T03:00:09+00:00", "event": "run_end",
             "payload": {"status": "llm_unavailable", "error": "LLM 不可用: All models failed"}},
        ],
    )
    body = run_client.get(f"/api/runs/{run_id}", headers=auth).json()
    assert body["error"] == "LLM 不可用: All models failed"

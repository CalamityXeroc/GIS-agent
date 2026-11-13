# -*- coding: utf-8 -*-
"""SSE 契约：六类事件到达、断线 since 补发、llm_delta 合并、重启后 trace 回放。"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import wait_until

from gis_cli.web.app import create_app
from gis_cli.web.event_bus import EventBus

SIX_KINDS = {"run_start", "turn", "tool_call", "tool_result", "verification", "run_end"}


def _parse_sse(response) -> list[dict]:
    """把 SSE 帧解析成 [{id, event, data}]；注释帧（: ping）忽略。"""
    frames: list[dict] = []
    event_name: str | None = None
    data: str | None = None
    last_id: int | None = None
    for raw in response.iter_lines():
        line = raw if isinstance(raw, str) else raw.decode("utf-8")
        if not line:
            if event_name is not None and data is not None:
                frames.append({"id": last_id, "event": event_name, "data": json.loads(data)})
            event_name, data = None, None
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data = value
        elif field == "id":
            try:
                last_id = int(value)
            except ValueError:
                last_id = None
    return frames


def _submit(run_client, auth, goal: str) -> str:
    response = run_client.post("/api/runs", json={"goal": goal}, headers=auth)
    assert response.status_code == 202, response.text
    return response.json()["run"]["id"]


def test_sse_streams_six_event_kinds(run_client, auth, manager) -> None:
    run_id = _submit(run_client, auth, "六类事件")
    assert manager.wait_for_idle(10.0)

    with run_client.stream("GET", f"/api/runs/{run_id}/events?since=0", headers=auth) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = _parse_sse(response)

    kinds = {frame["event"] for frame in frames}
    assert SIX_KINDS <= kinds, f"缺事件类型: {SIX_KINDS - kinds}"
    ids = [frame["id"] for frame in frames if frame["event"] not in {"hello"}]
    assert ids == sorted(ids), "seq 不是单调递增"
    assert frames[-1]["event"] == "run_end"
    assert frames[-1]["data"]["payload"]["status"] == "completed"


def test_since_only_replays_missing_segment(run_client, auth, manager, bus: EventBus) -> None:
    run_id = _submit(run_client, auth, "断线补发")
    assert manager.wait_for_idle(10.0)

    turn = next(item for item in bus.snapshot(run_id) if item.event == "turn")
    with run_client.stream(
        "GET", f"/api/runs/{run_id}/events?since={turn.seq}", headers=auth
    ) as response:
        frames = _parse_sse(response)

    replayed = [frame for frame in frames if frame["event"] != "hello"]
    assert replayed, "since 之后没有补发任何事件"
    assert min(frame["id"] for frame in replayed) > turn.seq
    assert "turn" not in {frame["event"] for frame in replayed}
    assert replayed[-1]["event"] == "run_end"


def test_last_event_id_header_is_honoured(run_client, auth, manager, bus: EventBus) -> None:
    run_id = _submit(run_client, auth, "Last-Event-ID")
    assert manager.wait_for_idle(10.0)

    first_tool = next(item for item in bus.snapshot(run_id) if item.event == "tool_call")
    headers = dict(auth)
    headers["Last-Event-ID"] = str(first_tool.seq)
    with run_client.stream("GET", f"/api/runs/{run_id}/events", headers=headers) as response:
        frames = _parse_sse(response)

    replayed = [frame for frame in frames if frame["event"] != "hello"]
    assert replayed and min(frame["id"] for frame in replayed) > first_tool.seq
    assert "run_start" not in {frame["event"] for frame in replayed}


def test_llm_delta_is_coalesced() -> None:
    bus = EventBus(delta_window_ms=50)
    for index in range(30):
        bus.push("run_delta", "llm_delta", {"kind": "content", "text": f"字{index}"})

    merged = wait_until(lambda: len(bus.snapshot("run_delta")) == 1, 3.0)
    assert merged, "合并窗口到期后仍没有事件落地"
    events = bus.snapshot("run_delta")
    assert len(events) < 30, "llm_delta 没有合并"
    assert events[0].payload["merged"] == 30
    assert events[0].payload["text"] == "".join(f"字{index}" for index in range(30))

    # 非 delta 事件会把挂起的增量先冲出去，保证时间顺序
    bus.push("run_delta", "tool_call", {"tool": "x"})
    kinds = [item.event for item in bus.snapshot("run_delta")]
    assert kinds == ["llm_delta", "tool_call"]


def test_trace_replay_after_restart(workspace: Path, token: str, auth) -> None:
    """服务重启（环形缓冲为空）时，SSE 仍能从 trace 回放历史。"""
    trace_dir = workspace / ".gis_agent" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"seq": 1, "ts": "2026-11-10T00:00:00+00:00", "event": "run_start", "payload": {"goal": "历史", "run_id": "run_hist"}},
        {"seq": 2, "ts": "2026-11-10T00:00:01+00:00", "event": "turn", "payload": {"turn": 1}},
        {"seq": 3, "ts": "2026-11-10T00:00:02+00:00", "event": "tool_call", "payload": {"tool": "write_text"}},
        {"seq": 4, "ts": "2026-11-10T00:00:03+00:00", "event": "verification", "payload": {"pass": True}},
        {"seq": 5, "ts": "2026-11-10T00:00:04+00:00", "event": "run_end", "payload": {"status": "completed"}},
    ]
    (trace_dir / "run_hist.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )

    app = create_app(workspace, token=token, heartbeat=0.2)
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        with client.stream("GET", "/api/runs/run_hist/events", headers=auth) as response:
            assert response.status_code == 200
            frames = _parse_sse(response)

    events = [frame["event"] for frame in frames if frame["event"] != "hello"]
    assert events == ["run_start", "turn", "tool_call", "verification", "run_end"]


def test_unknown_run_returns_404(run_client, auth) -> None:
    response = run_client.get("/api/runs/run_missing/events", headers=auth)
    assert response.status_code == 404

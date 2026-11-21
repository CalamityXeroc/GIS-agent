# -*- coding: utf-8 -*-
"""trace 只读 API：坏行/半截文件/seq 排序。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

TRACE_DIR = Path(".gis_agent") / "traces"


def _write_lines(workspace: Path, run_id: str, lines: list[str]) -> Path:
    trace_dir = workspace / TRACE_DIR
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{run_id}.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _event(seq: int, name: str, payload: dict, ts: str = "2026-01-01T00:00:00+00:00") -> str:
    return json.dumps(
        {"seq": seq, "ts": ts, "event": name, "payload": payload}, ensure_ascii=False
    )


def test_list_traces_skips_bad_lines_and_tolerates_partial_run(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    good = [
        _event(1, "turn", {"turn": 1}),
        "这不是 JSON",
        _event(2, "llm_result", {"usage": {"total_tokens": 42}}),
        _event(3, "verification", {"pass": True}),
        _event(
            4,
            "run_end",
            {
                "status": "completed",
                "turns": 1,
                "total_tokens": 42,
                "duration_seconds": 3.2,
                "started_at": "2026-01-01T00:00:00+00:00",
            },
        ),
        '{"seq": 5, "event": "turn", "payload": {"turn": 2}',  # 半行 JSON
    ]
    _write_lines(workspace, "run_good", good)
    _write_lines(
        workspace,
        "run_partial",
        [_event(1, "turn", {"turn": 1}, ts="2026-01-02T00:00:00+00:00")],
    )

    response = client.get("/api/traces", headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    by_id = {item["run_id"]: item for item in body["items"]}

    assert by_id["run_good"]["status"] == "completed"
    assert by_id["run_good"]["verdict"] == "pass"
    assert by_id["run_good"]["turns"] == 1
    assert by_id["run_good"]["tokens"] == 42
    assert by_id["run_good"]["duration_s"] == 3.2

    # 没有 run_end 的半截 run 不是 500，而是如实标 incomplete
    assert by_id["run_partial"]["status"] == "incomplete"
    assert by_id["run_partial"]["turns"] == 1
    assert by_id["run_partial"]["verdict"] is None


def test_fully_truncated_trace_does_not_500(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    _write_lines(
        workspace,
        "run_truncated",
        ['{"seq": 1, "event": "run_start", "payload": {"goal": "x"'],
    )

    listed = client.get("/api/traces", headers=auth)
    assert listed.status_code == 200
    assert listed.json()["items"][0]["status"] == "incomplete"

    detail = client.get("/api/traces/run_truncated", headers=auth)
    assert detail.status_code == 200
    assert detail.json()["events"] == []


def test_trace_detail_orders_events_by_seq(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    _write_lines(
        workspace,
        "run_out_of_order",
        [
            _event(3, "run_end", {"status": "completed", "turns": 1, "total_tokens": 7}),
            _event(1, "turn", {"turn": 1}),
            _event(2, "verification", {"pass": False}),
        ],
    )

    response = client.get("/api/traces/run_out_of_order", headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert [event["seq"] for event in body["events"]] == [1, 2, 3]
    assert body["verdict"] == "fail"


def test_missing_trace_dir_returns_empty_list(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    response = client.get("/api/traces", headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body["count"] == 0
    # S4 起支持分页：total 是筛选后的总数，count 仍是本页条数。
    assert body["total"] == 0


def test_unknown_run_is_404(client: TestClient, auth: dict[str, str]):
    assert client.get("/api/traces/nope", headers=auth).status_code == 404


def _three_runs(workspace: Path) -> None:
    _write_lines(
        workspace,
        "run_pass",
        [
            _event(1, "run_start", {"goal": "人口普查统计"}, ts="2026-11-01T10:00:00+00:00"),
            _event(2, "verification", {"pass": True}, ts="2026-11-01T10:01:00+00:00"),
            _event(
                3,
                "run_end",
                {"status": "completed", "duration_seconds": 60.0},
                ts="2026-11-01T10:02:00+00:00",
            ),
        ],
    )
    _write_lines(
        workspace,
        "run_fail",
        [
            _event(1, "run_start", {"goal": "叠加分析"}, ts="2026-11-05T10:00:00+00:00"),
            _event(2, "verification", {"pass": False, "problems": ["缺少成果"]}, ts="2026-11-05T10:01:00+00:00"),
            _event(
                3,
                "run_end",
                {"status": "completed", "duration_seconds": 90.0},
                ts="2026-11-05T10:02:00+00:00",
            ),
        ],
    )
    _write_lines(
        workspace,
        "run_stopped",
        [
            _event(1, "run_start", {"goal": "投影转换"}, ts="2026-11-08T10:00:00+00:00"),
            _event(2, "stop_requested", {"reason": "用户中断"}, ts="2026-11-08T10:01:00+00:00"),
        ],
    )


def test_list_filters_by_range_status_and_pass(workspace: Path, client: TestClient, auth: dict[str, str]):
    _three_runs(workspace)

    ids = lambda query: [item["run_id"] for item in client.get(f"/api/traces?{query}", headers=auth).json()["items"]]  # noqa: E731

    assert ids("") == ["run_stopped", "run_fail", "run_pass"]
    assert ids("since=2026-11-05") == ["run_stopped", "run_fail"]
    assert ids("until=2026-11-05") == ["run_fail", "run_pass"]
    assert ids("since=2026-11-05&until=2026-11-05") == ["run_fail"]
    assert ids("status=completed") == ["run_fail", "run_pass"]
    assert ids("status=completed,stopped") == ["run_stopped", "run_fail", "run_pass"]
    assert ids("pass=fail") == ["run_fail"]
    assert ids("pass=none") == ["run_stopped"]
    assert ids("q=叠加") == ["run_fail"]
    assert ids("q=RUN_PASS") == ["run_pass"]

    body = client.get("/api/traces?limit=1&offset=1", headers=auth).json()
    assert body["total"] == 3
    assert body["count"] == 1
    assert [item["run_id"] for item in body["items"]] == ["run_fail"]

    assert client.get("/api/traces?pass=maybe", headers=auth).status_code == 422
    assert client.get("/api/traces?limit=0", headers=auth).status_code == 422


def test_download_returns_raw_trace_and_report(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    _three_runs(workspace)
    (workspace / TRACE_DIR / "run_pass.md").write_text("# 报告", encoding="utf-8")

    jsonl = client.get("/api/traces/run_pass/download?kind=jsonl", headers=auth)
    assert jsonl.status_code == 200
    assert jsonl.text.strip().startswith('{"seq": 1')
    assert "attachment" in jsonl.headers["content-disposition"]

    report = client.get("/api/traces/run_pass/download?kind=md", headers=auth)
    assert report.status_code == 200
    assert report.text == "# 报告"

    assert client.get("/api/traces/run_pass/download?kind=zip", headers=auth).status_code == 422
    assert client.get("/api/traces/../secret/download", headers=auth).status_code == 404
    assert client.get("/api/traces/nope/download", headers=auth).status_code == 404

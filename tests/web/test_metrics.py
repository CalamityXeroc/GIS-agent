# -*- coding: utf-8 -*-
"""看板指标口径（S4 验收 1/2/3）。

5 个合成 trace 覆盖：通过 / ``run_end.status=completed`` 但验收没过的陷阱 /
失败 / 停止 / 无 usage 的 mock run；退化输入（空目录、只想 1 个、全坏）；
以及真 trace 上的性能上限。
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gis_cli.web import metrics
from gis_cli.web.trace_api import local_day

TRACE_DIR = Path(".gis_agent") / "traces"
TODAY = date(2026, 11, 10)


def _event(seq: int, name: str, payload: dict, ts: str) -> str:
    return json.dumps({"seq": seq, "ts": ts, "event": name, "payload": payload}, ensure_ascii=False)


def _write(workspace: Path, run_id: str, events: list[str]) -> Path:
    trace_dir = workspace / TRACE_DIR
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{run_id}.jsonl"
    path.write_text("\n".join(events) + "\n", encoding="utf-8")
    return path


def _run(
    workspace: Path,
    run_id: str,
    *,
    ts: str,
    status: str,
    duration: float,
    models: dict[str, tuple[int, int]] | None = None,
    verdict: bool | None = True,
    problems: list[str] | None = None,
    repairs: int = 0,
    tool_failures: int = 0,
    goal: str = "合成任务",
    turns: int = 2,
) -> None:
    """写一个完整 run：``models`` 是 ``{模型: (total_tokens, cached_tokens)}``。"""
    events = [_event(1, "run_start", {"goal": goal, "run_id": run_id}, ts)]
    seq = 2
    for _ in range(turns):
        events.append(_event(seq, "turn", {"turn": seq}, ts))
        seq += 1
    for model, (total, cached) in (models or {}).items():
        usage = {"total_tokens": total, "prompt_tokens": total, "completion_tokens": 0}
        if cached:
            usage["cached_tokens"] = cached
        events.append(_event(seq, "llm_result", {"model": model, "usage": usage}, ts))
        seq += 1
    for _ in range(tool_failures):
        events.append(_event(seq, "tool_result", {"tool": "t", "ok": False, "error": "失败"}, ts))
        seq += 1
    if verdict is not None:
        events.append(
            _event(seq, "verification", {"pass": verdict, "problems": problems or []}, ts)
        )
        seq += 1
    events.append(
        _event(
            seq,
            "run_end",
            {
                "run_id": run_id,
                "status": status,
                "turns": turns,
                "total_tokens": sum(total for total, _ in (models or {}).values()),
                "repairs": repairs,
                "duration_seconds": duration,
                "started_at": ts,
                "goal": goal,
            },
            ts,
        )
    )
    _write(workspace, run_id, events)


@pytest.fixture()
def five_runs(workspace: Path) -> Path:
    _run(
        workspace,
        "run_ok",
        ts="2026-11-08T10:00:00+00:00",
        status="completed",
        duration=100.0,
        models={"model-a": (1000, 400), "model-b": (200, 0)},
        verdict=True,
        repairs=1,
        goal="统计各县人口并出图",
    )
    # 关键样本：状态是 completed，但验收 pass=false —— 必须算失败。
    _run(
        workspace,
        "run_completed_but_failed",
        ts="2026-11-09T11:00:00+00:00",
        status="completed",
        duration=200.0,
        models={"model-a": (500, 0)},
        verdict=False,
        problems=["输出文件不存在：output/result.shp"],
        tool_failures=1,
        goal="裁剪研究区并导出 shp",
    )
    _run(
        workspace,
        "run_failed",
        ts="2026-11-09T12:00:00+00:00",
        status="failed",
        duration=300.0,
        models={"model-c": (300, 0)},
        verdict=None,
        repairs=2,
        goal="批量投影",
    )
    _run(
        workspace,
        "run_stopped",
        ts="2026-11-10T09:00:00+00:00",
        status="stopped",
        duration=400.0,
        models={"model-a": (700, 0)},
        verdict=False,
        problems=["坐标系不一致：EPSG:4326 与 4508"],
        goal="叠加分析",
    )
    _run(
        workspace,
        "run_mock",
        ts="2026-11-10T10:00:00+00:00",
        status="completed",
        duration=50.0,
        models={"mock-model": (0, 0)},
        verdict=True,
        goal="冒烟测试",
    )
    return workspace


def test_kpi_metrics_follow_the_plan(five_runs: Path) -> None:
    result = metrics.summary(five_runs, days=7, today=TODAY)
    kpis = result["kpis"]

    assert kpis["runs"] == 5
    # 通过只认 verification.pass —— completed 但 pass=false 的那条算失败。
    assert kpis["passed"] == 2
    assert kpis["failed"] == 2
    assert kpis["unverified"] == 1
    assert kpis["pass_rate"] == pytest.approx(0.4)

    # 耗时取 run_end.duration_seconds：p50 = 200，p90 = 300 + (400-300)*0.6
    assert kpis["duration_n"] == 5
    assert kpis["duration_p50_s"] == pytest.approx(200.0)
    assert kpis["duration_p90_s"] == pytest.approx(360.0)

    # token 按 llm_result.usage 求和，cached 单列
    assert kpis["tokens_total"] == 2700
    assert kpis["tokens_cached"] == 400
    assert kpis["cache_ratio"] == pytest.approx(0.1481, abs=1e-4)

    # 返工量：repairs 与 tool_result.ok=false 两个计数
    assert kpis["repairs"] == 3
    assert kpis["tool_failures"] == 1


def test_completed_but_failed_run_is_counted_as_failure(five_runs: Path) -> None:
    """单列成一条断言：这是"completed ≠ 过验收"的历史坑，回归时必须仍为失败。"""
    result = metrics.summary(five_runs, days=7, today=TODAY)
    daily = {row["date"]: row for row in result["daily"]}
    assert daily["2026-11-09"]["runs"] == 2
    assert daily["2026-11-09"]["passed"] == 0
    assert daily["2026-11-09"]["failed"] == 1  # completed + pass=false
    assert result["kpis"]["failed"] >= 1


def test_daily_buckets_and_model_split(five_runs: Path) -> None:
    result = metrics.summary(five_runs, days=7, today=TODAY)

    assert [row["date"] for row in result["daily"]] == [
        "2026-11-04",
        "2026-11-05",
        "2026-11-06",
        "2026-11-07",
        "2026-11-08",
        "2026-11-09",
        "2026-11-10",
    ]
    by_date = {row["date"]: row for row in result["daily"]}
    assert by_date["2026-11-04"]["runs"] == 0
    assert by_date["2026-11-08"]["by_model"] == {"model-a": 1000, "model-b": 200}
    assert by_date["2026-11-09"]["by_model"] == {"model-a": 500, "model-c": 300}
    assert by_date["2026-11-10"]["tokens"] == 700

    models = {row["model"]: row for row in result["models"]}
    assert models["model-a"]["tokens"] == 2200
    assert models["model-a"]["cached_tokens"] == 400
    assert models["model-a"]["runs"] == 3
    assert models["model-b"]["tokens"] == 200
    assert [row["model"] for row in result["models"]] == ["model-a", "model-c", "model-b", "mock-model"]


def test_gap_keywords_merge_without_llm(five_runs: Path) -> None:
    result = metrics.summary(five_runs, days=7, today=TODAY)
    gaps = {row["label"]: row["count"] for row in result["gaps"]}
    assert gaps == {"交付物缺失": 1, "坐标系/投影不一致": 1}


def test_window_excludes_old_runs(five_runs: Path) -> None:
    result = metrics.summary(five_runs, days=2, today=TODAY)
    assert result["window"]["since"] == "2026-11-09"
    assert result["kpis"]["runs"] == 4
    assert result["total_runs"] == 5
    assert result["runs_in_window"] == 4


def test_empty_workspace_returns_zeros_and_warning(workspace: Path) -> None:
    result = metrics.summary(workspace, days=7, today=TODAY)

    assert result["kpis"]["runs"] == 0
    assert result["kpis"]["pass_rate"] == 0.0
    assert result["kpis"]["duration_p50_s"] is None
    assert result["kpis"]["tokens_total"] == 0
    assert result["models"] == []
    assert result["gaps"] == []
    assert len(result["daily"]) == 7
    assert all(row["runs"] == 0 for row in result["daily"])
    assert result["warning"], "空目录必须给出 warning，前端才能显示空态原因"
    # 健康条不依赖 trace，空目录下也要有四项
    assert set(result["health"]) == {"kernel", "arcgis", "model", "queue"}


def test_single_trace_is_flagged_as_thin_sample(workspace: Path) -> None:
    _run(
        workspace,
        "run_only",
        ts="2026-11-10T10:00:00+00:00",
        status="completed",
        duration=42.0,
        models={"model-a": (10, 5)},
        verdict=True,
    )

    result = metrics.summary(workspace, days=7, today=TODAY)
    assert result["kpis"]["runs"] == 1
    assert result["kpis"]["pass_rate"] == 1.0
    assert result["kpis"]["duration_p50_s"] == 42.0
    assert result["warning"] and "1 个 run" in result["warning"]


def test_all_corrupt_traces_return_zeros_and_warning(workspace: Path) -> None:
    trace_dir = workspace / TRACE_DIR
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "run_bad1.jsonl").write_text('{"seq": 1, "event": "run_start"\n', encoding="utf-8")
    (trace_dir / "run_bad2.jsonl").write_text("", encoding="utf-8")
    (trace_dir / "run_bad3.jsonl").write_text("不是 JSON\n还是一行垃圾\n", encoding="utf-8")

    result = metrics.summary(workspace, days=7, today=TODAY)
    assert result["kpis"]["runs"] == 0
    assert result["kpis"]["tokens_total"] == 0
    assert result["warning"] and "无法解析" in result["warning"]


def test_cache_hit_avoids_reparse(five_runs: Path) -> None:
    """同一份文件解析两次：第二次不许再读盘（按 mtime+size 命中缓存）。"""
    metrics._CACHE.clear()
    first = metrics.summary(five_runs, days=7, today=TODAY)
    assert len(metrics._CACHE) == 5

    # 把文件内容换掉但保持 mtime/size 不变：命中缓存时结果不变。
    path = five_runs / TRACE_DIR / "run_ok.jsonl"
    stat = path.stat()
    original = path.read_bytes()
    path.write_bytes(b" " * len(original))
    os.utime(path, (stat.st_atime, stat.st_mtime))
    second = metrics.summary(five_runs, days=7, today=TODAY)
    assert second["kpis"] == first["kpis"]

    # 改大小 -> 缓存失效 -> 该 run 变成"不可解析"被剔除
    path.write_text("x", encoding="utf-8")
    third = metrics.summary(five_runs, days=7, today=TODAY)
    assert third["kpis"]["runs"] == 4


def test_local_day_bucketing_matches_local_midnight() -> None:
    """UTC 与本地跨日时，桶必须跟本地走（否则柱子会空一格）。"""
    # 2026-11-10T16:30Z 在 UTC+8 是 11-11 00:30：桶要跟本地走。
    ts = "2026-11-10T16:30:00+00:00"
    assert local_day(ts) == datetime.fromisoformat(ts).astimezone().date().isoformat()
    # 无时区/坏值不抛异常
    assert local_day("2026-11-10") == "2026-11-10"
    assert local_day(None) == ""


def test_summary_endpoint_returns_health_and_window(
    workspace: Path, client: TestClient, auth: dict[str, str]
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    _run(
        workspace,
        "run_api",
        ts=now,
        status="completed",
        duration=12.0,
        models={"model-a": (100, 20)},
        verdict=True,
    )

    response = client.get("/api/metrics/summary?days=7", headers=auth)
    assert response.status_code == 200
    body = response.json()

    assert body["window"]["days"] == 7
    assert body["kpis"]["runs"] == 1
    assert body["kpis"]["tokens_total"] == 100
    assert len(body["daily"]) == 7
    # 分桶是**本地时区**口径（引擎写 UTC）：本地 00:30 的 run 要落在"今天"那格。
    # 这里同时校验窗口总量，避免恰好跨本地午夜时把断言做脆。
    assert body["daily"][-1]["runs"] == 1
    assert sum(row["runs"] for row in body["daily"]) == 1
    assert set(body["health"]) == {"kernel", "arcgis", "model", "queue"}
    assert body["health"]["queue"]["length"] == 0
    assert body["warning"], "单条 run 也要提示样本不足"

    # days 越界直接 422，不要让一个手滑的 ?days=999999 扫全盘
    assert client.get("/api/metrics/summary?days=999", headers=auth).status_code == 422
    # 无 token 仍由统一中间件挡 401
    assert client.get("/api/metrics/summary").status_code == 401


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "workspace" / TRACE_DIR).is_dir(),
    reason="本机没有真实 trace（.gis_agent/ 不入库）",
)
def test_real_traces_aggregate_under_300ms() -> None:
    """验收 3：对现有真 trace 聚合 < 300ms（含首次解析）。"""
    real = Path(__file__).resolve().parents[2] / "workspace"
    metrics._CACHE.clear()

    started = time.perf_counter()
    cold = metrics.summary(real, days=90)
    cold_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    warm = metrics.summary(real, days=90)
    warm_ms = (time.perf_counter() - started) * 1000

    assert warm["total_runs"] >= 1
    assert cold["kpis"]["runs"] == warm["kpis"]["runs"]
    assert cold_ms < 300, f"冷启动聚合 {cold_ms:.1f}ms 超过 300ms"
    assert warm_ms < 300, f"命中缓存聚合 {warm_ms:.1f}ms 超过 300ms"

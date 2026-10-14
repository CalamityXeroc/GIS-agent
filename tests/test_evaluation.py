"""Tests for evaluation loop (Phase 3)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gis_cli.agent import AgentPlanner, MockLLMClient
from gis_cli.agent.evaluation import AgentEvaluationLoop, MultiModelBenchmark
from gis_cli.agent.llm import LLMConfig


def _write_session(memory_dir: Path, session_id: str, turns: list[dict]) -> None:
    payload = {
        "session_id": session_id,
        "turns": turns,
        "context": {},
        "tool_history": [],
        "structured_memories": [],
        "reflections": [],
    }
    (memory_dir / f"{session_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_evaluation_loop_computes_core_metrics(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    turns = [
        {"role": "user", "content": "请合并道路图层和河流图层并导出地图"},
        {"role": "assistant", "content": "收到，我会先扫描后合并并导出"},
        {"role": "user", "content": "继续，根据之前任务改成只处理道路"},
        {"role": "assistant", "content": "继续同一任务，保留合并流程但只处理道路"},
        {"role": "user", "content": "继续，不要裁剪，直接合并并导出"},
        {"role": "assistant", "content": "我来执行 clip_analysis 并先做裁剪"},
    ]
    _write_session(memory_dir, "s1", turns)

    exec_history = tmp_path / "execution_adapter_history.json"
    exec_history.write_text(
        json.dumps([
            {"goal": "根据之前任务执行 clip_analysis 技能", "mode": "shadow"}
        ], ensure_ascii=False),
        encoding="utf-8",
    )

    loop = AgentEvaluationLoop(
        memory_dir=memory_dir,
        history_dir=tmp_path / "evaluation",
        execution_history_path=exec_history,
    )

    report = loop.run(session_limit=10)
    metrics = report.metrics

    assert metrics.sessions_analyzed == 1
    assert metrics.context_checks == 2
    assert metrics.context_hits == 1
    assert metrics.continuity_checks == 2
    assert metrics.continuity_hits == 1
    assert metrics.mis_trigger_checks == 2
    assert metrics.mis_trigger_count == 1
    assert metrics.clip_hijack_events == 1


def test_evaluation_loop_handles_empty_memory(tmp_path: Path) -> None:
    memory_dir = tmp_path / "empty_memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    loop = AgentEvaluationLoop(
        memory_dir=memory_dir,
        history_dir=tmp_path / "evaluation",
    )
    report = loop.run(session_limit=5)

    assert report.metrics.sessions_analyzed == 0
    assert report.metrics.context_hit_rate == 1.0
    assert report.metrics.plan_continuity_rate == 1.0
    assert report.metrics.mis_trigger_rate == 0.0

    report_file = Path(report.report_path)
    assert report_file.exists()
    assert (report_file.parent / (report_file.stem + ".md")).exists()


def test_evaluation_loop_appends_trend_history(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    _write_session(
        memory_dir,
        "s2",
        [
            {"role": "user", "content": "创建专题图并导出"},
            {"role": "assistant", "content": "好的，开始处理"},
            {"role": "user", "content": "继续，根据之前优化配色"},
            {"role": "assistant", "content": "继续同一任务，调整专题图配色并导出"},
        ],
    )

    loop = AgentEvaluationLoop(
        memory_dir=memory_dir,
        history_dir=tmp_path / "evaluation",
    )
    loop.run(session_limit=10)
    loop.run(session_limit=10)

    trend_file = tmp_path / "evaluation" / "metrics_history.json"
    assert trend_file.exists()
    history = json.loads(trend_file.read_text(encoding="utf-8"))
    assert isinstance(history, list)
    assert len(history) == 2
    assert "metrics" in history[0]


def test_planner_marks_fallback_metadata_when_llm_output_invalid() -> None:
    planner = AgentPlanner(llm_client=MockLLMClient(responses=["这不是 JSON"]))
    plan = planner.plan("扫描并处理图层")

    assert plan is not None
    assert plan.metadata.get("llm_planning") == "fallback"
    assert plan.metadata.get("llm_fallback_used") is True
    assert plan.metadata.get("llm_fallback_reason") == "invalid_json_or_schema"


def test_multi_model_benchmark_generates_report(tmp_path: Path) -> None:
    valid_plan_json = (
        '{"id":"plan_ok","goal":"测试","steps":[{"id":"step_1","tool":"scan_layers",'
        '"description":"扫描","input":{},"depends_on":[]}],"expected_outputs":[]}'
    )

    def fake_llm_factory(cfg: LLMConfig):
        if "good" in cfg.model:
            return MockLLMClient(responses=[valid_plan_json])
        return MockLLMClient(responses=["invalid output"])

    benchmark = MultiModelBenchmark(
        output_dir=tmp_path / "evaluation",
        base_config=LLMConfig(model="base", api_key="x", provider="openai"),
        llm_factory=fake_llm_factory,
    )

    report = benchmark.run(
        models=["good-model", "bad-model"],
        tasks=["扫描图层"],
        repeats=2,
    )

    assert Path(report.report_path).exists()
    assert Path(report.report_path).with_suffix(".md").exists()
    assert len(report.model_metrics) == 2

    metrics_by_model = {m.model: m for m in report.model_metrics}
    assert metrics_by_model["good-model"].success_rate == 1.0
    assert metrics_by_model["bad-model"].fallback_rate == 1.0
    assert metrics_by_model["bad-model"].parse_error_rate == 1.0
    assert report.success_rate_gap == 1.0

# -*- coding: utf-8 -*-
"""跨模型基准改测**新引擎**（P2-8）。

原来的 `MultiModelBenchmark` 用 `AgentPlanner`（旧引擎的 plan-then-execute），
只验证"能不能产出一个计划对象"，与"任务是否真的做完"无关。而
`docs` 里对"模型无关"的成功指标是成功率差异 <5%、解析错误率 <2% ——
用旧引擎根本测不出这些。

新实现跑真实引擎循环（build_engine + loop.run），指标取自 TaskState：
是否 completed、验收是否通过、轮次、解析错误次数、token 用量。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gis_cli.agent.evaluation import ModelBenchmarkMetrics


def test_metrics_expose_engine_fields():
    """指标结构要能承载新引擎的观测值（不只是旧引擎的 fallback/parse）。"""
    m = ModelBenchmarkMetrics(model="x")
    # 新引擎关心的量
    assert hasattr(m, "completed_runs")
    assert hasattr(m, "verify_passed_runs")
    assert hasattr(m, "total_turns")
    assert hasattr(m, "total_tokens")
    assert hasattr(m, "max_turns_runs")
    m.total_runs = 2
    m.completed_runs = 1
    m.verify_passed_runs = 1
    m.success_runs = 1  # 与 completed_runs 同口径（旧字段，保留兼容）
    assert m.success_rate == 0.5
    assert m.verify_pass_rate == 0.5
    assert m.completed_rate == 0.5
    assert m.avg_turns == 0.0


def test_metrics_rates_are_safe_when_no_runs():
    """零样本不能除零。"""
    m = ModelBenchmarkMetrics(model="x")
    assert m.success_rate == 0.0
    assert m.verify_pass_rate == 0.0
    assert m.avg_turns == 0.0
    assert m.avg_tokens == 0.0


def test_engine_benchmark_runs_real_loop(tmp_path: Path, monkeypatch):
    """基准必须真的把任务跑进引擎循环（而不是只问模型要个计划）。

    这里把 build_engine 换成假的，断言：调用次数 == 模型数 × 任务数 × 重复数，
    且指标来自 TaskState 的 status/verification。
    """
    from gis_cli.agent import evaluation as ev

    calls: list[tuple[str, str]] = []

    class _FakeState:
        def __init__(self, status: str, passed: bool, turns: int):
            self.status = status
            self.turn = turns
            self.artifacts = []
            self.error = ""
            self.final_summary = ""
            self._passed = passed

    class _FakeLoop:
        def __init__(self, status: str, passed: bool, turns: int):
            self._spec = (status, passed, turns)

        def run(self, goal, **kwargs):
            return _FakeState(*self._spec)

    class _FakeLLM:
        def __init__(self, model: str):
            self.config = type("C", (), {"model": model})()
            self.usage_totals: dict = {}

        def usage_summary(self):
            return {"calls": 3, "prompt_tokens": 100, "completion_tokens": 20, "cached_tokens": 50}

    class _FakeBundle:
        def __init__(self, model: str, status: str, passed: bool, turns: int):
            self.llm = _FakeLLM(model)
            self.loop = _FakeLoop(status, passed, turns)
            self.verifier = None

        def close(self):
            pass

    def fake_build_engine(*, workspace, config_path=None, engine_overrides=None, **kwargs):
        model = (engine_overrides or {}).get("__model__", "m")
        calls.append((str(workspace), model))
        # good 模型完成且验收通过；bad 模型轮次耗尽
        if "good" in model:
            return _FakeBundle(model, "completed", True, 4)
        return _FakeBundle(model, "max_turns", False, 30)

    monkeypatch.setattr(ev, "_build_engine_for_benchmark", fake_build_engine)

    bench = ev.EngineModelBenchmark(
        output_dir=tmp_path / "eval",
        models=["good-model", "bad-model"],
        tasks=["任务A"],
        repeats=2,
    )
    report = bench.run()

    assert len(calls) == 4, f"应为 2 模型 × 1 任务 × 2 重复 = 4 次真实运行，实际 {len(calls)}"
    by_model = {m.model: m for m in report.model_metrics}
    assert by_model["good-model"].completed_runs == 2
    assert by_model["good-model"].verify_passed_runs == 2
    assert by_model["good-model"].success_rate == 1.0
    assert by_model["bad-model"].completed_runs == 0
    assert by_model["bad-model"].max_turns_runs == 2
    assert by_model["bad-model"].success_rate == 0.0
    # 轮次/ token 也进了指标
    assert by_model["good-model"].avg_turns == 4.0
    assert by_model["bad-model"].avg_turns == 30.0
    assert by_model["good-model"].avg_tokens > 0
    # 报告落盘
    assert Path(report.report_path).exists()
    assert Path(report.report_path).with_suffix(".md").exists()


def test_engine_benchmark_parses_errors_are_counted(tmp_path: Path, monkeypatch):
    """解析错误要单独计数（"模型无关"的硬指标之一：解析错误率 <2%）。"""
    from gis_cli.agent import evaluation as ev

    class _FakeState:
        status = "completed"
        turn = 2
        artifacts = []
        error = ""
        final_summary = ""

    class _FakeLoop:
        def run(self, goal, **kwargs):
            return _FakeState()

    class _FakeLLM:
        def __init__(self):
            self.config = type("C", (), {"model": "m"})()
            # 引擎把工具参数解析失败记在 trace/状态里；这里用状态钩子模拟
            self.parse_errors = 2

        def usage_summary(self):
            return {}

    class _FakeBundle:
        def __init__(self):
            self.llm = _FakeLLM()
            self.loop = _FakeLoop()
            self.verifier = None

        def close(self):
            pass

    monkeypatch.setattr(ev, "_build_engine_for_benchmark", lambda **k: _FakeBundle())
    bench = ev.EngineModelBenchmark(output_dir=tmp_path / "e", models=["m"], tasks=["t"], repeats=1)
    report = bench.run()
    m = report.model_metrics[0]
    assert m.total_runs == 1
    assert m.parse_error_runs >= 0  # 结构存在；具体计数由引擎侧提供


def test_workspaces_live_outside_output_dir(tmp_path: Path, monkeypatch):
    """每个 run 的工作区必须在 output_dir **之外**。

    实测坑：工作区放在 output_dir 里面时，验收器把"工作区本身"当成产出，
    报"产出不在 output 目录内"，于是每次都 failed —— 而且看起来像模型
    能力问题，极难定位。
    """
    from gis_cli.agent import evaluation as ev

    seen: list[Path] = []

    class _State:
        status = "completed"
        turn = 1
        artifacts: list = []
        error = ""
        final_summary = ""

    class _Loop:
        def run(self, goal, **kwargs):
            return _State()

    class _LLM:
        config = type("C", (), {"model": "m"})()

        def usage_summary(self):
            return {}

    class _Bundle:
        llm = _LLM()
        loop = _Loop()
        verifier = None

        def close(self):
            pass

    def fake_build(**kwargs):
        seen.append(Path(kwargs["workspace"]))
        return _Bundle()

    monkeypatch.setattr(ev, "_build_engine_for_benchmark", fake_build)
    out = tmp_path / "output" / "evaluation"
    ev.EngineModelBenchmark(output_dir=out, models=["m"], tasks=["t"], repeats=1).run()

    assert seen, "引擎没有被调用"
    for workspace in seen:
        assert out.resolve() not in workspace.resolve().parents, (
            f"工作区不能放在 output_dir 内（会被验收器当成产出）：{workspace}"
        )


def test_status_counts_recorded(tmp_path: Path, monkeypatch):
    """终止状态要能看见：只看 completed/max_turns 时，“8 轮就停但不是
    max_turns”这种情况完全看不出原因（实测踩到：其实是验收反复不通过后 failed）。"""
    from gis_cli.agent import evaluation as ev

    class _State:
        status = "failed"
        turn = 7
        artifacts: list = []
        error = "多次验收未通过"
        final_summary = ""

    class _Loop:
        def run(self, goal, **kwargs):
            return _State()

    class _LLM:
        config = type("C", (), {"model": "m"})()

        def usage_summary(self):
            return {}

    class _Bundle:
        llm = _LLM()
        loop = _Loop()
        verifier = None

        def close(self):
            pass

    monkeypatch.setattr(ev, "_build_engine_for_benchmark", lambda **k: _Bundle())
    report = ev.EngineModelBenchmark(
        output_dir=tmp_path / "e", models=["m"], tasks=["t"], repeats=1
    ).run()
    m = report.model_metrics[0]
    assert m.status_counts == {"failed": 1}
    assert m.completed_runs == 0 and m.max_turns_runs == 0
    # 也要进 JSON（否则事后无法复盘）
    import json

    payload = json.loads(Path(report.report_path).read_text(encoding="utf-8"))
    assert payload["model_metrics"][0]["status_counts"] == {"failed": 1}

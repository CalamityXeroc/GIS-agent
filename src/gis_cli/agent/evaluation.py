"""跨模型基准：用**真实引擎循环**比较不同模型做同一个任务的表现。

成功判据是"任务是否真的做完"（验收是否通过），不是"能不能产出一个计划
对象"。早期基于旧引擎 plan-then-execute 的评估类（EvaluationMetrics /
AgentEvaluationLoop / MultiModelBenchmark）已随旧引擎一起移除。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable



@dataclass
class ModelBenchmarkMetrics:
    model: str
    total_runs: int = 0
    success_runs: int = 0
    fallback_runs: int = 0
    parse_error_runs: int = 0
    total_latency_ms: float = 0.0
    prompt_optimizer_hits: int = 0
    baml_hits: int = 0
    # ---- 新引擎指标（跑真实任务循环，而不是只问模型要个计划）----
    #: 状态为 completed 的次数（引擎认为做完了）
    completed_runs: int = 0
    #: 验收真正通过的次数（比 completed 严：完成不等于过验收）
    verify_passed_runs: int = 0
    #: 轮次耗尽次数（模型没做完就撞上限）
    max_turns_runs: int = 0
    #: 模型不可用次数（网关/密钥问题，与模型能力无关，单列）
    llm_unavailable_runs: int = 0
    total_turns: int = 0
    total_tokens: int = 0
    #: 每次 run 的终止状态分布（``{status: 次数}``）。
    #:
    #: 为什么要记：只看 completed/max_turns 两个计数器时，“8 轮就停了但不是
    #: max_turns”这种情况完全看不出原因（实测踩到 —— 其实是验收反复不通过
    #: 后 failed）。有了分布就能直接看到 status=failed 而不是靠猜。
    status_counts: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.total_runs <= 0:
            return 0.0
        return self.success_runs / self.total_runs

    @property
    def verify_pass_rate(self) -> float:
        """验收通过率（新引擎的主指标）。"""
        if self.total_runs <= 0:
            return 0.0
        return self.verify_passed_runs / self.total_runs

    @property
    def completed_rate(self) -> float:
        if self.total_runs <= 0:
            return 0.0
        return self.completed_runs / self.total_runs

    @property
    def avg_turns(self) -> float:
        if self.total_runs <= 0:
            return 0.0
        return self.total_turns / self.total_runs

    @property
    def avg_tokens(self) -> float:
        if self.total_runs <= 0:
            return 0.0
        return self.total_tokens / self.total_runs

    @property
    def fallback_rate(self) -> float:
        if self.total_runs <= 0:
            return 0.0
        return self.fallback_runs / self.total_runs

    @property
    def parse_error_rate(self) -> float:
        if self.total_runs <= 0:
            return 0.0
        return self.parse_error_runs / self.total_runs

    @property
    def avg_latency_ms(self) -> float:
        if self.total_runs <= 0:
            return 0.0
        return self.total_latency_ms / self.total_runs

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "total_runs": self.total_runs,
            "success_runs": self.success_runs,
            "fallback_runs": self.fallback_runs,
            "parse_error_runs": self.parse_error_runs,
            "success_rate": round(self.success_rate, 4),
            "fallback_rate": round(self.fallback_rate, 4),
            "parse_error_rate": round(self.parse_error_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "prompt_optimizer_hits": self.prompt_optimizer_hits,
            "baml_hits": self.baml_hits,
            "completed_runs": self.completed_runs,
            "verify_passed_runs": self.verify_passed_runs,
            "max_turns_runs": self.max_turns_runs,
            "llm_unavailable_runs": self.llm_unavailable_runs,
            "verify_pass_rate": round(self.verify_pass_rate, 4),
            "completed_rate": round(self.completed_rate, 4),
            "avg_turns": round(self.avg_turns, 2),
            "avg_tokens": round(self.avg_tokens, 1),
            "status_counts": dict(self.status_counts),
        }


@dataclass
class MultiModelBenchmarkReport:
    report_id: str
    generated_at: str
    tasks: list[str]
    repeats: int
    model_metrics: list[ModelBenchmarkMetrics] = field(default_factory=list)
    report_path: str = ""

    @property
    def success_rate_gap(self) -> float:
        if len(self.model_metrics) <= 1:
            return 0.0
        rates = [m.success_rate for m in self.model_metrics]
        return max(rates) - min(rates)

    @property
    def max_parse_error_rate(self) -> float:
        if not self.model_metrics:
            return 0.0
        return max(m.parse_error_rate for m in self.model_metrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "tasks": self.tasks,
            "repeats": self.repeats,
            "success_rate_gap": round(self.success_rate_gap, 4),
            "max_parse_error_rate": round(self.max_parse_error_rate, 4),
            "model_metrics": [m.to_dict() for m in self.model_metrics],
            "report_path": self.report_path,
        }


def _build_engine_for_benchmark(
    *,
    workspace: Any,
    config_path: Any = None,
    engine_overrides: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """建引擎（单独抽出来，方便测试替换）。

    为什么不直接在类里调 build_engine：benchmark 要用假引擎做单测，
    而 build_engine 会拉 sqlite/内核/配方库，不适合在每个单测里真建。
    """
    from ..engine.runner import build_engine

    return build_engine(
        workspace=workspace,
        config_path=config_path,
        engine_overrides=engine_overrides,
        **kwargs,
    )


class EngineModelBenchmark:
    """跨模型基准：跑**真实引擎循环**，测"任务是否真的做完"。

    与旧的 :class:`MultiModelBenchmark` 的区别：

    ==================  ==========================  ========================
                        旧（MultiModelBenchmark）    新（本类）
    ==================  ==========================  ========================
    跑的什么             AgentPlanner.plan()         完整 loop.run() 循环
    成功判据             "产出了一个非空计划"          status=completed 且验收通过
    能否反映真实能力     不能（计划对≠做得对）          能
    指标                  fallback / parse error        验收通过率 / 轮次 / token
    ==================  ==========================  ========================

    为什么要改：`docs` 里对"模型无关"的成功指标是**成功率差异 <5%**、
    **解析错误率 <2%**。用旧实现根本测不出这些 —— 它只验证"能不能吐出一段
    合法 JSON"，而真正会翻车的是多轮工具调用、代码自修复、验收收敛。
    """

    def __init__(
        self,
        *,
        output_dir: Path | str,
        models: list[str],
        tasks: list[str],
        repeats: int = 1,
        config_path: Path | str | None = None,
        engine_overrides: dict[str, Any] | None = None,
        max_turns: int = 30,
        run_deadline_seconds: float = 1800.0,
    ):
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "model_benchmark"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        #: 每个 run 的独立工作区放在 **output_dir 之外**。
        #:
        #: 坑：放到 output_dir 里面时，验收器会把"工作区本身"当成产出，
        #: 报“产出不在 output 目录内”，于是每次都 failed（实测）。
        self.workspaces_dir = self.output_dir.parent / "benchmark_workspaces"
        self.models = [str(m).strip() for m in models if str(m).strip()]
        self.tasks = [str(t) for t in tasks if str(t).strip()]
        self.repeats = max(1, int(repeats))
        self.config_path = config_path
        self.engine_overrides = dict(engine_overrides or {})
        self.max_turns = max(1, int(max_turns))
        self.run_deadline_seconds = float(run_deadline_seconds)

    # ------------------------------------------------------------------ run
    def run(self) -> MultiModelBenchmarkReport:
        if not self.models:
            raise ValueError("models cannot be empty")
        if not self.tasks:
            raise ValueError("tasks cannot be empty")

        metrics = [self._run_model(model) for model in self.models]
        report = MultiModelBenchmarkReport(
            report_id=f"engine_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(timezone.utc).isoformat(),
            tasks=list(self.tasks),
            repeats=self.repeats,
            model_metrics=metrics,
        )
        json_path = self.reports_dir / f"{report.report_id}.json"
        json_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        md_path = self.reports_dir / f"{report.report_id}.md"
        md_path.write_text(self._render_markdown(report), encoding="utf-8")
        report.report_path = str(json_path)
        return report

    def _run_model(self, model: str) -> ModelBenchmarkMetrics:
        metrics = ModelBenchmarkMetrics(model=model)
        for _ in range(self.repeats):
            for index, task in enumerate(self.tasks):
                # 每个 (模型, 任务, 重复) 一个独立工作区：避免跨任务状态串味，
                # 也让"产物/目录/日志"互不干扰（基准公平性的基本要求）。
                workspace = (
                    self.workspaces_dir
                    / _safe_name(model)
                    / f"task{index + 1}_{_safe_name(task)[:40]}"
                    / f"rep{_}"
                )
                workspace.mkdir(parents=True, exist_ok=True)
                overrides = dict(self.engine_overrides)
                overrides["__model__"] = model
                overrides.setdefault("run_deadline_seconds", self.run_deadline_seconds)
                bundle = None
                started = time.perf_counter()
                try:
                    bundle = _build_engine_for_benchmark(
                        workspace=workspace,
                        config_path=self.config_path,
                        engine_overrides=overrides,
                    )
                    _apply_model(bundle, model)
                    _apply_max_turns(bundle, self.max_turns)
                    state = bundle.loop.run(task)
                except Exception as exc:  # pragma: no cover - 引擎自身异常
                    logger.warning("benchmark run crashed (%s / %s): %s", model, task, exc)
                    metrics.total_runs += 1
                    metrics.parse_error_runs += 0  # 崩溃不算解析错误
                    continue
                finally:
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    if bundle is not None:
                        # token 用量要在 close 之前取（close 可能清掉客户端）
                        tokens = _usage_tokens(bundle)
                        try:
                            bundle.close()
                        except Exception:
                            pass
                    else:
                        tokens = 0
                self._record(metrics, state, latency_ms, tokens)
        return metrics

    def _record(
        self,
        metrics: ModelBenchmarkMetrics,
        state: Any,
        latency_ms: float,
        tokens: int = 0,
    ) -> None:
        metrics.total_runs += 1
        metrics.total_latency_ms += latency_ms
        metrics.total_tokens += max(0, int(tokens or 0))
        status = str(getattr(state, "status", "") or "")
        metrics.status_counts[status] = metrics.status_counts.get(status, 0) + 1
        metrics.total_turns += int(getattr(state, "turn", 0) or 0)
        if status == "completed":
            metrics.completed_runs += 1
            metrics.success_runs += 1
            # 完成 ≠ 过验收：以最终是否被验收接受为准（新引擎的硬判据）
            if _verify_passed(state):
                metrics.verify_passed_runs += 1
        elif status == "max_turns":
            metrics.max_turns_runs += 1
        elif status == "llm_unavailable":
            metrics.llm_unavailable_runs += 1
        # 工具参数解析失败：引擎把它记成一次失败观察，状态里带 ArgumentParseError
        if _has_parse_error(state):
            metrics.parse_error_runs += 1

    # ------------------------------------------------------------- reporting
    def _render_markdown(self, report: MultiModelBenchmarkReport) -> str:
        lines = [
            f"# 跨模型基准（新引擎）: {report.report_id}",
            "",
            f"- 生成时间: {report.generated_at}",
            f"- 重复次数: {report.repeats}",
            f"- 成功率差异: {report.success_rate_gap:.2%}",
            f"- 最大解析错误率: {report.max_parse_error_rate:.2%}",
            "",
            "## 任务",
            "",
        ]
        lines.extend(f"- {task}" for task in report.tasks)
        lines.extend(
            [
                "",
                "## 指标",
                "",
                "| 模型 | 次数 | 验收通过率 | 完成率 | 平均轮次 | 平均 token | 解析错误率 | 平均耗时 |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for m in report.model_metrics:
            lines.append(
                f"| {m.model} | {m.total_runs} | {m.verify_pass_rate:.1%} | "
                f"{m.completed_rate:.1%} | {m.avg_turns:.1f} | {m.avg_tokens:.0f} | "
                f"{m.parse_error_rate:.1%} | {m.avg_latency_ms / 1000:.1f}s |"
            )
        lines.append("")
        lines.append(
            "> 验收通过率是新引擎的主指标：`completed` 只表示引擎认为做完了，"
            "真正的判据是验收器接受了产出。"
        )
        return "\n".join(lines)


def _safe_name(text: str) -> str:
    """把模型名/任务名变成安全的目录名。"""
    cleaned = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff._-]+", "_", str(text or ""))
    return cleaned.strip("_")[:60] or "x"


def _apply_model(bundle: Any, model: str) -> None:
    """把 bundle 的 LLM 客户端切到指定模型（基准的核心动作）。"""
    llm = getattr(bundle, "llm", None)
    config = getattr(llm, "config", None)
    if config is None:
        return
    try:
        config.model = model
        # 基准要测单模型能力，不能让 fallback 掩盖问题
        config.fallback_models = []
        config.routing_rules = {}
    except Exception:  # pragma: no cover - 配置对象不可变时忽略
        pass


def _apply_max_turns(bundle: Any, max_turns: int) -> None:
    loop = getattr(bundle, "loop", None)
    config = getattr(loop, "config", None)
    if config is not None:
        try:
            config.max_turns = max_turns
        except Exception:  # pragma: no cover
            pass


def _verify_passed(state: Any) -> bool:
    """从 TaskState 判断验收是否通过。

    引擎在 finish 时把验收结论写进 status/error；completed 且 error 为空
    表示验收接受了产出（被拒的话 status 会是 failed 或继续跑）。
    """
    if str(getattr(state, "status", "")) != "completed":
        return False
    return not str(getattr(state, "error", "") or "").strip()


def _has_parse_error(state: Any) -> bool:
    """状态里是否出现过工具参数解析失败。"""
    text = str(getattr(state, "error", "") or "")
    if "ArgumentParseError" in text or "参数解析失败" in text:
        return True
    for item in getattr(state, "tasklist", []) or []:
        detail = str(getattr(item, "detail", "") or "")
        if "ArgumentParseError" in detail:
            return True
    return False


def _usage_tokens(bundle: Any) -> int:
    """从 bundle 的 LLM 客户端取本 run 的 token 总量（取不到算 0）。"""
    llm = getattr(bundle, "llm", None)
    getter = getattr(llm, "usage_summary", None)
    if not callable(getter):
        return 0
    try:
        summary = getter() or {}
        return int(summary.get("prompt_tokens") or 0) + int(
            summary.get("completion_tokens") or 0
        )
    except Exception:  # pragma: no cover
        return 0

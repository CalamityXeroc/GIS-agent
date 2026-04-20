"""Evaluation loop for data-driven agent quality improvement.

Phase 3 goals:
- Evaluate context hit rate
- Evaluate plan continuity rate
- Evaluate mis-trigger rate
- Persist reports and trend history for continuous optimization
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .llm import LLMConfig, create_llm_client
from .planner import AgentPlanner


@dataclass
class EvaluationMetrics:
    sessions_analyzed: int
    followup_requests: int
    context_checks: int
    context_hits: int
    continuity_checks: int
    continuity_hits: int
    mis_trigger_checks: int
    mis_trigger_count: int
    clip_hijack_events: int = 0

    @property
    def context_hit_rate(self) -> float:
        if self.context_checks <= 0:
            return 1.0
        return self.context_hits / self.context_checks

    @property
    def plan_continuity_rate(self) -> float:
        if self.continuity_checks <= 0:
            return 1.0
        return self.continuity_hits / self.continuity_checks

    @property
    def mis_trigger_rate(self) -> float:
        if self.mis_trigger_checks <= 0:
            return 0.0
        return self.mis_trigger_count / self.mis_trigger_checks

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessions_analyzed": self.sessions_analyzed,
            "followup_requests": self.followup_requests,
            "context_checks": self.context_checks,
            "context_hits": self.context_hits,
            "context_hit_rate": round(self.context_hit_rate, 4),
            "continuity_checks": self.continuity_checks,
            "continuity_hits": self.continuity_hits,
            "plan_continuity_rate": round(self.plan_continuity_rate, 4),
            "mis_trigger_checks": self.mis_trigger_checks,
            "mis_trigger_count": self.mis_trigger_count,
            "mis_trigger_rate": round(self.mis_trigger_rate, 4),
            "clip_hijack_events": self.clip_hijack_events,
        }


@dataclass
class EvaluationReport:
    report_id: str
    generated_at: str
    metrics: EvaluationMetrics
    recommendations: list[str] = field(default_factory=list)
    report_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "metrics": self.metrics.to_dict(),
            "recommendations": self.recommendations,
            "report_path": self.report_path,
        }


class AgentEvaluationLoop:
    """Evaluate and track agent quality with persistent trend history."""

    def __init__(
        self,
        memory_dir: Path | str,
        history_dir: Path | str,
        execution_history_path: Path | str | None = None,
    ):
        self.memory_dir = Path(memory_dir)
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = self.history_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.trend_file = self.history_dir / "metrics_history.json"
        self.execution_history_path = Path(execution_history_path) if execution_history_path else None

    def run(self, session_limit: int = 30) -> EvaluationReport:
        sessions = self._load_sessions(session_limit=session_limit)
        metrics = self._compute_metrics(sessions)
        metrics.clip_hijack_events = self._count_clip_hijack_events()

        recommendations = self._build_recommendations(metrics)
        report_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        generated_at = datetime.now(timezone.utc).isoformat()

        report = EvaluationReport(
            report_id=report_id,
            generated_at=generated_at,
            metrics=metrics,
            recommendations=recommendations,
        )

        report_path = self.reports_dir / f"{report_id}.json"
        report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        md_path = self.reports_dir / f"{report_id}.md"
        md_path.write_text(self._render_markdown(report), encoding="utf-8")

        report.report_path = str(report_path)
        self._append_trend(report)
        return report

    def _load_sessions(self, session_limit: int) -> list[dict[str, Any]]:
        if not self.memory_dir.exists():
            return []
        files = sorted(self.memory_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        payloads: list[dict[str, Any]] = []
        for file_path in files[: max(1, int(session_limit))]:
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    payloads.append(data)
            except Exception:
                continue
        return payloads

    def _compute_metrics(self, sessions: list[dict[str, Any]]) -> EvaluationMetrics:
        metrics = EvaluationMetrics(
            sessions_analyzed=len(sessions),
            followup_requests=0,
            context_checks=0,
            context_hits=0,
            continuity_checks=0,
            continuity_hits=0,
            mis_trigger_checks=0,
            mis_trigger_count=0,
        )

        for session in sessions:
            turns = session.get("turns")
            if not isinstance(turns, list):
                continue

            for idx, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    continue
                if str(turn.get("role")) != "user":
                    continue

                user_text = str(turn.get("content") or "")
                if not self._is_followup_request(user_text):
                    continue

                metrics.followup_requests += 1

                assistant_text = self._find_next_assistant_text(turns, idx)
                anchor = self._find_prev_actionable_user(turns, idx)

                if anchor:
                    metrics.context_checks += 1
                    metrics.continuity_checks += 1

                    if self._is_context_hit(anchor, user_text, assistant_text):
                        metrics.context_hits += 1
                    if self._is_plan_continuous(anchor, assistant_text):
                        metrics.continuity_hits += 1

                metrics.mis_trigger_checks += 1
                if self._is_mis_trigger(user_text, assistant_text):
                    metrics.mis_trigger_count += 1

        return metrics

    def _is_followup_request(self, text: str) -> bool:
        lowered = text.lower()
        markers = [
            "继续", "根据之前", "基于之前", "接着", "后续", "在此基础", "上一步", "前面的",
            "follow up", "follow-up", "continue",
        ]
        return any(marker in lowered for marker in markers)

    def _looks_actionable(self, text: str) -> bool:
        lowered = text.lower()
        action_markers = [
            "制作", "创建", "生成", "导出", "整合", "合并", "投影", "变换", "检查", "分析",
            "裁剪", "缓冲", "叠加", "map", "merge", "project", "clip", "buffer", "overlay",
        ]
        return len(text) >= 50 or sum(1 for marker in action_markers if marker in lowered) >= 2

    def _find_next_assistant_text(self, turns: list[dict[str, Any]], user_idx: int) -> str:
        for turn in turns[user_idx + 1 :]:
            if not isinstance(turn, dict):
                continue
            if str(turn.get("role")) == "assistant":
                return str(turn.get("content") or "")
        return ""

    def _find_prev_actionable_user(self, turns: list[dict[str, Any]], user_idx: int) -> str:
        for i in range(user_idx - 1, -1, -1):
            turn = turns[i]
            if not isinstance(turn, dict):
                continue
            if str(turn.get("role")) != "user":
                continue
            content = str(turn.get("content") or "")
            if content and self._looks_actionable(content):
                return content
        return ""

    def _extract_keywords(self, text: str) -> set[str]:
        lowered = text.lower()
        eng = re.findall(r"[a-z0-9_]{3,}", lowered)
        cjk = [c for c in lowered if "\u4e00" <= c <= "\u9fff"]
        cjk_bi = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
        return set(eng + cjk_bi)

    def _is_context_hit(self, anchor: str, user_text: str, assistant_text: str) -> bool:
        if not assistant_text.strip():
            return False
        continuity_markers = ["历史任务", "同一任务", "继续规划", "在此基础", "继续", "follow-up"]
        if any(marker in assistant_text for marker in continuity_markers):
            return True

        anchor_tokens = self._extract_keywords(anchor)
        if not anchor_tokens:
            return False
        assistant_tokens = self._extract_keywords(assistant_text)
        overlap = anchor_tokens.intersection(assistant_tokens)
        return (len(overlap) / max(len(anchor_tokens), 1)) >= 0.12

    def _is_plan_continuous(self, anchor: str, assistant_text: str) -> bool:
        if not assistant_text.strip():
            return False
        # Strong continuity evidence: goal/plan mentions same core nouns
        anchor_tokens = self._extract_keywords(anchor)
        assistant_tokens = self._extract_keywords(assistant_text)
        if not anchor_tokens:
            return True
        overlap = anchor_tokens.intersection(assistant_tokens)
        return len(overlap) >= 2

    def _is_mis_trigger(self, user_text: str, assistant_text: str) -> bool:
        lowered_user = user_text.lower()
        lowered_assistant = assistant_text.lower()
        assistant_clip = any(k in lowered_assistant for k in ["clip_analysis", "裁剪", "clip"])
        if not assistant_clip:
            return False

        explicit_clip = re.search(r"(请|执行|进行|做).{0,4}(裁剪|clip|切割)", lowered_user) is not None
        negative_clip = any(
            phrase in lowered_user
            for phrase in ["不要裁剪", "不需要裁剪", "不用裁剪", "别裁剪", "禁止裁剪"]
        )
        if negative_clip:
            return True
        if explicit_clip:
            return False
        # If user didn't request clipping at all but assistant did, treat as mis-trigger.
        user_mentions_clip = any(k in lowered_user for k in ["裁剪", "clip", "切割"])
        return not user_mentions_clip

    def _count_clip_hijack_events(self) -> int:
        if not self.execution_history_path or not self.execution_history_path.exists():
            return 0
        try:
            payload = json.loads(self.execution_history_path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        if not isinstance(payload, list):
            return 0

        count = 0
        for item in payload:
            if not isinstance(item, dict):
                continue
            goal = str(item.get("goal") or "").lower()
            if "clip_analysis" in goal and "根据之前" in goal:
                count += 1
        return count

    def _build_recommendations(self, metrics: EvaluationMetrics) -> list[str]:
        tips: list[str] = []
        if metrics.context_hit_rate < 0.85:
            tips.append("提高 context hit: 将跟进请求的检索 top_k 从 8 提升到 12，并增强历史任务锚定权重。")
        if metrics.plan_continuity_rate < 0.85:
            tips.append("提高 continuity: 在 Planner 输入中增加最近 2 条 task_requirement 的显式排序注入。")
        if metrics.mis_trigger_rate > 0.05:
            tips.append("降低误触发: 扩展负向意图词表，并对 clip/buffer/overlay 增加二次确认门控。")
        if metrics.clip_hijack_events > 0:
            tips.append("检测到历史 clip hijack 事件：建议对 follow-up 语句默认禁用技能短路，仅保留显式命令放行。")
        if not tips:
            tips.append("指标健康：继续按周执行评测并观察趋势，避免回归。")
        return tips

    def _append_trend(self, report: EvaluationReport) -> None:
        history: list[dict[str, Any]] = []
        if self.trend_file.exists():
            try:
                loaded = json.loads(self.trend_file.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    history = [x for x in loaded if isinstance(x, dict)]
            except Exception:
                history = []

        history.append(
            {
                "report_id": report.report_id,
                "generated_at": report.generated_at,
                "metrics": report.metrics.to_dict(),
            }
        )
        history = history[-200:]
        self.trend_file.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    def _render_markdown(self, report: EvaluationReport) -> str:
        m = report.metrics
        lines = [
            f"# Agent Evaluation Report: {report.report_id}",
            "",
            f"- Generated At: {report.generated_at}",
            f"- Sessions Analyzed: {m.sessions_analyzed}",
            "",
            "## Metrics",
            "",
            f"- Context Hit Rate: {m.context_hit_rate:.2%} ({m.context_hits}/{m.context_checks})",
            f"- Plan Continuity Rate: {m.plan_continuity_rate:.2%} ({m.continuity_hits}/{m.continuity_checks})",
            f"- Mis-Trigger Rate: {m.mis_trigger_rate:.2%} ({m.mis_trigger_count}/{m.mis_trigger_checks})",
            f"- Clip Hijack Events (history): {m.clip_hijack_events}",
            "",
            "## Recommendations",
            "",
        ]
        for tip in report.recommendations:
            lines.append(f"- {tip}")
        lines.append("")
        return "\n".join(lines)


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

    @property
    def success_rate(self) -> float:
        if self.total_runs <= 0:
            return 0.0
        return self.success_runs / self.total_runs

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


class MultiModelBenchmark:
    """Cross-model planning benchmark for reliability parity tracking."""

    DEFAULT_TASKS = [
        "扫描输入目录并识别可用图层",
        "合并多个行政区图层并导出结果",
        "统一投影到 CGCS2000 并检查质量",
        "制作专题图并导出 PNG",
    ]

    def __init__(
        self,
        output_dir: Path | str,
        base_config: LLMConfig | None = None,
        llm_factory: Callable[[LLMConfig], Any] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = self.output_dir / "model_benchmark"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.base_config = base_config
        self.llm_factory = llm_factory or create_llm_client

    def run(
        self,
        models: list[str],
        tasks: list[str] | None = None,
        repeats: int = 1,
    ) -> MultiModelBenchmarkReport:
        selected_models = [m.strip() for m in models if str(m).strip()]
        if not selected_models:
            raise ValueError("models cannot be empty")

        task_list = tasks or list(self.DEFAULT_TASKS)
        effective_repeats = max(1, int(repeats))
        metrics_list = [self._run_model(model, task_list, effective_repeats) for model in selected_models]

        report_id = f"mm_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        generated_at = datetime.now(timezone.utc).isoformat()
        report = MultiModelBenchmarkReport(
            report_id=report_id,
            generated_at=generated_at,
            tasks=task_list,
            repeats=effective_repeats,
            model_metrics=metrics_list,
        )

        json_path = self.reports_dir / f"{report_id}.json"
        json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        md_path = self.reports_dir / f"{report_id}.md"
        md_path.write_text(self._render_markdown(report), encoding="utf-8")
        report.report_path = str(json_path)
        return report

    def _run_model(self, model: str, tasks: list[str], repeats: int) -> ModelBenchmarkMetrics:
        config = self._build_model_config(model)
        llm_client = self.llm_factory(config)
        planner = AgentPlanner(llm_client=llm_client)
        metrics = ModelBenchmarkMetrics(model=model)

        for _ in range(repeats):
            for task in tasks:
                start = time.perf_counter()
                plan = planner.plan(task_description=task, context={})
                latency_ms = (time.perf_counter() - start) * 1000.0

                metrics.total_runs += 1
                metrics.total_latency_ms += latency_ms

                fallback = bool((plan.metadata or {}).get("llm_fallback_used", False))
                if fallback:
                    metrics.fallback_runs += 1
                else:
                    if plan.steps:
                        metrics.success_runs += 1

                if (plan.metadata or {}).get("llm_fallback_reason") == "invalid_json_or_schema":
                    metrics.parse_error_runs += 1

        metrics.prompt_optimizer_hits = planner.prompt_adapter.metrics.prompt_optimizer_hits
        metrics.baml_hits = planner.plan_standardizer.metrics.baml_hits
        return metrics

    def _build_model_config(self, model: str) -> LLMConfig:
        cfg = self.base_config or LLMConfig.from_env()
        return LLMConfig(
            model=model,
            api_key=cfg.api_key,
            api_base=cfg.api_base,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            provider=cfg.provider,
            fallback_models=list(cfg.fallback_models),
            retry_count=cfg.retry_count,
            enable_prompt_optimizer=cfg.enable_prompt_optimizer,
            enable_baml_standardizer=cfg.enable_baml_standardizer,
        )

    def _render_markdown(self, report: MultiModelBenchmarkReport) -> str:
        lines = [
            f"# Multi-Model Benchmark: {report.report_id}",
            "",
            f"- Generated At: {report.generated_at}",
            f"- Repeats: {report.repeats}",
            f"- Success Rate Gap: {report.success_rate_gap:.2%}",
            f"- Max Parse Error Rate: {report.max_parse_error_rate:.2%}",
            "",
            "## Tasks",
            "",
        ]
        lines.extend([f"- {task}" for task in report.tasks])
        lines.extend(["", "## Model Metrics", ""])

        for metric in report.model_metrics:
            lines.extend(
                [
                    f"### {metric.model}",
                    f"- Runs: {metric.total_runs}",
                    f"- Success Rate: {metric.success_rate:.2%} ({metric.success_runs}/{metric.total_runs})",
                    f"- Fallback Rate: {metric.fallback_rate:.2%} ({metric.fallback_runs}/{metric.total_runs})",
                    f"- Parse Error Rate: {metric.parse_error_rate:.2%} ({metric.parse_error_runs}/{metric.total_runs})",
                    f"- Avg Latency: {metric.avg_latency_ms:.2f} ms",
                    f"- Prompt Optimizer Hits: {metric.prompt_optimizer_hits}",
                    f"- BAML Hits: {metric.baml_hits}",
                    "",
                ]
            )

        return "\n".join(lines)

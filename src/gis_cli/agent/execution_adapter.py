"""Execution adapter for gradual migration to LangChain execution layer."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .executor import ExecutionMode, ExecutionResult, ExecutionTrace, Executor
from .executor import StepTrace
from .planner import Plan


class ExecutionAdapter:
    """Adapter that keeps legacy executor behavior while enabling backend switch.

    Modes:
    - legacy: use existing Executor
    - langchain: try LangChain path, fallback to legacy on any error
    - shadow: run legacy, then best-effort run langchain path for comparison only
    """

    def __init__(
        self,
        *,
        context: Any,
        memory: Any = None,
        llm_client: Any = None,
        on_step_start: Callable[[Any], None] | None = None,
        on_step_complete: Callable[[Any, Any], None] | None = None,
        on_progress: Callable[[str, float], None] | None = None,
        config_path: Path | None = None,
        metrics_path: Path | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config_path = config_path or Path("config") / "execution_adapter_config.json"
        self.metrics_path = metrics_path or Path("config") / "execution_adapter_metrics.json"
        self.history_path = Path(str(self.metrics_path).replace("_metrics.json", "_history.json"))
        self._config = self._load_config()
        self.last_shadow_compare: dict[str, Any] = {}
        self.last_backend_used: str = "legacy"
        self.last_error: str = ""
        self.metrics: dict[str, int] = self._default_metrics()
        self._load_metrics()

        self._legacy = Executor(
            context=context,
            memory=memory,
            on_step_start=on_step_start,
            on_step_complete=on_step_complete,
            on_progress=on_progress,
        )
        self._on_progress = on_progress

    def _default_metrics(self) -> dict[str, int]:
        return {
            "runs": 0,
            "legacy_runs": 0,
            "shadow_runs": 0,
            "langchain_attempts": 0,
            "langchain_successes": 0,
            "langchain_fallbacks": 0,
        }

    def _load_metrics(self) -> None:
        if not self.metrics_path.exists():
            return
        try:
            payload = json.loads(self.metrics_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            for key in self.metrics:
                if key in payload:
                    self.metrics[key] = int(payload[key])
        except Exception:
            return

    def _save_metrics(self) -> None:
        try:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_path.write_text(
                json.dumps(self.metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            return

    def _append_history(self, event: dict[str, Any]) -> None:
        payload = []
        if self.history_path.exists():
            try:
                loaded = json.loads(self.history_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    payload = loaded
            except Exception:
                payload = []
        payload.append(event)
        max_events = int(self._config.get("max_history_events", 100))
        if max_events > 0:
            payload = payload[-max_events:]
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            return

    def load_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        try:
            payload = json.loads(self.history_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [x for x in payload if isinstance(x, dict)]
            return []
        except Exception:
            return []

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {"mode": "legacy"}
        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return {"mode": "legacy"}
            mode = str(payload.get("mode", "legacy")).strip().lower()
            if mode not in {"legacy", "langchain", "shadow"}:
                mode = "legacy"
            payload["mode"] = mode
            return payload
        except Exception:
            return {"mode": "legacy"}

    @property
    def on_progress(self) -> Callable[[str, float], None] | None:
        return self._on_progress

    @on_progress.setter
    def on_progress(self, callback: Callable[[str, float], None] | None) -> None:
        self._on_progress = callback
        self._legacy.on_progress = callback

    def _mode(self) -> str:
        return str(self._config.get("mode", "legacy")).strip().lower() or "legacy"

    def reload_config(self) -> dict[str, Any]:
        self._config = self._load_config()
        return self.status()

    def set_mode(self, mode: str, persist: bool = True) -> dict[str, Any]:
        new_mode = str(mode).strip().lower()
        if new_mode not in {"legacy", "langchain", "shadow"}:
            raise ValueError(f"Unsupported execution adapter mode: {mode}")
        self._config["mode"] = new_mode
        if persist:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(
                json.dumps(self._config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return self.status()

    def status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "mode": self._mode(),
            "config_path": str(self.config_path),
            "metrics_path": str(self.metrics_path),
            "history_path": str(self.history_path),
            "last_backend_used": self.last_backend_used,
            "last_error": self.last_error,
            "metrics": dict(self.metrics),
            "langchain_dispatch": "agentexecutor",
        }
        if self.last_shadow_compare:
            status["shadow_compare"] = self.last_shadow_compare
        history = self.load_history()
        status["history_count"] = len(history)
        if history:
            status["last_history"] = history[-1]
        status["guard_recommendation"] = self.guard_recommendation()
        return status

    def reset_metrics(self, persist: bool = True) -> dict[str, Any]:
        self.metrics = self._default_metrics()
        if persist:
            self._save_metrics()
        return self.status()

    def guard_recommendation(self) -> dict[str, Any]:
        guard_cfg = self._config.get("guard", {})
        if not isinstance(guard_cfg, dict):
            guard_cfg = {}
        min_attempts = int(guard_cfg.get("min_attempts", 10))
        max_fallback_rate = float(guard_cfg.get("max_fallback_rate", 0.2))
        suggested_mode = str(guard_cfg.get("suggested_mode", "langchain")).strip().lower()
        if suggested_mode not in {"legacy", "langchain", "shadow"}:
            suggested_mode = "langchain"

        attempts = int(self.metrics.get("langchain_attempts", 0))
        fallbacks = int(self.metrics.get("langchain_fallbacks", 0))
        fallback_rate = (fallbacks / attempts) if attempts > 0 else 1.0
        can_recommend = attempts >= min_attempts and fallback_rate <= max_fallback_rate
        target_mode = suggested_mode if can_recommend else "shadow"
        reason = (
            f"attempts={attempts}, fallback_rate={fallback_rate:.3f}, "
            f"threshold=<= {max_fallback_rate:.3f}, min_attempts={min_attempts}"
        )
        return {
            "can_recommend_switch": can_recommend,
            "target_mode": target_mode,
            "fallback_rate": round(fallback_rate, 4),
            "attempts": attempts,
            "fallbacks": fallbacks,
            "reason": reason,
        }

    def apply_guard_recommendation(self, persist: bool = True) -> dict[str, Any]:
        recommendation = self.guard_recommendation()
        target_mode = recommendation.get("target_mode", "shadow")
        return self.set_mode(str(target_mode), persist=persist)

    def execute(self, plan: Plan, mode: ExecutionMode = ExecutionMode.DRY_RUN) -> ExecutionResult:
        self.metrics["runs"] += 1
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": self._mode(),
            "plan_id": plan.id,
            "goal": plan.goal,
            "execution_mode": mode.value,
        }
        backend_mode = self._mode()
        if backend_mode == "legacy":
            self.metrics["legacy_runs"] += 1
            self.last_backend_used = "legacy"
            result = self._legacy.execute(plan, mode)
            self._save_metrics()
            event.update(
                {
                    "backend_used": self.last_backend_used,
                    "success": result.success,
                    "outputs": len(result.outputs),
                    "fallback": False,
                }
            )
            self._append_history(event)
            return result
        if backend_mode == "langchain":
            self.metrics["langchain_attempts"] += 1
            result = self._try_execute_langchain(plan, mode)
            if result:
                self.metrics["langchain_successes"] += 1
                self.last_backend_used = "langchain"
                self._save_metrics()
                event.update(
                    {
                        "backend_used": self.last_backend_used,
                        "success": result.success,
                        "outputs": len(result.outputs),
                        "fallback": False,
                    }
                )
                self._append_history(event)
                return result
            self.metrics["langchain_fallbacks"] += 1
            self.metrics["legacy_runs"] += 1
            self.last_backend_used = "legacy_fallback"
            fallback_result = self._legacy.execute(plan, mode)
            self._save_metrics()
            event.update(
                {
                    "backend_used": self.last_backend_used,
                    "success": fallback_result.success,
                    "outputs": len(fallback_result.outputs),
                    "fallback": True,
                    "error": self.last_error,
                }
            )
            self._append_history(event)
            return fallback_result

        # shadow mode
        self.metrics["shadow_runs"] += 1
        self.metrics["legacy_runs"] += 1
        legacy_result = self._legacy.execute(plan, mode)
        self.metrics["langchain_attempts"] += 1
        shadow_result = self._try_execute_langchain(plan, mode)
        if shadow_result:
            self.metrics["langchain_successes"] += 1
        else:
            self.metrics["langchain_fallbacks"] += 1
        self.last_shadow_compare = {
            "legacy_success": legacy_result.success,
            "legacy_outputs": len(legacy_result.outputs),
            "langchain_success": shadow_result.success if shadow_result else None,
            "langchain_outputs": len(shadow_result.outputs) if shadow_result else None,
        }
        self.last_backend_used = "legacy_shadow"
        self._save_metrics()
        event.update(
            {
                "backend_used": self.last_backend_used,
                "success": legacy_result.success,
                "outputs": len(legacy_result.outputs),
                "fallback": shadow_result is None,
                "shadow_compare": dict(self.last_shadow_compare),
                "error": self.last_error,
            }
        )
        self._append_history(event)
        return legacy_result

    def execute_single_tool(self, tool_name: str, input_data: dict[str, Any], dry_run: bool = True) -> Any:
        # Keep single-tool execution stable during migration.
        return self._legacy.execute_single_tool(tool_name, input_data, dry_run=dry_run)

    def _try_execute_langchain(self, plan: Plan, mode: ExecutionMode) -> ExecutionResult | None:
        """Best-effort LangChain AgentExecutor path with safe fallback."""
        try:
            from langchain.agents import AgentExecutor, create_react_agent
            from langchain_core.prompts import PromptTemplate
            from langchain_core.tools import Tool
        except Exception as exc:
            self.last_error = f"langchain_import_failed: {exc}"
            return None

        llm = self._build_langchain_llm()
        if llm is None:
            self.last_error = "langchain_llm_unavailable"
            return None

        trace = ExecutionTrace(plan_id=plan.id, mode=mode)
        all_outputs: list[str] = []
        try:
            while True:
                step = plan.next_step
                if step is None:
                    break

                captured: dict[str, Any] = {"trace": None, "result": None}

                def _dispatch_step(_: str = "") -> str:
                    step_trace, tool_result = self._legacy._execute_step(step, mode, plan)  # noqa: SLF001
                    captured["trace"] = step_trace
                    captured["result"] = tool_result
                    return json.dumps(
                        {
                            "success": bool(tool_result.success),
                            "error": tool_result.error,
                            "outputs": len(getattr(tool_result, "outputs", []) or []),
                        },
                        ensure_ascii=False,
                    )

                dispatch_tool = Tool(
                    name="dispatch_step",
                    description=(
                        "Execute current GIS plan step once. "
                        "Always call this tool exactly once and then provide final answer."
                    ),
                    func=_dispatch_step,
                    return_direct=False,
                )

                prompt = PromptTemplate.from_template(
                    "You are a workflow executor.\n"
                    "You must call one tool exactly once to execute the step.\n"
                    "Available tools:\n{tools}\n"
                    "Tool names: {tool_names}\n\n"
                    "Question: {input}\n"
                    "Thought:{agent_scratchpad}"
                )
                agent = create_react_agent(llm, [dispatch_tool], prompt)
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=[dispatch_tool],
                    verbose=False,
                    max_iterations=2,
                    handle_parsing_errors=True,
                )
                agent_executor.invoke(
                    {
                        "input": (
                            f"Execute step {step.id} [{step.tool}] {step.description}. "
                            "Call dispatch_step now."
                        )
                    }
                )

                step_trace = captured.get("trace")
                tool_result = captured.get("result")
                if not isinstance(step_trace, StepTrace) or tool_result is None:
                    self.last_error = f"agentexecutor_no_dispatch: {step.id}"
                    return None

                trace.add_step(step_trace)
                if tool_result.success:
                    step.complete(tool_result.data)
                    all_outputs.extend(getattr(tool_result, "outputs", []) or [])
                    self._legacy._report_progress(plan)  # noqa: SLF001
                else:
                    step.fail(tool_result.error or "Unknown error")
                    if not self._legacy._should_continue_after_failure(step, plan):  # noqa: SLF001
                        trace.complete(False, all_outputs, f"Step {step.id} failed: {tool_result.error}")
                        self.last_error = tool_result.error or "step_failed"
                        return ExecutionResult(
                            success=False,
                            plan=plan,
                            trace=trace,
                            outputs=all_outputs,
                            error=tool_result.error,
                        )

            if plan.has_failed:
                failed_steps = [s for s in plan.steps if s.status.value == "failed"]
                step_errors = [f"{s.tool}: {s.error}" for s in failed_steps if s.error]
                error_detail = "; ".join(step_errors) if step_errors else "Plan has failed steps"
                trace.complete(False, all_outputs, error_detail)
                self.last_error = error_detail
                return ExecutionResult(
                    success=False,
                    plan=plan,
                    trace=trace,
                    outputs=all_outputs,
                    error=error_detail,
                )

            trace.complete(True, all_outputs)
            self.last_error = ""
            return ExecutionResult(success=True, plan=plan, trace=trace, outputs=all_outputs)
        except Exception as exc:
            self.last_error = f"langchain_execution_failed: {exc}"
            return None

    def _build_langchain_llm(self) -> Any | None:
        try:
            from langchain_openai import ChatOpenAI
        except Exception:
            return None

        model = None
        api_key = None
        base_url = None
        temperature = 0.0

        cfg = getattr(self.llm_client, "config", None)
        if cfg is not None:
            model = getattr(cfg, "model", None)
            api_key = getattr(cfg, "api_key", None)
            base_url = getattr(cfg, "api_base", None)
            try:
                temperature = float(getattr(cfg, "temperature", 0.0))
            except Exception:
                temperature = 0.0

        model = model or os.environ.get("GIS_LLM_MODEL") or "gpt-4o-mini"
        api_key = api_key or os.environ.get("GIS_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = base_url or os.environ.get("GIS_LLM_API_BASE") or os.environ.get("OPENAI_API_BASE")
        if not api_key:
            return None

        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "temperature": temperature,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)


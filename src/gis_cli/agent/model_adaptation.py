"""Model-agnostic prompt/output adaptation utilities.

This module integrates optional open-source tooling with graceful fallback:
- prompt-optimizer: prompt rewriting for target model families
- BAML runtime clients (if generated in project): schema-first output validation

All integrations are best-effort. If external tools are unavailable, the core
agent behavior remains unchanged.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)

def _pydantic_fallback(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

@dataclass
class AdaptationMetrics:
    """Lightweight counters for adaptation/standardization diagnostics."""

    prompt_adapt_calls: int = 0
    prompt_optimizer_hits: int = 0
    baml_attempts: int = 0
    baml_hits: int = 0
    normalization_hits: int = 0


class PromptAdapter:
    """Adapt prompts across different model families.

    The adapter applies deterministic local constraints first, then optionally
    invokes prompt-optimizer when available.
    """

    def __init__(self, enable_prompt_optimizer: bool = True):
        self.enable_prompt_optimizer = enable_prompt_optimizer
        self.metrics = AdaptationMetrics()

    def adapt_prompt(self, prompt: str, task_type: str = "general", model_hint: str | None = None) -> str:
        self.metrics.prompt_adapt_calls += 1
        base = (prompt or "").strip()
        if not base:
            return ""

        adapted = self._apply_model_constraints(base, task_type=task_type, model_hint=model_hint)
        if self.enable_prompt_optimizer:
            optimized = self._try_prompt_optimizer(adapted, task_type=task_type, model_hint=model_hint)
            if optimized:
                self.metrics.prompt_optimizer_hits += 1
                return optimized
        return adapted

    def _apply_model_constraints(self, prompt: str, task_type: str, model_hint: str | None) -> str:
        model = (model_hint or "").lower()

        # These families benefit from shorter directives and hard output rules.
        compact_families = ["qwen", "deepseek", "minimax", "glm", "zhipu"]
        if any(name in model for name in compact_families):
            if task_type == "planning":
                return (
                    f"{prompt}\n\n"
                    "输出约束：\n"
                    "1) 仅输出 JSON 对象。\n"
                    "2) 不要 markdown，不要解释性文字。\n"
                    "3) 字段必须完整：id, goal, steps。"
                )
            if task_type == "intent":
                return (
                    f"{prompt}\n\n"
                    "输出约束：优先输出 {\"intent\": \"...\"}，否则只输出单个标签，不要解释。"
                )

        if "gpt" in model and task_type == "planning":
            return f"{prompt}\n\n请优先保证 JSON 字段类型稳定，避免 null 与字符串混用。"

        return prompt

    def _try_prompt_optimizer(self, prompt: str, task_type: str, model_hint: str | None) -> str | None:
        try:
            import prompt_optimizer as po  # type: ignore
        except Exception:
            return None

        optimize_fn = getattr(po, "optimize_prompt", None)
        if not callable(optimize_fn):
            return None

        attempts = [
            lambda: optimize_fn(prompt=prompt, model=model_hint or "", task_type=task_type),
            lambda: optimize_fn(prompt, model=model_hint or "", task_type=task_type),
            lambda: optimize_fn(prompt=prompt),
            lambda: optimize_fn(prompt),
        ]

        for call in attempts:
            try:
                result = call()
            except TypeError:
                continue
            except Exception:
                return None

            if isinstance(result, str) and result.strip():
                return result.strip()
            if isinstance(result, dict):
                for key in ("prompt", "optimized_prompt", "text", "content"):
                    value = result.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

        return None


class PlanStandardizer:
    """Standardize and validate plan payloads.

    If a project BAML client is available, it is used first. Otherwise this
    class applies deterministic schema normalization for planner stability.
    """

    def __init__(self, enable_baml: bool = True):
        self.enable_baml = enable_baml
        self.metrics = AdaptationMetrics()

    def standardize(self, plan_json: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(plan_json, dict):
            return None

        via_baml = self._try_baml_standardize(plan_json)
        if isinstance(via_baml, dict):
            plan_json = via_baml
            self.metrics.baml_hits += 1

        normalized = self._normalize_plan_dict(plan_json)
        if normalized is not None:
            self.metrics.normalization_hits += 1
        return normalized

    def _try_baml_standardize(self, plan_json: dict[str, Any]) -> dict[str, Any] | None:
        # Best-effort integration for generated BAML client projects.
        if not self.enable_baml:
            return None
        self.metrics.baml_attempts += 1
        try:
            from baml_client import b  # type: ignore
        except Exception:
            return None

        candidate_functions = [
            "normalize_gis_execution_plan",
            "standardize_gis_execution_plan",
            "validate_gis_execution_plan",
        ]

        source = json.dumps(plan_json, ensure_ascii=False, default=_pydantic_fallback)
        for fn_name in candidate_functions:
            fn = getattr(b, fn_name, None)
            if not callable(fn):
                continue
            try:
                result = fn(source)
            except Exception:
                continue

            if isinstance(result, dict):
                return result
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    continue

        return None

    def _normalize_plan_dict(self, plan_json: dict[str, Any]) -> dict[str, Any] | None:
        goal = str(plan_json.get("goal", "")).strip()
        if not goal:
            return None

        steps = plan_json.get("steps")
        if not isinstance(steps, list) or not steps:
            return None

        normalized_steps: list[dict[str, Any]] = []
        for idx, raw_step in enumerate(steps, start=1):
            if not isinstance(raw_step, dict):
                continue

            step_id = str(raw_step.get("id") or f"step_{idx}").strip()
            tool, step_input = self._extract_tool_and_input(raw_step)
            description = str(raw_step.get("description") or "").strip()
            if not tool or not description:
                continue

            if not isinstance(step_input, dict):
                step_input = {}

            depends_on = raw_step.get("depends_on")
            if not isinstance(depends_on, list):
                depends_on = []
            depends_on = [str(dep).strip() for dep in depends_on if str(dep).strip()]

            normalized_steps.append(
                {
                    "id": step_id,
                    "tool": tool,
                    "description": description,
                    "input": step_input,
                    "depends_on": depends_on,
                }
            )

        if not normalized_steps:
            return None

        normalized: dict[str, Any] = {
            "id": str(plan_json.get("id") or "").strip() or "plan_generated",
            "goal": goal,
            "steps": normalized_steps,
            "expected_outputs": plan_json.get("expected_outputs") if isinstance(plan_json.get("expected_outputs"), list) else [],
        }
        # Preserve expert_notes for expert mode planning
        if "expert_notes" in plan_json:
            normalized["expert_notes"] = plan_json["expert_notes"]
        return normalized

    def _extract_tool_and_input(self, step: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Extract canonical tool/input from heterogeneous step formats."""
        tool = str(step.get("tool") or "").strip()
        raw_input: Any = step.get("input")

        tool_call = step.get("tool_call")
        if isinstance(tool_call, dict):
            tool = str(tool_call.get("name") or tool_call.get("tool") or tool).strip()
            raw_input = tool_call.get("arguments", raw_input)

        function_call = step.get("function_call")
        if isinstance(function_call, dict):
            tool = str(function_call.get("name") or tool).strip()
            raw_input = function_call.get("arguments", raw_input)

        if not tool:
            tool = str(step.get("name") or "").strip()

        if isinstance(raw_input, dict):
            return tool, raw_input
        if isinstance(raw_input, str):
            text = raw_input.strip()
            if not text:
                return tool, {}
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return tool, parsed
            except Exception:
                return tool, {}
        return tool, {}


class BAMLBridge:
    """Best-effort bridge for BAML generated clients.

    This bridge is intentionally tolerant: if baml_client is unavailable or
    function signatures differ, it returns None and caller falls back.
    """

    def __init__(
        self,
        enabled: bool = True,
        function_map: dict[str, list[str]] | None = None,
        allow_builtin_fallback: bool = False,
    ):
        self.enabled = enabled
        self.function_map = function_map or {}
        self.allow_builtin_fallback = allow_builtin_fallback
        self.metrics = AdaptationMetrics()
        self._missing_client_warned = False
        self._builtin_fallbacks: dict[str, Any] = {
            "infer_gis_intent": self._builtin_noop,
            "classify_gis_intent": self._builtin_noop,
            "detect_gis_intent": self._builtin_noop,
            "refine_gis_task": self._builtin_noop,
            "normalize_gis_task": self._builtin_noop,
            "rewrite_gis_task": self._builtin_noop,
            "generate_gis_plan": self._builtin_noop,
            "create_gis_plan": self._builtin_noop,
            "generate_execution_plan": self._builtin_noop,
            "suggest_gis_recovery": self._builtin_noop,
            "generate_recovery_plan": self._builtin_noop,
            "create_recovery_plan": self._builtin_noop,
        }

    def infer_intent(self, message: str, valid_intents: list[str]) -> str | None:
        if not self.enabled:
            return None

        result = self._call_baml_function(
            self._candidate_functions("intent", ["infer_gis_intent", "classify_gis_intent", "detect_gis_intent"]),
            message=message,
            valid_intents=valid_intents,
        )
        if isinstance(result, str):
            text = result.strip().lower()
            if text in valid_intents:
                return text
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    intent = str(parsed.get("intent", "")).strip().lower()
                    return intent if intent in valid_intents else None
            except Exception:
                return None
        if isinstance(result, dict):
            intent = str(result.get("intent", "")).strip().lower()
            return intent if intent in valid_intents else None
        return None

    def refine_task(self, message: str) -> str | None:
        if not self.enabled:
            return None

        result = self._call_baml_function(
            self._candidate_functions("task_refine", ["refine_gis_task", "normalize_gis_task", "rewrite_gis_task"]),
            message=message,
        )
        if isinstance(result, str):
            text = result.strip()
            return text if text else None
        if isinstance(result, dict):
            for key in ("refined_task", "task", "content", "text"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def generate_plan(self, task_description: str, context: dict[str, Any] | None) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        context_payload = context if isinstance(context, dict) else {}
        result = self._call_baml_function(
            self._candidate_functions("planning", ["generate_gis_plan", "create_gis_plan", "generate_execution_plan"]),
            task_description=task_description,
            context=context_payload,
            context_json=json.dumps(context_payload, ensure_ascii=False, default=_pydantic_fallback),
        )
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None

    def suggest_recovery(
        self,
        plan_payload: dict[str, Any],
        failed_step_payload: dict[str, Any],
        remaining_steps_payload: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        result = self._call_baml_function(
            self._candidate_functions(
                "recovery",
                ["suggest_gis_recovery", "generate_recovery_plan", "create_recovery_plan"],
            ),
            plan=plan_payload,
            failed_step=failed_step_payload,
            remaining_steps=remaining_steps_payload,
            plan_json=json.dumps(plan_payload, ensure_ascii=False, default=_pydantic_fallback),
            failed_step_json=json.dumps(failed_step_payload, ensure_ascii=False, default=_pydantic_fallback),
            remaining_steps_json=json.dumps(remaining_steps_payload, ensure_ascii=False, default=_pydantic_fallback),
        )
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None

    def _candidate_functions(self, capability: str, defaults: list[str]) -> list[str]:
        key = str(capability or "").strip().lower()
        preferred = self.function_map.get(key, [])
        merged: list[str] = []
        for name in [*preferred, *defaults]:
            item = str(name).strip()
            if item and item not in merged:
                merged.append(item)
        return merged

    def _call_baml_function(self, candidate_functions: list[str], **kwargs: Any) -> Any:
        self.metrics.baml_attempts += 1
        b = self._load_baml_client()
        if b is None and not self.allow_builtin_fallback:
            return None

        for fn_name in candidate_functions:
            fn = self._resolve_function(b, fn_name)
            if not callable(fn):
                continue

            attempts = [
                lambda: fn(**kwargs),
                lambda: fn(*[kwargs[k] for k in ("task_description", "message", "context_json") if k in kwargs]),
                lambda: fn(*[kwargs[k] for k in ("message", "task_description") if k in kwargs]),
            ]
            for call in attempts:
                try:
                    result = call()
                    self.metrics.baml_hits += 1
                    return result
                except TypeError:
                    continue
                except Exception:
                    break

        return None

    def _resolve_function(self, client: Any | None, fn_name: str) -> Any | None:
        if client is not None:
            fn = getattr(client, fn_name, None)
            if callable(fn):
                return fn
        if self.allow_builtin_fallback:
            fn = self._builtin_fallbacks.get(fn_name)
            if callable(fn):
                return fn
        return None

    def _builtin_noop(self, *args: Any, **kwargs: Any) -> Any:
        # Keep behavior non-invasive: callers will naturally fall back to non-BAML paths.
        return None

    def _load_baml_client(self) -> Any | None:
        try:
            from baml_client import b  # type: ignore
        except Exception:
            if self.enabled and not self._missing_client_warned:
                logger.warning("BAML enabled but baml_client is unavailable; falling back to non-BAML paths")
                self._missing_client_warned = True
            return None
        return b

    def diagnostics(self) -> dict[str, Any]:
        """Return runtime diagnostics for BAML availability and function bindings."""
        client = self._load_baml_client()
        capabilities = {
            "intent": self._candidate_functions("intent", ["infer_gis_intent", "classify_gis_intent", "detect_gis_intent"]),
            "task_refine": self._candidate_functions("task_refine", ["refine_gis_task", "normalize_gis_task", "rewrite_gis_task"]),
            "planning": self._candidate_functions("planning", ["generate_gis_plan", "create_gis_plan", "generate_execution_plan"]),
            "recovery": self._candidate_functions("recovery", ["suggest_gis_recovery", "generate_recovery_plan", "create_recovery_plan"]),
        }

        functions: dict[str, list[dict[str, Any]]] = {}
        for capability, names in capabilities.items():
            entries: list[dict[str, Any]] = []
            for name in names:
                source = "missing"
                fn = None
                if client is not None:
                    fn = getattr(client, name, None)
                    if callable(fn):
                        source = "client"
                if fn is None and self.allow_builtin_fallback:
                    fn = self._builtin_fallbacks.get(name)
                    if callable(fn):
                        source = "builtin"
                entries.append(
                    {
                        "name": name,
                        "exists": fn is not None,
                        "callable": callable(fn),
                        "source": source,
                    }
                )
            functions[capability] = entries

        capability_ready: dict[str, bool] = {}
        for capability, entries in functions.items():
            capability_ready[capability] = any(
                bool(item.get("exists")) and bool(item.get("callable")) for item in entries
            )

        return {
            "enabled": self.enabled,
            "client_available": client is not None,
            "allow_builtin_fallback": self.allow_builtin_fallback,
            "function_map": self.function_map,
            "functions": functions,
            "capability_ready": capability_ready,
        }

    def validate_required_capabilities(self, required: list[str] | None = None) -> dict[str, Any]:
        """Validate required capabilities for strict preflight checks."""
        diag = self.diagnostics()
        capability_ready = diag.get("capability_ready", {}) if isinstance(diag.get("capability_ready"), dict) else {}

        required_list = required or ["intent", "task_refine", "planning", "recovery"]
        normalized_required = [str(x).strip().lower() for x in required_list if str(x).strip()]

        missing = [cap for cap in normalized_required if not bool(capability_ready.get(cap, False))]
        return {
            "required": normalized_required,
            "missing": missing,
            "ok": len(missing) == 0,
            "diagnostics": diag,
        }

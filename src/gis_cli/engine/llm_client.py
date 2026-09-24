# -*- coding: utf-8 -*-
"""Engine LLM client: robust OpenAI-compatible chat with tool calling.

This is the LLM layer for the new agent loop. It intentionally avoids the
legacy BAML/prompt-optimizer stack and focuses on:

- native tool calling (with graceful degradation handled by tool_codec)
- reasoning-model support (``reasoning_content``)
- strict single-flight concurrency (many gateways cap concurrent requests)
- retry with exponential backoff on 429/5xx/timeouts
- model fallback list
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when an LLM call fails after all retries."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass
class ToolCall:
    """A normalized tool call emitted by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""
    parse_error: str = ""


@dataclass
class LLMResponse:
    """Normalized chat completion response."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return int(self.usage.get("total_tokens", 0) or 0)


@dataclass
class EngineLLMConfig:
    """Configuration for the engine LLM client."""

    model: str = ""
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.2
    max_tokens: int = 16000
    timeout: float = 900.0
    # 单模型整轮（含重试）的绝对时间上限；网关拉锯时不再无限缠绵。
    total_timeout: float = 420.0
    fallback_models: list[str] = field(default_factory=list)
    # 熔断：同一模型连续失败（超时/连不上）达到阈值后冷却一段时间，先走备用模型。
    # 冷却时长按“连续冷却次数”指数升级（300→600→1200…上限 cooldown_max_seconds）：
    # 实测网关坏掉的主模型会反复“冷却到期→重试→再挂 420s”，不带升级的话每轮白烧 7~14 分钟。
    model_failure_threshold: int = 2
    model_cooldown_seconds: float = 300.0
    model_cooldown_max_seconds: float = 3600.0
    # 已经被冷却过的模型再次被尝试时，用更短的预算——网关卡死时每次尝试要烧满
    # total_timeout（实测 420s），反复重试会把整个任务预算吃光。
    reduced_timeout_seconds: float = 120.0
    # 多模态：模型能识图时，把产出图（地图/统计图）附进对话，让它自己看一眼核对；
    # 也可用于图面质检（中文是否变方框、图例是否被裁掉）。
    vision: bool = False
    routing_rules: dict[str, list[str]] = field(default_factory=dict)
    max_concurrency: int = 1
    retry_count: int = 6
    backoff_seconds: float = 8.0
    max_backoff_seconds: float = 90.0

    @classmethod
    def from_file(cls, path: str | Path) -> "EngineLLMConfig":
        data: dict[str, Any] = {}
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        engine = data.get("engine", {}) if isinstance(data.get("engine"), dict) else {}
        return cls(
            model=str(data.get("model", cls.model)),
            api_key=str(data.get("api_key", "") or ""),
            api_base=str(data.get("api_base", "") or ""),
            temperature=float(data.get("temperature", cls.temperature)),
            max_tokens=int(data.get("max_tokens", cls.max_tokens)),
            timeout=float(data.get("timeout", cls.timeout)),
            total_timeout=float(engine.get("request_total_timeout_seconds", data.get("total_timeout", 420.0))),
            vision=bool(data.get("vision", engine.get("vision", False))),
            fallback_models=[str(m) for m in data.get("fallback_models", []) if str(m).strip()],
            model_failure_threshold=int(engine.get("model_failure_threshold", data.get("model_failure_threshold", 2))),
            model_cooldown_seconds=float(engine.get("model_cooldown_seconds", data.get("model_cooldown_seconds", 300.0))),
            routing_rules={
                str(k).lower(): [str(v) for v in vals]
                for k, vals in (data.get("routing_rules") or {}).items()
                if isinstance(vals, list)
            },
            max_concurrency=max(1, int(engine.get("llm_max_concurrency", 1))),
            retry_count=max(1, int(engine.get("request_retry", data.get("retry_count", 6)))),
            backoff_seconds=float(engine.get("request_backoff_seconds", 8.0)),
            max_backoff_seconds=float(engine.get("request_max_backoff_seconds", 90.0)),
        )

    def models_for(self, task_type: str | None = None, explicit: str | None = None) -> list[str]:
        """Return ordered candidate models for a task type."""
        candidates: list[str] = []

        def _add(name: str | None) -> None:
            name = (name or "").strip()
            if name and name not in candidates:
                candidates.append(name)

        _add(explicit)
        key = (task_type or "").strip().lower()
        if key:
            for model in self.routing_rules.get(key, []):
                _add(model)
        _add(self.model)
        for model in self.fallback_models:
            _add(model)
        return candidates


_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class EngineLLMClient:
    """Single-flight, retrying chat client with tool-calling support."""

    def __init__(self, config: EngineLLMConfig):
        self.config = config
        self._client: Any = None
        self._lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(max(1, config.max_concurrency))
        self._consecutive_429 = 0
        self._failures: dict[str, int] = {}
        self._cooldown_until: dict[str, float] = {}
        self._cooldown_strikes: dict[str, int] = {}

    # ------------------------------------------------------------------ setup
    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise LLMError("openai package is required for EngineLLMClient") from exc
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base or None,
                timeout=self.config.timeout,
                max_retries=0,  # we own retries
            )
        return self._client

    # ------------------------------------------------------------------- chat
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        task_type: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send a chat completion, retrying transient failures.

        Raises ``LLMError`` when every candidate model has been exhausted.
        """
        candidates = self.config.models_for(task_type=task_type, explicit=model)
        if not candidates:
            raise LLMError(
                "未配置模型：请在 config/llm_config.json 中填写 model（可同时配置 fallback_models）"
            )
        candidates = self._order_by_health(candidates)

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                response = self._chat_once(
                    messages,
                    model=candidate,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                self._record_success(candidate)
                return response
            except LLMError as exc:
                last_error = exc
                self._record_failure(candidate, exc)
                logger.warning("model %s failed: %s", candidate, exc)
                continue
        raise LLMError(f"All models failed. Last error: {last_error}")

    # -------------------------------------------------------------- 模型熔断
    def _order_by_health(self, candidates: list[str]) -> list[str]:
        """把处于冷却期的模型排到最后（没别的选择时仍会尝试）。"""
        now = time.monotonic()

        def cooling(candidate: str) -> bool:
            until = self._cooldown_until.get(candidate, 0.0)
            if until and until <= now:
                self._cooldown_until.pop(candidate, None)
                self._failures.pop(candidate, None)
                return False
            return until > now

        healthy = [c for c in candidates if not cooling(c)]
        cooling_models = [c for c in candidates if cooling(c)]
        if cooling_models:
            logger.warning("模型冷却中，先试其他模型: %s", ", ".join(cooling_models))
        return healthy + cooling_models

    def _record_success(self, model: str) -> None:
        self._failures.pop(model, None)
        self._cooldown_until.pop(model, None)
        self._cooldown_strikes.pop(model, None)

    def _record_failure(self, model: str, exc: Exception) -> None:
        """连续超时/连接失败达阈值后冷却该模型（网关卡死时不再白等）。

        冷却时长指数升级：同一模型反复“冷却到期又失败”时，下一次冷却翻倍，
        避免整个 run 反复在坏模型上白烧 420s/次。
        """
        if getattr(exc, "status_code", None) not in (None, 429):
            return
        self._failures[model] = self._failures.get(model, 0) + 1
        if self._failures[model] < max(1, self.config.model_failure_threshold):
            return
        strikes = self._cooldown_strikes.get(model, 0)
        self._cooldown_strikes[model] = strikes + 1
        base = max(1.0, float(self.config.model_cooldown_seconds))
        cooldown = min(base * (2 ** strikes), max(base, float(self.config.model_cooldown_max_seconds)))
        self._cooldown_until[model] = time.monotonic() + cooldown
        logger.warning(
            "模型 %s 连续失败 %s 次（第 %s 次冷却），冷却 %.0f 秒",
            model,
            self._failures[model],
            strikes + 1,
            cooldown,
        )

    def _chat_once(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if response_format:
            payload["response_format"] = response_format

        last_error: Exception | None = None
        conn_failures = 0
        budget = float(self.config.total_timeout)
        if self._cooldown_strikes.get(model):
            # 该模型本回合已经超时/断连过 → 缩短单次尝试预算，快速失败换下一个模型
            budget = min(budget, float(self.config.reduced_timeout_seconds or budget))
        deadline = time.monotonic() + budget
        budget_exhausted = False
        for attempt in range(1, self.config.retry_count + 1):
            remaining_budget = deadline - time.monotonic()
            if remaining_budget <= 0:
                budget_exhausted = True
                break
            with self._semaphore:
                try:
                    response = self.client.chat.completions.create(
                        **payload, timeout=max(30.0, remaining_budget)
                    )
                    self._consecutive_429 = 0
                    return self._normalize(response, model)
                except Exception as exc:  # provider-specific exception types
                    status = _status_code(exc)
                    body = _error_body(exc)
                    retryable = status is None or status in _RETRYABLE_STATUS
                    if not retryable:
                        raise LLMError(
                            f"Non-retryable LLM error ({status}): {body or exc}",
                            status_code=status,
                            body=body,
                        ) from exc
                    last_error = exc
                    delay = self._backoff_delay(attempt, status)
                    if status == 429:
                        self._consecutive_429 += 1
                    elif status is None:
                        # Gateway unreachable: fail fast so the caller can switch models.
                        conn_failures += 1
                    logger.warning(
                        "LLM call failed (attempt %s/%s, status=%s): %s -- sleeping %.1fs",
                        attempt,
                        self.config.retry_count,
                        status,
                        body[:200] if body else exc,
                        delay,
                    )
                    if conn_failures >= 2:
                        break
                    time.sleep(delay)
        reason = (
            f"超过全局超时 {budget:.0f}s" if budget_exhausted
            else f"{self.config.retry_count} 次尝试"
        )
        raise LLMError(
            f"LLM call failed after {reason}: {last_error}",
            status_code=_status_code(last_error),
            body=_error_body(last_error),
        )

    def _backoff_delay(self, attempt: int, status: int | None) -> float:
        if status == 429:
            # Gateways with a hard concurrency cap need patient backoff.
            base = max(self.config.backoff_seconds, 10.0) * (1.6 ** min(self._consecutive_429, 4))
            delay = min(base * (2 ** (attempt - 1)), self.config.max_backoff_seconds)
            jittered = delay * (0.8 + 0.4 * random.random())
            return min(jittered, self.config.max_backoff_seconds)
        # Connection/DNS errors: short delay, switch model quickly.
        return min(3.0 * (2 ** (attempt - 1)), 15.0)

    @staticmethod
    def _normalize(response: Any, model: str) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message
        content = getattr(message, "content", None) or ""
        reasoning = getattr(message, "reasoning_content", None) or ""
        tool_calls: list[ToolCall] = []
        for raw_call in getattr(message, "tool_calls", None) or []:
            fn = getattr(raw_call, "function", None)
            raw_args = getattr(fn, "arguments", "") or ""
            args, parse_error = _parse_arguments(raw_args)
            tool_calls.append(
                ToolCall(
                    id=str(getattr(raw_call, "id", "") or f"call_{len(tool_calls)}"),
                    name=str(getattr(fn, "name", "") or ""),
                    arguments=args,
                    raw_arguments=raw_args,
                    parse_error=parse_error,
                )
            )
        usage_obj = getattr(response, "usage", None)
        usage = {}
        if usage_obj is not None:
            usage = {
                "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
            }
        return LLMResponse(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            finish_reason=str(getattr(choice, "finish_reason", "stop") or "stop"),
            usage=usage,
            model=model,
        )


def _parse_arguments(raw: str) -> tuple[dict[str, Any], str]:
    """Parse tool-call arguments, tolerating provider quirks."""
    text = (raw or "").strip()
    if not text:
        return {}, ""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, ""
        return {"value": parsed}, ""
    except Exception:
        pass
    try:
        import json5  # type: ignore

        parsed = json5.loads(text)
        if isinstance(parsed, dict):
            return parsed, ""
        return {"value": parsed}, ""
    except Exception as exc:
        return {}, f"argument JSON parse failed: {exc}"


def _status_code(exc: Exception | None) -> int | None:
    if exc is None:
        return None
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    match = re.search(r"\b(4\d{2}|5\d{2})\b", str(exc))
    return int(match.group(1)) if match else None


def _error_body(exc: Exception | None) -> str:
    if exc is None:
        return ""
    body = getattr(exc, "body", None)
    if body is not None:
        return json.dumps(body, ensure_ascii=False) if not isinstance(body, str) else body
    response = getattr(exc, "response", None)
    if response is not None:
        text = getattr(response, "text", "")
        if text:
            return str(text)
    return str(exc)


def default_config_path(workspace: str | Path | None = None) -> Path:
    """Resolve the engine config path (workspace first, then repo config)."""
    if workspace is not None:
        candidate = Path(workspace) / "config" / "llm_config.json"
        if candidate.exists():
            return candidate
    env_path = os.environ.get("GIS_LLM_CONFIG")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[3] / "config" / "llm_config.json"

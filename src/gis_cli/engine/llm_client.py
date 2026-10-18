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
    # 抗网关抖动：部分网关的负载均衡会把长连接踩到坏后端，表现为成串 504；
    # 关掉 keep-alive 让每次请求重新建连（默认 True，保持原行为）。
    http_keepalive: bool = True
    # 思考档位（reasoning_effort）。实测部分推理模型默认**不返回** reasoning_content，
    # 必须显式传该字段才有思考内容；空串 = 不传（兼容不支持它的网关）。
    reasoning_effort: str = ""
    #: 按任务类型覆盖思考档位，如 ``{"agent": "low", "planning": "medium"}``。
    #: 实测（deepseek-v4.1-flash-intl）交互轮换 low：首字 5.11s→2.37s、
    #: 总耗时 8.72s→5.80s，而输出质量相当；规划/验收这类“想清楚更重要”的
    #: 环节保持默认档位。空字典 = 全部用 reasoning_effort。
    reasoning_effort_by_task: dict[str, str] = field(default_factory=dict)
    # 流式输出：模型边生成边把正文/思考推给界面（用户不必盯着空屏等整段跑完）。
    # 组装后的返回值与非流式完全一致，调用方无感；网关不支持时自动降级。
    stream: bool = True

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
            reasoning_effort=str(
                engine.get("reasoning_effort", data.get("reasoning_effort", "")) or ""
            ).strip(),
            reasoning_effort_by_task={
                str(k).strip().lower(): str(v).strip()
                for k, v in (engine.get("reasoning_effort_by_task", {}) or {}).items()
                if str(v).strip()
            },
            stream=bool(engine.get("stream", data.get("stream", True))),
            http_keepalive=bool(data.get("http_keepalive", engine.get("http_keepalive", True))),
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

    def effort_for(self, task_type: str | None = None) -> str:
        """该任务类型的思考档位（按任务覆盖优先，否则用全局值）。"""
        key = (task_type or "").strip().lower()
        if key:
            override = self.reasoning_effort_by_task.get(key)
            if override:
                return override
        return (self.reasoning_effort or "").strip()

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


class _StreamUnsupported(RuntimeError):
    """上游不真的支持流式（忽略 stream 参数或直接报错）——内部的回退信号。"""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


def _usage_dict(raw_usage: Any) -> dict[str, int]:
    """把 SDK 的 usage 对象转成 dict（多取前缀缓存字段，供缓存命中率统计）。"""
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        try:
            out[key] = int(getattr(raw_usage, key, 0) or 0)
        except Exception:
            out[key] = 0
    for key in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "cached_tokens"):
        value = getattr(raw_usage, key, None)
        if value is not None:
            try:
                out[key] = int(value)
            except Exception:
                pass
    return out


def _looks_like_complete_response(obj: Any) -> bool:
    """判断上游给的是"完整响应"而不是 chunk 迭代器。

    为什么要显式判：SDK 的响应对象是 pydantic 模型（带 ``__iter__``），
    拿它去 ``for`` 会迭代出字段元组，看起来"有值"但绝不是 chunk；
    以前这种误判会静默地组出一个空回答。
    """
    if obj is None:
        return False
    # 迭代器/生成器就是我们要的流
    if hasattr(obj, "__next__"):
        return False
    # 已解析的完整响应：有 choices 且首个 choice 带 message（而非 delta）
    choices = getattr(obj, "choices", None)
    if choices:
        first = choices[0]
        if hasattr(first, "message"):
            return True
        if hasattr(first, "delta"):
            return False
    # 没法判断时：能当迭代用就当流
    return not hasattr(obj, "__iter__")


def _emit_delta(on_delta: Any, kind: str, text: str) -> None:
    """回调界面（异常一律吞掉：渲染绝不能把模型调用搞挂）。"""
    if on_delta is None:
        return
    try:
        on_delta(kind, text)
    except Exception:  # pragma: no cover - 渲染层问题不影响请求
        pass


def _timing_suffix(response: Any) -> str:
    """日志后缀：token 用量、是否流式、首字之前的等待看不见就写总量。"""
    try:
        usage = getattr(response, "usage", None) or {}
        if not isinstance(usage, dict):
            usage = {}
        raw = getattr(response, "raw", None) or {}
        bits = [f"in={usage.get('prompt_tokens', 0)}", f"out={usage.get('completion_tokens', 0)}"]
        cached = usage.get("prompt_cache_hit_tokens") or usage.get("cached_tokens") or 0
        if cached:
            bits.append(f"cached={cached}")
        if isinstance(raw, dict) and raw.get("streamed"):
            bits.append(f"chars={raw.get('emitted_chars', 0)}")
        return "  " + " ".join(bits)
    except Exception:  # pragma: no cover
        return ""


class EngineLLMClient:
    """Single-flight, retrying chat client with tool-calling support."""

    def __init__(self, config: EngineLLMConfig):
        self.config = config
        self._client: Any = None
        self._lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(max(1, config.max_concurrency))
        self._consecutive_429 = 0
        #: 用量统计：``{model: {calls, prompt_tokens, completion_tokens,
        #: cached_tokens, failures}}``。
        #:
        #: 为什么要聚合：以前只在日志后缀里写 ``cached=N``，跑完就没了 ——
        #: 无法回答“这个任务花了多少 token / 前缀缓存命中率多少 / 哪个模型
        #: 在烧钱”。聚合后由 loop 在 ``run_end`` 写进 trace，
        #: ``scripts/latency_report.py --cost`` 可汇总。
        self.usage_totals: dict[str, dict[str, int]] = {}
        #: 上游拒过图（400）后置位：调用方据此不再附图，省下每轮的无效请求
        self.vision_rejected = False
        #: 上游不吃流式（代理忽略 stream 或直接报错）后置位：后续一律走非流式
        self._stream_unsupported = False
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
            kwargs: dict[str, Any] = {}
            if not bool(getattr(self.config, "http_keepalive", True)):
                # 抗网关抖动：禁用长连接复用，避免连接池粘在被 LB 判死的后端上
                try:
                    import httpx

                    kwargs["http_client"] = httpx.Client(
                        timeout=self.config.timeout,
                        limits=httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0.0),
                    )
                except Exception:
                    pass
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base or None,
                timeout=self.config.timeout,
                max_retries=0,  # we own retries
                **kwargs,
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
        on_delta: Any = None,
    ) -> LLMResponse:
        """Send a chat completion, retrying transient failures.

        ``on_delta(kind, text)`` 在流式模式下被逐块调用（``kind`` 为
        ``"content"``/``"reasoning"``），供界面边生成边显示；不传则静默累积。
        **返回值与非流式逐字段一致**，调用方无需关心是否流式。

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
                    on_delta=on_delta,
                    task_type=task_type,
                )
                self._record_success(candidate)
                self._record_usage(candidate, response)
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

    # -------------------------------------------------------------- 用量统计
    def _record_usage(self, model: str, response: Any) -> None:
        """把一次成功调用的 token 用量累加到 ``usage_totals``。"""
        usage = getattr(response, "usage", None)
        if not isinstance(usage, dict):
            usage = {}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        cached = int(
            usage.get("prompt_cache_hit_tokens") or usage.get("cached_tokens") or 0
        )
        with self._lock:
            slot = self.usage_totals.setdefault(
                model,
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0},
            )
            slot["calls"] += 1
            slot["prompt_tokens"] += prompt
            slot["completion_tokens"] += completion
            slot["cached_tokens"] += cached

    def usage_summary(self) -> dict[str, Any]:
        """汇总用量（含缓存命中率）。无数据时返回空结构。

        命中率 = 命中缓存的 prompt token / 全部 prompt token。
        它直接对应钱：命中的部分通常按更低的价计费，命中率跌下来就说明
        提示词前缀在变（系统提示、数据目录摘要、日志摘要被改动）。
        """
        with self._lock:
            totals = {name: dict(values) for name, values in self.usage_totals.items()}
        prompt_all = sum(item["prompt_tokens"] for item in totals.values())
        cached_all = sum(item["cached_tokens"] for item in totals.values())
        return {
            "by_model": totals,
            "prompt_tokens": prompt_all,
            "completion_tokens": sum(i["completion_tokens"] for i in totals.values()),
            "cached_tokens": cached_all,
            "calls": sum(i["calls"] for i in totals.values()),
            "cache_hit_rate": (cached_all / prompt_all) if prompt_all else 0.0,
        }

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
        on_delta: Any = None,
        task_type: str | None = None,
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
        effort = self.config.effort_for(task_type)
        if effort:
            # 走 extra_body 而不是顶层：OpenAI SDK 对未知顶层字段会直接拒绝，
            # 而各家 OpenAI 兼容网关都从 body 里读这个字段。
            payload["extra_body"] = {"reasoning_effort": effort}

        last_error: Exception | None = None
        conn_failures = 0
        budget = float(self.config.total_timeout)
        if self._cooldown_strikes.get(model):
            # 该模型本回合已经超时/断连过 → 缩短单次尝试预算，快速失败换下一个模型
            budget = min(budget, float(self.config.reduced_timeout_seconds or budget))
        deadline = time.monotonic() + budget
        budget_exhausted = False
        vision_stripped = False
        for attempt in range(1, self.config.retry_count + 1):
            remaining_budget = deadline - time.monotonic()
            if remaining_budget <= 0:
                budget_exhausted = True
                break
            with self._semaphore:
                try:
                    started = time.perf_counter()
                    if self.config.stream and not self._stream_unsupported:
                        try:
                            response = self._stream_once(
                                payload, model, deadline=deadline, on_delta=on_delta
                            )
                        except _StreamUnsupported as exc:
                            # 网关不支持流式（有些代理直接忽略 stream 参数或报错）：
                            # 关掉后续流式尝试，本回合用非流式重来一次。
                            self._stream_unsupported = True
                            logger.info(
                                "上游不支持流式（%s），改用非流式：%s",
                                getattr(exc, "status", None),
                                str(exc)[:160],
                            )
                            continue
                    else:
                        response = self._normalize(
                            self.client.chat.completions.create(
                                **payload, timeout=max(30.0, remaining_budget)
                            ),
                            model,
                        )
                    self._consecutive_429 = 0
                    logger.info(
                        "[timing] llm.%-18s %7.2fs%s",
                        getattr(response, "model", model),
                        time.perf_counter() - started,
                        _timing_suffix(response),
                    )
                    return response
                except LLMError:
                    raise
                except Exception as exc:  # provider-specific exception types
                    status = _status_code(exc)
                    body = _error_body(exc)
                    retryable = status is None or status in _RETRYABLE_STATUS
                    if not retryable:
                        # 识图是**可选增强**，不能让附图把整个任务拖死。实测上游会拒图：
                        # 400 "Please start a new conversation, replace the image, and try again"，
                        # 以前直接抛 LLMError → 整轮 run 判 llm_unavailable（14 届 4a-1 实测，
                        # 只跑了 7 轮就死，而栅格产物其实已经做完了）。
                        # 这里把图摘掉重试一次：文字与统计数据还在，任务可以继续。
                        if not vision_stripped and _has_images(payload["messages"]):
                            payload["messages"] = _strip_images(payload["messages"])
                            vision_stripped = True
                            # 记住“上游不吃图”：否则每轮附图都要白碰一次 400。
                            self.vision_rejected = True
                            logger.warning(
                                "上游拒绝图片（%s），已摘掉图片重试：%s", status, (body or "")[:160]
                            )
                            continue
                        # 网关直接拒 stream 参数（400/422）时不能让任务死：
                        # 关掉流式并用同一个模型重试一次（非流式语义完全一样，
                        # 只是拿不到逐字输出）。
                        if self.config.stream and not self._stream_unsupported:
                            self._stream_unsupported = True
                            logger.warning(
                                "上游拒绝流式（%s），改用非流式重试：%s", status, (body or "")[:160]
                            )
                            continue
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

    # ------------------------------------------------------------------ 流式
    def _stream_once(
        self,
        payload: dict[str, Any],
        model: str,
        *,
        deadline: float,
        on_delta: Any = None,
    ) -> LLMResponse:
        """流式请求，边收边回调，最终组装成与非流式等价的 LLMResponse。

        兼容要点（实测 workbuddy 反代 / CodeBuddy 上游）：
        - ``delta.content`` / ``delta.reasoning_content`` 逐块到达；
        - ``delta.tool_calls`` 分片：首块带 ``id``/``name``，后续块只补 ``arguments``
          片段且可能不带 ``index`` 以外字段 —— 必须按 ``index`` 累积拼接；
        - ``usage`` 只出现在末尾那个 chunk（有时只在 ``[DONE]`` 前最后一个）。

        上游忽略 ``stream`` 参数（当成非流式返回）时抬 ``_StreamUnsupported``，
        由调用方回退到非流式重试。
        """
        remaining = max(30.0, deadline - time.monotonic())
        stream = self.client.chat.completions.create(
            **payload, stream=True, timeout=remaining
        )

        if _looks_like_complete_response(stream):
            # 上游把 stream 参数忽略了，直接返回完整响应。
            # 这里必须显式识别：SDK 的响应对象是 pydantic 模型，带 __iter__，
            # 直接 for 循环会静默地迭代出 (字段名, 值) 元组而不是 chunk。
            # 既然完整结果已经拿到，就按非流式处理（不再多发一次请求）。
            self._stream_unsupported = True
            logger.info("上游忽略了 stream 参数（返回完整响应），按非流式处理")
            return self._normalize(stream, model)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        # index -> {id, name, arguments}；arguments 是分片拼起来的字符串
        calls: dict[int, dict[str, str]] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}
        emitted = 0
        saw_chunk = False
        for chunk in stream:
            saw_chunk = True
            raw_usage = getattr(chunk, "usage", None)
            if raw_usage is not None:
                usage = _usage_dict(raw_usage)
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = str(choice.finish_reason)
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            text = getattr(delta, "content", None)
            if text:
                content_parts.append(str(text))
                emitted += len(str(text))
                _emit_delta(on_delta, "content", str(text))
            think = getattr(delta, "reasoning_content", None)
            if think:
                reasoning_parts.append(str(think))
                emitted += len(str(think))
                _emit_delta(on_delta, "reasoning", str(think))
            for raw_call in getattr(delta, "tool_calls", None) or []:
                index = int(getattr(raw_call, "index", 0) or 0)
                slot = calls.setdefault(
                    index, {"id": "", "name": "", "arguments": ""}
                )
                call_id = getattr(raw_call, "id", None)
                if call_id:
                    slot["id"] = str(call_id)
                fn = getattr(raw_call, "function", None)
                if fn is not None:
                    name = getattr(fn, "name", None)
                    if name:
                        slot["name"] = str(name)
                    args = getattr(fn, "arguments", None)
                    if args:
                        slot["arguments"] += str(args)
        if not saw_chunk:
            # 一个 chunk 都没收到：上游可能把 stream 忽略了。交给非流式重试。
            raise _StreamUnsupported("stream returned no chunks (upstream may ignore 'stream')")

        tool_calls: list[ToolCall] = []
        for index in sorted(calls):
            slot = calls[index]
            args, parse_error = _parse_arguments(slot["arguments"])
            tool_calls.append(
                ToolCall(
                    id=slot["id"] or f"call_{index}",
                    name=slot["name"],
                    arguments=args,
                    raw_arguments=slot["arguments"],
                    parse_error=parse_error,
                )
            )
        return LLMResponse(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=model,
            raw={"streamed": True, "emitted_chars": emitted},
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
        usage: dict[str, int] = {}
        if usage_obj is not None:
            usage = _usage_dict(usage_obj)
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


def _has_images(messages: list[dict[str, Any]]) -> bool:
    """对话里有没有多模态图片块。"""
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "image_url" for part in content
        ):
            return True
    return False


def _strip_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把对话里的图片块换成一行文字说明，保留其余文本。

    上游拒图时用：摘掉图片后对话仍然完整（文字、工具结果、统计数据都在），
    只是模型看不到图。比让整个 run 判 llm_unavailable 强得多。
    """
    stripped: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            stripped.append(message)
            continue
        texts = [
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        image_count = sum(
            1 for part in content if isinstance(part, dict) and part.get("type") == "image_url"
        )
        if image_count:
            texts.append(f"（{image_count} 张图片已因上游不支持而省略，请依据文本与统计结果判断）")
        merged = "\n".join(t for t in texts if t).strip()
        stripped.append({**message, "content": merged or "（图片已省略）"})
    return stripped


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

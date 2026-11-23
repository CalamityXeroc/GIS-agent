# -*- coding: utf-8 -*-
"""Anthropic Messages API 适配层。

引擎内部的对话一律用 OpenAI 形状（``role`` / ``tool_calls`` / ``role="tool"``），
而 Anthropic 官方接口是另一套协议：

- 系统提示单独一个 ``system`` 字段，不进 ``messages``；
- 工具调用是 ``assistant`` 里的 ``tool_use`` 块，结果回填成 ``user`` 里的
  ``tool_result`` 块（且连续的多个结果**必须合并进同一条 user 消息**）；
- 工具声明用 ``input_schema``（不是 ``parameters``），``tool_choice`` 取值是
  ``auto`` / ``any`` / ``tool``；
- ``max_tokens`` 必填，``stop_reason`` 取值是 ``end_turn`` / ``tool_use`` / ``max_tokens``；
- 流式是 SSE：``message_start`` → ``content_block_start`` / ``content_block_delta`` /
  ``content_block_stop`` → ``message_delta`` → ``message_stop``。

本模块只做「形状转换 + HTTP + 解析」，不 import ``llm_client``（避免循环依赖），
统一返回普通 dict，由调用方包成 ``LLMResponse``。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

logger = logging.getLogger(__name__)

#: Anthropic 要求的版本头（2023-06-01 至今稳定）。
ANTHROPIC_VERSION = "2023-06-01"

#: 官方默认地址；``/v1/messages`` 由本模块拼。
DEFAULT_BASE_URL = "https://api.anthropic.com"

#: 未显式给 max_tokens 时的兜底（Anthropic 必填）。
DEFAULT_MAX_TOKENS = 8192

#: 官方文档里的输出上限（Claude 3.5/3.7/4 系列单次最多 8192~64000，这里只用于提示）。
_MAX_TOKENS_CEILING = 64000

#: stop_reason → 引擎内部（OpenAI 语义）的 finish_reason。
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


class AnthropicHTTPError(RuntimeError):
    """上游返回非 2xx。

    带 ``status_code`` / ``body`` 是为了复用 ``llm_client`` 里既有的错误分类
    （401/403 判链路级、429/5xx 可重试）。
    """

    def __init__(self, message: str, *, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def normalize_base_url(base: str | None) -> str:
    """把用户填的各种写法收敛成 ``scheme://host[:port][/prefix]``（不带 ``/v1``）。

    - 空 → 官方地址；
    - 结尾的 ``/v1``、``/v1/messages``、``/`` 一律剥掉（用户很容易照 OpenAI 的习惯填）；
    - 自建反代保留路径前缀。
    """
    text = (base or "").strip() or DEFAULT_BASE_URL
    text = text.rstrip("/")
    for suffix in ("/v1/messages", "/messages", "/v1"):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)].rstrip("/")
            break
    return text


def is_anthropic_base(base: str | None) -> bool:
    """地址看起来是不是 Anthropic（官方或自建反代里带 anthropic 字样）。"""
    text = (base or "").strip().lower()
    if not text:
        return False
    return "anthropic" in text or "claude" in text


# --------------------------------------------------------------------- 请求侧


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                chunks.append(str(part.get("text") or ""))
        return "".join(chunks)
    return "" if content is None else str(content)


def _image_block(url: str) -> dict[str, Any] | None:
    """``data:image/png;base64,...`` → base64 source；http(s) → url source。"""
    text = (url or "").strip()
    if not text:
        return None
    match = re.match(r"^data:(image/[A-Za-z0-9.+-]+);base64,(.*)$", text, re.S)
    if match:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": match.group(1).lower(),
                "data": match.group(2),
            },
        }
    if text.startswith(("http://", "https://")):
        return {"type": "image", "source": {"type": "url", "url": text}}
    return None


def _user_blocks(content: Any) -> list[dict[str, Any]]:
    """user 消息的 content → Anthropic 内容块列表。"""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": _text_of(content)}]
    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            blocks.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            blocks.append({"type": "text", "text": str(part)})
            continue
        ptype = part.get("type")
        if ptype == "text":
            blocks.append({"type": "text", "text": str(part.get("text") or "")})
        elif ptype == "image_url":
            url = part.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            block = _image_block(str(url or ""))
            if block:
                blocks.append(block)
        elif ptype == "image" and isinstance(part.get("source"), dict):
            # 已经是 Anthropic 形状，原样保留（幂等）
            blocks.append(part)
        else:
            # 未知块（例如网关私有字段）降级成文本，不丢内容也不让上游 400
            blocks.append({"type": "text", "text": _text_of(part)})
    return blocks or [{"type": "text", "text": ""}]


def _tool_use_block(raw_call: dict[str, Any], index: int) -> dict[str, Any]:
    fn = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else {}
    raw_args = fn.get("arguments")
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args) if raw_args.strip() else {}
        except Exception:
            parsed = {"_raw": raw_args}
    elif isinstance(raw_args, dict):
        parsed = raw_args
    else:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {"value": parsed}
    return {
        "type": "tool_use",
        "id": str(raw_call.get("id") or f"toolu_{index}"),
        "name": str(fn.get("name") or raw_call.get("name") or ""),
        "input": parsed,
    }


def _tool_result_block(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": str(message.get("tool_call_id") or ""),
        "content": content if isinstance(content, (str, list)) else _text_of(content),
    }
    if message.get("is_error"):
        block["is_error"] = True
    return block


def _merge_adjacent(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并相邻同角色消息；Anthropic 要求 user/assistant 交替。"""
    merged: list[dict[str, Any]] = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"]:
            prev = merged[-1]
            prev_blocks = prev["content"] if isinstance(prev["content"], list) else [
                {"type": "text", "text": str(prev["content"])}
            ]
            cur_blocks = message["content"] if isinstance(message["content"], list) else [
                {"type": "text", "text": str(message["content"])}
            ]
            prev["content"] = [*prev_blocks, *cur_blocks]
            continue
        merged.append(message)
    # 首条必须是 user：Anthropic 不接受 assistant 开场
    if merged and merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": [{"type": "text", "text": "(continue)"}]})
    return merged


def to_anthropic_messages(messages: Iterable[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """OpenAI 形状的 messages → ``(system_text, anthropic_messages)``。"""
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for message in messages or []:
        role = str(message.get("role") or "user")
        if role in ("system", "developer"):
            text = _text_of(message.get("content")).strip()
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            block = _tool_result_block(message)
            last = converted[-1] if converted else None
            # 连续工具结果必须并进同一条 user 消息，否则上游报
            # "tool_result blocks must be in a user message"
            if (
                last
                and last["role"] == "user"
                and isinstance(last["content"], list)
                and all(b.get("type") == "tool_result" for b in last["content"])
            ):
                last["content"].append(block)
            else:
                converted.append({"role": "user", "content": [block]})
            continue
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            text = _text_of(message.get("content"))
            if text.strip():
                blocks.append({"type": "text", "text": text})
            for index, raw_call in enumerate(message.get("tool_calls") or []):
                if isinstance(raw_call, dict):
                    blocks.append(_tool_use_block(raw_call, index))
            if not blocks:
                # 空 assistant 消息会被上游拒；给一个空格占位
                blocks.append({"type": "text", "text": " "})
            converted.append({"role": "assistant", "content": blocks})
            continue
        converted.append({"role": "user", "content": _user_blocks(message.get("content"))})
    return "\n\n".join(system_parts).strip(), _merge_adjacent(converted)


def to_anthropic_tools(tools: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """OpenAI 工具声明 → Anthropic ``{name, description, input_schema}``。"""
    out: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        schema = fn.get("parameters")
        if not isinstance(schema, dict) or not schema:
            schema = {"type": "object", "properties": {}}
        out.append(
            {
                "name": name,
                "description": str(fn.get("description") or ""),
                "input_schema": schema,
            }
        )
    return out


def to_anthropic_tool_choice(choice: Any) -> dict[str, Any] | None:
    """``auto`` / ``required`` / ``none`` / ``{"type":"function",...}`` → Anthropic 形状。"""
    if choice is None:
        return None
    if isinstance(choice, str):
        lowered = choice.strip().lower()
        if lowered == "auto":
            return {"type": "auto"}
        if lowered in ("required", "any"):
            return {"type": "any"}
        if lowered == "none":
            return None
        return {"type": "auto"}
    if isinstance(choice, dict):
        fn = choice.get("function") if isinstance(choice.get("function"), dict) else {}
        name = str(fn.get("name") or choice.get("name") or "").strip()
        if name:
            return {"type": "tool", "name": name}
        kind = str(choice.get("type") or "").strip().lower()
        if kind in ("auto", "any", "tool"):
            return {"type": kind}
    return {"type": "auto"}


def build_request(
    *,
    model: str,
    messages: Iterable[dict[str, Any]],
    tools: Iterable[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    temperature: float | None = 0.2,
    max_tokens: int | None = None,
    stream: bool = True,
) -> dict[str, Any]:
    """拼 ``POST /v1/messages`` 的 body。"""
    system_text, anth_messages = to_anthropic_messages(messages)
    payload: dict[str, Any] = {
        "model": model,
        "messages": anth_messages,
        "max_tokens": int(max_tokens or DEFAULT_MAX_TOKENS),
    }
    if system_text:
        payload["system"] = system_text
    if temperature is not None:
        payload["temperature"] = float(temperature)
    anth_tools = to_anthropic_tools(tools)
    if anth_tools:
        payload["tools"] = anth_tools
        choice = to_anthropic_tool_choice(tool_choice)
        if choice:
            payload["tool_choice"] = choice
    if stream:
        payload["stream"] = True
    return payload


# --------------------------------------------------------------------- 响应侧


def _usage_of(raw: dict[str, Any] | None) -> dict[str, int]:
    """Anthropic usage → 引擎内部 usage（OpenAI 口径）。"""
    if not isinstance(raw, dict):
        return {}
    prompt = int(raw.get("input_tokens") or 0)
    completion = int(raw.get("output_tokens") or 0)
    cached = int(raw.get("cache_read_input_tokens") or 0)
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if cached:
        usage["cached_tokens"] = cached
    return usage


def _finish_reason(stop_reason: str | None) -> str:
    return _STOP_REASON_MAP.get(str(stop_reason or "end_turn"), str(stop_reason or "stop"))


def parse_response(data: dict[str, Any]) -> dict[str, Any]:
    """非流式响应 → 引擎内部形状。"""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            content_parts.append(str(block.get("text") or ""))
        elif btype == "thinking":
            reasoning_parts.append(str(block.get("thinking") or ""))
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id") or ""),
                    "name": str(block.get("name") or ""),
                    "arguments": block.get("input") if isinstance(block.get("input"), dict) else {},
                }
            )
    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "tool_calls": tool_calls,
        "finish_reason": _finish_reason(data.get("stop_reason")),
        "usage": _usage_of(data.get("usage")),
        "model": str(data.get("model") or ""),
    }


def parse_stream(lines: Iterable[str], on_delta: Any = None) -> dict[str, Any]:
    """SSE 行 → 引擎内部形状（正文/思考逐块回调 ``on_delta``）。

    容错：``ping`` / 未知事件忽略；单行 JSON 坏了不中断整条流。
    """
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, int] = {}
    finish_reason = "stop"
    model = ""
    #: 正在拼的 tool_use：``{index: {"id","name","json"}}``
    pending: dict[int, dict[str, Any]] = {}
    block_index = -1

    for raw_line in lines:
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line or line.startswith(":") or line.startswith("event:"):
            continue
        if not line.startswith("data:"):
            continue
        data_text = line[5:].strip()
        if not data_text or data_text == "[DONE]":
            continue
        try:
            event = json.loads(data_text)
        except Exception:
            logger.debug("跳过无法解析的 SSE 行：%s", data_text[:120])
            continue
        etype = str(event.get("type") or "")
        if etype == "message_start":
            message = event.get("message") or {}
            model = str(message.get("model") or model)
            start_usage = _usage_of(message.get("usage"))
            if start_usage:
                usage.update(start_usage)
        elif etype == "content_block_start":
            block_index = int(event.get("index", block_index + 1))
            block = event.get("content_block") or {}
            btype = block.get("type")
            if btype == "tool_use":
                pending[block_index] = {
                    "id": str(block.get("id") or ""),
                    "name": str(block.get("name") or ""),
                    "json": "",
                }
            elif btype == "thinking":
                # thinking 块的起始可能自带文本
                initial = str(block.get("thinking") or "")
                if initial:
                    reasoning.append(initial)
                    if on_delta:
                        on_delta("reasoning", initial)
        elif etype == "content_block_delta":
            index = int(event.get("index", block_index))
            delta = event.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                text = str(delta.get("text") or "")
                if text:
                    content.append(text)
                    if on_delta:
                        on_delta("content", text)
            elif dtype == "thinking_delta":
                text = str(delta.get("thinking") or "")
                if text:
                    reasoning.append(text)
                    if on_delta:
                        on_delta("reasoning", text)
            elif dtype == "input_json_delta":
                slot = pending.setdefault(index, {"id": "", "name": "", "json": ""})
                slot["json"] += str(delta.get("partial_json") or "")
        elif etype == "content_block_stop":
            index = int(event.get("index", block_index))
            slot = pending.pop(index, None)
            if slot is not None:
                raw_json = slot.get("json") or ""
                try:
                    parsed = json.loads(raw_json) if raw_json.strip() else {}
                except Exception:
                    parsed = {"_raw": raw_json}
                if not isinstance(parsed, dict):
                    parsed = {"value": parsed}
                tool_calls.append(
                    {"id": slot.get("id") or "", "name": slot.get("name") or "", "arguments": parsed}
                )
        elif etype == "message_delta":
            delta = event.get("delta") or {}
            if delta.get("stop_reason"):
                finish_reason = _finish_reason(delta.get("stop_reason"))
            delta_usage = _usage_of(event.get("usage"))
            if delta_usage:
                # 输出 token 只在 message_delta 里给全；输入 token 保留 message_start 的值
                merged = dict(usage)
                merged.update({k: v for k, v in delta_usage.items() if v})
                if merged.get("prompt_tokens") and merged.get("completion_tokens"):
                    merged["total_tokens"] = merged["prompt_tokens"] + merged["completion_tokens"]
                usage = merged
        elif etype == "error":
            error = event.get("error") or {}
            raise AnthropicHTTPError(
                f"stream error: {error.get('type')}: {error.get('message')}",
                status_code=int(error.get("status") or 0) or None,
                body=json.dumps(error, ensure_ascii=False),
            )

    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "usage": usage,
        "model": model,
    }


# ------------------------------------------------------------------ HTTP 客户端


class AnthropicAdapter:
    """最小 HTTP 适配器：只依赖 ``httpx``（openai 已传递依赖它）。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 900.0,
        http_client: Any = None,
    ):
        self.api_key = api_key or ""
        self.base_url = normalize_base_url(base_url)
        self.timeout = float(timeout or 900.0)
        self._client = http_client
        self._lock = None

    # -- 内部 ---------------------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/v1/messages"

    def headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
            "accept": "application/json",
        }

    @staticmethod
    def _raise_for_status(response: Any) -> None:
        status = int(getattr(response, "status_code", 0) or 0)
        if status < 400:
            return
        try:
            body = response.text
        except Exception:  # pragma: no cover - 假响应可能没有 text
            body = ""
        raise AnthropicHTTPError(
            f"Anthropic HTTP {status}: {(body or '')[:300]}",
            status_code=status,
            body=body,
        )

    # -- 对外 ---------------------------------------------------------------
    def complete(
        self,
        *,
        messages: Iterable[dict[str, Any]],
        model: str,
        tools: Iterable[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        temperature: float | None = 0.2,
        max_tokens: int | None = None,
        stream: bool = True,
        on_delta: Any = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """发一次请求，返回引擎内部形状（不重试，重试由上层负责）。"""
        want_stream = bool(stream)
        payload = build_request(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=want_stream,
        )
        call_timeout = float(timeout or self.timeout)
        if not want_stream:
            response = self.client.post(
                self.endpoint, headers=self.headers(), json=payload, timeout=call_timeout
            )
            self._raise_for_status(response)
            data = response.json()
            parsed = parse_response(data)
            parsed["model"] = parsed.get("model") or model
            return parsed
        with self.client.stream(
            "POST", self.endpoint, headers=self.headers(), json=payload, timeout=call_timeout
        ) as response:
            self._raise_for_status(response)
            parsed = parse_stream(response.iter_lines(), on_delta)
        parsed["model"] = parsed.get("model") or model
        return parsed

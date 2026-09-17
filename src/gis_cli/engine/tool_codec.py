# -*- coding: utf-8 -*-
"""Tool-call protocol adapters: native function calling with JSON fallback.

Users may run different models; not all support OpenAI-style tool calling.
``AutoCodec`` probes native support and falls back to a strict JSON action
protocol when needed, caching the decision per model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .llm_client import EngineLLMClient, LLMResponse, ToolCall


class Protocol(str, Enum):
    AUTO = "auto"
    NATIVE = "native"
    JSON = "json"


@dataclass
class Action:
    """A normalized action requested by the model."""

    kind: str  # "tool" | "final"
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    tool_call_id: str = ""
    reasoning: str = ""
    parse_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "tool": self.tool,
            "args": self.args,
            "content": self.content,
            "reasoning": self.reasoning,
            "parse_error": self.parse_error,
        }


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from free-form text."""
    if not text:
        return None
    candidates: list[str] = []
    for match in _JSON_BLOCK.finditer(text):
        candidates.append(match.group(1))
    stripped = text.strip()
    if stripped.startswith("{"):
        candidates.append(stripped)
    # Last resort: first balanced braces span.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        parsed = _parse_json(candidate)
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        import json5  # type: ignore

        return json5.loads(text)
    except Exception:
        return None


def actions_from_native(response: LLMResponse) -> list[Action]:
    """Convert native tool calls to actions."""
    actions: list[Action] = []
    for call in response.tool_calls:
        actions.append(
            Action(
                kind="tool",
                tool=call.name,
                args=call.arguments if isinstance(call.arguments, dict) else {},
                tool_call_id=call.id,
                reasoning=response.reasoning[:2000],
                parse_error=call.parse_error,
            )
        )
    if not actions and response.content.strip():
        # Model answered without acting; treat as a final message.
        actions.append(Action(kind="final", content=response.content, reasoning=response.reasoning[:2000]))
    return actions


def actions_from_json(response: LLMResponse) -> list[Action]:
    """Parse the JSON action protocol."""
    payload = extract_json_object(response.content)
    if payload is None:
        return [
            Action(
                kind="final",
                content=response.content,
                reasoning=response.reasoning[:2000],
                parse_error="no JSON action found",
            )
        ]
    # Support both {"tool": ...} and {"action": ...}
    tool = payload.get("tool") or payload.get("action") or ""
    args = payload.get("args") or payload.get("arguments") or {}
    if isinstance(args, str):
        parsed_args = _parse_json(args)
        args = parsed_args if isinstance(parsed_args, dict) else {}
    if not isinstance(args, dict):
        args = {}
    # Merge top-level keys (some models put params at top level).
    if not args:
        reserved = {"tool", "action", "args", "arguments", "thought", "reasoning", "content", "message"}
        args = {k: v for k, v in payload.items() if k not in reserved}
    if tool:
        return [
            Action(
                kind="tool",
                tool=str(tool),
                args=args,
                content=str(payload.get("message") or payload.get("content") or ""),
                reasoning=str(payload.get("thought") or payload.get("reasoning") or "")[:2000],
            )
        ]
    content = str(payload.get("message") or payload.get("content") or response.content)
    return [Action(kind="final", content=content, reasoning=str(payload.get("reasoning") or "")[:2000])]


class AutoCodec:
    """Model-adaptive action decoder."""

    def __init__(self, client: EngineLLMClient, protocol: Protocol | str = Protocol.AUTO):
        self.client = client
        self.protocol = Protocol(protocol) if not isinstance(protocol, Protocol) else protocol
        self._native_failures: dict[str, int] = {}
        self._resolved: dict[str, Protocol] = {}

    # ------------------------------------------------------------- protocol
    def protocol_for(self, model: str) -> Protocol:
        if self.protocol != Protocol.AUTO:
            return self.protocol
        resolved = self._resolved.get(model)
        if resolved is not None:
            return resolved
        if self._native_failures.get(model, 0) >= 2:
            self._resolved[model] = Protocol.JSON
            return Protocol.JSON
        return Protocol.NATIVE

    def note_native_failure(self, model: str) -> None:
        self._native_failures[model] = self._native_failures.get(model, 0) + 1

    # ----------------------------------------------------------------- call
    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        task_type: str = "agent",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[LLMResponse, list[Action]]:
        """Call the model and decode actions."""
        probe_model = model or self.client.config.model
        protocol = self.protocol_for(probe_model)

        if protocol == Protocol.NATIVE:
            response = self.client.chat(
                messages,
                tools=tools,
                task_type=task_type,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            actions = actions_from_native(response)
            if not response.tool_calls and response.content.strip():
                self.note_native_failure(response.model)
            return response, actions

        json_messages = _inject_json_protocol(messages, tools)
        response = self.client.chat(
            json_messages,
            task_type=task_type,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response, actions_from_json(response)


def _inject_json_protocol(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Add the JSON action protocol to the system message."""
    schema_text = json.dumps(tools, ensure_ascii=False, indent=2)
    instructions = (
        "\n\n## 动作协议（必须遵守）\n"
        "你只能通过输出一个 JSON 对象来行动，不要输出其他内容。\n"
        "调用工具时输出：{\"tool\": \"<工具名>\", \"args\": {...}, \"thought\": \"<简述理由>\"}\n"
        "任务完成时输出：{\"tool\": \"finish\", \"args\": {\"summary\": \"...\", \"methodology\": \"...\", \"artifacts\": [...]}}\n"
        "每次只输出一个 JSON 对象。可用工具定义如下：\n"
        f"{schema_text}\n"
    )
    patched: list[dict[str, Any]] = []
    injected = False
    for message in messages:
        if not injected and message.get("role") == "system":
            patched.append({**message, "content": str(message.get("content", "")) + instructions})
            injected = True
        else:
            patched.append(message)
    if not injected:
        patched.insert(0, {"role": "system", "content": instructions})
    return patched

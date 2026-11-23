# -*- coding: utf-8 -*-
"""Anthropic Messages API 适配层测试（零网络，全部用假客户端）。"""
from __future__ import annotations

import json

import pytest

from gis_cli.engine.anthropic_adapter import (
    ANTHROPIC_VERSION,
    AnthropicAdapter,
    AnthropicHTTPError,
    build_request,
    is_anthropic_base,
    normalize_base_url,
    parse_response,
    parse_stream,
    to_anthropic_messages,
    to_anthropic_tool_choice,
    to_anthropic_tools,
)
from gis_cli.engine.llm_client import EngineLLMClient, EngineLLMConfig, LLMError

# --------------------------------------------------------------------- 假客户端


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: dict | None = None, lines=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self._lines = list(lines or [])
        self.text = text or (json.dumps(self._payload) if self._payload else "")

    def json(self) -> dict:
        return self._payload

    def iter_lines(self):
        return iter(self._lines)


class _FakeCtx:
    def __init__(self, response: _FakeResponse):
        self._response = response

    def __enter__(self) -> _FakeResponse:
        return self._response

    def __exit__(self, *exc) -> bool:
        return False


class _FakeHTTP:
    """记录请求的假 httpx.Client。"""

    def __init__(self, response: _FakeResponse):
        self.response = response
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self.response

    def stream(self, method, url, headers=None, json=None, timeout=None):
        self.calls.append(
            {"method": method, "url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return _FakeCtx(self.response)


# ------------------------------------------------------------------ 消息转换


def test_system_prompt_is_lifted_out_of_messages():
    system, messages = to_anthropic_messages(
        [
            {"role": "system", "content": "你是 GIS 助手"},
            {"role": "user", "content": "画一张图"},
        ]
    )
    assert system == "你是 GIS 助手"
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "画一张图"}]}]


def test_assistant_tool_calls_become_tool_use_blocks():
    _, messages = to_anthropic_messages(
        [
            {"role": "user", "content": "跑一下"},
            {
                "role": "assistant",
                "content": "我来调用工具",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run_recipe", "arguments": '{"recipe_id": "buffer"}'},
                    }
                ],
            },
        ]
    )
    assistant = messages[-1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "text", "text": "我来调用工具"}
    tool_use = assistant["content"][1]
    assert tool_use["type"] == "tool_use"
    assert tool_use["id"] == "call_1"
    assert tool_use["name"] == "run_recipe"
    assert tool_use["input"] == {"recipe_id": "buffer"}


def test_consecutive_tool_results_merge_into_one_user_message():
    """Anthropic 要求 tool_result 必须在同一条 user 消息里，不能一条一个。"""
    _, messages = to_anthropic_messages(
        [
            {"role": "user", "content": "并行跑两个"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "a", "function": {"name": "t1", "arguments": "{}"}},
                    {"id": "b", "function": {"name": "t2", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "结果 A"},
            {"role": "tool", "tool_call_id": "b", "content": "结果 B"},
        ]
    )
    tail = messages[-1]
    assert tail["role"] == "user"
    assert [b["type"] for b in tail["content"]] == ["tool_result", "tool_result"]
    assert [b["tool_use_id"] for b in tail["content"]] == ["a", "b"]
    assert tail["content"][0]["content"] == "结果 A"


def test_tool_error_flag_is_preserved():
    _, messages = to_anthropic_messages(
        [
            {"role": "user", "content": "x"},
            {"role": "tool", "tool_call_id": "a", "content": "boom", "is_error": True},
        ]
    )
    # 相邻 user 消息会被合并，所以按类型找 tool_result 块，不能取 content[0]
    blocks = [b for m in messages for b in m["content"] if b.get("type") == "tool_result"]
    assert len(blocks) == 1
    assert blocks[0]["is_error"] is True
    assert blocks[0]["tool_use_id"] == "a"


def test_data_url_image_becomes_base64_source():
    _, messages = to_anthropic_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看图"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                    {"type": "image_url", "image_url": {"url": "https://x/y.jpg"}},
                ],
            }
        ]
    )
    blocks = messages[0]["content"]
    assert blocks[0] == {"type": "text", "text": "看图"}
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAA"},
    }
    assert blocks[2] == {"type": "image", "source": {"type": "url", "url": "https://x/y.jpg"}}


def test_first_message_must_be_user():
    """上游不接受 assistant 开场；适配层补一条占位 user。"""
    _, messages = to_anthropic_messages([{"role": "assistant", "content": "先说话"}])
    assert messages[0]["role"] == "user"


# ------------------------------------------------------------------ 工具声明


def test_tools_schema_is_renamed_to_input_schema():
    tools = to_anthropic_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "run_recipe",
                    "description": "跑配方",
                    "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
                },
            },
            {"type": "function", "function": {"name": "", "description": "无名字应丢弃"}},
        ]
    )
    assert len(tools) == 1
    assert tools[0]["name"] == "run_recipe"
    assert tools[0]["input_schema"]["properties"]["id"]["type"] == "string"


def test_tool_choice_mapping():
    assert to_anthropic_tool_choice("auto") == {"type": "auto"}
    assert to_anthropic_tool_choice("required") == {"type": "any"}
    assert to_anthropic_tool_choice("none") is None
    assert to_anthropic_tool_choice({"type": "function", "function": {"name": "t"}}) == {
        "type": "tool",
        "name": "t",
    }


def test_build_request_requires_max_tokens_and_sets_system():
    payload = build_request(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "u"}],
        tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
        tool_choice="auto",
        max_tokens=None,
        stream=True,
    )
    assert payload["max_tokens"] > 0
    assert payload["system"] == "S"
    assert payload["stream"] is True
    assert payload["tools"][0]["name"] == "t"
    assert payload["tool_choice"] == {"type": "auto"}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", "https://api.anthropic.com"),
        ("https://api.anthropic.com/", "https://api.anthropic.com"),
        ("https://api.anthropic.com/v1", "https://api.anthropic.com"),
        ("https://api.anthropic.com/v1/messages", "https://api.anthropic.com"),
        ("https://gw.example.com/anthropic/v1", "https://gw.example.com/anthropic"),
    ],
)
def test_base_url_normalisation(raw, expected):
    assert normalize_base_url(raw) == expected


def test_is_anthropic_base_detection():
    assert is_anthropic_base("https://api.anthropic.com") is True
    assert is_anthropic_base("https://api.deepseek.com/v1") is False
    assert is_anthropic_base("") is False


# ------------------------------------------------------------------ 响应解析


def test_parse_non_stream_response():
    parsed = parse_response(
        {
            "model": "claude-sonnet-4",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "我来算"},
                {"type": "tool_use", "id": "toolu_1", "name": "run_recipe", "input": {"a": 1}},
            ],
            "usage": {"input_tokens": 11, "output_tokens": 7, "cache_read_input_tokens": 3},
        }
    )
    assert parsed["content"] == "我来算"
    assert parsed["finish_reason"] == "tool_calls"
    assert parsed["tool_calls"][0]["arguments"] == {"a": 1}
    assert parsed["usage"]["total_tokens"] == 18
    assert parsed["usage"]["cached_tokens"] == 3


def test_parse_stream_assembles_text_tool_use_and_usage():
    deltas: list[tuple[str, str]] = []
    lines = [
        "event: message_start",
        'data: {"type":"message_start","message":{"model":"claude-sonnet-4","usage":{"input_tokens":9}}}',
        "",
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你好"}}',
        'data: {"type":"content_block_stop","index":0}',
        'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_9","name":"run_recipe"}}',
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"a\\":"}}',
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" 2}"}}',
        'data: {"type":"content_block_stop","index":1}',
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}',
        'data: {"type":"message_stop"}',
    ]
    parsed = parse_stream(lines, lambda kind, text: deltas.append((kind, text)))
    assert parsed["content"] == "你好"
    assert deltas == [("content", "你好")]
    assert parsed["finish_reason"] == "tool_calls"
    assert parsed["tool_calls"] == [{"id": "toolu_9", "name": "run_recipe", "arguments": {"a": 2}}]
    assert parsed["usage"]["prompt_tokens"] == 9
    assert parsed["usage"]["completion_tokens"] == 5
    assert parsed["model"] == "claude-sonnet-4"


def test_parse_stream_ignores_ping_and_broken_lines():
    parsed = parse_stream(
        [
            "event: ping",
            'data: {"type":"ping"}',
            "data: {broken json",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}',
        ]
    )
    assert parsed["content"] == "ok"


def test_parse_stream_thinking_becomes_reasoning():
    kinds: list[str] = []
    parsed = parse_stream(
        [
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"想一想"}}',
        ],
        lambda kind, text: kinds.append(kind),
    )
    assert parsed["reasoning"] == "想一想"
    assert kinds == ["reasoning"]


# ------------------------------------------------------------------ HTTP 适配


def test_adapter_non_stream_sends_headers_and_parses():
    fake = _FakeHTTP(
        _FakeResponse(
            payload={
                "model": "claude-sonnet-4",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
                "usage": {"input_tokens": 2, "output_tokens": 1},
            }
        )
    )
    adapter = AnthropicAdapter(api_key="sk-ant", base_url="https://api.anthropic.com/v1", http_client=fake)
    out = adapter.complete(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-sonnet-4",
        stream=False,
    )
    call = fake.calls[0]
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["headers"]["x-api-key"] == "sk-ant"
    assert call["headers"]["anthropic-version"] == ANTHROPIC_VERSION
    assert out["content"] == "done"
    assert out["finish_reason"] == "stop"


def test_adapter_stream_reads_sse_lines():
    fake = _FakeHTTP(
        _FakeResponse(
            lines=[
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"流"}}',
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}',
            ]
        )
    )
    adapter = AnthropicAdapter(api_key="k", base_url="https://api.anthropic.com", http_client=fake)
    chunks: list[str] = []
    out = adapter.complete(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-sonnet-4",
        stream=True,
        on_delta=lambda kind, text: chunks.append(text),
    )
    assert fake.calls[0]["method"] == "POST"
    assert out["content"] == "流"
    assert chunks == ["流"]


def test_adapter_raises_typed_error_on_401():
    fake = _FakeHTTP(_FakeResponse(status_code=401, text='{"error":"invalid api key"}'))
    adapter = AnthropicAdapter(api_key="bad", http_client=fake)
    with pytest.raises(AnthropicHTTPError) as excinfo:
        adapter.complete(messages=[{"role": "user", "content": "x"}], model="m", stream=False)
    assert excinfo.value.status_code == 401
    assert "invalid api key" in excinfo.value.body


# ------------------------------------------------------- 与 EngineLLMClient 集成


def test_provider_anthropic_routes_to_adapter(monkeypatch):
    cfg = EngineLLMConfig(
        model="claude-sonnet-4-20250514",
        api_key="sk-ant",
        api_base="https://api.anthropic.com",
        provider="anthropic",
        stream=True,
        retry_count=1,
    )
    client = EngineLLMClient(cfg)
    assert client._use_anthropic() is True

    class _StubAdapter:
        def __init__(self):
            self.calls: list[dict] = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "content": "调工具",
                "reasoning": "",
                "tool_calls": [{"id": "toolu_1", "name": "run_recipe", "arguments": {"a": 1}}],
                "finish_reason": "tool_calls",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                "model": "claude-sonnet-4-20250514",
            }

    stub = _StubAdapter()
    client._anthropic_adapter = stub
    response = client._chat_once(
        [{"role": "user", "content": "hi"}],
        model="claude-sonnet-4-20250514",
        tools=None,
        tool_choice=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    )
    assert response.content == "调工具"
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].name == "run_recipe"
    assert response.tool_calls[0].arguments == {"a": 1}
    assert response.total_tokens == 5
    assert stub.calls and stub.calls[0]["stream"] is True


def test_anthropic_401_is_provider_wide(monkeypatch):
    cfg = EngineLLMConfig(
        model="claude-sonnet-4-20250514",
        api_key="bad",
        api_base="https://api.anthropic.com",
        provider="anthropic",
        retry_count=1,
    )
    client = EngineLLMClient(cfg)

    class _StubAdapter:
        def complete(self, **kwargs):
            raise AnthropicHTTPError("Anthropic HTTP 401: bad key", status_code=401, body="bad key")

    client._anthropic_adapter = _StubAdapter()
    with pytest.raises(LLMError) as excinfo:
        client.chat([{"role": "user", "content": "x"}], model="claude-sonnet-4-20250514")
    assert excinfo.value.provider_wide is True
    assert excinfo.value.status_code == 401


def test_api_base_detection_without_provider_field():
    cfg = EngineLLMConfig(model="claude-sonnet-4", api_key="k", api_base="https://api.anthropic.com/v1")
    assert EngineLLMClient(cfg)._use_anthropic() is True

    cfg2 = EngineLLMConfig(model="deepseek-chat", api_key="k", api_base="http://127.0.0.1:8789/v1")
    assert EngineLLMClient(cfg2)._use_anthropic() is False


def test_config_from_file_reads_provider(tmp_path):
    path = tmp_path / "llm_config.json"
    path.write_text(
        json.dumps(
            {
                "provider": "Anthropic",
                "model": "claude-sonnet-4-20250514",
                "api_key": "k",
                "api_base": "https://api.anthropic.com",
            }
        ),
        encoding="utf-8",
    )
    cfg = EngineLLMConfig.from_file(path)
    assert cfg.provider == "anthropic"


def test_max_tokens_400_falls_back_to_8192():
    """Claude 3.5 的输出上限是 8192；配置里写 16000 会 400，应自动降级重试。"""
    cfg = EngineLLMConfig(
        model="claude-3-5-sonnet-20241022",
        api_key="k",
        api_base="https://api.anthropic.com",
        provider="anthropic",
        max_tokens=16000,
        retry_count=2,
    )
    client = EngineLLMClient(cfg)

    class _StubAdapter:
        def __init__(self):
            self.calls: list[dict] = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise AnthropicHTTPError(
                    "Anthropic HTTP 400: max_tokens: 16000 > 8192",
                    status_code=400,
                    body="max_tokens: 16000 > 8192",
                )
            return {
                "content": "ok",
                "reasoning": "",
                "tool_calls": [],
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "model": "claude-3-5-sonnet-20241022",
            }

    stub = _StubAdapter()
    client._anthropic_adapter = stub
    response = client._chat_once(
        [{"role": "user", "content": "hi"}],
        model="claude-3-5-sonnet-20241022",
        tools=None,
        tool_choice=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    )
    assert response.content == "ok"
    assert [call["max_tokens"] for call in stub.calls] == [16000, 8192]

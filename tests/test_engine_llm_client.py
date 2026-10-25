# -*- coding: utf-8 -*-
"""Unit tests for engine LLM client backoff/routing (no network)."""
from __future__ import annotations

from gis_cli.engine.llm_client import EngineLLMClient, EngineLLMConfig


def test_connection_errors_back_off_fast():
    client = EngineLLMClient(EngineLLMConfig(backoff_seconds=8.0, max_backoff_seconds=90.0))
    # connection error (status=None) must never wait long; we want to switch models
    assert client._backoff_delay(1, None) <= 6.0
    assert client._backoff_delay(4, None) <= 15.0


def test_429_backoff_grows_but_capped():
    client = EngineLLMClient(EngineLLMConfig(backoff_seconds=8.0, max_backoff_seconds=90.0))
    client._consecutive_429 = 3
    delay = client._backoff_delay(2, 429)
    assert 5.0 <= delay <= 90.0


def test_models_for_routing_order():
    cfg = EngineLLMConfig(
        model="gpt-4o-mini",
        fallback_models=["deepseek-chat"],
        routing_rules={
            "agent": ["deepseek-chat"],
            "agent_strong": ["gpt-4o-mini"],
        },
    )
    assert cfg.models_for(task_type="agent")[0] == "deepseek-chat"
    assert cfg.models_for(task_type="agent_strong")[0] == "gpt-4o-mini"
    # fallback model still reachable
    assert "gpt-4o-mini" in cfg.models_for(task_type="agent")


def test_total_timeout_caps_gateway_stall(monkeypatch):
    """429 风暴下整轮重试受全局超时约束，快速失败而不是无限缠绵。"""
    import time as _time

    from gis_cli.engine.llm_client import LLMError

    class _Err429(Exception):
        status_code = 429

    class _StubCompletions:
        def create(self, **kwargs):
            raise _Err429("rate limited")

    class _StubClient:
        class chat:
            completions = _StubCompletions()

    cfg = EngineLLMConfig(
        model="gpt-4o-mini",
        api_key="x",
        retry_count=100,
        backoff_seconds=0.01,
        max_backoff_seconds=0.05,
        total_timeout=0.4,
    )
    client = EngineLLMClient(cfg)
    client._client = _StubClient()
    client._backoff_delay = lambda attempt, status: 0.02  # 加速测试
    t0 = _time.monotonic()
    try:
        client.chat([{"role": "user", "content": "hi"}])
        raise AssertionError("expected LLMError")
    except LLMError as exc:
        elapsed = _time.monotonic() - t0
        assert "全局超时" in str(exc), str(exc)
        assert elapsed < 5.0, f"took {elapsed:.1f}s"


def test_reasoning_effort_is_sent_via_extra_body():
    """思考档位必须传到 body 的 reasoning_effort 里。

    实测：部分推理模型默认不返回 reasoning_content，不显式传 reasoning_effort
    就完全没有思考内容（界面只剩“模型自述”占位）。
    """
    seen: dict = {}

    class _StubCompletions:
        def create(self, **kwargs):
            seen.update(kwargs)
            return _FakeResponse()

    class _StubClient:
        class chat:
            completions = _StubCompletions()

    cfg = EngineLLMConfig(
        model="test-model", api_key="x", reasoning_effort="medium"
    )
    client = EngineLLMClient(cfg)
    client._client = _StubClient()
    client.chat([{"role": "user", "content": "hi"}])

    assert seen.get("extra_body") == {"reasoning_effort": "medium"}, seen.get("extra_body")
    # 不能塞到顶层：OpenAI SDK 对未知顶层字段会直接拒绝
    assert "reasoning_effort" not in seen


def test_image_rejection_strips_images_and_retries():
    """回归：上游拒图时摘掉图片重试，而不是把整个任务判死。

    实测（14 届 4a-1）：识图本是可选增强，但上游对图片回
    400 "Please start a new conversation, replace the image, and try again"，
    以前直接抛 LLMError → 整轮 run 判 llm_unavailable，只跑了 7 轮就死，
    而栅格产物其实已经做完了。
    """
    from gis_cli.engine.llm_client import EngineLLMClient, EngineLLMConfig

    calls: list[dict] = []

    class _Err400(Exception):
        status_code = 400
        body = '{"message": "upstream 400: replace the image"}'

    class _StubCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if _has_image_in(kwargs["messages"]):
                raise _Err400("replace the image")
            return _FakeResponse()

    class _StubClient:
        class chat:
            completions = _StubCompletions()

    cfg = EngineLLMConfig(model="test-model", api_key="x", retry_count=3)
    client = EngineLLMClient(cfg)
    client._client = _StubClient()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "看这张图"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}},
            ],
        }
    ]
    response = client.chat(messages)  # 不应抛异常
    assert response is not None
    assert len(calls) == 2, f"应先失败一次再摘图重试，实际 {len(calls)} 次"
    assert not _has_image_in(calls[-1]["messages"]), "重试时应该已经摘掉图片"
    assert "已因上游不支持而省略" in str(calls[-1]["messages"]), calls[-1]["messages"]
    assert client.vision_rejected is True


def _has_image_in(messages) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in content
        ):
            return True
    return False


def test_vision_disabled_after_rejection(monkeypatch, tmp_path):
    """上游拒过图后不再附图（省下每轮的无效 400）。"""
    from gis_cli.engine.loop import AgentLoop
    from gis_cli.engine.tools import EngineContext, build_default_registry

    class _LLM:
        class config:
            vision = True

        vision_rejected = True

    loop = AgentLoop(
        workspace=tmp_path,
        llm_client=_LLM(),
        registry=build_default_registry(),
        context=EngineContext(workspace=tmp_path),
    )
    assert loop._vision_enabled() is False


def test_reasoning_effort_omitted_when_unset():
    """没配就不传该字段（兼容不认它的网关）。"""
    seen: dict = {}

    class _StubCompletions:
        def create(self, **kwargs):
            seen.update(kwargs)
            return _FakeResponse()

    class _StubClient:
        class chat:
            completions = _StubCompletions()

    client = EngineLLMClient(EngineLLMConfig(model="m", api_key="x"))
    client._client = _StubClient()
    client.chat([{"role": "user", "content": "hi"}])
    assert "extra_body" not in seen


def test_reasoning_content_is_surfaced():
    """reasoning_content 要落到 LLMResponse.reasoning，否则界面没有思考可显示。"""

    class _StubCompletions:
        def create(self, **_kwargs):
            return _FakeResponse(reasoning="先看数据目录，再统计分组。")

    class _StubClient:
        class chat:
            completions = _StubCompletions()

    client = EngineLLMClient(EngineLLMConfig(model="m", api_key="x"))
    client._client = _StubClient()
    resp = client.chat([{"role": "user", "content": "hi"}])
    assert resp.reasoning == "先看数据目录，再统计分组。"


def test_config_reads_reasoning_effort_from_engine_section(tmp_path):
    import json

    path = tmp_path / "llm_config.json"
    path.write_text(
        json.dumps({"model": "m", "api_key": "k", "engine": {"reasoning_effort": "high"}}),
        encoding="utf-8",
    )
    assert EngineLLMConfig.from_file(path).reasoning_effort == "high"

    # 根段也认，且留空/缺失时为""
    path.write_text(json.dumps({"model": "m", "api_key": "k"}), encoding="utf-8")
    assert EngineLLMConfig.from_file(path).reasoning_effort == ""


def _FakeResponse(reasoning: str = "") -> object:
    """最小可用的 chat completion 响应（仿 OpenAI SDK 的对象形状）。"""

    class _Msg:
        def __init__(self):
            self.content = "ok"
            self.reasoning_content = reasoning
            self.tool_calls = None

    class _Choice:
        def __init__(self):
            self.message = _Msg()
            self.finish_reason = "stop"

    class _Usage:
        prompt_tokens = 1
        completion_tokens = 1
        total_tokens = 2

    class _Resp:
        def __init__(self):
            self.choices = [_Choice()]
            self.usage = _Usage()

    return _Resp()


def test_model_circuit_breaker_skips_stalled_primary():
    """主模型超时达阈值后进入冷却，下一次直接先走备用模型（不再白等）。"""
    from gis_cli.engine.llm_client import LLMError

    client = EngineLLMClient(
        EngineLLMConfig(
            model="primary-model",
            fallback_models=["backup-model"],
            model_failure_threshold=2,
            model_cooldown_seconds=300.0,
        )
    )
    client._record_failure("primary-model", LLMError("timeout", status_code=None))
    assert client._order_by_health(["primary-model", "backup-model"])[0] == "primary-model"
    client._record_failure("primary-model", LLMError("timeout", status_code=None))
    assert client._order_by_health(["primary-model", "backup-model"]) == ["backup-model", "primary-model"]
    # 成功一次后计数清零，冷却解除
    client._record_success("primary-model")
    assert client._order_by_health(["primary-model", "backup-model"])[0] == "primary-model"


def test_model_circuit_breaker_ignores_non_retryable():
    """参数/权限类错误不该被当成网关卡死而冷却模型。"""
    from gis_cli.engine.llm_client import LLMError

    client = EngineLLMClient(
        EngineLLMConfig(model="m", fallback_models=["b"], model_failure_threshold=1)
    )
    client._record_failure("m", LLMError("bad request", status_code=400))
    assert client._cooldown_until == {}


def test_model_cooldown_escalates_on_repeat_failures():
    """反复失败的模型冷却时间应指数升级（实测网关坏掉的主模型每轮白烧 420s 的修复）。"""
    from gis_cli.engine.llm_client import LLMError

    client = EngineLLMClient(
        EngineLLMConfig(
            model="m",
            fallback_models=["b"],
            model_failure_threshold=1,
            model_cooldown_seconds=100.0,
            model_cooldown_max_seconds=1000.0,
        )
    )
    import time as _t

    client._record_failure("m", LLMError("conn", status_code=None))
    first = client._cooldown_until["m"] - _t.monotonic()
    client._record_failure("m", LLMError("conn", status_code=None))
    second = client._cooldown_until["m"] - _t.monotonic()
    assert 90 <= first <= 115, first   # 首次冷却 ≈ 100s
    assert second >= first * 1.5, (first, second)  # 升级到 ≈ 200s
    for _ in range(10):
        client._record_failure("m", LLMError("conn", status_code=None))
    cap = client._cooldown_until["m"] - _t.monotonic()
    assert cap <= 1000.0, cap
    client._record_success("m")
    assert client._cooldown_until == {} and client._cooldown_strikes == {}


def test_repeat_offender_gets_reduced_attempt_budget(monkeypatch):
    """已被冷却过的模型再次被尝试时，单次预算应缩短（避免反复烧满 420s）。"""
    from gis_cli.engine.llm_client import EngineLLMClient, EngineLLMConfig, LLMError

    client = EngineLLMClient(
        EngineLLMConfig(
            model="m",
            api_key="test-key",
            api_base="http://127.0.0.1:9/v1",
            model_failure_threshold=1,
            total_timeout=420.0,
            reduced_timeout_seconds=60.0,
        )
    )
    seen: dict[str, float] = {}

    def fake_create(**kwargs):
        seen["timeout"] = kwargs.get("timeout")
        raise TimeoutError("Request timed out")

    monkeypatch.setattr(client.client.chat.completions, "create", fake_create)

    # 首次：没有失败史 → 用满 total_timeout（但至少 30s）
    try:
        client._chat_once(
            [{"role": "user", "content": "hi"}],
            model="m", tools=None, tool_choice=None, temperature=0, max_tokens=None, response_format=None,
        )
    except Exception:
        pass
    first = seen.get("timeout")
    client._record_failure("m", LLMError("conn", status_code=None))

    # 有失败史后：预算被压到 reduced_timeout_seconds
    try:
        client._chat_once(
            [{"role": "user", "content": "hi"}],
            model="m", tools=None, tool_choice=None, temperature=0, max_tokens=None, response_format=None,
        )
    except Exception:
        pass
    second = seen.get("timeout")
    assert first and second and second < first, (first, second)
    assert second <= 60.0, second


# --------------------------------------------------------------------- 流式
class _Delta:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Choice:
    def __init__(self, delta=None, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, delta=None, finish_reason=None, usage=None, choices=True):
        self.choices = [_Choice(delta, finish_reason)] if choices else []
        self.usage = usage


class _Usage:
    prompt_tokens = 11
    completion_tokens = 7
    total_tokens = 18
    prompt_cache_hit_tokens = 5


class _Fn:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _ToolCallDelta:
    def __init__(self, index=0, id=None, function=None):
        self.index = index
        self.id = id
        self.function = function


def _stream_client(chunks):
    """构造一个假的流式网关：调用 create 时按 chunks 逐块吐出。"""

    class _Completions:
        def create(self, **kwargs):
            return iter(chunks)

    class _Client:
        class chat:
            completions = _Completions()

    cfg = EngineLLMConfig(model="m", api_key="x", stream=True)
    client = EngineLLMClient(cfg)
    client._client = _Client()
    return client


def _nonstream_client(response_obj):
    class _Completions:
        def create(self, **kwargs):
            return response_obj

    class _Client:
        class chat:
            completions = _Completions()

    cfg = EngineLLMConfig(model="m", api_key="x", stream=False)
    client = EngineLLMClient(cfg)
    client._client = _Client()
    return client


def test_streaming_assembles_content_and_reasoning():
    """正文与思考分片按序拼装，usage 取自末尾 chunk。"""
    chunks = [
        _Chunk(_Delta(reasoning_content="Let me ")),
        _Chunk(_Delta(reasoning_content="think.")),
        _Chunk(_Delta(content="北京")),
        _Chunk(_Delta(content="市")),
        _Chunk(_Delta(), finish_reason="stop", usage=_Usage()),
    ]
    seen: list[tuple[str, str]] = []
    client = _stream_client(chunks)
    response = client.chat(
        [{"role": "user", "content": "hi"}], on_delta=lambda k, t: seen.append((k, t))
    )
    assert response.content == "北京市"
    assert response.reasoning == "Let me think."
    assert response.finish_reason == "stop"
    assert response.usage["total_tokens"] == 18
    # 增量按到达顺序回调，思考在前
    assert seen == [
        ("reasoning", "Let me "),
        ("reasoning", "think."),
        ("content", "北京"),
        ("content", "市"),
    ]


def test_streaming_tool_calls_fragments_merge():
    """tool_calls 分片：首块带 id/name，后续只补 arguments 片段。"""
    chunks = [
        _Chunk(_Delta(tool_calls=[_ToolCallDelta(0, id="call_1", function=_Fn("catalog_query", ""))])),
        _Chunk(_Delta(tool_calls=[_ToolCallDelta(0, function=_Fn(None, '{"que'))])),
        _Chunk(_Delta(tool_calls=[_ToolCallDelta(0, function=_Fn(None, 'ry": "北京"}'))])),
        _Chunk(_Delta(), finish_reason="tool_calls", usage=_Usage()),
    ]
    client = _stream_client(chunks)
    response = client.chat([{"role": "user", "content": "hi"}])
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "catalog_query"
    assert call.arguments == {"query": "北京"}
    assert call.parse_error == ""


def test_streaming_multiple_tool_calls_by_index():
    """多个并行工具调用按 index 各自累积，不串味。"""
    chunks = [
        _Chunk(_Delta(tool_calls=[
            _ToolCallDelta(0, id="a", function=_Fn("t1", '{"x":1}')),
            _ToolCallDelta(1, id="b", function=_Fn("t2", '{"y":')),
        ])),
        _Chunk(_Delta(tool_calls=[_ToolCallDelta(1, function=_Fn(None, "2}"))])),
        _Chunk(_Delta(), finish_reason="tool_calls", usage=_Usage()),
    ]
    client = _stream_client(chunks)
    response = client.chat([{"role": "user", "content": "hi"}])
    assert [c.name for c in response.tool_calls] == ["t1", "t2"]
    assert response.tool_calls[0].arguments == {"x": 1}
    assert response.tool_calls[1].arguments == {"y": 2}


def test_streaming_matches_nonstream_fields():
    """同一份内容：流式组装结果与非流式逐字段等价（调用方无感）。"""
    class _Msg:
        content = "北京市"
        reasoning_content = "Let me think."
        tool_calls = None

    class _RespChoice:
        message = _Msg()
        finish_reason = "stop"

    class _Resp:
        choices = [_RespChoice()]
        usage = _Usage()

    chunks = [
        _Chunk(_Delta(reasoning_content="Let me ")),
        _Chunk(_Delta(reasoning_content="think.")),
        _Chunk(_Delta(content="北京")),
        _Chunk(_Delta(content="市")),
        _Chunk(_Delta(), finish_reason="stop", usage=_Usage()),
    ]
    streamed = _stream_client(chunks).chat([{"role": "user", "content": "hi"}])
    plain = _nonstream_client(_Resp()).chat([{"role": "user", "content": "hi"}])
    assert streamed.content == plain.content
    assert streamed.reasoning == plain.reasoning
    assert streamed.finish_reason == plain.finish_reason
    assert streamed.usage == plain.usage
    assert len(streamed.tool_calls) == len(plain.tool_calls) == 0


def test_stream_ignored_by_gateway_falls_back_to_nonstream():
    """上游忽略 stream（返回空块流）→ 自动降级为非流式，任务不断。"""
    class _Msg:
        content = "非流式回答"
        reasoning_content = ""
        tool_calls = None

    class _RespChoice:
        message = _Msg()
        finish_reason = "stop"

    class _Resp:
        choices = [_RespChoice()]
        usage = _Usage()

    calls: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if kwargs.get("stream"):
                return iter([])  # 假装忽略了 stream 参数
            return _Resp()

    class _Client:
        class chat:
            completions = _Completions()

    cfg = EngineLLMConfig(model="m", api_key="x", stream=True, retry_count=2)
    client = EngineLLMClient(cfg)
    client._client = _Client()
    response = client.chat([{"role": "user", "content": "hi"}])
    assert response.content == "非流式回答"
    assert client._stream_unsupported is True
    # 先试了流式，再回落非流式
    assert [bool(c.get("stream")) for c in calls] == [True, False]


def test_stream_disabled_never_requests_stream():
    """配置关掉流式时请求里不带 stream=True。"""

    class _Msg:
        content = "x"
        reasoning_content = ""
        tool_calls = None

    class _RespChoice:
        message = _Msg()
        finish_reason = "stop"

    class _Resp:
        choices = [_RespChoice()]
        usage = _Usage()

    plain = _nonstream_client(_Resp())
    seen: dict = {}

    class _Completions:
        def create(self, **kwargs):
            seen.update(kwargs)
            return _Resp()

    class _Client:
        class chat:
            completions = _Completions()

    plain._client = _Client()
    plain.chat([{"role": "user", "content": "hi"}])
    assert not seen.get("stream"), seen


# ------------------------------------------------- 思考档位（按任务类型）


def test_effort_per_task_type_overrides_global():
    """按任务类型覆盖思考档位：交互轮用低档，规划/验收保持默认。

    实测（deepseek-v4.1-flash-intl）交互轮换 low：首字 5.11s→2.37s、
    总耗时 8.72s→5.80s，输出质量相当。
    """
    cfg = EngineLLMConfig(
        model="m",
        reasoning_effort="medium",
        reasoning_effort_by_task={"agent": "low", "verify": "high"},
    )
    assert cfg.effort_for("agent") == "low"
    assert cfg.effort_for("AGENT") == "low"  # 大小写不敏感
    assert cfg.effort_for("verify") == "high"
    assert cfg.effort_for("planning") == "medium"  # 没覆盖 → 全局
    assert cfg.effort_for(None) == "medium"
    assert cfg.effort_for("") == "medium"


def test_effort_by_task_read_from_config_file(tmp_path):
    """engine.reasoning_effort_by_task 能从配置文件读出来。"""
    import json

    path = tmp_path / "llm_config.json"
    path.write_text(
        json.dumps(
            {
                "model": "m",
                "api_key": "k",
                "engine": {
                    "reasoning_effort": "medium",
                    "reasoning_effort_by_task": {"agent": "low"},
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = EngineLLMConfig.from_file(path)
    assert cfg.effort_for("agent") == "low"
    assert cfg.effort_for("planning") == "medium"


def test_effort_by_task_empty_means_global_everywhere():
    """没配分档时所有任务都走全局值（默认行为不变）。"""
    cfg = EngineLLMConfig(model="m", reasoning_effort="high")
    assert cfg.effort_for("agent") == "high"
    assert cfg.effort_for("verify") == "high"


def test_chat_sends_effort_for_the_given_task_type():
    """实际请求体里的 reasoning_effort 要按 task_type 选。"""
    seen: dict = {}

    class _Msg:
        content = "ok"
        reasoning_content = ""
        tool_calls = None

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    class _Completions:
        def create(self, **kwargs):
            seen.update(kwargs)
            return _Resp()

    class _Client:
        class chat:
            completions = _Completions()

    cfg = EngineLLMConfig(
        model="m",
        api_key="x",
        stream=False,
        reasoning_effort="medium",
        reasoning_effort_by_task={"agent": "low"},
    )
    client = EngineLLMClient(cfg)
    client._client = _Client()
    client.chat([{"role": "user", "content": "hi"}], task_type="agent")
    assert seen.get("extra_body", {}).get("reasoning_effort") == "low"


# ------------------------------------------------------- 用量与缓存命中率统计
def test_usage_summary_aggregates_and_computes_hit_rate():
    """一次成功调用就要把 token 用量累加进来，并能算出缓存命中率。

    以前只在日志后缀里写 cached=N，跑完就没了 —— 无法回答"这个任务花了
    多少 token / 缓存命中率多少 / 哪个模型在烧钱"。
    """
    from gis_cli.engine.llm_client import EngineLLMClient, EngineLLMConfig, LLMResponse

    client = EngineLLMClient(EngineLLMConfig(model="m", api_key="k", api_base="http://x"))
    client._record_usage(
        "m",
        LLMResponse(
            content="hi",
            model="m",
            usage={"prompt_tokens": 1000, "completion_tokens": 50, "prompt_cache_hit_tokens": 800},
        ),
    )
    client._record_usage(
        "m",
        LLMResponse(
            content="hi",
            model="m",
            usage={"prompt_tokens": 500, "completion_tokens": 20, "cached_tokens": 0},
        ),
    )
    summary = client.usage_summary()
    assert summary["calls"] == 2
    assert summary["prompt_tokens"] == 1500
    assert summary["completion_tokens"] == 70
    assert summary["cached_tokens"] == 800
    assert abs(summary["cache_hit_rate"] - 800 / 1500) < 1e-9
    assert summary["by_model"]["m"]["calls"] == 2


def test_usage_summary_separates_models():
    """按模型分开累计（熔断切换后能看清是哪个模型在烧 token）。"""
    from gis_cli.engine.llm_client import EngineLLMClient, EngineLLMConfig, LLMResponse

    client = EngineLLMClient(EngineLLMConfig(model="a", api_key="k", api_base="http://x"))
    client._record_usage("a", LLMResponse(content="", model="a", usage={"prompt_tokens": 100}))
    client._record_usage("b", LLMResponse(content="", model="b", usage={"prompt_tokens": 300}))
    summary = client.usage_summary()
    assert summary["by_model"]["a"]["prompt_tokens"] == 100
    assert summary["by_model"]["b"]["prompt_tokens"] == 300
    assert summary["prompt_tokens"] == 400


def test_usage_summary_empty_is_safe():
    """没有任何调用时不能除零，返回结构化空值。"""
    from gis_cli.engine.llm_client import EngineLLMClient, EngineLLMConfig

    summary = EngineLLMClient(EngineLLMConfig(model="m", api_key="k", api_base="http://x")).usage_summary()
    assert summary["calls"] == 0
    assert summary["cache_hit_rate"] == 0.0
    assert summary["by_model"] == {}


# ------------------------------------------------ 上游卡死不能无限挂起
def test_stream_stall_is_bounded_by_deadline():
    """上游持续发 SSE 心跳（注释行）时，httpx 的 read timeout 永不触发。

    实测 workbuddy 反代卡死：420s 预算的调用挂了 2700s+，整个任务静默
    （14 届任务二a 实测）。看门狗必须到点强制关流。
    """
    import threading
    import time
    from types import SimpleNamespace

    client = EngineLLMClient(EngineLLMConfig(model="m", total_timeout=30, min_stream_budget_seconds=0.5))

    class _StallingStream:
        """无限产心跳注释行的流——SDK 会丢弃注释，for 循环拿不到 chunk。"""

        def __init__(self):
            self.closed = threading.Event()

        def __iter__(self):
            while not self.closed.is_set():
                time.sleep(0.05)
            raise RuntimeError("stream closed")

        def close(self):
            self.closed.set()

    stall = _StallingStream()
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: stall
    )))
    client._client = fake

    started = time.monotonic()
    try:
        client._stream_once({"model": "m"}, "m", deadline=time.monotonic() + 1.0)
        raise AssertionError("应当超时抛出，而不是返回")
    except TimeoutError as exc:
        assert "预算" in str(exc)
    elapsed = time.monotonic() - started
    assert elapsed < 15, f"卡了 {elapsed:.1f}s，说明看门狗没生效"


# ------------------------------------------------- 修复调用要能单独封顶
def test_budget_seconds_caps_a_single_call():
    """自动修复是加速手段：修不好就退回主循环，不该烧满 420s + 8 次退避。

    实测 14 届任务四a：第 7 轮卡在修复里 950s（420s 预算 + 长退避），
    整跑预算被一次修复吃掉。
    """
    import time
    from types import SimpleNamespace

    from gis_cli.engine.llm_client import EngineLLMClient, EngineLLMConfig

    cfg = EngineLLMConfig(
        model="m", total_timeout=600.0, retry_count=2,
        backoff_seconds=0.2, max_backoff_seconds=0.2, min_stream_budget_seconds=0.5,
    )
    client = EngineLLMClient(cfg)

    class _Slow:
        """每次都抛 429，逼它走退避重试。"""

        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise RuntimeError("429 rate limited")

    slow = _Slow()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=slow))

    started = time.monotonic()
    try:
        client.chat([{"role": "user", "content": "x"}], task_type="code_repair", budget_seconds=1.0)
        raise AssertionError("应当抛错")
    except Exception:
        pass
    elapsed = time.monotonic() - started
    assert elapsed < 15, f"封顶失效，耗时 {elapsed:.1f}s"


# ------------------------------------------- 修复不能把整跑预算吃光
def test_repair_stops_after_wall_clock_cap():
    """修不好就该回主循环换思路，而不是在同一个错上反复烧预算。

    实测 14 届任务二a：3000s 里 2700s 花在同一个 CSV 解析失败的反复修复上。
    """
    import time
    from types import SimpleNamespace

    from gis_cli.engine.repair import CodeRepairer
    from gis_cli.engine.state import Observation

    calls = {"llm": 0, "run": 0}

    class _LLM:
        def chat(self, *a, **kw):
            calls["llm"] += 1
            return SimpleNamespace(content='{"thought": "再试一次", "code": "print(1)"}')

    class _Runner:
        def run(self, *a, **kw):
            calls["run"] += 1
            time.sleep(0.2)  # 每次修复都慢
            return SimpleNamespace(
                ok=False, stdout="", stderr="", result=None, display_images=[],
                error={"type": "E", "message": f"fail-{calls['run']}"},
            )

    r = CodeRepairer(llm_client=_LLM(), code_runner=_Runner(), max_attempts=10,
                     max_repair_seconds=1.0, llm_budget_seconds=5.0)
    obs = Observation(ok=False, summary="boom", error={"type": "E", "message": "fail-0"})
    _, _, attempts = r.run(intent="x", code="print(0)", observation=obs)
    assert len(attempts) < 10, f"墙钟上限没生效，跑满 {len(attempts)} 次"


def test_vision_mode_auto_uses_model_capability():
    """`vision_mode: auto` 按模型名决定要不要附图；旧键 `vision: true` 仍当 on。"""
    import json
    import tempfile
    from pathlib import Path

    from gis_cli.engine.llm_client import EngineLLMConfig, model_supports_vision

    assert model_supports_vision("glm-5.3-flash") is False
    assert model_supports_vision("gpt-4o-2024-08-06") is True
    assert model_supports_vision("qwen2.5-vl-72b") is True

    assert EngineLLMConfig(model="glm-5.3-flash").effective_vision is False
    assert EngineLLMConfig(model="gpt-4o").effective_vision is True
    assert EngineLLMConfig(model="glm-5.3-flash", vision_mode="on").effective_vision is True
    assert EngineLLMConfig(model="gpt-4o", vision_mode="off").effective_vision is False

    # 配置文件：显式 vision_mode 优先；只有旧键 vision=true 时按 on 处理
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "llm_config.json"
        path.write_text(json.dumps({"model": "glm-5.3-flash", "vision": True, "engine": {"vision_mode": "auto"}}), encoding="utf-8")
        assert EngineLLMConfig.from_file(path).effective_vision is False

        path.write_text(json.dumps({"model": "glm-5.3-flash", "vision": True}), encoding="utf-8")
        assert EngineLLMConfig.from_file(path).vision_mode == "on"

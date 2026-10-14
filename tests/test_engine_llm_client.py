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

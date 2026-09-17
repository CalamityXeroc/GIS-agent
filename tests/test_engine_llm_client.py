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

# -*- coding: utf-8 -*-
"""配置写回契约（S5）：白名单 / 备份 / 还原 / 乐观并发 / 运行中锁 / 脱敏往返。

覆盖计划 §8 的「配置写入原子性」与 §9 的「配置写回打断在跑的 run」两条风险：
所有失败路径都断言**文件字节不变**，只有 2xx 才允许落盘。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gis_cli.engine.llm_client import EngineLLMConfig
from gis_cli.web import jobs
from gis_cli.web.app import create_app
from gis_cli.web.event_bus import EventBus

ROLES = (
    "agent",
    "agent_strong",
    "planning",
    "planning_expert",
    "intent",
    "task_refine",
    "recovery",
    "recovery_expert",
    "summarize",
    "verify",
)

RAW_KEY = "sk-live-0123456789abcdef"

BASE_CONFIG: dict[str, Any] = {
    "provider": "openai",
    "model": "model-a",
    "api_key": RAW_KEY,
    "api_base": "http://127.0.0.1:8789/v1",
    "temperature": 0.2,
    "max_tokens": 16000,
    "fallback_models": ["model-b"],
    "routing_rules": {role: ["model-a", "model-b"] for role in ROLES},
    "engine": {
        "max_turns": 60,
        "max_repairs": 4,
        "context_soft_limit_tokens": 120000,
        "kernel_idle_timeout_seconds": 1800,
        "kernel_exec_timeout_seconds": 600,
        "llm_max_concurrency": 1,
    },
}


@pytest.fixture()
def cfg(workspace: Path) -> Path:
    """写一份 CRLF 的配置（Windows 上引擎真实文件就是 CRLF）。"""
    path = workspace / "config" / "llm_config.json"
    path.parent.mkdir(parents=True)
    text = json.dumps(BASE_CONFIG, indent=2, ensure_ascii=False) + "\n"
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    return path


@pytest.fixture()
def config_client(workspace: Path, token: str, cfg: Path):
    with TestClient(create_app(workspace, token=token)) as client:
        yield client


@pytest.fixture()
def busy_client(workspace: Path, token: str, cfg: Path):
    """管理器里放一个 running 的 job：等价于“有 run 在跑”。"""
    manager = jobs.JobManager(
        workspace, bus=EventBus(), runner=lambda *args, **kwargs: None, persist=False
    )
    job = jobs.Job(id="run_busy01", goal="正在跑的任务")
    job.transition("running")
    manager._jobs[job.id] = job
    with TestClient(create_app(workspace, token=token, manager=manager)) as client:
        yield client


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get(client: TestClient, auth: dict[str, str]) -> dict[str, Any]:
    response = client.get("/api/config", headers=auth)
    assert response.status_code == 200, response.text
    return response.json()


def _put(
    client: TestClient,
    auth: dict[str, str],
    sha: str,
    patch: dict[str, Any] | None = None,
    **extra: Any,
):
    body: dict[str, Any] = {"base_sha256": sha, "patch": patch or {}}
    body.update(extra)
    return client.put("/api/config", headers=auth, json=body)


# ---------------------------------------------------------------- 读契约


def test_get_masks_secret_and_exposes_sha(config_client: TestClient, auth, cfg: Path):
    response = config_client.get("/api/config", headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert RAW_KEY not in response.text
    assert body["api_key"] == "sk-…cdef"
    assert body["sha256"] == _sha(cfg)
    assert body["_meta"]["busy"] is False
    assert body["_meta"]["path"].replace("\\", "/") == "config/llm_config.json"
    # 新建运行页只读展示服务端当前工作区
    assert body["_meta"]["workspace"] == str(cfg.parent.parent.resolve())
    assert body["engine"]["max_turns"] == 60


# ---------------------------------------------------------------- 白名单


@pytest.mark.parametrize("key", ["api_key", "vision", "provider_models"])
def test_put_rejects_forbidden_keys_and_keeps_bytes(
    config_client: TestClient, auth, cfg: Path, key: str
):
    before = cfg.read_bytes()
    body = _put(config_client, auth, _sha(cfg), {key: "whatever"})
    assert body.status_code == 400, body.text
    assert key in body.json()["detail"]
    assert cfg.read_bytes() == before
    assert not (cfg.parent / "backups").exists()


@pytest.mark.parametrize(
    "patch",
    [
        {"engine": {"llm_max_concurrency": 2}},
        {"engine": {"max_turns": "30"}},
        {"engine": {"max_turns": 0}},
        {"routing_rules": {"nope": ["model-a"]}},
        {"routing_rules": {"agent": []}},
        {"temperature": 9},
        {"max_tokens": -1},
        {"fallback_models": "model-b"},
    ],
)
def test_put_rejects_out_of_scope_values(
    config_client: TestClient, auth, cfg: Path, patch: dict[str, Any]
):
    before = cfg.read_bytes()
    response = _put(config_client, auth, _sha(cfg), patch)
    assert response.status_code == 400, response.text
    assert cfg.read_bytes() == before


def test_put_without_base_sha_is_428(config_client: TestClient, auth, cfg: Path):
    before = cfg.read_bytes()
    response = config_client.put(
        "/api/config", headers=auth, json={"patch": {"engine": {"max_turns": 30}}}
    )
    assert response.status_code == 428
    assert cfg.read_bytes() == before


# ---------------------------------------------------------------- 写 / 备份 / 还原


def test_put_writes_backs_up_and_revert_restores_exact_bytes(
    config_client: TestClient, auth, cfg: Path
):
    original = cfg.read_bytes()
    sha = _sha(cfg)

    response = _put(
        config_client,
        auth,
        sha,
        {
            "engine": {"max_turns": 30},
            "routing_rules": {"agent": ["model-b", "model-a"]},
            "fallback_models": ["model-c"],
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["ok"] is True
    assert result["backup"], "保存必须留一份备份"
    assert result["sha256"] == _sha(cfg) != sha

    saved = json.loads(cfg.read_text(encoding="utf-8"))
    assert saved["engine"]["max_turns"] == 30
    assert saved["routing_rules"]["agent"] == ["model-b", "model-a"]
    assert saved["fallback_models"] == ["model-c"]
    assert saved["api_key"] == RAW_KEY
    # 换行风格不被重排（Windows 文件是 CRLF）
    assert b"\r\n" in cfg.read_bytes()

    changes = {row["path"]: row for row in result["diff"]}
    assert changes["engine.max_turns"]["before"] == 60
    assert changes["engine.max_turns"]["after"] == 30
    assert changes["routing_rules.agent"]["after"] == ["model-b", "model-a"]
    assert changes["fallback_models"]["before"] == ["model-b"]
    assert "api_key" not in changes

    backup_file = cfg.parent / "backups" / result["backup"]
    assert backup_file.is_file()
    assert backup_file.read_bytes() == original

    listing = config_client.get("/api/config/backups", headers=auth)
    assert listing.status_code == 200
    names = [item["name"] for item in listing.json()["items"]]
    assert result["backup"] in names

    fresh = _get(config_client, auth)
    assert fresh["engine"]["max_turns"] == 30
    assert fresh["sha256"] == result["sha256"]

    revert = config_client.post(f"/api/config/revert/{result['backup']}", headers=auth)
    assert revert.status_code == 200, revert.text
    assert revert.json()["restored_from"] == result["backup"]
    assert cfg.read_bytes() == original, "还原必须逐字节回到原文"
    assert _get(config_client, auth)["engine"]["max_turns"] == 60


def test_noop_patch_does_not_create_backup(config_client: TestClient, auth, cfg: Path):
    before = cfg.read_bytes()
    response = _put(config_client, auth, _sha(cfg), {"engine": {"max_turns": 60}})
    assert response.status_code == 200
    assert response.json()["diff"] == []
    assert response.json()["backup"] is None
    assert cfg.read_bytes() == before
    assert not (cfg.parent / "backups").exists()


def test_revert_rejects_bad_names(config_client: TestClient, auth, cfg: Path):
    before = cfg.read_bytes()
    assert config_client.post("/api/config/revert/evil.json", headers=auth).status_code == 400
    assert (
        config_client.post("/api/config/revert/llm_config-missing.json", headers=auth).status_code
        == 404
    )
    assert (
        config_client.post("/api/config/revert/..%2Fetc%2Fpasswd", headers=auth).status_code
        in {400, 404, 405}
    )
    assert cfg.read_bytes() == before


# ---------------------------------------------------------------- 并发


def test_second_put_with_same_sha_is_409(config_client: TestClient, auth, cfg: Path):
    sha = _sha(cfg)
    first = _put(config_client, auth, sha, {"engine": {"max_turns": 30}})
    assert first.status_code == 200

    second = _put(config_client, auth, sha, {"engine": {"max_turns": 20}})
    assert second.status_code == 409
    assert "sha256" in second.json()["detail"]

    saved = json.loads(cfg.read_text(encoding="utf-8"))
    assert saved["engine"]["max_turns"] == 30, "过期请求不能覆盖新值"


# ---------------------------------------------------------------- 运行中锁


def test_put_is_409_while_a_run_is_active(busy_client: TestClient, auth, cfg: Path):
    before = cfg.read_bytes()
    response = _put(busy_client, auth, _sha(cfg), {"engine": {"max_turns": 30}})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "正在运行" in detail and "下一个 run" in detail
    assert cfg.read_bytes() == before
    assert _get(busy_client, auth)["_meta"]["busy"] is True


def test_revert_is_409_while_a_run_is_active(busy_client: TestClient, auth, cfg: Path):
    before = cfg.read_bytes()
    backup = cfg.parent / "backups" / "llm_config-20260101T000000_000.json"
    backup.parent.mkdir(parents=True)
    backup.write_bytes(before)
    response = busy_client.post(f"/api/config/revert/{backup.name}", headers=auth)
    assert response.status_code == 409
    assert cfg.read_bytes() == before


# ---------------------------------------------------------------- 密钥往返


def test_masked_key_roundtrip_never_overwrites_the_real_key(
    config_client: TestClient, auth, cfg: Path, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger="gis_cli.web.config_api")
    before = cfg.read_bytes()
    masked = _get(config_client, auth)["api_key"]
    assert masked == "sk-…cdef"

    rejected = _put(config_client, auth, _sha(cfg), {}, api_key=masked)
    assert rejected.status_code == 400
    assert "脱敏" in rejected.json()["detail"]
    assert cfg.read_bytes() == before

    # 留空 = 不改：即使带了别的合法改动，也只写合法键
    noop = _put(config_client, auth, _sha(cfg), {}, api_key="   ")
    assert noop.status_code == 200
    assert json.loads(cfg.read_text(encoding="utf-8"))["api_key"] == RAW_KEY

    new_key = "sk-live-ffffffff9999"
    saved = _put(
        config_client, auth, _sha(cfg), {"engine": {"max_repairs": 2}}, api_key=new_key
    )
    assert saved.status_code == 200, saved.text
    assert new_key not in saved.text, "响应体不能回显新 key"
    assert json.loads(cfg.read_text(encoding="utf-8"))["api_key"] == new_key
    assert _get(config_client, auth)["api_key"] == "sk-…9999"
    # 日志只记“是否改动”，不记值
    assert new_key not in caplog.text
    assert RAW_KEY not in caplog.text

    key_row = next(row for row in saved.json()["diff"] if row["path"] == "api_key")
    assert key_row["secret"] is True
    assert key_row["before"] == "sk-…cdef"
    assert key_row["after"] == "sk-…9999"


def test_put_rejects_unknown_body_fields(config_client: TestClient, auth, cfg: Path):
    """拼错字段名（patches）要报错，而不是“200 但什么都没写”。"""
    before = cfg.read_bytes()
    response = config_client.put(
        "/api/config",
        headers=auth,
        json={"base_sha256": _sha(cfg), "patches": {"engine": {"max_turns": 30}}},
    )
    assert response.status_code == 422
    assert cfg.read_bytes() == before


# ---------------------------------------------------------------- 连接级白名单


def test_put_accepts_connection_keys_and_updates_active_summary(
    config_client: TestClient, auth, cfg: Path
):
    response = _put(
        config_client,
        auth,
        _sha(cfg),
        {
            "provider": "custom",
            "model": "  deepseek-v4.1-flash  ",
            "api_base": "https://proxy.example.com/v1",
            "timeout": 300,
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["backup"], "真改动必须留备份"
    assert {"provider", "model", "api_base", "timeout"} <= {
        row["path"] for row in result["diff"]
    }

    saved = json.loads(cfg.read_text(encoding="utf-8"))
    assert saved["provider"] == "custom"
    assert saved["model"] == "deepseek-v4.1-flash", "model 要去首尾空白"
    assert saved["api_base"] == "https://proxy.example.com/v1"
    assert saved["timeout"] == 300
    assert saved["api_key"] == RAW_KEY, "改连接信息不能顺手清掉 key"

    fresh = _get(config_client, auth)
    assert fresh["_meta"]["active"] == {
        "provider": "custom",
        "model": "deepseek-v4.1-flash",
        "api_base": "https://proxy.example.com/v1",
        "key_tail": "cdef",
        "config_exists": True,
    }


def test_put_model_only_syncs_routing_rules(config_client: TestClient, auth, cfg: Path):
    """只改 model（配置页「模型与 API」卡片的默认路径）必须把路由一起指过去。

    引擎里 routing_rules 排在 model 前面（DAIR-265：只改 model 后新 run 6 次全是旧模型），
    所以保存后不仅要写出新 model，还要让 EngineLLMConfig.models_for() 真的选到它。
    """
    response = _put(config_client, auth, _sha(cfg), {"model": "model-c"})
    assert response.status_code == 200, response.text
    changes = {row["path"]: row for row in response.json()["diff"]}
    assert changes["model"]["after"] == "model-c"
    # 10 个角色全部同步（diff 里逐条列出，不能只显示 model 一行）
    for role in ROLES:
        row = changes[f"routing_rules.{role}"]
        assert row["before"] == ["model-a", "model-b"]
        assert row["after"] == ["model-c"]
    assert len(changes) == 1 + len(ROLES)

    saved = json.loads(cfg.read_text(encoding="utf-8"))
    assert saved["model"] == "model-c"
    assert all(models == ["model-c"] for models in saved["routing_rules"].values())

    engine_cfg = EngineLLMConfig.from_file(cfg)
    for role in ROLES:
        assert engine_cfg.models_for(task_type=role)[0] == "model-c", role


def test_put_model_change_keeps_explicitly_patched_role(
    config_client: TestClient, auth, cfg: Path
):
    """patch 里显式给了某个角色的路由，就按用户的来；其余角色才跟随 model。"""
    response = _put(
        config_client,
        auth,
        _sha(cfg),
        {"model": "model-c", "routing_rules": {"agent": ["model-b", "model-a"]}},
    )
    assert response.status_code == 200, response.text
    saved = json.loads(cfg.read_text(encoding="utf-8"))
    assert saved["routing_rules"]["agent"] == ["model-b", "model-a"]
    others = [role for role in ROLES if role != "agent"]
    assert all(saved["routing_rules"][role] == ["model-c"] for role in others)


def test_put_routing_only_does_not_touch_model(config_client: TestClient, auth, cfg: Path):
    """只改路由（没改 model）时不做联动：model 保持原值。"""
    response = _put(config_client, auth, _sha(cfg), {"routing_rules": {"verify": ["model-b"]}})
    assert response.status_code == 200, response.text
    assert {row["path"] for row in response.json()["diff"]} == {"routing_rules.verify"}
    saved = json.loads(cfg.read_text(encoding="utf-8"))
    assert saved["model"] == "model-a"
    assert saved["routing_rules"]["verify"] == ["model-b"]
    assert saved["routing_rules"]["agent"] == ["model-a", "model-b"]


@pytest.mark.parametrize(
    "api_base",
    ["", "https://api.openai.com/v1", "http://127.0.0.1:8789/v1", "http://127.0.0.1:8789/v1/"],
)
def test_api_base_accepts_empty_and_clean_http_urls(
    config_client: TestClient, auth, cfg: Path, api_base: str
):
    response = _put(config_client, auth, _sha(cfg), {"api_base": api_base})
    assert response.status_code == 200, response.text
    assert json.loads(cfg.read_text(encoding="utf-8"))["api_base"] == api_base


@pytest.mark.parametrize(
    "api_base",
    [
        "ftp://example.com/v1",
        "example.com/v1",
        "http://127.0.0.1:8789/v1?api_key=sk-abc",
        "https://example.com /v1",
        "https:///v1",
        "https://exa\nmple.com/v1",
        # ``#`` / NUL / userinfo 会被 SDK 静默丢弃或写进日志，必须落盘前拒掉。
        "https://example.com/v1#frag",
        "http://exa\x00mple.com/v1",
        "https://user:pass@example.com/v1",
        123,
    ],
)
def test_api_base_rejects_malformed_values(
    config_client: TestClient, auth, cfg: Path, api_base: Any
):
    before = cfg.read_bytes()
    response = _put(config_client, auth, _sha(cfg), {"api_base": api_base})
    assert response.status_code == 400, response.text
    assert "api_base" in response.json()["detail"]
    assert cfg.read_bytes() == before


@pytest.mark.parametrize("model", ["", "   ", "x" * 201, 123])
def test_model_rejects_empty_and_overlong(config_client: TestClient, auth, cfg: Path, model: Any):
    before = cfg.read_bytes()
    response = _put(config_client, auth, _sha(cfg), {"model": model})
    assert response.status_code == 400, response.text
    assert "model" in response.json()["detail"]
    assert cfg.read_bytes() == before


@pytest.mark.parametrize("provider", ["azure", "OpenAI", "", 123])
def test_provider_rejects_non_preset(config_client: TestClient, auth, cfg: Path, provider: Any):
    before = cfg.read_bytes()
    response = _put(config_client, auth, _sha(cfg), {"provider": provider})
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "provider" in detail or "供应商" in detail
    assert cfg.read_bytes() == before


@pytest.mark.parametrize("timeout", [4, 3601, "300", True])
def test_timeout_rejects_out_of_range(config_client: TestClient, auth, cfg: Path, timeout: Any):
    before = cfg.read_bytes()
    response = _put(config_client, auth, _sha(cfg), {"timeout": timeout})
    assert response.status_code == 400, response.text
    assert "timeout" in response.json()["detail"]
    assert cfg.read_bytes() == before


def test_timeout_accepts_bounds_and_keeps_number_type(config_client: TestClient, auth, cfg: Path):
    assert _put(config_client, auth, _sha(cfg), {"timeout": 5}).status_code == 200
    sha = _get(config_client, auth)["sha256"]
    assert _put(config_client, auth, sha, {"timeout": 3600}).status_code == 200
    assert json.loads(cfg.read_text(encoding="utf-8"))["timeout"] == 3600


def test_providers_endpoint_is_static_preset_table(config_client: TestClient, auth):
    response = config_client.get("/api/config/providers", headers=auth)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == len(body["items"])
    items = {item["id"]: item for item in body["items"]}
    for preset_id in (
        "openai",
        "deepseek",
        "anthropic",
        "dashscope",
        "moonshot",
        "zhipu",
        "ollama",
        "custom",
    ):
        assert preset_id in items
        assert set(items[preset_id]) == {
            "id",
            "name",
            "default_base",
            "key_required",
            "hint",
        }
    assert items["openai"]["default_base"] == "https://api.openai.com/v1"
    assert items["deepseek"]["default_base"] == "https://api.deepseek.com/v1"
    assert items["ollama"]["default_base"] == "http://127.0.0.1:11434/v1"
    assert items["ollama"]["key_required"] is False
    assert items["custom"]["default_base"] == ""
    assert "反代" in items["custom"]["hint"]


# ---------------------------------------------------------------- 测试连接


@pytest.fixture()
def fake_openai(monkeypatch: pytest.MonkeyPatch):
    """假 OpenAI 客户端：记录构造参数 + 可注入模型清单/异常，全程不打网络。"""
    import openai

    state: dict[str, Any] = {
        "calls": [],
        "ids": ["model-a", "model-b", "model-c"],
        "error": None,
    }

    class FakeModels:
        def list(self):
            if state["error"] is not None:
                raise state["error"]
            return SimpleNamespace(data=[SimpleNamespace(id=name) for name in state["ids"]])

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            state["calls"].append(kwargs)
            self.models = FakeModels()
            state["closed"] = 0

        def __enter__(self) -> "FakeOpenAI":
            return self

        def __exit__(self, *exc: Any) -> None:
            state["closed"] = int(state.get("closed") or 0) + 1

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    return state


def test_config_test_success_uses_saved_key(
    config_client: TestClient, auth, fake_openai: dict[str, Any]
):
    response = config_client.post(
        "/api/config/test",
        headers=auth,
        json={"provider": "custom", "api_base": "https://proxy.example.com/v1"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["models"] == ["model-a", "model-b", "model-c"]
    assert body["error"] == ""
    assert body["latency_ms"] >= 0
    assert body["key_source"] == "config"
    assert len(fake_openai["calls"]) == 1
    call = fake_openai["calls"][0]
    assert call["api_key"] == RAW_KEY
    assert call["base_url"] == "https://proxy.example.com/v1"
    assert call["max_retries"] == 0
    # 连接阶段 3s、单段 8s 封顶（DAIR-270：整次探测还要受外层 deadline 约束）
    # 内层 7.0 必须小于外层 TEST_DEADLINE=7.5，慢网关才能报出真实 httpx 超时原因
    assert (call["timeout"].connect, call["timeout"].read) == (3.0, 7.0)
    assert RAW_KEY not in response.text, "响应体不能回显 key"


def test_config_test_prefers_request_key_and_caps_models(
    config_client: TestClient, auth, fake_openai: dict[str, Any]
):
    fake_openai["ids"] = [f"m{index}" for index in range(30)]
    new_key = "sk-request-0123456789"
    response = config_client.post(
        "/api/config/test",
        headers=auth,
        json={"api_base": "http://127.0.0.1:8789/v1", "api_key": new_key},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["key_source"] == "request"
    assert len(body["models"]) == 20
    assert fake_openai["calls"][0]["api_key"] == new_key
    assert new_key not in response.text


def test_config_test_failure_is_200_and_sanitized(
    config_client: TestClient, auth, fake_openai: dict[str, Any]
):
    fake_openai["error"] = RuntimeError(
        f"Connection error for https://proxy.example.com/v1?api_key={RAW_KEY} "
        f"while using {RAW_KEY}"
    )
    response = config_client.post(
        "/api/config/test",
        headers=auth,
        json={"api_base": "https://proxy.example.com/v1"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is False
    assert body["models"] == []
    assert body["latency_ms"] >= 0
    assert body["error"], "失败必须给出原因"
    assert RAW_KEY not in response.text, "错误串不能回显完整 key"
    assert "?api_key" not in body["error"], "错误串里的查询串必须抹掉"
    assert "sk-***" in body["error"] or "***" in body["error"]


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({"api_base": ""}, "先填地址"),
        ({"api_base": "   "}, "先填地址"),
        ({"api_base": "ftp://example.com/v1"}, "http"),
        ({"api_base": "https://x.example/v1", "provider": "azure"}, "未知供应商"),
        (
            {"api_base": "https://x.example/v1", "api_key": "sk-…cdef"},
            "脱敏",
        ),
    ],
)
def test_config_test_rejects_bad_input_without_network(
    config_client: TestClient,
    auth,
    fake_openai: dict[str, Any],
    payload: dict[str, Any],
    fragment: str,
):
    response = config_client.post("/api/config/test", headers=auth, json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is False
    assert fragment in body["error"]
    assert fake_openai["calls"] == [], "非法输入不该发起探测"


def test_config_test_allows_ollama_with_saved_key(
    config_client: TestClient, auth, fake_openai: dict[str, Any]
):
    """没传 api_key = 沿用配置里的真 key；provider 只是元信息，不影响探测。"""
    response = config_client.post(
        "/api/config/test",
        headers=auth,
        json={"api_base": "http://127.0.0.1:11434/v1", "provider": "ollama"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["key_source"] == "config"
    assert fake_openai["calls"][0]["api_key"] == RAW_KEY


# ---------------------------------------------------------------- 空配置


def test_empty_config_reads_as_empty_and_can_be_created(workspace: Path, token: str, auth):
    target = workspace / "config" / "llm_config.json"
    with TestClient(create_app(workspace, token=token, config_path=target)) as client:
        first = client.get("/api/config", headers=auth)
        assert first.status_code == 200, first.text
        before = first.json()
        assert before["sha256"] == ""
        assert before["_meta"]["config_exists"] is False
        assert before["_meta"]["active"] == {
            "provider": "",
            "model": "",
            "api_base": "",
            "key_tail": "",
            "config_exists": False,
        }
        assert not target.exists(), "GET 不能凭空建文件"

        no_token = client.put(
            "/api/config", headers=auth, json={"patch": {"model": "m1"}}
        )
        assert no_token.status_code == 428
        assert not target.exists()

        created = client.put(
            "/api/config",
            headers=auth,
            json={
                "base_sha256": "",
                "patch": {
                    "provider": "custom",
                    "model": "m1",
                    "api_base": "http://127.0.0.1:8789/v1",
                    "timeout": 60,
                },
                "api_key": RAW_KEY,
            },
        )
        assert created.status_code == 200, created.text
        result = created.json()
        assert result["ok"] is True
        assert result["backup"] is None, "没有原文件就没有可备份的东西"
        assert not (target.parent / "backups").exists()
        assert target.is_file()

        saved = json.loads(target.read_text(encoding="utf-8"))
        assert saved["model"] == "m1"
        assert saved["api_key"] == RAW_KEY

        after = client.get("/api/config", headers=auth).json()
        assert after["sha256"] == result["sha256"] != ""
        assert after["api_key"] == "sk-…cdef"
        assert after["_meta"]["config_exists"] is True
        assert after["_meta"]["active"]["model"] == "m1"


def test_noop_put_on_missing_config_returns_empty_sha(workspace: Path, token: str, auth):
    """空配置下 no-op PUT 的令牌必须与 GET 一致（否则前端首次引导会 409）。"""
    target = workspace / "config" / "llm_config.json"
    with TestClient(create_app(workspace, token=token, config_path=target)) as client:
        response = client.put(
            "/api/config", headers=auth, json={"base_sha256": "", "patch": {}}
        )
        assert response.status_code == 200, response.text
        assert response.json()["sha256"] == ""
        assert not target.exists(), "no-op 不能凭空建文件"

        fetched = client.get("/api/config", headers=auth).json()
        assert fetched["sha256"] == ""
        assert fetched["_meta"]["config_exists"] is False


def test_config_test_on_corrupt_config_is_200_and_not_500(
    workspace: Path, token: str, auth, fake_openai: dict[str, Any]
):
    """配置损坏时 /test 仍要 200 + ok:false，不能把配置页打成 500。"""
    target = workspace / "config" / "llm_config.json"
    target.parent.mkdir(parents=True)
    target.write_text("{not json", encoding="utf-8")
    with TestClient(create_app(workspace, token=token, config_path=target)) as client:
        response = client.post(
            "/api/config/test",
            headers=auth,
            json={"api_base": "https://example.com/v1"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["ok"] is False
        assert body["models"] == []
        assert "unreadable" in body["error"]
        assert fake_openai["calls"] == [], "配置读不出来就不该发起探测"


def test_config_test_hard_deadline_cuts_slow_probe(
    config_client: TestClient, auth, fake_openai: dict[str, Any], monkeypatch: pytest.MonkeyPatch
):
    """慢网关不能把测试连接拖过设计界：外层 deadline 到点就返回 ok:false。

    httpx 的 timeout 是每段超时，不是整次请求的总时长；DAIR-270 实测真实反代
    成功路径 9912ms。这里把 deadline 压到 0.2s，验证接口本身有硬截止。
    """
    import time as _time

    from gis_cli.web import config_api

    class SlowModels:
        def list(self):
            _time.sleep(3.0)
            return SimpleNamespace(data=[SimpleNamespace(id="late-model")])

    monkeypatch.setattr(config_api, "TEST_DEADLINE", 0.2)

    import openai

    class SlowOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.models = SlowModels()

        def __enter__(self) -> "SlowOpenAI":
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    monkeypatch.setattr(openai, "OpenAI", SlowOpenAI)

    started = _time.perf_counter()
    response = config_client.post(
        "/api/config/test",
        headers=auth,
        json={"api_base": "https://proxy.example.com/v1"},
    )
    elapsed = _time.perf_counter() - started
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is False
    assert "0.2s" in body["error"] and "未返回" in body["error"]
    assert elapsed < 1.5, f"外层 deadline 没生效，耗时 {elapsed:.2f}s"


def test_probe_bypasses_system_proxy_for_loopback(
    config_client: TestClient, auth, fake_openai: dict[str, Any]
) -> None:
    """本机地址不走系统代理：Windows 注册表里的代理会给 localhost 加几秒甚至假 502。"""
    from gis_cli.web import config_api

    assert config_api._is_loopback("http://127.0.0.1:8789/v1") is True
    assert config_api._is_loopback("http://localhost:11434/v1") is True
    assert config_api._is_loopback("http://[::1]:8760/v1") is True
    assert config_api._is_loopback("https://api.openai.com/v1") is False
    assert config_api._is_loopback("") is False

    config_client.post(
        "/api/config/test",
        headers=auth,
        json={"api_base": "http://127.0.0.1:8789/v1"},
    )
    loopback_call = fake_openai["calls"][-1]
    assert loopback_call["http_client"].trust_env is False

    config_client.post(
        "/api/config/test",
        headers=auth,
        json={"api_base": "https://proxy.example.com/v1"},
    )
    # 远端地址沿用 SDK 自己的客户端（系统代理照旧生效）
    assert "http_client" not in fake_openai["calls"][-1]


def test_config_test_surfaces_connection_cause(
    config_client: TestClient, auth, fake_openai: dict[str, Any]
):
    """SDK 的 "Connection error." 太空：要把底层 WinError 带出来，用户才知道是连不上。"""
    cause = OSError("[WinError 10061] 目标计算机积极拒绝，无法连接。")
    error = RuntimeError("Connection error.")
    error.__cause__ = cause
    fake_openai["error"] = error

    response = config_client.post(
        "/api/config/test",
        headers=auth,
        json={"api_base": "http://127.0.0.1:9/v1"},
    )
    body = response.json()
    assert body["ok"] is False
    assert "Connection error." in body["error"]
    assert "10061" in body["error"]


def test_probe_timeout_is_below_outer_deadline(config_client: TestClient, auth, fake_openai: dict[str, Any]):
    """内层 httpx 超时必须先于外层 deadline 触发，慢网关才报得出真实原因（DAIR-271 建议 4）。"""
    from gis_cli.web import config_api

    assert config_api.TEST_TIMEOUT < config_api.TEST_DEADLINE
    assert config_api.TEST_CONNECT_TIMEOUT <= config_api.TEST_TIMEOUT


def test_probe_closes_the_client(config_client: TestClient, auth, fake_openai: dict[str, Any]):
    """每点一次「测试连接」不能留一个未关闭的 OpenAI/httpx 客户端（DAIR-271 建议 5）。"""
    response = config_client.post(
        "/api/config/test",
        headers=auth,
        json={"api_base": "http://127.0.0.1:8789/v1"},
    )
    assert response.json()["ok"] is True
    assert fake_openai["closed"] == 1

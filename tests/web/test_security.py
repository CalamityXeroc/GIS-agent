# -*- coding: utf-8 -*-
"""token 鉴权、路径白名单、密钥脱敏。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gis_cli.web.security import load_or_create_token, mask_secret, safe_path


def test_api_requires_token(client: TestClient, token: str):
    assert client.get("/api/traces").status_code == 401
    assert client.get("/api/traces", headers={"X-Web-Token": "wrong"}).status_code == 401
    assert client.get("/api/traces", headers={"X-Web-Token": token}).status_code == 200


def test_non_ascii_token_is_401_not_500(client: TestClient):
    assert client.get("/api/traces", params={"token": "中文"}).status_code == 401
    # 头按 latin-1 解码成 str，同样不能让它抛 TypeError 打成 500
    response = client.get("/api/traces", headers={b"X-Web-Token": b"\xff\xfe"})
    assert response.status_code == 401


def test_api_accepts_token_in_query(client: TestClient, token: str):
    response = client.get(f"/api/traces?token={token}")
    assert response.status_code == 200


def test_token_file_is_stable_and_32_bytes(workspace: Path):
    first = load_or_create_token(workspace)
    second = load_or_create_token(workspace)
    assert first == second
    assert len(first) == 64
    int(first, 16)  # 十六进制
    assert (workspace / ".gis_agent" / "web_token").read_text(encoding="utf-8").strip() == first


def test_path_traversal_is_403(workspace: Path, client: TestClient, auth: dict[str, str]):
    response = client.get(
        "/api/catalog/datasets/profile", params={"path": "../../x"}, headers=auth
    )
    assert response.status_code == 403


def test_safe_path_allows_only_workspace(workspace: Path):
    inside = workspace / "input" / "a.shp"
    assert safe_path(workspace, "input/a.shp") == inside.resolve()
    with pytest.raises(HTTPException) as exc:
        safe_path(workspace, "../../outside.shp")
    assert exc.value.status_code == 403


def test_mask_secret_keeps_only_last_four():
    assert mask_secret("sk-live-1234567890abcd") == "sk-…abcd"
    assert mask_secret("") == ""
    assert mask_secret("short") == "***"


def test_config_response_masks_api_key(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    config_dir = workspace / "config"
    config_dir.mkdir(parents=True)
    raw_key = "sk-live-1234567890abcd"
    nested_key = "sk-search-9876543210wxyz"
    (config_dir / "llm_config.json").write_text(
        json.dumps(
            {
                "model": "test-model",
                "api_key": raw_key,
                "web_search": {"api_key": nested_key, "enabled": True},
                "engine": {"max_turns": 60},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    response = client.get("/api/config", headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert raw_key not in response.text
    assert nested_key not in response.text
    assert body["api_key"] == "sk-…abcd"
    assert body["web_search"]["api_key"] == "sk-…wxyz"
    assert body["engine"] == {"max_turns": 60}


def test_missing_config_is_empty_not_404(
    workspace: Path, token: str, auth: dict[str, str]
):
    """空配置可用（DAIR-253）：GET 返回 {} + sha256:""，前端走首次配置引导。"""
    from gis_cli.web.app import create_app

    client = TestClient(
        create_app(workspace, token=token, config_path=workspace / "missing.json")
    )
    response = client.get("/api/config", headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert body["sha256"] == ""
    assert body["_meta"]["config_exists"] is False


def test_console_page_is_served(client: TestClient):
    """S2 起首页返回前端构建产物，不再是占位文案。"""
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="root"' in response.text
    assert "GIS Agent 控制台" in response.text

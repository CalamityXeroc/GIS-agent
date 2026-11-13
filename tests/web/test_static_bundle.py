# -*- coding: utf-8 -*-
"""S2 前端产物契约：产物存在、零外链、体积上限、SPA 子路由可直达。

「零外链」的判据是**产物不依赖任何外部网络资源**（离线可用），而不是
「文件里一个 http 字符都不许有」，因为下面这些字符串是功能常量或惰性文案，
删掉/改写会让 SVG 或报错信息坏掉，但都不会发起请求：

- ``http://www.w3.org/{2000/svg,1999/xlink,1998/Math/MathML}``、``XML/1998/namespace``
  —— ``createElementNS`` 必须逐字匹配的 XML 命名空间；
- ``https://react.dev/errors/`` —— React 只在控制台报错文案里拼接；
- ``http://localhost`` —— 路由库在 ``window.origin`` 为空时用的同源兜底值；
- ``https://tailwindcss.com`` —— Tailwind 的 MIT 许可证注释。

白名单之外的任何 http(s) 地址都算外链，测试直接失败。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gis_cli.web.app import STATIC_DIR, create_app
from gis_cli.web.security import load_or_create_token

#: 计划 §8：产物总体积上限 8MB。
MAX_TOTAL_BYTES = 8 * 1024 * 1024

#: 可扫描文本的产物后缀（woff2 是二进制，只计体积）。
TEXT_SUFFIXES = {".html", ".css", ".js", ".json", ".svg", ".map"}

URL_RE = re.compile(r"https?://[^\s\"'`)<>,;]+")

INERT_URL_PATTERNS = (
    re.compile(
        r"https?://www\.w3\.org/"
        r"(?:2000/svg|1999/xlink|1998/Math/MathML|XML/1998/namespace)"
    ),
    re.compile(r"https://react\.dev/errors/"),
    re.compile(r"http://localhost"),
    re.compile(r"https://tailwindcss\.com"),
)


def _bundle_files() -> list[Path]:
    assert STATIC_DIR.is_dir(), f"缺少前端产物目录：{STATIC_DIR}（先 npm run build）"
    return sorted(path for path in STATIC_DIR.rglob("*") if path.is_file())


def _read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="ignore")


def test_bundle_is_built() -> None:
    files = _bundle_files()
    index = STATIC_DIR / "index.html"
    assert index.is_file(), "static/index.html 不存在"
    html = index.read_text(encoding="utf-8")
    assert 'id="root"' in html
    assert "GIS Agent 控制台" in html
    # 产物必须带哈希资产，说明是真构建而不是占位页。
    assert re.search(r'/assets/index-[\w-]+\.js', html)
    assert len(files) >= 4


def test_asset_references_are_local() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="([^"]+)"', html)
    assert refs, "index.html 没有任何资源引用"
    for ref in refs:
        assert not ref.startswith(("http://", "https://", "//")), f"外链：{ref}"
        # 服务挂在 "/"，绝对同源路径才能在 /runs/demo-01 这类子路由上刷新可用。
        assert ref.startswith("/"), f"非站内绝对路径：{ref}"
        target = (STATIC_DIR / ref.lstrip("/")).resolve()
        assert target.is_file(), f"引用的资源不存在：{ref}"


def test_no_runtime_external_links() -> None:
    offenders: list[str] = []
    for path in _bundle_files():
        if path.suffix not in TEXT_SUFFIXES:
            continue
        for url in URL_RE.findall(_read_text(path)):
            if not any(pattern.match(url) for pattern in INERT_URL_PATTERNS):
                offenders.append(f"{path.relative_to(STATIC_DIR)}: {url}")
    assert not offenders, "产物里出现可加载的外链：\n" + "\n".join(offenders)


def test_bundle_size_under_limit() -> None:
    total = sum(path.stat().st_size for path in _bundle_files())
    assert total < MAX_TOTAL_BYTES, f"产物 {total / 1024 / 1024:.2f}MB 超过 8MB 上限"


def test_design_tokens_land_in_bundle() -> None:
    """token 原值必须真的编译进产物，避免「改了 CSS 没重新构建」。"""
    css = "".join(
        _read_text(path) for path in _bundle_files() if path.suffix == ".css"
    )
    # 浅色主色 + 深色主色（CSS 压缩器会把 0.692 归一化成 69.2%）。
    assert "243.716" in css, "缺少 --primary 浅色原值"
    assert "248.516" in css, "--primary 深色原值"
    assert "--radius:1rem" in css
    assert "--app-header-height:3rem" in css
    assert "--app-sidebar-width:13rem" in css
    assert "public-sans" in css.lower()


def test_spa_routes_fall_back_to_index(workspace: Path, token: str) -> None:
    """五个页面 + 导航目标都要能直接打开（刷新不 404）。"""
    client = TestClient(create_app(workspace, token=token))
    for route in (
        "/",
        "/runs",
        "/runs/new",
        "/runs/demo-01",
        "/data",
        "/packages",
        "/reports",
        "/config",
        "/env",
    ):
        response = client.get(route)
        assert response.status_code == 200, route
        assert 'id="root"' in response.text, route

    # 未知 API 不能被 SPA 回落吞成 200 HTML。
    response = client.get("/api/nope", headers={"X-Web-Token": token})
    assert response.status_code == 404

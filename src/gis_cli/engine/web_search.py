# -*- coding: utf-8 -*-
"""联网搜索：查 ArcPy/ArcGIS Pro 的最新 API、坐标系 WKID、报错信息等。

## 为什么单独一个模块

``engine/tools.py`` 已经 1300+ 行，再塞爬虫解析会更难读。搜索后端是
**可替换的实现细节**，把它隔离在一个文件里，tools.py 只负责注册工具。

## 后端现状（**实测**，不是照抄文档）

在本机网络环境下逐个测过可达性：

========================  ==========  ========================================
后端                       可达性       备注
========================  ==========  ========================================
DuckDuckGo (html/lite)     ✗ 超时       国内不可达
DuckDuckGo Instant Answer  ✗ 超时       国内不可达
Wikipedia API              ✗ 超时       国内不可达
Google / Searx             ✗ 超时       国内不可达
Bing (HTML)                ✓ 可达       但**返回被地域劫持**：查 "arcpy Buffer"
                                       返回百度知道的无关结果，无法用
百度 (HTML)                ✓ 可达       中文资料尚可，英文/GIS 专业内容差
Tavily API                 ✓（需 key）  专为 AI agent 做，质量最好
========================  ==========  ========================================

**结论**：免费后端在国内都不可靠。所以设计成"按顺序试，全失败就明确报告"，
而不是假装能用。真正想要稳定的联网搜索，配一个 Tavily key（配置
``web_search.TAVILY_API_KEY`` 或环境变量）。

## 设计约束（都来自真实使用场景）

- **不能阻断任务**：竞赛现场常常没网。搜索失败要返回一条明确的失败观察，
  让模型知道"这条路走不通，用已有知识继续"，而不是抛异常把 run 搞崩。
- **不能撑爆 context**：每条结果截断 snippet，总数限 5 条，总字符数封顶。
- **必须可并发**：标记 ``readonly=True``，与其他只读工具一起并发执行。
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

#: 单条摘要最多保留多少字符（搜索结果原文常常几百字，全塞进去很浪费）
SNIPPET_LIMIT = 400
#: 一次搜索最多返回几条
MAX_RESULTS = 8
#: 全部结果加起来的字符上限（防 context 膨胀）
TOTAL_LIMIT = 3000
#: 单个后端的网络超时（秒）。搜索是"锦上添花"，不值得让用户等太久。
TIMEOUT = 10.0
#: 最多试几个后端（避免没网时每个都等满 TIMEOUT）
MAX_BACKENDS = 2

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _UA,
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_TAG_RE = re.compile(r"<[^>]+>")


# ------------------------------------------------------------------ 工具函数
def _clean(text: str) -> str:
    """去 HTML 标签 + 反转义 + 压缩空白。"""
    text = _TAG_RE.sub(" ", text or "")
    text = html.unescape(text)
    return " ".join(text.split())


def _clip(text: str, limit: int = SNIPPET_LIMIT) -> str:
    text = _clean(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fetch(url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None) -> str:
    merged = dict(_HEADERS)
    if headers:
        merged.update(headers)
    request = urllib.request.Request(url, data=data, headers=merged, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read().decode("utf-8", errors="replace")


def _read_config_key(name: str) -> str:
    """从 config/llm_config.json 的 ``web_search`` 段读 key（可选）。"""
    for candidate in (
        os.path.join(os.getcwd(), "config", "llm_config.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "llm_config.json"),
    ):
        try:
            if not os.path.exists(candidate):
                continue
            with open(candidate, encoding="utf-8") as fh:
                data = json.load(fh)
            section = data.get("web_search") or {}
            value = section.get(name) if isinstance(section, dict) else None
            if value:
                return str(value)
        except Exception:
            continue
    return ""


def tavily_key() -> str:
    return os.environ.get("TAVILY_API_KEY", "") or _read_config_key("TAVILY_API_KEY")


# ------------------------------------------------------------------ 后端
def _search_tavily(query: str, limit: int, key: str) -> list[dict[str, str]]:
    """Tavily：专为 AI agent 做的搜索 API，质量最好（需要 key）。"""
    payload = json.dumps(
        {"api_key": key, "query": query, "max_results": limit, "search_depth": "basic"}
    ).encode("utf-8")
    raw = _fetch(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    data = json.loads(raw)
    out: list[dict[str, str]] = []
    for item in (data.get("results") or [])[:limit]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "title": _clean(str(item.get("title") or "")),
                "snippet": _clip(str(item.get("content") or "")),
                "url": str(item.get("url") or ""),
            }
        )
    return out


_DDG_BLOCK_RE = re.compile(r'<div[^>]*class="[^"]*result__body[^"]*"', re.IGNORECASE)
_DDG_LINK_RE = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_DDG_SNIPPET_RE = re.compile(
    r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)


def _unwrap_ddg_url(url: str) -> str:
    """DuckDuckGo 会把结果 URL 包成 ``//duckduckgo.com/l/?uddg=<真实地址>``。"""
    if "uddg=" not in url:
        return url
    try:
        query = urllib.parse.urlparse(url).query
        return urllib.parse.parse_qs(query).get("uddg", [url])[0] or url
    except Exception:
        return url


def _search_duckduckgo(query: str, limit: int) -> list[dict[str, str]]:
    """DuckDuckGo HTML 版。国内通常不可达，作为备选保留。"""
    page = _fetch("https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query))
    parts = _DDG_BLOCK_RE.split(page)
    blocks = parts[1:] if len(parts) > 1 else [page]
    out: list[dict[str, str]] = []
    for block in blocks:
        if len(out) >= limit:
            break
        link = _DDG_LINK_RE.search(block)
        if not link:
            continue
        href = _unwrap_ddg_url(html.unescape(link.group(1)))
        title = _clean(link.group(2))
        if not href.startswith("http") or not title:
            continue
        snippet_match = _DDG_SNIPPET_RE.search(block)
        out.append(
            {
                "title": title,
                "snippet": _clip(snippet_match.group(1)) if snippet_match else title,
                "url": href,
            }
        )
    return out


def _search_bing(query: str, limit: int) -> list[dict[str, str]]:
    """Bing HTML。

    **注意**：实测在本机网络环境下，Bing 对无 cookie 的自动化请求会返回
    **地域劫持的结果**（查 "arcpy Buffer" 返回百度知道的无关条目）。
    所以它排在 DuckDuckGo 之后，只作为最后尝试。
    """
    url = "https://www.bing.com/search?" + urllib.parse.urlencode(
        {"q": query, "cc": "us", "setlang": "en"}
    )
    page = _fetch(url)
    blocks = re.findall(r'<li class="b_algo".*?(?=<li class="b_algo"|</ol>)', page, re.DOTALL)
    out: list[dict[str, str]] = []
    for block in blocks[:limit]:
        match = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
        if not match:
            continue
        title = _clean(match.group(2))
        href = html.unescape(match.group(1))
        if not title or not href.startswith("http"):
            continue
        snippet_match = re.search(r"<p[^>]*>(.*?)</p>", block, re.DOTALL)
        out.append(
            {
                "title": title,
                "snippet": _clip(snippet_match.group(1)) if snippet_match else title,
                "url": href,
            }
        )
    return out


def _relevant(query: str, item: dict[str, str]) -> bool:
    """结果是否真的跟查询相关。

    **为什么必须做这一步（实测教训）**：本机网络环境下，Bing 对无 cookie
    的自动化请求会返回**完全无关的降级页面** —— 查 ``arcpy Buffer tool
    parameters`` 返回 NFL 官网、网易邮箱、百度知道。

    比"不可用"更糟：模型会把 NFL 当成 ArcPy 资料写进答案。宁可不给结果，
    也不能给错误结果。

    判据故意宽松（只要有任意一个关键词命中就算相关），因为搜索结果的
    措辞千差万别；但能拦住"完全不搭"的降级页。
    """
    haystack = f"{item.get('title', '')} {item.get('snippet', '')} {item.get('url', '')}".lower()
    words = [w for w in re.split(r"[\s,;:!?()\[\]{}'\"/\\\\|]+", query.lower()) if len(w) >= 3]
    if not words:
        return True
    return any(word in haystack for word in words)


def _filter_relevant(query: str, results: list[dict[str, str]]) -> list[dict[str, str]]:
    """过滤掉与查询完全无关的结果（见 :func:`_relevant` 里的实测说明）。"""
    return [item for item in results if _relevant(query, item)]


#: 免费后端清单：**名字 → 函数名**（不是函数对象）。
#:
#: 为什么存名字而不是对象：存对象会在模块导入时就把函数绑定死，
#: 测试里 monkeypatch ``_search_duckduckgo`` 就换不掉（踩过）。存名字、
#: 用时从 globals() 取，patch 才能生效。
_BACKEND_NAMES: tuple[str, ...] = ("_search_duckduckgo", "_search_bing")


# ------------------------------------------------------------------ 对外
def search(query: str, *, limit: int = 5) -> dict[str, Any]:
    """执行一次搜索。

    返回 ``{"query", "results", "backend", "error"}``；
    ``error`` 非空表示失败（调用方据此给模型一条明确的失败观察）。
    **不抛异常** —— 搜索失败不该让任务崩。
    """
    query = " ".join(str(query or "").split())
    if not query:
        return {"query": "", "results": [], "backend": "", "error": "查询为空"}
    limit = max(1, min(int(limit or 5), MAX_RESULTS))

    attempts: list[str] = []

    # 1) 配了 Tavily 就先用它（唯一可靠的选择）
    key = tavily_key()
    if key:
        try:
            results = _filter_relevant(query, _search_tavily(query, limit, key))
            if results:
                return {"query": query, "results": results, "backend": "tavily", "error": ""}
            attempts.append("tavily: 无相关结果")
        except Exception as exc:
            attempts.append(f"tavily: {type(exc).__name__}")

    # 2) 免费后端按顺序试，最多试 MAX_BACKENDS 个
    for func_name in _BACKEND_NAMES[:MAX_BACKENDS]:
        backend = globals().get(func_name)
        if backend is None:  # pragma: no cover - 防御
            continue
        name = func_name.removeprefix("_search_")
        try:
            raw = backend(query, limit)
            results = _filter_relevant(query, raw)
            if results:
                return {"query": query, "results": results, "backend": name, "error": ""}
            # 区分“真的没结果”与“返回了一堆不相干的”（后者是降级页）
            if raw:
                attempts.append(f"{name}: 返回 {len(raw)} 条但均与查询无关（降级页？）")
            else:
                attempts.append(f"{name}: 无结果")
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            attempts.append(f"{name}: 不可达（{reason}）")
        except Exception as exc:
            attempts.append(f"{name}: {type(exc).__name__}")

    return {
        "query": query,
        "results": [],
        "backend": "",
        "error": "；".join(attempts) or "无可用后端",
    }


def format_results(payload: dict[str, Any]) -> str:
    """把搜索结果格式化成给模型看的紧凑文本（总长受 TOTAL_LIMIT 约束）。"""
    results = payload.get("results") or []
    if not results:
        return ""
    lines: list[str] = []
    total = 0
    for index, item in enumerate(results, 1):
        block = (
            f"[{index}] {item.get('title', '')}\n"
            f"    {item.get('snippet', '')}\n"
            f"    {item.get('url', '')}"
        )
        if total + len(block) > TOTAL_LIMIT:
            lines.append(f"…（还有 {len(results) - index + 1} 条未显示）")
            break
        lines.append(block)
        total += len(block)
    return "\n".join(lines)

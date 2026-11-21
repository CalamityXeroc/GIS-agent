# -*- coding: utf-8 -*-
"""联网搜索工具（新引擎）。

设计约束（都来自真实使用场景，不是假想）：

1. **没网不能崩**：竞赛现场常常无网。搜索失败要返回一条明确的失败观察，
   让模型知道"这条路走不通，用已有知识继续" —— 而不是抛异常把整个 run 搞崩。
2. **不能撑爆 context**：搜索结果动辄上万字。单条摘要要截断、条数要限制、
   总字符要封顶，否则会把真正重要的上下文挤掉。
3. **解析要容错**：DuckDuckGo 是抓 HTML（不是官方 API），页面结构一变就
   返回空结果而不是抛异常。
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from gis_cli.engine import web_search as ws


# ------------------------------------------------------------------ 解析
def test_parse_duckduckgo_html_extracts_results(monkeypatch):
    """正常页面要能解析出 title/snippet/url，并解开 DDG 的跳转包装。"""
    page = """
    <div class="result__body">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fpro.arcgis.com%2Fbuffer.htm">
        Buffer (Analysis)—ArcGIS Pro
      </a>
      <a class="result__snippet">Buffer creates polygons at a specified distance.</a>
    </div>
    <div class="result__body">
      <a class="result__a" href="https://desktop.arcgis.com/xy.htm">XY Table To Point</a>
      <a class="result__snippet">Creates a point feature class from tabular data.</a>
    </div>
    """

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(ws, "_fetch", lambda *a, **k: page)
    results = ws._search_duckduckgo("buffer", 5)

    assert len(results) == 2
    assert results[0]["title"] == "Buffer (Analysis)—ArcGIS Pro"
    # uddg 包装必须被解开
    assert results[0]["url"] == "https://pro.arcgis.com/buffer.htm"
    assert "specified distance" in results[0]["snippet"]
    assert results[1]["url"] == "https://desktop.arcgis.com/xy.htm"


def test_parse_survives_page_structure_change(monkeypatch):
    """页面结构变了（没有 result__body）→ 返回空，不能抛。"""

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(ws, "_fetch", lambda *a, **k: "<html>totally new layout</html>")
    assert ws._search_duckduckgo("x", 5) == []


# ------------------------------------------------------------------ 失败降级
def test_search_returns_error_instead_of_raising_on_network_failure(monkeypatch):
    """没网时必须返回 error 字段，而不是抛异常（这是常规路径，不是异常路径）。"""

    def _boom(*args, **kwargs):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(ws, "_fetch", _boom)
    monkeypatch.setattr(ws, "tavily_key", lambda: "")  # 确保走免费后端
    payload = ws.search("arcpy buffer")
    assert payload["results"] == []
    assert payload["error"], "必须给出失败原因"
    assert "网络" in payload["error"] or "no network" in payload["error"]


def test_empty_query_is_rejected():
    payload = ws.search("   ")
    assert payload["results"] == []
    assert payload["error"] == "查询为空"


def test_tavily_failure_falls_back_to_duckduckgo(monkeypatch):
    """配了 Tavily 但它挂了 → 自动退回免费的 DDG，而不是直接失败。"""
    monkeypatch.setattr(ws, "tavily_key", lambda: "fake-key")

    def _tavily_boom(*args, **kwargs):
        raise RuntimeError("tavily down")

    monkeypatch.setattr(ws, "_search_tavily", _tavily_boom)
    monkeypatch.setattr(
        ws,
        "_search_duckduckgo",
        lambda q, n: [{"title": "ok", "snippet": "s", "url": "https://x"}],
    )
    monkeypatch.setattr(ws, "_search_bing", lambda q, n: [])
    payload = ws.search("q")
    assert payload["backend"] == "duckduckgo"
    assert len(payload["results"]) == 1


# ------------------------------------------------------------------ 输出约束
def test_snippet_is_clipped():
    """单条摘要要截断，否则一条就能吃掉半个上下文。"""
    long_text = "x" * 5000
    clipped = ws._clip(long_text)
    assert len(clipped) <= ws.SNIPPET_LIMIT
    assert clipped.endswith("…")


def test_format_results_respects_total_limit():
    """多条结果加起来不能超过 TOTAL_LIMIT。"""
    payload = {
        "results": [
            {"title": f"标题{i}", "snippet": "y" * 400, "url": f"https://e/{i}"}
            for i in range(8)
        ]
    }
    text = ws.format_results(payload)
    assert len(text) <= ws.TOTAL_LIMIT + 200  # 允许最后一行"还有 N 条"的余量
    assert "未显示" in text, "超出预算时要说明还有几条没显示"


def test_max_results_is_capped(monkeypatch):
    """调用方要 100 条也只给 MAX_RESULTS 条（防滥用）。"""
    captured: dict[str, int] = {}

    def _fake_ddg(query, limit):
        captured["limit"] = limit
        return [{"title": "t", "snippet": "s", "url": "https://x"}]

    monkeypatch.setattr(ws, "tavily_key", lambda: "")
    monkeypatch.setattr(ws, "_search_duckduckgo", _fake_ddg)
    monkeypatch.setattr(ws, "_search_bing", lambda q, n: [])
    ws.search("q", limit=100)
    assert captured["limit"] == ws.MAX_RESULTS


# ------------------------------------------------------------------ 工具集成
def test_web_search_tool_registered_readonly():
    """工具要注册，且标为只读（可与其他只读工具并发）。"""
    from gis_cli.engine.tools import build_default_registry

    registry = build_default_registry()
    assert "web_search" in registry.names()
    assert registry.is_readonly("web_search")


def test_web_search_tool_reports_failure_without_crashing(monkeypatch):
    """工具层：搜索失败要返回 ok=False 的观察 + 明确提示，不能抛。"""
    from gis_cli.engine.tools import EngineContext, build_default_registry

    monkeypatch.setattr(
        ws, "search", lambda q, limit=5: {"query": q, "results": [], "backend": "x", "error": "没网"}
    )
    registry = build_default_registry()
    ctx = EngineContext(workspace=".")
    obs = registry.execute("web_search", {"query": "arcpy"}, ctx)
    assert obs.ok is False
    assert "没网" in obs.summary
    # 必须告诉模型"别反复重试"
    assert obs.hint and "不要" in obs.hint


def test_web_search_tool_returns_results(monkeypatch):
    """工具层：成功时把结果文本放进 data，供模型阅读。"""
    from gis_cli.engine.tools import EngineContext, build_default_registry

    monkeypatch.setattr(
        ws,
        "search",
        lambda q, limit=5: {
            "query": q,
            "backend": "duckduckgo",
            "error": "",
            "results": [{"title": "Buffer", "snippet": "creates polygons", "url": "https://x"}],
        },
    )
    registry = build_default_registry()
    obs = registry.execute("web_search", {"query": "buffer"}, EngineContext(workspace="."))
    assert obs.ok is True
    assert "Buffer" in obs.data["text"]
    assert obs.data["backend"] == "duckduckgo"


def test_error_message_lists_each_backend_attempt(monkeypatch):
    """全失败时要说清楚**每个**后端怎么失败的（用户才知道该配 Tavily）。"""
    import urllib.error

    def _boom(*a, **k):
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(ws, "tavily_key", lambda: "")
    monkeypatch.setattr(ws, "_fetch", _boom)
    payload = ws.search("q")
    assert "duckduckgo" in payload["error"]
    assert "bing" in payload["error"]


def test_backend_registry_uses_names_not_objects():
    """后端清单必须存**函数名**，否则 monkeypatch 换不掉（踩过）。"""
    assert all(isinstance(name, str) for name in ws._BACKEND_NAMES)
    for name in ws._BACKEND_NAMES:
        assert callable(getattr(ws, name, None)), name


# ---------------------------------------------------------- 相关性校验
def test_irrelevant_results_are_filtered_out(monkeypatch):
    """**最重要的保护**：返回一堆与查询无关的结果时，必须全部丢掉。

    实测教训：本机网络环境下 Bing 对自动化请求返回降级页 —— 查
    "arcpy Buffer tool parameters" 返回 NFL 官网/网易邮箱/百度知道。
    这比"搜索不可用"更糟：模型会把 NFL 当成 ArcPy 资料写进答案。
    """
    monkeypatch.setattr(ws, "tavily_key", lambda: "")
    monkeypatch.setattr(
        ws,
        "_search_duckduckgo",
        lambda q, n: [
            {"title": "NFL Official Site", "snippet": "football", "url": "https://nfl.com"},
            {"title": "网易邮箱", "snippet": "邮件", "url": "https://mail.163.com"},
        ],
    )
    monkeypatch.setattr(ws, "_search_bing", lambda q, n: [])
    payload = ws.search("arcpy Buffer tool parameters")
    assert payload["results"] == [], f"无关结果必须被过滤掉: {payload['results']}"
    assert "无关" in payload["error"], payload["error"]


def test_relevant_results_pass_through(monkeypatch):
    """相关的结果要保留（不能因为过滤太狠把好东西也扔了）。"""
    monkeypatch.setattr(ws, "tavily_key", lambda: "")
    monkeypatch.setattr(
        ws,
        "_search_duckduckgo",
        lambda q, n: [
            {"title": "Buffer (Analysis)—ArcGIS Pro", "snippet": "arcpy buffer creates polygons",
             "url": "https://pro.arcgis.com/buffer"},
        ],
    )
    monkeypatch.setattr(ws, "_search_bing", lambda q, n: [])
    payload = ws.search("arcpy Buffer tool parameters")
    assert len(payload["results"]) == 1
    assert payload["backend"] == "duckduckgo"


def test_relevance_check_is_lenient():
    """相关性判据要宽松：结果里出现**任意**一个查询词就算相关。

    搜索结果的措辞千差万别，判据太严会把正常结果也滤掉。
    """
    item = {"title": "ArcPy 缓冲区分析教程", "snippet": "使用 Buffer 工具", "url": "https://x"}
    assert ws._relevant("arcpy buffer", item)
    assert ws._relevant("buffer", item)
    # 完全不搭的才判无关
    assert not ws._relevant("quantum chromodynamics", item)

# -*- coding: utf-8 -*-
"""会话持久化测试：写入/读取、列表、上下文前言、导出。"""

from __future__ import annotations

import time

from gis_cli.ui.session_store import SessionStore, summarize_records


def test_append_and_load_roundtrip(tmp_path):
    store = SessionStore(tmp_path)
    store.append("s1", "user", "把点位做核密度")
    store.append("s1", "tool", "run_recipe 完成", tool="run_recipe")
    store.append("s1", "assistant", "已交付 4 张密度图", status="completed")

    records = store.load("s1")
    assert [r.role for r in records] == ["user", "tool", "assistant"]
    assert records[0].text == "把点位做核密度"
    assert records[1].meta["tool"] == "run_recipe"
    assert records[2].meta["status"] == "completed"
    assert records[0].ts  # 有时间戳


def test_long_text_is_clipped(tmp_path):
    store = SessionStore(tmp_path)
    store.append("s1", "tool", "x" * 9000)
    text = store.load("s1")[0].text
    assert len(text) == 4000 and text.endswith("…")


def test_missing_session_returns_empty(tmp_path):
    store = SessionStore(tmp_path)
    assert store.load("nope") == []
    assert store.context_prefix("nope") == ""


def test_list_sessions_and_latest(tmp_path):
    store = SessionStore(tmp_path)
    store.append("old", "user", "旧任务")
    time.sleep(0.01)
    store.append("new", "user", "新任务做影像粗匹配")

    items = store.list_sessions()
    assert [item.session_id for item in items][0] == "new"
    assert items[0].records == 1
    assert "新任务" in items[0].title
    assert store.latest_id() == "new"


def test_context_prefix_contains_recent_turns(tmp_path):
    store = SessionStore(tmp_path)
    store.append("s1", "user", "统计物种点位数")
    store.append("s1", "assistant", "已交付统计表 output/统计.csv", status="completed")

    prefix = store.context_prefix("s1")
    assert "接续会话 s1" in prefix
    assert "统计物种点位数" in prefix
    assert "统计.csv" in prefix
    assert prefix.rstrip().endswith("[上下文结束]")


def test_context_prefix_respects_budget(tmp_path):
    store = SessionStore(tmp_path)
    for i in range(30):
        store.append("s1", "user", f"第 {i} 条很长的历史消息" + "细节" * 60)
    prefix = store.context_prefix("s1", max_records=14, max_chars=400)
    assert len(prefix) < 1200  # 有预算上限，不会把整个历史塞进去


def test_export_writes_markdown(tmp_path):
    store = SessionStore(tmp_path)
    store.append("s1", "user", "问题")
    store.append("s1", "assistant", "回答")
    path = store.export("s1")
    assert path is not None and path.exists()
    text = path.read_text(encoding="utf-8")
    assert "# GIS Agent 会话 s1" in text and "**你**" not in text  # 标签在行内
    assert "问题" in text and "回答" in text


def test_session_id_is_sanitized(tmp_path):
    store = SessionStore(tmp_path)
    store.append("../../evil", "user", "x")
    files = list((tmp_path / ".gis_agent" / "sessions").glob("*.jsonl"))
    assert len(files) == 1
    assert files[0].name == "evil.jsonl"  # 路径穿越被清掉


def test_summarize_records_is_compact(tmp_path):
    store = SessionStore(tmp_path)
    store.append("s1", "user", "问题一二三")
    store.append("s1", "assistant", "答复")
    text = summarize_records(store.load("s1"))
    assert "你" in text and "Agent" in text
    assert len(text.splitlines()) == 2

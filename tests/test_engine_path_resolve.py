# -*- coding: utf-8 -*-
"""路径解析回归：修掉"工作区名被拼两遍"导致的读文档失败。

事故现场（用户日志）：
    read_document path=workspace\\input\\13第十三届…\\B上午试题.pdf
    ✗ 文档不存在: workspace\\workspace\\input\\13第十三届…\\B上午试题.pdf
模型连续三轮在同一个错误路径上打转 —— 旧实现只做 ``ctx.workspace / path``。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gis_cli.engine.tools import _missing_path_message, _resolve_path


class _Ctx:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace


@pytest.fixture()
def ctx(tmp_path: Path) -> _Ctx:
    ws = tmp_path / "workspace"
    (ws / "input" / "第14届真题数据").mkdir(parents=True)
    (ws / "input" / "第14届真题数据" / "README.md").write_text("hello", encoding="utf-8")
    return _Ctx(ws)


def test_plain_relative_path(ctx):
    hit, _ = _resolve_path(ctx, "input/第14届真题数据/README.md")
    assert hit is not None and hit.name == "README.md"


def test_path_with_workspace_prefix_is_not_doubled(ctx):
    """核心回归：路径里已带工作区名时，不能再拼一次。"""
    hit, tried = _resolve_path(ctx, "workspace/input/第14届真题数据/README.md")
    assert hit is not None and hit.read_text(encoding="utf-8") == "hello"
    assert all("workspace/workspace" not in str(p).replace("\\", "/") for p in tried)


def test_windows_backslash_path(ctx):
    hit, _ = _resolve_path(ctx, r"workspace\input\第14届真题数据\README.md")
    assert hit is not None


def test_absolute_path(ctx):
    target = ctx.workspace / "input" / "第14届真题数据" / "README.md"
    hit, _ = _resolve_path(ctx, str(target))
    assert hit == target


def test_quoted_path_is_stripped(ctx):
    hit, _ = _resolve_path(ctx, '"input/第14届真题数据/README.md"')
    assert hit is not None


def test_missing_path_reports_what_was_tried(ctx):
    hit, tried = _resolve_path(ctx, "input/不存在.md")
    assert hit is None
    message = _missing_path_message("input/不存在.md", tried)
    assert "路径不存在" in message
    assert "也已尝试" in message  # 让模型知道去哪找，别再瞎试
    assert "workspace/workspace" not in message.replace("\\", "/")


def test_empty_path_returns_nothing(ctx):
    hit, tried = _resolve_path(ctx, "   ")
    assert hit is None and tried == []

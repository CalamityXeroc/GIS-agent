# -*- coding: utf-8 -*-
"""Select 前端 guard：``web/src`` 源码里不允许再出现原生 ``<select>``。

原生下拉的弹出层取 select 自身的 used ``background-color``，``bg-transparent``
会回落白底，深色主题下漏白且无法用 CSS 覆盖（见 ``components/ui/select.tsx``
文件头注释）。所有下拉都必须走 ``components/ui/select.tsx`` 的 ``Select``，
这条测试挡住「又写回原生 select」的回退。

只扫 ``.ts/.tsx`` 源码并跳过注释行——``select.tsx`` 文件头注释里提到
``<select>`` 正是在解释为什么不这么写。
"""

from __future__ import annotations

import re
from pathlib import Path

WEB_SRC = Path(__file__).resolve().parents[2] / "web" / "src"
SOURCE_SUFFIXES = {".ts", ".tsx"}

# 标签名后可以是空白 / ``>`` / ``/`` 或行尾（多行 JSX 首行常是裸 ``<select``）；
# 大小写敏感——JSX 里 ``<Select>`` 是组件，不是原生标签。
NATIVE_SELECT = re.compile(r"<select(?=[\s>/]|$)")
COMMENT_PREFIXES = ("//", "*", "/*", "{/*")


def _source_files() -> list[Path]:
    assert WEB_SRC.is_dir(), f"缺少前端源码目录：{WEB_SRC}"
    return sorted(
        path
        for path in WEB_SRC.rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    )


def test_web_src_has_no_native_select() -> None:
    files = _source_files()
    # 防止路径写错时测试静默通过。
    assert any(path.name == "select.tsx" for path in files), "未扫到 components/ui/select.tsx"

    offenders: list[str] = []
    for path in files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith(COMMENT_PREFIXES):
                continue
            if NATIVE_SELECT.search(line):
                offenders.append(f"{path.relative_to(WEB_SRC.parent.parent)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "web/src 里不允许出现原生 <select>（弹层会漏白），请改用 "
        "components/ui/select.tsx 的 Select：\n" + "\n".join(offenders)
    )


def test_guard_pattern_hits_multiline_native_select() -> None:
    # 多行 JSX 的首行只有裸 ``<select``，行尾没有字符，旧正则 ``[\s>/]`` 会漏报。
    assert NATIVE_SELECT.search("<select")
    assert NATIVE_SELECT.search("<select value={v}>")


def test_guard_pattern_ignores_select_component() -> None:
    # 单行写的 ``<Select ... />`` 是 components/ui/select.tsx 的组件；``<selected>`` 不是标签。
    assert not NATIVE_SELECT.search("<Select value={x} onValueChange={f} />")
    assert not NATIVE_SELECT.search("<selected>")

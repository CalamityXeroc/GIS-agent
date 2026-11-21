# -*- coding: utf-8 -*-
"""``一键启动.bat`` 的静态体检。

这些 bug 只有真跑批处理才会暴露，而 CI 里不会跑 .bat —— 所以把关键
约束做成断言（实测踩过的坑，见每条 docstring）。

批处理的坑：``echo [OK] numpy>=1.26 installed`` 里的 ``>`` 是**重定向符**，
不是字面量。结果是在当前目录生成一个名为 ``1.26`` 的文件，屏幕上显示
``[OK] numpy installed``（版本号被吞）。``pip install numpy>=1.26`` 同理：
pip 只收到 ``numpy``，版本约束静默丢失。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

BAT = Path(__file__).resolve().parents[1] / "一键启动.bat"


def _text() -> str:
    if not BAT.exists():  # pragma: no cover - 发布包可能不带 bat
        pytest.skip("一键启动.bat 不存在")
    return BAT.read_bytes().decode("utf-8")


def test_bat_uses_crlf():
    """必须是 CRLF：LF 行尾会让 cmd 解析错乱（实测标签跳转失效）。"""
    raw = BAT.read_bytes()
    if b"\r\n" not in raw:  # pragma: no cover
        pytest.skip("bat 不存在或是 LF 检出版本")
    lone_lf = raw.replace(b"\r\n", b"").count(b"\n")
    assert lone_lf == 0, f"发现 {lone_lf} 个裸 LF，批处理要求全 CRLF"


def test_no_unquoted_redirection_in_echo():
    """echo 里不能出现裸 ``>`` / ``<``（会把包规格当重定向，生成垃圾文件）。"""
    for lineno, line in enumerate(_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped.lower().startswith("echo"):
            continue
        # 允许 ^> 转义写法
        bad = re.search(r"(?<!\^)[<>]", stripped)
        assert not bad, (
            f"第 {lineno} 行 echo 含裸重定向符（会生成垃圾文件）：{stripped!r}"
        )


def test_pip_install_quotes_version_spec():
    """pip install 的包规格必须加引号，否则 ``>=`` 里的 ``>`` 变成重定向。"""
    for lineno, line in enumerate(_text().splitlines(), 1):
        if "pip install" not in line or "%~2" not in line:
            continue
        assert '"%~2"' in line, (
            f"第 {lineno} 行 pip install 的 %~2 未加引号（版本约束会被吞）：{line.strip()!r}"
        )


def test_ensure_subroutine_does_not_echo_package_spec():
    """:ensure 子程序回显包规格（含 >=）同样会生成垃圾文件，须用导入名。"""
    text = _text()
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.lower().startswith("echo") and "%~2" in stripped:
            pytest.fail(
                f"第 {lineno} 行 echo 用了 %~2（包规格，含 >=）：{stripped!r}"
            )

# -*- coding: utf-8 -*-
"""控制台兼容回归（乱码事故防线）。

事故：用户控制台不支持 ANSI，但 rich 仍输出 ``\\x1b[38;5;246m``，屏幕上显示成
``?[38;5;246m``（用户说的"246m"就是色号 246）。第一次修复靠"能否开启 VT"判断，
在用户环境里判断为"能"，于是照旧上色 → 问题依旧。

**现在的策略：默认纯文本**。只有明确认得的高级终端（Windows Terminal / VS Code /
ConEmu / mintty 等）才自动上色；用户可用 ``--color`` / ``GIS_AGENT_FORCE_COLOR=1``
强制，或用 ``--no-color`` / ``NO_COLOR`` / ``GIS_AGENT_PLAIN`` 强制关闭。
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console as RichConsole

from gis_cli.ui import console_env
from gis_cli.ui import events as ev
from gis_cli.ui import render as rd
from gis_cli.ui import theme as th

_TERM_KEYS = (
    "WT_SESSION",
    "TERM_PROGRAM",
    "ConEmuANSI",
    "WEZTERM_EXECUTABLE",
    "ALACRITTY_WINDOW_ID",
    "KITTY_WINDOW_ID",
    "VSCODE_INJECTION",
    "TERM",
    "NO_COLOR",
    "GIS_AGENT_PLAIN",
    "GIS_AGENT_FORCE_COLOR",
    "GIS_AGENT_ASCII",
)


@pytest.fixture()
def clean_env(monkeypatch):
    for key in _TERM_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


class FakeTTY(io.StringIO):
    """冒充"真实终端"：isatty 为真。"""

    encoding = "utf-8"

    def isatty(self) -> bool:  # noqa: D102
        return True


@pytest.fixture()
def tty(monkeypatch):
    """冒充"真实终端"。

    同时打两个位置：``sys.stdout``（真实路径）与 ``console_env._stdout_is_tty``
    （自有的判定接缝）—— pytest 的输出捕获会在 fixture 之后重新绑定 stdout，
    只打 sys.stdout 会不稳定（实测只在 -s 下通过）。
    """
    stream = FakeTTY()
    monkeypatch.setattr("sys.stdout", stream)
    monkeypatch.setattr(console_env, "_stdout_is_tty", lambda: True)
    return stream


# --------------------------------------------------------------- 策略表


def test_plain_by_default_on_ordinary_windows_console(clean_env, tty):
    """普通 cmd / PowerShell：**不上色**（这是乱码事故的关键回归）。"""
    assert console_env.terminal_family() in ("windows-console", "unknown", "term:")
    assert console_env.ansi_ok() is False


def test_color_on_windows_terminal(clean_env, tty):
    clean_env.setenv("WT_SESSION", "1")
    assert console_env.terminal_family() == "windows-terminal"
    assert console_env.ansi_ok() is True


def test_force_color_overrides_everything(clean_env, tty):
    clean_env.setenv("GIS_AGENT_FORCE_COLOR", "1")
    assert console_env.ansi_ok() is True


def test_plain_wins_over_force(clean_env, tty):
    clean_env.setenv("GIS_AGENT_FORCE_COLOR", "1")
    clean_env.setenv("GIS_AGENT_PLAIN", "1")
    assert console_env.ansi_ok() is False


def test_no_color_disables(clean_env, tty):
    clean_env.setenv("WT_SESSION", "1")
    clean_env.setenv("NO_COLOR", "1")
    assert console_env.ansi_ok() is False


def test_not_tty_never_colors(clean_env, monkeypatch):
    monkeypatch.setattr(console_env, "_stdout_is_tty", lambda: False)
    monkeypatch.setenv("GIS_AGENT_FORCE_COLOR", "1")
    assert console_env.ansi_ok() is False  # 重定向到文件时连强制也不给颜色码


# --------------------------------------------------------------- 无转义输出


def _render_sample(console: RichConsole) -> str:
    console.file = io.StringIO()
    console.print(rd.banner(model="deepseek-flash", api_base="https://api.deepseek.com", workspace="workspace", recipes=47))
    console.print(rd.render_unit(ev.RenderUnit("step", "run_recipe", lines=["recipe_id=kernel_density"], meta={}), elapsed=1.2))
    console.print(rd.render_unit(ev.RenderUnit("result", "执行成功", ok=True)))
    console.print(rd.render_unit(ev.RenderUnit("verify", "验收通过（7/7 断言）", ok=True)))
    console.print(rd.result_card({"status": "completed", "artifacts": ["output/a.jpg"], "turn": 3}))
    console.print(rd.footer_text(ev.StatusTracker(model="deepseek-flash", status="running", turn=3), width=100))
    return console.file.getvalue()


def test_no_escape_sequences_on_plain_console(clean_env, tty):
    console = th.make_console("dark", force_terminal=True)
    out = _render_sample(console)
    assert "\x1b[" not in out, "纯文本模式下不得出现任何 ANSI 转义序列"
    assert "?[" not in out  # 事故现场字样
    assert "38;5;" not in out


def test_console_with_forced_color_has_escapes(clean_env, tty):
    console = th.make_console("dark", color=True, force_terminal=True)
    out = _render_sample(console)
    assert "\x1b[" in out  # 显式要求时才上色


def test_explicit_no_color_argument_wins(clean_env, tty):
    console = th.make_console("dark", no_color=True, color=True, force_terminal=True)
    assert "\x1b[" not in _render_sample(console)


# --------------------------------------------------------------- 符号降级


def test_glyphs_unicode_vs_ascii(clean_env, monkeypatch):
    monkeypatch.setattr(console_env, "unicode_ok", lambda: True)
    assert th.glyph("step") == "⏺" and th.glyph("ok") == "✓"
    monkeypatch.setattr(console_env, "unicode_ok", lambda: False)
    assert th.glyph("step") == "*" and th.glyph("ok") == "v" and th.glyph("rule") == "-"


def test_ascii_glyphs_render_without_unicode(clean_env, monkeypatch, tty):
    monkeypatch.setattr(console_env, "unicode_ok", lambda: False)
    console = th.make_console("dark", force_terminal=True)
    console.file = io.StringIO()
    console.print(rd.render_unit(ev.RenderUnit("step", "run_recipe", lines=["x"], meta={})))
    console.print(rd.render_unit(ev.RenderUnit("result", "出错", ok=False)))
    out = console.file.getvalue()
    assert "*" in out and "⏺" not in out and "└" not in out and "✗" not in out


# --------------------------------------------------------------- 编码一致


def test_configure_stdio_follows_console_encoding(clean_env, monkeypatch):
    """GBK 控制台上不得把 Python 流强切成 UTF-8（否则整屏乱码）。"""
    monkeypatch.setattr(console_env, "output_codepage", lambda: 936)
    monkeypatch.setattr(console_env, "console_encoding", lambda: "gbk")
    monkeypatch.setattr(console_env, "ensure_utf8_console", lambda **_: False)

    class FakeStream:
        def __init__(self):
            self.kwargs = {}

        def reconfigure(self, **kwargs):
            self.kwargs = kwargs

    out, err = FakeStream(), FakeStream()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.stderr", err)
    info = console_env.configure_stdio()
    assert info["encoding"] == "gbk"
    assert out.kwargs.get("encoding") == "gbk"
    assert out.kwargs.get("errors") == "replace"


def test_configure_stdio_uses_utf8_console(clean_env, monkeypatch):
    monkeypatch.setattr(console_env, "output_codepage", lambda: 65001)
    monkeypatch.setattr(console_env, "console_encoding", lambda: "utf-8")

    class FakeStream:
        def __init__(self):
            self.kwargs = {}

        def reconfigure(self, **kwargs):
            self.kwargs = kwargs

    out = FakeStream()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.stderr", FakeStream())
    assert console_env.configure_stdio()["encoding"] == "utf-8"
    assert out.kwargs.get("encoding") == "utf-8"


def test_describe_reports_everything(clean_env, tty, monkeypatch):
    monkeypatch.setattr(console_env, "output_codepage", lambda: 936)
    text = console_env.describe()
    assert "936" in text and "颜色" in text and "符号" in text

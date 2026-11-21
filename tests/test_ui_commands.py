# -*- coding: utf-8 -*-
"""斜杠命令注册表测试。

锁住两件以前漂移过的事：
1. 菜单、``/help``、实际实现三者**同一数据源**（以前是两张表，``/color``
   ``/session`` ``/resume`` 一度只在 if 链里、菜单选不到）；
2. **每条命令都真的接了处理器**（``/skills`` 曾经能选、点了永远输出"（无）"）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gis_cli.ui.commands import COMMANDS, CommandRegistry, CommandSpec, normalize_args


def test_all_commands_have_summary_and_unique_name():
    names = [spec.name for spec in COMMANDS]
    assert len(names) == len(set(names)), "命令名不能重复"
    assert all(spec.summary.strip() for spec in COMMANDS)
    assert all(not spec.name.startswith("/") for spec in COMMANDS), "name 不带斜杠"


def test_slash_name_has_single_leading_slash():
    registry = CommandRegistry((CommandSpec("model", "模型"),))
    assert registry.all()[0].slash_name == "/model"


def test_find_accepts_slash_and_bare_and_aliases():
    registry = CommandRegistry(
        (CommandSpec("exit", "退出", aliases=("quit", "q")), CommandSpec("model", "模型"))
    )
    for word in ("/exit", "exit", "/quit", "quit", "/q", "EXIT"):
        assert registry.find(word) is not None, word
    assert registry.find("/model").name == "model"
    assert registry.find("/nope") is None
    assert registry.find("") is None


def test_menu_items_and_help_come_from_same_source():
    registry = CommandRegistry(
        (
            CommandSpec("model", "查看或切换模型", usage="/model [名字]", args=(("m1", ""),)),
            CommandSpec("exit", "退出"),
        )
    )
    menu = dict(registry.menu_items())
    # 有参数候选时菜单带占位提示
    assert menu["/model"] == "查看或切换模型 <值>"
    assert menu["/exit"] == "退出"
    help_text = registry.help_text(keys=("  Ctrl+O        明细",))
    assert "查看或切换模型" in help_text and "/model [名字]" in help_text
    assert "Ctrl+O" in help_text


def test_help_skips_usage_line_when_absent():
    registry = CommandRegistry((CommandSpec("exit", "退出"),))
    assert "用法：" not in registry.help_text()


def test_arg_choices_static_and_lazy():
    static = CommandSpec("theme", "配色", args=(("dark", "深色"), ("light", "浅色")))
    calls: list[int] = []
    lazy = CommandSpec("model", "模型", args=lambda: (calls.append(1), (("m1", "配置"),))[1])

    assert normalize_args(static.args) == [("dark", "深色"), ("light", "浅色")]
    assert normalize_args(lazy.args) == [("m1", "配置")]
    assert calls == [1], "惰性候选应只在求值时调用一次"


def test_lazy_arg_choices_failure_is_not_fatal():
    """候选函数炸了（网关超时等）不能把菜单搞崩，退化成空候选。"""

    def _boom():
        raise RuntimeError("网关挂了")

    assert normalize_args(_boom) == []


def test_with_handlers_keeps_missing_ones():
    registry = CommandRegistry((CommandSpec("a", "甲"), CommandSpec("b", "乙")))
    merged = registry.with_handlers({"a": lambda _arg: True})
    assert merged.find("a").handler is not None
    assert merged.find("b").handler is None


def test_with_handlers_can_override_args():
    registry = CommandRegistry((CommandSpec("model", "模型"),))
    merged = registry.with_handlers({}, arg_providers={"model": lambda: (("x", "X"),)})
    assert normalize_args(merged.find("model").args) == [("x", "X")]


def test_prompt_slash_commands_derive_from_commands():
    """``ui.prompt.SLASH_COMMANDS`` 必须从 COMMANDS 派生（菜单与实现不分叉）。"""
    from gis_cli.ui import prompt as ui_prompt

    assert len(ui_prompt.SLASH_COMMANDS) == len([s for s in COMMANDS if s.visible])
    assert ui_prompt.SLASH_COMMANDS[0][0] == "/help"


def test_interactive_registers_handler_for_every_command():
    """每条命令都要真的接到处理器（锁住"/skills 能选但没用"这类问题）。"""
    from gis_cli.agent import interactive as interactive_mod

    chat = interactive_mod.InteractiveChat(workspace=Path("."), theme_mode="dark", no_color=True)
    registry = chat._command_registry()
    missing = [spec.name for spec in registry.all() if spec.handler is None]
    assert missing == [], f"这些命令没接处理器：{missing}"


def test_every_command_is_handled_without_unknown_warning(tmp_path):
    """逐条执行：不应落到"未知命令"分支。"""
    import io

    from rich.console import Console

    from gis_cli.agent import interactive as interactive_mod

    chat = interactive_mod.InteractiveChat(workspace=tmp_path, theme_mode="dark", no_color=True)
    buffer = io.StringIO()
    chat.console = Console(file=buffer, width=120, no_color=True)
    exit_codes: dict[str, bool] = {}
    for spec in chat._command_registry().all():
        buffer.truncate(0)
        buffer.seek(0)
        keep = chat._handle_command("/" + spec.name)
        exit_codes[spec.name] = keep
        assert "未知命令" not in buffer.getvalue(), f"/{spec.name} 未被处理"
    assert exit_codes["exit"] is False, "/exit 应返回 False（退出）"
    assert all(v for k, v in exit_codes.items() if k != "exit")


@pytest.mark.parametrize("word", ["/quit", "/q"])
def test_aliases_exit(word, tmp_path):
    from gis_cli.agent import interactive as interactive_mod

    chat = interactive_mod.InteractiveChat(workspace=tmp_path, theme_mode="dark", no_color=True)
    assert chat._handle_command(word) is False

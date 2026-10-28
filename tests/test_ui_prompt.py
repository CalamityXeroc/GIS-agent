# -*- coding: utf-8 -*-
"""输入框测试（prompt_toolkit 可用时）。无依赖则整体跳过。

用 prompt_toolkit 的无头设施（create_pipe_input + DummyOutput）验证真实按键行为：
Enter 提交、Alt+Enter/Ctrl+J 换行、Ctrl+C 清空或抛出、补全内容。
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from gis_cli.ui import prompt as ui_prompt

pt = pytest.importorskip("prompt_toolkit", reason="prompt_toolkit 未安装（用 ArcGIS 解释器可跑）")


# --------------------------------------------------------------- 纯逻辑


def test_slash_command_completer_matches_prefix():
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document

    completer = ui_prompt.SlashCommandCompleter()
    buffer = Buffer()
    document = Document("/mo", cursor_position=3)
    names = [c.text for c in completer.get_completions(document, None)]  # type: ignore[arg-type]
    assert "/model" in names
    assert all(name.startswith("/mo") for name in names)

    # 已输入空格后不再提示命令
    document = Document("/model gpt", cursor_position=10)
    assert list(completer.get_completions(document, None)) == []  # type: ignore[arg-type]
    _ = buffer


def test_slash_command_completer_lists_all_on_bare_slash():
    """输入单个 ``/`` 就应该列出全部命令（这是"输入斜杠弹菜单"的关键）。"""
    from prompt_toolkit.document import Document

    completer = ui_prompt.SlashCommandCompleter()
    names = [c.text for c in completer.get_completions(Document("/"), None)]  # type: ignore[arg-type]
    assert names == [name for name, _ in ui_prompt.SLASH_COMMANDS]
    # 每条都要带中文说明，否则菜单里只剩命令名、看不出干什么
    metas = [c.display_meta_text for c in completer.get_completions(Document("/"), None)]  # type: ignore[arg-type]
    assert all(meta.strip() for meta in metas)


def test_slash_command_completer_filters_and_stops_at_space():
    from prompt_toolkit.document import Document

    completer = ui_prompt.SlashCommandCompleter()
    names = [c.text for c in completer.get_completions(Document("/mo"), None)]  # type: ignore[arg-type]
    assert names == ["/model"]

    # 进入参数区后不再列命令（否则会盖住参数补全）
    assert list(completer.get_completions(Document("/model gp"), None)) == []  # type: ignore[arg-type]


def test_slash_command_completer_is_case_insensitive():
    from prompt_toolkit.document import Document

    completer = ui_prompt.SlashCommandCompleter()
    names = [c.text for c in completer.get_completions(Document("/EX"), None)]  # type: ignore[arg-type]
    assert names == ["/export", "/exit"]


def test_slash_command_completer_ignores_plain_task_text():
    from prompt_toolkit.document import Document

    completer = ui_prompt.SlashCommandCompleter()
    assert list(completer.get_completions(Document("统计点位数"), None)) == []  # type: ignore[arg-type]


def test_at_file_completer_uses_provider(tmp_path):
    from prompt_toolkit.document import Document

    (tmp_path / "animal_points.shp").write_text("x", encoding="utf-8")
    (tmp_path / "dem.tif").write_text("x", encoding="utf-8")
    provider = ui_prompt.workspace_file_provider(tmp_path)
    completer = ui_prompt.AtFileCompleter(provider)
    completer.refresh()

    doc = Document("分析 @ani", cursor_position=len("分析 @ani"))
    hits = [c.text for c in completer.get_completions(doc, None)]  # type: ignore[arg-type]
    assert any("animal_points.shp" in h for h in hits)


def test_workspace_file_provider_returns_relative_paths(tmp_path):
    sub = tmp_path / "input"
    sub.mkdir()
    (sub / "a.shp").write_text("x", encoding="utf-8")
    (sub / "b.tif").write_text("x", encoding="utf-8")
    (sub / "notes.md").write_text("x", encoding="utf-8")  # 不在白名单
    found = ui_prompt.workspace_file_provider(tmp_path)()
    assert "input/a.shp" in found and "input/b.tif" in found
    assert all(not item.endswith(".md") for item in found)


def test_mixed_completer_prefers_first_non_empty():
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.document import Document

    class Empty(Completer):
        def get_completions(self, document, complete_event):
            return iter(())

    class One(Completer):
        def get_completions(self, document, complete_event):
            yield Completion("hit", start_position=0)

    mixed = ui_prompt.MixedCompleter([Empty(), One()])
    assert [c.text for c in mixed.get_completions(Document("x"), None)] == ["hit"]  # type: ignore[arg-type]


def test_every_menu_command_is_handled_by_interactive(monkeypatch, tmp_path):
    """菜单里能选到的每个命令，`_handle_command` 都必须真的认识。

    以前菜单（ui/prompt.py）与实现（agent/interactive.py 的 if 链）是两张表，
    /color /session /resume 就一直不在菜单里——这条测试防漂移。
    """
    from rich.console import Console

    from gis_cli.agent import interactive as agent_interactive

    chat = agent_interactive.InteractiveChat(workspace=tmp_path)
    printed: list[str] = []
    chat.console = Console(file=io.StringIO(), width=200)
    monkeypatch.setattr(chat, "_print_inventory", lambda _k: printed.append("inv"))
    monkeypatch.setattr(chat, "_print_status", lambda: printed.append("status"))
    monkeypatch.setattr(chat, "_print_session", lambda: printed.append("session"))
    monkeypatch.setattr(chat, "_resume", lambda _a: printed.append("resume"))

    for name, _desc in ui_prompt.SLASH_COMMANDS:
        printed.clear()
        keep_going = chat._handle_command(name)
        if name in ("/exit", "/quit", "/q"):
            assert keep_going is False, f"{name} 应该退出"
            continue
        assert keep_going is True, f"{name} 不应导致退出"
        assert "未知命令" not in _console_text(chat), f"{name} 未被 _handle_command 处理"


def _console_text(chat) -> str:
    stream = chat.console.file
    try:
        return stream.getvalue()
    except Exception:  # pragma: no cover - 非内存流
        return ""


def test_slash_menu_never_falls_through_to_file_completion(tmp_path):
    """回归：输入 ``/`` 必须只给命令，不能掉到路径补全。

    实测过的 bug：用 DynamicCompleter + get_app().current_buffer 判条件，条件一不成立
    就掉到 PathCompleter，而 ``/`` 正是文件系统根目录 → 输入 ``/`` 弹出 304 个文件候选。
    """
    from prompt_toolkit.document import Document

    commands = (("/model", "查看/切换模型"), ("/exit", "退出"))
    from prompt_toolkit.completion import Completer, Completion

    class NoisyFiles(Completer):
        """假装是路径补全：对任何输入都给一大堆东西。"""

        def get_completions(self, document, complete_event):
            yield Completion("!!!noise.txt", start_position=0)

    completer = ui_prompt.SlashMenuCompleter(commands, fallback=NoisyFiles())

    for text in ("/", "/mo", "/EX"):
        got = [c.text for c in completer.get_completions(Document(text), None)]  # type: ignore[arg-type]
        assert got, f"{text} 应有命令候选"
        assert all(name.startswith("/") for name in got), got
        assert "!!!noise.txt" not in got, f"{text} 不应回落到文件补全"

    # 非斜杠输入才走 fallback
    got = [c.text for c in completer.get_completions(Document("statistics"), None)]  # type: ignore[arg-type]
    assert got == ["!!!noise.txt"]


def test_slash_menu_stops_at_space_and_newline(tmp_path):
    from prompt_toolkit.document import Document

    completer = ui_prompt.SlashMenuCompleter((("/model", "模型"),))
    # 参数区 / 多行：不再抢着当命令
    assert list(completer.get_completions(Document("/model gp"), None)) == []  # type: ignore[arg-type]
    assert list(completer.get_completions(Document("/model\nx"), None)) == []  # type: ignore[arg-type]
    # 行首空白仍算命令
    names = [c.text for c in completer.get_completions(Document("  /mo"), None)]  # type: ignore[arg-type]
    assert names == ["/model"]


def test_complete_while_typing_not_disabled_by_history_search(tmp_path, pipe_input):
    """回归：边打边弹菜单不能被 enable_history_search 静默关掉。

    prompt_toolkit 内部把两者设成互斥（见 PromptSession._create_default_buffer）：
        Condition(complete_while_typing and not enable_history_search ...)
    一旦开了 enable_history_search，输入 `/` 就永远不弹命令菜单，
    而单测直接调 completer.get_completions() 是看不出来的——必须在真 session 上验。
    """
    from prompt_toolkit.output import DummyOutput

    reader = ui_prompt.PromptInput(tmp_path, mode="dark")
    session = reader.session(input=pipe_input, output=DummyOutput())
    buffer = session.default_buffer
    assert buffer.complete_while_typing(), (
        "complete_while_typing 被关掉了——输入 / 不会弹菜单（多半是开了 enable_history_search）"
    )
    assert session.enable_history_search is False


def test_format_help_lists_commands_and_keys():
    text = ui_prompt.format_help()
    assert "/model" in text and "/exit" in text
    assert "Shift+Enter" in text and "Ctrl+O" in text
    # 菜单交互方式要写在帮助里（用户靠这个知道能用 ↑↓）
    assert "↑↓" in text and "Enter 应用" in text
    assert "Ctrl+R" in text and "Ctrl+L" in text


def test_slash_commands_table_is_complete_and_described():
    """菜单命令表不能漏项、不能空说明。"""
    names = [name for name, _ in ui_prompt.SLASH_COMMANDS]
    assert len(names) == len(set(names)), "命令名不能重复"
    assert all(name.startswith("/") for name in names)
    assert all(desc.strip() for _, desc in ui_prompt.SLASH_COMMANDS)
    # 这些是 interactive._handle_command 实际支持的，少一个菜单就会漏
    for expected in ("/color", "/session", "/resume", "/export", "/trace"):
        assert expected in names, f"菜单缺少 {expected}"


def test_degrade_notice_reports_missing_prompt_toolkit(monkeypatch):
    monkeypatch.setattr(ui_prompt, "_PT_AVAILABLE", False)
    monkeypatch.setattr(ui_prompt, "_PT_IMPORT_ERROR", "No module named 'prompt_toolkit'")
    monkeypatch.delenv("GIS_AGENT_ALLOW_PLAIN_INPUT", raising=False)
    notice = ui_prompt.degrade_notice()
    assert "prompt_toolkit" in notice and "pip install" in notice

    # 可用时不该有噪音
    monkeypatch.setattr(ui_prompt, "_PT_AVAILABLE", True)
    assert ui_prompt.degrade_notice() == ""


def test_degrade_notice_can_be_silenced(monkeypatch):
    monkeypatch.setattr(ui_prompt, "_PT_AVAILABLE", False)
    monkeypatch.setenv("GIS_AGENT_ALLOW_PLAIN_INPUT", "1")
    assert ui_prompt.degrade_notice() == ""


def test_prompt_style_no_color_is_empty():
    style = ui_prompt.prompt_style("dark", no_color=True)
    assert style is not None  # 有对象但不含颜色规则


# --------------------------------------------------------------- 无头按键


@pytest.fixture()
def pipe_input():
    from prompt_toolkit.input import create_pipe_input

    with create_pipe_input() as pipe:
        yield pipe


#: prompt_toolkit 内部把 Enter 记作 c-m，写测试时用友好名
_KEY_ALIASES = {"enter": "c-m"}


def _find_binding(reader, *keys: str):
    """从 PromptInput 的 KeyBindings 里取指定键序列的处理函数。

    ``keys`` 要写完整序列（如 ``("escape",)`` 与 ``("escape", "enter")`` 是两条
    不同的绑定，不能靠"包含"混淆）；驱动完整 Application 需要事件循环，
    所以这里只取回调、手工造事件。
    """
    wanted = tuple(_KEY_ALIASES.get(k, k) for k in keys)
    bindings = reader._key_bindings()
    for binding in bindings.bindings:
        actual = tuple(str(getattr(k, "value", k)) for k in binding.keys)
        if actual == wanted:
            return binding.handler
    raise AssertionError(f"找不到按键绑定 {wanted}")


def _session(tmp_path: Path, pipe):
    from prompt_toolkit.output import DummyOutput

    reader = ui_prompt.PromptInput(tmp_path, mode="dark")
    return reader, reader.session(input=pipe, output=DummyOutput())


def test_enter_submits_single_line(tmp_path, pipe_input):
    reader, session = _session(tmp_path, pipe_input)
    pipe_input.send_text("统计点位数\r")
    assert session.prompt("› ") == "统计点位数"


def test_ctrl_j_inserts_newline_before_submit(tmp_path, pipe_input):
    reader, session = _session(tmp_path, pipe_input)
    pipe_input.send_text("第一行\x0a第二行\r")  # Ctrl+J 换行，再 Enter 提交
    assert session.prompt("› ") == "第一行\n第二行"


def test_alt_enter_inserts_newline(tmp_path, pipe_input):
    reader, session = _session(tmp_path, pipe_input)
    pipe_input.send_text("A\x1b\rB\r")  # Esc+Enter 换行
    assert session.prompt("› ") == "A\nB"


def test_ctrl_c_with_text_clears_instead_of_raising(tmp_path, pipe_input):
    reader, session = _session(tmp_path, pipe_input)
    # 输入文字 → Ctrl+C（清空）→ 再输入 → Enter
    pipe_input.send_text("写错的内容\x03对的内容\r")
    assert session.prompt("› ") == "对的内容"


def test_ctrl_c_on_empty_raises_keyboard_interrupt(tmp_path, pipe_input):
    reader, session = _session(tmp_path, pipe_input)
    pipe_input.send_text("\x03")
    with pytest.raises(KeyboardInterrupt):
        session.prompt("› ")


def test_ctrl_d_raises_eof(tmp_path, pipe_input):
    reader, session = _session(tmp_path, pipe_input)
    pipe_input.send_text("\x04")
    with pytest.raises(EOFError):
        session.prompt("› ")


def test_ctrl_o_triggers_toggle_callback(tmp_path, pipe_input):
    from prompt_toolkit.output import DummyOutput

    calls: list[str] = []
    reader = ui_prompt.PromptInput(tmp_path, mode="dark", on_toggle_output=lambda: calls.append("t"))
    session = reader.session(input=pipe_input, output=DummyOutput())
    pipe_input.send_text("abc\x0f\r")  # Ctrl+O 后 Enter
    assert session.prompt("› ") == "abc"
    assert calls == ["t"]


def test_enter_applies_slash_completion_before_submitting():
    """菜单开着时 Enter 只应用候选、不提交（pi/Claude 手感）。

    直接测按键回调本身：驱动一个真的 Application 需要事件循环，而这里要验的是
    "看 complete_state 决定分支"这一句逻辑。
    """
    applied: list[str] = []

    class _Buffer:
        complete_state = None
        text = "/mo"

        def apply_completion(self, completion):
            applied.append(completion.text)

        def insert_text(self, _text):
            raise AssertionError("Enter 不应直接插换行")

        def validate_and_handle(self):  # 不应被调用
            raise AssertionError("菜单开着时不应直接提交")

    class _Event:
        current_buffer = _Buffer()

    class _Completion:
        text = "/model"

    reader = ui_prompt.PromptInput(Path("."))
    handler = _find_binding(reader, "enter")
    _Event.current_buffer.complete_state = type("S", (), {"current_completion": _Completion()})()
    handler(_Event())
    assert applied == ["/model"]

    # 菜单关着时正常提交
    _Event.current_buffer.complete_state = None

    class _Buffer2(_Buffer):
        def validate_and_handle(self):
            applied.append("submit")

    class _Event2:
        current_buffer = _Buffer2()

    handler(_Event2())
    assert applied == ["/model", "submit"]


def test_escape_cancels_menu_before_interrupting():
    cancelled: list[str] = []
    interrupted: list[str] = []

    class _Buffer:
        complete_state = object()  # 菜单开着

        def cancel_completion(self):
            cancelled.append("c")

        def insert_text(self, _text):
            raise AssertionError("Esc 不应插换行")

    class _Event:
        current_buffer = _Buffer()

        class app:  # noqa: N801 - 模拟 pt 事件对象
            @staticmethod
            def invalidate():
                pass

    reader = ui_prompt.PromptInput(
        Path("."), on_interrupt=lambda: interrupted.append("i") or True
    )
    handler = _find_binding(reader, "escape")
    handler(_Event())
    assert cancelled == ["c"] and interrupted == []

    # 菜单关着时才轮到中断回调
    _Event.current_buffer.complete_state = None
    handler(_Event())
    assert interrupted == ["i"]


def test_history_file_is_written_under_workspace(tmp_path, pipe_input):
    reader, session = _session(tmp_path, pipe_input)
    pipe_input.send_text("保存历史\r")
    session.prompt("› ")
    # FileHistory 落盘位置（异步写入，允许文件尚未出现但目录应已建）
    assert (tmp_path / ".gis_agent").exists()


# --------------------------------------------------------------- 布局（自建输入框）


def test_chat_prompt_accepts_live_status_callable():
    """状态行要传 callable（每帧重算），否则运行中轮次/token 不会跳。"""
    from gis_cli.ui import tui as ui_tui

    calls: list[int] = []

    def provider() -> str:
        calls.append(1)
        return f"model · 第 {len(calls)} 轮 · ↑0 ↓0"

    prompt = ui_tui.ChatPrompt(status_provider=provider)
    assert prompt._status_provider() != prompt._status_provider(), "每次调用应重新求值"


def test_input_height_grows_with_lines(monkeypatch, tmp_path, pipe_input):
    """回归：多行输入时框跟着变长，而不是另开一个框。

    实测 bug：输入区写死 Dimension(min=1) + dont_extend_height=True，
    第二行只能溢出到框外，看起来像“每换行多一个框”。
    """
    from prompt_toolkit.document import Document
    from prompt_toolkit.output import DummyOutput

    from gis_cli.ui import tui as ui_tui

    prompt = ui_tui.ChatPrompt(input=pipe_input, output=DummyOutput())
    prompt.prepare()
    buffer = prompt.buffer

    assert prompt._content_lines() == 1
    # 用 set_document 而不是 insert_text：后者会拉起补全协程，
    # 需要事件循环（实测报 "no running event loop"）。
    text = "第一行\n第二行\n第三行"
    buffer.set_document(Document(text, cursor_position=len(text)), bypass_readonly=True)
    assert prompt._content_lines() == 3
    assert prompt._input_height().min >= 3


def test_content_lines_counts_cjk_width(monkeypatch, tmp_path, pipe_input):
    """折行要按显示宽度算（中文占两格），否则长中文会跑到框外。"""
    from prompt_toolkit.document import Document
    from prompt_toolkit.output import DummyOutput

    from gis_cli.ui import tui as ui_tui

    prompt = ui_tui.ChatPrompt(input=pipe_input, output=DummyOutput())
    prompt.prepare()
    monkeypatch.setattr(prompt, "_available_width", lambda: 23)  # 减 PREFIX_WIDTH 后 = 20
    text = "中文" * 10  # 20 个汉字 = 40 格 → 2 行
    prompt.buffer.set_document(Document(text, cursor_position=len(text)), bypass_readonly=True)
    assert prompt._content_lines() == 2


def test_borders_are_pure_lines():
    """上下边框是纯线：不含“输入”、不含快捷键提示。"""
    from gis_cli.ui import tui as ui_tui

    prompt = ui_tui.ChatPrompt(status_provider=lambda: "m · ↑0 ↓0")
    for side in ("top", "bottom"):
        window = prompt._border_line_window(side=side)
        text = window.content.text()
        joined = "".join(fragment for _, fragment in text)
        assert "输入" not in joined
        for word in ("Enter", "Ctrl+", "Esc"):
            assert word not in joined
        assert joined.startswith("╭" if side == "top" else "╰")


def test_chat_prompt_has_no_hint_provider():
    """快捷键提示永不在界面上显示：构造参数里不该再有 hint_provider。"""
    import inspect

    from gis_cli.ui import tui as ui_tui

    params = inspect.signature(ui_tui.ChatPrompt.__init__).parameters
    assert "hint_provider" not in params
    assert "top_title" not in params


# --------------------------------------------------------------- 降级路径


def test_fallback_read_when_prompt_toolkit_missing(monkeypatch):
    monkeypatch.setattr(ui_prompt, "_PT_AVAILABLE", False)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "降级输入")
    assert ui_prompt.PromptInput(Path(".")).read_line("› ") == "降级输入"

    def raise_eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    assert ui_prompt.PromptInput(Path(".")).read_line("› ") is None


def test_available_reports_import_state():
    assert ui_prompt.available() is True  # 本文件已 importorskip
    assert isinstance(ui_prompt.import_error(), str)


# ------------------------------------------------- rich Live 的 stdout 代理（DAIR-37）


def test_interactive_terminal_survives_live_status_proxy(install_fake_tty):
    """``Console.status()`` 内部也得认得出终端：rich 期间 stdout 是 FileProxy。

    真 conhost 上 ``ChatPlanGate`` 靠这条判定决定要不要停下来等人；而 ``FileProxy``
    继承 ``io.TextIOBase``（``isatty()`` 恒为 False），不解代理就会把真终端判成
    非交互 —— 计划卡从不暂停（DAIR-37）。
    """
    from rich.console import Console

    fake_out = install_fake_tty()
    assert fake_out.isatty() is True
    assert ui_prompt.interactive_terminal() is True  # 前提：外层确实是终端

    console = Console(file=io.StringIO(), width=100, no_color=True, force_terminal=True)
    with console.status("启动引擎…"):
        assert type(sys.stdout).__name__ == "FileProxy", "前提：Live 期间 stdout 真的被代理了"
        assert ui_prompt.interactive_terminal() is True, "Live 内部不得把终端判成非交互"

    assert ui_prompt.interactive_terminal() is True  # 退出 with 后恢复
    assert fake_out.isatty() is True

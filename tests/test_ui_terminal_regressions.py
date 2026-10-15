# -*- coding: utf-8 -*-
"""界面回归：流式显示、输入框生命周期、排队语义。

为什么要有这一组测试：这三类问题都**只在真实终端里可见**，普通单测（用
StringIO 当输出）全都会漏 —— 实测就是这么漏掉的：

1. **流式乱码**：rich ``Live`` 会写光标控制序列（``ESC[2K`` 清行、``ESC[1A``
   上移）。普通 Windows 控制台不解析 ANSI，用户就看到满屏 ``?[2K?[1A?[2K``
   和重复的 ``◆ 思考 · 0s``。而且 ``Live`` 还会**捕获进程的 stderr 重定向**，
   在没有真实 fd 的环境里甚至直接静默退出。
2. **多个输入框**：``Application(full_screen=False)`` 若未设 ``erase_when_done``，
   提交后旧框留在屏幕上，主循环再画一个 → 出现两个框。
3. **回车被补全吃掉**：兜底 ``PathCompleter`` 把任意文本当文件路径匹配，
   一旦匹配到，第一次回车只是"接受候选"而不提交。
"""

from __future__ import annotations

import io

import pytest

from gis_cli.ui import events as ev
from gis_cli.ui import render as rd


class _Tty(io.StringIO):
    """假装是终端（is_terminal=True），但没有真实 fd。"""

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        raise OSError("no fileno")


def _vt_output(rows: int = 30, cols: int = 100):
    """造一个 Vt100 输出（不需要真实控制台）——用来构造 prompt_toolkit 应用。

    Windows 上直接给 output=None 会去找真控制台，抛 NoConsoleScreenBufferError。
    """
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.output.vt100 import Vt100_Output

    sink = _Tty()
    output = Vt100_Output(sink, lambda: Size(rows=rows, columns=cols), term="xterm-256color")
    return output, sink


def _completer_of(reader: Any) -> Any:
    """拿到 PromptInput 实际用的补全器（兼容两种构造路径）。"""
    prompt = reader._chat_prompt(busy=False, status="")
    return prompt.completer


def _find_enter_handler(prompt: Any) -> Any:
    """从 ChatPrompt 的绑定里找出 enter 的处理函数。"""
    from prompt_toolkit.keys import Keys

    bindings = prompt._key_bindings(prompt.buffer)
    for binding in bindings.bindings:
        keys = list(binding.keys)
        if keys == [Keys.ControlM] or keys == ["enter"] or keys == [Keys.Enter]:
            return binding.handler
        if any(k in (Keys.ControlM, Keys.Enter) or k == "enter" for k in keys):
            return binding.handler
    return None


def _no_ansi_console(width: int = 80):
    """构造"是 tty 但项目判定不上色"的 Console —— 普通 Windows 控制台的等价物。

    这正是关键组合：``Console.is_terminal`` 为真（rich 因此会发控制序列），
    而 ``color_system=None``（我们不上色）。仅靠 no_color 拦不住控制序列。
    """
    from rich.console import Console

    buffer = _Tty()
    console = Console(
        file=buffer,
        force_terminal=True,
        color_system=None,
        width=width,
        highlight=False,
    )
    return console, buffer


# --------------------------------------------------------- 1) 流式不乱码


def test_streaming_emits_no_cursor_control_sequences_without_ansi():
    """无 ANSI 终端下，流式显示不能输出任何光标控制序列。

    这是用户报的 ``?[2K?[1A?[2K`` 满屏乱码的直接复现。
    """
    console, buffer = _no_ansi_console()
    view = rd.TranscriptView(console, ev.StatusTracker())

    for piece in ("先看数据", "目录，", "再统计分组。"):
        view.live_delta("reasoning", piece)
    view.end_live()

    out = buffer.getvalue()
    esc = "\x1b"
    assert esc not in out, f"输出里出现了 ANSI 转义：{out[:200]!r}"
    assert esc + "[2K" not in out, "出现了清行序列（终端不解析时会显示成 ?[2K）"
    assert esc + "[1A" not in out, "出现了光标上移序列"
    # 内容本身仍要完整可见（降级不等于丢信息）
    assert "先看数据" in out and "再统计分组。" in out


def test_streaming_does_not_repeat_the_same_header_every_frame():
    """不能每帧打一遍 ``◆ 思考 · 0s``（用户看到的就是这个刷屏）。

    降级路径下只在块结束时输出一次标题。
    """
    console, buffer = _no_ansi_console()
    view = rd.TranscriptView(console, ev.StatusTracker())
    for index in range(20):
        view.live_delta("reasoning", f"片段{index}")
    view.end_live()
    out = buffer.getvalue()
    assert out.count("思考") <= 1, f"思考标题重复了 {out.count('思考')} 次"


def test_streaming_works_in_non_interactive_mode_without_ansi():
    """非交互（loop 模式）下同样不能出现控制序列。

    这里用非 tty 的 Console（重定向到文件）——历史上这类场景会走到
    Live 的"禁用"分支，但要确保真的没有序列漏出来。
    """
    buffer = io.StringIO()
    from rich.console import Console

    console = Console(file=buffer, no_color=True, color_system=None, width=80)
    view = rd.TranscriptView(console, ev.StatusTracker())
    view.live_delta("content", "结论：")
    view.live_delta("content", "分区正确。")
    view.end_live()
    out = buffer.getvalue()
    assert "\x1b" not in out, f"非交互输出里出现了 ANSI：{out[:200]!r}"
    assert "分区正确。" in out


# ------------------------------------------------ 2) 一个框 / 一次回车


def test_chat_prompt_erases_frame_when_done():
    """提交一行后 Application 要擦掉自己画的框（否则屏幕会叠出多个框）。

    用户报"启动后有两个聊天框"就是这个：旧框没擦 + 主循环又画一个。
    """
    from gis_cli.ui import tui

    prompt = tui.ChatPrompt(input=None, output=_vt_output()[0])
    app = prompt.prepare()
    assert getattr(app, "erase_when_done", False) is True, (
        "Application 必须设 erase_when_done=True，否则提交后旧输入框会留在屏幕上"
    )


def test_enter_submits_without_completion_navigation():
    """没主动导航菜单时，一次回车就应提交（不能被补全吃掉）。"""
    from prompt_toolkit.buffer import Buffer

    plain = Buffer(name="t")
    fired: list[str] = []
    plain.accept_handler = lambda buf: (fired.append(buf.text), True)[1]
    plain.text = "统计各社区老年人口"
    assert plain.complete_state is None
    plain.validate_and_handle()
    assert fired == ["统计各社区老年人口"], "一次回车必须提交"


def test_enter_binding_ignores_unnavigated_menu():
    """菜单弹出（自动高亮首项）但用户没按过 ↑↓ 时，回车必须直接提交。

    旧写法用 ``current_completion is not None`` 判断，而菜单一弹出就有选中项，
    于是普通句子也会被吃掉一次回车（用户实测要按两次）。
    """
    from gis_cli.ui import tui

    prompt = tui.ChatPrompt(input=None, output=_vt_output()[0])
    prompt.prepare()

    submitted: list[str] = []
    applied: list[str] = []

    class _State:
        # 菜单开着且自动高亮了首项，但用户没导航过
        complete_index = None
        current_completion = type("C", (), {"text": "候选"})()
        completions = [current_completion]

    class _Buf:
        complete_state = _State()
        text = "统计各社区老年人口"

        def apply_completion(self, completion):
            applied.append(completion.text)

        def validate_and_handle(self):
            submitted.append(self.text)

    class _App:
        @staticmethod
        def exit(**_kwargs):
            return None

    class _Event:
        current_buffer = _Buf()
        app = _App()

    handler = _find_enter_handler(prompt)
    assert handler is not None, "找不到 enter 的绑定处理函数"
    handler(_Event())
    assert submitted == ["统计各社区老年人口"], "应直接提交，而不是应用候选"
    assert not applied, f"不该应用候选，实际应用了 {applied}"


def test_enter_binding_applies_after_user_navigation():
    """用户按过 ↑↓ 选了候选时，回车应该先应用它（保留菜单手感）。"""
    from gis_cli.ui import tui

    prompt = tui.ChatPrompt(input=None, output=_vt_output()[0])
    prompt.prepare()

    submitted: list[str] = []
    applied: list[str] = []

    class _State:
        complete_index = 0  # 用户导航过
        current_completion = type("C", (), {"text": "/help"})()
        completions = [current_completion]

    class _Buf:
        complete_state = _State()
        text = "/he"

        def apply_completion(self, completion):
            applied.append(completion.text)

        def validate_and_handle(self):
            submitted.append(self.text)

    class _App:
        @staticmethod
        def exit(**_kwargs):
            return None

    class _Event:
        current_buffer = _Buf()
        app = _App()

    handler = _find_enter_handler(prompt)
    handler(_Event())
    assert applied == ["/help"], "导航过菜单时应先应用候选"
    assert not submitted, "应用候选时不应同时提交"


def test_path_completer_is_not_in_fallback_chain():
    """兜底补全链里不能有 PathCompleter。

    它把任意文本当文件路径去匹配（``/`` 就是文件系统根），于是普通句子
    也可能匹配到候选 → 第一次回车变成"接受候选"，用户要按两次。
    """
    from pathlib import Path

    from gis_cli.ui import prompt as ui_prompt

    reader = ui_prompt.PromptInput(Path("."), boxed=True)
    chain = _completer_of(reader)
    seen: list[str] = []

    def _collect(node, depth=0):
        if node is None or depth > 6:
            return
        seen.append(type(node).__name__)
        for child in getattr(node, "_completers", []) or []:
            _collect(child, depth + 1)
        _collect(getattr(node, "fallback", None), depth + 1)

    _collect(chain)
    assert seen, "没拿到补全器"
    assert "PathCompleter" not in seen, f"兜底链里仍有 PathCompleter：{seen}"


def test_ordinary_sentence_yields_no_file_completions():
    """普通句子不应产生任何文件候选（这是"回车被吃掉"的触发条件）。"""
    from pathlib import Path

    from prompt_toolkit.document import Document

    from gis_cli.ui import prompt as ui_prompt

    reader = ui_prompt.PromptInput(Path("."), boxed=True)
    completer = _completer_of(reader)
    completions = list(completer.get_completions(Document("统计各社区老年人口并出图"), None))
    assert not completions, f"普通句子不该有候选，实际拿到 {[c.text for c in completions][:5]}"


# --------------------------------------------------------- 3) 排队语义


def test_queued_messages_are_merged_into_one_task():
    """运行中的多次输入要合并成**一个**后续任务，而不是各自成任务。

    用户的实际用法：为了完成同一件事会补充多次说明；要换任务会直接开新对话。
    """
    from gis_cli.ui.session_runner import EngineSession

    class _Loop:
        def inject_user_message(self, text: str) -> bool:
            return False  # 注入失败 → 走排队

        def request_stop(self, reason: str = "") -> bool:
            return True

    class _Bundle:
        loop = _Loop()
        llm = type("L", (), {"config": type("C", (), {"model": "m", "api_base": ""})()})()
        workspace = "."

    session = EngineSession(_Bundle())
    session._thread = type("T", (), {"is_alive": lambda self: True})()  # 假装在跑

    session.guide("补充一")
    session.guide("补充二")
    # 两条排队消息要合并成**一条**任务（用 ``\n\n`` 分隔），而不是各自成任务
    merged = session.merged_task("原始目标")
    assert merged == "原始目标\n\n补充一\n补充二", repr(merged)
    # 取走之后就空了（不会重复执行）
    assert session.take_queued() == []


def test_queue_is_visible_to_the_user():
    """排队要有可见反馈（否则用户以为消息丢了，会重复输入）。"""
    from gis_cli.ui.session_runner import EngineSession

    class _Loop:
        def inject_user_message(self, text: str) -> bool:
            return False

    class _Bundle:
        loop = _Loop()
        llm = type("L", (), {"config": type("C", (), {"model": "m", "api_base": ""})()})()
        workspace = "."

    session = EngineSession(_Bundle())
    session._thread = type("T", (), {"is_alive": lambda self: True})()
    session.guide("第一条")
    assert session.queued_count == 1
    assert "排队" in session.queue_hint()

# -*- coding: utf-8 -*-
"""自建 prompt_toolkit 布局：四边完整的输入框（真框，不是拼字符串）。

## 为什么要自建

原先的"框"是拼出来的：上边框写进 prompt 文本，下边框靠 ``bottom_toolbar``。
但 prompt_toolkit 给 ``bottom_toolbar`` 加了门禁（``shortcuts/prompt.py``）::

    filter=... & renderer_height_is_known

``renderer_height_is_known`` 要先知道"光标下方还有几行"（``renderer.py``）：
Win32 有原生 API，Vt100 只能等终端回 CPR（光标位置应答）。**终端不回 CPR 时
下边框永远不显示**——实测就是这样，所以看到"框只有上半部分"。

自建布局把下边框做成普通 ``Window``，不依赖那个门禁，四边就都稳定了。
左/右竖线用固定宽度的 ``Window(char=...)`` 做"栏"，这样**补全菜单那些行
也有左右边界**（菜单是浮层，本身不带边栏）。

## 结构

::

    ╭────────────────────────────────────────────────╮   ← 纯线
    │  › 用户输入（多行时框跟着变长，不是另开一个框）     │
    │       /help      /model     ...                │   ← 菜单（框内浮层）
    ╰────────────────────────────────────────────────╯   ← 纯线
    deepseek-v4.1-flash · 运行中 · 第 3 轮 · 12s · ↑13.4k ↓805 · 工具 2

- 上下边框是**纯线**：不放“输入”标题、不放快捷键提示、不放状态。
- 状态行（模型/轮次/用时/token/工具）**常驻在框下**，且随运行实时刷新。
- 快捷键提示不在界面上显示；键位说明只在 ``/help`` 与输入 `/` 的菜单里。
"""

from __future__ import annotations

import math
from typing import Any, Callable

try:  # pragma: no cover - 环境相关
    from prompt_toolkit.application import Application
    from prompt_toolkit.application.current import get_app
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document
    from prompt_toolkit.enums import SEARCH_BUFFER
    from prompt_toolkit.filters import Condition, has_focus, is_done
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Float, FloatContainer, HSplit, VSplit, Window
    from prompt_toolkit.layout.containers import ConditionalContainer
    from prompt_toolkit.layout.controls import (
        BufferControl,
        FormattedTextControl,
        SearchBufferControl,
    )
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.layout.layout import Layout
    from prompt_toolkit.layout.menus import MultiColumnCompletionsMenu
    from prompt_toolkit.layout.processors import (
        AppendAutoSuggestion,
        ConditionalProcessor,
        HighlightSelectionProcessor,
        ReverseSearchProcessor,
    )
    from prompt_toolkit.widgets.toolbars import SearchToolbar

    _PT_AVAILABLE = True
except Exception:  # pragma: no cover - 降级环境
    _PT_AVAILABLE = False


#: 行前缀宽度（`` › `` 或续行缩进 3 格）——算折行时要扣掉
PREFIX_WIDTH = 3
#: 菜单打开时输入区预留的行数（官方 PromptSession 的 reserve_space_for_menu）
MENU_RESERVE_ROWS = 20
#: 输入区最多长到几行（再长就由控件内部滚动，免得把屏幕吃光）
MAX_INPUT_LINES = 12
#: 应用定时重绘间隔：状态行要跟着运行进度跳动（tracker 由工作线程更新）
REFRESH_INTERVAL = 0.4
#: 左/右边栏宽度（1 格符号 + 1 格留白）
RAIL_WIDTH = 2


class ChatPrompt:
    """四边完整的聊天式输入框。

    每次 :meth:`read` 都重建 Buffer/Layout/Application，避免跨次运行的
    补全状态、光标位置互相污染（官方 PromptSession 也要靠 ``reset`` 达到同样目的）。
    """

    def __init__(
        self,
        *,
        history: Any = None,
        completer: Any = None,
        style: Any = None,
        extra_bindings: Any = None,
        status_provider: Callable[[], str] | None = None,
        on_toggle_output: Callable[[], None] | None = None,
        on_interrupt: Callable[[], bool] | None = None,
        marker: str = "›",
        rail_char: str = "│",
        fill_char: str = "─",
        corner: tuple[str, str, str, str] = ("╭", "╮", "╰", "╯"),
        reserve_menu: int = MENU_RESERVE_ROWS,
        max_input_lines: int = MAX_INPUT_LINES,
        refresh_interval: float = REFRESH_INTERVAL,
        input: Any = None,
        output: Any = None,
    ) -> None:
        if not _PT_AVAILABLE:  # pragma: no cover - 调用方应先查 available()
            raise RuntimeError("prompt_toolkit 不可用")
        self._history = history
        self._completer = completer
        self._style = style
        self._extra_bindings = extra_bindings
        self._status_provider = status_provider or (lambda: "")
        self._on_toggle_output = on_toggle_output
        self._on_interrupt = on_interrupt
        self._marker = marker
        self._rail = rail_char
        self._fill = fill_char
        self._corners = corner
        self._reserve_menu = max(0, int(reserve_menu))
        self._max_input_lines = max(1, int(max_input_lines))
        self._refresh_interval = max(0.0, float(refresh_interval))
        self._input = input
        self._output = output
        #: 最近一次构建的输入 buffer（测试与上层要用）
        self.buffer: Any = None

    # ------------------------------------------------- 兼容 PromptSession 的读法

    @property
    def default_buffer(self) -> Any:
        """兼容 PromptSession.default_buffer（测试与上层都依赖这个名字）。

        必须带上本实例的 io：不带就会去开真实控制台（无头环境下直接抛
        NoConsoleScreenBufferError）。
        """
        if self.buffer is None:
            self._build(input=self._input, output=self._output)
        return self.buffer

    @property
    def completer(self) -> Any:
        return self._completer

    #: 与 complete_while_typing 互斥，本实现固定关闭（见 prompt.py 注释）
    enable_history_search = False

    def prompt(self, *_args: Any, **_kwargs: Any) -> str:
        """兼容 PromptSession.prompt()：跑一次并返回输入文本。"""
        return self.read()

    # ------------------------------------------------------------- 对外

    def read(
        self,
        *,
        input: Any = None,
        output: Any = None,
        default: str = "",
        pre_run: Callable[[], None] | None = None,
    ) -> str:
        """显示输入框并读取一次输入。EOF 抛 ``EOFError``，中断抛 ``KeyboardInterrupt``。"""
        app = self.prepare(input=input, output=output, default=default, pre_run=pre_run)
        return app.run()

    def prepare(
        self,
        *,
        input: Any = None,
        output: Any = None,
        default: str = "",
        pre_run: Callable[[], None] | None = None,
    ) -> Any:
        """建好 Application 并返回它（同时把 ``buffer`` 暴露出来）。

        单独拆一步是因为外部（截图工具/无头测试）需要在**输入进行中**
        就能读到 buffer（补全状态、当前文本）。
        """
        return self._build(
            input=input if input is not None else self._input,
            output=output if output is not None else self._output,
            default=default,
            pre_run=pre_run,
        )

    # ------------------------------------------------------------- 构建

    def _build(
        self,
        *,
        input: Any = None,
        output: Any = None,
        default: str = "",
        pre_run: Callable[[], None] | None = None,
    ) -> Any:
        buffer = Buffer(
            name="chat-input",
            history=self._history,
            completer=self._completer,
            auto_suggest=AutoSuggestFromHistory(),
            # 边打边弹菜单：输入 / 就列出命令，不靠 Tab。
            # 注意 enable_history_search 与它互斥（见 prompt.py 里的说明）。
            complete_while_typing=True,
            enable_history_search=False,
            multiline=True,
            document=Document(default, cursor_position=len(default)),
        )
        self.buffer = buffer
        buffer.accept_handler = self._accept

        search_buffer = Buffer(name=SEARCH_BUFFER)
        search_control = SearchBufferControl(
            buffer=search_buffer,
            input_processors=[ReverseSearchProcessor()],
            ignore_case=True,
        )

        content = FloatContainer(
            HSplit([self._input_window(buffer, search_control)]),
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    transparent=True,
                    content=MultiColumnCompletionsMenu(
                        show_meta=True,
                        extra_filter=has_focus(buffer) & ~is_done,
                    ),
                ),
            ],
        )

        root = HSplit(
            [
                self._border_line_window(side="top"),
                VSplit(
                    [
                        self._rail_window(),
                        Window(width=1),  # 留白
                        content,
                        Window(width=1),  # 留白
                        self._rail_window(),
                    ]
                ),
                self._border_line_window(side="bottom"),
                # 框下常驻状态行（模型 / 状态 / 轮次 / 用时 / token / 工具）
                self._status_window(),
                # 反向搜索状态栏（Ctrl+R），平时不占位
                ConditionalContainer(
                    SearchToolbar(search_buffer, ignore_case=True),
                    Condition(lambda: get_app().layout.current_control is search_control),
                ),
            ]
        )

        bindings = self._key_bindings(buffer)
        app: Any = Application(
            layout=Layout(root, focused_element=content),
            key_bindings=bindings,
            style=self._style,
            full_screen=False,  # 走"就地重绘"，与滚动转录共存
            mouse_support=False,
            # 定时重绘：状态行要跟着运行进度跳动（tracker 由工作线程更新，
            # 不主动重绘就看不到轮次/token 变化）
            refresh_interval=self._refresh_interval,
            input=input,
            output=output,
        )
        if pre_run is not None:
            app.pre_run_callables.append(pre_run)
        return app

    def _accept(self, buffer: Any) -> bool:
        get_app().exit(result=buffer.document.text, style="class:accepted")
        return True

    # ------------------------------------------------------------- 部件

    def _rail_window(self) -> Any:
        """左/右侧竖线栏（固定宽度，跨整个框高，菜单行也有边界）。"""
        return Window(width=1, char=self._rail, style="class:frame")

    def _input_window(self, buffer: Any, search_control: Any) -> Any:
        """输入区：多行编辑，每行都有 ``›`` / 缩进前缀。"""
        control = BufferControl(
            buffer=buffer,
            search_buffer_control=search_control,
            # 传 list：BufferControl 内部会自己 merge_processors
            input_processors=[
                HighlightSelectionProcessor(),
                ConditionalProcessor(
                    AppendAutoSuggestion(), has_focus(buffer) & ~is_done
                ),
            ],
            include_default_input_processors=False,
        )
        return Window(
            control,
            height=self._input_height,
            get_line_prefix=self._line_prefix,
            wrap_lines=True,
            # 不写 dont_extend_height：那会把高度钉在 min 上，多行内容溢出到框外，
            # 看起来就像“每换行另开一个框”（用户反馈的问题）。
        )

    def _input_height(self) -> Any:
        """输入区高度 = 内容行数 + 补全菜单行数。

        为什么不写死 1 行（也不写 ``dont_extend_height``）：那样多行输入会溢出
        到框外，看起来像“每换行另开一个新框”（用户反馈的问题）；要的是
        **框跟着变长**。

        菜单是 Float（浮层），不会把容器撑高，所以必须在这里把菜单行数加进去，
        否则菜单会被裁掉（实测确认）。
        """
        rows = self._content_lines()
        state = self.buffer.complete_state
        if state is not None and state.completions:
            rows += min(_menu_rows(state, self._available_width()), self._reserve_menu)
        return Dimension(min=min(rows, self._max_input_lines))

    def _content_lines(self) -> int:
        """输入内容会占几行（含中文宽度与自动折行）。

        不能用 ``document.line_count``：那只是逻辑行数，一行长文自动折行后会
        超出框外。所以按显示宽度算每行需几个视觉行。
        """
        try:
            text = self.buffer.document.text or ""
        except Exception:  # pragma: no cover - buffer 未就绪
            return 1
        available = max(10, self._available_width() - PREFIX_WIDTH)
        total = 0
        for raw_line in text.split("\n"):
            total += max(1, math.ceil(_display_width(raw_line) / available))
        return max(1, total)

    def _available_width(self) -> int:
        """输入区可用宽度（终端宽度减去左右侧栏与留白）。"""
        return max(20, _term_columns() - 2 * (RAIL_WIDTH + 1))

    def _line_prefix(self, lineno: int, wrap_count: int) -> Any:
        """行前缀：首行 ``› ``，续行缩进对齐（竖线由侧栏画，这里不加）。"""
        if lineno == 0 and wrap_count == 0:
            return FormattedText([("class:prompt", f" {self._marker} ")])
        return FormattedText([("class:frame", "   ")])

    def _border_line_window(self, *, side: str) -> Any:
        """上/下边框：**纯线**（不带标题、提示、状态）。

        用户明确要求：“聊天框是两条纯线”“快捷键提示全删”。所以这里不再往边框里
        嵌任何文字；状态信息改放到框下单独的常驻行（见 :meth:`_status_window`）。
        """
        left, right = (
            (self._corners[0], self._corners[1])
            if side == "top"
            else (self._corners[2], self._corners[3])
        )

        def _render() -> Any:
            width = _term_columns()
            return FormattedText(
                [("class:frame", f"{left}{self._fill * max(0, width - 2)}{right}")]
            )

        return Window(
            FormattedTextControl(_render),
            height=1,
            dont_extend_height=True,
            style="class:frame",
        )

    def _status_window(self) -> Any:
        """框下常驻状态行（模型 · 状态 · 轮次 · 用时 · token · 工具）。

        一直显示（用户要求），且随运行实时刷新（靠 Application.refresh_interval）。
        超宽时截断加省略号，避免换行把框挤下去。
        """

        def _render() -> Any:
            text = " ".join(str(self._status_provider() or "").split())
            width = _term_columns()
            if _display_width(text) > width - 1:
                text = _clip_to_width(text, width - 2) + "…"
            return FormattedText([("class:hint", text)])

        return Window(
            FormattedTextControl(_render),
            height=1,
            dont_extend_height=True,
            style="class:hint",
        )

    # ------------------------------------------------------------- 按键

    def _key_bindings(self, buffer: Any) -> Any:
        bindings = KeyBindings()

        @bindings.add("enter")
        def _(event: Any) -> None:
            """菜单开着时先应用当前候选，再按一次才提交（pi/Claude 手感）。"""
            buf = event.current_buffer
            state = buf.complete_state
            if state is not None and state.current_completion is not None:
                buf.apply_completion(state.current_completion)
                return
            buf.validate_and_handle()

        @bindings.add("escape", "enter")
        @bindings.add("c-j")
        def _(event: Any) -> None:
            """Alt+Enter / Esc+Enter / Ctrl+J 换行。"""
            event.current_buffer.insert_text("\n")

        @bindings.add("escape")
        def _(event: Any) -> None:
            """菜单开着先关菜单；否则请求中断当前任务。"""
            buf = event.current_buffer
            if buf.complete_state is not None:
                buf.cancel_completion()
                return
            if self._on_interrupt is not None:
                try:
                    if not self._on_interrupt():
                        return
                except Exception:
                    return
            event.app.invalidate()

        @bindings.add("c-c")
        def _(event: Any) -> None:
            """有内容清空；空内容退出（KeyboardInterrupt）交给上层处理。"""
            buf = event.current_buffer
            if buf.text:
                buf.reset()
                return
            # 不能直接 raise：自建 Application 不会把 key 回调里的异常
            # 当成"退出"（官方 PromptSession 也是用 exit(exception=...)）。
            event.app.exit(exception=KeyboardInterrupt(), style="class:exiting")

        @bindings.add("c-d")
        def _(event: Any) -> None:
            """空内容时退出（EOF）；有内容时删除光标处字符。"""
            buff = event.current_buffer
            if not buff.text:
                event.app.exit(exception=EOFError(), style="class:exiting")
                return
            buff.delete()

        @bindings.add("c-o")
        def _(event: Any) -> None:
            if self._on_toggle_output is not None:
                self._on_toggle_output()

        @bindings.add("c-l")
        def _(event: Any) -> None:
            event.app.renderer.clear()

        @bindings.add("c-r")
        def _(event: Any) -> None:
            event.current_buffer.start_reverse_incremental_search()

        if self._extra_bindings is not None:
            return _merge_bindings(self._extra_bindings, bindings)
        return bindings


# ------------------------------------------------------------------ 工具


def _menu_rows(state: Any, available_width: int) -> int:
    """补全菜单会占几行（含 meta 行）。

    复刻 ``MultiColumnCompletionMenuControl.create_content`` 的列宽规则：
    列宽先取最长候选的显示宽度（+1 间隔），超过 suggested_max_column_width(30)
    就按比例折半，最后用 ``(宽度 - 3) // 列宽`` 得可见列数。
    这里必须用**渲染时**的列宽，不能用 ``preferred_height``：后者用的是未折半的
    列宽，会多算一行（实测多出一行空白）。
    """
    import math

    margin = 3  # _required_margin：右边留白 + 箭头位
    suggested = 30
    try:
        from prompt_toolkit.utils import get_cwidth

        column_width = max(get_cwidth(c.display_text) for c in state.completions) + 1
    except Exception:  # pragma: no cover - 降级估算
        column_width = 24
    column_width = min(available_width - margin, column_width)
    if column_width > suggested:
        column_width //= column_width // suggested
    visible_columns = max(1, (available_width - margin) // max(1, column_width))
    rows = int(math.ceil(len(state.completions) / float(visible_columns)))
    # 说明行只在"真的选中了某一项"时才渲染（complete_index 非空）。
    # 刚弹出菜单时还没选中，多算这行会留一条空白（实测）。
    has_selection = getattr(state, "complete_index", None) is not None
    has_meta = has_selection and any(getattr(c, "display_meta", "") for c in state.completions)
    return max(1, rows + (1 if has_meta else 0))


def _display_width(text: str) -> int:
    """字符串的显示宽度（中文/全角算 2 格）。"""
    try:
        from prompt_toolkit.utils import get_cwidth

        return get_cwidth(text)
    except Exception:  # pragma: no cover - 降级
        return len(text)


def _clip_to_width(text: str, limit: int) -> str:
    """按显示宽度截断（不把中文截成半个字）。"""
    if limit <= 0:
        return ""
    kept: list[str] = []
    used = 0
    for char in text:
        width = _display_width(char)
        if used + width > limit:
            break
        kept.append(char)
        used += width
    return "".join(kept)


def _term_columns(default: int = 100) -> int:
    """当前终端宽度（拿不到就用默认值）。"""
    try:
        return max(40, get_app().output.get_size().columns)
    except Exception:  # pragma: no cover - 无活动应用
        return default


def _merge_bindings(first: Any, second: Any) -> Any:
    from prompt_toolkit.key_binding import merge_key_bindings

    return merge_key_bindings([first, second])


__all__ = ["ChatPrompt", "available"]


def available() -> bool:
    """自建输入框是否可用（prompt_toolkit 装上了就行）。"""
    return _PT_AVAILABLE

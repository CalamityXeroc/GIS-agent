# -*- coding: utf-8 -*-
"""终端输入框（prompt_toolkit）。

替代原来"裸 ``input()`` + 两次回车结束多行"的启发式，提供：

- 多行编辑：``Enter`` 发送，``Shift+Enter`` / ``Alt+Enter`` / ``Ctrl+J`` 换行；
- 历史：上下键浏览，落盘 ``workspace/.gis_agent/history.txt``；``Ctrl+R`` 反向搜索；
- 补全：输入 ``/`` 即弹出**命令菜单**（↑↓ 选择、Enter 应用、Esc 取消），
  ``@`` 补文件引用，``Tab`` 补路径；
- 中断：``Ctrl+C`` 清空当前输入（空行时抛 ``KeyboardInterrupt``）、``Ctrl+D`` 退出；
- 其他：``Ctrl+O`` 切换工具输出、``Ctrl+L`` 清屏；
- 运行期输出：``patch_stdout`` 保证后台线程打印不会冲乱输入框。

``prompt_toolkit`` 是**必装依赖**（见 ``pyproject.toml``）；意外缺失时 :func:`degrade_notice`
会返回一行警告，而不是默默退化成只有一个 ``›`` 的 ``input()``（这个静默降级曾导致
"输入框怎么只有一个大于号"排查了很久）。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .console_env import interactive_terminal as _console_interactive_terminal

logger = logging.getLogger(__name__)

#: ``patch_stdout`` 在无真实控制台时抛的错误（Windows 的 NoConsoleScreenBufferError
#: 是 OSError 子类，但不同平台/版本会变，所以宁可宽一点：这里只用于“换个读法”，
#: 不是掩盖真错误——真的读失败会在下一行原样抛出去）。
_CONSOLE_PATCH_ERRORS: tuple[type[BaseException], ...] = (OSError, RuntimeError)

try:  # pragma: no cover - 环境相关
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion, PathCompleter
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.shortcuts import CompleteStyle
    from prompt_toolkit.styles import Style

    _PT_AVAILABLE = True
    _PT_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - 环境相关
    _PT_AVAILABLE = False
    _PT_IMPORT_ERROR = str(exc)

    # 降级占位：让本模块在任何环境下都能 import（否则未装 prompt_toolkit 时
    # 连 "能不能用输入框" 都判断不了——实测 CI/venv 环境会直接 ImportError）。
    class Completer:  # type: ignore[no-redef]
        def get_completions(self, document, complete_event):
            return iter(())

    class Completion:  # type: ignore[no-redef]
        def __init__(self, text: str = "", start_position: int = 0, display_meta: str = "") -> None:
            self.text = text
            self.start_position = start_position
            self.display_meta = display_meta

    class PathCompleter(Completer):  # type: ignore[no-redef]
        def __init__(self, **_kwargs) -> None:
            pass

    class CompleteStyle:  # type: ignore[no-redef]
        COLUMN = "COLUMN"
        MULTI_COLUMN = "MULTI_COLUMN"
        READLINE_LIKE = "READLINE_LIKE"

    class _MissingHistory:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    FileHistory = InMemoryHistory = _MissingHistory  # type: ignore[assignment,misc]

    def KeyBindings():  # type: ignore[no-redef]
        class _Bindings:
            def add(self, *_args, **_kwargs):
                def _decorator(func):
                    return func

                return _decorator

        return _Bindings()

    def patch_stdout(*_args, **_kwargs):  # type: ignore[no-redef]
        import contextlib

        return contextlib.nullcontext()

    class Style:  # type: ignore[no-redef]
        @staticmethod
        def from_dict(_mapping):
            return None

    class PromptSession:  # type: ignore[no-redef]
        def __init__(self, **_kwargs) -> None:
            raise RuntimeError(f"prompt_toolkit 不可用: {_PT_IMPORT_ERROR}")

    def AutoSuggestFromHistory():  # type: ignore[no-redef]
        return None

#: 内置斜杠命令（命令名, 说明）——供补全与 /help 使用。
#: 这张表必须与 ``interactive._handle_command`` 能处理的命令保持一致（有测试锁定）。
#: 内置斜杠命令（命令名, 说明）——**从 commands.COMMANDS 派生**，保证菜单、
#: ``/help``、实际实现三者永不分叉（以前是两张表，漂移过）。
def _slash_commands() -> tuple[tuple[str, str], ...]:
    try:
        from .commands import COMMANDS

        return tuple((spec.slash_name, spec.summary) for spec in COMMANDS if spec.visible)
    except Exception:  # pragma: no cover - 极端降级环境
        return (("/help", "显示帮助与快捷键"), ("/exit", "退出"))


SLASH_COMMANDS: tuple[tuple[str, str], ...] = _slash_commands()


def available() -> bool:
    """prompt_toolkit 是否可用。"""
    return _PT_AVAILABLE


def import_error() -> str:
    return _PT_IMPORT_ERROR


def degrade_notice() -> str:
    """输入框不可用时给用户的警告（空串表示一切正常）。

    静默降级成 ``› `` 曾让"只有一个大于号"变成一个查了很久的谜——
    这里把降级摆到台面上，一眼能定位。
    """
    import os

    if _PT_AVAILABLE:
        return ""
    if os.environ.get("GIS_AGENT_ALLOW_PLAIN_INPUT"):
        return ""
    return (
        f"输入框/补全不可用（prompt_toolkit: {_PT_IMPORT_ERROR or '未安装'}），"
        "已降级为纯文本输入。安装后重启即可：pip install 'prompt-toolkit>=3.0.43'"
    )


# ------------------------------------------------------------------ 补全器


class SlashMenuCompleter(Completer):
    """斜杠命令菜单的调度器：`/` 开头走命令表，否则走后面的补全器。

    为什么不用 ``DynamicCompleter`` + ``get_app().current_buffer``：实测那样在
    独立 Buffer（测试、嵌入场景）里会误判——条件不成立就掉到 `PathCompleter`，
    而 ``/`` 正好是文件系统根目录，于是输入 ``/`` 会弹出 300+ 个文件候选。
    改用**从 document 自身判断**（纯函数、可测、不依赖全局应用状态）。

    参数补全：``/theme `` 之后列出配色、``/model `` 之后列出模型——与 pi 一致。
    ``registry`` 为空时退化成只补命令名（保留旧行为，测试/降级用）。
    """

    #: 行首 `/<命令名>` 形式：命令名里不允许空格
    _PATTERN = re.compile(r"^\s*/[^\s]*$")
    #: `/命令 参数前缀` 形式（命令与参数之间至少一个空格）
    _ARG_PATTERN = re.compile(r"^\s*/(?P<cmd>[^\s]+)\s+(?P<arg>[^\s]*)$")

    def __init__(
        self,
        commands: Sequence[tuple[str, str]] = SLASH_COMMANDS,
        fallback: Completer | None = None,
        registry: Any = None,
    ) -> None:
        self.commands = list(commands)
        self.fallback = fallback
        self.registry = registry
        self._command_completer = SlashCommandCompleter(self.commands)

    def get_completions(self, document, complete_event):  # noqa: D401 - pt 接口
        text = document.text_before_cursor
        if self._PATTERN.match(text):
            yield from self._command_completer.get_completions(document, complete_event)
            return
        matched_args = self._ARG_PATTERN.match(text)
        if matched_args is not None:
            produced = False
            for completion in self._arg_completions(matched_args):
                produced = True
                yield completion
            if produced:
                return
        if self.fallback is not None:
            yield from self.fallback.get_completions(document, complete_event)

    def _arg_completions(self, matched: Any) -> Any:
        """按命令名给出参数候选（说明里带上该参数是干什么的）。"""
        from .commands import normalize_args

        if self.registry is None:
            return []
        spec = self.registry.find(matched.group("cmd"))
        if spec is None or not spec.args:
            return []
        prefix = matched.group("arg")
        lowered = prefix.lower()
        return [
            Completion(value, start_position=-len(prefix), display=value, display_meta=desc)
            for value, desc in normalize_args(spec.args)
            if value.lower().startswith(lowered)
        ]


class SlashCommandCompleter(Completer):
    """行首 ``/`` 补全命令（输入斜杠即列出全部命令，与 pi / Claude Code 一致）。

    触发条件：**光标前的内容 trim 后以 ``/`` 开头**，且 ``/`` 与光标之间只有一个词。
    刚输入 ``/`` 时词为空 → 返回全部命令（而不是什么都不给），这样菜单会立刻展开。
    """

    def __init__(self, commands: Sequence[tuple[str, str]] = SLASH_COMMANDS) -> None:
        self.commands = list(commands)

    def get_completions(self, document, complete_event):  # noqa: D401 - pt 接口
        text = document.text_before_cursor
        stripped = text.strip()
        if not stripped.startswith("/"):
            return
        # 只在"命令名"这一段补全：一旦出现空格，说明进入参数区，不再列命令
        word = stripped.split(" ", 1)[0]
        if " " in stripped:
            return
        lowered = word.lower()
        for name, desc in self.commands:
            if name.lower().startswith(lowered):
                yield Completion(
                    name,
                    start_position=-len(word),
                    display=name,
                    display_meta=desc,
                )


class AtFileCompleter(Completer):
    """``@`` 之后的文件引用补全（相对工作区）。"""

    def __init__(self, provider: Callable[[], Iterable[str]]) -> None:
        self._provider = provider
        self._cache: list[str] = []

    def refresh(self) -> None:
        try:
            self._cache = sorted({str(item) for item in self._provider()})
        except Exception:
            self._cache = []

    def get_completions(self, document, complete_event):  # noqa: D401 - pt 接口
        text = document.text_before_cursor
        idx = text.rfind("@")
        if idx < 0:
            return
        token = text[idx + 1:]
        if any(ch.isspace() for ch in token):
            return
        if not self._cache:
            self.refresh()
        lowered = token.lower()
        for path in self._cache:
            if lowered in path.lower():
                yield Completion(path, start_position=-len(token), display_meta="文件")


class MixedCompleter(Completer):
    """先试补全器，按顺序返回第一个有结果的那个（避免命令与文件互相污染）。"""

    def __init__(self, completers: Sequence[Completer]) -> None:
        self._completers = list(completers)

    def get_completions(self, document, complete_event):  # noqa: D401 - pt 接口
        for completer in self._completers:
            produced = False
            for completion in completer.get_completions(document, complete_event):
                produced = True
                yield completion
            if produced:
                return


def workspace_file_provider(workspace: Path, limit: int = 400) -> Callable[[], list[str]]:
    """扫描工作区（input/ 优先）里的文件名，供 ``@`` 补全。"""

    def _scan() -> list[str]:
        root = Path(workspace)
        found: list[str] = []
        for base in (root / "input", root / "workspace" / "input", root):
            if not base.exists():
                continue
            for path in sorted(base.rglob("*")):
                if path.is_file() and path.suffix.lower() in {
                    ".shp", ".gdb", ".tif", ".tiff", ".csv", ".txt", ".json", ".geojson", ".dbf", ".xlsx", ".jpg", ".png",
                }:
                    try:
                        rel = path.relative_to(root)
                    except Exception:
                        rel = path
                    found.append(str(rel).replace("\\", "/"))
                if len(found) >= limit:
                    return found
        return found

    return _scan


# ------------------------------------------------------------------ 样式


def prompt_style(mode: str = "system", no_color: bool = False) -> Any:
    """把 rich 的角色色映射成 prompt_toolkit 样式。"""
    if not _PT_AVAILABLE:  # pragma: no cover
        return None
    if no_color:
        return Style.from_dict({"prompt": "", "bottom-toolbar": "", "frame": "", "hint": ""})
    dark = (mode or "system") != "light"
    accent = "ansicyan bold" if dark else "ansiblue bold"
    hint = "ansibrightblack"
    # 补全菜单：当前项用反色高亮（pi / Claude Code 的选中观感），说明列压暗
    selected = "ansicyan bold" if dark else "ansiblue bold"
    return Style.from_dict(
        {
            "prompt": accent,
            "frame.border": "ansibrightblack",
            "frame.label": accent,
            "bottom-toolbar": "bg:default " + hint,
            "hint": hint,
            "completion-menu.completion": "bg:default " + hint,
            "completion-menu.completion.current": f"bg:default {selected} reverse",
            "completion-menu.meta.completion": hint,
            "completion-menu.meta.completion.current": selected,
            "scrollbar.background": "bg:default",
            "scrollbar.button": "bg:default " + hint,
        }
    )


# ------------------------------------------------------------------ 输入框


class PromptInput:
    """交互输入：多行编辑 + 历史 + 补全 + 快捷键。"""

    def __init__(
        self,
        workspace: Path,
        *,
        mode: str = "system",
        no_color: bool = False,
        busy: bool = False,
        on_toggle_output: Callable[[], None] | None = None,
        on_interrupt: Callable[[], bool] | None = None,
        extra_commands: Sequence[tuple[str, str]] = (),
        boxed: bool = True,
        input: Any = None,
        output: Any = None,
        registry: Any = None,
        stream_provider: Callable[[], str] | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.mode = mode
        self.no_color = no_color
        self.busy = busy
        self.on_toggle_output = on_toggle_output
        self.on_interrupt = on_interrupt
        #: 是否画"聊天框"（真框：四边 + 菜单在框内）
        self.boxed = boxed
        #: 可注入的 prompt_toolkit 输入/输出（测试用；生产留 None）
        self._input = input
        self._output = output
        #: 命令注册表（带参数候选）；由 agent 层注入，ui 层不反向依赖它
        self.registry = registry
        #: 流式实时预览的数据源（模型正在生成的文本尾部；空串 = 没在流式）
        self._stream_provider = stream_provider
        #: 最近一次输入框的 Buffer（测试与上层要用）
        self._buffer: Any = None
        self._session = None
        self._commands = list(SLASH_COMMANDS) + list(extra_commands)

    # -- 构造（延迟到首次使用，便于测试注入 input/output） -----------------

    def session(self, *, input: Any = None, output: Any = None) -> Any:
        if self._session is None:
            if not _PT_AVAILABLE:  # pragma: no cover
                raise RuntimeError(f"prompt_toolkit 不可用: {_PT_IMPORT_ERROR}")
            history_path = self.workspace / ".gis_agent" / "history.txt"
            try:
                history_path.parent.mkdir(parents=True, exist_ok=True)
                history = FileHistory(str(history_path))
            except Exception:
                history = InMemoryHistory()
            self._session = PromptSession(
                history=history,
                auto_suggest=AutoSuggestFromHistory()
                if not _is_dumb_terminal()
                else None,
                # 兜底补全器里**不放 PathCompleter**：它把任意文本当文件路径匹配，
                # 于是普通句子也会命中候选 → 第一次回车只是“接受候选”而不提交，
                # 用户得按两次回车（实测反馈）。文件补全保留在 ``@`` 触发的那条链上。
                completer=SlashMenuCompleter(
                    self._commands,
                    fallback=AtFileCompleter(workspace_file_provider(self.workspace)),
                ),
                key_bindings=self._key_bindings(),
                style=prompt_style(self.mode, self.no_color),
                # 边打边弹菜单：斜杠命令要"输入 / 就列出全部"，不能只靠 Tab。
                # dumb 终端本来就没有 tty，早被 _interactive_tty() 挡在门外了。
                complete_while_typing=True,
                complete_style=CompleteStyle.MULTI_COLUMN,
                # 伪框 + 状态栏 + 菜单同时在场，预留行数不足会把菜单裁掉
                reserve_space_for_menu=10,
                # 注意：不能开 enable_history_search！prompt_toolkit 内部用
                # Condition(complete_while_typing and not enable_history_search)
                # 把它和"边打边弹菜单"设成互斥（见 _create_default_buffer），
                # 开了它输入 / 就永远不弹命令菜单。↑↓ 翻历史本来就默认可用，
                # 反向搜索另有 Ctrl+R 绑定。
                enable_history_search=False,
                mouse_support=False,
                input=input,
                output=output,
            )
        return self._session

    def _key_bindings(self) -> Any:
        bindings = KeyBindings()

        @bindings.add("enter")
        def _(event):  # noqa: ANN001 - pt 事件
            """Enter：菜单开着时先"应用当前候选"（pi/Claude 手感），再按才发送。"""
            buffer = event.current_buffer
            state = buffer.complete_state
            if state is not None and state.current_completion is not None:
                buffer.apply_completion(state.current_completion)
                return
            buffer.validate_and_handle()

        @bindings.add("escape", "enter")
        @bindings.add("c-j")
        def _(event):  # noqa: ANN001
            """Alt+Enter / Esc+Enter / Ctrl+J 换行。"""
            event.current_buffer.insert_text("\n")

        @bindings.add("c-o")
        def _(event):  # noqa: ANN001
            if self.on_toggle_output is not None:
                self.on_toggle_output()

        @bindings.add("c-l")
        def _(event):  # noqa: ANN001
            """Ctrl+L 清屏（Claude/Codex 习惯）。"""
            event.app.renderer.clear()

        @bindings.add("c-r")
        def _(event):  # noqa: ANN001
            """Ctrl+R 反向增量搜索历史。"""
            event.current_buffer.start_reverse_incremental_search()

        @bindings.add("escape")
        def _(event):  # noqa: ANN001
            """Esc：菜单开着先关菜单；否则运行中=请求中断；空闲时无动作。"""
            buffer = event.current_buffer
            if buffer.complete_state is not None:
                buffer.cancel_completion()
                return
            if self.on_interrupt is not None:
                try:
                    handled = bool(self.on_interrupt())
                except Exception:
                    handled = False
                if not handled:
                    return
            event.app.invalidate()

        @bindings.add("c-c")
        def _(event):  # noqa: ANN001
            """有内容清空，空内容抛 KeyboardInterrupt（交给上层决定是否退出）。"""
            buffer = event.app.current_buffer
            if buffer.text:
                buffer.reset()
                return
            raise KeyboardInterrupt

        return bindings

    # -- 读取 ---------------------------------------------------------

    def _chat_prompt(self, *, busy: bool, status: str | Callable[[], str]) -> Any:
        """自建输入框（四边完整的真框）。

        历史：以前是拼字符串的"伪框"——上边框写进 prompt 文本、下边框靠
        ``bottom_toolbar``。而 ``bottom_toolbar`` 被 prompt_toolkit 用
        ``renderer_height_is_known`` 门禁锁住（需要终端回 CPR），**终端不回
        CPR 时下边框永远不显示**，所以只能看到"框的上半部分"。
        自建布局把下边框做成普通 Window，不依赖那个门禁。
        """
        from . import tui as ui_tui

        history_path = self.workspace / ".gis_agent" / "history.txt"
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(history_path))
        except Exception:
            history = InMemoryHistory()
        return ui_tui.ChatPrompt(
            history=history,
            # 补全链：行首 ``/`` 走命令表；``@`` 走文件补全。
            # **不放 PathCompleter**：它把任意文本当文件路径匹配，普通句子命中了候选
            # 时，第一次回车就只是“接受候选”而不提交（用户实测要按两次回车）。
            completer=SlashMenuCompleter(
                self._commands,
                fallback=AtFileCompleter(workspace_file_provider(self.workspace)),
                registry=self.registry,
            ),
            style=prompt_style(self.mode, self.no_color),
            # 状态行每帧重算：传进来的可能是 callable（运行中要实时跳轮次/token）
            status_provider=status if callable(status) else (lambda: str(status or "")),
            on_toggle_output=self.on_toggle_output,
            on_interrupt=self.on_interrupt,
            input=self._input,
            output=self._output,
            stream_provider=self._stream_provider,
        )

    def read_line(
        self,
        prompt_text: str = "",
        *,
        busy: bool = False,
        status: str | Callable[[], str] = "",
    ) -> str | None:
        """读一行；EOF（Ctrl+D）返回 ``None``。

        非交互场景（stdin 不是终端，例如 ``echo "任务" | gis-agent chat``）
        直接用简单读取：prompt_toolkit 在管道/重定向下无法附着控制台，会直接报错。
        """
        if not _PT_AVAILABLE or not _interactive_tty():
            return _fallback_read(prompt_text)
        if not self.boxed:
            # 不用框时走官方 PromptSession（保持原有降级能力）
            return self._read_line_plain(prompt_text, busy=busy, status=status)
        try:
            prompt = self._chat_prompt(busy=busy, status=status)
            # 先 prepare（建好 Application 并把 buffer 挂到 prompt.buffer），
            # 再 run：外部（截图工具/无头测试）需要在输入进行中就能读到 buffer。
            app = prompt.prepare()
            self._buffer = prompt.buffer
            return app.run()
        except EOFError:
            return None
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # 控制台不支持时退回简单读取，而不是让整个会话崩掉
            if _is_console_error(exc):
                return _fallback_read(prompt_text)
            raise

    def _read_line_plain(
        self, prompt_text: str = "", *, busy: bool = False, status: str = ""
    ) -> str | None:
        """无框版读取（官方 PromptSession，保留作兼容与降级）。"""
        session = self.session()
        frame = ""
        try:
            return session.prompt(_prompt_markup(prompt_text, busy=busy, frame=frame))
        except EOFError:
            return None

    def _toolbar(self, *, busy: bool, width: int) -> Any:
        """底部工具栏（旧实现遗留，供 ``_read_line_plain`` 与测试使用）。"""
        from . import render as ui_render

        hints = ui_render.hint_text(busy=busy)
        line = box_bottom(hints=hints, width=width)

        def _render():  # pragma: no cover - 依赖终端
            return [("class:frame", line)]

        return _render

    def read_line_with_stdout_patch(
        self,
        prompt_text: str = "",
        *,
        busy: bool = False,
        status: str | Callable[[], str] = "",
    ) -> str | None:
        """带 ``patch_stdout`` 的读取（后台线程打印不会冲乱输入框）。

        ``patch_stdout`` 是给**真实控制台**用的：它要把后台线程的 print 重定向到
        “输入框上方”。它一构造就会去找 Windows 控制台，**注入自定义 input/output
        （测试、无头场景）时会直接抛 NoConsoleScreenBufferError**，那属于误报，
        不应该把它当成“控制台不支持”而降级到纯文本（实测就是这么误降级的）。
        """
        if not _PT_AVAILABLE or not _interactive_tty():
            return _fallback_read(prompt_text)
        if self._input is not None or self._output is not None:
            # 注入了自定义 io（无头/测试）：没有真实控制台可 patch，直接读
            return self.read_line(prompt_text, busy=busy, status=status)
        try:
            with patch_stdout():
                return self.read_line(prompt_text, busy=busy, status=status)
        except _CONSOLE_PATCH_ERRORS as exc:
            logger.debug("patch_stdout 不可用，改用直读: %s", exc)
            return self.read_line(prompt_text, busy=busy, status=status)

    def bottom_toolbar(self, text: str) -> Any:
        """底部状态行（返回 prompt_toolkit 可用的可调用对象）。"""

        def _toolbar():  # pragma: no cover - 依赖终端
            return [("class:hint", text)]

        return _toolbar

    def add_commands(self, commands: Sequence[tuple[str, str]]) -> None:
        self._commands.extend(commands)
        self._session = None  # 重建以刷新补全


# ------------------------------------------------------------------ 降级


def _is_dumb_terminal() -> bool:
    import os

    return os.environ.get("TERM", "").lower() in ("dumb", "") and not os.environ.get("WT_SESSION")


def _interactive_tty() -> bool:
    """标准输入与输出都是终端（判定实现在 :mod:`gis_cli.ui.console_env`）。"""
    return _console_interactive_terminal()


def interactive_terminal() -> bool:
    """当前会话能不能真的在终端里收键（prompt_toolkit 可用 + 两端都是终端）。

    计划卡闸门（``ChatPlanGate``）要据此决定“停下来等 s/e/q”还是“非 TTY 降级”：
    没有可读的终端时停下来就是挂死。

    与 ``plan_card.can_ask()`` 只差 ``prompt_toolkit`` 这一条：chat 的输入框本体就是
    ``PromptSession``，缺了就只剩 ``input()``，计划卡无法在输入框里等人；而 loop 的
    ``PlanGate`` 本来就读 ``input()``，不需要它。这条**唯一的分歧**有用例锁定
    （DAIR-38）。
    """
    return bool(_PT_AVAILABLE and _interactive_tty())


def _is_console_error(exc: BaseException) -> bool:
    """识别"当前控制台不支持全屏交互"这类错误（Windows 控制台/管道常见）。"""
    name = type(exc).__name__
    if name in ("NoConsoleScreenBufferError", "EOFError", "OSError"):
        return True
    text = str(exc).lower()
    return "console" in text or "win32" in text or "terminal" in text


def _term_width(default: int = 100) -> int:
    try:
        import shutil

        return max(60, shutil.get_terminal_size((default, 24)).columns)
    except Exception:
        return default


def _rule(width: int) -> str:
    """横线：Unicode 制表线优先，老终端降级为 ASCII。"""
    try:
        from .theme import glyph

        return glyph("rule") * max(0, width)
    except Exception:  # pragma: no cover
        return "-" * max(0, width)


def _fit(left: str, right: str, width: int, filler: str = "") -> str:
    """``╭─ 左 ───── 右 ─╮``：把 left 与 right 之间用横线铺满。"""
    try:
        from rich.cells import cell_len
    except Exception:  # pragma: no cover
        cell_len = len  # type: ignore[assignment]
    line = filler or _rule(1)
    pad = max(0, width - cell_len(left) - cell_len(right) - 1)
    return f"{left}{line * pad}{right}"


def box_frame(*, status: str = "", width: int = 0) -> str:
    """上边框：``╭─ 输入 ────── 状态 ─╮``。"""
    width = width or _term_width()
    rule = _rule(1)
    right = f" {status} {rule}╮" if status else f"{rule}╮"
    return _fit("╭─ 输入 ", right, width)


def box_bottom(*, hints: str = "", width: int = 0) -> str:
    """下边框：``╰─ 快捷键提示 ─────╯``（提示在左，横线铺到右端）。"""
    width = width or _term_width()
    rule = _rule(1)
    left = f"╰─ {hints} " if hints else "╰─ "
    return _fit(left, "╯", width)


def _prompt_markup(prompt_text: str, *, busy: bool, frame: str = "") -> Any:
    """多行提示符：上边框 + ``│ › ``（busy 时仍用同一标记，状态在边框右侧）。"""
    marker = prompt_text or "›"
    try:
        from prompt_toolkit.formatted_text import FormattedText

        chunks: list[tuple[str, str]] = []
        if frame:
            chunks.append(("class:frame", frame + "\n"))
        chunks.append(("class:frame", "│ "))
        chunks.append(("class:prompt", f"{marker} "))
        return FormattedText(chunks)
    except Exception:  # pragma: no cover
        return f"{frame}\n│ {marker} " if frame else f"{marker} "


def _fallback_read(prompt_text: str) -> str | None:
    """没有 prompt_toolkit 时的降级读取（管道/重定向场景）。

    注意：``prompt_text`` 为空时**不画任何标记**——否则在"转录直接接输入"的场景里
    会看到 ``› › 下一段内容`` 这种重复标记（实测）。
    """
    marker = f"{prompt_text} " if prompt_text else ""
    try:
        return input(marker)
    except EOFError:
        return None


def format_help() -> str:
    """``/help`` 文本。"""
    lines = ["可用命令（输入 / 会自动列出，↑↓ 选择、Enter 应用、Esc 取消）："]
    lines += [f"  {name:<12} {desc}" for name, desc in SLASH_COMMANDS]
    lines += [
        "",
        "快捷键：",
        "  Enter         发送（补全菜单开着时先应用候选）",
        "  Shift+Enter   换行（Alt+Enter / Ctrl+J 亦可）",
        "  Tab           补全路径 / 命令",
        "  @             引用工作区文件",
        "  Ctrl+O        展开或收起工具输出",
        "  Ctrl+R        反向搜索历史",
        "  Ctrl+L        清屏",
        "  Ctrl+C        清空当前输入（空行时退出）",
        "  Ctrl+D        退出",
    ]
    return "\n".join(lines)


__all__ = [
    "SLASH_COMMANDS",
    "AtFileCompleter",
    "MixedCompleter",
    "PromptInput",
    "SlashCommandCompleter",
    "SlashMenuCompleter",
    "available",
    "degrade_notice",
    "format_help",
    "import_error",
    "prompt_style",
    "workspace_file_provider",
]

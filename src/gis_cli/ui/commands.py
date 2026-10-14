# -*- coding: utf-8 -*-
"""斜杠命令注册表：命令的**唯一**数据源。

## 为什么要有它

以前命令信息散在两处：``ui/prompt.py:SLASH_COMMANDS``（菜单/帮助用的一张表）
和 ``agent/interactive.py:_handle_command``（一长串 if 链）。两边靠一条测试勉强
对齐，实际已经漂移过：``/color`` ``/session`` ``/resume`` 一度只存在于 if 链里，
菜单里根本选不到；``/skills`` 在菜单里能选、点下去却永远输出"（无）"。

这里把「命令名 / 说明 / 参数规格 / 处理器」收成一张表，菜单、``/help``、
分发全部从它生成，**加一条命令只改一个地方**。

## 参数补全

``args`` 描述参数的候选来源，菜单会跟着显示：``/theme <Tab>`` 出三个配色、
``/model <Tab>`` 出模型列表。候选可以是静态列表，也可以是取候选的函数
（函数在按下补全时才调用，避免启动就打网络）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

#: 参数候选：静态列表，或"取候选的函数"（惰性求值）
ArgChoices = Sequence[tuple[str, str]] | Callable[[], Sequence[tuple[str, str]]]

#: 快捷键说明（``/help`` 用；放这里避免再开一个模块）
KEY_HINT_LINES: tuple[str, ...] = (
    "  Enter         发送（补全菜单开着时先应用候选）",
    "  Shift+Enter   换行（Alt+Enter / Ctrl+J 亦可）",
    "  Tab           补全命令 / 参数 / 路径",
    "  ↑ ↓           菜单里选项；没有菜单时翻历史",
    "  @             引用工作区文件",
    "  Ctrl+O        展开或收起工具输出",
    "  Ctrl+R        反向搜索历史",
    "  Ctrl+L        清屏",
    "  Ctrl+C        清空当前输入（空行时退出）",
    "  Ctrl+D        退出",
)


@dataclass(frozen=True)
class CommandSpec:
    """一条斜杠命令。

    ``name`` 不带前导斜杠（``"model"``），对外展示时再补——
    否则会出现 ``//model`` 这种（实测踩过）。
    """

    name: str
    summary: str
    #: 用法示例（``/help`` 与参数提示用），空则只显示 ``/name``
    usage: str = ""
    #: 命令别名（都能命中，不单独出现在菜单里；同样不带斜杠）
    aliases: tuple[str, ...] = ()
    #: 参数候选：``(值, 说明)`` 列表或返回它的函数
    args: ArgChoices = ()
    #: 处理器：接收参数文本，返回 False 表示要退出会话
    handler: Callable[[str], bool] | None = field(default=None, compare=False)
    #: 是否在菜单里显示（别名/隐藏命令设 False）
    visible: bool = True

    @property
    def slash_name(self) -> str:
        return f"/{self.name.lstrip('/')}"

    @property
    def arg_hint(self) -> str:
        """参数占位提示（没有参数则为空）。"""
        return " <值>" if self.args else ""


def normalize_args(choices: ArgChoices) -> list[tuple[str, str]]:
    """把 ``args`` 求值成 ``(值, 说明)`` 列表（惰性函数在这里才调用）。"""
    if callable(choices):
        try:
            return list(choices())
        except Exception:
            return []
    return list(choices)


class CommandRegistry:
    """命令表：查找、菜单、帮助、参数补全都从这里出。"""

    def __init__(self, commands: Sequence[CommandSpec] = ()) -> None:
        self._commands: list[CommandSpec] = list(commands)

    def with_handlers(
        self,
        handlers: dict[str, Callable[[str], bool]],
        arg_providers: dict[str, Callable[[], Sequence[tuple[str, str]]]] | None = None,
    ) -> "CommandRegistry":
        """返回一份挂上处理器（和动态参数候选）的副本。

        设计：命令的**名字/说明/静态参数**只在 :data:`COMMANDS` 里写一次；
        处理器与"需要查运行时才能得到的候选"（模型清单、会话列表）由 agent 层注入，
        这样 ui 层不用反向依赖 agent。

        候选用**惰性函数**：网关探测有几秒超时，不能在每次打字时跑（会卡住输入）。
        """
        from dataclasses import replace

        providers = arg_providers or {}
        merged: list[CommandSpec] = []
        for spec in self._commands:
            key = spec.name.lstrip("/")
            handler = handlers.get(key)
            provider = providers.get(key)
            if handler is None and provider is None:
                merged.append(spec)
                continue
            merged.append(
                replace(
                    spec,
                    handler=handler if handler is not None else spec.handler,
                    args=provider if provider is not None else spec.args,
                )
            )
        return CommandRegistry(merged)

    # ------------------------------------------------------------- 维护

    def register(self, spec: CommandSpec) -> None:
        self._commands.append(spec)

    def extend(self, specs: Sequence[CommandSpec]) -> None:
        self._commands.extend(specs)

    def all(self) -> list[CommandSpec]:
        """全部命令（含隐藏的别名项）。"""
        return list(self._commands)

    def visible(self) -> list[CommandSpec]:
        """菜单里应显示的命令。"""
        return [c for c in self._commands if c.visible]

    # ------------------------------------------------------------- 查找

    def find(self, word: str) -> CommandSpec | None:
        """按名字或别名查找（大小写不敏感；``/help`` 与 ``help`` 都认）。"""
        key = (word or "").strip().lower().lstrip("/")
        if not key:
            return None
        for spec in self._commands:
            if key == spec.name.lower().lstrip("/"):
                return spec
            if key in (alias.lower().lstrip("/") for alias in spec.aliases):
                return spec
        return None

    # ------------------------------------------------------------- 输出

    def menu_items(self) -> list[tuple[str, str]]:
        """菜单候选项：``(/命令, 说明)``，说明里带参数提示。"""
        return [
            (spec.slash_name, spec.summary + spec.arg_hint) for spec in self.visible()
        ]

    def help_text(self, *, keys: Sequence[str] = ()) -> str:
        """``/help`` 文本（命令段由本表生成，快捷键段由调用方给）。"""
        shown = self.visible()
        width = max((len(spec.name.lstrip("/")) for spec in shown), default=8) + 1
        lines = ["可用命令（输入 / 会自动列出，↑↓ 选择、Enter 应用、Esc 取消）："]
        for spec in shown:
            name = spec.name.lstrip("/")
            lines.append(f"  {name:<{width}} {spec.summary}")
            if spec.usage:
                lines.append(f"  {' ' * width} 用法：{spec.usage}")
        if keys:
            lines += ["", "快捷键："]
            lines += list(keys)
        return "\n".join(lines)


__all__ = [
    "COMMANDS",
    "KEY_HINT_LINES",
    "ArgChoices",
    "CommandRegistry",
    "CommandSpec",
    "normalize_args",
]


#: 全部斜杠命令（**唯一数据源**：菜单、/help、分发都读它）。
#: 处理器在 agent 层用 ``CommandRegistry.with_handlers()`` 挂上。
COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("help", "显示帮助与快捷键"),
    CommandSpec("model", "查看或切换模型", usage="/model [模型名]", args=()),
    CommandSpec(
        "theme",
        "切换配色",
        usage="/theme <system|dark|light>",
        args=(("system", "跟随终端"), ("dark", "深色"), ("light", "浅色")),
    ),
    CommandSpec(
        "color",
        "开关颜色",
        usage="/color <on|off>",
        args=(("on", "强制上色"), ("off", "纯文本，不输出颜色码")),
    ),
    CommandSpec("workspace", "显示工作区", usage="/workspace"),
    CommandSpec("status", "显示引擎与数据目录状态"),
    CommandSpec("tools", "列出可用工具"),
    CommandSpec("skills", "列出可用技能（可用 /skill:名字 手动加载）"),
    CommandSpec("recipes", "列出内置配方"),
    CommandSpec("trace", "显示最近一次运行的追踪文件", usage="/trace [full]"),
    CommandSpec("session", "显示当前会话记录"),
    CommandSpec("resume", "切换到历史会话继续", usage="/resume [序号或 id]"),
    CommandSpec("clear", "清空屏幕"),
    CommandSpec("export", "导出本次会话转录"),
    CommandSpec("exit", "退出", aliases=("quit", "q")),
)

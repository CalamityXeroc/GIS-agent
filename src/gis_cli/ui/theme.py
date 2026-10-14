# -*- coding: utf-8 -*-
"""主题与 Console 工厂。

设计目标（对齐 pi / Crush 这类终端 agent 的观感）：
- 颜色按**角色**分配（提示符/助手/工具/成功/警告/错误/次要/强调/路径/数值），
  而不是在调用点随手写 ``[cyan]``；
- 支持三档模式：``system``（跟随终端明暗）/ ``dark`` / ``light``，可用环境变量覆盖；
- 尊重 ``NO_COLOR`` 与 ``--no-color``，在 GBK/老终端下自动降级（无颜色、无花哨边框）。

零新增依赖：只用 rich（ArcGIS Pro 环境自带 14.x）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.theme import Theme

ThemeMode = Literal["system", "dark", "light"]

#: 角色名 → 颜色。放在一处，调用点只写角色，便于统一调整。
_DARK = {
    "gis.prompt": "bold bright_cyan",
    "gis.agent": "bright_white",
    "gis.tool": "cyan",
    "gis.tool_name": "bold cyan",
    "gis.ok": "bold green",
    "gis.warn": "bold yellow",
    "gis.err": "bold red",
    "gis.dim": "grey58",
    "gis.accent": "bright_magenta",
    "gis.path": "bright_blue",
    "gis.num": "bold bright_yellow",
    "gis.status": "grey70",
    "gis.status.key": "grey50",
    "gis.panel.border": "grey35",
    "gis.panel.title": "bold bright_white",
    "gis.code": "grey70",
    # —— 消息角色（对齐主流 CLI：回答最亮、思考最暗、系统居中）——
    "gis.user": "bold bright_white",
    "gis.user.tag": "bold bright_cyan",
    "gis.thought": "grey62",
    "gis.thought.tag": "grey50",
    "gis.thought.bar": "grey35",
    "gis.answer": "bright_white",
    "gis.answer.tag": "bold bright_green",
    "gis.system": "grey58",
    "gis.tool.summary": "grey62",
}

_LIGHT = {
    "gis.prompt": "bold cyan",
    "gis.agent": "black",
    "gis.tool": "blue",
    "gis.tool_name": "bold blue",
    "gis.ok": "bold green4",
    "gis.warn": "bold dark_orange3",
    "gis.err": "bold red3",
    "gis.dim": "grey35",
    "gis.accent": "magenta",
    "gis.path": "blue",
    "gis.num": "bold dark_orange3",
    "gis.status": "grey30",
    "gis.status.key": "grey50",
    "gis.panel.border": "grey62",
    "gis.panel.title": "bold black",
    "gis.code": "grey30",
    "gis.user": "bold black",
    "gis.user.tag": "bold cyan",
    "gis.thought": "grey35",
    "gis.thought.tag": "grey50",
    "gis.thought.bar": "grey62",
    "gis.answer": "black",
    "gis.answer.tag": "bold green4",
    "gis.system": "grey30",
    "gis.tool.summary": "grey35",
}


def detect_theme_mode(explicit: str | None = None) -> Literal["dark", "light"]:
    """决定实际用深色还是浅色。

    优先级：显式参数 > ``GIS_AGENT_THEME`` > ``COLORFGBG`` > 默认 dark。
    ``COLORFGBG`` 形如 ``"15;0"``（前景;背景），背景 0-6 视为深色。
    """
    candidate = (explicit or os.environ.get("GIS_AGENT_THEME") or "").strip().lower()
    if candidate in ("dark", "light"):
        return candidate  # type: ignore[return-value]
    fgbg = os.environ.get("COLORFGBG", "").strip()
    if fgbg:
        try:
            bg = int(fgbg.split(";")[-1])
            return "dark" if bg <= 6 else "light"
        except Exception:
            pass
    return "dark"


def no_color_requested(explicit: bool = False) -> bool:
    """``NO_COLOR`` 约定（https://no-color.org/）或显式关闭。"""
    if explicit:
        return True
    return bool(os.environ.get("NO_COLOR"))


def build_theme(mode: str = "system") -> Theme:
    """按模式生成 rich Theme。``system`` 会走 :func:`detect_theme_mode`。"""
    resolved = detect_theme_mode(None if mode == "system" else mode)
    palette = _DARK if resolved == "dark" else _LIGHT
    return Theme(dict(palette))


def make_console(
    mode: str = "system",
    *,
    no_color: bool = False,
    color: bool | None = None,
    width: int | None = None,
    force_terminal: bool | None = None,
    stderr: bool = False,
) -> Console:
    """创建统一风格的 Console。

    颜色三态（实测教训：曾因误判终端支持 ANSI，用户屏幕上直接出现 ``?[38;5;246m``）：

    - ``color=None``（默认）：由 :func:`console_env.ansi_ok` 严格判定，
      **默认不上色**，只有明确认得的高级终端才自动上色；
    - ``color=True``：用户显式要求（``--color`` / ``/color on``），无论终端如何都上色；
    - ``color=False`` 或 ``no_color=True``：纯文本，不输出任何转义序列。

    符号集由 :func:`glyphs` 控制（Unicode 不可用时自动 ASCII）。
    """
    from .console_env import ansi_ok

    if color is True:
        want_color = True
    elif color is False:
        want_color = False
    else:
        want_color = ansi_ok()
    if no_color_requested(no_color):
        want_color = False
    return Console(
        theme=build_theme(mode),
        no_color=not want_color,
        color_system=("auto" if want_color else None),  # None = 完全不输出 ANSI 转义序列
        width=width,
        force_terminal=force_terminal,
        stderr=stderr,
        soft_wrap=False,
        highlight=False,
        emoji=box_ok(),
        markup=True,
    )



def engine_version() -> str:
    """版本号：优先仓库 pyproject（editable 安装的 dist-info 常常滞后），其次已安装元数据。"""
    candidates: list[str] = []
    try:
        import tomllib

        for base in Path(__file__).resolve().parents:
            pyproject = base / "pyproject.toml"
            if pyproject.exists():
                with pyproject.open("rb") as fh:
                    data = tomllib.load(fh)
                candidates.append(str((data.get("project") or {}).get("version") or ""))
                break
    except Exception:
        pass
    try:
        from importlib.metadata import version as _v

        candidates.append(_v("gis-task-cli"))
    except Exception:
        pass
    for item in candidates:
        if item:
            return item
    return ""

#: 符号集：Unicode 好看，ASCII 保底（老控制台/GBK 字体缺字形时用）
_GLYPHS_UNICODE = {
    "step": "⏺",
    "cont": "└",
    "ok": "✓",
    "bad": "✗",
    "warn": "!",
    "bullet": "•",
    "dot": "·",
    "rule": "─",
    "arrow": "→",
    "thought": "◆",
    "answer": "●",
    "user": "▸",
    "bar": "│",
}
_GLYPHS_ASCII = {
    "step": "*",
    "cont": "|",
    "ok": "v",
    "bad": "x",
    "warn": "!",
    "bullet": "-",
    "dot": ".",
    "rule": "-",
    "arrow": "->",
    "thought": "*",
    "answer": "o",
    "user": ">",
    "bar": "|",
}


def glyphs() -> dict[str, str]:
    """按当前控制台能力返回符号集（Unicode 或 ASCII 保底）。"""
    try:
        from .console_env import unicode_ok

        return dict(_GLYPHS_UNICODE if unicode_ok() else _GLYPHS_ASCII)
    except Exception:  # pragma: no cover
        return dict(_GLYPHS_UNICODE)


def glyph(name: str) -> str:
    """取单个符号（未知名称返回空串，不影响排版）。"""
    return glyphs().get(name, "")


def box_ok() -> bool:
    """是否适合画制表符边框（rich Panel）——ASCII 模式下改用简单分隔线。"""
    try:
        from .console_env import unicode_ok

        return bool(unicode_ok())
    except Exception:  # pragma: no cover
        return True


def theme_diagnostics() -> dict[str, Any]:
    """给 ``/status``、``/debug`` 用的自检信息。"""
    mode = detect_theme_mode()
    return {
        "mode": mode,
        "env_theme": os.environ.get("GIS_AGENT_THEME", ""),
        "colorfgbg": os.environ.get("COLORFGBG", ""),
        "no_color": no_color_requested(),
        "term": os.environ.get("TERM", ""),
        "wt_session": os.environ.get("WT_SESSION", ""),
    }


__all__ = [
    "ThemeMode",
    "box_ok",
    "build_theme",
    "detect_theme_mode",
    "engine_version",
    "glyph",
    "glyphs",
    "make_console",
    "no_color_requested",
    "theme_diagnostics",
]

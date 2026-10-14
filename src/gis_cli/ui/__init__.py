# -*- coding: utf-8 -*-
"""终端界面层（渲染/主题/输入）。

分层：
- :mod:`gis_cli.ui.theme`   主题与 Console 工厂（颜色按角色分配，支持 system/dark/light）
- :mod:`gis_cli.ui.events`  引擎事件 → 渲染单元（纯逻辑，可单测）+ 状态行累积器
- :mod:`gis_cli.ui.render`  渲染单元 → rich 组件（横幅 / 结果卡片 / 状态行）

约定：本包**不依赖** prompt_toolkit（输入框在 ``gis_cli.ui.prompt``，按需导入），
因此 TUI、纯文本输出、基准脚本可以共用同一套渲染逻辑。
"""

from __future__ import annotations

from .events import RenderUnit, StatusTracker, cleanup_error, describe_args, format_duration, format_tokens, to_unit
from .logging_setup import setup_engine_logging
from .render import ToolOutputStore, banner, footer_text, hint_line, render_unit, result_card
from .theme import build_theme, detect_theme_mode, make_console, no_color_requested, theme_diagnostics

__all__ = [
    "RenderUnit",
    "StatusTracker",
    "ToolOutputStore",
    "banner",
    "build_theme",
    "cleanup_error",
    "describe_args",
    "detect_theme_mode",
    "footer_text",
    "format_duration",
    "format_tokens",
    "hint_line",
    "make_console",
    "no_color_requested",
    "render_unit",
    "result_card",
    "setup_engine_logging",
    "theme_diagnostics",
    "to_unit",
]

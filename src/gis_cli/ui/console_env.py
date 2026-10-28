# -*- coding: utf-8 -*-
"""控制台环境自适应（中文 Windows 控制台的重点照顾对象）。

## 两次踩坑记录（决定了现在的策略）

1. **不要把 stdout 无条件重配置成 UTF-8**：代码页 936 的窗口里，Python 写出的 UTF-8
   字节会被按 GBK 解码 → 整屏乱码。
2. **不要靠"能不能开 VT"来推断终端会不会解析 ANSI**：实测有环境 SetConsoleMode
   返回成功、rich 于是输出 ``\\x1b[38;5;246m``，但终端（PowerShell/老 conhost/包装终端）
   根本不当转义处理 —— 屏幕上直接显示 ``?[38;5;246m``（用户口中的"246m"就是色号 246）。

## 现在的策略（宁可少色，不可乱码）

- **默认纯文本**：不输出任何 ANSI 转义序列。
- 只有在**明确认得**的高级终端里才自动上色（Windows Terminal / VS Code / ConEmu /
  mintty 等），判定依据是环境变量，而不是 VT 探测。
- 用户可显式覆盖：``--color`` 或 ``GIS_AGENT_FORCE_COLOR=1`` 强制上色；
  ``--no-color`` / ``NO_COLOR`` / ``GIS_AGENT_PLAIN=1`` 强制纯文本。
- 符号单独降级：代码页/字体不支持时用 ASCII（见 ``theme.glyphs``）。
"""

from __future__ import annotations

import os
import sys
from typing import Any

#: 明确支持 ANSI 的终端特征（值 → 家族名）
_MODERN_TERMINALS = {
    "WT_SESSION": "windows-terminal",
    "TERM_PROGRAM": "term-program",
    "ConEmuANSI": "conemu",
    "WEZTERM_EXECUTABLE": "wezterm",
    "ALACRITTY_WINDOW_ID": "alacritty",
    "KITTY_WINDOW_ID": "kitty",
    "VSCODE_INJECTION": "vscode",
    "GIT_SSH_COMMAND": "",  # 占位，避免误判
}

_VT_STATE: bool | None = None


def _is_windows() -> bool:
    return os.name == "nt"


def output_codepage() -> int:
    """控制台输出代码页（非 Windows 或取不到时返回 0）。"""
    if not _is_windows():
        return 0
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetConsoleOutputCP())
    except Exception:
        return 0


def _set_codepage(cp: int) -> bool:
    if not _is_windows():
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        ok_out = bool(kernel32.SetConsoleOutputCP(cp))
        try:
            kernel32.SetConsoleCP(cp)
        except Exception:
            pass
        return ok_out
    except Exception:
        return False


def ensure_utf8_console(preferred: int = 65001) -> bool:
    """尽量把控制台切到 UTF-8；返回切换后是否处于 UTF-8。"""
    if not _is_windows():
        return True
    current = output_codepage()
    if current == preferred:
        return True
    if _set_codepage(preferred):
        return output_codepage() == preferred
    return output_codepage() == preferred


def console_encoding() -> str:
    """控制台"真正会用"的编码名（用于配置 Python 流）。"""
    if not _is_windows():
        return (os.environ.get("PYTHONIOENCODING") or "utf-8").split(":")[0] or "utf-8"
    cp = output_codepage()
    if cp == 65001:
        return "utf-8"
    if cp == 936:
        return "gbk"
    if cp:
        return f"cp{cp}"
    try:
        import ctypes

        return f"cp{int(ctypes.windll.kernel32.GetACP())}"
    except Exception:
        return "utf-8"


def terminal_family() -> str:
    """识别终端家族（用于决定要不要上色）。"""
    if os.environ.get("WT_SESSION"):
        return "windows-terminal"
    if os.environ.get("TERM_PROGRAM"):
        return f"term:{os.environ['TERM_PROGRAM']}"
    if str(os.environ.get("ConEmuANSI", "")).upper() == "ON":
        return "conemu"
    if os.environ.get("WEZTERM_EXECUTABLE"):
        return "wezterm"
    if os.environ.get("ALACRITTY_WINDOW_ID"):
        return "alacritty"
    if os.environ.get("KITTY_WINDOW_ID"):
        return "kitty"
    if os.environ.get("VSCODE_INJECTION"):
        return "vscode"
    if os.environ.get("TERM"):
        return f"term:{os.environ['TERM']}"
    if _is_windows():
        return "windows-console"
    return "unknown"


def stdout_is_tty() -> bool:
    """当前 stdout 是不是终端。

    ``rich`` 的 ``Live``/``status()`` 会把 ``sys.stdout`` 换成 ``FileProxy``（退出
    ``with`` 才恢复），而它继承 ``io.TextIOBase``、``isatty()`` 恒为 False。不先解
    代理，凡是在 Live 内部判定的“这是不是终端”都会被带偏（DAIR-37/38）。

    这是**唯一**一份实现：``ui/prompt.py`` 与 ``ui/plan_card.py`` 都转调这里，
    免得日后只修一边。
    """
    try:
        stream = getattr(sys.stdout, "rich_proxied_file", sys.stdout)
        return bool(stream.isatty())
    except Exception:
        return False


def stdin_is_tty() -> bool:
    """当前 stdin 是不是终端。"""
    try:
        return bool(sys.stdin.isatty())
    except Exception:  # pragma: no cover - 极端环境
        return False


def interactive_terminal() -> bool:
    """能不能真的停下来问人：stdin **和** stdout 都是终端。

    只看 stdin 会漏掉 ``gis-agent loop "…" | tee run.log`` 这类用法：提示进了管道，
    屏幕上什么都没有，人不知道要按 Enter，进程就默默卡在那里。
    """
    return stdin_is_tty() and stdout_is_tty()


def force_color_requested() -> bool:
    """用户显式要求上色（``--color`` 会写这个环境变量 / 直接传参）。"""
    return str(os.environ.get("GIS_AGENT_FORCE_COLOR", "")).strip().lower() in ("1", "true", "yes", "on")


def plain_requested() -> bool:
    """用户显式要求纯文本。"""
    if os.environ.get("NO_COLOR"):
        return True
    return str(os.environ.get("GIS_AGENT_PLAIN", "")).strip().lower() in ("1", "true", "yes", "on")


def ansi_ok() -> bool:
    """当前是否可以安全输出 ANSI 颜色码。

    策略：**默认否**；只有"明确认得的高级终端 + 是 TTY"才自动为真，
    或者用户显式 ``GIS_AGENT_FORCE_COLOR=1`` 强制。
    """
    if plain_requested():
        return False
    if not stdout_is_tty():
        return False
    if force_color_requested():
        return True
    family = terminal_family()
    if family == "windows-console":
        # 普通 cmd / PowerShell 窗口：默认不上色（实测会被显示成 ?[38;5;..m）
        return False
    if family in ("unknown", "term:dumb", ""):
        return False
    return True


def _reconfigure(stream: Any, encoding: str) -> None:
    try:
        stream.reconfigure(encoding=encoding, errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def configure_stdio(*, force_utf8: bool = False) -> dict[str, Any]:
    """让 Python 的输出编码与控制台一致（并尽量把控制台切到 UTF-8）。"""
    switched = False
    if force_utf8 or _is_windows():
        switched = ensure_utf8_console()
    encoding = console_encoding()
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            _reconfigure(stream, encoding)
    stdin = getattr(sys, "stdin", None)
    if stdin is not None:
        try:
            stdin.reconfigure(encoding=encoding, errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    return {
        "codepage": output_codepage(),
        "encoding": encoding,
        "switched_to_utf8": switched,
        "unicode_ok": unicode_ok(),
        "ansi_ok": ansi_ok(),
        "terminal": terminal_family(),
    }


def unicode_ok() -> bool:
    """当前环境能否安全输出 ``⏺ ─ ✓`` 这类符号（字体与代码页两方面）。"""
    if str(os.environ.get("GIS_AGENT_ASCII", "")).strip().lower() in ("1", "true", "yes"):
        return False
    if not _is_windows():
        return True
    cp = output_codepage()
    if cp == 65001:
        return True
    if cp == 936:
        # GBK 能显示 ─ ✓ 等常见符号，实测用户终端正常；仅对缺字形风险高的项降级
        return True
    if not cp:
        return (getattr(sys.stdout, "encoding", "") or "").lower().startswith("utf")
    return False


def describe() -> str:
    """一行诊断串，便于 ``/status`` 与排障。"""
    return (
        f"控制台 {terminal_family()} · 代码页 {output_codepage() or '-'}"
        f" · 输出编码 {getattr(sys.stdout, 'encoding', '?') or '?'}"
        f" · 颜色 {'on' if ansi_ok() else 'off'}"
        f" · 符号 {'unicode' if unicode_ok() else 'ascii'}"
    )


__all__ = [
    "ansi_ok",
    "configure_stdio",
    "console_encoding",
    "describe",
    "ensure_utf8_console",
    "force_color_requested",
    "output_codepage",
    "plain_requested",
    "terminal_family",
    "unicode_ok",
]

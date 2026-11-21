"""``--new-window``：把这次调用丢进一个新终端窗口。

## 为什么不写进 typer

用户要的是 ``giscli --new-window chat``：开关出现在**任何子命令前面或后面**
都要认。typer 只解析它注册过的参数，为了一个纯启动层的能力去改每个子命令的
签名（还把 ``--new-window`` 混进 ``--help`` 的业务选项里）并不划算。所以在
:func:`gis_cli.agent.cli.main` 里先把它摘掉，再把剩下的 argv 原样交给 typer——
子命令的解析规则一行都不用动。

## 两种开窗方式

1. 有 ``wt.exe``（Windows Terminal）：``wt -d <cwd> cmd /k giscli …``
2. 没有：``cmd /c start "" /D <cwd> cmd /k giscli …``

两者都显式带 ``/D``/``-d`` 指定工作目录，新窗口的 cwd 与当前一致（``pi`` 的
默认语义是占当前终端，这里只多给一个可选的新窗口）。``cmd /k`` 让窗口在
命令结束后不自动关闭，报错才看得见。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

#: 触发「新窗口」的开关（本模块独占，不注册进 typer）
NEW_WINDOW_FLAG = "--new-window"
#: 只打印将要执行的命令行，不真的开窗（测试与排障用）
DRY_RUN_FLAG = "--dry-run"
#: 新窗口里执行的命令名：PATH 上的 shim 自己（scripts\\giscli.cmd）
LAUNCHER_NAME = "giscli"
#: Windows Terminal 的固定安装位置（PATH 上不一定有）
_WT_RELATIVE = Path("Microsoft") / "WindowsApps" / "wt.exe"


def split_flags(argv: Sequence[str]) -> tuple[list[str], bool, bool]:
    """摘掉本层的开关，返回 ``(剩余 argv, 是否新窗口, 是否 dry-run)``。

    ``--dry-run`` **只在同时给了** ``--new-window`` 时才被摘掉：``resume-run``
    自己也有 ``--dry-run``（只看有没有断点），无条件摘会把它吃掉。
    """
    rest = [arg for arg in argv if arg != NEW_WINDOW_FLAG]
    new_window = len(rest) != len(argv)
    if not new_window:
        return rest, False, False
    dry_run = DRY_RUN_FLAG in rest
    if dry_run:
        rest = [arg for arg in rest if arg != DRY_RUN_FLAG]
    return rest, True, dry_run


def find_window_host(which=shutil.which, environ=None) -> str | None:
    """找 ``wt.exe``：先看 PATH，再看 WindowsApps 的固定位置。"""
    found = which("wt")
    if found:
        return found
    env = os.environ if environ is None else environ
    local = (env.get("LOCALAPPDATA") or "").strip()
    if not local:
        return None
    candidate = Path(local) / _WT_RELATIVE
    return str(candidate) if candidate.is_file() else None


#: cmd.exe 会当语法解释的字符：出现任何一个就必须整段包进双引号
_CMD_UNSAFE = frozenset('"&|^<>()')


def cmd_quote(part: str) -> str:
    """把一段文本转义成 cmd 命令行里的一个词（能原样粘回终端）。

    Python 的 ``list2cmdline`` 只在有空白时加引号，``a&whoami`` 这种参数会
    原样落进 cmd，``&`` 把命令行截断成两条（实测 ``cmd /c echo a&whoami``
    会去执行 whoami）。所以进 cmd 的每段自己先包好：含空白或 cmd 元字符就
    整段双引号包裹，内部 ``"`` 按 cmd 的规则写成 ``""``。
    """
    if part and not any(ch in _CMD_UNSAFE or ch.isspace() for ch in part):
        return part
    return '"' + part.replace('"', '""') + '"'


def build_new_window_command(
    argv: Sequence[str],
    *,
    cwd: Path | str,
    window_host: str | None,
    launcher: str = LAUNCHER_NAME,
) -> list[str]:
    """拼出开窗命令行。纯函数（不碰环境、不启动进程）以便断言。

    新窗口里真正解析这行的是 **cmd.exe**，不是 Python，所以每一段（argv、
    cwd、launcher、window_host）都得先过 :func:`cmd_quote`——否则
    ``--workspace a&whoami`` 里的 ``&`` 会被 cmd 当成第二条命令。
    """
    cwd_text = cmd_quote(str(cwd))
    tail = [cmd_quote(launcher), *(cmd_quote(arg) for arg in argv)]
    if window_host:
        return [cmd_quote(window_host), "-d", cwd_text, "cmd", "/k", *tail]
    # ``start`` 的第一个带引号参数是**窗口标题**，不写就会把 ``/D`` 当标题吞掉，
    # 所以这个 ``""`` 是必需的（不能省成空串——format_command 只加空格）。
    return ["cmd", "/c", "start", '""', "/D", cwd_text, "cmd", "/k", *tail]


def format_command(command: Sequence[str]) -> str:
    """命令行 → 可直接粘回终端的字符串。

    各段在 :func:`build_new_window_command` 里已经转义过，这里只加空格：
    再包一层引号会把 ``"a&whoami"`` 变成 cmd 不认的 ``\"a&whoami\"``。
    """
    return " ".join(command)


def relaunch(argv: Sequence[str]) -> int:
    """``--new-window`` 的执行体：dry-run 只打印，其余开窗并返回 0/1。"""
    rest, new_window, dry_run = split_flags(argv)
    if not new_window:  # 调用方保证会进这里，兜一下避免静默开窗
        return 2
    command = build_new_window_command(
        rest, cwd=Path.cwd(), window_host=find_window_host()
    )
    text = format_command(command)
    if dry_run:
        print(text)
        return 0
    try:
        # 给整串而不是 list：``Popen(list)`` 会让 list2cmdline 再包一层引号，
        # 把上面手写的 ``"a&whoami"`` 变成 cmd 看不懂的 ``\"a&whoami\"``。
        subprocess.Popen(text)
    except OSError as exc:
        # 开窗失败不能静默：至少把可手动执行的那行交给用户
        print(f"打开新窗口失败：{exc}", file=sys.stderr)
        print(f"可手动执行：{text}", file=sys.stderr)
        return 1
    return 0

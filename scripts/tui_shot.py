# -*- coding: utf-8 -*-
"""抓一帧输入框界面并断言四边完整（终端模拟）。

为什么需要它：管道/重定向场景走**降级路径**（根本不画框），"跑一遍截图"
看到的是降级输出；单测也看不到"输入时是否真的弹菜单"（那在 session/布局配置里）。
实测就因为缺少这类验证，漏掉过两个 bug：下边框被 hidden（终端不支持 CPR
时 prompt_toolkit 会藏起 bottom_toolbar）、以及输入 `/` 不弹菜单
（enable_history_search 与 complete_while_typing 互斥）。

用法（Git Bash 下务必带 MSYS_NO_PATHCONV，否则 `--text "/"` 会被转成 `E:/Git/`）：
    MSYS_NO_PATHCONV=1 python scripts/tui_shot.py --text "/"
    MSYS_NO_PATHCONV=1 python scripts/tui_shot.py --text "/" --assert
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def capture(
    *,
    text: str = "",
    rows: int = 22,
    cols: int = 96,
    status: str = "deepseek-v4.1-flash · 空闲 · 未开始 · 0s · ↑0 ↓0",
    wait: float = 1.8,
) -> tuple[str, dict]:
    """真跑一次输入，返回 (屏幕文本, 诊断信息)。"""
    import pyte
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output

    from gis_cli.ui import prompt as ui_prompt

    ui_prompt._interactive_tty = lambda: True  # 模拟终端，强制走真交互路径

    class _Sink:
        def __init__(self) -> None:
            self.chunks: list[str] = []

        def write(self, data: str) -> None:
            self.chunks.append(data)

        def flush(self) -> None:
            pass

        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            raise OSError("sink has no fileno")

    sink = _Sink()
    diag: dict = {}

    with create_pipe_input() as pipe:
        output = Vt100_Output(
            sink, lambda: Size(rows=rows, columns=cols), term="xterm-256color"
        )
        reader = ui_prompt.PromptInput(
            Path(ROOT),
            mode="dark",
            boxed=True,
            input=pipe,
            output=output,
            registry=_demo_registry(),
        )

        def _run() -> None:
            try:
                diag["value"] = reader.read_line("", busy=False, status=status)
            except BaseException as exc:  # pragma: no cover - 诊断用
                diag["error"] = repr(exc)

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

        # 等应用真的接管输入（buffer 建好）再发按键；否则按键会被丢掉，
        # 屏幕上会出现上一个用例的残留（实测过，很难查）。
        deadline = time.time() + max(wait, 5.0)
        while time.time() < deadline:
            time.sleep(0.15)
            if _find_buffer(reader) is not None:
                break

        if text:
            pipe.send_text(text)
            for _ in range(20):  # 等补全线程填好 complete_state
                time.sleep(0.2)
                buf = _find_buffer(reader)
                if buf is not None and buf.complete_state is not None:
                    break
            time.sleep(0.4)
        buf = _find_buffer(reader)
        if buf is not None:
            diag["buffer"] = buf.text
            state = buf.complete_state
            diag["menu_open"] = state is not None
            diag["candidates"] = [c.text for c in state.completions[:8]] if state else []

        screen = pyte.Screen(cols, rows)
        pyte.ByteStream(screen).feed("".join(sink.chunks).encode("utf-8", "replace"))
        lines = [line.rstrip() for line in screen.display]
        lines = [line for line in lines if "cursor position requests" not in line]
        while lines and not lines[-1]:
            lines.pop()

        pipe.send_text("\r")  # 让 read_line 返回，线程退出
        worker.join(timeout=5.0)

    return "\n".join(lines), diag


def _demo_registry() -> object:
    """截图用的命令注册表：只需参数候选（不需要真会话/引擎）。

    不注入 registry 时 ``/theme `` 这类参数菜单不会出现（实测漏过），
    所以截图要和真实使用一致。
    """
    from gis_cli.ui.commands import COMMANDS, CommandRegistry

    return CommandRegistry(COMMANDS)


def _find_buffer(reader) -> object | None:
    """从 PromptInput 里拿到当前生效的输入 buffer。

    自建输入框把 buffer 放在 ``_buffer``；兼容旧路径的 ``PromptSession``。
    注意：不能在 app 跑起来之前访问（那时还没 build）。
    """
    buf = getattr(reader, "_buffer", None)
    if buf is not None:
        return buf
    session = getattr(reader, "_session", None)
    if session is not None:
        return getattr(session, "default_buffer", None) or getattr(session, "buffer", None)
    return None


def check(screen: str, diag: dict) -> list[str]:
    """断言布局：两条纯线 + 框下状态行 + 菜单；返回问题列表（空 = 全过）。"""
    problems: list[str] = []
    lines = [line for line in screen.splitlines() if line.strip()]
    joined = "\n".join(lines)

    if not lines:
        return ["屏幕完全是空的"]
    if "╭" not in joined and "+" not in joined:
        problems.append("缺上边框")
    if "╰" not in joined and "+" not in joined:
        problems.append("缺下边框")
    if "│" not in joined and "|" not in joined:
        problems.append("缺左/右竖线")

    # 上下边框必须是**纯线**：不能嵌提示文字（用户要求全删）
    for row in lines:
        if not ("╭" in row or "╰" in row):
            continue
        for word in ("Enter", "Shift+Enter", "Ctrl+", "Esc", "命令 ·", "输入"):
            if word in row:
                problems.append(f"边框里残留提示文字 {word!r}：{row.strip()[:40]}")
                break

    # 竖线必须出现在输入行与菜单行上（菜单是浮层，最容易缺边栏）
    border_rows = [line for line in lines if "╭" in line or "╰" in line]
    inner_rows = [line for line in lines if line not in border_rows]
    if inner_rows and not any("│" in row for row in inner_rows):
        problems.append("输入/菜单行没有竖线（框是断的）")

    # 框下必须有状态行（模型名与 token 计数）
    if border_rows:
        bottom_index = max(lines.index(row) for row in border_rows if "╰" in row)
        below = lines[bottom_index + 1 :]
        if not below:
            problems.append("框下没有状态行")
        elif not any("↓" in row or "↑" in row for row in below):
            problems.append(f"框下行不像状态行：{below[0].strip()[:40]!r}")

    text = diag.get("buffer")
    if text and text.strip().startswith("/"):
        if not diag.get("menu_open"):
            problems.append(f"输入 {text!r} 没有弹出补全菜单")
        elif not diag.get("candidates"):
            problems.append("菜单打开了但没有候选")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="/", help="要输入的文本（注意 MSYS 路径转换）")
    ap.add_argument("--stdin-text", action="store_true", help="从 stdin 读文本，避开 MSYS")
    ap.add_argument("--status", default="deepseek-v4.1-flash · 空闲 · 未开始 · 0s · ↑0 ↓0")
    ap.add_argument("--rows", type=int, default=22)
    ap.add_argument("--cols", type=int, default=96)
    ap.add_argument("--assert", dest="do_assert", action="store_true", help="断言四边完整")
    args = ap.parse_args()

    text = sys.stdin.readline().rstrip("\n") if args.stdin_text else args.text

    screen, diag = capture(text=text, rows=args.rows, cols=args.cols, status=args.status)
    print(
        f"[诊断] 输入={text!r} buffer={diag.get('buffer')!r} "
        f"menu={diag.get('menu_open')} candidates={diag.get('candidates')}",
        file=sys.stderr,
    )
    print(screen)

    if args.do_assert:
        problems = check(screen, diag)
        if problems:
            print("\n[断言失败]", file=sys.stderr)
            for item in problems:
                print(f"  ✗ {item}", file=sys.stderr)
            return 1
        print("\n[断言通过] 四边完整、菜单正常", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

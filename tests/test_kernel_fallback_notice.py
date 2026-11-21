# -*- coding: utf-8 -*-
"""内核起不来时终端要有一行提示 + 横幅不得再谎报 ready（DAIR-320）。

真机现象（DAIR-318 第 4 项，`benchmark/reports/dair318_20261119/rev2/kernelfail4.1_after.screen.txt`）：

- 内核 16~20s 内死掉，``.gis_agent/logs/kernel.log`` 里有完整 traceback；
- 终端上一个字都没有（提示只写进 ``engine.log``）；
- 横幅全程写着 ``内核  ready``。

根因是 ``CodeRunner._ensure_kernel`` 捕获 ``KernelError`` 后只 ``logger.warning``
就回退，而横幅只看“执行器在不在”。这里锁住三件事：一行带 ``kernel.log`` 路径的
提示、只在状态变化时触发一次、横幅报实际状态。
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from gis_cli.runtime import code_runner as runner_mod
from gis_cli.runtime.code_runner import CodeRunner
from gis_cli.runtime.kernel import KernelError
from gis_cli.ui import wiring


class _BrokenKernel:
    """start() 必定失败的内核替身（与真 ArcpyKernel 一样先开日志再起子进程）。"""

    def __init__(self, *, workspace: str | None = None, **_kwargs) -> None:
        self.log_path: Path | None = Path(workspace) / ".gis_agent" / "logs" / "kernel.log" if workspace else None

    def start(self) -> None:
        raise KernelError("Kernel died before replying to kernel_info")

    def is_alive(self) -> bool:
        return False

    def shutdown(self) -> None:
        return None


class _ReadyKernel(_BrokenKernel):
    def start(self) -> None:
        return None

    def is_alive(self) -> bool:
        return True


def _runner(tmp_path: Path) -> CodeRunner:
    return CodeRunner(workspace=tmp_path, state_dir=tmp_path / "kernel", python_executable="python")


def test_start_failure_reports_once_with_kernel_log_path(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "ArcpyKernel", _BrokenKernel)
    runner = _runner(tmp_path)
    notices: list[str] = []
    runner.on_kernel_state_change = lambda updated: notices.append(updated.kernel_notice)

    assert runner._ensure_kernel() is None, "起不来就必须回退"
    assert runner.kernel_state == "unavailable"
    assert len(notices) == 1, "状态变化只通知一次"
    log_path = tmp_path / ".gis_agent" / "logs" / "kernel.log"
    assert str(log_path) in notices[0], "提示里要能直接看到 kernel.log 路径"
    assert "kernel.log" in notices[0] and "一次性子进程" in notices[0]

    # 后续每次回退都再打一遍 = 刷屏；不应重复通知
    assert runner._ensure_kernel() is None
    assert len(notices) == 1


def test_no_log_path_falls_back_to_error_message(tmp_path, monkeypatch):
    """日志打不开时（比如缺 jupyter_client）提示退化成异常本身，仍有可操作信息。"""

    class _NoLog(_BrokenKernel):
        def __init__(self, **_kwargs) -> None:
            self.log_path = None

    monkeypatch.setattr(runner_mod, "ArcpyKernel", _NoLog)
    runner = _runner(tmp_path)
    runner._ensure_kernel()

    assert runner.kernel_log_path is None
    assert "Kernel died before replying to kernel_info" in runner.kernel_notice


def test_ready_kernel_is_reported_as_ready_and_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "ArcpyKernel", _ReadyKernel)
    runner = _runner(tmp_path)
    notified: list[object] = []
    runner.on_kernel_state_change = notified.append

    assert runner._ensure_kernel() is not None
    assert runner.kernel_state == "ready"
    assert runner.kernel_notice == ""
    assert notified == [], "起来了就不该打扰用户"
    assert wiring.kernel_state_label(runner) == "就绪"


def test_banner_reports_actual_kernel_state(tmp_path, monkeypatch):
    """横幅以前只看 code_runner 在不在就写 ready（DAIR-318 的判定依据）。"""
    monkeypatch.setattr(runner_mod, "ArcpyKernel", _BrokenKernel)
    runner = _runner(tmp_path)
    bundle = SimpleNamespace(code_runner=runner)

    assert wiring.engine_meta(bundle)["kernel"] == "按需启动", "还没起内核时不能写 ready"
    runner._ensure_kernel()
    assert wiring.engine_meta(bundle)["kernel"] == "不可用（已回退）"


def test_chat_prints_kernel_notice_only_once(tmp_path):
    """交互会话把提示打到转录里，且同一条提示只出现一次。"""
    pytest.importorskip("prompt_toolkit", reason="prompt_toolkit 未安装（用 ArcGIS 解释器可跑）")
    from gis_cli.agent import interactive as interactive_mod

    chat = interactive_mod.InteractiveChat(workspace=tmp_path, no_color=True)
    buffer = io.StringIO()
    chat.console.file = buffer

    notice = (
        "内核不可用，已回退一次性子进程（每步约 15s）。"
        f"traceback 见 {tmp_path / '.gis_agent' / 'logs' / 'kernel.log'}"
    )
    runner = SimpleNamespace(kernel_notice=notice)
    chat._on_kernel_state_change(runner)
    chat._on_kernel_state_change(runner)

    text = buffer.getvalue()
    assert text.count("内核不可用") == 1, "同一条提示不得重复打印"
    assert "kernel.log" in text


def test_chat_build_attaches_kernel_state_hook(monkeypatch, tmp_path):
    """``_build()`` 必须把回调挂到执行器上，否则提示依旧只进日志。"""
    pytest.importorskip("prompt_toolkit", reason="prompt_toolkit 未安装（用 ArcGIS 解释器可跑）")
    from gis_cli.agent import interactive as interactive_mod

    class _Runner:
        on_kernel_state_change = None

    class _Loop:
        recorder = None
        state = None

        class config:  # noqa: N801 - 与真 bundle 的属性形状一致
            max_turns = 60

    bundle = SimpleNamespace(code_runner=_Runner(), loop=_Loop(), llm=None, catalog=None, recipes=None)
    monkeypatch.setattr("gis_cli.engine.runner.build_engine", lambda **_kwargs: bundle)

    chat = interactive_mod.InteractiveChat(workspace=tmp_path, no_color=True)
    assert chat._build() == 0
    assert bundle.code_runner.on_kernel_state_change == chat._on_kernel_state_change


def test_chat_run_prints_early_notice_once_after_banner(monkeypatch, tmp_path):
    """``--refresh-catalog`` 那种“启动阶段内核就已经失败”也要在终端看到提示。

    这时钩子还没挂上（``build_engine`` 里扫描目录就起了内核），提示是先备在
    执行器里的；``run()`` 打印横幅后补打一次，且只打一次。
    """
    pytest.importorskip("prompt_toolkit", reason="prompt_toolkit 未安装（用 ArcGIS 解释器可跑）")
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from rich.console import Console as RichConsole

    from gis_cli.agent import interactive as interactive_mod
    from gis_cli.ui import prompt as ui_prompt
    from gis_cli.ui import theme as ui_theme

    buffer = io.StringIO()
    monkeypatch.setattr(
        ui_theme,
        "make_console",
        lambda *_a, **_k: RichConsole(
            # 足够宽：这里验的是“打了几次/顺序”，不是换行（超宽路径会被折行）
            file=buffer,
            width=200,
            no_color=True,
            theme=ui_theme.build_theme("dark"),
        ),
    )
    monkeypatch.setattr(ui_prompt, "_interactive_tty", lambda: True)

    log_path = tmp_path / ".gis_agent" / "logs" / "kernel.log"
    notice = f"内核不可用，已回退一次性子进程（每步约 15s）。traceback 见 {log_path}"

    class _Runner:
        on_kernel_state_change = None
        kernel_state = "unavailable"
        kernel_notice = notice
        kernel_log_path = log_path

    class _Loop:
        recorder = None
        state = None

        class config:  # noqa: N801 - 与真 bundle 的属性形状一致
            max_turns = 60

    class _Bundle:
        code_runner = _Runner()
        loop = _Loop()
        llm = None
        catalog = None
        recipes = None

        def close(self) -> None:
            return None

    monkeypatch.setattr("gis_cli.engine.runner.build_engine", lambda **_kwargs: _Bundle())

    with create_pipe_input() as pipe:
        pipe.send_text("/exit\r")
        chat = interactive_mod.InteractiveChat(
            workspace=tmp_path, no_color=True, prompt_input=pipe, prompt_output=DummyOutput()
        )
        assert chat.run("") == 0

    text = buffer.getvalue()
    assert text.count("内核不可用") == 1, "同一提示只能出现一次"
    assert str(log_path) in text
    assert "不可用（已回退）" in text, "横幅要报实际状态"
    assert text.index("内核不可用") > text.index("GIS Agent"), "提示在横幅之后"


def test_status_line_says_kernel_unavailable(tmp_path):
    """横幅是静态的，内核挂掉后状态行要一直说真话。"""
    pytest.importorskip("prompt_toolkit", reason="prompt_toolkit 未安装（用 ArcGIS 解释器可跑）")
    from gis_cli.agent import interactive as interactive_mod

    chat = interactive_mod.InteractiveChat(workspace=tmp_path, no_color=True)
    chat.bundle = SimpleNamespace(code_runner=SimpleNamespace(kernel_state="ready"))
    assert "内核不可用" not in chat._status_text()

    chat.bundle = SimpleNamespace(code_runner=SimpleNamespace(kernel_state="unavailable"))
    assert "内核不可用" in chat._status_text()

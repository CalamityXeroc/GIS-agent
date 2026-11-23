# -*- coding: utf-8 -*-
"""CLI 引擎路径的转录测试：用假引擎喂事件，检查界面输出。

不依赖真实 LLM/ArcGIS：monkeypatch ``build_engine`` 返回假 bundle，
并通过它捕获 CLI 传入的 ``on_event`` 回调，模拟一轮真实工作流。
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console as RichConsole

from gis_cli.agent import cli as agent_cli
from gis_cli.ui import theme as ui_theme


class FakeConsole:
    """把 rich renderable 渲染成纯文本（宽 100、无颜色），便于断言。"""

    def __init__(self) -> None:
        self.buffer = io.StringIO()
        self.width = 100
        self._console = RichConsole(
            file=self.buffer,
            width=self.width,
            no_color=True,
            force_terminal=False,
            highlight=False,
            theme=ui_theme.build_theme("dark"),
        )

    def print(self, *args: Any, **_kwargs: Any) -> None:
        self._console.print(*args)

    def status(self, *_args: Any, **_kwargs: Any):
        console = self

        class _Status:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *_exc):
                return False

            def stop(self_inner):
                return None

            def update(self_inner, *_a, **_k):
                return None

        console._console.print("（status…）")
        return _Status()

    @property
    def text(self) -> str:
        return self.buffer.getvalue()


class FakeState:
    def __init__(self, status: str = "completed") -> None:
        self.status = status
        self.turn = 3
        self.artifacts = ["output/成果图/密度_1982_1986.jpg"]
        self.tasklist = [
            {"id": "t1", "title": "四期核密度", "status": "done"},
            {"id": "t2", "title": "统一图例出图", "status": "done"},
        ]
        self.methodology = "核密度：邻域 100km² → 半径 √(A/π)≈5642m"
        self.final_summary = "已交付 4 张密度图"
        self.error = ""
        self.requirements: list[Any] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "turn": self.turn,
            "artifacts": list(self.artifacts),
            "tasklist": [dict(item) for item in self.tasklist],
            "methodology": self.methodology,
            "final_summary": self.final_summary,
            "error": self.error,
        }


class FakeRecorder:
    jsonl_path = Path("workspace/.gis_agent/traces/run_test.jsonl")


class FakeLoop:
    def __init__(self, on_event) -> None:
        self._on_event = on_event
        self.recorder = FakeRecorder()
        self.state = type("S", (), {"run_id": "run_test"})()

        class _Cfg:
            max_turns = 60

        self.config = _Cfg()

    def run(self, task: str) -> FakeState:
        emit = self._on_event or (lambda *_: None)
        emit("run_start", {"goal": task, "run_id": "run_test"})
        emit("turn", {"turn": 1})
        emit("llm_result", {"model": "glm-5.3-flash", "actions": [{"tool": "run_recipe"}], "usage": {"prompt_tokens": 1200, "completion_tokens": 300}})
        emit("tool_call", {"tool": "run_recipe", "args": {"recipe_id": "kernel_density", "cell_size": 100}})
        emit("tool_result", {"tool": "run_recipe", "ok": True, "summary": "4 期核密度 → output/result_data.gdb/kde_*"})
        emit("tool_call", {"tool": "execute_code", "args": {"code": "print(1)"}})
        emit("tool_result", {"tool": "execute_code", "ok": False, "summary": "NameError: name 'dem' is not defined"})
        emit("repair_attempt", {"attempt": 1, "error": "NameError: name 'dem' is not defined"})
        emit("verification", {"pass": True, "summary": "验收通过（7/7 断言）"})
        return FakeState()


class FakeBundle:
    def __init__(self, on_event) -> None:
        self.loop = FakeLoop(on_event)
        self.llm = type("L", (), {"config": type("C", (), {"model": "glm-5.3-flash", "api_base": "https://tokenrhythm.studio/v1"})()})()
        self.catalog = type("Cat", (), {"counts": lambda self_: {"vector": 1, "table": 1}})()
        self.code_runner = object()
        self.recipes = type("R", (), {"all": lambda self_: [1] * 47})()
        self.workspace = Path("workspace")

    def close(self) -> None:
        return None


@pytest.fixture()
def fake_cli(monkeypatch, tmp_path):
    console = FakeConsole()
    monkeypatch.setattr(ui_theme, "make_console", lambda *a, **k: console)

    captured: dict[str, Any] = {}

    def fake_build_engine(*, workspace, config_path=None, refresh_catalog=True, on_event=None, **_kw):
        captured["on_event"] = on_event
        captured["workspace"] = workspace
        return FakeBundle(on_event)

    import gis_cli.engine.runner as runner_mod

    monkeypatch.setattr(runner_mod, "build_engine", fake_build_engine)
    return console, captured


def test_engine_task_renders_banner_steps_results_card_and_footer(fake_cli):
    console, _captured = fake_cli
    code = agent_cli._run_engine_task("统计物种点位数", Path("workspace"), None, True, 0, "dark", True)

    assert code == 0
    text = console.text
    # 启动横幅
    assert "GIS Agent" in text
    assert "glm-5.3-flash" in text and "tokenrhythm.studio" in text
    assert "47 个" in text
    # 任务行只出现一次（事件驱动渲染，避免重复打印）
    assert text.count("统计物种点位数") == 1
    # 工具动作与结果（结构化渲染）
    assert "run_recipe" in text and "recipe_id=kernel_density" in text
    assert "4 期核密度" in text
    assert "NameError" in text
    assert "自动修复第 1 次" in text
    assert "验收" in text and "通过" in text
    # 结果卡片 + 状态行
    assert "密度_1982_1986.jpg" in text
    assert "已交付 4 张密度图" in text
    assert "追踪" in text
    assert "第 3 轮" in text and "已完成" in text


def test_engine_task_reports_failure_status_with_exit_code(fake_cli, monkeypatch):
    console, _captured = fake_cli

    class FailingState(FakeState):
        def __init__(self) -> None:
            super().__init__(status="llm_unavailable")
            self.artifacts = []
            self.final_summary = ""
            self.error = 'All models failed: {"code": "INSUFFICIENT_BALANCE", "message": "余额不足", "traceId": "t-1"}'

    def fake_run(self, task: str) -> FailingState:
        emit = self._on_event or (lambda *_: None)
        emit("run_start", {"goal": task, "run_id": "r"})
        emit(
            "llm_error",
            {"error": 'Non-retryable LLM error (402): {"code": "INSUFFICIENT_BALANCE", "message": "余额不足", "traceId": "t-1"}'},
        )
        return FailingState()

    monkeypatch.setattr(FakeLoop, "run", fake_run)
    code = agent_cli._run_engine_task("任意任务", Path("workspace"), None, True, 0, "dark", True)

    assert code == 1
    text = console.text
    assert "余额不足" in text
    assert "traceId" not in text  # 网关噪声被清理
    assert "llm_unavailable" in text


# ------------------------------------------------- 内核状态 / 提示（DAIR-320、DAIR-323）

#: 真实文案的形状（``CodeRunner._mark_unavailable``）：一行，带 kernel.log 路径。
KERNEL_NOTICE = "内核不可用，已回退一次性子进程（每步约 15s）。traceback 见 ws/.gis_agent/logs/kernel.log"


class FakeKernelRunner:
    """带内核状态的假执行器（``CodeRunner`` 界面侧的四个属性 + 回调）。"""

    def __init__(self, state: str = "unavailable", notice: str = KERNEL_NOTICE) -> None:
        self.kernel_state = state
        self.kernel_notice = notice
        self.kernel_log_path = Path("ws/.gis_agent/logs/kernel.log") if notice else None
        self.on_kernel_state_change: Any = None

    def fail(self) -> None:
        """模拟预热线程里内核起不来：状态落定后自己回调界面。"""
        self.kernel_state = "unavailable"
        self.kernel_notice = KERNEL_NOTICE
        self.kernel_log_path = Path("ws/.gis_agent/logs/kernel.log")
        if callable(self.on_kernel_state_change):
            self.on_kernel_state_change(self)


def _install_kernel_runner(monkeypatch, runner: FakeKernelRunner) -> None:
    """在 ``fake_cli`` 的假 bundle 上换掉 ``code_runner``（其余装配不动）。"""
    import gis_cli.engine.runner as runner_mod

    fixture_build = runner_mod.build_engine

    def build(**kwargs):
        bundle = fixture_build(**kwargs)
        bundle.code_runner = runner
        return bundle

    monkeypatch.setattr(runner_mod, "build_engine", build)


def test_engine_task_banner_reports_kernel_state_and_one_notice(fake_cli, monkeypatch):
    """横幅报内核实际状态；启动阶段就已失败时，提示补打在横幅之后且只打一次。"""
    console, _captured = fake_cli
    runner = FakeKernelRunner()  # 钩子挂上时状态已定：没有回调，只能靠横幅后补打
    _install_kernel_runner(monkeypatch, runner)

    code = agent_cli._run_engine_task("统计物种点位数", Path("workspace"), None, True, 0, "dark", True)

    assert code == 0
    text = console.text
    assert "不可用（已回退）" in text, "横幅要报实际状态，不能写死 ready"
    assert text.count(KERNEL_NOTICE) == 1, "同一条提示只能打一次"
    assert text.index(KERNEL_NOTICE) > text.index("GIS Agent"), "提示在横幅之后"


def test_engine_task_prints_kernel_notice_when_kernel_fails_during_run(fake_cli, monkeypatch):
    """任务进行中内核才失败（预热线程）：提示由挂上的回调打出，且只打一次。"""
    console, _captured = fake_cli
    runner = FakeKernelRunner(state="ready", notice="")
    _install_kernel_runner(monkeypatch, runner)

    original_run = FakeLoop.run

    def run_and_fail_kernel(self, task):
        runner.fail()
        return original_run(self, task)

    monkeypatch.setattr(FakeLoop, "run", run_and_fail_kernel)

    code = agent_cli._run_engine_task("统计物种点位数", Path("workspace"), None, True, 0, "dark", True)

    assert code == 0
    text = console.text
    assert "就绪" in text, "横幅是启动时的真实状态（那时内核还在）"
    assert text.count(KERNEL_NOTICE) == 1, "回调没挂上就收不到提示"
    assert text.index(KERNEL_NOTICE) > text.index("GIS Agent"), "提示在横幅之后"


# ------------------------------------------------- 计划卡片闸门（DAIR-37）


def test_engine_task_plan_gate_follows_real_terminal(monkeypatch, tmp_path, install_fake_tty):
    """真终端 + 没加 --yes 必须停下来等确认；--yes 强制不问（CI / 脚本）。

    rich 的 ``status()`` 会把 sys.stdout 换成 ``FileProxy``（``isatty()`` 恒为 False），
    闸门若在 with 内部判定，真 conhost 上也会被误判成非交互（DAIR-37）。
    """
    import gis_cli.engine.runner as runner_mod

    assert install_fake_tty().isatty() is True
    console = RichConsole(
        file=io.StringIO(), width=100, no_color=True, force_terminal=True, theme=ui_theme.build_theme("dark")
    )
    monkeypatch.setattr(ui_theme, "make_console", lambda *a, **k: console)

    gates: list[Any] = []

    def fake_build_engine(*, workspace, config_path=None, refresh_catalog=True, on_event=None, plan_gate=None, **_kw):
        gates.append(plan_gate)
        return FakeBundle(on_event)

    monkeypatch.setattr(runner_mod, "build_engine", fake_build_engine)

    assert agent_cli._run_engine_task("统计物种点位数", tmp_path, None, True, 0, "dark", True) == 0
    assert gates[0].interactive is True, "真终端 + 没加 --yes：必须停下来等确认"

    assert agent_cli._run_engine_task("统计物种点位数", tmp_path, None, True, 0, "dark", True, yes=True) == 0
    assert gates[1].interactive is False, "--yes 就是不要等（CI / 脚本）"


def test_module_console_comes_from_the_project_factory():
    """子命令的输出必须走项目 Console 工厂（主题 + 默认不上色的策略）。

    历史上这里是裸 ``Console()``：绕过了"默认不发光标序列"的策略，在不解析
    ANSI 的窗口里会打出 ``?[38;5;..m``（DAIR-312）。
    """
    style = agent_cli.console.get_style("gis.tool")
    assert style.color is not None, "console 没挂上项目主题 → 说明又退回裸 Console() 了"

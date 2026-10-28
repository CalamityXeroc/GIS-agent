# -*- coding: utf-8 -*-
"""交互式对话的端到端流程测试（无头）。

用 prompt_toolkit 的 pipe input 喂按键，用假引擎（monkeypatch build_engine）替代真实
ArcGIS/LLM，验证：横幅 → 任务执行 → 工具动作/结果 → 结果卡片 → 状态行 → /status →
会话落盘 → /exit 退出。prompt_toolkit 缺失时跳过。
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import pytest

pt = pytest.importorskip("prompt_toolkit", reason="prompt_toolkit 未安装（用 ArcGIS 解释器可跑）")

from rich.console import Console as RichConsole  # noqa: E402

from gis_cli.agent import interactive as interactive_mod  # noqa: E402
from gis_cli.ui import prompt as ui_prompt  # noqa: E402
from gis_cli.ui import theme as ui_theme  # noqa: E402


class FakeState:
    def __init__(self) -> None:
        self.status = "completed"
        self.turn = 4
        self.artifacts = ["output/成果图/密度_1982_1986.jpg"]
        self.tasklist = [{"id": "t1", "title": "四期核密度", "status": "done"}]
        self.methodology = "核密度 IDW"
        self.final_summary = "已交付 4 张密度图"
        self.error = ""
        self.requirements: list[Any] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "turn": self.turn,
            "artifacts": list(self.artifacts),
            "tasklist": [dict(x) for x in self.tasklist],
            "methodology": self.methodology,
            "final_summary": self.final_summary,
            "error": self.error,
        }


class FakeLoop:
    def __init__(self, on_event) -> None:
        self._on_event = on_event
        self.recorder = type("R", (), {"jsonl_path": Path("workspace/.gis_agent/traces/run_x.jsonl")})()
        self.state = type("S", (), {"run_id": "run_x"})()

        class _Cfg:
            max_turns = 60

        self.config = _Cfg()

    def inject_user_message(self, text: str) -> bool:
        return True

    def request_stop(self, reason: str = "") -> bool:
        return True

    def run(self, task: str) -> FakeState:
        emit = self._on_event or (lambda *_: None)
        emit("run_start", {"goal": task, "run_id": "run_x"})
        emit("turn", {"turn": 1})
        emit("tool_call", {"tool": "run_recipe", "args": {"recipe_id": "kernel_density"}})
        emit("tool_result", {"tool": "run_recipe", "ok": True, "summary": "4 期核密度已生成"})
        emit("verification", {"pass": True, "summary": "验收通过（7/7）"})
        emit("run_end", FakeState().to_dict())
        return FakeState()


class FakeBundle:
    def __init__(self, on_event) -> None:
        self.loop = FakeLoop(on_event)
        self.workspace = Path("workspace")
        self.catalog = type("Cat", (), {"counts": lambda self_: {"vector": 1}})()
        self.recipes = type("R", (), {"all": lambda self_: [1] * 47})()
        self.code_runner = object()
        self.llm = type("L", (), {"config": type("C", (), {"model": "glm-5.3-flash", "api_base": "https://x/v1"})()})()

    def close(self) -> None:
        return None


class PlanAwareLoop(FakeLoop):
    """到计划卡闸门就停下来问（复刻真引擎的 plan_pending 行为）。

    真引擎那边由 ``ui.plan_card.ChatPlanGate`` 实现：引擎线程不读 stdin，
    把请求挂到 gate.pending 后停在 Event 上；这里就是按这条约定写的假引擎。
    """

    def __init__(self, on_event, gate) -> None:
        super().__init__(on_event)
        self._gate = gate
        self.cards: list[Any] = []
        self.decisions: list[Any] = []

    def run(self, task: str) -> FakeState:
        from gis_cli.ui.plan_card import build_plan_card

        revisions: list[str] = []
        while True:
            card = build_plan_card(task, [], [], workspace=None, revisions=revisions)
            self.cards.append(card)
            # 与真引擎同口径：闸门不交互（非 TTY）时卡片底栏是“未确认，直接开始”
            interactive = bool(getattr(self._gate, "interactive", False))
            self._on_event(
                "plan_card", {"text": card.render(interactive=interactive), "round": len(self.cards)}
            )
            decision = self._gate.ask(card)  # 停在这里等主线程按 s/e/q
            self.decisions.append(decision)
            if decision.action == "cancel":
                state = FakeState()
                state.status = "cancelled"
                state.turn = 0
                state.final_summary = ""
                state.error = "计划未确认（用户取消）"
                return state
            if decision.action != "revise":
                break
            revisions.append(decision.text)
        return super().run(task)


class PlanFakeBundle(FakeBundle):
    def __init__(self, on_event, gate) -> None:
        super().__init__(on_event)
        self.loop = PlanAwareLoop(on_event, gate)


def _run_chat_with_plan(
    monkeypatch,
    tmp_path: Path,
    steps,
    task: str = "把街道人口做成专题图 output/人口图.png",
    *,
    terminal: bool = True,
):
    """跑一次真 chat；每一步先等引擎真进了暂停态，再像人一样送按键。

    ``steps``：``(等暂停?, 按键)``。直接一次性把键全倒进管道会跟“引擎进入
    暂停态”赛跑（真人是看到卡片才按键），所以这里按真实节奏喂。
    """
    import threading
    import time

    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    buffer = io.StringIO()
    _patch_console(monkeypatch, buffer)

    import gis_cli.engine.runner as runner_mod

    def _build(*, workspace, config_path=None, refresh_catalog=True, on_event=None, plan_gate=None, **_k):
        return PlanFakeBundle(on_event, plan_gate)

    monkeypatch.setattr(runner_mod, "build_engine", _build)
    monkeypatch.setattr(ui_prompt, "_interactive_tty", lambda: True)
    if not terminal:
        # 非 TTY（管道/重定向）：输入框读不到键 → 闸门不得暂停
        monkeypatch.setattr(ui_prompt, "interactive_terminal", lambda: False)

    with create_pipe_input() as pipe:
        chat = interactive_mod.InteractiveChat(
            workspace=tmp_path,
            theme_mode="dark",
            no_color=True,
            prompt_input=pipe,
            prompt_output=DummyOutput(),
        )

        def _await(condition, timeout: float = 10.0) -> bool:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if condition():
                    return True
                time.sleep(0.01)
            return False

        def _feed() -> None:
            pipe.send_text(task + "\r")
            seen = None
            for wait_pause, chunk in steps:
                if wait_pause:
                    def _new_pause() -> bool:
                        nonlocal seen
                        gate = chat.plan_gate
                        request = gate.pending if gate is not None else None
                        if request is not None and request is not seen:
                            seen = request
                            return True
                        return False

                    _await(_new_pause)
                pipe.send_text(chunk)
            # 等这一轮真跑完再退出，否则 /exit 会抢在结果卡片之前
            _await(lambda: chat.session is not None and not chat.session.busy)
            pipe.send_text("/exit\r")

        threading.Thread(target=_feed, daemon=True).start()
        code = chat.run("")

    return code, buffer.getvalue(), chat


def _patch_console(monkeypatch, buffer: io.StringIO, *, force_terminal: bool = False):
    def _make_console(*_a, **_k):
        return RichConsole(
            file=buffer,
            width=100,
            no_color=True,
            force_terminal=force_terminal,
            theme=ui_theme.build_theme("dark"),
        )

    monkeypatch.setattr(ui_theme, "make_console", _make_console)


def _run_chat(monkeypatch, tmp_path: Path, keys: str) -> tuple[int, str, Path]:
    """在 pipe input 上跑一次交互会话，返回 (退出码, 屏幕文本, 会话文件)。"""
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    buffer = io.StringIO()
    _patch_console(monkeypatch, buffer)

    import gis_cli.engine.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "build_engine",
        lambda *, workspace, config_path=None, refresh_catalog=True, on_event=None, **_k: FakeBundle(on_event),
    )

    # 测试环境不是 TTY：显式打开交互路径，让 pipe 输入生效
    monkeypatch.setattr(ui_prompt, "_interactive_tty", lambda: True)

    with create_pipe_input() as pipe:
        # 通过构造参数注入 pipe：生产和测试走**同一条**代码路径，
        # 不替换 session()/_chat_prompt 之类的内部方法（那样测不到真东西）。
        chat = interactive_mod.InteractiveChat(
            workspace=tmp_path,
            theme_mode="dark",
            no_color=True,
            prompt_input=pipe,
            prompt_output=DummyOutput(),
        )
        pipe.send_text(keys)
        code = chat.run("")

    session_file = chat.sessions.path_for(chat.session_id)
    return code, buffer.getvalue(), session_file


def test_chat_runs_task_renders_and_saves_session(monkeypatch, tmp_path):
    keys = "把点位做核密度\r/status\r/exit\r"
    code, text, session_file = _run_chat(monkeypatch, tmp_path, keys)

    assert code == 0
    # 横幅与提示
    assert "GIS Agent" in text and "glm-5.3-flash" in text
    assert "会话" in text
    # 工具动作与结果（结构化渲染）
    assert "run_recipe" in text and "4 期核密度已生成" in text
    assert "验收" in text
    # 结果卡片 + 状态行
    assert "密度_1982_1986.jpg" in text and "已交付 4 张密度图" in text
    assert "已完成" in text
    # /status 输出
    assert "数据目录" in text and "配方" in text
    # 收尾提示
    assert "会话已保存" in text

    # 会话落盘：含用户与助手记录
    assert session_file.exists()
    body = session_file.read_text(encoding="utf-8")
    assert "把点位做核密度" in body
    assert "已交付 4 张密度图" in body


def test_chat_help_and_export_commands(monkeypatch, tmp_path):
    keys = "先做个统计任务\r/help\r/export\r/exit\r"
    code, text, session_file = _run_chat(monkeypatch, tmp_path, keys)

    assert code == 0
    assert "/model" in text and "Shift+Enter" in text  # /help 内容
    assert "已导出" in text  # /export（先有任务记录才能导出）
    assert (session_file.parent / f"{session_file.stem}.md").exists()


def test_chat_export_without_records_is_friendly(monkeypatch, tmp_path):
    keys = "/export\r/exit\r"
    code, text, _ = _run_chat(monkeypatch, tmp_path, keys)
    assert code == 0
    assert "无内容可导出" in text


def test_chat_unknown_command_warns(monkeypatch, tmp_path):
    keys = "/nope\r/exit\r"
    code, text, _ = _run_chat(monkeypatch, tmp_path, keys)
    assert code == 0
    assert "未知命令" in text


def test_chat_eof_exits_cleanly(monkeypatch, tmp_path):
    code, text, _ = _run_chat(monkeypatch, tmp_path, "\x04")  # Ctrl+D
    assert code == 0
    assert "会话已保存" in text


def test_chat_resume_uses_previous_context(monkeypatch, tmp_path):
    # 先造一次会话
    _, _, session_file = _run_chat(monkeypatch, tmp_path, "第一次任务：统计物种\r/exit\r")
    assert session_file.exists()

    # --continue 应带上上下文前言
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    buffer = io.StringIO()
    _patch_console(monkeypatch, buffer)
    import gis_cli.engine.runner as runner_mod

    captured: dict[str, str] = {}

    class RecordingLoop(FakeLoop):
        def run(self, task: str) -> FakeState:
            captured["task"] = task
            return super().run(task)

    class RecordingBundle(FakeBundle):
        def __init__(self, on_event) -> None:
            super().__init__(on_event)
            self.loop = RecordingLoop(on_event)

    monkeypatch.setattr(
        runner_mod,
        "build_engine",
        lambda *, workspace, config_path=None, refresh_catalog=True, on_event=None, **_k: RecordingBundle(on_event),
    )

    # 测试环境不是 TTY：显式打开交互路径，让 pipe 输入生效
    monkeypatch.setattr(ui_prompt, "_interactive_tty", lambda: True)

    with create_pipe_input() as pipe:
        chat = interactive_mod.InteractiveChat(
            workspace=tmp_path,
            theme_mode="dark",
            no_color=True,
            continue_last=True,
            prompt_input=pipe,
            prompt_output=DummyOutput(),
        )
        pipe.send_text("接着上次继续做迁移图\r/exit\r")
        chat.run("")

    assert "接续会话" in captured["task"]
    assert "接着上次继续做迁移图" in captured["task"]


# ------------------------------------------------------------------ 计划卡确认（DAIR-30）


def test_chat_plan_card_stops_and_waits_for_s(monkeypatch, tmp_path):
    """卡片打印后停下来等按键：按 s 才开始跑（以前是直接写“已自动确认”）。"""
    code, text, chat = _run_chat_with_plan(monkeypatch, tmp_path, [(True, "s\r")])

    assert code == 0
    assert "┌ 计划确认" in text and "[s]/Enter 开始" in text
    assert "已自动确认" not in text, "没人按过键就不能写“已自动确认”"
    assert "未确认，直接开始" not in text, "真等人确认时底栏不该是降级文案"
    assert "（计划确认：开始执行）" in text
    # 确认之后任务才真的跑起来
    assert "4 期核密度已生成" in text
    assert [decision.action for decision in chat.bundle.loop.decisions] == ["start"]
    # 顺序：卡片先出来，确认在后
    assert text.index("┌ 计划确认") < text.index("（计划确认：开始执行）")


def test_chat_plan_card_revise_rerenders_then_starts(monkeypatch, tmp_path):
    """e 改一句：重抽重渲染再问一次，按 s 才继续。"""
    steps = [(True, "e\r"), (False, "只统计城关镇\r"), (True, "s\r")]
    code, text, chat = _run_chat_with_plan(monkeypatch, tmp_path, steps)

    loop = chat.bundle.loop
    assert code == 0
    assert [decision.action for decision in loop.decisions] == ["revise", "start"]
    assert loop.decisions[0].text == "只统计城关镇"
    assert len(loop.cards) == 2, "改完必须重新渲染并再确认一次"
    assert "修改：只统计城关镇" in loop.cards[1].rows()[0][1]
    # 第二张卡真的打到了转录里（不是只改了屏幕上的第一张）
    assert text.count("┌ 计划确认") == 2
    assert "（计划确认：按修改重抽计划 —— 只统计城关镇）" in text
    assert "4 期核密度已生成" in text


def test_chat_plan_card_q_cancels_before_any_work(monkeypatch, tmp_path):
    """q 取消：一轮都不跑。"""
    code, text, chat = _run_chat_with_plan(monkeypatch, tmp_path, [(True, "q\r")])

    loop = chat.bundle.loop
    assert code == 0
    assert [decision.action for decision in loop.decisions] == ["cancel"]
    assert "（计划确认：取消，这次任务不开始）" in text
    assert "4 期核密度已生成" not in text, "取消了就不该开始执行"
    assert "已取消" in text  # 结果卡片的状态行


def test_chat_plan_card_degrades_when_terminal_cannot_read_keys(monkeypatch, tmp_path):
    """非 TTY 降级路径：不问、不阻塞，但底栏必须写“未确认，直接开始”。"""
    code, text, chat = _run_chat_with_plan(monkeypatch, tmp_path, [], terminal=False)

    assert code == 0
    assert "┌ 计划确认" in text, "降级也要把卡片打出来（看不看得见和要不要等是两件事）"
    assert "未确认，直接开始" in text
    assert "已自动确认" not in text
    assert [decision.action for decision in chat.bundle.loop.decisions] == ["start"], "不应该多问一次"
    assert "4 期核密度已生成" in text


def test_chat_gate_pauses_on_real_terminal(monkeypatch, tmp_path, install_fake_tty):
    """真终端语义（不 patch ``_interactive_tty``）：闸门必须停下来等确认（DAIR-37）。

    rich 的 ``status()`` 会把 sys.stdout 换成 ``FileProxy``（``isatty()`` 恒为 False）。
    闸门若在 with 内部判 tty，真 conhost 上也会退化成“未确认，直接开始”——而旧用例
    把 console 建在 force_terminal=False 下（天真的 Live 不代理 stdout），压根看不见。
    这里特意让 console 真的走 Live 代理，并把“判定发生时 stdout 是谁”也盯住。
    """
    import gis_cli.engine.runner as runner_mod

    assert install_fake_tty().isatty() is True
    _patch_console(monkeypatch, io.StringIO(), force_terminal=True)
    monkeypatch.setattr(
        runner_mod,
        "build_engine",
        lambda *, workspace, config_path=None, refresh_catalog=True, on_event=None, plan_gate=None, **_k: FakeBundle(
            on_event
        ),
    )

    seen: list[str] = []
    real_terminal = ui_prompt.interactive_terminal

    def _spy() -> bool:
        seen.append(type(sys.stdout).__name__)
        return real_terminal()

    monkeypatch.setattr(ui_prompt, "interactive_terminal", _spy)

    chat = interactive_mod.InteractiveChat(workspace=tmp_path, theme_mode="dark", no_color=True)
    assert chat._build() == 0

    assert seen and "FileProxy" not in seen, f"闸门不能落在 status(Live) 里判 tty：{seen}"
    assert chat.plan_gate.pauses is True, "真终端上计划卡必须停下来等确认"

# -*- coding: utf-8 -*-
"""交互式对话的端到端流程测试（无头）。

用 prompt_toolkit 的 pipe input 喂按键，用假引擎（monkeypatch build_engine）替代真实
ArcGIS/LLM，验证：横幅 → 任务执行 → 工具动作/结果 → 结果卡片 → 状态行 → /status →
会话落盘 → /exit 退出。prompt_toolkit 缺失时跳过。
"""

from __future__ import annotations

import io
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


def _patch_console(monkeypatch, buffer: io.StringIO):
    def _make_console(*_a, **_k):
        return RichConsole(file=buffer, width=100, no_color=True, force_terminal=False, theme=ui_theme.build_theme("dark"))

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

# -*- coding: utf-8 -*-
"""``gis_cli.web.autostart`` 的行为测试：真 socket 判活，spawn 全程用假 Popen。

不 spawn 真子进程、不打外网、不依赖 ArcGIS：所有 ``subprocess.Popen`` 都被替换，
只断言"要起什么命令、带什么参数、状态文件写成什么"。
"""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gis_cli.web import autostart
from gis_cli.web.security import TOKEN_FILE, load_or_create_token


class FakeProc:
    """假子进程：poll() 返回 ``code``（None = 还活着）。"""

    def __init__(self, pid: int = 4321, code: int | None = None) -> None:
        self.pid = pid
        self.code = code

    def poll(self) -> int | None:
        return self.code


def _record_popen(monkeypatch: pytest.MonkeyPatch, code: int | None = None) -> dict[str, Any]:
    """把 Popen 换成记录器：返回 {"args": ..., "kwargs": ...}。"""
    seen: dict[str, Any] = {}

    def fake_popen(args: list[str], **kwargs: Any) -> FakeProc:
        seen["args"] = list(args)
        seen["kwargs"] = kwargs
        return FakeProc(code=code)

    monkeypatch.setattr(autostart.subprocess, "Popen", fake_popen)
    return seen


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ------------------------------------------------------------------ is_running


def test_is_running_true_when_listening() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = int(server.getsockname()[1])
        assert autostart.is_running("127.0.0.1", port) is True


def test_is_running_false_when_nothing_listens() -> None:
    assert autostart.is_running("127.0.0.1", _free_port()) is False


# --------------------------------------------------------------- ensure_panel


def test_ensure_panel_reuses_running_instance_without_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / autostart.STATE_DIRNAME
    state.mkdir()
    (state / autostart.PID_FILE).write_text(f"{os.getpid()}\n", encoding="utf-8")
    (state / autostart.PORT_FILE).write_text("8765\n", encoding="utf-8")

    monkeypatch.setattr(autostart, "is_running", lambda host, port: True)

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("已在跑的面板不该再 spawn")

    monkeypatch.setattr(autostart.subprocess, "Popen", boom)

    status = autostart.ensure_panel(tmp_path)

    assert status.ok is True
    assert status.already_running is True
    assert status.port == 8765
    token = (state / TOKEN_FILE).read_text(encoding="utf-8").strip()
    assert status.url == f"http://127.0.0.1:8765/?token={token}"


def test_ensure_panel_waits_for_starting_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pid 活着但端口还没起来（连着启动两次）→ 等它就绪并复用，不起第二份。"""
    state = tmp_path / autostart.STATE_DIRNAME
    state.mkdir()
    (state / autostart.PID_FILE).write_text(f"{os.getpid()}\n", encoding="utf-8")
    (state / autostart.PORT_FILE).write_text("8760\n", encoding="utf-8")
    probes = {"count": 0}

    def fake_is_running(host: str, port: int) -> bool:
        probes["count"] += 1
        return probes["count"] >= 3  # 前两次还没监听，第三次起来了

    monkeypatch.setattr(autostart, "is_running", fake_is_running)

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("正在启动的面板不该被第二份取代")

    monkeypatch.setattr(autostart.subprocess, "Popen", boom)

    status = autostart.ensure_panel(tmp_path)

    assert status.ok is True
    assert status.already_running is True
    assert status.port == 8760


def test_ensure_panel_spawns_when_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: dict[str, Any] = {}
    seen = _record_popen(monkeypatch)
    monkeypatch.setattr(autostart, "_interpreter", lambda: "pythonw.exe")

    def fake_is_running(host: str, port: int) -> bool:
        return "port" in spawned and port == spawned["port"]

    def fake_popen(args: list[str], **kwargs: Any) -> FakeProc:
        spawned["port"] = int(args[args.index("--port") + 1])
        seen["args"] = list(args)
        seen["kwargs"] = kwargs
        return FakeProc(pid=4321)

    monkeypatch.setattr(autostart, "is_running", fake_is_running)
    monkeypatch.setattr(autostart.subprocess, "Popen", fake_popen)

    status = autostart.ensure_panel(tmp_path, port=8760)

    assert status.ok is True
    assert status.already_running is False
    assert status.port == 8760
    assert status.pid == 4321
    state = tmp_path / autostart.STATE_DIRNAME
    assert (state / autostart.PID_FILE).read_text(encoding="utf-8").strip() == "4321"
    assert (state / autostart.PORT_FILE).read_text(encoding="utf-8").strip() == "8760"
    assert (state / autostart.LOG_FILE).exists()
    assert status.url == autostart.panel_url(tmp_path, 8760)


def test_ensure_panel_skips_port_taken_by_foreign_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / autostart.STATE_DIRNAME
    state.mkdir()
    # pid 文件指向一个已退出的进程：8760 在听，但不是我们的面板
    (state / autostart.PID_FILE).write_text("999999\n", encoding="utf-8")
    monkeypatch.setattr(autostart, "_pid_alive", lambda pid: False)

    spawned: dict[str, Any] = {}
    seen = _record_popen(monkeypatch)
    monkeypatch.setattr(autostart, "_interpreter", lambda: "pythonw.exe")

    def fake_is_running(host: str, port: int) -> bool:
        if port == 8760:
            return True  # 别的程序占着
        return "port" in spawned and port == spawned["port"]

    def fake_popen(args: list[str], **kwargs: Any) -> FakeProc:
        spawned["port"] = int(args[args.index("--port") + 1])
        seen["args"] = list(args)
        return FakeProc(pid=5555)

    monkeypatch.setattr(autostart, "is_running", fake_is_running)
    monkeypatch.setattr(autostart.subprocess, "Popen", fake_popen)

    status = autostart.ensure_panel(tmp_path, port=8760)

    assert status.ok is True
    assert status.port == 8761
    assert spawned["port"] == 8761
    assert (state / autostart.PORT_FILE).read_text(encoding="utf-8").strip() == "8761"


def test_ensure_panel_reports_timeout_with_log_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_popen(monkeypatch)
    monkeypatch.setattr(autostart, "_interpreter", lambda: "pythonw.exe")
    monkeypatch.setattr(autostart, "is_running", lambda host, port: False)

    status = autostart.ensure_panel(tmp_path, wait_seconds=0.05)

    assert status.ok is False
    assert status.error
    assert str(tmp_path / autostart.STATE_DIRNAME / autostart.LOG_FILE) in status.error
    # 超时也要留 pid：下一次启动据此判活/清僵尸
    pid_file = tmp_path / autostart.STATE_DIRNAME / autostart.PID_FILE
    assert pid_file.exists()


def test_ensure_panel_reports_dead_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_popen(monkeypatch, code=3)
    monkeypatch.setattr(autostart, "_interpreter", lambda: "pythonw.exe")
    monkeypatch.setattr(autostart, "is_running", lambda host, port: False)

    status = autostart.ensure_panel(tmp_path, wait_seconds=0.5)

    assert status.ok is False
    assert "code=3" in status.error


def test_ensure_panel_clears_stale_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = tmp_path / autostart.STATE_DIRNAME
    state.mkdir()
    (state / autostart.PID_FILE).write_text("999999\n", encoding="utf-8")
    monkeypatch.setattr(autostart, "_pid_alive", lambda pid: False)
    _record_popen(monkeypatch)
    monkeypatch.setattr(autostart, "_interpreter", lambda: "pythonw.exe")
    monkeypatch.setattr(autostart, "is_running", lambda host, port: False)

    autostart.ensure_panel(tmp_path, wait_seconds=0.05)

    # 陈旧 pid 被替换成新起的进程，而不是继续沿用
    assert (state / autostart.PID_FILE).read_text(encoding="utf-8").strip() == "4321"


# --------------------------------------------------------------------- _spawn


def test_spawn_command_line_has_no_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _record_popen(monkeypatch)
    monkeypatch.setattr(autostart, "_interpreter", lambda: "pythonw.exe")
    token = load_or_create_token(tmp_path)

    autostart._spawn(tmp_path, "127.0.0.1", 8760, tmp_path / autostart.LOG_FILE)

    args = seen["args"]
    assert args[0] == "pythonw.exe"
    assert args[1:4] == ["-m", "gis_cli.agent.cli", "web"]
    assert "--no-browser" in args
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--port") + 1] == "8760"
    assert args[args.index("--workspace") + 1] == str(tmp_path)
    # 面板自己从 workspace 读 token，命令行里不该出现任何凭据
    assert "--token" not in args
    assert not any(token in arg for arg in args)
    assert seen["kwargs"]["env"]["GIS_NO_WEB"] == "1"
    assert seen["kwargs"]["stdin"] is subprocess.DEVNULL


def test_spawn_detaches_from_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _record_popen(monkeypatch)
    monkeypatch.setattr(autostart, "_interpreter", lambda: "python.exe")

    autostart._spawn(tmp_path, "127.0.0.1", 8760, tmp_path / autostart.LOG_FILE)

    kwargs = seen["kwargs"]
    if os.name == "nt":
        flags = kwargs["creationflags"]
        assert flags & subprocess.CREATE_NO_WINDOW
        assert flags & subprocess.DETACHED_PROCESS
        assert "start_new_session" not in kwargs
    else:
        assert kwargs["start_new_session"] is True


@pytest.mark.skipif(os.name != "nt", reason="Windows 解释器选择")
def test_interpreter_falls_back_to_python_exe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "python.exe"
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(autostart.sys, "executable", str(exe))

    assert autostart._interpreter() == str(exe)  # 没有 pythonw.exe → 退回 python.exe

    (tmp_path / "pythonw.exe").write_text("", encoding="utf-8")
    assert autostart._interpreter() == str(tmp_path / "pythonw.exe")


# ----------------------------------------------------------------- stop_panel


def test_stop_panel_without_pid_file(tmp_path: Path) -> None:
    message = autostart.stop_panel(tmp_path)

    assert "未在运行" in message
    assert not (tmp_path / autostart.STATE_DIRNAME / autostart.PID_FILE).exists()


def test_stop_panel_cleans_stale_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = tmp_path / autostart.STATE_DIRNAME
    state.mkdir()
    (state / autostart.PID_FILE).write_text("999999\n", encoding="utf-8")
    (state / autostart.PORT_FILE).write_text("8760\n", encoding="utf-8")
    monkeypatch.setattr(autostart, "_pid_alive", lambda pid: False)

    message = autostart.stop_panel(tmp_path)

    assert "未在运行" in message
    assert not (state / autostart.PID_FILE).exists()
    assert not (state / autostart.PORT_FILE).exists()


def test_stop_panel_kills_and_cleans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = tmp_path / autostart.STATE_DIRNAME
    state.mkdir()
    (state / autostart.PID_FILE).write_text(f"{os.getpid()}\n", encoding="utf-8")
    (state / autostart.PORT_FILE).write_text("8760\n", encoding="utf-8")
    killed: list[int] = []
    monkeypatch.setattr(autostart, "_kill_tree", lambda pid: killed.append(pid) or True)
    monkeypatch.setattr(autostart, "_wait_pid_gone", lambda pid, timeout: True)

    message = autostart.stop_panel(tmp_path)

    assert killed == [os.getpid()]
    assert "已停止" in message
    assert not (state / autostart.PID_FILE).exists()
    assert not (state / autostart.PORT_FILE).exists()


def test_panel_url_creates_and_reuses_token(tmp_path: Path) -> None:
    url = autostart.panel_url(tmp_path, 8760)

    token = load_or_create_token(tmp_path)
    assert token in url
    assert url == f"http://127.0.0.1:8760/?token={token}"


# --------------------------------------- 手动 web 也落 pid（DAIR-270）


def test_manual_web_state_is_written_and_stop_can_find_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """手动 ``gis-agent web`` 起的实例也要有 pid 文件，``--stop`` 才能停掉它。"""
    pid = os.getpid()
    autostart.write_panel_state(tmp_path, pid, 8765)

    state = tmp_path / autostart.STATE_DIRNAME
    assert (state / autostart.PID_FILE).read_text(encoding="utf-8").strip() == str(pid)
    assert (state / autostart.PORT_FILE).read_text(encoding="utf-8").strip() == "8765"

    killed: list[int] = []
    monkeypatch.setattr(autostart, "_kill_tree", lambda target: killed.append(target) or True)
    monkeypatch.setattr(autostart, "_wait_pid_gone", lambda target, timeout: True)

    message = autostart.stop_panel(tmp_path)

    assert killed == [pid]
    assert "已停止" in message
    assert not (state / autostart.PID_FILE).exists()
    assert not (state / autostart.PORT_FILE).exists()


def test_restore_panel_state_keeps_newer_instance_files(tmp_path: Path) -> None:
    """老实例退出时不能删掉新实例写的 pid/port。"""
    autostart.write_panel_state(tmp_path, os.getpid() + 1, 8761)

    autostart.restore_panel_state(tmp_path, os.getpid(), (None, None))  # 不是自己的记录 → 不动

    state = tmp_path / autostart.STATE_DIRNAME
    assert (state / autostart.PID_FILE).read_text(encoding="utf-8").strip() == str(
        os.getpid() + 1
    )
    assert (state / autostart.PORT_FILE).read_text(encoding="utf-8").strip() == "8761"

    autostart.restore_panel_state(tmp_path, os.getpid() + 1, (None, None))  # 是自己的 → 清掉
    assert not (state / autostart.PID_FILE).exists()
    assert not (state / autostart.PORT_FILE).exists()


def test_restore_panel_state_brings_back_previous_live_instance(tmp_path: Path, monkeypatch) -> None:
    """自己抢占失败（端口被占）退出时，把上一个还活着的实例的记录写回去。"""
    previous = (4242, 8760)
    autostart.write_panel_state(tmp_path, os.getpid(), 8761)
    monkeypatch.setattr(autostart, "_pid_alive", lambda pid: pid == 4242)

    autostart.restore_panel_state(tmp_path, os.getpid(), previous)

    state = tmp_path / autostart.STATE_DIRNAME
    assert (state / autostart.PID_FILE).read_text(encoding="utf-8").strip() == "4242"
    assert (state / autostart.PORT_FILE).read_text(encoding="utf-8").strip() == "8760"


def test_restore_panel_state_drops_previous_dead_instance(tmp_path: Path, monkeypatch) -> None:
    """上一个实例已经死了就别写回去，免得 `--stop` 一直报残留。"""
    autostart.write_panel_state(tmp_path, os.getpid(), 8761)
    monkeypatch.setattr(autostart, "_pid_alive", lambda pid: False)

    autostart.restore_panel_state(tmp_path, os.getpid(), (4242, 8760))

    state = tmp_path / autostart.STATE_DIRNAME
    assert not (state / autostart.PID_FILE).exists()
    assert not (state / autostart.PORT_FILE).exists()


# ------------------------------- 手动 web 的 pid 生命周期（DAIR-271 P1）


def _web_cmd(tmp_path: Path, port: int = 8760) -> None:
    from gis_cli.agent import cli as agent_cli

    agent_cli.web_cmd(
        host="127.0.0.1",
        port=port,
        no_browser=True,
        stop=False,
        token="test-token",
        workspace=tmp_path,
        config=None,
    )


def test_web_cmd_bind_failure_keeps_running_panel_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """端口被占（绑定失败）不能删掉正在运行实例的 web.pid / web.port。

    旧实现先写 pid 再 ``uvicorn.run``：绑定失败时 ``finally`` 的清理匹配到自己的 pid，
    把**另一个活着的实例**的记录删了，``web --stop`` 从此报「没有 pid 文件」。
    """
    state = tmp_path / autostart.STATE_DIRNAME
    state.mkdir()
    (state / autostart.PID_FILE).write_text("999999\n", encoding="utf-8")
    (state / autostart.PORT_FILE).write_text("8760\n", encoding="utf-8")

    def bind_failed(*args: Any, **kwargs: Any) -> None:
        raise OSError(10048, "address already in use")

    # 999999 是「另一个正在跑的面板」：对它判活
    monkeypatch.setattr(autostart, "_pid_alive", lambda pid: pid == 999999)
    monkeypatch.setattr("uvicorn.run", bind_failed)
    with pytest.raises(OSError):
        _web_cmd(tmp_path)

    assert (state / autostart.PID_FILE).read_text(encoding="utf-8").strip() == "999999"
    assert (state / autostart.PORT_FILE).read_text(encoding="utf-8").strip() == "8760"


def test_web_cmd_writes_panel_state_and_clears_on_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """手动 web 启动时落 pid/port（uvicorn.run 之前），正常退出时清掉自己的记录。"""
    state = tmp_path / autostart.STATE_DIRNAME
    seen: dict[str, Any] = {}

    def fake_run(app: Any, **kwargs: Any) -> None:
        seen["pid"] = (state / autostart.PID_FILE).read_text(encoding="utf-8").strip()
        seen["port"] = (state / autostart.PORT_FILE).read_text(encoding="utf-8").strip()
        with TestClient(app):  # 顺带确认 app 本身能正常起停
            pass

    from fastapi.testclient import TestClient

    monkeypatch.setattr("uvicorn.run", fake_run)
    _web_cmd(tmp_path, port=8761)

    assert seen["pid"] == str(os.getpid())
    assert seen["port"] == "8761"
    assert not (state / autostart.PID_FILE).exists(), "uvicorn.run 返回后应清掉自己的 pid/port"
    assert not (state / autostart.PORT_FILE).exists()

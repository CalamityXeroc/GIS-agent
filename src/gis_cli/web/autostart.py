# -*- coding: utf-8 -*-
"""Web 控制台的后台自启 / 判活 / 停止（无窗口 + 单实例 + 日志）。

``gis-agent chat`` 启动时调 :func:`ensure_panel`：已经有一个面板在监听就直接复用，
否则用 ``pythonw.exe`` 把 ``gis-agent web`` 拉起来（无控制台窗口、脱离 chat 的进程组），
输出追加到 ``<workspace>/.gis_agent/web.log``，pid 与实际端口落在同目录，下次启动据此判活。

三条硬约束：

- 面板必须**脱离** chat：chat 退出、终端关闭都不能带走它（用户要回看历史），
  所以 Windows 用 ``pythonw.exe``／``DETACHED_PROCESS``，POSIX 用 ``start_new_session``；
- 判定「已在跑」只靠 pid 文件 + TCP 探测，不去请求 API（探活不该依赖 token/HTTP）；
- 任何失败都只返回原因，绝不抛异常给 chat —— 控制台是附加能力，不能把对话拖死。

这个模块只 import 标准库 + :mod:`gis_cli.web.security`：它不碰终端渲染层
（``tests/web/test_import_guard.py`` 的红线），也不 import ``agent.cli``（靠 ``-m`` 调起）。
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

from .security import load_or_create_token

#: 状态文件所在目录名（相对 workspace）。
STATE_DIRNAME = ".gis_agent"
#: 面板进程 pid。
PID_FILE = "web.pid"
#: 面板**实际**监听端口（被别的程序占用时会顺延，和默认值不同）。
PORT_FILE = "web.port"
#: 面板 stdout/stderr 追加到这里。
LOG_FILE = "web.log"

#: TCP 探活超时：只连本机回环，0.3s 足够，且不会让 chat 启动卡住。
PROBE_TIMEOUT = 0.3
#: 默认端口被占时的顺延区间上界（含）。
PORT_SCAN_START = 8761
PORT_SCAN_END = 8770
#: 拉起后轮询间隔。
POLL_INTERVAL = 0.05
#: 上次的 pid 还活着但端口没起来时，等它启动完再决定（防连点两次启动撞出两份面板）。
GRACE_SECONDS = 5.0
#: ``--stop`` 等进程真正退出的上限。
STOP_TIMEOUT = 5.0


@dataclass(frozen=True)
class PanelStatus:
    """``ensure_panel`` 的结果：成功时给 URL/端口，失败时给原因和日志路径。"""

    ok: bool
    url: str = ""
    port: int | None = None
    already_running: bool = False
    pid: int | None = None
    error: str = ""
    log_path: Path | None = field(default=None, repr=False)


def state_dir(workspace: str | Path) -> Path:
    """``<workspace>/.gis_agent``（不落盘，调用方自己按需创建）。"""
    return Path(workspace) / STATE_DIRNAME


def panel_url(workspace: str | Path, port: int, host: str = "127.0.0.1") -> str:
    """带 token 的控制台 URL（token 复用 ``load_or_create_token``，重启不变）。"""
    token = load_or_create_token(workspace)
    return f"http://{host}:{int(port)}/?token={token}"


def is_running(host: str = "127.0.0.1", port: int = 8760) -> bool:
    """端口上有没有人在监听（TCP 连一下，不发 HTTP、不需要 token）。"""
    try:
        with socket.create_connection((host, int(port)), timeout=PROBE_TIMEOUT):
            return True
    except (OSError, ValueError):
        return False


def ensure_panel(
    workspace: str | Path,
    port: int = 8760,
    open_browser: bool = False,
    wait_seconds: float = 15.0,
    host: str = "127.0.0.1",
) -> PanelStatus:
    """保证 workspace 有一个控制台在跑，返回它的 URL/端口。

    - 已有面板（pid 存活 + 端口在听）→ 复用，``already_running=True``，不 spawn；
    - 端口被别的程序占用 → 顺延 8761..8770，并把实际端口写进 ``web.port``；
    - 拉起后 ``wait_seconds`` 内没监听 → 返回失败原因（带 ``web.log`` 路径）。
    """
    ws = Path(workspace)
    state = state_dir(ws)
    state.mkdir(parents=True, exist_ok=True)
    pid = _read_int(state / PID_FILE)
    current_port = _read_int(state / PORT_FILE) or int(port)

    # pid 存活就算我们的面板：端口在听就直接复用；没在听也先等一会儿——
    # 大概率是正在启动（用户连点两次启动），否则第二份进程会去撞同一个端口然后退出。
    grace = min(GRACE_SECONDS, float(wait_seconds))
    if pid and _pid_alive(pid) and (
        is_running(host, current_port) or _wait_listening(host, current_port, grace)
    ):
        return PanelStatus(
            ok=True,
            url=panel_url(ws, current_port, host),
            port=current_port,
            already_running=True,
            pid=pid,
            log_path=state / LOG_FILE,
        )

    # 僵尸：pid 文件在、进程已经没了 → 清掉，重新拉起
    if pid and not _pid_alive(pid):
        _unlink(state / PID_FILE)

    candidates = _candidate_ports(current_port, int(port))
    for candidate in candidates:
        if is_running(host, candidate):
            continue  # 被别的程序（或不是我们 pid 记录的那个面板）占着 → 顺延
        return _start(ws, state, host, candidate, wait_seconds, open_browser)

    return PanelStatus(
        ok=False,
        error=f"端口 {candidates[0]}..{PORT_SCAN_END} 都被占用，控制台未启动",
        log_path=state / LOG_FILE,
    )


def stop_panel(workspace: str | Path, host: str = "127.0.0.1") -> str:
    """按 ``web.pid`` 停掉面板（含子孙进程），清理 pid/port 文件。返回给用户看的一句话。"""
    state = state_dir(workspace)
    pid = _read_int(state / PID_FILE)
    if not pid:
        return "控制台未在运行（没有 pid 文件）"
    if not _pid_alive(pid):
        _cleanup(state)
        return f"控制台未在运行（pid {pid} 已退出，已清理残留状态）"

    try:
        killed = _kill_tree(pid)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"控制台停止失败：无法结束 pid {pid}（{exc}）"
    if not killed:
        return f"控制台停止失败：pid {pid} 杀不掉（可能权限不足）"

    _wait_pid_gone(pid, STOP_TIMEOUT)
    _cleanup(state)
    return f"控制台已停止（pid {pid}）"


# ---------------------------------------------------------------- 内部实现


def write_panel_state(workspace: str | Path, pid: int, port: int) -> None:
    """面板进程落 ``web.pid`` / ``web.port``。

    不光自启路径（:func:`_start`）要写：手动 ``gis-agent web`` 起的实例同样得写，
    否则 ``web --stop`` 会说「没有 pid 文件」，而端口其实还在听（DAIR-270）。
    """
    state = state_dir(workspace)
    state.mkdir(parents=True, exist_ok=True)
    _write_int(state / PID_FILE, pid)
    _write_int(state / PORT_FILE, port)


def read_panel_state(workspace: str | Path) -> tuple[int | None, int | None]:
    """当前记录的面板 ``(pid, port)``，没有就是 ``(None, None)``。"""
    state = state_dir(workspace)
    return _read_int(state / PID_FILE), _read_int(state / PORT_FILE)


def restore_panel_state(
    workspace: str | Path, pid: int, previous: tuple[int | None, int | None]
) -> None:
    """进程退出时收尾自己的状态；若自己是「抢占失败」的后来者，把上一个活着的实例写回。

    为什么需要它（DAIR-271 P1）：``uvicorn`` 是**先跑 lifespan startup、再 bind**，
    所以 pid 只能在 ``uvicorn.run`` 之前落盘。端口被占启动失败时，直接删文件会把
    **另一个正在跑的实例**的记录抹掉，``web --stop`` 从此报「没有 pid 文件」而面板
    还在监听。这里把启动前的快照写回去（只认还活着的 pid，死记录照常清掉）。
    """
    state = state_dir(workspace)
    if _read_int(state / PID_FILE) != int(pid):
        return  # 已经不是我们写的（被新实例覆盖 / 被 --stop 清掉）
    old_pid, old_port = previous
    if old_pid and int(old_pid) != int(pid) and _pid_alive(int(old_pid)):
        _write_int(state / PID_FILE, int(old_pid))
        if old_port:
            _write_int(state / PORT_FILE, int(old_port))
        return
    _unlink(state / PID_FILE)
    _unlink(state / PORT_FILE)


def _candidate_ports(*preferred: int) -> list[int]:
    """先试上次/指定端口（保持端口稳定），再顺延 8761..8770。"""
    ports: list[int] = []
    for value in (*preferred, *range(PORT_SCAN_START, PORT_SCAN_END + 1)):
        if value and value not in ports:
            ports.append(int(value))
    return ports


def _start(
    ws: Path,
    state: Path,
    host: str,
    port: int,
    wait_seconds: float,
    open_browser: bool,
) -> PanelStatus:
    log_path = state / LOG_FILE
    try:
        proc = _spawn(ws, host, port, log_path)
    except OSError as exc:
        return PanelStatus(ok=False, port=port, error=f"无法启动控制台进程：{exc}", log_path=log_path)

    # 先落 pid：即使后面等超时，下一次启动也能据此判活／清僵尸
    _write_int(state / PID_FILE, proc.pid)

    deadline = time.monotonic() + max(float(wait_seconds), 0.0)
    while True:
        if is_running(host, port):
            _write_int(state / PORT_FILE, port)
            url = panel_url(ws, port, host)
            if open_browser:
                webbrowser.open(url)
            return PanelStatus(
                ok=True, url=url, port=port, already_running=False, pid=proc.pid, log_path=log_path
            )
        code = proc.poll()
        if code is not None:
            return PanelStatus(
                ok=False,
                port=port,
                pid=proc.pid,
                error=f"控制台进程已退出（code={code}），日志见 {log_path}",
                log_path=log_path,
            )
        if time.monotonic() >= deadline:
            return PanelStatus(
                ok=False,
                port=port,
                pid=proc.pid,
                error=f"等待 {wait_seconds:g}s 仍未监听 {host}:{port}，日志见 {log_path}",
                log_path=log_path,
            )
        time.sleep(POLL_INTERVAL)


def _spawn(ws: Path, host: str, port: int, log_path: Path) -> subprocess.Popen:
    """无窗口拉起 ``web`` 子进程，stdout/stderr 追加进 ``web.log``。"""
    args = [
        _interpreter(),
        "-m",
        "gis_cli.agent.cli",
        "web",
        "--no-browser",
        "--host",
        host,
        "--port",
        str(int(port)),
        "--workspace",
        str(ws),
    ]
    env = dict(os.environ)
    # 子进程别再自启面板（防递归）；日志文件按 utf-8 写，子进程输出也统一成 utf-8
    env["GIS_NO_WEB"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    popen_kwargs: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "stderr": subprocess.STDOUT,
        "cwd": str(ws),
        "env": env,
    }
    if os.name == "nt":
        # 无控制台窗口：python.exe 走 CREATE_NO_WINDOW；DETACHED_PROCESS 让它不跟着
        # chat 的控制台一起结束。pythonw.exe 本来就无窗口，这两个 flag 无害。
        popen_kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        # 自成进程组：chat 退出后收不到 SIGHUP，--stop 才能 killpg 一次带走整棵树
        popen_kwargs["start_new_session"] = True

    log_handle = open(log_path, "a", encoding="utf-8", errors="replace")
    try:
        return subprocess.Popen(args, stdout=log_handle, **popen_kwargs)
    finally:
        log_handle.close()


def _interpreter() -> str:
    """Windows 优先同目录 ``pythonw.exe``（GUI 子系统，绝不闪黑窗），缺失才退回 ``python.exe``。"""
    if os.name != "nt":
        return sys.executable
    exe = Path(sys.executable)
    gui = exe.with_name("pythonw.exe")
    return str(gui) if gui.exists() else str(exe)


def _kill_tree(pid: int) -> bool:
    """结束 pid 及其子孙；返回 pid 是否确实退出了。"""
    if os.name == "nt":
        # Windows 没有进程组信号；/T 才带子孙，/F 保证不弹确认框
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            timeout=15,
            check=False,
        )
    else:
        import signal

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)  # start_new_session → pid 就是 pgid
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        if _wait_pid_gone(pid, STOP_TIMEOUT / 2):
            return True
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return _wait_pid_gone(pid, STOP_TIMEOUT / 2)
    return _wait_pid_gone(pid, STOP_TIMEOUT)


def _wait_listening(host: str, port: int, timeout: float) -> bool:
    """等端口开始监听（最多 ``timeout`` 秒）。"""
    deadline = time.monotonic() + max(float(timeout), 0.0)
    while True:
        if is_running(host, port):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_INTERVAL)


def _wait_pid_gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + max(float(timeout), 0.0)
    while _pid_alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_INTERVAL)
    return True


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:  # 存在但没权限发信号
            return True
        return True
    return _windows_pid_alive(pid)


def _windows_pid_alive(pid: int) -> bool:  # pragma: no cover - 平台相关
    """``os.kill(pid, 0)`` 在 Windows 上是 TerminateProcess，这里必须走 Win32。

    坑：``OpenProcess`` 默认 restype 是 ``c_int``，64 位下句柄会被截断，
    必须显式声明 ``wintypes.HANDLE``。
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    SYNCHRONIZE = 0x00100000
    WAIT_TIMEOUT = 0x00000102
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _write_int(path: Path, value: int) -> None:
    path.write_text(f"{int(value)}\n", encoding="utf-8")


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _cleanup(state: Path) -> None:
    _unlink(state / PID_FILE)
    _unlink(state / PORT_FILE)

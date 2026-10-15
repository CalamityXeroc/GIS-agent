# -*- coding: utf-8 -*-
"""Code execution facade: persistent kernel first, one-shot subprocess fallback."""

from __future__ import annotations

import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .kernel import ArcpyKernel, ExecResult, KernelError, KernelTimeout

logger = logging.getLogger(__name__)


class CodeRunner:
    """Runs ArcPy code with a warm kernel and safe fallback."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        python_executable: str | None = None,
        state_dir: str | Path | None = None,
        exec_timeout: float = 600.0,
        allow_subprocess_fallback: bool = True,
    ):
        self.workspace = Path(workspace)
        self.python_executable = python_executable
        self.state_dir = state_dir
        self.exec_timeout = exec_timeout
        self.allow_subprocess_fallback = allow_subprocess_fallback
        self._kernel: ArcpyKernel | None = None
        self._kernel_failed = False
        self.last_backend = "none"
        #: 内核启动很快（实测 arcpy 导入就要 8~11s）：预热与扫描并行，
        #: 首个 execute_code 就不用再等。锁保证不会起两个内核。
        self._kernel_lock = threading.Lock()
        self._prewarm_thread: threading.Thread | None = None
        #: 预热是**一次性**动作，用显式标志而不是“线程还活着吗”判断：
        #: 工作线程可能在调用方等到锁之前就已结束，那时 ``is_alive()`` 已是
        #: False，只看它会又起一个内核（实测竞争）。
        self._prewarm_started = False

    # ---------------------------------------------------------------- 预热
    def prewarm(self) -> bool:
        """后台启动内核（幂等）。已在跑或已就绪时返回 False。

        为什么要预热：内核冷启动实测 11.6~17.6s（大头是 ``import arcpy``）。
        启动时不预热的话，用户第一个要跑代码的任务会白等这一段；
        预热后可与数据目录扫描、首轮模型调用同时进行。
        没有 jupyter_client 时这里会快速失败并置 ``_kernel_failed``，
        后续自动退回一次性子进程（与旧行为一致）。
        """
        if self._kernel_failed:
            return False
        if self._kernel is not None and self._kernel.is_alive():
            return False
        with self._kernel_lock:
            # 先看一次性标志：线程可能已经结束，但那次预热已经做过了
            if self._prewarm_started:
                return False
            if self._kernel_failed:
                return False
            if self._kernel is not None and self._kernel.is_alive():
                return False
            self._prewarm_started = True
            self._prewarm_thread = threading.Thread(
                target=self._ensure_kernel, name="gis-agent-kernel-prewarm", daemon=True
            )
            self._prewarm_thread.start()
            return True

    def wait_prewarm(self, timeout: float | None = None) -> None:
        """等预热完成（不需要精确同步时可不调）。"""
        thread = self._prewarm_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    def _ensure_kernel(self) -> ArcpyKernel | None:
        if self._kernel_failed:
            return None
        if self._kernel is not None and self._kernel.is_alive():
            return self._kernel
        with self._kernel_lock:
            # 双重检查：预热线程与 execute_code 可能同时到达
            if self._kernel_failed:
                return None
            if self._kernel is not None and self._kernel.is_alive():
                return self._kernel
            try:
                from .timing import StageTimer

                timer = StageTimer("kernel.prewarm")
                self._kernel = ArcpyKernel(
                    python_executable=self.python_executable,
                    state_dir=self.state_dir,
                    workspace=str(self.workspace),
                )
                self._kernel.start()
                self.last_backend = "kernel"
                timer.done(ok=True)
                return self._kernel
            except Exception as exc:
                logger.warning("persistent kernel unavailable (%s); falling back to subprocess", exc)
                self._kernel_failed = True
                return None

    # ----------------------------------------------------------------- public
    def run(
        self,
        code: str,
        *,
        timeout: float | None = None,
        workspace: str | Path | None = None,
    ) -> ExecResult:
        """Execute code, preferring the warm kernel."""
        timeout = float(timeout or self.exec_timeout)
        ws = str(workspace or self.workspace)
        kernel = self._ensure_kernel()
        if kernel is not None:
            try:
                return kernel.execute(code, timeout=timeout, workspace=ws)
            except KernelTimeout as exc:
                return ExecResult(
                    ok=False,
                    error={"type": "TimeoutExpired", "message": str(exc), "traceback": ""},
                )
            except KernelError as exc:
                logger.warning("kernel execution failed: %s", exc)
                self._kernel_failed = True
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("kernel crashed: %s", exc)
                self._kernel_failed = True
        if not self.allow_subprocess_fallback:
            return ExecResult(
                ok=False,
                error={"type": "KernelUnavailable", "message": "persistent kernel unavailable"},
            )
        return self._run_subprocess(code, timeout=timeout, workspace=ws)

    def execute(self, code: str, *, timeout: float = 600.0, workspace: str | None = None) -> ExecResult:
        """Kernel-compatible alias used by the catalog scanner."""
        return self.run(code, timeout=timeout, workspace=workspace)

    def _run_subprocess(self, code: str, *, timeout: float, workspace: str) -> ExecResult:
        from ..arcpy_bridge import run_arcpy_code

        started = time.time()
        result = run_arcpy_code(
            code,
            workspace=workspace,
            timeout_seconds=int(max(10, timeout)),
            python_executable=self.python_executable,
        )
        self.last_backend = "subprocess"
        return ExecResult(
            ok=result.status == "success",
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            result=result.data,
            error=result.error,
            duration_ms=int((time.time() - started) * 1000),
        )

    def reset(self) -> None:
        """Reset kernel state between tasks."""
        if self._kernel is not None and self._kernel.is_alive():
            try:
                self._kernel.reset()
            except Exception as exc:
                logger.warning("kernel reset failed: %s", exc)

    def restart(self) -> None:
        if self._kernel is not None:
            try:
                self._kernel.restart()
                self._kernel_failed = False
                return
            except Exception as exc:
                logger.warning("kernel restart failed: %s", exc)
        self._kernel_failed = False
        self._kernel = None

    def shutdown(self) -> None:
        """关内核。先等在途的预热收尾，避免与半启动的内核抢 shm（实测会抦 zmq 报
        ``Bad address``）。
        """
        self._kernel_failed = True  # 关掉后不再自动重启
        self.wait_prewarm(timeout=120)
        if self._kernel is not None:
            self._kernel.shutdown()
            self._kernel = None

    @property
    def backend(self) -> str:
        return self.last_backend


# --------------------------------------------------------------------- backup
_DESTRUCTIVE_MARKERS = (
    "arcpy.management.Delete",
    "arcpy.Delete_management",
    "arcpy.management.Format",
    "arcpy.Convert_",
    "shutil.rmtree",
    "os.remove(",
    "os.unlink(",
    "os.rmdir(",
    ".unlink(",
)


def looks_destructive(code: str) -> bool:
    """Heuristic check for code that deletes/overwrites data."""
    return any(marker in code for marker in _DESTRUCTIVE_MARKERS)


def _is_volatile(name: str) -> bool:
    """ArcGIS 的瞬时锁文件（还在写时会创建，写完/进程退出后消失）。

    实测：备份一个 result_data.gdb 时，copytree 会把 ``*.sr.lock`` / ``*.rd.lock``
    也列进去，但它们很快就被 ArcGIS 删掉，于是复制到一半报
    ``[WinError 2] 系统找不到指定的文件``，一整排警告刷屏（还白写了一个
    半成品的备份目录）。这些锁文件本身没有备份价值。
    """
    lowered = name.lower()
    return lowered.endswith((".sr.lock", ".rd.lock", ".lock", ".lockfile"))


def backup_paths(paths: list[str], backup_root: str | Path) -> list[str]:
    """Copy existing files/dirs into a timestamped backup folder.

    Returns the list of created backup paths. Best-effort: failures are
    reported by omission, not raised.
    """
    root = Path(backup_root) / time.strftime("%Y%m%d_%H%M%S")
    created: list[str] = []
    for raw in paths:
        src = Path(raw)
        if not src.exists():
            continue
        root.mkdir(parents=True, exist_ok=True)
        dest = root / src.name
        try:
            if src.is_dir():
                if dest.exists():
                    shutil.rmtree(dest, ignore_errors=True)
                # 跳过瞬时锁文件：复制到一半它们就没了，只会刷一堆 WinError 2
                shutil.copytree(
                    src, dest, ignore=shutil.ignore_patterns("*.lock", "*.lockfile")
                )
            else:
                if _is_volatile(src.name):
                    continue
                shutil.copy2(src, dest)
            created.append(str(dest))
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("backup failed for %s: %s", src, exc)
    return created

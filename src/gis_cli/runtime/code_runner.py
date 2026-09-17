# -*- coding: utf-8 -*-
"""Code execution facade: persistent kernel first, one-shot subprocess fallback."""

from __future__ import annotations

import logging
import shutil
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

    # ---------------------------------------------------------------- kernel
    def _ensure_kernel(self) -> ArcpyKernel | None:
        if self._kernel_failed:
            return None
        if self._kernel is not None and self._kernel.is_alive():
            return self._kernel
        try:
            self._kernel = ArcpyKernel(
                python_executable=self.python_executable,
                state_dir=self.state_dir,
                workspace=str(self.workspace),
            )
            self._kernel.start()
            self.last_backend = "kernel"
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
                shutil.copytree(src, dest)
            else:
                shutil.copy2(src, dest)
            created.append(str(dest))
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("backup failed for %s: %s", src, exc)
    return created

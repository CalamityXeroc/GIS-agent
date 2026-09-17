# -*- coding: utf-8 -*-
"""Persistent ArcPy kernel.

Runs a long-lived Jupyter kernel inside the ArcGIS Pro Python environment so
that ``arcpy`` is imported once and Python state survives across agent steps.

Design notes
------------
- The kernel is launched through a generated kernelspec that pins the ArcGIS
  Pro interpreter, so the agent process itself does not have to be that
  interpreter.
- Every call is single-flight; the caller owns the session.
- On timeout we try a soft interrupt first; if the kernel does not come back
  promptly the caller may ``restart()``.
- All payloads crossing the boundary are JSON-serializable.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RESULT_VAR = "__arcpy_engine_result__"

_PRELUDE = f'''
import builtins as _bi
{_RESULT_VAR} = None

def set_result(value):
    global {_RESULT_VAR}
    {_RESULT_VAR} = value

_bi.{_RESULT_VAR} = None
'''


class KernelError(RuntimeError):
    """Base kernel failure."""


_NOISE_MARKERS = (
    "Unable to load code page translation table",
    "Code page conversion is off",
)


def _strip_arcgis_noise(text: str) -> str:
    """Drop ArcGIS locale warnings that pollute kernel stdout."""
    if not text or not any(marker in text for marker in _NOISE_MARKERS):
        return text
    kept = [
        line
        for line in text.splitlines(keepends=True)
        if not any(marker in line for marker in _NOISE_MARKERS)
    ]
    return "".join(kept)


class KernelTimeout(KernelError):
    """Execution exceeded its timeout."""


@dataclass
class ExecResult:
    """Result of one kernel execution."""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    result: Any = None
    error: dict[str, Any] | None = None
    duration_ms: int = 0
    interrupted: bool = False
    display_images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "interrupted": self.interrupted,
            "display_images": self.display_images,
        }


class ArcpyKernel:
    """A persistent Jupyter kernel for ArcPy code execution."""

    def __init__(
        self,
        *,
        python_executable: str | None = None,
        state_dir: str | Path | None = None,
        kernel_name: str = "arcpy-engine",
        startup_timeout: float = 180.0,
        workspace: str | None = None,
        watchdog_grace: float = 45.0,
    ):
        self.python_executable = python_executable or self._discover_arcgis_python()
        self.state_dir = Path(state_dir or Path.home() / ".gis_agent" / "kernel")
        self.kernel_name = kernel_name
        self.startup_timeout = startup_timeout
        self.workspace = workspace
        self.watchdog_grace = watchdog_grace
        self._km: Any = None
        self._kc: Any = None
        self._started = False
        self._namespace_dirty = False
        self._exec_count = 0

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _discover_arcgis_python() -> str:
        try:
            from ..arcpy_bridge import discover_arcgis_pro_python

            return discover_arcgis_pro_python().python_executable
        except Exception as exc:
            raise KernelError(
                "ArcGIS Pro Python not found. Set ARCGIS_PRO_PYTHON or install ArcGIS Pro."
            ) from exc

    def _ensure_kernelspec(self) -> None:
        """Create a kernelspec pinned to the ArcGIS Pro interpreter."""
        spec_dir = self.state_dir / "kernels" / self.kernel_name
        spec_dir.mkdir(parents=True, exist_ok=True)
        kernel_json = {
            "argv": [
                self.python_executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ],
            "display_name": "ArcPy Engine (ArcGIS Pro)",
            "language": "python",
            "env": {
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "FOR_DISABLE_CONSOLE_CTRL_HANDLER": "1",
            },
        }
        (spec_dir / "kernel.json").write_text(
            json.dumps(kernel_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        existing = os.environ.get("JUPYTER_PATH", "")
        paths = [p for p in existing.split(os.pathsep) if p]
        if str(self.state_dir) not in paths:
            os.environ["JUPYTER_PATH"] = os.pathsep.join([str(self.state_dir), *paths])

    # ------------------------------------------------------------ life cycle
    def start(self) -> None:
        """Start the kernel and verify arcpy is importable."""
        if self._started and self.is_alive():
            return
        self._ensure_kernelspec()
        try:
            from jupyter_client.manager import KernelManager
        except ImportError as exc:  # pragma: no cover
            raise KernelError(
                "jupyter_client is required for the persistent kernel. "
                "Install it in the ArcGIS Pro Python environment."
            ) from exc

        self._km = KernelManager(kernel_name=self.kernel_name)
        self._km.start_kernel()
        self._kc = self._km.client()
        self._kc.start_channels()
        try:
            self._kc.wait_for_ready(timeout=self.startup_timeout)
        except Exception as exc:
            self.shutdown()
            raise KernelError(f"Kernel did not become ready: {exc}") from exc

        self._started = True
        self._exec_count = 0
        self._namespace_dirty = False
        self._run_prelude()
        probe = self.execute("import arcpy; print('arcpy', arcpy.GetInstallInfo()['Version'])", timeout=120)
        if not probe.ok:
            self.shutdown()
            raise KernelError(f"Kernel started but arcpy is unavailable: {probe.error}")

    def _run_prelude(self) -> None:
        self._execute_raw(_PRELUDE, timeout=60)

    def is_alive(self) -> bool:
        if not self._started or self._km is None:
            return False
        try:
            return bool(self._km.is_alive())
        except Exception:
            return False

    def restart(self) -> None:
        """Restart the kernel, dropping all state."""
        if self._km is None:
            self.start()
            return
        try:
            self._km.restart_kernel(now=True)
        except Exception:
            self.shutdown()
            self.start()
            return
        try:
            self._kc.start_channels()
            self._kc.wait_for_ready(timeout=self.startup_timeout)
        except Exception as exc:
            raise KernelError(f"Kernel restart failed: {exc}") from exc
        self._started = True
        self._exec_count = 0
        self._namespace_dirty = False
        self._run_prelude()

    def shutdown(self) -> None:
        try:
            if self._kc is not None:
                self._kc.stop_channels()
        except Exception:
            pass
        try:
            if self._km is not None:
                self._km.shutdown_kernel(now=True)
        except Exception:
            pass
        self._km = None
        self._kc = None
        self._started = False

    def interrupt(self) -> None:
        """Send a soft interrupt to running code.

        Never blocks for long: ``interrupt_kernel`` waits for a ZMQ reply
        which a wedged kernel never sends, so it runs in a daemon thread.
        """
        try:
            km = self._km
            if km is not None:
                t = threading.Thread(target=km.interrupt_kernel, daemon=True)
                t.start()
                t.join(timeout=4.0)
        except Exception as exc:
            logger.warning("kernel interrupt failed: %s", exc)

    def _destroy_kernel(self) -> None:
        """Hard-stop the kernel process (used when soft interrupt fails)."""
        km = self._km
        if km is not None:
            try:
                km.kill()
            except Exception as exc:
                logger.warning("kernel hard kill failed: %s", exc)
        self._started = False
        logger.warning("kernel process destroyed (wedged); will restart on next use")

    # ------------------------------------------------------------- execution
    def execute(
        self,
        code: str,
        *,
        timeout: float = 600.0,
        workspace: str | None = None,
    ) -> ExecResult:
        """Execute user code in the persistent namespace.

        ``set_result(value)`` inside the code is captured and returned.
        """
        if not self.is_alive():
            self.start()
        ws = workspace or self.workspace
        prefix = ""
        if ws:
            safe = ws.replace("\\", "\\\\").replace("'", "\\'")
            prefix = (
                "import arcpy\n"
                f"arcpy.env.workspace = r'{safe}'\n"
                "arcpy.env.overwriteOutput = True\n"
            )
        started = time.time()
        outcome = self._execute_raw(prefix + code, timeout=timeout)
        outcome.duration_ms = int((time.time() - started) * 1000)
        if outcome.ok:
            outcome.result = self._fetch_result()
        return outcome

    def _execute_raw(self, code: str, *, timeout: float) -> ExecResult:
        """Execute code and collect iopub output until idle."""
        if self._kc is None:
            raise KernelError("Kernel is not running")
        self._exec_count += 1
        msg_id = self._kc.execute(code)
        stdout: list[str] = []
        stderr: list[str] = []
        images: list[str] = []
        error: dict[str, Any] | None = None
        deadline = time.time() + timeout
        interrupted = False

        # Watchdog: even if the iopub/control channels wedge (native arcpy
        # blocking), hard-kill the kernel process so execution is bounded.
        hard_timeout = timeout + self.watchdog_grace
        km = self._km
        done = threading.Event()

        def _watchdog() -> None:
            if not done.wait(hard_timeout) and km is not None:
                logger.warning(
                    "watchdog: wedged kernel after %ss, hard-killing process", hard_timeout
                )
                try:
                    km.kill()
                except Exception as exc:  # pragma: no cover - best effort
                    logger.warning("watchdog kill failed: %s", exc)
                self._started = False

        threading.Thread(target=_watchdog, daemon=True).start()
        try:
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    if not interrupted:
                        interrupted = True
                        self.interrupt()
                        deadline = time.time() + 15.0  # grace period after interrupt
                        continue
                    self._destroy_kernel()
                    raise KernelTimeout(f"Execution exceeded {timeout}s and did not stop after interrupt")
                try:
                    msg = self._kc.get_iopub_msg(timeout=min(remaining, 1.0))
                except Exception:
                    continue
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                mtype = msg.get("msg_type")
                content = msg.get("content", {}) or {}
                if mtype == "stream":
                    if content.get("name") == "stderr":
                        stderr.append(content.get("text", ""))
                    else:
                        stdout.append(_strip_arcgis_noise(content.get("text", "")))
                elif mtype == "error":
                    error = {
                        "type": content.get("ename", "Error"),
                        "message": content.get("evalue", ""),
                        "traceback": "\n".join(content.get("traceback", []) or []),
                    }
                elif mtype in {"display_data", "execute_result"}:
                    data = content.get("data", {}) or {}
                    if "image/png" in data:
                        images.append(str(data["image/png"]))
                    if "text/plain" in data and mtype == "execute_result":
                        stdout.append(str(data["text/plain"]))
                elif mtype == "status" and content.get("execution_state") == "idle":
                    break
        finally:
            done.set()

        # Drain the shell reply so it does not leak into the next call.
        try:
            self._kc.get_shell_msg(timeout=5.0)
        except Exception:
            pass

        return ExecResult(
            ok=error is None,
            stdout="".join(stdout),
            stderr="".join(stderr),
            error=error,
            interrupted=interrupted,
            display_images=images,
        )

    def _fetch_result(self) -> Any:
        """Fetch and JSON-decode the value passed to set_result()."""
        marker = "__ENGINE_RESULT__"
        outcome = self._execute_raw(
            f"import json as _json; print('{marker}' + _json.dumps(globals().get('{_RESULT_VAR}'), default=str))",
            timeout=60,
        )
        if not outcome.ok:
            return None
        text = outcome.stdout or ""
        idx = text.rfind(marker)
        if idx < 0:
            return None
        payload = text[idx + len(marker):].strip()
        if not payload or payload == "null":
            return None
        try:
            return json.loads(payload)
        except Exception:
            return payload

    # --------------------------------------------------------------- helpers
    def reset(self) -> None:
        """Clear user namespace but keep arcpy imported."""
        self._execute_raw(
            "for _k in [k for k in list(globals()) if not k.startswith('__') and k not in "
            "('set_result', 'arcpy', '_bi', 'builtins')]:\n"
            "    del globals()[_k]\n",
            timeout=60,
        )
        self._run_prelude()
        self._namespace_dirty = False

    def run_arcpy(self, code: str, *, timeout: float = 600.0, workspace: str | None = None) -> ExecResult:
        """Convenience wrapper requiring arcpy to be importable."""
        return self.execute(code, timeout=timeout, workspace=workspace)

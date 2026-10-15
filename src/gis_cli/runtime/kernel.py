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
import sys
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

# 中文字体与绘图环境：matplotlib 默认字体没有中文字形，图里中文会变方框（tofu）。
# 统一配置一次（缺 matplotlib 时静默跳过），保证 agent 自绘图的中文可读。
try:
    from gis_cli.runtime.mpl_setup import configure as _mpl_configure
    _mpl_configure()
except Exception:
    pass
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
        #: 执行串行化锁。
        #:
        #: **必须**：ZMQ socket 不是线程安全的。预热把内核建在后台线程、
        #: 主线程随后执行代码时，两边的 socket 操作会交叉写坏 wire 协议。
        #: 实测症状（可复现）：内核侧报 ``Invalid Signature: b'<IDS|MSG>'``
        #: / ``DELIM not in msg_list``，客户端侧报 zmq ``Bad address``。
        #: 锁把"发消息 + 收 iopub/shell + 取结果"整段封起来。
        self._exec_lock = threading.RLock()

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
        from .timing import StageTimer

        _timer = StageTimer("kernel.start")
        self._ensure_kernelspec()
        try:
            from jupyter_client.manager import KernelManager
        except ImportError as exc:  # pragma: no cover
            # 这里是**跑 agent 的那个解释器**缺包，不是内核解释器缺包。
            # 以前的文案写“Install it in the ArcGIS Pro Python environment”会把人
            # 带去错的地方：ArcGIS 环境本来就有，缺的往往是 venv。
            raise KernelError(
                "persistent kernel needs jupyter_client in the interpreter running the "
                f"agent ({sys.executable}); install it there (pip install jupyter-client). "
                "Without it every step falls back to a fresh subprocess that re-imports "
                "arcpy (~15s) and loses cross-step variables."
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
        # 分解：进程+连接就绪 vs 导入 arcpy（后者是大头，实测十几秒）
        _timer.done(startup=".", exec_ms=probe.duration_ms)

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
        """Shut down the kernel.

        **先让 ArcGIS 把数据刷盘，再杀内核**。实测（14 届 4a-1）：直接
        ``shutdown_kernel(now=True)`` 硬杀时，GDB 里刚写完的栅格还没落盘，
        内核一死文件就跟着没了，目录里只剩 ``*.sr.lock``；模型上一轮明明
        用 ``arcpy.ListRasters()`` 确认过四个 kde_* 都在，交付时却报“缺失”。
        所以先给内核一点时间收尾。
        """
        self._flush_arcpy_writes()
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

    def _flush_arcpy_writes(self) -> None:
        """Best-effort: let ArcGIS finish writing before the kernel is killed.

        实测（14 届 4a-1）：``shutdown_kernel(now=True)`` 硬杀时，GDB 里刚写完的
        栅格尚未落盘，内核一死数据就丢了，只留下 ``*.sr.lock``——模型上一轮
        还用 ``arcpy.ListRasters()`` 确认过四个 kde_* 都在，交付时却报“缺失”。

        做法（全部容忍失败）：删临时引用 → 触发 GC（释放游标/数据集句柄，
        ArcGIS 在对象析构时提交）→ 再等一拍。不跑重型检查，避免关机变慢。
        """
        if not self._started or self._kc is None:
            return
        try:
            self.execute(
                "import gc\n"
                "gc.collect()\n"
                "set_result({'flushed': True})",
                timeout=30,
            )
        except Exception as exc:  # pragma: no cover - 关机不能因刷盘失败而中断
            logger.debug("flush before shutdown failed: %s", exc)
        time.sleep(0.5)

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
        整个“执行 + 取结果”过程是**串行的**（见 ``_exec_lock`` 的说明）。
        """
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
        # 整段串行化（发消息→收 iopub→取结果）：ZMQ socket 不是线程安全的，
        # 与预热线程或另一个 execute 交叉操作会写坏 wire 协议（实测现象：
        # 内核侧 Invalid Signature / DELIM not in msg_list，客户端 zmq Bad address）。
        with self._exec_lock:
            if not self.is_alive():
                self.start()
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

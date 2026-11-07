# -*- coding: utf-8 -*-
"""GDAL 代码执行器：把 ``[gdal]`` 配方的代码送进独立 venv 里的解释器跑。

为什么不能复用 ``runtime/code_runner.py``：那套执行器从头到尾都是 ArcGIS 的
（持久内核启动时 ``import arcpy``，子进程也走 ``run_arcpy_code``）。GDAL 栈装在
另一个 venv（``.venv-gdal``，见 ``docs/p0-1b-env.md``），两者互不兼容。

所以这里每次起一个**一次性子进程**，但协议与 ``CodeRunner`` 的子进程路径是
**同一份**：driver 脚本直接拿 ``arcpy_bridge._build_runner_script()``，只把
``require_arcpy`` 关掉 —— ``set_result(value)`` 的返回体、stdout/stderr 捕获、
异常回传（``{"type","message","traceback"}``）因此逐项一致，不另立协议。

与 ``CodeRunner`` 有意的三点差别：

1. **cwd 就是 workspace**：GDAL 侧没有 ``arcpy.env.workspace``，代码里的
   ``"input/x.shp"`` 只能靠进程工作目录落到工作区；
2. **清掉指向 venv 之外的 ``PROJ_LIB`` / ``PROJ_DATA`` / ``GDAL_DATA``**：本机
   PostgreSQL 的 proj.db layout 太旧，不清会以 "EPSG 4326 unknown" 的形式炸
   （见 ``docs/p0-1b-env.md`` §4.1）；
3. **超时杀进程树**：GDAL 代码常自己再起子进程（multiprocessing / 命令行），
   只杀 direct child 会留下攥着 .tif / .gpkg 句柄的孤儿，下一次写就 PermissionError。

本模块**不在顶层 import 任何第三方 GIS 包**（也不 import arcpy）：GDAL 栈是可选
依赖，缺了只是 ``[gdal]`` 配方报“需 GDAL 后端”，引擎与 47 个 arcpy 配方照常工作。
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from ..runtime.kernel import ExecResult

logger = logging.getLogger(__name__)

#: 覆盖解释器路径的环境变量（优先于缺省的 ``.venv-gdal``）
PYTHON_ENV_VAR = "GIS_GDAL_PYTHON"

#: 缺省 venv 目录名（DAIR-56 建的，见 docs/p0-1b-env.md）
GDAL_VENV_DIRNAME = ".venv-gdal"

#: 缺环境时指给人看的那份文档
ENV_DOC = "docs/p0-1b-env.md"

#: 这些是**数据目录**而非解释器：指向 venv 之外时必须丢掉（见模块 docstring 第 2 点）
_FOREIGN_ENV_KEYS = ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA")


def _venv_python(venv: Path) -> Path:
    """venv 里的解释器路径（Windows 与 POSIX 两种布局）。"""
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def default_python_executable() -> Path | None:
    """缺省解释器：``GIS_GDAL_PYTHON`` → 项目根 ``.venv-gdal`` → 当前目录 ``.venv-gdal``。

    都找不到返回 ``None``（不抛异常：缺环境是**可预期**的降级，不是 bug）。
    项目根按 ``src/gis_cli/runner/gdal_runner.py`` 往上 4 层推；
    editable 安装与源码树内运行都对，装进 site-packages 时该候选自然落空，
    由 ``GIS_GDAL_PYTHON`` 兜底。
    """
    explicit = os.environ.get(PYTHON_ENV_VAR)
    if explicit:
        return Path(explicit).expanduser()
    for root in (Path(__file__).resolve().parents[3], Path.cwd()):
        candidate = _venv_python(root / GDAL_VENV_DIRNAME)
        if candidate.is_file():
            return candidate
    return None


def unavailable_message(configured: str | None = None) -> str:
    """缺解释器时的准话（不要 ``FileNotFoundError`` 直出）。"""
    detail = f"（{PYTHON_ENV_VAR}={configured}）" if configured else ""
    return (
        f"需 GDAL 后端：未找到 {GDAL_VENV_DIRNAME}，见 {ENV_DOC}{detail}。"
        f"按文档建好 {GDAL_VENV_DIRNAME}，或用环境变量 {PYTHON_ENV_VAR} 指向 GDAL 解释器。"
    )


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """杀掉子进程**及其子孙**，并等到主进程确实退出。

    只 ``proc.kill()`` 会漏掉它自己再起的进程（GDAL 代码常用 multiprocessing /
    命令行子进程），留下攥着 .tif / .gpkg 句柄的孤儿。
    """
    if os.name == "nt":
        # Windows 没有进程组信号；taskkill /T 才会带着子孙一起走
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - 兜底
            logger.warning("taskkill 失败（pid=%s）: %s", proc.pid, exc)
    else:
        # start_new_session=True 让子进程自成进程组，killpg 才能一次带走整棵树
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError) as exc:  # pragma: no cover - 已退出
            logger.warning("killpg 失败（pid=%s）: %s", proc.pid, exc)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - 树没杀干净
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("子进程 %s 杀不掉（已尽力）", proc.pid)


def _drain(proc: subprocess.Popen) -> tuple[str, str]:
    """收尾读取管道；**有界**等待，不被残留孙进程攥住的管道挂死。"""
    try:
        out, err = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - 管道仍被占
        logger.warning("子进程 %s 的管道未关闭，放弃收集输出", proc.pid)
        return "", ""
    return out or "", err or ""


def _build_env(interpreter: Path) -> dict[str, str]:
    """子进程环境：复用 arcpy 侧那份清理，再补上 PROJ/GDAL 数据目录的把关。"""
    from ..arcpy_bridge import _build_subprocess_env

    env = _build_subprocess_env()
    venv_root = str(interpreter.resolve().parent.parent).lower()
    for key in _FOREIGN_ENV_KEYS:
        value = env.get(key)
        if value and not str(Path(value)).lower().startswith(venv_root):
            env.pop(key, None)
    return env


class GdalCodeRunner:
    """在 ``.venv-gdal`` 里跑一次代码；接口与 ``runtime.code_runner.CodeRunner`` 对齐。"""

    def __init__(
        self,
        *,
        workspace: str | Path,
        python_executable: str | Path | None = None,
        exec_timeout: float = 600.0,
    ):
        self.workspace = Path(workspace)
        self.python_executable = str(python_executable) if python_executable else None
        self.exec_timeout = float(exec_timeout)
        self.last_backend = "none"

    # ------------------------------------------------------------- 能力声明
    @property
    def backend(self) -> str:
        """上一次实际走的执行路径（``subprocess`` / ``unavailable`` / ``none``）。"""
        return self.last_backend

    @property
    def backends(self) -> set[str]:
        """这个执行器只能跑 GDAL：配方层的路由就看这一个声明。"""
        return {"gdal"}

    @property
    def interpreter(self) -> Path | None:
        """解析后的解释器路径（可能不存在；不存在时 :meth:`execute` 报明确错误）。"""
        if self.python_executable:
            return Path(self.python_executable).expanduser()
        return default_python_executable()

    # ------------------------------------------------------------------ 执行
    def execute(
        self,
        code: str,
        *,
        timeout: float = 600.0,
        workspace: str | Path | None = None,
    ) -> ExecResult:
        """跑一段代码。失败（含缺环境 / 超时 / 用户代码抛异常）一律走 ``ExecResult``。"""
        started = time.time()
        ws = Path(workspace or self.workspace)
        interpreter = self.interpreter
        if interpreter is None or not interpreter.is_file():
            self.last_backend = "unavailable"
            return ExecResult(
                ok=False,
                error={
                    "type": "GdalBackendUnavailable",
                    "message": unavailable_message(
                        str(interpreter) if interpreter is not None else None
                    ),
                    "traceback": "",
                },
                duration_ms=int((time.time() - started) * 1000),
            )
        ws.mkdir(parents=True, exist_ok=True)
        timeout = float(timeout or self.exec_timeout)
        with tempfile.TemporaryDirectory(prefix="gis-gdal-") as temp_dir:
            temp_path = Path(temp_dir)
            payload_path = temp_path / "payload.json"
            result_path = temp_path / "result.json"
            driver_path = temp_path / "runner.py"
            payload_path.write_text(
                json.dumps(
                    {
                        "code": code,
                        "workspace": str(ws),
                        # 同 CodeRunner 的子进程路径：只差“不 require arcpy”
                        "require_arcpy": False,
                        # 让 .venv-gdal 也能 import gis_cli（driver 里的中文字体配置要用）
                        "python_paths": [str(Path(__file__).resolve().parents[2])],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            from ..arcpy_bridge import _build_runner_script

            driver_path.write_text(_build_runner_script(), encoding="utf-8")
            command = [
                str(interpreter),
                str(driver_path),
                str(payload_path),
                str(result_path),
            ]
            popen_kwargs: dict[str, object] = {}
            if os.name != "nt":
                popen_kwargs["start_new_session"] = True  # 为 killpg 建进程组
            else:
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    cwd=str(ws),
                    env=_build_env(interpreter),
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    **popen_kwargs,
                )
            except OSError as exc:
                self.last_backend = "unavailable"
                return ExecResult(
                    ok=False,
                    error={
                        "type": "InterpreterUnavailable",
                        "message": f"无法启动 GDAL 解释器 {interpreter}: {exc}",
                        "traceback": "",
                    },
                    duration_ms=int((time.time() - started) * 1000),
                )
            self.last_backend = "subprocess"
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_process_tree(proc)
                stdout, stderr = _drain(proc)
                return ExecResult(
                    ok=False,
                    stdout=stdout,
                    stderr=stderr,
                    error={
                        "type": "TimeoutExpired",
                        "message": f"执行超时，超过 {timeout:.0f} 秒。已杀掉子进程树。",
                        "traceback": "",
                    },
                    duration_ms=int((time.time() - started) * 1000),
                )
            # 结果文件在临时目录里 —— 必须在 with 结束前读完
            if result_path.exists():
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    payload = {
                        "status": "error",
                        "stdout": stdout,
                        "stderr": stderr,
                        "data": None,
                        "error": {"type": "RunnerError", "message": f"结果文件无法解析: {exc}"},
                    }
            else:
                payload = {
                    "status": "error",
                    "stdout": stdout,
                    "stderr": stderr,
                    "data": None,
                    "error": {
                        "type": "RunnerError",
                        "message": (
                            "子进程未生成结果文件"
                            f"（exit={proc.returncode}）。stderr 末尾：{(stderr or '').strip()[-500:]}"
                        ),
                    },
                }
        ok = payload.get("status") == "success"
        return ExecResult(
            ok=ok,
            stdout=payload.get("stdout") or "",
            stderr=payload.get("stderr") or stderr or "",
            result=payload.get("data"),
            error=None if ok else (payload.get("error") or {"type": "RunnerError", "message": ""}),
            duration_ms=int((time.time() - started) * 1000),
        )

    def run(
        self,
        code: str,
        *,
        timeout: float | None = None,
        workspace: str | Path | None = None,
    ) -> ExecResult:
        """``recipes/store.py`` 调的是 ``run()``；行为与 :meth:`execute` 相同。"""
        return self.execute(
            code,
            timeout=float(timeout or self.exec_timeout),
            workspace=workspace,
        )

    # ------------------------------------------------------- 与 CodeRunner 同接口
    def reset(self) -> None:
        """一次性子进程没有持久状态可清（接口对齐用）。"""

    def shutdown(self) -> None:
        """一次性子进程没有常驻资源可关（接口对齐用）。"""

# -*- coding: utf-8 -*-
"""GdalCodeRunner：把代码送进独立 GDAL 解释器跑。

**测试不依赖真 GDAL**：协议相关的路径（正常返回 / set_result 缺失 / 抛异常 /
超时被杀 / 解释器不存在）一律用**替身解释器**跑 —— 开发 venv 的 ``sys.executable``，
它同样没装 GDAL 栈，正好证明执行器自己不 import 任何 GIS 包。真 GDAL 只出现在
最后一条集成用例里，``.venv-gdal`` 不在就 skip。

超时用例会真起一个孙进程（``python -c "time.sleep(60)"``），验证杀的是**整棵树**
而不只是 direct child —— 只杀 direct child 会留下攥着 .tif / .gpkg 句柄的孤儿。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gis_cli.recipes.schema import Recipe
from gis_cli.recipes.store import RecipeLibrary
from gis_cli.runner.gdal_runner import (
    GDAL_VENV_DIRNAME,
    PYTHON_ENV_VAR,
    GdalCodeRunner,
    default_python_executable,
    unavailable_message,
)

#: 替身解释器：开发 venv 的 python。协议完全走真 driver，只是没装 GDAL。
STAND_IN = sys.executable


def _runner(tmp_path: Path, **kwargs) -> GdalCodeRunner:
    kwargs.setdefault("python_executable", STAND_IN)
    return GdalCodeRunner(workspace=tmp_path, **kwargs)


def _pid_alive(pid: int) -> bool:
    """进程还活着吗（不依赖 psutil；Windows 用 tasklist，其它用 signal 0）。"""
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
        )
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_dead(pids: list[int], *, budget: float = 5.0) -> list[int]:
    """等这些 pid 退出，返回仍然活着的那些（超预算就带着证据返回）。"""
    deadline = time.time() + budget
    alive = [pid for pid in pids if _pid_alive(pid)]
    while alive and time.time() < deadline:
        time.sleep(0.2)
        alive = [pid for pid in alive if _pid_alive(pid)]
    return alive


# ------------------------------------------------------------------ 能力声明
def test_backends_is_gdal_only(tmp_path):
    """声明成 ``{"gdal"}``：配方层才敢把 ``[gdal]`` 配方交给它。"""
    assert _runner(tmp_path).backends == {"gdal"}


# ---------------------------------------------------- 协议五条路径（替身解释器）
def test_normal_return_captures_result_and_stdout(tmp_path):
    result = _runner(tmp_path).execute(
        "import sys\n"
        "print('hello stdout')\n"
        "print('hello stderr', file=sys.stderr)\n"
        "set_result({'count': 3, 'name': '东区'})\n",
        timeout=60,
    )
    assert result.ok is True, result.error
    assert result.result == {"count": 3, "name": "东区"}
    assert "hello stdout" in result.stdout
    assert "hello stderr" in result.stderr
    assert result.duration_ms >= 0


def test_missing_set_result_is_not_an_error(tmp_path):
    """没调 ``set_result`` = 返回 ``None``，与 CodeRunner 一致（不是失败）。"""
    result = _runner(tmp_path).execute("print('no result here')\n", timeout=60)
    assert result.ok is True, result.error
    assert result.result is None
    assert "no result here" in result.stdout


def test_exception_is_reported_in_same_shape(tmp_path):
    """异常回传 ``{"type","message","traceback"}`` —— 与 CodeRunner 逐项对齐。"""
    result = _runner(tmp_path).execute(
        "print('before')\nraise ValueError('boom 中文')\n", timeout=60
    )
    assert result.ok is False
    error = result.error or {}
    assert error.get("type") == "ValueError"
    assert "boom 中文" in error.get("message", "")
    assert "ValueError" in error.get("traceback", "")
    assert "before" in result.stdout


def test_timeout_kills_whole_process_tree(tmp_path):
    """超时必须杀**整棵树**，并确认进程都退了。"""
    pid_file = tmp_path / "pids.json"
    code = (
        "import json, os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w', encoding='utf-8').write("
        "json.dumps([os.getpid(), child.pid]))\n"
        "time.sleep(60)\n"
    )
    started = time.time()
    result = _runner(tmp_path).execute(code, timeout=3)
    elapsed = time.time() - started
    assert result.ok is False
    assert (result.error or {}).get("type") == "TimeoutExpired"
    # 真被超时打断，而不是别的错误（代码自己要 sleep 60s）
    assert elapsed < 30, f"超时没有生效（耗时 {elapsed:.1f}s）"
    assert pid_file.exists(), "子进程没来得及写 pid 文件，用例前提不成立"
    pids = json.loads(pid_file.read_text(encoding="utf-8"))
    assert _wait_dead(pids) == [], f"进程树没杀干净，仍活着: {_wait_dead(pids)}"


def test_missing_interpreter_gives_clear_message(tmp_path):
    """缺解释器：明确说“需 GDAL 后端”、指到文档，不要 FileNotFoundError。"""
    missing = tmp_path / "no-such-venv" / "python.exe"
    runner = GdalCodeRunner(workspace=tmp_path, python_executable=str(missing))
    result = runner.execute("set_result(1)\n", timeout=30)
    assert result.ok is False
    error = result.error or {}
    assert error.get("type") == "GdalBackendUnavailable"
    assert "需 GDAL 后端" in error.get("message", "")
    assert "docs/p0-1b-env.md" in error.get("message", "")
    assert str(missing) in error.get("message", "")
    # 缺环境不是“跑过又失败”：路径标记要说清楚
    assert runner.backend == "unavailable"


def test_broken_interpreter_does_not_raise(tmp_path):
    """解释器路径存在但不是可执行文件 → ``InterpreterUnavailable``，不抛 OSError。"""
    bogus = tmp_path / ("fake-python.exe" if os.name == "nt" else "fake-python")
    bogus.write_text("not an interpreter\n", encoding="utf-8")
    result = GdalCodeRunner(workspace=tmp_path, python_executable=str(bogus)).execute(
        "set_result(1)\n", timeout=30
    )
    assert result.ok is False
    assert (result.error or {}).get("type") == "InterpreterUnavailable"


# ------------------------------------------------------------ 工作目录 / 环境
def test_workspace_is_process_cwd(tmp_path):
    """相对路径以 workspace 为根（GDAL 侧没有 arcpy.env.workspace）。"""
    ws = tmp_path / "ws"
    result = _runner(tmp_path).execute(
        "import os\n"
        "open('relative_output.txt', 'w', encoding='utf-8').write('ok')\n"
        "set_result({'cwd': os.getcwd()})\n",
        timeout=60,
        workspace=str(ws),
    )
    assert result.ok is True, result.error
    assert Path(result.result["cwd"]).resolve() == ws.resolve()
    assert (ws / "relative_output.txt").read_text(encoding="utf-8") == "ok"


def test_foreign_proj_env_is_dropped(tmp_path, monkeypatch):
    """指向 venv 之外的 PROJ/GDAL 数据目录必须丢掉（docs/p0-1b-env.md §4.1）。"""
    monkeypatch.setenv("PROJ_LIB", r"E:\PostgreSQL\share\contrib\postgis-3.6\proj")
    result = _runner(tmp_path).execute(
        "import os\nset_result({'proj_lib': os.environ.get('PROJ_LIB')})\n", timeout=60
    )
    assert result.ok is True, result.error
    assert result.result == {"proj_lib": None}


def test_proj_env_inside_venv_is_kept(tmp_path, monkeypatch):
    """venv 自己的数据目录要保留 —— 只清“指向外面”的那些。"""
    venv_root = Path(STAND_IN).resolve().parent.parent
    inside = str(venv_root / "share" / "proj")
    monkeypatch.setenv("PROJ_LIB", inside)
    result = _runner(tmp_path).execute(
        "import os\nset_result({'proj_lib': os.environ.get('PROJ_LIB')})\n", timeout=60
    )
    assert result.ok is True, result.error
    assert Path(result.result["proj_lib"]).resolve() == Path(inside).resolve()


# ------------------------------------------------------------ 解释器路径可配
def test_env_var_overrides_interpreter(tmp_path, monkeypatch):
    monkeypatch.setenv(PYTHON_ENV_VAR, STAND_IN)
    runner = GdalCodeRunner(workspace=tmp_path)
    assert runner.interpreter is not None
    assert Path(runner.interpreter).resolve() == Path(STAND_IN).resolve()


def test_default_interpreter_points_at_venv_gdal(tmp_path, monkeypatch):
    """没设环境变量时缺省落到项目根的 ``.venv-gdal``（存在才断言具体路径）。"""
    monkeypatch.delenv(PYTHON_ENV_VAR, raising=False)
    interpreter = default_python_executable()
    if interpreter is None:
        pytest.skip(f"本机没有 {GDAL_VENV_DIRNAME}（见 docs/p0-1b-env.md），只验证不抛异常")
    assert interpreter.name.lower().startswith("python")
    assert GDAL_VENV_DIRNAME in str(interpreter)
    assert interpreter.is_file()


# ------------------------------------------------------------------ 配方路由
def test_store_routes_gdal_recipe_to_gdal_runner(tmp_path):
    """``backend: [gdal]`` 的配方能选中这个执行器，并把代码真正跑起来。"""
    runner = _runner(tmp_path)
    lib = RecipeLibrary(user_dir=tmp_path / "recipes", code_runner=runner)
    lib.load()
    recipe = Recipe(
        id="gdal_probe",
        name="gdal_probe",
        backend=["gdal"],
        # gdal 后端要有自己的模板：只写 code_template 会被 render 当场拒掉
        # （不回落 arcpy 代码，见 Recipe.template_for）。
        code_template_gdal="set_result({'ran': True})\n",
    )
    lib._recipes[recipe.id] = recipe  # 直接放进内存库（不走磁盘）
    ctx = SimpleNamespace(workspace=tmp_path, code_runner=runner, exec_timeout=30)

    observation = lib.run("gdal_probe", {}, ctx)

    assert observation.ok is True, observation.summary
    assert observation.data["backend"] == "gdal"
    assert observation.data["result"] == {"ran": True}
    assert runner.backend == "subprocess"


# ------------------------------------------------------------ 真 GDAL（可跳过）
def test_real_gdal_interpreter_runs_geopandas(tmp_path):
    """``.venv-gdal`` 存在时跑一条真代码；不存在就 skip 并说明原因。"""
    interpreter = default_python_executable()
    if interpreter is None or not Path(interpreter).is_file():
        pytest.skip(
            f"未找到 {GDAL_VENV_DIRNAME} 解释器（见 docs/p0-1b-env.md），跳过真 GDAL 集成用例"
        )
    result = GdalCodeRunner(workspace=tmp_path).execute(
        "import geopandas\n"
        "print(geopandas.__version__)\n"
        "set_result({'geopandas': geopandas.__version__})\n",
        timeout=180,
    )
    assert result.ok is True, result.error
    version = result.result["geopandas"]
    assert version and version[0].isdigit()
    assert version in result.stdout

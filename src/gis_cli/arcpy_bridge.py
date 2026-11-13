# -*- coding: utf-8 -*-
"""ArcPy Bridge - 桥接 AI Agent 与 ArcGIS Pro Python 环境

参考 ArcGIS-Pro-Bridge-MCP-Server 实现
通过子进程调用 ArcGIS Pro 自带的 Python 环境执行 ArcPy 代码
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict, replace
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from typing import Any

try:
    import winreg
except ImportError:
    winreg = None


# === Constants ===

DEFAULT_TIMEOUT_SECONDS = 300

#: ``propy.bat`` 兜底探活的超时：真 ``import arcpy`` 冷启动实测 11.9s，给足余量。
LAUNCHER_PROBE_TIMEOUT_SECONDS = 180

ARCGIS_REGISTRY_PATHS = (
    r"SOFTWARE\ESRI\ArcGISPro",
    r"SOFTWARE\WOW6432Node\ESRI\ArcGISPro",
)
#: 版本号文本里的数字串（用于把候选按版本降序排：多版本并存选最高）。
_VERSION_IN_TEXT = re.compile(r"(\d+(?:\.\d+)*)")

#: 注册表同一个键下的四个值**都要**读：升级安装后 ``InstallDir`` / ``PythonCondaRoot``
#: 可能仍指向**已被删除**的旧版本，而 ``Version`` / ``PythonCondaEnv`` 是好的
#: （本机 3.7 实测：``InstallDir=E:\ArcGISPro3.6\``，``Version=3.7.0.1901``）。
REGISTRY_VALUE_NAMES = ("InstallDir", "PythonCondaRoot", "PythonCondaEnv", "Version")

#: 安装目录 → conda 环境 → 解释器。环境名由注册表 ``PythonCondaEnv`` 给，
#: **不能写死** ``arcgispro-py3``：用户克隆环境（``arcgispro-py3-clone``）是 Pro 的常规操作。
ARCGIS_PYTHON_ENV_SUBDIR = Path("bin/Python/envs")
ARCGIS_PYTHON_EXE_NAME = "python.exe"
DEFAULT_CONDA_ENV = "arcgispro-py3"
#: ``propy.bat`` 只当**最后**兜底：实测 3.7 上它会因注册表指向旧版本而落到系统
#: ``C:\Program Files\python\python.exe``（那里没有 arcpy），所以必须强制探活。
ARCGIS_PROPy_LAUNCHER = Path("bin/Python/scripts/propy.bat")
#: 「这个目录真的是 Pro」的判据（Pro 3.x 实测存在），**不启子进程**即可排除伪候选。
ARCGIS_INSTALL_MARKER = Path("Resources/ArcPy/arcpy/__init__.py")
#: 目录探测的已知根目录：只扫**一层** glob，禁止递归、禁止全盘扫
#: （本仓库有 ``find`` 挂死前科，见 AGENTS.md）。
ARCGIS_PRO_SCAN_ROOTS = (
    Path("E:/"),
    Path("D:/"),
    Path("E:/ArcGIS"),
    Path("D:/ArcGIS"),
    Path("C:/Program Files/ArcGIS"),
    Path("C:/Program Files (x86)/ArcGIS"),
)
ARCGIS_PRO_SCAN_PATTERNS = ("ArcGISPro*", "Pro*")
#: 目录探测只认这种名字（``ArcGISPro3.7`` / ``Pro`` / ``Pro3.7``）。
#: 盘根下的 ``Pro*`` 会把 ``Program Files`` / ``ProgramData`` 也匹配进来，
#: 它们显然不是 Pro，不能进候选表污染 doctor 的输出。
ARCGIS_PRO_DIR_NAME = re.compile(r"^(?:arcgispro|pro)[\d.\-_]*$", re.IGNORECASE)
#: 老候选表里的默认安装位置：目录探测的 glob 覆盖不到两级布局的个别机器，兜一下。
ARCGIS_PRO_COMPAT_DIRS = (
    Path(r"C:\Program Files\ArcGIS\Pro"),
    Path(r"D:\ArcGIS\Pro"),
)
ASIA_NORTH_ALBERS_EQUAL_AREA_CONIC_WKT = (
    'PROJCS["Asia_North_Albers_Equal_Area_Conic",'
    'GEOGCS["GCS_China_Geodetic_Coordinate_System_2000",'
    'DATUM["D_China_2000",SPHEROID["CGCS2000",6378137.0,298.257222101]],'
    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Albers"],'
    'PARAMETER["False_Easting",0.0],'
    'PARAMETER["False_Northing",0.0],'
    'PARAMETER["Central_Meridian",105.0],'
    'PARAMETER["Standard_Parallel_1",25.0],'
    'PARAMETER["Standard_Parallel_2",47.0],'
    'PARAMETER["Latitude_Of_Origin",0.0],'
    'UNIT["Meter",1.0]]'
)


# === Data Classes ===

@dataclass
class ArcGISPythonInfo:
    """ArcGIS Pro Python 环境信息"""
    install_dir: str
    python_executable: str
    source: str


@dataclass
class ArcPyExecutionResult:
    """ArcPy 执行结果"""
    status: str  # 'success' or 'error'
    exit_code: int
    python_executable: str
    stdout: str
    stderr: str
    data: Any | None = None
    error: dict[str, Any] | None = None
    hint: str | None = None


class ArcGISDiscoveryError(RuntimeError):
    """未能发现 ArcGIS Pro 环境"""
    pass


# === Environment Discovery ===

@dataclass(frozen=True)
class _InstallHint:
    """一条「Pro 可能装在哪」的线索（环境变量 / 注册表 / 目录探测各给一条）。"""

    source: str
    install_dir: str
    conda_env: str = DEFAULT_CONDA_ENV
    version: tuple[int, ...] = ()
    #: 环境变量直指解释器时用（不能拿 ``<install>\envs\<env>`` 拼，路径可能不是标准布局）
    python_executable: str = ""
    #: 用户显式指定（环境变量）：永远排在其他候选前面
    explicit: bool = False


@dataclass(frozen=True)
class _PythonCandidate:
    """一个解释器候选；``launcher`` 为真表示这是 ``propy.bat``，必须探活。"""

    source: str
    python_path: str
    install_dir: str
    version: tuple[int, ...] = ()
    explicit: bool = False
    launcher: bool = False


def _normalize_path(value: str | os.PathLike[str]) -> str:
    """标准化路径"""
    return str(Path(value).expanduser().resolve(strict=False))


def _version_key(text: str) -> tuple[int, ...]:
    """把 ``3.7`` / ``3.7.0.1901`` / ``ArcGISPro3.7`` 里的版本号抠成可排序的元组。

    抠不到（如 ``Pro``）返回空元组：排序时自然排在有版本号的后面。
    """
    match = _VERSION_IN_TEXT.search(str(text))
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split(".") if part.isdigit())


def _guess_install_dir_from_python(python_path: str | os.PathLike[str]) -> str:
    """从解释器路径推断 ArcGIS Pro 安装目录。

    ``<install>\\bin\\Python\\envs\\<环境名>\\python.exe`` 是 Pro 的固定布局；
    环境名**不能写死**（用户克隆的 ``arcgispro-py3-clone`` 是常规操作）。
    """
    normalized_path = Path(_normalize_path(python_path))
    parts = normalized_path.parts
    lowered = [part.lower() for part in parts]
    for index in range(len(parts) - 4, 0, -1):
        if tuple(lowered[index:index + 3]) == ("bin", "python", "envs"):
            return _normalize_path(Path(*parts[:index]))
    return _normalize_path(normalized_path.parent)


def _iter_env_install_hints() -> list[_InstallHint]:
    """从环境变量找安装线索。

    ``ARCGIS_PRO_PYTHON`` 是硬覆盖：显式指定的解释器永远排在候选第一位
    （即使它的安装目录不是标准布局、即使同机还装着别的 Pro）。
    """
    hints: list[_InstallHint] = []

    explicit_python = os.environ.get("ARCGIS_PRO_PYTHON")
    if explicit_python:
        normalized = _normalize_path(Path(explicit_python).expanduser())
        hints.append(_InstallHint(
            source="env:ARCGIS_PRO_PYTHON",
            install_dir=_guess_install_dir_from_python(normalized),
            python_executable=normalized,
            explicit=True,
        ))

    install_dir = os.environ.get("ARCGIS_PRO_INSTALL_DIR")
    if install_dir:
        hints.append(_InstallHint(
            source="env:ARCGIS_PRO_INSTALL_DIR",
            install_dir=_normalize_path(Path(install_dir).expanduser()),
            explicit=True,
        ))

    return hints


def _install_dir_from_install_value(value: str | None) -> str:
    """注册表 ``InstallDir`` → 安装目录（原样标准化）。"""
    if not value:
        return ""
    return _normalize_path(Path(str(value)).expanduser())


def _install_dir_from_conda_root(value: str | None) -> str:
    r"""``PythonCondaRoot``（``<安装目录>\bin\Python``）→ 安装目录。

    形状不对就返回空串（宁缺毋滥）：宁可少一条候选，也不要拿个半截路径
    去拼 ``<install>\bin\Python\envs\...``。
    """
    if not value:
        return ""
    path = Path(_normalize_path(Path(str(value)).expanduser()))
    if tuple(part.lower() for part in path.parts[-2:]) == ("bin", "python"):
        return _normalize_path(path.parents[1])
    return ""


def _iter_registry_install_dirs() -> list[_InstallHint]:
    r"""从 Windows 注册表查找安装线索。

    只信 ``InstallDir`` 是 3.7 适配前的老病根：升级安装后它可能仍指向**已被删除**
    的旧版本（本机 ``E:\ArcGISPro3.6\``，而 3.7 实际装在 ``E:\ArcGISPro3.7``）。
    所以同一个键下的 ``PythonCondaRoot`` / ``PythonCondaEnv`` / ``Version`` 一起读：
    其中任何一个坏了，还有别的线索可用。
    """
    if winreg is None:
        return []

    discovered: list[_InstallHint] = []
    seen: set[tuple[str, str]] = set()
    registry_views = [0]
    key_read = getattr(winreg, "KEY_READ", 0)

    for extra_flag_name in ("KEY_WOW64_64KEY", "KEY_WOW64_32KEY"):
        extra_flag = getattr(winreg, extra_flag_name, 0)
        if extra_flag:
            registry_views.append(extra_flag)

    for registry_path in ARCGIS_REGISTRY_PATHS:
        for view_flag in registry_views:
            values: dict[str, str] = {}
            try:
                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE, registry_path, 0, key_read | view_flag
                ) as key:
                    for value_name in REGISTRY_VALUE_NAMES:
                        try:
                            values[value_name], _ = winreg.QueryValueEx(key, value_name)
                        except OSError:
                            continue
            except OSError:
                continue
            if not values:
                continue

            version = _version_key(values.get("Version", ""))
            conda_env = str(values.get("PythonCondaEnv") or "").strip() or DEFAULT_CONDA_ENV
            for value_name, install_dir in (
                ("InstallDir", _install_dir_from_install_value(values.get("InstallDir"))),
                ("PythonCondaRoot", _install_dir_from_conda_root(values.get("PythonCondaRoot"))),
            ):
                if not install_dir or (value_name, install_dir) in seen:
                    continue
                seen.add((value_name, install_dir))
                discovered.append(_InstallHint(
                    source=f"registry:{registry_path}:{value_name}",
                    install_dir=install_dir,
                    conda_env=conda_env,
                    version=version,
                ))

    return discovered


def _iter_filesystem_install_dirs() -> list[_InstallHint]:
    """目录探测：已知根目录 + **一层** glob。

    禁止递归、禁止全盘扫（本仓库有 ``find`` 挂死前科）。
    """
    hints: list[_InstallHint] = []
    seen: set[str] = set()

    for root in ARCGIS_PRO_SCAN_ROOTS:
        for pattern in ARCGIS_PRO_SCAN_PATTERNS:
            try:
                matches = sorted(root.glob(pattern))
            except OSError:  # pragma: no cover - 盘符不存在 / 无权限
                continue
            for match in matches:
                if not ARCGIS_PRO_DIR_NAME.match(match.name):
                    continue
                if not match.is_dir():
                    continue
                normalized = _normalize_path(match)
                if normalized in seen:
                    continue
                seen.add(normalized)
                hints.append(_InstallHint(
                    source=f"filesystem:scan:{normalized}",
                    install_dir=normalized,
                    version=_version_key(match.name),
                ))

    for compat in ARCGIS_PRO_COMPAT_DIRS:
        normalized = _normalize_path(compat)
        if normalized in seen:
            continue
        seen.add(normalized)
        hints.append(_InstallHint(
            source="filesystem:compat",
            install_dir=normalized,
            version=_version_key(Path(normalized).name),
        ))

    return hints


def _python_paths_for(hint: _InstallHint) -> list[str]:
    """一条安装线索 → 可能的解释器路径。

    只收 ``python.exe``：``propy.bat`` 是另一个东西（启动器），见
    ``_iter_python_candidates`` 的 launcher 项。
    """
    if hint.python_executable:
        return [hint.python_executable]

    env_root = Path(hint.install_dir) / ARCGIS_PYTHON_ENV_SUBDIR
    env_names: list[str] = [hint.conda_env] if hint.conda_env else []
    try:
        # 克隆环境（arcgispro-py3-clone）也要收：注册表里的环境名可能早就不用了
        for child in sorted(env_root.iterdir()):
            if child.is_dir() and child.name not in env_names:
                env_names.append(child.name)
    except OSError:
        pass
    return [_normalize_path(env_root / name / ARCGIS_PYTHON_EXE_NAME) for name in env_names]


def _iter_python_candidates() -> list[_PythonCandidate]:
    """所有解释器候选，**按优先级排序**（不检查存在性、不启子进程）。

    顺序：环境变量（硬覆盖）→ 注册表（版本降序，同版本内 ``InstallDir`` 先于
    ``PythonCondaRoot``）→ 目录探测（版本降序：多版本并存选最高）→ ``propy.bat``
    最后兜底。适配前的旧候选只增不减，相对顺序不变。
    """
    candidates: list[_PythonCandidate] = []
    seen: set[str] = set()

    def _add(hint: _InstallHint, python_path: str, *, launcher: bool = False) -> None:
        normalized = _normalize_path(python_path)
        if normalized in seen:
            return
        seen.add(normalized)
        candidates.append(_PythonCandidate(
            source=hint.source,
            python_path=normalized,
            install_dir=_normalize_path(hint.install_dir),
            version=hint.version,
            explicit=hint.explicit,
            launcher=launcher,
        ))

    registry_hints = sorted(
        _iter_registry_install_dirs(), key=lambda hint: hint.version, reverse=True
    )
    filesystem_hints = sorted(
        _iter_filesystem_install_dirs(), key=lambda hint: hint.version, reverse=True
    )

    for hint in (*_iter_env_install_hints(), *registry_hints, *filesystem_hints):
        for python_path in _python_paths_for(hint):
            _add(hint, python_path)

    # 启动器最后兜底（discover 会强制探活，见 _probe_launcher）
    for hint in (*registry_hints, *filesystem_hints):
        _add(
            replace(hint, source=f"launcher:propy.bat:{hint.install_dir}"),
            str(Path(hint.install_dir) / ARCGIS_PROPy_LAUNCHER),
            launcher=True,
        )

    return candidates


def _build_python_candidates() -> list[tuple[str, str, str]]:
    """``(来源, 解释器, 安装目录)`` 候选表，按优先级排序。

    只收 ``python.exe``（``propy.bat`` 不算解释器）。doctor 用这份表列
    「这台机器上还装着哪些 Pro」。
    """
    return [
        (candidate.source, candidate.python_path, candidate.install_dir)
        for candidate in _iter_python_candidates()
        if not candidate.launcher
    ]


def _is_pro_install(install_dir: str | os.PathLike[str]) -> bool:
    """这个目录是不是**真的** Pro 安装（看 arcpy 包本体，不启子进程）。"""
    return (Path(install_dir) / ARCGIS_INSTALL_MARKER).is_file()


def discovery_routes() -> list[str]:
    r"""「试过哪几条路」的人类可读清单。

    doctor / 持久内核 / 发现失败异常**共用这一份**，避免三处文案各自漂移
    （历史问题：doctor 那份写死了 ``E:\ArcGISPro3.6``，3.7 升级后就在误导人）。
    """
    roots = "、".join(str(root) for root in ARCGIS_PRO_SCAN_ROOTS)
    return [
        "环境变量 ARCGIS_PRO_PYTHON / ARCGIS_PRO_INSTALL_DIR",
        r"注册表 SOFTWARE\ESRI\ArcGISPro 的 InstallDir / PythonCondaRoot / PythonCondaEnv",
        f"目录探测 {roots} 下的 ArcGISPro* / Pro*（各一层）",
        r"启动器兜底 <安装目录>\bin\Python\scripts\propy.bat（必须能 import arcpy）",
    ]


def _discovery_error() -> ArcGISDiscoveryError:
    """发现失败：把试过的路说清（这四个字直接告诉用户下一步改哪里）。"""
    return ArcGISDiscoveryError(
        "未找到 ArcGIS Pro Python 解释器。已试过："
        + "；".join(discovery_routes())
        + "。装在别处就用环境变量指定："
        r"set ARCGIS_PRO_PYTHON=<安装目录>\bin\Python\envs\arcgispro-py3\python.exe"
    )


def _probe_launcher(candidate: _PythonCandidate) -> ArcGISPythonInfo | None:
    r"""``propy.bat`` 兜底：**必须**真能 ``import arcpy`` 才算数。

    实测 3.7 的 ``propy.bat`` 会因注册表指向旧版本而落到系统
    ``C:\Program Files\python\python.exe``；那里没有 arcpy 却照样能跑完
    用户代码，``require_arcpy=False`` 时就是**在错误的解释器里静默“成功”**。
    """
    code = dedent('''
        import arcpy
        info = arcpy.GetInstallInfo()
        set_result({
            "install_dir": info.get("InstallDir", ""),
            "version": info.get("Version", ""),
        })
    ''')
    result = run_arcpy_code(
        code,
        python_executable=candidate.python_path,
        timeout_seconds=LAUNCHER_PROBE_TIMEOUT_SECONDS,
    )
    if result.status != "success" or not isinstance(result.data, dict):
        return None
    return ArcGISPythonInfo(
        install_dir=_normalize_path(result.data.get("install_dir") or candidate.install_dir),
        python_executable=candidate.python_path,
        source=candidate.source,
    )


@lru_cache(maxsize=1)
def discover_arcgis_pro_python() -> ArcGISPythonInfo:
    r"""自动发现 ArcGIS Pro Python 解释器。

    四级：① 环境变量（``ARCGIS_PRO_PYTHON`` 硬覆盖，永远第一）
    → ② 注册表（``InstallDir`` / ``PythonCondaRoot`` + ``PythonCondaEnv``，版本降序）
    → ③ 已知根目录的一层 glob（本机 ``E:\ArcGISPro3.7`` 靠这条命中）
    → ④ ``propy.bat`` 兜底（强制探活）。

    ①②③ 内部再分两轮：先只认「``Resources\ArcPy\arcpy\__init__.py`` 在」的真 Pro
    （不启子进程），都不合格才退回「``python.exe`` 存在即可」的老口径 —— 既排除
    残留目录里的伪候选，又不会在非标准布局的机器上突然找不到 Pro。
    """
    all_candidates = _iter_python_candidates()
    python_candidates = [candidate for candidate in all_candidates if not candidate.launcher]
    explicit = [candidate for candidate in python_candidates if candidate.explicit]
    discovered = [candidate for candidate in python_candidates if not candidate.explicit]

    for group in (
        explicit,
        [candidate for candidate in discovered if _is_pro_install(candidate.install_dir)],
        discovered,
    ):
        for candidate in group:
            if Path(candidate.python_path).exists():
                return ArcGISPythonInfo(
                    install_dir=candidate.install_dir,
                    python_executable=candidate.python_path,
                    source=candidate.source,
                )

    for candidate in all_candidates:
        if candidate.launcher and Path(candidate.python_path).exists():
            info = _probe_launcher(candidate)
            if info is not None:
                return info

    raise _discovery_error()


def clear_discovery_cache() -> None:
    """清理发现缓存"""
    discover_arcgis_pro_python.cache_clear()


# === Runner Script ===

def _build_runner_script() -> str:
    """生成在 ArcGIS Python 环境中执行的包装脚本"""
    return dedent('''
        # -*- coding: utf-8 -*-
        from __future__ import annotations

        import contextlib
        import io
        import json
        import os
        import sys
        import traceback
        from pathlib import Path

        # Read payload
        payload_path = Path(sys.argv[1])
        result_path = Path(sys.argv[2])

        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        code = payload["code"]
        workspace = payload.get("workspace")
        require_arcpy = payload.get("require_arcpy", True)
        python_paths = payload.get("python_paths", [])

        # Add extra Python paths for importing helper modules
        for p in python_paths:
            if p not in sys.path:
                sys.path.insert(0, p)

        # Prepare execution namespace
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        namespace = {
            "__name__": "__main__",
            "__file__": str(payload_path.with_name("user_code.py")),
        }
        namespace["__arcgis_result__"] = None
        namespace["__result__"] = None  # 兼容旧代码

        def set_result(value):
            namespace["__arcgis_result__"] = value

        namespace["set_result"] = set_result
        error = None
        status = "success"

        # 中文字体/绘图环境：与持久内核保持一致（matplotlib 默认字体无中文字形，
        # 自绘图里的中文会变方框）。缺 matplotlib 时静默跳过。
        try:
            from gis_cli.runtime.mpl_setup import configure as _mpl_configure
            _mpl_configure()
        except Exception:
            pass

        try:
            if require_arcpy:
                import arcpy
                namespace["arcpy"] = arcpy
                arcpy.env.overwriteOutput = True
                if workspace:
                    arcpy.env.workspace = workspace

            with (
                contextlib.redirect_stdout(stdout_buffer),
                contextlib.redirect_stderr(stderr_buffer),
            ):
                exec(compile(code, namespace["__file__"], "exec"), namespace, namespace)
                
        except Exception as exc:
            status = "error"
            error = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }

        # Write result
        result = {
            "status": status,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "data": namespace.get("__arcgis_result__"),
            "error": error,
            "workspace": workspace,
        }
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        sys.exit(0 if status == "success" else 1)
    ''').strip()


def _build_subprocess_env() -> dict[str, str]:
    """构建子进程环境变量，移除可能干扰 ArcPy 的变量"""
    env = dict(os.environ)
    
    # 移除可能干扰的环境变量
    drop_keys = {
        "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__"
    }
    drop_prefixes = ("UV_", "TRAE_")
    
    for key in list(env.keys()):
        if key in drop_keys or any(key.startswith(p) for p in drop_prefixes):
            env.pop(key, None)
    
    # 设置 UTF-8 编码
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    
    return env


def _build_execution_hint(stderr: str, error: dict[str, Any] | None) -> str | None:
    """根据错误信息生成提示"""
    combined = "\n".join(filter(None, [
        stderr,
        (error or {}).get("message", ""),
        (error or {}).get("traceback", "")
    ]))
    lowered = combined.lower()
    
    if "schema lock" in lowered or "cannot acquire a lock" in lowered:
        return "检测到数据锁定问题，请关闭占用该数据的图层或编辑会话后重试。"
    if "license" in lowered and "not available" in lowered:
        return "检测到许可不可用，请确认 ArcGIS Pro 已登录并具有对应工具许可。"
    if "module not found" in lowered and "arcpy" in lowered:
        return "无法导入 arcpy，请确认使用的是 ArcGIS Pro 自带 Python 环境。"
    if "feature class" in lowered and "does not exist" in lowered:
        return "要素类不存在，请检查数据路径是否正确。"
    
    return None


# === Main Execution Function ===

def run_arcpy_code(
    code: str,
    *,
    workspace: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    python_executable: str | None = None,
    require_arcpy: bool = True,
) -> ArcPyExecutionResult:
    """在 ArcGIS Pro Python 环境中执行代码
    
    Args:
        code: 要执行的 Python/ArcPy 代码
        workspace: ArcPy 工作空间路径
        timeout_seconds: 超时时间
        python_executable: 指定 Python 解释器路径
        require_arcpy: 是否需要导入 arcpy
        
    Returns:
        ArcPyExecutionResult 包含执行状态、输出和数据
    """
    # 获取 Python 解释器
    resolved_python = python_executable
    if resolved_python is None:
        try:
            info = discover_arcgis_pro_python()
            resolved_python = info.python_executable
        except ArcGISDiscoveryError as e:
            return ArcPyExecutionResult(
                status="error",
                exit_code=-1,
                python_executable="",
                stdout="",
                stderr=str(e),
                error={"type": "DiscoveryError", "message": str(e)},
                hint=(
                    "ArcGIS Pro 未安装或装在非默认位置：用 ARCGIS_PRO_PYTHON 指定解释器，"
                    "或跑 gis-agent doctor 看已试过哪些路径。"
                )
            )
    
    # 创建临时目录执行
    with tempfile.TemporaryDirectory(prefix="gis-agent-") as temp_dir:
        temp_path = Path(temp_dir)
        payload_path = temp_path / "payload.json"
        result_path = temp_path / "result.json"
        runner_path = temp_path / "runner.py"
        
        # 写入 payload
        arcpy_bridge_dir = Path(__file__).parent.resolve()
        src_dir = arcpy_bridge_dir.parent  # src/gis_cli/ 的父级 → src/
        payload = {
            "code": code,
            "workspace": workspace,
            "require_arcpy": require_arcpy,
            "python_paths": [str(src_dir)],
        }
        payload_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        # 写入 runner 脚本
        runner_path.write_text(_build_runner_script(), encoding="utf-8")
        
        # 执行子进程
        try:
            completed = subprocess.run(
                [resolved_python, str(runner_path), str(payload_path), str(result_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
                cwd=str(temp_path),
                env=_build_subprocess_env(),
                timeout=timeout_seconds,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            return ArcPyExecutionResult(
                status="error",
                exit_code=-1,
                python_executable=_normalize_path(resolved_python),
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                error={
                    "type": "TimeoutExpired",
                    "message": f"执行超时，超过 {timeout_seconds} 秒。"
                },
                hint="请缩小处理范围或提高 timeout_seconds。"
            )
        
        # 读取结果
        if result_path.exists():
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result_payload = {
                "status": "error",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "data": None,
                "error": {
                    "type": "RunnerError",
                    "message": "子进程未生成结果文件。"
                },
                "workspace": workspace,
            }
        
        hint = _build_execution_hint(
            result_payload.get("stderr", ""),
            result_payload.get("error")
        )
        
        return ArcPyExecutionResult(
            status=result_payload.get("status", "error"),
            exit_code=completed.returncode,
            python_executable=_normalize_path(resolved_python),
            stdout=result_payload.get("stdout", ""),
            stderr=result_payload.get("stderr", completed.stderr),
            data=result_payload.get("data"),
            error=result_payload.get("error"),
            hint=hint,
        )


def result_to_dict(result: ArcPyExecutionResult) -> dict[str, Any]:
    """将执行结果转换为字典"""
    return asdict(result)


# === Convenience Functions ===

def check_arcpy_available() -> ArcPyExecutionResult:
    """检查 ArcPy 是否可用"""
    code = dedent('''
        import arcpy
        info = {
            "version": arcpy.GetInstallInfo()["Version"],
            "product": arcpy.GetInstallInfo()["ProductName"],
            "license": arcpy.ProductInfo(),
        }
        set_result(info)
        print(f"ArcGIS Pro {info['version']} - {info['license']}")
    ''')
    return run_arcpy_code(code, timeout_seconds=60)


def scan_workspace_layers(workspace: str) -> ArcPyExecutionResult:
    """扫描工作空间中的图层"""
    code = dedent(f'''
        import arcpy
        import os
        
        workspace = r"{workspace}"
        arcpy.env.workspace = workspace
        
        layers = []
        
        def _read_layer_info(path, name):
            """Extract full layer metadata including fields and CRS."""
            desc = arcpy.Describe(path)
            sr = desc.spatialReference
            fields = []
            for f in arcpy.ListFields(path):
                info = {{
                    "name": f.name,
                    "type": f.type,
                }}
                # Sample values for non-geometry, non-OID fields
                if f.type not in ("Geometry", "OID", "Blob", "Raster"):
                    try:
                        samples = []
                        with arcpy.da.SearchCursor(path, [f.name]) as cursor:
                            for row in cursor:
                                val = row[0]
                                if val is not None and str(val).strip():
                                    samples.append(str(val))
                                    if len(samples) >= 3:
                                        break
                            del cursor
                        if samples:
                            info["samples"] = samples
                    except Exception:
                        pass
                fields.append(info)
            _shape_type = desc.shapeType if hasattr(desc, "shapeType") else None
            try:
                _cnt = int(arcpy.management.GetCount(path).getOutput(0))
            except Exception:
                _cnt = None
            return {{
                "name": name,
                "type": _shape_type,
                "path": path,
                "feature_type": "FeatureClass",
                "spatial_reference": sr.name if sr else None,
                "wkid": sr.factoryCode if sr else None,
                "has_z": desc.hasZ,
                "has_m": desc.hasM,
                "fields": fields,
                "feature_count": _cnt,
            }}

        # 扫描要素类
        for fc in arcpy.ListFeatureClasses() or []:
            try:
                full_path = os.path.join(workspace, fc)
                layers.append(_read_layer_info(full_path, fc))
            except:
                pass

        # 递归扫描子目录
        for dirpath, dirnames, filenames in arcpy.da.Walk(workspace, datatype="FeatureClass"):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                if full_path not in [l["path"] for l in layers]:
                    try:
                        layers.append(_read_layer_info(full_path, filename))
                    except:
                        pass
        
        set_result({{"layers": layers, "count": len(layers), "workspace": workspace}})
        print(f"Found {{len(layers)}} layers in {{workspace}}")
    ''')
    return run_arcpy_code(code, workspace=workspace, timeout_seconds=120)


def merge_layers(
    input_layers: list[str],
    output_path: str,
    workspace: str | None = None,
    overwrite_output: bool = False,
) -> ArcPyExecutionResult:
    """合并多个图层"""
    # 转义路径
    inputs_str = "; ".join(input_layers)
    
    code = dedent(f'''
        import arcpy
        
        input_layers = r"{inputs_str}".split("; ")
        output_path = r"{output_path}"
        overwrite_output = {repr(bool(overwrite_output))}

        if overwrite_output and arcpy.Exists(output_path):
            arcpy.management.Delete(output_path)
        
        # 执行合并
        arcpy.management.Merge(input_layers, output_path)
        
        # 获取结果信息
        count = int(arcpy.management.GetCount(output_path)[0])
        desc = arcpy.Describe(output_path)
        
        result = {{
            "output_path": output_path,
            "feature_count": count,
            "geometry_type": desc.shapeType,
            "input_count": len(input_layers),
        }}
        set_result(result)
        print(f"Merged {{len(input_layers)}} layers into {{output_path}} ({{count}} features)")
    ''')
    return run_arcpy_code(code, workspace=workspace, timeout_seconds=300)


def quality_check(input_path: str) -> ArcPyExecutionResult:
    """执行质量检查"""
    code = dedent(f'''
        import arcpy
        
        input_path = r"{input_path}"
        
        # 检查几何有效性
        desc = arcpy.Describe(input_path)
        count = int(arcpy.management.GetCount(input_path)[0])
        
        # 检查空几何
        null_geometry = 0
        invalid_geometry = 0
        
        with arcpy.da.SearchCursor(input_path, ["SHAPE@"]) as cursor:
            for row in cursor:
                if row[0] is None:
                    null_geometry += 1
                elif not row[0].isMultipart and row[0].partCount == 0:
                    invalid_geometry += 1
        
        result = {{
            "input_path": input_path,
            "feature_count": count,
            "geometry_type": desc.shapeType,
            "null_geometry": null_geometry,
            "invalid_geometry": invalid_geometry,
            "spatial_reference": desc.spatialReference.name if desc.spatialReference else None,
            "is_valid": null_geometry == 0 and invalid_geometry == 0,
        }}
        set_result(result)
        
        status = "VALID" if result["is_valid"] else "ISSUES FOUND"
        print(f"Quality check: {{status}} - {{count}} features, {{null_geometry}} null, {{invalid_geometry}} invalid")
    ''')
    return run_arcpy_code(code, timeout_seconds=180)


def project_layer(
    input_path: str,
    output_path: str,
    target_srs: str = "CGCS2000",
    overwrite_output: bool = False,
) -> ArcPyExecutionResult:
    """投影变换"""
    code = dedent(f'''
        import arcpy
        
        input_path = r"{input_path}"
        output_path = r"{output_path}"
        target_srs = "{target_srs}"
        overwrite_output = {repr(bool(overwrite_output))}
        
        # 获取目标坐标系（兼容名称/EPSG/WKID）
        target_srs_norm = str(target_srs).strip()
        target_srs_upper = target_srs_norm.upper()
        wkid_map = {{
            "CGCS2000": 4490,
            "WGS84": 4326,
            "WGS_1984": 4326,
            "EPSG:4326": 4326,
            "WGS_1984_WEB_MERCATOR_AUXILIARY_SPHERE": 3857,
            "WGS 1984 WEB MERCATOR AUXILIARY SPHERE": 3857,
            "WEB_MERCATOR": 3857,
            "EPSG:3857": 3857,
            "ASIA_NORTH_ALBERS_EQUAL_AREA_CONIC": 102025,
        }}
        known_wkt_map = {{
            "ASIA_NORTH_ALBERS_EQUAL_AREA_CONIC": {json.dumps(ASIA_NORTH_ALBERS_EQUAL_AREA_CONIC_WKT)},
            "ASIA NORTH ALBERS EQUAL AREA CONIC": {json.dumps(ASIA_NORTH_ALBERS_EQUAL_AREA_CONIC_WKT)},
        }}

        wkid = wkid_map.get(target_srs_upper)
        if wkid is None and target_srs_upper.isdigit():
            wkid = int(target_srs_upper)
        if wkid is None and target_srs_upper.startswith("EPSG:") and target_srs_upper.split(":", 1)[1].isdigit():
            wkid = int(target_srs_upper.split(":", 1)[1])

        if wkid is not None:
            out_sr = arcpy.SpatialReference(wkid)
        else:
            known_wkt = known_wkt_map.get(target_srs_upper)
            if known_wkt:
                out_sr = arcpy.SpatialReference()
                out_sr.loadFromString(known_wkt)
            else:
                out_sr = arcpy.SpatialReference()
                try:
                    out_sr.loadFromString(target_srs_norm)
                except Exception:
                    try:
                        out_sr = arcpy.SpatialReference(target_srs_norm)
                    except Exception as _sr_err:
                        raise ValueError(
                            f"无法解析目标坐标系: {{target_srs_norm}}。请使用 EPSG 代码、WKT，或 Asia_North_Albers_Equal_Area_Conic"
                        ) from _sr_err
        
        # 执行投影
        if overwrite_output and arcpy.Exists(output_path):
            arcpy.management.Delete(output_path)
        arcpy.management.Project(input_path, output_path, out_sr)
        
        # 获取结果信息
        count = int(arcpy.management.GetCount(output_path)[0])
        desc = arcpy.Describe(output_path)
        
        result = {{
            "input_path": input_path,
            "output_path": output_path,
            "feature_count": count,
            "target_srs": desc.spatialReference.name,
        }}
        set_result(result)
        print(f"Projected to {{desc.spatialReference.name}}: {{count}} features")
    ''')
    return run_arcpy_code(code, timeout_seconds=300)


def _graduated_renderer_block(field_name: str, input_path: str, color_ramp_name: str) -> str:
    """分级设色渲染片段（从大模板抽出，便于与分类设色共用布局/工程逻辑）。"""
    return f'''
sym = layer.symbology
sym.updateRenderer("GraduatedColorsRenderer")
# 字段名容错：大小写/下划线差异时自动匹配真实字段
_field_name = "{field_name}".strip()
_all_fields = [f.name for f in arcpy.ListFields(r"{input_path}")]
if _field_name:
    _field_norm = _field_name.lower().replace("_", "")
    for _f in _all_fields:
        if _f.lower() == _field_name.lower() or _f.lower().replace("_", "") == _field_norm:
            _field_name = _f
            break
sym.renderer.classificationField = _field_name
sym.renderer.breakCount = 5
sym.renderer.classificationMethod = "NaturalBreaks"
# 注意：ArcGIS Pro 3.6 的 GraduatedColorsRenderer 在设置 classificationField/
# breakCount/classificationMethod 后自动计算断点，无需调用 classify()
ramps = aprx.listColorRamps()
_target_ramp = "{color_ramp_name}".strip()
_chosen = None
def _infer_hue(_text: str) -> str:
    _t = (_text or "").lower().replace(" ", "").replace("-", "").replace("_", "")
    if any(_k in _t for _k in ("rdylgn", "redyellowgreen", "红黄绿")):
        return "redyellowgreen"
    if any(_k in _t for _k in ("rdbu", "redblue", "红蓝")):
        return "redblue"
    if any(_k in _t for _k in ("green", "ylgn", "绿")):
        return "green"
    if any(_k in _t for _k in ("blue", "blues", "gnbu", "pubu", "bupu", "ylgnbu", "蓝")):
        return "blue"
    if any(_k in _t for _k in ("red", "reds", "orrd", "ylorrd", "红")):
        return "red"
    if any(_k in _t for _k in ("orange", "oranges", "ylorbr", "橙")):
        return "orange"
    if any(_k in _t for _k in ("purple", "purples", "purd", "rdpu", "紫")):
        return "purple"
    if any(_k in _t for _k in ("brown", "brbg", "棕")):
        return "brown"
    if any(_k in _t for _k in ("grey", "gray", "greys", "灰")):
        return "gray"
    return ""

_target_hue = _infer_hue(_target_ramp)
if _target_ramp and ramps:
    for _r in ramps:
        if _target_ramp.lower() in _r.name.lower():
            _chosen = _r
            break
if _chosen is None and _target_hue and ramps:
    for _r in ramps:
        if _infer_hue(_r.name) == _target_hue:
            _chosen = _r
            break
if _chosen is None and ramps:
    _chosen = ramps[0]
if _chosen is not None:
    sym.renderer.colorRamp = _chosen

# 强制色相回退：当指定色相与实际匹配色带不一致时，直接写入对应分级色
_chosen_hue = _infer_hue(_chosen.name) if _chosen is not None else ""
_force_hue = bool(_target_hue) and (_chosen_hue != _target_hue)
if _force_hue:
    _palettes = {{
        "green": [
            {{"RGB": [232, 246, 225, 100]}},
            {{"RGB": [189, 228, 176, 100]}},
            {{"RGB": [142, 209, 128, 100]}},
            {{"RGB": [86, 173, 80, 100]}},
            {{"RGB": [35, 125, 50, 100]}},
        ],
        "blue": [
            {{"RGB": [222, 235, 247, 100]}},
            {{"RGB": [158, 202, 225, 100]}},
            {{"RGB": [107, 174, 214, 100]}},
            {{"RGB": [49, 130, 189, 100]}},
            {{"RGB": [8, 81, 156, 100]}},
        ],
        "red": [
            {{"RGB": [254, 224, 210, 100]}},
            {{"RGB": [252, 146, 114, 100]}},
            {{"RGB": [251, 106, 74, 100]}},
            {{"RGB": [222, 45, 38, 100]}},
            {{"RGB": [165, 15, 21, 100]}},
        ],
        "orange": [
            {{"RGB": [254, 237, 222, 100]}},
            {{"RGB": [253, 190, 133, 100]}},
            {{"RGB": [253, 141, 60, 100]}},
            {{"RGB": [230, 85, 13, 100]}},
            {{"RGB": [166, 54, 3, 100]}},
        ],
        "purple": [
            {{"RGB": [239, 237, 245, 100]}},
            {{"RGB": [188, 189, 220, 100]}},
            {{"RGB": [158, 154, 200, 100]}},
            {{"RGB": [117, 107, 177, 100]}},
            {{"RGB": [84, 39, 143, 100]}},
        ],
        "brown": [
            {{"RGB": [245, 245, 245, 100]}},
            {{"RGB": [216, 179, 101, 100]}},
            {{"RGB": [191, 129, 45, 100]}},
            {{"RGB": [140, 81, 10, 100]}},
            {{"RGB": [84, 48, 5, 100]}},
        ],
        "gray": [
            {{"RGB": [240, 240, 240, 100]}},
            {{"RGB": [189, 189, 189, 100]}},
            {{"RGB": [150, 150, 150, 100]}},
            {{"RGB": [99, 99, 99, 100]}},
            {{"RGB": [37, 37, 37, 100]}},
        ],
        "redblue": [
            {{"RGB": [178, 24, 43, 100]}},
            {{"RGB": [239, 138, 98, 100]}},
            {{"RGB": [247, 247, 247, 100]}},
            {{"RGB": [103, 169, 207, 100]}},
            {{"RGB": [33, 102, 172, 100]}},
        ],
        "redyellowgreen": [
            {{"RGB": [165, 0, 38, 100]}},
            {{"RGB": [244, 109, 67, 100]}},
            {{"RGB": [255, 255, 191, 100]}},
            {{"RGB": [166, 217, 106, 100]}},
            {{"RGB": [26, 150, 65, 100]}},
        ],
    }}
    _palette = _palettes.get(_target_hue)
    if _palette and hasattr(sym.renderer, "classBreaks") and sym.renderer.classBreaks:
        for _idx, _cb in enumerate(sym.renderer.classBreaks):
            _cb.symbol.color = _palette[min(_idx, len(_palette) - 1)]
'''


def _unique_renderer_block(field_name: str, input_path: str, category_colors: str, other_color: str,
                           default_color: str = "") -> str:
    """唯一值（分类）设色渲染片段：按值显式配色，如「标杆社区:#1F77B4;需整改社区:#2CA02C」。"""
    return f'''
sym = layer.symbology
sym.updateRenderer("UniqueValueRenderer")
# 字段名容错：大小写/下划线差异时自动匹配真实字段
_field_name = "{field_name}".strip()
_all_fields = [f.name for f in arcpy.ListFields(r"{input_path}")]
if _field_name:
    _field_norm = _field_name.lower().replace("_", "")
    for _f in _all_fields:
        if _f.lower() == _field_name.lower() or _f.lower().replace("_", "") == _field_norm:
            _field_name = _f
            break
if _field_name not in _all_fields:
    raise ValueError(f"分类字段不存在: {{_field_name}}；可用字段: {{_all_fields}}")
sym.renderer.fields = [_field_name]

def _hex_to_rgb(_h: str):
    _h = str(_h).strip().lstrip("#")
    return [int(_h[0:2], 16), int(_h[2:4], 16), int(_h[4:6], 16), 100]

_color_map = {{}}
for _item in "{category_colors}".split(";"):
    if ":" in _item:
        _k, _v = _item.split(":", 1)
        _color_map[_k.strip()] = _v.strip()
_default = "{default_color}".strip() or "{other_color}".strip() or "#D9D9D9"
_applied = {{}}
for _grp in sym.renderer.groups:
    for _cls in _grp.items:
        _vals = list(_cls.values) if _cls.values else []
        _key = ""
        if _vals:
            _v0 = _vals[0]
            _key = str(_v0[0]) if isinstance(_v0, (list, tuple)) else str(_v0)
        _hex = _color_map.get(_key, _default)
        _cls.symbol.color = {{"RGB": _hex_to_rgb(_hex)}}
        _cls.label = _key or str(getattr(_cls, "label", ""))
        _applied[_key] = _hex
print(f"唯一值配色: {{_applied}}")
'''


def build_graduated_colors_code(
    input_path: str,
    field_name: str,
    output_path: str,
    page_width: int = 297,
    page_height: int = 420,
    title: str = "",
    color_ramp_name: str = "",
    legend_style: str = "",
    scale_bar_style: str = "",
    north_arrow_style: str = "",
    aprx_path: str = "",
    renderer_mode: str = "graduated",
    category_colors: str = "",
    other_color: str = "",
    default_color: str = "",
    **kwargs,
) -> str:
    """生成分级设色+布局+导出的 ArcPy 代码字符串（可在 execute_code 中直接运行）

    Args:
        input_path: 输入要素类路径
        field_name: 分级字段名
        output_path: 输出 JPG 路径
        page_width: 页面宽度 mm（A3=297, A4竖版=210）
        page_height: 页面高度 mm（A3=420, A4竖版=297）
        title: 地图标题（空字符串时自动根据 field_name 生成）
        color_ramp_name: 色带名称（如 YlOrRd），空字符串时使用默认
        legend_style: 图例样式关键词（如"Legend 1"），空=默认
        scale_bar_style: 比例尺样式关键词（如"Scale Bar 1"），空=默认
        north_arrow_style: 指北针样式关键词（如"North Arrow 1"），空=默认
        aprx_path: 工程文件（.aprx）保存路径；空则与输出图同名同目录，
                   如 output/old_pct_map.jpg → output/old_pct_map.aprx。
                   已存在的工程会被复制一份再修改（不会改动原工程）。
    Returns:
        可直接在 ArcPro Python 中执行的代码字符串
    """
    _title = (title or "").strip()
    if not _title:
        _title = f"{{field_name}} 分类图" if renderer_mode == "unique" else f"{{field_name}} 分级图"
    if str(renderer_mode).lower() == "unique":
        _renderer_block = _unique_renderer_block(field_name, input_path, category_colors, other_color, default_color)
    else:
        _renderer_block = _graduated_renderer_block(field_name, input_path, color_ramp_name)
    code = f'''
from arcpy import mp, cim
import arcpy
import os

# 数据范围 → 自动选择纸张
_desc = arcpy.Describe(r"{input_path}")
_ext = _desc.extent
_asp = _ext.width / _ext.height
if _asp > 1.2:
    PAGE_W, PAGE_H = 420, 297  # A3 横版
elif _asp < 0.8:
    PAGE_W, PAGE_H = 210, 297  # A4 竖版
else:
    PAGE_W, PAGE_H = 297, 420  # A3 横版

# 要素数量 → 自适应图例大小
_fc = int(arcpy.management.GetCount(r"{input_path}").getOutput(0))
_mar = int(PAGE_W * 0.04)              # 图廓边距 4%
_lg_h = 35 if _fc > 50 else 45         # 要素多→图例小
_lg_w = 90 if _fc > 50 else 110        # 要素多→图例窄
_sb_h = 20
_sb_w = 100
_title_h = 14                           # 图名高度
_gap = 5                                # 元素间距

# 各元素位置（全在图廓内，百分比计算）
NL_L, NL_B = _mar, _mar
NL_R, NL_T = PAGE_W - _mar, PAGE_H - _mar

MF_L = NL_L + _gap
MF_R = NL_R - _gap
MF_B = NL_B + _lg_h + _gap * 3          # 底部留图例空间
MF_T = NL_T - _title_h - _gap * 4       # 顶部留图名空间

NA_X = MF_R - 15
NA_Y = MF_T - 15

LG_L = NL_L + _gap
LG_B = NL_B + _gap
LG_R = LG_L + _lg_w
LG_T = LG_B + _lg_h

SB_R = NL_R - _gap
SB_B = NL_B + _gap
SB_L = SB_R - _sb_w
SB_T = SB_B + _sb_h

# 目标工程文件：默认与导出图同名（同目录），也可由调用方指定
import glob
import shutil as _shutil
_out_dir = os.path.dirname(os.path.abspath(r"{output_path}")) or "."
os.makedirs(_out_dir, exist_ok=True)
_aprx_target = r"{aprx_path}".strip()
if not _aprx_target:
    _aprx_target = os.path.join(
        _out_dir, os.path.splitext(os.path.basename(r"{output_path}"))[0] + ".aprx"
    )
_aprx_target = os.path.abspath(_aprx_target)

# 基座工程：指定且存在的工程 → 复制一份再改（绝不改动原工程）；否则用 ArcGIS 空白模板
_base = _aprx_target if os.path.exists(_aprx_target) else None
if _base is not None:
    _tmp_swap = _aprx_target + ".base"
    _shutil.copy2(_base, _tmp_swap)
    os.remove(_aprx_target)
    _base = _tmp_swap
aprx = None
if _base is not None:
    try:
        _shutil.copy2(_base, _aprx_target)
        aprx = mp.ArcGISProject(_aprx_target)
        print(f"复用已有工程副本: {{_base}} -> {{_aprx_target}}")
    except Exception as _exc:
        print(f"现有工程无法复用({{_exc}})，改用空白模板")
        aprx = None

if _base is not None and "_tmp_swap" in dir() and os.path.exists(_tmp_swap):
    # 复用已有工程时会产生临时副本，用完即删（否则留下 .aprx.base）
    try:
        os.remove(_tmp_swap)
    except Exception:
        pass

if aprx is None:
    _install = ""
    try:
        _install = arcpy.GetInstallInfo().get("InstallDir", "") or ""
    except Exception:
        _install = ""
    _tmpl = None
    if _install:
        _cand = os.path.join(_install, "Resources", "ArcToolBox", "Services",
                            "routingservices", "data", "Blank.aprx")
        if os.path.exists(_cand):
            _tmpl = _cand
        else:
            _found = glob.glob(os.path.join(_install, "Resources", "**", "Blank.aprx"), recursive=True)
            _tmpl = _found[0] if _found else None
    if _tmpl is None:
        raise RuntimeError("未找到 .aprx 项目文件，且未找到 ArcGIS 空白模板 Blank.aprx")
    _shutil.copy2(_tmpl, _aprx_target)
    print(f"使用空白模板创建工程: {{_aprx_target}}")
    aprx = mp.ArcGISProject(_aprx_target)
m = aprx.listMaps()[0]
# 清除地图中已有图层（避免旧数据干扰）
for _old_lyr in list(m.listLayers()):
    m.removeLayer(_old_lyr)

# 添加数据 + 分级设色
layer = m.addDataFromPath(r"{input_path}")
{_renderer_block}
layer.symbology = sym
# 启用标注
try:
    _lc_list = layer.listLabelClasses()
    if _lc_list:
        for _lc in _lc_list:
            _lc.expression = "name"
            _lc.visible = True
            layer.showLabels = True
except Exception:
    pass

# 选择样式（按关键词匹配，回退默认）
all_legends = aprx.listStyleItems("ArcGIS 2D", "LEGEND")
all_sbs = aprx.listStyleItems("ArcGIS 2D", "SCALE_BAR")
all_nas = aprx.listStyleItems("ArcGIS 2D", "NORTH_ARROW")

def _pick_style(items, keyword, fallback_idx=0):
    if keyword and items:
        for _i, _item in enumerate(items):
            if keyword.lower() in _item.name.lower():
                return _item
    return items[fallback_idx] if items else None

legend_style_item = _pick_style(all_legends, "{legend_style}")
sb_style_item = _pick_style(all_sbs, "{scale_bar_style}")
na_style_item = _pick_style(all_nas, "{north_arrow_style}")

# 创建布局
layout = aprx.createLayout(PAGE_W, PAGE_H, "MILLIMETER")
mf_geom = arcpy.Polygon(arcpy.Array([
    arcpy.Point(MF_L, MF_B), arcpy.Point(MF_R, MF_B),
    arcpy.Point(MF_R, MF_T), arcpy.Point(MF_L, MF_T),
    arcpy.Point(MF_L, MF_B),
]))
# MapFrame will be created after setDefinition

# 添加图廓线（CIM 方式）
_nl_d = layout.getDefinition("V3")
_nl_line = cim.CIMLineGraphic()
_nl_line.symbol = cim.CIMLineSymbol()
_nl_stroke = cim.CIMSolidStroke()
_nl_stroke.color = cim.CIMRGBColor()
_nl_stroke.color.red = 0; _nl_stroke.color.green = 0; _nl_stroke.color.blue = 0
_nl_stroke.width = 0.5
_nl_line.symbol.symbolLayers = [_nl_stroke]
_nl_line.shape = arcpy.Polygon(arcpy.Array([
    arcpy.Point(NL_L, NL_B), arcpy.Point(NL_R, NL_B),
    arcpy.Point(NL_R, NL_T), arcpy.Point(NL_L, NL_T),
    arcpy.Point(NL_L, NL_B),
]))
_nl_el = cim.CIMGraphicElement()
_nl_el.name = "Neatline"
_nl_el.graphic = _nl_line
_nl_d.elements.append(_nl_el)

# 添加图名（CIMGraphicElement 包裹文本图形，确保可渲染）
_title_pg = cim.CIMParagraphTextGraphic()
_title_pg.text = "{_title}"
_title_sym = cim.CIMTextSymbol()
_title_sym.fontFamilyName = "微软雅黑"
_title_sym.fontSize = 20
_title_sym.bold = True
_title_sym.horizontalAlignment = "Center"
_title_sym.color = cim.CIMRGBColor()
_title_sym.color.red = 0; _title_sym.color.green = 0; _title_sym.color.blue = 0
_title_pg.symbol = _title_sym
_title_cy = (NL_T + MF_T) / 2  # 图廓顶部与地图框顶部之间居中
_title_pg.shape = arcpy.Polygon(arcpy.Array([
    arcpy.Point(NL_L + _gap, _title_cy - _title_h // 2),
    arcpy.Point(NL_R - _gap, _title_cy - _title_h // 2),
    arcpy.Point(NL_R - _gap, _title_cy + _title_h // 2),
    arcpy.Point(NL_L + _gap, _title_cy + _title_h // 2),
    arcpy.Point(NL_L + _gap, _title_cy - _title_h // 2),
]))
_title_el = cim.CIMGraphicElement()
_title_el.name = "Map Title"
_title_el.graphic = _title_pg
_nl_d.elements.append(_title_el)
layout.setDefinition(_nl_d)
# 在 CIM 元素之后创建 MapFrame（避免 setDefinition 重置）
mf = layout.createMapFrame(mf_geom, m, "Main Map")
mf.camera.setExtent(mf.getLayerExtent(layer))

# 添加图例/指北针/比例尺（全在图廓内）
layout.createMapSurroundElement(arcpy.Point(NA_X, NA_Y), "NORTH_ARROW", mf, na_style_item)
layout.createMapSurroundElement(arcpy.Polygon(arcpy.Array([
    arcpy.Point(LG_L, LG_B), arcpy.Point(LG_R, LG_B),
    arcpy.Point(LG_R, LG_T), arcpy.Point(LG_L, LG_T),
    arcpy.Point(LG_L, LG_B),
])), "LEGEND", mf, legend_style_item)
layout.createMapSurroundElement(arcpy.Polygon(arcpy.Array([
    arcpy.Point(SB_L, SB_B), arcpy.Point(SB_R, SB_B),
    arcpy.Point(SB_R, SB_T), arcpy.Point(SB_L, SB_T),
    arcpy.Point(SB_L, SB_B),
])), "SCALE_BAR", mf, sb_style_item)

# 地图与布局命名（便于在 ArcGIS Pro 中识别）
_bad_chars = r'\\/:*?"<>|'
_safe_name = "".join(("_" if _c in _bad_chars else _c) for _c in "{_title}").strip() or "{field_name} 专题图"
try:
    m.name = _safe_name[:50]
except Exception:
    pass
try:
    layout.name = _safe_name[:50]
except Exception:
    pass

# 导出（自动处理后缀）
import os.path as _osp
_out = r"{output_path}"
# 先落盘工程：样式与布局必须保存，用户才能用 ArcGIS Pro 打开继续修改
aprx.save()
if not _osp.exists(_aprx_target):
    raise RuntimeError(f"工程保存失败: {{_aprx_target}}")
_aprx_size = _osp.getsize(_aprx_target)
print(f"APRX 工程已保存: {{_aprx_target}} ({{_aprx_size}} bytes)")
# 确保输出路径以 .jpg 结尾（arcpy 可能追加 .jpg）
_out_base, _out_ext = _osp.splitext(_out)
if _out_ext.lower() not in (".jpg", ".jpeg"):
    _out = _out_base + ".jpg"
layout.exportToJPEG(_out, resolution=250)
# arcpy exportToJPEG 有时会追加 .jpg
_out_candidates = [_out, _out + ".jpg", _out + ".jpeg"]
_found = [p for p in _out_candidates if _osp.exists(p)]
if _found:
    _actual = _found[0]
    sz = _osp.getsize(_actual)
    print(f"JPG导出成功: {{_actual}} ({{sz}} bytes)")
    # 如果用户要求的是 PDF，额外导出 PDF
    if _out_ext.lower() == ".pdf":
        _pdf = r"{output_path}"
        layout.exportToPDF(_pdf, resolution=200)
        if _osp.exists(_pdf):
            sz2 = _osp.getsize(_pdf)
            print(f"PDF导出成功: {{_pdf}} ({{sz2}} bytes)")
            set_result({{"output": _pdf, "size": sz2, "success": True, "jpg_also": _actual,
                        "aprx": _aprx_target, "aprx_size": _aprx_size, "aprx_saved": True}})
        else:
            set_result({{"output": _actual, "size": sz, "success": True, "note": "PDF导出失败，返回JPG",
                        "aprx": _aprx_target, "aprx_size": _aprx_size, "aprx_saved": True}})
    else:
        set_result({{"output": _actual, "size": sz, "success": True,
                    "aprx": _aprx_target, "aprx_size": _aprx_size, "aprx_saved": True}})
else:
    raise RuntimeError(f"JPG 文件未生成，尝试路径: {{_out_candidates}}")
'''
    return code.strip()


# === Test Function ===

if __name__ == "__main__":
    print("Testing ArcPy Bridge...")
    print("-" * 50)
    
    # Test discovery
    try:
        info = discover_arcgis_pro_python()
        print(f"Found ArcGIS Pro Python:")
        print(f"  Install Dir: {info.install_dir}")
        print(f"  Python: {info.python_executable}")
        print(f"  Source: {info.source}")
    except ArcGISDiscoveryError as e:
        print(f"Discovery failed: {e}")
        sys.exit(1)
    
    print("-" * 50)
    
    # Test ArcPy availability
    print("Checking ArcPy availability...")
    result = check_arcpy_available()
    print(f"Status: {result.status}")
    if result.data:
        print(f"Data: {result.data}")
    if result.error:
        print(f"Error: {result.error}")
    if result.hint:
        print(f"Hint: {result.hint}")
    
    print("-" * 50)
    print("Done!")

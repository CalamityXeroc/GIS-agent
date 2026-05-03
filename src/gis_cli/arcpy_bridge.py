# -*- coding: utf-8 -*-
"""ArcPy Bridge - 桥接 AI Agent 与 ArcGIS Pro Python 环境

参考 ArcGIS-Pro-Bridge-MCP-Server 实现
通过子进程调用 ArcGIS Pro 自带的 Python 环境执行 ArcPy 代码
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
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
ARCGIS_REGISTRY_PATHS = (
    r"SOFTWARE\ESRI\ArcGISPro",
    r"SOFTWARE\WOW6432Node\ESRI\ArcGISPro",
)
ARCGIS_PYTHON_RELATIVE_PATHS = (
    Path("bin/Python/envs/arcgispro-py3/python.exe"),
    Path("bin/Python/scripts/propy.bat"),
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

def _normalize_path(value: str | os.PathLike[str]) -> str:
    """标准化路径"""
    return str(Path(value).expanduser().resolve(strict=False))


def _guess_install_dir_from_python(python_path: str | os.PathLike[str]) -> str:
    """从 Python 路径推断 ArcGIS Pro 安装目录"""
    normalized_path = Path(_normalize_path(python_path))
    parts = normalized_path.parts
    marker = ("bin", "Python", "envs", "arcgispro-py3")
    if len(parts) >= len(marker) + 1 and tuple(parts[-5:-1]) == marker:
        return _normalize_path(normalized_path.parents[4])
    return _normalize_path(normalized_path.parent)


def _iter_env_python_candidates() -> list[tuple[str, str, str]]:
    """从环境变量查找 Python 候选"""
    candidates: list[tuple[str, str, str]] = []
    
    # Check explicit environment variable
    explicit_python = os.environ.get("ARCGIS_PRO_PYTHON")
    if explicit_python:
        python_path = Path(explicit_python).expanduser()
        normalized = _normalize_path(python_path)
        candidates.append((
            "env:ARCGIS_PRO_PYTHON",
            normalized,
            _guess_install_dir_from_python(normalized)
        ))
    
    # Check install dir environment variable
    install_dir = os.environ.get("ARCGIS_PRO_INSTALL_DIR")
    if install_dir:
        install_path = Path(install_dir).expanduser()
        normalized_install = _normalize_path(install_path)
        for relative_path in ARCGIS_PYTHON_RELATIVE_PATHS:
            candidates.append((
                "env:ARCGIS_PRO_INSTALL_DIR",
                _normalize_path(install_path / relative_path),
                normalized_install
            ))
    
    return candidates


def _iter_registry_install_dirs() -> list[tuple[str, str]]:
    """从 Windows 注册表查找安装目录"""
    if winreg is None:
        return []
    
    discovered: list[tuple[str, str]] = []
    registry_views = [0]
    key_read = getattr(winreg, "KEY_READ", 0)
    
    for extra_flag_name in ("KEY_WOW64_64KEY", "KEY_WOW64_32KEY"):
        extra_flag = getattr(winreg, extra_flag_name, 0)
        if extra_flag:
            registry_views.append(extra_flag)
    
    for registry_path in ARCGIS_REGISTRY_PATHS:
        for view_flag in registry_views:
            try:
                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE, registry_path, 0, key_read | view_flag
                ) as key:
                    install_dir, _ = winreg.QueryValueEx(key, "InstallDir")
                    discovered.append((
                        f"registry:{registry_path}",
                        _normalize_path(install_dir)
                    ))
            except OSError:
                continue
    
    return discovered


def _build_python_candidates() -> list[tuple[str, str, str]]:
    """构建所有 Python 候选路径"""
    candidates: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    
    # From environment variables
    for source, python_path, install_dir in _iter_env_python_candidates():
        normalized = _normalize_path(python_path)
        if normalized not in seen:
            seen.add(normalized)
            candidates.append((source, normalized, install_dir))
    
    # From registry
    for source, install_dir in _iter_registry_install_dirs():
        install_path = Path(install_dir)
        for relative_path in ARCGIS_PYTHON_RELATIVE_PATHS:
            python_path = _normalize_path(install_path / relative_path)
            if python_path not in seen:
                seen.add(python_path)
                candidates.append((source, python_path, _normalize_path(install_path)))
    
    # Filesystem fallback - common locations
    fallback_paths = [
        Path(r"C:\Program Files\ArcGIS\Pro"),
        Path(r"E:\ArcGISPro3.6"),  # 用户的安装路径
        Path(r"D:\ArcGIS\Pro"),
    ]
    
    for fallback in fallback_paths:
        for relative_path in ARCGIS_PYTHON_RELATIVE_PATHS:
            python_path = _normalize_path(fallback / relative_path)
            if python_path not in seen:
                seen.add(python_path)
                candidates.append(("filesystem:fallback", python_path, _normalize_path(fallback)))
    
    return candidates


@lru_cache(maxsize=1)
def discover_arcgis_pro_python() -> ArcGISPythonInfo:
    """自动发现 ArcGIS Pro Python 解释器"""
    for source, python_path, install_dir in _build_python_candidates():
        if Path(python_path).exists():
            return ArcGISPythonInfo(
                install_dir=_normalize_path(install_dir),
                python_executable=python_path,
                source=source
            )
    
    raise ArcGISDiscoveryError(
        "未找到 ArcGIS Pro Python 解释器。请确认已安装 ArcGIS Pro，"
        "或通过 ARCGIS_PRO_PYTHON / ARCGIS_PRO_INSTALL_DIR 环境变量提供路径。"
    )


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
                hint="请安装 ArcGIS Pro 或设置 ARCGIS_PRO_PYTHON 环境变量。"
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
            return {{
                "name": name,
                "type": desc.shapeType,
                "path": path,
                "feature_type": "FeatureClass",
                "spatial_reference": sr.name if sr else None,
                "wkid": sr.factoryCode if sr else None,
                "has_z": desc.hasZ,
                "has_m": desc.hasM,
                "fields": fields,
                "feature_count": arcpy.management.GetCount(path).getOutput(0),
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
    Returns:
        可直接在 ArcPro Python 中执行的代码字符串
    """
    _title = (title or "").strip()
    if not _title:
        _title = f"{{field_name}} 分级图"
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

# 自动查找 aprx（从数据文件所在目录向上搜索）
import glob
_input_dir = os.path.dirname(r"{input_path}")
_search_roots = [_input_dir]
_p = _input_dir
for _ in range(5):  # 向上搜5层
    _p = os.path.dirname(_p)
    _search_roots.append(_p)
_aprx_files = []
for _r in _search_roots:
    _aprx_files.extend(glob.glob(os.path.join(_r, "**", "*.aprx"), recursive=True))
prj = _aprx_files[0] if _aprx_files else None
if prj is None or not os.path.exists(prj):
    raise RuntimeError("未找到 .aprx 项目文件")
aprx = mp.ArcGISProject(prj)
m = aprx.listMaps()[0]

# 添加数据 + 分级设色
layer = m.addDataFromPath(r"{input_path}")
sym = layer.symbology
sym.updateRenderer("GraduatedColorsRenderer")
sym.renderer.classificationField = "{field_name}"
sym.renderer.breakCount = 5
sym.renderer.classificationMethod = "NaturalBreaks"
# 注意：ArcGIS Pro 3.6 的 GraduatedColorsRenderer 在设置 classificationField/
# breakCount/classificationMethod 后自动计算断点，无需调用 classify()
ramps = aprx.listColorRamps()
if ramps:
    _target_ramp = "{color_ramp_name}"
    _chosen = None
    if _target_ramp:
        for _r in ramps:
            if _target_ramp.lower() in _r.name.lower():
                _chosen = _r
                break
    sym.renderer.colorRamp = _chosen or ramps[0]
layer.symbology = sym

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
mf = layout.createMapFrame(mf_geom, m, "Main Map")
mf.camera.setExtent(mf.getLayerExtent(layer))  # 缩放到数据范围

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
_nl_d.elements.append(_nl_line)

# 添加图名（CIM, 在图廓内顶部）
_title_tg = cim.CIMTextGraphic()
_title_tg.text = "{_title}"
_title_sym = cim.CIMTextSymbol()
_title_sym.fontFamilyName = "微软雅黑"
_title_sym.fontSize = 20
_title_sym.bold = True
_title_sym.horizontalAlignment = "Center"
_title_tg.symbol = _title_sym
_title_cy = (NL_T + MF_T) / 2  # 图廓顶部与地图框顶部之间居中
_title_tg.shape = arcpy.Polygon(arcpy.Array([
    arcpy.Point(NL_L + _gap, _title_cy - _title_h // 2),
    arcpy.Point(NL_R - _gap, _title_cy - _title_h // 2),
    arcpy.Point(NL_R - _gap, _title_cy + _title_h // 2),
    arcpy.Point(NL_L + _gap, _title_cy + _title_h // 2),
    arcpy.Point(NL_L + _gap, _title_cy - _title_h // 2),
]))
_nl_d.elements.append(_title_tg)
layout.setDefinition(_nl_d)

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

# 导出（自动处理后缀）
import os.path as _osp
_out = r"{output_path}"
# 确保输出路径以 .jpg 结尾（arcpy 可能追加 .jpg）
_out_base, _out_ext = _osp.splitext(_out)
if _out_ext.lower() not in (".jpg", ".jpeg"):
    _out = _out_base + ".jpg"
layout.exportToJPEG(_out, resolution=200)
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
            set_result({{"output": _pdf, "size": sz2, "success": True, "jpg_also": _actual}})
        else:
            set_result({{"output": _actual, "size": sz, "success": True, "note": "PDF导出失败，返回JPG"}})
    else:
        set_result({{"output": _actual, "size": sz, "success": True}})
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

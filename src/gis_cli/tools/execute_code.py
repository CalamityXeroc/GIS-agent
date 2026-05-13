# -*- coding: utf-8 -*-
"""通用 ArcPy 代码执行工具

这是最核心的扩展性工具 - 允许 LLM 生成并执行任意 ArcPy 代码，
从而解决任何 GIS 分析和制图问题，不受预定义工具的限制。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ..core.tool import Tool, ToolCategory, ToolResult, ToolContext
from ..core.registry import register_tool
from ..arcpy_bridge import run_arcpy_code, discover_arcgis_pro_python, ArcGISDiscoveryError


def _is_gis_output_path(path: str) -> bool:
    """判断路径是否为 GIS 输出文件/数据集。

    支持：
    - 常规文件扩展名：.shp, .gdb, .pdf, .png, .tif, .tiff, .csv, .xlsx, .dbf, .gpkg 等
    - GDB 内要素类/表：路径中包含 '.gdb/' 或 '.gdb\\'（如 xxx.gdb/FeatureName）
    """
    import os
    path_lower = path.lower().replace('\\', '/')

    # GDB 内部要素类/表（无扩展名但路径含 .gdb/）
    if '.gdb/' in path_lower:
        return True

    # 常规 GIS 文件扩展名
    gis_extensions = {
        '.shp', '.gdb', '.pdf', '.png', '.tif', '.tiff',
        '.csv', '.xlsx', '.dbf', '.gpkg', '.lyrx', '.mapx',
        '.jpg', '.jpeg', '.bmp', '.svg',
    }
    _, ext = os.path.splitext(path_lower)
    return ext in gis_extensions


def _normalize_output_path(path: str, workspace: str | None = None) -> str | None:
    """标准化输出路径，兼容未带 .gdb 后缀的返回值。"""
    import os
    from pathlib import Path

    if not isinstance(path, str):
        return None
    raw = path.strip()
    if not raw:
        return None

    # 直接是 GIS 输出路径
    if _is_gis_output_path(raw):
        return raw

    candidates: list[Path] = []
    p = Path(raw)
    if p.is_absolute():
        candidates.append(p)
    else:
        if workspace:
            candidates.append(Path(workspace) / raw)
        candidates.append(Path.cwd() / raw)

    for c in candidates:
        if c.exists():
            return str(c)

    # 兼容 CreateFileGDB 返回“数据库”而实际落盘为“数据库.gdb”
    for c in candidates:
        c_str = str(c)
        if not c_str.lower().endswith(".gdb"):
            gdb_candidate = c_str + ".gdb"
            if os.path.exists(gdb_candidate):
                return gdb_candidate

    return None


def _extract_output_paths(value: Any, workspace: str | None = None) -> list[str]:
    """递归提取 set_result() 返回数据中的输出路径。"""
    outputs: list[str] = []
    seen: set[str] = set()

    def _append(candidate: str | None) -> None:
        if isinstance(candidate, str) and candidate.strip() and candidate not in seen:
            seen.add(candidate)
            outputs.append(candidate)

    def _walk(node: Any) -> None:
        if isinstance(node, str):
            _append(_normalize_output_path(node, workspace=workspace))
            return
        if isinstance(node, dict):
            for v in node.values():
                _walk(v)
            return
        if isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(value)
    return outputs


class ExecuteCodeInput(BaseModel):
    """执行 ArcPy 代码的输入参数"""
    
    code: str = Field(
        ...,
        description="要执行的 Python/ArcPy 代码。代码中可以使用 arcpy 模块和 set_result() 函数返回结果。"
    )
    workspace: str | None = Field(
        None,
        description="工作空间路径（可选）。设置后 arcpy.env.workspace 会自动指向该路径。"
    )
    timeout_seconds: int = Field(
        300,
        description="执行超时时间（秒），默认 300 秒",
        ge=10,
        le=3600
    )
    description: str | None = Field(
        None,
        description="代码功能描述（可选），用于日志记录"
    )


@dataclass
class ExecuteCodeOutput:
    """执行结果"""
    success: bool
    status: str  # 'success' or 'error'
    stdout: str = ""
    stderr: str = ""
    result: Any = None  # 通过 set_result() 设置的返回值
    error_message: str | None = None
    error_type: str | None = None
    error_traceback: str | None = None
    execution_time: float = 0.0


@register_tool
class ExecuteCodeTool(Tool[ExecuteCodeInput, ExecuteCodeOutput]):
    """通用 ArcPy 代码执行工具
    
    这是 GIS Agent 的核心扩展能力 - 允许执行任意 ArcPy 代码。
    LLM 可以根据用户需求动态生成代码来解决各种 GIS 问题：
    
    - 空间分析（缓冲区、叠加、网络分析等）
    - 数据转换（格式转换、坐标系变换等）
    - 地图制图（符号化、布局、导出等）
    - 数据管理（创建、编辑、删除等）
    - 自定义处理流程
    
    使用示例：
    ```python
    # 代码中可以使用 arcpy 和 set_result()
    import arcpy
    
    # 执行分析
    result = arcpy.Buffer_analysis("input.shp", "output.shp", "100 Meters")
    
    # 返回结果
    set_result({
        "output": "output.shp",
        "count": arcpy.GetCount_management("output.shp")[0]
    })
    ```
    """
    
    name = "execute_code"
    description = "执行任意 ArcPy 代码，实现灵活的 GIS 分析和处理"
    category = ToolCategory.BATCH_PROCESSING
    input_model = ExecuteCodeInput
    
    # 不需要 ArcPy 预检测，因为我们会自己处理
    requires_arcpy = False
    
    def validate_input(self, input_data: ExecuteCodeInput):
        """验证输入
        
        安全策略（信任模型）：
        由于 LLM 生成代码的可控性，采用宽松的安全策略：
        
        允许：
        - subprocess：调用外部工具（Word、LibreOffice、其他应用等）
        - open()：文件 I/O 操作
        - os 模块：文件系统操作
        - import：模块导入
        
        不允许：
        - 无（完全信任 LLM 生成的代码）
        
        注意：用户应确保 LLM 配置正确，避免恶意提示词
        """
        from ..core.tool import ValidationResult
        
        if not input_data.code or not input_data.code.strip():
            return ValidationResult.failure("代码不能为空")
        
        # 最小化安全检查 - 仅防止明显的意外情况
        # （实际的安全性应通过 LLM 配置和用户监督来保证）
        
        return ValidationResult.success()

    @staticmethod
    def _fix_common_api_mistakes(code: str) -> str:
        """Fix common ArcPy API mistakes in LLM-generated code before execution."""
        import re

        replacements = [
            # GraduatedColorsRenderer: .field → .classificationField
            (r".renderer\.field\s*=", ".renderer.classificationField ="),
            # GraduatedColorsRenderer: .numClasses → .breakCount
            (r".renderer\.numClasses\s*=", ".renderer.breakCount ="),
            # ArcGISProject.maps → .listMaps()  (Pro 3.6 没有 .maps 属性)
            (r"(\w+)\.maps(?=\s*\[)", r"\1.listMaps()"),
            # Camera.setToExtent → .setExtent  (Pro 3.6 API 名称不同)
            (r"\.setToExtent\s*\(", ".setExtent("),
            # DescribeData.featureCount → arcpy.management.GetCount
            (r"(\w+)\.featureCount\b", r"int(arcpy.management.GetCount(\1).getOutput(0))"),
        ]
        fixed = code
        for pattern, replacement in replacements:
            fixed = re.sub(pattern, replacement, fixed)

        # Fix doubled workspace segments in paths (e.g. "workspace\\workspace\\output")
        fixed = re.sub(r'(workspace)[\\/]\1', r'\1', fixed, flags=re.IGNORECASE)

        # Auto-fix common layout/map-frame mistakes that lead to blank maps
        # or ArcGIS Pro API AttributeError:
        # - Map.camera (should use MapFrame.camera)
        # - layout.mapFrame.elementExtent (use layout.listElements("MAPFRAME_ELEMENT"))
        map_vars = set(
            re.findall(
                r"^\s*([A-Za-z_]\w*)\s*=\s*.*?\.listMaps\(\)\s*\[\s*0\s*\]",
                fixed,
                flags=re.MULTILINE,
            )
        )
        layout_vars = set(
            re.findall(
                r"^\s*([A-Za-z_]\w*)\s*=\s*.*?\.listLayouts\(\)\s*\[\s*0\s*\]",
                fixed,
                flags=re.MULTILINE,
            )
        )

        # Replace "<camera_var> = <map_var>.camera" with MapFrame-based camera setup.
        if map_vars and layout_vars:
            layout_var = next(iter(layout_vars))
            for map_var in map_vars:
                pattern = re.compile(
                    rf"^(?P<indent>\s*)(?P<lhs>[A-Za-z_]\w*)\s*=\s*{re.escape(map_var)}\.camera\s*$",
                    flags=re.MULTILINE,
                )

                def _replace_map_camera(match: re.Match[str]) -> str:
                    indent = match.group("indent")
                    lhs = match.group("lhs")
                    return (
                        f'{indent}_mapframes = {layout_var}.listElements("MAPFRAME_ELEMENT")\n'
                        f"{indent}if not _mapframes:\n"
                        f'{indent}    raise RuntimeError("布局中未找到 MAPFRAME_ELEMENT，无法设置地图范围")\n'
                        f"{indent}mf = _mapframes[0]\n"
                        f"{indent}mf.map = {map_var}\n"
                        f"{indent}{lhs} = mf.camera"
                    )

                fixed = pattern.sub(_replace_map_camera, fixed)

        # Replace "camera.setExtent(layout.mapFrame.elementExtent)" pattern.
        pattern_layout_mapframe = re.compile(
            r"^(?P<indent>\s*)(?P<cam>[A-Za-z_]\w*)\.setExtent\(\s*(?P<layout>[A-Za-z_]\w*)\.mapFrame\.elementExtent\s*\)\s*$",
            flags=re.MULTILINE,
        )

        def _replace_layout_mapframe_extent(match: re.Match[str]) -> str:
            indent = match.group("indent")
            cam = match.group("cam")
            layout_var = match.group("layout")
            map_var = next(iter(map_vars), "m")
            return (
                f'{indent}_mapframes = {layout_var}.listElements("MAPFRAME_ELEMENT")\n'
                f"{indent}if not _mapframes:\n"
                f'{indent}    raise RuntimeError("布局中未找到 MAPFRAME_ELEMENT，无法设置地图范围")\n'
                f"{indent}mf = _mapframes[0]\n"
                f"{indent}mf.map = {map_var}\n"
                f"{indent}_target_layer = layer if 'layer' in locals() else ({map_var}.listLayers()[0] if {map_var}.listLayers() else None)\n"
                f"{indent}if _target_layer:\n"
                f"{indent}    {cam}.setExtent(mf.getLayerExtent(_target_layer, False, True))"
            )

        fixed = pattern_layout_mapframe.sub(_replace_layout_mapframe_extent, fixed)

        # If code exports a layout but never binds map frame to the active map,
        # inject a safe binding snippet after layout acquisition.
        has_layout_export = bool(
            re.search(r"\b[A-Za-z_]\w*\.exportTo(?:JPEG|PNG|PDF|TIFF)\s*\(", fixed)
        )
        has_create_mapframe = "createMapFrame(" in fixed
        if has_layout_export and map_vars and layout_vars and not has_create_mapframe:
            map_var = next(iter(map_vars))
            layout_var = next(iter(layout_vars))
            if not re.search(rf"\.\s*map\s*=\s*{re.escape(map_var)}\b", fixed):
                layout_line = re.compile(
                    rf"^(?P<indent>\s*){re.escape(layout_var)}\s*=\s*.*?\.listLayouts\(\)\s*\[\s*0\s*\]\s*$",
                    flags=re.MULTILINE,
                )

                def _inject_binding(match: re.Match[str]) -> str:
                    line = match.group(0)
                    indent = match.group("indent")
                    injection = (
                        f'\n{indent}_mapframes = {layout_var}.listElements("MAPFRAME_ELEMENT")\n'
                        f"{indent}if _mapframes:\n"
                        f"{indent}    mf = _mapframes[0]\n"
                        f"{indent}    mf.map = {map_var}\n"
                        f"{indent}    _target_layer = layer if 'layer' in locals() else ({map_var}.listLayers()[0] if {map_var}.listLayers() else None)\n"
                        f"{indent}    if _target_layer:\n"
                        f"{indent}        mf.camera.setExtent(mf.getLayerExtent(_target_layer, False, True))"
                    )
                    return line + injection

                fixed = layout_line.sub(_inject_binding, fixed, count=1)

        return fixed

    @staticmethod
    def _check_known_bad_patterns(code: str) -> list[str]:
        """Check code for known ArcPy API errors before execution.

        Returns a list of human-readable error descriptions.
        Empty list = no known issues found.
        """
        import re
        errors: list[str] = []

        # Map.camera — Map has no .camera in Pro 3.6, use MapFrame
        map_vars = set(
            re.findall(
                r"^\s*([A-Za-z_]\w*)\s*=\s*.*?\.listMaps\(\)\s*\[\s*0\s*\]",
                code,
                flags=re.MULTILINE,
            )
        )
        for map_var in map_vars:
            if re.search(rf"\b{re.escape(map_var)}\.camera\b", code):
                errors.append(
                    "Map 对象没有 .camera 属性。请通过 MapFrame 的 camera 设置范围，"
                    "例如 mf = layout.listElements('MAPFRAME_ELEMENT')[0]；mf.map = m；mf.camera.setExtent(...)"
                )
                break

        # layout.mapFrame does not exist in ArcGIS Pro 3.6 API
        if re.search(r"\b[A-Za-z_]\w*\.mapFrame\b", code):
            errors.append(
                "布局对象没有 mapFrame 属性。请改用 layout.listElements('MAPFRAME_ELEMENT') 获取地图框。"
            )

        # arcpy.mp.MapDocument — doesn't exist in Pro 3.6
        if 'MapDocument' in code:
            errors.append("arcpy.mp.MapDocument 在 ArcGIS Pro 3.6 中不存在。请使用 arcpy.mp.ArcGISProject")

        # DescribeData.featureCount — doesn't exist on DescribeData
        if re.search(r'\.featureCount\b', code):
            errors.append("DescribeData 没有 featureCount 属性，请改用 arcpy.management.GetCount(path).getOutput(0) 获取要素数量")

        # classificationField = "占位符文本" — LLM 写了占位符而非实际字段名
        placeholder_match = re.search(r'classificationField\s*=\s*"([^"]*)"', code)
        if placeholder_match:
            field_val = placeholder_match.group(1)
            if '字段名' in field_val:
                errors.append(f'分类字段名 "{field_val}" 是占位符文本，不是数据中的实际字段名。请先用 scan_layers 扫描数据确认可用字段。')

        return errors

    def call(
        self,
        input_data: ExecuteCodeInput,
        context: ToolContext
    ) -> ToolResult[ExecuteCodeOutput]:
        """执行代码"""
        import time
        start_time = time.time()
        
        # 检查 ArcGIS Pro 环境
        try:
            arcgis_info = discover_arcgis_pro_python()
        except ArcGISDiscoveryError as e:
            return ToolResult.fail(
                f"无法找到 ArcGIS Pro 环境: {e}",
                "arcgis_discovery_error"
            )
        
        # Dry-run 模式
        if context.dry_run:
            code_preview = input_data.code[:200] + "..." if len(input_data.code) > 200 else input_data.code
            return ToolResult.ok(
                ExecuteCodeOutput(
                    success=True,
                    status="dry_run",
                    stdout=f"[DRY RUN] 将执行代码:\n{code_preview}"
                ),
                outputs=[]
            )
        
        # 确定工作空间（防止 .aprx 被误传为 workspace）
        workspace = input_data.workspace
        if workspace and workspace.lower().endswith('.aprx'):
            from pathlib import Path
            gdb_candidate = workspace[:-5] + '.gdb'
            if Path(gdb_candidate).exists():
                workspace = gdb_candidate
            else:
                workspace = str(Path(workspace).parent)
        
        # 预执行代码修复+检测
        fixed_code = self._fix_common_api_mistakes(input_data.code)

        # 预执行代码检测：在运行前发现已知错误模式
        check_errors = self._check_known_bad_patterns(fixed_code)
        if check_errors:
            return ToolResult.fail(
                f"代码预检测发现以下已知错误，已拦截执行：\n"
                + "\n".join(f"- {e}" for e in check_errors)
                + "\n\n请修改代码后重试，或使用 build_graduated_colors_code() 生成正确代码。",
                "code_precheck_failed"
            )

        # 执行代码
        desc = input_data.description or "执行 ArcPy 代码"

        try:
            result = run_arcpy_code(
                fixed_code,
                workspace=workspace,
                timeout_seconds=input_data.timeout_seconds
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return ToolResult.fail(
                f"执行失败: [{type(e).__name__}] {e}\n\nTraceback:\n{tb}",
                "execution_error"
            )
        
        execution_time = time.time() - start_time
        
        # 处理结果
        if result.status == "success":
            output = ExecuteCodeOutput(
                success=True,
                status="success",
                stdout=result.stdout,
                stderr=result.stderr,
                result=result.data,
                execution_time=execution_time
            )
            
            # 尝试从结果中提取输出文件（支持 GDB 要素类等无扩展名路径）
            outputs = _extract_output_paths(result.data, workspace=workspace)
            
            return ToolResult.ok(
                output,
                outputs=outputs
            )
        else:
            error_info = result.error or {}
            output = ExecuteCodeOutput(
                success=False,
                status="error",
                stdout=result.stdout,
                stderr=result.stderr,
                error_message=error_info.get("message", "未知错误"),
                error_type=error_info.get("type", "Error"),
                error_traceback=error_info.get("traceback"),
                execution_time=execution_time
            )

            # 构建详细错误信息，包含 stderr、hint、traceback
            error_parts = [f"执行失败: [{output.error_type}] {output.error_message}"]
            if result.stderr and result.stderr.strip():
                error_parts.append(f"stderr: {result.stderr.strip()[:500]}")
            if result.hint:
                error_parts.append(f"提示: {result.hint}")
            if output.error_traceback:
                # 只取最后 500 字符的 traceback，避免过长
                tb_tail = output.error_traceback.strip()[-500:]
                error_parts.append(f"traceback: ...{tb_tail}")

            return ToolResult.fail(
                "\n".join(error_parts),
                "execution_failed"
            )

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
        """验证输入"""
        from ..core.tool import ValidationResult
        
        if not input_data.code or not input_data.code.strip():
            return ValidationResult.failure("代码不能为空")
        
        # 基本安全检查 - 禁止危险操作
        dangerous_patterns = [
            "os.system(",
            "subprocess.",
            "eval(",
            "__import__",
            "exec(",  # 我们自己用 exec，但不允许嵌套
            "open(",  # 文件操作应通过 arcpy
            "shutil.rmtree",
        ]
        
        code_lower = input_data.code.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                return ValidationResult.failure(f"安全限制：不允许使用 {pattern}")
        
        return ValidationResult.success()
    
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
        
        # 执行代码
        desc = input_data.description or "执行 ArcPy 代码"
        
        try:
            result = run_arcpy_code(
                input_data.code,
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

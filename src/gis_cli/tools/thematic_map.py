"""ThematicMapTool - High-level thematic map creation tool.

Wraps build_graduated_colors_code to create graduated-color thematic maps
without requiring LLM to write ArcPy code directly, eliminating API errors
like .field/.camera/.maps misuse that plague execute_code.
"""

from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field

from ..core import (
    Tool,
    ToolCategory,
    ToolContext,
    ToolResult,
    ValidationResult,
    PermissionResult,
    register_tool,
)

TOOL_NAME = "thematic_map"

DESCRIPTION = """创建分级设色专题图（等高封装工具）

直接传入数据路径、字段名和输出路径即可生成完整的专题图，自动处理：
- 地图坐标系同步
- 分级设色渲染（NaturalBreaks / 5级）
- 色带选择和匹配（支持中英文色带名如"绿色"/Greens/"红黄绿"/YlOrRd）
- 地图三要素（图名、图例、指北针、比例尺）
- 导出为 JPG

这个工具内部使用已验证的 ArcPy 代码，比 execute_code 更可靠，强烈推荐使用。
"""

SEARCH_HINT = "thematic map graduated colors choropleth 专题图 分级设色 制图"


class ThematicMapInput(BaseModel):
    """Input schema for ThematicMapTool."""
    input_path: str = Field(
        description="输入图层路径（.shp 或 GDB 要素类），如 workspace/output/province_forest_coverage.shp"
    )
    field_name: str = Field(
        description="分级设色所用的数值字段名，如 forest_cov（大小写和空格不敏感）"
    )
    output_path: str = Field(
        description="输出 JPG 路径，如 workspace/output/thematic_map.jpg"
    )
    title: str = Field(
        default="",
        description="图名（可选，为空时根据字段名自动生成）"
    )
    color_ramp: str = Field(
        default="Greens",
        description="色带名称，支持中英文：Greens/绿色、YlOrRd/红黄绿、Blues/蓝色、OrRd/橙红等"
    )
    label_field: str = Field(
        default="",
        description="标注字段名（可选），如 name 字段可用于标注省名"
    )
    resolution: int = Field(
        default=200,
        description="导出分辨率（DPI），默认 200，范围 72-600"
    )


class ThematicMapOutput(BaseModel):
    """Output schema for ThematicMapTool."""
    output_path: str = ""
    file_size: int = 0
    field_used: str = ""
    message: str = ""


@register_tool
class ThematicMapTool(Tool[ThematicMapInput, ThematicMapOutput]):
    """Tool to create graduated-color thematic maps."""

    name = TOOL_NAME
    description = DESCRIPTION
    category = ToolCategory.CARTOGRAPHY
    search_hint = SEARCH_HINT
    input_model = ThematicMapInput

    def is_read_only(self) -> bool:
        return False

    def validate_input(self, input_data: ThematicMapInput) -> ValidationResult:
        if not input_data.input_path or not input_data.input_path.strip():
            return ValidationResult.failure("input_path 不能为空")
        if not input_data.field_name or not input_data.field_name.strip():
            return ValidationResult.failure("field_name 不能为空")
        if not input_data.output_path or not input_data.output_path.strip():
            return ValidationResult.failure("output_path 不能为空")
        if input_data.resolution < 72 or input_data.resolution > 600:
            return ValidationResult.failure("resolution 必须在 72-600 之间")
        return ValidationResult.success()

    def call(
        self,
        input_data: ThematicMapInput,
        context: ToolContext
    ) -> ToolResult[ThematicMapOutput]:
        """Execute thematic map creation."""
        if context.dry_run:
            return ToolResult.ok(
                ThematicMapOutput(
                    output_path=input_data.output_path,
                    message=f"[DRY RUN] 将创建专题图: {input_data.field_name} @ {input_data.input_path}"
                ),
                outputs=[]
            )

        if not context.arcpy_available:
            return ToolResult.fail(
                "ArcPy 不可用，无法创建专题图。请确保在 ArcGIS Pro 环境中运行。",
                "no_arcpy"
            )

        try:
            from ..arcpy_bridge import run_arcpy_code, build_graduated_colors_code

            # Generate verified code using the bridge function
            code = build_graduated_colors_code(
                input_path=input_data.input_path,
                field_name=input_data.field_name,
                output_path=input_data.output_path,
                title=input_data.title,
                color_ramp_name=input_data.color_ramp,
            )

            # Run in ArcPy subprocess
            result = run_arcpy_code(
                code,
                workspace=str(Path(input_data.input_path).parent),
                timeout_seconds=300
            )

            if result.status == "success":
                out_path = Path(input_data.output_path)
                file_size = out_path.stat().st_size if out_path.exists() else 0
                return ToolResult.ok(
                    ThematicMapOutput(
                        output_path=str(out_path),
                        file_size=file_size,
                        field_used=input_data.field_name,
                        message=f"专题图已生成: {out_path.name} ({file_size/1024:.0f} KB)"
                    ),
                    outputs=[str(out_path)]
                )
            else:
                err = result.error or {}
                return ToolResult.fail(
                    f"专题图生成失败: {err.get('message', '未知错误')}",
                    "execution_failed"
                )

        except Exception as e:
            import traceback
            return ToolResult.fail(
                f"创建专题图时出错: {e}\n{traceback.format_exc()[-500:]}",
                "tool_error"
            )

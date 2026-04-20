"""ProjectLayersTool - Project layers to target coordinate system.

Uses ArcPy Bridge to execute projection via ArcGIS Pro Python environment.
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


TOOL_NAME = "project_layers"

DESCRIPTION = """Project GIS layers to a target coordinate system.

This tool transforms coordinate systems for GIS layers:
- Supports multiple input/output projections
- Handles datum transformations
- Preserves feature attributes

Common target projections:
- CGCS2000 (China Geodetic Coordinate System 2000)
- WGS84 (EPSG:4326)
- Asia_North_Albers_Equal_Area_Conic (for China regional analysis)
"""

SEARCH_HINT = "project transform coordinate system projection CRS EPSG datum"


class ProjectLayersInput(BaseModel):
    """Input schema for ProjectLayersTool."""
    input_path: str = Field(description="Input layer path")
    output_path: str = Field(description="Output path for projected layer")
    target_srs: str = Field(
        default="CGCS2000",
        description="Target spatial reference: CGCS2000, WGS84, or EPSG code"
    )
    transform_method: str | None = Field(
        default=None,
        description="Optional datum transformation method"
    )
    overwrite_output: bool = Field(
        default=False,
        description="Whether to overwrite existing output dataset"
    )


class ProjectLayersOutput(BaseModel):
    """Output schema for ProjectLayersTool."""
    input_path: str
    output_path: str
    source_srs: str = ""
    target_srs: str = ""
    feature_count: int = 0
    transformed: bool = False


@register_tool
class ProjectLayersTool(Tool[ProjectLayersInput, ProjectLayersOutput]):
    """Tool to project GIS layers to target coordinate system."""
    
    name = TOOL_NAME
    description = DESCRIPTION
    category = ToolCategory.DATA_INTEGRATION
    search_hint = SEARCH_HINT
    input_model = ProjectLayersInput
    
    def is_read_only(self) -> bool:
        return False
    
    def get_activity_description(self, input_data: ProjectLayersInput) -> str:
        return f"Projecting to {input_data.target_srs}"
    
    def validate_input(self, input_data: ProjectLayersInput) -> ValidationResult:
        if not self._is_supported_input_path(input_data.input_path):
            return ValidationResult.failure(
                f"Input layer not found: {input_data.input_path}",
                error_code=1
            )
        return ValidationResult.success()

    def _is_supported_input_path(self, input_path: str) -> bool:
        """Accept filesystem paths and GDB feature class paths."""
        p = Path(input_path)
        if p.exists():
            return True
        # Feature class under geodatabase may not be visible as real FS file.
        # e.g. C:\\data\\demo.gdb\\roads
        return ".gdb\\" in input_path.lower()
    
    def call(
        self,
        input_data: ProjectLayersInput,
        context: ToolContext
    ) -> ToolResult[ProjectLayersOutput]:
        """Execute projection."""
        if context.dry_run:
            return self._dry_run(input_data)
        
        # Use ArcPy Bridge for actual execution
        return self._project_with_bridge(input_data, context)
    
    def _dry_run(self, input_data: ProjectLayersInput) -> ToolResult[ProjectLayersOutput]:
        output = ProjectLayersOutput(
            input_path=input_data.input_path,
            output_path=input_data.output_path,
            target_srs=input_data.target_srs,
            transformed=False
        )
        return ToolResult.ok(output)
    
    def _project_with_bridge(
        self,
        input_data: ProjectLayersInput,
        context: ToolContext
    ) -> ToolResult[ProjectLayersOutput]:
        """Project using ArcPy Bridge."""
        try:
            from ..arcpy_bridge import project_layer
            
            # Ensure output directory exists
            output_path = Path(input_data.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            result = project_layer(
                input_path=input_data.input_path,
                output_path=str(output_path),
                target_srs=input_data.target_srs,
                overwrite_output=input_data.overwrite_output,
            )
            
            if result.status == "success" and result.data:
                data = result.data
                output = ProjectLayersOutput(
                    input_path=input_data.input_path,
                    output_path=str(output_path),
                    source_srs=data.get("source_srs", ""),
                    target_srs=data.get("target_srs", input_data.target_srs),
                    feature_count=data.get("feature_count", 0),
                    transformed=True
                )
                return ToolResult.ok(output, outputs=[str(output_path)])
            else:
                error_msg = "Projection failed"
                if result.error:
                    error_msg = result.error.get("message", str(result.error))
                if result.hint:
                    error_msg += f" ({result.hint})"
                return ToolResult.fail(error_msg, "projection_error")
            
        except Exception as e:
            return ToolResult.fail(f"Projection failed: {e}", "execution_error")
    
    def render_tool_result_message(self, result: ToolResult[ProjectLayersOutput]) -> str:
        if not result.success:
            return f"Projection failed: {result.error}"
        
        data = result.data
        if data is None:
            return "No data returned"
        
        if data.transformed:
            return f"Projected {data.source_srs} -> {data.target_srs} ({data.feature_count} features)"
        else:
            return f"Already in {data.target_srs} (copied {data.feature_count} features)"

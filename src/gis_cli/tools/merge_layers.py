"""MergeLayersTool - Merge multiple layers into one.

Uses ArcPy Bridge to execute merge via ArcGIS Pro Python environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
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


# === Constants ===

TOOL_NAME = "merge_layers"

DESCRIPTION = """Merge multiple GIS layers into a single output layer.

This tool combines features from multiple input layers into one output:
- Preserves all attributes from input layers
- Handles coordinate system differences
- Supports shapefiles and geodatabase feature classes

Use this tool when you need to:
- Combine multiple tile/sheet data files
- Aggregate data from different sources
- Create a unified dataset from parts
"""

SEARCH_HINT = "merge combine union append layers features"


# === Schemas ===

class MergeLayersInput(BaseModel):
    """Input schema for MergeLayersTool."""
    input_layers: list[str] = Field(
        description="List of input layer paths to merge"
    )
    output_path: str = Field(
        description="Output path for merged layer"
    )
    overwrite_output: bool = Field(
        default=False,
        description="Whether to overwrite existing output dataset if it already exists"
    )
    layer_type: str = Field(
        default="UNKNOWN",
        description="Type of layers being merged (e.g., AGNP, BOUA)"
    )


class MergeLayersOutput(BaseModel):
    """Output schema for MergeLayersTool."""
    output_path: str
    input_count: int
    feature_count: int = 0
    geometry_type: str = ""
    success: bool = True
    message: str = ""


# === Tool Implementation ===

@register_tool
class MergeLayersTool(Tool[MergeLayersInput, MergeLayersOutput]):
    """Tool to merge multiple GIS layers."""
    
    name = TOOL_NAME
    description = DESCRIPTION
    category = ToolCategory.DATA_INTEGRATION
    search_hint = SEARCH_HINT
    input_model = MergeLayersInput
    
    def is_read_only(self) -> bool:
        return False  # Creates new files
    
    def get_activity_description(self, input_data: MergeLayersInput) -> str:
        count = len(input_data.input_layers)
        return f"Merging {count} {input_data.layer_type} layers"
    
    def validate_input(self, input_data: MergeLayersInput) -> ValidationResult:
        """Validate input layers exist."""
        if not input_data.input_layers:
            return ValidationResult.failure("No input layers specified", error_code=1)

        resolved_layers = self._expand_input_layers(input_data.input_layers)
        if not resolved_layers:
            return ValidationResult.failure(
                "No supported input datasets found. Provide layer files/feature classes or a folder containing .shp files.",
                error_code=2
            )

        return ValidationResult.success()
    
    def check_permissions(
        self,
        input_data: MergeLayersInput,
        context: ToolContext
    ) -> PermissionResult:
        """Check write permissions for output."""
        output_dir = Path(input_data.output_path).parent
        if not output_dir.exists():
            return PermissionResult.confirm(
                f"Output directory will be created: {output_dir}"
            )
        return PermissionResult.allow()
    
    def call(
        self,
        input_data: MergeLayersInput,
        context: ToolContext
    ) -> ToolResult[MergeLayersOutput]:
        """Execute layer merging."""
        import time
        start_time = time.time()

        resolved_layers = self._expand_input_layers(input_data.input_layers)
        if not resolved_layers:
            return ToolResult.fail(
                "No valid input layers found after expanding input paths",
                "no_inputs"
            )
        
        # Check if dry run
        if context.dry_run:
            return self._dry_run(input_data, resolved_layers)
        
        # Use ArcPy Bridge for actual execution
        return self._merge_with_bridge(input_data, context, resolved_layers)
    
    def _dry_run(self, input_data: MergeLayersInput, resolved_layers: list[str]) -> ToolResult[MergeLayersOutput]:
        """Simulate merge without executing."""
        mode_text = " (overwrite enabled)" if input_data.overwrite_output else ""
        output = MergeLayersOutput(
            output_path=input_data.output_path,
            input_count=len(resolved_layers),
            success=True,
            message=f"Would merge {len(resolved_layers)} layers to {input_data.output_path}{mode_text}"
        )
        return ToolResult.ok(output)
    
    def _merge_with_bridge(
        self,
        input_data: MergeLayersInput,
        context: ToolContext,
        resolved_layers: list[str],
    ) -> ToolResult[MergeLayersOutput]:
        """Merge using ArcPy Bridge."""
        try:
            from ..arcpy_bridge import merge_layers
            
            # Ensure output directory exists and use absolute path
            output_path = Path(input_data.output_path)
            if not output_path.is_absolute():
                output_path = Path.cwd() / output_path
            output_path = output_path.resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Group layers by geometry type to avoid merge errors
            grouped = self._group_by_geometry(resolved_layers)
            
            if not grouped:
                return ToolResult.fail("Could not determine layer geometry types", "geometry_error")
            
            # If only one geometry type, merge all
            if len(grouped) == 1:
                geom_type, layers = list(grouped.items())[0]
                result = merge_layers(
                    input_layers=layers,
                    output_path=str(output_path),
                    overwrite_output=input_data.overwrite_output,
                )
            else:
                # Multiple geometry types - merge by type or use most common
                # For now, use the type with most layers
                most_common_type = max(grouped, key=lambda k: len(grouped[k]))
                layers_to_merge = grouped[most_common_type]
                
                # Update output path to indicate geometry type
                stem = output_path.stem
                suffix = output_path.suffix
                new_path = output_path.parent / f"{stem}_{most_common_type}{suffix}"
                
                result = merge_layers(
                    input_layers=layers_to_merge,
                    output_path=str(new_path),
                    overwrite_output=input_data.overwrite_output,
                )
                
                # Add info about skipped layers
                if result.status == "success" and result.data:
                    skipped = sum(len(v) for k, v in grouped.items() if k != most_common_type)
                    result.data["skipped_count"] = skipped
                    result.data["message"] = f"Merged {len(layers_to_merge)} {most_common_type} layers (skipped {skipped} layers of other geometry types)"
                    output_path = new_path
            
            if result.status == "success" and result.data:
                output = MergeLayersOutput(
                    output_path=str(output_path),
                    input_count=result.data.get("input_count", len(resolved_layers)),
                    feature_count=result.data.get("feature_count", 0),
                    geometry_type=result.data.get("geometry_type", ""),
                    success=True,
                    message=result.data.get("message", f"Merged {len(resolved_layers)} layers successfully")
                )
                return ToolResult.ok(output, outputs=[str(output_path)])
            else:
                error_msg = "Merge failed"
                if result.error:
                    error_msg = result.error.get("message", str(result.error))
                if result.hint:
                    error_msg += f" ({result.hint})"
                return ToolResult.fail(error_msg, "merge_error")
            
        except Exception as e:
            return ToolResult.fail(f"Merge failed: {e}", "execution_error")
    
    def _group_by_geometry(self, layers: list[str]) -> dict[str, list[str]]:
        """Group layers by geometry type."""
        try:
            from ..arcpy_bridge import run_arcpy_code
            
            # Build code to get geometry types
            code = f'''
layers = {layers!r}
result = {{}}
for layer in layers:
    try:
        desc = arcpy.Describe(layer)
        geom_type = desc.shapeType if hasattr(desc, 'shapeType') else 'Unknown'
        if geom_type not in result:
            result[geom_type] = []
        result[geom_type].append(layer)
    except:
        pass
set_result(result)
'''
            result = run_arcpy_code(code, timeout_seconds=120)
            if result.status == "success" and result.data:
                return result.data
        except Exception:
            pass
        
        # Fallback: return all layers as single group
        return {"Unknown": layers}

    def _expand_input_layers(self, input_layers: list[str]) -> list[str]:
        """Expand folder inputs into concrete layer datasets.

        ArcPy Merge expects datasets (e.g. .shp / feature class), not a directory path.
        """
        expanded: list[str] = []
        seen: set[str] = set()

        for raw in input_layers:
            p = Path(raw)
            if not p.exists():
                continue

            if p.is_file():
                layer_path = str(p)
                if layer_path not in seen:
                    seen.add(layer_path)
                    expanded.append(layer_path)
                continue

            # Directory input: recursively collect shapefiles as merge candidates.
            for shp in p.rglob("*.shp"):
                shp_path = str(shp)
                if shp_path in seen:
                    continue
                seen.add(shp_path)
                expanded.append(shp_path)

        return expanded
    
    # === Rendering ===
    
    def render_tool_use_message(self, input_data: MergeLayersInput) -> str:
        count = len(input_data.input_layers)
        return f"Merging {count} {input_data.layer_type} layers..."
    
    def render_tool_result_message(self, result: ToolResult[MergeLayersOutput]) -> str:
        if not result.success:
            return f"Merge failed: {result.error}"
        
        data = result.data
        if data is None:
            return "No data returned"
        
        return f"Merged {data.input_count} layers -> {data.output_path} ({data.feature_count} features)"

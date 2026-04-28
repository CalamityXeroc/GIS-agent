"""ScanLayersTool - Scan directory for GIS layers.

Uses ArcPy Bridge to scan directories via ArcGIS Pro Python environment.
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


# === Constants (prompt.py style) ===

TOOL_NAME = "scan_layers"

DESCRIPTION = """Scan a directory to discover GIS layers (shapefiles, geodatabase feature classes, etc.).

This tool recursively searches for GIS data files and returns information about:
- File paths
- Layer types (point, line, polygon)
- Coordinate systems
- Feature counts (if available)

Use this tool when you need to:
- Discover available data in a directory
- Inventory GIS layers before processing
- Validate input data exists
"""

SEARCH_HINT = "find layers discover data inventory shapefiles geodatabase"


# === Input/Output Schemas ===

class ScanLayersInput(BaseModel):
    """Input schema for ScanLayersTool."""
    path: str = Field(
        default="workspace/input",
        description="Directory path to scan for GIS layers"
    )
    pattern: str = Field(
        default="**/*.shp",
        description="Glob pattern for file matching"
    )
    include_subdirs: bool = Field(
        default=True,
        description="Whether to search subdirectories"
    )


class LayerInfo(BaseModel):
    """Information about a discovered layer."""
    path: str
    name: str
    type: str  # Point, Polyline, Polygon, etc.
    extension: str
    spatial_reference: str | None = None
    feature_count: int | None = None
    fields: list[dict] = []  # e.g. [{"name": "FIELD1", "type": "String"}, ...]
    wkid: int | None = None  # CRS factory code (e.g. 4326 for WGS84)
    has_z: bool = False
    has_m: bool = False


class ScanLayersOutput(BaseModel):
    """Output schema for ScanLayersTool."""
    layers: list[LayerInfo]
    total_count: int
    by_type: dict[str, int]
    duration_ms: int = 0


# === Tool Implementation ===

@register_tool
class ScanLayersTool(Tool[ScanLayersInput, ScanLayersOutput]):
    """Tool to scan directories for GIS layers."""
    
    name = TOOL_NAME
    description = DESCRIPTION
    category = ToolCategory.FILE_OPERATION
    search_hint = SEARCH_HINT
    input_model = ScanLayersInput
    
    def is_read_only(self) -> bool:
        return True
    
    def is_concurrency_safe(self) -> bool:
        return True
    
    def get_activity_description(self, input_data: ScanLayersInput) -> str:
        return f"Scanning {input_data.path} for layers"
    
    def get_tool_use_summary(self, input_data: ScanLayersInput) -> str:
        return f"scan_layers({input_data.path}, {input_data.pattern})"
    
    def validate_input(self, input_data: ScanLayersInput) -> ValidationResult:
        """Validate that the directory exists."""
        path = Path(input_data.path)
        if not path.exists():
            return ValidationResult.failure(
                f"Directory does not exist: {input_data.path}",
                error_code=1
            )
        if not path.is_dir():
            return ValidationResult.failure(
                f"Path is not a directory: {input_data.path}",
                error_code=2
            )
        return ValidationResult.success()
    
    def check_permissions(
        self,
        input_data: ScanLayersInput,
        context: ToolContext
    ) -> PermissionResult:
        """Check read permissions for directory."""
        return PermissionResult.allow()
    
    def call(
        self,
        input_data: ScanLayersInput,
        context: ToolContext
    ) -> ToolResult[ScanLayersOutput]:
        """Execute layer scanning."""
        import time
        start_time = time.time()
        
        path = Path(input_data.path).resolve()
        
        # Try ArcPy-based scanning first (more accurate for GIS data)
        if context.arcpy_available and not context.dry_run:
            try:
                from ..arcpy_bridge import scan_workspace_layers
                result = scan_workspace_layers(str(path))
                
                if result.status == "success" and result.data:
                    layers = []
                    by_type: dict[str, int] = {}
                    
                    for layer_data in result.data.get("layers", []):
                        layer_type = layer_data.get("type", "Unknown")
                        layer_info = LayerInfo(
                            path=layer_data.get("path", ""),
                            name=layer_data.get("name", ""),
                            type=layer_type,
                            extension=Path(layer_data.get("path", "")).suffix.lower(),
                            spatial_reference=layer_data.get("spatial_reference"),
                            feature_count=layer_data.get("feature_count"),
                            fields=layer_data.get("fields", []),
                            wkid=layer_data.get("wkid"),
                            has_z=layer_data.get("has_z", False),
                            has_m=layer_data.get("has_m", False),
                        )
                        layers.append(layer_info)
                        by_type[layer_type] = by_type.get(layer_type, 0) + 1
                    
                    duration_ms = int((time.time() - start_time) * 1000)
                    output = ScanLayersOutput(
                        layers=layers,
                        total_count=len(layers),
                        by_type=by_type,
                        duration_ms=duration_ms
                    )
                    return ToolResult.ok(
                        data=output,
                        outputs=[l.path for l in layers[:10]],
                        duration_ms=duration_ms
                    )
            except Exception:
                pass  # Fall back to file-based scanning
        
        # File-based scanning (works without ArcPy)
        layers: list[LayerInfo] = []
        by_type: dict[str, int] = {}
        
        # Supported extensions
        extensions = [".shp", ".gdb", ".gpkg", ".geojson", ".aprx"]
        
        try:
            for ext in extensions:
                glob_pattern = f"**/*{ext}" if input_data.include_subdirs else f"*{ext}"
                for file_path in path.glob(glob_pattern):
                    # Skip auxiliary files
                    if file_path.suffix.lower() in [".dbf", ".shx", ".prj", ".cpg"]:
                        continue
                    
                    # Determine layer type from filename
                    stem_upper = file_path.stem.upper()
                    layer_type = "Unknown"
                    type_mapping = {
                        "AGNP": "Point",
                        "BOUA": "Polygon",
                        "BOUL": "Polyline",
                        "HYDL": "Polyline",
                        "HYDA": "Polygon",
                        "RESA": "Polygon",
                        "RESL": "Polyline",
                    }
                    for pattern, ltype in type_mapping.items():
                        if pattern in stem_upper:
                            layer_type = ltype
                            break
                    if file_path.suffix.lower() == ".aprx":
                        layer_type = "Project"
                    
                    layer_info = LayerInfo(
                        path=str(file_path),
                        name=file_path.stem,
                        type=layer_type,
                        extension=file_path.suffix.lower()
                    )
                    layers.append(layer_info)
                    
                    # Count by type
                    by_type[layer_type] = by_type.get(layer_type, 0) + 1
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            output = ScanLayersOutput(
                layers=layers,
                total_count=len(layers),
                by_type=by_type,
                duration_ms=duration_ms
            )
            
            return ToolResult.ok(
                data=output,
                outputs=[str(l.path) for l in layers[:10]],
                duration_ms=duration_ms
            )
            
        except Exception as e:
            return ToolResult.fail(str(e), "scan_error")
    
    # === Rendering ===
    
    def render_tool_use_message(self, input_data: ScanLayersInput) -> str:
        return f"Scanning {input_data.path} for GIS layers..."
    
    def render_tool_result_message(self, result: ToolResult[ScanLayersOutput]) -> str:
        if not result.success:
            return f"Scan failed: {result.error}"
        
        data = result.data
        if data is None:
            return "No data returned"
        
        lines = [f"Found {data.total_count} layers in {data.duration_ms}ms"]
        
        for layer_type, count in sorted(data.by_type.items()):
            lines.append(f"  - {layer_type}: {count}")
        
        return "\n".join(lines)

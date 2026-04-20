"""ExportMapTool - Export map layouts to PDF/PNG."""

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


TOOL_NAME = "export_map"

DESCRIPTION = """Export map layouts to PDF, PNG, or other formats.

This tool exports ArcGIS Pro map layouts:
- Supports PDF, PNG, JPEG, TIFF, and other formats
- Configurable resolution (DPI)
- Layout or map view export

Use this tool for:
- Creating publication-ready maps
- Exporting thematic maps
- Generating map images for reports
"""

SEARCH_HINT = "export map layout PDF PNG image print cartography"


class ExportMapInput(BaseModel):
    """Input schema for ExportMapTool."""
    project_path: str | None = Field(
        default=None,
        description="ArcGIS Pro project (.aprx) path, or None for current project"
    )
    output_path: str = Field(
        description="Output file path (extension determines format)"
    )
    format: str = Field(
        default="PDF",
        description="Export format: PDF, PNG, JPEG, TIFF"
    )
    resolution: int = Field(
        default=300,
        description="Output resolution in DPI"
    )
    layout_name: str | None = Field(
        default=None,
        description="Layout name to export (None for first/default)"
    )
    overwrite_output: bool = Field(
        default=False,
        description="Whether to overwrite existing output file"
    )


class ExportMapOutput(BaseModel):
    """Output schema for ExportMapTool."""
    output_path: str
    format: str
    resolution: int
    width_px: int = 0
    height_px: int = 0
    file_size_kb: int = 0


@register_tool
class ExportMapTool(Tool[ExportMapInput, ExportMapOutput]):
    """Tool to export map layouts."""
    
    name = TOOL_NAME
    description = DESCRIPTION
    category = ToolCategory.CARTOGRAPHY
    search_hint = SEARCH_HINT
    input_model = ExportMapInput
    
    def is_read_only(self) -> bool:
        return False  # Creates output files
    
    def get_activity_description(self, input_data: ExportMapInput) -> str:
        return f"Exporting map to {input_data.format}"
    
    def validate_input(self, input_data: ExportMapInput) -> ValidationResult:
        # Validate format
        valid_formats = {"PDF", "PNG", "JPEG", "TIFF", "BMP", "GIF"}
        if input_data.format.upper() not in valid_formats:
            return ValidationResult.failure(
                f"Invalid format: {input_data.format}. Valid: {', '.join(valid_formats)}",
                error_code=1
            )
        
        # Validate resolution
        if input_data.resolution < 72 or input_data.resolution > 600:
            return ValidationResult.failure(
                f"Resolution must be 72-600 DPI, got: {input_data.resolution}",
                error_code=2
            )
        
        return ValidationResult.success()
    
    def call(
        self,
        input_data: ExportMapInput,
        context: ToolContext
    ) -> ToolResult[ExportMapOutput]:
        """Execute map export."""
        if context.dry_run:
            return self._dry_run(input_data)
        
        if context.arcpy_available:
            return self._export_with_arcpy(input_data, context)
        else:
            return self._export_fallback(input_data, context)
    
    def _dry_run(self, input_data: ExportMapInput) -> ToolResult[ExportMapOutput]:
        output = ExportMapOutput(
            output_path=input_data.output_path,
            format=input_data.format,
            resolution=input_data.resolution
        )
        return ToolResult.ok(output)
    
    def _export_with_arcpy(
        self,
        input_data: ExportMapInput,
        context: ToolContext
    ) -> ToolResult[ExportMapOutput]:
        try:
            import arcpy
            
            # Get project
            if input_data.project_path:
                aprx = arcpy.mp.ArcGISProject(input_data.project_path)
            else:
                discovered_project = self._discover_project_path(context)
                if discovered_project:
                    aprx = arcpy.mp.ArcGISProject(discovered_project)
                else:
                    try:
                        aprx = arcpy.mp.ArcGISProject("CURRENT")
                    except Exception:
                        return ToolResult.fail(
                            "No ArcGIS Pro project available. 当前在独立 CLI 环境中无法使用 CURRENT；请在任务中提供 project_path 指向 .aprx，或在 ArcGIS Pro 内运行。",
                            "no_project"
                        )
            
            # Ensure output directory exists
            output_path = Path(input_data.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if input_data.overwrite_output and output_path.exists():
                output_path.unlink()

            maps = aprx.listMaps()
            if maps:
                self._ensure_map_has_data(maps[0], context)
            
            # Get layout
            layouts = aprx.listLayouts()
            if not layouts:
                # Try exporting map view instead
                if maps:
                    return self._export_map_view(maps[0], input_data, context)
                else:
                    return ToolResult.fail(
                        "No layouts or maps in project. 可在 .aprx 中先创建地图/版式，或将 project_path 指向包含地图内容的项目。",
                        "no_content"
                    )
            
            layout = layouts[0]
            if input_data.layout_name:
                matching = [l for l in layouts if l.name == input_data.layout_name]
                if matching:
                    layout = matching[0]
            
            # Export based on format
            fmt = input_data.format.upper()
            if fmt == "PDF":
                layout.exportToPDF(
                    str(output_path),
                    resolution=input_data.resolution
                )
            elif fmt == "PNG":
                layout.exportToPNG(
                    str(output_path),
                    resolution=input_data.resolution
                )
            elif fmt == "JPEG":
                layout.exportToJPEG(
                    str(output_path),
                    resolution=input_data.resolution
                )
            elif fmt == "TIFF":
                layout.exportToTIFF(
                    str(output_path),
                    resolution=input_data.resolution
                )
            else:
                layout.exportToPNG(
                    str(output_path),
                    resolution=input_data.resolution
                )
            
            # Get file size
            file_size_kb = 0
            if output_path.exists():
                file_size_kb = int(output_path.stat().st_size / 1024)
            
            output = ExportMapOutput(
                output_path=str(output_path),
                format=fmt,
                resolution=input_data.resolution,
                file_size_kb=file_size_kb
            )
            
            return ToolResult.ok(output, outputs=[str(output_path)])
            
        except Exception as e:
            return ToolResult.fail(f"Export failed: {e}", "export_error")

    def _discover_project_path(self, context: ToolContext) -> str | None:
        """Try to auto-discover a .aprx under workspace/input and subdirectories."""
        roots = self._candidate_roots(context)
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            aprx_files = sorted(root.rglob("*.aprx"))
            if aprx_files:
                return str(aprx_files[0])
        return None

    def _candidate_roots(self, context: ToolContext) -> list[Path]:
        """Build candidate roots for project/data discovery in priority order."""
        roots: list[Path] = []

        meta_input = context.get_state("input_folder")
        if isinstance(meta_input, str) and meta_input.strip():
            roots.append(Path(meta_input))

        cwd = Path(context.working_directory or ".")
        roots.extend([cwd / "input", cwd / "workspace" / "input", cwd])

        deduped: list[Path] = []
        seen: set[str] = set()
        for p in roots:
            key = str(p.resolve()) if p.exists() else str(p)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)
        return deduped

    def _discover_input_shapefiles(self, context: ToolContext, limit: int = 30) -> list[str]:
        """Discover shapefiles recursively from input/workspace roots."""
        results: list[str] = []
        seen: set[str] = set()
        for root in self._candidate_roots(context):
            if not root.exists() or not root.is_dir():
                continue
            for shp in root.rglob("*.shp"):
                path_text = str(shp)
                if path_text in seen:
                    continue
                seen.add(path_text)
                results.append(path_text)
                if len(results) >= max(1, limit):
                    return results
        return results

    def _ensure_map_has_data(self, map_obj, context: ToolContext) -> int:
        """If map has no layers, try loading shapefiles from input recursively."""
        try:
            existing_layers = map_obj.listLayers()
        except Exception:
            existing_layers = []
        if existing_layers:
            return 0

        added = 0
        for shp_path in self._discover_input_shapefiles(context, limit=20):
            try:
                map_obj.addDataFromPath(shp_path)
                added += 1
            except Exception:
                continue
        return added
    
    def _export_map_view(self, map_obj, input_data: ExportMapInput, context: ToolContext):
        """Export map view when no layout is available."""
        try:
            import arcpy
            
            output_path = Path(input_data.output_path)
            if input_data.overwrite_output and output_path.exists():
                output_path.unlink()
            
            # Export map frame
            map_obj.exportToPNG(
                str(output_path),
                resolution=input_data.resolution
            )
            
            file_size_kb = 0
            if output_path.exists():
                file_size_kb = int(output_path.stat().st_size / 1024)
            
            output = ExportMapOutput(
                output_path=str(output_path),
                format="PNG",  # Map view exports to PNG
                resolution=input_data.resolution,
                file_size_kb=file_size_kb
            )
            
            return ToolResult.ok(output, outputs=[str(output_path)])
            
        except Exception as e:
            return ToolResult.fail(f"Map view export failed: {e}", "export_error")
    
    def _export_fallback(
        self,
        input_data: ExportMapInput,
        context: ToolContext
    ) -> ToolResult[ExportMapOutput]:
        """Fallback when ArcPy not available."""
        # Create a summary file instead
        output_path = Path(input_data.output_path)
        if input_data.overwrite_output and output_path.exists():
            output_path.unlink()
        summary_path = output_path.parent / f"{output_path.stem}_export_pending.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary_path.write_text(
            f"Map export pending (ArcPy not available)\n"
            f"Format: {input_data.format}\n"
            f"Resolution: {input_data.resolution} DPI\n"
            f"Target: {input_data.output_path}\n",
            encoding="utf-8"
        )
        
        return ToolResult.fail(
            "ArcPy not available for map export",
            "arcpy_unavailable"
        )
    
    def render_tool_result_message(self, result: ToolResult[ExportMapOutput]) -> str:
        if not result.success:
            return f"❌ Export failed: {result.error}"
        
        data = result.data
        if data is None:
            return "⚠️ No data returned"
        
        return f"✅ Exported {data.format} ({data.resolution} DPI, {data.file_size_kb} KB) → {data.output_path}"

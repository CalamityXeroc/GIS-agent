"""Bundled skills - Pre-built workflows for common GIS tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Skill, SkillContext, SkillResult, register_skill
from ..core import ToolRegistry, ToolContext


@register_skill
class ThematicMapSkill(Skill):
    """Create thematic maps with data integration.
    
    This skill combines data scanning, merging, projection,
    and map export into a complete workflow.
    """
    
    name = "thematic_map"
    description = "Create a thematic map from multiple data sources"
    tags = ["map", "cartography", "thematic", "export", "pdf"]
    
    steps = [
        "Scan data sources",
        "Merge layers",
        "Project to target CRS",
        "Apply symbology",
        "Export map"
    ]
    
    def get_required_inputs(self) -> list[str]:
        return ["data_sources", "output_path"]
    
    def get_optional_inputs(self) -> dict[str, Any]:
        return {
            "target_crs": "Asia_North_Albers_Equal_Area_Conic",
            "output_format": "PDF",
            "resolution": 300,
            "title": "Thematic Map"
        }
    
    def validate_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str | None]:
        if "data_sources" not in inputs:
            return False, "Missing required input: data_sources"
        if "output_path" not in inputs:
            return False, "Missing required input: output_path"
        
        data_sources = inputs["data_sources"]
        if not isinstance(data_sources, list) or len(data_sources) == 0:
            return False, "data_sources must be a non-empty list"
        
        return True, None
    
    def execute(
        self,
        inputs: dict[str, Any],
        context: SkillContext
    ) -> SkillResult:
        """Execute thematic map workflow."""
        context.total_steps = len(self.steps)
        outputs = []
        
        try:
            registry = ToolRegistry.instance()
            tool_context = ToolContext(
                workspace=context.workspace,
                output_dir=context.output_dir,
                arcpy_available=context.arcpy_available,
                dry_run=context.dry_run
            )
            
            # Step 1: Scan data sources
            context.current_step = 1
            scan_tool = registry.get("scan_layers")
            if scan_tool:
                for source in inputs["data_sources"]:
                    scan_result = scan_tool.call(
                        scan_tool.input_model(directory=source),
                        tool_context
                    )
                    if scan_result.success:
                        context.step_results["scan"] = scan_result.data
            
            # Step 2: Merge layers
            context.current_step = 2
            merge_tool = registry.get("merge_layers")
            if merge_tool:
                merge_result = merge_tool.call(
                    merge_tool.input_model(
                        input_layers=inputs["data_sources"],
                        output_path=str(Path(context.output_dir) / "merged.shp")
                    ),
                    tool_context
                )
                if merge_result.success:
                    context.step_results["merge"] = merge_result.data
                    outputs.extend(merge_result.outputs)
            
            # Step 3: Project
            context.current_step = 3
            project_tool = registry.get("project_layers")
            merged_path = context.step_results.get("merge", {})
            if project_tool and hasattr(merged_path, "output_path"):
                project_result = project_tool.call(
                    project_tool.input_model(
                        input_path=merged_path.output_path,
                        output_path=str(Path(context.output_dir) / "projected.shp"),
                        target_srs=inputs.get("target_crs", "Asia_North_Albers_Equal_Area_Conic")
                    ),
                    tool_context
                )
                if project_result.success:
                    context.step_results["project"] = project_result.data
                    outputs.extend(project_result.outputs)
            
            # Step 4: Apply symbology (placeholder - needs project context)
            context.current_step = 4
            context.step_results["symbology"] = {"status": "skipped", "reason": "requires ArcGIS Pro project"}
            
            # Step 5: Export map
            context.current_step = 5
            export_tool = registry.get("export_map")
            if export_tool:
                export_result = export_tool.call(
                    export_tool.input_model(
                        output_path=inputs["output_path"],
                        format=inputs.get("output_format", "PDF"),
                        resolution=inputs.get("resolution", 300)
                    ),
                    tool_context
                )
                if export_result.success:
                    context.step_results["export"] = export_result.data
                    outputs.extend(export_result.outputs)
            
            return SkillResult.ok(
                data=context.step_results,
                outputs=outputs,
                steps_completed=context.current_step,
                steps_total=context.total_steps
            )
            
        except Exception as e:
            return SkillResult.fail(
                error=str(e),
                error_code="skill_error",
                steps_completed=context.current_step,
                steps_total=context.total_steps
            )


@register_skill
class DataIntegrationSkill(Skill):
    """Integrate multiple GIS data sources.
    
    This skill handles data discovery, format conversion,
    coordinate transformation, and quality validation.
    """
    
    name = "data_integration"
    description = "Integrate and harmonize multiple GIS data sources"
    tags = ["data", "integration", "merge", "transform", "etl"]
    
    steps = [
        "Scan source directories",
        "Validate data quality",
        "Transform coordinates",
        "Merge datasets",
        "Generate summary report"
    ]
    
    def get_required_inputs(self) -> list[str]:
        return ["source_dirs", "output_dir"]
    
    def get_optional_inputs(self) -> dict[str, Any]:
        return {
            "target_crs": None,  # Keep original if not specified
            "validate_geometry": True,
            "fix_errors": False
        }
    
    def validate_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str | None]:
        if "source_dirs" not in inputs:
            return False, "Missing required input: source_dirs"
        if "output_dir" not in inputs:
            return False, "Missing required input: output_dir"
        
        return True, None
    
    def execute(
        self,
        inputs: dict[str, Any],
        context: SkillContext
    ) -> SkillResult:
        """Execute data integration workflow."""
        context.total_steps = len(self.steps)
        outputs = []
        all_layers = []
        
        try:
            registry = ToolRegistry.instance()
            tool_context = ToolContext(
                workspace=context.workspace,
                output_dir=context.output_dir,
                arcpy_available=context.arcpy_available,
                dry_run=context.dry_run
            )
            
            # Step 1: Scan sources
            context.current_step = 1
            scan_tool = registry.get("scan_layers")
            if scan_tool:
                for source_dir in inputs["source_dirs"]:
                    scan_result = scan_tool.call(
                        scan_tool.input_model(directory=source_dir),
                        tool_context
                    )
                    if scan_result.success and scan_result.data:
                        all_layers.extend(scan_result.data.layers)
            
            context.step_results["scan"] = {"layer_count": len(all_layers)}
            
            # Step 2: Validate quality
            context.current_step = 2
            quality_tool = registry.get("quality_check")
            quality_issues = []
            if quality_tool and inputs.get("validate_geometry", True):
                for layer in all_layers[:10]:  # Limit to first 10
                    qc_result = quality_tool.call(
                        quality_tool.input_model(
                            input_path=layer.path,
                            fix_errors=inputs.get("fix_errors", False)
                        ),
                        tool_context
                    )
                    if qc_result.success and qc_result.data:
                        quality_issues.extend(qc_result.data.issues)
            
            context.step_results["quality"] = {"issues_found": len(quality_issues)}
            
            # Step 3: Transform coordinates
            context.current_step = 3
            project_tool = registry.get("project_layers")
            projected_paths = []
            target_crs = inputs.get("target_crs")
            
            if project_tool and target_crs:
                for i, layer in enumerate(all_layers):
                    out_path = str(Path(inputs["output_dir"]) / f"projected_{i}.shp")
                    proj_result = project_tool.call(
                        project_tool.input_model(
                            input_path=layer.path,
                            output_path=out_path,
                            target_srs=target_crs
                        ),
                        tool_context
                    )
                    if proj_result.success:
                        projected_paths.append(out_path)
                        outputs.extend(proj_result.outputs)
            
            context.step_results["transform"] = {"projected_count": len(projected_paths)}
            
            # Step 4: Merge datasets
            context.current_step = 4
            merge_tool = registry.get("merge_layers")
            layers_to_merge = projected_paths if projected_paths else [l.path for l in all_layers]
            
            if merge_tool and layers_to_merge:
                merge_out = str(Path(inputs["output_dir"]) / "integrated.shp")
                merge_result = merge_tool.call(
                    merge_tool.input_model(
                        input_layers=layers_to_merge,
                        output_path=merge_out
                    ),
                    tool_context
                )
                if merge_result.success:
                    context.step_results["merge"] = merge_result.data
                    outputs.extend(merge_result.outputs)
            
            # Step 5: Generate summary
            context.current_step = 5
            summary = {
                "sources_scanned": len(inputs["source_dirs"]),
                "layers_found": len(all_layers),
                "quality_issues": len(quality_issues),
                "layers_projected": len(projected_paths),
                "outputs_created": len(outputs)
            }
            context.step_results["summary"] = summary
            
            # Write summary report
            report_path = Path(inputs["output_dir"]) / "integration_report.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_lines = [
                "Data Integration Report",
                "=" * 40,
                f"Sources scanned: {summary['sources_scanned']}",
                f"Layers found: {summary['layers_found']}",
                f"Quality issues: {summary['quality_issues']}",
                f"Layers projected: {summary['layers_projected']}",
                f"Outputs created: {summary['outputs_created']}",
            ]
            report_path.write_text("\n".join(report_lines), encoding="utf-8")
            outputs.append(str(report_path))
            
            return SkillResult.ok(
                data=context.step_results,
                outputs=outputs,
                steps_completed=context.current_step,
                steps_total=context.total_steps
            )
            
        except Exception as e:
            return SkillResult.fail(
                error=str(e),
                error_code="skill_error",
                steps_completed=context.current_step,
                steps_total=context.total_steps
            )


@register_skill
class QualityAssuranceSkill(Skill):
    """Comprehensive data quality assurance workflow.
    
    This skill performs thorough quality checks and generates
    detailed reports for GIS data validation.
    """
    
    name = "quality_assurance"
    description = "Perform comprehensive quality checks on GIS data"
    tags = ["quality", "validation", "qa", "qc", "check", "report"]
    
    steps = [
        "Scan input data",
        "Check geometry validity",
        "Validate attributes",
        "Check coordinate system",
        "Generate QA report"
    ]
    
    def get_required_inputs(self) -> list[str]:
        return ["input_path"]
    
    def get_optional_inputs(self) -> dict[str, Any]:
        return {
            "output_report": None,
            "fix_errors": False,
            "check_types": ["geometry", "topology", "crs"]
        }
    
    def validate_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str | None]:
        if "input_path" not in inputs:
            return False, "Missing required input: input_path"
        
        input_path = Path(inputs["input_path"])
        if not input_path.exists():
            return False, f"Input path does not exist: {input_path}"
        
        return True, None
    
    def execute(
        self,
        inputs: dict[str, Any],
        context: SkillContext
    ) -> SkillResult:
        """Execute quality assurance workflow."""
        context.total_steps = len(self.steps)
        outputs = []
        
        try:
            registry = ToolRegistry.instance()
            tool_context = ToolContext(
                workspace=context.workspace,
                output_dir=context.output_dir,
                arcpy_available=context.arcpy_available,
                dry_run=context.dry_run
            )
            
            input_path = Path(inputs["input_path"])
            layers_to_check = []
            
            # Step 1: Scan input
            context.current_step = 1
            if input_path.is_dir():
                scan_tool = registry.get("scan_layers")
                if scan_tool:
                    scan_result = scan_tool.call(
                        scan_tool.input_model(directory=str(input_path)),
                        tool_context
                    )
                    if scan_result.success and scan_result.data:
                        layers_to_check = [l.path for l in scan_result.data.layers]
            else:
                layers_to_check = [str(input_path)]
            
            context.step_results["scan"] = {"layers_found": len(layers_to_check)}
            
            # Steps 2-4: Run quality checks
            quality_tool = registry.get("quality_check")
            all_results = []
            
            for i, layer_path in enumerate(layers_to_check):
                context.current_step = 2 + (i % 3)  # Cycle through steps 2-4
                
                if quality_tool:
                    report_path = None
                    if inputs.get("output_report"):
                        report_dir = Path(inputs["output_report"]).parent
                        report_path = str(report_dir / f"qa_report_{i}.txt")
                    
                    qc_result = quality_tool.call(
                        quality_tool.input_model(
                            input_path=layer_path,
                            fix_errors=inputs.get("fix_errors", False),
                            check_types=inputs.get("check_types", ["geometry", "topology", "crs"]),
                            output_report=report_path
                        ),
                        tool_context
                    )
                    if qc_result.success:
                        all_results.append({
                            "layer": layer_path,
                            "passed": qc_result.data.passed if qc_result.data else False,
                            "issues": qc_result.data.issues_found if qc_result.data else 0
                        })
                        outputs.extend(qc_result.outputs)
            
            # Step 5: Generate summary report
            context.current_step = 5
            
            passed_count = sum(1 for r in all_results if r["passed"])
            total_issues = sum(r["issues"] for r in all_results)
            
            summary = {
                "layers_checked": len(layers_to_check),
                "layers_passed": passed_count,
                "layers_failed": len(layers_to_check) - passed_count,
                "total_issues": total_issues,
                "overall_passed": passed_count == len(layers_to_check)
            }
            
            context.step_results["summary"] = summary
            
            # Write summary report if output path specified
            if inputs.get("output_report"):
                report_path = Path(inputs["output_report"])
                report_path.parent.mkdir(parents=True, exist_ok=True)
                
                lines = [
                    "Quality Assurance Summary Report",
                    "=" * 50,
                    f"Layers checked: {summary['layers_checked']}",
                    f"Passed: {summary['layers_passed']}",
                    f"Failed: {summary['layers_failed']}",
                    f"Total issues: {summary['total_issues']}",
                    "",
                    f"Overall status: {'PASSED' if summary['overall_passed'] else 'FAILED'}",
                    "",
                    "Layer Details:",
                    "-" * 40,
                ]
                
                for result in all_results:
                    status = "PASS" if result["passed"] else "FAIL"
                    lines.append(f"[{status}] {result['layer']} ({result['issues']} issues)")
                
                report_path.write_text("\n".join(lines), encoding="utf-8")
                outputs.append(str(report_path))
            
            return SkillResult.ok(
                data=context.step_results,
                outputs=outputs,
                steps_completed=context.current_step,
                steps_total=context.total_steps
            )
            
        except Exception as e:
            return SkillResult.fail(
                error=str(e),
                error_code="skill_error",
                steps_completed=context.current_step,
                steps_total=context.total_steps
            )

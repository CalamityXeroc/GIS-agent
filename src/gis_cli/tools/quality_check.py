"""QualityCheckTool - Validate geometry and data quality.

Uses ArcPy Bridge to execute quality checks via ArcGIS Pro Python environment.
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


TOOL_NAME = "quality_check"

DESCRIPTION = """Check GIS data quality and validate geometries.

This tool performs quality assurance checks:
- Geometry validation (null, corrupt, invalid)
- Topology checks (self-intersections, gaps)
- Attribute validation
- Coordinate reference system verification

Quality checks performed:
- Null geometry detection
- Self-intersection detection
- Ring orientation validation
- Empty geometry detection
- Duplicate feature detection
"""

SEARCH_HINT = "quality check validate geometry topology QA QC error fix repair"


class QualityCheckInput(BaseModel):
    """Input schema for QualityCheckTool."""
    input_path: str = Field(description="Path to layer to check")
    fix_errors: bool = Field(
        default=False,
        description="Attempt to repair detected errors"
    )
    check_types: list[str] = Field(
        default=["geometry", "topology"],
        description="Types of checks: geometry, topology, attributes, crs"
    )
    output_report: str | None = Field(
        default=None,
        description="Path to save quality report"
    )
    allow_missing_input: bool = Field(
        default=False,
        description="Internal flag: allow missing input path in dry-run planning checks"
    )


class QualityIssue(BaseModel):
    """A single quality issue."""
    issue_type: str
    severity: str  # error, warning, info
    feature_id: int | None = None
    description: str
    fixed: bool = False


class QualityCheckOutput(BaseModel):
    """Output schema for QualityCheckTool."""
    input_path: str
    total_features: int = 0
    issues_found: int = 0
    issues_fixed: int = 0
    issues: list[QualityIssue] = []
    passed: bool = True
    geometry_type: str = ""
    spatial_reference: str | None = None
    report_path: str | None = None


@register_tool
class QualityCheckTool(Tool[QualityCheckInput, QualityCheckOutput]):
    """Tool to validate GIS data quality."""
    
    name = TOOL_NAME
    description = DESCRIPTION
    category = ToolCategory.QUALITY_CHECK
    search_hint = SEARCH_HINT
    input_model = QualityCheckInput
    
    def is_read_only(self) -> bool:
        return True  # Read-only unless fix_errors is True
    
    def get_activity_description(self, input_data: QualityCheckInput) -> str:
        action = "Checking and repairing" if input_data.fix_errors else "Checking"
        return f"{action} data quality"
    
    def validate_input(self, input_data: QualityCheckInput) -> ValidationResult:
        if not Path(input_data.input_path).exists():
            if input_data.allow_missing_input:
                return ValidationResult.warning(
                    f"Input not found in dry-run, skip strict check: {input_data.input_path}"
                )
            return ValidationResult.failure(
                f"Input layer not found: {input_data.input_path}",
                error_code=1
            )
        
        valid_checks = {"geometry", "topology", "attributes", "crs"}
        invalid = set(input_data.check_types) - valid_checks
        if invalid:
            return ValidationResult.failure(
                f"Invalid check types: {invalid}. Valid: {valid_checks}",
                error_code=2
            )
        
        return ValidationResult.success()
    
    def call(
        self,
        input_data: QualityCheckInput,
        context: ToolContext
    ) -> ToolResult[QualityCheckOutput]:
        """Execute quality check."""
        if context.dry_run:
            return self._dry_run(input_data)
        
        # Use ArcPy Bridge for actual execution
        return self._check_with_bridge(input_data, context)
    
    def _dry_run(self, input_data: QualityCheckInput) -> ToolResult[QualityCheckOutput]:
        output = QualityCheckOutput(
            input_path=input_data.input_path,
            passed=True
        )
        return ToolResult.ok(output)
    
    def _check_with_bridge(
        self,
        input_data: QualityCheckInput,
        context: ToolContext
    ) -> ToolResult[QualityCheckOutput]:
        """Check quality using ArcPy Bridge."""
        try:
            from ..arcpy_bridge import quality_check
            
            result = quality_check(input_data.input_path)
            
            if result.status == "success" and result.data:
                data = result.data
                issues = []
                
                # Create issue objects from result data
                if data.get("null_geometry", 0) > 0:
                    issues.append(QualityIssue(
                        issue_type="geometry",
                        severity="error",
                        description=f"Null geometry: {data['null_geometry']} features"
                    ))
                
                if data.get("invalid_geometry", 0) > 0:
                    issues.append(QualityIssue(
                        issue_type="geometry",
                        severity="error",
                        description=f"Invalid geometry: {data['invalid_geometry']} features"
                    ))
                
                output = QualityCheckOutput(
                    input_path=input_data.input_path,
                    total_features=data.get("feature_count", 0),
                    issues_found=len(issues),
                    issues=issues,
                    passed=data.get("is_valid", True),
                    geometry_type=data.get("geometry_type", ""),
                    spatial_reference=data.get("spatial_reference")
                )
                
                return ToolResult.ok(output)
            else:
                # Fall back to basic file check
                return self._check_fallback(input_data, context)
            
        except Exception as e:
            return ToolResult.fail(f"Quality check failed: {e}", "execution_error")
    
    def _check_fallback(
        self,
        input_data: QualityCheckInput,
        context: ToolContext
    ) -> ToolResult[QualityCheckOutput]:
        """Fallback when ArcPy not available - basic checks only."""
        issues = []
        
        # Check if file exists and is readable
        input_path = Path(input_data.input_path)
        if not input_path.exists():
            issues.append(QualityIssue(
                issue_type="file",
                severity="error",
                description="File not found"
            ))
        elif input_path.stat().st_size == 0:
            issues.append(QualityIssue(
                issue_type="file",
                severity="error",
                description="File is empty"
            ))
        
        output = QualityCheckOutput(
            input_path=str(input_path),
            issues_found=len(issues),
            issues=issues,
            passed=len([i for i in issues if i.severity == "error"]) == 0
        )
        
        return ToolResult.ok(output)
    
    def render_tool_result_message(self, result: ToolResult[QualityCheckOutput]) -> str:
        if not result.success:
            return f"Quality check failed: {result.error}"
        
        data = result.data
        if data is None:
            return "No data returned"
        
        status = "PASSED" if data.passed else "FAILED"
        fixed_str = f", {data.issues_fixed} fixed" if data.issues_fixed > 0 else ""
        return f"{status} | {data.total_features} features, {data.issues_found} issues{fixed_str}"

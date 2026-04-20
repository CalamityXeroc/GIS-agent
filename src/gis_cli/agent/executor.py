"""Agent executor - Execute plans with tools.

Executes GIS workflow plans by calling tools and handling errors.
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from enum import Enum

from ..core import ToolRegistry, ToolContext, ToolResult, ExecutionContext
from .context_hub import ContextHub
from .planner import Plan, PlanStep, StepStatus


class ExecutionMode(str, Enum):
    """Execution mode."""
    DRY_RUN = "dry_run"
    EXECUTE = "execute"


@dataclass
class StepTrace:
    """Trace of a single step execution."""
    step_id: str
    tool: str
    started_at: datetime
    completed_at: datetime | None = None
    status: str = "running"
    input: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    error: str | None = None
    duration_ms: int = 0


@dataclass
class ExecutionTrace:
    """Complete trace of a plan execution."""
    plan_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    mode: ExecutionMode = ExecutionMode.DRY_RUN
    steps: list[StepTrace] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    success: bool = False
    error: str | None = None
    
    @property
    def duration_ms(self) -> int:
        """Total execution duration."""
        if not self.completed_at:
            return 0
        return int((self.completed_at - self.started_at).total_seconds() * 1000)
    
    def add_step(self, trace: StepTrace) -> None:
        """Add a step trace."""
        self.steps.append(trace)
    
    def complete(self, success: bool, outputs: list[str] | None = None, error: str | None = None):
        """Mark execution as complete."""
        self.completed_at = datetime.now(timezone.utc)
        self.success = success
        self.error = error
        if outputs:
            self.outputs = outputs
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "mode": self.mode.value,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "outputs": self.outputs,
            "steps": [
                {
                    "step_id": s.step_id,
                    "tool": s.tool,
                    "status": s.status,
                    "duration_ms": s.duration_ms,
                    "error": s.error
                }
                for s in self.steps
            ]
        }


@dataclass
class ExecutionResult:
    """Result of plan execution."""
    success: bool
    plan: Plan
    trace: ExecutionTrace
    outputs: list[str] = field(default_factory=list)
    error: str | None = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "error": self.error,
            "outputs": self.outputs,
            "plan": self.plan.to_dict(),
            "trace": self.trace.to_dict()
        }


class Executor:
    """Executes GIS workflow plans.
    
    Features:
    - Step-by-step execution
    - Error handling and recovery
    - Progress callbacks
    - Dry-run support
    """
    
    def __init__(
        self,
        context: ExecutionContext | None = None,
        memory: Any = None,
        on_step_start: Callable[[PlanStep], None] | None = None,
        on_step_complete: Callable[[PlanStep, ToolResult], None] | None = None,
        on_progress: Callable[[str, float], None] | None = None
    ):
        self.context = context or ExecutionContext.build()
        self.on_step_start = on_step_start
        self.on_step_complete = on_step_complete
        self.on_progress = on_progress
        self.registry = ToolRegistry.instance()
        workspace = self.context.workspace.workspace_path or Path.cwd()
        self.context_hub = ContextHub(workspace=workspace, memory=memory)
    
    def execute(
        self,
        plan: Plan,
        mode: ExecutionMode = ExecutionMode.DRY_RUN
    ) -> ExecutionResult:
        """Execute a plan.
        
        Args:
            plan: The plan to execute
            mode: Execution mode (dry_run or execute)
        
        Returns:
            ExecutionResult with success status and outputs
        """
        trace = ExecutionTrace(
            plan_id=plan.id,
            mode=mode
        )
        
        all_outputs = []
        
        try:
            while True:
                # Get next step
                step = plan.next_step
                if step is None:
                    break
                
                # Execute step
                step_trace, result = self._execute_step(step, mode, plan)
                trace.add_step(step_trace)
                
                if result.success:
                    step.complete(result.data)
                    all_outputs.extend(result.outputs)
                    
                    # Report progress
                    self._report_progress(plan)
                else:
                    step.fail(result.error or "Unknown error")
                    
                    # Check if we should continue after failure
                    if not self._should_continue_after_failure(step, plan):
                        trace.complete(False, all_outputs, f"Step {step.id} failed: {result.error}")
                        return ExecutionResult(
                            success=False,
                            plan=plan,
                            trace=trace,
                            outputs=all_outputs,
                            error=result.error
                        )
            
            # Check final status
            if plan.has_failed:
                # Collect all step errors for better error message
                failed_steps = [s for s in plan.steps if s.status == StepStatus.FAILED]
                step_errors = [f"{s.tool}: {s.error}" for s in failed_steps if s.error]
                error_detail = "; ".join(step_errors) if step_errors else "Plan has failed steps"
                trace.complete(False, all_outputs, error_detail)
                return ExecutionResult(
                    success=False,
                    plan=plan,
                    trace=trace,
                    outputs=all_outputs,
                    error=error_detail
                )
            
            trace.complete(True, all_outputs)
            return ExecutionResult(
                success=True,
                plan=plan,
                trace=trace,
                outputs=all_outputs
            )
            
        except Exception as e:
            trace.complete(False, all_outputs, str(e))
            return ExecutionResult(
                success=False,
                plan=plan,
                trace=trace,
                outputs=all_outputs,
                error=str(e)
            )
    
    def _execute_step(
        self,
        step: PlanStep,
        mode: ExecutionMode,
        plan: Plan
    ) -> tuple[StepTrace, ToolResult]:
        """Execute a single step."""
        effective_tool_name, effective_input = self._unwrap_tool_call_payload(step.tool, step.input)
        step_trace = StepTrace(
            step_id=step.id,
            tool=effective_tool_name,
            started_at=datetime.now(timezone.utc),
            input=effective_input
        )
        
        # Notify step start
        step.start()
        if self.on_step_start:
            self.on_step_start(step)
        
        try:
            # Resolve input parameters first
            resolved_input, unresolved_refs = self._resolve_input(effective_input, plan)
            if unresolved_refs:
                reason = self._build_unresolved_reference_error(unresolved_refs)
                result = ToolResult.fail(reason, "unresolved_reference")
                step_trace.completed_at = datetime.now(timezone.utc)
                step_trace.duration_ms = int(
                    (step_trace.completed_at - step_trace.started_at).total_seconds() * 1000
                )
                step_trace.status = "failed"
                step_trace.error = reason
                if self.on_step_complete:
                    self.on_step_complete(step, result)
                return step_trace, result
            
            # Normalize parameter names to match tool's expected schema
            resolved_input = self._normalize_params(effective_tool_name, resolved_input)

            # Tool-specific input fallback from execution context/previous results
            if effective_tool_name in {"quality_check", "project_layers"}:
                current_input = resolved_input.get("input_path")
                needs_infer = "input_path" not in resolved_input
                if effective_tool_name == "project_layers" and isinstance(current_input, str) and current_input.strip():
                    # If planner produced a stale/default path that doesn't exist, prefer inferred paths
                    current_path = Path(current_input)
                    if (not current_path.exists()) and ".gdb\\" not in current_input.lower():
                        needs_infer = True
                if needs_infer:
                    inferred_input = self._infer_input_path(step, plan)
                    if inferred_input:
                        resolved_input["input_path"] = inferred_input
            if effective_tool_name == "quality_check" and mode == ExecutionMode.DRY_RUN:
                resolved_input.setdefault("allow_missing_input", True)
            if effective_tool_name == "project_layers" and "output_path" not in resolved_input:
                inferred_output = self._infer_project_output_path(step, resolved_input.get("input_path"))
                if inferred_output:
                    resolved_input["output_path"] = inferred_output

            preflight = self._run_step_preflight(effective_tool_name, resolved_input, step, plan)
            resolved_input = preflight.get("input", resolved_input)
            if preflight.get("blocked"):
                reason = str(preflight.get("reason") or "Preflight blocked")
                result = ToolResult.fail(reason, "clarification_required")
                step_trace.completed_at = datetime.now(timezone.utc)
                step_trace.duration_ms = int(
                    (step_trace.completed_at - step_trace.started_at).total_seconds() * 1000
                )
                step_trace.status = "failed"
                step_trace.error = reason
                if self.on_step_complete:
                    self.on_step_complete(step, result)
                return step_trace, result
             
            # Special case: merge_layers with filter_pattern
            # Convert to execute_code for dynamic filtering
            actual_tool_name = effective_tool_name
            if effective_tool_name == "merge_layers" and "_filter_pattern" in resolved_input:
                filter_pattern = resolved_input.pop("_filter_pattern")
                output_path = resolved_input.get("output_path", "./workspace/output/merged.shp")
                actual_tool_name, resolved_input = self._resolve_merge_with_filter(
                    step, plan, filter_pattern, output_path
                )
            
            # Get tool
            tool = self.registry.get(actual_tool_name)
            if tool is None:
                result = ToolResult.fail(f"Tool not found: {actual_tool_name}", "tool_not_found")
            else:
                # Build tool context
                workspace_path = self.context.workspace.workspace_path or "."
                snapshot = self.context_hub.discover()
                input_folder = self.context_hub.best_input_root(snapshot)
                tool_context = ToolContext(
                    dry_run=(mode == ExecutionMode.DRY_RUN),
                    arcpy_available=self.context.arcpy_available,
                    working_directory=workspace_path,
                    metadata={
                        "workspace_path": workspace_path,
                        "input_folder": input_folder,
                        "project_path": self.context.workspace.project_path or "",
                        "context_snapshot": snapshot.to_dict(),
                    },
                )

                # Execute via standardized pipeline: parse -> validate -> permission -> call
                result = tool.execute(resolved_input, tool_context)
            
            # Update trace
            step_trace.completed_at = datetime.now(timezone.utc)
            step_trace.duration_ms = int(
                (step_trace.completed_at - step_trace.started_at).total_seconds() * 1000
            )
            
            if result.success:
                step_trace.status = "completed"
                step_trace.output = result.data
            else:
                step_trace.status = "failed"
                step_trace.error = result.error
            
            # Notify step complete
            if self.on_step_complete:
                self.on_step_complete(step, result)
            
            return step_trace, result
            
        except Exception as e:
            # Format the error message better for Pydantic validation errors
            error_msg = str(e)
            if "validation error" in error_msg.lower():
                # Make Pydantic errors more readable
                error_msg = f"参数验证失败: {error_msg}"
            
            step_trace.completed_at = datetime.now(timezone.utc)
            step_trace.status = "failed"
            step_trace.error = error_msg
            step_trace.duration_ms = int(
                (step_trace.completed_at - step_trace.started_at).total_seconds() * 1000
            )
            
            return step_trace, ToolResult.fail(error_msg, "execution_error")

    def _unwrap_tool_call_payload(
        self,
        default_tool_name: str,
        payload: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        """Normalize model-specific tool-call envelopes to (tool_name, args).

        Supported shapes:
        - {"tool_call": {"name": "merge_layers", "arguments": {...}}}
        - {"function_call": {"name": "merge_layers", "arguments": "{...}"}}
        - {"tool": "merge_layers", "arguments": {...}}
        - {"name": "merge_layers", "arguments": {...}}
        """
        if not isinstance(payload, dict):
            return default_tool_name, {}

        tool_name = str(default_tool_name or "").strip()
        args_payload: Any = payload

        # Envelope 1: tool_call / function_call
        for envelope_key in ("tool_call", "function_call"):
            candidate = payload.get(envelope_key)
            if isinstance(candidate, dict):
                name = candidate.get("name") or candidate.get("tool")
                if isinstance(name, str) and name.strip():
                    tool_name = name.strip()
                args_payload = candidate.get("arguments", candidate)
                break

        # Envelope 2: direct "tool"/"name" with arguments
        if args_payload is payload:
            direct_name = payload.get("tool") or payload.get("name")
            if isinstance(direct_name, str) and direct_name.strip() and "arguments" in payload:
                tool_name = direct_name.strip()
                args_payload = payload.get("arguments")

        arguments = self._decode_tool_arguments(args_payload)
        if not isinstance(arguments, dict):
            arguments = {}

        # Preserve extra top-level fields as fallback params when arguments is empty.
        if not arguments and args_payload is payload:
            arguments = {
                k: v
                for k, v in payload.items()
                if k not in {"tool", "name", "arguments", "tool_call", "function_call"}
            }

        return tool_name or default_tool_name, arguments

    def _decode_tool_arguments(self, args_payload: Any) -> Any:
        """Decode arguments payload that might be dict or JSON string."""
        if isinstance(args_payload, dict):
            return args_payload
        if isinstance(args_payload, str):
            text = args_payload.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _run_step_preflight(
        self,
        tool_name: str,
        normalized_input: dict[str, Any],
        step: PlanStep,
        plan: Plan,
    ) -> dict[str, Any]:
        """Preflight guard: auto-fill recoverable gaps and block hard-missing inputs."""
        result = {"input": dict(normalized_input), "blocked": False, "reason": ""}
        payload = result["input"]
        snapshot = self.context_hub.discover()

        if tool_name == "scan_layers":
            if not payload.get("path"):
                payload["path"] = self.context_hub.best_input_root(snapshot)
            payload.setdefault("include_subdirs", True)
            return result

        if tool_name == "export_map":
            if not payload.get("project_path"):
                project_path = self.context_hub.best_project_path(snapshot)
                if project_path:
                    payload["project_path"] = project_path
            if not payload.get("output_path"):
                payload["output_path"] = str(Path(snapshot.output_root) / "map.pdf")

            if not payload.get("project_path"):
                result["blocked"] = True
                result["reason"] = (
                    "缺少可用 ArcGIS Pro 项目(.aprx)。请提供 project_path，"
                    "或将 .aprx 放到 input 子目录后重试。"
                )
            return result

        if tool_name in {"project_layers", "quality_check"}:
            current_input = payload.get("input_path")
            if not isinstance(current_input, str) or not current_input.strip():
                inferred = self._infer_input_path(step, plan)
                if inferred:
                    payload["input_path"] = inferred
                elif snapshot.shapefiles:
                    payload["input_path"] = snapshot.shapefiles[0]

            if tool_name == "project_layers" and not payload.get("output_path"):
                payload["output_path"] = self._infer_project_output_path(step, payload.get("input_path"))

            if tool_name == "project_layers" and not payload.get("target_srs"):
                payload["target_srs"] = self.context.user.default_crs

            if not payload.get("input_path"):
                result["blocked"] = True
                result["reason"] = "缺少 input_path，未能从历史步骤或输入目录自动推断。"
            return result

        if tool_name == "merge_layers":
            layers = payload.get("input_layers")
            if isinstance(layers, str):
                layers = [layers]
            if not isinstance(layers, list) or not layers:
                best_root = self.context_hub.best_input_root(snapshot)
                payload["input_layers"] = [best_root]
            if not payload.get("output_path"):
                payload["output_path"] = str(Path(snapshot.output_root) / "merged.shp")
            return result

        return result
    
    def _resolve_input(self, input_spec: dict, plan: Plan) -> tuple[dict, list[str]]:
        """Resolve input parameters, including references to previous step outputs."""
        import re
        resolved = {}
        unresolved: list[str] = []
        
        for key, value in input_spec.items():
            if isinstance(value, str):
                # Check for reference patterns
                resolved_value = value
                
                # Pattern 1: $step_id.field (e.g., $step_1.layers)
                if value.startswith("$") and "{" not in value:
                    ref = value[1:]
                    parts = ref.split(".", 1)
                    step_id = parts[0]
                    field = parts[1] if len(parts) > 1 else None
                    
                    step = plan.get_step(step_id)
                    if not step:
                        unresolved.append(f"{key}: 引用步骤 {step_id} 不存在")
                    elif step.result is None:
                        unresolved.append(f"{key}: 引用步骤 {step_id} 尚无可用输出")
                    elif field:
                        if hasattr(step.result, field):
                            resolved_value = getattr(step.result, field)
                        elif isinstance(step.result, dict) and field in step.result:
                            resolved_value = step.result[field]
                        else:
                            unresolved.append(f"{key}: 步骤 {step_id} 缺少字段 {field}")
                    else:
                        resolved_value = step.result
                
                # Pattern 2: ${step_id.output.field} or ${step_1.result.field}
                # Also handles: {step_id.result.field}
                elif "{" in value and "}" in value:
                    # Match patterns like ${step_1.output.layers} or {step_1.result.layers}
                    patterns = [
                        r"\$?\{(step_\d+)\.output\.(\w+)\}",    # ${step_1.output.layers}
                        r"\$?\{(step_\d+)\.result\.(\w+)\}",    # {step_1.result.layers}
                        r"\$?\{(step_\d+)\.(\w+)\}",            # {step_1.layers}
                    ]
                    
                    for pattern in patterns:
                        match = re.match(pattern, value)
                        if match:
                            step_id = match.group(1)
                            field = match.group(2)
                            
                            step = plan.get_step(step_id)
                            if not step:
                                unresolved.append(f"{key}: 引用步骤 {step_id} 不存在")
                                break
                            if step.result is None:
                                unresolved.append(f"{key}: 引用步骤 {step_id} 尚无可用输出")
                                break
                            # Try to get field from result object
                            if hasattr(step.result, field):
                                resolved_value = getattr(step.result, field)
                                break
                            if isinstance(step.result, dict) and field in step.result:
                                resolved_value = step.result[field]
                                break
                            unresolved.append(f"{key}: 步骤 {step_id} 缺少字段 {field}")
                            break
                
                # Post-process resolved value: convert LayerInfo objects to paths
                resolved_value = self._convert_layer_infos(resolved_value)
                resolved[key] = resolved_value
            else:
                resolved[key] = value
        
        return resolved, unresolved

    def _build_unresolved_reference_error(self, unresolved_refs: list[str]) -> str:
        unique = list(dict.fromkeys(unresolved_refs))
        details = "\n".join(f"- {item}" for item in unique)
        return (
            "步骤输入存在未解析引用，已阻止继续执行。\n"
            f"{details}\n"
            "请先确认依赖步骤已成功执行，并返回被引用的字段。"
        )
    
    def _convert_layer_infos(self, value):
        """Convert LayerInfo objects to path strings for tool input."""
        # Import here to avoid circular imports
        try:
            from ..tools.scan_layers import LayerInfo
        except ImportError:
            return value
        
        if isinstance(value, LayerInfo):
            # Single LayerInfo - return path
            return value.path
        elif isinstance(value, list):
            # List of items - convert each LayerInfo to path
            result = []
            for item in value:
                if isinstance(item, LayerInfo):
                    result.append(item.path)
                else:
                    result.append(item)
            return result
        
        return value

    def _infer_input_path(self, step: PlanStep, plan: Plan) -> str | None:
        """Infer an input path from dependency results or workspace defaults."""
        candidates: list[str] = []
        typed_candidates: list[tuple[str, str]] = []

        # 1) Prefer direct dependencies
        for dep_id in reversed(step.depends_on):
            dep = plan.get_step(dep_id)
            if dep and dep.result:
                candidates.extend(self._extract_candidate_paths(dep.result))
                typed_candidates.extend(self._extract_typed_candidate_paths(dep.result))

        # 2) Then any completed step outputs
        for done_step in reversed(plan.get_completed_steps()):
            if done_step.result:
                candidates.extend(self._extract_candidate_paths(done_step.result))
                typed_candidates.extend(self._extract_typed_candidate_paths(done_step.result))

        selected_by_geometry = self._select_candidate_by_geometry_hint(step, typed_candidates)
        if selected_by_geometry:
            return selected_by_geometry

        for path in candidates:
            if isinstance(path, str) and path.strip():
                return path

        # 3) Finally workspace-level defaults
        workspace = self.context.workspace.workspace_path
        if workspace:
            ws = Path(str(workspace))
            defaults = [
                ws / "output" / "projected.shp",
                ws / "output" / "merged.shp",
                ws / "input",
            ]
            for p in defaults:
                if p.exists():
                    return str(p)
        return None

    def _extract_candidate_paths(self, result: Any) -> list[str]:
        """Extract possible layer paths from step result payloads."""
        paths: list[str] = []

        # Pydantic-like objects
        for attr in ("output_path", "input_path", "path"):
            value = getattr(result, attr, None)
            if isinstance(value, str) and value.strip():
                paths.append(value)

        # dict payload
        if isinstance(result, dict):
            for key in ("output_path", "input_path", "path", "output"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    paths.append(value)
            layers = result.get("layers")
            if isinstance(layers, list):
                for layer in layers:
                    layer_path = None
                    if isinstance(layer, dict):
                        layer_path = layer.get("path")
                    else:
                        layer_path = getattr(layer, "path", None)
                    if isinstance(layer_path, str) and layer_path.strip():
                        paths.append(layer_path)

        # object with layers attr
        layers = getattr(result, "layers", None)
        if isinstance(layers, list):
            for layer in layers:
                layer_path = getattr(layer, "path", None)
                if isinstance(layer_path, str) and layer_path.strip():
                    paths.append(layer_path)

        return paths

    def _extract_typed_candidate_paths(self, result: Any) -> list[tuple[str, str]]:
        """Extract candidate paths with geometry type hints (if available)."""
        typed: list[tuple[str, str]] = []

        def _append(path: Any, geom_type: Any) -> None:
            if isinstance(path, str) and path.strip():
                typed.append((path, str(geom_type or "")))

        if isinstance(result, dict):
            layers = result.get("layers")
            if isinstance(layers, list):
                for layer in layers:
                    if isinstance(layer, dict):
                        _append(layer.get("path"), layer.get("type"))
                    else:
                        _append(getattr(layer, "path", None), getattr(layer, "type", None))
            return typed

        layers = getattr(result, "layers", None)
        if isinstance(layers, list):
            for layer in layers:
                _append(getattr(layer, "path", None), getattr(layer, "type", None))

        return typed

    def _select_candidate_by_geometry_hint(
        self,
        step: PlanStep,
        typed_candidates: list[tuple[str, str]],
    ) -> str | None:
        """Select line/polygon path when step description implies geometry type."""
        if not typed_candidates:
            return None

        hint = f"{step.description} {step.tool}".lower()
        wants_line = ("线" in hint) or ("line" in hint) or ("polyline" in hint)
        wants_polygon = ("面" in hint) or ("polygon" in hint)

        if not wants_line and not wants_polygon:
            return None

        for path, geom_type in typed_candidates:
            g = geom_type.lower()
            if wants_line and ("polyline" in g or "line" in g):
                return path
            if wants_polygon and "polygon" in g:
                return path

        return None

    def _infer_project_output_path(self, step: PlanStep, input_path: str | None) -> str | None:
        """Infer a stable output path for project_layers when LLM omitted it."""
        workspace = self.context.workspace.workspace_path
        base_dir = Path(str(workspace)) / "output" if workspace else Path("./workspace/output")
        hint = f"{step.description} {step.tool}".lower()

        suffix = "projected"
        if ("线" in hint) or ("line" in hint) or ("polyline" in hint):
            suffix = "projected_line"
        elif ("面" in hint) or ("polygon" in hint):
            suffix = "projected_polygon"

        # Preserve source basename when available and not a bare directory.
        if input_path:
            src = Path(input_path)
            if src.name and src.suffix:
                return str(base_dir / f"{src.stem}_{suffix}.shp")
            if src.name and ".gdb" in input_path.lower():
                return str(base_dir / f"{src.name}_{suffix}.shp")

        return str(base_dir / f"{suffix}.shp")
    
    def _normalize_params(self, tool_name: str, params: dict) -> dict:
        """Normalize parameter names to match tool's expected schema.
        
        LLM may generate parameter names that are slightly different from
        what the tool expects. This method maps common variations.
        """
        # Parameter name mappings for each tool
        # Format: {tool_name: {alias: correct_name}}
        param_mappings = {
            "merge_layers": {
                "inputs": "input_layers",
                "input": "input_layers",
                "layers": "input_layers",
                "input_paths": "input_layers",
                "source_layers": "input_layers",
                "output": "output_path",
                "out_path": "output_path",
                "type": "layer_type",
                "overwrite": "overwrite_output",
                "force": "overwrite_output",
                "replace": "overwrite_output",
                "覆盖": "overwrite_output",
            },
            "scan_layers": {
                "directory": "path",
                "folder": "path",
                "input_path": "path",
                "input": "path",
                "recursive": "include_subdirs",
                "recurse": "include_subdirs",
                "subdirs": "include_subdirs",
            },
            "project_layers": {
                "input": "input_path",
                "source": "input_path",
                "source_path": "input_path",
                "input_layers": "input_path",
                "layers": "input_path",
                "input_paths": "input_path",
                "source_layers": "input_path",
                "output": "output_path",
                "out": "output_path",
                "out_path": "output_path",
                "output_place": "output_path",
                "output_dir": "output_path",
                "destination": "output_path",
                "dest": "output_path",
                "target_path": "output_path",
                "target": "target_srs",
                "crs": "target_srs",
                "srs": "target_srs",
                "projection": "target_srs",
                "overwrite": "overwrite_output",
                "force": "overwrite_output",
                "replace": "overwrite_output",
                "覆盖": "overwrite_output",
            },
            "quality_check": {
                "input": "input_path",
                "path": "input_path",
                "source": "input_path",
                "input_layers": "input_path",
                "layers": "input_path",
                "input_paths": "input_path",
                "source_layers": "input_path",
                "checks": "check_types",
                "fix": "fix_errors",
                "repair": "fix_errors",
                "report": "output_report",
                "coordinate_system": "check_types",
                "coordinate_reference_system": "check_types",
                "crs_check": "check_types",
            },
            "export_map": {
                "project": "project_path",
                "aprx": "project_path",
                "output": "output_path",
                "out_path": "output_path",
                "dpi": "resolution",
                "layout": "layout_name",
                "overwrite": "overwrite_output",
                "force": "overwrite_output",
                "replace": "overwrite_output",
                "覆盖": "overwrite_output",
            },
            "execute_code": {
                "script": "code",
                "python_code": "code",
                "code_snippet": "code",
                "task": "description",
                "task_description": "description",
                "workdir": "workspace",
                "working_dir": "workspace",
                "timeout": "timeout_seconds",
            },
        }
        
        # Get mappings for this tool
        mappings = param_mappings.get(tool_name, {})
        
        # Create normalized params
        normalized = {}
        for key, value in params.items():
            # Check if this key needs to be mapped
            correct_key = mappings.get(key, key)
            normalized[correct_key] = value

        # Generic fallback: if planner/LLM nests actual args under "input"
        if "input" in normalized and isinstance(normalized["input"], dict):
            nested = normalized.pop("input")
            for key, value in nested.items():
                correct_key = mappings.get(key, key)
                # do not clobber already-normalized explicit top-level values
                normalized.setdefault(correct_key, value)

        # If alias mapping turned "input" into a dict-valued canonical field
        # (e.g. project_layers: input -> input_path), flatten it as nested args.
        if "input_path" in normalized and isinstance(normalized["input_path"], dict):
            nested = normalized.pop("input_path")
            for key, value in nested.items():
                correct_key = mappings.get(key, key)
                normalized.setdefault(correct_key, value)
        
        # Special handling: ensure input_layers is a list for merge_layers
        if tool_name == "merge_layers" and "input_layers" in normalized:
            if isinstance(normalized["input_layers"], str):
                normalized["input_layers"] = [normalized["input_layers"]]

        # Common placeholder/LLM artifacts normalization
        # e.g. "从 step_2 的输出图层", "{step_2.output.layers}", "$step_2.layers"
        def _looks_like_placeholder(val: Any) -> bool:
            if not isinstance(val, str):
                return False
            text = val.strip()
            if not text:
                return True
            return (
                text.startswith("$step_")
                or ("{step_" in text and "}" in text)
                or ("step_" in text and ("输出" in text or "output" in text.lower()))
                or text.lower() in {"none", "null", "未提供", "待补充"}
                or "待确认" in text
                or "请提供输入图层路径" in text
            )

        # Special handling: quality_check requires a single input_path string
        if tool_name == "quality_check" and "input_path" in normalized:
            if isinstance(normalized["input_path"], list):
                # LLM sometimes passes input_layers/list for quality_check
                # Use first resolved layer path as the check target.
                if normalized["input_path"]:
                    normalized["input_path"] = normalized["input_path"][0]
            elif _looks_like_placeholder(normalized["input_path"]):
                normalized.pop("input_path", None)

        if tool_name == "quality_check":
            if "check_types" in normalized and isinstance(normalized["check_types"], str):
                normalized["check_types"] = [normalized["check_types"]]

            if "check_types" in normalized and isinstance(normalized["check_types"], list):
                check_alias = {
                    "coordinate_system": "crs",
                    "coordinate reference system": "crs",
                    "coordinate": "crs",
                    "projection": "crs",
                    "geometry_validity": "geometry",
                    "topology_validity": "topology",
                }
                normalized["check_types"] = [
                    check_alias.get(str(item).strip().lower(), item)
                    for item in normalized["check_types"]
                ]

            if "input_path" not in normalized:
                # Prefer explicit source path aliases
                for alias in ("input_layers", "layers", "source_layers", "source_path", "source", "input"):
                    if alias in params:
                        value = params[alias]
                        if isinstance(value, list):
                            candidates = [v for v in value if isinstance(v, str) and v.strip() and not _looks_like_placeholder(v)]
                            if candidates:
                                normalized["input_path"] = candidates[0]
                                break
                        elif isinstance(value, str) and value.strip() and not _looks_like_placeholder(value):
                            normalized["input_path"] = value
                            break

            if "input_path" not in normalized:
                # Recovery step may only provide workspace-style defaults
                workspace = normalized.get("workspace") or params.get("workspace")
                if isinstance(workspace, str) and workspace.strip():
                    normalized["input_path"] = str(Path(workspace) / "projected.shp")

        # Special handling: project_layers requires single input_path and output_path
        if tool_name == "project_layers" and "input_path" in normalized:
            if isinstance(normalized["input_path"], list) and normalized["input_path"]:
                normalized["input_path"] = normalized["input_path"][0]
            elif _looks_like_placeholder(normalized["input_path"]):
                normalized.pop("input_path", None)
        
        if tool_name == "project_layers":
            # Heuristic: some LLM plans provide only workspace/output directory
            # instead of explicit output_path.
            if "output_path" not in normalized:
                workspace = normalized.get("workspace")
                if isinstance(workspace, str) and workspace.strip():
                    normalized["output_path"] = str(Path(workspace) / "projected.shp")

            # Heuristic: if input_path still missing, salvage from commonly-seen aliases.
            if "input_path" not in normalized:
                for alias in ("input_layers", "layers", "source_layers", "source_path", "source", "input"):
                    if alias in params:
                        value = params[alias]
                        if isinstance(value, list):
                            candidates = [v for v in value if isinstance(v, str) and v.strip() and not _looks_like_placeholder(v)]
                            if candidates:
                                normalized["input_path"] = candidates[0]
                                break
                        if isinstance(value, str) and value.strip() and not _looks_like_placeholder(value):
                            normalized["input_path"] = value
                            break

            # Fallback: if still missing, try deriving from output path/workspace
            if "input_path" not in normalized:
                output_path = normalized.get("output_path")
                if isinstance(output_path, str) and output_path.strip():
                    output_parent = Path(output_path).parent
                    if str(output_parent) and str(output_parent) != ".":
                        normalized["input_path"] = str(output_parent / "merged.shp")
                if "input_path" not in normalized:
                    workspace = normalized.get("workspace") or params.get("workspace")
                    if isinstance(workspace, str) and workspace.strip():
                        normalized["input_path"] = str(Path(workspace) / "merged.shp")

        if tool_name == "scan_layers":
            path_value = normalized.get("path")
            if isinstance(path_value, str) and path_value.strip():
                p = Path(path_value)
                if not p.exists():
                    alt = Path.cwd() / path_value
                    if alt.exists():
                        normalized["path"] = str(alt)
                    else:
                        # Agent-like autodiscovery: infer common workspace input folders
                        candidates = [
                            Path.cwd() / "workspace" / "input",
                            Path.cwd() / "input",
                        ]
                        # If current path ends with input, also try workspace\\input peer
                        if p.name.lower() == "input" and p.parent:
                            candidates.append(p.parent / "workspace" / "input")
                        for c in candidates:
                            if c.exists() and c.is_dir():
                                normalized["path"] = str(c)
                                break

        # Special handling: filter_pattern for merge_layers
        # If filter_pattern is provided but no input_layers, we need to use execute_code instead
        if tool_name == "merge_layers" and "filter_pattern" in normalized:
            if "input_layers" not in normalized:
                # This will cause validation error - the planner should use execute_code instead
                # But we can provide a hint by setting a placeholder
                normalized["_filter_pattern"] = normalized.pop("filter_pattern")

        # Special handling: execute_code requires `code`
        if tool_name == "execute_code" and "code" not in normalized:
            description = str(normalized.get("description", "")).strip()
            task_text = " ".join([
                description,
                str(params.get("task", "")),
                str(params.get("task_description", "")),
            ]).lower()
            create_gdb_markers = [
                "文件地理数据库", "地理数据库", "创建数据库", "create geodatabase", "file geodatabase", "createfilegdb", ".gdb"
            ]
            is_create_gdb_task = any(marker in task_text for marker in create_gdb_markers)

            if is_create_gdb_task:
                workspace = normalized.get("workspace") or params.get("workspace")
                default_output_dir = str(Path(workspace)) if isinstance(workspace, str) and workspace.strip() else "./workspace/output"

                requested_output = (
                    normalized.get("output_path")
                    or params.get("output_path")
                    or params.get("gdb_path")
                    or params.get("output")
                    or str(Path(default_output_dir) / "GIS_Database.gdb")
                )

                normalized["code"] = (
                    "import arcpy\n"
                    "import os\n"
                    "import time\n"
                    f"requested_output = {str(requested_output)!r}\n"
                    "if not requested_output.lower().endswith('.gdb'):\n"
                    "    requested_output = requested_output + '.gdb'\n"
                    "output_dir = os.path.dirname(requested_output) or './workspace/output'\n"
                    "os.makedirs(output_dir, exist_ok=True)\n"
                    "gdb_name = os.path.basename(requested_output)\n"
                    "target = os.path.join(output_dir, gdb_name)\n"
                    "if arcpy.Exists(target):\n"
                    "    ts = time.strftime('%Y%m%d_%H%M%S')\n"
                    "    stem = os.path.splitext(gdb_name)[0]\n"
                    "    gdb_name = f'{stem}_{ts}.gdb'\n"
                    "    target = os.path.join(output_dir, gdb_name)\n"
                    "arcpy.management.CreateFileGDB(output_dir, gdb_name)\n"
                    "set_result({\n"
                    "    'output': target,\n"
                    "    'gdb_path': target,\n"
                    "    'created': True\n"
                    "})\n"
                    "print(f'Created FileGDB: {target}')\n"
                )
            else:
                input_layers = normalized.get("input_layers") or params.get("input_layers")
                target_srs = normalized.get("target_srs") or params.get("target_srs") or "CGCS2000"

                if isinstance(input_layers, list) and input_layers:
                    source_expr = repr(str(input_layers[0]))
                elif isinstance(input_layers, str) and input_layers.strip():
                    source_expr = repr(input_layers.strip())
                else:
                    inferred = None
                    workspace = normalized.get("workspace") or params.get("workspace")
                    if isinstance(workspace, str) and workspace.strip():
                        inferred = str(Path(workspace) / "input")
                    source_expr = repr(inferred or "./workspace/input")

                # Safe fallback snippet: minimal ArcPy task stub with explicit result contract
                normalized["code"] = (
                    "import arcpy\n"
                    f"task = {description!r}\n"
                    f"input_source = {source_expr}\n"
                    f"target_srs = {str(target_srs)!r}\n"
                    "set_result({\n"
                    "    'task': task,\n"
                    "    'input_source': input_source,\n"
                    "    'target_srs': target_srs,\n"
                    "    'status': 'fallback_code_generated'\n"
                    "})\n"
                )

        if tool_name == "export_map":
            current_project = normalized.get("project_path")
            if isinstance(current_project, str) and _looks_like_placeholder(current_project):
                normalized.pop("project_path", None)

            if "project_path" not in normalized:
                context_project = self.context.workspace.project_path
                if isinstance(context_project, str) and context_project.strip() and Path(context_project).exists():
                    normalized["project_path"] = context_project

            if "project_path" not in normalized:
                workspace_hint = (
                    self.context.workspace.workspace_path
                    or normalized.get("workspace")
                    or params.get("workspace")
                )
                if isinstance(workspace_hint, str) and workspace_hint.strip():
                    ws = Path(workspace_hint)
                    if ws.exists() and ws.is_dir():
                        aprx_files = sorted(ws.rglob("*.aprx"))
                        if aprx_files:
                            normalized["project_path"] = str(aprx_files[0])
        
        return normalized
    
    def _resolve_merge_with_filter(
        self,
        step: PlanStep,
        plan: Plan,
        filter_pattern: str,
        output_path: str
    ) -> tuple[str, dict]:
        """Convert a merge_layers step with filter_pattern to execute_code.
        
        When LLM generates a merge_layers with filter_pattern, we need to
        convert it to execute_code that can filter layers from scan results.
        """
        # Get scan_layers result from previous step
        scan_step = None
        for prev_id in step.depends_on:
            prev_step = plan.get_step(prev_id)
            if prev_step and prev_step.tool == "scan_layers" and prev_step.result:
                scan_step = prev_step
                break
        
        # Generate code to filter and merge
        code = f'''
import arcpy
import os

# 从扫描结果中筛选图层
filter_pattern = "{filter_pattern.lower()}"
output_path = r"{output_path}"

# 获取输入路径
input_folder = r"./workspace/input"
layers_to_merge = []

# 遍历文件夹查找匹配的图层
for root, dirs, files in os.walk(input_folder):
    for f in files:
        if f.lower().endswith('.shp') and filter_pattern in f.lower():
            layers_to_merge.append(os.path.join(root, f))

if not layers_to_merge:
    raise ValueError(f"未找到匹配 '{{filter_pattern}}' 的图层")

# 执行合并
arcpy.Merge_management(layers_to_merge, output_path)

# 统计结果
count = int(arcpy.GetCount_management(output_path)[0])
set_result({{
    "output": output_path,
    "feature_count": count,
    "merged_layers": len(layers_to_merge),
    "filter": filter_pattern
}})
'''
        return "execute_code", {
            "code": code.strip(),
            "workspace": str(Path(output_path).parent),
            "description": f"筛选并合并包含 '{filter_pattern}' 的图层"
        }
    
    def _should_continue_after_failure(self, failed_step: PlanStep, plan: Plan) -> bool:
        """Determine if execution should continue after a step failure."""
        # Check if any remaining steps depend on the failed step
        for step in plan.get_pending_steps():
            if failed_step.id in step.depends_on:
                # Dependent step exists, cannot continue
                return False
        
        # No dependencies, can skip and continue
        return True
    
    def _report_progress(self, plan: Plan) -> None:
        """Report execution progress."""
        if self.on_progress:
            completed = len(plan.get_completed_steps())
            total = len(plan.steps)
            percent = (completed / total * 100) if total > 0 else 0
            message = f"Completed {completed}/{total} steps"
            self.on_progress(message, percent)
    
    def execute_single_tool(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        dry_run: bool = True
    ) -> ToolResult:
        """Execute a single tool directly."""
        effective_tool_name, effective_input = self._unwrap_tool_call_payload(tool_name, input_data)

        tool = self.registry.get(effective_tool_name)
        if tool is None:
            return ToolResult.fail(f"Tool not found: {effective_tool_name}", "tool_not_found")
        
        tool_context = ToolContext(
            dry_run=dry_run,
            arcpy_available=self.context.arcpy_available,
            working_directory=self.context.workspace.workspace_path or "."
        )
        
        try:
            normalized_input = self._normalize_params(effective_tool_name, effective_input)
            return tool.execute(normalized_input, tool_context)
        except Exception as e:
            return ToolResult.fail(str(e), "execution_error")

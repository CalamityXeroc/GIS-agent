"""Agent planner - LLM-based task planning.

Plans multi-step GIS workflows by selecting appropriate tools and skills.
"""

from __future__ import annotations

import json
import re
import os
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from enum import Enum

from .model_adaptation import BAMLBridge, PromptAdapter, PlanStandardizer

# 默认输出路径配置
DEFAULT_OUTPUT_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "workspace", "output")
)
if not os.path.isabs(DEFAULT_OUTPUT_FOLDER):
    # 如果路径不绝对，使用相对于当前工作目录的路径
    DEFAULT_OUTPUT_FOLDER = os.path.abspath(os.path.join(os.getcwd(), "workspace", "output"))


class StepStatus(str, Enum):
    """Status of a plan step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in an execution plan."""
    id: str
    tool: str
    description: str
    input: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    def start(self) -> None:
        """Mark step as running."""
        self.status = StepStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
    
    def complete(self, result: Any) -> None:
        """Mark step as completed."""
        self.status = StepStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now(timezone.utc)
    
    def fail(self, error: str) -> None:
        """Mark step as failed."""
        self.status = StepStatus.FAILED
        self.error = error
        self.completed_at = datetime.now(timezone.utc)
    
    def skip(self, reason: str = "") -> None:
        """Mark step as skipped."""
        self.status = StepStatus.SKIPPED
        self.error = reason
        self.completed_at = datetime.now(timezone.utc)
    
    @property
    def duration_ms(self) -> int:
        """Get step duration in milliseconds."""
        if not self.started_at:
            return 0
        end = self.completed_at or datetime.now(timezone.utc)
        return int((end - self.started_at).total_seconds() * 1000)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result_dict = self.result
        if hasattr(result_dict, "model_dump"):
            result_dict = result_dict.model_dump()
        elif hasattr(result_dict, "dict"):
            result_dict = result_dict.dict()
            
        return {
            "id": self.id,
            "tool": self.tool,
            "description": self.description,
            "input": self.input,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": result_dict,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms
        }


@dataclass
class Plan:
    """An execution plan consisting of multiple steps."""
    id: str
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_complete(self) -> bool:
        """Check if all steps are complete or skipped."""
        return all(
            s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
            for s in self.steps
        )
    
    @property
    def has_failed(self) -> bool:
        """Check if any step has failed."""
        return any(s.status == StepStatus.FAILED for s in self.steps)
    
    @property
    def current_step(self) -> PlanStep | None:
        """Get the current running step."""
        for step in self.steps:
            if step.status == StepStatus.RUNNING:
                return step
        return None
    
    @property
    def next_step(self) -> PlanStep | None:
        """Get the next pending step that has all dependencies satisfied."""
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            
            # Check dependencies
            deps_satisfied = all(
                self.get_step(dep_id) and 
                self.get_step(dep_id).status == StepStatus.COMPLETED
                for dep_id in step.depends_on
            )
            
            if deps_satisfied:
                return step
        return None
    
    def get_step(self, step_id: str) -> PlanStep | None:
        """Get step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def get_completed_steps(self) -> list[PlanStep]:
        """Get all completed steps."""
        return [s for s in self.steps if s.status == StepStatus.COMPLETED]
    
    def get_failed_steps(self) -> list[PlanStep]:
        """Get all failed steps."""
        return [s for s in self.steps if s.status == StepStatus.FAILED]
    
    def get_pending_steps(self) -> list[PlanStep]:
        """Get all pending steps."""
        return [s for s in self.steps if s.status == StepStatus.PENDING]
    
    def summary(self) -> dict[str, Any]:
        """Get plan summary."""
        return {
            "id": self.id,
            "goal": self.goal,
            "total_steps": len(self.steps),
            "completed": len(self.get_completed_steps()),
            "failed": len(self.get_failed_steps()),
            "pending": len(self.get_pending_steps()),
            "is_complete": self.is_complete,
            "has_failed": self.has_failed
        }
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "expected_outputs": self.expected_outputs,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "summary": self.summary()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        """Create Plan from dictionary."""
        steps: list[PlanStep] = []
        for s in data.get("steps", []):
            if not isinstance(s, dict):
                continue
            step = PlanStep(
                id=s.get("id", ""),
                tool=s.get("tool", ""),
                description=s.get("description", ""),
                input=s.get("input", {}) if isinstance(s.get("input"), dict) else {},
                depends_on=s.get("depends_on", []) if isinstance(s.get("depends_on"), list) else [],
            )
            status_raw = s.get("status")
            if isinstance(status_raw, str):
                try:
                    step.status = StepStatus(status_raw)
                except ValueError:
                    step.status = StepStatus.PENDING
            step.result = s.get("result")
            error_value = s.get("error")
            step.error = str(error_value) if error_value else None
            started_raw = s.get("started_at")
            completed_raw = s.get("completed_at")
            if isinstance(started_raw, str) and started_raw:
                try:
                    step.started_at = datetime.fromisoformat(started_raw)
                except ValueError:
                    step.started_at = None
            if isinstance(completed_raw, str) and completed_raw:
                try:
                    step.completed_at = datetime.fromisoformat(completed_raw)
                except ValueError:
                    step.completed_at = None
            steps.append(step)

        plan = cls(
            id=data.get("id", f"plan_{datetime.now().strftime('%Y%m%d%H%M%S')}"),
            goal=data.get("goal", ""),
            steps=steps,
            expected_outputs=data.get("expected_outputs", [])
        )
        created_raw = data.get("created_at")
        if isinstance(created_raw, str) and created_raw:
            try:
                plan.created_at = datetime.fromisoformat(created_raw)
            except ValueError:
                pass
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            plan.metadata = metadata
        return plan


class AgentPlanner:
    """LLM-based planner for GIS tasks.
    
    Uses LLM to:
    1. Understand the task
    2. Break it into steps
    3. Select appropriate tools
    4. Generate execution plan
    """
    
    def __init__(
        self,
        llm_client: Any = None,
        available_tools: list[str] | None = None,
        available_skills: list[str] | None = None
    ):
        self.llm_client = llm_client
        self.available_tools = available_tools or [
            "scan_layers",
            "merge_layers",
            "project_layers",
            "export_map",
            "quality_check",
            "execute_code"  # 核心扩展工具
        ]
        self.available_skills = available_skills or [
            "thematic_map",
            "data_integration",
            "quality_assurance"
        ]
        config = getattr(self.llm_client, "config", None)
        enable_prompt_optimizer = True
        enable_baml_standardizer = True
        enable_baml_builtin_fallback = False
        baml_function_map: dict[str, list[str]] = {}
        if config is not None:
            enable_prompt_optimizer = bool(getattr(config, "enable_prompt_optimizer", True))
            enable_baml_standardizer = bool(getattr(config, "enable_baml_standardizer", True))
            enable_baml_builtin_fallback = bool(getattr(config, "enable_baml_builtin_fallback", True))
            raw_map = getattr(config, "baml_functions", None)
            if isinstance(raw_map, dict):
                baml_function_map = raw_map

        self.prompt_adapter = PromptAdapter(enable_prompt_optimizer=enable_prompt_optimizer)
        self.plan_standardizer = PlanStandardizer(enable_baml=enable_baml_standardizer)
        self.baml_bridge = BAMLBridge(
            enabled=enable_baml_standardizer,
            function_map=baml_function_map,
            allow_builtin_fallback=enable_baml_builtin_fallback,
        )
    
    def plan(
        self,
        task_description: str,
        context: dict[str, Any] | None = None,
        expert_mode: bool = True,
    ) -> Plan:
        """Generate an execution plan for a task.

        If LLM is available, uses it for planning.
        In expert mode, injects GIS domain knowledge into the prompt.
        Otherwise falls back to rule-based planning.
        """
        if self.llm_client:
            if expert_mode:
                return self._plan_with_llm_expert(task_description, context)
            return self._plan_with_llm(task_description, context)
        else:
            return self._plan_with_rules(task_description, context)
    
    def _plan_with_llm(
        self,
        task_description: str,
        context: dict[str, Any] | None
    ) -> Plan:
        """Generate plan using LLM."""
        from .prompts import SystemPrompts

        prompts = self._build_planning_prompts(task_description, context)

        # Call LLM with progressive constraints
        fallback_reason = "invalid_json_or_schema"

        # Prefer BAML-generated function contracts when available.
        baml_plan = self.baml_bridge.generate_plan(task_description, context)
        if isinstance(baml_plan, dict):
            standardized = self.plan_standardizer.standardize(baml_plan)
            if self._is_valid_plan_json(standardized):
                plan = Plan.from_dict(standardized)
                plan.metadata["llm_planning"] = "success_baml"
                plan.metadata["llm_fallback_used"] = False
                return plan

        try:
            for prompt in prompts:
                response = self.llm_client.chat(
                    messages=[
                        {"role": "system", "content": SystemPrompts.SYSTEM_CN},
                        {"role": "user", "content": prompt}
                    ],
                    task_type="planning",
                )

                # Parse response
                plan_json = self._extract_plan_json(response)
                if self._is_valid_plan_json(plan_json):
                    plan = Plan.from_dict(plan_json)
                    plan.metadata["llm_planning"] = "success"
                    plan.metadata["llm_fallback_used"] = False
                    return plan
        except Exception as e:
            print(f"LLM planning failed: {e}, falling back to rules")
            fallback_reason = "llm_exception"
        
        plan = self._plan_with_rules(task_description, context)
        plan.metadata["llm_planning"] = "fallback"
        plan.metadata["llm_fallback_used"] = True
        plan.metadata["llm_fallback_reason"] = fallback_reason
        return plan

    def _plan_with_llm_expert(
        self,
        task_description: str,
        context: dict[str, Any] | None,
    ) -> Plan:
        """Generate plan using LLM with GIS domain knowledge injection."""
        from .prompts import SystemPrompts
        from .gis_domain_prompts import GISDomainPrompts

        # 1. Determine which GIS domain sections are relevant
        domain_knowledge = GISDomainPrompts.get_relevant_sections(task_description)

        # 2. Extract data_schema from context if available (strip from context_json to avoid duplication)
        data_schema = ""
        context_for_json = dict(context) if context else {}
        if context and "data_schema" in context:
            data_schema = context["data_schema"]
            del context_for_json["data_schema"]

        # 3. Build enhanced expert planning prompts
        context_json = json.dumps(context_for_json, ensure_ascii=False, indent=2) if context else "{}"
        prompts = self._build_expert_planning_prompts(
            task_description, domain_knowledge, context_json, data_schema
        )

        fallback_reason = "invalid_json_or_schema"

        try:
            for prompt in prompts:
                response = self.llm_client.chat(
                    messages=[
                        {"role": "system", "content": SystemPrompts.SYSTEM_EXPERT_CN},
                        {"role": "user", "content": prompt}
                    ],
                    task_type="planning_expert",
                )

                plan_json = self._extract_plan_json(response)
                if self._is_valid_plan_json(plan_json):
                    # Extract expert_notes into plan metadata
                    expert_notes = plan_json.pop("expert_notes", {})
                    plan = Plan.from_dict(plan_json)
                    plan.metadata["expert_notes"] = expert_notes
                    plan.metadata["llm_planning"] = "success_expert"
                    plan.metadata["llm_fallback_used"] = False

                    # Self-verify plan quality
                    verification = self._verify_expert_plan(
                        plan, task_description
                    )
                    plan.metadata["plan_verification"] = verification
                    if not verification.get("pass", False):
                        issues = verification.get("issues", [])
                        if issues:
                            plan.metadata["plan_warnings"] = issues

                    return plan
        except Exception as e:
            print(f"[Expert] LLM planning failed: {e}, falling back to standard LLM")
            fallback_reason = "expert_llm_exception"

        # Fallback: try standard LLM planning
        plan = self._plan_with_llm(task_description, context)
        if plan.metadata.get("llm_planning") != "fallback":
            # standard LLM succeeded, tag it as expert fallback
            plan.metadata["llm_planning"] = "expert_fallback_standard"
        return plan

    def _verify_expert_plan(
        self,
        plan: Plan,
        task_description: str,
    ) -> dict[str, Any]:
        """Self-verify plan quality before presenting to user.

        Checks the generated plan against a GIS best-practice checklist
        and returns any issues found. This catches omissions that the
        initial prompt didn't prevent (e.g. missing projection, missing
        map elements).
        """
        if not self.llm_client or not plan.steps:
            return {"pass": True, "issues": [], "note": "skipped_no_llm"}

        # Build verification checklist based on task content
        task_lower = task_description.lower()
        checks: list[str] = []

        # Check 1: Does the plan have map output elements?
        has_map_export = any(
            s.tool == "export_map" for s in plan.steps
        )
        has_execute_code = any(
            s.tool == "execute_code" for s in plan.steps
        )
        expert_notes = plan.metadata.get("expert_notes", {})

        if task_lower in ("", "general"):
            return {"pass": True, "issues": [], "note": "skipped_generic"}

        # 1. Projection check: distance/area tasks need projection
        if any(kw in task_lower for kw in ["面积", "距离", "长度", "缓冲区", "buffer", "area", "distance"]):
            if not any("project" in s.tool or "投影" in s.description for s in plan.steps):
                checks.append("任务涉及距离/面积计算，但计划中没有投影步骤")

        # 2. Map completeness check
        if has_map_export:
            elements = expert_notes.get("cartographic_elements", [])
            if not elements:
                checks.append("专题图缺少地图要素说明（图名/图例/比例尺/指北针）")
            else:
                element_text = " ".join(elements).lower()
                missing = []
                for req in ["图名", "图例", "比例尺", "指北针"]:
                    if req not in element_text and req not in str(plan.steps):
                        missing.append(req)
                if missing:
                    checks.append(f"地图缺少: {', '.join(missing)}")

        # 3. Color scheme check for thematic maps
        if has_map_export:
            color = expert_notes.get("color_scheme", "")
            if not color:
                checks.append("专题图未指定配色方案")

        # 4. Analysis plan check
        if has_execute_code and not expert_notes.get("analysis", "").strip():
            checks.append("execute_code 步骤缺少分析方案说明")

        result = {
            "pass": len(checks) == 0,
            "issues": checks,
            "note": "ok" if len(checks) == 0 else f"发现 {len(checks)} 个问题",
        }
        return result

    def _build_expert_planning_prompts(
        self,
        task_description: str,
        domain_knowledge: str,
        context_json: str,
        data_schema: str = "",
    ) -> list[str]:
        """Build progressive prompts for expert mode planning."""
        from .prompts import SystemPrompts

        base_prompt = SystemPrompts.get_planning_prompt(
            task_description,
            expert_mode=True,
            domain_knowledge=domain_knowledge,
            context_json=context_json,
            data_schema=data_schema,
        )

        strict_retry = (
            f"{base_prompt}\n\n"
            "重要：仅输出一个合法 JSON 对象，不要包含 markdown 代码块，不要解释。\n"
            "JSON 必须包含字段：id, goal, steps, expected_outputs, expert_notes。\n"
            "每个 step 必须包含：id, tool, description, input, depends_on。\n"
            "所有参数值必须具体明确，不要使用占位符。"
        )

        model_hint = self._get_model_hint()
        return [
            self.prompt_adapter.adapt_prompt(base_prompt, task_type="planning", model_hint=model_hint),
            self.prompt_adapter.adapt_prompt(strict_retry, task_type="planning", model_hint=model_hint),
        ]

    def _get_model_hint(self) -> str:
        """Get model hint string for prompt adapter."""
        if self.llm_client and hasattr(self.llm_client, 'config'):
            return f"{self.llm_client.config.provider}/{self.llm_client.config.model}"
        return ""

    def _build_planning_prompts(
        self,
        task_description: str,
        context: dict[str, Any] | None,
    ) -> list[str]:
        """Build progressive prompts: normal first, strict JSON retry second."""
        from .prompts import SystemPrompts

        base_prompt = SystemPrompts.get_planning_prompt(task_description)
        if context:
            base_prompt += f"\n\n当前上下文：\n{json.dumps(context, ensure_ascii=False, indent=2)}"

        strict_retry = (
            f"{base_prompt}\n\n"
            "重要：仅输出一个合法 JSON 对象，不要包含 markdown 代码块，不要解释。\n"
            "JSON 必须至少包含字段：id, goal, steps。\n"
            "steps 必须是数组，每个 step 必须包含：id, tool, description, input, depends_on。"
        )
        model_hint = self._get_model_hint()
        return [
            self.prompt_adapter.adapt_prompt(base_prompt, task_type="planning", model_hint=model_hint),
            self.prompt_adapter.adapt_prompt(strict_retry, task_type="planning", model_hint=model_hint),
        ]

    def _get_model_hint(self) -> str | None:
        config = getattr(self.llm_client, "config", None)
        if config is None:
            return None
        model = getattr(config, "model", None)
        return str(model).strip() if model else None

    def _is_valid_plan_json(self, plan_json: dict | None) -> bool:
        if not isinstance(plan_json, dict):
            return False
        if not isinstance(plan_json.get("goal"), str) or not str(plan_json.get("goal", "")).strip():
            return False
        steps = plan_json.get("steps")
        if not isinstance(steps, list) or not steps:
            return False
        for step in steps:
            if not isinstance(step, dict):
                return False
            if not step.get("id") or not step.get("tool") or not step.get("description"):
                return False
            if "input" in step and not isinstance(step.get("input"), dict):
                return False
            if "depends_on" in step and not isinstance(step.get("depends_on"), list):
                return False
        return True
    
    def _plan_with_rules(
        self,
        task_description: str,
        context: dict[str, Any] | None
    ) -> Plan:
        """Generate plan using rule-based logic."""
        task_lower = task_description.lower()
        is_followup_or_meta = self._is_followup_or_meta_request(task_lower)
        is_complex_multistage = self._is_complex_multistage_task(task_lower)
        reflection_hints = context.get("reflection_hints", []) if context else []
        steps = []
        step_id = 0
        
        # Extract context info
        input_folder = context.get("input_folder", "./workspace/input") if context else "./workspace/input"
        output_folder = context.get("output_folder", DEFAULT_OUTPUT_FOLDER) if context else DEFAULT_OUTPUT_FOLDER
        input_files = context.get("input_files", []) if context else []
        
        # === 首先检查是否是筛选合并任务（优先级最高）===
        if self._is_filter_merge_task(task_lower) and not is_followup_or_meta:
            filter_plan = self._create_filter_merge_plan(task_lower, input_folder, output_folder)
            if filter_plan is not None:
                return filter_plan
        
        # === 然后检查是否匹配 SKILL.md 自定义技能 ===
        # 对于综合多阶段任务，禁止单技能短路，避免只执行局部动作。
        if not is_complex_multistage and not is_followup_or_meta:
            skill_plan = self._try_match_custom_skill(task_description, context)
            if skill_plan:
                return skill_plan
        
        # Try to get first valid input file
        default_input = input_folder
        if input_files:
            # Prefer shapefiles or geodatabase
            for f in input_files:
                if f.get("type") in ["shapefile", "geodatabase", "geojson"]:
                    default_input = f["path"]
                    break
            if default_input == input_folder and input_files:
                # Use first file if no preferred type found
                default_input = input_files[0]["path"]
        
        def add_step(tool: str, desc: str, input_data: dict = None, deps: list = None):
            nonlocal step_id
            step_id += 1
            steps.append(PlanStep(
                id=f"step_{step_id}",
                tool=tool,
                description=desc,
                input=input_data or {},
                depends_on=deps or []
            ))
        
        # Output path helper
        output_folder = context.get("output_folder", DEFAULT_OUTPUT_FOLDER) if context else DEFAULT_OUTPUT_FOLDER

        # 创建文件地理数据库（高优先级专用路径，避免落入通用占位 execute_code）
        create_gdb_markers = [
            "文件地理数据库", "地理数据库", "创建数据库", "create geodatabase", "file geodatabase", "createfilegdb", ".gdb"
        ]
        if (
            any(marker in task_lower for marker in create_gdb_markers)
            and not is_complex_multistage
            and not is_followup_or_meta
        ):
            create_gdb_code = f'''
import arcpy
import os
import time

output_dir = r"{output_folder}"
os.makedirs(output_dir, exist_ok=True)

base_name = "GIS_Database.gdb"
target = os.path.join(output_dir, base_name)

# 如果已存在同名 GDB，则自动创建带时间戳的新库，保证“创建新库”语义
if arcpy.Exists(target):
    ts = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"GIS_Database_{{ts}}.gdb"
    target = os.path.join(output_dir, base_name)

arcpy.management.CreateFileGDB(output_dir, base_name)

set_result({{
    "output": target,
    "gdb_path": target,
    "created": True
}})
print(f"Created FileGDB: {{target}}")
'''

            add_step("execute_code", "使用 ArcPy 创建文件地理数据库", {
                "code": create_gdb_code.strip(),
                "workspace": output_folder,
                "description": "创建文件地理数据库"
            })

            return Plan(
                id=f"plan_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                goal=task_description,
                steps=steps,
                expected_outputs=[f"{output_folder}/GIS_Database.gdb", "execution_log.txt"]
            )
        
        # Determine workflow based on keywords
        # 专题图 / 制图 / 导出
        if any(kw in task_lower for kw in ["专题图", "制图", "地图", "导出", "map", "export"]):
            add_step("scan_layers", "扫描输入数据", {"path": input_folder, "include_subdirs": True})
            add_step("merge_layers", "合并图层数据", {
                "input_layers": [input_folder],
                "output_path": f"{output_folder}/merged.shp"
            }, ["step_1"])
            add_step("project_layers", "统一投影坐标系", {
                "input_path": f"{output_folder}/merged.shp",
                "output_path": f"{output_folder}/projected.shp",
                "target_srs": "EPSG:4326"
            }, ["step_2"])
            add_step("quality_check", "数据质量检查", {"input_path": f"{output_folder}/projected.shp"}, ["step_3"])
            add_step("export_map", "导出地图", {
                "output_path": f"{output_folder}/map.jpg",
                "format": "JPG"
            }, ["step_4"])
        
        # 数据整合 / 入库（普通合并，不涉及名称筛选）
        if any(kw in task_lower for kw in ["整合", "入库", "integrate"]) or (
            any(kw in task_lower for kw in ["合并", "merge"]) and not self._is_filter_merge_task(task_lower)
        ):
            add_step("scan_layers", "扫描输入数据", {"path": input_folder, "include_subdirs": True})
            add_step("quality_check", "数据质量预检", {"input_path": default_input}, ["step_1"])
            add_step("merge_layers", "合并图层数据", {
                "input_layers": [input_folder],
                "output_path": f"{output_folder}/merged.shp"
            }, ["step_2"])
            add_step("project_layers", "统一投影坐标系", {
                "input_path": f"{output_folder}/merged.shp",
                "output_path": f"{output_folder}/projected.shp"
            }, ["step_3"])
            add_step("quality_check", "数据质量终检", {"input_path": f"{output_folder}/projected.shp"}, ["step_4"])
        
        # 质量检查
        elif any(kw in task_lower for kw in ["检查", "质量", "验证", "check", "quality"]):
            add_step("scan_layers", "扫描输入数据", {"path": input_folder, "include_subdirs": True})
            add_step("quality_check", "几何有效性检查", 
                    {"input_path": default_input, "check_types": ["geometry"]}, ["step_1"])
            add_step("quality_check", "拓扑检查", 
                    {"input_path": default_input, "check_types": ["topology"]}, ["step_2"])
            add_step("quality_check", "坐标系检查", 
                    {"input_path": default_input, "check_types": ["crs"]}, ["step_3"])
        
        # 投影变换
        elif any(kw in task_lower for kw in ["投影", "坐标", "变换", "project", "transform"]):
            add_step("scan_layers", "扫描输入数据", {"path": input_folder, "include_subdirs": True})
            add_step("project_layers", "投影变换", {
                "input_path": default_input,
                "output_path": f"{output_folder}/projected.shp",
                "target_srs": "EPSG:4326"
            }, ["step_1"])
            add_step("quality_check", "验证变换结果", {"input_path": f"{output_folder}/projected.shp"}, ["step_2"])
        
        # 扫描 / 查看数据
        elif any(kw in task_lower for kw in ["扫描", "看看", "查看", "有什么", "图层", "数据", "scan", "list", "show"]):
            add_step("scan_layers", "扫描输入数据", {"path": input_folder, "include_subdirs": True})
        
        # 缓冲区分析
        elif any(kw in task_lower for kw in ["缓冲", "buffer"]):
            add_step("scan_layers", "扫描输入数据", {"path": input_folder, "include_subdirs": True})
            # 使用 execute_code 执行缓冲区分析
            buffer_code = f'''
import arcpy
# 获取输入数据
in_features = r"{default_input}"
out_features = r"{output_folder}/buffer_result.shp"
buffer_distance = "500 Meters"  # 默认缓冲距离

# 执行缓冲区分析
arcpy.Buffer_analysis(in_features, out_features, buffer_distance)

# 统计结果
count = arcpy.GetCount_management(out_features)[0]
set_result({{"output": out_features, "feature_count": count}})
'''
            add_step("execute_code", "执行缓冲区分析", {
                "code": buffer_code.strip(),
                "workspace": output_folder,
                "description": "缓冲区分析"
            }, ["step_1"])
        
        # 裁剪分析
        elif (
            any(kw in task_lower for kw in ["裁剪", "clip", "切割"])
            and not self._contains_negative_tool_intent(task_lower, ["裁剪", "clip", "切割"])
            and not self._should_avoid_tool_from_reflections(task_lower, reflection_hints, "clip")
        ):
            add_step("scan_layers", "扫描输入数据", {"path": input_folder, "include_subdirs": True})
            clip_code = f'''
import arcpy
# 输入图层和裁剪范围需要用户指定
in_features = r"{default_input}"
clip_features = r"{default_input}"  # 裁剪范围
out_features = r"{output_folder}/clip_result.shp"

# 执行裁剪
arcpy.Clip_analysis(in_features, clip_features, out_features)

# 统计结果
count = arcpy.GetCount_management(out_features)[0]
set_result({{"output": out_features, "feature_count": count}})
'''
            add_step("execute_code", "执行裁剪分析", {
                "code": clip_code.strip(),
                "workspace": output_folder,
                "description": "裁剪分析"
            }, ["step_1"])
        
        # 叠加分析（相交、联合等）
        elif any(kw in task_lower for kw in ["叠加", "相交", "联合", "intersect", "union", "overlay"]):
            add_step("scan_layers", "扫描输入数据", {"path": input_folder, "include_subdirs": True})
            overlay_code = f'''
import arcpy
# 输入图层
in_features = r"{default_input}"
overlay_features = r"{default_input}"
out_features = r"{output_folder}/overlay_result.shp"

# 执行相交分析
arcpy.Intersect_analysis([in_features, overlay_features], out_features)

# 统计结果
count = arcpy.GetCount_management(out_features)[0]
set_result({{"output": out_features, "feature_count": count}})
'''
            add_step("execute_code", "执行叠加分析", {
                "code": overlay_code.strip(),
                "workspace": output_folder,
                "description": "叠加分析"
            }, ["step_1"])
        
        # 默认：如果有 LLM，尝试用 execute_code 生成代码
        # 否则回退到扫描+检查
        else:
            add_step("scan_layers", "扫描输入数据", {"path": input_folder, "include_subdirs": True})
            # 对于无法识别的任务，添加一个占位的 execute_code 步骤
            # 实际代码由 LLM 在执行时生成
            add_step("execute_code", f"执行自定义分析: {task_description[:50]}", {
                "code": f'''
import arcpy
# TODO: 根据任务需求编写代码
# 任务: {task_description[:100]}
print("请使用 LLM 模式生成具体代码")
set_result({{"status": "需要LLM生成代码", "task": "{task_description[:50]}"}})
''',
                "workspace": output_folder,
                "description": task_description[:100]
            }, ["step_1"])
        
        return Plan(
            id=f"plan_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            goal=task_description,
            steps=steps,
            expected_outputs=["output.shp", "execution_log.txt"]
        )
    
    def _try_match_custom_skill(
        self,
        task_description: str,
        context: dict[str, Any] | None
    ) -> Plan | None:
        """尝试匹配 SKILL.md 自定义技能。
        
        Returns:
            匹配的执行计划，或 None
        """
        from ..skills import get_skill_loader

        task_lower = task_description.lower()
        if self._is_followup_or_meta_request(task_lower):
            return None

        # 长文本/多目标请求中，避免被单技能误抢占。
        if self._is_complex_multistage_task(task_lower):
            return None
        
        loader = get_skill_loader()
        skill_match = loader.find_best_match_with_score(
            task_description,
            min_score=5,
            only_executable=True,
        )
        if not skill_match:
            return None
        if not skill_match.name_hit and not skill_match.trigger_hits:
            return None
        if len(task_lower) >= 60 and (not skill_match.name_hit) and len(skill_match.trigger_hits) < 2:
            return None
        skill = skill_match.skill
        
        # 找到匹配的技能，生成执行计划
        output_folder = context.get("output_folder", DEFAULT_OUTPUT_FOLDER) if context else DEFAULT_OUTPUT_FOLDER
        input_folder = context.get("input_folder", "./workspace/input") if context else "./workspace/input"
        
        # 使用技能的代码模板
        if skill.code_template:
            # 准备默认参数
            default_params = {
                "input_layer": input_folder,
                "output_path": f"{output_folder}/{skill.name}_result.shp",
                "buffer_distance": "500 Meters",
                "dissolve_option": "NONE",
                "clip_layer": input_folder,
                "input_layers": input_folder,
                "join_attributes": "ALL",
                "field_name": "NEW_FIELD",
                "expression": "'value'",
                "expression_type": "PYTHON3",
                "code_block": "",
                "dissolve_field": "",
                "statistics_fields": "",
                "document_path": (context.get("document_path") if context else "") or "",
                "document_summary": (context.get("document_summary") if context else "") or "",
            }
            
            # 合并可选参数的默认值
            default_params.update(skill.optional_inputs)
            
            # 生成代码
            code = loader.generate_code(skill, default_params)
            
            # 创建执行计划
            steps = [
                PlanStep(
                    id="step_1",
                    tool="scan_layers",
                    description="扫描输入数据",
                    input={"path": input_folder, "include_subdirs": True},
                    depends_on=[]
                ),
                PlanStep(
                    id="step_2",
                    tool="execute_code",
                    description=f"执行 {skill.name}: {skill.description}",
                    input={
                        "code": code,
                        "workspace": output_folder,
                        "description": skill.description
                    },
                    depends_on=["step_1"]
                )
            ]
            
            return Plan(
                id=f"plan_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                goal=f"使用 {skill.name} 技能: {skill.description}",
                steps=steps,
                expected_outputs=[f"{output_folder}/{skill.name}_result.shp"],
                metadata={
                    "skill": skill.name,
                    "skill_file": str(skill.source_file) if skill.source_file else None,
                    "skill_match_score": skill_match.score,
                    "skill_match_trigger_hits": skill_match.trigger_hits,
                    "skill_match_tag_hits": skill_match.tag_hits,
                    "skill_match_name_hit": skill_match.name_hit,
                }
            )
        
        return None

    def _is_followup_or_meta_request(self, task_lower: str) -> bool:
        """Detect follow-up/meta instructions that should not be skill-short-circuited."""
        markers = [
            "根据之前", "基于之前", "按之前", "继续", "接着", "后续", "在此基础", "前面的",
            "规划任务", "继续规划", "继续操作", "继续处理", "继续执行",
        ]
        return any(marker in task_lower for marker in markers)

    def _should_avoid_tool_from_reflections(
        self,
        task_lower: str,
        reflection_hints: list[dict[str, Any]],
        tool_keyword: str,
    ) -> bool:
        """Use reflection hints to avoid known accidental tool selection.

        Explicit user requests always take precedence.
        """
        if not reflection_hints:
            return False

        if re.search(r"(请|执行|进行|做).{0,4}(裁剪|clip|切割)", task_lower):
            return False

        for hint in reflection_hints:
            related_tools = [str(t).lower() for t in (hint.get("related_tools") or [])]
            issue_text = str(hint.get("issue") or "").lower()
            action_hint = str(hint.get("action_hint") or "").lower()
            confidence = float(hint.get("confidence") or 0.0)

            mentions_tool = tool_keyword in related_tools or tool_keyword in issue_text or tool_keyword in action_hint
            indicates_mistrigger = any(k in issue_text for k in ["误触发", "误判", "hijack", "抢占"])
            if mentions_tool and indicates_mistrigger and confidence >= 0.6:
                return True

        return False

    def _contains_negative_tool_intent(self, task_lower: str, tokens: list[str]) -> bool:
        """Detect requests like '不要裁剪' to avoid opposite execution."""
        negatives = ["不要", "不需要", "不用", "别", "禁止"]
        for token in tokens:
            if token not in task_lower:
                continue
            for n in negatives:
                if f"{n}{token}" in task_lower or f"{token}{n}" in task_lower:
                    return True
        return False
    
    def _is_filter_merge_task(self, task_lower: str) -> bool:
        """判断是否是需要按名称筛选的合并任务。
        
        识别模式：
        - "将所有 boua 合并" / "把 boul 合并"
        - "合并所有名称包含 xxx 的图层"
        - "分别合并 xxx 和 yyy"
        """
        # 必须包含合并关键词
        if not any(kw in task_lower for kw in ["合并", "合成", "merge"]):
            return False

        # 复杂多阶段需求（如完整竞赛题）不应被“筛选合并”短路
        if self._is_complex_multistage_task(task_lower):
            return False
        
        # 检查是否有明确的筛选模式
        filter_patterns = [
            r"所有\s*(\w+)\s*(和|与|及|,|，|\s)+\s*(\w+)",  # 所有 boua 和 boul
            r"将.*?(\w+).*?合并",                            # 将 boua 合并
            r"把.*?(\w+).*?合并",                            # 把 boua 合并
            r"分别.*?(\w+).*?(\w+)",                        # 分别合并 xxx 和 yyy
            r"(\w+)\s*(和|与|及)\s*(\w+).*?分别",           # boua 和 boul 分别
            r"命名为.*?(行政区|边界|道路)",                  # 命名为行政区
        ]
        
        import re
        for pattern in filter_patterns:
            if re.search(pattern, task_lower):
                return True
        
        # 检查常见的图层名称模式
        layer_name_hints = ["boua", "boul", "resa", "resl", "hyda", "hydl", "vega", "vegl"]
        if any(hint in task_lower for hint in layer_name_hints):
            return True
        
        return False

    def _is_complex_multistage_task(self, task_lower: str) -> bool:
        """判断是否是包含多个制图阶段的综合任务。"""
        # 题目式长文特征
        long_text = len(task_lower) >= 400
        has_structured_sections = any(token in task_lower for token in ["一、", "二、", "三、", "四、", "分析要求", "数据说明", "分)"])

        # 多阶段制图关键词（命中越多越可能是综合任务）
        stage_keywords = [
            "数据库", "投影", "配准", "拓扑", "修复", "符号", "标注",
            "箭头", "图例", "比例尺", "导出", "制图", "布局", "地图"
        ]
        stage_hits = sum(1 for kw in stage_keywords if kw in task_lower)

        return (long_text and has_structured_sections) or stage_hits >= 5
    
    def _extract_filter_patterns(self, task_lower: str) -> list[tuple[str, str]]:
        """从任务描述中提取筛选模式和输出名称。
        
        Returns:
            [(pattern, output_name), ...] 如 [("boua", "行政区面"), ("boul", "行政区线")]
        """
        results = []
        
        # 常见的图层名称到中文名称的映射
        layer_name_mapping = {
            "boua": "行政区面",
            "boul": "行政区线",
            "resa": "居民地面",
            "resl": "居民地线",
            "hyda": "水系面",
            "hydl": "水系线",
            "vega": "植被面",
            "vegl": "植被线",
            "lrda": "地貌面",
            "lrdl": "地貌线",
            "tera": "地形面",
            "terl": "地形线",
        }
        
        # 仅在“合并/整合”语句片段中提取图层关键词，避免长任务中误触发
        merge_segments = re.findall(r"[^。；\n]*(?:合并|整合|merge)[^。；\n]*", task_lower)
        if not merge_segments:
            merge_segments = [task_lower]

        for segment in merge_segments:
            for key, default_name in layer_name_mapping.items():
                if key not in segment:
                    continue
                # 检查用户是否在任务中提到了中文名称
                if default_name in segment or default_name in task_lower:
                    results.append((key, default_name))
                elif "行政区面" in task_lower and key == "boua":
                    results.append((key, "行政区面"))
                elif "行政区线" in task_lower and key == "boul":
                    results.append((key, "行政区线"))
                else:
                    results.append((key, default_name))

        # 去重，保持顺序
        deduped: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in results:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        
        return deduped
    
    def _create_filter_merge_plan(
        self,
        task_lower: str,
        input_folder: str,
        output_folder: str
    ) -> Plan | None:
        """创建筛选合并计划。
        
        生成使用 execute_code 的计划，按名称模式筛选图层并合并。
        """
        filters = self._extract_filter_patterns(task_lower)
        
        if not filters:
            # 如果没有提取到模式，返回 None 让其他规则处理
            return None
        
        steps = [
            PlanStep(
                id="step_1",
                tool="scan_layers",
                description="扫描输入数据",
                input={"path": input_folder, "include_subdirs": True},
                depends_on=[]
            )
        ]
        
        for i, (pattern, output_name) in enumerate(filters):
            filter_code = f'''import arcpy
import os

# 筛选并合并包含 '{pattern}' 的图层
filter_pattern = "{pattern.lower()}"
output_path = r"{output_folder}/{output_name}.shp"
input_folder = r"{input_folder}"

layers_to_merge = []
for root, dirs, files in os.walk(input_folder):
    for f in files:
        if f.lower().endswith('.shp') and filter_pattern in f.lower():
            layers_to_merge.append(os.path.join(root, f))

if not layers_to_merge:
    raise ValueError(f"未找到匹配 '{{filter_pattern}}' 的图层")

print(f"找到 {{len(layers_to_merge)}} 个图层: {{layers_to_merge}}")
arcpy.Merge_management(layers_to_merge, output_path)

count = int(arcpy.GetCount_management(output_path)[0])
set_result({{"output": output_path, "feature_count": count, "merged_layers": len(layers_to_merge)}})
'''
            steps.append(PlanStep(
                id=f"step_{i + 2}",
                tool="execute_code",
                description=f"筛选并合并包含 '{pattern}' 的图层为 '{output_name}'",
                input={
                    "code": filter_code.strip(),
                    "workspace": output_folder,
                    "description": f"合并 {pattern} 图层为 {output_name}"
                },
                depends_on=["step_1"]
            ))
        
        output_names = [f[1] for f in filters]
        return Plan(
            id=f"plan_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            goal=f"筛选合并图层: {', '.join(output_names)}",
            steps=steps,
            expected_outputs=[f"{output_folder}/{name}.shp" for name in output_names],
            metadata={"filter_merge": True, "patterns": [f[0] for f in filters]}
        )
    
    def _extract_plan_json(self, response: str) -> dict | None:
        """Extract plan JSON from LLM response."""
        # Try to find JSON in code block
        pattern = r"```(?:plan|json)?\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            repaired = self._repair_json_text(match)
            try:
                parsed = json.loads(repaired)
                standardized = self.plan_standardizer.standardize(parsed)
                if standardized:
                    return standardized
            except json.JSONDecodeError:
                # Optional relaxed parser
                try:
                    import json5  # type: ignore

                    parsed = json5.loads(repaired)
                    if isinstance(parsed, dict):
                        standardized = self.plan_standardizer.standardize(parsed)
                        if standardized:
                            return standardized
                except Exception:
                    continue
        
        # Try to parse entire response as JSON
        repaired_response = self._repair_json_text(response)
        try:
            parsed = json.loads(repaired_response)
            standardized = self.plan_standardizer.standardize(parsed)
            if standardized:
                return standardized
        except json.JSONDecodeError:
            pass

        try:
            import json5  # type: ignore

            parsed = json5.loads(repaired_response)
            if isinstance(parsed, dict):
                standardized = self.plan_standardizer.standardize(parsed)
                if standardized:
                    return standardized
        except Exception:
            pass
        
        return None

    def _repair_json_text(self, text: str) -> str:
        """Best-effort JSON cleanup for verbose/loose model outputs."""
        value = (text or "").strip()
        if not value:
            return value

        # Remove BOM / smart quotes / markdown remnants.
        value = value.replace("\ufeff", "")
        value = value.replace("“", '"').replace("”", '"').replace("’", "'")

        # Trim leading prose before first JSON bracket.
        first_obj = value.find("{")
        first_arr = value.find("[")
        starts = [x for x in [first_obj, first_arr] if x >= 0]
        if starts:
            value = value[min(starts):]

        # Trim trailing prose after final JSON bracket.
        last_obj = value.rfind("}")
        last_arr = value.rfind("]")
        end = max(last_obj, last_arr)
        if end >= 0:
            value = value[: end + 1]

        # Remove trailing commas before closing braces/brackets.
        value = re.sub(r",\s*([}\]])", r"\1", value)
        return value
    
    def refine_plan(
        self,
        plan: Plan,
        feedback: str
    ) -> Plan:
        """Refine a plan based on feedback."""
        if self.llm_client:
            # Use LLM to refine
            prompt = f"""原计划：
{json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)}

用户反馈：{feedback}

请根据反馈修改计划，输出新的计划 JSON。
"""
            try:
                response = self.llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    task_type="planning_refine",
                )
                plan_json = self._extract_plan_json(response)
                if plan_json:
                    return Plan.from_dict(plan_json)
            except Exception:
                pass
        
        # Return original plan if refinement fails
        return plan
    
    def suggest_recovery(
        self,
        plan: Plan,
        failed_step: PlanStep
    ) -> Plan | None:
        """Suggest a recovery plan for a failed step."""
        remaining_steps = [
            PlanStep(
                id=s.id,
                tool=s.tool,
                description=s.description,
                input=s.input,
                depends_on=[d for d in s.depends_on if d != failed_step.id]
            )
            for s in plan.steps
            if s.status == StepStatus.PENDING
        ]

        if not remaining_steps:
            return None

        failed_error = (failed_step.error or "").lower()

        # 关键步骤失败时，禁止自动恢复跳过或替换，交由用户修正输入后重试
        if self._is_critical_step(failed_step):
            return None

        # Prefer BAML recovery suggestion for non-critical failures.
        baml_recovery = self.baml_bridge.suggest_recovery(
            plan_payload=plan.to_dict(),
            failed_step_payload={
                "id": failed_step.id,
                "tool": failed_step.tool,
                "description": failed_step.description,
                "input": failed_step.input,
                "error": failed_step.error,
            },
            remaining_steps_payload=[s.to_dict() for s in remaining_steps],
        )
        if isinstance(baml_recovery, dict):
            standardized = self.plan_standardizer.standardize(baml_recovery)
            if self._is_valid_plan_json(standardized):
                recovered = Plan.from_dict(standardized)
                recovered.metadata.setdefault("recovery_strategy", "baml_recovery")
                recovered.metadata.setdefault("failed_step", failed_step.id)
                return recovered

        # 策略1: 参数类失败 -> 先做质量检查再继续
        if any(k in failed_error for k in ["validation", "参数", "field required", "missing"]):
            qc_step = PlanStep(
                id="recovery_qc_1",
                tool="quality_check",
                description=f"恢复检查：验证失败步骤 {failed_step.id} 的输入数据",
                input={"input_path": "./workspace/input"},
                depends_on=[]
            )
            for s in remaining_steps:
                s.depends_on = ["recovery_qc_1"] + [d for d in s.depends_on if d != "recovery_qc_1"]
            return Plan(
                id=f"recovery_{plan.id}",
                goal=f"Recovery: validate inputs after {failed_step.id}",
                steps=[qc_step, *remaining_steps],
                metadata={"recovery_strategy": "validate_then_continue", "failed_step": failed_step.id}
            )

        # 策略2: ArcPy/执行环境失败 -> 回退 execute_code
        if any(k in failed_error for k in ["arcpy", "environment", "execution_error", "tool_not_found"]):
            fallback_step = PlanStep(
                id="recovery_exec_1",
                tool="execute_code",
                description=f"恢复执行：回退到 execute_code 处理 {failed_step.id}",
                input={
                    "code": (
                        "import arcpy\n"
                        "print('Recovery fallback for failed step')\n"
                        "set_result({'recovery': 'execute_code_fallback', 'failed_step': '" + failed_step.id + "'})"
                    ),
                    "workspace": DEFAULT_OUTPUT_FOLDER,
                    "description": f"Recovery fallback for {failed_step.id}"
                },
                depends_on=[]
            )
            return Plan(
                id=f"recovery_{plan.id}",
                goal=f"Recovery: execute_code fallback for {failed_step.id}",
                steps=[fallback_step],
                metadata={"recovery_strategy": "execute_code_fallback", "failed_step": failed_step.id}
            )

        # 默认策略：跳过失败步骤继续（仅限非关键步骤）
        if self._is_critical_step(failed_step):
            return None

        return Plan(
            id=f"recovery_{plan.id}",
            goal=f"Recovery: skip {failed_step.id}",
            steps=remaining_steps,
            metadata={"recovery_strategy": "skip_failed_step", "failed_step": failed_step.id}
        )

    def _is_critical_step(self, step: PlanStep) -> bool:
        """Whether a failed step is critical and must not be auto-skipped."""
        critical_tools = {"scan_layers", "merge_layers", "project_layers", "execute_code", "export_map"}
        return step.id == "step_1" or step.tool in critical_tools

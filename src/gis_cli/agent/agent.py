"""GIS Agent - Main agent class.

The GISAgent is the main entry point for interacting with the GIS CLI
through natural language. It combines:
- Memory for conversation context
- Planner for task decomposition
- Executor for tool execution
- LLM integration for understanding and reasoning
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Generator

from ..core import ToolRegistry, ExecutionContext
from ..document_parser import DocumentTaskParser
from ..skills import SkillRegistry, get_skill_loader
from .. import tools as _  # Auto-register tools on import
from .memory import Memory, MemoryStore, ConversationTurn, Role
from .state_manager import LangGraphStateManager
from .context_hub import ContextHub, ContextSnapshot
from .planner import AgentPlanner, Plan, PlanStep, StepStatus
from .executor import ExecutionResult, ExecutionMode, ExecutionTrace, StepTrace
from .execution_adapter import ExecutionAdapter
from .prompts import SystemPrompts, PromptBuilder, build_agent_prompt
from .model_adaptation import BAMLBridge, PromptAdapter


@dataclass
class AgentConfig:
    """Configuration for GIS Agent."""
    
    # Session settings
    session_id: str = field(default_factory=lambda: f"session_{uuid.uuid4().hex[:8]}")
    language: str = "cn"
    
    # Memory settings
    memory_path: Path | None = None
    max_conversation_turns: int = 100

    # State management (LangGraph-compatible)
    state_path: Path | None = None
    enable_state_manager: bool = True
    
    # Execution settings
    default_mode: ExecutionMode = ExecutionMode.DRY_RUN
    auto_execute: bool = False
    max_recovery_attempts: int = 3
    
    # LLM settings
    llm_enabled: bool = True
    llm_model: str = "gpt-4"
    llm_api_key: str | None = None
    llm_api_base: str | None = None
    
    # Workspace
    workspace_path: Path | None = None
    output_dir: Path | None = None

    # Expert mode (default: enabled)
    expert_mode: bool = True


@dataclass 
class AgentResponse:
    """Response from the agent."""
    
    content: str
    action_taken: str | None = None
    plan: Plan | None = None
    execution_result: ExecutionResult | None = None
    tool_calls: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "action_taken": self.action_taken,
            "plan": self.plan.to_dict() if self.plan else None,
            "execution_result": self.execution_result.to_dict() if self.execution_result else None,
            "tool_calls": self.tool_calls,
            "suggestions": self.suggestions
        }


class GISAgent:
    """Main GIS Agent class.
    
    The agent can:
    - Understand natural language GIS requests
    - Plan multi-step workflows
    - Execute plans using tools
    - Maintain conversation context
    - Learn from feedback
    
    Example usage:
        ```python
        agent = GISAgent()
        
        # Simple query
        response = agent.chat("扫描当前目录的 GIS 数据")
        print(response.content)
        
        # Multi-turn conversation
        agent.chat("整合这些数据")
        agent.chat("投影到 CGCS2000")
        agent.chat("导出专题图")
        
        # Execute with streaming
        for chunk in agent.stream("制作广西专题图"):
            print(chunk, end="", flush=True)
        ```
    """
    
    def __init__(
        self,
        config: AgentConfig | None = None,
        llm_client: Any = None
    ):
        self.config = config or AgentConfig()
        
        # Auto-load LLM client if not provided
        if llm_client is None and self.config.llm_enabled:
            llm_client = self._auto_load_llm_client()
        
        self.llm_client = llm_client
        llm_cfg = getattr(self.llm_client, "config", None)
        enable_prompt_optimizer = bool(getattr(llm_cfg, "enable_prompt_optimizer", True))
        enable_baml_standardizer = bool(getattr(llm_cfg, "enable_baml_standardizer", True))
        baml_function_map = getattr(llm_cfg, "baml_functions", {}) if llm_cfg is not None else {}
        baml_builtin_fallback = bool(getattr(llm_cfg, "enable_baml_builtin_fallback", True)) if llm_cfg is not None else False
        if not isinstance(baml_function_map, dict):
            baml_function_map = {}
        self.prompt_adapter = PromptAdapter(enable_prompt_optimizer=enable_prompt_optimizer)
        self.baml_bridge = BAMLBridge(
            enabled=enable_baml_standardizer,
            function_map=baml_function_map,
            allow_builtin_fallback=baml_builtin_fallback,
        )
        
        # Load tools and skills first (needed for planner init)
        self.tool_registry = ToolRegistry.instance()
        self.skill_registry = SkillRegistry.instance()
        
        # Load SKILL.md files from workspace/skills/
        workspace_root = self.config.workspace_path or Path.cwd()
        self.skill_loader = get_skill_loader(workspace_root / "workspace" / "skills")
        self._load_custom_skills()
        
        # Initialize components
        self._init_memory()
        self._init_state_manager()
        self._init_context()
        self._init_planner()
        self._init_executor()
        
        # Current state
        self.current_plan: Plan | None = None
        self.last_result: ExecutionResult | None = None
        self._pending_task: str | None = None  # For suggestion-triggered tasks
        self._pending_confirmation_mode: ExecutionMode | None = None
        self._restore_runtime_state()
    
    def _load_custom_skills(self) -> None:
        """Load SKILL.md files from workspace/skills/ directory."""
        count = self.skill_loader.load()
        if count > 0:
            # Log loaded skills (silent by default)
            pass
    
    def _auto_load_llm_client(self) -> Any:
        """Auto-load LLM client from config file or environment."""
        from .llm import create_llm_client, LLMConfig
        
        # Try to load from config file first
        config_path = Path("config/llm_config.json")
        if config_path.exists():
            try:
                llm_config = LLMConfig.from_file(str(config_path))
                if llm_config.api_key and llm_config.api_key != "YOUR_API_KEY_HERE":
                    return create_llm_client(llm_config)
            except Exception:
                pass
        
        # Try environment variables
        try:
            llm_config = LLMConfig.from_env()
            if llm_config.api_key:
                return create_llm_client(llm_config)
        except Exception:
            pass
        
        return None
    
    def _init_memory(self) -> None:
        """Initialize memory system."""
        store = None
        if self.config.memory_path:
            store = MemoryStore(self.config.memory_path)
        
        self.memory = Memory(
            session_id=self.config.session_id,
            store=store,
            max_turns=self.config.max_conversation_turns
        )
        
        # Load existing memory if available
        self.memory.load()

    def _init_state_manager(self) -> None:
        """Initialize workflow state manager."""
        self.thread_id = f"thread_{self.config.session_id}"
        self.state_manager = None
        if not self.config.enable_state_manager:
            return

        self.state_manager = LangGraphStateManager(self.config.state_path)
        self.state_manager.upsert_state(
            self.thread_id,
            session_id=self.config.session_id,
            status="initialized",
        )
    
    def _init_context(self) -> None:
        """Initialize execution context."""
        workspace = self.config.workspace_path or Path.cwd()
        self.context = ExecutionContext.build(
            workspace=workspace,
            dry_run=(self.config.default_mode == ExecutionMode.DRY_RUN)
        )
        self.context_hub = ContextHub(workspace=workspace, memory=self.memory)
    
    def _init_planner(self) -> None:
        """Initialize planner."""
        available_tools = [t.name for t in self.tool_registry.list_tools()]
        available_skills = [s.name for s in self.skill_registry.list_skills()]
        
        # Add custom skills from SKILL.md files
        custom_skills = [s.name for s in self.skill_loader.list_skills()]
        available_skills.extend(custom_skills)
        
        self.planner = AgentPlanner(
            llm_client=self.llm_client,
            available_tools=available_tools,
            available_skills=available_skills
        )

    def reload_custom_skills(self) -> int:
        """Reload SKILL.md custom skills and refresh planner skill list."""
        count = self.skill_loader.reload()
        self._init_planner()
        return count
    
    def _init_executor(self) -> None:
        """Initialize executor."""
        self.executor = ExecutionAdapter(
            context=self.context,
            memory=self.memory,
            llm_client=self.llm_client,
            on_step_start=self._on_step_start,
            on_step_complete=self._on_step_complete,
            on_progress=self._on_progress
        )
    
    # === Main Interface ===
    
    def chat(self, message: str) -> AgentResponse:
        """Process a chat message and return a response.
        
        This is the main entry point for interacting with the agent.
        
        Args:
            message: User's natural language message
            
        Returns:
            AgentResponse with the agent's response
        """
        # Add to memory
        self.memory.add_user_message(message)
        self._update_context_from_message(message)
        self._extract_structured_memory_from_message(message)
        
        # Analyze intent
        intent = self._analyze_intent(message)
        
        # Check for pending task from suggestion selection
        task_message = message
        if self._pending_task:
            task_message = self._pending_task
            self._pending_task = None
        
        # Generate response based on intent
        if intent == "execute_task":
            response = self._handle_task_execution(task_message)
        elif intent == "query_status":
            response = self._handle_status_query(message)
        elif intent == "tool_help":
            response = self._handle_tool_help(message)
        elif intent == "confirm_action":
            response = self._handle_confirmation(message)
        elif intent == "preview":
            response = self._handle_preview(message)
        elif intent == "cancel":
            response = self._handle_cancel(message)
        elif intent == "skip_step":
            response = self._handle_skip_step(message)
        else:
            response = self._handle_general_query(message)
        
        # Add response to memory
        self.memory.add_assistant_message(response.content)
        self._persist_runtime_state()
        
        # Save memory
        self.memory.save()
        
        return response
    
    def stream(self, message: str) -> Generator[str, None, AgentResponse]:
        """Process a message with streaming response.
        
        Yields response chunks as they are generated.
        Returns the final AgentResponse when complete.
        """
        # Add to memory
        self.memory.add_user_message(message)
        
        # For now, just yield the full response
        # TODO: Implement true streaming with LLM
        response = self._handle_task_execution(message)
        
        # Yield content in chunks
        for chunk in self._chunk_response(response.content):
            yield chunk
        
        # Add to memory
        self.memory.add_assistant_message(response.content)
        self.memory.save()
        
        return response
    
    def execute_plan(
        self,
        plan: Plan | None = None,
        mode: ExecutionMode | None = None
    ) -> ExecutionResult:
        """Execute a plan.
        
        Args:
            plan: Plan to execute (uses current_plan if None)
            mode: Execution mode (uses config default if None)
            
        Returns:
            ExecutionResult with execution details
        """
        plan = plan or self.current_plan
        if plan is None:
            raise ValueError("No plan to execute")
        
        mode = mode or self.config.default_mode
        result = self.executor.execute(plan, mode)
        self.last_result = result
        
        return result

    def _execute_with_recovery(
        self,
        plan: Plan,
        mode: ExecutionMode
    ) -> tuple[ExecutionResult, list[str]]:
        """Execute plan with automatic recovery attempts."""
        notes: list[str] = []
        current_plan = plan
        attempts = 0
        seen_failures: dict[str, int] = {}

        result = self.execute_plan(current_plan, mode=mode)

        if result.success and self._is_false_success_after_recovery(result):
            notes.append("执行结果判定为假成功：未检测到可交付产物。")
            result.success = False
            result.error = "执行完成但未检测到可交付输出。"

        while (not result.success) and attempts < self.config.max_recovery_attempts:
            failed_steps = current_plan.get_failed_steps()
            if not failed_steps:
                break

            failed_step = failed_steps[0]
            fingerprint = self._build_failure_fingerprint(failed_step, result)
            seen_failures[fingerprint] = seen_failures.get(fingerprint, 0) + 1
            if seen_failures[fingerprint] >= 2:
                notes.append(
                    f"自动恢复终止：检测到重复失败（步骤 {failed_step.id}）且错误未变化，已停止循环重试。"
                )
                break

            if self._attempt_autonomous_recovery(current_plan, failed_step, result, notes):
                attempts += 1
                notes.append(
                    f"自动恢复第 {attempts} 次：策略=autonomous，失败步骤={failed_step.id}"
                )
                self.current_plan = current_plan
                result = self.execute_plan(current_plan, mode=mode)
                if result.success and self._is_false_success_after_recovery(result):
                    notes.append("自动恢复结果判定为假成功：关键产物缺失。")
                    result.success = False
                    result.error = "恢复后关键步骤未产出有效结果，已阻止假成功。"
                continue

            recovery_plan = self.planner.suggest_recovery(current_plan, failed_step)
            if recovery_plan is None or not recovery_plan.steps:
                notes.append(
                    f"自动恢复终止：失败步骤 {failed_step.id} 为关键步骤，禁止 skip。"
                )
                break

            attempts += 1
            strategy = recovery_plan.metadata.get("recovery_strategy", "unknown")
            notes.append(
                f"自动恢复第 {attempts} 次：策略={strategy}，失败步骤={failed_step.id}"
            )

            current_plan = recovery_plan
            self.current_plan = current_plan
            result = self.execute_plan(current_plan, mode=mode)

            if result.success:
                if self._is_false_success_after_recovery(result):
                    notes.append("自动恢复结果判定为假成功：关键产物缺失。")
                    result.success = False
                    result.error = "恢复后关键步骤未产出有效结果，已阻止假成功。"
                    break
                notes.append("自动恢复成功")
                break

        return result, notes

    def _build_failure_fingerprint(self, failed_step: PlanStep, result: ExecutionResult) -> str:
        error_text = ((result.error or "") + " " + (failed_step.error or "")).lower()
        compact = " ".join(error_text.split())
        return f"{failed_step.id}|{failed_step.tool}|{compact[:220]}"

    def _attempt_autonomous_recovery(
        self,
        plan: Plan,
        failed_step: PlanStep,
        result: ExecutionResult,
        notes: list[str],
    ) -> bool:
        """Apply deterministic self-healing policies before planner-based recovery."""
        error_text = " ".join([
            (result.error or ""),
            (failed_step.error or ""),
        ]).lower()

        if self._is_output_exists_error(error_text):
            updated = self._apply_overwrite_retry_policy(plan)
            if updated > 0:
                notes.append(f"自动恢复：检测到输出已存在冲突，已启用覆盖策略并更新 {updated} 个步骤。")
                return True

        if failed_step.tool == "export_map":
            if "no arcgis pro project available" in error_text or "no_project" in error_text or ".aprx" in error_text:
                snapshot = self.context_hub.discover()
                project_path = self.context_hub.best_project_path(snapshot)
                if not project_path:
                    notes.append("自动恢复终止：未发现可用 .aprx，无法继续 export_map。")
                    return False
                for step in plan.steps:
                    if step.tool != "export_map":
                        continue
                    if step.status in {StepStatus.FAILED, StepStatus.PENDING}:
                        step.input["project_path"] = project_path
                        if step.status == StepStatus.FAILED:
                            step.status = StepStatus.PENDING
                            step.error = None
                            step.started_at = None
                            step.completed_at = None
                notes.append("自动恢复：已自动注入 project_path 并重试 export_map。")
                return True

        if failed_step.tool == "merge_layers":
            no_input_markers = ["no input layers", "no valid input layers", "no supported input datasets", "no_inputs"]
            if any(marker in error_text for marker in no_input_markers):
                snapshot = self.context_hub.discover()
                best_root = self.context_hub.best_input_root(snapshot)
                failed_step.input["input_layers"] = [best_root]
                failed_step.status = StepStatus.PENDING
                failed_step.error = None
                failed_step.started_at = None
                failed_step.completed_at = None
                notes.append("自动恢复：merge_layers 输入缺失，已回退到自动发现的输入目录重试。")
                return True

        if failed_step.tool in {"project_layers", "quality_check"}:
            if "input_path" in error_text and ("missing" in error_text or "缺少" in error_text or "required" in error_text):
                snapshot = self.context_hub.discover()
                inferred = snapshot.shapefiles[0] if snapshot.shapefiles else ""
                if inferred:
                    failed_step.input["input_path"] = inferred
                    failed_step.status = StepStatus.PENDING
                    failed_step.error = None
                    failed_step.started_at = None
                    failed_step.completed_at = None
                    notes.append("自动恢复：已为步骤补齐 input_path 并重试。")
                    return True

        return False

    def _is_false_success_after_recovery(self, result: ExecutionResult) -> bool:
        """Detect success states that actually produced no actionable output."""
        if not result.success:
            return False
        if result.outputs:
            return False
        delivered_outputs = self._collect_output_candidates(result)
        if any(self._looks_like_deliverable_path(p) for p in delivered_outputs):
            return False
        if not result.trace or not result.trace.steps:
            return True
        # If all executed steps are only quality checks, we treat as non-delivery.
        executed_tools = [s.tool for s in result.trace.steps if s.status == "completed"]
        if not executed_tools:
            return True
        non_deliver_tools = {"quality_check"}
        if all(t in non_deliver_tools for t in executed_tools):
            return True
        # 检查 execute_code 步骤是否通过 set_result() 返回了数据
        # 如果有数据，说明代码确实产出了结果，不应判定为假成功
        for step in result.trace.steps:
            if step.tool == "execute_code" and step.status == "completed":
                payload = step.output
                result_data = None
                if isinstance(payload, dict):
                    result_data = payload.get("result")
                else:
                    result_data = getattr(payload, "result", None)
                if result_data:  # set_result() 返回了非空数据
                    return False
        # If workflow includes deliverable tools but outputs are empty, treat as false success.
        deliverable_tools = {"merge_layers", "project_layers", "export_map", "execute_code"}
        return any(t in deliverable_tools for t in executed_tools)
    
    def create_plan(self, task_description: str) -> Plan:
        """Create an execution plan for a task.
        
        Args:
            task_description: Natural language task description
            
        Returns:
            Plan with steps to execute
        """
        workspace = self.config.workspace_path or Path.cwd()
        context_workspace = self.memory.get_context("workspace")
        if isinstance(context_workspace, str) and context_workspace.strip():
            workspace = Path(context_workspace)

        context_input_folder = self.memory.get_context("input_folder")
        if isinstance(context_input_folder, str) and context_input_folder.strip():
            input_folder = self._resolve_best_input_folder(context_input_folder, workspace)
        else:
            input_folder = self._resolve_best_input_folder(str(workspace / "input"), workspace)

        context = {
            "workspace": str(workspace),
            "input_folder": input_folder,
            "output_folder": str(workspace / "output"),
            "user_specified_input": self.memory.get_context("input_folder", ""),
            "user_specified_output": self.memory.get_context("output_folder", ""),
            "input_files": self._collect_recent_input_files(),
            "document_path": self.memory.get_context("document_path"),
            "document_constraints": self.memory.get_context("document_constraints", []),
            "document_summary": self.memory.get_context("document_summary", ""),
            "arcpy_available": self.context.arcpy_available,
            "structured_memories": [m.to_dict() for m in self.memory.search_structured_memories(task_description, top_k=8)],
            "reflection_hints": self.memory.get_reflection_hints(task_description, top_k=4),
        }

        snapshot = self.context_hub.discover()
        context.update(self.context_hub.build_planner_payload(snapshot))
        
        plan = self.planner.plan(task_description, context, expert_mode=self.config.expert_mode)
        if plan is None:
            plan = Plan(
                id=f"plan_fallback_{int(time.time())}",
                goal=task_description,
                steps=[
                    PlanStep(
                        id="step_1",
                        tool="scan_layers",
                        description="扫描输入数据",
                        input={"path": input_folder, "include_subdirs": True},
                        depends_on=[],
                    )
                ],
                expected_outputs=[],
                metadata={"fallback_reason": "planner_returned_none"},
            )
        self._preflight_plan(plan, task_description)
        self._inject_smart_defaults(plan, snapshot)
        self.current_plan = plan
        self._persist_runtime_state()
        self._update_workflow_state(
            status="planned",
            current_plan_id=plan.id,
            current_goal=plan.goal,
            metadata={"steps": len(plan.steps)},
            event="plan_created",
            event_payload={"plan_id": plan.id, "goal": plan.goal[:160]},
        )
        
        return plan

    def _inject_smart_defaults(self, plan: Plan, snapshot: ContextSnapshot) -> None:
        """Inject context-aware defaults after planning.

        This keeps the existing planner framework unchanged while making step
        inputs robust in heterogeneous real-world workspaces.
        """
        best_project = self.context_hub.best_project_path(snapshot)
        best_input_root = self.context_hub.best_input_root(snapshot)

        for step in plan.steps:
            if step.tool == "scan_layers":
                current_path = step.input.get("path")
                if not isinstance(current_path, str) or not current_path.strip():
                    step.input["path"] = best_input_root
                step.input.setdefault("include_subdirs", True)

            if step.tool == "export_map":
                if best_project and not step.input.get("project_path"):
                    step.input["project_path"] = best_project
                if not step.input.get("output_path"):
                    step.input["output_path"] = str(Path(snapshot.output_root) / "map.pdf")
    
    def call_tool(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        dry_run: bool = True
    ) -> Any:
        """Call a tool directly.
        
        Args:
            tool_name: Name of the tool to call
            input_data: Input parameters for the tool
            dry_run: Whether to run in dry-run mode
            
        Returns:
            Tool result
        """
        result = self.executor.execute_single_tool(tool_name, input_data, dry_run)
        
        # Record in memory
        self.memory.add_tool_call(tool_name, input_data, result.data if result.success else result.error)
        
        return result
    
    # === Intent Analysis ===
    
    def _analyze_intent(self, message: str) -> str:
        """Analyze user intent from message.
        
        Uses LLM if available for better understanding, falls back to rules.
        """
        message_lower = message.lower().strip()

        if self._message_implies_document_usage(message_lower):
            return "execute_task"

        if self._looks_like_affirmative(message_lower) and self._has_recent_document_hint():
            return "execute_task"

        # If a plan is pending, continuation/retry phrases should resume execution
        # instead of being interpreted as status queries by the LLM.
        if self.current_plan and not self.current_plan.is_complete:
            stripped_pending = message_lower.strip()
            resume_markers = [
                "继续执行", "继续", "重试", "retry", "resume", "接着执行", "继续跑",
                "再试", "再试试", "换个方法", "换个方式", "试试别的", "继续之前", "继续上一个",
                "重写代码", "代码重写", "重写", "重做", "重新", "换一种", "换种", "换一下",
                # Confirmation markers when plan is pending
                "执行", "执行吧", "确认执行", "可以执行", "开始执行", "开始吧",
                "可以的", "好的", "好", "行", "是", "嗯", "ok", "yes",
            ]
            if any(marker in stripped_pending for marker in resume_markers):
                return "confirm_action"
            if self._contains_overwrite_intent(stripped_pending):
                return "confirm_action"

            # If user is urging action while plan is pending, treat as execute confirmation.
            # Example: "扫啊" / "你倒是扫啊" / "赶紧执行".
            pending_status_markers = ["好了吗", "完成了吗", "进度", "状态", "结果", "了吗", "?", "？"]
            action_push_markers = [
                "扫", "扫描", "执行", "开始", "跑", "run", "做", "制图", "出图", "导出", "继续",
            ]
            urgency_markers = ["赶紧", "快点", "马上", "立刻", "现在就", "倒是"]
            if any(marker in stripped_pending for marker in action_push_markers):
                if len(stripped_pending) <= 12 or any(marker in stripped_pending for marker in urgency_markers):
                    return "confirm_action"

            if any(marker in stripped_pending for marker in pending_status_markers):
                return "query_status"
        
        # Check if user is selecting from suggestions (numeric input)
        if message_lower in ["1", "2", "3", "4", "5"]:
            return self._handle_suggestion_selection(message_lower)
        
        # Use LLM for intent analysis if available
        if self.llm_client:
            try:
                intent = self._analyze_intent_with_llm(message)
                if intent:
                    # Guardrail: do not let verbose/general chat responses hijack
                    # clearly actionable commands.
                    rule_intent = self._analyze_intent_with_rules(message_lower)
                    if intent == "general" and rule_intent == "execute_task":
                        return "execute_task"
                    return intent
            except Exception:
                # Fall back to rule-based
                pass
        
        # Rule-based fallback
        return self._analyze_intent_with_rules(message_lower)
    
    def _analyze_intent_with_llm(self, message: str) -> str | None:
        """Use LLM to analyze user intent."""
        tools_list = [t.name for t in self.tool_registry.list_tools()]
        skills_list = self._get_available_skill_names()
        valid_intents = ["execute_task", "confirm_action", "preview", "cancel", "query_status", "tool_help", "general"]

        baml_intent = self.baml_bridge.infer_intent(message, valid_intents)
        if baml_intent:
            return baml_intent
        
        system_prompt = f"""你是 GIS Agent 的意图分析器。根据用户输入判断意图类型。

可用工具: {', '.join(tools_list)}
可用技能: {', '.join(skills_list)}

意图类型:
- general: 问候、闲聊、询问功能（"你好"、"你是谁"、"你能做什么"）
- execute_task: 用户想执行具体 GIS 任务（如"制作地图"、"处理数据"、"分析图层"）
- confirm_action: 用户确认执行当前计划（"执行"、"确认"、"好的"）
- preview: 用户想预览/模拟执行（"预览"、"dry-run"）
- cancel: 用户想取消当前操作（"取消"、"算了"）
- query_status: 用户查询状态/进度（"进度如何"、"结果"）
- tool_help: 用户寻求详细使用文档（"帮助文档"、"使用手册"）

重要:
- "你好"、"你可以做什么"、"功能介绍"  general（友好对话）
- "制作专题图"、"合并图层"  execute_task（具体任务）
- "帮助文档"、"使用手册"  tool_help（需要文档）

输出要求（必须遵守）:
- 优先输出 JSON: {"intent": "execute_task"}
- 若非 JSON，只返回一个意图标签（小写英文）
- 不要解释。"""
        
        model_hint = self._get_llm_model_hint()
        adapted_system_prompt = self.prompt_adapter.adapt_prompt(
            system_prompt,
            task_type="intent",
            model_hint=model_hint,
        )

        try:
            response = self.llm_client.chat([
                {"role": "system", "content": adapted_system_prompt},
                {"role": "user", "content": message}
            ], task_type="intent")
            
            intent = self._extract_intent_from_llm_response(response)
            
            if intent in valid_intents:
                return intent
        except Exception:
            pass
        
        return None

    def _extract_intent_from_llm_response(self, response: str) -> str:
        """Normalize intent output across different model styles.

        Accepts strict labels, JSON payloads, and mixed natural-language responses.
        """
        text = (response or "").strip()
        if not text:
            return ""

        valid_intents = [
            "execute_task", "confirm_action", "preview", "cancel",
            "query_status", "tool_help", "general"
        ]

        # 1) Prefer JSON payload: {"intent": "execute_task"}
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                raw = str(payload.get("intent", "")).strip().lower()
                if raw in valid_intents:
                    return raw
        except Exception:
            pass

        lowered = text.lower()

        # 2) Exact match
        if lowered in valid_intents:
            return lowered

        # 3) Prefix/contains fallback for verbose models
        for intent in valid_intents:
            if lowered.startswith(intent):
                return intent
        for intent in valid_intents:
            if intent in lowered:
                return intent

        # 4) Chinese synonyms fallback
        zh_map = {
            "执行任务": "execute_task",
            "任务执行": "execute_task",
            "确认操作": "confirm_action",
            "确认执行": "confirm_action",
            "预览": "preview",
            "取消": "cancel",
            "状态查询": "query_status",
            "查询状态": "query_status",
            "工具帮助": "tool_help",
            "帮助": "tool_help",
            "通用": "general",
            "闲聊": "general",
        }
        for zh_key, mapped in zh_map.items():
            if zh_key in lowered:
                return mapped

        return ""
    
    def _analyze_intent_with_rules(self, message_lower: str) -> str:
        """Rule-based intent analysis (fallback)."""
        # Follow-up/retry phrasing should be treated as actionable task continuation.
        followup_task_markers = [
            "继续之前", "继续上一个", "继续前面的", "继续这个任务", "继续任务",
            "再试", "再试试", "换个方法", "换个方式", "试试别的", "接着做",
            "就在input", "就在之前的文件夹", "之前的文件夹", "完成这个任务",
        ]
        if any(m in message_lower for m in followup_task_markers):
            return "execute_task"

        # Question/bug-report style messages should not be mistaken for execution requests.
        # Example: "你为什么只执行合并图层" is feedback, not a new task.
        question_markers = ["为什么", "为何", "怎么回事", "没看到", "看不到", "是不是", "bug", "问题", "?", "？"]
        task_markers = [
            "执行", "继续", "重试", "再试", "换个方法", "换个方式", "扫描", "扫", "制图", "出图", "地图",
            "图层", "投影", "合并", "导出", "任务", "arcpy", "input", "之前",
        ]
        starts_with_command = message_lower.strip().startswith((
            "请", "帮我", "我要", "开始", "执行", "确认", "扫描", "合并", "制作", "导出", "投影", "检查"
        ))
        has_task_marker = any(m in message_lower for m in task_markers)
        if any(m in message_lower for m in question_markers) and not starts_with_command and not has_task_marker:
            return "general"

        # Greeting/introduction (should go to general, not tool_help)
        if any(kw in message_lower for kw in [
            "你好", "您好", "hi", "hello",
            "能做什么", "可以做什么", "能干什么", "可以干什么",
            "功能", "介绍一下"
        ]):
            return "general"
        
        # Task execution keywords
        if any(kw in message_lower for kw in [
            "制作", "创建", "生成", "导出", "整合", "合并", 
            "投影", "变换", "转换", "转化", "坐标系", "检查", "扫描", "分析",
            "制图", "绘制", "出图", "地图", "行政区划", "图例", "比例尺", "指北针", "扫", "扫一下",
            "继续", "重试", "再试", "换个方法", "换个方式", "继续之前", "之前任务",
            "create", "make", "export", "merge", "project", "check", "scan"
        ]):
            return "execute_task"

        # Common short imperative forms that often appear in follow-up turns.
        if message_lower.strip() in {"扫", "扫啊", "扫描", "扫描啊", "制图", "出图", "画图"}:
            return "execute_task"

        # Generic actionable requests should also route to task execution.
        # The task handler will decide whether to execute GIS flow or fall back
        # to skill-first/general handling.
        if any(kw in message_lower for kw in [
            "帮我", "请你", "请帮", "给我", "写", "整理", "总结", "翻译", "提取", "计算", "生成",
            "implement", "build", "write", "summarize", "translate", "extract"
        ]):
            return "execute_task"
        
        # Cancel/abort
        if any(kw in message_lower for kw in [
            "取消", "算了", "不要", "停止",
            "cancel", "abort", "stop", "nevermind"
        ]):
            return "cancel"
        
        # Preview/dry-run
        if any(kw in message_lower for kw in [
            "预览", "dry-run", "dry run", "模拟"
        ]):
            return "preview"
        
        # Status query
        if any(kw in message_lower for kw in [
            "状态", "进度", "结果", "输出",
            "status", "progress", "result", "output"
        ]):
            return "query_status"
        
        # Help request (specific help keywords only)
        if any(kw in message_lower for kw in [
            "帮助文档", "使用说明", "教程",
            "help documentation", "manual", "tutorial"
        ]):
            return "tool_help"
        
        # Confirmation (strict matching to avoid hijacking follow-up task messages)
        stripped = message_lower.strip()
        overwrite_confirm_words = ["可以", "确认", "执行", "重试", "继续", "直接", "强制"]
        if self._contains_overwrite_intent(stripped) and any(w in stripped for w in overwrite_confirm_words):
            return "confirm_action"

        if (
            any(kw in stripped for kw in ["确认", "执行", "confirm", "execute"])
            or stripped in {"1", "是", "好的", "好", "可以", "yes", "ok"}
        ):
            return "confirm_action"
        
        return "general"
    
    def _handle_suggestion_selection(self, selection: str) -> str:
        """Handle numeric suggestion selection, context-aware."""
        idx = int(selection) - 1

        # Check if we have a current plan (user is selecting action for plan)
        if self.current_plan and not self.current_plan.is_complete:
            if self.current_plan.has_failed:
                # Recovery suggestions: ["查看详细错误信息", "重试失败的步骤", "跳过失败步骤继续"]
                recovery_actions = ["query_status", "confirm_action", "skip_step"]
                if 0 <= idx < len(recovery_actions):
                    if recovery_actions[idx] == "confirm_action":
                        self._pending_confirmation_mode = ExecutionMode.EXECUTE
                    return recovery_actions[idx]

            # Normal suggestions: ["执行计划", "预览 dry-run", "取消"]
            if idx == 0:
                self._pending_confirmation_mode = ExecutionMode.EXECUTE
                return "confirm_action"  # Execute
            elif idx == 1:
                return "preview"  # Dry-run
            elif idx == 2:
                return "cancel"  # Cancel
        
        # Default suggestions: ["扫描 GIS 数据", "制作专题图", "数据质量检查"]
        if idx == 0:
            self._pending_task = "扫描当前工作目录的 GIS 数据"
            return "execute_task"
        elif idx == 1:
            self._pending_task = "制作专题图"
            return "execute_task"
        elif idx == 2:
            self._pending_task = "执行数据质量检查"
            return "execute_task"
        
        return "general"
    
    # === Intent Handlers ===
    
    def _handle_task_execution(self, message: str) -> AgentResponse:
        """Handle task execution request.
        
        Uses LLM to understand natural language and create appropriate plan.
        """
        planning_message, used_followup_context = self._prepare_task_for_planning(message)

        # GIS-first policy: always attempt GIS planning for actionable requests.
        # This ensures user requests like "继续进行行政区划图的制作" route to execution,
        # not filtered out by overly-strict GIS keyword detection.

        planning_message, doc_enrichment_note = self._enrich_task_with_word_document(planning_message)
        planning_message, skill_enrichment_note = self._enrich_task_with_custom_skill_context(planning_message)

        # Use LLM to refine/understand the task if available
        if self.llm_client and not used_followup_context:
            try:
                refined_task = self._refine_task_with_llm(planning_message)
                if refined_task:
                    planning_message = refined_task
            except Exception:
                pass  # Use original message
        
        # Create plan
        plan = self.create_plan(planning_message)
        
        # Build response
        plan_summary = self._format_plan_summary(plan)
        
        if self.config.auto_execute:
            # Execute immediately
            result, recovery_notes = self._execute_with_recovery(
                plan,
                self.config.default_mode
            )
            self._record_execution_reflection(plan, result, recovery_notes)
            
            if result.success:
                delivered_outputs = self._collect_output_candidates(result)
                if delivered_outputs:
                    outputs_text = self._format_outputs(delivered_outputs)
                    content = f"任务已完成！\n\n{plan_summary}\n\n输出文件：\n{outputs_text}"
                else:
                    content = (
                        f"任务执行完成，但未产出可交付文件。\n\n{plan_summary}\n\n"
                        + self._build_non_delivery_diagnosis(result)
                    )
                if recovery_notes:
                    content += "\n\n恢复日志：\n" + "\n".join(f"- {n}" for n in recovery_notes)
            else:
                content = f"任务执行失败：{result.error}\n\n{plan_summary}"
                if recovery_notes:
                    content += "\n\n恢复日志：\n" + "\n".join(f"- {n}" for n in recovery_notes)
            
            return AgentResponse(
                content=content,
                action_taken="execute",
                plan=plan,
                execution_result=result,
                suggestions=self._get_recovery_suggestions(result) if not result.success else []
            )
        else:
            # Preview mode
            content = f"已创建执行计划：\n\n{plan_summary}\n\n输入'执行'或'确认'开始执行，或输入'取消'取消计划。"
            if doc_enrichment_note:
                content += f"\n\n文档上下文：{doc_enrichment_note}"
            if skill_enrichment_note:
                content += f"\n\n技能上下文：{skill_enrichment_note}"
            
            return AgentResponse(
                content=content,
                action_taken="plan",
                plan=plan,
                suggestions=["执行计划", "预览 dry-run", "取消"]
            )

    def _looks_like_gis_task(self, message: str) -> bool:
        """Heuristic GIS-domain detection used for routing guardrails."""
        text = (message or "").strip().lower()
        if not text:
            return False
        gis_markers = [
            "gis", "地图", "图层", "地理", "空间", "坐标", "投影", "aprx", "arcpy",
            "shp", ".shp", "gdb", ".gdb", "geojson", "栅格", "矢量", "缓冲区", "叠加分析",
            "专题图", "制图", "区划", "行政区", "行政区划",
            "scan_layers", "merge_layers", "project_layers", "quality_check", "export_map",
        ]
        return any(marker in text for marker in gis_markers)

    def _enrich_task_with_custom_skill_context(self, planning_message: str) -> tuple[str, str]:
        """Inject matched custom-skill context into planning text.

        - Executable skills: planner will consume directly via _try_match_custom_skill.
        - Reference-style generic skills: inject concise guidance so planner/LLM can use it.
        """
        skill_match = self.skill_loader.find_best_match_with_score(
            planning_message,
            min_score=5,
            only_executable=False,
        )
        if not skill_match:
            return planning_message, ""

        skill = skill_match.skill
        if getattr(skill, "is_executable", False):
            return planning_message, f"已匹配可执行技能 {skill.name}"

        trigger_text = ", ".join(skill.triggers[:6]) if skill.triggers else "（未定义触发词）"
        steps_text = "\n".join(f"- {x}" for x in skill.steps[:6]) if skill.steps else "- （未定义步骤）"
        template_preview = (skill.code_template or "").strip()
        if len(template_preview) > 380:
            template_preview = template_preview[:380] + "\n..."
        if not template_preview:
            template_preview = "（无代码模板）"

        enriched = (
            f"{planning_message}\n\n"
            f"[通用技能参考]\n"
            f"- skill: {skill.name}\n"
            f"- description: {skill.description}\n"
            f"- triggers: {trigger_text}\n\n"
            f"[skill步骤建议]\n{steps_text}\n\n"
            f"[skill代码/片段参考]\n{template_preview}\n"
        )
        note = f"已匹配通用技能 {skill.name}（参考模式）"
        return enriched, note

    def _enrich_task_with_word_document(self, planning_message: str) -> tuple[str, str]:
        """If message contains .docx/.pdf path, extract text and merge parsed requirements."""
        doc_candidate = self._extract_document_path_from_message(planning_message)
        if not doc_candidate:
            return planning_message, ""

        resolved_doc = self._resolve_doc_path(doc_candidate)
        ext = Path(resolved_doc).suffix.lower()
        if ext == ".pdf":
            read_tool = "read_pdf"
            source_kind = "pdf"
        else:
            read_tool = "read_word"
            source_kind = "word"

        read_result = self.executor.execute_single_tool(
            read_tool,
            {"file_path": resolved_doc, "max_chars": 15000},
            dry_run=False,
        )
        if not read_result.success or read_result.data is None:
            return planning_message, f"文档解析失败（{resolved_doc}）：{read_result.error or 'unknown error'}"

        raw_text = getattr(read_result.data, "text", "")
        if not isinstance(raw_text, str) or not raw_text.strip():
            return planning_message, f"文档为空或未提取到文本（{resolved_doc}）"

        parser = DocumentTaskParser()
        try:
            parsed = parser.parse_text(raw_text)
        except Exception as e:
            return planning_message, f"文档内容提炼失败（{resolved_doc}）：{e}"

        self.memory.set_context("document_path", resolved_doc)
        self.memory.set_context("document_constraints", parsed.constraints[:20])
        self.memory.set_context("document_summary", parsed.normalized_prompt[:600])
        self.memory.add_structured_memory(
            "document_requirement",
            parsed.normalized_prompt[:500],
            metadata={"source": source_kind, "path": resolved_doc},
            importance=3,
        )

        constraints_text = "\n".join(f"- {item}" for item in parsed.constraints[:8])
        if not constraints_text:
            constraints_text = "- （文档未提取到显式约束）"

        enriched = (
            f"{planning_message}\n\n"
            f"[文档来源]\n- {resolved_doc}\n\n"
            f"[文档提炼任务]\n{parsed.normalized_prompt}\n\n"
            f"[文档约束]\n{constraints_text}"
        )
        note = f"已读取文档 {Path(resolved_doc).name}，并注入任务约束"
        return enriched, note

    def _extract_document_path_from_message(self, message: str) -> str | None:
        """Extract first .docx/.pdf path from user message.

        Skips paths that look like output files (containing /output/ or \\output\\)
        since those are typically export targets, not input documents.
        """
        patterns = [
            r'([A-Za-z]:\\[^\n\r"\']+?\.(?:docx|pdf))',
            r'([A-Za-z]:/[^\n\r"\']+?\.(?:docx|pdf))',
            r'((?:\.\\|\./|workspace\\|workspace/|input\\|input/)[^\n\r"\']+?\.(?:docx|pdf))',
        ]
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                path = match.group(1).strip()
                # Skip output paths — user is referring to a file to be created, not an input doc
                if re.search(r'[/\\]output[/\\]', path, re.IGNORECASE):
                    continue
                return path
        return self._infer_document_path_from_workspace(message)

    def _infer_document_path_from_workspace(self, message: str) -> str | None:
        """Infer document path when user mentions a doc without explicit path."""
        if not self._message_implies_document_usage(message):
            return None

        workspace = self.config.workspace_path or Path.cwd()
        normalized = message.lower()
        want_pdf = "pdf" in normalized
        want_docx = "docx" in normalized or "word" in normalized
        allowed_suffixes = [".pdf", ".docx"]
        if want_pdf and not want_docx:
            allowed_suffixes = [".pdf"]
        elif want_docx and not want_pdf:
            allowed_suffixes = [".docx"]

        # Prefer conventional input folders first.
        search_roots = [
            workspace / "input",
            workspace / "workspace" / "input",
            workspace,
        ]

        candidates: list[Path] = []
        seen: set[str] = set()
        for root in search_roots:
            if not root.exists() or not root.is_dir():
                continue

            for suffix in allowed_suffixes:
                for file_path in root.glob(f"*{suffix}"):
                    resolved = str(file_path.resolve())
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    candidates.append(file_path)

        if not candidates:
            return None

        # Score candidates by explicit filename mention and input-folder preference.
        scored: list[tuple[int, Path]] = []
        for file_path in candidates:
            score = 0
            stem = file_path.stem.lower()
            if stem and stem in normalized:
                score += 100

            parent_lower = str(file_path.parent).lower().replace("\\", "/")
            if "/input" in parent_lower or parent_lower.endswith("/input"):
                score += 20

            if file_path.suffix.lower() == ".pdf" and want_pdf:
                score += 10
            if file_path.suffix.lower() == ".docx" and want_docx:
                score += 10

            scored.append((score, file_path))

        scored.sort(key=lambda item: item[0], reverse=True)

        if len(scored) == 1:
            return str(scored[0][1])

        # If top score is strictly better, select it; otherwise avoid ambiguous guess.
        if scored[0][0] > scored[1][0]:
            return str(scored[0][1])
        return None

    def _message_implies_document_usage(self, message: str) -> bool:
        """Whether a message likely refers to document-driven tasking.

        Only triggers on explicit user intent — never auto-detect from
        output paths or broad keywords like 'pdf' or '文档'.
        """
        lowered = message.lower()
        doc_markers = [
            "根据文档", "按文档", "从文档", "读一下", "读取",
        ]
        return any(marker in lowered for marker in doc_markers)

    def _looks_like_affirmative(self, message: str) -> bool:
        """Check whether message is a short confirmation phrase."""
        stripped = message.strip().lower()
        if stripped in {
            "好", "好的", "可以", "是", "行", "嗯", "读吧", "继续", "继续吧",
            "ok", "yes", "confirm", "execute",
        }:
            return True
        # Partial match for longer confirmations like "可以的，执行吧"
        confirm_keywords = ["可以的", "可以执行", "确认执行", "开始执行", "执行吧",
                           "好的执行", "好的开始", "可以开始"]
        return any(kw in stripped for kw in confirm_keywords)

    def _has_recent_document_hint(self) -> bool:
        """Check recent user turns for document/path hints."""
        for turn in reversed(self.memory.turns[-8:]):
            if turn.role != Role.USER:
                continue
            text = (turn.content or "").strip()
            if not text:
                continue
            if self._message_implies_document_usage(text):
                return True
            if re.search(r"(?:\\.pdf|\\.docx)", text, re.IGNORECASE):
                return True
        return False

    def _extract_docx_path_from_message(self, message: str) -> str | None:
        """Extract the first .docx path from user message."""
        matched = self._extract_document_path_from_message(message)
        if matched and matched.lower().endswith(".docx"):
            return matched

        patterns = [
            r'([A-Za-z]:\\[^\n\r"\']+?\.docx)',
            r'([A-Za-z]:/[^\n\r"\']+?\.docx)',
            r'((?:\.\\|\.\/|workspace\\|workspace/|input\\|input/)[^\n\r"\']+?\.docx)',
        ]
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _resolve_doc_path(self, candidate: str) -> str:
        """Resolve document path against known workspace roots when needed."""
        raw = candidate.strip().strip('"\'')
        p = Path(raw)
        if p.is_absolute() and p.exists():
            return str(p)

        workspace = self.config.workspace_path or Path.cwd()
        candidate_paths = [
            workspace / raw,
            Path.cwd() / raw,
            workspace / "workspace" / "input" / Path(raw).name,
        ]
        for item in candidate_paths:
            if item.exists():
                return str(item)

        # keep best-effort absolute path for downstream error reporting
        return str((workspace / raw).resolve())
    
    def _refine_task_with_llm(self, message: str) -> str | None:
        """Use LLM to understand and refine task description into GIS terms."""
        baml_refined = self.baml_bridge.refine_task(message)
        if baml_refined:
            refined = self._sanitize_refined_task(baml_refined)
            return refined if refined else None

        tools_desc = "\n".join([
            f"- {t.name}: {t.description[:100]}"
            for t in self.tool_registry.list_tools()
        ])

        skills_desc = "\n".join(self._get_available_skill_lines())
        
        system_prompt = f"""你是 GIS 任务理解助手。将用户的自然语言转换为具体的 GIS 任务描述。

可用工具:
{tools_desc}

可用技能:
{skills_desc}

任务要求:
1. 理解用户意图，即使用词不专业
2. 将模糊描述转换为明确的 GIS 操作
3. 识别需要的工具和操作顺序
4. 用清晰简洁的语言描述任务

输出要求:
- 只返回 refined 任务描述
- 不要解释过程，不要分点，不要 markdown
- 建议不超过 300 字"""
        
        model_hint = self._get_llm_model_hint()
        adapted_system_prompt = self.prompt_adapter.adapt_prompt(
            system_prompt,
            task_type="task_refine",
            model_hint=model_hint,
        )

        try:
            response = self.llm_client.chat([
                {"role": "system", "content": adapted_system_prompt},
                {"role": "user", "content": f"用户输入: {message}\n\n请转换为GIS任务描述:"}
            ], task_type="task_refine")
            
            refined = self._sanitize_refined_task(response)
            return refined if refined else None
        except Exception:
            return None

    def _sanitize_refined_task(self, text: str) -> str:
        """Trim noisy model output while preserving actionable task content."""
        value = (text or "").strip()
        if not value:
            return ""

        # Strip common prefixes from non-Claude models.
        for prefix in ["输出：", "输出:", "refined:", "结果：", "结果:", "任务描述：", "任务描述:"]:
            if value.lower().startswith(prefix.lower()):
                value = value[len(prefix):].strip()
                break

        # Avoid very long rambling responses polluting planner input.
        if len(value) > 900:
            # keep first coherent chunk
            splitters = ["\n\n", "\n", "。", ". "]
            for sep in splitters:
                parts = [p.strip() for p in value.split(sep) if p.strip()]
                if parts:
                    candidate = parts[0]
                    if 20 <= len(candidate) <= 600:
                        return candidate
            return value[:600]

        return value

    def _get_llm_model_hint(self) -> str | None:
        """Get active model name for model-specific prompt adaptation."""
        config = getattr(self.llm_client, "config", None)
        if config is None:
            return None
        model = getattr(config, "model", None)
        if not model:
            return None
        return str(model).strip()
    
    def _handle_status_query(self, message: str) -> AgentResponse:
        """Handle status query."""
        if self.last_result:
            result = self.last_result
            status = "成功" if result.success else "失败"
            content = f"上次执行状态：{status}\n\n"
            
            if result.success:
                delivered_outputs = self._collect_output_candidates(result)
                if delivered_outputs:
                    content += "输出文件：\n" + "\n".join(f"- {o}" for o in delivered_outputs)
                else:
                    content += "输出文件：\n（无）\n\n"
                    if self._is_why_empty_output_query(message):
                        content += self._build_non_delivery_diagnosis(result)
                    else:
                        content += "提示：这是“执行成功但无产物”的情况，通常只完成了检查步骤。你可以问“为什么输出是空的”，我会给出具体诊断。"
            else:
                content += f"错误信息：{result.error}"
            
            return AgentResponse(
                content=content,
                action_taken="query",
                execution_result=result
            )
        elif self.current_plan:
            summary = self.current_plan.summary()
            content = f"当前计划状态：\n- 总步骤：{summary['total_steps']}\n- 已完成：{summary['completed']}\n- 失败：{summary['failed']}\n- 待执行：{summary['pending']}"
            
            return AgentResponse(
                content=content,
                action_taken="query",
                plan=self.current_plan
            )
        else:
            return AgentResponse(
                content="当前没有正在执行的任务。请输入您的 GIS 任务需求。",
                action_taken="query"
            )
    
    def _handle_tool_help(self, message: str) -> AgentResponse:
        """Handle help request."""
        # Tool descriptions in Chinese
        tool_desc_cn = {
            "scan_layers": "扫描目录中的 GIS 图层（Shapefile、GeoDatabase 等）",
            "merge_layers": "合并多个 GIS 图层为单个输出图层",
            "project_layers": "将 GIS 图层投影到目标坐标系",
            "export_map": "导出地图布局为 PDF、PNG 或其他格式",
            "quality_check": "检查 GIS 数据质量并验证几何有效性",
            "read_word": "读取 Word 文档（.docx）并提取文本内容",
            "read_pdf": "读取 PDF 文档（.pdf）并提取文本内容",
        }
        
        tools = self.tool_registry.list_tools()
        builtin_skills = self.skill_registry.list_skills()
        custom_skills = self.skill_loader.list_skills()
        
        content = "##  使用文档\n\n"
        content += f"### 可用工具（{len(tools)}个）\n\n"
        for tool in tools:
            desc = tool_desc_cn.get(tool.name, tool.description[:100])
            content += f"- **{tool.name}**: {desc}\n"
        
        total_skills = len(builtin_skills) + len(custom_skills)
        content += f"\n### 可用技能（{total_skills}个）\n\n"

        builtin_desc = {
            "thematic_map": "专题图制作（扫描合并投影导出）",
            "data_integration": "数据集成（扫描验证变换合并）",
            "quality_assurance": "质量保证（扫描检查报告）",
        }

        for skill in builtin_skills:
            desc = builtin_desc.get(skill.name, skill.description)
            content += f"- **{skill.name}**: {desc}\n"

        for skill in custom_skills:
            skill_type = "可执行" if getattr(skill, "is_executable", False) else "文档参考"
            content += f"- **{skill.name}** [{skill_type}]\n"
        
        content += "\n### 使用示例\n\n"
        content += "- 「扫描我的数据文件夹」\n"
        content += "- 「把这些图层合并成一张」\n"
        content += "- 「检查数据质量」\n"
        content += "- 「制作专题图」\n"
        
        content += "\n **提示**：直接用普通话描述你想做什么，不需要专业术语！\n"
        
        return AgentResponse(
            content=content,
            action_taken="help"
        )
    
    def _handle_confirmation(self, message: str) -> AgentResponse:
        """Handle confirmation response."""
        if self.current_plan and not self.current_plan.is_complete:
            if self.current_plan.has_failed and self._contains_retry_intent(message):
                user_feedback = self._extract_user_feedback(message)
                retry_prompt = self._build_retry_replan_prompt(
                    self.current_plan, self.last_result, user_feedback
                )
                return self._handle_task_execution(retry_prompt)

            # Execute the pending plan
            mode = self._pending_confirmation_mode or ExecutionMode.EXECUTE
            self._pending_confirmation_mode = None

            overwrite_notes: list[str] = []
            if self._contains_overwrite_intent(message):
                updated = self._apply_overwrite_retry_policy(self.current_plan)
                if updated > 0:
                    overwrite_notes.append(f"已启用覆盖策略并更新 {updated} 个步骤。")
                elif self.current_plan.has_failed:
                    overwrite_notes.append("已收到覆盖指令，但未识别到可重试的输出冲突步骤。")

            result, recovery_notes = self._execute_with_recovery(self.current_plan, mode)
            self._record_execution_reflection(self.current_plan, result, recovery_notes)
            
            if result.success:
                delivered_outputs = self._collect_output_candidates(result)
                if delivered_outputs:
                    outputs_text = self._format_outputs(delivered_outputs)
                    content = f"任务执行成功！\n\n输出文件：\n{outputs_text}"
                else:
                    content = "任务执行完成，但未产出可交付文件。\n\n" + self._build_non_delivery_diagnosis(result)
                if recovery_notes:
                    content += "\n\n恢复日志：\n" + "\n".join(f"- {n}" for n in recovery_notes)
            else:
                content = f"任务执行失败：{result.error}"
                if recovery_notes:
                    content += "\n\n恢复日志：\n" + "\n".join(f"- {n}" for n in recovery_notes)

            if overwrite_notes:
                content += "\n\n覆盖策略：\n" + "\n".join(f"- {n}" for n in overwrite_notes)
            
            return AgentResponse(
                content=content,
                action_taken="execute",
                plan=self.current_plan,
                execution_result=result,
                suggestions=self._get_recovery_suggestions(result) if not result.success else []
            )
        else:
            if self._message_implies_document_usage(message) or (
                self._looks_like_affirmative(message) and self._has_recent_document_hint()
            ):
                doc_task = message
                if not self._message_implies_document_usage(doc_task):
                    doc_task = "根据文档需求规划 GIS 任务并执行"
                return self._handle_task_execution(doc_task)

            return AgentResponse(
                content=(
                    "当前没有待确认的操作。\n"
                    "如果任务写在文档里，可以直接说：\n"
                    "- 根据 input/xxx.pdf 规划任务\n"
                    "- 读取 input 中的试题文档并按要求执行"
                ),
                action_taken="confirm"
            )
    
    def _handle_preview(self, message: str) -> AgentResponse:
        """Handle preview/dry-run request."""
        if self.current_plan and not self.current_plan.is_complete:
            # Execute in dry-run mode
            result, recovery_notes = self._execute_with_recovery(
                self.current_plan,
                ExecutionMode.DRY_RUN
            )
            
            if result.success:
                content = f"预览完成（dry-run 模式）\n\n计划步骤：\n"
                for i, step in enumerate(self.current_plan.steps, 1):
                    status = "" if step.status == "completed" else ""
                    content += f"  {i}. {status} [{step.tool}] {step.description}\n"
                delivered_outputs = self._collect_output_candidates(result)
                content += "\n模拟输出：\n" + self._format_outputs(delivered_outputs)
                if recovery_notes:
                    content += "\n\n恢复日志：\n" + "\n".join(f"- {n}" for n in recovery_notes)
            else:
                content = f"预览失败：{result.error}"
                if recovery_notes:
                    content += "\n\n恢复日志：\n" + "\n".join(f"- {n}" for n in recovery_notes)
            
            return AgentResponse(
                content=content,
                action_taken="preview",
                plan=self.current_plan,
                execution_result=result,
                suggestions=["执行计划", "取消"]
            )
        else:
            return AgentResponse(
                content="当前没有待预览的计划。请先创建一个任务。",
                action_taken="preview"
            )
    
    def _handle_cancel(self, message: str) -> AgentResponse:
        """Handle cancel request."""
        if self.current_plan:
            plan_goal = self.current_plan.goal
            self.current_plan = None
            self._persist_runtime_state()
            self._update_workflow_state(
                status="cancelled",
                current_plan_id="",
                metadata={"reason": "user_cancel"},
                event="plan_cancelled",
                event_payload={"goal": plan_goal[:160]},
            )
            return AgentResponse(
                content=f"已取消计划：{plan_goal}",
                action_taken="cancel",
                suggestions=["扫描 GIS 数据", "制作专题图", "数据质量检查"]
            )
        else:
            return AgentResponse(
                content="当前没有待取消的计划。",
                action_taken="cancel"
            )

    def _handle_skip_step(self, message: str) -> AgentResponse:
        """Skip the failed step(s) and continue execution."""
        if not self.current_plan:
            return AgentResponse(
                content="当前没有正在执行的计划。",
                action_taken="skip"
            )

        skipped = []
        for step in self.current_plan.get_failed_steps():
            step.status = StepStatus.SKIPPED
            step.error = None
            skipped.append(step.id)

        if not skipped:
            return AgentResponse(
                content="没有可跳过的失败步骤。",
                action_taken="skip"
            )

        result, recovery_notes = self._execute_with_recovery(
            self.current_plan, ExecutionMode.EXECUTE
        )

        content = f"已跳过步骤：{', '.join(skipped)}。"
        if result.success:
            delivered = self._collect_output_candidates(result)
            if delivered:
                content += f"\n\n后续步骤执行成功！\n\n输出文件：\n{self._format_outputs(delivered)}"
            else:
                content += "\n\n后续步骤执行完成。"
        else:
            content += f"\n\n后续步骤执行失败：{result.error}"

        if recovery_notes:
            content += "\n\n恢复日志：\n" + "\n".join(f"- {n}" for n in recovery_notes)

        return AgentResponse(
            content=content,
            action_taken="skip",
            plan=self.current_plan,
            execution_result=result,
        )

    def _handle_general_query(self, message: str) -> AgentResponse:
        """Handle general query."""
        """Handle general query."""
        if self.last_result and self._is_why_empty_output_query(message):
            return self._handle_status_query(message)

        llm_init_error = self.memory.get_context("llm_init_error", "")
        if not self.llm_client and isinstance(llm_init_error, str) and llm_init_error.strip():
            return AgentResponse(
                content=(
                    "当前未启用模型推理能力，正在使用离线兜底回复。\n"
                    f"原因：{llm_init_error}\n\n"
                    "请先修复 LLM 初始化问题后再重试。"
                ),
                action_taken="chat",
                suggestions=["扫描 GIS 数据", "制作专题图", "数据质量检查"],
            )

        # Check if user is asking for help/introduction
        is_greeting = any(kw in message.lower() for kw in [
            "你好", "hello", "hi", "帮助", "help", "怎么用", "如何使用",
            "能做什么", "what can", "介绍", "introduce"
        ])
        
        # Use LLM if available
        if self.llm_client:
            # Build detailed system prompt for better understanding
            tools_desc = "\n".join([
                f"- {t.name}: {t.description}"
                for t in self.tool_registry.list_tools()
            ])
            skills_desc = "\n".join(self._get_available_skill_lines())
            
            # Enhanced system prompt
            system_prompt = SystemPrompts.get_system_prompt(self.config.language)
            
            # Add available tools/skills info
            enhanced_prompt = f"""{system_prompt}

## 当前可用工具：
{tools_desc}

## 当前可用技能：
{skills_desc}

## 对话历史：
{self._format_conversation_history()}

请根据用户的输入，友好专业地回应。记住：
1. 默认用中文回答
2. 对新手要详细引导
3. 对专家要高效执行
4. 不确定时主动询问细节
"""
            
            try:
                response = self.llm_client.chat([
                    {"role": "system", "content": enhanced_prompt},
                    {"role": "user", "content": message}
                ])
                
                # Add helpful suggestions based on context
                suggestions = []
                if is_greeting or not self.memory.turns:
                    suggestions = [
                        " 扫描我的数据文件夹",
                        " 制作专题图", 
                        " 检查数据质量",
                        " 查看使用教程"
                    ]
                
                return AgentResponse(
                    content=response,
                    action_taken="chat",
                    suggestions=suggestions
                )
            except Exception as e:
                error_msg = str(e).strip() or e.__class__.__name__
                self.memory.set_context("last_llm_chat_error", error_msg)
                return AgentResponse(
                    content=(
                        "模型调用失败，当前无法生成实时回复。\n"
                        f"错误：{error_msg}\n\n"
                        "请检查 llm_config.json 的 provider/model/api_base 是否匹配，或稍后重试。"
                    ),
                    action_taken="chat",
                    suggestions=["查看当前模型配置", "重试", "执行 GIS 任务"],
                )
        
        # Fallback response for new users
        if is_greeting or not self.memory.turns:
            content = """你好！我是 GIS Agent，一个地理数据处理助手 

我可以帮你：
 扫描和整理 GIS 数据
 制作专题地图
 检查数据质量
 合并和处理图层

**新手快速开始：**
1. 告诉我你的数据在哪里（如"数据在 ./data 文件夹"）
2. 告诉我你想做什么（用普通话描述就行）
3. 我会制定计划，征求你的确认后执行

**示例任务：**

有什么我可以帮你的吗？"""
            
            return AgentResponse(
                content=content,
                action_taken="chat",
                suggestions=[
                    " 扫描我的数据文件夹",
                    " 制作专题图",
                    " 检查数据质量"
                ]
            )
        
        # Fallback for other queries
        return AgentResponse(
            content=f"我理解您想了解：{message}\n\n作为 GIS 助手，我可以帮助您处理地理数据、制作地图和进行空间分析。\n\n请告诉我具体想做什么，我会制定详细的执行计划。如果您不确定怎么描述，可以说说您手头有什么数据，想达成什么目标。",
            action_taken="chat",
            suggestions=["扫描 GIS 数据", "制作专题图", "数据质量检查"]
        )
    
    def _format_conversation_history(self) -> str:
        """Format recent conversation for LLM context."""
        recent_turns = self.memory.turns[-10:]  # Last 10 turns (5 pairs of user/assistant)
        if not recent_turns:
            return "(这是首次对话)"

        history = []
        for turn in recent_turns:
            role_label = "用户" if turn.role.value == "user" else "助手"
            content_preview = turn.content[:200] if turn.content else ""
            history.append(f"{role_label}: {content_preview}")

        return "\n".join(history)

    def _get_available_skill_names(self) -> list[str]:
        """Get all available skill names (builtin + custom)."""
        names: list[str] = []
        for skill in self.skill_registry.list_skills():
            if skill.name not in names:
                names.append(skill.name)
        for skill in self.skill_loader.list_skills():
            if skill.name not in names:
                names.append(skill.name)
        return names

    def _get_available_skill_lines(self) -> list[str]:
        """Get human-readable skill lines for prompts (builtin + custom)."""
        lines: list[str] = []
        for skill in self.skill_registry.list_skills():
            lines.append(f"- {skill.name}: {skill.description}")
        for skill in self.skill_loader.list_skills():
            skill_type = "可执行" if getattr(skill, "is_executable", False) else "文档参考"
            lines.append(f"- {skill.name}: {skill.description}（{skill_type}）")
        return lines
    
    # === Helper Methods ===
    
    def _format_plan_summary(self, plan: Plan) -> str:
        """Format plan as readable summary."""
        lines = [f"**目标**: {plan.goal}", "", "**执行步骤**:"]

        for i, step in enumerate(plan.steps, 1):
            status_icon = ""
            if step.status.value == "completed":
                status_icon = ""
            elif step.status.value == "failed":
                status_icon = ""
            elif step.status.value == "running":
                status_icon = ""

            lines.append(f"{i}. {status_icon} [{step.tool}] {step.description}")

        # Add plan warnings from self-verification
        warnings = plan.metadata.get("plan_warnings", [])
        if warnings:
            lines.append("")
            lines.append("**⚠️ 规划建议**:")
            for w in warnings:
                lines.append(f"- {w}")

        return "\n".join(lines)

    def _format_outputs(self, outputs: list[str]) -> str:
        """Format output file list for user display."""
        if not outputs:
            return "（未检测到输出文件；可能只完成了检查步骤，或工具未返回输出路径）"
        return "\n".join(f"- {o}" for o in outputs)

    def _looks_like_deliverable_path(self, value: str) -> bool:
        """Heuristic: whether a string likely points to a deliverable output path."""
        if not isinstance(value, str):
            return False
        text = value.strip()
        if not text:
            return False

        lower = text.lower().replace("\\", "/")
        if ".gdb/" in lower or lower.endswith(".gdb"):
            return True

        suffixes = (
            ".shp", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff",
            ".csv", ".xlsx", ".dbf", ".gpkg", ".lyrx", ".mapx",
        )
        if lower.endswith(suffixes):
            return True

        # 兜底：允许已存在路径被识别为交付结果
        try:
            p = Path(text)
            if p.exists():
                return True
            if not lower.endswith(".gdb") and Path(text + ".gdb").exists():
                return True
        except Exception:
            return False

        return False

    def _collect_output_candidates(self, result: ExecutionResult) -> list[str]:
        """Collect output candidates from result.outputs and step payloads."""
        merged: list[str] = []
        seen: set[str] = set()

        for item in result.outputs or []:
            if isinstance(item, str) and item.strip() and item not in seen:
                seen.add(item)
                merged.append(item)

        if not result.trace or not result.trace.steps:
            return merged

        def _append_path(path_value: Any) -> None:
            if isinstance(path_value, str) and path_value.strip() and path_value not in seen:
                seen.add(path_value)
                merged.append(path_value)

        for step in result.trace.steps:
            payload = step.output
            if payload is None:
                continue

            if isinstance(payload, dict):
                for key in ("output_path", "output", "path", "report_path", "file_path"):
                    _append_path(payload.get(key))
                # 深入 execute_code 的 result 字段（set_result() 返回的数据）
                result_data = payload.get("result")
                if isinstance(result_data, dict):
                    for val in result_data.values():
                        if isinstance(val, str) and val.strip():
                            _append_path(val)
                elif isinstance(result_data, str) and result_data.strip():
                    _append_path(result_data)
                continue

            for attr in ("output_path", "path", "report_path", "file_path"):
                _append_path(getattr(payload, attr, None))
            # 深入 execute_code 的 result 字段（set_result() 返回的数据）
            result_data = getattr(payload, "result", None)
            if isinstance(result_data, dict):
                for val in result_data.values():
                    if isinstance(val, str) and val.strip():
                        _append_path(val)
            elif isinstance(result_data, str) and result_data.strip():
                _append_path(result_data)

        return merged

    def _is_why_empty_output_query(self, message: str) -> bool:
        lowered = message.lower()
        why_markers = ["为什么", "为何", "怎么", "why"]
        empty_markers = ["空", "为空", "没输出", "无输出", "output", "输出文件"]
        return any(w in lowered for w in why_markers) and any(e in lowered for e in empty_markers)

    def _build_non_delivery_diagnosis(self, result: ExecutionResult) -> str:
        """Explain why execution succeeded without deliverable outputs."""
        if not result.trace or not result.trace.steps:
            return (
                "诊断：执行链路未记录详细步骤，无法定位产物缺失点。\n"
                "建议：开启详细执行日志后重试。"
            )

        completed_tools = [s.tool for s in result.trace.steps if s.status == "completed"]
        tool_line = " -> ".join(completed_tools) if completed_tools else "（无已完成步骤）"

        reasons: list[str] = []
        suggestions: list[str] = []
        non_deliver_tools = {"scan_layers", "quality_check"}
        if completed_tools and all(t in non_deliver_tools for t in completed_tools):
            reasons.append("本次只执行了扫描/检查类步骤，它们默认不生成交付文件。")
            suggestions.append("在任务中明确要求导出步骤（例如“导出地图到 PNG/PDF”）。")

        if "export_map" in completed_tools:
            reasons.append("export_map 可能执行了，但工具未返回输出路径。")
            suggestions.append("检查 export_map 的 output_path 和 format 是否明确设置。")

        if not reasons:
            reasons.append("工具执行成功但未上报产物路径，属于“成功无交付”场景。")
            suggestions.append("让工具返回 output_path，或在执行后补充产物收集规则。")

        lines = [
            "诊断：",
            f"- 已完成步骤: {tool_line}",
        ]
        lines.extend([f"- 可能原因: {r}" for r in reasons])
        lines.append("")
        lines.append("建议：")
        lines.extend([f"- {s}" for s in suggestions])
        return "\n".join(lines)
    
    def _get_recovery_suggestions(self, result: ExecutionResult) -> list[str]:
        """Get recovery suggestions for failed execution."""
        suggestions = []

        if not result.success:
            suggestions.append("查看详细错误信息")
            suggestions.append("重试失败的步骤")
            suggestions.append("跳过失败步骤继续")

            if self._is_output_exists_error(result.error):
                suggestions.append("直接覆盖现有输出并重试")

            # Only suggest switching to ArcGIS Pro if not already using it
            if not self.context.arcpy_available:
                if "arcpy" in (result.error or "").lower():
                    suggestions.append("切换到 ArcGIS Pro 环境执行")

        return suggestions

    def _contains_overwrite_intent(self, message: str) -> bool:
        lowered = message.lower()
        markers = [
            "覆盖", "直接覆盖", "强制覆盖", "覆盖文件", "覆盖输出", "替换现有", "replace",
            "overwrite", "force overwrite"
        ]
        return any(m in lowered for m in markers)

    def _contains_retry_intent(self, message: str) -> bool:
        lowered = message.lower().strip()
        markers = [
            "继续", "继续执行", "继续跑", "重试", "再试", "再试试",
            "换个方法", "换个方式", "试试别的",
            "重写代码", "代码重写", "重写", "重做", "重新",
            "换一种", "换种", "换一下",
            "retry", "resume", "try again", "redo",
        ]
        return any(m in lowered for m in markers)

    def _build_retry_replan_prompt(
        self,
        plan: Plan,
        result: ExecutionResult | None,
        user_feedback: str = "",
    ) -> str:
        failed_steps = plan.get_failed_steps()
        failed_lines: list[str] = []
        for step in failed_steps[:3]:
            err = (step.error or "未知错误").strip()
            failed_lines.append(f"- 步骤 {step.id}（{step.tool}）：{err[:220]}")

        if not failed_lines and result and result.error:
            failed_lines.append(f"- 上次错误：{str(result.error)[:220]}")

        failure_text = "\n".join(failed_lines) if failed_lines else "- 上次执行失败，但未记录到明确失败步骤"

        feedback_section = ""
        if user_feedback:
            feedback_section = (
                "\n用户修正意见（必须采纳）：\n"
                f"{user_feedback}\n"
            )

        return (
            "请重新规划并执行同一任务，避免重复之前失败路径。\n"
            f"任务目标：{plan.goal}\n"
            "上次失败摘要：\n"
            f"{failure_text}\n"
            f"{feedback_section}"
            "要求：\n"
            "1. 优先保留原目标，不要机械复用原失败步骤。\n"
            "2. 必要时调整工具选择或参数后再执行。\n"
            "3. 若存在可替代路径，优先选择成功率更高的方案。\n"
            "4. 用户修正意见优先级最高，必须严格遵循。"
        )

    def _extract_user_feedback(self, message: str) -> str:
        """Extract substantive user feedback from retry message.

        Strips known retry/action keywords to isolate the user's specific
        guidance about how to fix the issue.

        Examples:
        - "换个方法用KEEP_COMMON" → "用KEEP_COMMON"
        - "重写代码，用arcpy.KEEP_COMMON" → "用arcpy.KEEP_COMMON"
        - "重试" → "" (no specific feedback)
        """
        retry_keywords = [
            "继续执行", "继续跑", "继续",
            "重试", "重写代码", "代码重写", "重写", "重做", "重新",
            "再试", "再试试",
            "换个方法", "换个方式", "试试别的",
            "换一种", "换种", "换一下",
            "retry", "resume", "try again", "redo",
        ]
        feedback = message
        for kw in sorted(retry_keywords, key=len, reverse=True):
            feedback = feedback.replace(kw, "")
        feedback = feedback.strip().strip("，,。.!！；;:：")
        return feedback

    def _is_output_exists_error(self, error_text: str | None) -> bool:
        lowered = (error_text or "").lower()
        return (
            "000725" in lowered
            or "already exists" in lowered
            or "已存在" in lowered
            or "dataset" in lowered and "exists" in lowered
        )

    def _apply_overwrite_retry_policy(self, plan: Plan) -> int:
        """Inject overwrite flags and requeue failed output-conflict steps."""
        updated_steps = 0
        output_tools = {"merge_layers", "project_layers", "export_map"}

        for step in plan.steps:
            output_path = step.input.get("output_path")
            has_output_path = isinstance(output_path, str) and bool(output_path.strip())
            is_output_step = step.tool in output_tools or has_output_path

            if not is_output_step:
                continue

            if step.status == StepStatus.PENDING and step.tool in output_tools:
                if step.input.get("overwrite_output") is not True:
                    step.input["overwrite_output"] = True
                    updated_steps += 1
                continue

            if step.status == StepStatus.FAILED and self._is_output_exists_error(step.error):
                if step.input.get("overwrite_output") is not True:
                    step.input["overwrite_output"] = True
                    updated_steps += 1
                step.status = StepStatus.PENDING
                step.error = None
                step.started_at = None
                step.completed_at = None

        return updated_steps
    
    def _chunk_response(self, content: str, chunk_size: int = 50) -> Generator[str, None, None]:
        """Chunk response for streaming."""
        for i in range(0, len(content), chunk_size):
            yield content[i:i+chunk_size]

    def _preflight_plan(self, plan: Plan, task_description: str) -> None:
        """System-level plan governance before execution."""
        self._apply_scan_path_policy(plan)
        self._apply_target_srs_policy(plan, task_description)
        self._hydrate_plan_inputs_from_memory(plan)

    def _apply_scan_path_policy(self, plan: Plan) -> None:
        """Normalize all scan step paths to the best available existing directory."""
        workspace = self.config.workspace_path or Path.cwd()
        context_workspace = self.memory.get_context("workspace")
        if isinstance(context_workspace, str) and context_workspace.strip():
            workspace = Path(context_workspace)

        for step in plan.steps:
            if step.tool != "scan_layers":
                continue
            raw_path = step.input.get("path", str(workspace / "input"))
            if isinstance(raw_path, str) and raw_path.strip():
                step.input["path"] = self._resolve_best_input_folder(raw_path, workspace)

    def _resolve_best_input_folder(self, candidate: str, workspace: Path) -> str:
        """Find a usable input directory from candidate/context/common conventions."""
        def _is_output_like(path: Path) -> bool:
            """Check if path looks like an output directory (only used for auto-fallback)."""
            return any(part.lower() in {"output", "outputs"} for part in path.parts)

        raw = Path(candidate)
        # Priority 1: User-explicit path (do NOT reject output dirs — user knows what they want)
        if raw.exists() and raw.is_dir():
            return str(raw)

        memory_input = self.memory.get_context("input_folder")
        if isinstance(memory_input, str) and memory_input.strip():
            p = Path(memory_input)
            if p.exists() and p.is_dir():
                return str(p)

        common_candidates = [
            workspace / "input",
            workspace / "workspace" / "input",
            Path.cwd() / "workspace" / "input",
            Path.cwd() / "input",
        ]
        # Auto-fallback: skip output-like dirs but NOT for user-specified paths
        for p in common_candidates:
            if p.exists() and p.is_dir() and not _is_output_like(p):
                return str(p)

        # If user/planner wrote "...\\input" but actual path is "...\\workspace\\input"
        if raw.name.lower() == "input":
            alt = raw.parent / "workspace" / "input"
            if alt.exists() and alt.is_dir():
                return str(alt)

        return str(raw)

    def _apply_target_srs_policy(self, plan: Plan, task_description: str) -> None:
        target_srs = self._extract_target_srs_from_text(task_description)
        if not target_srs:
            return
        for step in plan.steps:
            if step.tool == "project_layers":
                step.input["target_srs"] = target_srs

    def _extract_target_srs_from_text(self, text: str) -> str | None:
        lowered = text.lower()
        if (
            "asia_north_albers_equal_area_conic" in lowered
            or "asia north albers equal area conic" in lowered
            or ("albers" in lowered and "105" in lowered and "25" in lowered and "47" in lowered)
        ):
            return "Asia_North_Albers_Equal_Area_Conic"
        if "web_mercator_auxiliary_sphere" in lowered or "web mercator" in lowered or "3857" in lowered:
            return "WGS_1984_Web_Mercator_Auxiliary_Sphere"
        if "cgcs2000" in lowered or "4490" in lowered:
            return "CGCS2000"
        if "wgs84" in lowered or "wgs 84" in lowered or "4326" in lowered:
            return "WGS84"
        return None

    def _collect_recent_input_files(self) -> list[dict[str, str]]:
        """Collect recent layer paths from tool history for follow-up tasks."""
        collected: list[dict[str, str]] = []

        for item in reversed(self.memory.get_tool_results("scan_layers")):
            output = item.get("output")
            layers = getattr(output, "layers", None)
            if isinstance(output, dict):
                layers = output.get("layers", layers)
            if isinstance(layers, list):
                for layer in layers:
                    if isinstance(layer, dict):
                        path = layer.get("path")
                        layer_type = layer.get("type", "Unknown")
                    else:
                        path = getattr(layer, "path", None)
                        layer_type = getattr(layer, "type", "Unknown")
                    if isinstance(path, str) and path.strip():
                        collected.append({"path": path, "type": str(layer_type or "Unknown")})
                if collected:
                    return collected

        # Fallback to last execution outputs
        if self.last_result and self.last_result.outputs:
            for out in self.last_result.outputs:
                if isinstance(out, str) and out.strip():
                    collected.append({"path": out, "type": "Unknown"})
        return collected

    def _prepare_task_for_planning(self, message: str) -> tuple[str, bool]:
        """Merge follow-up request with recent primary requirement when possible.

        Returns:
            (planning_text, used_followup_context)
        """
        current = message.strip()
        if not current:
            return message, False

        if not self._is_followup_request(current):
            return current, False

        anchor, anchor_index = self._find_recent_primary_task_with_index(current)
        if not anchor:
            return current, False

        chain_constraints = self._collect_followup_constraints(anchor_index, current)
        chain_text = ""
        if chain_constraints:
            chain_text = "\n\n已记录续接约束：\n" + "\n".join(f"- {item}" for item in chain_constraints)

        merged = (
            "请在同一任务链路中继续规划，不要重置为无关的单步技能。\n"
            f"历史任务：\n{anchor}"
            f"{chain_text}\n\n"
            f"本次补充要求：\n{current}"
        )
        return merged, True

    def _is_followup_request(self, message: str) -> bool:
        lowered = message.lower()
        markers = [
            "继续", "接着", "后续", "在此基础", "根据之前", "基于之前", "按之前",
            "继续操作", "继续处理", "继续执行", "上一步", "前面的", "follow up", "follow-up",
        ]
        return any(marker in lowered for marker in markers)

    def _find_recent_primary_task(self, current_message: str) -> str | None:
        """Find the latest actionable user requirement before current follow-up."""
        task, _ = self._find_recent_primary_task_with_index(current_message)
        return task

    def _find_recent_primary_task_with_index(self, current_message: str) -> tuple[str | None, int]:
        """Find latest actionable requirement and its turn index."""
        turns = self.memory.turns
        if not turns:
            return None, -1

        current_text = current_message.strip()
        for idx in range(len(turns) - 1, -1, -1):
            turn = turns[idx]
            if turn.role != Role.USER:
                continue
            candidate = (turn.content or "").strip()
            if not candidate or candidate == current_text:
                continue
            if self._is_followup_request(candidate):
                continue
            if self._looks_like_meta_or_feedback(candidate):
                continue
            if self._looks_like_actionable_requirement(candidate):
                return candidate, idx
        return None, -1

    def _collect_followup_constraints(self, anchor_index: int, current_message: str) -> list[str]:
        """Collect incremental user constraints after anchor requirement."""
        if anchor_index < 0:
            return []

        current_text = current_message.strip()
        constraints: list[str] = []
        for idx in range(anchor_index + 1, len(self.memory.turns)):
            turn = self.memory.turns[idx]
            if turn.role != Role.USER:
                continue
            candidate = (turn.content or "").strip()
            if not candidate or candidate == current_text:
                continue
            if self._looks_like_meta_or_feedback(candidate):
                continue
            if (
                self._is_followup_request(candidate)
                or self._looks_like_constraint_statement(candidate)
                or self._looks_like_actionable_requirement(candidate)
            ):
                constraints.append(candidate)

        return list(dict.fromkeys(constraints))

    def _looks_like_constraint_statement(self, text: str) -> bool:
        lowered = text.lower()
        markers = [
            "不要", "请勿", "不需要", "避免", "必须", "需要", "优先", "保持", "沿用", "仅", "只",
            "不要裁剪", "不要重置", "continue with", "keep", "avoid",
        ]
        return any(marker in lowered for marker in markers)

    def _looks_like_meta_or_feedback(self, text: str) -> bool:
        lowered = text.lower()
        meta_markers = [
            "规划任务", "给我计划", "怎么回事", "为什么", "bug", "问题", "识别不到", "看不到", "?", "？",
        ]
        return any(marker in lowered for marker in meta_markers)

    def _looks_like_actionable_requirement(self, text: str) -> bool:
        lowered = text.lower()
        if len(lowered) >= 120:
            return True

        action_markers = [
            "制作", "创建", "生成", "导出", "整合", "合并", "投影", "变换", "检查", "分析",
            "裁剪", "缓冲", "叠加", "map", "merge", "project", "clip", "buffer", "overlay",
        ]
        structured_markers = ["一、", "二、", "三、", "1.", "2.", "3.", "分析要求", "数据说明"]

        action_hits = sum(1 for marker in action_markers if marker in lowered)
        structured_hit = any(marker in text for marker in structured_markers)
        return structured_hit or action_hits >= 2

    def _hydrate_plan_inputs_from_memory(self, plan: Plan) -> None:
        """Fill missing/placeholder input paths from recent memory context."""
        recent_files = self._collect_recent_input_files()
        context_input_folder = self.memory.get_context("input_folder")
        preferred_input_folder = context_input_folder.strip() if isinstance(context_input_folder, str) else ""

        for step in plan.steps:
            if step.tool == "scan_layers" and preferred_input_folder:
                current = step.input.get("path")
                current_text = str(current).strip() if isinstance(current, str) else ""
                # Force explicit user-provided folder over generic defaults like "input"/"./workspace/input"
                if (not current_text) or current_text.lower() in {
                    "input",
                    "./input",
                    ".\\input",
                    "workspace/input",
                    ".\\workspace\\input",
                    "./workspace/input",
                }:
                    step.input["path"] = preferred_input_folder

        if not recent_files:
            return

        for step in plan.steps:
            if step.tool not in {"project_layers", "quality_check"}:
                continue

            current = step.input.get("input_path")
            if isinstance(current, str) and current.strip() and not self._looks_like_placeholder_path(current):
                continue

            selected = self._pick_recent_path_for_step(step, recent_files)
            if selected:
                step.input["input_path"] = selected

    def _pick_recent_path_for_step(self, step: PlanStep, recent_files: list[dict[str, str]]) -> str | None:
        hint = f"{step.description} {step.tool}".lower()
        wants_line = ("线" in hint) or ("line" in hint) or ("polyline" in hint)
        wants_polygon = ("面" in hint) or ("polygon" in hint)

        if wants_line:
            for item in recent_files:
                t = str(item.get("type", "")).lower()
                if "polyline" in t or "line" in t:
                    return item.get("path")
        if wants_polygon:
            for item in recent_files:
                t = str(item.get("type", "")).lower()
                if "polygon" in t:
                    return item.get("path")

        return recent_files[0].get("path")

    def _looks_like_placeholder_path(self, value: str) -> bool:
        text = value.strip().lower()
        if not text:
            return True
        return (
            "待确认" in value
            or "请提供输入图层路径" in value
            or text.startswith("$step_")
            or ("{step_" in text and "}" in text)
        )

    def _update_context_from_message(self, message: str) -> None:
        """Extract and persist user-provided path hints from conversation."""
        workspace = self.config.workspace_path or Path.cwd()
        windows_abs = re.findall(r"[A-Za-z]:\\[^\n\r\"']+", message)
        relative_candidates = re.findall(
            r"(?i)(?:\.?\\|\./|workspace\\|workspace/|input\\|input/|data\\|data/)[^\n\r\"']+",
            message,
        )
        quoted_paths = re.findall(r"['\"]([^'\"]*(?:/|\\)[^'\"]*)['\"]", message)

        candidates: list[str] = []
        candidates.extend([p.strip() for p in windows_abs if p.strip()])
        candidates.extend([p.strip() for p in relative_candidates if p.strip()])
        candidates.extend([p.strip() for p in quoted_paths if p.strip()])

        def normalize_path(raw: str) -> Path | None:
            try:
                text = raw.strip().strip("\"'")
                if not text:
                    return None
                p = Path(text)
                if p.is_absolute():
                    return p
                return (workspace / p).resolve()
            except Exception:
                return None

        lowered = message.lower()
        prefer_input = any(k in lowered for k in ["输入", "数据", "扫描", "scan", "input", "data"])
        explicit_output = any(k in lowered for k in ["输出", "导出", "output", "export"])

        normalized_candidates: list[Path] = []
        for raw in candidates:
            p = normalize_path(raw)
            if p is not None:
                normalized_candidates.append(p)

        # Pick the latest input-like path, avoid output-path pollution.
        for p in reversed(normalized_candidates):
            parts_lower = [part.lower() for part in p.parts]
            is_input_like = any(part in {"input", "data", "workspace"} for part in parts_lower)
            is_output_like = any(part in {"output", "outputs"} for part in parts_lower)

            # If user explicitly says "output"/"导出"/"输出", store as both output_folder and input_folder
            if is_output_like and (explicit_output or not prefer_input):
                self.memory.set_context("output_folder", str(p))
                self.memory.set_context("input_folder", str(p))
                break

            # Skip output paths in auto-detect mode
            if is_output_like and not prefer_input:
                continue

            if is_input_like or prefer_input or not explicit_output:
                self.memory.set_context("input_folder", str(p))
                p_str = str(p).lower()
                marker = "\\workspace\\input"
                idx = p_str.find(marker)
                if idx > 0:
                    workspace_root = str(p)[:idx + len("\\workspace")]
                    self.memory.set_context("workspace", workspace_root)
                break

        target_srs = self._extract_target_srs_from_text(message)
        if target_srs:
            self.memory.set_context("target_srs", target_srs)
        if self.current_plan and not self.current_plan.is_complete:
            self._apply_scan_path_policy(self.current_plan)

    def _extract_structured_memory_from_message(self, message: str) -> None:
        """Capture durable structured facts from user input."""
        msg = message.strip()
        if not msg:
            return

        if self._looks_like_actionable_requirement(msg):
            self.memory.add_structured_memory(
                "task_requirement",
                msg[:500],
                metadata={"source": "user_message"},
                importance=3,
            )

        target_srs = self._extract_target_srs_from_text(msg)
        if target_srs:
            self.memory.add_structured_memory(
                "projection_constraint",
                f"目标坐标系: {target_srs}",
                metadata={"target_srs": target_srs},
                importance=4,
            )

        candidate_paths = re.findall(r"[A-Za-z]:\\[^\n\r\"']+", msg)
        if candidate_paths:
            self.memory.add_structured_memory(
                "data_path",
                candidate_paths[-1].strip(),
                metadata={"source": "user_message"},
                importance=4,
            )

    def _record_execution_reflection(
        self,
        plan: Plan,
        result: ExecutionResult,
        recovery_notes: list[str] | None = None,
    ) -> None:
        """Record structured outcomes and reflection notes from execution."""
        recovery_notes = recovery_notes or []
        related_tools = [s.tool for s in (result.trace.steps if result.trace else []) if getattr(s, "tool", None)]
        related_tools = list(dict.fromkeys(related_tools))

        self._update_workflow_state(
            status="completed" if result.success else "failed",
            current_plan_id=plan.id,
            current_goal=plan.goal,
            last_result_success=result.success,
            last_error=result.error or "",
            metadata={
                "outputs_count": len(result.outputs),
                "recovery_notes": recovery_notes,
            },
            event="execution_completed" if result.success else "execution_failed",
            event_payload={
                "plan_id": plan.id,
                "error": (result.error or "")[:200],
                "tools": related_tools,
            },
        )

        if result.success:
            self.memory.add_structured_memory(
                "workflow_outcome",
                f"任务成功: {plan.goal[:160]}",
                metadata={
                    "outputs": result.outputs,
                    "tools": related_tools,
                    "recovery_notes": recovery_notes,
                },
                importance=2,
            )
            if recovery_notes:
                self.memory.add_reflection(
                    trigger="success",
                    issue="任务通过自动恢复后成功完成",
                    lesson="恢复策略有效，但应优先在规划阶段规避高风险输入。",
                    action_hint="当输入路径/参数不稳定时，先执行扫描与质量检查。",
                    related_tools=related_tools,
                    confidence=0.65,
                )
            self._persist_runtime_state()
            return

        error_text = (result.error or "未知错误")[:300]
        action_hint = "失败后先检查输入路径、坐标系和依赖步骤输出。"
        if "000732" in error_text or "不存在" in error_text:
            action_hint = "优先验证输入路径有效性；目录输入需先展开为具体图层文件。"
        elif "spatialreference" in error_text.lower() or "createfromfile" in error_text.lower():
            action_hint = "统一将目标坐标系标准化为别名/WKID/WKT，再执行投影步骤。"
        elif "validation" in error_text.lower() or "参数" in error_text:
            action_hint = "执行前先补齐关键参数并做 schema 校验。"

        self.memory.add_structured_memory(
            "error_pattern",
            error_text,
            metadata={"plan_goal": plan.goal[:160], "tools": related_tools},
            importance=4,
        )
        self.memory.add_reflection(
            trigger="failure",
            issue=f"执行失败: {error_text}",
            lesson="规划阶段应注入更多结构化上下文与防错约束。",
            action_hint=action_hint,
            related_tools=related_tools,
            confidence=0.8,
        )
        self._persist_runtime_state()

    def _persist_runtime_state(self) -> None:
        """Persist current runtime state into memory/state manager."""
        runtime_state = {
            "current_plan": self._serialize_plan_state(self.current_plan),
            "last_result": self._serialize_execution_result_state(self.last_result),
        }
        self.memory.set_context("runtime_state", runtime_state)
        if self.state_manager:
            self.state_manager.upsert_state(self.thread_id, runtime_state=runtime_state)

    def _restore_runtime_state(self) -> None:
        """Restore runtime state from memory or state manager."""
        payload = self.memory.get_context("runtime_state")
        if not isinstance(payload, dict) and self.state_manager:
            state = self.state_manager.get_state(self.thread_id)
            if state and isinstance(state.metadata, dict):
                payload = state.metadata.get("runtime_state")

        if not isinstance(payload, dict):
            return

        restored_plan = self._deserialize_plan_state(payload.get("current_plan"))
        if restored_plan is not None:
            self.current_plan = restored_plan

        restored_result = self._deserialize_execution_result_state(payload.get("last_result"), self.current_plan)
        if restored_result is not None:
            self.last_result = restored_result

    def _serialize_plan_state(self, plan: Plan | None) -> dict[str, Any] | None:
        if plan is None:
            return None

        return {
            "id": plan.id,
            "goal": plan.goal,
            "expected_outputs": self._json_safe(plan.expected_outputs),
            "metadata": self._json_safe(plan.metadata),
            "created_at": plan.created_at.isoformat(),
            "steps": [
                {
                    "id": step.id,
                    "tool": step.tool,
                    "description": step.description,
                    "input": self._json_safe(step.input),
                    "depends_on": self._json_safe(step.depends_on),
                    "status": step.status.value,
                    "result": self._json_safe(step.result),
                    "error": step.error,
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                }
                for step in plan.steps
            ],
        }

    def _deserialize_plan_state(self, payload: Any) -> Plan | None:
        if not isinstance(payload, dict):
            return None

        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list):
            raw_steps = []

        steps: list[PlanStep] = []
        for raw in raw_steps:
            if not isinstance(raw, dict):
                continue
            step = PlanStep(
                id=str(raw.get("id") or ""),
                tool=str(raw.get("tool") or ""),
                description=str(raw.get("description") or ""),
                input=dict(raw.get("input") or {}),
                depends_on=[str(x) for x in list(raw.get("depends_on") or [])],
            )
            status_raw = str(raw.get("status") or StepStatus.PENDING.value)
            if status_raw in StepStatus._value2member_map_:
                step.status = StepStatus(status_raw)
            step.result = raw.get("result")
            error_raw = raw.get("error")
            step.error = str(error_raw) if error_raw else None
            step.started_at = self._parse_iso_datetime(raw.get("started_at"))
            step.completed_at = self._parse_iso_datetime(raw.get("completed_at"))
            steps.append(step)

        plan = Plan(
            id=str(payload.get("id") or f"plan_restored_{int(time.time())}"),
            goal=str(payload.get("goal") or ""),
            steps=steps,
            expected_outputs=[str(x) for x in list(payload.get("expected_outputs") or [])],
            metadata=dict(payload.get("metadata") or {}),
        )
        created_at = self._parse_iso_datetime(payload.get("created_at"))
        if created_at:
            plan.created_at = created_at
        return plan

    def _serialize_execution_result_state(self, result: ExecutionResult | None) -> dict[str, Any] | None:
        if result is None:
            return None

        trace_payload = {
            "plan_id": result.trace.plan_id,
            "started_at": result.trace.started_at.isoformat(),
            "completed_at": result.trace.completed_at.isoformat() if result.trace.completed_at else None,
            "mode": result.trace.mode.value,
            "outputs": self._json_safe(result.trace.outputs),
            "success": result.trace.success,
            "error": result.trace.error,
            "steps": [
                {
                    "step_id": step.step_id,
                    "tool": step.tool,
                    "started_at": step.started_at.isoformat(),
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                    "status": step.status,
                    "input": self._json_safe(step.input),
                    "output": self._json_safe(step.output),
                    "error": step.error,
                    "duration_ms": step.duration_ms,
                }
                for step in result.trace.steps
            ],
        }

        return {
            "success": result.success,
            "outputs": self._json_safe(result.outputs),
            "error": result.error,
            "plan": self._serialize_plan_state(result.plan),
            "trace": trace_payload,
        }

    def _deserialize_execution_result_state(
        self,
        payload: Any,
        fallback_plan: Plan | None,
    ) -> ExecutionResult | None:
        if not isinstance(payload, dict):
            return None

        plan = self._deserialize_plan_state(payload.get("plan")) or fallback_plan
        if plan is None:
            return None

        trace_raw = payload.get("trace")
        if not isinstance(trace_raw, dict):
            trace_raw = {}

        mode_raw = str(trace_raw.get("mode") or ExecutionMode.DRY_RUN.value)
        mode = ExecutionMode.DRY_RUN
        if mode_raw in ExecutionMode._value2member_map_:
            mode = ExecutionMode(mode_raw)

        trace = ExecutionTrace(
            plan_id=str(trace_raw.get("plan_id") or plan.id),
            mode=mode,
            started_at=self._parse_iso_datetime(trace_raw.get("started_at")) or datetime.now(timezone.utc),
        )
        trace.completed_at = self._parse_iso_datetime(trace_raw.get("completed_at"))
        trace.outputs = [str(x) for x in list(trace_raw.get("outputs") or [])]
        trace.success = bool(trace_raw.get("success"))
        trace_error = trace_raw.get("error")
        trace.error = str(trace_error) if trace_error else None

        for raw in list(trace_raw.get("steps") or []):
            if not isinstance(raw, dict):
                continue
            step_trace = StepTrace(
                step_id=str(raw.get("step_id") or ""),
                tool=str(raw.get("tool") or ""),
                started_at=self._parse_iso_datetime(raw.get("started_at")) or datetime.now(timezone.utc),
                input=dict(raw.get("input") or {}),
            )
            step_trace.completed_at = self._parse_iso_datetime(raw.get("completed_at"))
            step_trace.status = str(raw.get("status") or "completed")
            step_trace.output = raw.get("output")
            step_error = raw.get("error")
            step_trace.error = str(step_error) if step_error else None
            try:
                step_trace.duration_ms = int(raw.get("duration_ms") or 0)
            except (TypeError, ValueError):
                step_trace.duration_ms = 0
            trace.steps.append(step_trace)

        outputs = [str(x) for x in list(payload.get("outputs") or trace.outputs)]
        result_error = payload.get("error")
        return ExecutionResult(
            success=bool(payload.get("success")),
            plan=plan,
            trace=trace,
            outputs=outputs,
            error=str(result_error) if result_error else None,
        )

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(v) for v in value]
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return self._json_safe(to_dict())
            except Exception:
                return str(value)
        return str(value)

    def _parse_iso_datetime(self, raw: Any) -> datetime | None:
        if not isinstance(raw, str) or not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    def _update_workflow_state(self, **kwargs: Any) -> None:
        """Update persistent workflow state if manager is enabled."""
        if not self.state_manager:
            return

        event = kwargs.pop("event", None)
        event_payload = kwargs.pop("event_payload", None)
        self.state_manager.upsert_state(self.thread_id, **kwargs)
        if isinstance(event, str) and event:
            self.state_manager.record_event(self.thread_id, event, event_payload or {})
    
    # === Callbacks ===
    
    def _on_step_start(self, step: PlanStep) -> None:
        """Called when a step starts."""
        pass
    
    def _on_step_complete(self, step: PlanStep, result: Any) -> None:
        """Called when a step completes."""
        pass
    
    def _on_progress(self, message: str, percent: float) -> None:
        """Called to report progress."""
        pass
    
    # === Session Management ===
    
    def reset(self) -> None:
        """Reset agent state."""
        self.memory.clear()
        self.current_plan = None
        self.last_result = None
        self._persist_runtime_state()
        self._update_workflow_state(
            status="reset",
            current_plan_id="",
            current_goal="",
            last_result_success=None,
            last_error="",
            metadata={},
            event="session_reset",
            event_payload={},
        )
    
    def save_session(self) -> None:
        """Save current session."""
        self._persist_runtime_state()
        self.memory.save()
    
    def load_session(self, session_id: str) -> None:
        """Load a previous session."""
        self.config.session_id = session_id
        self.memory.session_id = session_id
        self.memory.load()
        self.thread_id = f"thread_{session_id}"
        self._update_workflow_state(
            session_id=session_id,
            status="session_loaded",
            event="session_loaded",
            event_payload={"session_id": session_id},
        )
        self.current_plan = None
        self.last_result = None
        self._restore_runtime_state()
    
    def get_session_summary(self) -> dict:
        """Get summary of current session."""
        execution_adapter_status = None
        if hasattr(self.executor, "status"):
            try:
                execution_adapter_status = self.executor.status()  # type: ignore[assignment]
            except Exception:
                execution_adapter_status = None
        return {
            "session_id": self.config.session_id,
            "memory": self.memory.summary(),
            "current_plan": self.current_plan.summary() if self.current_plan else None,
            "last_result": self.last_result.to_dict() if self.last_result else None,
            "execution_adapter": execution_adapter_status,
            "state_manager": self.state_manager.get_summary(self.thread_id) if self.state_manager else None,
        }

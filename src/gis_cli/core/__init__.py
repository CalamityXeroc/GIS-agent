"""Core module for GIS CLI.

This module provides the foundational architecture inspired by Claude Code:
- Tool: Base class for all tools with schema validation and permissions
- Task: Task state machine with lifecycle management  
- Registry: Tool registration and discovery
"""

from .tool import (
    Tool,
    ToolBuilder,
    ToolCategory,
    ToolContext,
    ToolResult,
    ValidationResult,
    PermissionResult,
    build_tool,
)

from .task import (
    Task,
    TaskContext,
    TaskManager,
    TaskNode,
    TaskStatus,
    TaskType,
    TaskStateBase,
    is_terminal_status,
    generate_task_id,
)

from .registry import (
    ToolRegistry,
    assemble_tool_pool,
    get_tools_json_schema,
    register_tool,
    tool,
)

from .context import (
    SystemContext,
    UserContext,
    WorkspaceContext,
    ExecutionContext,
)

__all__ = [
    # Tool
    "Tool",
    "ToolBuilder", 
    "ToolCategory",
    "ToolContext",
    "ToolResult",
    "ValidationResult",
    "PermissionResult",
    "build_tool",
    # Task
    "Task",
    "TaskContext",
    "TaskManager",
    "TaskNode",
    "TaskStatus",
    "TaskType",
    "TaskStateBase",
    "is_terminal_status",
    "generate_task_id",
    # Registry
    "ToolRegistry",
    "assemble_tool_pool",
    "get_tools_json_schema",
    "register_tool",
    "tool",
    # Context
    "SystemContext",
    "UserContext",
    "WorkspaceContext",
    "ExecutionContext",
]

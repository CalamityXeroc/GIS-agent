"""Base Tool class for GIS CLI.

Inspired by Claude Code's Tool architecture, this provides a standardized
interface for all GIS processing tools with:
- Input/output schema validation (using Pydantic)
- Permission checking
- Execution with context
- Result rendering
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar, Callable
from pydantic import BaseModel, ValidationError


class ToolCategory(str, Enum):
    """Tool categories for organization and filtering."""
    DATA_INTEGRATION = "data_integration"
    SPATIAL_ANALYSIS = "spatial_analysis"
    CARTOGRAPHY = "cartography"
    QUALITY_CHECK = "quality_check"
    FILE_OPERATION = "file_operation"
    BATCH_PROCESSING = "batch_processing"


class ValidationResult:
    """Result of input validation."""
    def __init__(self, valid: bool, message: str = "", error_code: int = 0):
        self.valid = valid
        self.message = message
        self.error_code = error_code
    
    @classmethod
    def success(cls) -> "ValidationResult":
        return cls(valid=True)
    
    @classmethod
    def failure(cls, message: str, error_code: int = 1) -> "ValidationResult":
        return cls(valid=False, message=message, error_code=error_code)

    @classmethod
    def warning(cls, message: str, error_code: int = 0) -> "ValidationResult":
        """Return a non-blocking validation warning."""
        return cls(valid=True, message=message, error_code=error_code)


class PermissionResult:
    """Result of permission check."""
    def __init__(self, allowed: bool, reason: str = "", requires_confirmation: bool = False):
        self.allowed = allowed
        self.reason = reason
        self.requires_confirmation = requires_confirmation
    
    @classmethod
    def allow(cls) -> "PermissionResult":
        return cls(allowed=True)
    
    @classmethod
    def deny(cls, reason: str) -> "PermissionResult":
        return cls(allowed=False, reason=reason)
    
    @classmethod
    def confirm(cls, reason: str) -> "PermissionResult":
        return cls(allowed=True, reason=reason, requires_confirmation=True)


InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT")


@dataclass
class ToolContext:
    """Context passed to tool execution."""
    abort_signal: bool = False
    dry_run: bool = False
    working_directory: str = "."
    arcpy_available: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get_state(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)
    
    def set_state(self, key: str, value: Any) -> None:
        self.metadata[key] = value


@dataclass
class ToolResult(Generic[OutputT]):
    """Result of tool execution."""
    success: bool
    data: OutputT | None = None
    error: str | None = None
    error_code: str | None = None
    outputs: list[str] = field(default_factory=list)
    duration_ms: int = 0
    
    @classmethod
    def ok(cls, data: OutputT, outputs: list[str] | None = None, duration_ms: int = 0) -> "ToolResult[OutputT]":
        return cls(success=True, data=data, outputs=outputs or [], duration_ms=duration_ms)
    
    @classmethod
    def fail(cls, error: str, error_code: str = "execution_error") -> "ToolResult[OutputT]":
        return cls(success=False, error=error, error_code=error_code)


class Tool(ABC, Generic[InputT, OutputT]):
    """Base class for all GIS tools.
    
    Inspired by Claude Code's buildTool pattern, each tool defines:
    - name: unique identifier
    - description: what the tool does
    - category: for organization
    - input_schema: Pydantic model for input validation
    - validate_input: custom validation beyond schema
    - check_permissions: permission requirements
    - call: actual execution logic
    - render methods: for UI display
    """
    
    # Class attributes to be overridden
    name: str = "base_tool"
    description: str = "Base tool"
    category: ToolCategory = ToolCategory.FILE_OPERATION
    search_hint: str = ""
    
    # Schema types (override in subclass)
    input_model: type[InputT] | None = None
    
    def __init__(self):
        self._registered = False
    
    @property
    def user_facing_name(self) -> str:
        """Human-readable name for display."""
        return self.name.replace("_", " ").title()
    
    def get_activity_description(self, input_data: InputT) -> str:
        """Short description of current activity for status display."""
        return f"Running {self.user_facing_name}"
    
    def get_tool_use_summary(self, input_data: InputT) -> str:
        """Summary of tool use for logging."""
        return f"{self.name}"
    
    # === Schema and Validation ===
    
    def parse_input(self, raw_input: dict[str, Any]) -> InputT:
        """Parse and validate input against schema."""
        if self.input_model is None:
            raise NotImplementedError(f"Tool {self.name} must define input_model")
        try:
            return self.input_model.model_validate(raw_input)
        except ValidationError as e:
            raise ValueError(f"Invalid input for {self.name}: {e}")
    
    def validate_input(self, input_data: InputT) -> ValidationResult:
        """Custom validation beyond schema validation."""
        return ValidationResult.success()
    
    # === Permissions ===
    
    def check_permissions(self, input_data: InputT, context: ToolContext) -> PermissionResult:
        """Check if tool execution is allowed."""
        return PermissionResult.allow()
    
    def is_read_only(self) -> bool:
        """Whether this tool only reads data (no modifications)."""
        return False
    
    def is_concurrency_safe(self) -> bool:
        """Whether this tool can run concurrently with others."""
        return self.is_read_only()
    
    # === Execution ===
    
    @abstractmethod
    def call(self, input_data: InputT, context: ToolContext) -> ToolResult[OutputT]:
        """Execute the tool. Must be implemented by each tool."""
        pass
    
    def execute(self, raw_input: dict[str, Any], context: ToolContext) -> ToolResult[OutputT]:
        """Full execution pipeline: parse -> validate -> check permissions -> call."""
        import time
        start_time = time.time()
        
        # Parse input
        try:
            input_data = self.parse_input(raw_input)
        except ValueError as e:
            return ToolResult.fail(str(e), "input_parse_error")
        
        # Validate
        validation = self.validate_input(input_data)
        if not validation.valid:
            return ToolResult.fail(validation.message, f"validation_error_{validation.error_code}")
        
        # Check permissions
        permission = self.check_permissions(input_data, context)
        if not permission.allowed:
            return ToolResult.fail(permission.reason, "permission_denied")
        
        # Execute
        try:
            result = self.call(input_data, context)
            result.duration_ms = int((time.time() - start_time) * 1000)
            return result
        except Exception as e:
            return ToolResult.fail(str(e), "execution_exception")
    
    # === Rendering ===
    
    def render_tool_use_message(self, input_data: InputT) -> str:
        """Render message when tool is being used."""
        return f"Using {self.user_facing_name}..."
    
    def render_tool_result_message(self, result: ToolResult[OutputT]) -> str:
        """Render message after tool completes."""
        if result.success:
            return f"{self.user_facing_name} completed successfully"
        return f"{self.user_facing_name} failed: {result.error}"
    
    def render_tool_use_error_message(self, error: str) -> str:
        """Render error message."""
        return f"Error in {self.user_facing_name}: {error}"
    
    def to_json_schema(self) -> dict[str, Any]:
        """Export tool definition as JSON schema (for LLM tool use)."""
        schema: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }
        if self.input_model:
            schema["input_schema"] = self.input_model.model_json_schema()
        return schema


class ToolBuilder:
    """Builder for creating tools with fluent API."""
    
    def __init__(self, name: str):
        self._name = name
        self._description = ""
        self._category = ToolCategory.FILE_OPERATION
        self._search_hint = ""
        self._input_model: type[BaseModel] | None = None
        self._call_fn: Callable[..., ToolResult] | None = None
        self._validate_fn: Callable[..., ValidationResult] | None = None
        self._permission_fn: Callable[..., PermissionResult] | None = None
        self._is_read_only = False
    
    def with_description(self, desc: str) -> "ToolBuilder":
        self._description = desc
        return self
    
    def with_category(self, cat: ToolCategory) -> "ToolBuilder":
        self._category = cat
        return self
    
    def with_search_hint(self, hint: str) -> "ToolBuilder":
        self._search_hint = hint
        return self
    
    def with_input_schema(self, model: type[BaseModel]) -> "ToolBuilder":
        self._input_model = model
        return self
    
    def with_call(self, fn: Callable[..., ToolResult]) -> "ToolBuilder":
        self._call_fn = fn
        return self
    
    def with_validate(self, fn: Callable[..., ValidationResult]) -> "ToolBuilder":
        self._validate_fn = fn
        return self
    
    def with_permissions(self, fn: Callable[..., PermissionResult]) -> "ToolBuilder":
        self._permission_fn = fn
        return self
    
    def as_read_only(self, is_read_only: bool = True) -> "ToolBuilder":
        self._is_read_only = is_read_only
        return self
    
    def build(self) -> type[Tool]:
        """Build and return the Tool class."""
        builder = self
        
        class BuiltTool(Tool):
            name = builder._name
            description = builder._description
            category = builder._category
            search_hint = builder._search_hint
            input_model = builder._input_model
            
            def is_read_only(self) -> bool:
                return builder._is_read_only
            
            def validate_input(self, input_data) -> ValidationResult:
                if builder._validate_fn:
                    return builder._validate_fn(input_data)
                return ValidationResult.success()
            
            def check_permissions(self, input_data, context: ToolContext) -> PermissionResult:
                if builder._permission_fn:
                    return builder._permission_fn(input_data, context)
                return PermissionResult.allow()
            
            def call(self, input_data, context: ToolContext) -> ToolResult:
                if builder._call_fn:
                    return builder._call_fn(input_data, context)
                return ToolResult.fail("Tool not implemented", "not_implemented")
        
        return BuiltTool


def build_tool(name: str) -> ToolBuilder:
    """Convenience function to create a ToolBuilder."""
    return ToolBuilder(name)

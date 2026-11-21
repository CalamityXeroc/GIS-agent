"""Base skill infrastructure."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar
from pydantic import BaseModel


T = TypeVar("T")


@dataclass
class SkillContext:
    """Execution context for skills."""
    
    workspace: str = ""
    output_dir: str = ""
    arcpy_available: bool = False
    dry_run: bool = False
    verbose: bool = False
    
    # Workflow state
    current_step: int = 0
    total_steps: int = 0
    
    # Tool results from previous steps
    step_results: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult(Generic[T]):
    """Result from skill execution."""
    
    success: bool
    data: T | None = None
    error: str | None = None
    error_code: str | None = None
    outputs: list[str] = field(default_factory=list)
    steps_completed: int = 0
    steps_total: int = 0
    
    @classmethod
    def ok(
        cls,
        data: T,
        outputs: list[str] | None = None,
        steps_completed: int = 0,
        steps_total: int = 0
    ) -> "SkillResult[T]":
        return cls(
            success=True,
            data=data,
            outputs=outputs or [],
            steps_completed=steps_completed,
            steps_total=steps_total
        )
    
    @classmethod
    def fail(
        cls,
        error: str,
        error_code: str | None = None,
        steps_completed: int = 0,
        steps_total: int = 0
    ) -> "SkillResult[T]":
        return cls(
            success=False,
            error=error,
            error_code=error_code,
            steps_completed=steps_completed,
            steps_total=steps_total
        )


class Skill(abc.ABC):
    """Base class for skills.
    
    Skills combine multiple tools to achieve complex workflows.
    They are higher-level than individual tools.
    """
    
    # Skill metadata
    name: str = ""
    description: str = ""
    tags: list[str] = []
    
    # Workflow steps
    steps: list[str] = []
    
    @abc.abstractmethod
    def execute(
        self,
        inputs: dict[str, Any],
        context: SkillContext
    ) -> SkillResult:
        """Execute the skill workflow."""
        pass
    
    @abc.abstractmethod
    def validate_inputs(self, inputs: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate skill inputs before execution."""
        pass
    
    def get_required_inputs(self) -> list[str]:
        """Return list of required input keys."""
        return []
    
    def get_optional_inputs(self) -> dict[str, Any]:
        """Return dict of optional inputs with defaults."""
        return {}
    
    def render_progress(self, context: SkillContext) -> str:
        """Render progress message."""
        if context.total_steps > 0:
            pct = int((context.current_step / context.total_steps) * 100)
            step_name = self.steps[context.current_step - 1] if context.current_step > 0 else ""
            return f"[{pct}%] Step {context.current_step}/{context.total_steps}: {step_name}"
        return ""
    
    def render_result(self, result: SkillResult) -> str:
        """Render final result message."""
        if result.success:
            return f"✅ {self.name} completed ({result.steps_completed}/{result.steps_total} steps)"
        else:
            return f"❌ {self.name} failed: {result.error}"


class SkillRegistry:
    """Registry for skill discovery and lookup."""
    
    _instance: "SkillRegistry | None" = None
    _skills: dict[str, type[Skill]] = {}
    
    @classmethod
    def instance(cls) -> "SkillRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def register(cls, skill_class: type[Skill]) -> type[Skill]:
        """Register a skill class."""
        registry = cls.instance()
        registry._skills[skill_class.name] = skill_class
        return skill_class
    
    @classmethod
    def get(cls, name: str) -> type[Skill] | None:
        """Get skill by name."""
        registry = cls.instance()
        return registry._skills.get(name)
    
    @classmethod
    def list_skills(cls) -> list[type[Skill]]:
        """List all registered skills."""
        registry = cls.instance()
        return list(registry._skills.values())
    
    @classmethod
    def search(cls, query: str) -> list[type[Skill]]:
        """Search skills by name, description, or tags."""
        registry = cls.instance()
        query_lower = query.lower()
        results = []
        
        for skill_cls in registry._skills.values():
            # Search in name
            if query_lower in skill_cls.name.lower():
                results.append(skill_cls)
                continue
            
            # Search in description
            if query_lower in skill_cls.description.lower():
                results.append(skill_cls)
                continue
            
            # Search in tags
            for tag in skill_cls.tags:
                if query_lower in tag.lower():
                    results.append(skill_cls)
                    break
        
        return results


def register_skill(cls: type[Skill]) -> type[Skill]:
    """Decorator to register a skill."""
    return SkillRegistry.register(cls)

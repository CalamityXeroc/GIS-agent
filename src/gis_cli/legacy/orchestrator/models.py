from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    REVIEW = "review"
    COMPLETED = "completed"
    FAILED = "failed"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PlanStep:
    order: int
    title: str
    status: str = "pending"


@dataclass
class TaskPlan:
    intent: str
    confidence: float
    planner_mode: str = "rule"
    missing_parameters: list[str] = field(default_factory=list)
    clarifying_questions: list[str] = field(default_factory=list)
    steps: list[PlanStep] = field(default_factory=list)


@dataclass
class TaskRecord:
    task_id: str
    prompt: str
    status: TaskStatus
    risk: RiskLevel
    created_at: str
    updated_at: str
    plan: TaskPlan | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    task_id: str
    success: bool
    dry_run: bool
    message: str
    outputs: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

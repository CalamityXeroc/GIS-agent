"""Task state machine for GIS CLI.

Inspired by Claude Code's Task.ts, this provides:
- Task types and status transitions
- Task lifecycle management
- Task context for execution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from uuid import uuid4


class TaskType(str, Enum):
    """Types of tasks that can be executed."""
    DATA_INTEGRATION = "data_integration"
    SPATIAL_ANALYSIS = "spatial_analysis"
    CARTOGRAPHY = "cartography"
    BATCH_PROCESSING = "batch_processing"
    QUALITY_CHECK = "quality_check"
    WORKFLOW = "workflow"  # Multi-step workflow


class TaskStatus(str, Enum):
    """Task lifecycle states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


def is_terminal_status(status: TaskStatus) -> bool:
    """Check if task is in a terminal state (won't transition further)."""
    return status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED)


# Task ID prefixes for different types
TASK_ID_PREFIXES = {
    TaskType.DATA_INTEGRATION: "d",
    TaskType.SPATIAL_ANALYSIS: "s",
    TaskType.CARTOGRAPHY: "c",
    TaskType.BATCH_PROCESSING: "b",
    TaskType.QUALITY_CHECK: "q",
    TaskType.WORKFLOW: "w",
}


def generate_task_id(task_type: TaskType) -> str:
    """Generate a unique task ID with type prefix."""
    prefix = TASK_ID_PREFIXES.get(task_type, "x")
    # Use first 8 chars of UUID for uniqueness
    unique_part = uuid4().hex[:8]
    return f"{prefix}{unique_part}"


@dataclass
class TaskStateBase:
    """Base fields shared by all task states."""
    id: str
    type: TaskType
    status: TaskStatus
    description: str
    start_time: datetime
    end_time: datetime | None = None
    total_paused_ms: int = 0
    output_file: str | None = None
    notified: bool = False
    
    @property
    def duration_ms(self) -> int:
        """Calculate task duration in milliseconds."""
        end = self.end_time or datetime.now(timezone.utc)
        delta = end - self.start_time
        return int(delta.total_seconds() * 1000) - self.total_paused_ms


@dataclass
class TaskNode:
    """A single step/node in a task workflow."""
    name: str
    status: str = "pending"  # pending, running, executed, skipped, failed
    reason: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    outputs: list[str] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> int:
        if not self.start_time:
            return 0
        end = self.end_time or datetime.now(timezone.utc)
        return int((end - self.start_time).total_seconds() * 1000)


@dataclass
class TaskContext:
    """Context for task execution."""
    task_id: str
    abort_requested: bool = False
    dry_run: bool = False
    working_directory: str = "."
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Callbacks
    on_progress: Callable[[str, float], None] | None = None
    on_node_start: Callable[[str], None] | None = None
    on_node_complete: Callable[[str, bool], None] | None = None
    
    def request_abort(self) -> None:
        """Request task abortion."""
        self.abort_requested = True
    
    def report_progress(self, message: str, percent: float = 0.0) -> None:
        """Report progress update."""
        if self.on_progress:
            self.on_progress(message, percent)


@dataclass
class Task:
    """A GIS task with full lifecycle management."""
    id: str
    type: TaskType
    description: str
    prompt: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    # Execution details
    nodes: list[TaskNode] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    error: str | None = None
    error_code: str | None = None
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        prompt: str,
        description: str,
        task_type: TaskType = TaskType.WORKFLOW,
        **kwargs
    ) -> "Task":
        """Create a new task."""
        task_id = generate_task_id(task_type)
        return cls(
            id=task_id,
            type=task_type,
            description=description,
            prompt=prompt,
            **kwargs
        )
    
    # === State Transitions ===
    
    def start(self) -> None:
        """Transition to running state."""
        if self.status != TaskStatus.PENDING:
            raise ValueError(f"Cannot start task in {self.status} state")
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
        self.updated_at = self.started_at
    
    def complete(self, outputs: list[str] | None = None) -> None:
        """Transition to completed state."""
        if self.status != TaskStatus.RUNNING:
            raise ValueError(f"Cannot complete task in {self.status} state")
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = self.completed_at
        if outputs:
            self.outputs = outputs
    
    def fail(self, error: str, error_code: str = "execution_error") -> None:
        """Transition to failed state."""
        if self.status != TaskStatus.RUNNING:
            raise ValueError(f"Cannot fail task in {self.status} state")
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = self.completed_at
        self.error = error
        self.error_code = error_code
    
    def kill(self) -> None:
        """Force kill the task."""
        if is_terminal_status(self.status):
            return  # Already done
        self.status = TaskStatus.KILLED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = self.completed_at
    
    # === Node Management ===
    
    def add_node(self, name: str) -> TaskNode:
        """Add a new node to the task."""
        node = TaskNode(name=name)
        self.nodes.append(node)
        return node
    
    def get_node(self, name: str) -> TaskNode | None:
        """Get a node by name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None
    
    def mark_node(self, name: str, status: str, reason: str | None = None) -> None:
        """Update node status."""
        node = self.get_node(name)
        if node:
            node.status = status
            node.reason = reason
            if status == "running" and not node.start_time:
                node.start_time = datetime.now(timezone.utc)
            elif status in ("executed", "failed", "skipped"):
                node.end_time = datetime.now(timezone.utc)
    
    def get_failed_nodes(self) -> list[TaskNode]:
        """Get all failed nodes."""
        return [n for n in self.nodes if n.status == "failed"]
    
    def get_pending_nodes(self) -> list[TaskNode]:
        """Get all pending nodes."""
        return [n for n in self.nodes if n.status == "pending"]
    
    # === Serialization ===
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "prompt": self.prompt,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "nodes": [
                {
                    "name": n.name,
                    "status": n.status,
                    "reason": n.reason,
                    "outputs": n.outputs,
                    "duration_ms": n.duration_ms,
                }
                for n in self.nodes
            ],
            "outputs": self.outputs,
            "error": self.error,
            "error_code": self.error_code,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """Create from dictionary."""
        task = cls(
            id=data["id"],
            type=TaskType(data["type"]),
            description=data["description"],
            prompt=data["prompt"],
            status=TaskStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            outputs=data.get("outputs", []),
            error=data.get("error"),
            error_code=data.get("error_code"),
            metadata=data.get("metadata", {}),
        )
        if data.get("started_at"):
            task.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            task.completed_at = datetime.fromisoformat(data["completed_at"])
        
        for node_data in data.get("nodes", []):
            node = task.add_node(node_data["name"])
            node.status = node_data["status"]
            node.reason = node_data.get("reason")
            node.outputs = node_data.get("outputs", [])
        
        return task


class TaskManager:
    """Manages task lifecycle and persistence."""
    
    def __init__(self, storage_root: str = ".gis-cli"):
        self.storage_root = storage_root
        self._tasks: dict[str, Task] = {}
    
    def create_task(
        self,
        prompt: str,
        description: str,
        task_type: TaskType = TaskType.WORKFLOW,
    ) -> Task:
        """Create and register a new task."""
        task = Task.create(prompt, description, task_type)
        self._tasks[task.id] = task
        return task
    
    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)
    
    def list_tasks(
        self,
        status: TaskStatus | None = None,
        task_type: TaskType | None = None,
    ) -> list[Task]:
        """List tasks with optional filtering."""
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        if task_type:
            tasks = [t for t in tasks if t.type == task_type]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)
    
    def kill_task(self, task_id: str) -> bool:
        """Kill a running task."""
        task = self.get_task(task_id)
        if task:
            task.kill()
            return True
        return False

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import PlanStep, RiskLevel, TaskPlan, TaskRecord, TaskStatus


class TaskStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(".gis-cli") / "tasks"
        self.root.mkdir(parents=True, exist_ok=True)

    def _task_file(self, task_id: str) -> Path:
        return self.root / f"{task_id}.json"

    def save(self, record: TaskRecord) -> None:
        payload = asdict(record)
        self._task_file(record.task_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, task_id: str) -> TaskRecord:
        payload = json.loads(self._task_file(task_id).read_text(encoding="utf-8"))
        plan = None
        if payload.get("plan"):
            plan_data = payload["plan"]
            plan = TaskPlan(
                intent=plan_data["intent"],
                confidence=plan_data["confidence"],
                planner_mode=plan_data.get("planner_mode", "rule"),
                missing_parameters=plan_data.get("missing_parameters", []),
                clarifying_questions=plan_data.get("clarifying_questions", []),
                steps=[
                    PlanStep(
                        order=step["order"],
                        title=step["title"],
                        status=step.get("status", "pending"),
                    )
                    for step in plan_data.get("steps", [])
                ],
            )
        return TaskRecord(
            task_id=payload["task_id"],
            prompt=payload["prompt"],
            status=TaskStatus(payload["status"]),
            risk=RiskLevel(payload["risk"]),
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            plan=plan,
            metadata=payload.get("metadata", {}),
        )

    def list_task_ids(self) -> list[str]:
        return [p.stem for p in self.root.glob("*.json")]

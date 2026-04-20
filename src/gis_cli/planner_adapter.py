from __future__ import annotations

import json
from pathlib import Path

from .catalog import IntentCatalog
from .models import TaskPlan
from .planner import IntentPlanner


class PlannerAdapter:
    """Adapter layer for planner migration.

    This keeps orchestrator call-sites stable while allowing planner backend
    switching by config.
    """

    def __init__(
        self,
        catalog: IntentCatalog | None = None,
        llm_rag_config_path: Path | None = None,
        adapter_config_path: Path | None = None,
    ) -> None:
        self.catalog = catalog or IntentCatalog()
        self.llm_rag_config_path = llm_rag_config_path or Path("config") / "llm_rag_config.json"
        self.adapter_config_path = adapter_config_path or Path("config") / "planner_adapter_config.json"
        self.adapter_config = self._load_adapter_config()

        # Legacy planner remains source of truth for now.
        self._legacy = IntentPlanner(self.catalog, llm_rag_config_path=self.llm_rag_config_path)
        # M1 compatibility: "standard" mode still points to compatible planner implementation.
        self._standard = IntentPlanner(self.catalog, llm_rag_config_path=self.llm_rag_config_path)
        self._last_shadow_compare: dict[str, str] = {}

    def _load_adapter_config(self) -> dict:
        if not self.adapter_config_path.exists():
            return {"mode": "legacy"}
        try:
            payload = json.loads(self.adapter_config_path.read_text(encoding="utf-8"))
        except Exception:
            return {"mode": "legacy"}
        if not isinstance(payload, dict):
            return {"mode": "legacy"}
        mode = str(payload.get("mode", "legacy")).strip().lower()
        if mode not in {"legacy", "standard", "shadow"}:
            mode = "legacy"
        payload["mode"] = mode
        return payload

    def _active_mode(self) -> str:
        return str(self.adapter_config.get("mode", "legacy")).strip().lower() or "legacy"

    def _active_planner(self) -> IntentPlanner:
        return self._legacy if self._active_mode() == "legacy" else self._standard

    def build_plan(self, prompt: str) -> TaskPlan:
        mode = self._active_mode()
        if mode == "shadow":
            # Execute both paths for future comparison without changing runtime behavior.
            standard_plan = self._standard.build_plan(prompt)
            legacy_plan = self._legacy.build_plan(prompt)
            self._last_shadow_compare = {
                "legacy_intent": legacy_plan.intent,
                "standard_intent": standard_plan.intent,
                "legacy_mode": legacy_plan.planner_mode,
                "standard_mode": standard_plan.planner_mode,
            }
            return standard_plan
        return self._active_planner().build_plan(prompt)

    def get_llm_rag_status(self) -> dict:
        status = self._active_planner().get_llm_rag_status()
        status["planner_adapter_mode"] = self._active_mode()
        status["planner_adapter_config_path"] = str(self.adapter_config_path)
        if self._last_shadow_compare:
            status["shadow_compare"] = dict(self._last_shadow_compare)
        return status

    def reload_llm_rag_config(self) -> dict:
        self.adapter_config = self._load_adapter_config()
        self._legacy.reload_llm_rag_config()
        self._standard.reload_llm_rag_config()
        return self.get_llm_rag_status()


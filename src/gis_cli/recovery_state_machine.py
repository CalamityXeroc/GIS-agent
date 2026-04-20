from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RecoveryInput:
    nodes: list[str]
    base_rerun_nodes: set[str]
    node_reasons: dict[str, str]
    auto_apply: bool
    effective_rules: dict[str, Any]


class RecoveryStateMachine:
    """Deterministic recovery strategy state machine."""

    def run(self, recovery_input: RecoveryInput) -> dict[str, Any]:
        strategy = {
            "auto_apply": recovery_input.auto_apply,
            "applied": False,
            "applied_rules": [],
            "skipped_nodes": [],
            "replaced_nodes": [],
            "effective_nodes": list(recovery_input.nodes),
            "force_dry_run": False,
            "decision_log": [],
        }

        if not recovery_input.auto_apply:
            strategy["decision_log"].append({"state": "idle", "decision": "auto_apply_disabled"})
            return strategy

        rule_flags = recovery_input.effective_rules.get("rule_flags", {})
        if not isinstance(rule_flags, dict):
            rule_flags = {}
        rule_priorities = recovery_input.effective_rules.get("rule_priorities", {})
        if not isinstance(rule_priorities, dict):
            rule_priorities = {}
        fallback_rules = recovery_input.effective_rules.get("implementation_fallback_nodes", {})
        if not isinstance(fallback_rules, dict):
            fallback_rules = {}

        def flag(name: str, default: bool = True) -> bool:
            return bool(rule_flags.get(name, default))

        def priority(name: str, default: int = 0) -> int:
            try:
                return int(rule_priorities.get(name, default))
            except Exception:
                return default

        effective_nodes: list[str] = []
        skipped_nodes: list[dict[str, str]] = []
        replaced_nodes: list[dict[str, object]] = []
        applied_rules: list[str] = []
        decision_log: list[dict[str, object]] = []

        for node in recovery_input.nodes:
            reason = str(recovery_input.node_reasons.get(node, "")).lower()
            if node in recovery_input.base_rerun_nodes and reason == "implementation_pending":
                replacement_nodes = fallback_rules.get(node, [])
                can_replace = flag("replace_implementation_pending_with_fallback", True)
                can_skip = flag("skip_implementation_pending", True)
                replace_prio = priority("replace_implementation_pending_with_fallback", 100)
                skip_prio = priority("skip_implementation_pending", 50)

                if replacement_nodes and can_replace and (not can_skip or replace_prio >= skip_prio):
                    replaced_nodes.append(
                        {
                            "node": node,
                            "reason": "implementation_pending",
                            "action": "replace_node",
                            "replacements": replacement_nodes,
                        }
                    )
                    if "replace_implementation_pending_with_fallback" not in applied_rules:
                        applied_rules.append("replace_implementation_pending_with_fallback")
                    for replacement in replacement_nodes:
                        if replacement not in effective_nodes:
                            effective_nodes.append(replacement)
                    decision_log.append(
                        {
                            "state": "evaluate_node",
                            "node": node,
                            "decision": "replace",
                            "rule": "replace_implementation_pending_with_fallback",
                            "priority": replace_prio,
                            "replacements": replacement_nodes,
                        }
                    )
                    continue

                if can_skip:
                    skipped_nodes.append(
                        {
                            "node": node,
                            "reason": "implementation_pending",
                            "action": "skip_node",
                        }
                    )
                    if "skip_implementation_pending" not in applied_rules:
                        applied_rules.append("skip_implementation_pending")
                    decision_log.append(
                        {
                            "state": "evaluate_node",
                            "node": node,
                            "decision": "skip",
                            "rule": "skip_implementation_pending",
                            "priority": skip_prio,
                        }
                    )
                    continue

                effective_nodes.append(node)
                decision_log.append(
                    {
                        "state": "evaluate_node",
                        "node": node,
                        "decision": "keep",
                        "rule": "none_enabled",
                    }
                )
                continue

            if node in recovery_input.base_rerun_nodes and reason == "arcpy_unavailable":
                if flag("force_dry_run_when_arcpy_unavailable", True):
                    strategy["force_dry_run"] = True
                    if "force_dry_run_when_arcpy_unavailable" not in applied_rules:
                        applied_rules.append("force_dry_run_when_arcpy_unavailable")
                    decision_log.append(
                        {
                            "state": "evaluate_node",
                            "node": node,
                            "decision": "force_dry_run",
                            "rule": "force_dry_run_when_arcpy_unavailable",
                            "priority": priority("force_dry_run_when_arcpy_unavailable", 100),
                        }
                    )
            effective_nodes.append(node)

        strategy["effective_nodes"] = effective_nodes
        strategy["skipped_nodes"] = skipped_nodes
        strategy["replaced_nodes"] = replaced_nodes
        strategy["applied_rules"] = applied_rules
        strategy["applied"] = bool(applied_rules)
        strategy["decision_log"] = decision_log
        return strategy


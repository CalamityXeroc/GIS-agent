from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AssemblyResult:
    runner_profile: str
    workflow_nodes: list[str] = field(default_factory=list)
    unresolved_capabilities: list[str] = field(default_factory=list)


class WorkflowAssembler:
    """Assemble executable node hints from intent and capability signals."""

    CAPABILITY_TO_NODES: dict[str, list[str]] = {
        "projection_transform": ["ensure_target_projection"],
        "geodatabase_write": ["ensure_workspace_gdb", "write_outputs"],
        "topology_check": ["check_geometry"],
        "geometry_repair": ["repair_geometry"],
        "map_export": ["export_map_layout"],
        "cartography_layout": ["apply_layout_template"],
        "labeling": ["apply_label_rules"],
        "buffer_analysis": ["buffer_analysis"],
        "overlay_analysis": ["overlay_analysis"],
        "spatial_join": ["spatial_join"],
        "batch_execution": ["batch_dispatch"],
    }

    INTENT_BASE_NODES: dict[str, list[str]] = {
        "data_integration": [
            "scan_inputs",
            "merge_by_layer",
            "field_cleanup",
            "ensure_target_projection",
            "write_outputs",
        ],
        "thematic_mapping": [
            "scan_inputs",
            "apply_symbology",
            "apply_label_rules",
            "apply_layout_template",
            "export_map_layout",
        ],
        "spatial_analysis": [
            "scan_inputs",
            "prepare_analysis_inputs",
            "run_spatial_tools",
            "write_outputs",
        ],
        "batch_processing": [
            "scan_inputs",
            "build_batch_units",
            "batch_dispatch",
            "write_outputs",
        ],
        "quality_check": [
            "scan_inputs",
            "check_geometry",
            "repair_geometry",
            "build_quality_report",
        ],
    }

    def assemble(self, intent: str, required_capabilities: list[str] | None = None) -> AssemblyResult:
        required_capabilities = required_capabilities or []
        nodes = list(self.INTENT_BASE_NODES.get(intent, ["scan_inputs", "write_outputs"]))
        unresolved: list[str] = []

        for cap in required_capabilities:
            mapped = self.CAPABILITY_TO_NODES.get(cap)
            if not mapped:
                unresolved.append(cap)
                continue
            for node in mapped:
                if node not in nodes:
                    nodes.append(node)

        profile = self._decide_profile(intent, required_capabilities)
        return AssemblyResult(
            runner_profile=profile,
            workflow_nodes=nodes,
            unresolved_capabilities=unresolved,
        )

    def _decide_profile(self, intent: str, required_capabilities: list[str]) -> str:
        if "batch_execution" in required_capabilities or intent == "batch_processing":
            return "batch"
        if intent == "thematic_mapping":
            return "mapping"
        if intent == "spatial_analysis":
            return "analysis"
        if intent == "quality_check":
            return "quality"
        return "integration"

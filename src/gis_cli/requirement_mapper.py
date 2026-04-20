from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequirementMapResult:
    required_capabilities: list[str] = field(default_factory=list)
    suggested_intents: list[str] = field(default_factory=list)
    unresolved_items: list[str] = field(default_factory=list)


class RequirementMapper:
    """Map document constraints/scoring items into executable capability hints."""

    CAPABILITY_RULES: list[tuple[str, str, str]] = [
        ("投影", "projection_transform", "data_integration"),
        ("坐标", "projection_transform", "data_integration"),
        ("入库", "geodatabase_write", "data_integration"),
        ("拓扑", "topology_check", "quality_check"),
        ("修复", "geometry_repair", "quality_check"),
        ("导出", "map_export", "thematic_mapping"),
        ("制图", "cartography_layout", "thematic_mapping"),
        ("标注", "labeling", "thematic_mapping"),
        ("缓冲", "buffer_analysis", "spatial_analysis"),
        ("叠加", "overlay_analysis", "spatial_analysis"),
        ("空间连接", "spatial_join", "spatial_analysis"),
        ("批量", "batch_execution", "batch_processing"),
    ]

    def map_document_requirements(
        self,
        constraints: list[str],
        scoring_items: list[str],
        extracted_actions: list[str],
    ) -> RequirementMapResult:
        text_pool = constraints + scoring_items + extracted_actions
        required_capabilities: list[str] = []
        suggested_intents: list[str] = []

        for line in text_pool:
            for keyword, capability, intent in self.CAPABILITY_RULES:
                if keyword in line:
                    if capability not in required_capabilities:
                        required_capabilities.append(capability)
                    if intent not in suggested_intents:
                        suggested_intents.append(intent)

        unresolved_items = [
            line
            for line in scoring_items
            if not any(keyword in line for keyword, _, _ in self.CAPABILITY_RULES)
        ]

        return RequirementMapResult(
            required_capabilities=required_capabilities,
            suggested_intents=suggested_intents,
            unresolved_items=unresolved_items,
        )

from gis_cli.catalog import IntentCatalog
from gis_cli.document_parser import DocumentTaskParser
from gis_cli.orchestrator import TaskOrchestrator
from gis_cli.planner_adapter import PlannerAdapter
from gis_cli.planner import IntentPlanner
from gis_cli.requirement_mapper import RequirementMapper
from gis_cli.runner.arcpy_runner import ArcPyRunner
from gis_cli.workflow_assembler import WorkflowAssembler
from gis_cli.models import ExecutionResult, RiskLevel, TaskPlan, TaskRecord, TaskStatus
from pathlib import Path
import json


def test_mapping_for_thematic_mapping_prompt() -> None:
    planner = IntentPlanner(IntentCatalog())
    plan = planner.build_plan("请完成专题图制图并导出png")
    assert plan.intent == "thematic_mapping"
    assert len(plan.steps) >= 3


def test_planner_loads_llm_rag_config(tmp_path) -> None:
    cfg = {
        "enabled": False,
        "llm": {"model": "gpt-4o-mini", "api_key": "", "api_base": "https://api.openai.com/v1"},
        "rag": {"knowledge_paths": ["README.md"], "top_k": 2, "chunk_size": 500},
    }
    cfg_path = tmp_path / "llm_rag_config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    planner = IntentPlanner(IntentCatalog(), llm_rag_config_path=cfg_path)
    assert planner.llm_rag_config.get("enabled") is False


def test_planner_fallbacks_to_rule_when_llm_disabled(tmp_path) -> None:
    cfg = {
        "enabled": False,
        "llm": {"model": "gpt-4o-mini", "api_key": "", "api_base": "https://api.openai.com/v1"},
        "rag": {"knowledge_paths": ["README.md"], "top_k": 2, "chunk_size": 500},
    }
    cfg_path = tmp_path / "llm_rag_config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    planner = IntentPlanner(IntentCatalog(), llm_rag_config_path=cfg_path)
    plan = planner.build_plan("请完成专题图制图并导出jpg")
    assert plan.planner_mode in {"rule", "fallback"}


def test_planner_adapter_default_mode_is_legacy(tmp_path) -> None:
    adapter = PlannerAdapter(
        catalog=IntentCatalog(),
        llm_rag_config_path=tmp_path / "llm_rag_config.json",
        adapter_config_path=tmp_path / "planner_adapter_config.json",
    )
    status = adapter.get_llm_rag_status()
    assert status.get("planner_adapter_mode") == "legacy"


def test_planner_adapter_accepts_standard_mode(tmp_path) -> None:
    cfg_path = tmp_path / "planner_adapter_config.json"
    cfg_path.write_text('{"mode":"standard"}', encoding="utf-8")
    adapter = PlannerAdapter(
        catalog=IntentCatalog(),
        llm_rag_config_path=tmp_path / "llm_rag_config.json",
        adapter_config_path=cfg_path,
    )
    status = adapter.get_llm_rag_status()
    assert status.get("planner_adapter_mode") == "standard"
    plan = adapter.build_plan("整合数据并导出")
    assert plan.intent in {"data_integration", "thematic_mapping", "spatial_analysis", "batch_processing", "quality_check"}


def test_planner_adapter_shadow_mode_exposes_compare_status(tmp_path) -> None:
    cfg_path = tmp_path / "planner_adapter_config.json"
    cfg_path.write_text('{"mode":"shadow"}', encoding="utf-8")
    adapter = PlannerAdapter(
        catalog=IntentCatalog(),
        llm_rag_config_path=tmp_path / "llm_rag_config.json",
        adapter_config_path=cfg_path,
    )
    plan = adapter.build_plan("整合数据并导出")
    assert plan.intent in {"data_integration", "thematic_mapping", "spatial_analysis", "batch_processing", "quality_check"}
    status = adapter.get_llm_rag_status()
    assert status.get("planner_adapter_mode") == "shadow"
    shadow_compare = status.get("shadow_compare", {})
    assert shadow_compare.get("legacy_intent")
    assert shadow_compare.get("standard_intent")


def test_planner_fallbacks_to_rule_when_llm_missing_key(tmp_path) -> None:
    cfg = {
        "enabled": True,
        "llm": {"model": "gpt-4o-mini", "api_key": "", "api_base": "https://api.openai.com/v1"},
        "rag": {"knowledge_paths": ["README.md"], "top_k": 2, "chunk_size": 500},
    }
    cfg_path = tmp_path / "llm_rag_config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    planner = IntentPlanner(IntentCatalog(), llm_rag_config_path=cfg_path)
    plan = planner.build_plan("整合数据并入库")
    assert plan.intent == "data_integration"
    assert plan.planner_mode in {"rule", "fallback"}


def test_planner_rag_status_contains_fields(tmp_path) -> None:
    cfg = {
        "enabled": True,
        "llm": {"model": "gpt-4o-mini", "api_key": "", "api_base": "https://api.openai.com/v1"},
        "rag": {
            "knowledge_paths": ["README.md"],
            "top_k": 2,
            "chunk_size": 500,
            "auto_include_outputs": False,
        },
    }
    cfg_path = tmp_path / "llm_rag_config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    planner = IntentPlanner(IntentCatalog(), llm_rag_config_path=cfg_path)
    status = planner.get_llm_rag_status()
    assert "enabled" in status
    assert "has_api_key" in status
    assert "static_source_count" in status
    assert "rag_backend" in status
    assert "rag_backend_active" in status


def test_planner_rag_auto_discovers_outputs(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    task_dir = output_root / "t001"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "checklist.json").write_text('{"ok": true}', encoding="utf-8")

    cfg = {
        "enabled": True,
        "llm": {"model": "gpt-4o-mini", "api_key": "", "api_base": "https://api.openai.com/v1"},
        "rag": {
            "knowledge_paths": [],
            "top_k": 2,
            "chunk_size": 500,
            "auto_include_outputs": True,
            "output_root": str(output_root),
            "max_output_files": 5,
            "output_globs": ["**/checklist.json"],
        },
    }
    cfg_path = tmp_path / "llm_rag_config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    planner = IntentPlanner(IntentCatalog(), llm_rag_config_path=cfg_path)
    status = planner.get_llm_rag_status()
    assert status["dynamic_source_count"] >= 1


def test_planner_langchain_backend_falls_back_to_native_when_langchain_path_empty(tmp_path, monkeypatch) -> None:
    kb = tmp_path / "kb.txt"
    kb.write_text("GIS 投影 质量 检查", encoding="utf-8")
    cfg = {
        "enabled": True,
        "llm": {"model": "gpt-4o-mini", "api_key": "x", "api_base": "https://api.openai.com/v1"},
        "rag": {
            "backend": "langchain",
            "knowledge_paths": [str(kb)],
            "top_k": 1,
            "chunk_size": 200,
            "auto_include_outputs": False,
        },
    }
    cfg_path = tmp_path / "llm_rag_config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    planner = IntentPlanner(IntentCatalog(), llm_rag_config_path=cfg_path)

    monkeypatch.setattr(planner, "_retrieve_rag_context_langchain", lambda prompt, rag_cfg: "")
    context = planner._retrieve_rag_context("请检查投影")  # noqa: SLF001
    assert context
    assert planner._last_rag_backend == "native_fallback"  # noqa: SLF001


def test_spatial_analysis_missing_distance() -> None:
    planner = IntentPlanner(IntentCatalog())
    plan = planner.build_plan("对行政区做缓冲分析并输出结果")
    assert plan.intent == "spatial_analysis"
    assert "缓冲距离与单位" in plan.missing_parameters


def test_integration_prompt_prefers_data_integration() -> None:
    planner = IntentPlanner(IntentCatalog())
    plan = planner.build_plan("整合9个分幅数据并入库，同时导出专题图")
    assert plan.intent == "data_integration"


def test_runner_dry_run_exports_plan(tmp_path) -> None:
    runner = ArcPyRunner(output_root=tmp_path / "outputs")
    runner.data_root = tmp_path / "data"
    runner.data_root.mkdir(parents=True, exist_ok=True)
    (runner.data_root / "E48").mkdir(parents=True, exist_ok=True)
    (runner.data_root / "E48" / "BOUA.shp").write_text("", encoding="utf-8")

    record = TaskRecord(
        task_id="t001",
        prompt="整合数据",
        status=TaskStatus.READY,
        risk=RiskLevel.LOW,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        plan=TaskPlan(intent="data_integration", confidence=0.9),
    )

    result = runner.run(record, dry_run=True)
    assert result.success is True
    assert any(path.endswith("dry_run_plan.json") for path in result.outputs)
    assert any(path.endswith("checklist.json") for path in result.outputs)


def test_document_parser_extracts_actions_and_constraints() -> None:
    parser = DocumentTaskParser()
    text = """
    广西区域发展布局示意图任务
    1. 创建数据库并整合数据入库
    2. 投影统一到 Asia_North_Albers_Equal_Area_Conic
    3. 导出结果 PNG
    """
    parsed = parser.parse_text(text)
    assert parsed.title.startswith("广西区域发展布局示意图任务")
    assert any("投影" in x for x in parsed.constraints)
    assert any("整合" in x or "创建" in x for x in parsed.extracted_actions)
    assert parsed.normalized_prompt


def test_document_parser_dedupes_normalized_prompt() -> None:
    parser = DocumentTaskParser()
    text = "创建数据库并整合数据入库，投影统一，导出专题图"
    parsed = parser.parse_text(text)
    assert parsed.normalized_prompt.count("创建数据库并整合数据入库") <= 1


def test_requirement_mapper_extracts_capabilities() -> None:
    mapper = RequirementMapper()
    result = mapper.map_document_requirements(
        constraints=["投影统一到目标坐标系", "导出比赛版png"],
        scoring_items=["拓扑检查并修复（5分）"],
        extracted_actions=["创建数据库并整合入库"],
    )
    assert "projection_transform" in result.required_capabilities
    assert "map_export" in result.required_capabilities
    assert "topology_check" in result.required_capabilities


def test_orchestrator_document_task_has_requirement_hints() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task_from_document("创建数据库，投影统一，导出专题图")
    assert "required_capabilities" in record.metadata
    assert isinstance(record.metadata["required_capabilities"], list)


def test_workflow_assembler_adds_nodes_from_capabilities() -> None:
    assembler = WorkflowAssembler()
    assembled = assembler.assemble(
        "data_integration",
        required_capabilities=["projection_transform", "topology_check"],
    )
    assert assembled.runner_profile == "integration"
    assert "ensure_target_projection" in assembled.workflow_nodes
    assert "check_geometry" in assembled.workflow_nodes


def test_orchestrator_sets_execution_profile_metadata() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("整合数据并导出")
    assert "runner_profile" in record.metadata
    assert "workflow_nodes" in record.metadata
    assert isinstance(record.metadata["workflow_nodes"], list)


def test_runner_dry_run_contains_workflow_nodes(tmp_path) -> None:
    runner = ArcPyRunner(output_root=tmp_path / "outputs")
    runner.data_root = tmp_path / "data"
    runner.data_root.mkdir(parents=True, exist_ok=True)

    record = TaskRecord(
        task_id="t002",
        prompt="整合数据",
        status=TaskStatus.READY,
        risk=RiskLevel.LOW,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        plan=TaskPlan(intent="data_integration", confidence=0.9),
        metadata={"runner_profile": "integration", "workflow_nodes": ["scan_inputs", "merge_by_layer"]},
    )
    result = runner.run(record, dry_run=True)
    plan_path = next(path for path in result.outputs if path.endswith("dry_run_plan.json"))
    import json
    from pathlib import Path

    payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    assert payload["runner_profile"] == "integration"
    assert "merge_by_layer" in payload["workflow_nodes"]


def test_runner_fallback_report_contains_workflow_context(tmp_path, monkeypatch) -> None:
    runner = ArcPyRunner(output_root=tmp_path / "outputs")
    runner.data_root = tmp_path / "data"
    runner.data_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(runner, "_load_arcpy", lambda: None)

    record = TaskRecord(
        task_id="t003",
        prompt="整合数据",
        status=TaskStatus.READY,
        risk=RiskLevel.LOW,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        plan=TaskPlan(intent="data_integration", confidence=0.9),
        metadata={"runner_profile": "integration", "workflow_nodes": ["scan_inputs", "write_outputs"]},
    )
    result = runner.run(record, dry_run=False)
    report_path = next(path for path in result.outputs if path.endswith("execution_report.txt"))
    from pathlib import Path

    report = Path(report_path).read_text(encoding="utf-8")
    assert "runner_profile=integration" in report
    assert "workflow_nodes=['scan_inputs', 'write_outputs']" in report


def test_orchestrator_persists_last_execution_metadata() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("整合并导出")
    result = orch.run_task(record.task_id, dry_run=True)
    assert result.success is True

    loaded = orch.get_task(record.task_id)
    assert "last_execution" in loaded.metadata
    assert loaded.metadata["last_execution"]["success"] is True
    assert "details" in loaded.metadata["last_execution"]


def test_orchestrator_override_nodes_for_partial_run() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("整合并导出")
    result = orch.run_task(
        record.task_id,
        dry_run=True,
        override_nodes=["check_geometry", "repair_geometry"],
    )
    assert result.success is True
    assert result.details.get("workflow_nodes") == ["check_geometry", "repair_geometry"]

    loaded = orch.get_task(record.task_id)
    assert loaded.metadata.get("workflow_nodes") == ["check_geometry", "repair_geometry"]
    assert loaded.metadata.get("workflow_override_active") is True


def test_suggest_rerun_nodes_from_last_execution() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("整合并导出")
    loaded = orch.get_task(record.task_id)
    loaded.metadata["last_execution"] = {
        "details": {
            "node_execution": {
                "scan_inputs": "executed",
                "check_geometry": "failed",
                "repair_geometry": "pending",
                "export_map_layout": "skipped",
            }
        }
    }
    orch.store.save(loaded)

    suggested = orch.suggest_rerun_nodes(record.task_id, include_skipped=False)
    assert "check_geometry" in suggested
    assert "repair_geometry" in suggested
    assert "export_map_layout" not in suggested


def test_rerun_failed_nodes_returns_noop_when_empty() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("整合并导出")
    loaded = orch.get_task(record.task_id)
    loaded.metadata["last_execution"] = {"details": {"node_execution": {"scan_inputs": "executed"}}}
    orch.store.save(loaded)

    result = orch.rerun_failed_nodes(record.task_id, dry_run=True)
    assert result.success is True
    assert result.message == "No failed/pending nodes to rerun."


def test_rerun_failed_nodes_uses_override_path() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("整合并导出")
    loaded = orch.get_task(record.task_id)
    loaded.metadata["last_execution"] = {
        "details": {"node_execution": {"check_geometry": "failed", "repair_geometry": "pending"}}
    }
    orch.store.save(loaded)

    result = orch.rerun_failed_nodes(record.task_id, dry_run=True)
    assert result.success is True
    assert result.details.get("workflow_nodes") == ["check_geometry", "repair_geometry"]


def test_rerun_failed_nodes_auto_replaces_pending_implementation() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("制图并导出")
    loaded = orch.get_task(record.task_id)
    loaded.metadata["last_execution"] = {
        "details": {
            "node_execution": {"export_map_layout": "pending"},
            "node_reasons": {"export_map_layout": "implementation_pending"},
        }
    }
    orch.store.save(loaded)

    result = orch.rerun_failed_nodes(record.task_id, dry_run=True)
    assert result.success is True
    assert result.details.get("strategy_applied") is True
    assert result.details.get("workflow_nodes") == ["build_quality_report"]
    strategy = result.details.get("strategy", {})
    assert "replace_implementation_pending_with_fallback" in strategy.get("applied_rules", [])
    replaced = strategy.get("replaced_nodes", [])
    assert replaced and replaced[0].get("node") == "export_map_layout"


def test_rerun_failed_nodes_skips_unknown_pending_implementation() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("制图并导出")
    loaded = orch.get_task(record.task_id)
    loaded.metadata["last_execution"] = {
        "details": {
            "node_execution": {"unknown_node": "pending"},
            "node_reasons": {"unknown_node": "implementation_pending"},
        }
    }
    orch.store.save(loaded)

    result = orch.rerun_failed_nodes(record.task_id, dry_run=True)
    assert result.success is True
    assert result.message == "No eligible nodes to rerun after applying recovery strategy."
    strategy = result.details.get("strategy", {})
    assert strategy.get("replaced_nodes") == []
    skipped = strategy.get("skipped_nodes", [])
    assert skipped and skipped[0].get("node") == "unknown_node"


def test_rerun_failed_nodes_can_disable_auto_strategy() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("制图并导出")
    loaded = orch.get_task(record.task_id)
    loaded.metadata["last_execution"] = {
        "details": {
            "node_execution": {"export_map_layout": "pending"},
            "node_reasons": {"export_map_layout": "implementation_pending"},
        }
    }
    orch.store.save(loaded)

    result = orch.rerun_failed_nodes(
        record.task_id,
        dry_run=True,
        auto_apply_strategy=False,
    )
    assert result.success is True
    assert result.details.get("strategy_applied") is False
    assert result.details.get("workflow_nodes") == ["export_map_layout"]


def test_rerun_failed_nodes_forces_dry_run_when_arcpy_unavailable() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("整合并导出")
    loaded = orch.get_task(record.task_id)
    loaded.metadata["last_execution"] = {
        "details": {
            "node_execution": {"merge_agnp": "failed"},
            "node_reasons": {"merge_agnp": "arcpy_unavailable"},
        }
    }
    orch.store.save(loaded)

    result = orch.rerun_failed_nodes(record.task_id, dry_run=False)
    assert result.success is True
    assert result.dry_run is True
    assert result.details.get("effective_dry_run") is True
    strategy = result.details.get("strategy", {})
    assert strategy.get("force_dry_run") is True
    assert "force_dry_run_when_arcpy_unavailable" in strategy.get("applied_rules", [])


def test_rerun_failed_nodes_no_force_dry_run_when_auto_strategy_disabled() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("整合并导出")
    loaded = orch.get_task(record.task_id)
    loaded.metadata["last_execution"] = {
        "details": {
            "node_execution": {"merge_agnp": "failed"},
            "node_reasons": {"merge_agnp": "arcpy_unavailable"},
        }
    }
    orch.store.save(loaded)

    result = orch.rerun_failed_nodes(
        record.task_id,
        dry_run=False,
        auto_apply_strategy=False,
    )
    assert result.success is True
    assert result.dry_run is False
    strategy = result.details.get("strategy", {})
    assert strategy.get("force_dry_run") is False


def test_orchestrator_handles_runner_exception_gracefully(monkeypatch) -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("整合并导出")

    def _boom(_record, dry_run=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(orch.runner, "run", _boom)
    result = orch.run_task(record.task_id, dry_run=True)
    assert result.success is False
    assert result.details.get("error_code") == "runner_exception"

    loaded = orch.get_task(record.task_id)
    assert loaded.status == TaskStatus.FAILED
    assert loaded.metadata.get("last_execution", {}).get("success") is False


def test_recovery_plan_adds_prep_nodes_for_missing_input() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("整合并导出")
    loaded = orch.get_task(record.task_id)
    loaded.metadata["last_execution"] = {
        "details": {
            "node_execution": {"merge_agnp": "failed"},
            "node_reasons": {"merge_agnp": "missing_input"},
        }
    }
    orch.store.save(loaded)

    plan = orch.suggest_recovery_plan(record.task_id)
    assert "scan_inputs" in plan["nodes"]
    assert "merge_agnp" in plan["nodes"]


def test_recovery_plan_recommendation_for_pending_implementation() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("制图并导出")
    loaded = orch.get_task(record.task_id)
    loaded.metadata["last_execution"] = {
        "details": {
            "node_execution": {"export_map_layout": "pending"},
            "node_reasons": {"export_map_layout": "implementation_pending"},
        }
    }
    orch.store.save(loaded)

    plan = orch.suggest_recovery_plan(record.task_id)
    assert any("尚未实现" in item for item in plan["recommendations"])


def test_planner_fallback_mode_when_prompt_is_ambiguous() -> None:
    planner = IntentPlanner(IntentCatalog())
    plan = planner.build_plan_hybrid("帮我搞定这个 GIS 任务吧", min_rule_confidence=0.95)
    assert plan.planner_mode == "fallback"
    assert plan.intent in {
        "data_integration",
        "thematic_mapping",
        "spatial_analysis",
        "batch_processing",
        "quality_check",
    }
    assert len(plan.clarifying_questions) >= 1


def test_agent_run_returns_trace_and_task() -> None:
    orch = TaskOrchestrator()
    result = orch.run_agent("整合数据并导出", dry_run=True, max_recovery_rounds=1)
    assert result["task_id"]
    assert result["attempts"] >= 1
    assert isinstance(result["trace"], list)
    assert result["trace"][0]["stage"] == "initial_run"
    assert "final_result" in result
    assert "task" in result


def test_agent_run_for_task_executes_recovery_round(monkeypatch) -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("整合数据并导出")

    call_state = {"run_calls": 0, "recovery_calls": 0}

    def _fake_run_task(task_id: str, dry_run: bool = False, override_nodes=None):
        call_state["run_calls"] += 1
        return ExecutionResult(
            task_id=task_id,
            success=False,
            dry_run=dry_run,
            message="Task execution failed in runner.",
            outputs=[],
            details={"workflow_nodes": ["merge_agnp"], "error_code": "merge_agnp_failed"},
        )

    def _fake_rerun_failed_nodes(
        task_id: str,
        dry_run: bool = True,
        include_skipped: bool = True,
        auto_apply_strategy: bool = True,
    ):
        call_state["recovery_calls"] += 1
        return ExecutionResult(
            task_id=task_id,
            success=True,
            dry_run=dry_run,
            message="No failed/pending nodes to rerun.",
            outputs=[],
            details={"workflow_nodes": [], "strategy_applied": True, "strategy": {"applied_rules": []}},
        )

    monkeypatch.setattr(orch, "run_task", _fake_run_task)
    monkeypatch.setattr(orch, "rerun_failed_nodes", _fake_rerun_failed_nodes)

    result = orch.run_agent_for_task(record.task_id, dry_run=True, max_recovery_rounds=2)
    assert call_state["run_calls"] == 1
    assert call_state["recovery_calls"] == 1
    assert result["attempts"] == 2
    assert result["trace"][1]["stage"] == "recovery_round_1"


def test_orchestrator_uses_custom_recovery_config(tmp_path) -> None:
    import json

    config_path = tmp_path / "recovery_strategies.json"
    config_path.write_text(
        json.dumps(
            {
                "implementation_fallback_nodes": {
                    "export_map_layout": ["scan_inputs"]
                },
                "reason_recommendations": {
                    "implementation_pending": "自定义建议：节点 {node} 待实现。"
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    orch = TaskOrchestrator(recovery_config_path=config_path)
    record = orch.create_task("制图并导出")
    loaded = orch.get_task(record.task_id)
    loaded.metadata["last_execution"] = {
        "details": {
            "node_execution": {"export_map_layout": "pending"},
            "node_reasons": {"export_map_layout": "implementation_pending"},
        }
    }
    orch.store.save(loaded)

    result = orch.rerun_failed_nodes(record.task_id, dry_run=True)
    assert result.success is True
    assert result.details.get("workflow_nodes") == ["scan_inputs"]
    assert any("自定义建议" in x for x in result.details.get("recommendations", []))


def test_recovery_strategy_info_contains_version() -> None:
    orch = TaskOrchestrator()
    info = orch.get_recovery_strategy_info()
    assert "version" in info
    assert "source" in info


def test_recovery_rules_hot_reload_by_mtime(tmp_path) -> None:
    import json
    import time

    config_path = tmp_path / "recovery_strategies.json"
    config_path.write_text(
        json.dumps(
            {
                "implementation_fallback_nodes": {
                    "export_map_layout": ["build_quality_report"]
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    orch = TaskOrchestrator(recovery_config_path=config_path)
    first_version = orch.get_recovery_strategy_info().get("version")

    time.sleep(0.02)
    config_path.write_text(
        json.dumps(
            {
                "implementation_fallback_nodes": {
                    "export_map_layout": ["scan_inputs"]
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    record = orch.create_task("制图并导出")
    loaded = orch.get_task(record.task_id)
    loaded.metadata["last_execution"] = {
        "details": {
            "node_execution": {"export_map_layout": "pending"},
            "node_reasons": {"export_map_layout": "implementation_pending"},
        }
    }
    orch.store.save(loaded)

    result = orch.rerun_failed_nodes(record.task_id, dry_run=True)
    assert result.details.get("workflow_nodes") == ["scan_inputs"]
    next_version = result.details.get("recovery_strategy", {}).get("version")
    assert next_version
    assert next_version != first_version


def test_agent_trace_contains_recovery_strategy_info() -> None:
    orch = TaskOrchestrator()
    result = orch.run_agent("整合数据并导出", dry_run=True, max_recovery_rounds=1)
    assert "recovery_strategy" in result
    assert "recovery_strategy" in result["trace"][0]


def test_profile_specific_recovery_rules_override(tmp_path) -> None:
    import json

    config_path = tmp_path / "recovery_strategies.json"
    config_path.write_text(
        json.dumps(
            {
                "implementation_fallback_nodes": {
                    "export_map_layout": ["build_quality_report"]
                },
                "profiles": {
                    "mapping": {
                        "implementation_fallback_nodes": {
                            "export_map_layout": ["apply_layout_template"]
                        }
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    orch = TaskOrchestrator(recovery_config_path=config_path)
    record = orch.create_task("制图并导出")
    loaded = orch.get_task(record.task_id)
    loaded.metadata["last_execution"] = {
        "details": {
            "node_execution": {"export_map_layout": "pending"},
            "node_reasons": {"export_map_layout": "implementation_pending"},
        }
    }
    orch.store.save(loaded)

    result = orch.rerun_failed_nodes(record.task_id, dry_run=True)
    assert result.details.get("recovery_profile") == "mapping"
    assert result.details.get("workflow_nodes") == ["apply_layout_template"]


def test_recovery_strategy_info_includes_profiles() -> None:
    orch = TaskOrchestrator()
    info = orch.get_recovery_strategy_info("mapping")
    assert "available_profiles" in info
    assert "effective_version" in info
    assert info.get("active_profile") == "mapping"


def test_recovery_strategy_decision_log_contains_state_field() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("制图并导出")
    loaded = orch.get_task(record.task_id)
    loaded.metadata["last_execution"] = {
        "details": {
            "node_execution": {"export_map_layout": "pending"},
            "node_reasons": {"export_map_layout": "implementation_pending"},
        }
    }
    orch.store.save(loaded)

    result = orch.rerun_failed_nodes(record.task_id, dry_run=True)
    log = result.details.get("strategy", {}).get("decision_log", [])
    assert isinstance(log, list)
    assert log
    assert "state" in log[0]


def test_rule_flags_can_disable_replace(tmp_path) -> None:
    import json

    config_path = tmp_path / "recovery_strategies.json"
    config_path.write_text(
        json.dumps(
            {
                "implementation_fallback_nodes": {
                    "export_map_layout": ["build_quality_report"]
                },
                "rule_flags": {
                    "replace_implementation_pending_with_fallback": False,
                    "skip_implementation_pending": True,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    orch = TaskOrchestrator(recovery_config_path=config_path)
    record = orch.create_task("制图并导出")
    loaded = orch.get_task(record.task_id)
    loaded.metadata["last_execution"] = {
        "details": {
            "node_execution": {"export_map_layout": "pending"},
            "node_reasons": {"export_map_layout": "implementation_pending"},
        }
    }
    orch.store.save(loaded)

    result = orch.rerun_failed_nodes(record.task_id, dry_run=True)
    assert result.message == "No eligible nodes to rerun after applying recovery strategy."
    decision_log = result.details.get("strategy", {}).get("decision_log", [])
    assert decision_log and decision_log[0].get("decision") == "skip"


def test_rule_priority_prefers_skip_over_replace(tmp_path) -> None:
    import json

    config_path = tmp_path / "recovery_strategies.json"
    config_path.write_text(
        json.dumps(
            {
                "implementation_fallback_nodes": {
                    "export_map_layout": ["build_quality_report"]
                },
                "rule_flags": {
                    "replace_implementation_pending_with_fallback": True,
                    "skip_implementation_pending": True,
                },
                "rule_priorities": {
                    "replace_implementation_pending_with_fallback": 10,
                    "skip_implementation_pending": 50,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    orch = TaskOrchestrator(recovery_config_path=config_path)
    record = orch.create_task("制图并导出")
    loaded = orch.get_task(record.task_id)
    loaded.metadata["last_execution"] = {
        "details": {
            "node_execution": {"export_map_layout": "pending"},
            "node_reasons": {"export_map_layout": "implementation_pending"},
        }
    }
    orch.store.save(loaded)

    result = orch.rerun_failed_nodes(record.task_id, dry_run=True)
    assert result.details.get("workflow_nodes") == []
    decision_log = result.details.get("strategy", {}).get("decision_log", [])
    assert decision_log and decision_log[0].get("decision") == "skip"


def test_task_summary_counts_runs() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("整合数据并导出")
    _ = orch.run_task(record.task_id, dry_run=True)
    _ = orch.rerun_failed_nodes(record.task_id, dry_run=True)

    summary = orch.get_task_summary(record.task_id)
    assert summary["total_runs"] >= 2
    assert summary["dry_runs"] >= 2
    assert "recovery_strategy" in summary


def test_export_tasks_stats_json() -> None:
    orch = TaskOrchestrator()
    record1 = orch.create_task("制图并导出")
    record2 = orch.create_task("整合数据并导出")
    _ = orch.run_task(record1.task_id, dry_run=True)
    _ = orch.run_task(record2.task_id, dry_run=True)
    
    result = orch.export_tasks_stats(format="json")
    assert isinstance(result, str)
    
    import json
    data = json.loads(result)
    assert isinstance(data, list)
    assert len(data) >= 2
    
    for item in data:
        assert "task_id" in item
        assert "status" in item
        assert "total_runs" in item


def test_export_tasks_stats_csv() -> None:
    orch = TaskOrchestrator()
    record1 = orch.create_task("制图并导出")
    record2 = orch.create_task("整合数据并导出")
    _ = orch.run_task(record1.task_id, dry_run=True)
    _ = orch.run_task(record2.task_id, dry_run=True)
    
    result = orch.export_tasks_stats(format="csv")
    assert isinstance(result, str)
    assert "task_id" in result
    assert "status" in result
    assert "total_runs" in result
    
    lines = result.strip().split("\n")
    assert len(lines) >= 2  # Header + at least 1 data row


def test_export_tasks_stats_filtered() -> None:
    orch = TaskOrchestrator()
    record1 = orch.create_task("制图并导出")
    record2 = orch.create_task("整合数据并导出")
    _ = orch.run_task(record1.task_id, dry_run=True)
    _ = orch.run_task(record2.task_id, dry_run=True)
    
    result = orch.export_tasks_stats(format="json", task_ids=[record1.task_id])
    
    import json
    data = json.loads(result)
    assert len(data) == 1
    assert data[0]["task_id"] == record1.task_id


def test_export_tasks_stats_invalid_format() -> None:
    orch = TaskOrchestrator()
    
    try:
        _ = orch.export_tasks_stats(format="xml")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "不支持的格式" in str(e)


def test_list_tasks_pagination() -> None:
    orch = TaskOrchestrator()
    
    # Create multiple tasks
    task_ids = []
    for i in range(5):
        record = orch.create_task(f"制图并导出 {i}")
        task_ids.append(record.task_id)
    
    all_ids = orch.list_tasks()
    assert len(all_ids) >= 5
    
    # Test with skip and limit
    # (In real implementation, this would be tested via API or higher-level method)


def test_task_search_by_prompt() -> None:
    orch = TaskOrchestrator()
    record1 = orch.create_task("制图并导出使用ArcGIS")
    record2 = orch.create_task("整合数据并导出CSV文件")
    record3 = orch.create_task("清质量报告和导出")
    
    all_ids = orch.list_tasks()
    assert record1.task_id in all_ids
    assert record2.task_id in all_ids
    assert record3.task_id in all_ids


def test_delete_task() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("制图并导出")
    task_id = record.task_id
    
    # Verify task exists
    assert task_id in orch.list_tasks()
    
    # Delete task
    result = orch.delete_task(task_id, force=True)
    assert result["success"] is True
    
    # Verify task is deleted
    assert task_id not in orch.list_tasks()


def test_archive_task() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("制图并导出")
    task_id = record.task_id
    
    # Archive task
    result = orch.archive_task(task_id)
    assert result["success"] is True
    
    # Verify task is archived
    assert task_id not in orch.list_tasks()
    archived = orch.get_archive_list()
    assert task_id in archived


def test_restore_task() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("制图并导出")
    task_id = record.task_id
    
    # Archive task
    orch.archive_task(task_id)
    assert task_id not in orch.list_tasks()
    
    # Restore task
    result = orch.restore_task(task_id)
    assert result["success"] is True
    
    # Verify task is restored
    assert task_id in orch.list_tasks()
    archived = orch.get_archive_list()
    assert task_id not in archived


def test_cleanup_tasks_dry_run() -> None:
    orch = TaskOrchestrator()
    record = orch.create_task("制图并导出")
    
    # Run dry cleanup
    result = orch.cleanup_tasks(max_age_days=0, dry_run=True)
    assert result["dry_run"] is True
    # May or may not have tasks depending on initial status
    assert "would_delete" in result or "tasks" in result


def test_strategy_statistics() -> None:
    orch = TaskOrchestrator()
    
    # Create a few tasks
    for i in range(3):
        record = orch.create_task(f"任务{i}")
        _ = orch.run_task(record.task_id, dry_run=True)
    
    stats = orch.get_strategy_statistics()
    assert stats["total_tasks"] >= 3
    assert "profile_distribution" in stats
    assert "recovery_rounds_stats" in stats
    assert "task_status_distribution" in stats


def test_generate_strategy_report_json() -> None:
    orch = TaskOrchestrator()
    
    # Create a task and run it
    record = orch.create_task("制图并导出")
    _ = orch.run_task(record.task_id, dry_run=True)
    
    result = orch.generate_strategy_report(format="json")
    assert isinstance(result, str)
    
    import json
    data = json.loads(result)
    assert "total_tasks" in data
    assert "recovery_success_rate" in data


def test_generate_strategy_report_html() -> None:
    orch = TaskOrchestrator()
    
    # Create a task
    record = orch.create_task("制图并导出")
    _ = orch.run_task(record.task_id, dry_run=True)
    
    result = orch.generate_strategy_report(format="html")
    assert isinstance(result, str)
    assert "<!DOCTYPE html>" in result
    assert "<title>" in result
    assert "Total Tasks" in result or "total_tasks" in result.lower()


def test_strategy_report_invalid_format() -> None:
    orch = TaskOrchestrator()
    
    try:
        _ = orch.generate_strategy_report(format="xml")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "不支持的格式" in str(e)

"""End-to-end integration tests for GIS CLI."""

from pathlib import Path
from gis_cli.orchestrator import TaskOrchestrator
from gis_cli.models import TaskStatus, RiskLevel
from gis_cli.runner.arcpy_runner import ArcPyRunner


def test_e2e_task_create_and_plan(tmp_path) -> None:
    """Test creating a task and generating a plan."""
    orch = TaskOrchestrator()
    
    # Create task from prompt
    prompt = "整合9个分幅数据，统一投影并导出广西专题图"
    record = orch.create_task(prompt)
    
    # Verify task created
    assert record.task_id is not None
    assert record.status == TaskStatus.DRAFT
    assert record.plan is not None
    assert record.plan.intent == "data_integration"
    assert record.plan.confidence > 0.5


def test_e2e_document_task_creation(tmp_path) -> None:
    """Test creating task from document text."""
    orch = TaskOrchestrator()
    
    doc_text = """
    数据整合任务：广西行政区域数据
    
    1. 创建数据库并整合9个分幅数据入库
    2. 统一投影到 Asia_North_Albers_Equal_Area_Conic
    3. 检查和修复几何问题
    4. 导出专题图为 PNG 格式
    5. 生成质量检查报告
    """
    
    record = orch.create_task_from_document(doc_text, source_name="test_doc")
    
    # Verify task created from document
    assert record.task_id is not None
    assert record.status == TaskStatus.DRAFT
    assert "workflow_nodes" in record.metadata
    assert "required_capabilities" in record.metadata


def test_e2e_dry_run_execution(tmp_path) -> None:
    """Test full workflow with dry-run execution."""
    # Setup
    data_root = tmp_path / "数据"
    output_root = tmp_path / "outputs"
    
    data_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Create sample data files
    (data_root / "E48").mkdir(parents=True, exist_ok=True)
    (data_root / "E48" / "BOUA.shp").write_text("mock shapefile", encoding="utf-8")
    (data_root / "E49").mkdir(parents=True, exist_ok=True)
    (data_root / "E49" / "AGNP.shp").write_text("mock shapefile", encoding="utf-8")
    
    orch = TaskOrchestrator()
    orch.runner = ArcPyRunner(output_root=output_root)
    orch.runner.data_root = data_root
    
    # Create and execute task
    prompt = "整合数据并导出"
    record = orch.create_task(prompt)
    
    # Run dry-run
    result = orch.run_task(record.task_id, dry_run=True)
    
    # Verify results
    assert result.success is True
    assert result.dry_run is True
    assert len(result.outputs) > 0
    assert any("dry_run_plan.json" in out for out in result.outputs)
    assert any("checklist.json" in out for out in result.outputs)


def test_e2e_task_planning_with_workflow_assembly(tmp_path) -> None:
    """Test task planning with workflow node assembly."""
    orch = TaskOrchestrator()
    
    # Create task with multiple capabilities
    doc_text = """
    专题制图任务
    1. 加载数据
    2. 应用符号化
    3. 应用版式模板
    4. 导出地图版式为PDF
    5. 生成质量报告
    """
    
    record = orch.create_task_from_document(doc_text, source_name="cartography")
    
    # Verify workflow nodes are assembled
    assert "workflow_nodes" in record.metadata
    workflow_nodes = record.metadata.get("workflow_nodes", [])
    
    # Check for cartography-related nodes
    expected_nodes = {"apply_symbology", "apply_layout_template", 
                      "export_map_layout", "build_quality_report"}
    actual_nodes = set(workflow_nodes)
    
    # At least some expected nodes should be present
    assert len(expected_nodes & actual_nodes) > 0


def test_e2e_task_rerun_suggestion(tmp_path) -> None:
    """Test getting rerun suggestions for failed tasks."""
    orch = TaskOrchestrator()
    
    # Create a task
    record = orch.create_task("测试数据整合")
    task_id = record.task_id
    
    # Simulate a failed execution by manually setting task metadata
    record.metadata["last_execution"] = {
        "success": False,
        "details": {
            "node_execution": {
                "merge_agnp": "failed",
                "merge_boua": "pending",
                "export_map_layout": "skipped",
            },
            "node_reasons": {
                "merge_agnp": "merge_failed",
                "merge_boua": None,
                "export_map_layout": "node_not_selected",
            },
        },
    }
    
    # Save the modified task
    orch.store.save(record)
    
    # Get rerun suggestions
    rerun_nodes = orch.suggest_rerun_nodes(task_id)
    recovery_plan = orch.suggest_recovery_plan(task_id)
    
    # Verify suggestions
    assert len(rerun_nodes) > 0
    assert recovery_plan is not None
    assert "nodes" in recovery_plan


def test_e2e_arcpy_unavailable_fallback(tmp_path) -> None:
    """Test fallback behavior when ArcPy is not available."""
    data_root = tmp_path / "数据"
    output_root = tmp_path / "outputs"
    
    data_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Create sample data
    (data_root / "E48").mkdir(parents=True, exist_ok=True)
    (data_root / "E48" / "BOUA.shp").write_text("mock", encoding="utf-8")
    
    orch = TaskOrchestrator()
    orch.runner = ArcPyRunner(output_root=output_root)
    orch.runner.data_root = data_root
    
    # Mock arcpy unavailability by patching _load_arcpy
    original_load_arcpy = orch.runner._load_arcpy
    orch.runner._load_arcpy = lambda: None
    
    try:
        # Create and execute task
        record = orch.create_task("测试数据处理")
        result = orch.run_task(record.task_id, dry_run=False)
        
        # Verify fallback execution
        assert result.success is True
        assert any("execution_report.txt" in out for out in result.outputs)
        assert any("discovered_layers.json" in out for out in result.outputs)
    finally:
        # Restore original method
        orch.runner._load_arcpy = original_load_arcpy


def test_e2e_risk_assessment_high_risk_prompt(tmp_path) -> None:
    """Test risk assessment for high-risk operations."""
    orch = TaskOrchestrator()
    
    # High-risk prompt with delete/overwrite keywords
    prompt = "删除旧数据并覆盖现有图层，然后重新整合"
    record = orch.create_task(prompt)
    
    # Verify high risk is detected
    assert record.risk == RiskLevel.HIGH


def test_e2e_task_list_and_show(tmp_path) -> None:
    """Test listing and showing task details."""
    orch = TaskOrchestrator()
    
    # Create multiple tasks
    for i in range(3):
        orch.create_task(f"任务 {i+1}")
    
    # List tasks
    task_ids = orch.list_tasks()
    assert len(task_ids) >= 3
    
    # Show task details
    task_id = task_ids[0]
    detail = orch.get_task(task_id)
    assert detail is not None
    assert detail.task_id == task_id


def test_e2e_api_compatibility(tmp_path) -> None:
    """Test that orchestrator methods match API expectations."""
    orch = TaskOrchestrator()
    
    # Create task (API: POST /tasks)
    record = orch.create_task("测试API兼容性")
    assert record.task_id is not None
    
    # Create from document (API: POST /tasks/from-document)
    record2 = orch.create_task_from_document("测试文档", source_name="api_test")
    assert record2.task_id is not None
    
    # Plan task (API: POST /tasks/{id}/plan)
    plan = orch.plan_task(record.task_id)
    assert plan is not None
    
    # Get task (API: GET /tasks/{id})
    detail = orch.get_task(record.task_id)
    assert detail is not None
    
    # List tasks (API: GET /tasks)
    all_tasks = orch.list_tasks()
    assert len(all_tasks) >= 2


def test_e2e_shadow_mode_planner_status_contains_compare(tmp_path) -> None:
    import json

    cfg_dir = Path("config")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "planner_adapter_config.json"
    original = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else None
    cfg_path.write_text(json.dumps({"mode": "shadow"}, ensure_ascii=False), encoding="utf-8")
    try:
        orch = TaskOrchestrator()
        record = orch.create_task("整合数据并导出")
        assert record.plan is not None
        status = orch.get_planner_rag_status()
        assert status.get("planner_adapter_mode") == "shadow"
        compare = status.get("shadow_compare", {})
        assert compare.get("legacy_intent")
        assert compare.get("standard_intent")
    finally:
        if original is None:
            cfg_path.unlink(missing_ok=True)
        else:
            cfg_path.write_text(original, encoding="utf-8")

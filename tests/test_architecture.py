"""Tests for new tool architecture.

Tests the Claude Code-inspired architecture:
- Tool base class
- Tool registry
- Task state machine
- Skills system
- Context injection
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gis_cli.core import (
    Tool,
    ToolCategory,
    ToolContext,
    ToolResult,
    ToolRegistry,
    ValidationResult,
    PermissionResult,
    Task,
    TaskStatus,
    TaskManager,
    ExecutionContext,
    SystemContext,
    UserContext,
    WorkspaceContext,
)

from gis_cli.tools import (
    ScanLayersTool,
    MergeLayersTool,
    ProjectLayersTool,
    ExportMapTool,
    QualityCheckTool,
)

from gis_cli.skills import (
    Skill,
    SkillContext,
    SkillResult,
    SkillRegistry,
    ThematicMapSkill,
    DataIntegrationSkill,
    QualityAssuranceSkill,
)


# === Tool Tests ===

class TestToolRegistry:
    """Test tool registration and discovery."""
    
    def test_registry_singleton(self):
        r1 = ToolRegistry.instance()
        r2 = ToolRegistry.instance()
        assert r1 is r2
    
    def test_all_tools_registered(self):
        registry = ToolRegistry.instance()
        tools = registry.list_tools()
        tool_names = [t.name for t in tools]
        
        assert "scan_layers" in tool_names
        assert "merge_layers" in tool_names
        assert "project_layers" in tool_names
        assert "export_map" in tool_names
        assert "quality_check" in tool_names
    
    def test_get_tool_by_name(self):
        registry = ToolRegistry.instance()
        tool = registry.get("scan_layers")
        
        assert tool is not None
        assert tool.name == "scan_layers"
        # scan_layers is FILE_OPERATION category
        assert tool.category == ToolCategory.FILE_OPERATION
    
    def test_search_tools(self):
        registry = ToolRegistry.instance()
        results = registry.search("scan")
        
        assert len(results) >= 1
        assert any(t.name == "scan_layers" for t in results)
    
    def test_get_by_category(self):
        registry = ToolRegistry.instance()
        data_tools = registry.get_by_category(ToolCategory.DATA_INTEGRATION)
        
        assert isinstance(data_tools, list)
        assert all(t.category == ToolCategory.DATA_INTEGRATION for t in data_tools)


class TestToolResult:
    """Test ToolResult factory methods."""
    
    def test_ok_result(self):
        result = ToolResult.ok({"count": 5}, outputs=["out.shp"])
        
        assert result.success is True
        assert result.data == {"count": 5}
        assert result.outputs == ["out.shp"]
        assert result.error is None
    
    def test_fail_result(self):
        result = ToolResult.fail("Something went wrong", "test_error")
        
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.error_code == "test_error"
        assert result.data is None


class TestToolValidation:
    """Test tool input validation."""
    
    def test_validation_success(self):
        result = ValidationResult.success()
        
        assert result.valid is True
        # message is empty string on success, not None
        assert result.message == ""
    
    def test_validation_failure(self):
        result = ValidationResult.failure("Invalid input", error_code=1)
        
        assert result.valid is False
        assert result.message == "Invalid input"
        assert result.error_code == 1


class TestToolExecution:
    """Test tool execution."""
    
    def test_scan_layers_dry_run(self, tmp_path):
        # Create test directory with some files
        (tmp_path / "test.shp").touch()
        (tmp_path / "test.dbf").touch()
        (tmp_path / "test.shx").touch()
        
        tool = ScanLayersTool()
        context = ToolContext(
            working_directory=str(tmp_path),
            dry_run=True,
            arcpy_available=False
        )
        
        result = tool.call(
            tool.input_model(directory=str(tmp_path)),
            context
        )
        
        assert result.success is True
        assert result.data is not None
    
    def test_quality_check_dry_run(self, tmp_path):
        # Create test file
        test_file = tmp_path / "test.shp"
        test_file.touch()
        
        tool = QualityCheckTool()
        context = ToolContext(
            working_directory=str(tmp_path),
            dry_run=True,
            arcpy_available=False
        )
        
        result = tool.call(
            tool.input_model(input_path=str(test_file)),
            context
        )
        
        assert result.success is True


# === Task Tests ===

class TestTaskStatus:
    """Test task status transitions."""
    
    def test_all_statuses_exist(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.KILLED.value == "killed"


class TestTask:
    """Test task lifecycle."""
    
    def test_create_task(self):
        task = Task.create(
            prompt="Test prompt",
            description="A test task"
        )
        
        assert task.id is not None
        assert task.status == TaskStatus.PENDING
    
    def test_task_start(self):
        task = Task.create(prompt="Test", description="Test")
        task.start()
        
        assert task.status == TaskStatus.RUNNING
        assert task.started_at is not None
    
    def test_task_complete(self):
        task = Task.create(prompt="Test", description="Test")
        task.start()
        task.complete(outputs=["result.shp"])
        
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert "result.shp" in task.outputs
    
    def test_task_fail(self):
        task = Task.create(prompt="Test", description="Test")
        task.start()
        task.fail("Something went wrong")
        
        assert task.status == TaskStatus.FAILED
        assert task.error == "Something went wrong"


# === Skills Tests ===

class TestSkillRegistry:
    """Test skill registration and discovery."""
    
    def test_registry_singleton(self):
        r1 = SkillRegistry.instance()
        r2 = SkillRegistry.instance()
        assert r1 is r2
    
    def test_all_skills_registered(self):
        registry = SkillRegistry.instance()
        skills = registry.list_skills()
        skill_names = [s.name for s in skills]
        
        assert "thematic_map" in skill_names
        assert "data_integration" in skill_names
        assert "quality_assurance" in skill_names
    
    def test_get_skill_by_name(self):
        registry = SkillRegistry.instance()
        skill_cls = registry.get("thematic_map")
        
        assert skill_cls is not None
        assert skill_cls.name == "thematic_map"
    
    def test_search_skills(self):
        registry = SkillRegistry.instance()
        results = registry.search("map")
        
        assert len(results) >= 1
        assert any(s.name == "thematic_map" for s in results)


class TestSkillResult:
    """Test SkillResult factory methods."""
    
    def test_ok_result(self):
        result = SkillResult.ok(
            {"data": "value"},
            outputs=["out.pdf"],
            steps_completed=5,
            steps_total=5
        )
        
        assert result.success is True
        assert result.data == {"data": "value"}
        assert result.steps_completed == 5
    
    def test_fail_result(self):
        result = SkillResult.fail(
            "Skill failed",
            "skill_error",
            steps_completed=2,
            steps_total=5
        )
        
        assert result.success is False
        assert result.error == "Skill failed"
        assert result.steps_completed == 2


class TestSkillValidation:
    """Test skill input validation."""
    
    def test_thematic_map_validation_missing_sources(self):
        skill = ThematicMapSkill()
        valid, error = skill.validate_inputs({})
        
        assert valid is False
        assert "data_sources" in error
    
    def test_thematic_map_validation_success(self):
        skill = ThematicMapSkill()
        valid, error = skill.validate_inputs({
            "data_sources": ["dir1", "dir2"],
            "output_path": "out.pdf"
        })
        
        assert valid is True
        assert error is None
    
    def test_data_integration_required_inputs(self):
        skill = DataIntegrationSkill()
        required = skill.get_required_inputs()
        
        assert "source_dirs" in required
        assert "output_dir" in required


# === Context Tests ===

class TestSystemContext:
    """Test system context detection."""
    
    def test_detect_system_context(self):
        ctx = SystemContext.detect()
        
        assert ctx.os_name != ""
        assert ctx.python_version != ""
        assert ctx.home_dir != ""


class TestUserContext:
    """Test user context."""
    
    def test_load_user_context(self):
        ctx = UserContext.load()
        
        assert ctx.username != ""
        assert ctx.default_crs != ""
    
    def test_save_and_load_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        
        ctx = UserContext()
        ctx.default_crs = "WGS84"
        ctx.default_resolution = 150
        ctx.save(config_path)
        
        loaded = UserContext.load(config_path)
        assert loaded.default_crs == "WGS84"
        assert loaded.default_resolution == 150


class TestWorkspaceContext:
    """Test workspace context scanning."""
    
    def test_scan_empty_workspace(self, tmp_path):
        ctx = WorkspaceContext.scan(tmp_path)
        
        assert ctx.workspace_path == str(tmp_path)
        assert len(ctx.shapefiles) == 0
    
    def test_scan_workspace_with_files(self, tmp_path):
        # Create test files
        (tmp_path / "layer1.shp").touch()
        (tmp_path / "layer2.shp").touch()
        (tmp_path / "raster.tif").touch()
        
        ctx = WorkspaceContext.scan(tmp_path)
        
        assert len(ctx.shapefiles) == 2
        assert len(ctx.rasters) == 1


class TestExecutionContext:
    """Test combined execution context."""
    
    def test_build_execution_context(self, tmp_path):
        ctx = ExecutionContext.build(
            workspace=tmp_path,
            dry_run=True,
            verbose=True
        )
        
        assert ctx.dry_run is True
        assert ctx.verbose is True
        assert ctx.system.os_name != ""
    
    def test_context_summary(self, tmp_path):
        ctx = ExecutionContext.build(workspace=tmp_path)
        summary = ctx.summary()
        
        assert "System:" in summary
        assert "ArcPy:" in summary


# === Integration Tests ===

class TestToolSkillIntegration:
    """Test integration between tools and skills."""
    
    def test_skill_can_find_tools(self):
        """Skills should be able to locate tools from registry."""
        registry = ToolRegistry.instance()
        
        skill = DataIntegrationSkill()
        # Check that skills can use tools in their steps
        assert "scan_layers" in [t.name for t in registry.list_tools()]
        assert "quality_check" in [t.name for t in registry.list_tools()]
        assert "merge_layers" in [t.name for t in registry.list_tools()]
    
    def test_context_flows_to_tools(self, tmp_path):
        """Execution context should properly flow to tool calls."""
        exec_ctx = ExecutionContext.build(
            workspace=tmp_path,
            dry_run=True
        )
        
        tool_ctx = ToolContext(
            working_directory=exec_ctx.workspace.workspace_path,
            arcpy_available=exec_ctx.arcpy_available,
            dry_run=exec_ctx.dry_run
        )
        
        assert tool_ctx.working_directory == str(tmp_path)
        assert tool_ctx.dry_run is True

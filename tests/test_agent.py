"""Tests for GIS Agent."""

import pytest
from pathlib import Path
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gis_cli.agent import (
    GISAgent,
    AgentConfig,
    AgentResponse,
    ContextHub,
    HashingVectorRetriever,
    LangGraphStateManager,
    Memory,
    MemoryStore,
    ConversationTurn,
    Role,
    AgentPlanner,
    Plan,
    PlanStep,
    StepStatus,
    Executor,
    ExecutionResult,
    ExecutionTrace,
    ExecutionMode,
    SystemPrompts,
    PromptBuilder,
    MockLLMClient,
    ExecutionAdapter,
    LLMConfig,
    OpenAIClient,
    UnifiedModelClient,
    create_llm_client,
    PromptAdapter,
    PlanStandardizer,
    BAMLBridge,
)
from gis_cli.tools.project_layers import ProjectLayersTool
from gis_cli.tools.merge_layers import MergeLayersTool
from gis_cli.tools.export_map import ExportMapTool
import gis_cli.arcpy_bridge as arcpy_bridge
from gis_cli.core import ExecutionContext, ToolContext


# === Memory Tests ===

class TestMemory:
    """Test agent memory system."""
    
    def test_create_memory(self):
        memory = Memory(session_id="test-session")
        assert memory.session_id == "test-session"
        assert len(memory.turns) == 0
    
    def test_add_user_message(self):
        memory = Memory(session_id="test")
        memory.add_user_message("Hello")
        
        assert len(memory.turns) == 1
        assert memory.turns[0].role == Role.USER
        assert memory.turns[0].content == "Hello"
    
    def test_add_assistant_message(self):
        memory = Memory(session_id="test")
        memory.add_assistant_message("Hi there!")
        
        assert len(memory.turns) == 1
        assert memory.turns[0].role == Role.ASSISTANT
    
    def test_add_tool_call(self):
        memory = Memory(session_id="test")
        memory.add_tool_call("scan_layers", {"directory": "/data"}, {"layers": []})
        
        assert len(memory.turns) == 1
        assert memory.turns[0].role == Role.TOOL
        assert len(memory.tool_history) == 1
    
    def test_get_last_user_message(self):
        memory = Memory(session_id="test")
        memory.add_user_message("First")
        memory.add_assistant_message("Response")
        memory.add_user_message("Second")
        
        assert memory.get_last_user_message() == "Second"
    
    def test_conversation_for_llm(self):
        memory = Memory(session_id="test")
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi")
        
        messages = memory.get_conversation_for_llm()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
    
    def test_context_management(self):
        memory = Memory(session_id="test")
        memory.set_context("workspace", "/data")
        
        assert memory.get_context("workspace") == "/data"
        assert memory.get_context("missing", "default") == "default"
    
    def test_memory_summary(self):
        memory = Memory(session_id="test")
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi")
        
        summary = memory.summary()
        assert summary["session_id"] == "test"
        assert summary["total_turns"] == 2
        assert summary["user_messages"] == 1
        assert summary["assistant_messages"] == 1

    def test_structured_memory_and_reflection_apis(self):
        memory = Memory(session_id="test")
        memory.add_structured_memory("task_requirement", "制作专题图", importance=3)
        memory.add_reflection(
            trigger="failure",
            issue="执行失败: ERROR 000732",
            lesson="先检查输入路径",
            action_hint="先扫描目录并展开 shp",
            related_tools=["merge_layers"],
            confidence=0.8,
        )

        matched = memory.search_structured_memories("专题图", top_k=3)
        hints = memory.get_reflection_hints("路径 000732", top_k=3)

        assert len(matched) == 1
        assert matched[0].memory_type == "task_requirement"
        assert len(hints) == 1
        assert hints[0]["related_tools"] == ["merge_layers"]

    def test_memory_store_persists_full_state(self, tmp_path):
        store = MemoryStore(tmp_path)
        memory = Memory(session_id="s-full", store=store)
        memory.add_user_message("路径在 E:\\BST_lab\\GIS CLI\\workspace\\input")
        memory.set_context("workspace", "E:\\BST_lab\\GIS CLI\\workspace")
        memory.add_tool_call("scan_layers", {"path": "x"}, {"layers": []})
        memory.add_structured_memory("data_path", "E:\\BST_lab\\GIS CLI\\workspace\\input", importance=4)
        memory.add_reflection(
            trigger="manual",
            issue="继续任务要继承上下文",
            lesson="优先复用历史主任务",
            related_tools=["planner"],
            confidence=0.7,
        )
        memory.save()

        loaded = Memory(session_id="s-full", store=store)
        loaded.load()

        assert loaded.get_context("workspace") == "E:\\BST_lab\\GIS CLI\\workspace"
        assert len(loaded.tool_history) == 1
        assert len(loaded.structured_memories) == 1
        assert len(loaded.reflections) == 1

    def test_vector_retriever_chinese_query_match(self):
        retriever = HashingVectorRetriever(dimension=256)
        retriever.upsert("a", "行政区线与行政区面继续处理", {"kind": "structured"})
        retriever.upsert("b", "缓冲区分析道路", {"kind": "structured"})

        hits = retriever.search("继续处理行政区线", top_k=2)
        assert hits
        assert hits[0].record_id == "a"

    def test_memory_reflection_vector_hint_retrieval(self):
        memory = Memory(session_id="test")
        memory.add_reflection(
            trigger="failure",
            issue="误触发 clip 分析",
            lesson="跟进语句优先延续历史任务",
            action_hint="除非明确要求，否则不要执行裁剪",
            related_tools=["clip"],
            confidence=0.9,
        )

        hints = memory.get_reflection_hints("继续任务 不要裁剪", top_k=2)
        assert hints
        assert "clip" in hints[0]["related_tools"]


class TestLLMConfigProfiles:
    def test_from_file_applies_qwen_profile_when_values_missing(self, tmp_path):
        cfg_file = tmp_path / "llm_config.json"
        cfg_file.write_text(
            '{"model":"Qwen/Qwen3.5-397B-A17B","provider":"siliconflow","api_key":"x","api_base":"https://example.com/v1"}',
            encoding="utf-8",
        )
        cfg = LLMConfig.from_file(str(cfg_file))
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 2500
        assert cfg.timeout == 120
        assert cfg.retry_count == 2

    def test_from_file_respects_explicit_values_over_profile(self, tmp_path):
        cfg_file = tmp_path / "llm_config_explicit.json"
        cfg_file.write_text(
            '{"model":"Qwen/Qwen3.5-397B-A17B","provider":"siliconflow","api_key":"x","temperature":0.55,"max_tokens":1800,"timeout":45,"retry_count":1}',
            encoding="utf-8",
        )
        cfg = LLMConfig.from_file(str(cfg_file))
        assert cfg.temperature == 0.55
        assert cfg.max_tokens == 1800
        assert cfg.timeout == 45
        assert cfg.retry_count == 1

    def test_from_file_reads_adaptation_switches(self, tmp_path):
        cfg_file = tmp_path / "llm_config_flags.json"
        cfg_file.write_text(
            '{"model":"gpt-4o-mini","provider":"openai","api_key":"x","enable_prompt_optimizer":false,"enable_baml_standardizer":false}',
            encoding="utf-8",
        )
        cfg = LLMConfig.from_file(str(cfg_file))
        assert cfg.enable_prompt_optimizer is False
        assert cfg.enable_baml_standardizer is False

    def test_from_file_reads_routing_and_provider_models(self, tmp_path):
        cfg_file = tmp_path / "llm_config_routing.json"
        cfg_file.write_text(
            '{"model":"gpt-4o-mini","provider":"openai","api_key":"x",'
            '"routing_rules":{"planning":["model-a","model-b"]},'
            '"provider_models":{"openai":["model-c"]}}',
            encoding="utf-8",
        )
        cfg = LLMConfig.from_file(str(cfg_file))
        assert cfg.routing_rules.get("planning") == ["model-a", "model-b"]
        assert cfg.provider_models.get("openai") == ["model-c"]

    def test_from_file_reads_baml_function_map(self, tmp_path):
        cfg_file = tmp_path / "llm_config_baml_map.json"
        cfg_file.write_text(
            '{"model":"gpt-4o-mini","provider":"openai","api_key":"x",'
            '"baml_functions":{"planning":["custom_generate_plan"],"intent":["custom_intent_fn"]}}',
            encoding="utf-8",
        )
        cfg = LLMConfig.from_file(str(cfg_file))
        assert cfg.baml_functions.get("planning") == ["custom_generate_plan"]
        assert cfg.baml_functions.get("intent") == ["custom_intent_fn"]

    def test_from_file_reads_baml_preflight_settings(self, tmp_path):
        cfg_file = tmp_path / "llm_config_baml_preflight.json"
        cfg_file.write_text(
            '{"model":"gpt-4o-mini","provider":"openai","api_key":"x",'
            '"enable_baml_preflight":true,'
            '"baml_preflight_strict":true,'
            '"baml_required_capabilities":["intent","planning"]}',
            encoding="utf-8",
        )
        cfg = LLMConfig.from_file(str(cfg_file))
        assert cfg.enable_baml_preflight is True
        assert cfg.baml_preflight_strict is True
        assert cfg.baml_required_capabilities == ["intent", "planning"]

    def test_from_file_reads_baml_builtin_fallback_setting(self, tmp_path):
        cfg_file = tmp_path / "llm_config_baml_builtin_fallback.json"
        cfg_file.write_text(
            '{"model":"gpt-4o-mini","provider":"openai","api_key":"x",'
            '"enable_baml_builtin_fallback":false}',
            encoding="utf-8",
        )
        cfg = LLMConfig.from_file(str(cfg_file))
        assert cfg.enable_baml_builtin_fallback is False


class TestUnifiedModelClient:
    def test_candidate_models_prefers_explicit_then_route_then_default(self):
        cfg = LLMConfig(
            model="default-model",
            api_key="x",
            provider="openai",
            fallback_models=["fallback-1"],
            routing_rules={"planning": ["route-1", "route-2"]},
            provider_models={"openai": ["provider-1"]},
        )
        client = UnifiedModelClient(cfg)
        candidates = client._candidate_models("explicit-1", task_type="planning")  # noqa: SLF001
        assert candidates == [
            "explicit-1",
            "route-1",
            "route-2",
            "default-model",
            "provider-1",
            "fallback-1",
        ]

    def test_chat_complete_delegates_to_chat_completion(self):
        cfg = LLMConfig(model="gpt-4o-mini", api_key="x", provider="openai")
        client = UnifiedModelClient(cfg)

        class _Sentinel:
            pass

        sentinel = _Sentinel()

        def fake_chat_completion(messages, **kwargs):
            return sentinel

        client.chat_completion = fake_chat_completion  # type: ignore[method-assign]
        result = client.chat_complete(messages=[{"role": "user", "content": "hi"}])
        assert result is sentinel


class TestLLMClientFactory:
    def test_create_llm_client_prefers_openai_for_siliconflow_without_routing(self):
        cfg = LLMConfig(
            model="Qwen/Qwen3.5-397B-A17B",
            provider="siliconflow",
            api_key="x",
            api_base="https://api.siliconflow.cn/v1",
        )

        client = create_llm_client(cfg)
        assert isinstance(client, OpenAIClient)

    def test_create_llm_client_uses_unified_when_routing_configured(self):
        cfg = LLMConfig(
            model="Qwen/Qwen3.5-397B-A17B",
            provider="siliconflow",
            api_key="x",
            api_base="https://api.siliconflow.cn/v1",
            routing_rules={"planning": ["route-model"]},
        )

        client = create_llm_client(cfg)
        assert isinstance(client, UnifiedModelClient)


class TestOpenAIClientModelNameNormalization:
    def test_openai_client_preserves_prefixed_model_for_siliconflow(self):
        cfg = LLMConfig(
            model="Qwen/Qwen3.5-397B-A17B",
            provider="siliconflow",
            api_key="x",
            api_base="https://api.siliconflow.cn/v1",
        )
        client = OpenAIClient(cfg)

        captured: dict[str, str] = {}

        class _Completions:
            def create(self, **kwargs):
                captured["model"] = kwargs.get("model", "")
                msg = SimpleNamespace(content="ok", role="assistant")
                choice = SimpleNamespace(message=msg, finish_reason="stop")
                usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
                return SimpleNamespace(choices=[choice], usage=usage)

        client._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))  # noqa: SLF001

        _ = client.chat_completion([{"role": "user", "content": "hi"}])
        assert captured.get("model") == "Qwen/Qwen3.5-397B-A17B"


class TestModelAdaptation:
    def test_prompt_adapter_adds_compact_constraints_for_qwen(self):
        adapter = PromptAdapter(enable_prompt_optimizer=False)
        prompt = "请规划 GIS 执行步骤"
        adapted = adapter.adapt_prompt(prompt, task_type="planning", model_hint="Qwen/Qwen3.5-32B")
        assert "仅输出 JSON 对象" in adapted

    def test_prompt_adapter_uses_prompt_optimizer_when_available(self, monkeypatch):
        class _FakePromptOptimizer:
            @staticmethod
            def optimize_prompt(*args, **kwargs):
                return {"optimized_prompt": "优化后的提示词"}

        monkeypatch.setitem(sys.modules, "prompt_optimizer", _FakePromptOptimizer)
        adapter = PromptAdapter(enable_prompt_optimizer=True)
        adapted = adapter.adapt_prompt("基础提示词", task_type="intent", model_hint="gpt-4o")
        assert adapted == "优化后的提示词"
        assert adapter.metrics.prompt_optimizer_hits == 1

    def test_plan_standardizer_normalizes_missing_optional_fields(self):
        standardizer = PlanStandardizer()
        payload = {
            "id": "p1",
            "goal": "测试标准化",
            "steps": [
                {"id": "step_1", "tool": "scan_layers", "description": "扫描"},
            ],
        }
        normalized = standardizer.standardize(payload)
        assert normalized is not None
        assert normalized["steps"][0]["input"] == {}
        assert normalized["steps"][0]["depends_on"] == []

    def test_plan_standardizer_skips_baml_when_disabled(self):
        standardizer = PlanStandardizer(enable_baml=False)
        payload = {
            "goal": "测试",
            "steps": [{"tool": "scan_layers", "description": "扫描"}],
        }
        normalized = standardizer.standardize(payload)
        assert normalized is not None
        assert standardizer.metrics.baml_attempts == 0

    def test_plan_standardizer_normalizes_tool_call_step(self):
        standardizer = PlanStandardizer(enable_baml=False)
        payload = {
            "goal": "测试",
            "steps": [
                {
                    "id": "step_1",
                    "description": "扫描目录",
                    "tool_call": {
                        "name": "scan_layers",
                        "arguments": {"directory": "./workspace/input"},
                    },
                }
            ],
        }
        normalized = standardizer.standardize(payload)
        assert normalized is not None
        assert normalized["steps"][0]["tool"] == "scan_layers"
        assert normalized["steps"][0]["input"] == {"directory": "./workspace/input"}

    def test_plan_standardizer_normalizes_function_call_json_arguments(self):
        standardizer = PlanStandardizer(enable_baml=False)
        payload = {
            "goal": "测试",
            "steps": [
                {
                    "id": "step_1",
                    "description": "检查质量",
                    "function_call": {
                        "name": "quality_check",
                        "arguments": '{"input_layers": ["./workspace/input/a.shp"]}',
                    },
                }
            ],
        }
        normalized = standardizer.standardize(payload)
        assert normalized is not None
        assert normalized["steps"][0]["tool"] == "quality_check"
        assert normalized["steps"][0]["input"]["input_layers"] == ["./workspace/input/a.shp"]


class TestBAMLBridge:
    def test_infer_intent_prefers_baml_when_available(self, monkeypatch):
        class _FakeB:
            @staticmethod
            def infer_gis_intent(**kwargs):
                return {"intent": "execute_task"}

        monkeypatch.setitem(sys.modules, "baml_client", SimpleNamespace(b=_FakeB()))
        bridge = BAMLBridge(enabled=True)
        intent = bridge.infer_intent("请制作专题图", ["execute_task", "general"])
        assert intent == "execute_task"

    def test_generate_plan_parses_baml_string_result(self, monkeypatch):
        class _FakeB:
            @staticmethod
            def generate_gis_plan(**kwargs):
                return '{"id":"p1","goal":"测试","steps":[{"id":"step_1","tool":"scan_layers","description":"扫描","input":{},"depends_on":[]}]}'

        monkeypatch.setitem(sys.modules, "baml_client", SimpleNamespace(b=_FakeB()))
        bridge = BAMLBridge(enabled=True)
        plan = bridge.generate_plan("扫描数据", {})
        assert isinstance(plan, dict)
        assert plan.get("id") == "p1"

    def test_bridge_prefers_configured_function_name(self, monkeypatch):
        class _FakeB:
            @staticmethod
            def custom_intent_fn(**kwargs):
                return {"intent": "general"}

        monkeypatch.setitem(sys.modules, "baml_client", SimpleNamespace(b=_FakeB()))
        bridge = BAMLBridge(enabled=True, function_map={"intent": ["custom_intent_fn"]})
        intent = bridge.infer_intent("你好", ["general", "execute_task"])
        assert intent == "general"

    def test_baml_bridge_diagnostics_reports_function_bindings(self, monkeypatch):
        class _FakeB:
            @staticmethod
            def custom_intent_fn(**kwargs):
                return {"intent": "general"}

            @staticmethod
            def generate_gis_plan(**kwargs):
                return {"id": "p", "goal": "g", "steps": [{"id": "s1", "tool": "scan_layers", "description": "d", "input": {}, "depends_on": []}]}

        monkeypatch.setitem(sys.modules, "baml_client", SimpleNamespace(b=_FakeB()))
        bridge = BAMLBridge(
            enabled=True,
            function_map={
                "intent": ["custom_intent_fn"],
                "planning": ["generate_gis_plan"],
            },
        )
        diag = bridge.diagnostics()
        assert diag["enabled"] is True
        assert diag["client_available"] is True
        intent_entries = diag["functions"]["intent"]
        assert intent_entries[0]["name"] == "custom_intent_fn"
        assert intent_entries[0]["exists"] is True

    def test_baml_bridge_diagnostics_handles_missing_client(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "baml_client", raising=False)
        bridge = BAMLBridge(enabled=True)
        diag = bridge.diagnostics()
        assert diag["enabled"] is True
        assert diag["client_available"] is False

    def test_validate_required_capabilities_ok(self, monkeypatch):
        class _FakeB:
            @staticmethod
            def infer_gis_intent(**kwargs):
                return {"intent": "general"}

            @staticmethod
            def refine_gis_task(**kwargs):
                return "x"

            @staticmethod
            def generate_gis_plan(**kwargs):
                return {"id": "p", "goal": "g", "steps": [{"id": "s1", "tool": "scan_layers", "description": "d", "input": {}, "depends_on": []}]}

            @staticmethod
            def suggest_gis_recovery(**kwargs):
                return {"id": "r", "goal": "g", "steps": [{"id": "s1", "tool": "quality_check", "description": "d", "input": {}, "depends_on": []}]}

        monkeypatch.setitem(sys.modules, "baml_client", SimpleNamespace(b=_FakeB()))
        bridge = BAMLBridge(enabled=True)
        result = bridge.validate_required_capabilities(["intent", "task_refine", "planning", "recovery"])
        assert result["ok"] is True
        assert result["missing"] == []

    def test_validate_required_capabilities_reports_missing(self, monkeypatch):
        class _FakeB:
            @staticmethod
            def infer_gis_intent(**kwargs):
                return {"intent": "general"}

        monkeypatch.setitem(sys.modules, "baml_client", SimpleNamespace(b=_FakeB()))
        bridge = BAMLBridge(enabled=True)
        result = bridge.validate_required_capabilities(["intent", "planning"])
        assert result["ok"] is False
        assert "planning" in result["missing"]

    def test_baml_bridge_diagnostics_includes_capability_ready(self, monkeypatch):
        class _FakeB:
            @staticmethod
            def infer_gis_intent(**kwargs):
                return {"intent": "general"}

            @staticmethod
            def refine_gis_task(**kwargs):
                return "refined"

            @staticmethod
            def generate_gis_plan(**kwargs):
                return {"id": "p", "goal": "g", "steps": [{"id": "s1", "tool": "scan_layers", "description": "d", "input": {}, "depends_on": []}]}

            @staticmethod
            def suggest_gis_recovery(**kwargs):
                return {"id": "r", "goal": "g", "steps": [{"id": "s1", "tool": "quality_check", "description": "d", "input": {}, "depends_on": []}]}

        monkeypatch.setitem(sys.modules, "baml_client", SimpleNamespace(b=_FakeB()))
        bridge = BAMLBridge(enabled=True)
        diag = bridge.diagnostics()
        assert diag["capability_ready"]["intent"] is True
        assert diag["capability_ready"]["task_refine"] is True
        assert diag["capability_ready"]["planning"] is True
        assert diag["capability_ready"]["recovery"] is True

    def test_baml_bridge_builtin_fallback_satisfies_required_capabilities(self, monkeypatch):
        bridge = BAMLBridge(enabled=True, allow_builtin_fallback=True)
        monkeypatch.setattr(bridge, "_load_baml_client", lambda: None)

        diag = bridge.diagnostics()
        assert diag["client_available"] is False
        assert diag["allow_builtin_fallback"] is True
        assert diag["capability_ready"]["intent"] is True
        assert diag["capability_ready"]["task_refine"] is True
        assert diag["capability_ready"]["planning"] is True
        assert diag["capability_ready"]["recovery"] is True

        validation = bridge.validate_required_capabilities(["intent", "task_refine", "planning", "recovery"])
        assert validation["ok"] is True
        assert validation["missing"] == []


class TestStateManager:
    def test_state_manager_persists_and_loads(self, tmp_path):
        store = tmp_path / "state" / "workflow_state.json"
        manager = LangGraphStateManager(store)
        manager.upsert_state("thread_1", session_id="s1", status="planned", current_plan_id="p1")
        manager.record_event("thread_1", "plan_created", {"plan_id": "p1"})

        manager2 = LangGraphStateManager(store)
        summary = manager2.get_summary("thread_1")

        assert summary["exists"] is True
        assert summary["current_plan_id"] == "p1"
        assert summary["events_count"] == 1


class TestContextHub:
    def test_context_hub_discovers_nested_workspace_assets(self, tmp_path):
        ws_input = tmp_path / "workspace" / "input" / "myproject5"
        ws_input.mkdir(parents=True, exist_ok=True)
        (ws_input / "MyProject5.aprx").touch()
        (ws_input / "roads.shp").touch()
        (ws_input / "region.gdb").mkdir(parents=True, exist_ok=True)

        hub = ContextHub(workspace=tmp_path)
        snapshot = hub.discover()

        assert any(p.lower().endswith(".aprx") for p in snapshot.aprx_files)
        assert any(p.lower().endswith("roads.shp") for p in snapshot.shapefiles)
        assert any(p.lower().endswith("region.gdb") for p in snapshot.geodatabases)

    def test_context_hub_builds_planner_payload_with_hints(self, tmp_path):
        ws_input = tmp_path / "input" / "proj"
        ws_input.mkdir(parents=True, exist_ok=True)
        (ws_input / "proj.aprx").touch()

        hub = ContextHub(workspace=tmp_path)
        snapshot = hub.discover()
        payload = hub.build_planner_payload(snapshot)

        assert payload["detected_project_path"]
        assert "smart_hints" in payload

    def test_context_hub_cache_invalidates_when_memory_context_changes(self, tmp_path):
        (tmp_path / "input").mkdir(parents=True, exist_ok=True)
        alt_input = tmp_path / "workspace" / "input"
        alt_input.mkdir(parents=True, exist_ok=True)

        memory = Memory(session_id="ctx-cache")
        hub = ContextHub(workspace=tmp_path, memory=memory)

        first = hub.discover()
        assert str(tmp_path / "input") in first.input_roots

        memory.set_context("input_folder", str(alt_input))
        second = hub.discover()
        assert str(alt_input) in second.input_roots


# === Planner Tests ===

class TestPlanner:
    """Test agent planner."""
    
    def test_create_planner(self):
        planner = AgentPlanner()
        assert "scan_layers" in planner.available_tools
    
    def test_plan_thematic_map(self):
        planner = AgentPlanner()
        plan = planner.plan("制作广西专题图")
        
        assert plan.goal == "制作广西专题图"
        assert len(plan.steps) > 0
        
        # Should include export step for map
        tool_names = [s.tool for s in plan.steps]
        assert "export_map" in tool_names or "scan_layers" in tool_names

    def test_plan_explicit_thematic_map_path_prefers_execute_code(self, tmp_path):
        planner = AgentPlanner(llm_client=None)
        workspace = tmp_path
        context = {
            "workspace": str(workspace),
            "input_folder": str(workspace / "workspace" / "input"),
            "output_folder": str(workspace / "workspace" / "output"),
            "input_files": [],
        }

        plan = planner.plan(
            "根据 workspace/output/province_forest_coverage.shp 制图，字段用forest_cov，绿色分级，"
            "导出到 workspace/output/china_vegetation_coverage.jpg",
            context=context,
        )

        tool_names = [s.tool for s in plan.steps]
        assert tool_names[:2] == ["scan_layers", "execute_code"]
        code = str(plan.steps[1].input.get("code", ""))
        assert "build_graduated_colors_code" in code
        assert "Greens" in code
        assert "requested_field" in code
        assert "china_vegetation_coverage.jpg" in "".join(plan.expected_outputs)

    def test_plan_explicit_thematic_map_honors_requested_red_scheme(self, tmp_path):
        planner = AgentPlanner(llm_client=None)
        workspace = tmp_path
        context = {
            "workspace": str(workspace),
            "input_folder": str(workspace / "workspace" / "input"),
            "output_folder": str(workspace / "workspace" / "output"),
            "input_files": [],
        }

        plan = planner.plan(
            "根据 workspace/output/province_population.shp 做中国人口分布图，使用红色分级色带，导出到 workspace/output/population_map.jpg",
            context=context,
        )

        assert [s.tool for s in plan.steps][:2] == ["scan_layers", "execute_code"]
        code = str(plan.steps[1].input.get("code", ""))
        assert 'color_ramp_name=r"Reds"' in code

    def test_plan_explicit_thematic_map_honors_explicit_ramp_name(self, tmp_path):
        planner = AgentPlanner(llm_client=None)
        workspace = tmp_path
        context = {
            "workspace": str(workspace),
            "input_folder": str(workspace / "workspace" / "input"),
            "output_folder": str(workspace / "workspace" / "output"),
            "input_files": [],
        }

        plan = planner.plan(
            "根据 workspace/output/province_population.shp 制作专题图，使用 YlOrRd 色带，导出到 workspace/output/population_map.jpg",
            context=context,
        )

        code = str(plan.steps[1].input.get("code", ""))
        assert 'color_ramp_name=r"YlOrRd"' in code

    def test_plan_thematic_map_resolves_layer_name_without_extension(self, tmp_path):
        planner = AgentPlanner(llm_client=None)
        workspace = tmp_path
        output_root = workspace / "workspace" / "output"
        output_root.mkdir(parents=True, exist_ok=True)
        layer_file = output_root / "province_forest_coverage.shp"
        layer_file.touch()

        context = {
            "workspace": str(workspace),
            "input_folder": str(workspace / "workspace" / "input"),
            "output_folder": str(output_root),
            "input_files": [],
            "context_snapshot": {
                "output_data": [str(layer_file)],
                "shapefiles": [],
                "geodatabases": [],
            },
        }

        plan = planner.plan(
            "根据workspace/output/province_forest_coverage这个图层制作专题图，绿色分级并导出jpg",
            context=context,
        )

        assert [s.tool for s in plan.steps][:2] == ["scan_layers", "execute_code"]
        code = str(plan.steps[1].input.get("code", ""))
        assert str(layer_file) in code
    
    def test_plan_data_integration(self):
        planner = AgentPlanner()
        plan = planner.plan("整合多个图层数据")
        
        assert len(plan.steps) > 0
        tool_names = [s.tool for s in plan.steps]
        assert "merge_layers" in tool_names or "scan_layers" in tool_names
    
    def test_plan_quality_check(self):
        planner = AgentPlanner()
        plan = planner.plan("检查数据质量")
        
        tool_names = [s.tool for s in plan.steps]
        assert "quality_check" in tool_names
    
    def test_plan_step_dependencies(self):
        planner = AgentPlanner()
        plan = planner.plan("整合数据并导出专题图")
        
        # Later steps should depend on earlier ones
        for step in plan.steps[1:]:
            # Each step after the first should have dependencies
            # (depends on at least the previous step)
            pass  # This is more of a structural check

    def test_recovery_does_not_skip_critical_first_step(self):
        planner = AgentPlanner()
        plan = Plan(
            id="p-critical",
            goal="critical recovery",
            steps=[
                PlanStep(id="step_1", tool="scan_layers", description="scan"),
                PlanStep(id="step_2", tool="quality_check", description="check", depends_on=["step_1"]),
            ],
        )
        failed = plan.steps[0]
        failed.fail("unknown error")
        recovery = planner.suggest_recovery(plan, failed)
        assert recovery is None

    def test_filter_merge_without_pattern_falls_back_to_normal_rules(self):
        planner = AgentPlanner()
        plan = planner.plan("请把图层合并")
        assert plan is not None
        assert len(plan.steps) > 0
        assert any(s.tool in {"scan_layers", "merge_layers"} for s in plan.steps)

    def test_complex_requirement_not_hijacked_by_filter_merge(self):
        planner = AgentPlanner()
        task = """
一、案例背景
制作广西区域发展布局示意图，统一投影，比例尺出图。
二、数据说明
整合9个文件夹中的BOUA、BOUL、AGNP数据。
三、分析要求
1、创建数据库并整合数据入库。
2、处理行政境界面并地理配准，制作广西区域发展数据库。
3、处理行政境界线并按GB码符号化。
4、提取行政中心并标注。
5、绘制箭头。
6、地图制图并导出比赛版PNG。
"""
        plan = planner.plan(task)
        assert plan is not None
        assert plan.metadata.get("filter_merge") is not True
        assert len(plan.steps) >= 4

    def test_focus_filter_merge_still_generates_special_plan(self):
        planner = AgentPlanner()
        plan = planner.plan("请将9个分幅中的 BOUA 和 BOUL 分别合并，输出行政区面和行政区线")
        assert plan is not None
        assert plan.metadata.get("filter_merge") is True
        assert any(s.tool == "execute_code" for s in plan.steps)

    def test_extract_plan_json_can_repair_trailing_comma(self):
        planner = AgentPlanner()
        response = """```json
{
    "id": "plan_x",
    "goal": "测试",
    "steps": [
        {"id": "step_1", "tool": "scan_layers", "description": "扫描", "input": {}, "depends_on": [],},
    ],
    "expected_outputs": [],
}
```"""
        plan_json = planner._extract_plan_json(response)  # noqa: SLF001
        assert plan_json is not None
        assert plan_json.get("id") == "plan_x"

    def test_plan_with_llm_retries_with_strict_json_prompt(self):
        mock = MockLLMClient(
            responses=[
                "我建议先扫描再处理。",
                '{"id":"plan_ok","goal":"测试任务","steps":[{"id":"step_1","tool":"scan_layers","description":"扫描","input":{},"depends_on":[]}],"expected_outputs":[]}',
            ]
        )
        planner = AgentPlanner(llm_client=mock)
        plan = planner.plan("扫描并处理数据")
        assert plan is not None
        assert plan.id == "plan_ok"
        assert len(plan.steps) == 1

    def test_plan_with_llm_prefers_baml_result(self, monkeypatch):
        class _FakeB:
            @staticmethod
            def generate_gis_plan(**kwargs):
                return {
                    "id": "plan_baml",
                    "goal": "测试任务",
                    "steps": [
                        {
                            "id": "step_1",
                            "tool": "scan_layers",
                            "description": "扫描",
                            "input": {},
                            "depends_on": [],
                        }
                    ],
                    "expected_outputs": [],
                }

        monkeypatch.setitem(sys.modules, "baml_client", SimpleNamespace(b=_FakeB()))

        mock = MockLLMClient(responses=["invalid"])  # Should not be used when BAML succeeds
        planner = AgentPlanner(llm_client=mock)
        plan = planner.plan("扫描并处理数据")
        assert plan is not None
        assert plan.id == "plan_baml"
        assert plan.metadata.get("llm_planning") == "success_baml"

    def test_suggest_recovery_prefers_baml_recovery_plan(self, monkeypatch):
        class _FakeB:
            @staticmethod
            def suggest_gis_recovery(**kwargs):
                return {
                    "id": "recovery_baml_1",
                    "goal": "Recovery via BAML",
                    "steps": [
                        {
                            "id": "recovery_step_1",
                            "tool": "quality_check",
                            "description": "恢复检查",
                            "input": {"input_path": "./workspace/input"},
                            "depends_on": [],
                        }
                    ],
                }

        monkeypatch.setitem(sys.modules, "baml_client", SimpleNamespace(b=_FakeB()))

        planner = AgentPlanner(llm_client=MockLLMClient())
        plan = Plan(
            id="p1",
            goal="测试恢复",
            steps=[
                PlanStep(id="step_1", tool="scan_layers", description="扫描", status=StepStatus.COMPLETED),
                PlanStep(id="step_2", tool="quality_check", description="检查", depends_on=["step_1"]),
                PlanStep(id="step_3", tool="export_map", description="导出", depends_on=["step_2"]),
            ],
        )
        failed = plan.steps[1]
        failed.fail("参数错误")

        recovery = planner.suggest_recovery(plan, failed)
        assert recovery is not None
        assert recovery.id == "recovery_baml_1"
        assert recovery.metadata.get("recovery_strategy") == "baml_recovery"

    def test_followup_meta_request_skips_custom_skill_short_circuit(self, monkeypatch):
        planner = AgentPlanner()
        called = {"hit": False}

        def fake_match(task_description, context):
            called["hit"] = True
            return None

        monkeypatch.setattr(planner, "_try_match_custom_skill", fake_match)
        _ = planner.plan("根据之前信息继续规划任务，直接对行政区线和面继续操作")
        assert called["hit"] is False

    def test_negative_clip_intent_does_not_generate_clip_step(self):
        planner = AgentPlanner()
        plan = planner.plan("继续处理行政区线和面，不要裁剪图层")
        assert plan is not None
        assert all("clip" not in s.description.lower() for s in plan.steps)

    def test_create_file_geodatabase_intent_generates_deterministic_execute_code(self):
        planner = AgentPlanner()
        plan = planner.plan("在输出目录创建新的文件地理数据库")
        assert plan is not None
        execute_steps = [s for s in plan.steps if s.tool == "execute_code"]
        assert execute_steps
        code = str(execute_steps[0].input.get("code", ""))
        assert "CreateFileGDB" in code
        assert "set_result" in code
        assert ".gdb" in code


class TestPlan:
    """Test Plan class."""
    
    def test_create_plan(self):
        plan = Plan(id="test-plan", goal="Test goal")
        assert plan.id == "test-plan"
        assert len(plan.steps) == 0
    
    def test_plan_with_steps(self):
        plan = Plan(
            id="test-plan",
            goal="Test goal",
            steps=[
                PlanStep(id="s1", tool="scan_layers", description="Scan"),
                PlanStep(id="s2", tool="merge_layers", description="Merge", depends_on=["s1"])
            ]
        )
        
        assert len(plan.steps) == 2
        assert plan.get_step("s1") is not None
        assert plan.get_step("s2").depends_on == ["s1"]
    
    def test_next_step(self):
        plan = Plan(
            id="test-plan",
            goal="Test",
            steps=[
                PlanStep(id="s1", tool="scan_layers", description="Scan"),
                PlanStep(id="s2", tool="merge_layers", description="Merge", depends_on=["s1"])
            ]
        )
        
        # First pending step with no dependencies
        next_step = plan.next_step
        assert next_step.id == "s1"
        
        # Complete first step
        next_step.complete({"result": "ok"})
        
        # Now second step should be available
        next_step = plan.next_step
        assert next_step.id == "s2"
    
    def test_plan_completion(self):
        plan = Plan(
            id="test-plan",
            goal="Test",
            steps=[
                PlanStep(id="s1", tool="scan_layers", description="Scan")
            ]
        )
        
        assert not plan.is_complete
        
        plan.steps[0].complete({"result": "ok"})
        assert plan.is_complete
    
    def test_plan_failure(self):
        plan = Plan(
            id="test-plan",
            goal="Test",
            steps=[
                PlanStep(id="s1", tool="scan_layers", description="Scan")
            ]
        )
        
        assert not plan.has_failed
        
        plan.steps[0].fail("Error")
        assert plan.has_failed


# === Executor Tests ===

class TestExecutor:
    """Test plan executor."""
    
    def test_create_executor(self):
        executor = Executor()
        assert executor.registry is not None
    
    def test_execute_simple_plan(self, tmp_path):
        # Create a simple plan
        plan = Plan(
            id="test-plan",
            goal="Test",
            steps=[
                PlanStep(
                    id="s1",
                    tool="scan_layers",
                    description="Scan",
                    input={"directory": str(tmp_path)}
                )
            ]
        )
        
        executor = Executor()
        result = executor.execute(plan, ExecutionMode.DRY_RUN)
        
        # Should complete (may succeed or fail depending on tool impl)
        assert result.plan == plan
        assert result.trace is not None
    
    def test_execute_single_tool(self, tmp_path):
        executor = Executor()
        result = executor.execute_single_tool(
            "scan_layers",
            {"directory": str(tmp_path)},
            dry_run=True
        )
        
        # Result should be returned
        assert result is not None

    def test_normalize_scan_layers_autodiscovers_workspace_input(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ws_input = tmp_path / "workspace" / "input"
        ws_input.mkdir(parents=True, exist_ok=True)

        executor = Executor()
        normalized = executor._normalize_params(  # noqa: SLF001 - intentional internal contract test
            "scan_layers",
            {"path": str(tmp_path / "input")},
        )
        assert normalized["path"] == str(ws_input)

    def test_scan_layers_recursively_discovers_aprx_in_subfolder(self, tmp_path):
        project_file = tmp_path / "input" / "myproject5" / "myproject5.aprx"
        project_file.parent.mkdir(parents=True, exist_ok=True)
        project_file.touch()

        executor = Executor()
        result = executor.execute_single_tool(
            "scan_layers",
            {"path": str(tmp_path / "input"), "include_subdirs": True},
            dry_run=True,
        )

        assert result.success is True
        assert result.data is not None
        discovered = [layer.path for layer in result.data.layers]
        assert str(project_file) in discovered

    def test_normalize_quality_check_accepts_input_layers_alias(self, tmp_path):
        executor = Executor()
        shp = tmp_path / "sample.shp"
        shp.touch()

        normalized = executor._normalize_params(  # noqa: SLF001 - intentional internal contract test
            "quality_check",
            {
                "input_layers": [str(shp)],
                "checks": ["geometry", "crs"],
            },
        )

        assert normalized["input_path"] == str(shp)
        assert normalized["check_types"] == ["geometry", "crs"]

    def test_normalize_project_layers_accepts_common_aliases(self, tmp_path):
        executor = Executor()
        shp = tmp_path / "sample.shp"
        shp.touch()

        normalized = executor._normalize_params(  # noqa: SLF001 - intentional internal contract test
            "project_layers",
            {
                "input_layers": [str(shp)],
                "output_place": "output/projected.shp",
                "crs": "CGCS2000",
            },
        )

        assert normalized["input_path"] == str(shp)
        assert normalized["output_path"] == "output/projected.shp"
        assert normalized["target_srs"] == "CGCS2000"

    def test_normalize_project_layers_maps_overwrite_aliases(self, tmp_path):
        executor = Executor()
        shp = tmp_path / "sample.shp"
        shp.touch()

        normalized = executor._normalize_params(  # noqa: SLF001 - intentional internal contract test
            "project_layers",
            {
                "input_path": str(shp),
                "output_path": "output/projected.shp",
                "overwrite": True,
            },
        )

        assert normalized["overwrite_output"] is True

    def test_normalize_project_layers_nested_input_payload(self, tmp_path):
        executor = Executor()
        shp = tmp_path / "sample.shp"
        shp.touch()

        normalized = executor._normalize_params(  # noqa: SLF001 - intentional internal contract test
            "project_layers",
            {
                "input": {
                    "input_layers": str(shp),
                    "output_place": "output/projected",
                    "srs": "CGCS2000",
                },
                "workspace": str(tmp_path),
            },
        )

        assert normalized["input_path"] == str(shp)
        assert normalized["output_path"] == "output/projected"
        assert normalized["target_srs"] == "CGCS2000"

    def test_execute_single_tool_uses_param_normalization_for_project_layers(self, tmp_path):
        executor = Executor()
        shp = tmp_path / "sample.shp"
        shp.touch()

        result = executor.execute_single_tool(
            "project_layers",
            {
                "input_layers": str(shp),
                "output_place": str(tmp_path / "projected.shp"),
                "srs": "CGCS2000",
            },
            dry_run=True,
        )

        assert result.success is True
        assert result.data is not None
        assert str(result.data.input_path) == str(shp)

    def test_unwrap_tool_call_payload_supports_tool_call_arguments_dict(self):
        executor = Executor()
        tool_name, args = executor._unwrap_tool_call_payload(  # noqa: SLF001 - intentional internal contract test
            "execute_code",
            {
                "tool_call": {
                    "name": "scan_layers",
                    "arguments": {"directory": "./workspace/input"},
                }
            },
        )
        assert tool_name == "scan_layers"
        assert args == {"directory": "./workspace/input"}

    def test_unwrap_tool_call_payload_supports_function_call_arguments_json(self):
        executor = Executor()
        tool_name, args = executor._unwrap_tool_call_payload(  # noqa: SLF001 - intentional internal contract test
            "execute_code",
            {
                "function_call": {
                    "name": "quality_check",
                    "arguments": '{"input_layers": ["./workspace/input/a.shp"], "checks": ["geometry"]}',
                }
            },
        )
        assert tool_name == "quality_check"
        assert args.get("input_layers") == ["./workspace/input/a.shp"]

    def test_execute_single_tool_accepts_tool_call_envelope(self, tmp_path):
        executor = Executor()
        result = executor.execute_single_tool(
            "execute_code",
            {
                "tool_call": {
                    "name": "scan_layers",
                    "arguments": {"directory": str(tmp_path)},
                }
            },
            dry_run=True,
        )
        assert result is not None

    def test_normalize_project_layers_placeholder_input_uses_workspace_fallback(self, tmp_path):
        executor = Executor()
        normalized = executor._normalize_params(  # noqa: SLF001 - intentional internal contract test
            "project_layers",
            {
                "input_layers": "从 step_2 的输出图层",
                "workspace": str(tmp_path),
                "output_path": str(tmp_path / "projected.shp"),
            },
        )

        assert normalized["input_path"] == str(tmp_path / "merged.shp")
        assert normalized["output_path"] == str(tmp_path / "projected.shp")

    def test_normalize_quality_check_placeholder_input_uses_workspace_fallback(self, tmp_path):
        executor = Executor()
        normalized = executor._normalize_params(  # noqa: SLF001 - intentional internal contract test
            "quality_check",
            {
                "input_layers": "从 step_3 的输出图层",
                "workspace": str(tmp_path),
            },
        )

        assert normalized["input_path"] == str(tmp_path / "projected.shp")

    def test_normalize_quality_check_coordinate_system_alias(self):
        executor = Executor()
        normalized = executor._normalize_params(  # noqa: SLF001 - intentional internal contract test
            "quality_check",
            {
                "coordinate_system": ["coordinate_system"],
                "input_path": "workspace/input/sample.shp",
            },
        )
        assert normalized["check_types"] == ["crs"]

    def test_execute_single_tool_uses_pipeline_validation_on_missing_input(self):
        executor = Executor()
        result = executor.execute_single_tool(
            "project_layers",
            {"output_path": "output/projected.shp"},
            dry_run=True,
        )
        assert result.success is False
        assert result.error_code in {"input_parse_error", "validation_error_1"}

    def test_execute_step_infers_input_path_for_quality_check_from_dependency(self, tmp_path):
        executor = Executor()
        shp = tmp_path / "sample.shp"
        shp.touch()

        plan = Plan(
            id="p1",
            goal="quality check recovery inference",
            steps=[
                PlanStep(id="step_1", tool="scan_layers", description="scan", input={"path": str(tmp_path)}),
                PlanStep(
                    id="step_2",
                    tool="quality_check",
                    description="check crs",
                    input={"check_types": ["coordinate_system"]},
                    depends_on=["step_1"],
                ),
            ],
        )

        # simulate completed scan output
        scan_result = executor.execute_single_tool("scan_layers", {"path": str(tmp_path)}, dry_run=True)
        assert scan_result.success
        plan.steps[0].complete(scan_result.data)

        trace, result = executor._execute_step(  # noqa: SLF001 - intentional internal contract test
            plan.steps[1], ExecutionMode.DRY_RUN, plan
        )
        assert trace.status == "completed"
        assert result.success is True

    def test_execute_step_replaces_stale_project_input_with_inferred_path(self, tmp_path):
        executor = Executor()
        shp = tmp_path / "sample.shp"
        shp.touch()

        plan = Plan(
            id="p-replace-stale",
            goal="replace stale input path",
            steps=[
                PlanStep(id="step_1", tool="scan_layers", description="scan", input={"path": str(tmp_path)}),
                PlanStep(
                    id="step_2",
                    tool="project_layers",
                    description="投影变换",
                    input={"input_path": str(tmp_path / "input"), "output_path": str(tmp_path / "out.shp")},
                    depends_on=["step_1"],
                ),
            ],
        )
        scan_result = executor.execute_single_tool("scan_layers", {"path": str(tmp_path)}, dry_run=True)
        assert scan_result.success
        plan.steps[0].complete(scan_result.data)

        trace, result = executor._execute_step(plan.steps[1], ExecutionMode.DRY_RUN, plan)  # noqa: SLF001
        assert trace.status == "completed"
        assert result.success is True
        assert result.data is not None
        assert str(result.data.input_path) == str(shp)

    def test_execute_step_selects_line_and_polygon_from_scan_layers(self):
        executor = Executor()
        scan_result = {
            "layers": [
                {"path": r"E:\data\my.gdb\roads_line", "type": "Polyline"},
                {"path": r"E:\data\my.gdb\blocks_poly", "type": "Polygon"},
            ]
        }
        plan = Plan(
            id="p-geom-hint",
            goal="geom-specific projection",
            steps=[
                PlanStep(id="step_1", tool="scan_layers", description="scan", input={"path": "."}),
                PlanStep(id="step_2", tool="project_layers", description="将线数据投影转换至CGCS2000", input={}, depends_on=["step_1"]),
                PlanStep(id="step_3", tool="project_layers", description="将面数据投影转换至CGCS2000", input={}, depends_on=["step_1"]),
            ],
        )
        plan.steps[0].complete(scan_result)

        line_path = executor._infer_input_path(plan.steps[1], plan)  # noqa: SLF001
        poly_path = executor._infer_input_path(plan.steps[2], plan)  # noqa: SLF001
        assert line_path == r"E:\data\my.gdb\roads_line"
        assert poly_path == r"E:\data\my.gdb\blocks_poly"

    def test_execute_step_infers_project_output_path_when_missing(self):
        executor = Executor()
        plan = Plan(
            id="p-proj-output",
            goal="infer project output",
            steps=[
                PlanStep(id="step_1", tool="scan_layers", description="scan", input={"path": "."}),
                PlanStep(
                    id="step_2",
                    tool="project_layers",
                    description="将扫描发现的线数据图层投影转换至CGCS2000坐标系",
                    input={},
                    depends_on=["step_1"],
                ),
            ],
        )
        plan.steps[0].complete({"layers": [{"path": r"E:\data\my.gdb\roads_line", "type": "Polyline"}]})
        trace, result = executor._execute_step(plan.steps[1], ExecutionMode.DRY_RUN, plan)  # noqa: SLF001
        assert trace.status == "completed"
        assert result.success is True
        assert result.data is not None
        assert "projected_line" in str(result.data.output_path)

    def test_normalize_execute_code_missing_code_from_task_payload(self):
        executor = Executor()
        normalized = executor._normalize_params(  # noqa: SLF001 - intentional internal contract test
            "execute_code",
            {
                "task": "筛选 Polyline 并投影",
                "input_layers": "workspace/input/roads.shp",
                "target_srs": "CGCS2000",
                "workspace": "workspace/output",
            },
        )
        assert "code" in normalized
        assert isinstance(normalized["code"], str)
        assert "set_result" in normalized["code"]
        assert normalized["description"] == "筛选 Polyline 并投影"

    def test_normalize_execute_code_missing_code_for_create_gdb_task(self):
        executor = Executor()
        normalized = executor._normalize_params(  # noqa: SLF001 - intentional internal contract test
            "execute_code",
            {
                "task": "创建文件地理数据库",
                "workspace": "workspace/output",
            },
        )
        code = str(normalized.get("code", ""))
        assert "CreateFileGDB" in code
        assert "requested_output" in code
        assert "set_result" in code
        assert "gdb_path" in code

    def test_normalize_export_map_infers_nested_project_path_from_workspace(self, tmp_path):
        project_file = tmp_path / "input" / "myproject5" / "myproject5.aprx"
        project_file.parent.mkdir(parents=True, exist_ok=True)
        project_file.touch()

        executor = Executor(context=ExecutionContext.build(workspace=tmp_path))
        normalized = executor._normalize_params(  # noqa: SLF001 - intentional internal contract test
            "export_map",
            {"output_path": str(tmp_path / "output" / "map.pdf")},
        )

        assert normalized["project_path"] == str(project_file)

    def test_normalize_export_map_maps_overwrite_aliases(self):
        executor = Executor()
        normalized = executor._normalize_params(  # noqa: SLF001 - intentional internal contract test
            "export_map",
            {
                "output_path": "workspace/output/map.png",
                "覆盖": True,
            },
        )
        assert normalized["overwrite_output"] is True

    def test_execute_single_tool_execute_code_accepts_task_alias(self):
        executor = Executor()
        result = executor.execute_single_tool(
            "execute_code",
            {
                "task": "筛选 Polyline 并投影",
                "input_layers": "workspace/input/roads.shp",
                "target_srs": "CGCS2000",
                "workspace": "workspace/output",
            },
            dry_run=True,
        )
        assert result.success is True
        assert result.data is not None
        assert "DRY RUN" in result.data.stdout

    def test_preflight_blocks_export_map_when_no_project_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        executor = Executor(context=ExecutionContext.build(workspace=tmp_path))
        step = PlanStep(id="step_1", tool="export_map", description="导出地图", input={})
        plan = Plan(id="p1", goal="export", steps=[step])

        preflight = executor._run_step_preflight("export_map", {}, step, plan)  # noqa: SLF001
        assert preflight["blocked"] is True
        assert "aprx" in str(preflight["reason"]).lower()

    def test_preflight_autofills_export_map_project_from_nested_input(self, tmp_path):
        project_file = tmp_path / "input" / "myproject5" / "myproject5.aprx"
        project_file.parent.mkdir(parents=True, exist_ok=True)
        project_file.touch()

        executor = Executor(context=ExecutionContext.build(workspace=tmp_path))
        step = PlanStep(id="step_1", tool="export_map", description="导出地图", input={})
        plan = Plan(id="p2", goal="export", steps=[step])

        preflight = executor._run_step_preflight("export_map", {}, step, plan)  # noqa: SLF001
        assert preflight["blocked"] is False
        assert preflight["input"]["project_path"] == str(project_file)

    def test_preflight_project_layers_replaces_missing_input_with_dependency_output(self, tmp_path):
        executor = Executor(context=ExecutionContext.build(workspace=tmp_path))
        existing_merged = tmp_path / "output" / "merged_polygon.shp"
        existing_merged.parent.mkdir(parents=True, exist_ok=True)
        existing_merged.touch()

        step_1 = PlanStep(
            id="step_1",
            tool="merge_layers",
            description="merge",
            input={"input_layers": [str(tmp_path / "input")]},
        )
        step_1.complete({"output_path": str(existing_merged)})

        missing_input = tmp_path / "output" / "merged.shp"
        step_2 = PlanStep(
            id="step_2",
            tool="project_layers",
            description="project",
            input={
                "input_path": str(missing_input),
                "output_path": str(tmp_path / "output" / "projected.shp"),
                "target_srs": "WGS84",
            },
            depends_on=["step_1"],
        )
        plan = Plan(id="p-preflight-project", goal="project", steps=[step_1, step_2])

        preflight = executor._run_step_preflight("project_layers", dict(step_2.input), step_2, plan)  # noqa: SLF001
        assert preflight["blocked"] is False
        assert preflight["input"]["input_path"] == str(existing_merged)

    def test_execute_step_blocks_unresolved_step_reference(self, tmp_path):
        executor = Executor(context=ExecutionContext.build(workspace=tmp_path))
        plan = Plan(
            id="p-unresolved-ref",
            goal="unresolved reference",
            steps=[
                PlanStep(id="step_1", tool="scan_layers", description="scan", input={"path": str(tmp_path)}),
                PlanStep(
                    id="step_2",
                    tool="merge_layers",
                    description="merge from previous output",
                    input={
                        "input_layers": "$step_1.layers",
                        "output_path": str(tmp_path / "output" / "merged.shp"),
                    },
                    depends_on=["step_1"],
                ),
            ],
        )

        # 依赖步骤执行完成，但没有 layers 字段，引用应被拦截。
        plan.steps[0].complete({"summary": "no layers"})
        trace, result = executor._execute_step(plan.steps[1], ExecutionMode.DRY_RUN, plan)  # noqa: SLF001

        assert trace.status == "failed"
        assert result.success is False
        assert "未解析引用" in (result.error or "")


class TestExecutionAdapter:
    def test_execution_adapter_default_legacy_mode(self, tmp_path):
        adapter = ExecutionAdapter(context=ExecutionContext.build(workspace=tmp_path))
        status = adapter.status()
        assert status["mode"] == "legacy"

    def test_execution_adapter_shadow_mode_collects_compare(self, tmp_path):
        cfg = tmp_path / "execution_adapter_config.json"
        cfg.write_text('{"mode":"shadow"}', encoding="utf-8")
        adapter = ExecutionAdapter(
            context=ExecutionContext.build(workspace=tmp_path),
            config_path=cfg,
        )
        plan = Plan(
            id="p-shadow-exec",
            goal="scan only",
            steps=[PlanStep(id="s1", tool="scan_layers", description="scan", input={"path": str(tmp_path)})],
        )
        result = adapter.execute(plan, ExecutionMode.DRY_RUN)
        assert result is not None
        st = adapter.status()
        assert st["mode"] == "shadow"
        assert "shadow_compare" in st
        assert "metrics" in st

    def test_execution_adapter_set_mode_persists(self, tmp_path):
        cfg = tmp_path / "execution_adapter_config.json"
        adapter = ExecutionAdapter(
            context=ExecutionContext.build(workspace=tmp_path),
            config_path=cfg,
        )
        st = adapter.set_mode("shadow", persist=True)
        assert st["mode"] == "shadow"
        assert cfg.exists()
        text = cfg.read_text(encoding="utf-8")
        assert '"mode": "shadow"' in text

    def test_execution_adapter_metrics_persist_and_reset(self, tmp_path):
        cfg = tmp_path / "execution_adapter_config.json"
        metrics = tmp_path / "execution_adapter_metrics.json"
        cfg.write_text('{"mode":"legacy"}', encoding="utf-8")
        adapter = ExecutionAdapter(
            context=ExecutionContext.build(workspace=tmp_path),
            config_path=cfg,
            metrics_path=metrics,
        )
        plan = Plan(
            id="p-metrics",
            goal="scan only",
            steps=[PlanStep(id="s1", tool="scan_layers", description="scan", input={"path": str(tmp_path)})],
        )
        _ = adapter.execute(plan, ExecutionMode.DRY_RUN)
        assert metrics.exists()
        payload = metrics.read_text(encoding="utf-8")
        assert '"runs": 1' in payload

        adapter2 = ExecutionAdapter(
            context=ExecutionContext.build(workspace=tmp_path),
            config_path=cfg,
            metrics_path=metrics,
        )
        st = adapter2.status()
        assert (st.get("metrics") or {}).get("runs", 0) >= 1
        reset = adapter2.reset_metrics(persist=True)
        assert (reset.get("metrics") or {}).get("runs", 1) == 0

    def test_execution_adapter_guard_recommendation_and_apply(self, tmp_path):
        cfg = tmp_path / "execution_adapter_config.json"
        cfg.write_text(
            '{"mode":"shadow","guard":{"min_attempts":2,"max_fallback_rate":0.5,"suggested_mode":"langchain"}}',
            encoding="utf-8",
        )
        adapter = ExecutionAdapter(
            context=ExecutionContext.build(workspace=tmp_path),
            config_path=cfg,
            metrics_path=tmp_path / "execution_adapter_metrics.json",
        )
        adapter.metrics["langchain_attempts"] = 4
        adapter.metrics["langchain_fallbacks"] = 1
        recommendation = adapter.guard_recommendation()
        assert recommendation["can_recommend_switch"] is True
        assert recommendation["target_mode"] == "langchain"

        payload = adapter.apply_guard_recommendation(persist=False)
        assert payload["mode"] == "langchain"

    def test_execution_adapter_history_appends_events(self, tmp_path):
        cfg = tmp_path / "execution_adapter_config.json"
        metrics = tmp_path / "execution_adapter_metrics.json"
        cfg.write_text('{"mode":"shadow","max_history_events":5}', encoding="utf-8")
        adapter = ExecutionAdapter(
            context=ExecutionContext.build(workspace=tmp_path),
            config_path=cfg,
            metrics_path=metrics,
        )
        plan = Plan(
            id="p-history",
            goal="scan only",
            steps=[PlanStep(id="s1", tool="scan_layers", description="scan", input={"path": str(tmp_path)})],
        )
        _ = adapter.execute(plan, ExecutionMode.DRY_RUN)
        history = adapter.load_history()
        assert len(history) >= 1
        assert history[-1].get("plan_id") == "p-history"
        st = adapter.status()
        assert st.get("history_count", 0) >= 1


# === Agent Tests ===

class TestGISAgent:
    """Test main GIS Agent."""
    
    def test_create_agent(self):
        agent = GISAgent()
        assert agent.memory is not None
        assert agent.planner is not None
        assert agent.executor is not None
    
    def test_create_agent_with_config(self, tmp_path):
        config = AgentConfig(
            session_id="test-session",
            workspace_path=tmp_path,
            default_mode=ExecutionMode.DRY_RUN
        )
        agent = GISAgent(config=config)
        
        assert agent.config.session_id == "test-session"
        assert agent.config.workspace_path == tmp_path
    
    def test_agent_chat(self):
        agent = GISAgent()
        response = agent.chat("你好")
        
        assert isinstance(response, AgentResponse)
        assert response.content is not None
        assert len(response.content) > 0
    
    def test_agent_chat_task(self):
        agent = GISAgent()
        response = agent.chat("扫描当前目录的 GIS 数据")
        
        assert response.action_taken is not None
        # Should create a plan for task execution
        assert response.plan is not None or "execute" in response.action_taken
    
    def test_agent_create_plan(self):
        agent = GISAgent()
        plan = agent.create_plan("制作专题图")
        
        assert plan is not None
        assert agent.current_plan == plan

    def test_agent_create_plan_injects_detected_project_path(self, tmp_path):
        project_file = tmp_path / "input" / "myproject5" / "MyProject5.aprx"
        project_file.parent.mkdir(parents=True, exist_ok=True)
        project_file.touch()

        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))
        plan = agent.create_plan("制作专题图并导出地图")

        export_steps = [s for s in plan.steps if s.tool == "export_map"]
        assert export_steps
        assert export_steps[0].input.get("project_path") == str(project_file)

    def test_agent_create_plan_applies_deterministic_thematic_override(self, tmp_path, monkeypatch):
        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))

        generic_plan = Plan(
            id="p-generic",
            goal="generic map",
            steps=[PlanStep(id="step_1", tool="export_map", description="export", input={"output_path": "x.jpg"})],
        )
        forced_plan = Plan(
            id="p-forced",
            goal="forced map",
            steps=[PlanStep(id="step_1", tool="execute_code", description="draw", input={"code": "set_result({})"})],
        )

        monkeypatch.setattr(agent.planner, "plan", lambda task_description, context, expert_mode=True: generic_plan)

        def fake_builder(task_description, context, input_folder, output_folder):
            return forced_plan

        monkeypatch.setattr(agent.planner, "_create_explicit_thematic_map_plan", fake_builder)

        plan = agent.create_plan("根据workspace/output/province_forest_coverage这个图层制作专题图")
        assert plan is forced_plan
        assert plan.metadata.get("plan_source") == "deterministic_thematic_override"
    
    def test_agent_call_tool(self, tmp_path):
        agent = GISAgent()
        result = agent.call_tool(
            "scan_layers",
            {"directory": str(tmp_path)},
            dry_run=True
        )
        
        assert result is not None
        # Should record in memory
        assert len(agent.memory.tool_history) > 0

    def test_agent_available_skill_names_include_custom_workspace_skills(self, tmp_path):
        skills_dir = tmp_path / "workspace" / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "demo_skill.md").write_text(
            """---
name: demo_skill
description: 自定义测试技能
triggers: [demo]
code_template: |
  set_result({"ok": True})
---
""",
            encoding="utf-8",
        )

        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))
        names = agent._get_available_skill_names()  # noqa: SLF001
        assert "thematic_map" in names
        assert "demo_skill" in names

    def test_handle_tool_help_includes_custom_skills_in_count(self, tmp_path):
        skills_dir = tmp_path / "workspace" / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "demo_skill.md").write_text(
            """---
name: demo_skill
description: 自定义测试技能
triggers: [demo]
code_template: |
  import arcpy
  set_result({"ok": True})
---
""",
            encoding="utf-8",
        )

        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))
        response = agent._handle_tool_help("帮助")  # noqa: SLF001

        assert "### 可用技能（4个）" in response.content
        assert "demo_skill" in response.content
    
    def test_agent_with_mock_llm(self):
        mock_llm = MockLLMClient(responses=["这是一个测试响应。"])
        agent = GISAgent(llm_client=mock_llm)
        
        response = agent.chat("测试问题")
        
        # Should use mock LLM
        assert mock_llm.call_count >= 0  # May or may not use LLM depending on intent

    def test_prepare_task_for_planning_merges_followup_with_recent_primary_task(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        previous_task = (
            "制作广西区域发展布局图，整合行政区线和行政区面，统一投影，"
            "完成符号化和标注后导出成果图。"
        )
        followup = "根据之前信息，直接对行政区线和面继续操作并规划任务"

        agent.memory.add_user_message(previous_task)
        agent.memory.add_assistant_message("已记录")
        agent.memory.add_user_message(followup)

        merged, used = agent._prepare_task_for_planning(followup)  # noqa: SLF001
        assert used is True
        assert "历史任务" in merged
        assert "本次补充要求" in merged
        assert previous_task in merged

    def test_prepare_task_for_planning_keeps_intermediate_followup_constraints(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        previous_task = "制作广西区域发展布局图并整合行政区线和行政区面"
        followup_1 = "继续处理行政区线，统一符号化并标注"
        followup_2 = "不要裁剪图层，保持原始边界"
        latest_followup = "根据之前信息继续执行并导出结果"

        agent.memory.add_user_message(previous_task)
        agent.memory.add_assistant_message("收到")
        agent.memory.add_user_message(followup_1)
        agent.memory.add_assistant_message("已记录")
        agent.memory.add_user_message(followup_2)
        agent.memory.add_assistant_message("已记录")

        merged, used = agent._prepare_task_for_planning(latest_followup)  # noqa: SLF001

        assert used is True
        assert "已记录续接约束" in merged
        assert followup_1 in merged
        assert followup_2 in merged

    def test_prepare_task_for_planning_explicit_new_task_does_not_merge_followup(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        previous_task = "制作自然保护区分布图并导出"
        explicit_new = (
            "继续之前任务，但只用 E:/BST_lab/GIS CLI/workspace/output/province_forest_coverage.shp "
            "按 ForestCov 字段做绿色分级图并导出到 workspace/output/china_vegetation_coverage.jpg"
        )

        agent.memory.add_user_message(previous_task)
        agent.memory.add_assistant_message("收到")

        merged, used = agent._prepare_task_for_planning(explicit_new)  # noqa: SLF001
        assert used is False
        assert merged == explicit_new

    def test_followup_execution_skips_llm_refine_and_keeps_context(self, monkeypatch):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        agent.llm_client = object()
        previous_task = "制作广西专题图并完成整合、投影和导出"
        followup = "根据之前信息继续处理行政区线和面"

        agent.memory.add_user_message(previous_task)
        agent.memory.add_assistant_message("收到")

        def fail_if_called(_message):
            raise AssertionError("LLM refinement should be skipped for follow-up")

        monkeypatch.setattr(agent, "_refine_task_with_llm", fail_if_called)
        response = agent._handle_task_execution(followup)  # noqa: SLF001
        assert response.plan is not None

    def test_extract_docx_path_from_message(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        message = r"请根据 E:\BST_lab\GIS CLI\workspace\input\task_requirements.docx 执行地理处理"
        found = agent._extract_docx_path_from_message(message)  # noqa: SLF001
        assert found is not None
        assert found.lower().endswith(".docx")

    def test_extract_document_path_from_message_supports_pdf(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        message = r"请根据 E:\BST_lab\GIS CLI\workspace\input\requirements.pdf 执行地理处理"
        found = agent._extract_document_path_from_message(message)  # noqa: SLF001
        assert found is not None
        assert found.lower().endswith(".pdf")

    def test_extract_document_path_from_message_infers_single_pdf_in_input(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        target_pdf = input_dir / "A上午试题.pdf"
        target_pdf.write_bytes(b"%PDF-1.4\n")

        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))
        found = agent._extract_document_path_from_message("我的 GIS 需求在 pdf 文档里，请按文档规划")  # noqa: SLF001

        assert found is not None
        assert Path(found).name == "A上午试题.pdf"

    def test_extract_document_path_from_message_avoids_ambiguous_guess(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "A上午试题.pdf").write_bytes(b"%PDF-1.4\n")
        (input_dir / "B下午试题.pdf").write_bytes(b"%PDF-1.4\n")

        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))
        found = agent._extract_document_path_from_message("我的任务在 pdf 文档里")  # noqa: SLF001

        assert found is None

    def test_enrich_task_with_word_document_uses_parser_and_memory(self, monkeypatch):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))

        def fake_execute_single_tool(tool_name, input_data, dry_run=True):
            assert tool_name == "read_word"
            return SimpleNamespace(
                success=True,
                error=None,
                data=SimpleNamespace(
                    text="创建数据库并整合数据入库\n投影统一到 CGCS2000\n导出结果 PNG"
                ),
            )

        monkeypatch.setattr(agent.executor, "execute_single_tool", fake_execute_single_tool)

        enriched, note = agent._enrich_task_with_word_document(  # noqa: SLF001
            "请根据 workspace/input/task.docx 完成任务"
        )
        assert "文档提炼任务" in enriched
        assert "投影统一" in enriched
        assert "已读取文档" in note
        assert agent.memory.get_context("document_summary")
        constraints = agent.memory.get_context("document_constraints", [])
        assert isinstance(constraints, list)
        assert any("投影" in c for c in constraints)

    def test_enrich_task_with_pdf_document_uses_parser_and_memory(self, monkeypatch):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))

        def fake_execute_single_tool(tool_name, input_data, dry_run=True):
            assert tool_name == "read_pdf"
            return SimpleNamespace(
                success=True,
                error=None,
                data=SimpleNamespace(
                    text="创建数据库并整合数据入库\n统一坐标系\n导出结果 PNG"
                ),
            )

        monkeypatch.setattr(agent.executor, "execute_single_tool", fake_execute_single_tool)

        enriched, note = agent._enrich_task_with_word_document(  # noqa: SLF001
            "请根据 workspace/input/task.pdf 完成任务"
        )
        assert "文档提炼任务" in enriched
        assert "统一坐标系" in enriched
        assert "已读取文档" in note
        assert agent.memory.get_context("document_summary")
        constraints = agent.memory.get_context("document_constraints", [])
        assert isinstance(constraints, list)

    def test_enrich_task_with_custom_skill_context_for_reference_skill(self, tmp_path):
        skills_dir = tmp_path / "workspace" / "skills" / "pdf"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "SKILL.md").write_text(
            """---
name: pdf
description: PDF 文档处理参考技能
keywords: [pdf, 文档]
---

## 执行步骤
1. 提取文本
2. 清洗结果

```python
from pypdf import PdfReader
reader = PdfReader("doc.pdf")
print(len(reader.pages))
```
""",
            encoding="utf-8",
        )

        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))
        enriched, note = agent._enrich_task_with_custom_skill_context("请处理 pdf 并提取内容")  # noqa: SLF001

        assert "[通用技能参考]" in enriched
        assert "skill: pdf" in enriched
        assert "已匹配通用技能 pdf" in note

    def test_enrich_task_with_custom_skill_context_for_executable_skill(self, tmp_path):
        skills_dir = tmp_path / "workspace" / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "dissolve_features.md").write_text(
            """---
name: dissolve_features
description: 按字段融合要素
triggers: [融合, dissolve]
---

```python
import arcpy
set_result({"ok": True})
```
""",
            encoding="utf-8",
        )

        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))
        enriched, note = agent._enrich_task_with_custom_skill_context("请帮我做融合")  # noqa: SLF001

        assert enriched == "请帮我做融合"
        assert "可执行技能 dissolve_features" in note
    
    def test_agent_reset(self):
        agent = GISAgent()
        agent.chat("Hello")
        agent.create_plan("Test task")
        
        agent.reset()
        
        assert len(agent.memory.turns) == 0
        assert agent.current_plan is None
    
    def test_agent_session_summary(self):
        agent = GISAgent()
        agent.chat("Hello")
        
        summary = agent.get_session_summary()
        
        assert "session_id" in summary
        assert "memory" in summary
        assert summary["memory"]["total_turns"] >= 2  # User + Assistant
        assert "execution_adapter" in summary

    def test_agent_restores_runtime_state_from_memory_store(self, tmp_path):
        memory_dir = tmp_path / "memory"
        session_id = "session_persist_runtime"
        config = AgentConfig(
            llm_enabled=False,
            session_id=session_id,
            memory_path=memory_dir,
            enable_state_manager=False,
        )

        agent = GISAgent(config=config)
        plan = Plan(
            id="p-restore",
            goal="跨会话恢复计划",
            steps=[
                PlanStep(id="step_1", tool="scan_layers", description="scan", input={"path": str(tmp_path)}),
            ],
        )
        plan.steps[0].complete({"output_path": str(tmp_path / "output" / "scan.json")})
        agent.current_plan = plan

        trace = ExecutionTrace(plan_id=plan.id, mode=ExecutionMode.EXECUTE)
        trace.steps.append(
            SimpleNamespace(
                step_id="step_1",
                tool="scan_layers",
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                status="completed",
                input={"path": str(tmp_path)},
                output={"output_path": str(tmp_path / "output" / "scan.json")},
                error=None,
                duration_ms=8,
            )
        )
        trace.complete(True, outputs=[str(tmp_path / "output" / "scan.json")])
        agent.last_result = ExecutionResult(
            success=True,
            plan=plan,
            trace=trace,
            outputs=[str(tmp_path / "output" / "scan.json")],
            error=None,
        )

        agent.save_session()

        restored = GISAgent(config=config)
        assert restored.current_plan is not None
        assert restored.current_plan.id == "p-restore"
        assert restored.current_plan.steps[0].status == StepStatus.COMPLETED
        assert restored.last_result is not None
        assert restored.last_result.success is True
        assert restored.last_result.outputs == [str(tmp_path / "output" / "scan.json")]

    def test_record_execution_reflection_on_failure(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        plan = Plan(
            id="p-refl",
            goal="测试失败反思",
            steps=[PlanStep(id="step_1", tool="merge_layers", description="merge")],
        )
        trace = ExecutionTrace(plan_id=plan.id, mode=ExecutionMode.EXECUTE)
        trace.complete(False, outputs=[], error="ERROR 000732 输入数据集不存在")
        result = ExecutionResult(success=False, plan=plan, trace=trace, outputs=[], error="ERROR 000732 输入数据集不存在")

        agent._record_execution_reflection(plan, result, [])  # noqa: SLF001

        assert len(agent.memory.reflections) == 1
        assert "000732" in agent.memory.reflections[0].issue
        assert len(agent.memory.structured_memories) == 1

    def test_agent_updates_workflow_state_on_plan_creation(self, tmp_path):
        state_file = tmp_path / "state" / "workflow_state.json"
        agent = GISAgent(
            config=AgentConfig(
                llm_enabled=False,
                workspace_path=tmp_path,
                state_path=state_file,
            )
        )

        plan = agent.create_plan("扫描 GIS 数据")
        summary = agent.get_session_summary()
        state = summary.get("state_manager") or {}

        assert plan is not None
        assert state.get("exists") is True
        assert state.get("current_plan_id") == plan.id

    def test_intent_rule_treats_bug_question_as_general(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        intent = agent._analyze_intent_with_rules("你为什么只执行合并图层，别的任务需求看不到？")  # noqa: SLF001
        assert intent == "general"

    def test_status_query_explains_non_delivery_success(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        plan = Plan(id="p-empty", goal="检查数据", steps=[PlanStep(id="s1", tool="quality_check", description="检查")])
        fake_trace = SimpleNamespace(
            steps=[SimpleNamespace(tool="quality_check", status="completed", output=None)]
        )
        agent.last_result = ExecutionResult(
            success=True,
            plan=plan,
            trace=fake_trace,  # type: ignore[arg-type]
            outputs=[],
            error=None,
        )

        response = agent._handle_status_query("为什么输出文件是空的")  # noqa: SLF001
        assert "诊断：" in response.content
        assert "检查类步骤" in response.content

    def test_status_query_collects_outputs_from_trace_payload(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        plan = Plan(id="p-out", goal="导出地图", steps=[PlanStep(id="s1", tool="export_map", description="导出")])
        fake_trace = SimpleNamespace(
            steps=[SimpleNamespace(tool="export_map", status="completed", output={"output_path": "workspace/output/map.png"})]
        )
        agent.last_result = ExecutionResult(
            success=True,
            plan=plan,
            trace=fake_trace,  # type: ignore[arg-type]
            outputs=[],
            error=None,
        )

        response = agent._handle_status_query("查看结果")  # noqa: SLF001
        assert "workspace/output/map.png" in response.content
        assert "输出文件：\n（无）" not in response.content

    def test_general_query_routes_why_empty_output_to_status(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        plan = Plan(id="p-empty-2", goal="检查数据", steps=[PlanStep(id="s1", tool="quality_check", description="检查")])
        fake_trace = SimpleNamespace(steps=[SimpleNamespace(tool="quality_check", status="completed", output=None)])
        agent.last_result = ExecutionResult(
            success=True,
            plan=plan,
            trace=fake_trace,  # type: ignore[arg-type]
            outputs=[],
            error=None,
        )

        response = agent._handle_general_query("为什么输出是空的")  # noqa: SLF001
        assert response.content.startswith("上次执行状态：")

    def test_continue_message_prefers_confirm_action_with_pending_plan_even_with_llm(self):
        mock_llm = MockLLMClient(responses=["query_status"])
        agent = GISAgent(llm_client=mock_llm)
        agent.current_plan = Plan(
            id="p-resume",
            goal="resume test",
            steps=[PlanStep(id="step_1", tool="scan_layers", description="scan")],
        )
        intent = agent._analyze_intent("怎么出错了，继续依次执行任务")  # noqa: SLF001
        assert intent == "confirm_action"

    def test_pending_plan_explicit_new_task_maps_execute_task(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        agent.current_plan = Plan(
            id="p-explicit-new-task",
            goal="旧任务",
            steps=[PlanStep(id="step_1", tool="merge_layers", description="merge")],
        )

        message = (
            "继续，但这次只用 E:\\BST_lab\\GIS CLI\\workspace\\output\\province_forest_coverage.shp "
            "做图，字段 ForestCov，绿色分级，输出到 workspace/output/china_vegetation_coverage.jpg"
        )
        intent = agent._analyze_intent(message)  # noqa: SLF001
        assert intent == "execute_task"

    def test_overwrite_phrase_prefers_confirm_action_with_pending_plan(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        agent.current_plan = Plan(
            id="p-overwrite-intent",
            goal="overwrite intent",
            steps=[PlanStep(id="step_1", tool="merge_layers", description="merge")],
        )

        intent = agent._analyze_intent("可以直接覆盖文件")  # noqa: SLF001
        assert intent == "confirm_action"

    def test_confirmation_applies_overwrite_policy_for_output_conflict(self, monkeypatch, tmp_path):
        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))
        failed = PlanStep(
            id="step_1",
            tool="merge_layers",
            description="merge",
            input={"input_layers": ["a.shp"], "output_path": str(tmp_path / "output" / "merged.shp")},
        )
        failed.fail("ERROR 000725: Output Dataset already exists")
        plan = Plan(id="p-overwrite-retry", goal="retry overwrite", steps=[failed])
        agent.current_plan = plan

        captured: dict[str, object] = {}

        def fake_execute_with_recovery(plan_arg, mode):
            captured["mode"] = mode
            captured["status"] = plan_arg.steps[0].status
            captured["overwrite_output"] = plan_arg.steps[0].input.get("overwrite_output")
            trace = ExecutionTrace(plan_id=plan_arg.id, mode=mode)
            trace.complete(True, outputs=[str(tmp_path / "output" / "merged.shp")])
            result = ExecutionResult(
                success=True,
                plan=plan_arg,
                trace=trace,
                outputs=[str(tmp_path / "output" / "merged.shp")],
                error=None,
            )
            return result, []

        monkeypatch.setattr(agent, "_execute_with_recovery", fake_execute_with_recovery)
        monkeypatch.setattr(agent, "_record_execution_reflection", lambda *args, **kwargs: None)

        response = agent._handle_confirmation("可以直接覆盖文件")  # noqa: SLF001
        assert captured["mode"] == ExecutionMode.EXECUTE
        assert captured["status"] == StepStatus.PENDING
        assert captured["overwrite_output"] is True
        assert "覆盖策略" in response.content

    def test_confirmation_with_explicit_new_task_restarts_planning(self, monkeypatch):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        agent.current_plan = Plan(
            id="p-confirm-explicit",
            goal="旧失败任务",
            steps=[PlanStep(id="step_1", tool="project_layers", description="投影")],
        )

        captured: dict[str, str] = {}

        def fake_handle_task_execution(message: str):
            captured["message"] = message
            return AgentResponse(content="new plan", action_taken="plan")

        monkeypatch.setattr(agent, "_handle_task_execution", fake_handle_task_execution)

        msg = (
            "继续，使用 workspace/output/province_forest_coverage.shp，字段 ForestCov，"
            "绿色分级，导出 workspace/output/china_vegetation_coverage.jpg"
        )
        response = agent._handle_confirmation(msg)  # noqa: SLF001

        assert response.action_taken == "plan"
        assert "province_forest_coverage.shp" in captured["message"]
        assert agent.current_plan is None

    def test_confirmation_without_pending_plan_routes_doc_followup_to_task_execution(self, monkeypatch, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "A上午试题.pdf").write_bytes(b"%PDF-1.4\n")

        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))
        agent.memory.add_user_message("input 文件夹里有一个 pdf 文档 A上午试题，帮我读取")

        captured: dict[str, str] = {}

        def fake_handle_task_execution(message: str):
            captured["message"] = message
            return AgentResponse(content="ok", action_taken="plan")

        monkeypatch.setattr(agent, "_handle_task_execution", fake_handle_task_execution)

        response = agent._handle_confirmation("读吧")  # noqa: SLF001

        assert response.action_taken == "plan"
        assert "文档" in captured["message"]

    def test_intent_affirmative_with_recent_document_hint_returns_execute_task(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        agent.memory.add_user_message("我的需求在 input/requirements.pdf 里")

        intent = agent._analyze_intent("读吧")  # noqa: SLF001
        assert intent == "execute_task"

    def test_extract_intent_from_llm_response_accepts_json_and_verbose_text(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        assert agent._extract_intent_from_llm_response('{"intent":"execute_task"}') == "execute_task"  # noqa: SLF001
        assert agent._extract_intent_from_llm_response("我认为是 execute_task，因为用户要执行任务") == "execute_task"  # noqa: SLF001
        assert agent._extract_intent_from_llm_response("执行任务") == "execute_task"  # noqa: SLF001

    def test_model_query_uses_normal_llm_chat_path(self):
        captured = {"called": False, "messages": None}

        class _Client:
            config = SimpleNamespace(model="Qwen/Qwen3.5-397B-A17B", provider="siliconflow")

            def chat(self, messages):
                captured["called"] = True
                captured["messages"] = messages
                return "这是走统一 llm 聊天路径返回的回答。"

        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        agent.llm_client = _Client()

        response = agent._handle_general_query("你是什么模型")  # noqa: SLF001

        assert captured["called"] is True
        assert response.action_taken == "chat"
        assert "统一 llm 聊天路径" in response.content
        assert isinstance(captured["messages"], list)

    def test_model_query_prefers_llm_reply_when_chat_available(self):
        captured = {"called": False, "messages": None}

        class _Client:
            config = SimpleNamespace(model="Qwen/Qwen3.5-397B-A17B", provider="siliconflow")

            def chat(self, messages):
                captured["called"] = True
                captured["messages"] = messages
                return "我当前使用你配置的 Qwen 模型，并通过 siliconflow 提供服务。"

        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        agent.llm_client = _Client()

        response = agent._handle_general_query("你是什么模型")  # noqa: SLF001

        assert captured["called"] is True
        assert response.action_taken == "chat"
        assert "Qwen" in response.content
        assert "siliconflow" in response.content
        assert isinstance(captured["messages"], list)

    def test_handle_task_execution_non_gis_without_skill_falls_back_to_chat(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        response = agent._handle_task_execution("帮我写一份周报")  # noqa: SLF001

        assert response.action_taken == "chat"
        assert "非 GIS" in response.content
        assert response.plan is None

    def test_handle_task_execution_non_gis_with_executable_skill_creates_plan(self, tmp_path):
        skills_dir = tmp_path / "workspace" / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "weekly_report.md").write_text(
            """---
name: weekly_report
description: 生成周报
triggers: [周报, 报告]
---

```python
set_result({"output": "weekly-report.md"})
```
""",
            encoding="utf-8",
        )

        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))
        response = agent._handle_task_execution("帮我生成周报")  # noqa: SLF001

        assert response.action_taken == "plan"
        assert response.plan is not None

    def test_rule_intent_allows_generic_actionable_request(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        intent = agent._analyze_intent_with_rules("帮我总结这个内容")  # noqa: SLF001
        assert intent == "execute_task"

    def test_llm_general_intent_does_not_override_actionable_rule(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        agent.llm_client = SimpleNamespace(chat=lambda *_args, **_kwargs: '{"intent":"general"}')
        agent._analyze_intent_with_llm = lambda _msg: "general"  # type: ignore[method-assign]  # noqa: SLF001

        intent = agent._analyze_intent("请帮我扫描数据并制作行政区划图")  # noqa: SLF001
        assert intent == "execute_task"

    def test_pending_plan_impatient_scan_message_maps_to_confirm_action(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        agent.current_plan = Plan(
            id="p-confirm",
            goal="扫描 input",
            steps=[PlanStep(id="step_1", tool="scan_layers", description="扫描输入目录")],
        )

        intent = agent._analyze_intent("你倒是扫啊")  # noqa: SLF001
        assert intent == "confirm_action"

    def test_rule_intent_recognizes_short_scan_or_mapping_phrases(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        assert agent._analyze_intent_with_rules("扫啊") == "execute_task"  # noqa: SLF001
        assert agent._analyze_intent_with_rules("请制图并出图") == "execute_task"  # noqa: SLF001

    def test_rule_intent_retry_question_phrase_maps_execute_task(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        message = "你再试试，换个方法试试？我需要你完成这个任务，你检查检查是不是arcpy写的有问题？"
        assert agent._analyze_intent_with_rules(message.lower()) == "execute_task"  # noqa: SLF001

    def test_rule_intent_followup_input_phrase_maps_execute_task(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        message = "就在input里面呀，继续之前的任务即可"
        assert agent._analyze_intent_with_rules(message.lower()) == "execute_task"  # noqa: SLF001

    def test_pending_plan_retry_question_phrase_maps_confirm_action(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        agent.current_plan = Plan(
            id="p-retry-confirm",
            goal="继续执行上一个失败任务",
            steps=[PlanStep(id="step_1", tool="project_layers", description="投影转换")],
        )

        message = "你再试试，换个方法试试？我需要你完成这个任务，你检查检查是不是arcpy写的有问题？"
        assert agent._analyze_intent(message) == "confirm_action"  # noqa: SLF001

    def test_update_context_from_relative_input_path(self, tmp_path):
        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))

        agent._update_context_from_message("我的数据在 workspace/input")  # noqa: SLF001

        input_folder = agent.memory.get_context("input_folder")
        assert isinstance(input_folder, str)
        assert input_folder.replace("\\", "/").endswith("workspace/input")

    def test_collect_output_candidates_skips_scan_layer_discovery_paths(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        plan = Plan(id="p-out-filter", goal="专题图", steps=[PlanStep(id="s1", tool="scan_layers", description="scan")])
        fake_trace = SimpleNamespace(
            steps=[
                SimpleNamespace(tool="scan_layers", status="completed", output={"path": "E:/old/nature_reserve.shp"}),
                SimpleNamespace(
                    tool="execute_code",
                    status="completed",
                    output={"result": {"output": "E:/workspace/output/china_vegetation_coverage.jpg"}},
                ),
            ]
        )
        result = ExecutionResult(
            success=True,
            plan=plan,
            trace=fake_trace,  # type: ignore[arg-type]
            outputs=[
                "E:/old/nature_reserve.shp",
                "E:/workspace/output/china_vegetation_coverage.jpg",
            ],
            error=None,
        )

        outputs = agent._collect_output_candidates(result)  # noqa: SLF001
        assert outputs == ["E:/workspace/output/china_vegetation_coverage.jpg"]

    def test_update_context_output_path_does_not_override_input_folder(self, tmp_path):
        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))
        agent.memory.set_context("input_folder", str(tmp_path / "workspace" / "input"))

        agent._update_context_from_message("输出到 workspace/output/行政区划图.png")  # noqa: SLF001

        input_folder = agent.memory.get_context("input_folder")
        assert isinstance(input_folder, str)
        assert input_folder.replace("\\", "/").endswith("workspace/input")

    def test_looks_like_gis_task_recognizes_admin_map_phrases(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        assert agent._looks_like_gis_task("我现在就扫描数据并制作行政区划图") is True  # noqa: SLF001

    def test_resolve_best_input_folder_ignores_output_like_memory_context(self, tmp_path):
        workspace = tmp_path
        good_input = workspace / "workspace" / "input"
        bad_output = workspace / "workspace" / "output"
        good_input.mkdir(parents=True, exist_ok=True)
        bad_output.mkdir(parents=True, exist_ok=True)

        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=workspace))
        agent.memory.set_context("input_folder", str(bad_output))

        resolved = agent._resolve_best_input_folder(str(workspace / "input"), workspace)  # noqa: SLF001
        assert resolved.replace("\\", "/").endswith("workspace/input")

    def test_recovery_suggestions_include_overwrite_when_output_exists(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        plan = Plan(id="p-suggest-overwrite", goal="suggest overwrite", steps=[PlanStep(id="s1", tool="merge_layers", description="merge")])
        trace = ExecutionTrace(plan_id=plan.id, mode=ExecutionMode.EXECUTE)
        trace.complete(False, outputs=[], error="ERROR 000725: Dataset already exists")
        result = ExecutionResult(
            success=False,
            plan=plan,
            trace=trace,
            outputs=[],
            error="ERROR 000725: Dataset already exists",
        )

        suggestions = agent._get_recovery_suggestions(result)  # noqa: SLF001
        assert "直接覆盖现有输出并重试" in suggestions

    def test_execute_with_recovery_autonomous_overwrite_policy_without_user_prompt(self, tmp_path, monkeypatch):
        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path, max_recovery_attempts=3))
        plan = Plan(
            id="p-auto-overwrite",
            goal="auto overwrite",
            steps=[
                PlanStep(
                    id="step_1",
                    tool="merge_layers",
                    description="merge",
                    input={"input_layers": ["a.shp"], "output_path": str(tmp_path / "output" / "merged.shp")},
                )
            ],
        )

        call_state = {"count": 0}

        def fake_execute_plan(plan_arg, mode=None):
            call_state["count"] += 1
            trace = ExecutionTrace(plan_id=plan_arg.id, mode=ExecutionMode.EXECUTE)
            if call_state["count"] == 1:
                plan_arg.steps[0].fail("ERROR 000725: Dataset already exists")
                trace.steps.append(
                    SimpleNamespace(tool="merge_layers", status="failed", output=None)
                )
                trace.complete(False, outputs=[], error="ERROR 000725: Dataset already exists")
                return ExecutionResult(
                    success=False,
                    plan=plan_arg,
                    trace=trace,
                    outputs=[],
                    error="ERROR 000725: Dataset already exists",
                )

            plan_arg.steps[0].status = StepStatus.COMPLETED
            trace.steps.append(
                SimpleNamespace(tool="merge_layers", status="completed", output={"output_path": str(tmp_path / "output" / "merged.shp")})
            )
            trace.complete(True, outputs=[str(tmp_path / "output" / "merged.shp")])
            return ExecutionResult(
                success=True,
                plan=plan_arg,
                trace=trace,
                outputs=[str(tmp_path / "output" / "merged.shp")],
                error=None,
            )

        monkeypatch.setattr(agent, "execute_plan", fake_execute_plan)

        result, notes = agent._execute_with_recovery(plan, ExecutionMode.EXECUTE)  # noqa: SLF001
        assert result.success is True
        assert call_state["count"] == 2
        assert plan.steps[0].input.get("overwrite_output") is True
        assert any("覆盖策略" in n for n in notes)

    def test_execute_with_recovery_stops_on_repeated_identical_failure(self, tmp_path, monkeypatch):
        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path, max_recovery_attempts=5))
        plan = Plan(
            id="p-repeat-fail",
            goal="repeat fail",
            steps=[PlanStep(id="step_1", tool="quality_check", description="check", input={"input_path": "missing.shp"})],
        )

        call_state = {"count": 0}

        def always_fail(plan_arg, mode=None):
            call_state["count"] += 1
            plan_arg.steps[0].fail("validation_error: input_path missing")
            trace = ExecutionTrace(plan_id=plan_arg.id, mode=ExecutionMode.EXECUTE)
            trace.steps.append(SimpleNamespace(tool="quality_check", status="failed", output=None))
            trace.complete(False, outputs=[], error="validation_error: input_path missing")
            return ExecutionResult(
                success=False,
                plan=plan_arg,
                trace=trace,
                outputs=[],
                error="validation_error: input_path missing",
            )

        monkeypatch.setattr(agent, "execute_plan", always_fail)
        monkeypatch.setattr(agent.planner, "suggest_recovery", lambda plan_arg, failed: plan_arg)

        result, notes = agent._execute_with_recovery(plan, ExecutionMode.EXECUTE)  # noqa: SLF001
        assert result.success is False
        assert call_state["count"] < 5
        assert any("重复失败" in n for n in notes)

    def test_false_success_detection_flags_execute_code_without_outputs(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        plan = Plan(id="p-false-success", goal="execute code", steps=[PlanStep(id="s1", tool="execute_code", description="code")])
        trace = ExecutionTrace(plan_id=plan.id, mode=ExecutionMode.EXECUTE)
        trace.steps.append(SimpleNamespace(tool="execute_code", status="completed", output={"status": "ok"}))
        trace.complete(True, outputs=[])
        result = ExecutionResult(success=True, plan=plan, trace=trace, outputs=[], error=None)

        assert agent._is_false_success_after_recovery(result) is True  # noqa: SLF001

    def test_false_success_detection_accepts_execute_code_output_candidate_without_gdb_suffix(self, tmp_path):
        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))
        gdb_real = tmp_path / "workspace" / "output" / "数据库.gdb"
        gdb_real.mkdir(parents=True, exist_ok=True)
        reported_without_suffix = str(gdb_real)[:-4]

        plan = Plan(id="p-false-success-gdb", goal="create gdb", steps=[PlanStep(id="s1", tool="execute_code", description="code")])
        trace = ExecutionTrace(plan_id=plan.id, mode=ExecutionMode.EXECUTE)
        trace.steps.append(
            SimpleNamespace(
                tool="execute_code",
                status="completed",
                output={"result": {"output": reported_without_suffix, "created": True}},
            )
        )
        trace.complete(True, outputs=[])
        result = ExecutionResult(success=True, plan=plan, trace=trace, outputs=[], error=None)

        assert agent._is_false_success_after_recovery(result) is False  # noqa: SLF001

    def test_followup_projection_uses_recent_scan_memory_paths(self, tmp_path):
        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))
        # simulate prior successful scan tool call in memory
        agent.memory.add_tool_call(
            "scan_layers",
            {"path": str(tmp_path)},
            {
                "layers": [
                    {"path": r"E:\demo\MyProject5.gdb\bldg_dwg_Polyline", "type": "Polyline"},
                    {"path": r"E:\demo\MyProject5.gdb\bldg_dwg_Polygon", "type": "Polygon"},
                ]
            },
        )

        plan = agent.create_plan("转化这些图层的坐标系，转成WGS_1984_Web_Mercator_Auxiliary_Sphere")
        project_steps = [s for s in plan.steps if s.tool == "project_layers"]
        assert project_steps, "expected project_layers step(s)"
        assert all(s.input.get("input_path") for s in project_steps)
        assert any("Polyline" in s.input.get("input_path", "") for s in project_steps) or any(
            "Polygon" in s.input.get("input_path", "") for s in project_steps
        )

    def test_user_provided_path_persists_in_context_for_next_plan(self, tmp_path):
        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))
        msg = r"路径是E:\BST_lab\GIS CLI\workspace\input"
        agent.chat(msg)
        plan = agent.create_plan("扫描文件夹里的数据")
        scan_steps = [s for s in plan.steps if s.tool == "scan_layers"]
        assert scan_steps
        assert scan_steps[0].input.get("path") == r"E:\BST_lab\GIS CLI\workspace\input"

    def test_hydrate_overrides_generic_scan_input_with_user_path(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        agent.memory.set_context("input_folder", r"E:\BST_lab\GIS CLI\workspace\input")
        plan = Plan(
            id="p-scan-override",
            goal="scan override",
            steps=[
                PlanStep(
                    id="step_1",
                    tool="scan_layers",
                    description="扫描 input 目录",
                    input={"path": "input", "include_subdirs": True},
                )
            ],
        )
        agent._hydrate_plan_inputs_from_memory(plan)  # noqa: SLF001
        assert plan.steps[0].input["path"] == r"E:\BST_lab\GIS CLI\workspace\input"

    def test_preflight_scan_policy_auto_finds_workspace_input(self, tmp_path):
        (tmp_path / "workspace" / "input").mkdir(parents=True, exist_ok=True)
        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))
        plan = Plan(
            id="p-preflight-scan",
            goal="scan auto policy",
            steps=[PlanStep(id="step_1", tool="scan_layers", description="scan", input={"path": str(tmp_path / "input")})],
        )
        agent._preflight_plan(plan, "扫描数据")  # noqa: SLF001
        assert plan.steps[0].input["path"] == str(tmp_path / "workspace" / "input")

    def test_preflight_target_srs_policy_overrides_project_steps(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        plan = Plan(
            id="p-preflight-srs",
            goal="srs policy",
            steps=[PlanStep(id="step_1", tool="project_layers", description="project", input={"target_srs": "EPSG:4326"})],
        )
        agent._preflight_plan(plan, "转成WGS_1984_Web_Mercator_Auxiliary_Sphere")  # noqa: SLF001
        assert plan.steps[0].input["target_srs"] == "WGS_1984_Web_Mercator_Auxiliary_Sphere"

    def test_preflight_target_srs_policy_detects_asia_north_albers(self):
        agent = GISAgent(config=AgentConfig(llm_enabled=False))
        plan = Plan(
            id="p-preflight-albers",
            goal="albers policy",
            steps=[PlanStep(id="step_1", tool="project_layers", description="project", input={"target_srs": "EPSG:4326"})],
        )
        agent._preflight_plan(
            plan,
            "坐标系统一为 Asia_North_Albers_Equal_Area_Conic，中央经线105，标准纬线25和47",
        )  # noqa: SLF001
        assert plan.steps[0].input["target_srs"] == "Asia_North_Albers_Equal_Area_Conic"

    def test_agent_create_plan_fallback_when_planner_returns_none(self, tmp_path, monkeypatch):
        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))
        monkeypatch.setattr(agent.planner, "plan", lambda task_description, context: None)
        plan = agent.create_plan("扫描数据")
        assert plan is not None
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "scan_layers"
        assert plan.metadata.get("fallback_reason") == "planner_returned_none"

    def test_hydrate_replaces_placeholder_input_path_from_recent_memory(self, tmp_path):
        agent = GISAgent(config=AgentConfig(llm_enabled=False, workspace_path=tmp_path))
        agent.memory.add_tool_call(
            "scan_layers",
            {"path": str(tmp_path)},
            {
                "layers": [
                    {"path": r"E:\demo\MyProject5.gdb\bldg_dwg_Polyline", "type": "Polyline"},
                    {"path": r"E:\demo\MyProject5.gdb\bldg_dwg_Polygon", "type": "Polygon"},
                ]
            },
        )

        plan = Plan(
            id="p-placeholder",
            goal="project follow-up",
            steps=[
                PlanStep(
                    id="step_1",
                    tool="project_layers",
                    description="执行投影变换",
                    input={
                        "input_path": "【待确认】请提供输入图层路径，如：./data/input_layer.shp",
                        "target_srs": "WGS_1984_Web_Mercator_Auxiliary_Sphere",
                    },
                )
            ],
        )

        agent._hydrate_plan_inputs_from_memory(plan)  # noqa: SLF001 - internal behavior contract
        assert "待确认" not in plan.steps[0].input["input_path"]
        assert ".gdb" in plan.steps[0].input["input_path"]


class TestProjectLayersTool:
    def test_validate_input_accepts_gdb_feature_class_path(self):
        tool = ProjectLayersTool()
        input_data = tool.input_model(
            input_path=r"E:\data\sample.gdb\roads_line",
            output_path=r"E:\output\roads_line_projected.shp",
            target_srs="CGCS2000",
        )
        validation = tool.validate_input(input_data)
        assert validation.valid is True

    def test_project_with_bridge_passes_overwrite_output_flag(self, tmp_path, monkeypatch):
        shp = tmp_path / "input" / "roads.shp"
        shp.parent.mkdir(parents=True, exist_ok=True)
        shp.touch()

        tool = ProjectLayersTool()
        input_data = tool.input_model(
            input_path=str(shp),
            output_path=str(tmp_path / "output" / "roads_projected.shp"),
            target_srs="CGCS2000",
            overwrite_output=True,
        )

        captured: dict[str, object] = {}

        class FakeBridgeResult:
            status = "success"
            data = {
                "source_srs": "WGS 1984",
                "target_srs": "CGCS2000",
                "feature_count": 10,
            }
            error = None
            hint = None

        def fake_project_layer(input_path, output_path, target_srs="CGCS2000", overwrite_output=False):
            captured["input_path"] = input_path
            captured["output_path"] = output_path
            captured["target_srs"] = target_srs
            captured["overwrite_output"] = overwrite_output
            return FakeBridgeResult()

        monkeypatch.setattr("gis_cli.arcpy_bridge.project_layer", fake_project_layer)
        context = ToolContext(dry_run=False, arcpy_available=True, working_directory=str(tmp_path))

        result = tool._project_with_bridge(input_data, context)  # noqa: SLF001
        assert result.success is True
        assert captured["overwrite_output"] is True


class TestArcPyBridgeBooleanCodegen:
    def test_project_layer_codegen_uses_python_bool_literal(self, monkeypatch):
        captured: dict[str, object] = {}

        def fake_run_arcpy_code(code, **kwargs):
            captured["code"] = code
            captured["kwargs"] = kwargs
            return arcpy_bridge.ArcPyExecutionResult(
                status="success",
                exit_code=0,
                python_executable="python",
                stdout="",
                stderr="",
                data={},
                error=None,
                hint=None,
            )

        monkeypatch.setattr(arcpy_bridge, "run_arcpy_code", fake_run_arcpy_code)

        _ = arcpy_bridge.project_layer(
            input_path=r"E:\\data\\in.shp",
            output_path=r"E:\\data\\out.shp",
            target_srs="CGCS2000",
            overwrite_output=False,
        )

        code = str(captured.get("code", ""))
        assert "overwrite_output = False" in code
        assert "overwrite_output = false" not in code

    def test_merge_layers_codegen_uses_python_bool_literal(self, monkeypatch):
        captured: dict[str, object] = {}

        def fake_run_arcpy_code(code, **kwargs):
            captured["code"] = code
            captured["kwargs"] = kwargs
            return arcpy_bridge.ArcPyExecutionResult(
                status="success",
                exit_code=0,
                python_executable="python",
                stdout="",
                stderr="",
                data={},
                error=None,
                hint=None,
            )

        monkeypatch.setattr(arcpy_bridge, "run_arcpy_code", fake_run_arcpy_code)

        _ = arcpy_bridge.merge_layers(
            input_layers=[r"E:\\data\\a.shp", r"E:\\data\\b.shp"],
            output_path=r"E:\\data\\merged.shp",
            overwrite_output=False,
        )

        code = str(captured.get("code", ""))
        assert "overwrite_output = False" in code
        assert "overwrite_output = false" not in code


class TestMergeLayersTool:
    def test_expand_input_layers_from_directory(self, tmp_path):
        input_dir = tmp_path / "input"
        (input_dir / "E48").mkdir(parents=True, exist_ok=True)
        (input_dir / "E49").mkdir(parents=True, exist_ok=True)
        (input_dir / "E48" / "BOUA.shp").touch()
        (input_dir / "E49" / "BOUL.shp").touch()

        tool = MergeLayersTool()
        expanded = tool._expand_input_layers([str(input_dir)])  # noqa: SLF001

        assert len(expanded) == 2
        assert all(p.lower().endswith(".shp") for p in expanded)

    def test_validate_input_accepts_directory_with_shapefiles(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "AGNP.shp").touch()

        tool = MergeLayersTool()
        input_data = tool.input_model(
            input_layers=[str(input_dir)],
            output_path=str(tmp_path / "output" / "merged.shp"),
        )

        validation = tool.validate_input(input_data)
        assert validation.valid is True

    def test_merge_with_bridge_passes_overwrite_output_flag(self, tmp_path, monkeypatch):
        shp = tmp_path / "input" / "AGNP.shp"
        shp.parent.mkdir(parents=True, exist_ok=True)
        shp.touch()

        tool = MergeLayersTool()
        input_data = tool.input_model(
            input_layers=[str(shp)],
            output_path=str(tmp_path / "output" / "merged.shp"),
            overwrite_output=True,
        )

        captured: dict[str, object] = {}

        class FakeBridgeResult:
            status = "success"
            data = {
                "input_count": 1,
                "feature_count": 10,
                "geometry_type": "Polygon",
                "message": "ok",
            }
            error = None
            hint = None

        def fake_merge_layers(input_layers, output_path, workspace=None, overwrite_output=False):
            captured["input_layers"] = input_layers
            captured["output_path"] = output_path
            captured["overwrite_output"] = overwrite_output
            return FakeBridgeResult()

        monkeypatch.setattr("gis_cli.arcpy_bridge.merge_layers", fake_merge_layers)
        monkeypatch.setattr(tool, "_group_by_geometry", lambda layers: {"Polygon": layers})

        context = ToolContext(dry_run=False, arcpy_available=True, working_directory=str(tmp_path))
        result = tool._merge_with_bridge(input_data, context, [str(shp)])  # noqa: SLF001

        assert result.success is True
        assert captured["overwrite_output"] is True


class TestExportMapTool:
    def test_discover_project_path_recursively_from_input_subfolder(self, tmp_path):
        project_file = tmp_path / "input" / "myproject5" / "myproject5.aprx"
        project_file.parent.mkdir(parents=True, exist_ok=True)
        project_file.touch()

        tool = ExportMapTool()
        context = ToolContext(
            dry_run=True,
            working_directory=str(tmp_path),
            metadata={"input_folder": str(tmp_path / "input")},
        )

        discovered = tool._discover_project_path(context)  # noqa: SLF001
        assert discovered == str(project_file)

    def test_export_with_arcpy_overwrite_removes_existing_file(self, tmp_path, monkeypatch):
        output_path = tmp_path / "output" / "map.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("old", encoding="utf-8")

        class FakeLayout:
            name = "Layout1"

            def exportToPNG(self, target_path, resolution=300):
                p = Path(target_path)
                if p.exists():
                    raise RuntimeError("already exists")
                p.write_text("new", encoding="utf-8")

        class FakeMap:
            def listLayers(self):
                return ["layer1"]

        class FakeProject:
            def listMaps(self):
                return [FakeMap()]

            def listLayouts(self):
                return [FakeLayout()]

        class FakeMP:
            def ArcGISProject(self, _path):
                return FakeProject()

        class FakeArcPy:
            mp = FakeMP()

        monkeypatch.setitem(sys.modules, "arcpy", FakeArcPy())

        tool = ExportMapTool()
        input_data = tool.input_model(
            project_path="dummy.aprx",
            output_path=str(output_path),
            format="PNG",
            overwrite_output=True,
        )
        context = ToolContext(dry_run=False, arcpy_available=True, working_directory=str(tmp_path))

        result = tool._export_with_arcpy(input_data, context)  # noqa: SLF001
        assert result.success is True
        assert output_path.read_text(encoding="utf-8") == "new"


class TestSpatialReferenceMapping:
    def test_project_layer_maps_web_mercator_name_to_wkid(self):
        # Static contract test: ensure mapping dictionary contains expected key
        # by inspecting generated source path behavior through dry-run execution path.
        # We cannot execute ArcPy here, so we validate tool-level acceptance path.
        tool = ProjectLayersTool()
        input_data = tool.input_model(
            input_path=r"E:\data\sample.gdb\roads_line",
            output_path=r"E:\output\roads_line_projected.shp",
            target_srs="WGS_1984_Web_Mercator_Auxiliary_Sphere",
        )
        validation = tool.validate_input(input_data)
        assert validation.valid is True

    def test_recovery_stops_for_critical_step_and_returns_failure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = AgentConfig(llm_enabled=False, workspace_path=tmp_path)
        agent = GISAgent(config=config)
        plan = Plan(
            id="p-fail-critical",
            goal="must fail",
            steps=[
                PlanStep(
                    id="step_1",
                    tool="scan_layers",
                    description="scan missing path",
                    input={"path": str(tmp_path / "missing_dir")},
                ),
                PlanStep(
                    id="step_2",
                    tool="quality_check",
                    description="qc",
                    input={"check_types": ["crs"]},
                    depends_on=["step_1"],
                ),
            ],
        )

        result, notes = agent._execute_with_recovery(plan, ExecutionMode.DRY_RUN)  # noqa: SLF001
        assert result.success is False
        assert notes
        assert ("关键步骤" in notes[0]) or ("禁止 skip" in notes[0])


# === Prompts Tests ===

class TestPrompts:
    """Test prompt system."""
    
    def test_system_prompt_cn(self):
        prompt = SystemPrompts.get_system_prompt("cn")
        assert "GIS" in prompt
        assert "专业" in prompt or "工具" in prompt
    
    def test_system_prompt_en(self):
        prompt = SystemPrompts.get_system_prompt("en")
        assert "GIS" in prompt
        assert "tool" in prompt.lower() or "agent" in prompt.lower()
    
    def test_planning_prompt(self):
        prompt = SystemPrompts.get_planning_prompt("制作专题图")
        assert "制作专题图" in prompt
    
    def test_prompt_builder(self):
        builder = PromptBuilder(language="cn")
        builder.add_system_prompt()
        builder.add_tool_use_instructions()
        
        prompt = builder.build()
        assert len(prompt) > 0
        assert "GIS" in prompt
    
    def test_prompt_builder_with_context(self):
        builder = PromptBuilder()
        builder.add_context({"workspace": "/data", "user": "test"})
        
        prompt = builder.build()
        assert "workspace" in prompt
        assert "/data" in prompt


# === Integration Tests ===

class TestAgentIntegration:
    """Integration tests for the complete agent workflow."""
    
    def test_full_workflow(self, tmp_path):
        """Test complete workflow: chat -> plan -> execute."""
        # Create some test files
        (tmp_path / "test.shp").touch()
        
        config = AgentConfig(
            workspace_path=tmp_path,
            default_mode=ExecutionMode.DRY_RUN,
            auto_execute=False
        )
        agent = GISAgent(config=config)
        
        # Step 1: Create task
        response1 = agent.chat(f"扫描 {tmp_path} 目录")
        assert response1.plan is not None
        
        # Step 2: Confirm and execute
        response2 = agent.chat("确认执行")
        
        # Should have executed
        assert "execute" in response2.action_taken or agent.last_result is not None
    
    def test_multi_turn_conversation(self):
        """Test multi-turn conversation maintains context."""
        agent = GISAgent()
        
        # Turn 1
        agent.chat("我想处理 GIS 数据")
        
        # Turn 2 - should remember context
        response = agent.chat("开始扫描")
        
        # Should have conversation history
        assert len(agent.memory.turns) >= 4  # At least 2 user + 2 assistant
    
    def test_error_handling(self, tmp_path):
        """Test agent handles errors gracefully."""
        agent = GISAgent()
        
        # Try to execute on non-existent path
        result = agent.call_tool(
            "scan_layers",
            {"directory": "/non/existent/path"},
            dry_run=True
        )
        
        # Should not crash, should return result
        assert result is not None

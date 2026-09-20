# -*- coding: utf-8 -*-
"""错误记忆 / 操作目录 / 项目日志 的单元测试（不需要 ArcGIS）。"""
from __future__ import annotations

import json
from pathlib import Path

from gis_cli.engine.llm_client import LLMResponse, ToolCall
from gis_cli.engine.loop import AgentLoop, LoopConfig
from gis_cli.engine.project_log import build_entry, digest, log_path, read_recent, record_run
from gis_cli.engine.tools import EngineContext, build_default_registry
from gis_cli.recipes.store import RecipeLibrary
from gis_cli.runtime.error_memory import enrich_hint, match
from gis_cli.trace.recorder import RunRecorder


# ------------------------------------------------------------------ 错误记忆
def test_error_memory_matches_sa_module_trap():
    hints = match("AttributeError: module 'arcpy.sa' has no attribute 'GetRasterProperties'")
    keys = [h.key for h in hints]
    assert "sa-module" in keys
    rendered = hints[0].render()
    assert "arcpy.management" in rendered


def test_error_memory_matches_field_and_gdb_traps():
    assert [h.key for h in match("ERROR 000539: 无效字段 常住人口")] == ["field-539"]
    assert "gdb" in match("ERROR 000354: 名称无效")[0].fix.lower()
    assert [h.key for h in match("ERROR 000732: 输入不存在")] == ["path-732"]


def test_error_memory_matches_layer_and_domain_traps():
    layer = match("ERROR 001628: 该值并非图层或具有连接的表")
    assert layer and layer[0].key == "need-layer-1628"
    assert "MakeFeatureLayer" in layer[0].fix
    domain = match("ERROR 000864: 属性类型: 输入不在定义域内")
    assert domain and domain[0].key == "wrong-type-864"


def test_error_memory_ignores_unknown_text_and_enriches_once():
    assert match("一切正常") == []
    first = enrich_hint("", "ERROR 000622: 参数无效")
    assert first.startswith("[错误记忆/param-622]")
    merged = enrich_hint(first, "ERROR 000622: 参数无效")
    assert merged.count("[错误记忆/param-622]") == 1
    assert merged.startswith("[错误记忆/param-622]")


def test_error_memory_caps_hint_count():
    text = "ERROR 000539 字段无效; ERROR 000354 名称无效; ERROR 000622 参数无效; ERROR 000732 输入不存在"
    assert len(match(text, limit=2)) == 2


# ------------------------------------------------------------------ 操作目录
def test_recipe_catalog_groups_by_category_and_lists_params():
    lib = RecipeLibrary()
    lib.load()
    catalog = lib.catalog_digest()
    assert "栅格" in catalog and "raster_calculator" in catalog
    assert "必填:" in catalog
    # 老配方没有 category 字段时应回落到 domain，不至于丢失
    assert "project_to_crs" in catalog and "buffer_dissolve" in catalog
    assert len(catalog) <= 5200


def test_recipe_category_defaults_to_domain():
    lib = RecipeLibrary()
    lib.load()
    by_id = {r.id: r for r in lib.all()}
    assert by_id["raster_calculator"].category == "raster"
    assert by_id["project_to_crs"].category  # 由 domain 回落，不能为空


# ------------------------------------------------------------------ 项目日志
def test_project_log_record_and_read(tmp_path: Path):
    assert read_recent(tmp_path) == ""
    assert digest(tmp_path) == ""
    record_run(
        tmp_path,
        goal="做一张图",
        status="completed",
        turns=5,
        artifacts=[str(tmp_path / "output" / "a.jpg")],
        summary="图已产出，面积合计 100 km²",
        hints=["[错误记忆/field-539] 字段名或表达式不合法 → 先查真实字段名"],
        run_id="run_1",
    )
    text = read_recent(tmp_path)
    assert "做一张图" in text
    assert "已完成" in text
    assert "output" in text and "a.jpg" in text
    assert "字段名或表达式不合法" in text
    assert log_path(tmp_path).exists()
    assert "项目日志" in digest(tmp_path)


def test_project_log_keeps_recent_entries_only(tmp_path: Path):
    for index in range(6):
        record_run(
            tmp_path,
            goal=f"任务 {index}",
            status="completed",
            turns=index,
            artifacts=[],
            summary="x" * 400,
        )
    recent = read_recent(tmp_path, max_chars=900)
    assert "任务 5" in recent
    assert "任务 0" not in recent


def test_project_log_entry_is_deterministic_shape():
    entry = build_entry(
        goal="目标",
        status="max_turns",
        turns=7,
        artifacts=[],
        workspace=".",
        summary="",
        run_id="r",
    )
    assert entry.startswith("## ")
    assert "轮次耗尽" in entry
    assert "产物: 0 个" in entry


# ------------------------------------------------------------------ 循环接线
class ScriptedLLM:
    class _Cfg:
        model = "mock"

    config = _Cfg()

    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, messages, **kwargs):
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(
            content=json.dumps({"tool": "finish", "args": {"summary": "done"}}), model="mock"
        )


class FailingRegistry:
    """所有工具都失败，用来验证错误记忆接线。"""

    def __init__(self, base):
        self.base = base

    def schemas(self):
        return self.base.schemas()

    def execute(self, tool, args, ctx):
        if tool == "finish":
            return self.base.execute(tool, args, ctx)
        from gis_cli.engine.state import Observation

        return Observation(
            ok=False,
            summary="工具执行失败",
            error={"type": "ExecuteError", "message": "module 'arcpy.sa' has no attribute 'GetRasterProperties'"},
        )


class StubVerifier:
    def verify(self, state):
        return {"pass": True, "summary": "通过"}


def _tc(name, args):
    return LLMResponse(
        tool_calls=[ToolCall(id=f"c_{name}", name=name, arguments=args, raw_arguments=json.dumps(args))],
        model="mock",
    )


def _make_loop(tmp_path: Path, responses, registry=None):
    llm = ScriptedLLM(responses)
    base = build_default_registry()
    loop = AgentLoop(
        workspace=tmp_path,
        llm_client=llm,
        registry=registry or base,
        context=EngineContext(workspace=tmp_path),
        verifier=StubVerifier(),
        recorder=RunRecorder("run_unit", tmp_path / "traces"),
        config=LoopConfig(max_turns=6),
        auto_extract_requirements=False,
    )
    return loop, llm


def test_system_prompt_contains_discipline_and_catalog(tmp_path: Path):
    lib = RecipeLibrary()
    lib.load()
    loop, _ = _make_loop(
        tmp_path,
        [_tc("finish", {"summary": "ok"})],
    )
    loop.recipes = lib
    loop.run("测试目标")
    prompt = loop._messages[0]["content"]
    assert "方法论纪律" in prompt
    assert "先看后算" in prompt
    assert "操作目录" in prompt
    assert "raster_calculator" in prompt
    assert f"共 {len(lib.all())} 个" in prompt


def test_failure_observation_gets_error_memory_hint(tmp_path: Path):
    base = build_default_registry()
    loop, _ = _make_loop(
        tmp_path,
        [
            _tc("catalog_query", {"query": "x"}),
            _tc("finish", {"summary": "ok"}),
        ],
        registry=FailingRegistry(base),
    )
    state = loop.run("测试失败注入")
    assert state.status == "completed"
    messages = json.dumps(loop._messages, ensure_ascii=False)
    assert "错误记忆/sa-module" in messages


def test_run_writes_project_log_and_resumes_context(tmp_path: Path):
    loop, _ = _make_loop(
        tmp_path,
        [_tc("finish", {"summary": "已产出图与统计表"})],
    )
    loop.run("第一个任务")
    text = read_recent(tmp_path)
    assert "第一个任务" in text
    assert "已产出图与统计表" in text

    loop2, _ = _make_loop(tmp_path, [_tc("finish", {"summary": "第二个任务完成"})])
    loop2.run("第二个任务")
    first_user = loop2._messages[1]["content"]
    assert "项目日志" in first_user and "第一个任务" in first_user

def test_workspace_hygiene_warns_once_on_stray_dir(tmp_path: Path):
    """Windows 路径转义错误会在工作区根目录造出假目录，必须尽早提示且只提示一次。"""
    loop, _ = _make_loop(tmp_path, [_tc("finish", {"summary": "ok"})])

    from gis_cli.engine.state import Observation

    stray = tmp_path / "data"
    stray.mkdir()
    observation = Observation(ok=True, summary="执行成功")
    loop._check_workspace_hygiene(observation)
    assert "工作区根目录出现预期外的项" in observation.hint
    assert "data" in observation.hint

    again = Observation(ok=True, summary="执行成功")
    loop._check_workspace_hygiene(again)
    assert again.hint == ""

    (tmp_path / "output").mkdir()
    (tmp_path / "input").mkdir()
    clean = Observation(ok=True, summary="执行成功")
    loop._check_workspace_hygiene(clean)
    assert clean.hint == ""


def test_verifier_prunes_deleted_artifacts(tmp_path: Path):
    """已删除的产物应从产物清单剔除，否则验收永远报'不在 output 目录内'。"""
    from gis_cli.engine.state import TaskState
    from gis_cli.engine.verifier import Verifier

    (tmp_path / "output").mkdir()
    good = tmp_path / "output" / "result.csv"
    good.write_text("a,b\n1,2\n", encoding="utf-8")
    gone = tmp_path / "data" / "stray" / "fake.csv"
    gone.parent.mkdir(parents=True)
    gone.write_text("x", encoding="utf-8")

    state = TaskState(run_id="r", goal="g", workspace=str(tmp_path))
    state.artifacts = [str(good), str(gone)]
    gone.unlink()

    verifier = Verifier(workspace=tmp_path)
    verdict = verifier.verify(state)
    assert str(gone) not in state.artifacts
    assert not any("不在 output 目录内" in str(p) for p in verdict.get("problems", []))


def test_soft_recipe_hint_on_successful_code():
    """代码成功但操作有对应配方时，给一句不阻断的提示（提高可复现性）。"""
    from gis_cli.engine.tools import EngineContext, _recipe_soft_hint

    lib = RecipeLibrary()
    lib.load()
    ctx = EngineContext(workspace=Path("."), recipes=lib)
    hint = _recipe_soft_hint(ctx, "相交绿地和社区", "arcpy.analysis.Intersect([a, b], out)")
    assert "intersect_layers" in hint and "不阻断" in hint
    assert _recipe_soft_hint(ctx, "读一张表", "print(1)") == ""


def test_error_memory_matches_layer_select_hang():
    hints = match("调用 SelectLayerByAttribute 后执行超时：Execution exceeded 120.0s and did not stop after interrupt")
    assert hints and hints[0].key == "layer-select-hang"
    assert "FeatureClassToFeatureClass" in hints[0].fix


def test_benchmark_runner_survives_arcpy_datetime_pollution():
    """arcpy 会把调用方命名空间里的 datetime 换成模块，runner 不能用裸 datetime 名字。

    基准目录只存在于开发环境；发布版没有 benchmark/，此时跳过（回归防护仍在开发版生效）。
    """
    runner_path = Path("benchmark/runner.py")
    if not runner_path.exists():
        import pytest

        pytest.skip("发布版不含 benchmark/，跳过该回归防护")
    src = runner_path.read_text(encoding="utf-8")
    assert "import datetime as dt" in src
    assert "\nfrom datetime import datetime" not in src
    assert "datetime.now()" not in src.replace("dt.datetime.now()", "")


def test_new_assertion_kinds_registered_and_fail_gracefully(tmp_path: Path):
    """新增断言（字段按值计数 / 工程核验）在无 arcpy 或文件缺失时应失败而不是抛异常。"""
    from gis_cli.recipes.store import RecipeLibrary

    lib = RecipeLibrary()
    missing = str(tmp_path / "nope.shp")
    r1 = lib.run_assertions(
        [{"type": "field_value_counts", "path": missing, "field": "类别", "value": {"标杆社区": 32}}]
    )[0]
    assert r1.ok is False and "实际=None" in r1.detail
    r2 = lib.run_assertions([{"type": "aprx_map_check", "path": str(tmp_path / "nope.aprx")}])[0]
    assert r2.ok is False and "不存在" in r2.detail


def test_error_memory_matches_aprx_open_hang():
    hints = match("调用 mp.ArcGISProject 后执行超时：Execution exceeded 300.0s and did not stop after interrupt")
    assert hints and any(h.key == "aprx-open-hang" for h in hints)
    assert "check_map_project" in hints[0].fix


def test_layer_select_rule_does_not_steal_other_timeouts():
    """泛化超时文本不该归因到图层选择规则（否则会给出错误建议）。"""
    keys = [h.key for h in match("Execution exceeded 300.0s and did not stop after interrupt")]
    assert "layer-select-hang" not in keys

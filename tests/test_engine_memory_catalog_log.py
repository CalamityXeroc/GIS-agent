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


def test_verifier_drops_artifacts_outside_workspace(tmp_path):
    """回归：外部路径不能当成 blocked 问题（14 届 4a-1 空转到轮次上限）。

    ArcGIS 模板工程与用户 Documents 下的工程被登记后，验收每轮都报
    “产出不在 output 目录内”，断言 7/7 全过却永远完成不了。
    """
    from gis_cli.engine.state import TaskState
    from gis_cli.engine.verifier import Verifier

    out = tmp_path / "output" / "成果图"
    out.mkdir(parents=True)
    good = out / "密度_1982_1986.jpg"
    good.write_bytes(b"\xff\xd8\xff" + b"0" * 20_000)  # 假 JPG（足够长，避开空白判定）

    state = TaskState(run_id="r", goal="g", workspace=str(tmp_path))
    state.artifacts = [
        str(good),
        r"E:\ArcGISPro3.6\Resources\ArcToolBox\Services\routingservices\data\Blank.aprx",
        r"C:\Users\<user>\Documents\ArcGIS\Projects\MyProject\MyProject.aprx",
    ]

    verifier = Verifier(workspace=tmp_path)
    verdict = verifier.verify(state)
    assert not any("不在 output 目录内" in str(p) for p in verdict.get("problems", [])), verdict
    assert not any("Blank.aprx" in str(p) or "MyProject" in str(p) for p in state.artifacts), state.artifacts
    assert any("密度_1982_1986.jpg" in str(p) for p in state.artifacts), state.artifacts


def test_finish_ignores_artifacts_outside_workspace(tmp_path):
    """finish 传的 artifacts 若在 workspace 外，不能登记进产物清单。"""
    from gis_cli.engine.state import TaskState
    from gis_cli.engine.tools import EngineContext, build_default_registry

    state = TaskState(run_id="r", goal="g", workspace=str(tmp_path))
    ctx = EngineContext(workspace=tmp_path, state=state)
    registry = build_default_registry()
    good = tmp_path / "output" / "a.gdb"
    good.mkdir(parents=True)
    input_shp = tmp_path / "input" / "animal_points.shp"
    input_shp.parent.mkdir(parents=True)
    input_shp.write_text("x", encoding="utf-8")
    registry.execute(
        "finish",
        {
            "summary": "done",
            "artifacts": [
                str(good),
                str(input_shp),
                r"C:\Users\<user>\Documents\ArcGIS\Projects\MyProject\MyProject.aprx",
            ],
        },
        ctx,
    )
    assert not any("MyProject" in str(p) for p in state.artifacts), state.artifacts
    assert not any("animal_points" in str(p) for p in state.artifacts), state.artifacts
    assert any("a.gdb" in str(p) for p in state.artifacts), state.artifacts


def test_finish_keeps_spec_json_and_gdb_members(tmp_path):
    """回归：finish 里的 .spec.json 与 GDB 内要素不能被过滤掉。

    实测（14 届 4a-1）：模型在 finish 里正确列了 16 个产物（含 4 份 .spec.json、
    4 个 GDB 内栅格），但登记时的过滤把它们全丢了——GIS_OUTPUT_EXT 没有 .json，
    而 GDB 内要素本来就没扩展名。结果语义验收看不到证据，反复说“缺失 .spec.json”，
    agent 空转到轮次上限。
    """
    from gis_cli.engine.state import TaskState
    from gis_cli.engine.tools import EngineContext, build_default_registry

    state = TaskState(run_id="r", goal="g", workspace=str(tmp_path))
    ctx = EngineContext(workspace=tmp_path, state=state)
    registry = build_default_registry()

    (tmp_path / "output" / "成果图").mkdir(parents=True)
    (tmp_path / "output" / "result_data.gdb").mkdir(parents=True)
    for name in ("密度_1982_1986.spec.json", "密度_1982_1986.jpg"):
        (tmp_path / "output" / "成果图" / name).write_text("x", encoding="utf-8")
    # GDB 内要素在文件系统上不存在，但路径本身是合法的产物
    gdb_member = str(tmp_path / "output" / "result_data.gdb" / "kde_1982_1986")

    registry.execute(
        "finish",
        {
            "summary": "done",
            "artifacts": [
                gdb_member,
                str(tmp_path / "output" / "成果图" / "密度_1982_1986.spec.json"),
                str(tmp_path / "output" / "成果图" / "密度_1982_1986.jpg"),
            ],
        },
        ctx,
    )
    assert any("kde_1982_1986" in str(p) for p in state.artifacts), state.artifacts
    assert any("spec.json" in str(p) for p in state.artifacts), state.artifacts
    assert any(str(p).endswith(".jpg") for p in state.artifacts), state.artifacts


def test_semantic_check_includes_spec_contents(tmp_path):
    """回归：语义验收必须看到 .spec.json 的**实际内容**（而不只是路径）。

    实测（14 届 4a-1）：以前只把 structural（路径+存在性）给模型，模型看不到
    class_bounds 的实际值，于是无论 agent 做什么都回“无法确认四图分级是否一致”，
    agent 空转到轮次上限（验收 problems 明明是空的）。
    """
    from gis_cli.engine.state import Requirement, TaskState
    from gis_cli.engine.verifier import Verifier

    spec_dir = tmp_path / "output" / "成果图"
    spec_dir.mkdir(parents=True)
    (spec_dir / "a.spec.json").write_text(
        json.dumps(
            {"decisions": [{"key": "explicit_bounds", "value": [0, 2, 4, 6, 8]}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    seen: dict = {}

    class _LLM:
        def chat(self, messages, **kwargs):
            seen["prompt"] = str(messages[-1]["content"])

            class _R:
                content = '{"pass": true, "per_requirement": [], "missing": []}'

            return _R()

    state = TaskState(run_id="r", goal="g", workspace=str(tmp_path))
    state.requirements = [Requirement(id="r1", text="四图共用同一图例")]

    verdict = Verifier(workspace=tmp_path, llm_client=_LLM()).semantic_check(state, [])
    assert verdict.get("pass") is True, verdict
    assert "explicit_bounds" in seen["prompt"], "spec.json 的实际内容必须进 prompt"
    assert "a.spec.json" in seen["prompt"], seen["prompt"][:400]


def test_artifact_exists_keeps_gdb_members_without_arcpy(tmp_path):
    """回归：没有 arcpy 时不能把 GDB 内要素判成“不存在”。

    实测（14 届 4a-1）：Verifier 跑在 benchmark 的 venv 里，venv 没装 arcpy，
    arcpy.Exists 抛 ModuleNotFoundError → 旧实现返回 False → 四个 kde_* 栅格
    被当成“已删除的产物”全部剔除，验收只能说“缺失栅格元数据”，agent 空转。
    现在判不了就当作存在，交给后面跑在内核里的 structural_check 去实测。
    """
    from gis_cli.engine.verifier import _artifact_exists

    # GDB 内成员：文件系统不存在，但不应被剔除
    assert _artifact_exists(str(tmp_path / "output" / "a.gdb" / "kde_1982_1986")) is True
    # 普通文件：确实不存在就该是 False
    assert _artifact_exists(str(tmp_path / "output" / "nope.jpg")) is False
    # 真实存在的普通文件：True
    real = tmp_path / "output" / "real.jpg"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"x")
    assert _artifact_exists(str(real)) is True


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
    # 无内核且无 arcpy 时必须透传真实错误（不得静默吞掉只剩空 {}）
    assert r1.ok is False and "字段读取失败" in r1.detail and "arcpy" in r1.detail
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


# ------------------------------------------------------------------ 断言：字段计数走内核
class _FakeOutcome:
    def __init__(self, result=None, ok=True, error=""):
        self.result = result
        self.ok = ok
        self.error = error


class _FakeCodeRunner:
    """记录收到的内核代码，返回预设结果（不依赖 arcpy）。"""

    def __init__(self, result=None, ok=True, error=""):
        self.result = result
        self.ok = ok
        self.error = error
        self.codes: list[str] = []

    def run(self, code, timeout=0):
        self.codes.append(code)
        return _FakeOutcome(result=self.result, ok=self.ok, error=self.error)


def test_field_counts_all_prefers_kernel(tmp_path):
    """field_value_counts 必须走 code_runner（内核），不能在断言进程内 import arcpy。"""
    lib = RecipeLibrary()
    runner = _FakeCodeRunner(result={"counts": {"标杆社区": 32, "需整改社区": 500, "其他": 1809}})
    lib.code_runner = runner
    counts, err = lib._field_counts_all("gdb://fc", "类别")
    assert err == ""
    assert counts == {"标杆社区": 32, "需整改社区": 500, "其他": 1809}
    assert runner.codes and "SearchCursor" in runner.codes[0]


def test_field_value_counts_between_surfaces_kernel_error(tmp_path):
    """内核失败时错误文本必须透传到 detail，不能只剩空 {}（14 届实测教训）。"""
    lib = RecipeLibrary()
    lib.code_runner = _FakeCodeRunner(ok=False, error="kernel destroyed")
    results = lib.run_assertions(
        [
            {
                "type": "field_value_counts_between",
                "path": "gdb://fc",
                "field": "NAME",
                "value": {"Giraffe": [1700, 1850]},
            }
        ]
    )
    assert results[0].ok is False
    # 内核失败后落到进程内兕底；无 arcpy 的测试环境里进程内也会失败，但错误必须透传（不能是空 {}）
    assert results[0].detail.startswith("字段读取失败: ")
    assert results[0].detail != "字段读取失败: {}"
    assert "arcpy" in results[0].detail or "kernel destroyed" in results[0].detail


def test_field_value_counts_between_band_check(tmp_path):
    """区间断言：类别计数落在给定区间内通过，越界/缺失判失败。"""
    lib = RecipeLibrary()
    lib.code_runner = _FakeCodeRunner(result={"counts": {"Giraffe": 1793, "Gazelle": 1159, "Zebra": 16}})
    ok_res = lib.run_assertions(
        [
            {
                "type": "field_value_counts_between",
                "path": "gdb://fc",
                "field": "NAME",
                "value": {"Giraffe": [1700, 1850], "Gazelle": [1100, 1220], "Zebra": [10, 40]},
            }
        ]
    )
    assert ok_res[0].ok is True
    bad_res = lib.run_assertions(
        [
            {
                "type": "field_value_counts_between",
                "path": "gdb://fc",
                "field": "NAME",
                "value": {"Giraffe": [1800, 1850]},
            }
        ]
    )
    assert bad_res[0].ok is False


class _WarnOutcome:
    """成功但带 stderr 告警的执行结果（模拟 matplotlib 缺中文字形）。"""

    ok = True
    stdout = "绘图完成"
    stderr = "Glyph 20013 (\N{CJK UNIFIED IDEOGRAPH-4E2D}) missing from font(s) DejaVu Sans."
    result = {"output": "output/图.png"}
    duration_ms = 12
    display_images: list = []
    error = None


def test_execute_code_surfaces_stderr_warnings(tmp_path):
    """成功执行但 stderr 有告警必须回喂模型（否则'中文变方框'的图会一路通过验收）。"""
    from gis_cli.engine.tools import EngineContext, build_default_registry

    class _Runner:
        def run(self, code, timeout=None, workspace=None):
            return _WarnOutcome()

    ctx = EngineContext(workspace=tmp_path)
    ctx.code_runner = _Runner()
    registry = build_default_registry()
    obs = registry.execute(
        "execute_code", {"code": "import matplotlib.pyplot as plt", "description": "画对比图"}, ctx
    )
    assert obs.ok is True
    assert "stderr" in obs.data, "成功路径也要带上 stderr 告警"
    assert "mpl-cjk" in obs.hint, obs.hint


# ------------------------------------------------- .spec.json 读入（崩溃回归）


def test_shrink_json_always_yields_valid_json():
    """超长验收材料缩水后必须是合法 JSON。

    实测崩溃：旧代码把 json.dumps 的结果直接切到 max_chars，截在字符串中间
    → json.loads 抛 JSONDecodeError → 整个 run 在验收阶段挂掉
    （一个内容不完整的 .spec.json 就能把任务拖死）。
    """
    import json

    from gis_cli.engine.verifier import _shrink_json

    big = {
        "class_bounds": [0, 1, 2, 3, 4, 5],
        "title": "标题" * 3000,
        "decisions": ["说明" * 400] * 30,
        "renderer": {"type": "graduated", "ramp": "Reds", "extra": "x" * 5000},
    }
    for budget in (50, 200, 1181, 4000, 24000, 200000):
        shrunk = _shrink_json(big, budget)
        text = json.dumps(shrunk, ensure_ascii=False)
        parsed = json.loads(text)  # 不抛即通过
        assert isinstance(parsed, dict)
        assert "class_bounds" in parsed, "关键字段不能丢"


def test_spec_contents_survives_broken_and_huge_spec(tmp_path):
    """_read_spec_contents 面对超大/畸形 .spec.json 都不能抛异常。"""
    import json

    from gis_cli.engine.verifier import Verifier

    out = tmp_path / "output"
    out.mkdir(parents=True)
    # 超大但合法
    (out / "big.spec.json").write_text(
        json.dumps({"class_bounds": [0, 1, 2], "title": "x" * 8000}, ensure_ascii=False),
        encoding="utf-8",
    )
    # 畸形（不是合法 JSON）
    (out / "broken.spec.json").write_text("{not json at all", encoding="utf-8")

    verifier = Verifier(workspace=tmp_path)
    result = verifier._read_spec_contents()
    assert isinstance(result, list)
    paths = [item.get("path") for item in result]
    # 超大那个要能读到，且里面是合法结构
    assert any("big.spec.json" in str(p) for p in paths), paths
    for item in result:
        assert isinstance(item.get("content"), (dict, str))


# ------------------------------------------------- 识图质检（批量 + 缓存）


class _VisionLLM:
    """假的多模态客户端：记录调用次数，返回可预测的 JSON。"""

    def __init__(self, *, vision=True, per_image_ok=True):
        self.calls: list[list[dict]] = []
        self.vision_rejected = False
        self.config = type("C", (), {"vision": vision})()

        # 需要一个能被 from_file 之类读到的 config.model
        class _Resp:
            content = ""
            model = "vision-stub"

        self._per_image_ok = per_image_ok

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        # 数一下这次请求带了几张图
        images = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                images += sum(
                    1 for part in content if isinstance(part, dict) and part.get("type") == "image_url"
                )
        import json as _json

        class _R:
            model = "vision-stub"

        _R.content = _json.dumps(
            {"images": [{"index": i + 1, "text_ok": True, "issues": [], "summary": f"ok{i+1}"} for i in range(images)]}
        )
        return _R()


def _make_images(tmp_path, count=3):
    from PIL import Image

    paths = []
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        path = out / f"图{index + 1}.jpg"
        Image.new("RGB", (60, 40), (index * 40 % 255, 90, 140)).save(path)
        paths.append(path)
    return paths


def test_vision_batch_checks_all_images_in_one_call(tmp_path):
    """多张图应一次调用看完（实测逐张串行时 5 张图要 200s+）。"""
    from gis_cli.engine.verifier import Verifier

    paths = _make_images(tmp_path, 3)
    llm = _VisionLLM()
    verifier = Verifier(workspace=tmp_path, llm_client=llm, vision_mode="warn")
    verdicts = verifier.vision_check_images(paths)
    assert len(llm.calls) == 1, f"应只调用一次，实际 {len(llm.calls)} 次"
    assert set(verdicts) == {0, 1, 2}
    assert all(v["ok"] for v in verdicts.values())


def test_vision_verdict_is_cached_between_verify_calls(tmp_path):
    """同一批产物验两次时，第二遍必须命中缓存（不再看图）。

    一个 run 里 verify() 会被调两次（verify_outputs 与 finish），
    没有缓存就要把每张图看两遍。
    """
    from gis_cli.engine.verifier import Verifier

    paths = _make_images(tmp_path, 2)
    llm = _VisionLLM()
    verifier = Verifier(workspace=tmp_path, llm_client=llm, vision_mode="warn")
    records = [{"path": str(p), "image": {"ok": True}} for p in paths]

    verifier._vision_verdicts(records)
    assert len(llm.calls) == 1, "第一遍应只调一次（批量）"
    assert all("vision" in r for r in records)

    # 第二遍（模拟 finish 再次验收）：命中缓存，不应再发请求
    records2 = [{"path": str(p), "image": {"ok": True}} for p in paths]
    verifier._vision_verdicts(records2)
    assert len(llm.calls) == 1, "第二遍应命中缓存"
    assert all("vision" in r for r in records2)


def test_vision_cache_invalidates_when_image_changes(tmp_path):
    """图片重出（mtime 变了）后缓存要失效，否则新图的问题看不出来。"""
    import os
    import time

    from gis_cli.engine.verifier import Verifier

    paths = _make_images(tmp_path, 1)
    llm = _VisionLLM()
    verifier = Verifier(workspace=tmp_path, llm_client=llm, vision_mode="warn")
    records = [{"path": str(paths[0]), "image": {"ok": True}}]
    verifier._vision_verdicts(records)
    assert len(llm.calls) == 1

    future = time.time() + 60
    os.utime(paths[0], (future, future))
    verifier._vision_verdicts([{"path": str(paths[0]), "image": {"ok": True}}])
    assert len(llm.calls) == 2, "图片变了必须重新看"


def test_vision_check_skipped_when_disabled(tmp_path):
    """关掉识图 / 模型不支持识图时，不应产生任何请求，且不能报错。"""
    from gis_cli.engine.verifier import Verifier

    paths = _make_images(tmp_path, 2)
    llm = _VisionLLM(vision=False)
    verifier = Verifier(workspace=tmp_path, llm_client=llm, vision_mode="warn")
    assert verifier.vision_check_images(paths) == {}
    assert llm.calls == []
    verdict = verifier.vision_check_image(paths[0])
    assert verdict.get("skipped") is True


def test_vision_check_returns_empty_on_bad_reply(tmp_path):
    """模型回复不是预期 JSON 时不能抛异常（退回逐张路径）。"""
    from gis_cli.engine.verifier import Verifier

    paths = _make_images(tmp_path, 2)

    class _BadLLM(_VisionLLM):
        def chat(self, messages, **kwargs):
            self.calls.append(messages)

            class _R:
                model = "vision-stub"
                content = "这不是 JSON"

            return _R()

    verifier = Verifier(workspace=tmp_path, llm_client=_BadLLM(), vision_mode="warn")
    assert verifier.vision_check_images(paths) == {}

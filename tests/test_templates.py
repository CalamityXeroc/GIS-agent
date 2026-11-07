# -*- coding: utf-8 -*-
"""民生模板库骨架：manifest schema / 加载器 / 参数绑定 / 逐步结果 / 缺后端明确报错。

执行侧全部用**假配方 + 假执行器**，不依赖真 arcpy、也不依赖真 GDAL：
``output_xxx = '路径'`` 这一行就是假执行器认的产物（它会把文件落盘），
这样模板断言（``file_exists``）真的在读文件，而不是被 mock 掉。

后端门禁的用例照 ``tests/test_recipe_backend_routing.py`` 的写法：
先断言 ``not runner.codes`` —— 缺后端时**一行配方代码都没执行**。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gis_cli.engine.state import Observation
from gis_cli.engine.tools import EngineContext, build_default_registry
from gis_cli.recipes.schema import Recipe, RecipeParam
from gis_cli.recipes.store import RecipeLibrary
from gis_cli.templates.schema import (
    Template,
    TemplateBindingError,
    TemplateManifestError,
    parse_binding,
)
from gis_cli.templates.store import TemplateLibrary, _out_path_problem

ROOT = Path(__file__).resolve().parents[1]

#: 假执行器从渲染后的代码里抠产物路径（``output_path = '...'`` / ``output_table = '...'``）。
_ASSIGN = re.compile(r"^\s*(output\w*)\s*=\s*(.+?)\s*$", re.MULTILINE)


def _output_from_code(code: str) -> str | None:
    found = None
    for _name, literal in _ASSIGN.findall(code):
        try:
            value = ast.literal_eval(literal)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, str):
            found = value
    return found


class _Runner:
    """假执行器：声明自己能跑哪些后端；run() 把产物路径写到磁盘。"""

    def __init__(
        self,
        backends: set[str] | None = None,
        *,
        fail_on: int | None = None,
        extra_result: dict | None = None,
    ):
        if backends is not None:
            self.backends = set(backends)
        self.codes: list[str] = []
        self.fail_on = fail_on
        #: 额外结果键（如 project_to_crs 的 feature_count）：真实配方从代码里 set_result，
        #: 假执行器不跑代码，只能由用例喂进来 —— 断言绑 $steps.x.feature_count 时要用。
        self.extra_result = dict(extra_result or {})

    def run(self, code, *, timeout=None, workspace=None):
        self.codes.append(code)
        out = _output_from_code(code)
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text("x", encoding="utf-8")
        if self.fail_on is not None and len(self.codes) == self.fail_on:
            return SimpleNamespace(
                ok=False,
                stdout="",
                stderr="boom",
                result=None,
                error={"type": "Boom", "message": "第二步炸了"},
                display_images=[],
            )
        return SimpleNamespace(
            ok=True,
            stdout="",
            stderr="",
            result={**({"output": out} if out else {}), **self.extra_result},
            error=None,
            display_images=[],
        )


def _recipe(
    rid: str,
    *,
    params: tuple[str, ...],
    output_param: str,
    backend: tuple[str, ...] = ("arcpy",),
) -> Recipe:
    """假配方：每个参数都渲染成一行赋值（好验绑定），代码永不真执行。"""
    lines = "\n".join(f"{name} = {{{{{name}}}}}" for name in params)
    code = f"{lines}\nset_result({{'output': {output_param}}})\n"
    return Recipe(
        id=rid,
        name=rid,
        backend=list(backend),
        params={name: RecipeParam(name, "string", name == output_param) for name in params},
        code_template=code,
        # 双后端都给：本文件验的是模板层，不重复验“配方有没有 GDAL 版”
        code_template_gdal=code,
    )


def _project_recipe(backend: tuple[str, ...] = ("arcpy",)) -> Recipe:
    return _recipe(
        "project_to_crs",
        params=("input_path", "output_path", "target_srid"),
        output_param="output_path",
        backend=backend,
    )


def _zonal_recipe(backend: tuple[str, ...] = ("arcpy",)) -> Recipe:
    return _recipe(
        "zonal_statistics",
        params=("zone_path", "raster_path", "output_table", "statistics"),
        output_param="output_table",
        backend=backend,
    )


def _recipes(*recipes: Recipe, code_runner=None) -> RecipeLibrary:
    empty = Path("__no_such_dir__")
    library = RecipeLibrary(builtin_dir=empty, user_dir=empty, code_runner=code_runner)
    library.load()
    for recipe in recipes:
        library._recipes[recipe.id] = recipe  # 直接放进内存库（不走磁盘）
    return library


def _demo_manifest(
    *,
    backend: tuple[str, ...] = ("arcpy",),
    zone_ref: str = "$steps.project.output",
    assertions: list[dict] | None = None,
) -> dict:
    return {
        "id": "demo",
        "name": "演示模板",
        "description": "两步玩具链：投影 → 分区统计",
        "backend": list(backend),
        "data": {
            "zones": {"type": "vector", "required": True, "purpose": "分区面"},
            "raster": {"type": "raster", "required": True, "purpose": "数值栅格"},
            "target_srid": {"type": "integer", "required": False, "default": 4527},
        },
        "pipeline": [
            {
                "id": "project",
                "recipe_id": "project_to_crs",
                "params": {
                    "input_path": "$data.zones",
                    "output_path": "$out/zones.shp",
                    "target_srid": "$data.target_srid",
                },
            },
            {
                "id": "zonal",
                "recipe_id": "zonal_statistics",
                "params": {
                    "zone_path": zone_ref,
                    "raster_path": "$data.raster",
                    "output_table": "$out/zonal.dbf",
                    "statistics": "SUM",
                },
            },
        ],
        "assertions": (
            assertions
            if assertions is not None
            else [
                {"type": "file_exists", "path": "$out/zones.shp"},
                {"type": "file_exists", "path": "$steps.zonal.output"},
            ]
        ),
    }


def _library(tmp_path: Path, manifest: dict, recipes: RecipeLibrary) -> TemplateLibrary:
    folder = tmp_path / "templates" / str(manifest["id"])
    folder.mkdir(parents=True)
    (folder / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    library = TemplateLibrary(templates_dir=tmp_path / "templates", recipes=recipes)
    library.load()
    return library


def _ctx(tmp_path: Path, recipes: RecipeLibrary, runner) -> SimpleNamespace:
    return SimpleNamespace(
        workspace=tmp_path, recipes=recipes, code_runner=runner, templates=None, exec_timeout=10
    )


def _demo_data(tmp_path: Path) -> dict:
    return {"zones": str(tmp_path / "input" / "zones.shp"), "raster": str(tmp_path / "input" / "pop.tif")}


# ------------------------------------------------------------------ schema
def test_parse_binding_accepts_the_three_documented_forms():
    assert parse_binding("$data.zones").name == "zones"
    assert parse_binding("$data.zones").kind == "data"

    step = parse_binding("$steps.project.output")
    assert (step.kind, step.step, step.key) == ("step", "project", "output")

    out = parse_binding("$out/maps/a.png")
    assert (out.kind, out.path) == ("out", "maps/a.png")
    assert parse_binding("$out").path == ""

    assert parse_binding("input/zones.shp") is None
    assert parse_binding(4527) is None

    # 以 $ 开头但认不出：必须报错，不能当字面量悄悄传下去
    with pytest.raises(TemplateBindingError) as excinfo:
        parse_binding("$dat.zones")
    assert "$data.<输入名>" in str(excinfo.value)


def test_manifest_rejects_bad_shapes_with_readable_errors():
    base = _demo_manifest()

    with pytest.raises(TemplateManifestError, match="模板 id"):
        Template.from_dict({**base, "id": "Demo Template"})
    with pytest.raises(TemplateManifestError, match="pipeline"):
        Template.from_dict({**base, "pipeline": []})
    with pytest.raises(TemplateManifestError, match="步骤 id 重复"):
        Template.from_dict(
            {
                **base,
                "pipeline": base["pipeline"] + [base["pipeline"][0]],
            }
        )
    with pytest.raises(TemplateManifestError, match="类型"):
        Template.from_dict(
            {**base, "data": {**base["data"], "zones": {"type": "shapefile"}}}
        )
    with pytest.raises(TemplateManifestError, match="recipe_id"):
        Template.from_dict({**base, "pipeline": [{"id": "only"}]})


def test_manifest_to_dict_lists_inputs_and_backends():
    template = Template.from_dict(_demo_manifest(backend=("arcpy", "gdal")))
    info = template.to_dict()
    assert info["backend"] == ["arcpy", "gdal"]
    assert info["data"]["zones"] == {
        "purpose": "分区面",
        "type": "vector",
        "required": True,
        "default": None,
    }
    assert [step["recipe_id"] for step in info["pipeline"]] == [
        "project_to_crs",
        "zonal_statistics",
    ]


def test_shipped_manifests_point_at_real_recipes_and_params():
    """仓库里的玩具模板必须和真实配方对得上（不然它们只是几篇 YAML）。"""
    shipped = TemplateLibrary().all()
    assert shipped, "仓库里一个模板都没加载到（templates/ 目录不对？）"
    library = RecipeLibrary()
    for demo in shipped:
        for step in demo.pipeline:
            recipe = library.get(step.recipe_id)
            assert recipe is not None, f"{demo.id} 引用了不存在的配方: {step.recipe_id}"
            assert set(recipe.backend) & set(demo.backend), f"{step.recipe_id} 与模板后端无交集"
            for name in step.params:
                assert name in recipe.params, f"{demo.id} 给了 {step.recipe_id} 不认识的参数: {name}"
            required = {
                name
                for name, spec in recipe.params.items()
                if spec.required and spec.default is None
            }
            assert required <= set(step.params), f"{step.recipe_id} 的必填参数没绑: {required}"


def test_shipped_dual_backend_demo_runs_on_a_gdal_only_runner(tmp_path: Path):
    """双后端玩具模板（仓库里的 manifest）在 GDAL-only 执行器下真的能跑通整条链。"""
    recipes = _recipes(
        _project_recipe(backend=("arcpy", "gdal")),
        _recipe(
            "buffer_dissolve",
            params=("input_path", "output_path", "distance", "dissolve_type"),
            output_param="output_path",
            backend=("arcpy", "gdal"),
        ),
    )
    library = TemplateLibrary(recipes=recipes)  # 用仓库里真实的 manifest，不用夹具
    runner = _Runner({"gdal"})
    data = {"points": str(tmp_path / "input" / "points.shp")}

    observation = library.run("demo_project_then_buffer", data, _ctx(tmp_path, recipes, runner))

    assert observation.ok, observation.summary
    assert observation.data["backend"] == "gdal"
    first, second = observation.data["steps"]
    assert first["params"]["target_srid"] == 32610  # manifest 里的默认值
    assert second["recipe_id"] == "buffer_dissolve"
    assert second["params"]["input_path"] == first["output"], "上一步产物没绑给下一步"
    assert second["params"]["distance"] == "1000 Meters"
    assert all(item["ok"] for item in observation.data["assertions"])


# ------------------------------------------------------------ list_templates
def test_list_templates_shows_inputs_backends_and_runnability(tmp_path: Path):
    recipes = _recipes(_project_recipe(), _zonal_recipe())
    library = _library(tmp_path, _demo_manifest(backend=("arcpy",)), recipes)
    registry = build_default_registry()
    ctx = EngineContext(
        workspace=tmp_path, recipes=recipes, code_runner=_Runner({"gdal"}), templates=library
    )

    observation = registry.execute("list_templates", {"query": "demo"}, ctx)

    assert observation.ok, observation.summary
    template = observation.data["templates"][0]
    assert template["id"] == "demo"
    assert template["backend"] == ["arcpy"]
    assert set(template["data"]) == {"zones", "raster", "target_srid"}
    assert template["data"]["zones"]["type"] == "vector"
    assert template["data"]["zones"]["purpose"] == "分区面"
    assert template["data"]["target_srid"]["required"] is False
    # 纯 GDAL 机器上这个 arcpy 模板跑不了：列表里要能看出来
    assert template["runnable"] is False

    # 执行器没声明能力时不许猜
    untyped = registry.execute(
        "list_templates",
        {},
        EngineContext(workspace=tmp_path, code_runner=_Runner(), templates=library),
    )
    assert untyped.data["templates"][0]["runnable"] is None


# ------------------------------------- 最小可用路径：链跑通 + 绑定真的生效
def test_run_template_walks_the_chain_and_binds_upstream_output(tmp_path: Path):
    recipes = _recipes(_project_recipe(), _zonal_recipe())
    library = _library(tmp_path, _demo_manifest(), recipes)
    runner = _Runner({"arcpy"})
    ctx = _ctx(tmp_path, recipes, runner)

    observation = library.run("demo", _demo_data(tmp_path), ctx)

    assert observation.ok, observation.summary
    assert observation.data["backend"] == "arcpy"
    steps = observation.data["steps"]
    assert [step["step"] for step in steps] == ["project", "zonal"]
    assert [step["recipe_id"] for step in steps] == ["project_to_crs", "zonal_statistics"]

    first, second = steps
    assert first["ok"] is True and second["ok"] is True
    assert first["params"]["input_path"] == _demo_data(tmp_path)["zones"]
    # 输入默认值也走绑定：target_srid 没传 → manifest 默认 4527
    assert first["params"]["target_srid"] == 4527
    assert first["output"] == str(tmp_path / "output" / "demo" / "zones.shp")
    # 上游产物被下游正确引用
    assert second["params"]["zone_path"] == first["output"]
    assert second["output"] == str(tmp_path / "output" / "demo" / "zonal.dbf")
    # 绑定真的进了渲染后的代码（不是只写在记录里）
    assert repr(first["output"]) in runner.codes[1]

    # 模板断言真的执行了：两条 file_exists 都过（假执行器把文件写出来了）
    assertions = observation.data["assertions"]
    assert len(assertions) == 2
    assert all(item["ok"] for item in assertions)
    assert second["output"] in observation.artifacts


def test_run_template_tool_is_the_same_path(tmp_path: Path):
    """工具面（run_template）与库 API 同一条路：模型看到的就是逐步结果。"""
    recipes = _recipes(_project_recipe(), _zonal_recipe())
    library = _library(tmp_path, _demo_manifest(), recipes)
    ctx = EngineContext(
        workspace=tmp_path, recipes=recipes, code_runner=_Runner({"arcpy"}), templates=library
    )

    observation = build_default_registry().execute(
        "run_template", {"template_id": "demo", "data": _demo_data(tmp_path)}, ctx
    )

    assert observation.ok, observation.summary
    assert len(observation.data["steps"]) == 2
    assert observation.data["output_dir"] == str(tmp_path / "output" / "demo")


def test_template_level_assertion_failure_turns_the_run_red(tmp_path: Path):
    recipes = _recipes(_project_recipe(), _zonal_recipe())
    manifest = _demo_manifest(
        assertions=[{"type": "file_exists", "path": "$out/missing_report.md"}]
    )
    library = _library(tmp_path, manifest, recipes)
    ctx = _ctx(tmp_path, recipes, _Runner({"arcpy"}))

    observation = library.run("demo", _demo_data(tmp_path), ctx)

    assert not observation.ok
    assert "断言 0/1 通过" in observation.summary
    assert observation.data["assertions"][0]["ok"] is False
    # 配方链本身是跑完的：断言失败不等于步骤失败
    assert [step["ok"] for step in observation.data["steps"]] == [True, True]


# ------------------------------------------------------------ 绑定的反例
def test_unknown_upstream_step_reports_human_error_and_runs_nothing(tmp_path: Path):
    recipes = _recipes(_project_recipe(), _zonal_recipe())
    library = _library(tmp_path, _demo_manifest(zone_ref="$steps.projct.output"), recipes)
    runner = _Runner({"arcpy"})
    ctx = _ctx(tmp_path, recipes, runner)

    observation = library.run("demo", _demo_data(tmp_path), ctx)

    assert not observation.ok
    assert observation.error["type"] == "TemplateBindingError"
    assert "projct" in observation.summary
    assert "可用: project" in observation.summary
    assert not runner.codes, "绑定写错时必须一步都不跑"


def test_forward_reference_to_a_later_step_is_rejected(tmp_path: Path):
    recipes = _recipes(_project_recipe(), _zonal_recipe())
    # 第 1 步引用第 2 步的产物：顺序反了
    manifest = _demo_manifest()
    manifest["pipeline"][0]["params"]["input_path"] = "$steps.zonal.output"
    library = _library(tmp_path, manifest, recipes)
    runner = _Runner({"arcpy"})

    observation = library.run("demo", _demo_data(tmp_path), _ctx(tmp_path, recipes, runner))

    assert not observation.ok
    assert observation.error["type"] == "TemplateBindingError"
    assert "没有名为 'zonal' 的步骤" in observation.summary
    assert "只能引用" in observation.summary
    assert not runner.codes


def test_misspelled_binding_namespace_is_rejected_before_execution(tmp_path: Path):
    recipes = _recipes(_project_recipe(), _zonal_recipe())
    manifest = _demo_manifest()
    manifest["pipeline"][0]["params"]["input_path"] = "$dat.zones"
    library = _library(tmp_path, manifest, recipes)
    runner = _Runner({"arcpy"})

    observation = library.run("demo", _demo_data(tmp_path), _ctx(tmp_path, recipes, runner))

    assert not observation.ok
    assert observation.error["type"] == "TemplateBindingError"
    assert "$data.<输入名>" in observation.summary
    assert not runner.codes


def test_missing_required_input_names_what_to_pass(tmp_path: Path):
    recipes = _recipes(_project_recipe(), _zonal_recipe())
    library = _library(tmp_path, _demo_manifest(), recipes)
    runner = _Runner({"arcpy"})
    data = {"raster": _demo_data(tmp_path)["raster"]}

    observation = library.run("demo", data, _ctx(tmp_path, recipes, runner))

    assert not observation.ok
    assert observation.error["type"] == "TemplateDataError"
    assert "zones" in observation.summary and "必需输入" in observation.summary
    assert not runner.codes


def test_unknown_input_name_is_rejected(tmp_path: Path):
    recipes = _recipes(_project_recipe(), _zonal_recipe())
    library = _library(tmp_path, _demo_manifest(), recipes)
    runner = _Runner({"arcpy"})
    data = {**_demo_data(tmp_path), "zone": "input/typo.shp"}

    observation = library.run("demo", data, _ctx(tmp_path, recipes, runner))

    assert not observation.ok
    assert "zone" in observation.summary and "可用" in observation.summary
    assert not runner.codes


def test_untouched_recipe_param_typo_is_rejected(tmp_path: Path):
    """manifest 里把 statistics 写成 statistic：必须报错，不能静默用配方默认值。"""
    recipes = _recipes(_project_recipe(), _zonal_recipe())
    manifest = _demo_manifest()
    manifest["pipeline"][1]["params"]["statistic"] = "SUM"
    library = _library(tmp_path, manifest, recipes)
    runner = _Runner({"arcpy"})

    observation = library.run("demo", _demo_data(tmp_path), _ctx(tmp_path, recipes, runner))

    assert not observation.ok
    assert "statistic" in observation.summary and "不认识的参数" in observation.summary
    assert not runner.codes


# ------------------------------------------- $out 绑定不许跳出输出目录
@pytest.mark.parametrize(
    "binding",
    [
        "$out/../x.shp",
        "$out/a/../../x.shp",
        # Windows 上 /etc/x.shp 没有盘符但锚在盘根：out_dir / '/etc/x.shp' 会丢掉 out_dir
        "$out//etc/x.shp",
        "$out/\\etc\\x.shp",
        # 盘符相对路径（C:x.shp）同样会落到别处
        "$out/C:x.shp",
        # 反斜杠写法连绑定都认不出，在解析层就报错
        "$out\\etc\\x.shp",
    ],
)
def test_out_binding_that_escapes_the_output_dir_is_rejected_before_any_code_runs(
    tmp_path: Path, binding: str
):
    recipes = _recipes(_project_recipe(), _zonal_recipe())
    manifest = _demo_manifest()
    manifest["pipeline"][0]["params"]["output_path"] = binding
    library = _library(tmp_path, manifest, recipes)
    runner = _Runner({"arcpy"})

    observation = library.run("demo", _demo_data(tmp_path), _ctx(tmp_path, recipes, runner))

    assert not observation.ok
    assert observation.error["type"] == "TemplateBindingError"
    assert not runner.codes, "越界的 $out 绑定必须一行配方代码都不跑"


def test_out_binding_inside_the_output_dir_is_allowed(tmp_path: Path):
    """放行侧：``$out/a/b.shp`` 落在输出目录下；``$out``（空相对路径）就是输出目录本身。"""
    assert parse_binding("$out").path == ""
    assert _out_path_problem("") is None

    recipes = _recipes(_project_recipe(), _zonal_recipe())
    manifest = _demo_manifest(assertions=[])
    manifest["pipeline"][0]["params"]["output_path"] = "$out/a/b.shp"
    library = _library(tmp_path, manifest, recipes)
    runner = _Runner({"arcpy"})

    observation = library.run("demo", _demo_data(tmp_path), _ctx(tmp_path, recipes, runner))

    assert observation.ok, observation.summary
    out_dir = tmp_path / "output" / "demo"
    assert observation.data["steps"][0]["params"]["output_path"] == str(out_dir / "a" / "b.shp")
    assert (out_dir / "a" / "b.shp").is_file()
    assert runner.codes


# --------------------------------------------------- 缺后端：明确报错 + 零执行
def test_arcpy_only_template_on_gdal_runner_says_needs_arcgis_and_runs_nothing(tmp_path: Path):
    recipes = _recipes(_project_recipe(), _zonal_recipe())
    library = _library(tmp_path, _demo_manifest(backend=("arcpy",)), recipes)
    runner = _Runner({"gdal"})
    ctx = _ctx(tmp_path, recipes, runner)

    observation = library.run("demo", _demo_data(tmp_path), ctx)

    assert not observation.ok
    assert observation.error["type"] == "BackendUnavailable"
    assert "需 ArcGIS" in observation.summary
    assert observation.error["required"] == ["arcpy"]
    assert observation.error["available"] == ["gdal"]
    assert not runner.codes, "缺后端时一行配方代码都不该执行"


def test_later_step_needing_arcpy_is_caught_before_the_first_step_runs(tmp_path: Path):
    """模板声明 [arcpy, gdal]、第一步能跑，但第二步配方只有 arcpy —— 不许跑到一半才炸。"""
    recipes = _recipes(
        _project_recipe(backend=("gdal",)), _zonal_recipe(backend=("arcpy",))
    )
    library = _library(tmp_path, _demo_manifest(backend=("arcpy", "gdal")), recipes)
    runner = _Runner({"gdal"})

    observation = library.run("demo", _demo_data(tmp_path), _ctx(tmp_path, recipes, runner))

    assert not observation.ok
    assert observation.error["type"] == "BackendUnavailable"
    assert "需 ArcGIS" in observation.summary
    assert "第 2 步" in observation.summary
    assert not runner.codes


def test_undeclared_runner_is_not_gated(tmp_path: Path):
    """执行器没声明能力 → 不设门禁（与 P0-1a 的历史行为一致）。"""
    recipes = _recipes(_project_recipe(), _zonal_recipe())
    library = _library(tmp_path, _demo_manifest(backend=("arcpy",)), recipes)
    runner = _Runner()  # 没有 backends 声明

    observation = library.run("demo", _demo_data(tmp_path), _ctx(tmp_path, recipes, runner))

    assert observation.ok, observation.summary
    assert observation.data["backend"] == "arcpy"
    assert runner.codes


# ------------------------------------------------------- 某步失败就停链、留记录
def test_step_failure_stops_the_chain_and_keeps_step_records(tmp_path: Path):
    recipes = _recipes(_project_recipe(), _zonal_recipe())
    library = _library(tmp_path, _demo_manifest(), recipes)
    runner = _Runner({"arcpy"}, fail_on=2)

    observation = library.run("demo", _demo_data(tmp_path), _ctx(tmp_path, recipes, runner))

    assert not observation.ok
    assert "第 2 步" in observation.summary
    steps = observation.data["steps"]
    assert [step["ok"] for step in steps] == [True, False]
    assert steps[1]["error"]["type"] == "Boom"
    # 第 1 步的产物仍留在记录里，便于定位
    assert steps[0]["output"]


# ============================================ 首批真实业务模板（W4 第 2 步）
# 下面两个 manifest 是**仓库里的真文件**（templates/<id>/manifest.yaml），用例只换配方与
# 执行器：假配方的 id 与参数名照真实配方声明（``test_shipped_manifests_point_at_real_recipes_and_params``
# 也会拿真配方库再对一遍），所以链断了、绑定错位、断言写歪都会在这里红。
# 数值断言（``field_stats_between``）不是被 mock 掉的：它向执行器要 sum/min/max/mean，
# ``_StatsRunner`` 按 ``columns`` 真算一遍回传，因此「断言能过」与「断言能红」都算得出来。
_STAT_PATH = re.compile(r"^_p = (.+)$", re.MULTILINE)
_STAT_FIELD = re.compile(r"^_f = (.+)$", re.MULTILINE)


class _StatsRunner(_Runner):
    """假执行器 + 数值断言的取数端（``columns`` = {产物路径: {字段: [值...]}}）。"""

    def __init__(
        self,
        backends: set[str] | None = None,
        *,
        columns=None,
        fail_on: int | None = None,
        extra_result: dict | None = None,
    ):
        super().__init__(backends, fail_on=fail_on, extra_result=extra_result)
        self.columns = {str(path): dict(fields) for path, fields in (columns or {}).items()}

    def run(self, code, *, timeout=None, workspace=None):
        if "SearchCursor" not in code:  # 普通配方代码：照旧把产物写到磁盘
            return super().run(code, timeout=timeout, workspace=workspace)
        path = ast.literal_eval(_STAT_PATH.search(code).group(1))
        field = ast.literal_eval(_STAT_FIELD.search(code).group(1))
        # 与真实 ``_field_stat`` 一致：数值统计只吃能转成 float 的值，
        # distinct 数的是**原值**（文本字段如 CATEGORY 也能数出 distinct）。
        raw = list(self.columns.get(str(path), {}).get(str(field), []))
        values = []
        for item in raw:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                pass
        result = {"sum": None, "min": None, "max": None, "mean": None,
                  "count": len(raw), "null": 0, "distinct": len({str(v) for v in raw}),
                  "null_rate": 0.0}
        if values:
            result.update(sum=sum(values), min=min(values), max=max(values),
                          mean=sum(values) / len(values))
        return SimpleNamespace(
            ok=True, stdout="", stderr="", result=result, error=None, display_images=[]
        )


def _chain_recipe(
    rid: str,
    *,
    params: tuple[str, ...],
    required: tuple[str, ...],
    output_expr: str,
    emit: str | None = None,
) -> Recipe:
    """假配方：参数逐行赋值（好验绑定），产物路径与额外结果按给定代码表达式回报。

    ``emit`` 是给假执行器抠路径用的**字面量**赋值行：``field_calculate`` 这类
    「原地改」的配方没有 output_path 参数，得补一行 ``output_path = {{input_path}}``。
    """
    lines = [f"{name} = {{{{{name}}}}}" for name in params]
    if emit:
        lines.append(emit)
    code = "\n".join(lines) + f"\nset_result({{'output': {output_expr}}})\n"
    return Recipe(
        id=rid,
        name=rid,
        backend=["arcpy"],
        params={name: RecipeParam(name, "string", name in required) for name in params},
        code_template=code,
        code_template_gdal=code,
    )


def _business_recipes(*, code_runner=None) -> RecipeLibrary:
    """两个业务模板用到的 7 个配方（参数名照真实配方，必填项照真实声明）。"""
    return _recipes(
        _chain_recipe(
            "project_to_crs",
            params=("input_path", "output_path", "target_srid"),
            required=("input_path", "output_path"),
            output_expr="output_path",
        ),
        _chain_recipe(
            "buffer_dissolve",
            params=("input_path", "output_path", "distance", "dissolve_type"),
            required=("input_path", "output_path", "distance"),
            output_expr="output_path",
        ),
        _chain_recipe(
            "clip_layer",
            params=("input_path", "clip_path", "output_path"),
            required=("input_path", "clip_path", "output_path"),
            output_expr="output_path",
        ),
        _chain_recipe(
            "field_calculate",
            params=("input_path", "field_name", "expression", "field_type"),
            required=("input_path", "field_name", "expression"),
            output_expr="output_path",
            emit="output_path = {{input_path}}",
        ),
        _chain_recipe(
            "aggregate_to_zones",
            params=("zones_path", "source_path", "output_path", "stat_fields", "stat_type"),
            required=("zones_path", "source_path", "output_path", "stat_fields"),
            output_expr="output_path",
        ),
        _chain_recipe(
            "zonal_statistics",
            params=("zone_path", "raster_path", "output_table", "zone_field", "statistics"),
            required=("zone_path", "raster_path", "output_table"),
            output_expr="output_table",
        ),
        _chain_recipe(
            "summarize_stats",
            params=("in_table", "out_table", "stat_fields", "case_field"),
            required=("in_table", "out_table", "stat_fields", "case_field"),
            output_expr="out_table",
            # 真实配方的产物参数叫 out_table（不是 output_path）：假执行器只认 output* 变量，
            # 补一行字面量赋值才能抠到产物路径
            emit="output_table = {{out_table}}",
        ),
        _chain_recipe(
            "attribute_join_table",
            params=(
                "target_path",
                "target_key",
                "join_table",
                "join_key",
                "join_fields",
                "output_path",
            ),
            required=("target_path", "target_key", "join_table", "join_key", "output_path"),
            output_expr="output_path",
        ),
        code_runner=code_runner,
    )


def _elderly_data(tmp_path: Path) -> dict:
    return {
        "communities": str(tmp_path / "input" / "communities.shp"),
        "facilities": str(tmp_path / "input" / "facilities.shp"),
        "elderly_raster": str(tmp_path / "input" / "elderly.tif"),
    }


def _green_data(tmp_path: Path) -> dict:
    return {
        "communities": str(tmp_path / "input" / "communities.shp"),
        "green_spaces": str(tmp_path / "input" / "green.shp"),
        "population_raster": str(tmp_path / "input" / "population.tif"),
    }


def _run_shipped(tmp_path: Path, template_id: str, data: dict, *, columns=None, feature_count: int = 3):
    """跑仓库里真实的 manifest（配方与执行器是假的）。返回 (observation, runner)。"""
    runner = _StatsRunner(
        {"arcpy"}, columns=columns, extra_result={"feature_count": feature_count}
    )
    recipes = _business_recipes(code_runner=runner)
    library = TemplateLibrary(recipes=recipes)
    observation = library.run(template_id, data, _ctx(tmp_path, recipes, runner))
    return observation, runner


def _assert_steps_landed_in_output_dir(observation, out_dir: Path) -> None:
    for step in observation.data["steps"]:
        assert step["ok"], step
        path = Path(step["output"])
        assert path.parent == out_dir, f"{step['step']} 的产物没落在 {out_dir}：{path}"
        assert path.is_file(), f"{step['step']} 的产物没落盘：{path}"


def test_elderly_care_template_runs_the_whole_chain(tmp_path: Path):
    out = tmp_path / "output" / "elderly_care_coverage"
    data = _elderly_data(tmp_path)
    observation, runner = _run_shipped(
        tmp_path,
        "elderly_care_coverage",
        data,
        columns={
            str(out / "community_metrics.shp"): {"COVER_RATE": [0.0, 0.35, 0.8]},
            str(out / "elderly_by_community.dbf"): {"SUM": [120.0, 80.0, 300.0]},
            # 守恒断言：归属片段数合计 = covered_parts 的要素数（下面 feature_count=3）
            str(out / "coverage_by_community.shp"): {"COUNT_COVA": [1.0, 1.0, 1.0]},
        },
    )

    assert observation.ok, observation.summary
    steps = observation.data["steps"]
    assert [step["step"] for step in steps] == [
        "project_communities",
        "project_facilities",
        "service_area",
        "covered_parts",
        "covered_area",
        "coverage_stats",
        "coverage_by_community",
        "elderly_by_community",
        "community_metrics",
        "coverage_rate",
    ]
    _assert_steps_landed_in_output_dir(observation, out)

    by_id = {step["step"]: step for step in steps}
    # $data 绑定 + manifest 里的默认值
    assert by_id["project_communities"]["params"]["input_path"] == data["communities"]
    assert by_id["project_facilities"]["params"]["input_path"] == data["facilities"]
    assert by_id["project_communities"]["params"]["target_srid"] == 4527
    assert by_id["service_area"]["params"]["distance"] == "800 Meters"
    assert by_id["elderly_by_community"]["params"]["zone_field"] == "COMM_ID"
    # $steps 绑定：下游拿到的必须是上游**产物路径**，不是写死的名字
    assert by_id["service_area"]["params"]["input_path"] == by_id["project_facilities"]["output"]
    assert by_id["covered_parts"]["params"]["clip_path"] == by_id["service_area"]["output"]
    assert by_id["covered_parts"]["params"]["input_path"] == by_id["project_communities"]["output"]
    # DAIR-171：覆盖片段按自带 COMM_ID 属性聚合，不做第二次空间归属
    assert by_id["coverage_stats"]["recipe_id"] == "summarize_stats"
    assert by_id["coverage_stats"]["params"]["in_table"] == by_id["covered_area"]["output"]
    assert by_id["coverage_stats"]["params"]["case_field"] == "COMM_ID"
    assert by_id["coverage_by_community"]["recipe_id"] == "attribute_join_table"
    assert by_id["coverage_by_community"]["params"]["target_path"] == by_id["project_communities"]["output"]
    assert by_id["coverage_by_community"]["params"]["join_table"] == by_id["coverage_stats"]["output"]
    assert by_id["elderly_by_community"]["params"]["zone_path"] == by_id["project_communities"]["output"]
    assert by_id["community_metrics"]["params"]["join_table"] == by_id["elderly_by_community"]["output"]
    assert by_id["coverage_rate"]["params"]["input_path"] == by_id["community_metrics"]["output"]
    # 原地改的步骤（field_calculate）输出就是输入
    assert by_id["covered_area"]["output"] == by_id["covered_parts"]["output"]
    # 模板级断言真的跑了：6 条全绿
    assert len(observation.data["assertions"]) == 6
    assert [item["ok"] for item in observation.data["assertions"]] == [True] * 6
    assert runner.codes, "链一步都没执行"


def test_green_space_equity_template_runs_the_whole_chain(tmp_path: Path):
    out = tmp_path / "output" / "green_space_equity"
    data = _green_data(tmp_path)
    observation, _runner = _run_shipped(
        tmp_path,
        "green_space_equity",
        data,
        feature_count=7,  # 输入 7 块绿地：守恒断言的上界绑的就是这个数
        columns={
            str(out / "green_by_community.shp"): {"SRC_COUNT": [3.0, 2.0, 1.0]},
            str(out / "population_by_community.dbf"): {"SUM": [900.0, 1200.0, 400.0]},
            str(out / "green_equity.shp"): {
                "GPC": [12.0, 4.5, 30.0],
                "GREEN_GAP": [0.0, 1.0, 0.0],
            },
        },
    )

    assert observation.ok, observation.summary
    steps = observation.data["steps"]
    assert [step["step"] for step in steps] == [
        "project_communities",
        "project_green",
        "green_area",
        "green_by_community",
        "population_by_community",
        "green_equity",
        "green_per_capita",
        "green_gap",
    ]
    _assert_steps_landed_in_output_dir(observation, out)

    by_id = {step["step"]: step for step in steps}
    assert by_id["project_green"]["params"]["input_path"] == data["green_spaces"]
    assert by_id["green_area"]["params"]["input_path"] == by_id["project_green"]["output"]
    assert by_id["green_by_community"]["params"]["zones_path"] == by_id["project_communities"]["output"]
    assert by_id["green_by_community"]["params"]["source_path"] == by_id["green_area"]["output"]
    assert by_id["green_equity"]["params"]["join_table"] == by_id["population_by_community"]["output"]
    assert by_id["green_per_capita"]["params"]["input_path"] == by_id["green_equity"]["output"]
    assert by_id["green_gap"]["params"]["input_path"] == by_id["green_per_capita"]["output"]
    # 缺口标记是「有人口且人均绿地 < 9」：口径写进 manifest，不能被静默改掉
    assert "GPC! < 9.0" in by_id["green_gap"]["params"]["expression"]

    assert len(observation.data["assertions"]) == 6
    assert [item["ok"] for item in observation.data["assertions"]] == [True] * 6


def test_coverage_rate_assertion_really_fails_on_bad_data(tmp_path: Path):
    """覆盖率 1.4 > 1：覆盖面积不可能大于社区面积，这条区间断言必须红（不是恒真）。"""
    out = tmp_path / "output" / "elderly_care_coverage"
    observation, _runner = _run_shipped(
        tmp_path,
        "elderly_care_coverage",
        _elderly_data(tmp_path),
        columns={
            str(out / "community_metrics.shp"): {"COVER_RATE": [0.2, 1.4]},
            str(out / "elderly_by_community.dbf"): {"SUM": [10.0, 20.0]},
            # 守恒断言这条案例里是绿的（1+2 = 3 = covered_parts 要素数），红的只能是覆盖率
            str(out / "coverage_by_community.shp"): {"COUNT_COVA": [1.0, 2.0]},
        },
    )

    assert not observation.ok
    failed = [item for item in observation.data["assertions"] if not item["ok"]]
    assert [item["kind"] for item in failed] == ["field_stats_between"]
    assert "COVER_RATE" in failed[0]["detail"] and "1.4" in failed[0]["detail"]
    # 链本身是跑完的：断言失败不等于步骤失败
    assert [step["ok"] for step in observation.data["steps"]] == [True] * 10


def test_elderly_conservation_assertion_catches_lost_or_double_counted_parts(tmp_path: Path):
    """归属片段数合计 5 > covered_parts 的 3 个片段：重复计入（DAIR-171 那类错）必须红。"""
    out = tmp_path / "output" / "elderly_care_coverage"
    for parts, why in (([3.0, 2.0], "多算"), ([1.0, 1.0], "漏算")):
        observation, _runner = _run_shipped(
            tmp_path,
            "elderly_care_coverage",
            _elderly_data(tmp_path),
            columns={
                str(out / "community_metrics.shp"): {"COVER_RATE": [0.2, 0.5]},
                str(out / "elderly_by_community.dbf"): {"SUM": [10.0, 20.0]},
                str(out / "coverage_by_community.shp"): {"COUNT_COVA": parts},
            },
        )

        assert not observation.ok, why
        failed = [item for item in observation.data["assertions"] if not item["ok"]]
        assert [item["kind"] for item in failed] == ["field_stats_between"]
        assert "COUNT_COVA" in failed[0]["detail"], why


def test_green_conservation_assertion_catches_double_counting(tmp_path: Path):
    """归属斑块数合计 9 > 输入斑块总数 7：SpatialJoin 扇出/重复计数，守恒断言必须红。"""
    out = tmp_path / "output" / "green_space_equity"
    observation, _runner = _run_shipped(
        tmp_path,
        "green_space_equity",
        _green_data(tmp_path),
        feature_count=7,
        columns={
            str(out / "green_by_community.shp"): {"SRC_COUNT": [4.0, 3.0, 2.0]},
            str(out / "population_by_community.dbf"): {"SUM": [900.0, 1200.0, 400.0]},
            str(out / "green_equity.shp"): {
                "GPC": [12.0, 4.5, 30.0],
                "GREEN_GAP": [0.0, 1.0, 0.0],
            },
        },
    )

    assert not observation.ok
    failed = [item for item in observation.data["assertions"] if not item["ok"]]
    assert [item["kind"] for item in failed] == ["field_stats_between"]
    assert "SRC_COUNT" in failed[0]["detail"] and "9.0" in failed[0]["detail"]
    assert [step["ok"] for step in observation.data["steps"]] == [True] * 8


@pytest.mark.parametrize(
    "template_id, data_factory",
    [("elderly_care_coverage", _elderly_data), ("green_space_equity", _green_data)],
)
def test_business_templates_need_arcgis_and_run_nothing_on_a_gdal_runner(
    tmp_path: Path, template_id: str, data_factory
):
    """业务模板是 arcpy-only：GDAL-only 机器上必须预检就报「需 ArcGIS」，一步都不跑。"""
    recipes = _business_recipes()
    library = TemplateLibrary(recipes=recipes)
    runner = _Runner({"gdal"})

    observation = library.run(template_id, data_factory(tmp_path), _ctx(tmp_path, recipes, runner))

    assert not observation.ok
    assert observation.error["type"] == "BackendUnavailable"
    assert "需 ArcGIS" in observation.summary
    assert observation.error["required"] == ["arcpy"]
    assert not runner.codes, "缺后端时一行配方代码都不该执行"


def test_list_templates_shows_the_two_business_templates_with_readable_inputs(tmp_path: Path):
    recipes = _business_recipes()
    library = TemplateLibrary(recipes=recipes)  # 仓库里真实的 templates/ 目录
    registry = build_default_registry()
    ctx = EngineContext(
        workspace=tmp_path, recipes=recipes, code_runner=_Runner({"gdal"}), templates=library
    )

    observation = registry.execute("list_templates", {}, ctx)

    assert observation.ok, observation.summary
    templates = {item["id"]: item for item in observation.data["templates"]}
    # 首批两个业务模板仍按原口径断言；第 3 批的 4 个模板由下面的
    # ``test_batch3_list_templates_shows_inputs_backends_and_runnability`` 管。
    assert {
        "demo_project_then_buffer",
        "demo_project_then_zonal",
        "elderly_care_coverage",
        "green_space_equity",
    } <= set(templates)
    elderly = templates["elderly_care_coverage"]
    assert elderly["backend"] == ["arcpy"]
    assert elderly["runnable"] is False  # 纯 GDAL 机器上跑不了
    assert set(elderly["data"]) == {
        "communities",
        "elderly_raster",
        "facilities",
        "target_srid",
        "service_radius",
        "community_id_field",
    }
    assert elderly["data"]["elderly_raster"]["required"] is True
    assert elderly["data"]["service_radius"]["default"] == "800 Meters"
    assert len(elderly["pipeline"]) == 10

    green = templates["green_space_equity"]
    assert set(green["data"]) == {
        "communities",
        "green_spaces",
        "population_raster",
        "target_srid",
        "community_id_field",
    }
    assert green["data"]["target_srid"]["required"] is False
    assert len(green["pipeline"]) == 8

    # 数据清单与口径要人能读：每一项都有 purpose，描述里写了做什么
    for info in (elderly, green):
        for name, spec in info["data"].items():
            assert spec["purpose"], f"{info['id']}.{name} 没有 purpose"
        assert info["description"], f"{info['id']} 没有 description"


# ==================== .shp 字段名 10 字符上限（静态守卫，不依赖 arcpy、不落真 .shp）
# `.shp` 的属性表是 dBASE，字段名物理上限 10 字符：超了 arcpy 只会**静默截断**（或被拒），
# 而后续步骤与断言引用的都是 manifest 里那个**没截断**的长名字 → 链一定断在中间。
# 所以这条规则不能靠「跑一遍真 .shp 才发现」，得在 manifest 层静态挡住。
_SHP_FIELD_LIMIT = 10


def _step_output_binding(step: dict) -> str | None:
    """步骤的产物绑定**原样**字符串（不解析）。

    ``output_path`` / ``output_table`` 是配方的产物参数；``field_calculate`` 这类原地改的
    配方没有产物参数，产物就是它的 ``input_path``。
    """
    params = step.get("params") or {}
    for key in ("output_path", "output_table"):
        if isinstance(params.get(key), str):
            return params[key]
    input_path = params.get("input_path")
    return input_path if isinstance(input_path, str) else None


def _resolved_output(step: dict, steps: dict, seen: frozenset = frozenset()) -> str | None:
    """把步骤产物绑定解析成最终路径（``$steps.x.output`` 递归追到上游）。"""
    binding = _step_output_binding(step)
    if binding is None or not binding.startswith("$steps."):
        return binding
    upstream = binding[len("$steps.") :].split(".")[0]
    if upstream in seen or upstream not in steps:
        return None
    return _resolved_output(steps[upstream], steps, seen | {upstream})


def _long_shp_field_names(manifest: dict) -> list[str]:
    """manifest 会写到 ``*.shp`` 产物上的 >10 字符字段名（空 list = 合规）。"""
    steps = {step["id"]: step for step in manifest.get("pipeline") or [] if "id" in step}
    problems = []
    for step_id, step in steps.items():
        path = _resolved_output(step, steps)
        if not path or not path.lower().endswith(".shp"):
            continue
        params = step.get("params") or {}
        names: list[str] = []
        if step.get("recipe_id") == "field_calculate":
            names.append(str(params.get("field_name", "")))
        elif step.get("recipe_id") == "aggregate_to_zones":
            # 聚合产物字段名是 <stat_type>_<stat_fields>，默认 SUM_ 占 4 个字符
            stat = str(params.get("stat_type") or "SUM")
            names += [
                f"{stat}_{field.strip()}"
                for field in str(params.get("stat_fields", "")).split(",")
                if field.strip()
            ]
        elif step.get("recipe_id") == "attribute_join_table":
            # 挂接会把挂接表的字段名**原样**落到 .shp 上（DAIR-171 的 COUNT_COVA 就刚好 10 个字符）
            names += [field.strip() for field in str(params.get("join_fields", "")).split(",") if field.strip()]
        problems += [
            f"{step_id}: {name}（{len(name)} 字符）"
            for name in names
            if name and not name.startswith("$") and len(name) > _SHP_FIELD_LIMIT
        ]
    return problems


def test_shipped_manifests_keep_shp_field_names_within_the_ten_char_limit():
    """仓库里每个 manifest 的 ``*.shp`` 产物字段名都必须 ≤10 字符。"""
    manifests = sorted((ROOT / "templates").glob("*/manifest.y*ml"))
    assert manifests, "templates/ 下一个 manifest 都没找到"
    problems = [
        f"{path.parent.name}/{item}"
        for path in manifests
        for item in _long_shp_field_names(yaml.safe_load(path.read_text(encoding="utf-8")))
    ]
    assert problems == [], f".shp 字段名超过 10 字符（dBASE 上限）：{problems}"

    # 守卫自己也得能红：把 DAIR-168 报的那个 bug 原样喂进来，必须被点名
    broken = {
        "pipeline": [
            {
                "id": "covered_parts",
                "recipe_id": "clip_layer",
                "params": {
                    "input_path": "$data.communities",
                    "clip_path": "$data.service_area",
                    "output_path": "$out/covered_parts.shp",
                },
            },
            {
                # 原地改：产物路径要顺着 input_path 追到 covered_parts.shp
                "id": "covered_area",
                "recipe_id": "field_calculate",
                "params": {
                    "input_path": "$steps.covered_parts.output",
                    "field_name": "COVER_AREA",  # 恰好 10 字符：合法
                    "expression": "!shape.area@squaremeters!",
                },
            },
            {
                "id": "coverage_by_community",
                "recipe_id": "aggregate_to_zones",
                "params": {
                    "zones_path": "$data.communities",
                    "source_path": "$steps.covered_area.output",
                    "output_path": "$out/coverage_by_community.shp",
                    "stat_fields": "COVER_AREA",
                },
            },
        ]
    }
    assert _long_shp_field_names(broken) == [
        "coverage_by_community: SUM_COVER_AREA（14 字符）"
    ]

    # 10 字符上限只属于 .shp：同样的长字段名写到 .gpkg 图层上不该被这条守卫误杀
    broken["pipeline"][2]["params"]["output_path"] = "$out/coverage_by_community.gpkg/coverage"
    assert _long_shp_field_names(broken) == []

    # 挂接落下来的字段名也归这条守卫管：挂接表字段名会原样写到 .shp 上
    joined = {
        "pipeline": [
            {
                "id": "coverage_stats",
                "recipe_id": "summarize_stats",
                "params": {
                    "in_table": "$data.communities",
                    "out_table": "$out/coverage_stats.dbf",
                    "case_field": "COMM_ID",
                    "stat_fields": "COVA SUM;COVA COUNT",
                },
            },
            {
                "id": "coverage_by_community",
                "recipe_id": "attribute_join_table",
                "params": {
                    "target_path": "$data.communities",
                    "target_key": "COMM_ID",
                    "join_table": "$steps.coverage_stats.output",
                    "join_key": "COMM_ID",
                    "join_fields": "SUM_COVA,COUNT_COVERAGE",  # COUNT_COVERAGE 14 字符：写完会被截断
                    "output_path": "$out/coverage_by_community.shp",
                },
            },
        ]
    }
    assert _long_shp_field_names(joined) == [
        "coverage_by_community: COUNT_COVERAGE（14 字符）"
    ]


# ==================== 第二批业务模板（W4 第 3 批：4 条 arcpy-only 链）
# 这 4 个 manifest 也是**仓库里的真文件**。与上一节同样的做法：配方与执行器是假的，
# 但配方的 id / 参数名照真实声明（``test_shipped_manifests_point_at_real_recipes_and_params``
# 会拿真配方库再对一遍），所以链断了、绑定错位、断言写歪都会在这里红。
#
# 与上一节唯一的区别是取数端：这一批的断言里有 ``fields_present`` /
# ``feature_count_between`` / ``crs_wkid``，它们走 ``_describe``（一段 ``arcpy.Describe``
# 代码），不走 SearchCursor。所以假执行器要多答一路 Describe 快照——
# 默认「路径存在、要素数 feature_count、坐标系 wkid、字段 fields」，可按路径覆盖。
_DESC_PATH = re.compile(r"^_p = (.+)$", re.MULTILINE)


class _Batch3Runner(_StatsRunner):
    """假执行器 + Describe 快照 + 制图产物的附属文件（.jpg 真图 / 同名 .aprx / .json 内容）。"""

    def __init__(
        self,
        backends: set[str] | None = None,
        *,
        columns=None,
        feature_count: int = 3,
        fields: tuple[str, ...] = (),
        describe: dict | None = None,
        wkid: int = 4527,
        **kwargs,
    ):
        super().__init__(
            backends, columns=columns, extra_result={"feature_count": feature_count}, **kwargs
        )
        self.feature_count = feature_count
        self.fields = list(fields)
        self.wkid = wkid
        #: 按路径覆盖 Describe 快照（如 {".../city_pop": {"wkid": 4326}}）
        self.describe = {str(k): dict(v) for k, v in (describe or {}).items()}

    def _record(self, path: str) -> dict:
        record = {
            "exists": True,
            "wkid": self.wkid,
            "crs_name": "CGCS2000_3_Degree_GK_Zone_38",
            "geom_type": "Polygon",
            "feature_count": self.feature_count,
            "fields": ["OBJECTID", "Shape"] + self.fields,
            "extent": {"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0},
            "raster_min": None,
            "raster_max": None,
        }
        record.update(self.describe.get(str(path), {}))
        return record

    def run(self, code, *, timeout=None, workspace=None):
        # Describe 快照（_describe_code 里有 arcpy.Describe(_p)、没有 SearchCursor）
        if "arcpy.Describe(_p)" in code and "SearchCursor" not in code:
            path = ast.literal_eval(_DESC_PATH.search(code).group(1))
            return SimpleNamespace(
                ok=True, stdout="", stderr="", result=self._record(str(path)), error=None, display_images=[]
            )
        outcome = super().run(code, timeout=timeout, workspace=workspace)
        out = _output_from_code(code)
        if outcome.ok and out:
            path = Path(out)
            suffix = path.suffix.lower()
            if suffix == ".jpg":  # image_not_blank 要真图：写一张带噪声的 JPEG
                _write_noise_jpeg(path)
                aprx = path.with_suffix(".aprx")
                aprx.write_text("fake-proj", encoding="utf-8")
                outcome.result = {**(outcome.result or {}), "aprx": str(aprx), "aprx_saved": True}
            elif suffix == ".json":  # file_contains：写一份带 most_similar 的最小 JSON
                path.write_text(
                    json.dumps({"most_similar": "长沙市", "ranking": []}, ensure_ascii=False),
                    encoding="utf-8",
                )
        return outcome


def _write_noise_jpeg(path: Path) -> None:
    from PIL import Image  # type: ignore
    import numpy as np  # type: ignore

    rng = np.random.default_rng(7)
    array = rng.integers(0, 255, size=(24, 24, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="JPEG")


def _batch3_recipes(*, code_runner=None) -> RecipeLibrary:
    """第二批业务模板用到的配方（参数名与必填项照真实声明，缺的补进上一节的 8 个里）。"""
    library = _business_recipes(code_runner=code_runner)
    extra = [
        _chain_recipe(
            "create_file_gdb",
            params=("gdb_path", "recreate"),
            required=("gdb_path",),
            output_expr="gdb_path",
        ),
        _chain_recipe(
            "feature_to_centroid",
            params=("input_path", "output_path", "location"),
            required=("input_path", "output_path"),
            output_expr="output_path",
        ),
        _chain_recipe(
            "point_buffer_raster_sum",
            params=("points_path", "id_field", "raster_path", "radius", "output_table", "nodata_as_zero"),
            required=("points_path", "id_field", "raster_path", "radius", "output_table"),
            output_expr="output_table",
        ),
        _chain_recipe(
            "rank_top_bottom_flag",
            params=("input_path", "rank_field", "output_path", "top_n", "bottom_n",
                    "bottom_include_null", "second_field"),
            required=("input_path", "rank_field", "output_path"),
            output_expr="output_path",
        ),
        _chain_recipe(
            "select_by_attribute",
            params=("input_path", "output_path", "where_clause", "expect_min"),
            required=("input_path", "output_path", "where_clause"),
            output_expr="output_path",
        ),
        _chain_recipe(
            "category_map",
            params=("input_path", "category_field", "output_path", "category_colors",
                    "other_color", "title", "aprx_path"),
            required=("input_path", "category_field", "output_path"),
            output_expr="output_path",
        ),
        _chain_recipe(
            "csv_points_spatial_join",
            params=("csv_path", "lon_field", "lat_field", "target_path", "output_path",
                    "source_srid", "target_srid"),
            required=("csv_path", "lon_field", "lat_field", "target_path", "output_path"),
            output_expr="output_path",
        ),
        _chain_recipe(
            "hotspot_analysis",
            params=("input_path", "field_name", "output_path", "conceptualization", "standardization"),
            required=("input_path", "field_name", "output_path"),
            output_expr="output_path",
        ),
        _chain_recipe(
            "census_city_indicators",
            params=("county_path", "census_csv", "city_path", "output_path", "pop_field"),
            required=("county_path", "census_csv", "city_path", "output_path"),
            output_expr="output_path",
        ),
        _chain_recipe(
            "similarity_ranking",
            params=("input_path", "fields", "reference_name", "output_json", "name_field", "top_n"),
            required=("input_path", "fields", "reference_name", "output_json"),
            output_expr="output_json",
        ),
        _chain_recipe(
            "graduated_colors_map",
            params=("input_path", "field_name", "output_path", "title", "purpose", "aprx_path"),
            required=("input_path", "field_name", "output_path"),
            output_expr="output_path",
        ),
    ]
    for recipe in extra:
        library._recipes[recipe.id] = recipe
    return library


def _run_shipped3(
    tmp_path: Path,
    template_id: str,
    data: dict,
    *,
    columns=None,
    feature_count: int = 3,
    fields: tuple[str, ...] = (),
    describe: dict | None = None,
    wkid: int = 4527,
):
    """跑仓库里真实的 manifest（配方与执行器是假的）。返回 (observation, runner)。"""
    runner = _Batch3Runner(
        {"arcpy"},
        columns=columns,
        feature_count=feature_count,
        fields=fields,
        describe=describe,
        wkid=wkid,
    )
    recipes = _batch3_recipes(code_runner=runner)
    library = TemplateLibrary(recipes=recipes)
    observation = library.run(template_id, data, _ctx(tmp_path, recipes, runner))
    return observation, runner


def _svc_data(tmp_path: Path) -> dict:
    return {
        "communities": str(tmp_path / "input" / "community_green.shp"),
        "elderly_raster": str(tmp_path / "input" / "elderly.tif"),
    }


def _city_data(tmp_path: Path) -> dict:
    return {
        "population_csv": str(tmp_path / "input" / "population.CSV"),
        "county": str(tmp_path / "input" / "county.shp"),
        "city": str(tmp_path / "input" / "city.shp"),
    }


def _census_data(tmp_path: Path) -> dict:
    return {
        **_city_data(tmp_path),
        "census_csv": str(tmp_path / "input" / "population_census.csv"),
    }


def _ranked_data(tmp_path: Path) -> dict:
    return {"community_service": str(tmp_path / "input" / "community_rank.gdb" / "community_rank")}


def test_public_service_value_rank_runs_the_whole_chain(tmp_path: Path):
    out = tmp_path / "output" / "public_service_value_rank"
    data = _svc_data(tmp_path)
    observation, runner = _run_shipped3(
        tmp_path,
        "public_service_value_rank",
        data,
        feature_count=3,
        fields=("COMM_ID", "GREEN_AREA", "ELDER_COMM", "SUM_VALUE", "SERVICE_VALUE",
                "TOP_FLAG", "BOTTOM_FLAG"),
        columns={
            str(out / "result_data.gdb" / "buffer_elder_tbl"): {"SUM_VALUE": [10.0, 20.0, 30.0]},
            str(out / "result_data.gdb" / "community_rank"): {
                "TOP_FLAG": [1.0, 0.0, 0.0], "BOTTOM_FLAG": [1.0, 0.0, 0.0]
            },
        },
    )

    assert observation.ok, observation.summary
    steps = observation.data["steps"]
    assert [step["step"] for step in steps] == [
        "result_gdb",
        "project_communities",
        "centroids",
        "service_pop",
        "service_join",
        "service_value",
        "rank",
        "top_communities",
        "bottom_communities",
    ]
    assert [step["recipe_id"] for step in steps] == [
        "create_file_gdb",
        "project_to_crs",
        "feature_to_centroid",
        "point_buffer_raster_sum",
        "attribute_join_table",
        "field_calculate",
        "rank_top_bottom_flag",
        "select_by_attribute",
        "select_by_attribute",
    ]
    by_id = {step["step"]: step for step in steps}
    # 默认值：投影到 4527、500 米半径、前后各 500 名
    assert by_id["project_communities"]["params"]["target_srid"] == 4527
    assert by_id["service_pop"]["params"]["radius"] == 500
    assert by_id["rank"]["params"]["top_n"] == 500
    assert by_id["rank"]["params"]["bottom_n"] == 500
    # 中心点从投影后的社区面转出来：点与栅格的坐标系必须逐字一致
    assert by_id["centroids"]["params"]["input_path"] == by_id["project_communities"]["output"]
    assert by_id["service_pop"]["params"]["points_path"] == by_id["centroids"]["output"]
    assert by_id["service_pop"]["params"]["raster_path"] == data["elderly_raster"]
    # 服务人口按 COMM_ID 挂回社区，服务价值原地算在挂接结果上
    assert by_id["service_join"]["params"]["join_table"] == by_id["service_pop"]["output"]
    assert by_id["service_join"]["params"]["target_path"] == by_id["project_communities"]["output"]
    assert by_id["service_value"]["params"]["input_path"] == by_id["service_join"]["output"]
    assert by_id["service_value"]["output"] == by_id["service_join"]["output"]
    # 无绿地留空值、不填 0：口径写进 manifest，不能被静默改掉
    assert "else None" in by_id["service_value"]["params"]["expression"]
    # 排名：空值当最差，空值内部按 ELDER_COMM 降序（整改迫切性）
    assert by_id["rank"]["params"]["bottom_include_null"] is True
    assert by_id["rank"]["params"]["second_field"] == "ELDER_COMM"
    assert by_id["top_communities"]["params"]["where_clause"] == "TOP_FLAG = 1"
    assert by_id["bottom_communities"]["params"]["where_clause"] == "BOTTOM_FLAG = 1"
    # 8 条模板断言全绿（含名额上界与要素数守恒），不是被 mock 掉的
    assert len(observation.data["assertions"]) == 8
    assert [item["ok"] for item in observation.data["assertions"]] == [True] * 8
    assert runner.codes, "链一步都没执行"


def test_value_rank_assertions_catch_bad_flags_and_empty_service_pop(tmp_path: Path):
    """名额超限 / 服务人口全 0：两条不同断言各红一次。"""
    cases = [
        ({"SUM_VALUE": [10.0], "TOP_FLAG": [1.0] * 600, "BOTTOM_FLAG": [1.0]}, "TOP_FLAG"),
        ({"SUM_VALUE": [10.0], "TOP_FLAG": [1.0], "BOTTOM_FLAG": [1.0] * 600}, "BOTTOM_FLAG"),
        ({"SUM_VALUE": [0.0, 0.0], "TOP_FLAG": [1.0], "BOTTOM_FLAG": [1.0]}, "SUM_VALUE"),
    ]
    for index, (rank_columns, expected) in enumerate(cases):
        case_dir = tmp_path / f"case{index}"  # 每条 case 一个空输出目录（假 .gdb 守卫）
        out = case_dir / "output" / "public_service_value_rank"
        observation, _runner = _run_shipped3(
            case_dir,
            "public_service_value_rank",
            _svc_data(case_dir),
            feature_count=3,
            fields=("SUM_VALUE", "SERVICE_VALUE", "TOP_FLAG", "BOTTOM_FLAG"),
            columns={
                str(out / "result_data.gdb" / "buffer_elder_tbl"):
                    {"SUM_VALUE": rank_columns["SUM_VALUE"]},
                str(out / "result_data.gdb" / "community_rank"):
                    {k: v for k, v in rank_columns.items() if k != "SUM_VALUE"},
            },
        )

        assert not observation.ok, expected
        failed = [item for item in observation.data["assertions"] if not item["ok"]]
        assert len(failed) == 1, (expected, failed)
        assert expected in failed[0]["detail"]
        assert [step["ok"] for step in observation.data["steps"]] == [True] * 9


def test_value_rank_describe_guard_catches_more_ranked_features_than_input(tmp_path: Path):
    """排名产物比输入社区还多（CopyFeatures 出了扇出）：要素数守恒断言必须红。"""
    out = tmp_path / "output" / "public_service_value_rank"
    observation, _runner = _run_shipped3(
        tmp_path,
        "public_service_value_rank",
        _svc_data(tmp_path),
        feature_count=3,
        fields=("SUM_VALUE", "SERVICE_VALUE", "TOP_FLAG", "BOTTOM_FLAG"),
        describe={str(out / "result_data.gdb" / "community_rank"): {"feature_count": 9}},
        columns={str(out / "result_data.gdb" / "buffer_elder_tbl"): {"SUM_VALUE": [10.0]},
                 str(out / "result_data.gdb" / "community_rank"): {"TOP_FLAG": [1.0],
                                                                  "BOTTOM_FLAG": [1.0]}},
    )

    assert not observation.ok
    failed = [item for item in observation.data["assertions"] if not item["ok"]]
    assert [item["kind"] for item in failed] == ["feature_count_between"]
    assert "count=9" in failed[0]["detail"]


def test_map_quality_eval_builds_the_category_field_and_asks_for_the_map(tmp_path: Path):
    """分类图模板：类别来自 TOP_FLAG/BOTTOM_FLAG；工程核验交给真 ArcGIS。"""
    out = tmp_path / "output" / "map_quality_eval"
    observation, runner = _run_shipped3(
        tmp_path,
        "map_quality_eval",
        _ranked_data(tmp_path),
        feature_count=5,
        fields=("CATEGORY", "TOP_FLAG", "BOTTOM_FLAG"),
        columns={str(out / "community_eval.gdb" / "community_eval"):
                 {"CATEGORY": ["标杆社区", "需整改社区", "其他"]}},
    )

    assert observation.data["steps"] and all(step["ok"] for step in observation.data["steps"])
    steps = observation.data["steps"]
    assert [step["step"] for step in steps] == ["copy_input", "category", "map"]
    by_id = {step["step"]: step for step in steps}
    # 先复制再改：不在调用方的图层上原地加字段
    assert by_id["copy_input"]["params"]["input_path"].endswith("community_rank")
    assert by_id["category"]["params"]["input_path"] == by_id["copy_input"]["output"]
    assert by_id["category"]["params"]["field_name"] == "CATEGORY"
    assert "标杆社区" in by_id["category"]["params"]["expression"]
    # 制图口径：唯一值 + 两类固定配色 + 浅灰其余
    assert by_id["map"]["params"]["category_field"] == "CATEGORY"
    assert by_id["map"]["params"]["category_colors"] == "标杆社区:#1F77B4;需整改社区:#2CA02C"
    assert by_id["map"]["params"]["other_color"] == "#E8E8E8"
    assert by_id["map"]["params"]["title"] == "社区服务价值评价"

    # 产物相关的断言全绿；aprx_map_check 在 .venv 里没有 arcpy，只能「未核验」——
    # 真工程核验在 arcpy 冷启动里跑（验收记录见 docs/templates.md）。
    failed = [item for item in observation.data["assertions"] if not item["ok"]]
    assert [item["kind"] for item in failed] == ["aprx_map_check"]
    assert "未核验" in failed[0]["detail"]
    assert len(observation.data["assertions"]) == 6
    assert (out / "community_eval_map.jpg").is_file()
    assert (out / "community_eval_map.aprx").is_file()
    assert runner.codes


def test_map_quality_eval_assertion_goes_red_when_only_one_category_survives(tmp_path: Path):
    """类别只剩一种（标记字段读错）时，field_distinct_gt 必须红。"""
    out = tmp_path / "output" / "map_quality_eval"
    observation, _runner = _run_shipped3(
        tmp_path,
        "map_quality_eval",
        _ranked_data(tmp_path),
        feature_count=5,
        fields=("CATEGORY",),
        columns={str(out / "community_eval.gdb" / "community_eval"):
                 {"CATEGORY": ["标杆社区", "标杆社区", "标杆社区"]}},
    )

    assert not observation.ok
    failed = [item for item in observation.data["assertions"] if not item["ok"]]
    # 类别断言必须红；aprx 在 .venv 里只能「未核验」（真核验在 arcpy 冷启动里）
    assert [item["kind"] for item in failed] == ["field_distinct_gt", "aprx_map_check"]
    assert "distinct(CATEGORY)=1" in failed[0]["detail"]


def test_city_hotspot_aggregate_runs_the_whole_chain(tmp_path: Path):
    out = tmp_path / "output" / "city_hotspot_aggregate"
    data = _city_data(tmp_path)
    observation, runner = _run_shipped3(
        tmp_path,
        "city_hotspot_aggregate",
        data,
        feature_count=49,
        fields=("name", "SUM_Y2011", "SUM_Y2012", "SUM_Y2013", "SUM_Y2020", "SRC_COUNT",
                "GiZScore", "GiPValue", "Gi_Bin"),
        wkid=4508,
        columns={
            str(out / "result_data.gdb" / "city_pop"): {"SRC_COUNT": [3.0, 4.0],
                                                        "SUM_Y2011": [100.0, 200.0]},
            str(out / "result_data.gdb" / "city_hot_2011"): {"Gi_Bin": [3.0, -2.0, 0.0]},
            str(out / "result_data.gdb" / "city_hot_2013"): {"Gi_Bin": [1.0, -1.0]},
        },
    )

    assert observation.ok, observation.summary
    steps = observation.data["steps"]
    assert [step["step"] for step in steps] == [
        "county_pop", "city_proj", "city_pop", "hotspot_2011", "hotspot_2012", "hotspot_2013",
    ]
    by_id = {step["step"]: step for step in steps}
    # CSV 转点在区县面上做：缺省经纬度列名与 4326 源坐标系写死在 manifest 里
    assert by_id["county_pop"]["params"]["csv_path"] == data["population_csv"]
    assert by_id["county_pop"]["params"]["target_path"] == data["county"]
    assert by_id["county_pop"]["params"]["source_srid"] == 4326
    assert by_id["county_pop"]["params"]["target_srid"] == 4508
    assert by_id["city_proj"]["params"]["target_srid"] == 4508
    assert by_id["city_pop"]["params"]["zones_path"] == by_id["city_proj"]["output"]
    assert by_id["city_pop"]["params"]["source_path"] == by_id["county_pop"]["output"]
    assert by_id["city_pop"]["params"]["stat_fields"] == "Y2011,Y2012,Y2013,Y2020"
    # 三年各一次 Gi*，分析字段是上一步汇总出来的 SUM_Y*
    assert [by_id[f"hotspot_{year}"]["params"]["field_name"]
            for year in (2011, 2012, 2013)] == ["SUM_Y2011", "SUM_Y2012", "SUM_Y2013"]
    assert all(by_id[f"hotspot_{year}"]["params"]["input_path"] == by_id["city_pop"]["output"]
               for year in (2011, 2012, 2013))
    assert len(observation.data["assertions"]) == 9
    assert [item["ok"] for item in observation.data["assertions"]] == [True] * 9
    assert runner.codes


def test_hotspot_assertions_catch_wrong_crs_and_a_single_gi_bin(tmp_path: Path):
    """坐标系没投影（4326）/ Gi_Bin 只有一个取值：两条断言各红一次。"""
    crs_dir = tmp_path / "crs"
    out = crs_dir / "output" / "city_hotspot_aggregate"
    base_columns = {
        str(out / "result_data.gdb" / "city_pop"): {"SRC_COUNT": [3.0], "SUM_Y2011": [100.0]},
        str(out / "result_data.gdb" / "city_hot_2011"): {"Gi_Bin": [3.0, -1.0]},
        str(out / "result_data.gdb" / "city_hot_2013"): {"Gi_Bin": [1.0, -1.0]},
    }
    observation, _runner = _run_shipped3(
        crs_dir,
        "city_hotspot_aggregate",
        _city_data(crs_dir),
        feature_count=49,
        fields=("SUM_Y2011", "SUM_Y2012", "SUM_Y2013", "SUM_Y2020", "SRC_COUNT",
                "GiZScore", "GiPValue", "Gi_Bin"),
        wkid=4326,  # 负例：汇总结果没投影（还是 4326）
        columns=base_columns,
    )
    assert not observation.ok
    failed = [item for item in observation.data["assertions"] if not item["ok"]]
    assert [item["kind"] for item in failed] == ["crs_wkid"]
    assert "4326" in failed[0]["detail"]

    single_dir = tmp_path / "single"
    single_out = single_dir / "output" / "city_hotspot_aggregate"
    observation, _runner = _run_shipped3(
        single_dir,
        "city_hotspot_aggregate",
        _city_data(single_dir),
        feature_count=49,
        fields=("SUM_Y2011", "SUM_Y2012", "SUM_Y2013", "SUM_Y2020", "SRC_COUNT",
                "GiZScore", "GiPValue", "Gi_Bin"),
        wkid=4508,
        columns={
            str(single_out / "result_data.gdb" / "city_pop"): {"SRC_COUNT": [3.0],
                                                               "SUM_Y2011": [100.0]},
            str(single_out / "result_data.gdb" / "city_hot_2011"): {"Gi_Bin": [3.0, -1.0]},
            str(single_out / "result_data.gdb" / "city_hot_2013"): {"Gi_Bin": [1.0]},
        },
    )
    assert not observation.ok
    failed = [item for item in observation.data["assertions"] if not item["ok"]]
    assert [item["kind"] for item in failed] == ["field_distinct_gt"]
    assert "Gi_Bin" in failed[0]["detail"]


def test_census_similarity_runs_the_whole_chain(tmp_path: Path):
    out = tmp_path / "output" / "census_indicator_similarity"
    data = _census_data(tmp_path)
    observation, runner = _run_shipped3(
        tmp_path,
        "census_indicator_similarity",
        data,
        feature_count=49,
        fields=("name", "OLD_PCT", "FERT_PCT", "SEX_RATIO", "SRC_COUNT", "SUM_Y2020"),
        wkid=4508,
        columns={
            str(out / "result_data.gdb" / "city_census"): {
                "SRC_COUNT": [3.0, 4.0], "SUM_Y2020": [1000.0, 2000.0],
                "OLD_PCT": [10.5, 11.2], "FERT_PCT": [20.0, 22.0],
            },
        },
    )

    assert observation.ok, observation.summary
    steps = observation.data["steps"]
    assert [step["step"] for step in steps] == [
        "county_pop", "city_proj", "city_census", "similarity", "map_old", "map_fert", "map_sex",
    ]
    by_id = {step["step"]: step for step in steps}
    assert by_id["county_pop"]["params"]["csv_path"] == data["population_csv"]
    assert by_id["city_census"]["params"]["census_csv"] == data["census_csv"]
    assert by_id["city_census"]["params"]["county_path"] == by_id["county_pop"]["output"]
    assert by_id["city_census"]["params"]["city_path"] == by_id["city_proj"]["output"]
    assert by_id["city_census"]["params"]["pop_field"] == "Y2020"
    # 相似度：三个指标 + 参照城市，输入是上一步的 city_census
    assert by_id["similarity"]["params"]["input_path"] == by_id["city_census"]["output"]
    assert by_id["similarity"]["params"]["fields"] == "OLD_PCT,FERT_PCT,SEX_RATIO"
    assert by_id["similarity"]["params"]["reference_name"] == "郑州市"
    # 三张图各取一个字段，都读 city_census（不是回头去读原始表）
    assert [(by_id[name]["params"]["field_name"], by_id[name]["params"]["output_path"]
             .endswith(f"{field}.jpg"))
            for name, field in (("map_old", "old_pct_map"), ("map_fert", "fert_pct_map"),
                                ("map_sex", "sex_ratio_map"))] == [
        ("OLD_PCT", True), ("FERT_PCT", True), ("SEX_RATIO", True)
    ]
    assert by_id["map_old"]["params"]["field_name"] == "OLD_PCT"
    assert by_id["map_fert"]["params"]["field_name"] == "FERT_PCT"
    assert by_id["map_sex"]["params"]["field_name"] == "SEX_RATIO"
    assert all(by_id[name]["params"]["input_path"] == by_id["city_census"]["output"]
               for name in ("map_old", "map_fert", "map_sex"))
    assert len(observation.data["assertions"]) == 17
    assert [item["ok"] for item in observation.data["assertions"]] == [True] * 17
    assert runner.codes


def test_census_assertion_catches_percent_scale(tmp_path: Path):
    """占比忘了 /100（均值 200%）：field_stats_between 必须红。"""
    out = tmp_path / "output" / "census_indicator_similarity"
    observation, _runner = _run_shipped3(
        tmp_path,
        "census_indicator_similarity",
        _census_data(tmp_path),
        feature_count=49,
        fields=("OLD_PCT", "FERT_PCT", "SEX_RATIO", "SRC_COUNT", "SUM_Y2020"),
        wkid=4508,
        columns={
            str(out / "result_data.gdb" / "city_census"): {
                "SRC_COUNT": [3.0], "SUM_Y2020": [1000.0],
                "OLD_PCT": [199.0, 201.0], "FERT_PCT": [20.0],
            },
        },
    )
    assert not observation.ok
    failed = [item for item in observation.data["assertions"] if not item["ok"]]
    assert [item["kind"] for item in failed] == ["field_stats_between"]
    assert "OLD_PCT" in failed[0]["detail"]


@pytest.mark.parametrize(
    "template_id, data_factory",
    [
        ("public_service_value_rank", _svc_data),
        ("map_quality_eval", _ranked_data),
        ("city_hotspot_aggregate", _city_data),
        ("census_indicator_similarity", _census_data),
    ],
)
def test_batch3_templates_need_arcgis_and_run_nothing_on_a_gdal_runner(
    tmp_path: Path, template_id: str, data_factory
):
    """这 4 条链都是 arcpy-only：GDAL-only 机器上必须预检就报「需 ArcGIS」，一步都不跑。"""
    recipes = _batch3_recipes()
    library = TemplateLibrary(recipes=recipes)
    runner = _Runner({"gdal"})

    observation = library.run(template_id, data_factory(tmp_path), _ctx(tmp_path, recipes, runner))

    assert not observation.ok
    assert observation.error["type"] == "BackendUnavailable"
    assert "需 ArcGIS" in observation.summary
    assert observation.error["required"] == ["arcpy"]
    assert not runner.codes, "缺后端时一行配方代码都不该执行"


def test_batch3_list_templates_shows_inputs_backends_and_runnability(tmp_path: Path):
    """8 个模板都要能在 list_templates 里看到需要什么数据、能不能在本机跑。"""
    recipes = _batch3_recipes()
    library = TemplateLibrary(recipes=recipes)  # 仓库里真实的 templates/ 目录
    ctx = EngineContext(
        workspace=tmp_path, recipes=recipes, code_runner=_Runner({"gdal"}), templates=library
    )

    observation = build_default_registry().execute("list_templates", {}, ctx)

    assert observation.ok, observation.summary
    templates = {item["id"]: item for item in observation.data["templates"]}
    assert set(templates) == {
        "demo_project_then_buffer",
        "demo_project_then_zonal",
        "elderly_care_coverage",
        "green_space_equity",
        "public_service_value_rank",
        "map_quality_eval",
        "city_hotspot_aggregate",
        "census_indicator_similarity",
    }
    expected = {
        "public_service_value_rank": (
            {"communities", "elderly_raster", "target_srid", "service_radius",
             "community_id_field", "top_n", "bottom_n"},
            9,
        ),
        "map_quality_eval": ({"community_service", "map_title"}, 3),
        "city_hotspot_aggregate": (
            {"population_csv", "county", "city", "target_srid", "lon_field", "lat_field"}, 6
        ),
        "census_indicator_similarity": (
            {"population_csv", "census_csv", "county", "city", "target_srid",
             "reference_name", "lon_field", "lat_field"}, 7
        ),
    }
    for tid, (data_names, steps) in expected.items():
        info = templates[tid]
        # 4 条都是 arcpy-only：纯 GDAL 机器上跑不了，列表里要能看出来
        assert info["backend"] == ["arcpy"]
        assert info["runnable"] is False
        assert set(info["data"]) == data_names
        assert len(info["pipeline"]) == steps
        for name, spec in info["data"].items():
            assert spec["purpose"], f"{tid}.{name} 没有 purpose"
        assert info["description"], f"{tid} 没有 description"

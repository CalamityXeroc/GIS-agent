# -*- coding: utf-8 -*-
"""每后端一份代码模板：``Recipe.render(..., backend=)`` 与 store 的路由。

三条必须锁死的边界：

1. **老配方零变化**：47 个 arcpy 配方的渲染输出与改动前逐字相同
   （快照 ``tests/snapshots/builtin_render_arcpy.json``，改动前生成）；
2. **不回落**：选了 gdal 但没写 gdal 模板 → 明确报错，绝不把 arcpy 代码交给
   .venv-gdal（那会在 ``import arcpy`` 上炸，报一个与配方内容无关的错）；
3. **路由咬合**：``RecipeLibrary.run`` 把实际选中的后端传进 ``render``，执行的
   确实是那个后端的模板。

全部不依赖真 arcpy / 真 GDAL —— CI 机器两者都没有。
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gis_cli.recipes.fragments import FRAGMENTS, fragment_names
from gis_cli.recipes.gdal_checks import gdal_vector_check
from gis_cli.recipes.schema import Recipe
from gis_cli.recipes.store import RecipeLibrary

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(__file__).resolve().parent / "snapshots" / "builtin_render_arcpy.json"


class _Runner:
    """假执行器：声明能力 + 记录每段被执行的代码。"""

    def __init__(self, backends: set[str] | None = None, result: dict | None = None):
        if backends is not None:
            self.backends = set(backends)
        self.result = result if result is not None else {"ok": True}
        self.codes: list[str] = []

    def run(self, code, *, timeout=None, workspace=None):
        self.codes.append(code)
        return SimpleNamespace(
            ok=True, stdout="", stderr="", result=self.result, error=None, display_images=[]
        )


def _render_params(recipe: Recipe) -> dict:
    """每个参数给一个确定值（与快照生成时同一套规则）。"""
    params = {}
    for name, spec in recipe.params.items():
        if spec.default is not None:
            params[name] = spec.default
        elif spec.type in {"integer", "number"}:
            params[name] = 1
        else:
            params[name] = f"dummy_{name}"
    return params


def _library(tmp_path: Path, recipe: Recipe, runner) -> RecipeLibrary:
    lib = RecipeLibrary(user_dir=tmp_path / "recipes", code_runner=runner)
    lib.load()
    lib._recipes[recipe.id] = recipe
    return lib


def _ctx(tmp_path: Path, runner) -> SimpleNamespace:
    return SimpleNamespace(workspace=tmp_path, code_runner=runner, exec_timeout=10)


# --------------------------------------------------- 老配方：逐字不变的快照
def test_arcpy_render_is_byte_identical_to_snapshot():
    """47 个内置配方的 arcpy 渲染（sha256 + 长度）与改动前一致。"""
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    lib = RecipeLibrary()
    lib.load()
    got = {}
    for recipe in lib.all():
        code = recipe.render(_render_params(recipe))  # 默认后端 = arcpy
        got[recipe.id] = {
            "sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "chars": len(code),
        }
        # 显式写 backend="arcpy" 必须与默认调用逐字相同
        assert recipe.render(_render_params(recipe), backend="arcpy") == code, recipe.id

    assert set(got) == set(snapshot), (
        f"内置配方集合变了：新增 {sorted(set(got) - set(snapshot))}，"
        f"消失 {sorted(set(snapshot) - set(got))}"
    )
    changed = {rid: (snapshot[rid], got[rid]) for rid in snapshot if snapshot[rid] != got[rid]}
    assert not changed, f"{len(changed)} 个配方的 arcpy 渲染输出变了: {sorted(changed)[:5]}"


# ------------------------------------------------- 全库守门：模板必须是合法 Python
def test_every_builtin_template_compiles():
    """47 个配方的 arcpy / gdal 模板渲染后都必须能 ``ast.parse``。

    快照测试只比 sha256 —— 模板坏成什么样子它都照样绿，``category_map`` 的
    ``if`` 缺 body 就这样藏了 6 周：渲染出来必然 IndentationError，任何机器都跑不了
    第一步，而全部测试仍然通过。这里直接把渲染结果交给解析器。
    """
    lib = RecipeLibrary()
    lib.load()

    broken: dict[str, str] = {}
    backends: set[str] = set()
    for recipe in lib.all():
        for backend in ("arcpy", "gdal"):
            # 没有 GDAL 版是显式声明（由 test_gdal_render_without_gdal_template_raises 管）
            if backend == "gdal" and not recipe.code_template_gdal:
                continue
            backends.add(backend)
            try:
                ast.parse(recipe.render(_render_params(recipe), backend=backend))
            except SyntaxError as exc:
                broken[f"{recipe.id}:{backend}"] = (
                    f"{exc.msg} @ line {exc.lineno}: {(exc.text or '').strip()}"
                )
            except Exception as exc:  # 渲染本身炸了同样是模板的问题
                broken[f"{recipe.id}:{backend}"] = f"{type(exc).__name__}: {exc}"

    assert len(lib.all()) >= 47, f"内置配方只剩 {len(lib.all())} 个，守门测试可能空转"
    assert backends == {"arcpy", "gdal"}, f"两套模板都要覆盖，实际只跑了 {backends}"
    assert not broken, f"{len(broken)} 个模板渲染后不是合法 Python: {broken}"


# ---------------------------------------------------------- 按后端选模板
def _two_templates(**kwargs) -> Recipe:
    return Recipe(
        id="probe",
        name="probe",
        code_template="import arcpy\nx = {{value}}\nset_result({'arcpy': x})\n",
        code_template_gdal="import pyogrio\nx = {{value}}\nset_result({'gdal': x})\n",
        params={},
        **kwargs,
    )


def test_render_picks_template_per_backend():
    recipe = _two_templates(backend=["arcpy", "gdal"])
    arcpy_code = recipe.render({"value": 1})
    gdal_code = recipe.render({"value": 1}, backend="gdal")

    assert arcpy_code.startswith("import arcpy") and "import pyogrio" not in arcpy_code
    assert gdal_code.startswith("import pyogrio") and "import arcpy" not in gdal_code
    # 参数替换两条模板一致生效
    assert "x = 1" in arcpy_code and "x = 1" in gdal_code


def test_gdal_render_without_gdal_template_raises():
    """没写 GDAL 版 → 报错，**不回落** arcpy 代码（回落会在 .venv-gdal 里撞 import arcpy）。"""
    recipe = _two_templates(backend=["arcpy", "gdal"])
    recipe.code_template_gdal = ""

    with pytest.raises(ValueError) as excinfo:
        recipe.render({"value": 1}, backend="gdal")
    message = str(excinfo.value)
    assert "没有 GDAL 版代码" in message and "probe" in message


def test_unknown_backend_raises():
    with pytest.raises(ValueError) as excinfo:
        _two_templates().render({}, backend="qgis")
    assert "未知执行后端" in str(excinfo.value)


# --------------------------------------------------------------- 公共代码片段
def test_fragment_expands_at_the_placeholder_indent():
    """``{{@zone_layer@}}`` 展开成 fragments.py 的源码，并按占位行的缩进对齐。

    按占位行缩进是关键：片段里既有 ``def`` 也有 ``if``/``raise``，错一层就是 IndentationError。
    """
    recipe = Recipe(
        id="probe",
        name="probe",
        code_template="import arcpy, os\n\ndef outer():\n    {{@zone_layer@}}\n    return 1\n",
    )

    code = recipe.render({})

    assert "{{@" not in code and "def resolve_zone_layer(path):" in code
    assert "\ndef resolve_zone_layer" not in code, "片段没跟着缩进，会掉回顶格"
    ast.parse(code)


def test_unknown_fragment_raises_and_lists_the_available_ones():
    """片段名写错要当场报错（可用名一并给出），不能把占位符原样交给执行器。"""
    recipe = Recipe(id="probe", name="probe", code_template="{{@no_such_fragment@}}\n")

    with pytest.raises(ValueError) as excinfo:
        recipe.render({})

    message = str(excinfo.value)
    assert "no_such_fragment" in message and "zone_layer" in message


def test_inline_fragment_raises_instead_of_breaking_the_syntax():
    """``x = {{@zone_layer@}}`` 必须当场报人话错误。

    以前会把整段 ``def ...`` 拼在 ``x = `` 后面，拼成 ``x = def resolve_zone_layer(path):``
    ——SyntaxError 的报错点在执行器里，离真正的原因（占位符没独占一行）很远。
    """
    recipe = Recipe(
        id="probe", name="probe", code_template="import arcpy, os\nx = {{@zone_layer@}}\n"
    )

    with pytest.raises(ValueError) as excinfo:
        recipe.render({})

    message = str(excinfo.value)
    assert "独占一行" in message and "zone_layer" in message


def test_two_fragments_on_one_line_raise_instead_of_leaving_the_second():
    """一行两个占位符也必须报错：以前只展开第一个，第二个原样留在代码里。"""
    recipe = Recipe(
        id="probe", name="probe", code_template="{{@zone_layer@}}{{@zone_layer@}}\n"
    )

    with pytest.raises(ValueError) as excinfo:
        recipe.render({})

    message = str(excinfo.value)
    assert "独占一行" in message and "2 个" in message


def test_zone_container_logic_lives_in_one_place():
    """7 个配方的容器判定来自同一片段，渲染结果里片段体只出现一次（不是七份副本）。"""
    lib = RecipeLibrary()
    lib.load()
    for recipe_id in (
        "zonal_statistics",
        "zonal_statistics_raster",
        "zone_raster_sum",
        "aggregate_to_zones",
        "attribute_join_table",
        "project_to_crs",
        "point_buffer_raster_sum",
    ):
        recipe = lib.get(recipe_id)
        assert recipe is not None, recipe_id
        code = recipe.render(_render_params(recipe))
        assert code.count("def resolve_zone_layer(path):") == 1, recipe_id
        assert code.count("resolve_zone_layer(") == 2, recipe_id  # 定义 + 一次调用
        assert "_zone_layers" not in code, f"{recipe_id} 又抄了一份容器判定"
        ast.parse(code)

    # 模板里只允许引用真实存在的片段名
    for recipe in lib.all():
        for template in (recipe.code_template, recipe.code_template_gdal):
            for name in fragment_names(template):
                assert name in FRAGMENTS, f"{recipe.id} 引用了未知片段 {name}"


def test_gdal_template_survives_dict_roundtrip():
    recipe = _two_templates(backend=["arcpy", "gdal"])
    data = recipe.to_dict(with_code=True)
    assert data["code_template_gdal"] == recipe.code_template_gdal
    assert Recipe.from_dict(data).code_template_gdal == recipe.code_template_gdal

    # 老配方（没有 gdal 模板）的 to_dict 形态不变：不凭空多一个键
    legacy = Recipe(id="legacy", name="legacy", code_template="print(1)\n")
    assert "code_template_gdal" not in legacy.to_dict(with_code=True)


# ------------------------------------------------- store 把选中的后端传下去
def test_store_renders_the_selected_backend_template(tmp_path: Path):
    runner = _Runner({"arcpy", "gdal"})
    gdal_recipe = _two_templates(backend=["gdal", "arcpy"])
    arcpy_recipe = _two_templates(backend=["arcpy", "gdal"])

    obs_gdal = _library(tmp_path, gdal_recipe, runner).run(
        gdal_recipe.id, {"value": 1}, _ctx(tmp_path, runner)
    )
    assert obs_gdal.ok, obs_gdal.summary
    assert obs_gdal.data["backend"] == "gdal"
    assert "import pyogrio" in runner.codes[-1] and "import arcpy" not in runner.codes[-1]

    obs_arcpy = _library(tmp_path, arcpy_recipe, runner).run(
        arcpy_recipe.id, {"value": 1}, _ctx(tmp_path, runner)
    )
    assert obs_arcpy.ok, obs_arcpy.summary
    assert obs_arcpy.data["backend"] == "arcpy"
    assert "import arcpy" in runner.codes[-1] and "import pyogrio" not in runner.codes[-1]


def test_store_reports_missing_gdal_template_without_running(tmp_path: Path):
    runner = _Runner({"gdal"})
    recipe = _two_templates(backend=["gdal"])
    recipe.code_template_gdal = ""

    obs = _library(tmp_path, recipe, runner).run(recipe.id, {}, _ctx(tmp_path, runner))

    assert not obs.ok
    assert "没有 GDAL 版代码" in obs.summary
    assert not runner.codes, "没有模板时一行代码都不该执行"


# ------------------------------------------------------ buffer_dissolve 本身
def test_buffer_dissolve_declares_two_backends_and_renders_both():
    recipe = Recipe.from_yaml(ROOT / "src/gis_cli/recipes/builtin/buffer_dissolve.yaml")
    assert recipe.backend == ["arcpy", "gdal"]
    assert "arcpy.analysis.Buffer" in recipe.render(_render_params(recipe))

    gdal_code = recipe.render(_render_params(recipe), backend="gdal")
    assert "gpd.read_file" in gdal_code and "import arcpy" not in gdal_code
    # GPKG 不是 shp：shapefile 字段名会截断到 10 字符
    assert 'driver="GPKG"' in gdal_code


# --------------------------------------------------------- GDAL 侧断言口径
class _GdalInfoRunner:
    """假 GDAL 执行器：只回传 read_info 的结果，用来验断言口径。"""

    def __init__(self, payload: dict):
        self.payload = payload
        self.codes: list[str] = []

    def run(self, code, *, timeout=None, workspace=None):
        self.codes.append(code)
        return SimpleNamespace(ok=True, stdout="", stderr="", result=self.payload, error=None)


@pytest.mark.parametrize(
    "kind,expected,want_ok",
    [
        ("vector_exists", None, True),
        ("feature_count_gt", 2, True),
        ("feature_count_gt", 3, False),
        ("feature_count_equal", 3, True),
        ("feature_count_lt", 4, True),
        ("feature_count_between", [1, 5], True),
        ("feature_count_between", [5, 9], False),
    ],
)
def test_gdal_vector_check_counts_with_pyogrio(kind, expected, want_ok):
    runner = _GdalInfoRunner({"exists": True, "feature_count": 3, "crs": "EPSG:32650"})
    ok, detail = gdal_vector_check(runner, kind, "out.gpkg", expected)
    assert ok is want_ok
    assert "pyogrio" in runner.codes[0] and "arcpy" not in runner.codes[0]
    assert "count=" in detail or kind == "vector_exists"


def test_gdal_vector_check_leaves_other_kinds_alone():
    """不接管的 kind（file_exists 等与后端无关）返回 None，调用方走内置实现。"""
    runner = _GdalInfoRunner({"exists": True, "feature_count": 3})
    assert gdal_vector_check(runner, "file_exists", "x.txt", None) is None
    assert not runner.codes
    # 没有执行器 → 也不接管
    assert gdal_vector_check(None, "feature_count_gt", "x.gpkg", 0) is None


def test_missing_dataset_is_reported_not_counted_as_zero_features():
    runner = _GdalInfoRunner({"exists": False, "reason": "文件不存在"})
    ok, detail = gdal_vector_check(runner, "feature_count_gt", "missing.gpkg", 0)
    assert ok is False
    assert "不存在" in detail

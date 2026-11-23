# -*- coding: utf-8 -*-
"""P0-1a 后端契约：配方声明 backend，执行与断言按声明路由。

全部用假 runner / 假断言实现，**不依赖真 arcpy、也不依赖真 GDAL** ——
CI 机器两者都没有（``find_spec("arcpy")`` / ``find_spec("osgeo")`` 都是 None）。

四条路由路径 = arcpy 有/无 × gdal 有/无；外加"缺省即 arcpy"的锁死用例。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gis_cli.recipes import store as store_mod
from gis_cli.recipes.backends import (
    DEFAULT_BACKEND,
    normalize_backends,
    pick_backend,
    runner_backends,
    unavailable_message,
)
from gis_cli.recipes.schema import Recipe, RecipeParam
from gis_cli.recipes.store import RecipeLibrary, ValidationResult

ROOT = Path(__file__).resolve().parents[1]


class _Runner:
    """假执行器：声明自己能跑哪些后端；run() 只记账，不真跑代码。"""

    def __init__(self, backends: set[str] | None = None):
        if backends is not None:
            self.backends = set(backends)
        self.codes: list[str] = []

    def run(self, code, *, timeout=None, workspace=None):
        self.codes.append(code)
        return SimpleNamespace(
            ok=True, stdout="", stderr="", result={"ok": True}, error=None, display_images=[]
        )


def _recipe(rid: str = "probe", *, backend=None, validation=None) -> Recipe:
    kwargs = {} if backend is None else {"backend": list(backend)}
    return Recipe(
        id=rid,
        name=rid,
        # 两个模板都给：本文件验的是**路由**，不验模板缺没缺（那个由
        # test_recipe_backend_templates.py 管）。只给 arcpy 模板的话，
        # 带 gdal 的用例会先在 render 上撞“没有 GDAL 版代码”。
        code_template="print('probe')\n",
        code_template_gdal="print('probe gdal')\n",
        params={"out": RecipeParam("out", "string", False, "output/x.txt", "产物")},
        validation=list(validation or []),
        **kwargs,
    )


def _library(tmp_path: Path, recipe: Recipe, runner) -> RecipeLibrary:
    lib = RecipeLibrary(user_dir=tmp_path / "recipes", code_runner=runner)
    lib.load()
    lib._recipes[recipe.id] = recipe  # 直接放进内存库（不走磁盘）
    return lib


def _ctx(tmp_path: Path, runner) -> SimpleNamespace:
    return SimpleNamespace(workspace=tmp_path, code_runner=runner, exec_timeout=10)


# ------------------------------------------------------------------ 契约本身
def test_recipe_backend_defaults_to_arcpy():
    """缺省即 arcpy：新增配方忘写 backend 时，绝不能默认成 GDAL。"""
    assert Recipe(id="x", name="x").backend == ["arcpy"]
    assert Recipe.from_dict({"id": "x", "name": "x"}).backend == ["arcpy"]
    assert Recipe.from_dict({"id": "x", "name": "x", "backend": "gdal"}).backend == ["gdal"]
    assert Recipe.from_dict({"id": "x", "name": "x", "backend": ["gdal", "arcpy"]}).backend == [
        "gdal",
        "arcpy",
    ]
    # 写错/写空的名字回落到 arcpy：拼错一个词不该让配方变成"本机不可用"
    assert normalize_backends(" GDAL ") == ["gdal"]
    assert normalize_backends([]) == ["arcpy"]
    assert normalize_backends(["ogr", "arcpy", "arcpy"]) == ["arcpy"]
    assert normalize_backends(["whatever"]) == ["arcpy"]


def test_builtin_recipes_keep_arcpy_and_only_migrated_ones_have_gdal():
    """47 个现有 manifest 里只有已移植的配方新增了 gdal 后端。

    P0-1b 每移植一个配方，这里就多一个例外 —— 其余配方仍声明 ``[arcpy]``，
    一个都不能被默默改成 GDAL。
    """
    lib = RecipeLibrary()
    recipes = lib.all()
    assert len(recipes) == 47
    assert {r.id: r.backend for r in recipes if r.backend != ["arcpy"]} == {
        "buffer_dissolve": ["arcpy", "gdal"],
        "aggregate_to_zones": ["arcpy", "gdal"],
        "zonal_statistics": ["arcpy", "gdal"],
        "zonal_statistics_raster": ["arcpy", "gdal"],
        "polygon_to_raster": ["arcpy", "gdal"],
        "raster_calculator": ["arcpy", "gdal"],
        "attribute_join_table": ["arcpy", "gdal"],
        "project_to_crs": ["arcpy", "gdal"],
        "point_buffer_raster_sum": ["arcpy", "gdal"],
    }


def test_runner_backends_only_trusts_declaration():
    assert runner_backends(_Runner({"arcpy"})) == {"arcpy"}
    # 声明写成单个字符串也认（manifest 里 "backend: gdal" 的形态）
    assert runner_backends(SimpleNamespace(backends="gdal")) == {"gdal"}
    assert runner_backends(_Runner(set())) == set()
    # 没声明 = 不知道，不设门禁（历史默认：所有执行器都是 arcpy）
    assert runner_backends(_Runner()) is None
    assert runner_backends(None) is None


def test_pick_backend_follows_declared_order():
    assert pick_backend(["gdal", "arcpy"], {"arcpy", "gdal"}) == "gdal"
    assert pick_backend(["gdal", "arcpy"], {"arcpy"}) == "arcpy"
    assert pick_backend(["gdal"], {"arcpy"}) is None
    assert pick_backend(["arcpy"], None) == "arcpy"


# ------------------------------------------------------- 四条路由路径（有/无）
def test_arcpy_recipe_runs_when_only_arcpy_available(tmp_path: Path):
    """arcpy 有 × gdal 无：[arcpy] 配方照常跑。"""
    recipe, runner = _recipe(), _Runner({"arcpy"})
    lib = _library(tmp_path, recipe, runner)

    obs = lib.run(recipe.id, {}, _ctx(tmp_path, runner))

    assert obs.ok, obs.summary
    assert obs.data["backend"] == "arcpy"
    assert runner.codes, "配方代码应该交给执行器"


def test_gdal_recipe_refused_when_only_arcpy_available(tmp_path: Path):
    """arcpy 有 × gdal 无：[gdal] 配方必须当场说"需 GDAL 后端"，不许丢给 arcpy 内核。"""
    recipe, runner = _recipe(backend=["gdal"]), _Runner({"arcpy"})
    lib = _library(tmp_path, recipe, runner)

    obs = lib.run(recipe.id, {}, _ctx(tmp_path, runner))

    assert not obs.ok
    assert obs.error["type"] == "BackendUnavailable"
    assert obs.error["required"] == ["gdal"]
    assert obs.error["available"] == ["arcpy"]
    assert "需 GDAL 后端" in obs.summary
    assert not runner.codes, "缺后端时一行代码都不该执行"


def test_gdal_recipe_runs_when_only_gdal_available(tmp_path: Path):
    """arcpy 无 × gdal 有：[gdal] 配方交给 GDAL 执行器。"""
    recipe, runner = _recipe(backend=["gdal"]), _Runner({"gdal"})
    lib = _library(tmp_path, recipe, runner)

    obs = lib.run(recipe.id, {}, _ctx(tmp_path, runner))

    assert obs.ok, obs.summary
    assert obs.data["backend"] == "gdal"
    assert runner.codes


def test_arcpy_recipe_refused_when_only_gdal_available(tmp_path: Path):
    """arcpy 无 × gdal 有：[arcpy] 配方必须说"需 ArcGIS"。"""
    recipe, runner = _recipe(backend=["arcpy"]), _Runner({"gdal"})
    lib = _library(tmp_path, recipe, runner)

    obs = lib.run(recipe.id, {}, _ctx(tmp_path, runner))

    assert not obs.ok
    assert obs.error["type"] == "BackendUnavailable"
    assert "需 ArcGIS" in obs.summary
    assert "arcpy" in obs.error["message"]
    assert not runner.codes


def test_default_backend_recipe_is_refused_on_gdal_only_runner(tmp_path: Path):
    """缺省即 arcpy 的行为锁死：没写 backend 的配方在纯 GDAL 机器上不可跑。

    如果哪天缺省被改成 gdal，这条会“跑成功”，测试立刻红。
    """
    recipe = Recipe(id="no_backend", name="没写 backend", code_template="print(1)\n")
    runner = _Runner({"gdal"})
    lib = _library(tmp_path, recipe, runner)

    obs = lib.run(recipe.id, {}, _ctx(tmp_path, runner))

    assert not obs.ok
    assert "需 ArcGIS" in obs.summary
    assert not runner.codes


def test_no_backend_at_all_reports_each_requirement(tmp_path: Path):
    """arcpy 无 × gdal 无：两个配方各自报自己缺什么，不糊成一句话。"""
    runner = _Runner(set())
    arcpy_recipe = _recipe("needs_arcpy", backend=["arcpy"])
    gdal_recipe = _recipe("needs_gdal", backend=["gdal"])

    arcpy_obs = _library(tmp_path, arcpy_recipe, runner).run(
        arcpy_recipe.id, {}, _ctx(tmp_path, runner)
    )
    gdal_obs = _library(tmp_path, gdal_recipe, runner).run(gdal_recipe.id, {}, _ctx(tmp_path, runner))

    assert "需 ArcGIS" in arcpy_obs.summary
    assert "需 GDAL 后端" in gdal_obs.summary
    assert not runner.codes


def test_multi_backend_recipe_picks_declared_order(tmp_path: Path):
    """[arcpy, gdal] / [gdal, arcpy]：按声明顺序挑本机能跑的那个。"""
    both = _Runner({"arcpy", "gdal"})
    gdal_first = _recipe("gdal_first", backend=["gdal", "arcpy"])
    arcpy_first = _recipe("arcpy_first", backend=["arcpy", "gdal"])

    obs_gdal_first = _library(tmp_path, gdal_first, both).run(
        gdal_first.id, {}, _ctx(tmp_path, both)
    )
    obs_arcpy_first = _library(tmp_path, arcpy_first, both).run(
        arcpy_first.id, {}, _ctx(tmp_path, both)
    )
    assert obs_gdal_first.data["backend"] == "gdal"
    assert obs_arcpy_first.data["backend"] == "arcpy"

    arcpy_only = _Runner({"arcpy"})
    obs_fallback = _library(tmp_path, gdal_first, arcpy_only).run(
        gdal_first.id, {}, _ctx(tmp_path, arcpy_only)
    )
    assert obs_fallback.ok, obs_fallback.summary
    assert obs_fallback.data["backend"] == "arcpy"


def test_undeclared_runner_is_not_gated(tmp_path: Path):
    """执行器没声明能力时不设门禁（历史行为，注入的假 runner 不受影响）。"""
    recipe, runner = _recipe(backend=["arcpy"]), _Runner()
    lib = _library(tmp_path, recipe, runner)

    obs = lib.run(recipe.id, {}, _ctx(tmp_path, runner))

    assert obs.ok, obs.summary
    assert obs.data["backend"] == "arcpy"


# ------------------------------------------------------- 模型侧看得到 backend
def test_backend_is_visible_in_recipe_listing(tmp_path: Path):
    runner = _Runner({"arcpy"})
    lib = _library(tmp_path, _recipe("gdal_probe", backend=["gdal"]), runner)

    hits = lib.search("gdal_probe")
    assert hits and hits[0]["backend"] == ["gdal"]

    catalog = lib.catalog_digest()
    line = next(item for item in catalog.splitlines() if "gdal_probe:" in item)
    assert "后端: gdal" in line
    # 本机没有 gdal 后端 → 目录里要能看出来跑不了
    assert "本机缺此后端" in line


def test_catalog_keeps_arcpy_lines_unchanged(tmp_path: Path):
    """arcpy 配方（缺省值）在目录里一个字都不加，否则 47 行会把 5200 字预算吃光。"""
    lib = _library(tmp_path, _recipe("plain"), _Runner({"arcpy"}))
    line = next(item for item in lib.catalog_digest().splitlines() if "plain:" in item)
    assert "后端" not in line

    # 执行器没声明能力 → 连"本机缺此后端"也不猜（多后端配方自己的“后端: …”
    # 标注是配方声明，与本机能不能跑无关）
    untyped = _library(tmp_path, _recipe("plain"), _Runner())
    untyped_line = next(item for item in untyped.catalog_digest().splitlines() if "plain:" in item)
    assert "后端" not in untyped_line
    assert "本机缺此后端" not in untyped.catalog_digest()


# ----------------------------------------------------------- 断言按后端分离
def _fake_gdal_check(lib, kind, target, expected, raw, result):
    return ValidationResult(kind, target, False, "gdal 实现（假）")


def test_gdal_assertion_impl_is_used_for_gdal_recipe(tmp_path: Path, monkeypatch):
    """同一 kind 允许按后端各一套实现：gdal 路由走 gdal 实现。"""
    monkeypatch.setitem(store_mod._BACKEND_CHECKS, ("gdal", "file_exists"), _fake_gdal_check)
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "x.txt").write_text("ok", encoding="utf-8")

    recipe = _recipe(backend=["gdal"], validation=[{"type": "file_exists", "path": "{{out}}"}])
    runner = _Runner({"gdal"})
    obs = _library(tmp_path, recipe, runner).run(recipe.id, {}, _ctx(tmp_path, runner))

    assert not obs.ok
    assert obs.data["assertions"][0]["detail"] == "gdal 实现（假）"


def test_arcpy_assertions_unchanged_when_gdal_impl_registered(tmp_path: Path, monkeypatch):
    """登记了 gdal 实现，arcpy 路由仍然走内置实现 —— 现有断言行为一行不变。"""
    monkeypatch.setitem(store_mod._BACKEND_CHECKS, ("gdal", "file_exists"), _fake_gdal_check)
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "x.txt").write_text("ok", encoding="utf-8")

    recipe = _recipe(backend=["arcpy"], validation=[{"type": "file_exists", "path": "{{out}}"}])
    runner = _Runner({"arcpy"})
    obs = _library(tmp_path, recipe, runner).run(recipe.id, {}, _ctx(tmp_path, runner))

    assert obs.ok, obs.summary
    assert obs.data["assertions"][0]["detail"] == ""

    # 公开入口同样能按后端挑实现
    raw = [{"type": "file_exists", "path": str(tmp_path / "output" / "x.txt")}]
    lib = RecipeLibrary()
    assert lib.run_assertions(raw)[0].ok is True
    assert lib.run_assertions(raw, backend="gdal")[0].detail == "gdal 实现（假）"


def test_unavailable_message_names_every_missing_backend():
    text = unavailable_message(["arcpy", "gdal"], set())
    assert "需 ArcGIS" in text and "需 GDAL 后端" in text


class _GdalInfoRunner:
    """假 GDAL 执行器：pyogrio 那句回固定 payload，别的代码不真跑。"""

    def __init__(self, payload: dict):
        self.backends = {"gdal"}
        self.payload = payload
        self.codes: list[str] = []

    def run(self, code, *, timeout=None, workspace=None):
        self.codes.append(code)
        return SimpleNamespace(
            ok=True, stdout="", stderr="", result=self.payload, error=None, display_images=[]
        )


def test_gdal_assertions_use_the_runner_that_ran_the_code(tmp_path: Path):
    """要素数必须问**执行了这段代码**的执行器（ctx 注入的那个），不是库自带的那个。

    两者不同时（库带 arcpy 执行器、ctx 注入 gdal 执行器），拿库里的执行器去数
    要素，得到的是另一个解释器环境里的答案。DAIR-68 非阻塞项 1。
    """
    recipe = _recipe(
        backend=["gdal"],
        validation=[{"type": "feature_count_equal", "path": "{{out}}", "value": 7}],
    )
    lib_runner = _Runner({"arcpy"})  # 库自带：只回 {'ok': True}，数不出要素
    ctx_runner = _GdalInfoRunner({"exists": True, "feature_count": 7})
    lib = RecipeLibrary(user_dir=tmp_path / "recipes", code_runner=lib_runner)
    lib.load()
    lib._recipes[recipe.id] = recipe

    obs = lib.run(recipe.id, {}, _ctx(tmp_path, ctx_runner))

    assert obs.ok, obs.summary
    assert obs.data["assertions"][0]["detail"] == "count=7 expected=7"
    assert any("pyogrio" in code for code in ctx_runner.codes), "要素数要在 gdal 执行器里数"
    assert not any("pyogrio" in code for code in lib_runner.codes), "不该拿库里的执行器去数"


# --------------------------------------------------------- GDAL 依赖必须可选
def test_gdal_dependency_is_optional():
    """GDAL 只能进 optional-dependencies，且必须是自带 libgdal 的 wheel。

    为什么不是 ``GDAL>=3.6``：那是 osgeo 的源码包，Windows 上没 wheel
    （实测 `pip download --only-binary :all: GDAL` → "from versions: none"），
    列它等于让 `pip install gis-task-cli[gdal]` 必然失败。能装的是 pyogrio /
    rasterio 这类把 libgdal 打进 wheel 的包（见 docs/p0-1b-env.md）。
    """
    import sys

    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - Python 3.10
        import tomli as tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    required = " ".join(project.get("dependencies") or []).lower()
    optional = project.get("optional-dependencies") or {}

    assert "gdal" in optional, "GDAL 必须声明成 optional extra（pip install gis-task-cli[gdal]）"
    extra = [dep.lower() for dep in optional["gdal"]]
    assert not any(dep.startswith("gdal") for dep in extra), (
        "extra 里别列 GDAL 源码包：Windows 没有 wheel，装它要求本机先有 libgdal/OSGeo4W"
    )
    assert any(dep.startswith("pyogrio") for dep in extra) and any(
        dep.startswith("rasterio") for dep in extra
    ), "GDAL extra 要列自带 libgdal 的 wheel（pyogrio + rasterio），否则装了也没有驱动"
    assert "gdal" not in required, "GDAL 不能进硬依赖：装不上就整包装不了"

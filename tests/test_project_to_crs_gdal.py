# -*- coding: utf-8 -*-
"""project_to_crs 的 GDAL 版真跑 + 双端逐点一致（坐标 / CRS / 轴序 / 容器 / 基准面变换）。

它是几乎所有链的**第一步**（`pm_task1_elderly_raster` 一上来就把 district 投影到 4548），
所以这一批真正要钉住的不是"投影能出数"，而是**两端投影出来的东西是同一份**。

**arcpy 与 pyproj 的基准面口径全部是 Pro 3.6 + PROJ 9.8.1 实测出来的**（探针见各用例）：

| 情形 | arcpy 实测（Pro 3.6） | pyproj 默认（PROJ 9.8.1） | 本配方 GDAL 版 |
|---|---|---|---|
| 同一基准面（4490→4548） | 只做投影 | 只做投影 | 同（逐点一致） |
| WGS84 ↔ CGCS2000 | 无变换（ListTransformations 空） | ballpark（无变换） | 同 |
| 北京54 → CGCS2000（4548/4490） | 无变换（ArcGIS 没有这对的变换） | ballpark（无变换） | 同（stdout 会说明） |
| 北京54 → WGS84（32650/4326） | **自动用 `Beijing_1954_To_WGS_1984_1`** | **ballpark（无变换）** | **不猜：报错要 `transformation`** |
| 显式给变换 | `_1` … `_6` | `Beijing 1954 to WGS 84 (1)` … `(6)` | 同一个操作（逐点差 <0.00003 m） |
| 投影算法 | 级数近似 | 精确 Krüger | 差 <1 mm（带内）~3.1 cm（离 CM 12°），相对 ~1e-8 |
| 越界坐标 | 写成**空几何**（数据静默丢失） | 照常给数学结果 | 不丢点（有意分歧） |

缺环境就 skip（与 ``test_attribute_join_table_gdal.py`` 同一套口径）：
``.venv-gdal`` 不在 → 所有真 GDAL 用例 skip；ArcGIS Pro 找不到 → 只有双端用例 skip。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gis_cli.recipes.schema import Recipe
from gis_cli.recipes.store import RecipeLibrary
from gis_cli.runner.gdal_runner import GdalCodeRunner, default_python_executable
from gis_cli.runtime.code_runner import CodeRunner

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "src/gis_cli/recipes/builtin/project_to_crs.yaml"

#: 真实数据（pm_task1_elderly_raster 的 district 面，WGS84，32 个街区）
DISTRICT = (
    ROOT
    / "workspace/input/13第十三届全国大学生GIS应用技能大赛（兰州交通大学）"
    / "数据/B下午/B下午/数据/district.shp"
)

#: 覆盖 3 个 3 度带的点：CM 105 / 111 / 117 / 123（含跨带边界 111、120 附近），
#: 外加北京城区的真实量级点（真实数据的 district 就在这一带）。
POINTS = [
    ("z35", 105.0, 35.0),
    ("z37", 111.0, 33.0),
    ("z39", 117.0, 36.0),
    ("z41", 123.0, 38.0),
    ("edge", 110.0, 30.0),
    ("bj", 116.4, 39.9),
]

#: 一个面（顶点数 > 3，覆盖"面要素逐顶点"而不只是点）
POLY_RING = [(116.0, 39.5), (117.0, 39.5), (117.0, 40.2), (116.0, 40.2), (116.0, 39.5)]

_MAKE_POINTS = "\n".join(
    [
        "import geopandas as gpd",
        "from shapely.geometry import Point",
        "gdf = gpd.GeoDataFrame({'pid': [r[0] for r in __ROWS__]},",
        "                       geometry=[Point(r[1], r[2]) for r in __ROWS__], crs=__CRS__)",
        "gdf.to_file(__PATH__, driver='GPKG', engine='pyogrio', layer='pts')",
        "set_result({'features': int(len(gdf))})",
    ]
)

_MAKE_POINTS_AND_ZONES = "\n".join(
    [
        "import geopandas as gpd",
        "from shapely.geometry import Point, Polygon",
        "pts = gpd.GeoDataFrame({'pid': [r[0] for r in __ROWS__]},",
        "                       geometry=[Point(r[1], r[2]) for r in __ROWS__], crs=__CRS__)",
        "pts.to_file(__PATH__, driver='GPKG', engine='pyogrio', layer='pts')",
        "zones = gpd.GeoDataFrame({'pid': ['zone']}, geometry=[Polygon(__RING__)], crs=__CRS__)",
        "zones.to_file(__PATH__, driver='GPKG', engine='pyogrio', layer='zones', mode='w')",
        "set_result({'features': int(len(pts))})",
    ]
)

_READ_GDAL = "\n".join(
    [
        "import geopandas as gpd",
        "import shapely",
        "g = gpd.read_file(__PATH__, layer=__LAYER__, engine='pyogrio')",
        "c = shapely.get_coordinates(g.geometry.values)",
        "set_result({'n': int(len(g)), 'epsg': g.crs.to_epsg(), 'crs': g.crs.name,",
        "            'coords': [[float(x), float(y)] for x, y in c]})",
    ]
)

_READ_ARCPY = "\n".join(
    [
        "import arcpy",
        "def _verts(geom):",
        "    if geom is None:",
        "        return []",
        "    if geom.type == 'point':",
        "        p = geom.firstPoint",
        "        return [[float(p.X), float(p.Y)]]",
        "    out = []",
        "    for i in range(geom.partCount):",
        "        for p in geom.getPart(i):",
        "            if p is not None:",
        "                out.append([float(p.X), float(p.Y)])",
        "    return out",
        "coords = []",
        "with arcpy.da.SearchCursor(__PATH__, ['SHAPE@']) as cur:",
        "    for (geom,) in cur:",
        "        coords.extend(_verts(geom))",
        "sr = arcpy.Describe(__PATH__).spatialReference",
        "set_result({'n': len(coords), 'epsg': sr.factoryCode, 'crs': sr.name, 'coords': coords})",
    ]
)

#: pyproj 的默认变换（不给 transformation 时 PROJ 会怎么走）——用来证明"两端默认不同"
_PROJ_DEFAULT = "\n".join(
    [
        "from pyproj import CRS, Transformer",
        "t = Transformer.from_crs(CRS.from_epsg(__SRC__), CRS.from_epsg(__DST__), always_xy=True)",
        "set_result({'coords': [list(map(float, t.transform(r[1], r[2]))) for r in __ROWS__]})",
    ]
)


# --------------------------------------------------------------------- 夹具
@pytest.fixture()
def gdal_runner(tmp_path: Path) -> GdalCodeRunner:
    interpreter = default_python_executable()
    if interpreter is None or not Path(interpreter).is_file():
        pytest.skip("未找到 .venv-gdal 解释器（见 docs/p0-1b-env.md），跳过真 GDAL 用例")
    return GdalCodeRunner(workspace=tmp_path, python_executable=interpreter)


@pytest.fixture()
def arcpy_runner(tmp_path: Path):
    from gis_cli.arcpy_bridge import ArcGISDiscoveryError, discover_arcgis_pro_python

    try:
        info = discover_arcgis_pro_python()
    except ArcGISDiscoveryError as exc:
        pytest.skip(f"本机没有 ArcGIS Pro（{exc}），跳过双端一致用例")
    runner = CodeRunner(
        workspace=tmp_path, python_executable=info.python_executable, exec_timeout=600
    )
    try:
        yield runner
    finally:
        runner.shutdown()


def _run(runner, tmp_path: Path, **params):
    recipe_lib = RecipeLibrary(code_runner=runner)
    recipe_lib.load()
    base = {
        "input_path": str(tmp_path / "pts.gpkg") + "/pts",
        "output_path": str(tmp_path / "out.gpkg"),
        "target_srid": 4548,
        "overwrite": True,
        "transformation": "",
    }
    base.update(params)
    ctx = SimpleNamespace(workspace=tmp_path, code_runner=runner, exec_timeout=600)
    return recipe_lib.run("project_to_crs", base, ctx)


def _make_points(runner, tmp_path: Path, *, srid: int = 4490, name: str = "pts.gpkg"):
    path = tmp_path / name
    outcome = runner.run(
        _MAKE_POINTS.replace("__ROWS__", repr(POINTS))
        .replace("__CRS__", repr(f"EPSG:{srid}"))
        .replace("__PATH__", repr(str(path))),
        timeout=180,
    )
    assert outcome.ok, outcome.error
    return path


def _make_two_layers(runner, tmp_path: Path):
    path = tmp_path / "multi.gpkg"
    outcome = runner.run(
        _MAKE_POINTS_AND_ZONES.replace("__ROWS__", repr(POINTS))
        .replace("__CRS__", repr("EPSG:4490"))
        .replace("__RING__", repr(POLY_RING))
        .replace("__PATH__", repr(str(path))),
        timeout=180,
    )
    assert outcome.ok, outcome.error
    return path


def _read_gdal(runner, path: str) -> dict:
    layer = None if str(path).lower().endswith(".shp") else Path(path).stem
    outcome = runner.run(
        _READ_GDAL.replace("__PATH__", repr(str(path))).replace("__LAYER__", repr(layer)),
        timeout=180,
    )
    assert outcome.ok, outcome.error
    return outcome.result


def _read_arcpy(runner, path: str) -> dict:
    outcome = runner.run(_READ_ARCPY.replace("__PATH__", repr(str(path))), timeout=300)
    assert outcome.ok, outcome.error
    return outcome.result


def _relative_worst(actual: list[list[float]], expected: list[list[float]]) -> float:
    """逐顶点最大相对差（以坐标量级为分母，下限 1.0：接近 0 的坐标不能把绝对差当相对差）。"""
    assert len(actual) == len(expected), f"顶点数不同: {len(actual)} != {len(expected)}"
    worst = 0.0
    for (ax, ay), (ex, ey) in zip(actual, expected):
        for a, e in ((ax, ex), (ay, ey)):
            worst = max(worst, abs(a - e) / max(abs(e), 1.0))
    return worst


def _assert_same(actual: list[list[float]], expected: list[list[float]], label: str) -> None:
    """P0-1c 口径：相对差 ≤1e-6（浮点，不卡位级）。"""
    worst = _relative_worst(actual, expected)
    assert worst <= 1e-6, f"{label}: 最大相对差 {worst:.3e} 超过 1e-6"


def _metres_between(a: list[list[float]], b: list[list[float]]) -> float:
    """两组**投影坐标**（米）逐点距离的最大值。"""
    worst = 0.0
    for (ax, ay), (bx, by) in zip(a, b):
        worst = max(worst, ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)
    return worst


# ------------------------------------------------------------- 模板本身
def test_recipe_declares_two_backends_and_renders_both():
    recipe = Recipe.from_yaml(RECIPE)
    assert recipe.backend == ["arcpy", "gdal"]
    params = {
        "input_path": "in/district.shp",
        "output_path": "out/district_prj.gpkg",
        "target_srid": 4548,
        "overwrite": True,
        "transformation": "Beijing 1954 to WGS 84 (1)",
    }
    arcpy_code = recipe.render(params)
    gdal_code = recipe.render(params, backend="gdal")

    # arcpy 版仍然走 Project，容器解析走共享片段（不是复制一份判断逻辑）
    assert "arcpy.management.Project" in arcpy_code
    assert "def resolve_zone_layer" in arcpy_code, "容器输入必须复用 fragments.py 的片段"
    assert "import geopandas" not in arcpy_code
    # 空 transformation 必须原样传给 Project（Pro 3.6 实测：空串与不传第 4 参等价，
    # 自动选变换的行为不变）；非空时同样透传
    assert "arcpy.management.Project(input_path, output_path, target_sr, transformation)" in arcpy_code
    # overwriteOutput=True 下 Project 会静默覆盖，overwrite=false 必须自己拦（见 DAIR-131）
    assert "if not overwrite:" in arcpy_code, "overwrite=false 必须拦在 Project 之前"

    assert "from pyproj import CRS" in gdal_code and "TransformerGroup" in gdal_code
    assert "import arcpy" not in gdal_code
    assert '".gpkg": "GPKG"' in gdal_code, "输出格式白名单必须写死在代码里"
    assert "always_xy=True" in gdal_code, "轴序口径必须显式钉住"
    for key in ("in/district.shp", "out/district_prj.gpkg", "Beijing 1954 to WGS 84 (1)"):
        assert key in arcpy_code and key in gdal_code


# ------------------------------------------------- 双端一致：投影 + 轴序
def test_gdal_projection_matches_arcpy_across_three_bands(gdal_runner, arcpy_runner, tmp_path: Path):
    """4490 → 4548：跨 3 个 3 度带的点 + 一个面，逐顶点相对差 ≤1e-6。

    同一基准面，两端都只做投影；实测最大绝对差 3.07 cm（lon=105，离 CM 117 有 12°，
    是 PROJ 精确 Krüger 与 ArcGIS 级数近似的差别），相对差 5.1e-8。
    """
    src = _make_points(gdal_runner, tmp_path)

    gdal_obs = _run(gdal_runner, tmp_path, input_path=str(src) + "/pts", output_path=str(tmp_path / "gdal.gpkg"))
    arcpy_obs = _run(arcpy_runner, tmp_path, input_path=str(src) + "/pts", output_path=str(tmp_path / "arcpy.shp"))

    assert gdal_obs.ok, gdal_obs.summary
    assert arcpy_obs.ok, arcpy_obs.summary
    assert gdal_obs.data["backend"] == "gdal"
    assert gdal_obs.data["result"]["feature_count"] == len(POINTS)
    assert gdal_obs.data["result"]["target_wkid"] == 4548
    # 同一基准面：GDAL 侧没有做任何基准面变换
    assert gdal_obs.data["result"]["transformation"] == ""

    got = _read_gdal(gdal_runner, gdal_obs.data["result"]["output"])
    want = _read_arcpy(arcpy_runner, arcpy_obs.data["result"]["output"])
    _assert_same(got["coords"], want["coords"], "4490->4548 逐顶点")
    assert len(got["coords"]) == len(want["coords"]) == len(POINTS)
    # CRS 元数据：两端都认成 4548（名字是各自后端的写法，不参与比对）
    assert got["epsg"] == want["epsg"] == 4548


def test_geographic_target_keeps_axis_order(gdal_runner, arcpy_runner, tmp_path: Path):
    """4548 → 4490（目标轴序是经纬度）：两端符号与量级一致，x 是经度、y 是纬度。

    轴序踩过的坑：pyproj 默认按 CRS 的轴序（EPSG:4490 是纬度在前），
    本配方全程 ``always_xy=True``，才与 arcpy 的 lon/lat 口径对齐。
    """
    src = _make_points(gdal_runner, tmp_path)

    gdal_obs = _run(
        gdal_runner, tmp_path, input_path=str(src) + "/pts", output_path=str(tmp_path / "geo.gpkg"),
        target_srid=4490,
    )
    arcpy_obs = _run(
        arcpy_runner, tmp_path, input_path=str(src) + "/pts", output_path=str(tmp_path / "geo.shp"),
        target_srid=4490,
    )
    assert gdal_obs.ok, gdal_obs.summary
    assert arcpy_obs.ok, arcpy_obs.summary

    got = _read_gdal(gdal_runner, gdal_obs.data["result"]["output"])
    want = _read_arcpy(arcpy_runner, arcpy_obs.data["result"]["output"])
    _assert_same(got["coords"], want["coords"], "4548->4490 逐顶点")
    assert got["epsg"] == want["epsg"] == 4490

    # 首点 lon=105 / lat=35：x 必须是经度（量级 100），y 是纬度（量级 10）
    x, y = got["coords"][0]
    assert 104.9 < x < 105.1 and 34.9 < y < 35.1, f"轴序不对: {(x, y)}"


# ----------------------------------------------------- 基准面变换：不猜
def test_gdal_refuses_to_guess_datum_transformation(gdal_runner, arcpy_runner, tmp_path: Path):
    """北京54 → 32650（WGS84 UTM）：两端默认不同，GDAL 侧必须报错而不是静默无变换。

    实测（Pro 3.6 / PROJ 9.8.1）：
    - arcpy 不给 transformation 时**自动**用了 ``Beijing_1954_To_WGS_1984_1``；
    - pyproj 的默认是 ballpark（= 无变换），与 arcpy 的默认差 42~74 m；
    - 所以 GDAL 侧既不猜 arcpy 选哪个、也不静默用 PROJ 的默认，直接报错列候选。
    """
    src = _make_points(gdal_runner, tmp_path, srid=4214, name="b54.gpkg")

    gdal_obs = _run(
        gdal_runner, tmp_path, input_path=str(src) + "/pts", output_path=str(tmp_path / "b54.gpkg"),
        target_srid=32650,
    )
    arcpy_obs = _run(
        arcpy_runner, tmp_path, input_path=str(src) + "/pts", output_path=str(tmp_path / "b54.shp"),
        target_srid=32650,
    )
    assert arcpy_obs.ok, arcpy_obs.summary
    assert not gdal_obs.ok, "GDAL 侧不许静默按无变换跑"
    message = str(gdal_obs.error["message"])
    assert "Beijing 1954 to WGS 84 (1)" in message, message
    assert "transformation" in message, message

    # PROJ 的默认（ballpark）与 arcpy 的自动选择差多少：证明"不报错就会错几十米"
    proj_default = gdal_runner.run(
        _PROJ_DEFAULT.replace("__SRC__", "4214").replace("__DST__", "32650").replace("__ROWS__", repr(POINTS)),
        timeout=180,
    )
    assert proj_default.ok, proj_default.error
    want = _read_arcpy(arcpy_runner, arcpy_obs.data["result"]["output"])
    gap = _metres_between(proj_default.result["coords"], want["coords"])
    assert gap > 10.0, f"PROJ 默认与 arcpy 默认只差 {gap:.3f} m，与探针结论（42~74 m）不符"


def test_explicit_transformation_is_the_same_operation_on_both_ends(gdal_runner, arcpy_runner, tmp_path: Path):
    """给了 transformation：两端必须用同一个变换，逐顶点相对差 ≤1e-6。

    ESRI 的 ``Beijing_1954_To_WGS_1984_1`` 与 PROJ 的 ``Beijing 1954 to WGS 84 (1)``
    是同一个操作（实测逐点差 <0.00003 m），所以两端各写各的名字，结果必须一致；
    GDAL 侧写 ESRI 名也认（归一化后同一串词）。
    """
    src = _make_points(gdal_runner, tmp_path, srid=4214, name="b54.gpkg")

    gdal_obs = _run(
        gdal_runner, tmp_path, input_path=str(src) + "/pts", output_path=str(tmp_path / "tf.gpkg"),
        target_srid=32650, transformation="Beijing 1954 to WGS 84 (1)",
    )
    arcpy_obs = _run(
        arcpy_runner, tmp_path, input_path=str(src) + "/pts", output_path=str(tmp_path / "tf.shp"),
        target_srid=32650, transformation="Beijing_1954_To_WGS_1984_1",
    )
    assert gdal_obs.ok, gdal_obs.summary
    assert arcpy_obs.ok, arcpy_obs.summary
    assert gdal_obs.data["result"]["transformation"] == "Beijing 1954 to WGS 84 (1)"

    got = _read_gdal(gdal_runner, gdal_obs.data["result"]["output"])
    want = _read_arcpy(arcpy_runner, arcpy_obs.data["result"]["output"])
    _assert_same(got["coords"], want["coords"], "4214->32650 显式 (1) 逐顶点")

    # GDAL 侧也认 ESRI 名：同一条变换、同一份坐标
    alias_obs = _run(
        gdal_runner, tmp_path, input_path=str(src) + "/pts", output_path=str(tmp_path / "alias.gpkg"),
        target_srid=32650, transformation="Beijing_1954_To_WGS_1984_1",
    )
    assert alias_obs.ok, alias_obs.summary
    assert alias_obs.data["result"]["transformation"] == "Beijing 1954 to WGS 84 (1)"
    alias = _read_gdal(gdal_runner, alias_obs.data["result"]["output"])
    assert alias["coords"] == got["coords"]

    # 反例：写一个这一对坐标系里不存在的变换名，要报错并列出候选（不能静默忽略）
    bad = _run(
        gdal_runner, tmp_path, input_path=str(src) + "/pts", output_path=str(tmp_path / "bad.gpkg"),
        target_srid=32650, transformation="Beijing_1954_To_WGS_1984_9",
    )
    assert not bad.ok
    assert "不在这一对坐标系的可用变换里" in str(bad.error["message"])


def test_datum_pair_without_any_transformation_runs_on_both_ends(gdal_runner, arcpy_runner, tmp_path: Path):
    """北京54 → 4548：ArcGIS 也没有北京54→CGCS2000 的变换，两端都按无变换执行。

    这条是"不猜"的另一半：**没有**候选时不该报错（否则这条链第一步就断了），
    但要在 stdout 里说清"按无变换执行"，并给出与 arcpy 逐点一致的结果。
    """
    src = _make_points(gdal_runner, tmp_path, srid=4214, name="b54.gpkg")

    gdal_obs = _run(
        gdal_runner, tmp_path, input_path=str(src) + "/pts", output_path=str(tmp_path / "null.gpkg"),
        target_srid=4548,
    )
    arcpy_obs = _run(
        arcpy_runner, tmp_path, input_path=str(src) + "/pts", output_path=str(tmp_path / "null.shp"),
        target_srid=4548,
    )
    assert gdal_obs.ok, gdal_obs.summary
    assert arcpy_obs.ok, arcpy_obs.summary
    assert gdal_obs.data["result"]["transformation"] == ""
    assert "没有可用的基准面变换" in gdal_obs.data["stdout"]

    got = _read_gdal(gdal_runner, gdal_obs.data["result"]["output"])
    want = _read_arcpy(arcpy_runner, arcpy_obs.data["result"]["output"])
    _assert_same(got["coords"], want["coords"], "4214->4548 无变换逐顶点")


# ------------------------------------------------------------- 容器输入
def test_container_input_single_layer_is_resolved(gdal_runner, arcpy_runner, tmp_path: Path):
    """.gpkg 单层容器（路径里没写图层名）：两端都自动取用其中唯一的图层。"""
    src = _make_points(gdal_runner, tmp_path)

    gdal_obs = _run(gdal_runner, tmp_path, input_path=str(src), output_path=str(tmp_path / "c.gpkg"))
    arcpy_obs = _run(arcpy_runner, tmp_path, input_path=str(src), output_path=str(tmp_path / "c.shp"))
    assert gdal_obs.ok, gdal_obs.summary
    assert arcpy_obs.ok, arcpy_obs.summary
    assert gdal_obs.data["result"]["feature_count"] == len(POINTS)
    assert "自动使用其中唯一的图层" in gdal_obs.data["stdout"]


def test_container_input_multi_layer_asks_for_a_layer(gdal_runner, arcpy_runner, tmp_path: Path):
    """多层容器：两端都报人话错误（列出图层名），不替用户挑。"""
    src = _make_two_layers(gdal_runner, tmp_path)

    gdal_obs = _run(gdal_runner, tmp_path, input_path=str(src), output_path=str(tmp_path / "m.gpkg"))
    arcpy_obs = _run(arcpy_runner, tmp_path, input_path=str(src), output_path=str(tmp_path / "m.shp"))
    for obs, backend in ((gdal_obs, "GDAL"), (arcpy_obs, "arcpy")):
        assert not obs.ok, f"{backend} 不该替用户在多层容器里挑图层"
        message = str(obs.error["message"])
        assert "多层容器" in message and "pts" in message and "zones" in message, message


def test_container_input_shapefile_folder_is_resolved(gdal_runner, arcpy_runner, tmp_path: Path):
    """shp 文件夹（容器）：两端都自动取用其中唯一的 .shp。"""
    src = _make_points(gdal_runner, tmp_path)
    folder = tmp_path / "shps"
    folder.mkdir()
    # 先用 GDAL 写一份 .shp 进文件夹（只有它一个图层）
    made = gdal_runner.run(
        "\n".join(
            [
                "import geopandas as gpd",
                "g = gpd.read_file(__SRC__, layer='pts', engine='pyogrio')",
                "g.to_file(__DST__, driver='ESRI Shapefile', engine='pyogrio')",
                "set_result({'ok': True})",
            ]
        )
        .replace("__SRC__", repr(str(src)))
        .replace("__DST__", repr(str(folder / "only.shp"))),
        timeout=180,
    )
    assert made.ok, made.error

    gdal_obs = _run(gdal_runner, tmp_path, input_path=str(folder), output_path=str(tmp_path / "f.gpkg"))
    arcpy_obs = _run(arcpy_runner, tmp_path, input_path=str(folder), output_path=str(tmp_path / "f.shp"))
    assert gdal_obs.ok, gdal_obs.summary
    assert arcpy_obs.ok, arcpy_obs.summary
    assert gdal_obs.data["result"]["feature_count"] == len(POINTS)


# ------------------------------------------------------- 输出格式 / 覆盖
def test_gdb_output_is_rejected_with_a_reason(gdal_runner, tmp_path: Path):
    """GDAL 没有 FileGDB 写驱动：.gdb 输出要明确拒绝，不许静默换成别的格式。"""
    obs = _run(
        gdal_runner, tmp_path, input_path=str(tmp_path / "pts.gpkg") + "/pts",
        output_path=str(tmp_path / "out.gdb" / "district_prj"),
    )
    assert not obs.ok
    message = str(obs.error["message"])
    assert ".gdb" in message and ".gpkg" in message and "arcpy" in message, message


def test_overwrite_false_refuses_existing_output(gdal_runner, tmp_path: Path):
    """overwrite=false 且输出已存在：报错而不是覆盖（与 arcpy 的行为对齐）。"""
    src = _make_points(gdal_runner, tmp_path)
    first = _run(gdal_runner, tmp_path, input_path=str(src) + "/pts", output_path=str(tmp_path / "o.gpkg"))
    assert first.ok, first.summary

    second = _run(
        gdal_runner, tmp_path, input_path=str(src) + "/pts", output_path=str(tmp_path / "o.gpkg"),
        overwrite=False,
    )
    assert not second.ok
    assert "overwrite=false" in str(second.error["message"])


def test_overwrite_false_refuses_existing_output_arcpy(gdal_runner, arcpy_runner, tmp_path: Path):
    """arcpy 端同款：`overwrite=false` 必须拦在写之前，不能让 Project 静默覆盖。

    `Project` 在 `arcpy.env.overwriteOutput = True` 下会直接覆盖已有输出（DAIR-131 实测：
    先 4508 跑一次，再同一 output_path 用 4548 + overwrite=false 重跑，返回 ok=True 且产物
    wkid 变成 4548）。所以两次跑用**不同 target_srid**，覆盖成功会直接体现在产物上。
    """
    src = _make_points(gdal_runner, tmp_path)
    out = str(tmp_path / "o.shp")
    first = _run(arcpy_runner, tmp_path, input_path=str(src) + "/pts", output_path=out, target_srid=4508)
    assert first.ok, first.summary

    second = _run(
        arcpy_runner, tmp_path, input_path=str(src) + "/pts", output_path=out,
        target_srid=4548, overwrite=False,
    )
    assert not second.ok, "overwrite=false 不该成功（会被 Project 静默覆盖）"
    assert "overwrite=false" in str(second.error["message"])
    assert _read_arcpy(arcpy_runner, out)["epsg"] == 4508, "已有产物被覆盖了"


# ------------------------------------------------------------- 越界坐标
def test_out_of_range_points_are_kept(gdal_runner, tmp_path: Path):
    """越界坐标：GDAL 侧不丢点（arcpy 实测会把落在目标范围外的点写成空几何）。

    有意分歧，写进 pitfalls：arcpy 的 Project 对 (0,0)/(179.9,89) 这类投影后落在
    目标 CRS 有效范围外的顶点给 null；pyproj 照常给数学结果。本配方选择"不静默丢点"。
    """
    rows = [("zero", 0.0, 0.0), ("far", 179.9, 89.0), ("ok", 116.4, 39.9)]
    src = tmp_path / "bad.gpkg"
    made = gdal_runner.run(
        _MAKE_POINTS.replace("__ROWS__", repr(rows)).replace("__CRS__", repr("EPSG:4326"))
        .replace("__PATH__", repr(str(src))),
        timeout=180,
    )
    assert made.ok, made.error

    obs = _run(gdal_runner, tmp_path, input_path=str(src) + "/pts", output_path=str(tmp_path / "bad_out.gpkg"))
    assert obs.ok, obs.summary
    got = _read_gdal(gdal_runner, obs.data["result"]["output"])
    assert got["n"] == 3, "越界点不能被静默丢掉"
    for x, y in got["coords"]:
        assert x == x and y == y and abs(x) < 1e9 and abs(y) < 1e9


# ------------------------------------------------------------- 真实数据
@pytest.mark.skipif(not DISTRICT.exists(), reason="本机没有 B下午 的 district.shp（基准数据未随仓库分发）")
def test_real_district_to_4548_matches_arcpy(gdal_runner, arcpy_runner, tmp_path: Path):
    """真实数据（pm_task1_elderly_raster 的 district 面，WGS84）→ 4548，双端逐顶点一致。

    只读输入，产物写 tmp_path。实测：32 个街区、8229 个顶点，最大绝对差 2.8e-9 m、
    最大相对差 6.3e-16——数据在 116.3~116.4E，离 CM 117 不到 1°，PROJ 与 ArcGIS 的
    级数差在这一小段里只有纳米级（跨带的合成点才差到 3 cm，见上一条用例）。
    """
    gdal_obs = _run(
        gdal_runner, tmp_path, input_path=str(DISTRICT), output_path=str(tmp_path / "district.gpkg")
    )
    arcpy_obs = _run(
        arcpy_runner, tmp_path, input_path=str(DISTRICT), output_path=str(tmp_path / "district.shp")
    )
    assert gdal_obs.ok, gdal_obs.summary
    assert arcpy_obs.ok, arcpy_obs.summary
    assert gdal_obs.data["result"]["feature_count"] == arcpy_obs.data["result"]["feature_count"] == 32
    assert gdal_obs.data["result"]["target_wkid"] == 4548

    got = _read_gdal(gdal_runner, gdal_obs.data["result"]["output"])
    want = _read_arcpy(arcpy_runner, arcpy_obs.data["result"]["output"])
    assert got["epsg"] == want["epsg"] == 4548
    # 顶点级比对（不是要素级）：32 个面的 8229 个顶点都要逐个对得上
    assert len(got["coords"]) == len(want["coords"]) > 1000
    _assert_same(got["coords"], want["coords"], "真实 district 逐顶点")


# ------------------------------------------------- crs_wkid 断言（GDAL 口径）
class _FakeRunner:
    """只回一个固定 payload 的假执行器：验断言读的是不是 GDAL 那条路。"""

    def __init__(self, payload: dict):
        self.payload = payload
        self.codes: list[str] = []

    def run(self, code, *, timeout=None, workspace=None):
        self.codes.append(code)
        return SimpleNamespace(ok=True, stdout="", result=self.payload, error=None)


def test_crs_wkid_assertion_reads_the_layer_crs_with_pyogrio():
    """``crs_wkid`` 必须由 GDAL 侧读图层 CRS。

    为什么：内置实现走 ``arcpy.Describe(path)``，而 GDAL 后端的产物是 ``x.gpkg`` 这个
    **容器路径**——Describe 只看容器、spatialReference 是 None，断言必挂（DAIR-129 实测：
    产物明明写对了 4548，断言仍报 wkid=None）。这里顺带锁住第二层置信度：ESRI 写法的
    WKT 常常只能靠 ``to_epsg(min_confidence=20)`` 认出来。
    """
    from gis_cli.recipes.gdal_checks import gdal_vector_check

    runner = _FakeRunner({"exists": True, "wkid": None, "low": 4548, "name": "CGCS2000_3_Degree_GK_CM_117E"})
    ok, detail = gdal_vector_check(runner, "crs_wkid", "out.gpkg", "4548")
    assert ok is True and "low=4548" in detail
    assert "pyogrio" in runner.codes[0] and "arcpy" not in runner.codes[0]

    # 码对不上就是失败（不是"有 CRS 就放行"）
    ok, _ = gdal_vector_check(runner, "crs_wkid", "out.gpkg", 32650)
    assert ok is False

    # 两层都认不出 → 失败并回报两个值，方便定位
    ok, detail = gdal_vector_check(_FakeRunner({"exists": True, "wkid": None, "low": None}), "crs_wkid", "o.gpkg", 4548)
    assert ok is False and "wkid=None" in detail and "low=None" in detail

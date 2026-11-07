# -*- coding: utf-8 -*-
"""point_buffer_raster_sum 的 GDAL 版真跑 + 双端一致 + 探针锁住的口径。

测试数据（EPSG:32650，米）：

    栅格  100 m 像元，30 列 × 10 行，像元值 = 行号*100 + 列号（行/列都不重复：
          一次 SUM 就能反推出「哪些像元进了圆」），原点左上 (0, 1000)，nodata = -9999
    点    A(500,500) / B(1500,500) / C(550,520)：A 与 C 的 500 m 圆几乎完全重叠

``REFERENCE`` 是**真 arcpy 跑出来的**（ArcGIS Pro 3.6，配方原样执行，见 DAIR-207 探针），
不是从 GDAL 侧抄回来的。它同时钉住几件事：

- 数组朝向：C 的 top/bottom 两种算法给 33680 / 35480，实测 33680 → 行 0 是顶（GDAL 侧一致）；
- 重叠缓冲：A 单独跑与 A+C 一起跑，A 那行逐值相同（每个点各算各的圆，重叠像元各点各记一次）；
- nodata：圆内 4 个 nodata 像元，nodata_as_zero 两档都是 34546.0 / 80（0 与 NaN 对 nansum 等价，
  CELL_COUNT 两档都把 nodata 像元算进去）；
- 空结果：点完全在栅格外仍出一行 0.0 / 0；
- 非点要素：arcpy 的 SHAPE@XY 对面要素返回几何中心（面输入与中心点同值）。

缺环境就 skip（与 ``test_zonal_statistics_gdal.py`` 同一套口径）：
``.venv-gdal`` 不在 → 所有真 GDAL 用例 skip；ArcGIS Pro 找不到 → 只有双端用例 skip。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from gis_cli.recipes.fragments import FRAGMENTS, fragment_names
from gis_cli.recipes.schema import Recipe
from gis_cli.recipes.store import RecipeLibrary
from gis_cli.runner.gdal_runner import GdalCodeRunner, default_python_executable
from gis_cli.runtime.code_runner import CodeRunner

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "src/gis_cli/recipes/builtin/point_buffer_raster_sum.yaml"

#: arcpy（Pro 3.6）实测：pid -> (SUM_VALUE, CELL_COUNT)，三点一起跑
REFERENCE = {1: (36360.0, 80), 2: (37160.0, 80), 3: (33680.0, 76)}
REFERENCE_TOTAL = 107200.0
#: arcpy 实测：圆内 4 个 nodata 像元，两档相同
NODATA_REFERENCE = (34546.0, 80)

_MAKE_INPUTS = "\n".join(
    [
        "import os",
        "import geopandas as gpd",
        "import numpy as np",
        "import pyproj",
        "import rasterio",
        "from rasterio.transform import from_origin",
        "from shapely.geometry import Point, Polygon",
        "",
        "_work = __WORK__",
        "_crs = 'EPSG:32650'",
        "_rows, _cols = 10, 30",
        "_base = np.array([[r * 100 + c for c in range(_cols)] for r in range(_rows)], dtype='float64')",
        "",
        "def _raster(name, data, crs='EPSG:32650', count=1):",
        "    path = os.path.join(_work, name)",
        "    with rasterio.open(path, 'w', driver='GTiff', height=_rows, width=_cols, count=count,",
        "                       dtype='float32', crs=crs, transform=from_origin(0, _rows * 100, 100, 100),",
        "                       nodata=-9999.0) as ds:",
        "        for band in range(count):",
        "            ds.write(data.astype('float32'), band + 1)",
        "    return path",
        "",
        "def _points(name, rows, layer='pts', crs='EPSG:32650'):",
        "    path = os.path.join(_work, name)",
        "    gdf = gpd.GeoDataFrame({'pid': [r[0] for r in rows], 'label': [r[1] for r in rows]},",
        "                           geometry=[Point(r[2], r[3]) for r in rows], crs=crs)",
        "    gdf.to_file(path, driver='GPKG', engine='pyogrio', layer=layer)",
        "    return path",
        "",
        "out = {'work': _work}",
        "out['raster'] = _raster('dem100.tif', _base)",
        "_nodata = _base.copy()",
        "_nodata[4:6, 3:5] = -9999.0",
        "out['raster_nodata'] = _raster('dem100_nodata.tif', _nodata)",
        "out['raster_multiband'] = _raster('dem100_multiband.tif', _base, count=2)",
        "out['raster_shift'] = _raster('dem100_shift.tif', _base + 1000.0)",
        "out['raster_4326'] = _raster('dem100_4326.tif', _base, crs='EPSG:4326')",
        "out['pts3'] = _points('pts3.gpkg', [(1, 'A', 500.0, 500.0), (2, 'B', 1500.0, 500.0),",
        "                                    (3, 'C', 550.0, 520.0)])",
        "out['pts_a_only'] = _points('pts_a_only.gpkg', [(1, 'A', 500.0, 500.0)])",
        "out['pts_c_only'] = _points('pts_c_only.gpkg', [(3, 'C', 550.0, 520.0)])",
        "out['pts_far'] = _points('pts_far.gpkg', [(9, 'FAR', -5000.0, 500.0)])",
        "out['pts_4326'] = _points('pts_4326.gpkg', [(1, 'A', 500.0, 500.0)], crs='EPSG:4326')",
        "out['pts_textid'] = _points('pts_textid.gpkg', [(1, 'A', 500.0, 500.0)])",
        "_text = gpd.read_file(out['pts_textid'], layer='pts')",
        "_text['pid'] = ['A001']",
        "_text.to_file(out['pts_textid'], driver='GPKG', engine='pyogrio', layer='pts')",
        "_null = gpd.GeoDataFrame({'pid': [1, 2]}, geometry=[Point(500, 500), None], crs='EPSG:32650')",
        "out['pts_null'] = os.path.join(_work, 'pts_null.gpkg')",
        "_null.to_file(out['pts_null'], driver='GPKG', engine='pyogrio', layer='pts')",
        "_poly = gpd.GeoDataFrame({'pid': [1]},",
        "    geometry=[Polygon([(400, 400), (600, 400), (600, 600), (400, 600)])], crs='EPSG:32650')",
        "out['poly'] = os.path.join(_work, 'poly.gpkg')",
        "_poly.to_file(out['poly'], driver='GPKG', engine='pyogrio', layer='poly')",
        # 同 EPSG、ESRI 写法 WKT 的点（坐标系比较不能是字符串比）
        "_alias = pyproj.CRS.from_wkt(pyproj.CRS('EPSG:32650').to_wkt(version='WKT1_ESRI'))",
        "out['pts_esri'] = os.path.join(_work, 'pts_esri.gpkg')",
        "gpd.GeoDataFrame({'pid': [1]}, geometry=[Point(500.0, 500.0)],",
        "    crs=_alias).to_file(out['pts_esri'], driver='GPKG', engine='pyogrio', layer='pts')",
        "out['pts_esri_wkt'] = _alias.to_wkt()",
        "out['pts_esri_epsg'] = _alias.to_epsg()",
        # 容器：shp 文件夹 / 多层 .gpkg
        "os.makedirs(os.path.join(_work, 'shpdir'), exist_ok=True)",
        "out['shpdir'] = os.path.join(_work, 'shpdir')",
        "gpd.GeoDataFrame({'pid': [1, 2]}, geometry=[Point(500, 500), Point(1500, 500)],",
        "    crs=_crs).to_file(os.path.join(out['shpdir'], 'pts.shp'), engine='pyogrio')",
        "out['multi'] = _points('multi.gpkg', [(1, 'A', 500.0, 500.0)], layer='layer_one')",
        "_second = gpd.GeoDataFrame({'pid': [2]}, geometry=[Point(1500.0, 500.0)], crs=_crs)",
        "_second.to_file(out['multi'], driver='GPKG', engine='pyogrio', layer='layer_two')",
        "out['gdb_out'] = os.path.join(_work, 'target.gdb', 'tbl')",
        "set_result(out)",
    ]
)

_READ_TABLE = "\n".join(
    [
        "import pandas as pd",
        "_p = __PATH__",
        "if _p.lower().endswith('.csv'):",
        "    _df = pd.read_csv(_p, encoding='utf-8-sig')",
        "else:",
        "    import pyogrio",
        "    _df = pyogrio.read_dataframe(_p)",
        "set_result({'columns': [str(c) for c in _df.columns],",
        "            'rows': [[None if (v is None or (isinstance(v, float) and v != v)) else v",
        "                      for v in r] for r in _df.itertuples(index=False, name=None)]})",
    ]
)


#: 真 .gdb 里的栅格（DAIR-217 B1/B2）。GDAL 没有 FGDB 写驱动，只能借 arcpy 造：
#: single = 单栅格库；multi = 两份栅格（dem / dem_shift，后者逐值 +1000，用来认到底读了哪份）
_MAKE_GDB_RASTERS = "\n".join(
    [
        "import os",
        "import arcpy",
        "arcpy.env.overwriteOutput = True",
        "_work = __WORK__",
        "",
        "def _gdb(name):",
        "    path = os.path.join(_work, name)",
        "    if not arcpy.Exists(path):",
        "        arcpy.management.CreateFileGDB(_work, name)",
        "    return path",
        "",
        "single = _gdb('raster_single.gdb')",
        "arcpy.management.CopyRaster(__TIF__, os.path.join(single, 'dem'))",
        "multi = _gdb('raster_multi.gdb')",
        "arcpy.management.CopyRaster(__TIF__, os.path.join(multi, 'dem'))",
        "arcpy.management.CopyRaster(__SHIFT__, os.path.join(multi, 'dem_shift'))",
        "out_gdb = _gdb('pb_out.gdb')",
        "set_result({'single': single, 'multi': multi, 'out': out_gdb})",
    ]
)

#: 要素类挂在要素数据集下的 .gdb（B5：两端对 ``x.gdb/数据集/要素类`` 的取舍相反）
_MAKE_GDB_POINTS_FD = "\n".join(
    [
        "import os",
        "import arcpy",
        "arcpy.env.overwriteOutput = True",
        "_work = __WORK__",
        "gdb = os.path.join(_work, 'pts_fds.gdb')",
        "if not arcpy.Exists(gdb):",
        "    arcpy.management.CreateFileGDB(_work, 'pts_fds.gdb')",
        "sr = arcpy.SpatialReference(32650)",
        "fd = os.path.join(gdb, 'fd')",
        "if not arcpy.Exists(fd):",
        "    arcpy.management.CreateFeatureDataset(gdb, 'fd', sr)",
        "fc = os.path.join(fd, 'pts')",
        "if arcpy.Exists(fc):",
        "    arcpy.management.Delete(fc)",
        "arcpy.management.CreateFeatureclass(fd, 'pts', 'POINT', spatial_reference=sr)",
        "arcpy.management.AddField(fc, 'pid', 'LONG')",
        "with arcpy.da.InsertCursor(fc, ['pid', 'SHAPE@XY']) as cur:",
        "    for pid, x, y in [(1, 500.0, 500.0), (2, 1500.0, 500.0), (3, 550.0, 520.0)]:",
        "        cur.insertRow([pid, (x, y)])",
        "set_result({'gdb': gdb, 'fc': fc})",
    ]
)


# --------------------------------------------------------------------- 夹具
@pytest.fixture(scope="module")
def work(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("point_buffer_raster_sum")


@pytest.fixture(scope="module")
def gdal_runner(work: Path) -> GdalCodeRunner:
    interpreter = default_python_executable()
    if interpreter is None or not Path(interpreter).is_file():
        pytest.skip("未找到 .venv-gdal 解释器（见 docs/p0-1b-env.md），跳过真 GDAL 用例")
    return GdalCodeRunner(workspace=work, python_executable=interpreter)


@pytest.fixture(scope="module")
def inputs(gdal_runner: GdalCodeRunner, work: Path) -> dict:
    made = gdal_runner.run(_MAKE_INPUTS.replace("__WORK__", repr(str(work))), timeout=600)
    assert made.ok, made.error
    return made.result


@pytest.fixture(scope="module")
def arcpy_runner(work: Path):
    from gis_cli.arcpy_bridge import ArcGISDiscoveryError, discover_arcgis_pro_python

    try:
        info = discover_arcgis_pro_python()
    except ArcGISDiscoveryError as exc:
        pytest.skip(f"本机没有 ArcGIS Pro（{exc}），跳过双端一致用例")
    runner = CodeRunner(workspace=work, python_executable=info.python_executable, exec_timeout=900)
    try:
        yield runner
    finally:
        runner.shutdown()


@pytest.fixture(scope="module")
def gdb_rasters(arcpy_runner, work: Path, inputs: dict) -> dict:
    """造真 .gdb 栅格（GDAL 写不了 FGDB，只能借 arcpy）。"""
    made = arcpy_runner.run(
        _MAKE_GDB_RASTERS.replace("__WORK__", repr(str(work)))
        .replace("__TIF__", repr(inputs["raster"]))
        .replace("__SHIFT__", repr(inputs["raster_shift"])),
        timeout=600,
    )
    assert made.ok, made.error
    return made.result


@pytest.fixture(scope="module")
def gdb_points_fd(arcpy_runner, work: Path) -> dict:
    """要素类在要素数据集里的点库（B5）。"""
    made = arcpy_runner.run(_MAKE_GDB_POINTS_FD.replace("__WORK__", repr(str(work))), timeout=600)
    assert made.ok, made.error
    return made.result


# --------------------------------------------------------------------- 工具
def _run(recipe_lib: RecipeLibrary, runner, work: Path, **params):
    recipe_lib.load()
    ctx = SimpleNamespace(workspace=work, code_runner=runner, exec_timeout=900)
    return recipe_lib.run("point_buffer_raster_sum", params, ctx)


def _run_gdal(gdal_runner, work: Path, name: str, **params):
    lib = RecipeLibrary(code_runner=gdal_runner)
    params.setdefault("output_table", str(work / name))
    return _run(lib, gdal_runner, work, **params)


def _table(gdal_runner, path) -> dict:
    """读回统计表 → {pid: (SUM_VALUE, CELL_COUNT)}（列顺序保持，供字段名断言）。"""
    got = gdal_runner.run(_READ_TABLE.replace("__PATH__", repr(str(path))), timeout=300)
    assert got.ok, got.error
    columns = got.result["columns"]
    table = {}
    for row in got.result["rows"]:
        record = dict(zip(columns, row))
        table[int(record[columns[0]])] = (float(record["SUM_VALUE"]), int(record["CELL_COUNT"]))
    return {"columns": columns, "rows": table}


def _base_params(inputs: dict, **over) -> dict:
    params = {
        "points_path": inputs["pts3"] + "/pts",
        "id_field": "pid",
        "raster_path": inputs["raster"],
        "radius": 500,
        "nodata_as_zero": True,
    }
    params.update(over)
    return params


# ------------------------------------------------------------- 模板本身
def test_recipe_declares_two_backends_and_renders_both():
    recipe = Recipe.from_yaml(RECIPE)
    assert recipe.backend == ["arcpy", "gdal"]
    params = {
        "points_path": "pts.gpkg",
        "id_field": "pid",
        "raster_path": "dem.tif",
        "radius": 500,
        "output_table": "out/stats.csv",
        "nodata_as_zero": True,
    }
    arcpy_code = recipe.render(params)
    gdal_code = recipe.render(params, backend="gdal")

    # arcpy 版补的是容器判定（共享片段），统计逻辑一个字没动
    assert "def resolve_zone_layer(path):" in arcpy_code
    assert "points_path = resolve_zone_layer(points_path)" in arcpy_code
    assert "arcpy.RasterToNumPyArray" in arcpy_code and "rasterio" not in arcpy_code
    # GDAL 版：同一套格心判定 + 输出格式白名单
    assert "from rasterio.windows import Window" in gdal_code and "import arcpy" not in gdal_code
    assert '_OUT_EXT = (".csv", ".gpkg")' in gdal_code
    assert "int(round(number))" in gdal_code  # 与 arcpy 的 int(round(float(pid))) 同一口径
    # DAIR-217 B1/B2：栅格路径要能翻译成 GDAL 认的写法，存在性判据不能再是 os.path.exists
    assert "def _resolve_raster_path(path):" in gdal_code
    assert 'OpenFileGDB:"' in gdal_code
    assert "os.path.exists(str(raster_path))" not in gdal_code
    for key in ("pts.gpkg", "dem.tif", "pid", "500"):
        assert key in arcpy_code and key in gdal_code
    # 片段占位符只在 arcpy 模板里、且独占一行（这是设计：混写渲染时直接报错）
    assert fragment_names(recipe.code_template) == {"zone_layer"}
    assert fragment_names(recipe.code_template_gdal) == set()
    assert "zone_layer" in FRAGMENTS


# --------------------------------------------------- GDAL 侧：逐值对上 arcpy
@pytest.mark.parametrize("ext", [".csv", ".gpkg"])
def test_values_match_arcpy_reference(gdal_runner, work: Path, inputs: dict, ext: str):
    """acceptance #1：3 点 + 500 m 缓冲 + 100 m 栅格，逐点值等于 arcpy 实测。"""
    outcome = _run_gdal(gdal_runner, work, f"ref{ext}", **_base_params(inputs))
    assert outcome.ok, outcome.summary

    table = _table(gdal_runner, work / f"ref{ext}")
    assert table["columns"] == ["pid", "SUM_VALUE", "CELL_COUNT"]
    assert table["rows"] == REFERENCE
    result = outcome.data["result"]
    assert result["point_count"] == 3
    assert result["points_with_data"] == 3
    assert result["total_value"] == REFERENCE_TOTAL
    # 断言走的是 GDAL 后端自己的读数实现（pyogrio），不是 arcpy
    assert [a["ok"] for a in outcome.data["assertions"]] == [True, True]


def test_overlapping_buffers_are_counted_once_per_point(gdal_runner, work: Path, inputs: dict):
    """acceptance #2：重叠缓冲按点累加（每个点各算各的圆），不是全局只算一次。"""
    both = _run_gdal(gdal_runner, work, "both.csv", **_base_params(inputs))
    a_only = _run_gdal(gdal_runner, work, "a.csv",
                       **_base_params(inputs, points_path=inputs["pts_a_only"] + "/pts"))
    c_only = _run_gdal(gdal_runner, work, "c.csv",
                       **_base_params(inputs, points_path=inputs["pts_c_only"] + "/pts"))
    for outcome in (both, a_only, c_only):
        assert outcome.ok, outcome.summary

    both_rows = _table(gdal_runner, work / "both.csv")["rows"]
    a_row = _table(gdal_runner, work / "a.csv")["rows"]
    c_row = _table(gdal_runner, work / "c.csv")["rows"]
    # A 与 C 的圆几乎完全重叠：A 那行不因 C 的存在而变化（各点独立计算）
    assert a_row[1] == both_rows[1] == REFERENCE[1]
    assert c_row[3] == both_rows[3] == REFERENCE[3]
    # 重叠像元在 A 与 C 两行里各记一次：合计 = 三个单位置之和，不是并集之和
    assert both.data["result"]["total_value"] == sum(v[0] for v in REFERENCE.values())
    assert both.data["result"]["total_value"] == (
        a_row[1][0] + c_row[3][0] + REFERENCE[2][0]
    )


@pytest.mark.parametrize("nodata_as_zero", [True, False])
def test_nodata_does_not_change_values(gdal_runner, work: Path, inputs: dict, nodata_as_zero):
    """圆内 4 个 nodata 像元：SUM 记 0 与记 NaN 等价，CELL_COUNT 两档都把它们算进去。"""
    outcome = _run_gdal(
        gdal_runner, work, f"nodata_{nodata_as_zero}.csv",
        **_base_params(inputs, points_path=inputs["pts_a_only"] + "/pts",
                       raster_path=inputs["raster_nodata"], nodata_as_zero=nodata_as_zero),
    )
    assert outcome.ok, outcome.summary
    assert _table(gdal_runner, work / f"nodata_{nodata_as_zero}.csv")["rows"][1] == NODATA_REFERENCE


def test_point_outside_raster_still_gets_a_row(gdal_runner, work: Path, inputs: dict):
    """点完全在栅格外：仍出一行 0.0 / 0（与 arcpy 一致，不是「不出行」）。"""
    outcome = _run_gdal(gdal_runner, work, "far.csv",
                        **_base_params(inputs, points_path=inputs["pts_far"] + "/pts"))
    assert outcome.ok, outcome.summary
    assert _table(gdal_runner, work / "far.csv")["rows"] == {9: (0.0, 0)}
    assert outcome.data["result"]["points_with_data"] == 0


def test_non_point_input_uses_geometry_centroid(gdal_runner, work: Path, inputs: dict):
    """面输入按几何中心算：arcpy 的 SHAPE@XY 返回的就是中心点（实测面与中心点同值）。"""
    outcome = _run_gdal(gdal_runner, work, "poly.csv",
                        **_base_params(inputs, points_path=inputs["poly"] + "/poly"))
    assert outcome.ok, outcome.summary
    assert _table(gdal_runner, work / "poly.csv")["rows"] == {1: REFERENCE[1]}


def test_duplicate_ids_are_written_as_separate_rows(gdal_runner, work: Path, inputs: dict):
    """同一 ID 的两个点各出一行（arcpy 也不去重）——行数按要素数，不按 ID 去重。"""
    dup = work / "dup.gpkg"
    made = gdal_runner.run(
        "\n".join([
            "import geopandas as gpd",
            "from shapely.geometry import Point",
            "gpd.GeoDataFrame({'pid': [7, 7]}, geometry=[Point(500, 500), Point(1500, 500)],",
            "    crs='EPSG:32650').to_file(__PATH__, driver='GPKG', engine='pyogrio', layer='pts')",
            "set_result({'ok': True})",
        ]).replace("__PATH__", repr(str(dup))),
        timeout=120,
    )
    assert made.ok, made.error
    outcome = _run_gdal(gdal_runner, work, "dup.csv",
                        **_base_params(inputs, points_path=str(dup) + "/pts"))
    assert outcome.ok, outcome.summary
    assert outcome.data["result"]["point_count"] == 2
    with open(work / "dup.csv", encoding="utf-8-sig") as fh:
        assert len(fh.read().strip().splitlines()) == 3  # 表头 + 2 行


# ------------------------------------------------------- 容器输入与格式白名单
def test_single_layer_container_is_used_automatically(gdal_runner, work: Path, inputs: dict):
    """acceptance #3：.gpkg 单层容器、shp 文件夹都不给图层名也能跑。"""
    gpkg = _run_gdal(gdal_runner, work, "container_gpkg.csv",
                     **_base_params(inputs, points_path=inputs["pts3"]))
    shp = _run_gdal(gdal_runner, work, "container_shp.csv",
                    **_base_params(inputs, points_path=inputs["shpdir"]))
    for outcome in (gpkg, shp):
        assert outcome.ok, outcome.summary
        assert "自动使用其中唯一的图层" in outcome.data["stdout"]
    assert _table(gdal_runner, work / "container_gpkg.csv")["rows"] == REFERENCE
    assert _table(gdal_runner, work / "container_shp.csv")["rows"] == {1: REFERENCE[1],
                                                                      2: REFERENCE[2]}


def test_multi_layer_container_asks_for_a_layer_by_name(gdal_runner, work: Path, inputs: dict):
    """多层容器给人话错误：列出图层名 + 怎么指定，不是 AttributeError。"""
    outcome = _run_gdal(gdal_runner, work, "multi.csv",
                        **_base_params(inputs, points_path=inputs["multi"]))
    assert not outcome.ok
    message = str(getattr(outcome, "error", outcome.summary))
    assert "layer_one" in message and "layer_two" in message
    assert "请指明图层" in message and "AttributeError" not in message


def test_layer_in_path_works_for_gpkg(gdal_runner, work: Path, inputs: dict):
    """``x.gpkg/图层名`` 这种写法两端都认（片段/报错文案给的就是它）。"""
    outcome = _run_gdal(gdal_runner, work, "layer_path.csv",
                        **_base_params(inputs, points_path=inputs["multi"] + "/layer_two"))
    assert outcome.ok, outcome.summary
    assert _table(gdal_runner, work / "layer_path.csv")["rows"] == {2: REFERENCE[2]}


def test_gdb_output_is_rejected_with_a_clear_message(gdal_runner, work: Path, inputs: dict):
    """acceptance：.gdb 输出白名单 + 明确报错，不静默换格式。"""
    outcome = _run_gdal(gdal_runner, work, "x", **_base_params(
        inputs, points_path=inputs["pts_a_only"] + "/pts", output_table=inputs["gdb_out"]))
    assert not outcome.ok
    message = str(getattr(outcome, "error", outcome.summary))
    assert "GDAL 后端写不了 gdb 表" in message and ".csv" in message
    assert "arcpy 后端" in message


# ------------------------------------------- .gdb 里的栅格（DAIR-217 B1/B2）
def test_gdb_raster_layer_path_works_with_both_separators(gdal_runner, work, inputs, gdb_rasters):
    """B1：arcpy 侧交出来的 ``x.gdb\层名`` 不能再报「栅格数据不存在」（数据在库里）。"""
    for sep, tag in (("\\", "backslash"), ("/", "slash")):
        outcome = _run_gdal(
            gdal_runner, work, f"gdb_layer_{tag}.csv",
            **_base_params(inputs, raster_path=gdb_rasters["single"] + sep + "dem"),
        )
        assert outcome.ok, outcome.summary
        assert _table(gdal_runner, work / f"gdb_layer_{tag}.csv")["rows"] == REFERENCE
        assert "栅格数据不存在" not in outcome.data["stdout"]


def test_plain_gdb_directory_with_a_raster_inside_still_opens_as_a_file(
    gdal_runner, work, inputs
):
    """``.gdb`` 结尾的**普通目录**里放散文件：按文件路径打开，不翻 OpenFileGDB。

    与 ``test_zonal_statistics_gdal.py`` 的同名用例同形态（DAIR-224）。
    """
    plain = work / "plain.gdb"
    plain.mkdir(exist_ok=True)
    shutil.copyfile(inputs["raster"], plain / "dem.tif")

    outcome = _run_gdal(gdal_runner, work, "plain_gdb.csv",
                        **_base_params(inputs, raster_path=str(plain) + "\\dem.tif"))

    assert outcome.ok, outcome.summary
    assert "栅格数据不存在" not in outcome.data["stdout"]
    assert _table(gdal_runner, work / "plain_gdb.csv")["rows"] == REFERENCE


def test_single_raster_gdb_container_is_used_automatically(gdal_runner, work, inputs, gdb_rasters):
    """B2：只给库本身且库里恰好一份栅格时可以直接用（多份才报错要你指明）。"""
    outcome = _run_gdal(gdal_runner, work, "gdb_container.csv",
                        **_base_params(inputs, raster_path=gdb_rasters["single"]))
    assert outcome.ok, outcome.summary
    assert "栅格数据不存在" not in outcome.data["stdout"]
    assert _table(gdal_runner, work / "gdb_container.csv")["rows"] == REFERENCE


def test_multi_raster_gdb_container_asks_for_a_raster_by_name(gdal_runner, work, inputs, gdb_rasters):
    """B2：多栅格库只给库本身 → 人话错误（列出栅格名 + 怎么指定），不是 No such file。"""
    outcome = _run_gdal(gdal_runner, work, "gdb_multi.csv",
                        **_base_params(inputs, raster_path=gdb_rasters["multi"]))
    assert not outcome.ok
    message = str(getattr(outcome, "error", outcome.summary))
    assert "dem" in message and "dem_shift" in message
    assert "请指明栅格" in message and "No such file" not in message


def test_multi_raster_gdb_with_explicit_layer_reads_that_raster(gdal_runner, work, inputs, gdb_rasters):
    """B2：``x.gdb\层名`` 要读到**指定的那一份**（dem_shift 逐值 +1000）。"""
    outcome = _run_gdal(gdal_runner, work, "gdb_multi_pick.csv",
                        **_base_params(inputs, raster_path=gdb_rasters["multi"] + "\\dem_shift"))
    assert outcome.ok, outcome.summary
    expected = {pid: (sum_v + 1000.0 * count, count) for pid, (sum_v, count) in REFERENCE.items()}
    assert _table(gdal_runner, work / "gdb_multi_pick.csv")["rows"] == expected


@pytest.mark.parametrize("form", ["missing_file", "missing_gdb"])
def test_unreadable_raster_reports_what_was_tried(gdal_runner, work, inputs, form):
    """存在性判据换成「能不能打开」之后，真不存在时仍要说清是哪个路径。"""
    path = (str(work / "nope.tif") if form == "missing_file"
            else str(work / "nope.gdb") + "\\dem")
    outcome = _run_gdal(gdal_runner, work, "missing.csv", **_base_params(inputs, raster_path=path))
    assert not outcome.ok
    message = str(getattr(outcome, "error", outcome.summary))
    assert "栅格数据不存在" in message and "nope" in message


def test_gdb_raster_dual_end_matches_arcpy(arcpy_runner, gdal_runner, work, inputs, gdb_rasters):
    """B1/B2 的真实回归：同一个 .gdb 栅格，arcpy 与 GDAL 逐点相同（DAIR-215 D 组跑不起来的那一格）。"""
    from gis_cli.recipes.store import RecipeLibrary as _Lib

    raster = gdb_rasters["single"] + "\\dem"
    arcpy_out = gdb_rasters["out"] + "\\gdb_raster_tbl"
    got_arcpy = _run(_Lib(code_runner=arcpy_runner), arcpy_runner, work,
                     **{**_base_params(inputs), "raster_path": raster, "output_table": arcpy_out})
    assert got_arcpy.ok, got_arcpy.summary
    got_gdal = _run_gdal(gdal_runner, work, "gdb_raster.csv",
                         **_base_params(inputs, raster_path=raster))
    assert got_gdal.ok, got_gdal.summary

    read = arcpy_runner.run(
        "\n".join([
            "import arcpy",
            f"names = [f.name for f in arcpy.ListFields({arcpy_out!r})",
            "         if f.type not in ('Geometry', 'OID', 'Blob', 'Raster')]",
            f"rows = [list(r) for r in arcpy.da.SearchCursor({arcpy_out!r}, names)]",
            "set_result({'columns': names, 'rows': rows})",
        ]),
        timeout=300,
    )
    assert read.ok, read.error
    assert read.result["columns"] == ["pid", "SUM_VALUE", "CELL_COUNT"]
    arcpy_rows = {int(r[0]): (float(r[1]), int(r[2])) for r in read.result["rows"]}
    gdal_rows = _table(gdal_runner, work / "gdb_raster.csv")["rows"]
    assert arcpy_rows == REFERENCE
    assert set(gdal_rows) == set(arcpy_rows)
    for pid, (a_sum, a_count) in arcpy_rows.items():
        g_sum, g_count = gdal_rows[pid]
        assert g_count == a_count, f"pid={pid} CELL_COUNT 不一致：{a_count} vs {g_count}"
        assert abs(g_sum - a_sum) <= 1e-9 * max(abs(a_sum), 1.0), f"pid={pid} SUM 不一致"
    a_total = got_arcpy.data["result"]["total_value"]
    g_total = got_gdal.data["result"]["total_value"]
    assert abs(g_total - a_total) <= 1e-9 * max(abs(a_total), 1.0)


# ------------------------------------ 要素数据集内的要素类（DAIR-217 B5）
def test_gdb_feature_dataset_container_is_used_automatically(gdal_runner, work, inputs, gdb_points_fd):
    """B5：要素类都在要素数据集里、只给库本身时 GDAL 自动取唯一图层。"""
    outcome = _run_gdal(gdal_runner, work, "fds_container.csv",
                        **_base_params(inputs, points_path=gdb_points_fd["gdb"]))
    assert outcome.ok, outcome.summary
    assert "自动使用其中唯一的图层" in outcome.data["stdout"]
    assert _table(gdal_runner, work / "fds_container.csv")["rows"] == REFERENCE


def test_gdb_feature_dataset_layer_path_gives_a_human_error(gdal_runner, work, inputs, gdb_points_fd):
    """B5：GDAL 的图层名不含要素数据集前缀——报错要说清，而不是丢 pyogrio 的 DataLayerError。"""
    outcome = _run_gdal(gdal_runner, work, "fds_layer.csv",
                        **_base_params(inputs, points_path=gdb_points_fd["gdb"] + "\\fd\\pts"))
    assert not outcome.ok
    message = str(getattr(outcome, "error", outcome.summary))
    assert "数据集前缀" in message and "pts" in message
    assert "arcpy 认" in message and "DataLayerError" not in message


# ------------------------------------------------------------- 输入侧的人话错误
def test_text_id_field_is_rejected_with_a_clear_message(gdal_runner, work: Path, inputs: dict):
    outcome = _run_gdal(gdal_runner, work, "textid.csv",
                        **_base_params(inputs, points_path=inputs["pts_textid"] + "/pts"))
    assert not outcome.ok
    message = str(getattr(outcome, "error", outcome.summary))
    assert "必须是数值型" in message and "A001" in message
    assert "int(round(float(pid)))" in message  # 说清 arcpy 侧是怎么落表的


def test_missing_id_field_lists_the_available_ones(gdal_runner, work: Path, inputs: dict):
    outcome = _run_gdal(gdal_runner, work, "nofield.csv",
                        **_base_params(inputs, id_field="nope"))
    assert not outcome.ok
    message = str(getattr(outcome, "error", outcome.summary))
    assert "缺少字段: nope" in message and "pid" in message


def test_multiband_raster_is_rejected_with_a_clear_message(gdal_runner, work: Path, inputs: dict):
    """arcpy 对多波段直接在 Describe 上报「方法 width 不存在」；GDAL 侧给可执行的报错。"""
    outcome = _run_gdal(gdal_runner, work, "bands.csv",
                        **_base_params(inputs, points_path=inputs["pts_a_only"] + "/pts",
                                       raster_path=inputs["raster_multiband"]))
    assert not outcome.ok
    message = str(getattr(outcome, "error", outcome.summary))
    assert "2 个波段" in message and "单波段" in message


def test_null_geometry_is_rejected_not_silently_skipped(gdal_runner, work: Path, inputs: dict):
    """几何 NULL：arcpy 抛 TypeError（SHAPE@XY 给 (None, None)，`if shp is None` 是死代码）。"""
    outcome = _run_gdal(gdal_runner, work, "nulls.csv",
                        **_base_params(inputs, points_path=inputs["pts_null"] + "/pts"))
    assert not outcome.ok
    message = str(getattr(outcome, "error", outcome.summary))
    assert "几何为空" in message and "第 2 行" in message


def test_crs_mismatch_is_rejected(gdal_runner, work: Path, inputs: dict):
    outcome = _run_gdal(gdal_runner, work, "crs.csv",
                        **_base_params(inputs, points_path=inputs["pts_4326"] + "/pts"))
    assert not outcome.ok
    message = str(getattr(outcome, "error", outcome.summary))
    assert "坐标系不一致" in message and "请先投影" in message


def test_same_epsg_in_esri_wkt_is_accepted(gdal_runner, work: Path, inputs: dict):
    """同 EPSG、ESRI 写法 WKT（ArcGIS 导出的 GeoTIFF 十有八九是这种）不能判成不一致。"""
    assert inputs["pts_esri_epsg"] == 32650
    outcome = _run_gdal(gdal_runner, work, "esri.csv",
                        **_base_params(inputs, points_path=inputs["pts_esri"] + "/pts"))
    assert outcome.ok, outcome.summary
    assert _table(gdal_runner, work / "esri.csv")["rows"] == {1: REFERENCE[1]}


def test_geographic_crs_warns_but_keeps_the_unit(gdal_runner, work: Path, inputs: dict):
    """地理坐标系下半径就是「度」：两端都不改单位（arcpy 实测不报错），GDAL 侧只多一行警告。"""
    outcome = _run_gdal(gdal_runner, work, "gcs.csv",
                        **_base_params(inputs, points_path=inputs["pts_4326"] + "/pts",
                                       raster_path=inputs["raster_4326"]))
    assert outcome.ok, outcome.summary
    assert "会按「度」计算" in outcome.data["stdout"]


def test_output_is_overwritten(gdal_runner, work: Path, inputs: dict):
    """同一个输出跑两次：覆盖而不是追加（arcpy 侧也是 Delete + CreateTable）。"""
    params = _base_params(inputs, points_path=inputs["pts_a_only"] + "/pts")
    for _ in range(2):
        outcome = _run_gdal(gdal_runner, work, "over.csv", **params)
        assert outcome.ok, outcome.summary
    with open(work / "over.csv", encoding="utf-8-sig") as fh:
        assert len(fh.read().strip().splitlines()) == 2  # 表头 + 1 行


# ------------------------------------------------------------------ 双端一致
def test_dual_end_values_are_identical(arcpy_runner, gdal_runner, work: Path, inputs: dict):
    """acceptance #1：同一份输入两端各跑一次，整数像元值/计数严格相等、浮点相对差 ≤1e-6。"""
    from gis_cli.recipes.store import RecipeLibrary as _Lib

    gdb = work / "dual.gdb"
    boot = arcpy_runner.run(
        "\n".join([
            "import arcpy",
            "arcpy.env.overwriteOutput = True",
            f"if not arcpy.Exists({str(gdb)!r}):",
            f"    arcpy.management.CreateFileGDB({str(work)!r}, 'dual.gdb')",
            "set_result({'ok': True})",
        ]),
        timeout=300,
    )
    assert boot.ok, boot.error

    base = _base_params(inputs)
    arcpy_out = str(gdb / "buffer_tbl")
    got_arcpy = _run(_Lib(code_runner=arcpy_runner), arcpy_runner, work,
                     **{**base, "output_table": arcpy_out})
    assert got_arcpy.ok, got_arcpy.summary

    got_gdal = _run_gdal(gdal_runner, work, "dual.csv", **base)
    assert got_gdal.ok, got_gdal.summary

    read_arcpy = arcpy_runner.run(
        "\n".join([
            "import arcpy",
            f"names = [f.name for f in arcpy.ListFields({arcpy_out!r})",
            "         if f.type not in ('Geometry', 'OID', 'Blob', 'Raster')]",
            f"rows = [list(r) for r in arcpy.da.SearchCursor({arcpy_out!r}, names)]",
            "set_result({'columns': names, 'rows': rows})",
        ]),
        timeout=300,
    )
    assert read_arcpy.ok, read_arcpy.error
    columns = read_arcpy.result["columns"]
    assert columns == ["pid", "SUM_VALUE", "CELL_COUNT"], "输出表字段名/顺序两端必须一致"
    arcpy_rows = {int(r[0]): (float(r[1]), int(r[2])) for r in read_arcpy.result["rows"]}
    gdal_rows = _table(gdal_runner, work / "dual.csv")["rows"]

    assert set(arcpy_rows) == set(gdal_rows) == set(REFERENCE)
    for pid in arcpy_rows:
        a_sum, a_count = arcpy_rows[pid]
        g_sum, g_count = gdal_rows[pid]
        assert a_count == g_count, f"pid={pid} CELL_COUNT 不一致：{a_count} vs {g_count}"
        assert abs(g_sum - a_sum) <= 1e-6 * max(abs(a_sum), 1.0), f"pid={pid} SUM 不一致"
        assert g_sum == a_sum, f"pid={pid} SUM 应当逐值相等：{a_sum} vs {g_sum}"
    assert got_arcpy.data["result"]["total_value"] == got_gdal.data["result"]["total_value"]


def test_dual_end_container_input(arcpy_runner, gdal_runner, work: Path, inputs: dict):
    """acceptance #3：容器输入两端都走通（arcpy 侧是 DAIR-207 补的共享片段）。"""
    from gis_cli.recipes.store import RecipeLibrary as _Lib

    gdb = work / "dual.gdb"
    for label, points_path in (("gpkg", inputs["pts3"]), ("shp", inputs["shpdir"])):
        arcpy_out = str(gdb / f"container_{label}")
        got = _run(_Lib(code_runner=arcpy_runner), arcpy_runner, work,
                   **{**_base_params(inputs), "points_path": points_path,
                      "output_table": arcpy_out})
        assert got.ok, f"{label}: {got.summary}"
        assert "自动使用其中唯一的图层" in got.data["stdout"]

        gdal_out = work / f"container_{label}.csv"
        got_g = _run_gdal(gdal_runner, work, gdal_out.name,
                          **{**_base_params(inputs), "points_path": points_path})
        assert got_g.ok, f"{label}: {got_g.summary}"

        read = arcpy_runner.run(
            "\n".join([
                "import arcpy",
                f"rows = [list(r) for r in arcpy.da.SearchCursor({arcpy_out!r},",
                "         ['pid', 'SUM_VALUE', 'CELL_COUNT'])]",
                "set_result({'rows': rows})",
            ]),
            timeout=300,
        )
        assert read.ok, read.error
        expected = {int(r[0]): (float(r[1]), int(r[2])) for r in read.result["rows"]}
        got_table = _table(gdal_runner, gdal_out)["rows"]
        assert got_table == expected, f"{label}: 双端容器输入结果不一致"


def test_dual_end_multi_layer_container_error(arcpy_runner, gdal_runner, work: Path, inputs: dict):
    """多层容器两端都给人话错误（不是 AttributeError），并都说清怎么指定图层。"""
    from gis_cli.recipes.store import RecipeLibrary as _Lib

    got_arcpy = _run(_Lib(code_runner=arcpy_runner), arcpy_runner, work,
                     **{**_base_params(inputs), "points_path": inputs["multi"],
                        "output_table": str(work / "dual.gdb" / "multi")})
    assert not got_arcpy.ok
    message = str(getattr(got_arcpy, "error", got_arcpy.summary))
    assert "多层容器" in message and "layer_one" in message and "AttributeError" not in message

    got_gdal = _run_gdal(gdal_runner, work, "multi_dual.csv",
                         **_base_params(inputs, points_path=inputs["multi"]))
    assert not got_gdal.ok
    assert "多层容器" in str(getattr(got_gdal, "error", got_gdal.summary))

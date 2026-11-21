# -*- coding: utf-8 -*-
"""zonal_statistics_raster 的 GDAL 版真跑 + **逐像元**双端一致。

测试数据（EPSG:32650，米）：

    分区面  三块 1 km × 1 km 方块，x 依次相接（Z1 0..1000 / Z2 1000..2000 / Z3 2000..3000）
    栅格    100 m 像元，30 列 × 10 行，像元值 = 列号（0..29），nodata = -9999

分区与栅格网格对齐，格心（50/150/…）不会压在分区边界上 —— 边界正好压格心时两端取舍
不同（见配方 pitfalls），那是本批验收口径「逐像元一致」唯一可复现的构造。

``REFERENCE`` / ``REFERENCE_STATS`` 那两组数是**真 arcpy 跑出来的**（ArcGIS Pro 3.6 的
ZonalStatistics，证据在 ``benchmark/reports/dair91_zonal_raster_20261029/probe_evidence*.json``），
不是从 GDAL 侧抄回来的。它们同时钉住 arcpy 的四条实测口径：

1. 分区内每个像元都写该分区的统计值（不是逐像元原值）；
2. 输出范围是**分区 bbox 外扩到栅格格线、再裁到栅格范围**，不是整幅栅格；
3. 输出永远是 float32 + nodata=-3.4028235e+38（不管输入 dtype / nodata）；
4. 输入 nodata 像元不计入统计，但落在分区内的那几格仍写该分区的统计值；
   整块只有 nodata 的分区，输出那几格仍是 nodata。

缺环境就 skip（与 ``test_zonal_statistics_gdal.py`` 同一套口径）：
``.venv-gdal`` 不在 → 所有真 GDAL 用例 skip；ArcGIS Pro 找不到 → 只有双端用例 skip。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gis_cli.recipes.gdal_checks import gdal_raster_check
from gis_cli.recipes.schema import Recipe
from gis_cli.recipes.store import RecipeLibrary
from gis_cli.runner.gdal_runner import GdalCodeRunner, default_python_executable
from gis_cli.runtime.code_runner import CodeRunner

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "src/gis_cli/recipes/builtin/zonal_statistics_raster.yaml"

#: (zone_id, 名字, 左下角 x)：三块 1 km × 1 km 方块，y 都是 0..1000
ZONE_SPECS = [(11, "Z1", 0), (22, "Z2", 1000), (33, "Z3", 2000)]

#: arcpy（Pro 3.6）实测：SUM 每块分区 100 个像元
REFERENCE = {11: 450.0, 22: 1450.0, 33: 2450.0}

#: arcpy（Pro 3.6）实测：其它统计量在 Z1（像元值 0..9 各 10 个）上的取值
REFERENCE_STATS = {
    "MAXIMUM": 9.0,
    "MINIMUM": 0.0,
    "RANGE": 9.0,
    "STD": 2.872281312942505,  # 总体标准差 sqrt(8.25)
    "MEDIAN": 4.5,
}

#: 输出 nodata：arcpy 实测恒为 float32 最小值
NODATA_OUT = -3.4028234663852886e38

_MAKE_SCENE = "\n".join(
    [
        "import geopandas as gpd",
        "import numpy as np",
        "import rasterio",
        "from rasterio.transform import from_origin",
        "from shapely.geometry import Polygon",
        "_boxes = __BOXES__",
        "zones = gpd.GeoDataFrame({'zone_id': [b[0] for b in _boxes],",
        "                          'name': [b[1] for b in _boxes]},",
        "    geometry=[Polygon(b[2]) for b in _boxes], crs='EPSG:32650')",
        "zones.to_file(__ZONES__, driver='GPKG', engine='pyogrio', layer='zones')",
        "data = np.tile(np.arange(__WIDTH__, dtype='float32'), (__ROWS__, 1))",
        "for _c in __NODATA_COLS__:",
        "    data[:, _c] = -9999.0",
        "with rasterio.open(__RASTER__, 'w', driver='GTiff', height=__ROWS__, width=__WIDTH__, count=1,",
        "                   dtype='float32', crs='EPSG:32650',",
        "                   transform=from_origin(0, __ROWS__ * 100, 100, 100), nodata=-9999.0) as ds:",
        "    ds.write(data, 1)",
        "set_result({'zones': str(__ZONES__), 'raster': str(__RASTER__)})",
    ]
)

_READ_RASTER = "\n".join(
    [
        "import numpy as np",
        "import rasterio",
        "with rasterio.open(__PATH__) as ds:",
        "    _a = ds.read(1, masked=True).astype('float64')",
        "    _t = ds.transform",
        "    _b = ds.bounds",
        "    set_result({'values': _a.filled(np.nan).tolist(),",
        "                'shape': [int(ds.height), int(ds.width)],",
        "                'cell': [abs(float(_t.a)), abs(float(_t.e))],",
        "                'extent': [float(_b.left), float(_b.bottom), float(_b.right), float(_b.top)],",
        "                'crs': ds.crs.to_string() if ds.crs else None,",
        "                'nodata': ds.nodata,",
        "                'n_valid': int(_a.count())})",
    ]
)

_READ_ARCPY_RASTER = "\n".join(
    [
        "import arcpy, numpy as np",
        "_d = arcpy.Describe(__PATH__)",
        "_a = arcpy.RasterToNumPyArray(__PATH__, nodata_to_value=np.nan).astype('float64')",
        "_e = _d.extent",
        "set_result({'values': _a.tolist(),",
        "            'shape': [int(_a.shape[0]), int(_a.shape[1])],",
        "            'cell': [float(_d.meanCellWidth), float(_d.meanCellHeight)],",
        "            'extent': [float(_e.XMin), float(_e.YMin), float(_e.XMax), float(_e.YMax)],",
        "            'crs': int(_d.spatialReference.factoryCode) if _d.spatialReference else None,",
        "            'n_valid': int(np.count_nonzero(~np.isnan(_a)))})",
    ]
)

#: arcpy 侧分区：真 .gdb 要素类（GDAL 版从同一份 gdb 读，双端就是“同一输入”）
_MAKE_ARCPY_ZONES = "\n".join(
    [
        "import arcpy",
        "arcpy.env.overwriteOutput = True",
        "sr = arcpy.SpatialReference(32650)",
        "gdb = __GDB__",
        "if not arcpy.Exists(gdb):",
        "    arcpy.management.CreateFileGDB(__PARENT__, __GDB_NAME__[:-4])",
        "_fc = gdb + '\\\\zones'",
        "if arcpy.Exists(_fc):",
        "    arcpy.management.Delete(_fc)",
        "arcpy.management.CreateFeatureclass(gdb, 'zones', 'POLYGON', spatial_reference=sr)",
        "arcpy.management.AddField(_fc, 'zone_id', __FIELD_TYPE__)",
        "_boxes = __BOXES__",
        "with arcpy.da.InsertCursor(_fc, ['SHAPE@', 'zone_id']) as cur:",
        "    for zid, name, pts in _boxes:",
        "        cur.insertRow([arcpy.Polygon(arcpy.Array([arcpy.Point(*p) for p in pts]), sr), zid])",
        "set_result({'zones': _fc})",
    ]
)


#: 真 .gdb 里的栅格（DAIR-221）。GDAL 没有 FGDB 写驱动，只能借 arcpy 造：
#: single = 单栅格库；multi = 两份栅格（dem / dem_shift，后者只是同一份的副本，
#: 用来验「多份不许瞎挑」的报错——这里只需要两个不同的名字）。
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
        "arcpy.management.CopyRaster(__TIF__, os.path.join(multi, 'dem_shift'))",
        "set_result({'single': single, 'multi': multi})",
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


@pytest.fixture()
def gdb_rasters(arcpy_runner, gdal_runner, tmp_path: Path) -> dict:
    """造真 .gdb 栅格（GDAL 写不了 FGDB，只能借 arcpy），顺带建好 .tif 基线竞景。"""
    _, raster = _make_scene(gdal_runner, tmp_path)
    made = arcpy_runner.run(
        _MAKE_GDB_RASTERS.replace("__WORK__", repr(str(tmp_path))).replace(
            "__TIF__", repr(str(raster))
        ),
        timeout=300,
    )
    assert made.ok, made.error
    made.result["raster"] = str(raster)
    return made.result


# --------------------------------------------------------------------- 工具
def _box(x0: int, x1: int) -> list[list[int]]:
    return [[x0, 0], [x1, 0], [x1, 1000], [x0, 1000]]


def _make_scene(
    runner,
    tmp_path: Path,
    boxes=ZONE_SPECS,
    *,
    width: int = 30,
    rows: int = 10,
    nodata_cols: tuple[int, ...] = (),
):
    """建一份场景（分区 .gpkg + 栅格 .tif），返回 (zones, raster)。"""
    zones, raster = tmp_path / "zones.gpkg", tmp_path / "dem.tif"
    spec = [
        (zid, name, _box(x0, x0 + 1000)) if isinstance(x0, int) else (zid, name, x0)
        for zid, name, x0 in boxes
    ]
    made = runner.run(
        _MAKE_SCENE.replace("__ZONES__", repr(str(zones)))
        .replace("__RASTER__", repr(str(raster)))
        .replace("__BOXES__", repr(spec))
        .replace("__WIDTH__", str(width))
        .replace("__ROWS__", str(rows))
        .replace("__NODATA_COLS__", repr(tuple(nodata_cols))),
        timeout=300,
    )
    assert made.ok, made.error
    return zones, raster


def _arcpy_zones(
    arcpy_runner, tmp_path: Path, boxes=None, field_type: str = "LONG", name: str = "zsr_inputs.gdb"
) -> str:
    boxes = boxes or ZONE_SPECS
    spec = [
        (zid, name, _box(x0, x0 + 1000)) if isinstance(x0, int) else (zid, name, x0)
        for zid, name, x0 in boxes
    ]
    gdb = tmp_path / name
    made = arcpy_runner.run(
        _MAKE_ARCPY_ZONES.replace("__GDB__", repr(str(gdb)))
        .replace("__PARENT__", repr(str(tmp_path)))
        .replace("__GDB_NAME__", repr(name))
        .replace("__BOXES__", repr(spec))
        .replace("__FIELD_TYPE__", repr(field_type)),
        timeout=300,
    )
    assert made.ok, made.error
    return made.result["zones"]


def _run(lib: RecipeLibrary, runner, tmp_path: Path, **params):
    lib.load()
    ctx = SimpleNamespace(workspace=tmp_path, code_runner=runner, exec_timeout=600)
    return lib.run("zonal_statistics_raster", params, ctx)


def _zonal(runner, tmp_path: Path, name: str, **overrides):
    params = {
        "zone_path": str(tmp_path / "zones.gpkg"),
        "raster_path": str(tmp_path / "dem.tif"),
        "output_path": str(tmp_path / name),
        "zone_field": "zone_id",
        "statistics": "SUM",
    }
    params.update(overrides)
    return _run(RecipeLibrary(code_runner=runner), runner, tmp_path, **params)


def _result(obs, *, expect_ok: bool = True) -> dict:
    """取配方返回值。

    ``expect_ok=False`` 用于**输出本身就是常量**的用例（单分区、RANGE/STD 三个分区同值）：
    配方自带的 ``raster_not_constant`` 断言会判否——那是配方在防"整幅同值"的坏产物，
    不是本用例要验的东西，这里只要求代码跑完、返回值在。
    """
    if expect_ok:
        assert obs.ok, obs.summary
    else:
        assert obs.data.get("result"), obs.summary
    return obs.data["result"]


def _raster(runner, path: str, *, arcpy: bool = False) -> dict:
    """读回输出栅格 → {values, shape, cell, extent, crs, nodata, n_valid}。"""
    code = _READ_ARCPY_RASTER if arcpy else _READ_RASTER
    outcome = runner.run(code.replace("__PATH__", repr(str(path))), timeout=300)
    assert outcome.ok, outcome.error
    grid = outcome.result
    grid["array"] = np.array(grid["values"], dtype="float64")
    return grid


def _assert_same_grid(gdal_grid: dict, arcpy_grid: dict) -> None:
    """输出网格（shape / 像元大小 / extent）逐项相同。"""
    assert gdal_grid["shape"] == arcpy_grid["shape"]
    np.testing.assert_allclose(gdal_grid["cell"], arcpy_grid["cell"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(gdal_grid["extent"], arcpy_grid["extent"], rtol=0, atol=1e-6)


def _assert_same_pixels(gdal_grid: dict, arcpy_grid: dict, *, rtol: float = 1e-6) -> None:
    """逐像元：nodata 掩膜完全一致，有效值在 float32 容差内相等。"""
    gdal_mask = np.isnan(gdal_grid["array"])
    arcpy_mask = np.isnan(arcpy_grid["array"])
    assert np.array_equal(gdal_mask, arcpy_mask), (
        f"nodata 掩膜不同：gdal {int(gdal_mask.sum())} 个 nodata，"
        f"arcpy {int(arcpy_mask.sum())} 个"
    )
    np.testing.assert_allclose(
        gdal_grid["array"][~gdal_mask], arcpy_grid["array"][~arcpy_mask], rtol=rtol, atol=1e-6
    )


# ------------------------------------------------------------- 模板本身
def test_recipe_declares_two_backends_and_renders_both():
    recipe = Recipe.from_yaml(RECIPE)
    assert recipe.backend == ["arcpy", "gdal"]
    params = {
        "zone_path": "z.gdb/zones",
        "raster_path": "d.tif",
        "output_path": "out/zs.tif",
        "zone_field": "zone_id",
        "statistics": "SUM",
    }
    arcpy_code = recipe.render(params)
    gdal_code = recipe.render(params, backend="gdal")

    assert "arcpy.sa.ZonalStatistics(" in arcpy_code and "rasterio" not in arcpy_code
    assert "from rasterio.features import rasterize" in gdal_code and "import arcpy" not in gdal_code
    for key in ("z.gdb/zones", "d.tif", "out/zs.tif", "zone_id", "SUM"):
        assert key in arcpy_code and key in gdal_code
    # DAIR-221：栅格路径要能翻译成 GDAL 认的写法，存在性判据不能再是 os.path.exists
    assert "def _resolve_raster_path(path):" in gdal_code
    assert 'OpenFileGDB:"' in gdal_code
    assert "os.path.exists(str(raster_path))" not in gdal_code
    # 单统计量的拒绝理由两端都写在代码里（不是只写在注释/文档里）
    assert "栅格版只支持单统计量" in arcpy_code and "栅格版只支持单统计量" in gdal_code


def test_gdal_template_keeps_the_three_layer_crs_equality():
    """CRS 判等必须与 zonal_statistics 的 GDAL 段同一套（EPSG → 低置信度 EPSG → pyproj 等价）。"""
    params = {
        "zone_path": "zones.gpkg",
        "raster_path": "dem.tif",
        "output_path": "out/zs.tif",
    }
    code = Recipe.from_yaml(RECIPE).render(params, backend="gdal")

    assert "to_epsg(min_confidence=20)" in code
    assert "equals(raster_crs, ignore_axis_order=True)" in code
    assert "same_crs = zone_epsg is not None and zone_epsg == raster_epsg" in code


# ----------------------------------------------------------- GDAL 侧真跑
@pytest.mark.parametrize("suffix", [".tif", ".img"])
def test_gdal_writes_zone_statistic_to_every_cell(gdal_runner, tmp_path: Path, suffix):
    """每个分区内的像元都写该分区的 SUM，网格与输入栅格逐项相同（arcpy 实测口径）。

    输出格式白名单里的两种都真跑一遍（.tif → GTiff，.img → HFA），不留没验证过的选项。
    """
    _make_scene(gdal_runner, tmp_path)
    obs = _zonal(gdal_runner, tmp_path, f"out/gdal_sum{suffix}")

    assert obs.ok, obs.summary
    result = obs.data["result"]
    assert result["statistics"] == "SUM" and result["zone_field"] == "zone_id"
    assert (result["zone_count"], result["zones_with_values"]) == (3, 3)
    assert result["min"] == 450.0 and result["max"] == 2450.0
    assert result["cell_size"] == 100.0

    grid = _raster(gdal_runner, result["output"])
    assert grid["shape"] == [10, 30]
    assert grid["crs"] == "EPSG:32650"
    assert grid["cell"] == [100.0, 100.0]
    assert grid["extent"] == [0.0, 0.0, 3000.0, 1000.0]
    assert grid["nodata"] == pytest.approx(NODATA_OUT)
    assert grid["n_valid"] == 300, "分区铺满整幅栅格，不该有 nodata"
    for zone_id, column in ((11, 0), (22, 10), (33, 20)):
        block = grid["array"][:, column:column + 10]
        assert np.all(block == REFERENCE[zone_id]), (zone_id, np.unique(block))


@pytest.mark.parametrize("statistics", sorted(REFERENCE_STATS))
def test_gdal_matches_arcpy_reference_for_each_statistic(
    gdal_runner, tmp_path: Path, statistics
):
    """MAXIMUM/MINIMUM/RANGE/STD/MEDIAN 与 arcpy 实测值一致（STD 是总体标准差）。"""
    _make_scene(gdal_runner, tmp_path)
    obs = _zonal(gdal_runner, tmp_path, f"out/gdal_{statistics}.tif", statistics=statistics)
    # RANGE/STD 在三块分区上同值（都是 9 / sqrt(8.25)），输出整幅同值 →
    # 配方自带的 raster_not_constant 会判否；这不是本用例要验的，只看代码跑没跑完。
    result = _result(obs, expect_ok=statistics not in ("RANGE", "STD"))
    grid = _raster(gdal_runner, result["output"])
    block = grid["array"][:, 0:10]
    expected = REFERENCE_STATS[statistics]
    assert np.allclose(block, expected, rtol=1e-6, atol=1e-6), (statistics, np.unique(block))


def test_gdal_output_window_follows_zone_bbox_not_the_whole_raster(gdal_runner, tmp_path: Path):
    """输出范围 = 分区 bbox（外扩到格线、裁到栅格），分区之间的缺口写 nodata。"""
    zones = [(11, "Z1", 0), (33, "Z3", 2000)]
    _make_scene(gdal_runner, tmp_path, boxes=zones, width=40)
    obs = _zonal(gdal_runner, tmp_path, "out/gdal_gap.tif")

    assert obs.ok, obs.summary
    grid = _raster(gdal_runner, obs.data["result"]["output"])
    # arcpy 实测：两块分区 bbox 是 0..3000，栅格有 40 列 → 输出裁到分区 bbox
    assert grid["shape"] == [10, 30]
    assert grid["extent"] == [0.0, 0.0, 3000.0, 1000.0]
    assert np.all(grid["array"][:, 0:10] == REFERENCE[11])
    assert np.all(grid["array"][:, 20:30] == REFERENCE[33])
    assert np.all(np.isnan(grid["array"][:, 10:20])), "分区之间的缺口必须是 nodata"


def test_gdal_window_snaps_outward_and_skips_the_boundary_centre_cell(
    gdal_runner, tmp_path: Path
):
    """分区 bbox 不齐格：输出按格线外扩，但边界正好压在格心上的那列不算（pitfall 已写）。

    arcpy 实测：分区 550..1250 → 输出 extent 500..1300（8 列），并把 x=550 那列也算进去
    （SUM=680）。GDAL 的栅格化规则是左闭右开，x=550 那列不算 → SUM=630、该列 nodata。
    这条差异只出现在「边界正好压格心」时，写进 pitfalls，这里锁住 GDAL 侧的真实行为。
    """
    zones = [(11, "Z1", [[550, 0], [1250, 0], [1250, 1000], [550, 1000]])]
    _make_scene(gdal_runner, tmp_path, boxes=zones, width=40)
    obs = _zonal(gdal_runner, tmp_path, "out/gdal_unaligned.tif")

    result = _result(obs, expect_ok=False)  # 单分区 → 输出整幅同值，raster_not_constant 会判否
    grid = _raster(gdal_runner, result["output"])
    assert grid["shape"] == [10, 8]
    assert grid["extent"] == [500.0, 0.0, 1300.0, 1000.0]
    assert np.all(np.isnan(grid["array"][:, 0])), "x=500..600（格心 550）不在分区内"
    assert np.all(grid["array"][:, 1:8] == 630.0), np.unique(grid["array"][:, 1:8])


def test_gdal_keeps_the_input_grid_for_non_square_cells(gdal_runner, tmp_path: Path):
    """像元非方形（100 × 50 m）时 GDAL 保持输入网格。

    arcpy 实测会把输出拉成方形像元（pop_density.tif 5.0774×5.0587 m → 输入 2512 行、
    输出 2503 行，见 pitfalls）。GDAL 侧不跟着拉方：输出网格 = 输入网格的窗口，
    否则下游按像元做分母/掩膜运算会对不上。
    """
    zones = tmp_path / "zones.gpkg"
    raster = tmp_path / "wide_cell.tif"
    made = gdal_runner.run(
        "\n".join(
            [
                "import geopandas as gpd",
                "import numpy as np",
                "import rasterio",
                "from rasterio.transform import from_origin",
                "from shapely.geometry import Polygon",
                "zones = gpd.GeoDataFrame({'zone_id': [11]},",
                "    geometry=[Polygon([(0, 0), (1000, 0), (1000, 250), (0, 250)])],",
                "    crs='EPSG:32650')",
                "zones.to_file(__ZONES__, driver='GPKG', engine='pyogrio', layer='zones')",
                "data = np.tile(np.arange(20, dtype='float32'), (10, 1))",
                "with rasterio.open(__RASTER__, 'w', driver='GTiff', height=10, width=20, count=1,",
                "                   dtype='float32', crs='EPSG:32650',",
                "                   transform=from_origin(0, 500, 100, 50)) as ds:",
                "    ds.write(data, 1)",
                "set_result({'ok': True})",
            ]
        )
        .replace("__ZONES__", repr(str(zones)))
        .replace("__RASTER__", repr(str(raster))),
        timeout=180,
    )
    assert made.ok, made.error

    obs = _zonal(
        gdal_runner, tmp_path, "out/gdal_wide.tif",
        zone_path=str(zones), raster_path=str(raster),
    )
    result = _result(obs, expect_ok=False)  # 单分区 → 整幅同值，raster_not_constant 判否
    grid = _raster(gdal_runner, result["output"])
    assert grid["cell"] == [100.0, 50.0], "GDAL 不拉方：保持输入的 100×50 像元"
    assert grid["shape"] == [5, 10]
    assert grid["extent"] == [0.0, 0.0, 1000.0, 250.0]
    assert np.all(grid["array"] == 225.0), np.unique(grid["array"])


def test_gdal_nodata_cells_inside_zone_are_excluded_from_sum_but_still_filled(
    gdal_runner, tmp_path: Path
):
    """输入 nodata 不计入统计，但落在分区内的那几格写的是分区统计值（arcpy 实测）。"""
    _make_scene(gdal_runner, tmp_path, nodata_cols=(5,))
    obs = _zonal(gdal_runner, tmp_path, "out/gdal_nodata.tif")

    assert obs.ok, obs.summary
    grid = _raster(gdal_runner, obs.data["result"]["output"])
    # Z1 覆盖列 0..9，其中列 5 整列 nodata：SUM = 450 - 5*10 = 400
    assert np.all(grid["array"][:, 0:10] == 400.0), np.unique(grid["array"][:, 0:10])
    assert np.all(grid["array"][:, 10:20] == 1450.0)
    assert np.all(grid["array"][:, 20:30] == 2450.0)
    assert not np.isnan(grid["array"]).any(), "分区内的 nodata 像元也要写分区统计值"


def test_gdal_zone_with_only_nodata_cells_stays_nodata(gdal_runner, tmp_path: Path):
    """整块只有 nodata 像元的分区：输出那几格仍是 nodata，不写 0（arcpy 实测）。"""
    zones = [(77, "Z7", [[500, 0], [600, 0], [600, 1000], [500, 1000]])]
    _make_scene(gdal_runner, tmp_path, boxes=zones, width=40, nodata_cols=(5,))
    obs = _zonal(gdal_runner, tmp_path, "out/gdal_only_nodata.tif")

    result = _result(obs, expect_ok=False)  # 整幅 nodata → 常量，raster_not_constant 会判否
    assert result["zone_count"] == 1 and result["zones_with_values"] == 0
    grid = _raster(gdal_runner, result["output"])
    assert grid["shape"] == [10, 1]
    assert np.isnan(grid["array"]).all()


def test_gdal_accepts_text_zone_field(gdal_runner, tmp_path: Path):
    """文本分区字段能用（arcpy 的 ZonalStatistics 也接受文本，两端一致）。"""
    made = gdal_runner.run(
        "\n".join(
            [
                "import geopandas as gpd",
                "from shapely.geometry import Polygon",
                "zones = gpd.GeoDataFrame({'code': ['A', 'B']},",
                "    geometry=[Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)]),",
                "              Polygon([(1000, 0), (2000, 0), (2000, 1000), (1000, 1000)])],",
                "    crs='EPSG:32650')",
                "zones.to_file(__ZONES__, driver='GPKG', engine='pyogrio', layer='zones')",
                "set_result({'ok': True})",
            ]
        ).replace("__ZONES__", repr(str(tmp_path / "text_zones.gpkg"))),
        timeout=180,
    )
    assert made.ok, made.error
    _make_scene(gdal_runner, tmp_path)
    obs = _zonal(
        gdal_runner,
        tmp_path,
        "out/gdal_text.tif",
        zone_path=str(tmp_path / "text_zones.gpkg"),
        zone_field="code",
    )

    assert obs.ok, obs.summary
    grid = _raster(gdal_runner, obs.data["result"]["output"])
    assert np.all(grid["array"][:, 0:10] == REFERENCE[11])
    assert np.all(grid["array"][:, 10:20] == REFERENCE[22])


def test_gdal_rejects_float_zone_field(gdal_runner, tmp_path: Path):
    """浮点分区字段两端都拒绝（arcpy 实测 ERROR 000864/000889），GDAL 版给同样的准话。"""
    made = gdal_runner.run(
        "\n".join(
            [
                "import geopandas as gpd",
                "from shapely.geometry import Polygon",
                "zones = gpd.GeoDataFrame({'zone_id': [11.5, 22.5]},",
                "    geometry=[Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)]),",
                "              Polygon([(1000, 0), (2000, 0), (2000, 1000), (1000, 1000)])],",
                "    crs='EPSG:32650')",
                "zones.to_file(__ZONES__, driver='GPKG', engine='pyogrio', layer='zones')",
                "set_result({'ok': True})",
            ]
        ).replace("__ZONES__", repr(str(tmp_path / "float_zones.gpkg"))),
        timeout=180,
    )
    assert made.ok, made.error
    _make_scene(gdal_runner, tmp_path)
    obs = _zonal(
        gdal_runner,
        tmp_path,
        "out/gdal_float.tif",
        zone_path=str(tmp_path / "float_zones.gpkg"),
    )

    assert not obs.ok
    assert "分区字段必须是整型或文本" in obs.summary
    assert not (tmp_path / "out/gdal_float.tif").exists()


def test_gdal_rejects_unknown_zone_field(gdal_runner, tmp_path: Path):
    _make_scene(gdal_runner, tmp_path)
    obs = _zonal(gdal_runner, tmp_path, "out/gdal_no_field.tif", zone_field="nope")

    assert not obs.ok
    assert "分区数据缺少字段: nope" in obs.summary and "zone_id" in obs.summary
    assert not (tmp_path / "out/gdal_no_field.tif").exists()


@pytest.mark.parametrize("statistics", ["SUM;MEAN", "MEAN;SUM"])
def test_gdal_rejects_multi_statistics(gdal_runner, tmp_path: Path, statistics):
    """栅格版只支持单统计量：多值明确报错，不留成第二个 D1。"""
    _make_scene(gdal_runner, tmp_path)
    obs = _zonal(gdal_runner, tmp_path, "out/gdal_multi.tif", statistics=statistics)

    assert not obs.ok
    assert "栅格版只支持单统计量" in obs.summary and "zonal_statistics" in obs.summary
    assert not (tmp_path / "out/gdal_multi.tif").exists()


@pytest.mark.parametrize(
    "statistics,want",
    [("COUNT", "COUNT 是表版的固定输出列"), ("PERCENTILE", "percentile_values"), ("ALL", "取值域里没有它")],
)
def test_gdal_rejects_statistics_outside_the_domain(
    gdal_runner, tmp_path: Path, statistics, want
):
    """COUNT/ALL/PERCENTILE 不在 ZonalStatistics 的取值域里（arcpy 实测 ERROR 000864）。"""
    _make_scene(gdal_runner, tmp_path)
    obs = _zonal(gdal_runner, tmp_path, "out/gdal_badstat.tif", statistics=statistics)

    assert not obs.ok
    assert want in obs.summary and "可用" in obs.summary
    assert not (tmp_path / "out/gdal_badstat.tif").exists()


@pytest.mark.parametrize(
    "output_path,want",
    [
        ("out/x.gdb/zs", "GDAL 后端写不了 gdb 里的栅格"),
        ("out/x.asc", "只支持 .tif / .tiff / .img"),
    ],
)
def test_gdal_rejects_output_formats_it_cannot_write(
    gdal_runner, tmp_path: Path, output_path, want
):
    _make_scene(gdal_runner, tmp_path)
    obs = _zonal(gdal_runner, tmp_path, output_path)

    assert not obs.ok
    assert want in obs.summary
    assert not (tmp_path / "out/x.gdb").exists()


def test_gdal_rejects_zones_that_miss_the_raster(gdal_runner, tmp_path: Path):
    """所有分区都在栅格外：arcpy 报 ERROR 010568，GDAL 版给明确 ValueError。"""
    zones = [(99, "far", [[5000, 5000], [6000, 5000], [6000, 6000], [5000, 6000]])]
    _make_scene(gdal_runner, tmp_path, boxes=zones)
    obs = _zonal(gdal_runner, tmp_path, "out/gdal_far.tif")

    assert not obs.ok
    assert "分区范围与栅格范围不重叠" in obs.summary and "010568" in obs.summary
    assert not (tmp_path / "out/gdal_far.tif").exists()


def test_gdal_empty_zone_field_uses_objectid(gdal_runner, tmp_path: Path):
    """空 zone_field = arcpy 的 OBJECTID：取源数据 FID，结果里 zone_field 写 OBJECTID。"""
    _make_scene(gdal_runner, tmp_path)
    obs = _zonal(gdal_runner, tmp_path, "out/gdal_oid.tif", zone_field="")

    assert obs.ok, obs.summary
    assert obs.data["result"]["zone_field"] == "OBJECTID"
    grid = _raster(gdal_runner, obs.data["result"]["output"])
    assert np.all(grid["array"][:, 0:10] == REFERENCE[11])


def test_gdal_accepts_esri_wkt_zone_crs(gdal_runner, tmp_path: Path):
    """同 EPSG、不同 WKT 写法（ESRI 别名）不算不一致——三层判等的第 2/3 层要真用上。"""
    made = gdal_runner.run(
        "\n".join(
            [
                "import geopandas as gpd",
                "import pyproj",
                "from shapely.geometry import Polygon",
                "_alias = pyproj.CRS.from_wkt(pyproj.CRS('EPSG:32650').to_wkt(version='WKT1_ESRI'))",
                "zones = gpd.GeoDataFrame({'zone_id': [11, 22]},",
                "    geometry=[Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)]),",
                "              Polygon([(1000, 0), (2000, 0), (2000, 1000), (1000, 1000)])],",
                "    crs=_alias)",
                "zones.to_file(__ZONES__, driver='GPKG', engine='pyogrio', layer='zones')",
                "set_result({'ok': True, 'epsg': zones.crs.to_epsg(),",
                "            'low': zones.crs.to_epsg(min_confidence=20)})",
            ]
        ).replace("__ZONES__", repr(str(tmp_path / "esri_zones.gpkg"))),
        timeout=180,
    )
    assert made.ok, made.error
    _make_scene(gdal_runner, tmp_path)
    obs = _zonal(
        gdal_runner,
        tmp_path,
        "out/gdal_esri.tif",
        zone_path=str(tmp_path / "esri_zones.gpkg"),
    )

    assert obs.ok, obs.summary
    grid = _raster(gdal_runner, obs.data["result"]["output"])
    assert np.all(grid["array"][:, 0:10] == REFERENCE[11])


def test_gdal_rejects_different_crs(gdal_runner, tmp_path: Path):
    """不同坐标系仍然拒绝（不是「都有 CRS 就放行」）。"""
    made = gdal_runner.run(
        "\n".join(
            [
                "import geopandas as gpd",
                "from shapely.geometry import Polygon",
                "zones = gpd.GeoDataFrame({'zone_id': [11]},",
                "    geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs='EPSG:4490')",
                "zones.to_file(__ZONES__, driver='GPKG', engine='pyogrio', layer='zones')",
                "set_result({'ok': True})",
            ]
        ).replace("__ZONES__", repr(str(tmp_path / "wgs_zones.gpkg"))),
        timeout=180,
    )
    assert made.ok, made.error
    _make_scene(gdal_runner, tmp_path)
    obs = _zonal(
        gdal_runner,
        tmp_path,
        "out/gdal_badcrs.tif",
        zone_path=str(tmp_path / "wgs_zones.gpkg"),
    )

    assert not obs.ok
    assert "坐标系不一致" in obs.summary
    assert not (tmp_path / "out/gdal_badcrs.tif").exists()


# ------------------------------------------------ raster_not_constant 的 GDAL 口径
def test_gdal_raster_check_reads_min_max_without_arcpy(gdal_runner, tmp_path: Path):
    """``raster_not_constant`` 在 GDAL 后端要自己读极值（arcpy Describe 在 .venv-gdal 里没有）。"""
    _make_scene(gdal_runner, tmp_path)
    obs = _zonal(gdal_runner, tmp_path, "out/gdal_check.tif")

    assert obs.ok, obs.summary
    kinds = {a["kind"]: a for a in obs.data["assertions"]}
    assert kinds["raster_not_constant"]["ok"] is True
    assert kinds["raster_not_constant"]["detail"].startswith("range=(450.0, 2450.0)")


def test_gdal_raster_check_flags_constant_and_all_nodata(gdal_runner, tmp_path: Path):
    """常量栅格、整幅 nodata 都判「不是非常量」——否则这条断言会被空图骗过去。"""
    paths = {}
    for name, fill in (("constant.tif", 7.0), ("empty.tif", -9999.0)):
        path = tmp_path / name
        made = gdal_runner.run(
            "\n".join(
                [
                    "import numpy as np",
                    "import rasterio",
                    "from rasterio.transform import from_origin",
                    "with rasterio.open(__PATH__, 'w', driver='GTiff', height=4, width=4, count=1,",
                    "                   dtype='float32', crs='EPSG:32650',",
                    "                   transform=from_origin(0, 400, 100, 100), nodata=-9999.0) as ds:",
                    "    ds.write(np.full((4, 4), __FILL__, dtype='float32'), 1)",
                    "set_result({'ok': True})",
                ]
            )
            .replace("__PATH__", repr(str(path)))
            .replace("__FILL__", repr(fill)),
            timeout=180,
        )
        assert made.ok, made.error
        paths[name] = str(path)

    ok, detail = gdal_raster_check(gdal_runner, "raster_not_constant", paths["constant.tif"], None)
    assert ok is False and "range=(7.0, 7.0)" in detail
    ok, detail = gdal_raster_check(gdal_runner, "raster_not_constant", paths["empty.tif"], None)
    assert ok is False and "range=(None, None)" in detail
    # 不接管的 kind 仍然返回 None（走内置实现）
    assert gdal_raster_check(gdal_runner, "file_exists", paths["constant.tif"], None) is None


# ------------------------------------------- GDAL 侧：.gdb 里的栅格（DAIR-221）
def _baseline_grid(gdal_runner, tmp_path: Path, gdb_rasters: dict, name: str) -> dict:
    """同一份栅格的 .tif 基线输出（逐像元对比的参照）。"""
    obs = _zonal(gdal_runner, tmp_path, name, raster_path=gdb_rasters["raster"])
    assert obs.ok, obs.summary
    return _raster(gdal_runner, obs.data["result"]["output"])


def test_gdb_raster_layer_path_works_with_both_separators(gdal_runner, tmp_path, gdb_rasters):
    """arcpy 侧交出来的 ``x.gdb\层名`` 不能再报「栅格数据不存在」（数据其实在库里）。

    两种分隔符都要跑通，输出栅格与同一份栅格的 .tif 基线逐像元相等。
    """
    baseline = _baseline_grid(gdal_runner, tmp_path, gdb_rasters, "out/gdb_base.tif")
    for sep, tag in (("\\", "backslash"), ("/", "slash")):
        obs = _zonal(
            gdal_runner, tmp_path, f"out/gdb_layer_{tag}.tif",
            raster_path=gdb_rasters["single"] + sep + "dem",
        )
        assert obs.ok, obs.summary
        assert "栅格数据不存在" not in obs.data["stdout"]
        grid = _raster(gdal_runner, obs.data["result"]["output"])
        _assert_same_grid(grid, baseline)
        _assert_same_pixels(grid, baseline)


def test_plain_gdb_directory_with_a_raster_inside_still_opens_as_a_file(gdal_runner, tmp_path):
    """``.gdb`` 结尾的**普通目录**里放散文件：按文件路径打开，不翻 OpenFileGDB。

    与 ``test_zonal_statistics_gdal.py`` 的同名用例同形态（DAIR-224）。
    """
    _, raster = _make_scene(gdal_runner, tmp_path)
    baseline = _raster(
        gdal_runner,
        _result(_zonal(gdal_runner, tmp_path, "out/plain_gdb_base.tif"))["output"],
    )
    plain = tmp_path / "plain.gdb"
    plain.mkdir()
    shutil.copyfile(raster, plain / "dem.tif")

    obs = _zonal(gdal_runner, tmp_path, "out/plain_gdb.tif",
                 raster_path=str(plain) + "\\dem.tif")

    assert obs.ok, obs.summary
    assert "栅格数据不存在" not in obs.data["stdout"]
    grid = _raster(gdal_runner, obs.data["result"]["output"])
    _assert_same_grid(grid, baseline)
    _assert_same_pixels(grid, baseline)


def test_single_raster_gdb_container_is_used_automatically(gdal_runner, tmp_path, gdb_rasters):
    """只给库本身且库里恰好一份栅格时直接取用（单栅格库 rasterio 本来就打得开）。"""
    baseline = _baseline_grid(gdal_runner, tmp_path, gdb_rasters, "out/gdb_container_base.tif")
    obs = _zonal(gdal_runner, tmp_path, "out/gdb_container.tif",
                 raster_path=gdb_rasters["single"])

    assert obs.ok, obs.summary
    assert "栅格数据不存在" not in obs.data["stdout"]
    grid = _raster(gdal_runner, obs.data["result"]["output"])
    _assert_same_pixels(grid, baseline)


def test_multi_raster_gdb_container_asks_for_a_raster_by_name(gdal_runner, tmp_path, gdb_rasters):
    """多栅格库只给库本体 → 人话错误（列出栅格名 + 怎么指定），不是 No such file。"""
    obs = _zonal(gdal_runner, tmp_path, "out/gdb_multi.tif", raster_path=gdb_rasters["multi"])

    assert not obs.ok
    assert "dem、dem_shift" in obs.summary
    assert "请指明栅格" in obs.summary and "No such file" not in obs.summary


@pytest.mark.parametrize("form", ["missing_file", "missing_gdb"])
def test_missing_raster_still_reports_it_does_not_exist(gdal_runner, tmp_path: Path, form):
    """存在性判据换成「能不能打开」之后，真不存在时仍要说清是哪个路径。"""
    _make_scene(gdal_runner, tmp_path)
    path = (str(tmp_path / "nope.tif") if form == "missing_file"
            else str(tmp_path / "nope.gdb") + "\\dem")
    obs = _zonal(gdal_runner, tmp_path, "out/missing.tif", raster_path=path)

    assert not obs.ok
    assert "栅格数据不存在" in obs.summary and "nope" in obs.summary


# --------------------------------------------------------------- 双端一致
@pytest.mark.parametrize("statistics", ["SUM", "MEAN"])
def test_arcpy_and_gdal_agree_pixel_by_pixel(
    gdal_runner, arcpy_runner, tmp_path: Path, statistics
):
    """同一份 .gdb 分区 + 同一份 .tif：网格相同、nodata 掩膜相同、逐像元值相等。"""
    zones_fc = _arcpy_zones(arcpy_runner, tmp_path)
    _make_scene(gdal_runner, tmp_path)

    gdal_obs = _zonal(
        gdal_runner, tmp_path, f"out/gdal_{statistics}.tif",
        zone_path=zones_fc, statistics=statistics,
    )
    arcpy_obs = _zonal(
        arcpy_runner, tmp_path, f"out/arcpy_{statistics}.tif",
        zone_path=zones_fc, statistics=statistics,
    )
    assert gdal_obs.ok, f"gdal/{statistics}: {gdal_obs.summary}"
    assert arcpy_obs.ok, f"arcpy/{statistics}: {arcpy_obs.summary}"

    gdal_grid = _raster(gdal_runner, gdal_obs.data["result"]["output"])
    arcpy_grid = _raster(arcpy_runner, arcpy_obs.data["result"]["output"], arcpy=True)

    _assert_same_grid(gdal_grid, arcpy_grid)
    assert gdal_grid["crs"] == "EPSG:32650" and arcpy_grid["crs"] == 32650
    assert gdal_grid["n_valid"] == arcpy_grid["n_valid"] == 300
    _assert_same_pixels(gdal_grid, arcpy_grid)


def test_arcpy_and_gdal_agree_on_window_gap_and_nodata(gdal_runner, arcpy_runner, tmp_path: Path):
    """带缺口 + 分区内 nodata 列的场景：输出范围、掩膜、逐像元值三样全对得上。"""
    zones = [(11, "Z1", 0), (33, "Z3", 2000)]
    zones_fc = _arcpy_zones(arcpy_runner, tmp_path, boxes=zones)
    _make_scene(gdal_runner, tmp_path, boxes=zones, width=40, nodata_cols=(5,))

    gdal_obs = _zonal(gdal_runner, tmp_path, "out/gdal_gap.tif", zone_path=zones_fc)
    arcpy_obs = _zonal(arcpy_runner, tmp_path, "out/arcpy_gap.tif", zone_path=zones_fc)
    assert gdal_obs.ok, f"gdal: {gdal_obs.summary}"
    assert arcpy_obs.ok, f"arcpy: {arcpy_obs.summary}"

    gdal_grid = _raster(gdal_runner, gdal_obs.data["result"]["output"])
    arcpy_grid = _raster(arcpy_runner, arcpy_obs.data["result"]["output"], arcpy=True)

    _assert_same_grid(gdal_grid, arcpy_grid)
    assert gdal_grid["shape"] == [10, 30], "输出裁到分区 bbox（0..3000），不是整幅 40 列"
    _assert_same_pixels(gdal_grid, arcpy_grid)
    # 缺口（列 10..19）两端都 nodata；分区内 nodata 列（列 5）两端都写分区统计值
    assert np.isnan(gdal_grid["array"][:, 10:20]).all()
    assert np.all(gdal_grid["array"][:, 5] == 400.0)


def test_arcpy_template_guards_follow_the_probed_domain(arcpy_runner, gdal_runner, tmp_path: Path):
    """arcpy 侧的三条守卫都按实测取值域来（一个内核跑完，不重复起进程）：

    1. 多统计量：模板直接报「只支持单统计量」，不甩 ERROR 000864；
    2. 文本分区字段：**能用**（实测 ZonalStatistics 接受文本，旧守卫反而拒了它）；
    3. 浮点分区字段：给准话（实测 ERROR 000864/000889），不让人去读 arcpy 的报错。
    """
    _make_scene(gdal_runner, tmp_path)

    # 1) 多统计量
    zones_fc = _arcpy_zones(arcpy_runner, tmp_path)
    multi = _zonal(
        arcpy_runner, tmp_path, "out/arcpy_multi.tif", zone_path=zones_fc, statistics="SUM;MEAN"
    )
    assert not multi.ok
    assert "栅格版只支持单统计量" in multi.summary
    assert not (tmp_path / "out/arcpy_multi.tif").exists()

    # 2) 文本分区字段
    text_boxes = [("A", "Z1", _box(0, 1000)), ("B", "Z2", _box(1000, 2000))]
    text_fc = _arcpy_zones(
        arcpy_runner, tmp_path, boxes=text_boxes, field_type="TEXT", name="zsr_text.gdb"
    )
    text_obs = _zonal(arcpy_runner, tmp_path, "out/arcpy_text.tif", zone_path=text_fc)
    assert text_obs.ok, text_obs.summary
    text_grid = _raster(arcpy_runner, text_obs.data["result"]["output"], arcpy=True)
    assert np.all(text_grid["array"][:, 0:10] == REFERENCE[11])
    assert np.all(text_grid["array"][:, 10:20] == REFERENCE[22])

    # 3) 浮点分区字段
    float_boxes = [(11.5, "Z1", _box(0, 1000)), (22.5, "Z2", _box(1000, 2000))]
    float_fc = _arcpy_zones(
        arcpy_runner, tmp_path, boxes=float_boxes, field_type="DOUBLE", name="zsr_float.gdb"
    )
    float_obs = _zonal(arcpy_runner, tmp_path, "out/arcpy_float.tif", zone_path=float_fc)
    assert not float_obs.ok
    assert "分区字段必须是整型或文本" in float_obs.summary and "000864" not in float_obs.summary
    assert not (tmp_path / "out/arcpy_float.tif").exists()


#: 重叠分区：A 先出现（x 0..600 = 列 0..5），B 后出现（x 400..1000 = 列 4..9），重叠列 4、5
OVERLAP_BOXES = [
    (11, "A", [[0, 0], [600, 0], [600, 1000], [0, 1000]]),
    (22, "B", [[400, 0], [1000, 0], [1000, 1000], [400, 1000]]),
]


def test_arcpy_and_gdal_agree_on_overlapping_zones_first_wins(
    gdal_runner, arcpy_runner, tmp_path: Path
):
    """重叠分区：arcpy **先到先得**，GDAL 版照此实现（``rasterize`` 反序喂几何）。

    A（先出现）x 0..600 = 列 0..5，全占 SUM = (0+1+2+3+4+5)×10 = 150；
    B（后出现）x 400..1000 = 列 4..9，全占 SUM = (4+5+6+7+8+9)×10 = 390。
    重叠列 4、5 归 A ⇒ B 的统计只用它实际占到的列 6..9 = (6+7+8+9)×10 = **300**（不是 390）。

    把实现里的 ``reversed(shapes)`` 去掉，重叠列会归 B → 列 0..3 = 60、列 4..9 = 390，
    这条用例必须红（此前没有任何用例会红）。
    """
    zones_fc = _arcpy_zones(arcpy_runner, tmp_path, boxes=OVERLAP_BOXES)
    _make_scene(gdal_runner, tmp_path, boxes=OVERLAP_BOXES, width=10, rows=10)

    gdal_obs = _zonal(gdal_runner, tmp_path, "out/gdal_overlap.tif", zone_path=zones_fc)
    arcpy_obs = _zonal(arcpy_runner, tmp_path, "out/arcpy_overlap.tif", zone_path=zones_fc)
    assert gdal_obs.ok, f"gdal: {gdal_obs.summary}"
    assert arcpy_obs.ok, f"arcpy: {arcpy_obs.summary}"

    gdal_grid = _raster(gdal_runner, gdal_obs.data["result"]["output"])
    arcpy_grid = _raster(arcpy_runner, arcpy_obs.data["result"]["output"], arcpy=True)

    _assert_same_grid(gdal_grid, arcpy_grid)
    assert gdal_grid["n_valid"] == arcpy_grid["n_valid"] == 100
    _assert_same_pixels(gdal_grid, arcpy_grid)

    # 先出现的 A 保留重叠列 4、5；后出现的 B 只统计它实际占到的列 6..9
    for grid in (gdal_grid, arcpy_grid):
        assert np.all(grid["array"][:, 0:6] == 150.0), np.unique(grid["array"][:, 0:6])
        assert np.all(grid["array"][:, 6:10] == 300.0), np.unique(grid["array"][:, 6:10])

    for obs in (gdal_obs, arcpy_obs):
        result = obs.data["result"]
        assert (result["min"], result["max"]) == (150.0, 300.0), result


# ------------------------------------------------- 缺后端的契约（P0-1a）
class _FakeRunner:
    """假执行器：只声明给定后端，run() 只记账、不真跑代码。"""

    def __init__(self, backends):
        self.backends = set(backends)
        self.codes: list[str] = []

    def run(self, code, *, timeout=None, workspace=None):
        self.codes.append(code)
        return SimpleNamespace(
            ok=True, stdout="", stderr="", result={"ok": True}, error=None, display_images=[]
        )


def test_arcpy_only_machine_runs_the_arcpy_template(tmp_path: Path):
    """本机只有 arcpy 时照声明顺序落到 arcpy 模板（配方不因为加了 gdal 就要求 GDAL）。"""
    runner = _FakeRunner({"arcpy"})
    lib = RecipeLibrary(code_runner=runner)
    lib.load()
    ctx = SimpleNamespace(workspace=tmp_path, code_runner=runner, exec_timeout=10)

    obs = lib.run(
        "zonal_statistics_raster",
        {"zone_path": "zones.gpkg", "raster_path": "dem.tif", "output_path": "out/zs.tif"},
        ctx,
    )

    assert obs.data["backend"] == "arcpy"
    # 第一段被执行的代码就是 arcpy 模板；后续的 codes 是断言层的 Describe 小段
    assert "arcpy.sa.ZonalStatistics(" in runner.codes[0]
    assert not any("rasterize" in code for code in runner.codes)


def test_no_backend_available_reports_and_executes_nothing(tmp_path: Path):
    """两个后端都没有：明确报 BackendUnavailable，一行代码都不执行（P0-1a 契约）。"""
    runner = _FakeRunner({"none"})
    lib = RecipeLibrary(code_runner=runner)
    lib.load()
    ctx = SimpleNamespace(workspace=tmp_path, code_runner=runner, exec_timeout=10)

    obs = lib.run(
        "zonal_statistics_raster",
        {"zone_path": "zones.gpkg", "raster_path": "dem.tif", "output_path": "out/zs.tif"},
        ctx,
    )

    assert not obs.ok
    assert obs.error["type"] == "BackendUnavailable"
    assert "需 ArcGIS（arcpy）" in obs.summary and "需 GDAL 后端" in obs.summary
    assert not runner.codes, "缺后端时不该把任何模板交给执行器"

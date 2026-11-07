# -*- coding: utf-8 -*-
"""zonal_statistics 的 GDAL 版真跑 + 双端一致（网格对齐的大面上 SUM/MEAN <1%、COUNT/AREA
严格相等；5 m 小面在 4 个 9×9 偏移半像元构造上逐分区卡 MEAN 相对差 ≤ 3%（比跨后端
p95 口径更严，见 zonal_statistics.yaml:393），COUNT 不承诺相等 —— 见文末 DAIR-139 用例）。

测试数据（EPSG:32650，米）：

    分区面  三块 1 km × 1 km 方块，x 依次相接（Z1 0..1000 / Z2 1000..2000 / Z3 2000..3000）
    栅格    100 m 像元，30 列 × 10 行，像元值 = 列号（0..29），nodata = -9999

分区与栅格网格对齐，格心（50/150/…）不会压在分区边界上 —— 边界正好压格心时两端的取舍
不同（见配方 pitfalls），那是本批验收口径「逐分区 SUM/MEAN 差 <1%」唯一可复现的构造。

``REFERENCE`` 那组数是**真 arcpy 跑出来的**（ArcGIS Pro 3.6 的 ZonalStatisticsAsTable），
不是从 GDAL 侧抄回来的。它同时钉住四件事：输出字段名（zone_id/COUNT/AREA/SUM/MEAN）、
COUNT 只数有效像元、AREA = 有效像元数 × 像元面积、没有像元的分区不出行。

缺环境就 skip（与 ``test_aggregate_to_zones_gdal.py`` 同一套口径）：
``.venv-gdal`` 不在 → 所有真 GDAL 用例 skip；ArcGIS Pro 找不到 → 只有双端用例 skip。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from gis_cli.recipes.schema import Recipe
from gis_cli.recipes.store import RecipeLibrary
from gis_cli.runner.gdal_runner import GdalCodeRunner, default_python_executable
from gis_cli.runtime.code_runner import CodeRunner

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "src/gis_cli/recipes/builtin/zonal_statistics.yaml"

#: (zone_id, name, 左下角 x)：三块 1 km × 1 km 方块，y 都是 0..1000
ZONE_SPECS = [(11, "Z1", 0), (22, "Z2", 1000), (33, "Z3", 2000)]

#: arcpy（Pro 3.6）实测值：每块分区 100 个像元、AREA 1e6 m²
REFERENCE = {
    11: {"COUNT": 100.0, "AREA": 1_000_000.0, "SUM": 450.0, "MEAN": 4.5},
    22: {"COUNT": 100.0, "AREA": 1_000_000.0, "SUM": 1450.0, "MEAN": 14.5},
    33: {"COUNT": 100.0, "AREA": 1_000_000.0, "SUM": 2450.0, "MEAN": 24.5},
}

_MAKE_INPUTS = "\n".join(
    [
        "import geopandas as gpd",
        "import numpy as np",
        "import rasterio",
        "from rasterio.transform import from_origin",
        "from shapely.geometry import Polygon",
        "_specs = __SPECS__",
        "zones = gpd.GeoDataFrame({'zone_id': [s[0] for s in _specs],",
        "                          'name': [s[1] for s in _specs]},",
        "    geometry=[Polygon([(s[2], 0), (s[2] + 1000, 0), (s[2] + 1000, 1000), (s[2], 1000)])",
        "              for s in _specs],",
        "    crs=__CRS__)",
        "zones.to_file(__ZONES__, driver='GPKG', engine='pyogrio', layer='zones')",
        "data = np.tile(np.arange(30, dtype='float32'), (10, 1))",
        "with rasterio.open(__RASTER__, 'w', driver='GTiff', height=10, width=30, count=1,",
        "                   dtype='float32', crs=__CRS__,",
        "                   transform=from_origin(0, 1000, 100, 100), nodata=-9999.0) as ds:",
        "    ds.write(data, 1)",
        "set_result({'zones': str(__ZONES__), 'raster': str(__RASTER__)})",
    ]
)

#: 边界情形：一块正常分区 + 一块不含任何格心的小面 + 一块完全在栅格外的面，
#: 栅格第 3 列（x 200..300）整列是 nodata。
_MAKE_EDGE_INPUTS = "\n".join(
    [
        "import geopandas as gpd",
        "import numpy as np",
        "import rasterio",
        "from rasterio.transform import from_origin",
        "from shapely.geometry import Polygon",
        "_boxes = [",
        "    (11, [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]),",
        "    (44, [(150, 150), (160, 150), (160, 160), (150, 160)]),",
        "    (55, [(5000, 5000), (6000, 5000), (6000, 6000), (5000, 6000)]),",
        "]",
        "zones = gpd.GeoDataFrame({'zone_id': [b[0] for b in _boxes]},",
        "    geometry=[Polygon(b[1]) for b in _boxes], crs='EPSG:32650')",
        "zones.to_file(__ZONES__, driver='GPKG', engine='pyogrio', layer='zones')",
        "data = np.tile(np.arange(30, dtype='float32'), (10, 1))",
        "data[:, 2] = -9999.0",
        "with rasterio.open(__RASTER__, 'w', driver='GTiff', height=10, width=30, count=1,",
        "                   dtype='float32', crs='EPSG:32650',",
        "                   transform=from_origin(0, 1000, 100, 100), nodata=-9999.0) as ds:",
        "    ds.write(data, 1)",
        "set_result({'zones': str(__ZONES__), 'raster': str(__RASTER__)})",
    ]
)

#: 同一块地的 ESRI 写法 WKT（EPSG 码相同、WKT 文本不同）：坐标系比较不能是字符串比
_MAKE_ESRI_WKT_ZONES = "\n".join(
    [
        "import geopandas as gpd",
        "import pyproj",
        "from shapely.geometry import Polygon",
        "_base = pyproj.CRS('EPSG:32650')",
        "_alias = pyproj.CRS.from_wkt(_base.to_wkt(version='WKT1_ESRI'))",
        "_specs = __SPECS__",
        "zones = gpd.GeoDataFrame({'zone_id': [s[0] for s in _specs]},",
        "    geometry=[Polygon([(s[2], 0), (s[2] + 1000, 0), (s[2] + 1000, 1000), (s[2], 1000)])",
        "              for s in _specs],",
        "    crs=_alias)",
        "zones.to_file(__PATH__, driver='GPKG', engine='pyogrio', layer='zones')",
        "set_result({'wkt': zones.crs.to_wkt(), 'base_wkt': _base.to_wkt(),",
        "            'epsg': zones.crs.to_epsg()})",
    ]
)

#: 真实世界最常见的坑（DAIR-87 基准撞出的那份「1km老年人口分布.tif」）：GeoTIFF 的 CRS 是
#: ArcGIS 写进去的 ESRI 写法 WKT（挂在 .aux.xml 的 <SRS> 上，GDAL 原样用，不重新认 EPSG）。
#: 它与 EPSG:4508 是同一个坐标系，但 pyproj 默认置信度认不出码、equals() 连
#: ignore_axis_order=True 都判 False —— 光靠「等价判断」救不回来。
_MAKE_ESRI_WKT_RASTER = "\n".join(
    [
        "import numpy as np",
        "import pyproj",
        "import rasterio",
        "from rasterio.transform import from_origin",
        "_base = pyproj.CRS('EPSG:4508')",
        "_esri = _base.to_wkt(version='WKT1_ESRI')",
        "with rasterio.open(__RASTER__, 'w', driver='GTiff', height=10, width=30, count=1,",
        "                   dtype='float32', transform=from_origin(0, 1000, 100, 100),",
        "                   nodata=-9999.0) as ds:",
        "    ds.write(np.tile(np.arange(30, dtype='float32'), (10, 1)), 1)",
        "with open(str(__RASTER__) + '.aux.xml', 'w', encoding='utf-8') as fh:",
        "    fh.write('<PAMDataset><SRS>' + _esri + '</SRS></PAMDataset>')",
        "with rasterio.open(__RASTER__) as ds:",
        "    _crs = pyproj.CRS.from_wkt(ds.crs.to_wkt())",
        "    set_result({'wkt': ds.crs.to_wkt(), 'base_wkt': _base.to_wkt(),",
        "                'epsg': _crs.to_epsg(), 'epsg_low': _crs.to_epsg(min_confidence=20),",
        "                'equals': _crs.equals(_base),",
        "                'equals_ignore_axis': _crs.equals(_base, ignore_axis_order=True)})",
    ]
)

_MAKE_GEOGRAPHIC_ZONES = "\n".join(
    [
        "import geopandas as gpd",
        "from shapely.geometry import Polygon",
        "gdf = gpd.GeoDataFrame({'zone_id': [11]},",
        "    geometry=[Polygon([(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)])], crs='EPSG:4490')",
        "gdf.to_file(__PATH__, driver='GPKG', engine='pyogrio', layer='zones')",
        "set_result({'ok': True})",
    ]
)

#: arcpy 侧输入：真 .gdb 要素类（GDAL 版从同一份 gdb 读，双端就是“同一输入”）
_MAKE_ARCPY_ZONES = "\n".join(
    [
        "import arcpy",
        "arcpy.env.overwriteOutput = True",
        "sr = arcpy.SpatialReference(32650)",
        "gdb = __GDB__",
        "if not arcpy.Exists(gdb):",
        "    arcpy.management.CreateFileGDB(__PARENT__, 'zonal_inputs.gdb')",
        "if arcpy.Exists(gdb + '\\\\zones'):",
        "    arcpy.management.Delete(gdb + '\\\\zones')",
        "arcpy.management.CreateFeatureclass(gdb, 'zones', 'POLYGON', spatial_reference=sr)",
        "arcpy.management.AddField(gdb + '\\\\zones', 'zone_id', 'LONG')",
        "arcpy.management.AddField(gdb + '\\\\zones', 'name', 'TEXT', field_length=50)",
        "for zid, name, x in __SPECS__:",
        "    pts = [(x, 0), (x + 1000, 0), (x + 1000, 1000), (x, 1000)]",
        "    with arcpy.da.InsertCursor(gdb + '\\\\zones', ['SHAPE@', 'zone_id', 'name']) as cur:",
        "        cur.insertRow([arcpy.Polygon(arcpy.Array([arcpy.Point(*p) for p in pts]), sr), zid, name])",
        "set_result({'zones': gdb + '\\\\zones', 'gdb': gdb})",
    ]
)

_READ_CSV = "\n".join(
    [
        "import pandas as pd",
        "df = pd.read_csv(__PATH__, encoding='utf-8-sig')",
        "set_result({'columns': [str(c) for c in df.columns],",
        "            'rows': [[None if (v is None or (isinstance(v, float) and v != v)) else v",
        "                      for v in row]",
        "                     for row in df.itertuples(index=False, name=None)]})",
    ]
)

_READ_GPKG = "\n".join(
    [
        "import pyogrio",
        "df = pyogrio.read_dataframe(__PATH__, layer=__LAYER__)",
        "set_result({'columns': [str(c) for c in df.columns],",
        "            'geometry': 'geometry' in df.columns,",
        "            'rows': [[None if (v is None or (isinstance(v, float) and v != v)) else v",
        "                      for v in row]",
        "                     for row in df.itertuples(index=False, name=None)]})",
    ]
)

_READ_ARCPY = "\n".join(
    [
        "import arcpy",
        "names = [f.name for f in arcpy.ListFields(__PATH__)",
        "         if f.type not in ('Geometry', 'OID', 'Blob', 'Raster')]",
        "rows = [list(r) for r in arcpy.da.SearchCursor(__PATH__, names)]",
        "set_result({'columns': names, 'rows': rows})",
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
    """造真 .gdb 栅格（GDAL 写不了 FGDB，只能借 arcpy），顺带建好 .tif 基线输入。"""
    _, raster = _make_inputs(gdal_runner, tmp_path)
    made = arcpy_runner.run(
        _MAKE_GDB_RASTERS.replace("__WORK__", repr(str(tmp_path))).replace(
            "__TIF__", repr(str(raster))
        ),
        timeout=300,
    )
    assert made.ok, made.error
    made.result["raster"] = str(raster)
    return made.result


def _run(recipe_lib: RecipeLibrary, runner, tmp_path: Path, **params):
    recipe_lib.load()
    ctx = SimpleNamespace(workspace=tmp_path, code_runner=runner, exec_timeout=600)
    return recipe_lib.run("zonal_statistics", params, ctx)


def _make_inputs(runner, tmp_path: Path, code: str = _MAKE_INPUTS, crs: str = "EPSG:32650"):
    zones, raster = tmp_path / "zones.gpkg", tmp_path / "dem.tif"
    made = runner.run(
        code.replace("__ZONES__", repr(str(zones)))
        .replace("__RASTER__", repr(str(raster)))
        .replace("__SPECS__", repr(ZONE_SPECS))
        .replace("__CRS__", repr(crs)),
        timeout=300,
    )
    assert made.ok, made.error
    return zones, raster


def _zonal(runner, tmp_path: Path, name: str, **overrides):
    params = {
        "zone_path": str(tmp_path / "zones.gpkg"),
        "raster_path": str(tmp_path / "dem.tif"),
        "output_table": str(tmp_path / name),
        "zone_field": "zone_id",
        "statistics": "SUM",
    }
    params.update(overrides)
    lib = RecipeLibrary(code_runner=runner)
    return _run(lib, runner, tmp_path, **params)


def _table(runner, path: str, *, arcpy: bool = False, layer: str | None = None) -> dict:
    """读回统计表 → {zone_id: {字段: 值}}（字段顺序保留在 dict 里）。"""
    if arcpy:
        code = _READ_ARCPY
    elif str(path).lower().endswith(".csv"):
        code = _READ_CSV
    else:
        code = _READ_GPKG.replace("__LAYER__", repr(layer or Path(path).stem))
    outcome = runner.run(code.replace("__PATH__", repr(str(path))), timeout=180)
    assert outcome.ok, outcome.error
    columns = outcome.result["columns"]
    table = {}
    for row in outcome.result["rows"]:
        record = dict(zip(columns, row))
        table[int(record["zone_id"])] = record
    return table


def _relative_diff(actual, expected) -> float | None:
    """两边都空 → None（视为一致）；一边空一边有值 → inf（一致不了）。"""
    if actual is None or expected is None:
        return None if actual is None and expected is None else float("inf")
    a, e = float(actual), float(expected)
    if e == 0:
        return 0.0 if abs(a) < 1e-9 else float("inf")
    return abs(a - e) / abs(e)


# ------------------------------------------------------------- 模板本身
def test_recipe_declares_two_backends_and_renders_both():
    recipe = Recipe.from_yaml(RECIPE)
    assert recipe.backend == ["arcpy", "gdal"]
    params = {
        "zone_path": "z.gdb/zones",
        "raster_path": "d.tif",
        "output_table": "out/stats.csv",
        "zone_field": "zone_id",
        "statistics": "SUM",
    }
    arcpy_code = recipe.render(params)
    gdal_code = recipe.render(params, backend="gdal")

    # arcpy 版一个字都没动（快照测试盯着的就是这一段）
    assert "arcpy.sa.ZonalStatisticsAsTable" in arcpy_code and "rasterio" not in arcpy_code
    assert "from rasterio.mask import mask" in gdal_code and "import arcpy" not in gdal_code
    # 统计类型 → 输出列名只写一处：arcpy 的 MAXIMUM/MINIMUM 与列名 MAX/MIN 在同一张表里
    assert '"MAXIMUM": ("MAX",)' in gdal_code and '"MINIMUM": ("MIN",)' in gdal_code
    # 输出格式是显式白名单（不静默换格式）
    assert 'out_ext not in (".csv", ".gpkg")' in gdal_code
    # DAIR-221：栅格路径要能翻译成 GDAL 认的写法，存在性判据不能再是 os.path.exists
    assert "def _resolve_raster_path(path):" in gdal_code
    assert 'OpenFileGDB:"' in gdal_code
    assert "os.path.exists(str(raster_path))" not in gdal_code
    for key in ("z.gdb/zones", "d.tif", "out/stats.csv", "zone_id", "SUM"):
        assert key in arcpy_code and key in gdal_code


def test_both_templates_split_semicolon_statistics():
    """多项统计（SUM;MEAN）：两端都按 ; 拆开。arcpy 一次只认一个统计类型，整串传会报 000864。"""
    recipe = Recipe.from_yaml(RECIPE)
    params = {
        "zone_path": "z.gdb/zones",
        "raster_path": "d.tif",
        "output_table": "out/stats.dbf",
        "zone_field": "zone_id",
        "statistics": "SUM;MEAN",
    }
    arcpy_code = recipe.render(params)
    gdal_code = recipe.render(params, backend="gdal")

    assert 'split(";")' in arcpy_code and 'split(";")' in gdal_code
    assert "statistics_type=statistics" not in arcpy_code, "不能把整串交给 ZonalStatisticsAsTable"
    assert "JoinField" in arcpy_code, "多项统计要并进同一张表，不能只跑第一个"


# ----------------------------------------------------------- GDAL 侧真跑
@pytest.mark.parametrize("statistics", ["SUM", "MEAN"])
def test_gdal_sum_and_mean_match_arcpy_reference_values(gdal_runner, tmp_path: Path, statistics):
    """逐分区与 arcpy 实测值对齐：COUNT/AREA 严格相等，SUM/MEAN 差 <1%（本批验收口径）。"""
    _make_inputs(gdal_runner, tmp_path)
    obs = _zonal(gdal_runner, tmp_path, f"out/{statistics}.csv", statistics=statistics)

    assert obs.ok, obs.summary
    payload = obs.data["result"]
    assert payload["record_count"] == 3
    assert payload["zone_field"] == "zone_id"

    table = _table(gdal_runner, payload["output"])
    assert set(table) == {11, 22, 33}
    for zone_id, expected in REFERENCE.items():
        assert table[zone_id]["COUNT"] == expected["COUNT"], (zone_id, table[zone_id])
        assert abs(table[zone_id]["AREA"] - expected["AREA"]) < 1e-6, (zone_id, table[zone_id])
        diff = _relative_diff(table[zone_id][statistics], expected[statistics])
        assert diff is not None and diff < 0.01, (statistics, zone_id, table[zone_id])
        print(f"{statistics} zone {zone_id}: arcpy={expected[statistics]} gdal={table[zone_id][statistics]}")

    # 两条断言都是 GDAL 口径真判过的（pyogrio 自己数行，不落到 arcpy 兜底）
    assertions = {a["kind"]: a for a in obs.data["assertions"]}
    assert assertions["vector_exists"]["ok"] is True
    assert assertions["feature_count_gt"]["ok"] is True
    assert assertions["feature_count_gt"]["detail"] == "count=3"


def test_gdal_csv_and_gpkg_hold_the_same_table(gdal_runner, tmp_path: Path):
    _make_inputs(gdal_runner, tmp_path)
    # zone_field 大小写不敏感（与 arcpy 的 ListFields 同口径）：输出列名用数据里的真实写法
    csv_obs = _zonal(gdal_runner, tmp_path, "out/a.csv", zone_field="ZONE_ID")
    gpkg_obs = _zonal(gdal_runner, tmp_path, "out/b.gpkg", zone_field="ZONE_ID")

    assert csv_obs.ok, csv_obs.summary
    assert gpkg_obs.ok, gpkg_obs.summary
    csv_table = _table(gdal_runner, csv_obs.data["result"]["output"])
    gpkg_table = _table(gdal_runner, gpkg_obs.data["result"]["output"], layer="b")

    assert list(csv_table[11]) == ["zone_id", "COUNT", "AREA", "SUM"]
    assert csv_table == gpkg_table, "同一次统计，.csv 与 .gpkg 的内容必须一致"

    # .gpkg 里是一张**表**（没有 geometry），不是要素类
    info = gdal_runner.run(
        "\n".join(
            [
                "import pyogrio",
                f"i = pyogrio.read_info({gpkg_obs.data['result']['output']!r}, layer='b')",
                "set_result({'driver': i.get('driver'), 'features': int(i['features']),",
                "            'geometry_type': i.get('geometry_type')})",
            ]
        ),
        timeout=120,
    )
    assert info.ok, info.error
    assert info.result["driver"] == "GPKG"
    assert info.result["features"] == 3
    assert info.result["geometry_type"] is None


def test_gdal_empty_zone_field_writes_objectid_column(gdal_runner, tmp_path: Path):
    """空 zone_field = arcpy 的 OBJECTID：列名与取值都要一致（交付说明承诺过、之前没测）。"""
    _make_inputs(gdal_runner, tmp_path)
    obs = _zonal(gdal_runner, tmp_path, "out/oid.csv", zone_field="")

    assert obs.ok, obs.summary
    assert obs.data["result"]["zone_field"] == "OBJECTID"

    raw = gdal_runner.run(
        _READ_CSV.replace("__PATH__", repr(obs.data["result"]["output"])), timeout=120
    )
    assert raw.ok, raw.error
    assert raw.result["columns"] == ["OBJECTID", "COUNT", "AREA", "SUM"]
    # 空 zone_field 取源数据 FID：三块分区依次是 1/2/3，统计值仍与 zone_id 那版一致
    rows = {int(row[0]): row[1:] for row in raw.result["rows"]}
    assert set(rows) == {1, 2, 3}
    for oid, expected in enumerate(REFERENCE.values(), start=1):
        assert rows[oid][0] == expected["COUNT"], (oid, rows[oid])
        assert abs(rows[oid][1] - expected["AREA"]) < 1e-6, (oid, rows[oid])
        assert abs(rows[oid][2] - expected["SUM"]) < 1e-9, (oid, rows[oid])


def test_gdal_statistic_names_match_arcpy_output(gdal_runner, tmp_path: Path):
    """统计类型 → 输出列名：arcpy 的 MAXIMUM/MINIMUM 落成 MAX/MIN，多项一次给齐。"""
    _make_inputs(gdal_runner, tmp_path)

    obs = _zonal(gdal_runner, tmp_path, "out/named.csv", statistics="MAXIMUM;MINIMUM")
    assert obs.ok, obs.summary
    table = _table(gdal_runner, obs.data["result"]["output"])
    assert list(table[11]) == ["zone_id", "COUNT", "AREA", "MAX", "MIN"]
    assert table[11]["MAX"] == 9.0 and table[11]["MIN"] == 0.0
    assert table[33]["MAX"] == 29.0 and table[33]["MIN"] == 20.0

    obs = _zonal(gdal_runner, tmp_path, "out/multi.csv", statistics="RANGE;MEDIAN;STD;MIN_MAX;MEAN_STD")
    assert obs.ok, obs.summary
    table = _table(gdal_runner, obs.data["result"]["output"])
    assert list(table[11]) == ["zone_id", "COUNT", "AREA", "RANGE", "MEDIAN", "STD", "MIN", "MAX", "MEAN"]
    # Z1 的像元值是 0..9：RANGE=9、MEDIAN=4.5、STD 用总体标准差 sqrt(8.25)（arcpy 实测口径）
    assert table[11]["RANGE"] == 9.0
    assert table[11]["MEDIAN"] == 4.5
    assert abs(table[11]["STD"] - 2.8722813232690143) < 1e-6, table[11]["STD"]  # 栅格是 float32


def test_gdal_nodata_and_degenerate_zones(gdal_runner, tmp_path: Path):
    """nodata 不算进 COUNT/AREA；没有像元的分区不出行（arcpy 实测同样丢行）。"""
    _make_inputs(gdal_runner, tmp_path, _MAKE_EDGE_INPUTS)
    obs = _zonal(gdal_runner, tmp_path, "out/edge.csv")

    assert obs.ok, obs.summary
    payload = obs.data["result"]
    assert payload["record_count"] == 1, "只有 zone 11 有像元；44（无格心）、55（在栅格外）都不出行"

    table = _table(gdal_runner, payload["output"])
    assert set(table) == {11}
    # 1 km 分区 = 100 个像元，第 3 列整列 nodata → 90 个有效像元，SUM = (0+1+3+…+9)×10
    assert table[11]["COUNT"] == 90.0
    assert abs(table[11]["AREA"] - 900_000.0) < 1e-6
    assert abs(table[11]["SUM"] - 430.0) < 1e-9, table[11]["SUM"]


# --------------------------------------------------------------- 拒绝分支
def test_gdal_rejects_crs_mismatch(gdal_runner, tmp_path: Path):
    _make_inputs(gdal_runner, tmp_path)
    geographic = tmp_path / "zones_geo.gpkg"
    made = gdal_runner.run(_MAKE_GEOGRAPHIC_ZONES.replace("__PATH__", repr(str(geographic))), timeout=180)
    assert made.ok, made.error

    obs = _zonal(gdal_runner, tmp_path, "out/crs.csv", zone_path=str(geographic))

    assert not obs.ok
    assert "坐标系不一致" in obs.summary
    assert not (tmp_path / "out/crs.csv").exists(), "报错时不应留下半成品"


def test_gdal_still_rejects_geographic_vs_projected_zones(gdal_runner, tmp_path: Path):
    """同一基准面的 4490（地理）vs 4508（投影）必须仍被拒：低置信度比较不能退化成「都有 CRS 就放行」。"""
    _make_inputs(gdal_runner, tmp_path, crs="EPSG:4508")
    geographic = tmp_path / "zones_geo.gpkg"
    made = gdal_runner.run(_MAKE_GEOGRAPHIC_ZONES.replace("__PATH__", repr(str(geographic))), timeout=180)
    assert made.ok, made.error

    obs = _zonal(gdal_runner, tmp_path, "out/geo_vs_proj.csv", zone_path=str(geographic))

    assert not obs.ok
    assert "坐标系不一致" in obs.summary
    assert not (tmp_path / "out/geo_vs_proj.csv").exists()


def test_gdal_accepts_same_epsg_with_different_wkt(gdal_runner, tmp_path: Path):
    """同 EPSG、WKT 文本不同（ESRI 写法）不算坐标系不一致：比较口径是等价，不是字符串。"""
    _make_inputs(gdal_runner, tmp_path)
    alias = tmp_path / "zones_esri_wkt.gpkg"
    made = gdal_runner.run(
        _MAKE_ESRI_WKT_ZONES.replace("__PATH__", repr(str(alias))).replace(
            "__SPECS__", repr(ZONE_SPECS)
        ),
        timeout=180,
    )
    assert made.ok, made.error
    assert made.result["epsg"] == 32650
    assert made.result["wkt"] != made.result["base_wkt"], "前提：两份 WKT 文本确实不同"

    obs = _zonal(gdal_runner, tmp_path, "out/alias.csv", zone_path=str(alias))

    assert obs.ok, obs.summary
    table = _table(gdal_runner, obs.data["result"]["output"])
    assert abs(table[11]["SUM"] - REFERENCE[11]["SUM"]) < 1e-9


def test_gdal_accepts_esri_wkt_raster_that_only_identifies_at_low_confidence(
    gdal_runner, tmp_path: Path
):
    """真实 GeoTIFF（ESRI 写法 WKT）：默认置信度认不出 EPSG 码，但坐标系确实相同 → 不能误判。

    这就是 DAIR-87 基准里那份「1km老年人口分布.tif」：pyproj 默认置信度给 None，
    equals() 也是 False，只有 min_confidence=20 才认得出 4508。
    """
    _make_inputs(gdal_runner, tmp_path, crs="EPSG:4508")
    raster = tmp_path / "dem.tif"
    made = gdal_runner.run(
        _MAKE_ESRI_WKT_RASTER.replace("__RASTER__", repr(str(raster))), timeout=300
    )
    assert made.ok, made.error
    # 前提：两份 WKT 文本确实不同、默认置信度确实认不出码 —— 否则这条测的是假等价
    assert made.result["wkt"] != made.result["base_wkt"]
    assert made.result["epsg"] is None and made.result["epsg_low"] == 4508
    assert made.result["equals"] is False and made.result["equals_ignore_axis"] is False

    obs = _zonal(gdal_runner, tmp_path, "out/esri_raster.csv", raster_path=str(raster))

    assert obs.ok, obs.summary
    table = _table(gdal_runner, obs.data["result"]["output"])
    assert abs(table[11]["SUM"] - REFERENCE[11]["SUM"]) < 1e-9


@pytest.mark.parametrize(
    "statistics,want",
    [
        ("ALL", "ALL 需要 percentile_values 参数"),
        ("PERCENTILE", "PERCENTILE 需要 percentile_values 参数"),
        ("SUM;MEDIAN;PCT90", "PCT90 在 GDAL 后端没有实现"),
    ],
)
def test_gdal_rejects_statistics_it_cannot_match(gdal_runner, tmp_path: Path, statistics, want):
    """arcpy 认的 ALL/PERCENTILE 要百分位（还得给 percentile_values），本配方没有 → 报错带原因。"""
    _make_inputs(gdal_runner, tmp_path)
    obs = _zonal(gdal_runner, tmp_path, "out/bad_stat.csv", statistics=statistics)

    assert not obs.ok
    assert want in obs.summary, obs.summary
    assert not (tmp_path / "out/bad_stat.csv").exists()


def test_gdal_says_count_is_already_a_column_not_unimplemented(gdal_runner, tmp_path: Path):
    """COUNT 是固定输出列，不是「没实现的统计量」——报错不能误导人去等实现。"""
    _make_inputs(gdal_runner, tmp_path)
    obs = _zonal(gdal_runner, tmp_path, "out/count_stat.csv", statistics="SUM;COUNT")

    assert not obs.ok
    assert "COUNT 已是固定输出列" in obs.summary, obs.summary
    # 只锁「别把 COUNT 说成没实现、让人去等」这一个意图：整句里出现「没有实现」不算问题
    # （例如将来写「MIN_MAX 没有实现的说明」），但把 COUNT 归到「没实现」就必须红。
    assert "COUNT 在 GDAL 后端没有实现" not in obs.summary, obs.summary
    assert not (tmp_path / "out/count_stat.csv").exists()


@pytest.mark.parametrize(
    "output_table,want",
    [("out/x.gdb/stats", "GDAL 后端不支持 gdb 表"), ("out/x.dbf", "只支持 .csv / .gpkg")],
)
def test_gdal_rejects_output_formats_it_cannot_write(gdal_runner, tmp_path: Path, output_table, want):
    _make_inputs(gdal_runner, tmp_path)
    obs = _zonal(gdal_runner, tmp_path, output_table)

    assert not obs.ok
    assert want in obs.summary and ".csv" in obs.summary
    assert not (tmp_path / "out/x.gdb").exists()


def test_gdal_rejects_unknown_zone_field(gdal_runner, tmp_path: Path):
    _make_inputs(gdal_runner, tmp_path)
    obs = _zonal(gdal_runner, tmp_path, "out/no_field.csv", zone_field="nope")

    assert not obs.ok
    assert "分区数据缺少字段: nope" in obs.summary and "zone_id" in obs.summary
    assert not (tmp_path / "out/no_field.csv").exists()


# ------------------------------------------- GDAL 侧：.gdb 里的栅格（DAIR-221）
def test_gdb_raster_layer_path_works_with_both_separators(gdal_runner, tmp_path, gdb_rasters):
    """arcpy 侧交出来的 ``x.gdb\层名`` 不能再报「栅格数据不存在」（数据其实在库里）。

    逐区分值必须等于同一份栅格的 .tif 基线（同一像元、同一网格，两端都不改数值）。
    """
    baseline = _table(
        gdal_runner,
        _zonal(gdal_runner, tmp_path, "out/gdb_base.csv",
               raster_path=gdb_rasters["raster"]).data["result"]["output"],
    )
    for sep, tag in (("\\", "backslash"), ("/", "slash")):
        obs = _zonal(
            gdal_runner, tmp_path, f"out/gdb_layer_{tag}.csv",
            raster_path=gdb_rasters["single"] + sep + "dem",
        )
        assert obs.ok, obs.summary
        assert "栅格数据不存在" not in obs.data["stdout"]
        assert _table(gdal_runner, obs.data["result"]["output"]) == baseline


def test_plain_gdb_directory_with_a_raster_inside_still_opens_as_a_file(gdal_runner, tmp_path):
    r"""``.gdb`` 结尾的**普通目录**里放散文件：按文件路径打开，不翻 OpenFileGDB。

    链式任务里输出目录常被传成 ``out.gdb``（普通目录），下游再把 ``out.gdb\dem.tif``
    喂回来——按「库 + 层名」翻会让 rasterio 打不开，报成「栅格数据不存在或打不开」。
    """
    _, raster = _make_inputs(gdal_runner, tmp_path)
    baseline = _table(
        gdal_runner,
        _zonal(gdal_runner, tmp_path, "out/plain_gdb_base.csv").data["result"]["output"],
    )
    plain = tmp_path / "plain.gdb"
    plain.mkdir()
    shutil.copyfile(raster, plain / "dem.tif")

    obs = _zonal(gdal_runner, tmp_path, "out/plain_gdb.csv",
                 raster_path=str(plain) + "\\dem.tif")

    assert obs.ok, obs.summary
    assert "栅格数据不存在" not in obs.data["stdout"]
    assert _table(gdal_runner, obs.data["result"]["output"]) == baseline


def test_single_raster_gdb_container_is_used_automatically(gdal_runner, tmp_path, gdb_rasters):
    """只给库本身且库里恰好一份栅格时直接取用（单栅格库 rasterio 本来就打得开）。"""
    baseline = _table(
        gdal_runner,
        _zonal(gdal_runner, tmp_path, "out/gdb_container_base.csv",
               raster_path=gdb_rasters["raster"]).data["result"]["output"],
    )
    obs = _zonal(gdal_runner, tmp_path, "out/gdb_container.csv",
                 raster_path=gdb_rasters["single"])

    assert obs.ok, obs.summary
    assert "栅格数据不存在" not in obs.data["stdout"]
    assert _table(gdal_runner, obs.data["result"]["output"]) == baseline


def test_multi_raster_gdb_container_asks_for_a_raster_by_name(gdal_runner, tmp_path, gdb_rasters):
    """多栅格库只给库本体 → 人话错误（列出栅格名 + 怎么指定），不是 No such file。"""
    obs = _zonal(gdal_runner, tmp_path, "out/gdb_multi.csv", raster_path=gdb_rasters["multi"])

    assert not obs.ok
    assert "dem、dem_shift" in obs.summary
    assert "请指明栅格" in obs.summary and "No such file" not in obs.summary


@pytest.mark.parametrize("form", ["missing_file", "missing_gdb"])
def test_missing_raster_still_reports_it_does_not_exist(gdal_runner, tmp_path: Path, form):
    """存在性判据换成「能不能打开」之后，真不存在时仍要说清是哪个路径。"""
    _make_inputs(gdal_runner, tmp_path)
    path = (str(tmp_path / "nope.tif") if form == "missing_file"
            else str(tmp_path / "nope.gdb") + "\\dem")
    obs = _zonal(gdal_runner, tmp_path, "out/missing.csv", raster_path=path)

    assert not obs.ok
    assert "栅格数据不存在" in obs.summary and "nope" in obs.summary


# --------------------------------------------------------------- 双端一致
def _arcpy_zones(arcpy_runner, tmp_path: Path, specs=ZONE_SPECS) -> str:
    gdb = tmp_path / "zonal_inputs.gdb"
    made = arcpy_runner.run(
        _MAKE_ARCPY_ZONES.replace("__GDB__", repr(str(gdb)))
        .replace("__PARENT__", repr(str(tmp_path)))
        .replace("__SPECS__", repr(specs)),
        timeout=300,
    )
    assert made.ok, made.error
    return made.result["zones"]


@pytest.mark.parametrize("statistics", ["SUM", "MEAN"])
def test_arcpy_and_gdal_agree_on_the_same_inputs(
    gdal_runner, arcpy_runner, tmp_path: Path, statistics
):
    """同一份 .gdb 分区 + 同一份 .tif：逐分区 SUM/MEAN 差 <1%，COUNT/AREA/字段名严格相等。"""
    zones_fc = _arcpy_zones(arcpy_runner, tmp_path)
    _, raster = _make_inputs(gdal_runner, tmp_path)

    gdal_obs = _zonal(
        gdal_runner,
        tmp_path,
        f"out/gdal_{statistics}.csv",
        zone_path=zones_fc,
        raster_path=str(raster),
        statistics=statistics,
    )
    arcpy_obs = _zonal(
        arcpy_runner,
        tmp_path,
        f"out/arcpy_{statistics}.dbf",
        zone_path=zones_fc,
        raster_path=str(raster),
        statistics=statistics,
    )
    assert gdal_obs.ok, f"gdal/{statistics}: {gdal_obs.summary}"
    assert arcpy_obs.ok, f"arcpy/{statistics}: {arcpy_obs.summary}"

    gdal_table = _table(gdal_runner, gdal_obs.data["result"]["output"])
    arcpy_table = _table(arcpy_runner, arcpy_obs.data["result"]["output"], arcpy=True)

    assert set(gdal_table) == set(arcpy_table) == {11, 22, 33}
    # 字段名与顺序一致：下游 join 认的是 zone_id（GDAL 侧不能叫 zone_code 之类）
    assert list(gdal_table[11]) == list(arcpy_table[11]) == ["zone_id", "COUNT", "AREA", statistics]

    for zone_id in (11, 22, 33):
        for field in ("COUNT", "AREA"):
            assert gdal_table[zone_id][field] == arcpy_table[zone_id][field], (
                statistics,
                zone_id,
                field,
                gdal_table[zone_id][field],
                arcpy_table[zone_id][field],
            )
        diff = _relative_diff(gdal_table[zone_id][statistics], arcpy_table[zone_id][statistics])
        assert diff is not None and diff < 0.01, (
            f"{statistics} zone {zone_id}: arcpy={arcpy_table[zone_id][statistics]} "
            f"gdal={gdal_table[zone_id][statistics]} diff={diff}"
        )
        print(
            f"{statistics} zone {zone_id}: arcpy={arcpy_table[zone_id][statistics]} "
            f"gdal={gdal_table[zone_id][statistics]} diff={diff:.4%}"
        )


@pytest.mark.parametrize("in_gdb", [False, True], ids=["dbf", "gdb"])
def test_arcpy_accepts_semicolon_statistics(
    gdal_runner, arcpy_runner, tmp_path: Path, in_gdb: bool
):
    """SUM;MEAN 在 arcpy 后端能跑通（DAIR-89 D1）：列名/列序与 GDAL 一致，数值与 arcpy 单跑一致。"""
    zones_fc = _arcpy_zones(arcpy_runner, tmp_path)
    _, raster = _make_inputs(gdal_runner, tmp_path)
    (tmp_path / "out").mkdir(exist_ok=True)  # arcpy 不建目录（会先报 000875）
    output = str(tmp_path / "zonal_inputs.gdb" / "multi_stats") if in_gdb else str(
        tmp_path / "out" / "multi.dbf"
    )

    arcpy_obs = _zonal(
        arcpy_runner,
        tmp_path,
        "out/multi.dbf",
        output_table=output,
        zone_path=zones_fc,
        raster_path=str(raster),
        statistics="SUM;MEAN",
    )
    assert arcpy_obs.ok, arcpy_obs.summary
    arcpy_table = _table(arcpy_runner, arcpy_obs.data["result"]["output"], arcpy=True)
    assert list(arcpy_table[11]) == ["zone_id", "COUNT", "AREA", "SUM", "MEAN"]
    for zone_id, expected in REFERENCE.items():
        assert arcpy_table[zone_id]["SUM"] == pytest.approx(expected["SUM"], rel=0.01)
        assert arcpy_table[zone_id]["MEAN"] == pytest.approx(expected["MEAN"], rel=0.01)
    assert not list(tmp_path.rglob("zz_zonal_tmp*")), "逐项统计的临时表必须清掉"

    gdal_obs = _zonal(
        gdal_runner,
        tmp_path,
        "out/multi.csv",
        zone_path=zones_fc,
        raster_path=str(raster),
        statistics="SUM;MEAN",
    )
    assert gdal_obs.ok, gdal_obs.summary
    gdal_table = _table(gdal_runner, gdal_obs.data["result"]["output"])
    assert list(gdal_table[11]) == list(arcpy_table[11]), "两端输出列名/列序必须一致"
    for zone_id in (11, 22, 33):
        for field in ("COUNT", "AREA"):
            assert gdal_table[zone_id][field] == arcpy_table[zone_id][field], (zone_id, field)
        assert _relative_diff(gdal_table[zone_id]["SUM"], arcpy_table[zone_id]["SUM"]) < 0.01


#: 多层 .gpkg（counties + towns）：DAIR-92 报的那条——直接把容器当分区输入
_MAKE_TWO_LAYER_GPKG = "\n".join(
    [
        "import geopandas as gpd",
        "from shapely.geometry import Polygon",
        "_specs = __SPECS__",
        "zones = gpd.GeoDataFrame({'zone_id': [s[0] for s in _specs],",
        "                          'name': [s[1] for s in _specs]},",
        "    geometry=[Polygon([(s[2], 0), (s[2] + 1000, 0), (s[2] + 1000, 1000), (s[2], 1000)])",
        "              for s in _specs],",
        "    crs='EPSG:32650')",
        "zones.to_file(__PATH__, driver='GPKG', engine='pyogrio', layer='counties')",
        "zones.to_file(__PATH__, driver='GPKG', engine='pyogrio', layer='towns')",
        "set_result({'layers': ['counties', 'towns']})",
    ]
)


def test_arcpy_rejects_multilayer_container_and_lists_its_layers(
    gdal_runner, arcpy_runner, tmp_path: Path
):
    """多层 .gpkg 直接当 zone_path：报一条人话错误（列出图层 + 怎么改），不是裸 AttributeError。

    arcpy 把没指明图层的 .gpkg 当工作空间（Describe 的 dataType=Workspace），上面没有
    spatialReference——以前这行裸取就是 `[AttributeError] DescribeData: 对象 spatialReference 不存在`。
    """
    multi = tmp_path / "multi.gpkg"
    made = gdal_runner.run(
        _MAKE_TWO_LAYER_GPKG.replace("__PATH__", repr(str(multi))).replace(
            "__SPECS__", repr(ZONE_SPECS)
        ),
        timeout=180,
    )
    assert made.ok, made.error
    _, raster = _make_inputs(gdal_runner, tmp_path)
    (tmp_path / "out").mkdir(exist_ok=True)  # 先建目录：下面「不留半成品」才是真断言

    obs = _zonal(
        arcpy_runner,
        tmp_path,
        "out/multi.dbf",
        zone_path=str(multi),  # 容器本身：没指明是哪一层
        raster_path=str(raster),
    )

    assert not obs.ok
    assert "多层容器" in obs.summary, obs.summary
    assert "counties" in obs.summary and "towns" in obs.summary, obs.summary
    assert "图层名" in obs.summary, obs.summary
    # 关键：不能是原来那条裸 AttributeError
    assert "AttributeError" not in obs.summary and "DescribeData" not in obs.summary, obs.summary
    assert not (tmp_path / "out/multi.dbf").exists(), "报错时不应留下半成品"


@pytest.mark.parametrize("container_kind", ["gdb", "gpkg"])
def test_arcpy_uses_the_only_layer_of_a_container(
    gdal_runner, arcpy_runner, tmp_path: Path, container_kind: str
):
    """容器里恰好一个要素类：自动取用并跑通（不因为「给的是库不是图层」就报错）。

    gpkg 那组是真实世界最常见的样子（GDAL 侧默认输出 .gpkg），顺带盖住两件事：
    gpkg 的图层名带 ``main.`` 前缀、gpkg 的整型是 BigInteger（要先转成 LONG 才能统计）。
    """
    if container_kind == "gdb":
        zones_fc = _arcpy_zones(arcpy_runner, tmp_path)  # .gdb 里只有 zones 一个要素类
        container = str(Path(zones_fc).parent)
        _, raster = _make_inputs(gdal_runner, tmp_path)
    else:
        zones_path, raster = _make_inputs(gdal_runner, tmp_path)  # zones.gpkg 只有 zones 一层
        container = str(zones_path)
    (tmp_path / "out").mkdir(exist_ok=True)

    obs = _zonal(
        arcpy_runner,
        tmp_path,
        f"out/one_layer_{container_kind}.dbf",
        zone_path=container,
        raster_path=str(raster),
        statistics="SUM;MEAN",
    )

    assert obs.ok, obs.summary
    table = _table(arcpy_runner, obs.data["result"]["output"], arcpy=True)
    assert list(table[11]) == ["zone_id", "COUNT", "AREA", "SUM", "MEAN"]
    for zone_id, expected in REFERENCE.items():
        assert table[zone_id]["SUM"] == pytest.approx(expected["SUM"], rel=0.01)
        assert table[zone_id]["MEAN"] == pytest.approx(expected["MEAN"], rel=0.01)


def test_arcpy_in_memory_output_table_leaves_no_temp_tables(
    gdal_runner, arcpy_runner, tmp_path: Path
):
    """临时表落点 _stat_tmp 的 in_memory 分支：输出在 in_memory 时 temp 表是 in_memory\\zz_zonal_tmp_N，
    跑完必须删掉（DAIR-92 #2：这条分支此前一行判断、没有用例）。

    直接跑渲染出的模板（不走 ``RecipeLibrary.run``）：配方的 vector_exists / feature_count_gt 按
    文件系统路径判，in_memory 输出一定栽在那两道闸上（与本用例要验的临时表落点无关）。
    """
    zones_fc = _arcpy_zones(arcpy_runner, tmp_path)
    _, raster = _make_inputs(gdal_runner, tmp_path)
    out_table = r"in_memory\zonal_out"

    recipe = Recipe.from_yaml(RECIPE)
    code = recipe.render(
        {
            "zone_path": zones_fc,
            "raster_path": str(raster),
            "output_table": out_table,
            "zone_field": "zone_id",
            "statistics": "SUM;MEAN",
        }
    )
    ran = arcpy_runner.run(code, timeout=600)

    assert ran.ok, ran.error
    assert ran.result["output"] == out_table and ran.result["record_count"] == 3
    # in_memory 表要跨 run 存活才能读回：退回一次性子进程时无法验证临时表清理（会明说 skip，不假绿）
    if arcpy_runner.last_backend != "kernel":
        pytest.skip(f"本机走了一次性子进程（{arcpy_runner.last_backend}），in_memory 表不跨 run 存活")
    table = _table(arcpy_runner, out_table, arcpy=True)
    assert list(table[11]) == ["zone_id", "COUNT", "AREA", "SUM", "MEAN"]
    for zone_id, expected in REFERENCE.items():
        assert table[zone_id]["SUM"] == pytest.approx(expected["SUM"], rel=0.01)
        assert table[zone_id]["MEAN"] == pytest.approx(expected["MEAN"], rel=0.01)

    check = arcpy_runner.run(
        "\n".join(
            [
                "import arcpy",
                "set_result({'tmp': arcpy.Exists(__TMP__), 'out': arcpy.Exists(__OUT__)})",
            ]
        )
        .replace("__TMP__", repr(r"in_memory\zz_zonal_tmp_1"))
        .replace("__OUT__", repr(out_table)),
        timeout=180,
    )
    assert check.ok, check.error
    assert check.result == {"tmp": False, "out": True}, check.result


def test_arcpy_empty_zone_field_with_multi_statistics(
    gdal_runner, arcpy_runner, tmp_path: Path
):
    """空 zone_field + 多统计量：OBJECTID 既是 JoinField 的键也是输出列，列名/列序/值都要对
    （DAIR-92 #3：单统计、多统计各自有用例，这组合没有）。"""
    zones_fc = _arcpy_zones(arcpy_runner, tmp_path)
    _, raster = _make_inputs(gdal_runner, tmp_path)
    (tmp_path / "out").mkdir(exist_ok=True)

    obs = _zonal(
        arcpy_runner,
        tmp_path,
        "out/oid_multi.dbf",
        zone_path=zones_fc,
        raster_path=str(raster),
        zone_field="",
        statistics="SUM;MEAN",
    )

    assert obs.ok, obs.summary
    read = arcpy_runner.run(
        _READ_ARCPY.replace("__PATH__", repr(obs.data["result"]["output"])), timeout=180
    )
    assert read.ok, read.error
    assert read.result["columns"] == ["OBJECTID", "COUNT", "AREA", "SUM", "MEAN"]
    rows = {int(row[0]): row[1:] for row in read.result["rows"]}
    assert list(rows) == [1, 2, 3], "空 zone_field 的键是 OBJECTID，按插入顺序 1..3"
    for oid, (zone_id, _, _) in enumerate(ZONE_SPECS, start=1):
        expected = REFERENCE[zone_id]
        assert rows[oid][0] == expected["COUNT"] and rows[oid][1] == expected["AREA"]
        assert rows[oid][2] == pytest.approx(expected["SUM"], rel=0.01)
        assert rows[oid][3] == pytest.approx(expected["MEAN"], rel=0.01)
    assert not list(tmp_path.rglob("zz_zonal_tmp*")), "多统计量的临时表必须清掉"


#: 分区源换成 .shp：OID 字段名是 FID（0 起），与 .gpkg 的 fid（1 起）都不是 OBJECTID
_MAKE_SHP_ZONES = "\n".join(
    [
        "import geopandas as gpd",
        "from shapely.geometry import Polygon",
        "_specs = __SPECS__",
        "zones = gpd.GeoDataFrame({'zone_id': [s[0] for s in _specs]},",
        "    geometry=[Polygon([(s[2], 0), (s[2] + 1000, 0), (s[2] + 1000, 1000), (s[2], 1000)])",
        "              for s in _specs], crs='EPSG:32650')",
        "zones.to_file(__PATH__, driver='ESRI Shapefile', engine='pyogrio')",
        "set_result({'path': str(__PATH__)})",
    ]
)


@pytest.mark.parametrize("container_kind", ["gpkg", "shp"])
def test_arcpy_empty_zone_field_uses_the_source_oid_field(
    gdal_runner, arcpy_runner, tmp_path: Path, container_kind: str
):
    """空 zone_field 要按分区源的 OID 字段名统计（.gpkg=fid、.shp=FID），不能写死 OBJECTID。

    DAIR-98 #1：以前这条路径死在第一条 ZonalStatisticsAsTable
    （ERROR 001000: 字段 OBJECTID 不存在）；.gpkg 容器还是 DAIR-92 之后最容易走到的入口。
    输出列名由 arcpy 决定（.dbf 里 dBASE 的保留名会被加下划线），所以断言的是
    「结果里的 zone_field == 输出表里真实的分区列名」，不硬编码列名 —— 下游 join 认的也是它。
    """
    zones_gpkg, raster = _make_inputs(gdal_runner, tmp_path)
    (tmp_path / "out").mkdir(exist_ok=True)
    if container_kind == "gpkg":
        container = str(zones_gpkg)  # 单层 .gpkg 容器：DAIR-92 让它自动取用唯一图层
    else:
        shp_dir = tmp_path / "shpfolder"
        shp_dir.mkdir()
        made = gdal_runner.run(
            _MAKE_SHP_ZONES.replace("__PATH__", repr(str(shp_dir / "zones.shp")))
            .replace("__SPECS__", repr(ZONE_SPECS)),
            timeout=180,
        )
        assert made.ok, made.error
        container = str(shp_dir)  # .shp 的"容器"就是它所在文件夹

    obs = _zonal(
        arcpy_runner,
        tmp_path,
        f"out/oid_{container_kind}.dbf",
        zone_path=container,
        raster_path=str(raster),
        zone_field="",
        statistics="SUM;MEAN",
    )

    assert obs.ok, obs.summary
    reported = obs.data["result"]["zone_field"]
    read = arcpy_runner.run(
        _READ_ARCPY.replace("__PATH__", repr(obs.data["result"]["output"])), timeout=180
    )
    assert read.ok, read.error
    columns = read.result["columns"]
    # 上报的分区列必须真实存在于输出表（以前上报 OBJECTID，而表里根本没有这列）
    assert reported in columns, (reported, columns)
    assert reported.upper().startswith(("FID", "OBJECTID")), columns
    assert columns == [reported, "COUNT", "AREA", "SUM", "MEAN"], columns

    rows = {int(row[columns.index(reported)]): dict(zip(columns, row)) for row in read.result["rows"]}
    # .gpkg 的 fid 从 1 开始，.shp 的 FID 从 0 开始（arcpy 实测）—— 分区值就是源的 OID
    first = 1 if container_kind == "gpkg" else 0
    assert set(rows) == set(range(first, first + len(ZONE_SPECS)))
    for oid, (zone_id, _, _) in zip(sorted(rows), ZONE_SPECS):
        expected = REFERENCE[zone_id]
        assert rows[oid]["COUNT"] == expected["COUNT"] and rows[oid]["AREA"] == expected["AREA"]
        assert rows[oid]["SUM"] == pytest.approx(expected["SUM"], rel=0.01)
        assert rows[oid]["MEAN"] == pytest.approx(expected["MEAN"], rel=0.01)
    assert not list(tmp_path.rglob("zz_zonal_tmp*")), "多统计量的临时表必须清掉"


#: OID 故意不连续（11/22/33）：空 zone_field 下分区值 = OID，join 是按分区值还是按行序，一测就露
_MAKE_ARCPY_ZONES_WITH_GAPS = "\n".join(
    [
        "import arcpy, os",
        "arcpy.env.overwriteOutput = True",
        "sr = arcpy.SpatialReference(32650)",
        "gdb = __GDB__",
        "if not arcpy.Exists(gdb):",
        "    arcpy.management.CreateFileGDB(os.path.dirname(gdb), os.path.basename(gdb))",
        "fc = os.path.join(gdb, 'gapped')",
        "if arcpy.Exists(fc):",
        "    arcpy.management.Delete(fc)",
        "arcpy.management.CreateFeatureclass(gdb, 'gapped', 'POLYGON', spatial_reference=sr)",
        "arcpy.management.AddField(fc, 'zone_id', 'LONG')",
        "arcpy.management.AddField(fc, 'name', 'TEXT', field_length=50)",
        "_square = [(0, 0), (1, 0), (1, 1), (0, 1)]",
        "with arcpy.da.InsertCursor(fc, ['SHAPE@']) as cur:",
        "    for _ in range(33):",
        "        cur.insertRow([arcpy.Polygon(arcpy.Array([arcpy.Point(*p) for p in _square]), sr)])",
        "_keep = {11: (11, 'Z1', 0), 22: (22, 'Z2', 1000), 33: (33, 'Z3', 2000)}",
        "with arcpy.da.UpdateCursor(fc, ['OID@', 'SHAPE@', 'zone_id', 'name']) as cur:",
        "    for oid, _shape, _zid, _name in cur:",
        "        if oid in _keep:",
        "            zid, nm, x = _keep[oid]",
        "            pts = [(x, 0), (x + 1000, 0), (x + 1000, 1000), (x, 1000)]",
        "            cur.updateRow([oid, arcpy.Polygon(arcpy.Array([arcpy.Point(*p) for p in pts]), sr), zid, nm])",
        "        else:",
        "            cur.deleteRow()",
        "set_result({'fc': fc, 'oids': [r[0] for r in arcpy.da.SearchCursor(fc, ['OID@'])]})",
    ]
)


def test_arcpy_gdb_output_with_empty_zone_field_joins_by_zone_value(
    gdal_runner, arcpy_runner, tmp_path: Path
):
    """空 zone_field + .gdb 输出：分区列被改名 OBJECTID_1，JoinField 必须按它走。

    DAIR-98 #2：以前 JoinField 拿 `zone_used="OBJECTID"` 当键 —— 在 .gdb 表里那是**表自身的
    OID**，两趟统计行序恰好一致时看着全对，一旦行数/行序不同就是静默错数。
    这里让分区源的 OID 不连续（11/22/33），断言「上报的 zone_field 就是输出表里的分区列，
    且按它取出来的分区值正是 11/22/33」—— 按行号 join 会得到 1/2/3，直接红。
    """
    gdb = tmp_path / "zonal_inputs.gdb"
    made = arcpy_runner.run(
        _MAKE_ARCPY_ZONES_WITH_GAPS.replace("__GDB__", repr(str(gdb))), timeout=300
    )
    assert made.ok, made.error
    assert made.result["oids"] == [11, 22, 33], made.result
    _, raster = _make_inputs(gdal_runner, tmp_path)

    obs = _zonal(
        arcpy_runner,
        tmp_path,
        "out/gapped.dbf",
        output_table=str(gdb / "gapped_stats"),
        zone_path=made.result["fc"],
        raster_path=str(raster),
        zone_field="",
        statistics="SUM;MEAN",
    )

    assert obs.ok, obs.summary
    reported = obs.data["result"]["zone_field"]
    read = arcpy_runner.run(
        _READ_ARCPY.replace("__PATH__", repr(obs.data["result"]["output"])), timeout=180
    )
    assert read.ok, read.error
    columns = read.result["columns"]
    assert reported == "OBJECTID_1", "OBJECTID 让给表自身 OID，分区列只能叫 OBJECTID_1"
    assert columns == [reported, "COUNT", "AREA", "SUM", "MEAN"], columns

    rows = {int(row[columns.index(reported)]): dict(zip(columns, row)) for row in read.result["rows"]}
    assert set(rows) == {11, 22, 33}, "按分区值 join；拿到 1/2/3 说明拿表自身 OID 当键了"
    for zone_id, expected in REFERENCE.items():
        assert rows[zone_id]["COUNT"] == expected["COUNT"]
        assert rows[zone_id]["AREA"] == expected["AREA"]
        assert rows[zone_id]["SUM"] == pytest.approx(expected["SUM"], rel=0.01)
        assert rows[zone_id]["MEAN"] == pytest.approx(expected["MEAN"], rel=0.01)
    assert not list(tmp_path.rglob("zz_zonal_tmp*")), "多统计量的临时表必须清掉"


def test_arcpy_and_gdal_agree_on_empty_zone_field(
    gdal_runner, arcpy_runner, tmp_path: Path
):
    """空 zone_field 两端都落成 OBJECTID 列、值也一样（arcpy 写 .dbf 的列名就是 OBJECTID）。"""
    zones_fc = _arcpy_zones(arcpy_runner, tmp_path)
    _, raster = _make_inputs(gdal_runner, tmp_path)

    gdal_obs = _zonal(
        gdal_runner, tmp_path, "out/oid_gdal.csv",
        zone_path=zones_fc, raster_path=str(raster), zone_field="",
    )
    arcpy_obs = _zonal(
        arcpy_runner, tmp_path, "out/oid_arcpy.dbf",
        zone_path=zones_fc, raster_path=str(raster), zone_field="",
    )
    assert gdal_obs.ok, f"gdal: {gdal_obs.summary}"
    assert arcpy_obs.ok, f"arcpy: {arcpy_obs.summary}"

    def _read(runner, path, code):
        outcome = runner.run(code.replace("__PATH__", repr(str(path))), timeout=180)
        assert outcome.ok, outcome.error
        return outcome.result["columns"], {int(r[0]): r[1:] for r in outcome.result["rows"]}

    gdal_columns, gdal_rows = _read(gdal_runner, gdal_obs.data["result"]["output"], _READ_CSV)
    arcpy_columns, arcpy_rows = _read(arcpy_runner, arcpy_obs.data["result"]["output"], _READ_ARCPY)

    assert gdal_columns == arcpy_columns == ["OBJECTID", "COUNT", "AREA", "SUM"]
    assert gdal_rows == arcpy_rows


#: 一块在栅格内（11）+ 一块完全在栅格外（55：x 5000..6000，栅格只到 3000）
OUTSIDE_ZONE_SPECS = [(11, "Z1", 0), (55, "Z5", 5000)]


def test_zones_outside_the_raster_are_skipped_by_both_backends(
    gdal_runner, arcpy_runner, tmp_path: Path
):
    """个别分区在栅格外：两端都跳过它、不出行，不报错（pitfalls 那条的前半句）。"""
    zones_fc = _arcpy_zones(arcpy_runner, tmp_path, OUTSIDE_ZONE_SPECS)
    _, raster = _make_inputs(gdal_runner, tmp_path)

    gdal_obs = _zonal(
        gdal_runner, tmp_path, "out/outside.csv", zone_path=zones_fc, raster_path=str(raster)
    )
    arcpy_obs = _zonal(
        arcpy_runner, tmp_path, "out/outside.dbf", zone_path=zones_fc, raster_path=str(raster)
    )
    assert gdal_obs.ok, f"gdal: {gdal_obs.summary}"
    assert arcpy_obs.ok, f"arcpy: {arcpy_obs.summary}"

    gdal_table = _table(gdal_runner, gdal_obs.data["result"]["output"])
    arcpy_table = _table(arcpy_runner, arcpy_obs.data["result"]["output"], arcpy=True)
    assert set(gdal_table) == set(arcpy_table) == {11}, "栅格外的 55 两端都不出行"
    assert gdal_table == arcpy_table
    assert gdal_table[11]["COUNT"] == 100.0 and gdal_table[11]["SUM"] == 450.0


def test_arcpy_errors_when_every_zone_is_outside_the_raster(
    gdal_runner, arcpy_runner, tmp_path: Path
):
    """全部分区在栅格外：arcpy 整体 ERROR 010568，GDAL 写 0 行表后栽在 feature_count_gt。"""
    zones_fc = _arcpy_zones(arcpy_runner, tmp_path, [(55, "Z5", 5000)])
    _, raster = _make_inputs(gdal_runner, tmp_path)

    # 先跑 GDAL：它顺手建出 out/（arcpy 不建目录，缺目录会先报 000875，把 010568 遮掉）
    gdal_obs = _zonal(
        gdal_runner, tmp_path, "out/all_out.csv", zone_path=zones_fc, raster_path=str(raster)
    )
    arcpy_obs = _zonal(
        arcpy_runner, tmp_path, "out/all_out.dbf", zone_path=zones_fc, raster_path=str(raster)
    )

    assert not arcpy_obs.ok
    assert "010568" in arcpy_obs.summary, arcpy_obs.summary
    assert not gdal_obs.ok
    assert "feature_count_gt" in gdal_obs.summary and "count=0" in gdal_obs.summary, gdal_obs.summary


# ------------------------------------------------- 缺后端的契约（P0-1a）
class _GdalOnlyRunner:
    """假 GDAL 执行器：只声明 gdal，用来验「配方没声明 gdal」时的报错。"""

    backends = {"gdal"}

    def __init__(self):
        self.codes: list[str] = []

    def run(self, code, *, timeout=None, workspace=None):
        self.codes.append(code)
        return SimpleNamespace(ok=True, stdout="", stderr="", result={"ok": True}, error=None)


def test_arcpy_only_recipe_still_reports_missing_backend(tmp_path: Path):
    """backend 里没有 gdal 的配方，在 gdal 后端下必须明确报错，且一行代码都不执行。

    用的配方是 ``create_file_gdb``：GDAL 根本没有写 FGDB 的驱动，它永远只能 arcpy。
    （以前拿 ``polygon_to_raster`` 当样本，本批把它改成双后端后就不能再这么用了。）
    """
    runner = _GdalOnlyRunner()
    lib = RecipeLibrary(code_runner=runner)
    lib.load()
    ctx = SimpleNamespace(workspace=tmp_path, code_runner=runner, exec_timeout=10)

    obs = lib.run("create_file_gdb", {"gdb_path": str(tmp_path / "out.gdb")}, ctx)

    assert not obs.ok
    assert obs.error["type"] == "BackendUnavailable"
    assert "需 ArcGIS" in obs.summary
    assert not runner.codes, "缺后端时不该把任何模板交给执行器"


# ------------------------------------------- 5 m 小面：逐分区 MEAN 相对差 ≤ 3%（DAIR-139 契约）
#: 5 m 栅格 + 四个 9×9 像元的小面，全部偏移半像元（2.5 m）：
#: 两端对边界像元的归属规则不同，COUNT 不承诺相等（DAIR-134 实测 2341 社区 × 5 m
#: 只有 10.0% 严格相等），浮点量按逐分区 MEAN 相对差 ≤ 3% 验收 —— 这是更严的构造性
#: 断言；跨后端分布口径见 zonal_statistics.yaml:393（p95 ≤ 3%，逐分区卡 3% 在 2341
#: 社区真实数据上不成立）。
FINE_ZONE_SPECS = [
    (1, 2.5, 2.5, 45.0),
    (2, 102.5, 2.5, 45.0),
    (3, 2.5, 102.5, 45.0),
    (4, 102.5, 102.5, 45.0),
]

_MAKE_FINE_RASTER = "\n".join(
    [
        "import numpy as np",
        "import rasterio",
        "from rasterio.transform import from_origin",
        "rows = np.arange(60, dtype='float64').reshape(60, 1)",
        "cols = np.arange(60, dtype='float64').reshape(1, 60)",
        "data = (1000.0 + np.mod(rows * 7 + cols * 13, 17)).astype('float32')",
        "with rasterio.open(__RASTER__, 'w', driver='GTiff', height=60, width=60, count=1,",
        "                   dtype='float32', crs='EPSG:32650',",
        "                   transform=from_origin(0.0, 300.0, 5.0, 5.0), nodata=-9999.0) as ds:",
        "    ds.write(data, 1)",
        "set_result({'raster': str(__RASTER__)})",
    ]
)

_MAKE_ARCPY_FINE_ZONES = "\n".join(
    [
        "import arcpy",
        "arcpy.env.overwriteOutput = True",
        "sr = arcpy.SpatialReference(32650)",
        "gdb = __GDB__",
        "if not arcpy.Exists(gdb):",
        "    arcpy.management.CreateFileGDB(__PARENT__, 'zonal_fine.gdb')",
        "if arcpy.Exists(gdb + '\\zones'):",
        "    arcpy.management.Delete(gdb + '\\zones')",
        "arcpy.management.CreateFeatureclass(gdb, 'zones', 'POLYGON', spatial_reference=sr)",
        "arcpy.management.AddField(gdb + '\\zones', 'zone_id', 'LONG')",
        "for zid, x0, y0, size in __SPECS__:",
        "    pts = [(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)]",
        "    with arcpy.da.InsertCursor(gdb + '\\zones', ['SHAPE@', 'zone_id']) as cur:",
        "        cur.insertRow([arcpy.Polygon(arcpy.Array([arcpy.Point(*p) for p in pts]), sr), zid])",
        "set_result({'zones': gdb + '\\zones', 'gdb': gdb})",
    ]
)


def _fine_raster(gdal_runner, tmp_path: Path) -> Path:
    raster = tmp_path / "fine5.tif"
    made = gdal_runner.run(_MAKE_FINE_RASTER.replace("__RASTER__", repr(str(raster))), timeout=300)
    assert made.ok, made.error
    return raster


def _fine_zones(arcpy_runner, tmp_path: Path) -> str:
    gdb = tmp_path / "zonal_fine.gdb"
    made = arcpy_runner.run(
        _MAKE_ARCPY_FINE_ZONES.replace("__GDB__", repr(str(gdb)))
        .replace("__PARENT__", repr(str(tmp_path)))
        .replace("__SPECS__", repr(FINE_ZONE_SPECS)),
        timeout=300,
    )
    assert made.ok, made.error
    return made.result["zones"]


def test_arcpy_and_gdal_mean_within_three_percent_on_five_meter_zones(
    gdal_runner, arcpy_runner, tmp_path: Path
):
    """5 m 小面偏移半像元：COUNT 不承诺相等，逐分区 MEAN 相对差远小于 3%，故按 ≤ 3% 卡。

    DAIR-134（2341 社区 × 5 m）实测 MEAN 中位 0.40%、p95 2.33%、max 14.2%，COUNT
    只有 10.0% 的分区严格相等；这里用四个 9×9 小面把「边界像元归属规则不同」
    稳定复现出来，逐分区实测相对差远小于 3%。跨后端分布口径见 zonal_statistics.yaml:393
    （p95 ≤ 3%，逐分区卡 3% 在 2341 社区真实数据上不成立），本用例是更严的构造性
    断言，因此按逐分区 ≤ 3% 卡，且不再断言 COUNT 相等。
    """
    raster = _fine_raster(gdal_runner, tmp_path)
    zones_fc = _fine_zones(arcpy_runner, tmp_path)

    gdal_obs = _zonal(gdal_runner, tmp_path, "out/fine_gdal.csv", zone_path=zones_fc,
                      raster_path=str(raster), statistics="MEAN")
    arcpy_obs = _zonal(arcpy_runner, tmp_path, "out/fine_arcpy.dbf", zone_path=zones_fc,
                       raster_path=str(raster), statistics="MEAN")
    assert gdal_obs.ok, f"gdal: {gdal_obs.summary}"
    assert arcpy_obs.ok, f"arcpy: {arcpy_obs.summary}"

    gdal_table = _table(gdal_runner, gdal_obs.data["result"]["output"])
    arcpy_table = _table(arcpy_runner, arcpy_obs.data["result"]["output"], arcpy=True)
    zone_ids = {z[0] for z in FINE_ZONE_SPECS}
    assert set(gdal_table) == set(arcpy_table) == zone_ids

    count_diff = 0
    for zid in sorted(zone_ids):
        gdal_row, arcpy_row = gdal_table[zid], arcpy_table[zid]
        if gdal_row["COUNT"] != arcpy_row["COUNT"]:
            count_diff += 1
        diff = _relative_diff(gdal_row["MEAN"], arcpy_row["MEAN"])
        print(f"zone {zid}: COUNT arcpy={arcpy_row['COUNT']} gdal={gdal_row['COUNT']} | "
              f"MEAN arcpy={arcpy_row['MEAN']} gdal={gdal_row['MEAN']} | rel={diff:.4%}")
        assert diff is not None and diff <= 0.03, (
            f"zone {zid} MEAN 相对差 {diff:.4%} 超过 3%："
            f"arcpy={arcpy_row['MEAN']} gdal={gdal_row['MEAN']}"
        )

    # 构造必须真的复现「边界像元归属不同」，否则这条用例卡的是空气
    assert count_diff >= 1, "偏移半像元的构造没有产生 COUNT 差异，用例没验证到契约"

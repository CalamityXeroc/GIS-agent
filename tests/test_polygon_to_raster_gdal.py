# -*- coding: utf-8 -*-
"""polygon_to_raster 的 GDAL 版真跑 + **逐像元**双端一致。

测试数据（EPSG:32650，米）：

    三块 1 km × 1 km 分区，x 依次相接（11: 0..1000 / 22: 1000..2000 / 33: 2000..3000）
    cell_size=100 → 30 列 × 10 行，格心在 50/150/…，不压在分区边界上

``REFERENCE_ALIGNED`` 那组数是**真 arcpy 跑出来的**（ArcGIS Pro 3.6，证据在
``benchmark/reports/dair105_polygon_raster_20261029/work/probe_evidence*.json``），
不是从 GDAL 侧抄回来的。它们同时钉住 arcpy 的四条实测口径：

1. 输出网格 origin = 输入 bbox 的 (minx, miny)、宽高 = round(bbox 宽/像元大小)
   （bbox 230..1780 → 输出 extent 230..1830、16 列；3000/37 → 81 列）；
2. ``cell_assignment`` 的取值域是 MAXIMUM_AREA / MAXIMUM_COMBINED_AREA / **CELL_CENTER**
   （传 CENTROID 报 ERROR 000800），两者的判据在「格心在 A 面、B 面占面积更大」时分道
   —— 所以 GDAL 后端只做 CELL_CENTER，另两个明确报错；
3. 整型输出按值域挑最省 dtype（11/22/33 → S8、nodata=-128），浮点是 F32/F64；
4. 取值字段为 NULL 的要素不参与栅格化（那块是 nodata），但几何照样进 bbox。

缺环境就 skip（与 ``test_zonal_statistics_raster_gdal.py`` 同一套口径）：
``.venv-gdal`` 不在 → 所有真 GDAL 用例 skip；ArcGIS Pro 找不到 → 只有双端用例 skip。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gis_cli.recipes.schema import Recipe
from gis_cli.recipes.store import RecipeLibrary
from gis_cli.runner.gdal_runner import GdalCodeRunner, default_python_executable
from gis_cli.runtime.code_runner import CodeRunner

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "src/gis_cli/recipes/builtin/polygon_to_raster.yaml"

#: (取值, 左下角 x)：三块 1 km 方块，y 都是 0..1000
ZONE_SPECS = [(11, 0), (22, 1000), (33, 2000)]

#: arcpy（Pro 3.6）实测：网格对齐的 3 块分区 + 100 m 像元
REFERENCE_ALIGNED = {"shape": [10, 30], "extent": [0.0, 0.0, 3000.0, 1000.0]}

#: 不齐格的 bbox：单个面 (230,170)-(1786,946) → 16×8、extent 230..1830 / 170..970
#: （右边界故意留出 6 米，让右上角那格的格心 (1780, 920) 落在面内：
#:  格心正好压在面边界上时两端取舍不同，见 pitfalls，别拿它当一致性用例）
UNALIGNED_BOX = [(230, 170), (1786, 170), (1786, 946), (230, 946)]
UNALIGNED_EXPECT = {"shape": [8, 16], "extent": [230.0, 170.0, 1830.0, 970.0]}

#: 输出 nodata：GDAL 版整型用 int32 最小值、浮点用与 arcpy F32 同一哨兵值
NODATA_INT = -2147483648
NODATA_FLOAT = -3.4028234663852886e38

_MAKE_GPKG = "\n".join(
    [
        "import pandas as pd",
        "import geopandas as gpd",
        "from shapely.geometry import Polygon",
        "_specs = __SPECS__",
        "values = pd.Series([s[0] for s in _specs], dtype=__DTYPE__)",
        "gdf = gpd.GeoDataFrame({__FIELD__: values},",
        "                       geometry=[Polygon(s[1]) for s in _specs], crs=__CRS__)",
        "gdf.to_file(__PATH__, driver='GPKG', engine='pyogrio', layer=__LAYER__)",
        "set_result({'ok': True, 'dtype': str(gdf[__FIELD__].dtype)})",
    ]
)

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
        "arcpy.management.AddField(_fc, 'val_d', 'DOUBLE')",
        "_specs = __SPECS__",
        "with arcpy.da.InsertCursor(_fc, ['SHAPE@', 'zone_id', 'val_d']) as cur:",
        "    for value, pts in _specs:",
        "        cur.insertRow([arcpy.Polygon(arcpy.Array([arcpy.Point(*p) for p in pts]), sr),",
        "                       value, None if value is None else float(value) + 0.5])",
        "set_result({'zones': _fc})",
    ]
)

_MAKE_ARCPY_GPKG = "\n".join(
    [
        "import arcpy",
        "arcpy.env.overwriteOutput = True",
        "sr = arcpy.SpatialReference(32650)",
        "_gpkg = __GPKG__",
        "if arcpy.Exists(_gpkg):",
        "    arcpy.management.Delete(_gpkg)",
        "arcpy.management.CreateSQLiteDatabase(_gpkg, 'GEOPACKAGE')",
        "arcpy.management.CreateFeatureclass(_gpkg, 'zones', 'POLYGON', spatial_reference=sr)",
        "_fc = _gpkg + '\\\\zones'",
        "arcpy.management.AddField(_fc, 'zone_id', 'LONG')",
        "with arcpy.da.InsertCursor(_fc, ['SHAPE@', 'zone_id']) as cur:",
        "    for value, pts in __SPECS__:",
        "        cur.insertRow([arcpy.Polygon(arcpy.Array([arcpy.Point(*p) for p in pts]), sr), value])",
        "set_result({'gpkg': _gpkg, 'zones': _fc})",
    ]
)

#: .gdb 的**要素数据集**里造要素类（``ds.gdb\fd\zones``）：GDAL 的图层名里没有 ``fd`` 这一段，
#: 而 arcpy 自己认这种写法——DAIR-109 的那条两段路径。
_MAKE_ARCPY_DATASET_ZONES = "\n".join(
    [
        "import arcpy, os",
        "arcpy.env.overwriteOutput = True",
        "sr = arcpy.SpatialReference(32650)",
        "gdb = __GDB__",
        "if not arcpy.Exists(gdb):",
        "    arcpy.management.CreateFileGDB(os.path.dirname(gdb), os.path.basename(gdb))",
        "arcpy.management.CreateFeatureDataset(gdb, __DATASET__, sr)",
        "_fd = os.path.join(gdb, __DATASET__)",
        "_fc = os.path.join(_fd, 'zones')",
        "if arcpy.Exists(_fc):",
        "    arcpy.management.Delete(_fc)",
        "arcpy.management.CreateFeatureclass(_fd, 'zones', 'POLYGON', spatial_reference=sr)",
        "arcpy.management.AddField(_fc, 'zone_id', 'LONG')",
        "with arcpy.da.InsertCursor(_fc, ['SHAPE@', 'zone_id']) as cur:",
        "    for value, pts in __SPECS__:",
        "        cur.insertRow([arcpy.Polygon(arcpy.Array([arcpy.Point(*p) for p in pts]), sr), value])",
        "set_result({'fc': _fc, 'gdb': gdb})",
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
        "                'origin': [float(_t.c), float(_t.f)],",
        "                'extent': [float(_b.left), float(_b.bottom), float(_b.right), float(_b.top)],",
        "                'crs': ds.crs.to_string() if ds.crs else None,",
        "                'dtype': ds.dtypes[0],",
        "                'nodata': ds.nodata,",
        "                'n_valid': int(_a.count())})",
    ]
)

_READ_ARCPY_RASTER = "\n".join(
    [
        "import arcpy, numpy as np",
        "_d = arcpy.Describe(__PATH__)",
        "_r = arcpy.Raster(__PATH__)",
        "_raw = arcpy.RasterToNumPyArray(__PATH__)",
        "_a = _raw.astype('float64')",
        "if _r.noDataValue is not None and not np.isnan(_r.noDataValue):",
        "    _a[_raw == _r.noDataValue] = np.nan",
        "_e = _d.extent",
        "set_result({'values': _a.tolist(),",
        "            'shape': [int(_a.shape[0]), int(_a.shape[1])],",
        "            'cell': [float(_d.meanCellWidth), float(_d.meanCellHeight)],",
        "            'extent': [float(_e.XMin), float(_e.YMin), float(_e.XMax), float(_e.YMax)],",
        "            'crs': int(_d.spatialReference.factoryCode) if _d.spatialReference else None,",
        "            'dtype': _d.pixelType,",
        "            'nodata': None if _r.noDataValue is None else float(_r.noDataValue),",
        "            'n_valid': int(np.count_nonzero(~np.isnan(_a)))})",
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


# --------------------------------------------------------------------- 工具
def _box(x0: int, x1: int, y0: int = 0, y1: int = 1000) -> list[list[int]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _make_gpkg(
    runner,
    tmp_path: Path,
    specs,
    *,
    field: str = "zone_id",
    dtype: str = "Int64",
    crs: str = "EPSG:32650",
    name: str = "zones.gpkg",
    layer: str = "zones",
) -> Path:
    """造一个单层 .gpkg；specs = [(取值, 顶点列表), ...]（取值可以是 None）。"""
    path = tmp_path / name
    made = runner.run(
        _MAKE_GPKG.replace("__SPECS__", repr(specs))
        .replace("__DTYPE__", repr(dtype))
        .replace("__FIELD__", repr(field))
        .replace("__CRS__", repr(crs))
        .replace("__PATH__", repr(str(path)))
        .replace("__LAYER__", repr(layer)),
        timeout=300,
    )
    assert made.ok, made.error
    return path


def _make_aligned_gpkg(runner, tmp_path: Path, **kwargs) -> Path:
    specs = [(value, _box(x0, x0 + 1000)) for value, x0 in ZONE_SPECS]
    return _make_gpkg(runner, tmp_path, specs, **kwargs)


def _arcpy_zones(
    arcpy_runner, tmp_path: Path, specs=None, *, field_type: str = "LONG", name: str = "ptr_inputs.gdb"
) -> str:
    """arcpy 侧的分区：真 .gdb 要素类（GDAL 版从同一份读，双端就是"同一输入"）。"""
    specs = specs or [(value, _box(x0, x0 + 1000)) for value, x0 in ZONE_SPECS]
    gdb = tmp_path / name
    made = arcpy_runner.run(
        _MAKE_ARCPY_ZONES.replace("__GDB__", repr(str(gdb)))
        .replace("__PARENT__", repr(str(tmp_path)))
        .replace("__GDB_NAME__", repr(name))
        .replace("__FIELD_TYPE__", repr(field_type))
        .replace("__SPECS__", repr(specs)),
        timeout=300,
    )
    assert made.ok, made.error
    return made.result["zones"]


def _arcpy_gpkg(arcpy_runner, tmp_path: Path, name: str = "single.gpkg",
                specs=None) -> Path:
    """arcpy 侧的 .gpkg（arcpy 的 Python 里没有 geopandas，只能用 CreateSQLiteDatabase）。"""
    specs = specs or [(value, _box(x0, x0 + 1000)) for value, x0 in ZONE_SPECS]
    path = tmp_path / name
    made = arcpy_runner.run(
        _MAKE_ARCPY_GPKG.replace("__GPKG__", repr(str(path))).replace("__SPECS__", repr(specs)),
        timeout=300,
    )
    assert made.ok, made.error
    return path


def _arcpy_dataset_zones(
    arcpy_runner, tmp_path: Path, *, name: str = "ds.gdb", dataset: str = "fd"
) -> str:
    """要素数据集里的要素类：返回 ``ds.gdb\\fd\\zones``（取值 11/22 各 1 km → 20 列 × 10 行）。"""
    specs = [(value, _box(x0, x0 + 1000)) for value, x0 in ZONE_SPECS[:2]]
    made = arcpy_runner.run(
        _MAKE_ARCPY_DATASET_ZONES.replace("__GDB__", repr(str(tmp_path / name)))
        .replace("__DATASET__", repr(dataset))
        .replace("__SPECS__", repr(specs)),
        timeout=300,
    )
    assert made.ok, made.error
    return made.result["fc"]


def _run(runner, tmp_path: Path, **params):
    lib = RecipeLibrary(code_runner=runner)
    lib.load()
    ctx = SimpleNamespace(workspace=tmp_path, code_runner=runner, exec_timeout=600)
    return lib.run("polygon_to_raster", params, ctx)


def _rasterize(runner, tmp_path: Path, name: str, **overrides):
    params = {
        "input_path": str(tmp_path / "zones.gpkg"),
        "value_field": "zone_id",
        "output_path": str(tmp_path / name),
        "cell_size": 100,
        "cell_assignment": "CELL_CENTER",
    }
    params.update(overrides)
    return _run(runner, tmp_path, **params)


def _result(obs, *, expect_ok: bool = True) -> dict:
    """取配方返回值。

    ``expect_ok=False`` 用于**输出本身就是常量**的用例（单个面只有一种取值）：
    配方自带的 ``raster_not_constant`` 断言会判否——那是配方在防"整幅同值"的坏产物，
    不是本用例要验的东西，这里只要求代码跑完、返回值在。
    """
    if expect_ok:
        assert obs.ok, obs.summary
    else:
        assert obs.data.get("result"), obs.summary
    return obs.data["result"]


def _raster(runner, path: str, *, arcpy: bool = False) -> dict:
    code = _READ_ARCPY_RASTER if arcpy else _READ_RASTER
    outcome = runner.run(code.replace("__PATH__", repr(str(path))), timeout=300)
    assert outcome.ok, outcome.error
    grid = outcome.result
    grid["array"] = np.array(grid["values"], dtype="float64")
    return grid


def _assert_same_grid(gdal_grid: dict, arcpy_grid: dict) -> None:
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
        "input_path": "z.gdb/zones",
        "value_field": "ELDER_N",
        "output_path": "out/elder.tif",
        "cell_size": 5,
        "cell_assignment": "CELL_CENTER",
    }
    arcpy_code = recipe.render(params)
    gdal_code = recipe.render(params, backend="gdal")

    assert "arcpy.conversion.PolygonToRaster(" in arcpy_code and "rasterio" not in arcpy_code
    assert "from rasterio.features import rasterize" in gdal_code and "import arcpy" not in gdal_code
    for key in ("z.gdb/zones", "ELDER_N", "out/elder.tif", "CELL_CENTER"):
        assert key in arcpy_code and key in gdal_code
    # 容器输入复用 DAIR-98 的片段（不另写一份容器判定）
    assert arcpy_code.count("def resolve_zone_layer(path):") == 1
    assert arcpy_code.count("resolve_zone_layer(") == 2  # 定义 + 一次调用
    # 两端都把 cell_assignment 的取值域与"另两个值 GDAL 做不出来"写在代码/文档里
    pitfalls = "".join(recipe.pitfalls)
    assert "MAXIMUM_AREA / MAXIMUM_COMBINED_AREA / CELL_CENTER" in pitfalls
    assert "GDAL 后端**只支持 CELL_CENTER**" in pitfalls
    assert "在 GDAL 后端做不出来" in gdal_code
    assert "MAXIMUM_COMBINED_AREA" in arcpy_code


def test_gdal_template_keeps_the_arcpy_grid_and_overlap_rules():
    """网格与重叠口径必须与 arcpy 实测一致：origin=bbox 左下角、round 宽高、反序喂几何。"""
    params = {
        "input_path": "zones.gpkg",
        "value_field": "zone_id",
        "output_path": "out/x.tif",
        "cell_size": 100,
        "cell_assignment": "CELL_CENTER",
    }
    code = Recipe.from_yaml(RECIPE).render(params, backend="gdal")

    assert "from_origin(minx, miny + height * cell_size, cell_size, cell_size)" in code
    assert "int((maxx - minx) / cell_size + 0.5)" in code
    assert "list(reversed(shapes))" in code
    assert "all_touched=False" in code


# ----------------------------------------------------------- GDAL 侧真跑
def test_gdal_writes_values_on_the_arcpy_grid(gdal_runner, tmp_path: Path):
    """3 块分区 + 100 m 像元：网格与 arcpy 实测逐项相同，每块写自己的值。"""
    _make_aligned_gpkg(gdal_runner, tmp_path)
    obs = _rasterize(gdal_runner, tmp_path, "out/elder.tif")

    assert obs.ok, obs.summary
    result = obs.data["result"]
    assert result["value_field"] == "zone_id"
    assert result["size"] == REFERENCE_ALIGNED["shape"][::-1]
    assert (result["min"], result["max"]) == (11.0, 33.0)
    assert result["cell_count"] == 300

    grid = _raster(gdal_runner, result["output"])
    assert grid["shape"] == REFERENCE_ALIGNED["shape"]
    assert grid["crs"] == "EPSG:32650"
    assert grid["cell"] == [100.0, 100.0]
    assert grid["extent"] == REFERENCE_ALIGNED["extent"]
    assert grid["origin"] == [0.0, 1000.0]
    assert grid["dtype"] == "int32"
    assert grid["nodata"] == NODATA_INT
    assert grid["n_valid"] == 300, "三块分区铺满整幅栅格，不该有 nodata"
    for value, column in ((11, 0), (22, 10), (33, 20)):
        block = grid["array"][:, column:column + 10]
        assert np.all(block == value), (value, np.unique(block))


def test_gdal_grid_origin_is_the_bbox_min_and_width_rounds_half_up(gdal_runner, tmp_path: Path):
    """origin = 输入 bbox 左下角（不外扩到格线），宽高 = round(bbox 宽 / 像元大小)。"""
    _make_gpkg(gdal_runner, tmp_path, [(5, UNALIGNED_BOX)])
    obs = _rasterize(gdal_runner, tmp_path, "out/un.tif")
    result = _result(obs, expect_ok=False)  # 单个面 → 整幅同值，raster_not_constant 判否

    grid = _raster(gdal_runner, result["output"])
    assert grid["shape"] == UNALIGNED_EXPECT["shape"], "bbox 230..1786 宽 1556/100=15.56 → 16 列"
    assert grid["extent"] == UNALIGNED_EXPECT["extent"]
    assert grid["origin"] == [230.0, 970.0], "origin = (bbox.minx, bbox.maxy)，不是对齐到 0"
    # 右边界留了 6 米：右上角那格的格心 (1780, 920) 在面内 → 128 格全有值（arcpy 实测同）
    assert grid["n_valid"] == 128
    assert np.all(grid["array"] == 5.0)


@pytest.mark.parametrize("bbox_w,expect_cols", [(330, 3), (350, 4), (355, 4), (440, 4), (450, 5)])
def test_gdal_width_rounds_like_arcpy(gdal_runner, tmp_path: Path, bbox_w, expect_cols):
    """0.5 格进位的实测表（arcpy：3.3→3、3.5→4、4.4→4、4.5→5 列）。"""
    _make_gpkg(gdal_runner, tmp_path, [(7, _box(0, bbox_w, 0, 300))])
    obs = _rasterize(gdal_runner, tmp_path, f"out/w{bbox_w}.tif")
    result = _result(obs, expect_ok=False)

    grid = _raster(gdal_runner, result["output"])
    assert grid["shape"] == [3, expect_cols], (bbox_w, grid["shape"])


def test_gdal_cell_size_37_keeps_square_cells_and_the_rounded_shape(gdal_runner, tmp_path: Path):
    """像元大小是参数标量 → 恒为方形；3000×1000 / 37 = 81.08×27.03 → 81 列 × 27 行。

    （arcpy 实测同：extent 0..2997 / 0..999。本配方不像 zonal_statistics_raster 那样
    会被输入栅格拉成方形像元——它根本没有输入栅格。）
    """
    _make_aligned_gpkg(gdal_runner, tmp_path)
    obs = _rasterize(gdal_runner, tmp_path, "out/cell37.tif", cell_size=37)
    result = _result(obs, expect_ok=False)  # 像元太细，整幅仍是三块常量 → 常量断言判否

    grid = _raster(gdal_runner, result["output"])
    assert grid["shape"] == [27, 81]
    assert grid["cell"] == [37.0, 37.0]
    assert grid["extent"] == [0.0, 0.0, 2997.0, 999.0]
    assert set(np.unique(grid["array"][~np.isnan(grid["array"])])) == {11.0, 22.0, 33.0}


def test_gdal_centroid_alias_is_cell_center(gdal_runner, tmp_path: Path):
    """CENTROID 是 Pro 3.6 之前的名字：按 CELL_CENTER 处理，输出逐像元相同。"""
    _make_gpkg(
        gdal_runner,
        tmp_path,
        [(1, _box(0, 1000)), (2, _box(1000, 2000))],
    )
    alias = _rasterize(gdal_runner, tmp_path, "out/alias.tif", cell_assignment="CENTROID")
    canonical = _rasterize(gdal_runner, tmp_path, "out/canonical.tif", cell_assignment="CELL_CENTER")
    assert _result(alias)["size"] == [20, 10]  # 与 CELL_CENTER 同一结果，先确认真跑了
    assert _result(canonical)["size"] == [20, 10]

    a = _raster(gdal_runner, alias.data["result"]["output"])
    b = _raster(gdal_runner, canonical.data["result"]["output"])
    _assert_same_grid(a, b)
    _assert_same_pixels(a, b)
    assert a["array"].tolist() == b["array"].tolist()


@pytest.mark.parametrize("assignment", ["MAXIMUM_AREA", "MAXIMUM_COMBINED_AREA", "maximum_area"])
def test_gdal_rejects_area_based_cell_assignment(gdal_runner, tmp_path: Path, assignment):
    """面积判据（含默认值 MAXIMUM_AREA）在 GDAL 后端明确报错，不静默退化成格心。"""
    _make_aligned_gpkg(gdal_runner, tmp_path)
    obs = _rasterize(gdal_runner, tmp_path, "out/area.tif", cell_assignment=assignment)

    assert not obs.ok
    assert "在 GDAL 后端做不出来" in obs.summary
    assert "CELL_CENTER" in obs.summary and "all_touched" in obs.summary
    assert "49% → nodata" in obs.summary, "报错里要写清 arcpy 的覆盖率门槛（实测事实）"
    assert not (tmp_path / "out/area.tif").exists()


def test_gdal_rejects_unknown_cell_assignment(gdal_runner, tmp_path: Path):
    _make_aligned_gpkg(gdal_runner, tmp_path)
    obs = _rasterize(gdal_runner, tmp_path, "out/bogus.tif", cell_assignment="BOGUS")

    assert not obs.ok
    assert "不认识" in obs.summary and "MAXIMUM_COMBINED_AREA" in obs.summary
    assert not (tmp_path / "out/bogus.tif").exists()


def test_gdal_overlapping_polygons_let_the_first_feature_win(gdal_runner, tmp_path: Path):
    """重叠面：arcpy 实测「先出现的要素优先」（插入顺序反过来结果就反过来），GDAL 照此实现。"""
    overlap = [(1, _box(0, 1000)), (2, _box(400, 1400))]
    _make_gpkg(gdal_runner, tmp_path, overlap)
    obs = _rasterize(gdal_runner, tmp_path, "out/overlap.tif")

    assert obs.ok, obs.summary
    grid = _raster(gdal_runner, obs.data["result"]["output"])
    assert grid["shape"] == [10, 14]  # bbox 0..1400 → 14 列
    # 重叠的列 4..9 归先出现的 value=1（后出现的 value=2 只占 10..13）
    assert np.all(grid["array"][:, :10] == 1.0), np.unique(grid["array"][:, :10])
    assert np.all(grid["array"][:, 10:14] == 2.0), np.unique(grid["array"][:, 10:14])

    # 同一几何、插入顺序反过来 → 重叠像元归后插入的（即当时的"第一个"）那条
    _make_gpkg(gdal_runner, tmp_path, list(reversed(overlap)), name="rev.gpkg")
    rev = _rasterize(gdal_runner, tmp_path, "out/rev.tif", input_path=str(tmp_path / "rev.gpkg"))
    assert rev.ok, rev.summary
    rev_grid = _raster(gdal_runner, rev.data["result"]["output"])
    assert np.all(rev_grid["array"][:, :4] == 1.0), np.unique(rev_grid["array"][:, :4])
    assert np.all(rev_grid["array"][:, 4:14] == 2.0), np.unique(rev_grid["array"][:, 4:14])


def test_gdal_skips_null_values_but_keeps_them_in_the_grid(gdal_runner, tmp_path: Path):
    """取值字段为 NULL 的要素不参与栅格化（那几列 nodata），但几何照样进 bbox。"""
    specs = [(11, _box(0, 1000)), (22, _box(1000, 2000)), (33, _box(2000, 3000)),
             (None, _box(3000, 4000))]
    _make_gpkg(gdal_runner, tmp_path, specs)
    obs = _rasterize(gdal_runner, tmp_path, "out/nulls.tif")

    assert obs.ok, obs.summary
    grid = _raster(gdal_runner, obs.data["result"]["output"])
    assert grid["shape"] == [10, 40], "空取值的那块照样进 bbox（arcpy 实测 40 列）"
    assert grid["n_valid"] == 300
    assert np.all(np.isnan(grid["array"][:, 30:40])), "NULL 取值那块应该是 nodata"
    for value, column in ((11, 0), (22, 10), (33, 20)):
        assert np.all(grid["array"][:, column:column + 10] == value)


def test_gdal_dtype_follows_the_value_field_type(gdal_runner, tmp_path: Path):
    """整型字段 → int32 + int32 最小值 nodata；浮点字段 → float64 + F32 哨兵。"""
    _make_aligned_gpkg(gdal_runner, tmp_path)
    int_obs = _rasterize(gdal_runner, tmp_path, "out/int.tif")
    assert int_obs.ok, int_obs.summary
    int_grid = _raster(gdal_runner, int_obs.data["result"]["output"])
    assert int_grid["dtype"] == "int32" and int_grid["nodata"] == NODATA_INT

    _make_gpkg(
        gdal_runner,
        tmp_path,
        [(11.5, _box(0, 1000)), (22.5, _box(1000, 2000))],
        field="val",
        dtype="float64",
        name="floats.gpkg",
    )
    float_obs = _rasterize(
        gdal_runner, tmp_path, "out/float.tif",
        input_path=str(tmp_path / "floats.gpkg"), value_field="val",
    )
    assert float_obs.ok, float_obs.summary
    float_grid = _raster(gdal_runner, float_obs.data["result"]["output"])
    assert float_grid["dtype"] == "float64"
    assert float_grid["nodata"] == pytest.approx(NODATA_FLOAT)
    assert set(np.unique(float_grid["array"])) == {11.5, 22.5}


def test_gdal_rejects_text_value_field(gdal_runner, tmp_path: Path):
    _make_gpkg(
        gdal_runner, tmp_path, [("A", _box(0, 1000))], field="name", dtype="object", name="txt.gpkg"
    )
    obs = _rasterize(
        gdal_runner, tmp_path, "out/txt.tif",
        input_path=str(tmp_path / "txt.gpkg"), value_field="name",
    )

    assert not obs.ok
    assert "取值字段必须是数值型" in obs.summary
    assert not (tmp_path / "out/txt.tif").exists()


def test_gdal_rejects_empty_value_field(gdal_runner, tmp_path: Path):
    """空 value_field 不回落到 OID（arcpy 实测 ERROR 000735），但要告诉人 OID 真名怎么取。"""
    _make_aligned_gpkg(gdal_runner, tmp_path)
    obs = _rasterize(gdal_runner, tmp_path, "out/empty.tif", value_field="")

    assert not obs.ok
    assert "value_field 不能为空" in obs.summary and "000735" in obs.summary
    assert "OBJECTID" in obs.summary and "fid" in obs.summary
    assert not (tmp_path / "out/empty.tif").exists()


def test_gdal_rejects_unknown_value_field(gdal_runner, tmp_path: Path):
    _make_aligned_gpkg(gdal_runner, tmp_path)
    obs = _rasterize(gdal_runner, tmp_path, "out/nofield.tif", value_field="nope")

    assert not obs.ok
    assert "字段不存在: nope" in obs.summary and "zone_id" in obs.summary
    assert not (tmp_path / "out/nofield.tif").exists()


def test_gdal_accepts_field_names_in_any_case(gdal_runner, tmp_path: Path):
    """字段名大小写不敏感（与 arcpy 的 ListFields 同一口径，两端都解析成真名）。"""
    _make_aligned_gpkg(gdal_runner, tmp_path)
    obs = _rasterize(gdal_runner, tmp_path, "out/case.tif", value_field="ZONE_ID")

    assert obs.ok, obs.summary
    assert obs.data["result"]["value_field"] == "zone_id"


def test_arcpy_accepts_field_names_in_any_case(gdal_runner, arcpy_runner, tmp_path: Path):
    """同一份 .gpkg、同一个 'ZONE_ID'：arcpy 侧也解析成真名并跑通（不是只有 GDAL 宽容）。"""
    gpkg = _make_aligned_gpkg(gdal_runner, tmp_path)
    obs = _rasterize(
        arcpy_runner, tmp_path, "out/case_arcpy.tif",
        input_path=str(gpkg), value_field="ZONE_ID",
    )

    assert obs.ok, obs.summary
    assert obs.data["result"]["value_field"] == "zone_id"


def test_gdal_rejects_geographic_crs(gdal_runner, tmp_path: Path):
    """地理坐标系下像元大小是"度"，必须先投影（与 arcpy 侧同一条守卫）。"""
    _make_gpkg(
        gdal_runner,
        tmp_path,
        [(1, [[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01]])],
        crs="EPSG:4490",
        name="geo.gpkg",
    )
    obs = _rasterize(gdal_runner, tmp_path, "out/geo.tif", input_path=str(tmp_path / "geo.gpkg"))

    assert not obs.ok
    assert "地理坐标系" in obs.summary
    assert not (tmp_path / "out/geo.tif").exists()


def test_gdal_rejects_non_polygon_input(gdal_runner, tmp_path: Path):
    path = tmp_path / "points.gpkg"
    made = gdal_runner.run(
        "\n".join(
            [
                "import geopandas as gpd",
                "from shapely.geometry import Point",
                "gdf = gpd.GeoDataFrame({'v': [1, 2]},",
                "                       geometry=[Point(0, 0), Point(100, 100)], crs='EPSG:32650')",
                f"gdf.to_file({str(path)!r}, driver='GPKG', engine='pyogrio', layer='pts')",
                "set_result({'ok': True})",
            ]
        ),
        timeout=300,
    )
    assert made.ok, made.error
    obs = _rasterize(gdal_runner, tmp_path, "out/pts.tif", input_path=str(path), value_field="v")

    assert not obs.ok
    assert "面转栅格需要面要素" in obs.summary
    assert not (tmp_path / "out/pts.tif").exists()


@pytest.mark.parametrize(
    "output_path,want",
    [
        ("out/x.gdb/ras", "GDAL 后端写不了 gdb 里的栅格"),
        ("out/x.asc", "只支持 .tif / .tiff / .img"),
    ],
)
def test_gdal_rejects_output_formats_it_cannot_write(gdal_runner, tmp_path: Path, output_path, want):
    _make_aligned_gpkg(gdal_runner, tmp_path)
    obs = _rasterize(gdal_runner, tmp_path, output_path)

    assert not obs.ok
    assert want in obs.summary
    assert not (tmp_path / "out/x.gdb").exists()


def test_gdal_rejects_cell_size_larger_than_the_extent(gdal_runner, tmp_path: Path):
    """bbox 比像元还小 → 一个像元都算不出来，报人话（不是 rastersize 的 ValueError）。"""
    _make_gpkg(gdal_runner, tmp_path, [(1, _box(0, 40, 0, 40))])
    obs = _rasterize(gdal_runner, tmp_path, "out/small.tif", cell_size=100)

    assert not obs.ok
    assert "比像元大小 100.0 还小" in obs.summary
    assert not (tmp_path / "out/small.tif").exists()


# ------------------------------------------------------- 容器输入（两端）
def test_gdal_reads_single_layer_containers(gdal_runner, tmp_path: Path):
    """.gpkg 给文件本身（不指图层）与"只有一个 .shp 的文件夹"都要能跑（arcpy 侧实测同）。"""
    gpkg = _make_aligned_gpkg(gdal_runner, tmp_path)
    shps = tmp_path / "shps"
    shps.mkdir()
    made = gdal_runner.run(
        "\n".join(
            [
                "import geopandas as gpd",
                f"gdf = gpd.read_file({str(gpkg)!r}, engine='pyogrio')",
                f"gdf.to_file({str(shps / 'zones.shp')!r}, engine='pyogrio')",
                "set_result({'ok': True})",
            ]
        ),
        timeout=300,
    )
    assert made.ok, made.error

    for label, path in (("gpkg", gpkg), ("folder", shps)):
        obs = _rasterize(gdal_runner, tmp_path, f"out/container_{label}.tif", input_path=str(path))
        assert obs.ok, f"{label}: {obs.summary}"
        grid = _raster(gdal_runner, obs.data["result"]["output"])
        assert grid["shape"] == REFERENCE_ALIGNED["shape"], label
        assert np.all(grid["array"][:, 10:20] == 22.0), label


def _make_multi_layer_gpkg(runner, tmp_path: Path, name: str = "multi.gpkg") -> Path:
    """两层 .gpkg：``zones`` = 三块对齐分区，``towns`` = 另一份（只为制造「多层」）。"""
    _make_gpkg(
        runner, tmp_path, [(value, _box(x0, x0 + 1000)) for value, x0 in ZONE_SPECS],
        name=name, layer="zones",
    )
    _make_gpkg(runner, tmp_path, [(1, _box(0, 1000))], name=name, layer="towns")
    return tmp_path / name


def test_gdal_rejects_multi_layer_containers(gdal_runner, tmp_path: Path):
    """多层容器不替你挑：报错里列出图层名并说明怎么指明（arcpy 侧片段同一口径）。"""
    multi = _make_multi_layer_gpkg(gdal_runner, tmp_path)
    obs = _rasterize(gdal_runner, tmp_path, "out/multi.tif", input_path=str(multi))

    assert not obs.ok
    assert "多层容器" in obs.summary and "zones" in obs.summary and "towns" in obs.summary
    assert "图层名" in obs.summary
    assert not (tmp_path / "out/multi.tif").exists()


def test_gdal_reads_the_layer_form_its_own_error_suggests(gdal_runner, tmp_path: Path):
    """多层容器的报错让人写 ``xxx.gpkg/图层名``：这个形状必须真能读。

    arcpy 侧的 ``resolve_zone_layer`` 返回的就是 ``os.path.join(容器, 图层名)``，
    所以两端对同一输入得认同一种写法（探针实测 arcpy 正反斜杠都认）。
    """
    multi = _make_multi_layer_gpkg(gdal_runner, tmp_path)

    for label, sep in (("fwd", "/"), ("bs", "\\")):
        obs = _rasterize(
            gdal_runner, tmp_path, f"out/picked_{label}.tif", input_path=f"{multi}{sep}zones"
        )
        assert obs.ok, f"{label}: {obs.summary}"
        grid = _raster(gdal_runner, obs.data["result"]["output"])
        assert grid["shape"] == REFERENCE_ALIGNED["shape"], label
        assert np.all(grid["array"][:, 0:10] == 11.0), label
        assert np.all(grid["array"][:, 20:30] == 33.0), label



def test_gdal_does_not_guess_a_layer_from_a_bogus_sub_path(gdal_runner, tmp_path: Path):
    """要素数据集只存在于 .gdb：.gpkg 上的多段图层名（``pkg.gpkg/子路径/zones``）一律当写错。

    退到末段那条路只为 ``xxx.gdb/数据集名/要素类名`` 开（GDAL 的图层名里没有数据集前缀）；
    在 .gpkg 上按末段猜，会把 ``pkg.gpkg/sub/zones`` 静默读成 ``zones``——宁可不认。
    """
    multi = _make_multi_layer_gpkg(gdal_runner, tmp_path)
    obs = _rasterize(gdal_runner, tmp_path, "out/bogus_sub.tif", input_path=f"{multi}/sub/zones")

    assert not obs.ok
    assert "没有图层" in obs.summary and "zones" in obs.summary and "towns" in obs.summary
    assert not (tmp_path / "out/bogus_sub.tif").exists()


# --------------------------------------------------------------- 双端一致
def test_arcpy_and_gdal_agree_pixel_by_pixel(gdal_runner, arcpy_runner, tmp_path: Path):
    """同一份 .gdb 要素类：网格相同、nodata 掩膜相同、逐像元值相等（整型严格相等）。"""
    zones_fc = _arcpy_zones(arcpy_runner, tmp_path)

    gdal_obs = _rasterize(gdal_runner, tmp_path, "out/gdal_cc.tif", input_path=zones_fc)
    arcpy_obs = _rasterize(arcpy_runner, tmp_path, "out/arcpy_cc.tif", input_path=zones_fc)
    assert gdal_obs.ok, f"gdal: {gdal_obs.summary}"
    assert arcpy_obs.ok, f"arcpy: {arcpy_obs.summary}"

    gdal_grid = _raster(gdal_runner, gdal_obs.data["result"]["output"])
    arcpy_grid = _raster(arcpy_runner, arcpy_obs.data["result"]["output"], arcpy=True)

    _assert_same_grid(gdal_grid, arcpy_grid)
    assert gdal_grid["crs"] == "EPSG:32650" and arcpy_grid["crs"] == 32650
    assert gdal_grid["n_valid"] == arcpy_grid["n_valid"] == 300
    _assert_same_pixels(gdal_grid, arcpy_grid)
    # 整型像元值严格相等（不是"在容差内"）
    assert np.array_equal(gdal_grid["array"], arcpy_grid["array"])


def test_arcpy_and_gdal_agree_on_an_unaligned_bbox(gdal_runner, arcpy_runner, tmp_path: Path):
    """不齐格的 bbox（230..1786 × 170..946）：网格 origin/宽高、掩膜、取值三样都对得上。"""
    zones_fc = _arcpy_zones(
        arcpy_runner, tmp_path, [(5, UNALIGNED_BOX)], name="ptr_unaligned.gdb"
    )

    gdal_obs = _rasterize(gdal_runner, tmp_path, "out/gdal_un.tif", input_path=zones_fc)
    arcpy_obs = _rasterize(arcpy_runner, tmp_path, "out/arcpy_un.tif", input_path=zones_fc)
    # 单个面 → 整幅同值，配方自带的 raster_not_constant 会判否（不是本用例要验的）
    gdal_grid = _raster(gdal_runner, _result(gdal_obs, expect_ok=False)["output"])
    arcpy_grid = _raster(arcpy_runner, _result(arcpy_obs, expect_ok=False)["output"], arcpy=True)

    _assert_same_grid(gdal_grid, arcpy_grid)
    assert gdal_grid["shape"] == UNALIGNED_EXPECT["shape"]
    assert gdal_grid["extent"] == UNALIGNED_EXPECT["extent"]
    _assert_same_pixels(gdal_grid, arcpy_grid)


def test_arcpy_area_modes_really_differ_from_cell_center(arcpy_runner, tmp_path: Path):
    """默认的 MAXIMUM_AREA 与 CELL_CENTER **真的不一样**——这就是 GDAL 侧不肯静默退化的理由。

    arcpy 实测（Pro 3.6）：同一个不齐格场景，CELL_CENTER 128 个有值像元，MAXIMUM_AREA 只有
    127 个（右上角那格只被面覆盖 42.6% < 50%，落 nodata）。
    """
    zones_fc = _arcpy_zones(arcpy_runner, tmp_path, [(5, UNALIGNED_BOX)], name="ptr_modes.gdb")

    cc = _rasterize(arcpy_runner, tmp_path, "out/arcpy_cc.tif", input_path=zones_fc)
    ma = _rasterize(
        arcpy_runner, tmp_path, "out/arcpy_ma.tif",
        input_path=zones_fc, cell_assignment="MAXIMUM_AREA",
    )
    cc_grid = _raster(arcpy_runner, _result(cc, expect_ok=False)["output"], arcpy=True)
    ma_grid = _raster(arcpy_runner, _result(ma, expect_ok=False)["output"], arcpy=True)

    assert cc_grid["n_valid"] == 128
    assert ma_grid["n_valid"] == 127, "MAXIMUM_AREA 带 ≥50% 覆盖率门槛（49% → nodata）"
    diff = np.isnan(cc_grid["array"]) != np.isnan(ma_grid["array"])
    assert int(diff.sum()) == 1 and diff[0, -1], "差异就在右上角那一格"


def test_arcpy_combined_area_pools_the_same_value_across_polygons(arcpy_runner, tmp_path: Path):
    """三种取值真的不等价：MAXIMUM_AREA 与 MAXIMUM_COMBINED_AREA 会分道（Pro 3.6 实测）。

    单个 100×100 像元、覆盖 86%：value 1 = 一块 30%；value 2 = 两块各 28%（合计 56%，
    每块都小于 30%）。实测 MAXIMUM_AREA → 1、MAXIMUM_COMBINED_AREA → 2；CELL_CENTER →
    nodata（格心 (50,50) 落在 x30..100 × y40..60 的空档里，它没有覆盖率门槛）。

    这就是 GDAL 后端拒绝这两个取值的依据：rasterio 的 all_touched 一种语义都表达不了。
    """
    specs = [
        (1, [[0, 0], [30, 0], [30, 100], [0, 100]]),
        (2, [[30, 0], [100, 0], [100, 40], [30, 40]]),
        (2, [[30, 60], [100, 60], [100, 100], [30, 100]]),
    ]
    zones_fc = _arcpy_zones(arcpy_runner, tmp_path, specs, name="ptr_modes2.gdb")

    def grid(mode: str) -> dict:
        obs = _rasterize(
            arcpy_runner, tmp_path, f"out/m_{mode}.tif",
            input_path=zones_fc, cell_assignment=mode,
        )
        # 1×1 且只有一种取值 → 配方自带的 raster_not_constant 会判否（不是本用例要验的）
        return _raster(arcpy_runner, _result(obs, expect_ok=False)["output"], arcpy=True)

    ma, combined, center = grid("MAXIMUM_AREA"), grid("MAXIMUM_COMBINED_AREA"), grid("CELL_CENTER")

    assert ma["shape"] == combined["shape"] == center["shape"] == [1, 1]
    assert ma["array"][0, 0] == 1.0, "单面最大是 value 1（30% > 28%）"
    assert combined["array"][0, 0] == 2.0, "同值合计是 value 2（56% > 30%）"
    assert np.isnan(center["array"][0, 0]), "格心落在空档里 → CELL_CENTER 给 nodata"


def test_arcpy_template_guards_follow_the_probed_domain(arcpy_runner, tmp_path: Path):
    """arcpy 侧两条守卫（一个内核跑完，不重复起进程）：

    1. CENTROID 是旧名 → 按 CELL_CENTER 跑通（旧配方文档写的就是它）；
    2. 非法 cell_assignment → 人话报错，不是 ERROR 000800。
    """
    zones_fc = _arcpy_zones(arcpy_runner, tmp_path)

    # 1) CENTROID → 能跑，且结果与 CELL_CENTER 相同
    alias = _rasterize(arcpy_runner, tmp_path, "out/alias.tif",
                       input_path=zones_fc, cell_assignment="CENTROID")
    assert alias.ok, alias.summary
    assert "旧名" in alias.data["stdout"]
    alias_grid = _raster(arcpy_runner, alias.data["result"]["output"], arcpy=True)
    cc = _rasterize(arcpy_runner, tmp_path, "out/cc.tif",
                    input_path=zones_fc, cell_assignment="CELL_CENTER")
    cc_grid = _raster(arcpy_runner, _result(cc)["output"], arcpy=True)
    assert np.array_equal(alias_grid["array"], cc_grid["array"])

    # 2) 非法取值
    bogus = _rasterize(arcpy_runner, tmp_path, "out/bogus.tif",
                       input_path=zones_fc, cell_assignment="BOGUS")
    assert not bogus.ok
    assert "不认识" in bogus.summary and "000800" not in bogus.summary
    assert not (tmp_path / "out/bogus.tif").exists()


def test_arcpy_empty_value_field_names_the_real_oid(arcpy_runner, gdal_runner, tmp_path: Path):
    """空 value_field → 人话报错，并给出**本数据**的 OID 真名（GDAL 建的 .gpkg 是 fid）。

    硬编码 OBJECTID 的提示会把人导到不存在的字段上——DAIR-98 的教训。
    """
    gpkg = _make_gpkg(
        gdal_runner, tmp_path, [(1, _box(0, 1000)), (2, _box(1000, 2000))], name="oid.gpkg"
    )
    empty = _rasterize(arcpy_runner, tmp_path, "out/empty.tif",
                       input_path=str(gpkg), value_field="")
    assert not empty.ok
    assert "value_field 不能为空" in empty.summary
    assert "本数据是 fid" in empty.summary
    assert not (tmp_path / "out/empty.tif").exists()


def test_arcpy_reads_single_layer_containers_and_rejects_multi(arcpy_runner, tmp_path: Path):
    """容器输入走共享片段：单层 .gpkg / 只有一个 .shp 的文件夹都能跑；多层要指明图层。"""
    one = _arcpy_gpkg(arcpy_runner, tmp_path, name="single.gpkg")
    gpkg_obs = _rasterize(arcpy_runner, tmp_path, "out/arcpy_gpkg.tif", input_path=str(one))
    assert gpkg_obs.ok, gpkg_obs.summary
    assert "自动使用其中唯一的图层" in gpkg_obs.data["stdout"]

    shps = tmp_path / "one_shp"
    shps.mkdir()
    made = arcpy_runner.run(
        "\n".join(
            [
                "import arcpy",
                f"arcpy.management.CopyFeatures({str(one / 'zones')!r},"
                f" {str(shps / 'zones.shp')!r})",
                "set_result({'ok': True})",
            ]
        ),
        timeout=300,
    )
    assert made.ok, made.error
    folder_obs = _rasterize(arcpy_runner, tmp_path, "out/arcpy_folder.tif", input_path=str(shps))
    assert folder_obs.ok, folder_obs.summary

    # 容器 + 图层（``.gpkg/图层名``）：GDAL 侧也认这个形状，两端对同一输入同一写法
    picked = _rasterize(arcpy_runner, tmp_path, "out/arcpy_picked.tif", input_path=f"{one}/zones")
    assert picked.ok, picked.summary

    multi = tmp_path / "two_shp"
    multi.mkdir()
    made = arcpy_runner.run(
        "\n".join(
            [
                "import arcpy",
                f"arcpy.management.CopyFeatures({str(one / 'zones')!r},"
                f" {str(multi / 'zones.shp')!r})",
                f"arcpy.management.CopyFeatures({str(one / 'zones')!r},"
                f" {str(multi / 'towns.shp')!r})",
                "set_result({'ok': True})",
            ]
        ),
        timeout=300,
    )
    assert made.ok, made.error
    bad = _rasterize(arcpy_runner, tmp_path, "out/arcpy_multi.tif", input_path=str(multi))

    assert not bad.ok
    assert "多层容器" in bad.summary and "zones" in bad.summary and "towns" in bad.summary
    assert "AttributeError" not in bad.summary and "DescribeData" not in bad.summary
    assert not (tmp_path / "out/arcpy_multi.tif").exists()


def test_both_backends_read_a_feature_class_inside_a_feature_dataset(
    gdal_runner, arcpy_runner, tmp_path: Path
):
    """``ds.gdb/要素数据集/要素类``（正反斜杠两种写法）：两端都要跑通，都是 20 列 × 10 行。

    GDAL 的图层名里没有数据集前缀（``pyogrio.list_layers(ds.gdb)`` 只给 ``zones``），
    整段读不到得退到末段再试；arcpy 自己解析这种写法。取值 11/22 各占一半 →
    2000×1000 / 100 = [20, 10]（DAIR-107 审查的实测表就是这个形状）。
    """
    _arcpy_dataset_zones(arcpy_runner, tmp_path)

    for label, sep in (("fwd", "/"), ("bs", "\\")):
        picked = f"{tmp_path / 'ds.gdb'}{sep}fd{sep}zones"
        for backend, runner in (("gdal", gdal_runner), ("arcpy", arcpy_runner)):
            obs = _rasterize(runner, tmp_path, f"out/{backend}_ds_{label}.tif", input_path=picked)
            assert obs.ok, f"{backend}/{label}: {obs.summary}"
            result = obs.data["result"]
            assert result["size"] == [20, 10], f"{backend}/{label}: {result['size']}"
            assert [result["min"], result["max"]] == [11, 22], f"{backend}/{label}"
            if backend == "gdal":
                # 要真走「退到末段」那条路，而不是碰巧能读
                assert "要素数据集前缀" in obs.data["stdout"], obs.data["stdout"]


def test_arcpy_container_with_only_feature_datasets_points_at_the_feature_class(
    arcpy_runner, tmp_path: Path
):
    """容器里只有要素数据集：报错要指到要素类本身，而不是「里面没有要素类」。

    实测 Pro 3.6：``ds.gdb`` 上 ``ListFeatureClasses()`` 返回 []（要素类在 ``fd`` 里，
    它不递归），``ListDatasets("*", "Feature")`` 返回 ``['fd']``。旧文案会把人往
    「.gpkg/图层名」引，那是错的方向。
    """
    _arcpy_dataset_zones(arcpy_runner, tmp_path)
    obs = _rasterize(
        arcpy_runner, tmp_path, "out/arcpy_ds_container.tif", input_path=str(tmp_path / "ds.gdb")
    )

    assert not obs.ok
    assert "要素数据集" in obs.summary and "fd" in obs.summary, obs.summary
    assert "数据集名/要素类名" in obs.summary, obs.summary
    assert not (tmp_path / "out/arcpy_ds_container.tif").exists()

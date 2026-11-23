# -*- coding: utf-8 -*-
"""三个分区配方（DAIR-98 #3）：分区输入给的是容器时，不再裸抛 AttributeError。

``zonal_statistics_raster`` / ``zone_raster_sum`` / ``aggregate_to_zones`` 以前直接

    sr = arcpy.Describe(zone_path).spatialReference

把 ``.gpkg`` / ``.gdb`` / 文件夹当分区输入时，arcpy 把容器当工作空间
（``Describe().dataType`` 是 ``Workspace`` / ``Folder``），上面根本没有 spatialReference ——
抛的是 ``[AttributeError] DescribeData: 方法 spatialReference 不存在``，不是人话。

容器判定现在抽成共享片段（``src/gis_cli/recipes/fragments.py``，模板里写 ``{{@zone_layer@}}``），
三个配方都先用它解析：容器里恰好一个要素类 → 自动取用并跑通；多个图层 → 人话错误要你指明图层。
这里对三个配方各跑这两种情形（真 arcpy）。

缺环境就 skip（与 ``test_zonal_statistics_gdal.py`` 同一套口径）：ArcGIS Pro 找不到 → 全 skip，
GDAL 侧造输入用的 ``.venv-gdal`` 不在 → 同样 skip。CI 机器两者都没有，by design。
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gis_cli.arcpy_bridge import ArcGISDiscoveryError, discover_arcgis_pro_python
from gis_cli.recipes.store import RecipeLibrary
from gis_cli.runner.gdal_runner import GdalCodeRunner, default_python_executable
from gis_cli.runtime.code_runner import CodeRunner

#: 分区面（EPSG:32650，三个 1 km 方块）+ 100 m 像元栅格（像元值 = 列号 0..29）
_MAKE_INPUTS = "\n".join(
    [
        "import numpy as np",
        "import rasterio",
        "from rasterio.transform import from_origin",
        "with rasterio.open(__RASTER__, 'w', driver='GTiff', height=10, width=30, count=1,",
        "                   dtype='float32', crs='EPSG:32650',",
        "                   transform=from_origin(0, 1000, 100, 100), nodata=-9999.0) as ds:",
        "    ds.write(np.tile(np.arange(30, dtype='float32'), (10, 1)), 1)",
        "set_result({'raster': str(__RASTER__)})",
    ]
)

#: .gdb 要素类：zone_id LONG + name TEXT（三个方块）；.gdb 里只有它一个 → 是「单层容器」
_MAKE_ARCPY_ZONES = "\n".join(
    [
        "import arcpy, os",
        "arcpy.env.overwriteOutput = True",
        "sr = arcpy.SpatialReference(32650)",
        "gdb = __GDB__",
        "if not arcpy.Exists(gdb):",
        "    arcpy.management.CreateFileGDB(os.path.dirname(gdb), os.path.basename(gdb))",
        "fc = os.path.join(gdb, 'zones')",
        "if arcpy.Exists(fc):",
        "    arcpy.management.Delete(fc)",
        "arcpy.management.CreateFeatureclass(gdb, 'zones', 'POLYGON', spatial_reference=sr)",
        "arcpy.management.AddField(fc, 'zone_id', 'LONG')",
        "arcpy.management.AddField(fc, 'name', 'TEXT', field_length=50)",
        "for zid, nm, x in __SPECS__:",
        "    pts = [(x, 0), (x + 1000, 0), (x + 1000, 1000), (x, 1000)]",
        "    with arcpy.da.InsertCursor(fc, ['SHAPE@', 'zone_id', 'name']) as cur:",
        "        cur.insertRow([arcpy.Polygon(arcpy.Array([arcpy.Point(*p) for p in pts]), sr), zid, nm])",
        "set_result({'gdb': gdb, 'fc': fc})",
    ]
)

#: 往同一个 .gdb 再加一层：容器变成「多层」，配方必须报人话错误而不是自己挑一层
_ADD_SECOND_LAYER = "\n".join(
    [
        "import arcpy, os",
        "gdb = __GDB__",
        "fc = os.path.join(gdb, 'towns')",
        "if arcpy.Exists(fc):",
        "    arcpy.management.Delete(fc)",
        "arcpy.management.CreateFeatureclass(gdb, 'towns', 'POLYGON',",
        "    spatial_reference=arcpy.SpatialReference(32650))",
        "_previous = arcpy.env.workspace",
        "arcpy.env.workspace = gdb",
        "try:",
        "    _layers = list(arcpy.ListFeatureClasses() or [])",
        "finally:",
        "    arcpy.env.workspace = _previous",
        "set_result({'layers': _layers})",
    ]
)

ZONE_SPECS = [(11, "Z1", 0), (22, "Z2", 1000), (33, "Z3", 2000)]


# --------------------------------------------------------------------- 夹具
@pytest.fixture()
def gdal_runner(tmp_path: Path) -> GdalCodeRunner:
    interpreter = default_python_executable()
    if interpreter is None or not Path(interpreter).is_file():
        pytest.skip("未找到 .venv-gdal 解释器（见 docs/p0-1b-env.md），跳过真 GDAL 用例")
    return GdalCodeRunner(workspace=tmp_path, python_executable=interpreter)


@pytest.fixture()
def arcpy_runner(tmp_path: Path):
    try:
        info = discover_arcgis_pro_python()
    except ArcGISDiscoveryError as exc:
        pytest.skip(f"本机没有 ArcGIS Pro（{exc}），跳过容器用例")
    runner = CodeRunner(
        workspace=tmp_path, python_executable=info.python_executable, exec_timeout=600
    )
    try:
        yield runner
    finally:
        runner.shutdown()


#: 输出另开一个 .gdb：写回 zone_inputs.gdb 会多出一层，把「单层容器」变成多层
_MAKE_OUTPUT_GDB = "\n".join(
    [
        "import arcpy, os",
        "gdb = __GDB__",
        "if not arcpy.Exists(gdb):",
        "    arcpy.management.CreateFileGDB(os.path.dirname(gdb), os.path.basename(gdb))",
        "set_result({'gdb': gdb})",
    ]
)


def _make_zones_gdb(arcpy_runner, tmp_path: Path) -> str:
    gdb = tmp_path / "zone_inputs.gdb"
    made = arcpy_runner.run(
        _MAKE_ARCPY_ZONES.replace("__GDB__", repr(str(gdb)))
        .replace("__SPECS__", repr(ZONE_SPECS)),
        timeout=300,
    )
    assert made.ok, made.error
    return made.result["gdb"]


def _add_second_layer(arcpy_runner, gdb: str) -> None:
    made = arcpy_runner.run(_ADD_SECOND_LAYER.replace("__GDB__", repr(gdb)), timeout=300)
    assert made.ok, made.error
    assert "towns" in made.result["layers"], made.result


def _run(recipe_id: str, runner, tmp_path: Path, **params):
    lib = RecipeLibrary(code_runner=runner)
    lib.load()
    ctx = SimpleNamespace(workspace=tmp_path, code_runner=runner, exec_timeout=600)
    return lib.run(recipe_id, params, ctx)


#: 每个配方的参数：分区输入一律给容器（.gdb），其余参数按配方要求补齐
_PARAMS = {
    "aggregate_to_zones": lambda gdb, tag, raster: {
        "zones_path": gdb,
        "source_path": os.path.join(gdb, "zones"),
        "output_path": os.path.join(str(Path(gdb).parent / "outputs.gdb"), tag),
        "stat_fields": "zone_id",
        "stat_type": "SUM",
    },
    "zonal_statistics_raster": lambda gdb, tag, raster: {
        "zone_path": gdb,
        "zone_field": "",
        "raster_path": raster,
        "output_path": str(Path(raster).parent / "out" / f"{tag}.tif"),
        "statistics": "SUM",
    },
    "zone_raster_sum": lambda gdb, tag, raster: {
        "zone_path": gdb,
        "zone_field": "zone_id",
        "raster_path": raster,
        "output_table": os.path.join(gdb, tag),
        "nodata_as_zero": True,
    },
}


@pytest.mark.parametrize("recipe_id", sorted(_PARAMS))
def test_zone_container_is_resolved_or_rejected_with_a_human_error(
    gdal_runner, arcpy_runner, tmp_path: Path, recipe_id: str
):
    """单层容器 → 自动取用并跑通；多层容器 → 人话错误（列图层名），不是 AttributeError。"""
    raster = tmp_path / "dem.tif"
    made = gdal_runner.run(_MAKE_INPUTS.replace("__RASTER__", repr(str(raster))), timeout=180)
    assert made.ok, made.error
    (tmp_path / "out").mkdir(exist_ok=True)
    gdb = _make_zones_gdb(arcpy_runner, tmp_path)
    out_gdb = tmp_path / "outputs.gdb"
    made = arcpy_runner.run(_MAKE_OUTPUT_GDB.replace("__GDB__", repr(str(out_gdb))), timeout=300)
    assert made.ok, made.error

    single = _run(recipe_id, arcpy_runner, tmp_path, **_PARAMS[recipe_id](gdb, "single", str(raster)))
    assert single.ok, single.summary
    assert "自动使用其中唯一的图层: zones" in (single.data.get("stdout") or ""), (
        single.data.get("stdout")
    )

    _add_second_layer(arcpy_runner, gdb)
    multi = _run(recipe_id, arcpy_runner, tmp_path, **_PARAMS[recipe_id](gdb, "multi", str(raster)))

    assert not multi.ok, multi.summary
    assert "多层容器" in multi.summary, multi.summary
    assert "zones" in multi.summary and "towns" in multi.summary, multi.summary
    assert "图层名" in multi.summary, multi.summary
    # 关键：不能还是原来那条裸 AttributeError
    assert "AttributeError" not in multi.summary and "DescribeData" not in multi.summary, multi.summary

# -*- coding: utf-8 -*-
"""``zone_raster_sum`` 跑完还原持久内核的 ``arcpy.env``（DAIR-138）。

DAIR-136 审查确认（``c5ce3ec3`` 之后）：47 个内置配方里只剩本配方仍然

    arcpy.env.snapRaster = raster_path
    arcpy.env.extent = raster_path
    arcpy.env.outputCoordinateSystem = raster_sr

且全文没有 ``finally`` 还原。持久内核是**跨配方共享**的，这三项留在内核里会：

- 同内核后续任何用到 ``arcpy.env.extent`` / ``snapRaster`` 的任务被带偏
  （``DAIR-134`` 的 R1 就是同一机制：参考栅格是另一套 CRS 时，
  随后的 ``zonal_statistics`` 直接报 ``ERROR 010568 范围无效``）；
- ``outputCoordinateSystem`` 更隐蔽：下一次落盘可能被静默重投影。

三条用例与 ``tests/test_raster_calculator_gdal.py`` 里 ``DAIR-135`` 的 3 条同一口径：

1. 不需要 Pro：渲染形状是「设置前存旧值 → ``finally`` 写回」，把 ``finally`` 换成
   ``pass`` 必须失败；
2. 真 arcpy：跑完 env 与跑前逐字段相同；
3. 真 arcpy：跑前 env 非 None 时还原成**那个值**（不是退化成清空）。

缺环境就 skip（与 ``test_zonal_statistics_gdal.py`` 同一套口径）：ArcGIS Pro 找不到 →
arcpy 用例全 skip；造输入用的 ``.venv-gdal`` 不在 → 同样 skip。
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gis_cli.arcpy_bridge import ArcGISDiscoveryError, discover_arcgis_pro_python
from gis_cli.recipes.schema import Recipe
from gis_cli.recipes.store import RecipeLibrary
from gis_cli.runner.gdal_runner import GdalCodeRunner, default_python_executable
from gis_cli.runtime.code_runner import CodeRunner

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "src" / "gis_cli" / "recipes" / "builtin" / "zone_raster_sum.yaml"

#: 100 m 像元、30 列 × 10 行的数值栅格：像元值 = 列号（与 ``test_recipe_zone_container`` 同一构造）
_MAKE_RASTER = "\n".join(
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

#: 跑前的"污染" env：与本次运行**不同**的 snapRaster / extent（同 CRS：这里只验还原，
#: 不把"跑前 CS 属于别的投影带"当成前置条件——那时第 1 步 CopyFeatures 会被重投影，
#: 属于另一类问题），范围完整包住三个分区。
_MAKE_PRESET_RASTER = "\n".join(
    [
        "import numpy as np",
        "import rasterio",
        "from rasterio.transform import from_origin",
        "with rasterio.open(__RASTER__, 'w', driver='GTiff', height=20, width=40, count=1,",
        "                   dtype='float32', crs='EPSG:32650',",
        "                   transform=from_origin(0, 2000, 100, 100)) as ds:",
        "    ds.write(np.ones((20, 40), dtype='float32'), 1)",
        "set_result({'raster': str(__RASTER__)})",
    ]
)

#: .gdb 要素类：zone_id LONG + name TEXT，三个 1 km 方块（各覆盖 10 列 / 100 个像元）
_MAKE_ZONES = "\n".join(
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

_MAKE_OUTPUT_GDB = "\n".join(
    [
        "import arcpy, os",
        "gdb = __GDB__",
        "if not arcpy.Exists(gdb):",
        "    arcpy.management.CreateFileGDB(os.path.dirname(gdb), os.path.basename(gdb))",
        "set_result({'gdb': gdb})",
    ]
)

ZONE_SPECS = [(11, "Z1", 0), (22, "Z2", 1000), (33, "Z3", 2000)]

#: 读持久内核里的三项 env（只回可比较的纯数据）
_READ_ARCPY_ENV = "\n".join(
    [
        "import arcpy, os",
        "_ext = arcpy.env.extent",
        "_snap = arcpy.env.snapRaster",
        "_cs = arcpy.env.outputCoordinateSystem",
        "set_result({",
        "    'extent': None if _ext is None else [float(_ext.XMin), float(_ext.YMin),",
        "                                                float(_ext.XMax), float(_ext.YMax)],",
        "    'snapRaster': None if _snap is None else os.path.normcase(os.path.abspath(str(_snap))),",
        "    'outputCoordinateSystem': None if _cs is None else int(_cs.factoryCode),",
        "})",
    ]
)

#: 把内核里的 env 先设成一套**非 None** 的值（pre-set 用例的前置条件）
_PRESET_ARCPY_ENV = "\n".join(
    [
        "import arcpy",
        "arcpy.env.snapRaster = __RASTER__",
        "arcpy.env.extent = arcpy.Describe(__RASTER__).extent",
        "arcpy.env.outputCoordinateSystem = arcpy.Describe(__RASTER__).spatialReference",
        "set_result({'ok': True})",
    ]
)

_RESET_ARCPY_ENV = "\n".join(
    [
        "import arcpy",
        "arcpy.env.snapRaster = None",
        "arcpy.env.extent = None",
        "arcpy.env.outputCoordinateSystem = None",
        "set_result({'ok': True})",
    ]
)


# --------------------------------------------------------------------- 夹具
@pytest.fixture()
def gdal_runner(tmp_path: Path) -> GdalCodeRunner:
    interpreter = default_python_executable()
    if interpreter is None or not Path(interpreter).is_file():
        pytest.skip("未找到 .venv-gdal 解释器（见 docs/p0-1b-env.md），跳过真 GDAL 用例")
    return GdalCodeRunner(workspace=tmp_path, python_executable=interpreter)


@pytest.fixture(scope="module")
def arcpy_runner(tmp_path_factory):
    """arcpy 内核模块级共享：两条真跑用例共用一次内核启动（单次启动实测 ~14s）。"""
    try:
        info = discover_arcgis_pro_python()
    except ArcGISDiscoveryError as exc:
        pytest.skip(f"本机没有 ArcGIS Pro（{exc}），跳过 env 还原用例")
    runner = CodeRunner(
        workspace=tmp_path_factory.mktemp("arcpy_kernel"),
        python_executable=info.python_executable,
        exec_timeout=600,
    )
    try:
        yield runner
    finally:
        runner.shutdown()


# --------------------------------------------------------------------- 工具
def _make_raster(gdal_runner, tmp_path: Path) -> str:
    raster = tmp_path / "value.tif"
    made = gdal_runner.run(_MAKE_RASTER.replace("__RASTER__", repr(str(raster))), timeout=180)
    assert made.ok, made.error
    return made.result["raster"]


def _make_preset_raster(gdal_runner, tmp_path: Path) -> str:
    raster = tmp_path / "preset.tif"
    made = gdal_runner.run(
        _MAKE_PRESET_RASTER.replace("__RASTER__", repr(str(raster))), timeout=180
    )
    assert made.ok, made.error
    return made.result["raster"]


def _run_recipe(arcpy_runner, tmp_path: Path, raster: str):
    """按真实配方链路跑一遍（含 ``{{@zone_layer@}}`` 容器解析与 validation）。"""
    gdb = tmp_path / "zone_inputs.gdb"
    made = arcpy_runner.run(
        _MAKE_ZONES.replace("__GDB__", repr(str(gdb))).replace("__SPECS__", repr(ZONE_SPECS)),
        timeout=300,
    )
    assert made.ok, made.error
    out_gdb = tmp_path / "outputs.gdb"
    made = arcpy_runner.run(_MAKE_OUTPUT_GDB.replace("__GDB__", repr(str(out_gdb))), timeout=300)
    assert made.ok, made.error

    lib = RecipeLibrary(code_runner=arcpy_runner)
    lib.load()
    ctx = SimpleNamespace(workspace=tmp_path, code_runner=arcpy_runner, exec_timeout=600)
    return lib.run(
        "zone_raster_sum",
        {
            "zone_path": str(gdb),
            "zone_field": "zone_id",
            "raster_path": raster,
            "output_table": os.path.join(str(out_gdb), "zone_sum"),
            "nodata_as_zero": True,
        },
        ctx,
    )


def _env(runner) -> dict:
    outcome = runner.run(_READ_ARCPY_ENV, timeout=120)
    assert outcome.ok, outcome.error
    return outcome.result


# ------------------------------------------- 渲染形状（不需要 Pro）
def test_arcpy_template_restores_the_persistent_env():
    """先存旧值、``finally`` 写回：把 ``finally`` 里的三行换成 ``pass`` 这条必须失败。"""
    recipe = Recipe.from_yaml(RECIPE)
    code = recipe.render({"zone_path": "z.gdb/zones", "zone_field": "zone_id",
                          "raster_path": "r.tif", "output_table": "o.gdb/t",
                          "nodata_as_zero": True})
    ast.parse(code)  # 套一层 try/finally 不能把缩进搞坏

    assert "_previous_snap_raster = arcpy.env.snapRaster" in code
    assert "_previous_extent = arcpy.env.extent" in code
    assert "_previous_output_cs = arcpy.env.outputCoordinateSystem" in code

    body = code[code.index("_previous_snap_raster = arcpy.env.snapRaster"):]
    # 先存后改：三处赋值必须在 try 里的改写之前
    for assign in ("arcpy.env.snapRaster = raster_path",
                   "arcpy.env.extent = raster_path",
                   "arcpy.env.outputCoordinateSystem = raster_sr"):
        assert body.index(assign) < body.index("finally:"), assign
    tail = body[body.index("finally:"):]
    assert "arcpy.env.snapRaster = _previous_snap_raster" in tail
    assert "arcpy.env.extent = _previous_extent" in tail
    assert "arcpy.env.outputCoordinateSystem = _previous_output_cs" in tail


# ------------------------------------------- 真 arcpy：env 不泄漏
def test_arcpy_zone_sum_restores_the_env_it_found(gdal_runner, arcpy_runner, tmp_path: Path):
    """跑完 ``snapRaster`` / ``extent`` / ``outputCoordinateSystem`` 回到跑前值。"""
    raster = _make_raster(gdal_runner, tmp_path)
    before = _env(arcpy_runner)
    try:
        obs = _run_recipe(arcpy_runner, tmp_path, raster)
        assert obs.ok, obs.summary
        result = obs.data["result"]
        # 行为没变：三个分区各 100 个像元，列号和 450 + 1450 + 2450
        assert result["zone_count"] == 3, result
        assert result["cell_size"] == 100.0, result
        assert result["total_value"] == 4350.0, result
        assert _env(arcpy_runner) == before, (
            "跑完没还原 env，同内核后续任务会被 snapRaster/extent/outputCoordinateSystem 污染"
        )
    finally:
        arcpy_runner.run(_RESET_ARCPY_ENV, timeout=120)


def test_arcpy_zone_sum_restores_a_preset_env(gdal_runner, arcpy_runner, tmp_path: Path):
    """跑前 env 已有值（与本次运行不同）时也要还原成**那个值**，不是一律清空。

    前置值是 ``preset.tif`` 那一套（同 CRS、范围 [0,0,4000,2000]，与本次的
    ``value.tif`` [0,0,3000,1000] 不同）。若只把 env 清空、或留在配方设的值上，
    这条会在 snapRaster / extent 两项上失败。
    """
    raster = _make_raster(gdal_runner, tmp_path)
    preset_raster = _make_preset_raster(gdal_runner, tmp_path)
    preset = arcpy_runner.run(
        _PRESET_ARCPY_ENV.replace("__RASTER__", repr(preset_raster)), timeout=120
    )
    assert preset.ok, preset.error
    before = _env(arcpy_runner)
    assert before["snapRaster"] == os.path.normcase(os.path.abspath(preset_raster)), before
    assert before["extent"] == [0.0, 0.0, 4000.0, 2000.0], before
    assert before["outputCoordinateSystem"] == 32650, before
    try:
        obs = _run_recipe(arcpy_runner, tmp_path, raster)
        assert obs.ok, obs.summary
        assert obs.data["result"]["total_value"] == 4350.0, obs.data["result"]
        after = _env(arcpy_runner)
        assert after == before, "有前置 env 时没还原成跑前值（或退化成清空/留在本次运行的值上）"
    finally:
        arcpy_runner.run(_RESET_ARCPY_ENV, timeout=120)

# -*- coding: utf-8 -*-
"""raster_calculator 的 GDAL 版真跑 + **逐像元**双端一致（同分辨率；混分辨率必须显式
给 resampling，给了之后两端先重采样到同一网格、仍逐像元一致 —— 见文末 DAIR-139 用例）。

测试数据（EPSG:32650，米，5 m 像元，10×10）：

    a   行值 0..9（第 0 行 0、第 9 行 9），锚在 (0,0)
    b   常量 2，锚在 (0,0)
    c   常量 100，锚在 (0,2.5) —— **半像元错位**
    d   常量 1000，锚在 (0,5.0) —— 另一个错位
    e   12 行（比 a 高 10 米），常量 7
    g   8 行（比 a 矮 10 米），行值 0..7
    h   10 行，锚在 (0,-2.5) —— 同尺寸、上边界更低
    p   12 列（比 a 宽 10 米），行值 0..9
    tall 10 行 × 8 列 = 80 px，原点 (0,50)  —— 与 wide 同像元数、同原点、不同形状
    wide  8 行 × 10 列 = 80 px，原点 (0,50)
    m   行值 0..9，**第 0 列与 (5,5) 是 NoData**
    i   int16 行值 0..9（问 dtype 口径）

``REFERENCE_*`` 那几组数是**真 arcpy 跑出来的**（ArcGIS Pro 3.6，证据在
``benchmark/reports/dair110_raster_calc_20261030/``：probe_arcpy.json /
probe_grid_rule.json / compare_both.json / verify_chain.json），不是从 GDAL 侧抄回来的。
它们同时钉住 arcpy 的四条实测口径：

1. **输出范围逐结点传播**（不是按全局算一遍）：每个子表达式都有自己的窗口，两幅合并
   时逐轴取交集，再落到「交集正好等于某一侧整幅 → 用那一侧；否则用原点更大那一侧」
   的格子上，窗口取该格子里**格心落在交集内**的像元（dair116 探针：`a+d` → 9 行
   @(0,50)、`a+narrow` → 6 列、`a+c+d` → 10 行 @(0,55) 而 `d+a+c` → 9 行 @(0,50)，
   arcpy 自己就不对称）；真实链 3c 输出 2220×2542@elder 也由此而来；
2. 其余输入被**最近邻**重采样到该网格（半像元错位时取格心所在像元）；
3. NoData 传播：比较/四则遇 NoData → NoData，**Con 的条件是 NoData → NoData**，
   `SetNull` 条件为真或 NoData → NoData，除零（0/0 与 x/0）→ NoData，`IsNull` 从不产 NoData；
4. 输出 nodata 恒为 float32 最小值 -3.4028234663852886e+38（dtype 两端不同：
   arcpy 按值域挑类型，GDAL 恒 float32 —— 只比网格、掩膜与数值）。

缺环境就 skip（与 ``test_zonal_statistics_raster_gdal.py`` 同一套口径）：
``.venv-gdal`` 不在 → 所有真 GDAL 用例 skip；ArcGIS Pro 找不到 → 只有双端用例 skip。
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gis_cli.recipes.schema import Recipe
from gis_cli.recipes.store import RecipeLibrary
from gis_cli.runner.gdal_runner import GdalCodeRunner, default_python_executable
from gis_cli.runtime.code_runner import CodeRunner

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "src/gis_cli/recipes/builtin/raster_calculator.yaml"

#: 输出 nodata：GDAL 版与 zonal_statistics_raster 同一哨兵（arcpy 实测也是它）
NODATA_OUT = -3.4028234663852886e38

#: 分摊那一步（3c）的表达式：模板里用的就是这条
ALLOC_EXPR = "Con(IsNull(zone_sum), 0, Con(zone_sum == 0, 0, density / zone_sum * elder))"

_MAKE_SCENE = "\n".join(
    [
        "import os",
        "import numpy as np",
        "import rasterio",
        "from rasterio.transform import from_origin",
        "_base = __DIR__",
        "rows = np.arange(10, dtype='float32').reshape(10, 1)",
        "out = {}",
        "",
        "def write(name, arr, x0, y0, dtype='float32', nodata=-9999.0):",
        "    path = os.path.join(_base, name + '.tif')",
        "    arr = np.asarray(arr, dtype=dtype)",
        "    tr = from_origin(x0, y0 + arr.shape[0] * 5.0, 5.0, 5.0)",
        "    with rasterio.open(path, 'w', driver='GTiff', height=arr.shape[0],",
        "                       width=arr.shape[1], count=1, dtype=dtype, crs='EPSG:32650',",
        "                       transform=tr, nodata=nodata) as ds:",
        "        ds.write(arr, 1)",
        "    out[name] = path",
        "",
        "write('a', np.repeat(rows, 10, axis=1), 0.0, 0.0)",
        "write('b', np.full((10, 10), 2.0), 0.0, 0.0)",
        "write('c', np.full((10, 10), 100.0), 0.0, 2.5)",
        "write('d', np.full((10, 10), 1000.0), 0.0, 5.0)",
        "write('e', np.full((12, 10), 7.0), 0.0, 0.0)",
        "write('g', np.repeat(rows[:8], 10, axis=1), 0.0, 0.0)",
        "write('h', np.repeat(rows, 10, axis=1), 0.0, -2.5)",
        "write('p', np.repeat(rows, 12, axis=1), 0.0, 0.0)",
        "write('tall', np.repeat(rows, 8, axis=1), 0.0, 0.0)",
        "write('wide', np.repeat(rows[:8], 10, axis=1), 0.0, 10.0)",
        # 20 行 x 6 列（120 px），x 只到 30 / x 在 2.5..32.5：钉「交集收窄」与「半格边界」
        "narrow_rows = np.arange(20, dtype='float32').reshape(20, 1)",
        "write('narrow', np.repeat(narrow_rows, 6, axis=1), 0.0, 0.0)",
        "write('narrow2', np.repeat(narrow_rows, 6, axis=1), 2.5, 0.0)",
        "m = np.repeat(rows, 10, axis=1).copy()",
        "m[:, 0] = -9999.0",
        "m[5, 5] = -9999.0",
        "write('m', m, 0.0, 0.0)",
        "write('i', np.repeat(np.arange(10, dtype='int16').reshape(10, 1), 10, axis=1),",
        "      0.0, 0.0, dtype='int16', nodata=-32768)",
        "set_result(out)",
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

#: arcpy（Pro 3.6）实测的网格选择：用例 → (shape, origin_y, 第 0 列像元值)
GRID_CASES = [
    ("两路对齐", "a={a},b={b}", "a + b", [10, 10], 50.0, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]),
    ("半像元错位跟错位那幅", "a={a},c={c}", "a + c", [10, 10], 52.5, [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]),
    ("错位反序同结果", "c={c},a={a}", "c + a", [10, 10], 52.5, [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]),
    ("三路错位在末尾", "a={a},b={b},c={c}", "a + b + c", [10, 10], 52.5, [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0]),
    ("三路三个原点取最高", "a={a},c={c},d={d}", "a + c + d", [10, 10], 55.0, [np.nan, 1100.0, 1101.0, 1102.0, 1103.0, 1104.0, 1105.0, 1106.0, 1107.0, 1108.0]),
    ("尺寸不同取像元少的", "a={a},e={e}", "a + e", [10, 10], 50.0, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
    ("更小那幅上边界低也归它", "a={a},g={g}", "a + g", [8, 10], 40.0, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
    ("同尺寸上边界更低的不选", "a={a},h={h}", "a + h", [10, 10], 50.0, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]),
    ("同高更宽的不选", "a={a},p={p}", "a + p", [10, 10], 50.0, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]),
    # 同像元数（80）+ 同原点（0,50）但形状不同：arcpy 实测取交集 8×8，与书写顺序无关
    ("同像元数同原点 tall 先", "tall={tall},wide={wide}", "tall + wide", [8, 8], 50.0, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]),
    ("同像元数同原点 wide 先", "wide={wide},tall={tall}", "wide + tall", [8, 8], 50.0, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]),
]

_READ_ARCPY_ENV = "\n".join(
    [
        "import arcpy",
        "_ext = arcpy.env.extent",
        "set_result({'extent': None if _ext is None else [float(_ext.XMin), float(_ext.YMin),",
        "                                         float(_ext.XMax), float(_ext.YMax)],",
        "            'cellSize': str(arcpy.env.cellSize)})",
    ]
)

#: 把内核里的 env 先设成一套**非 None** 的值（pre-set 用例的前置条件）
_PRESET_ARCPY_ENV = "\n".join(
    [
        "import arcpy",
        "arcpy.env.extent = arcpy.Describe(__REF__).extent",
        "arcpy.env.cellSize = 7",
        "set_result({'ok': True})",
    ]
)

_RESET_ARCPY_ENV = "\n".join(
    [
        "import arcpy",
        "arcpy.env.extent = None",
        "arcpy.env.cellSize = None",
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
    """arcpy 内核模块级共享：13 个双端用例共用一次内核启动（单次启动实测 ~14s）。"""
    from gis_cli.arcpy_bridge import ArcGISDiscoveryError, discover_arcgis_pro_python

    try:
        info = discover_arcgis_pro_python()
    except ArcGISDiscoveryError as exc:
        pytest.skip(f"本机没有 ArcGIS Pro（{exc}），跳过双端一致用例")
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
def _scene(runner, tmp_path: Path) -> dict[str, str]:
    made = runner.run(_MAKE_SCENE.replace("__DIR__", repr(str(tmp_path))), timeout=300)
    assert made.ok, made.error
    return made.result


def _run(runner, tmp_path: Path, inputs: str, expression: str, name: str, **params):
    lib = RecipeLibrary(code_runner=runner)
    lib.load()
    ctx = SimpleNamespace(workspace=tmp_path, code_runner=runner, exec_timeout=600)
    request = {"inputs": inputs, "expression": expression,
               "output_path": str(tmp_path / name)}
    request.update(params)
    return lib.run("raster_calculator", request, ctx)


def _result(obs) -> dict:
    assert obs.ok, obs.summary
    return obs.data["result"]


def _raster(runner, path: str) -> dict:
    outcome = runner.run(_READ_RASTER.replace("__PATH__", repr(str(path))), timeout=300)
    assert outcome.ok, outcome.error
    grid = outcome.result
    grid["array"] = np.array(grid["values"], dtype="float64")
    return grid


def _assert_same_grid(gdal_grid: dict, arcpy_grid: dict) -> None:
    assert gdal_grid["shape"] == arcpy_grid["shape"]
    np.testing.assert_allclose(gdal_grid["cell"], arcpy_grid["cell"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(gdal_grid["origin"], arcpy_grid["origin"], rtol=0, atol=1e-6)


def _assert_same_pixels(gdal_grid: dict, arcpy_grid: dict, *, rtol: float = 1e-6) -> None:
    """逐像元：nodata 掩膜完全一致，有效值在 float32 容差内相等（dtype 按设计不同）。"""
    gdal_mask = np.isnan(gdal_grid["array"])
    arcpy_mask = np.isnan(arcpy_grid["array"])
    assert np.array_equal(gdal_mask, arcpy_mask), (
        f"nodata 掩膜不同：gdal {int(gdal_mask.sum())} 个 nodata，"
        f"arcpy {int(arcpy_mask.sum())} 个"
    )
    both = ~gdal_mask
    np.testing.assert_allclose(
        gdal_grid["array"][both], arcpy_grid["array"][both], rtol=rtol, atol=1e-6
    )


# ------------------------------------------------------------- 模板本身
def test_recipe_declares_two_backends_and_renders_both():
    recipe = Recipe.from_yaml(RECIPE)
    assert recipe.backend == ["arcpy", "gdal"]
    params = {
        "inputs": "density=d.tif,zone_sum=z.tif,elder=e.tif",
        "expression": ALLOC_EXPR,
        "output_path": "out/alloc.tif",
        "cell_size": "",
        "extent_ref": "",
    }
    arcpy_code = recipe.render(params)
    gdal_code = recipe.render(params, backend="gdal")

    assert "arcpy.sa" in arcpy_code and "rasterio" not in arcpy_code
    assert "from rasterio.warp import Resampling, reproject" in gdal_code
    assert "import arcpy" not in gdal_code
    for key in ("density=d.tif,zone_sum=z.tif,elder=e.tif", "out/alloc.tif", "zone_sum == 0"):
        assert key in arcpy_code and key in gdal_code
    # GDAL 段不用 eval：不认识的函数/语法要报「不支持什么」，而不是静默当常量
    assert "eval(expression" in arcpy_code and "ast.parse" in gdal_code
    pitfalls = "".join(recipe.pitfalls)
    assert "逐结点范围传播" in pitfalls
    assert "float32 + nodata=-3.4028235e+38" in pitfalls
    assert "明确报错" in pitfalls


def test_arcpy_template_restores_the_persistent_env():
    """`extent_ref` / `cell_size` 只在本配方内部生效：先存旧值，`finally` 里写回。

    实测（DAIR-135，Pro 3.6）：`extent_ref=dem` 跑完后 DEM（UTM 10N）的范围留在持久
    内核的 `arcpy.env.extent` 里（`cellSize=30` 同样留着），同内核随后的
    `zonal_statistics`（EPSG:4508）直接 ERROR 010568 范围无效。这条用例不需要 Pro，
    用渲染结果锁住「改之前存、finally 里写回」的形状，把 finally 删掉就会失败。
    """
    recipe = Recipe.from_yaml(RECIPE)
    code = recipe.render({"inputs": "a=a.tif,b=b.tif", "expression": "a + b",
                          "output_path": "o.tif", "cell_size": "30", "extent_ref": "b"})
    ast.parse(code)  # 套一层 try/finally 不能把缩进搞坏
    assert "_previous_extent = arcpy.env.extent" in code
    assert "_previous_cell_size = arcpy.env.cellSize" in code
    tail = code[code.index("finally:"):]
    assert "arcpy.env.extent = _previous_extent" in tail
    assert "arcpy.env.cellSize = _previous_cell_size" in tail


def test_arcpy_mixed_branch_reads_aligned_inputs_without_resampling():
    """DAIR-144：混分辨率分支对「已在输出网格上」的输入直读，与 GDAL 版 ``_load`` 的 same 对称。

    arcpy 侧曾经对所有 ``used`` 输入无条件 ``_resample``：``extent_ref`` 指向的正是参与
    运算的那幅时，它本身就在输出网格上，bilinear 取不到右下邻格 → 末端最后一行/列与
    NoData 洞边缘被判 NoData（真实链 835/2,368,407 px = 0.0353%）。这条用例不需要 Pro，
    用渲染结果锁住短路形状，把 ``if _same`` 短路删掉就会失败。
    """
    recipe = Recipe.from_yaml(RECIPE)
    code = recipe.render({"inputs": "lc=lc.tif,dem=dem.tif",
                          "expression": "Con(lc == 2, dem, 0)", "output_path": "o.tif",
                          "cell_size": "30", "extent_ref": "dem", "resampling": "bilinear"})
    ast.parse(code)  # 多一段 if/else 不能把缩进搞坏
    assert "_src = _read_float(paths[name])" in code
    assert "_arr = _src if _same else _resample(" in code


def test_gdal_template_pins_the_arcpy_grid_and_nodata_rules():
    params = {"inputs": "a=a.tif,b=b.tif", "expression": "a + b", "output_path": "o.tif",
              "cell_size": "", "extent_ref": ""}
    code = Recipe.from_yaml(RECIPE).render(params, backend="gdal")

    # 输出网格：逐结点传播（交集 → 整幅优先 / 原点大者优先 → 取格心落在交集内的像元）
    assert "def _new_grid(a, b):" in code
    assert "def _snap(phase, lo, hi, cell, vertical):" in code
    assert "def _merge(grids):" in code
    assert "def _grid_eval(node):" in code
    assert "格心落在交集内" in code
    assert "Resampling.nearest" in code
    # nodata 哨兵与 zonal_statistics_raster 同一口径
    assert "_NODATA_OUT = -3.4028234663852886e+38" in code
    assert "nodata=_NODATA_OUT" in code
    # 除零 → NoData（arcpy 实测），不是 inf
    assert "ok = ok & (right != 0)" in code


# ----------------------------------------------------------- GDAL 侧真跑
def test_gdal_con_isnull_allocates_like_arcpy(gdal_runner, tmp_path: Path):
    """3c 的分摊表达式：NoData 掩膜写 0、0/0 落 NoData、其余 = a/m*b。"""
    paths = _scene(gdal_runner, tmp_path)
    obs = _run(gdal_runner, tmp_path, f"a={paths['a']},m={paths['m']},b={paths['b']}",
               "Con(IsNull(m), 0, a / m * b)", "out/alloc.tif")
    result = _result(obs)
    assert result["size"] == [10, 10]
    assert result["cell_size"] == 5.0
    assert result["nodata_px"] == 9  # arcpy 实测 n_valid=91
    assert result["min"] == 0.0 and result["max"] == 2.0

    grid = _raster(gdal_runner, result["output"])
    assert grid["origin"] == [0.0, 50.0]
    assert grid["crs"] == "EPSG:32650"
    assert grid["dtype"] == "float32"
    assert grid["nodata"] == NODATA_OUT
    array = grid["array"]
    assert np.all(array[:, 0] == 0.0), "m 第 0 列是 NoData → IsNull 分支写 0"
    assert array[5, 5] == 0.0, "m(5,5) 是 NoData → 同样写 0"
    assert np.all(np.isnan(array[0, 1:])), "第 0 行 a=0、m=0 → 0/0 → NoData（arcpy 实测）"
    assert np.allclose(array[1:5, 1:], 2.0)
    assert np.allclose(array[6:, 1:], 2.0)


def test_gdal_division_by_zero_is_nodata(gdal_runner, tmp_path: Path):
    paths = _scene(gdal_runner, tmp_path)
    result = _result(_run(gdal_runner, tmp_path, f"a={paths['a']},b={paths['b']}",
                          "a / (b - 2)", "out/divzero.tif"))
    assert result["nodata_px"] == 100 and result["min"] is None and result["max"] is None
    grid = _raster(gdal_runner, result["output"])
    assert np.all(np.isnan(grid["array"]))


def test_gdal_con_with_nodata_condition_propagates(gdal_runner, tmp_path: Path):
    paths = _scene(gdal_runner, tmp_path)
    result = _result(_run(gdal_runner, tmp_path, f"m={paths['m']},a={paths['a']},b={paths['b']}",
                          "Con(m > 3, a, b)", "out/con.tif"))
    assert result["nodata_px"] == 11  # arcpy 实测 n_valid=89
    array = _raster(gdal_runner, result["output"])["array"]
    assert np.all(np.isnan(array[:, 0])), "条件为 NoData → NoData（arcpy 实测）"
    assert np.all(np.isnan(array[5, 5]))
    assert np.allclose(array[0, 1:], 2.0), "条件为假 → 取假值"
    assert np.allclose(array[6:, 1:], np.arange(6, 10).reshape(4, 1))


def test_gdal_set_null_and_is_null(gdal_runner, tmp_path: Path):
    paths = _scene(gdal_runner, tmp_path)
    setnull = _result(_run(gdal_runner, tmp_path, f"m={paths['m']}", "SetNull(m > 3, 1)",
                           "out/setnull.tif"))
    assert setnull["nodata_px"] == 64  # arcpy 实测 n_valid=36
    array = _raster(gdal_runner, setnull["output"])["array"]
    assert np.all(np.isnan(array[:, 0])) and np.all(np.isnan(array[5, 5]))
    assert np.allclose(array[0, 1:], 1.0), "条件为假 → 写值"
    assert np.all(np.isnan(array[4:, 1:])), "条件为真 → NoData"

    isnull = _result(_run(gdal_runner, tmp_path, f"m={paths['m']}", "IsNull(m)",
                          "out/isnull.tif"))
    assert isnull["nodata_px"] == 0 and isnull["min"] == 0.0 and isnull["max"] == 1.0
    array = _raster(gdal_runner, isnull["output"])["array"]
    assert np.all(array[:, 0] == 1.0) and array[5, 5] == 1.0
    assert np.all(array[0, 1:] == 0.0)


def test_gdal_negative_condition_is_true(gdal_runner, tmp_path: Path):
    """条件**非 0 为真**（负值也算真）：Con 与 SetNull 同一口径（arcpy 实测）。"""
    paths = _scene(gdal_runner, tmp_path)
    # a 行值 0..9 → a-5 在第 0..4 行是负值
    con = _result(_run(gdal_runner, tmp_path, f"a={paths['a']}", "Con(a - 5, 1, 0)",
                       "out/negcon.tif"))
    assert con["nodata_px"] == 0 and con["min"] == 0.0 and con["max"] == 1.0
    array = _raster(gdal_runner, con["output"])["array"]
    assert np.all(array[:5, :] == 1.0), "负值 → 真（arcpy 实测是 1，不是 0）"
    assert np.all(array[5, :] == 0.0), "条件为 0 → 假"
    assert np.all(array[6:, :] == 1.0), "正值 → 真"

    setnull = _result(_run(gdal_runner, tmp_path, f"a={paths['a']}", "SetNull(a - 5, 1)",
                           "out/negsetnull.tif"))
    assert setnull["nodata_px"] == 90  # 只有 a-5 == 0 那一行（第 5 行）留下
    array = _raster(gdal_runner, setnull["output"])["array"]
    assert np.all(np.isnan(array[:5, :])) and np.all(np.isnan(array[6:, :]))
    assert np.all(array[5, :] == 1.0)


def test_gdal_comparison_on_nodata_stays_nodata(gdal_runner, tmp_path: Path):
    paths = _scene(gdal_runner, tmp_path)
    result = _result(_run(gdal_runner, tmp_path, f"m={paths['m']}", "m > 3", "out/gt.tif"))
    assert result["nodata_px"] == 11  # arcpy 实测 n_valid=89
    array = _raster(gdal_runner, result["output"])["array"]
    assert np.all(np.isnan(array[:, 0]))
    assert np.all(array[0, 1:] == 0.0) and np.all(array[9, 1:] == 1.0)


@pytest.mark.parametrize("label,inputs,expression,shape,origin_y,col0", GRID_CASES,
                         ids=[c[0] for c in GRID_CASES])
def test_gdal_output_grid_follows_the_arcpy_rule(gdal_runner, tmp_path: Path, label, inputs,
                                                 expression, shape, origin_y, col0):
    """网格选择与最近邻重采样：每条期望值都是 arcpy 实测（见模块 docstring）。"""
    paths = _scene(gdal_runner, tmp_path)
    result = _result(_run(gdal_runner, tmp_path, inputs.format(**paths), expression,
                          "out/grid.tif"))
    assert result["size"] == [shape[1], shape[0]], label
    grid = _raster(gdal_runner, result["output"])
    assert grid["shape"] == shape, label
    assert grid["origin"] == [0.0, origin_y], label
    assert grid["cell"] == [5.0, 5.0]
    np.testing.assert_allclose(grid["array"][:, 0], col0, rtol=1e-6, atol=1e-6, err_msg=label)


#: 逐结点范围传播（DAIR-116）：arcpy Pro 3.6 实测的窗口、原点与第 0 列。
#: 每条都对应 issue 表格里的一行；`a+c+d` / `d+a+c` 必须分别钉（arcpy 自己就不对称）。
EXTENT_CASES = [
    ("两路错位取交集 a+d", "a={a},d={d}", "a + d", [9, 10], [0.0, 50.0], 90,
     [1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0, 1007.0, 1008.0]),
    ("交集收窄 a+narrow", "a={a},narrow={narrow}", "a + narrow", [10, 6], [0.0, 50.0], 60,
     [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]),
    ("半格边界 a+narrow2", "a={a},narrow2={narrow2}", "a + narrow2", [10, 6], [2.5, 50.0], 60,
     [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]),
    ("三路 a+c+d", "a={a},c={c},d={d}", "a + c + d", [10, 10], [0.0, 55.0], 90,
     [np.nan, 1100.0, 1101.0, 1102.0, 1103.0, 1104.0, 1105.0, 1106.0, 1107.0, 1108.0]),
    ("三路反序 d+a+c", "d={d},a={a},c={c}", "d + a + c", [9, 10], [0.0, 50.0], 90,
     [1100.0, 1101.0, 1102.0, 1103.0, 1104.0, 1105.0, 1106.0, 1107.0, 1108.0]),
]


@pytest.mark.parametrize("label,inputs,expression,shape,origin,n_valid,col0", EXTENT_CASES,
                         ids=[c[0] for c in EXTENT_CASES])
def test_gdal_output_extent_is_the_per_node_intersection(gdal_runner, tmp_path: Path, label,
                                                        inputs, expression, shape, origin,
                                                        n_valid, col0):
    """输出范围逐结点传播：窗口、原点、有效像元数与第 0 列都按 arcpy 实测钉住。"""
    paths = _scene(gdal_runner, tmp_path)
    result = _result(_run(gdal_runner, tmp_path, inputs.format(**paths), expression,
                          "out/extent.tif"))
    assert result["size"] == [shape[1], shape[0]], label
    grid = _raster(gdal_runner, result["output"])
    assert grid["shape"] == shape, label
    assert grid["origin"] == origin, label
    assert grid["cell"] == [5.0, 5.0]
    assert grid["n_valid"] == n_valid, label
    np.testing.assert_allclose(grid["array"][:, 0], col0, rtol=1e-6, atol=1e-6, err_msg=label)


def test_gdal_extent_ref_and_cell_size_override_the_grid(gdal_runner, tmp_path: Path):
    """extent_ref 指定参考栅格、cell_size 给数值/输入名时，输出网格跟着它们走。"""
    paths = _scene(gdal_runner, tmp_path)
    inputs = f"a={paths['a']},b={paths['b']},c={paths['c']}"

    default = _result(_run(gdal_runner, tmp_path, inputs, "a + b + c", "out/d.tif"))
    assert default["grid_ref"] == "c"  # 默认按「逐结点传播」选中 c 的格子

    pinned = _result(_run(gdal_runner, tmp_path, inputs, "a + b + c", "out/p.tif",
                          extent_ref="a"))
    assert pinned["grid_ref"] == "a"
    assert _raster(gdal_runner, pinned["output"])["origin"] == [0.0, 50.0]

    coarse = _result(_run(gdal_runner, tmp_path, f"a={paths['a']},b={paths['b']}", "a + b",
                          "out/coarse.tif", cell_size="10"))
    assert coarse["cell_size"] == 10.0 and coarse["size"] == [5, 5]
    grid = _raster(gdal_runner, coarse["output"])
    assert grid["cell"] == [10.0, 10.0] and grid["origin"] == [0.0, 50.0]

    by_name = _result(_run(gdal_runner, tmp_path, f"a={paths['a']},b={paths['b']}", "a + b",
                           "out/byname.tif", cell_size="b"))
    assert by_name["cell_size"] == 5.0


def test_gdal_extent_ref_path_with_other_crs_is_rejected(gdal_runner, tmp_path: Path):
    """extent_ref 直接给路径时也要做 CRS 校验：范围按 ref 的坐标取，却写输入的 CRS 就是错位。"""
    paths = _scene(gdal_runner, tmp_path)
    ref = tmp_path / "ref3857.tif"
    made = gdal_runner.run("\n".join([
        "import numpy as np, rasterio",
        "from rasterio.transform import from_origin",
        f"with rasterio.open({str(ref)!r}, 'w', driver='GTiff', height=10, width=10,",
        "                   count=1, dtype='float32', crs='EPSG:3857',",
        "                   transform=from_origin(100.0, 200.0, 5.0, 5.0), nodata=-9999.0) as ds:",
        "    ds.write(np.ones((10, 10), dtype='float32'), 1)",
        "set_result({})",
    ]), timeout=300)
    assert made.ok, made.error
    obs = _run(gdal_runner, tmp_path, f"a={paths['a']}", "a + 1", "out/x.tif",
               extent_ref=str(ref))
    assert not obs.ok, "不同 CRS 的 extent_ref 应当报错，不能静默按 ref 的数值建窗口"
    assert "坐标系不一致" in obs.summary


def test_gdal_rejects_gdb_raster_and_unknown_extension(gdal_runner, tmp_path: Path):
    paths = _scene(gdal_runner, tmp_path)
    gdb = _run(gdal_runner, tmp_path, f"a={paths['a']}", "a + 1", "out/result.gdb/calc")
    assert not gdb.ok
    assert "gdb" in gdb.summary and "arcpy" in gdb.summary

    bad = _run(gdal_runner, tmp_path, f"a={paths['a']}", "a + 1", "out/calc.shp")
    assert not bad.ok
    assert "只支持 .tif" in bad.summary


def test_gdal_rejects_unknown_function_and_unknown_name(gdal_runner, tmp_path: Path):
    paths = _scene(gdal_runner, tmp_path)
    unknown = _run(gdal_runner, tmp_path, f"a={paths['a']}", "Float(a) + 1", "out/f.tif")
    assert not unknown.ok
    message = unknown.summary
    assert "Float" in message and "Con" in message, message
    assert "arcpy" in message, "报错要说清该换 arcpy 后端"

    missing = _run(gdal_runner, tmp_path, f"a={paths['a']}", "a + nope", "out/n.tif")
    assert not missing.ok
    assert "nope" in missing.summary and "'a'" in missing.summary


def test_gdal_rejects_and_or_not_and_bitwise_invert(gdal_runner, tmp_path: Path):
    """`and` / `or` / `not` 两端都不支持（arcpy 报 truth value is ambiguous）；`~` 是逻辑非。"""
    paths = _scene(gdal_runner, tmp_path)
    spec = f"a={paths['a']},m={paths['m']}"
    for expression, hint in (
        ("Con((a > 2) and (a < 6), a, 0)", "& / |"),
        ("Con((a > 2) or (a < 6), a, 0)", "& / |"),
        ("Con(not (a > 2), a, 0)", "& / |"),
        ("~Con(a > 5, 1, 0)", "逻辑非"),
        ("a ** 2", "不支持"),
    ):
        obs = _run(gdal_runner, tmp_path, spec, expression, "out/op.tif")
        assert not obs.ok, expression
        assert hint in obs.summary, (expression, obs.summary)

    # & / | 可用，且 NoData 照常传播（arcpy 实测口径）
    ok = _result(_run(gdal_runner, tmp_path, spec, "Con((m > 2) & (m < 6), a, 0)", "out/and.tif"))
    array = _raster(gdal_runner, ok["output"])["array"]
    assert np.all(np.isnan(array[:, 0]))  # m 第 0 列 NoData → 条件 NoData → NoData
    assert array[3, 1] == 3.0 and array[1, 1] == 0.0


# --------------------------------------------------------- 双端逐像元一致
DUAL_CASES = [
    ("分摊", "a={a},m={m},b={b}", "Con(IsNull(m), 0, a / m * b)"),
    ("除零", "a={a},b={b}", "a / (b - 2)"),
    ("Con 条件 nodata", "m={m},a={a},b={b}", "Con(m > 3, a, b)"),
    ("Con 两参数", "m={m},a={a}", "Con(m > 5, a)"),
    ("SetNull", "m={m}", "SetNull(m > 3, 1)"),
    ("IsNull", "m={m}", "IsNull(m)"),
    ("比较遇 nodata", "m={m}", "m > 3"),
    ("半像元错位", "a={a},c={c}", "a + c"),
    ("三路三个原点", "a={a},c={c},d={d}", "a + c + d"),
    ("像元少的说了算", "a={a},g={g}", "a + g"),
    ("布尔与", "m={m},a={a}", "Con((m > 2) & (m < 6), a, 0)"),
    ("整型加常量", "i={i}", "i + 1"),
    ("一元负号", "a={a}", "-a + 1"),
    ("负值条件 Con", "a={a}", "Con(a - 5, 1, 0)"),
    ("负值条件 SetNull", "a={a}", "SetNull(a - 5, 1)"),
    ("同像元数同原点", "tall={tall},wide={wide}", "tall + wide"),
    # 逐结点范围传播：两路错位取交集、交集被更窄的输入收窄、交集边界落在半格上
    ("两路错位取交集", "a={a},d={d}", "a + d"),
    ("交集收窄", "a={a},narrow={narrow}", "a + narrow"),
    ("交集半格边界", "a={a},narrow2={narrow2}", "a + narrow2"),
    # 三路反序（a+c+d 由上面「三路三个原点」钉住；arcpy 自己就不对称，两条要分别钉）
    ("三路反序 d+a+c", "d={d},a={a},c={c}", "d + a + c"),
]


@pytest.mark.parametrize("label,inputs,expression", DUAL_CASES,
                         ids=[c[0] for c in DUAL_CASES])
def test_gdal_matches_arcpy_pixel_by_pixel(gdal_runner, arcpy_runner, tmp_path: Path,
                                           label, inputs, expression):
    """同一份输入、同一表达式：网格 + nodata 掩膜 + 数值逐像元一致。"""
    paths = _scene(gdal_runner, tmp_path)
    spec = inputs.format(**paths)
    gdal_obs = _run(gdal_runner, tmp_path, spec, expression, "out/gdal.tif")
    arcpy_obs = _run(arcpy_runner, tmp_path, spec, expression, "out/arcpy.tif")
    gdal_grid = _raster(gdal_runner, _result(gdal_obs)["output"])
    arcpy_grid = _raster(gdal_runner, _result(arcpy_obs)["output"])

    _assert_same_grid(gdal_grid, arcpy_grid)
    _assert_same_pixels(gdal_grid, arcpy_grid)


# ------------------------------------------- 持久内核的 env 不泄漏（DAIR-135）
def _env(runner) -> dict:
    """读持久内核里的 ``arcpy.env.extent`` / ``cellSize``（可比较的纯数据）。"""
    outcome = runner.run(_READ_ARCPY_ENV, timeout=120)
    assert outcome.ok, outcome.error
    return outcome.result


def test_arcpy_extent_ref_restores_the_env_it_found(gdal_runner, arcpy_runner, tmp_path: Path):
    """``extent_ref`` 跑完，``arcpy.env.extent`` / ``cellSize`` 回到跑前值。

    DAIR-135 实测（Pro 3.6）：模板不还原时，DEM（UTM 10N）的范围与 ``cellSize=30``
    留在持久内核里，同内核随后的 ``zonal_statistics``（EPSG:4508）直接 ERROR 010568。
    """
    paths = _scene(gdal_runner, tmp_path)
    before = _env(arcpy_runner)
    try:
        _result(_run(arcpy_runner, tmp_path, f"a={paths['a']},b={paths['b']}", "a + b",
                     "out/env_default.tif", cell_size="10", extent_ref="b"))
        assert _env(arcpy_runner) == before, "跑完没还原 env，同内核后续任务会被 extent_ref 污染"
    finally:
        arcpy_runner.run(_RESET_ARCPY_ENV, timeout=120)


def test_arcpy_extent_ref_restores_a_preset_env(gdal_runner, arcpy_runner, tmp_path: Path):
    """跑前 env 已有值（不是 None）时也要还原成**那个值**，不是一律清空。"""
    paths = _scene(gdal_runner, tmp_path)
    preset = arcpy_runner.run(
        _PRESET_ARCPY_ENV.replace("__REF__", repr(paths["wide"])), timeout=120
    )
    assert preset.ok, preset.error
    before = _env(arcpy_runner)
    assert before["extent"] is not None and before["cellSize"] == "7", before
    try:
        _result(_run(arcpy_runner, tmp_path, f"a={paths['a']},b={paths['b']}", "a + b",
                     "out/env_preset.tif", cell_size="10", extent_ref="b"))
        after = _env(arcpy_runner)
        assert after == before, "有前置 env 时没还原成跑前值（或退化成清空）"
    finally:
        arcpy_runner.run(_RESET_ARCPY_ENV, timeout=120)


# --------------------------------------------------- 混分辨率：必须显式重采样（DAIR-139）
_MIXED_INPUTS = "\n".join(
    [
        "import os",
        "import numpy as np",
        "import rasterio",
        "from rasterio.transform import from_origin",
        "_base = __DIR__",
        "rng = np.random.default_rng(7)",
        "lc = rng.integers(1, 5, size=(30, 30)).astype('float32')",
        "# 一个 NoData 格：输出第 0 行第 0 列（格心 x=20 / y=255）会取到它",
        "lc[4, 2] = -9999.0",
        "lc_p = os.path.join(_base, 'lc10.tif')",
        "with rasterio.open(lc_p, 'w', driver='GTiff', height=30, width=30, count=1,",
        "                   dtype='float32', crs='EPSG:32650',",
        "                   transform=from_origin(0.0, 300.0, 10.0, 10.0), nodata=-9999.0) as ds:",
        "    ds.write(lc, 1)",
        "dem = (np.arange(81, dtype='float32').reshape(9, 9) + 1000.0)",
        "dem_p = os.path.join(_base, 'dem30.tif')",
        "with rasterio.open(dem_p, 'w', driver='GTiff', height=9, width=9, count=1,",
        "                   dtype='float32', crs='EPSG:32650',",
        "                   transform=from_origin(5.0, 270.0, 30.0, 30.0)) as ds:",
        "    ds.write(dem, 1)",
        "set_result({'lc': lc_p, 'dem': dem_p,",
        "            'lc_values': lc.tolist(), 'dem_values': dem.tolist()})",
    ]
)

MIXED_EXPR = "Con(lc == 2, dem, 0)"


def _mixed_inputs(runner, tmp_path: Path) -> dict:
    """10 m 分类栅格 + 30 m DEM，两套格网半像元错位（旧版两端取格口径不同的构造）。"""
    made = runner.run(_MIXED_INPUTS.replace("__DIR__", repr(str(tmp_path))), timeout=300)
    assert made.ok, made.error
    return made.result


def test_gdal_mixed_resolution_requires_explicit_resampling(gdal_runner, tmp_path: Path):
    """混分辨率不给 resampling → 明确报错（不再让两端各按自己的默认取格）。"""
    paths = _mixed_inputs(gdal_runner, tmp_path)
    inputs = f"lc={paths['lc']},dem={paths['dem']}"

    obs = _run(gdal_runner, tmp_path, inputs, MIXED_EXPR, "out/missing.tif", extent_ref="dem")
    assert not obs.ok, obs.summary
    assert "混分辨率必须先明确重采样口径" in obs.summary, obs.summary
    assert "lc=10 m" in obs.summary and "dem=30 m" in obs.summary, obs.summary
    assert not (tmp_path / "out/missing.tif").exists()

    bad = _run(gdal_runner, tmp_path, inputs, MIXED_EXPR, "out/bad.tif",
               extent_ref="dem", resampling="cubic")
    assert not bad.ok
    assert "只接受" in bad.summary and "nearest" in bad.summary, bad.summary

    ok = _result(_run(gdal_runner, tmp_path, inputs, MIXED_EXPR, "out/ok.tif",
                      extent_ref="dem", resampling="nearest"))
    # cell_size 为空时输出像元跟第一个输入（lc 的 10 m），extent_ref=dem 只定范围
    assert ok["cell_size"] == 10.0 and ok["size"] == [27, 27]


def test_gdal_mixed_resolution_nearest_follows_the_documented_rule(gdal_runner, tmp_path: Path):
    """显示 nearest 后按文档规则取格：格心落在哪个源像元就取哪个（边界归右/下一格）。"""
    paths = _mixed_inputs(gdal_runner, tmp_path)
    result = _result(_run(
        gdal_runner, tmp_path, f"lc={paths['lc']},dem={paths['dem']}", MIXED_EXPR,
        "out/nearest.tif", cell_size="30", extent_ref="dem", resampling="nearest",
    ))
    grid = _raster(gdal_runner, result["output"])
    assert grid["shape"] == [9, 9] and grid["origin"] == [5.0, 270.0]

    lc = np.array(paths["lc_values"], dtype="float64")
    dem = np.array(paths["dem_values"], dtype="float64")
    lc[lc == -9999.0] = np.nan
    xs = 5.0 + (np.arange(9) + 0.5) * 30.0
    ys = 270.0 - (np.arange(9) + 0.5) * 30.0
    ci = np.floor((xs - 0.0) / 10.0 + 1e-9).astype(int)
    ri = np.floor((300.0 - ys) / 10.0 + 1e-9).astype(int)
    picked = lc[np.ix_(ri, ci)]
    expected = np.where(picked == 2.0, dem, 0.0)
    expected = np.where(np.isfinite(picked), expected, np.nan)

    np.testing.assert_allclose(grid["array"], expected, rtol=0, atol=0, equal_nan=True)
    # 构造里 lc[4,2] 是 NoData，落在输出第 0 行第 0 列：两端都必须传播成 NoData
    assert np.isnan(grid["array"][0, 0])


@pytest.mark.parametrize("method", ["nearest", "bilinear"])
@pytest.mark.parametrize("use_extent_ref", [True, False], ids=["extent_ref", "intersection"])
def test_arcpy_and_gdal_agree_on_mixed_resolution(
    gdal_runner, arcpy_runner, tmp_path: Path, method, use_extent_ref
):
    """混分辨率 + 显式重采样：两端先重采样到同一网格，网格/掩膜/数值逐像元一致。"""
    paths = _mixed_inputs(gdal_runner, tmp_path)
    inputs = f"lc={paths['lc']},dem={paths['dem']}"
    kwargs = {"resampling": method}
    if use_extent_ref:
        kwargs["extent_ref"] = "dem"

    gdal_obs = _run(gdal_runner, tmp_path, inputs, MIXED_EXPR, f"out/gdal_{method}.tif", **kwargs)
    arcpy_obs = _run(arcpy_runner, tmp_path, inputs, MIXED_EXPR, f"out/arcpy_{method}.tif", **kwargs)
    assert gdal_obs.ok, f"gdal/{method}: {gdal_obs.summary}"
    assert arcpy_obs.ok, f"arcpy/{method}: {arcpy_obs.summary}"

    # 两端都用 GDAL 侧读回（arcpy 输出也是 .tif），逐像元比
    gdal_grid = _raster(gdal_runner, _result(gdal_obs)["output"])
    arcpy_grid = _raster(gdal_runner, _result(arcpy_obs)["output"])
    _assert_same_grid(gdal_grid, arcpy_grid)
    _assert_same_pixels(gdal_grid, arcpy_grid)


def test_arcpy_and_gdal_agree_when_the_ref_raster_sits_on_the_output_grid(
    gdal_runner, arcpy_runner, tmp_path: Path
):
    """DAIR-144：``extent_ref`` 那幅正好就是输出网格时，两端都直读、不再重采样。

    真实链（10 m 土地覆盖 + 30 m DEM，``Con(lc == 2, dem, 0)``，cell 30）实测 835 px
    掩膜差、全是 arcpy NoData / GDAL 有值：arcpy 侧无条件跑 ``_resample``，bilinear
    在网格末端取不到右下邻格，最后一行/列（以及 NoData 洞边缘）被判 NoData；
    GDAL 侧 ``_load`` 有 ``same`` 短路。这里把 dem 钉成输出网格本身（cell_size=30）。
    """
    paths = _mixed_inputs(gdal_runner, tmp_path)
    inputs = f"lc={paths['lc']},dem={paths['dem']}"
    kwargs = {"resampling": "bilinear", "extent_ref": "dem", "cell_size": "30"}
    gdal_obs = _run(gdal_runner, tmp_path, inputs, MIXED_EXPR, "out/gdal_aligned.tif", **kwargs)
    arcpy_obs = _run(arcpy_runner, tmp_path, inputs, MIXED_EXPR, "out/arcpy_aligned.tif", **kwargs)
    assert gdal_obs.ok, f"gdal: {gdal_obs.summary}"
    assert arcpy_obs.ok, f"arcpy: {arcpy_obs.summary}"

    gdal_grid = _raster(gdal_runner, _result(gdal_obs)["output"])
    arcpy_grid = _raster(gdal_runner, _result(arcpy_obs)["output"])
    # 输出网格就是 dem 的网格（extent_ref=dem + cell_size=30）
    assert gdal_grid["shape"] == [9, 9] and gdal_grid["origin"] == [5.0, 270.0]
    _assert_same_grid(gdal_grid, arcpy_grid)
    _assert_same_pixels(gdal_grid, arcpy_grid)
    # 除了 lc 自己那一格 NoData（构造里的 lc[4,2] → 输出 (0,0)），末端不该再被吃掉像元：
    # 修好前 arcpy 是 5 个（末行末列多 4 个，bilinear 取不到右下邻格）。
    assert int(np.isnan(gdal_grid["array"]).sum()) == 1
    assert int(np.isnan(arcpy_grid["array"]).sum()) == 1


def test_arcpy_and_gdal_both_reject_mixed_resolution_without_resampling(
    gdal_runner, arcpy_runner, tmp_path: Path
):
    """不给 resampling 时两端都要报「混分辨率必须先明确重采样口径」，不能一边静默跑。"""
    paths = _mixed_inputs(gdal_runner, tmp_path)
    inputs = f"lc={paths['lc']},dem={paths['dem']}"
    for label, runner in (("gdal", gdal_runner), ("arcpy", arcpy_runner)):
        obs = _run(runner, tmp_path, inputs, MIXED_EXPR, f"out/{label}_bad.tif",
                   extent_ref="dem")
        assert not obs.ok, f"{label} 没报错：{obs.summary}"
        assert "混分辨率必须先明确重采样口径" in obs.summary, f"{label}: {obs.summary}"


# ------------------------------------- 输入 CRS 不一致：两端都必须报错（DAIR-152）
_OTHER_RASTER = "\n".join(
    [
        "import os",
        "import numpy as np",
        "import rasterio",
        "from rasterio.transform import from_origin",
        "_p = os.path.join(__DIR__, __NAME__ + '.tif')",
        "with rasterio.open(_p, 'w', driver='GTiff', height=10, width=10, count=1,",
        "                   dtype='float32', crs=__CRS__,",
        "                   transform=from_origin(0.0, 50.0, 5.0, 5.0), nodata=-9999.0) as ds:",
        "    ds.write(np.full((10, 10), 3.0, dtype='float32'), 1)",
        "set_result({'path': _p})",
    ]
)


def _other_raster(runner, tmp_path: Path, name: str, crs) -> str:
    """与场景里 a 同 shape / 同 transform（10×10 @5 m，原点 (0,50)）的另一幅栅格。"""
    made = runner.run(
        _OTHER_RASTER.replace("__DIR__", repr(str(tmp_path)))
        .replace("__NAME__", repr(name))
        .replace("__CRS__", repr(crs)),
        timeout=300,
    )
    assert made.ok, made.error
    return made.result["path"]


@pytest.mark.parametrize(
    "crs, message",
    [("EPSG:32651", "坐标系不一致"), (None, "没有坐标系")],
    ids=["other_epsg", "no_crs"],
)
def test_arcpy_and_gdal_both_reject_inputs_with_bad_crs(
    gdal_runner, arcpy_runner, tmp_path: Path, crs, message
):
    """DAIR-152：同 shape / 同 transform / 不同 EPSG（或干脆没有 CRS）的输入，两端都要报错。

    arcpy 侧原先没有任何 CRS 校验（GDAL 侧一直有）：混分辨率分支按各自 Describe 出来的
    坐标数值直接建窗、重采样，两套 UTM 带的栅格被当成同一坐标空间**静默**出结果
    （DAIR-149 实测 10×10 全有效，GDAL 侧报「坐标系不一致」）。同 EPSG 不同 WKT 要放过，
    所以有 WKID 时比 WKID（factoryCode）而不是比名字；没有 WKID 时比投影参数（DAIR-157）。
    """
    paths = _scene(gdal_runner, tmp_path)
    other = _other_raster(gdal_runner, tmp_path, "other", crs)
    inputs = f"a={paths['a']},b={other}"
    for label, runner in (("gdal", gdal_runner), ("arcpy", arcpy_runner)):
        obs = _run(runner, tmp_path, inputs, "a + b", f"out/{label}_crs.tif")
        assert not obs.ok, f"{label} 没报错：{obs.summary}"
        assert message in obs.summary, f"{label}: {obs.summary}"
        assert not (tmp_path / f"out/{label}_crs.tif").exists()


def test_arcpy_and_gdal_both_reject_extent_ref_path_with_other_crs(
    gdal_runner, arcpy_runner, tmp_path: Path
):
    """路径形态的 extent_ref 不经过输入绑定：范围按 ref 的坐标取、输出却写输入的 CRS。

    GDAL 侧一直拦（DAIR-119），arcpy 侧原先只在混分辨率分支里才打开 ref、且不查 CRS。
    """
    paths = _scene(gdal_runner, tmp_path)
    other = _other_raster(gdal_runner, tmp_path, "ref51", "EPSG:32651")
    for label, runner in (("gdal", gdal_runner), ("arcpy", arcpy_runner)):
        obs = _run(runner, tmp_path, f"a={paths['a']}", "a + 1", f"out/{label}_ref.tif",
                   cell_size="5", extent_ref=other)
        assert not obs.ok, f"{label} 没报错：{obs.summary}"
        assert "坐标系不一致" in obs.summary, f"{label}: {obs.summary}"


# ------------------- 同 EPSG、不同 WKT 写法：两端都必须放过（DAIR-155）
# 直接把变体 WKT 交给 rasterio 写 GTiff 是**测不出**这条的：GDAL 会把三种写法归并回
# EPSG:32650 的 geokeys（实测写盘后两端读到的 WKT 文本逐字相同，DAIR-154 探针的三次
# 「放过」测的就是这种归一化后的同一份 CRS）。要让文本差异活到两端读的那一刻，只能用
# GDAL 的 PAM：GTiff 里照常写 32650，再把变体 WKT 放进 `.tif.aux.xml` 的 `<SRS>`。
_WKT_VARIANT_RASTER = "\n".join(
    [
        "import os, warnings",
        "import numpy as np, rasterio, pyproj",
        "from rasterio.transform import from_origin",
        "warnings.simplefilter('ignore')",
        "base = pyproj.CRS.from_epsg(32650)",
        "variants = {",
        "    'esri': base.to_wkt(version='WKT1_ESRI'),",
        "    'proj4': pyproj.CRS.from_proj4(base.to_proj4()).to_wkt(),",
        "    'noauth': base.to_wkt(version='WKT1_GDAL').replace(',AUTHORITY[\"EPSG\",\"32650\"]', ''),",
        "}",
        "out = {'wkt': dict(variants)}",
        "for name, wkt in variants.items():",
        "    path = os.path.join(__DIR__, 'wkt_' + name + '.tif')",
        "    with rasterio.open(path, 'w', driver='GTiff', height=10, width=10, count=1,",
        "                       dtype='float32', crs=base.to_wkt(),",
        "                       transform=from_origin(0.0, 50.0, 5.0, 5.0), nodata=-9999.0) as ds:",
        "        ds.write(np.full((10, 10), 3.0, dtype='float32'), 1)",
        "    with open(path + '.aux.xml', 'w', encoding='utf-8') as fh:",
        "        fh.write('<PAMDataset>\\n  <SRS>%s</SRS>\\n</PAMDataset>\\n'",
        "                 % wkt.replace('&', '&amp;').replace('<', '&lt;'))",
        "    out[name] = path",
        "set_result(out)",
    ]
)

_WKT_READBACK = "\n".join(
    [
        "import rasterio, pyproj",
        "base = pyproj.CRS.from_epsg(32650)",
        "out = {}",
        "for name, path in __PATHS__.items():",
        "    with rasterio.open(path) as ds:",
        "        wkt = ds.crs.to_wkt() if ds.crs else None",
        "        out[name] = {'wkt': wkt, 'epsg': ds.crs.to_epsg() if ds.crs else None,",
        "                     'same_epsg': bool(pyproj.CRS.from_wkt(wkt).equals(base, ignore_axis_order=True))}",
        "set_result(out)",
    ]
)

_ARCPY_SR_READBACK = "\n".join(
    [
        "import arcpy",
        "out = {}",
        "for name, path in __PATHS__.items():",
        "    sr = arcpy.Describe(path).spatialReference",
        "    out[name] = {'name': str(sr.name), 'code': int(getattr(sr, 'factoryCode', 0) or 0),",
        "                 'wkt': sr.exportToString()}",
        "set_result(out)",
    ]
)


def _wkt_variant_rasters(runner, tmp_path: Path) -> dict:
    """与场景里 a 同 shape / 同 transform（10×10 @5 m，原点 (0,50)）的三种 WKT 写法。"""
    made = runner.run(_WKT_VARIANT_RASTER.replace("__DIR__", repr(str(tmp_path))), timeout=300)
    assert made.ok, made.error
    return made.result


_PROJ4_PAM_RASTER = "\n".join(
    [
        "import os, warnings",
        "import numpy as np, rasterio, pyproj",
        "from rasterio.transform import from_origin",
        "warnings.simplefilter('ignore')",
        "base = pyproj.CRS.from_epsg(__EPSG__)",
        "path = os.path.join(__DIR__, __NAME__ + '.tif')",
        "with rasterio.open(path, 'w', driver='GTiff', height=__ROWS__, width=__COLS__,",
        "                   count=1, dtype='float32', crs=base.to_wkt(),",
        "                   transform=from_origin(__X0__, __Y0__, __CELL__, __CELL__),",
        "                   nodata=-9999.0) as ds:",
        "    ds.write(np.full((__ROWS__, __COLS__), 3.0, dtype='float32'), 1)",
        "wkt = pyproj.CRS.from_proj4(base.to_proj4()).to_wkt()",
        "with open(path + '.aux.xml', 'w', encoding='utf-8') as fh:",
        "    fh.write('<PAMDataset>\\n  <SRS>%s</SRS>\\n</PAMDataset>\\n'",
        "             % wkt.replace('&', '&amp;').replace('<', '&lt;'))",
        "set_result(path)",
    ]
)


def _proj4_pam_raster(
    runner, tmp_path: Path, epsg: int, name: str, *,
    rows: int = 10, cols: int = 10, cell: float = 5.0, x0: float = 0.0, y0: float = 50.0,
) -> str:
    """内嵌 EPSG:epsg、PAM `<SRS>` 换成该 EPSG 的 proj4 派生 WKT（无 WKID、名字 unknown）。"""
    script = _PROJ4_PAM_RASTER
    for token, value in (
        ("__EPSG__", str(epsg)), ("__NAME__", repr(name)),
        ("__ROWS__", str(rows)), ("__COLS__", str(cols)), ("__CELL__", repr(float(cell))),
        ("__X0__", repr(float(x0))), ("__Y0__", repr(float(y0))),
        ("__DIR__", repr(str(tmp_path))),
    ):
        script = script.replace(token, value)
    made = runner.run(script, timeout=300)
    assert made.ok, made.error
    return made.result


#: 两点等距（Two_Point_Equidistant）的 ESRI WKT。Esri 支持这个投影：arcpy 用
#: ``loadFromString`` 读进来后 ``longitudeOf1st/latitudeOf1st/longitudeOf2nd/latitudeOf2nd``
#: 分别读回 117/31/123/35（DAIR-159 探针）。名字两份故意相同、只让 1st 点经线不同，
#: 钉住「名字分不开、只有这四个参数能分开」这个反例。
_TPEQD_WKT = (
    'PROJCS["tpeqd_same",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]],PROJECTION["Two_Point_Equidistant"],'
    'PARAMETER["Longitude_Of_1st_Point",__LON1__],PARAMETER["Latitude_Of_1st_Point",31.0],'
    'PARAMETER["Longitude_Of_2nd_Point",123.0],PARAMETER["Latitude_Of_2nd_Point",35.0],'
    'PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],UNIT["Meter",1.0]]'
)

_TPEQD_PAM_RASTERS = "\n".join(
    [
        "import os, warnings",
        "import numpy as np, rasterio",
        "from rasterio.transform import from_origin",
        "warnings.simplefilter('ignore')",
        "out = {}",
        "for name, wkt in __WKTS__.items():",
        "    path = os.path.join(__DIR__, 'tpeqd_' + name + '.tif')",
        "    with rasterio.open(path, 'w', driver='GTiff', height=10, width=10, count=1,",
        "                       dtype='float32', transform=from_origin(0.0, 50.0, 5.0, 5.0),",
        "                       nodata=-9999.0) as ds:",
        "        ds.write(np.full((10, 10), 3.0, dtype='float32'), 1)",
        "    with open(path + '.aux.xml', 'w', encoding='utf-8') as fh:",
        "        fh.write('<PAMDataset>\\n  <SRS>%s</SRS>\\n</PAMDataset>\\n'",
        "                 % wkt.replace('&', '&amp;').replace('<', '&lt;'))",
        "    out[name] = path",
        "set_result(out)",
    ]
)


def _tpeqd_pam_rasters(runner, tmp_path: Path) -> dict:
    """同 shape / 同 transform，只有 1st 点经线不同（117 vs 119）的两点等距栅格。"""
    wkts = {tag: _TPEQD_WKT.replace("__LON1__", lon)
            for tag, lon in (("a", "117.0"), ("b", "119.0"))}
    script = (_TPEQD_PAM_RASTERS.replace("__WKTS__", repr(wkts))
              .replace("__DIR__", repr(str(tmp_path))))
    made = runner.run(script, timeout=300)
    assert made.ok, made.error
    return made.result


def _sr_facts(runner, paths: dict[str, str], script: str) -> dict:
    """让某一端自己报出它读到的 CRS：WKT 文本 / EPSG 码（GDAL）或名字与 WKID（arcpy）。"""
    outcome = runner.run(script.replace("__PATHS__", repr(paths)), timeout=300)
    assert outcome.ok, outcome.error
    return outcome.result


@pytest.mark.parametrize("variant", ["esri", "noauth"])
def test_same_epsg_with_different_wkt_text_is_accepted(
    gdal_runner, arcpy_runner, tmp_path: Path, variant
):
    """DAIR-155：同一 EPSG:32650、WKT 文本不同的两幅，两端都必须放过。

    先钉前提再断言行为（DAIR-89 的规矩）：三份 WKT 文本互不相同、变体被 GDAL 读成
    「等价于 32650 但文本不同」、被 ArcGIS 归一化回 32650（WKID 相同）。谁把判据收紧成
    「WKT 字符串相等」，`gdal_facts["variant"]["wkt"] != gdal_facts["ref"]["wkt"]` 这一条
    就说明会误伤；arcpy 侧的文本会被 ArcGIS 归一化，判据只能是 WKID（factoryCode）。
    """
    scene = _scene(gdal_runner, tmp_path)
    made = _wkt_variant_rasters(gdal_runner, tmp_path)
    ref, other = scene["a"], made[variant]

    texts = made["wkt"]
    assert len({texts["esri"], texts["proj4"], texts["noauth"]}) == 3, texts

    gdal_facts = _sr_facts(gdal_runner, {"ref": ref, "variant": other}, _WKT_READBACK)
    assert gdal_facts["variant"]["same_epsg"], gdal_facts
    assert gdal_facts["variant"]["wkt"] != gdal_facts["ref"]["wkt"], "变体与参考的 WKT 文本相同＝测的是假等价"

    arcpy_facts = _sr_facts(arcpy_runner, {"ref": ref, "variant": other}, _ARCPY_SR_READBACK)
    assert arcpy_facts["ref"]["code"] == arcpy_facts["variant"]["code"] == 32650, arcpy_facts

    for label, runner in (("gdal", gdal_runner), ("arcpy", arcpy_runner)):
        obs = _run(runner, tmp_path, f"a={ref},b={other}", "a + b",
                   f"out/wkt_{variant}_{label}.tif")
        assert obs.ok, f"{label} 把同 EPSG 的 {variant} 写法当成不一致：{obs.summary}"
        assert (tmp_path / f"out/wkt_{variant}_{label}.tif").exists()


def test_same_epsg_proj4_derived_wkt_is_accepted(gdal_runner, arcpy_runner, tmp_path: Path):
    """DAIR-155 第三种写法：proj4 派生 WKT（PROJ 字符串 → WKT2，没有 EPSG ID）。

    GDAL 侧按 pyproj 等价判据放过；arcpy 侧读到 WKID=0、名字 unknown。DAIR-157 把 arcpy
    判据从「无 WKID 比名字」改成「无 WKID 比投影参数」后，这条从 xfail 转绿（参数逐项相同：
    Central_Meridian 117、False_Easting 500000、Scale_Factor 0.9996、datum D_WGS_1984…）。
    """
    scene = _scene(gdal_runner, tmp_path)
    made = _wkt_variant_rasters(gdal_runner, tmp_path)
    ref, other = scene["a"], made["proj4"]

    gdal_facts = _sr_facts(gdal_runner, {"ref": ref, "variant": other}, _WKT_READBACK)
    assert gdal_facts["variant"]["same_epsg"], gdal_facts
    assert gdal_facts["variant"]["wkt"] != gdal_facts["ref"]["wkt"]

    for label, runner in (("gdal", gdal_runner), ("arcpy", arcpy_runner)):
        obs = _run(runner, tmp_path, f"a={ref},b={other}", "a + b",
                   f"out/wkt_proj4_{label}.tif")
        assert obs.ok, f"{label} 把等价的 proj4 派生 WKT 判成不一致：{obs.summary}"


def test_two_different_crs_without_wkid_are_rejected(gdal_runner, arcpy_runner, tmp_path: Path):
    """DAIR-157 误放方向：两幅都**没有 WKID**、名字都是 'unknown'，但中央经线 117 vs 123。

    旧判据在无 WKID 时比名字，`(0, 'unknown')` 两边相同 → 混分辨率分支按各自的坐标数值
    建窗重采样、**静默出结果**（与 DAIR-152 修掉的 WKID 场景同一条漏洞；本单实测修复前
    arcpy 侧 `ok=True`、29×29、mean 102）。新判据在无 WKID 时比投影参数：只差
    Central_Meridian 也必须拒绝。构造照抄 DAIR-152：10 m 40×40 @32650 + 30 m 10×10 @32651。
    """
    ref = _proj4_pam_raster(gdal_runner, tmp_path, 32650, "nowkid50", rows=40, cols=40,
                            cell=10.0, x0=499945.0, y0=3400055.0)
    other = _proj4_pam_raster(gdal_runner, tmp_path, 32651, "nowkid51", rows=10, cols=10,
                              cell=30.0, x0=500000.0, y0=3400000.0)

    # 先钉前提：arcpy 读回的都是「无 WKID + 同一个占位名字」，靠名字确实分不开
    facts = _sr_facts(arcpy_runner, {"ref": ref, "other": other}, _ARCPY_SR_READBACK)
    assert facts["ref"]["code"] == facts["other"]["code"] == 0, facts
    assert facts["ref"]["name"] == facts["other"]["name"] == "unknown", facts
    assert facts["ref"]["wkt"] != facts["other"]["wkt"], "两份 WKT 文本相同＝反例本身没构造出来"

    for label, runner in (("gdal", gdal_runner), ("arcpy", arcpy_runner)):
        obs = _run(runner, tmp_path, f"a={ref},b={other}", "a + b", f"out/nowkid_{label}.tif",
                   resampling="nearest")
        assert not obs.ok, f"{label} 把两个不同的无 WKID CRS 放过了：{obs.summary}"
        assert "坐标系不一致" in obs.summary, f"{label}: {obs.summary}"
        assert not (tmp_path / f"out/nowkid_{label}.tif").exists()


def test_two_point_equidistant_without_wkid_is_rejected(gdal_runner, arcpy_runner, tmp_path: Path):
    """DAIR-159：无 WKID、名字相同、只差 1st 点经线（117 vs 119）的两点等距必须拒绝。

    arcpy 侧的参数元组原先只收了 12 个数值参数，1st/2nd 点那四个（longitudeOf1st 等）
    没纳入 → 两份只差该参数的 CRS 判成一致。Esri 支持 Two_Point_Equidistant，属真实场景。
    文案也要能定位到差异（两边名字都是同一个占位名）。
    """
    paths = _tpeqd_pam_rasters(gdal_runner, tmp_path)

    # 先钉前提：WKID 都是 0、名字相同、WKT 文本不同（靠名字确实分不开）
    facts = _sr_facts(arcpy_runner, paths, _ARCPY_SR_READBACK)
    assert facts["a"]["code"] == facts["b"]["code"] == 0, facts
    assert facts["a"]["name"] == facts["b"]["name"], f"名字不同＝反例没构造出来: {facts}"
    assert facts["a"]["wkt"] != facts["b"]["wkt"], "两份 WKT 文本相同＝反例本身没构造出来"

    for label, runner in (("gdal", gdal_runner), ("arcpy", arcpy_runner)):
        obs = _run(runner, tmp_path, f"a={paths['a']},b={paths['b']}", "a + b",
                   f"out/tpeqd_{label}.tif")
        assert not obs.ok, f"{label} 把只差 1st 点经线的两点等距放过了：{obs.summary}"
        assert "坐标系不一致" in obs.summary, f"{label}: {obs.summary}"
        # 报错要能定位差异：两端名字相同，必须靠 1st 点参数区分
        assert ("1st" in obs.summary and "117" in obs.summary
                and "119" in obs.summary), f"{label}: {obs.summary}"
        assert not (tmp_path / f"out/tpeqd_{label}.tif").exists()


@pytest.mark.parametrize(
    "ref_epsg, other_epsg",
    [(32650, 4490), (4508, 4547)],
    ids=["geographic_vs_utm", "gauss_kruger_111e_vs_114e"],
)
def test_different_epsg_is_still_rejected(
    gdal_runner, arcpy_runner, tmp_path: Path, ref_epsg, other_epsg
):
    """反例：WKID 真的不同就不能被「放过」那条放过去。

    4490（CGCS2000 地理坐标系）vs 32650；4508 / 4547 同为 CGCS2000 高斯-克吕格、只差中央
    经线（CGCS2000_GK_CM_111E vs CGCS2000_3_Degree_GK_CM_114E）—— 同基准、同单位、同一套
    参数名，靠 WKT 里大半相同的前缀很容易被“看起来一样”糊过去。
    """
    ref = _other_raster(gdal_runner, tmp_path, f"ref{ref_epsg}", f"EPSG:{ref_epsg}")
    other = _other_raster(gdal_runner, tmp_path, f"other{other_epsg}", f"EPSG:{other_epsg}")

    facts = _sr_facts(gdal_runner, {"ref": ref, "other": other}, _WKT_READBACK)
    assert facts["ref"]["wkt"] != facts["other"]["wkt"], "两份 WKT 文本相同＝反例本身没构造出来"
    assert facts["ref"]["epsg"] == ref_epsg and facts["other"]["epsg"] == other_epsg, facts

    for label, runner in (("gdal", gdal_runner), ("arcpy", arcpy_runner)):
        obs = _run(runner, tmp_path, f"a={ref},b={other}", "a + b",
                   f"out/epsg_{ref_epsg}_{other_epsg}_{label}.tif")
        assert not obs.ok, f"{label} 没报错：{obs.summary}"
        assert "坐标系不一致" in obs.summary, f"{label}: {obs.summary}"
        assert not (tmp_path / f"out/epsg_{ref_epsg}_{other_epsg}_{label}.tif").exists()

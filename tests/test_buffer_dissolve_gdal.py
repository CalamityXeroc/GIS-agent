# -*- coding: utf-8 -*-
"""buffer_dissolve 的 GDAL 版真跑 + 双端一致（面积相对差 ≤1e-3）。

输入是现场生成的**投影坐标系**面（EPSG:32650，米）：三条 1 km × 1 km 的方块，
彼此分开，缓冲 500 m 后面积可手算：

    每个方块 = 1e6（本体）+ 4×1000×500（四边）+ π×500²（四角圆弧）= 3.785e6 m²
    三条     = 11,356,194 m²（圆弧按理想圆算；两种实现都是折线近似，允许 1% 偏差）

所以 ``total_area`` 不只能“有值”，还能对数；双端一致按 **1e-3 相对差**卡
（两端都用折线近似圆弧、顶点数不同，实测 1e-4 量级；1e-6 不是本配方的承诺，见 pitfalls）。

缺环境就 skip（与 ``test_gdal_runner.py`` 里真 GDAL 用例同一套口径）：
``.venv-gdal`` 不在 → 整个文件 skip；ArcGIS Pro 找不到 → 只有双端用例 skip。
CI 机器两者都没有，by design。
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from gis_cli.recipes.store import RecipeLibrary
from gis_cli.runner.gdal_runner import GdalCodeRunner, default_python_executable
from gis_cli.runtime.code_runner import CodeRunner

#: 三条方块的 500 m 缓冲区理论面积（m²）
THEORY_AREA = 3 * (1_000_000 + 4 * 1000 * 500 + math.pi * 500**2)

MAKE_ZONES = "\n".join(
    [
        "import geopandas as gpd",
        "from shapely.geometry import Polygon",
        "polys = [Polygon([(x, 0), (x, 1000), (x + 1000, 1000), (x + 1000, 0)])",
        "         for x in (0, 2000, 4000)]",
        "gdf = gpd.GeoDataFrame({'zone': ['a', 'b', 'c']}, geometry=polys, crs='EPSG:32650')",
        "gdf.to_file(__PATH__, driver='GPKG', engine='pyogrio', layer='zones')",
        "set_result({'features': int(len(gdf))})",
    ]
)

MAKE_GEOGRAPHIC = "\n".join(
    [
        "import geopandas as gpd",
        "from shapely.geometry import Polygon",
        "gdf = gpd.GeoDataFrame({'zone': ['a']},",
        "                       geometry=[Polygon([(0, 0), (0, 4), (4, 4), (4, 0)])],",
        "                       crs='EPSG:4490')",
        "gdf.to_file(__PATH__, driver='GPKG', engine='pyogrio', layer='zones_geo')",
        "set_result({'features': int(len(gdf))})",
    ]
)

MAKE_ZONES_WITH_NULL = "\n".join(
    [
        "import geopandas as gpd",
        "from shapely.geometry import Polygon",
        "polys = [Polygon([(x, 0), (x, 1000), (x + 1000, 1000), (x + 1000, 0)])",
        "         for x in (0, 2000, 4000, 6000)]",
        "# 第三条的 zone 故意留空：arcpy 的 LIST 会把它单独归成一个 NULL 组，",
        "# 而 pandas 的 groupby 默认 dropna=True 会把整组丢掉（DAIR-68 非阻塞项 3）",
        "gdf = gpd.GeoDataFrame({'zone': ['a', 'b', None, 'a']}, geometry=polys, crs='EPSG:32650')",
        "gdf.to_file(__PATH__, driver='GPKG', engine='pyogrio', layer='zones')",
        "set_result({'features': int(len(gdf))})",
    ]
)

READ_ZONE_VALUES = "\n".join(
    [
        "import geopandas as gpd",
        "gdf = gpd.read_file(__PATH__, engine='pyogrio')",
        "col = gdf[__FIELD__]",
        "set_result({'nulls': int(col.isna().sum()),",
        "            'values': sorted(str(v) for v in col.dropna())})",
    ]
)

READ_INFO = "\n".join(
    [
        "import pyogrio",
        "info = pyogrio.read_info(__PATH__)",
        "_fields = info.get('fields')",
        "set_result({'driver': info.get('driver'), 'features': int(info['features']),",
        "            'crs': str(info.get('crs') or ''),",
        "            'fields': [str(f) for f in (_fields if _fields is not None else [])]})",
    ]
)

MAKE_ARCPY_FC = "\n".join(
    [
        "import arcpy",
        "arcpy.env.overwriteOutput = True",
        "sr = arcpy.SpatialReference(32650)",
        "gdb = __GDB__",
        "if not arcpy.Exists(gdb):",
        "    arcpy.management.CreateFileGDB(__PARENT__, 'zones_arcpy.gdb')",
        "fc = gdb + r'\\zones'",
        "if arcpy.Exists(fc):",
        "    arcpy.management.Delete(fc)",
        "arcpy.management.CreateFeatureclass(gdb, 'zones', 'POLYGON', spatial_reference=sr)",
        "arcpy.management.AddField(fc, 'zone', 'TEXT', field_length=20)",
        "with arcpy.da.InsertCursor(fc, ['SHAPE@', 'zone']) as cur:",
        "    for x, z in ((0, 'a'), (2000, 'b'), (4000, 'c')):",
        "        pts = [arcpy.Point(x, 0), arcpy.Point(x, 1000),",
        "               arcpy.Point(x + 1000, 1000), arcpy.Point(x + 1000, 0)]",
        "        cur.insertRow([arcpy.Polygon(arcpy.Array(pts), sr), z])",
        "set_result({'count': int(arcpy.management.GetCount(fc)[0]), 'fc': fc})",
    ]
)

#: 分组字段带空值的 arcpy 版输入（同几何，第三条 zone 为 NULL）
MAKE_ARCPY_FC_WITH_NULL = "\n".join(
    [
        "import arcpy",
        "arcpy.env.overwriteOutput = True",
        "sr = arcpy.SpatialReference(32650)",
        "gdb = __GDB__",
        "if not arcpy.Exists(gdb):",
        "    arcpy.management.CreateFileGDB(__PARENT__, 'zones_null_arcpy.gdb')",
        "fc = gdb + r'\\zones'",
        "if arcpy.Exists(fc):",
        "    arcpy.management.Delete(fc)",
        "arcpy.management.CreateFeatureclass(gdb, 'zones', 'POLYGON', spatial_reference=sr)",
        "arcpy.management.AddField(fc, 'zone', 'TEXT', field_length=20)",
        "with arcpy.da.InsertCursor(fc, ['SHAPE@', 'zone']) as cur:",
        "    for x, z in ((0, 'a'), (2000, 'b'), (4000, None), (6000, 'a')):",
        "        pts = [arcpy.Point(x, 0), arcpy.Point(x, 1000),",
        "               arcpy.Point(x + 1000, 1000), arcpy.Point(x + 1000, 0)]",
        "        cur.insertRow([arcpy.Polygon(arcpy.Array(pts), sr), z])",
        "set_result({'count': int(arcpy.management.GetCount(fc)[0]), 'fc': fc})",
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


def _run(recipe_lib: RecipeLibrary, runner, tmp_path: Path, **params):
    recipe_lib.load()
    ctx = SimpleNamespace(workspace=tmp_path, code_runner=runner, exec_timeout=600)
    return recipe_lib.run("buffer_dissolve", params, ctx)


def _buffer(runner, recipe_lib, tmp_path: Path, name: str, **overrides):
    params = {
        "input_path": str(tmp_path / "zones.gpkg"),
        "output_path": str(tmp_path / name),
        "distance": "500 Meters",
        "dissolve_type": "ALL",
        "dissolve_field": "",
    }
    params.update(overrides)
    return _run(recipe_lib, runner, tmp_path, **params)


def _make_zones(runner, tmp_path: Path) -> Path:
    path = tmp_path / "zones.gpkg"
    made = runner.run(MAKE_ZONES.replace("__PATH__", repr(str(path))), timeout=180)
    assert made.ok, made.error
    return path


def _make_zones_with_null(runner, tmp_path: Path) -> Path:
    path = tmp_path / "zones_null.gpkg"
    made = runner.run(MAKE_ZONES_WITH_NULL.replace("__PATH__", repr(str(path))), timeout=180)
    assert made.ok, made.error
    return path


def _prepare_arcpy_out_gdb(arcpy_runner, tmp_path: Path) -> Path:
    """arcpy 的 Buffer 不会凭空建容器：输出库得先存在（实测写新建 .gpkg 报 cannot open）。"""
    out_gdb = tmp_path / "arcpy_out.gdb"
    made_gdb = arcpy_runner.run(
        "\n".join(
            [
                "import arcpy",
                f"if not arcpy.Exists({str(out_gdb)!r}):",
                f"    arcpy.management.CreateFileGDB({str(tmp_path)!r}, 'arcpy_out.gdb')",
                "set_result({'ok': True})",
            ]
        ),
        timeout=180,
    )
    assert made_gdb.ok, made_gdb.error
    return out_gdb


def _read_info(runner, path: Path) -> dict:
    outcome = runner.run(READ_INFO.replace("__PATH__", repr(str(path))), timeout=120)
    assert outcome.ok, outcome.error
    return outcome.result


# ----------------------------------------------------------- GDAL 侧真跑
def test_gdal_dissolve_all_writes_one_feature(gdal_runner, tmp_path: Path):
    _make_zones(gdal_runner, tmp_path)
    lib = RecipeLibrary(code_runner=gdal_runner)

    obs = _buffer(gdal_runner, lib, tmp_path, "buffer_all.gpkg", dissolve_type="ALL")

    assert obs.ok, obs.summary
    payload = obs.data["result"]
    assert payload["dissolve_option"] == "ALL"
    assert payload["feature_count"] == 1
    assert payload["kept_field"] is False
    assert payload["distance"] == "500 Meters"
    assert abs(payload["total_area"] - THEORY_AREA) / THEORY_AREA < 0.01, payload["total_area"]

    # 产物必须是真 GPKG（不是“文件存在”就算过）
    info = _read_info(gdal_runner, Path(payload["output"]))
    assert info["driver"] == "GPKG"
    assert info["features"] == 1
    assert "32650" in info["crs"]


def test_gdal_dissolve_none_keeps_every_feature(gdal_runner, tmp_path: Path):
    _make_zones(gdal_runner, tmp_path)
    lib = RecipeLibrary(code_runner=gdal_runner)

    obs = _buffer(gdal_runner, lib, tmp_path, "buffer_none.gpkg", dissolve_type="NONE")

    assert obs.ok, obs.summary
    payload = obs.data["result"]
    assert payload["dissolve_option"] == "NONE"
    assert payload["feature_count"] == 3, "NONE 必须保留每条要素（可回连原表）"
    assert payload["kept_field"] is False
    assert abs(payload["total_area"] - THEORY_AREA) / THEORY_AREA < 0.01


def test_gdal_dissolve_list_groups_by_field(gdal_runner, tmp_path: Path):
    _make_zones(gdal_runner, tmp_path)
    lib = RecipeLibrary(code_runner=gdal_runner)

    obs = _buffer(
        gdal_runner, lib, tmp_path, "buffer_list.gpkg", dissolve_type="LIST", dissolve_field="zone"
    )

    assert obs.ok, obs.summary
    payload = obs.data["result"]
    assert payload["dissolve_option"] == "LIST"
    assert payload["feature_count"] == 3  # 三个 zone 各一条
    assert payload["kept_field"] is True
    info = _read_info(gdal_runner, Path(payload["output"]))
    assert "zone" in info["fields"]


def test_gdal_dissolve_list_keeps_null_group(gdal_runner, tmp_path: Path):
    """分组字段为空的行不能丢：arcpy 的 LIST 会单独留一个 NULL 组（DAIR-68 非阻塞项 3）。"""
    _make_zones_with_null(gdal_runner, tmp_path)
    lib = RecipeLibrary(code_runner=gdal_runner)

    obs = _buffer(
        gdal_runner, lib, tmp_path, "buffer_list_null.gpkg",
        dissolve_type="LIST", dissolve_field="zone",
        input_path=str(tmp_path / "zones_null.gpkg"),
    )

    assert obs.ok, obs.summary
    payload = obs.data["result"]
    assert payload["dissolve_option"] == "LIST"
    assert payload["feature_count"] == 3, "a / b / NULL 三组，NULL 组不能丢"

    values = gdal_runner.run(
        READ_ZONE_VALUES.replace("__PATH__", repr(payload["output"])).replace(
            "__FIELD__", repr("zone")
        ),
        timeout=120,
    )
    assert values.ok, values.error
    assert values.result["nulls"] == 1, values.result
    assert values.result["values"] == ["a", "b"]


def test_gdal_distance_units_all_mean_the_same(gdal_runner, tmp_path: Path):
    """带单位的写法都要换算成米 —— ``1 Kilometers`` / ``1000 m`` / ``1km`` 与 ``1000 Meters`` 同面积。"""
    _make_zones(gdal_runner, tmp_path)
    lib = RecipeLibrary(code_runner=gdal_runner)

    areas: dict[str, float] = {}
    for index, distance in enumerate(["1000 Meters", "1 Kilometers", "1000 m", "1km"]):
        obs = _buffer(gdal_runner, lib, tmp_path, f"buffer_units_{index}.gpkg", distance=distance)
        assert obs.ok, f"{distance}: {obs.summary}"
        payload = obs.data["result"]
        assert payload["distance"] == distance
        areas[distance] = payload["total_area"]

    base = areas["1000 Meters"]
    assert base > 0
    for distance, area in areas.items():
        assert abs(area - base) / base < 1e-9, f"{distance} 与 1000 Meters 面积不一致: {areas}"


def test_gdal_rejects_geographic_crs(gdal_runner, tmp_path: Path):
    """地理坐标系必须拒绝：按度算缓冲区结果完全错，arcpy 版有这条校验，GDAL 版不能少。"""
    geo = tmp_path / "geo.gpkg"
    made = gdal_runner.run(MAKE_GEOGRAPHIC.replace("__PATH__", repr(str(geo))), timeout=180)
    assert made.ok, made.error
    lib = RecipeLibrary(code_runner=gdal_runner)

    obs = _buffer(gdal_runner, lib, tmp_path, "buffer_geo.gpkg", input_path=str(geo))

    assert not obs.ok
    assert "需要投影坐标系" in obs.summary
    assert not (tmp_path / "buffer_geo.gpkg").exists(), "报错时不应留下半成品"


# ------------------------------------------- .gdb 要素类路径（DAIR-217 B4）
def test_gdal_reads_gdb_feature_class_path(gdal_runner, arcpy_runner, tmp_path: Path):
    """B4：arcpy 侧交出来的 ``x.gdb\要素类`` 不能报「输入数据不存在」（两种分隔符都要认）。"""
    gdb = tmp_path / "zones_arcpy.gdb"  # MAKE_ARCPY_FC 只会建这个库名
    made = arcpy_runner.run(
        MAKE_ARCPY_FC.replace("__GDB__", repr(str(gdb))).replace("__PARENT__", repr(str(tmp_path))),
        timeout=300,
    )
    assert made.ok, made.error
    _make_zones(gdal_runner, tmp_path)  # 参照物走 .gpkg，不经过路径解析
    lib = RecipeLibrary(code_runner=gdal_runner)

    expected = _buffer(gdal_runner, lib, tmp_path, "path_ref.gpkg")
    assert expected.ok, expected.summary
    ref = expected.data["result"]

    for sep, tag in (("\\", "backslash"), ("/", "slash")):
        obs = _buffer(gdal_runner, lib, tmp_path, f"path_{tag}.gpkg",
                      input_path=str(gdb) + sep + "zones")
        assert obs.ok, obs.summary
        payload = obs.data["result"]
        assert payload["feature_count"] == ref["feature_count"] == 1  # ALL = 三块合成一个 MultiPolygon
        assert abs(payload["total_area"] - ref["total_area"]) <= 1e-12 * ref["total_area"]

    # 要素数据集前缀：GDAL 只认裸图层名 → 给人话错误，不丢 pyogrio 的 DataLayerError
    obs = _buffer(gdal_runner, lib, tmp_path, "path_fd.gpkg",
                  input_path=str(gdb) + "\\fd\\zones")
    assert not obs.ok
    message = str(getattr(obs, "error", obs.summary))
    assert "没有图层" in message and "zones" in message and "DataLayerError" not in message


# --------------------------------------------------------------- 双端一致
def test_arcpy_and_gdal_agree_within_one_e_3(gdal_runner, arcpy_runner, tmp_path: Path):
    """同一份几何：arcpy（CodeRunner 走 arcgispro-py3）与 GDAL 的要素数相等、面积相对差 < 1e-3。"""
    gdal_zones = _make_zones(gdal_runner, tmp_path)
    gdb = tmp_path / "zones_arcpy.gdb"
    made = arcpy_runner.run(
        MAKE_ARCPY_FC.replace("__GDB__", repr(str(gdb))).replace("__PARENT__", repr(str(tmp_path))),
        timeout=300,
    )
    assert made.ok, made.error
    arcpy_input = made.result["fc"]
    # arcpy 的 Buffer 不会凭空建容器：输出库得先存在（实测写新建 .gpkg 报 cannot open）
    out_gdb = _prepare_arcpy_out_gdb(arcpy_runner, tmp_path)

    gdal_lib = RecipeLibrary(code_runner=gdal_runner)
    arcpy_lib = RecipeLibrary(code_runner=arcpy_runner)
    gdal_lib.load()
    arcpy_lib.load()

    evidence = {}
    for option, field in (("ALL", ""), ("NONE", "zone"), ("LIST", "zone")):
        gdal_obs = _buffer(
            gdal_runner, gdal_lib, tmp_path, f"gdal_{option}.gpkg",
            dissolve_type=option, dissolve_field=field,
        )
        arcpy_obs = _run(
            arcpy_lib, arcpy_runner, tmp_path,
            input_path=arcpy_input,
            output_path=str(out_gdb / f"buffer_{option}"),
            distance="500 Meters",
            dissolve_type=option,
            dissolve_field=field,
        )
        assert gdal_obs.ok, f"gdal/{option}: {gdal_obs.summary}"
        assert arcpy_obs.ok, f"arcpy/{option}: {arcpy_obs.summary}"

        gdal_payload = gdal_obs.data["result"]
        arcpy_payload = arcpy_obs.data["result"]
        evidence[option] = (arcpy_payload["feature_count"], gdal_payload["feature_count"],
                            arcpy_payload["total_area"], gdal_payload["total_area"])

        assert gdal_payload["dissolve_option"] == arcpy_payload["dissolve_option"] == option
        assert gdal_payload["feature_count"] == arcpy_payload["feature_count"], evidence[option]
        diff = abs(gdal_payload["total_area"] - arcpy_payload["total_area"]) / arcpy_payload["total_area"]
        assert diff < 1e-3, f"{option} 面积差 {diff:.4%}：{evidence[option]}"

    print("双端对比（option, arcpy_count, gdal_count, arcpy_area, gdal_area）:")
    for option, row in evidence.items():
        print(f"  {option}: {row[0]} / {row[1]} | {row[2]:.3f} / {row[3]:.3f}")


def test_arcpy_and_gdal_agree_on_null_dissolve_group(gdal_runner, arcpy_runner, tmp_path: Path):
    """分组字段带空值的双端对比：两边都留 NULL 组，要素数相等、面积相对差 < 1e-3。

    这就是“pandas 默认丢空值组”会咬人的地方：dropna=True 时 GDAL 侧只剩 2 条，
    arcpy 侧 3 条，双端直接对不上（DAIR-68 非阻塞项 3）。
    """
    _make_zones_with_null(gdal_runner, tmp_path)
    gdb = tmp_path / "zones_null_arcpy.gdb"
    made = arcpy_runner.run(
        MAKE_ARCPY_FC_WITH_NULL.replace("__GDB__", repr(str(gdb))).replace(
            "__PARENT__", repr(str(tmp_path))
        ),
        timeout=300,
    )
    assert made.ok, made.error
    out_gdb = _prepare_arcpy_out_gdb(arcpy_runner, tmp_path)

    gdal_lib = RecipeLibrary(code_runner=gdal_runner)
    arcpy_lib = RecipeLibrary(code_runner=arcpy_runner)

    gdal_obs = _buffer(
        gdal_runner, gdal_lib, tmp_path, "gdal_null.gpkg",
        dissolve_type="LIST", dissolve_field="zone",
        input_path=str(tmp_path / "zones_null.gpkg"),
    )
    arcpy_obs = _run(
        arcpy_lib, arcpy_runner, tmp_path,
        input_path=made.result["fc"],
        output_path=str(out_gdb / "buffer_null"),
        distance="500 Meters",
        dissolve_type="LIST",
        dissolve_field="zone",
    )

    assert gdal_obs.ok, gdal_obs.summary
    assert arcpy_obs.ok, arcpy_obs.summary
    gdal_payload = gdal_obs.data["result"]
    arcpy_payload = arcpy_obs.data["result"]
    assert gdal_payload["feature_count"] == arcpy_payload["feature_count"] == 3, (
        arcpy_payload["feature_count"],
        gdal_payload["feature_count"],
    )
    diff = abs(gdal_payload["total_area"] - arcpy_payload["total_area"]) / arcpy_payload["total_area"]
    assert diff < 1e-3, f"NULL 组面积差 {diff:.4%}：{arcpy_payload['total_area']} / {gdal_payload['total_area']}"

    # 两边都必须是“一个 NULL 组 + a + b”，而不只是要素数碰巧相等
    arcpy_nulls = arcpy_runner.run(
        "\n".join(
            [
                "import arcpy",
                f"vals = [r[0] for r in arcpy.da.SearchCursor({str(out_gdb / 'buffer_null')!r}, ['zone'])]",
                "set_result({'nulls': sum(1 for v in vals if v is None), 'count': len(vals)})",
            ]
        ),
        timeout=180,
    )
    assert arcpy_nulls.ok, arcpy_nulls.error
    assert arcpy_nulls.result["nulls"] == 1, arcpy_nulls.result
    gdal_values = gdal_runner.run(
        READ_ZONE_VALUES.replace("__PATH__", repr(gdal_payload["output"])).replace(
            "__FIELD__", repr("zone")
        ),
        timeout=120,
    )
    assert gdal_values.ok, gdal_values.error
    assert gdal_values.result["nulls"] == 1, gdal_values.result
    print(
        f"NULL 组双端：arcpy={arcpy_payload['feature_count']} gdal={gdal_payload['feature_count']} "
        f"| {arcpy_payload['total_area']:.3f} / {gdal_payload['total_area']:.3f}"
    )

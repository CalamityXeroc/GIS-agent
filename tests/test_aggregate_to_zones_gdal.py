# -*- coding: utf-8 -*-
"""aggregate_to_zones 的 GDAL 版真跑 + 双端一致（SUM <1%、SRC_COUNT 严格相等）。

输入是现场生成的**投影坐标系**面（EPSG:32650，米）：分区是三个 10 km × 10 km 的方块，
源要素是五个小面，其中两个**跨分区边界**，归属只能靠中心点判：

    CityA   x ∈ [0, 10000)         CityB   x ∈ [10000, 20000)      CityC x ∈ [20000, 30000)
    s1  (1000..4000)   中心 2500  → A
    s2  (7000..12000)  中心 9500  → A   ← 跨边界：面积有 1/4 落进 B
    s3  (9000..13000)  中心 11000 → B   ← 跨边界：面积有 1/4 落在 A
    s4  (11000..14000) 中心 12500 → B，Y2011 为空
    s5  (50000..51000) 在所有分区之外

所以 ``SRC_COUNT`` / ``SUM_*`` 不只能“有值”，还能逐分区对数；``HAVE_THEIR_CENTER_IN``
是否真按中心点（而不是按相交/重叠面积）也由 s2/s3 卡住 —— 用 INTERSECT 时 s3 会落到 A。

缺环境就 skip（与 ``test_buffer_dissolve_gdal.py`` 同一套口径）：
``.venv-gdal`` 不在 → 所有真 GDAL 用例 skip；ArcGIS Pro 找不到 → 只有双端用例 skip。
CI 机器两者都没有，by design。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gis_cli.recipes.gdal_checks import gdal_vector_check
from gis_cli.recipes.schema import Recipe
from gis_cli.recipes.store import RecipeLibrary
from gis_cli.runner.gdal_runner import GdalCodeRunner, default_python_executable
from gis_cli.runtime.code_runner import CodeRunner

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "src/gis_cli/recipes/builtin/aggregate_to_zones.yaml"

#: 分区面（三个 10 km 方块，x 依次相接）
ZONE_COORDS = [
    ("CityA", 11, [(0, 0), (10000, 0), (10000, 10000), (0, 10000)]),
    ("CityB", 22, [(10000, 0), (20000, 0), (20000, 10000), (10000, 10000)]),
    ("CityC", 33, [(20000, 0), (30000, 0), (30000, 10000), (20000, 10000)]),
]

#: 源要素（id, Y2011, Y2012, 坐标）；s2/s3 跨边界，s4 的 Y2011 为空，s5 在所有分区之外
SOURCE_COORDS = [
    ("s1", 100.0, 10.0, [(1000, 1000), (4000, 1000), (4000, 4000), (1000, 4000)]),
    ("s2", 200.0, 20.0, [(7000, 1000), (12000, 1000), (12000, 4000), (7000, 4000)]),
    ("s3", 500.0, 30.0, [(9000, 1000), (13000, 1000), (13000, 4000), (9000, 4000)]),
    ("s4", None, 40.0, [(11000, 5000), (14000, 5000), (14000, 8000), (11000, 8000)]),
    ("s5", 999.0, 99.0, [(50000, 50000), (51000, 50000), (51000, 51000), (50000, 51000)]),
]

#: 中心点归属下每个分区的期望值（arcpy 与 GDAL 都必须落在这组数上）
EXPECTED = {
    "CityA": {"SUM_Y2011": 300.0, "SUM_Y2012": 30.0, "SRC_COUNT": 2},
    # s4 的 Y2011 为空：SRC_COUNT 数的是非空值（arcpy Statistics 的 COUNT 口径）
    "CityB": {"SUM_Y2011": 500.0, "SUM_Y2012": 70.0, "SRC_COUNT": 1},
    "CityC": {"SUM_Y2011": None, "SUM_Y2012": None, "SRC_COUNT": None},  # 没有源要素落入
}

_MAKE_GPKG = "\n".join(
    [
        "import geopandas as gpd",
        "from shapely.geometry import Polygon",
        f"_zones = {ZONE_COORDS!r}",
        f"_sources = {SOURCE_COORDS!r}",
        "zones = gpd.GeoDataFrame({'name': [z[0] for z in _zones],",
        "                          'pop_base': [z[1] for z in _zones]},",
        "                         geometry=[Polygon(z[2]) for z in _zones], crs='EPSG:32650')",
        "zones.to_file(__ZONES__, driver='GPKG', engine='pyogrio', layer='zones')",
        "src = gpd.GeoDataFrame({'id': [s[0] for s in _sources],",
        "                        'Y2011': [s[1] for s in _sources],",
        "                        'Y2012': [s[2] for s in _sources]},",
        "                       geometry=[Polygon(s[3]) for s in _sources], crs='EPSG:32650')",
        "src.to_file(__SOURCE__, driver='GPKG', engine='pyogrio', layer='source')",
        "set_result({'ok': True})",
    ]
)

_MAKE_ONLY_NULL = "\n".join(
    [
        "import geopandas as gpd",
        "from shapely.geometry import Polygon",
        "gdf = gpd.GeoDataFrame({'id': ['n1'], 'Y2011': [float('nan')], 'Y2012': [5.0]},",
        "    geometry=[Polygon([(11000, 1000), (14000, 1000), (14000, 4000), (11000, 4000)])],",
        "    crs='EPSG:32650')",
        "gdf.to_file(__PATH__, driver='GPKG', engine='pyogrio', layer='source')",
        "set_result({'ok': True})",
    ]
)

_MAKE_GEOGRAPHIC = "\n".join(
    [
        "import geopandas as gpd",
        "from shapely.geometry import Polygon",
        "gdf = gpd.GeoDataFrame({'id': ['g1'], 'Y2011': [1.0]},",
        "                       geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],",
        "                       crs='EPSG:4490')",
        "gdf.to_file(__PATH__, driver='GPKG', engine='pyogrio', layer='source_geo')",
        "set_result({'ok': True})",
    ]
)

#: arcpy 侧输入：真 .gdb 要素类（GDAL 版从同一份 gdb 读，双端就是“同一输入”）
_MAKE_ARCPY_INPUTS = "\n".join(
    [
        "import arcpy",
        "arcpy.env.overwriteOutput = True",
        "sr = arcpy.SpatialReference(32650)",
        "gdb = __GDB__",
        "if not arcpy.Exists(gdb):",
        "    arcpy.management.CreateFileGDB(__PARENT__, 'inputs.gdb')",
        "for name in ('zones', 'source'):",
        "    if arcpy.Exists(gdb + '\\\\' + name):",
        "        arcpy.management.Delete(gdb + '\\\\' + name)",
        "arcpy.management.CreateFeatureclass(gdb, 'zones', 'POLYGON', spatial_reference=sr)",
        "arcpy.management.AddField(gdb + '\\\\zones', 'name', 'TEXT', field_length=50)",
        "arcpy.management.AddField(gdb + '\\\\zones', 'pop_base', 'LONG')",
        f"for nm, pop, pts in {ZONE_COORDS!r}:",
        "    with arcpy.da.InsertCursor(gdb + '\\\\zones', ['SHAPE@', 'name', 'pop_base']) as cur:",
        "        cur.insertRow([arcpy.Polygon(arcpy.Array([arcpy.Point(*p) for p in pts]), sr), nm, pop])",
        "arcpy.management.CreateFeatureclass(gdb, 'source', 'POLYGON', spatial_reference=sr)",
        "arcpy.management.AddField(gdb + '\\\\source', 'id', 'TEXT', field_length=20)",
        "for f in ('Y2011', 'Y2012'):",
        "    arcpy.management.AddField(gdb + '\\\\source', f, 'DOUBLE')",
        f"for sid, y1, y2, pts in {SOURCE_COORDS!r}:",
        "    with arcpy.da.InsertCursor(gdb + '\\\\source', ['SHAPE@', 'id', 'Y2011', 'Y2012']) as cur:",
        "        cur.insertRow([arcpy.Polygon(arcpy.Array([arcpy.Point(*p) for p in pts]), sr), sid, y1, y2])",
        "set_result({'zones': gdb + '\\\\zones', 'source': gdb + '\\\\source'})",
    ]
)

_READ_GPKG = "\n".join(
    [
        "import geopandas as gpd",
        "gdf = gpd.read_file(__PATH__, engine='pyogrio')",
        "cols = [c for c in gdf.columns if c != 'geometry']",
        "rows = gdf[cols].itertuples(index=False, name=None)",
        "set_result({'columns': cols,",
        "            'rows': [[None if v is None or (isinstance(v, float) and v != v) else v",
        "                      for v in row] for row in rows]})",
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
    return recipe_lib.run("aggregate_to_zones", params, ctx)


def _make_gpkg(runner, tmp_path: Path) -> tuple[Path, Path]:
    zones, source = tmp_path / "zones.gpkg", tmp_path / "source.gpkg"
    made = runner.run(
        _MAKE_GPKG.replace("__ZONES__", repr(str(zones))).replace("__SOURCE__", repr(str(source))),
        timeout=180,
    )
    assert made.ok, made.error
    return zones, source


def _aggregate(runner, tmp_path: Path, name: str, **overrides):
    params = {
        "zones_path": str(tmp_path / "zones.gpkg"),
        "source_path": str(tmp_path / "source.gpkg"),
        "output_path": str(tmp_path / name),
        "stat_fields": "Y2011,Y2012",
        "stat_type": "SUM",
        "match_option": "HAVE_THEIR_CENTER_IN",
    }
    params.update(overrides)
    lib = RecipeLibrary(code_runner=runner)
    return _run(lib, runner, tmp_path, **params)


def _read_table(runner, path: str, *, arcpy: bool) -> dict[str, dict[str, object]]:
    code = (_READ_ARCPY if arcpy else _READ_GPKG).replace("__PATH__", repr(str(path)))
    outcome = runner.run(code, timeout=180)
    assert outcome.ok, outcome.error
    columns = outcome.result["columns"]
    table = {}
    for row in outcome.result["rows"]:
        record = dict(zip(columns, row))
        table[str(record["name"])] = record
    return table


def _relative_diff(actual, expected) -> float | None:
    """两边都空 → None（视为一致）；一边空一边有值 → inf（一致不了）。"""
    if actual is None or expected is None:
        return None if actual is None and expected is None else float("inf")
    a, e = float(actual), float(expected)
    if e == 0:
        return 0.0 if abs(a) < 1e-9 else float("inf")
    return abs(a - e) / abs(e)


def _same_count(actual, expected) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    return int(actual) == int(expected)


# ------------------------------------------------------------- 模板本身
def test_recipe_declares_two_backends_and_renders_both():
    recipe = Recipe.from_yaml(RECIPE)
    assert recipe.backend == ["arcpy", "gdal"]
    params = {
        "zones_path": "z.gdb/city",
        "source_path": "s.gdb/county",
        "output_path": "out/agg.gpkg",
        "stat_fields": "Y2011,Y2012",
    }
    arcpy_code = recipe.render(params)
    gdal_code = recipe.render(params, backend="gdal")

    # arcpy 版一个字都没动（快照测试盯着的就是这一段）
    assert "arcpy.analysis.SpatialJoin" in arcpy_code and "import geopandas" not in arcpy_code
    assert "gpd.read_file" in gdal_code and "import arcpy" not in gdal_code
    assert 'driver="GPKG"' in gdal_code
    assert "HAVE_THEIR_CENTER_IN" in gdal_code  # 中心点归属是默认语义，代码里必须显式在
    for key in ("zones_path", "output_path", "Y2011,Y2012"):
        assert key in arcpy_code and key in gdal_code


# ----------------------------------------------------------- GDAL 侧真跑
def test_gdal_sum_by_center_in_writes_expected_values(gdal_runner, tmp_path: Path):
    _make_gpkg(gdal_runner, tmp_path)
    obs = _aggregate(gdal_runner, tmp_path, "out_sum.gpkg")

    assert obs.ok, obs.summary
    payload = obs.data["result"]
    assert payload["feature_count"] == 3
    assert payload["zone_key"] == "name"
    assert payload["stat_fields"] == {"SUM": ["SUM_Y2011", "SUM_Y2012"]}

    table = _read_table(gdal_runner, payload["output"], arcpy=False)
    for zone, expected in EXPECTED.items():
        for field, value in expected.items():
            assert _relative_diff(table[zone][field], value) in (None, 0.0), (zone, field, table[zone])

    # 产物必须是真 GPKG（不是“文件存在”就算过）
    info = gdal_runner.run(
        "\n".join(
            [
                "import pyogrio",
                f"i = pyogrio.read_info({payload['output']!r})",
                "set_result({'driver': i.get('driver'), 'features': int(i['features']),",
                "            'crs': str(i.get('crs') or '')})",
            ]
        ),
        timeout=120,
    )
    assert info.ok, info.error
    assert info.result["driver"] == "GPKG"
    assert info.result["features"] == 3
    assert "32650" in info.result["crs"]

    # 4 条断言都是 GDAL 口径真判过的（fields_present / field_sum_gt 不能落到 arcpy 兜底）
    assertions = {a["kind"]: a for a in obs.data["assertions"]}
    assert assertions["feature_count_gt"]["detail"] == "count=3"
    assert assertions["fields_present"]["ok"] is True
    assert assertions["fields_present"]["detail"] == "missing=[]"
    assert assertions["field_sum_gt"]["ok"] is True
    assert assertions["field_sum_gt"]["detail"] == "sum(SRC_COUNT)=3.0 expected>0"


def test_gdal_center_in_counts_crossing_source_once_on_the_right_zone(gdal_runner, tmp_path: Path):
    """s3 跨 A/B 边界、中心在 B：只算进 B，且不被 A、B 重复计入。"""
    _make_gpkg(gdal_runner, tmp_path)
    obs = _aggregate(gdal_runner, tmp_path, "out_center.gpkg")

    assert obs.ok, obs.summary
    table = _read_table(gdal_runner, obs.data["result"]["output"], arcpy=False)
    assert table["CityA"]["SUM_Y2011"] == 300.0, "s2（中心在 A）应归 A"
    assert table["CityB"]["SUM_Y2011"] == 500.0, "s3（中心在 B、跨边界）应归 B"
    counts = [table[z]["SRC_COUNT"] for z in ("CityA", "CityB", "CityC")]
    assert counts == [2, 1, None], f"跨边界要素只算一次（不能被重复计）: {counts}"

    # 换成 INTERSECT 结果就必须不同：s2/s3 都跨边界，会按相交归到 A
    inter = _aggregate(gdal_runner, tmp_path, "out_intersect.gpkg", match_option="INTERSECT")
    assert inter.ok, inter.summary
    inter_table = _read_table(gdal_runner, inter.data["result"]["output"], arcpy=False)
    assert inter_table["CityA"]["SUM_Y2011"] == 800.0, inter_table["CityA"]
    assert inter_table["CityB"]["SUM_Y2011"] is None, inter_table["CityB"]
    assert inter_table["CityA"] != table["CityA"], "center-in 与 intersect 必须能区分开"


def test_gdal_stat_types_mean_max_min_count(gdal_runner, tmp_path: Path):
    _make_gpkg(gdal_runner, tmp_path)

    expected = {
        "MEAN": {"CityA": 150.0, "CityB": 500.0, "CityC": None},
        "MAX": {"CityA": 200.0, "CityB": 500.0, "CityC": None},
        "MIN": {"CityA": 100.0, "CityB": 500.0, "CityC": None},
        "COUNT": {"CityA": 2.0, "CityB": 1.0, "CityC": None},
    }
    for stat, zones in expected.items():
        obs = _aggregate(gdal_runner, tmp_path, f"out_{stat}.gpkg", stat_type=stat)
        assert obs.ok, f"{stat}: {obs.summary}"
        payload = obs.data["result"]
        assert payload["stat_fields"] == {stat: [f"{stat}_Y2011", f"{stat}_Y2012"]}
        table = _read_table(gdal_runner, payload["output"], arcpy=False)
        for zone, value in zones.items():
            assert _relative_diff(table[zone][f"{stat}_Y2011"], value) in (None, 0.0), (
                stat,
                zone,
                table[zone],
            )
        # SRC_COUNT 与统计类型无关：数的都是 stat_fields[0] 的非空值个数
        assert table["CityA"]["SRC_COUNT"] == 2
        assert table["CityB"]["SRC_COUNT"] == 1
        assert table["CityC"]["SRC_COUNT"] is None


def test_gdal_keeps_null_semantics(gdal_runner, tmp_path: Path):
    """Y2011 全空的分区：SUM 留空、COUNT 是 0；没有源要素落入的分区整行是空。"""
    _make_gpkg(gdal_runner, tmp_path)
    only_null = tmp_path / "source_only_null.gpkg"
    made = gdal_runner.run(_MAKE_ONLY_NULL.replace("__PATH__", repr(str(only_null))), timeout=180)
    assert made.ok, made.error

    obs = _aggregate(
        gdal_runner, tmp_path, "out_null.gpkg", source_path=str(only_null)
    )
    # 这次 SRC_COUNT 合计是 0，field_sum_gt(SRC_COUNT > 0) 必然不过：
    # 断言红是配方自己的口径，配方本身要“执行成功”。
    assert "执行成功" in obs.summary, obs.summary
    table = _read_table(gdal_runner, obs.data["result"]["output"], arcpy=False)
    assert table["CityB"]["SUM_Y2011"] is None, "全空值的 SUM 必须是空值，不能是 0"
    assert table["CityB"]["SUM_Y2012"] == 5.0
    assert table["CityB"]["SRC_COUNT"] == 0, "落入了要素但值全空 → COUNT 是 0（arcpy 口径）"
    assert table["CityC"]["SRC_COUNT"] is None, "没有源要素落入的分区是空值"


# --------------------------------------------------------------- 拒绝分支
def test_gdal_rejects_crs_mismatch(gdal_runner, tmp_path: Path):
    _make_gpkg(gdal_runner, tmp_path)
    geographic = tmp_path / "source_geo.gpkg"
    made = gdal_runner.run(_MAKE_GEOGRAPHIC.replace("__PATH__", repr(str(geographic))), timeout=180)
    assert made.ok, made.error

    obs = _aggregate(gdal_runner, tmp_path, "out_crs.gpkg", source_path=str(geographic))

    assert not obs.ok
    assert "坐标系不一致" in obs.summary
    assert not (tmp_path / "out_crs.gpkg").exists(), "报错时不应留下半成品"


def test_gdal_rejects_non_numeric_stat_field(gdal_runner, tmp_path: Path):
    _make_gpkg(gdal_runner, tmp_path)
    obs = _aggregate(gdal_runner, tmp_path, "out_text.gpkg", stat_fields="Y2011,id")

    assert not obs.ok
    assert "非数值字段" in obs.summary and "id" in obs.summary
    assert not (tmp_path / "out_text.gpkg").exists()


def test_gdal_rejects_unknown_stat_type_and_match_option(gdal_runner, tmp_path: Path):
    _make_gpkg(gdal_runner, tmp_path)

    bad_stat = _aggregate(gdal_runner, tmp_path, "out_stat.gpkg", stat_type="MEDIAN")
    assert not bad_stat.ok
    assert "stat_type 只支持" in bad_stat.summary

    bad_match = _aggregate(gdal_runner, tmp_path, "out_match.gpkg", match_option="CLOSEST")
    assert not bad_match.ok
    assert "match_option 不支持" in bad_match.summary


def test_gdal_rejects_non_gpkg_output(gdal_runner, tmp_path: Path):
    _make_gpkg(gdal_runner, tmp_path)
    obs = _aggregate(gdal_runner, tmp_path, "unused", output_path=str(tmp_path / "out.gdb" / "agg"))

    assert not obs.ok
    assert ".gpkg" in obs.summary


# --------------------------------------------------------------- 双端一致
def _arcpy_inputs(arcpy_runner, tmp_path: Path) -> tuple[str, str, Path]:
    gdb = tmp_path / "inputs.gdb"
    made = arcpy_runner.run(
        _MAKE_ARCPY_INPUTS.replace("__GDB__", repr(str(gdb))).replace(
            "__PARENT__", repr(str(tmp_path))
        ),
        timeout=300,
    )
    assert made.ok, made.error
    return made.result["zones"], made.result["source"], gdb


def test_arcpy_and_gdal_agree_on_the_same_gdb_input(gdal_runner, arcpy_runner, tmp_path: Path):
    """同一份 .gdb 要素类：要素数相等、每个 SUM_* 合计差 <1%、SRC_COUNT 严格相等。

    GDAL 侧直接读 arcpy 建出来的 gdb 要素类（``xxx.gdb\\图层``），因此输入是**同一份数据**，
    不是“两边各建一份坐标相同的数据”。
    """
    zones_fc, source_fc, gdb = _arcpy_inputs(arcpy_runner, tmp_path)
    arcpy_lib = RecipeLibrary(code_runner=arcpy_runner)
    gdal_lib = RecipeLibrary(code_runner=gdal_runner)
    arcpy_lib.load()
    gdal_lib.load()

    evidence: dict[tuple[str, str], tuple] = {}
    for stat in ("SUM", "MEAN", "MAX", "MIN"):
        gdal_obs = _run(
            gdal_lib,
            gdal_runner,
            tmp_path,
            zones_path=zones_fc,
            source_path=source_fc,
            output_path=str(tmp_path / f"gdal_{stat}.gpkg"),
            stat_fields="Y2011,Y2012",
            stat_type=stat,
            match_option="HAVE_THEIR_CENTER_IN",
        )
        arcpy_obs = _run(
            arcpy_lib,
            arcpy_runner,
            tmp_path,
            zones_path=zones_fc,
            source_path=source_fc,
            output_path=str(gdb / f"arcpy_{stat}"),
            stat_fields="Y2011,Y2012",
            stat_type=stat,
            match_option="HAVE_THEIR_CENTER_IN",
        )
        assert gdal_obs.ok, f"gdal/{stat}: {gdal_obs.summary}"
        assert arcpy_obs.ok, f"arcpy/{stat}: {arcpy_obs.summary}"
        assert (
            gdal_obs.data["result"]["feature_count"]
            == arcpy_obs.data["result"]["feature_count"]
        ), f"{stat}: 分区要素数不一致"

        gdal_table = _read_table(gdal_runner, gdal_obs.data["result"]["output"], arcpy=False)
        arcpy_table = _read_table(arcpy_runner, arcpy_obs.data["result"]["output"], arcpy=True)

        for zone in ("CityA", "CityB", "CityC"):
            assert set(gdal_table[zone]) >= {"name", "pop_base"}  # 原分区属性要带过来
            for suffix in ("Y2011", "Y2012"):
                field = f"{stat}_{suffix}"
                diff = _relative_diff(gdal_table[zone][field], arcpy_table[zone][field])
                evidence[(f"{stat}_{suffix}", zone)] = (
                    arcpy_table[zone][field],
                    gdal_table[zone][field],
                    diff,
                )
                assert diff is None or diff < 0.01, f"{field}/{zone} 差 {diff:.4%}"
            # SRC_COUNT 是计数：必须严格相等，不给容差
            assert _same_count(gdal_table[zone]["SRC_COUNT"], arcpy_table[zone]["SRC_COUNT"]), (
                stat,
                zone,
                gdal_table[zone]["SRC_COUNT"],
                arcpy_table[zone]["SRC_COUNT"],
            )

        # 逐字段的**合计**也要对得上（SUM 是验收硬指标，其余一并看住）
        arcpy_counts = [arcpy_table[z]["SRC_COUNT"] for z in ("CityA", "CityB", "CityC")]
        gdal_counts = [gdal_table[z]["SRC_COUNT"] for z in ("CityA", "CityB", "CityC")]
        arcpy_total_count = sum(int(v) for v in arcpy_counts if v is not None)
        gdal_total_count = sum(int(v) for v in gdal_counts if v is not None)
        evidence[("SRC_COUNT+合计", "ALL")] = (arcpy_total_count, gdal_total_count, 0.0)
        assert gdal_total_count == arcpy_total_count, (
            f"{stat}: SRC_COUNT 合计 {gdal_total_count} != {arcpy_total_count}（计数必须严格相等）"
        )

        for suffix in ("Y2011", "Y2012"):
            field = f"{stat}_{suffix}"
            arcpy_total = sum(
                float(arcpy_table[z][field]) for z in ("CityA", "CityB", "CityC")
                if arcpy_table[z][field] is not None
            )
            gdal_total = sum(
                float(gdal_table[z][field]) for z in ("CityA", "CityB", "CityC")
                if gdal_table[z][field] is not None
            )
            diff = _relative_diff(gdal_total, arcpy_total)
            evidence[(f"{field}+合计", "ALL")] = (arcpy_total, gdal_total, diff)
            assert diff is None or diff < 0.01, f"{field} 合计差 {diff:.4%}"

    print("双端对比（字段, 分区, arcpy, gdal, 相对差）:")
    for (field, zone), (arcpy_value, gdal_value, diff) in sorted(evidence.items()):
        shown = "None" if diff is None else f"{diff:.4%}"
        print(f"  {field:14s} {zone}: {arcpy_value} / {gdal_value}  diff={shown}")


def test_arcpy_and_gdal_agree_on_crossing_sources(gdal_runner, arcpy_runner, tmp_path: Path):
    """跨边界源要素（s2 中心在 A、s3 中心在 B）两端的归属必须一致。"""
    zones_fc, source_fc, gdb = _arcpy_inputs(arcpy_runner, tmp_path)
    arcpy_lib = RecipeLibrary(code_runner=arcpy_runner)
    gdal_lib = RecipeLibrary(code_runner=gdal_runner)

    gdal_obs = _run(
        gdal_lib,
        gdal_runner,
        tmp_path,
        zones_path=zones_fc,
        source_path=source_fc,
        output_path=str(tmp_path / "gdal_cross.gpkg"),
        stat_fields="Y2011",
        match_option="HAVE_THEIR_CENTER_IN",
    )
    arcpy_obs = _run(
        arcpy_lib,
        arcpy_runner,
        tmp_path,
        zones_path=zones_fc,
        source_path=source_fc,
        output_path=str(gdb / "arcpy_cross"),
        stat_fields="Y2011",
        match_option="HAVE_THEIR_CENTER_IN",
    )

    assert gdal_obs.ok, gdal_obs.summary
    assert arcpy_obs.ok, arcpy_obs.summary
    gdal_table = _read_table(gdal_runner, gdal_obs.data["result"]["output"], arcpy=False)
    arcpy_table = _read_table(arcpy_runner, arcpy_obs.data["result"]["output"], arcpy=True)

    for zone in ("CityA", "CityB", "CityC"):
        assert _relative_diff(gdal_table[zone]["SUM_Y2011"], arcpy_table[zone]["SUM_Y2011"]) in (
            None,
            0.0,
        ), (zone, gdal_table[zone], arcpy_table[zone])
        assert _same_count(gdal_table[zone]["SRC_COUNT"], arcpy_table[zone]["SRC_COUNT"]), zone

    assert gdal_table["CityA"]["SUM_Y2011"] == 300.0
    assert gdal_table["CityB"]["SUM_Y2011"] == 500.0
    print(
        "跨边界双端：arcpy="
        f"{[arcpy_table[z]['SUM_Y2011'] for z in ('CityA', 'CityB')]} "
        f"gdal={[gdal_table[z]['SUM_Y2011'] for z in ('CityA', 'CityB')]}"
    )


# --------------------------------------------------------- 断言侧的 GDAL 口径
class _FakeGdalRunner:
    """假 GDAL 执行器：只回固定 payload，用来验断言口径（不需要真环境）。"""

    backends = {"gdal"}

    def __init__(self, payload: dict):
        self.payload = payload
        self.codes: list[str] = []

    def run(self, code, *, timeout=None, workspace=None):
        self.codes.append(code)
        return SimpleNamespace(ok=True, stdout="", result=self.payload, error=None)


def test_gdal_fields_present_reads_fields_with_pyogrio():
    runner = _FakeGdalRunner(
        {"exists": True, "feature_count": 3, "fields": ["name", "SUM_Y2011", "SRC_COUNT"]}
    )
    ok, detail = gdal_vector_check(runner, "fields_present", "out.gpkg", ["SRC_COUNT"], {})
    assert ok is True and detail == "missing=[]"
    assert "pyogrio" in runner.codes[0] and "arcpy" not in runner.codes[0]

    ok, detail = gdal_vector_check(
        runner, "fields_present", "out.gpkg", ["SRC_COUNT", "NOPE"], {}
    )
    assert ok is False and detail == "missing=['NOPE']"


def test_gdal_field_sum_gt_sums_the_column_with_pyogrio():
    runner = _FakeGdalRunner({"exists": True, "found": True, "sum": 12.0})
    ok, detail = gdal_vector_check(runner, "field_sum_gt", "out.gpkg", 10, {"field": "SRC_COUNT"})
    assert ok is True
    assert detail == "sum(SRC_COUNT)=12.0 expected>10"
    assert "pyogrio" in runner.codes[0] and "arcpy" not in runner.codes[0]

    ok, _ = gdal_vector_check(runner, "field_sum_gt", "out.gpkg", 12, {"field": "SRC_COUNT"})
    assert ok is False, "相等不算大于"


def test_gdal_field_sum_gt_reports_missing_field_and_missing_dataset():
    runner = _FakeGdalRunner(
        {"exists": True, "found": False, "reason": "字段不存在", "fields": ["a"]}
    )
    ok, detail = gdal_vector_check(runner, "field_sum_gt", "out.gpkg", 0, {"field": "NOPE"})
    assert ok is False and "字段不存在" in detail and "['a']" in detail

    missing = _FakeGdalRunner({"exists": False, "found": False, "reason": "文件不存在"})
    ok, detail = gdal_vector_check(missing, "field_sum_gt", "out.gpkg", 0, {"field": "a"})
    assert ok is False and "不存在" in detail

    # 没给字段名 → 明确报错，不去猜
    ok, detail = gdal_vector_check(runner, "field_sum_gt", "out.gpkg", 0, {})
    assert ok is False and "field" in detail


def test_gdal_vector_check_still_ignores_unrelated_kinds():
    runner = _FakeGdalRunner({"exists": True, "feature_count": 1})
    assert gdal_vector_check(runner, "file_exists", "x.txt", None) is None
    # crs_wkid 已由 GDAL 侧接管（DAIR-129）；这里换一个仍然与后端无关的 kind
    assert gdal_vector_check(runner, "bbox_between", "x.gpkg", {"xmin": [0, 1]}) is None
    assert not runner.codes

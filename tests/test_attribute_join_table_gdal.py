# -*- coding: utf-8 -*-
"""attribute_join_table 的 GDAL 版真跑 + 双端逐值一致（行数/行序/每个字段值）。

它是「社区适老化评估」链的最后一环（``zonal_statistics`` 出统计表 → 按 OBJECTID 挂回
社区面），前七批把上游都打通了，这一步以前是 GDAL-only 机器上唯一的断点。

**arcpy 的 JoinField 行为全部是 Pro 3.6 实测出来的**（探针见各用例注释），GDAL 侧逐条对齐：

| 情形 | arcpy 实测 | GDAL 版 |
|---|---|---|
| 一对多（表里同键多行） | 行数不变，取表里**第一条** | 同样取第一条 |
| 字段重名 | 加 `_1`（`name`→`name_1`），原字段不动 | 同样加 `_1`、再撞加 `_2` |
| 空 join_fields | 挂接表全部字段（OID/几何除外），**连 join_key 也挂进来**（`code_1`） | 同 |
| 键不匹配 | 行保留、挂上的字段为空 | 同 |
| 文本键大小写 | `"A"` 不匹配 `"a"`（大小写敏感） | 同（pandas 本来就这样） |
| 文本键尾空格 | `"a  "` **能**匹配 `"a"`（ArcGIS 字符串比较忽略尾空格） | 匹配前 rstrip |
| 数值键 int vs double | `1` 能匹配 `1.0`，不能匹配 `1.5` | 同为数值时统一成 float 再比 |
| 文本键 vs 数值键 | **不报错**，静默匹配 0 行、字段全空 | **报人话错误**（有意分歧，见 pitfalls） |
| `fields` 里写了不存在的字段 | **静默忽略**（等于什么都没挂上） | 报人话错误（有意分歧） |
| 目标已有 `name` 与 `name_1` | 新字段叫 `name_12`（它的去重是「加后缀直到唯一」的怪实现） | `name_2`（有意分歧） |
| 索引字段 | 不需要 AddIndex（2000 行：1.769s vs 1.783s，无差别） | 无对应物，pandas 哈希 join |
| OID | 是**真字段**，但真名随容器变（.gdb=`OBJECTID`、.shp=`FID`、.gpkg=`fid`） | 按 read_info 的 `fid_column` 认，不写死 OBJECTID |

缺环境就 skip（与 ``test_aggregate_to_zones_gdal.py`` 同一套口径）：
``.venv-gdal`` 不在 → 所有真 GDAL 用例 skip；ArcGIS Pro 找不到 → 只有双端用例 skip。
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gis_cli.recipes.schema import Recipe
from gis_cli.recipes.store import RecipeLibrary
from gis_cli.runner.gdal_runner import GdalCodeRunner, default_python_executable
from gis_cli.runtime.code_runner import CodeRunner

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "src/gis_cli/recipes/builtin/attribute_join_table.yaml"

#: 目标面：4 条（code = a/b/c/d，name 与挂接表**同名字段**，用来卡重名后缀；
#: pop_base_long_name 超过 10 字符，用来卡 .shp 输出时字段名被截断后不能算成「新增字段」）
TARGET_ROWS = [
    ("a", "A-name", 1),
    ("b", "B-name", 2),
    ("c", "C-name", 3),
    ("d", "D-name", 4),
]

#: 挂接表：a 出现两次（一对多，arcpy 取第一条 10）、b 一次、zzz 挂不上任何目标行
STATS_ROWS = [
    ("a", 10, "J-A"),
    ("a", 11, "J-A2"),
    ("b", 20, "J-B"),
    ("zzz", 99, "J-Z"),
]

#: 双端期望（按目标行序）：c/d 挂不上 → 空；a 取第一条 10
EXPECTED_ROWS = [
    ["a", "a", 10, "J-A"],
    ["b", "b", 20, "J-B"],
    ["c", None, None, None],
    ["d", None, None, None],
]
EXPECTED_ADDED = ["code_1", "pop", "name_1"]

_MAKE_GPKG = "\n".join(
    [
        "import geopandas as gpd",
        "import pandas as pd",
        "import pyogrio",
        "from shapely.geometry import Polygon",
        "zones = gpd.GeoDataFrame({'code': [r[0] for r in __TARGET__],",
        "                          'name': [r[1] for r in __TARGET__],",
        "                          'pop_base_long_name': [r[2] for r in __TARGET__]},",
        "                         geometry=[Polygon([(i*100, 0), (i*100+50, 0), (i*100+50, 50), (i*100, 50)])",
        "                                   for i in range(len(__TARGET__))], crs='EPSG:32650')",
        "zones.to_file(__TARGET_PATH__, driver='GPKG', engine='pyogrio', layer='zones')",
        "stats = pd.DataFrame({'code': [r[0] for r in __STATS__],",
        "                      'pop': pd.array([r[1] for r in __STATS__], dtype='Int32'),",
        "                      'name': [r[2] for r in __STATS__]})",
        "pyogrio.write_dataframe(stats, __STATS_PATH__, driver='GPKG', layer='stats')",
        "set_result({'ok': True})",
    ]
)

#: arcpy 侧输入：.gdb 要素类 + .gdb 表（双端比的是「同一份逻辑数据」）
_MAKE_ARCPY_INPUTS = "\n".join(
    [
        "import arcpy, os",
        "arcpy.env.overwriteOutput = True",
        "sr = arcpy.SpatialReference(32650)",
        "gdb = __GDB__",
        "if not arcpy.Exists(gdb):",
        "    arcpy.management.CreateFileGDB(__PARENT__, os.path.basename(gdb))",
        "fc = os.path.join(gdb, 'zones')",
        "tbl = os.path.join(gdb, 'stats')",
        "for p in (fc, tbl):",
        "    if arcpy.Exists(p):",
        "        arcpy.management.Delete(p)",
        "arcpy.management.CreateFeatureclass(gdb, 'zones', 'POLYGON', spatial_reference=sr)",
        "for f, t in (('code', 'TEXT'), ('name', 'TEXT'), ('pop_base_long_name', 'LONG')):",
        "    arcpy.management.AddField(fc, f, t, field_length=50 if t == 'TEXT' else None)",
        "for i, (code, name, base) in enumerate(__TARGET__):",
        "    pts = [(i*100, 0), (i*100+50, 0), (i*100+50, 50), (i*100, 50)]",
        "    with arcpy.da.InsertCursor(fc, ['SHAPE@', 'code', 'name', 'pop_base_long_name']) as cur:",
        "        cur.insertRow([arcpy.Polygon(arcpy.Array([arcpy.Point(*p) for p in pts]), sr), code, name, base])",
        "arcpy.management.CreateTable(gdb, 'stats')",
        "arcpy.management.AddField(tbl, 'code', 'TEXT', field_length=50)",
        "arcpy.management.AddField(tbl, 'pop', 'LONG')",
        "arcpy.management.AddField(tbl, 'name', 'TEXT', field_length=50)",
        "for row in __STATS__:",
        "    with arcpy.da.InsertCursor(tbl, ['code', 'pop', 'name']) as cur:",
        "        cur.insertRow(list(row))",
        "set_result({'fc': fc, 'table': tbl})",
    ]
)

_READ_GPKG = "\n".join(
    [
        "import geopandas as gpd",
        "gdf = gpd.read_file(__PATH__, layer=__LAYER__, engine='pyogrio')",
        "cols = [c for c in __COLS__ if c in gdf.columns]",
        "rows = gdf[cols].itertuples(index=False, name=None)",
        "set_result({'columns': cols,",
        "            'rows': [[None if (v is None or (isinstance(v, float) and v != v)) else v",
        "                      for v in row] for row in rows]})",
    ]
)

_READ_ARCPY = "\n".join(
    [
        "import arcpy",
        "names = [n for n in __COLS__ if n in {f.name for f in arcpy.ListFields(__PATH__)}]",
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
    return recipe_lib.run("attribute_join_table", params, ctx)


def _join(runner, tmp_path: Path, name: str, **overrides):
    params = {
        "target_path": str(tmp_path / "zones.gpkg") + "/zones",
        "target_key": "code",
        "join_table": str(tmp_path / "stats.gpkg") + "/stats",
        "join_key": "code",
        "output_path": str(tmp_path / name),
    }
    params.update(overrides)
    return _run(RecipeLibrary(code_runner=runner), runner, tmp_path, **params)


def _make_gpkg(runner, tmp_path: Path) -> tuple[Path, Path]:
    zones, stats = tmp_path / "zones.gpkg", tmp_path / "stats.gpkg"
    made = runner.run(
        _MAKE_GPKG.replace("__TARGET__", repr(TARGET_ROWS))
        .replace("__STATS__", repr(STATS_ROWS))
        .replace("__TARGET_PATH__", repr(str(zones)))
        .replace("__STATS_PATH__", repr(str(stats))),
        timeout=180,
    )
    assert made.ok, made.error
    return zones, stats


def _read_rows(runner, path: str, cols: list[str], *, arcpy: bool, layer: str | None = None) -> list[list]:
    code = (
        (_READ_ARCPY if arcpy else _READ_GPKG)
        .replace("__PATH__", repr(str(path)))
        .replace("__COLS__", repr(cols))
        .replace("__LAYER__", repr(layer))
    )
    outcome = runner.run(code, timeout=180)
    assert outcome.ok, outcome.error
    return outcome.result["rows"]


def _relative_diff(actual, expected) -> float | None:
    """两边都空 → None（视为一致）；一边空一边有值 → inf（一致不了）。"""
    if actual is None or expected is None:
        return None if actual is None and expected is None else float("inf")
    a, e = float(actual), float(expected)
    if e == 0:
        return 0.0 if abs(a) < 1e-9 else float("inf")
    return abs(a - e) / abs(e)


def _assert_rows_equal(actual: list[list], expected: list[list], label: str) -> None:
    """行数、行序、每个字段值：整数严格相等，浮点相对差 ≤1e-6（P0-1c 口径）。"""
    assert len(actual) == len(expected), f"{label}: 行数 {len(actual)} != {len(expected)}"
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert len(got) == len(want), f"{label} 第 {index} 行: {got} != {want}"
        for col, (a, e) in enumerate(zip(got, want)):
            if isinstance(e, str) or e is None:
                assert a == e, f"{label} 第 {index} 行第 {col} 列: {a!r} != {e!r}"
            else:
                assert _relative_diff(a, e) in (None, 0.0), (
                    f"{label} 第 {index} 行第 {col} 列: {a!r} != {e!r}"
                )


# ------------------------------------------------------------- 模板本身
def test_recipe_declares_two_backends_and_renders_both():
    recipe = Recipe.from_yaml(RECIPE)
    assert recipe.backend == ["arcpy", "gdal"]
    params = {
        "target_path": "zones.gdb/district",
        "target_key": "OBJECTID",
        "join_table": "stats.gdb/zonal",
        "join_key": "OBJECTID",
        "output_path": "out/joined.gpkg",
    }
    arcpy_code = recipe.render(params)
    gdal_code = recipe.render(params, backend="gdal")

    # arcpy 版仍然走 JoinField，容器解析走共享片段（不是复制一份判断逻辑）
    assert "arcpy.management.JoinField" in arcpy_code
    assert "def resolve_zone_layer" in arcpy_code, "目标容器的解析必须复用 fragments.py 的片段"
    assert "import geopandas" not in arcpy_code

    assert "gpd.read_file" in gdal_code and "import arcpy" not in gdal_code
    assert "merge(" in gdal_code and "JoinField(" not in gdal_code
    assert '".gpkg": "GPKG"' in gdal_code, "输出格式白名单必须写死在代码里"
    for key in ("zones.gdb/district", "stats.gdb/zonal", "out/joined.gpkg", "OBJECTID"):
        assert key in arcpy_code and key in gdal_code


# ----------------------------------------------------------- GDAL 侧真跑
def test_gdal_join_matches_arcpy_measured_behaviour(gdal_runner, tmp_path: Path):
    """一对多取第一条 / 键不匹配留空 / 重名加 _1 / 空 join_fields 连键一起挂。"""
    _make_gpkg(gdal_runner, tmp_path)
    obs = _join(gdal_runner, tmp_path, "out/joined.gpkg")

    assert obs.ok, obs.summary
    payload = obs.data["result"]
    assert payload["feature_count"] == 4
    assert payload["added_fields"] == EXPECTED_ADDED
    # 与 arcpy 同口径：数的是非空键值个数，不是真匹配上的行数
    assert payload["key_match_count"] == 4

    rows = _read_rows(gdal_runner, payload["output"], ["code", "code_1", "pop", "name_1"], arcpy=False, layer="joined")
    _assert_rows_equal(rows, EXPECTED_ROWS, "GDAL")

    # 产物必须是真 GPKG 且挂上的 pop 是整型：源字段是 Integer（int32），带 NULL 也得写回
    # INTEGER，不能因为「有空值」被 pandas 升成 float 而悄悄变成 REAL。
    info = gdal_runner.run(
        "\n".join(
            [
                "import pyogrio",
                f"i = pyogrio.read_info({payload['output']!r}, layer='joined')",
                "set_result({'driver': i.get('driver'), 'features': int(i['features']),",
                "            'types': dict(zip([str(f) for f in i['fields']], [str(t) for t in i['ogr_types']]))})",
            ]
        ),
        timeout=120,
    )
    assert info.ok, info.error
    assert info.result["driver"] == "GPKG"
    assert info.result["features"] == 4
    assert info.result["types"]["pop"] == "OFTInteger", info.result["types"]
    # 两条断言都是 GDAL 口径真判过的
    assertions = {a["kind"]: a for a in obs.data["assertions"]}
    assert assertions["feature_count_gt"]["detail"] == "count=4"
    assert assertions["vector_exists"]["ok"] is True


def test_gdal_join_only_requested_fields_and_numeric_key(gdal_runner, tmp_path: Path):
    """join_fields 只挂指定字段（键不再挂一遍）；数值键 int 对 int 严格相等。"""
    _make_gpkg(gdal_runner, tmp_path)
    obs = _join(gdal_runner, tmp_path, "out/pop_only.gpkg", join_fields="pop")

    assert obs.ok, obs.summary
    payload = obs.data["result"]
    assert payload["added_fields"] == ["pop"]
    rows = _read_rows(gdal_runner, payload["output"], ["code", "pop"], arcpy=False, layer="pop_only")
    _assert_rows_equal(rows, [["a", 10], ["b", 20], ["c", None], ["d", None]], "GDAL")

    # 数值键：目标 code 换成整型，挂接表键也是整型
    made = gdal_runner.run(
        "\n".join(
            [
                "import geopandas as gpd",
                "import pandas as pd",
                "import pyogrio",
                "from shapely.geometry import Polygon",
                "zones = gpd.GeoDataFrame({'code': [1, 2, 3]},",
                "    geometry=[Polygon([(i*100, 0), (i*100+50, 0), (i*100+50, 50), (i*100, 50)]) for i in range(3)],",
                "    crs='EPSG:32650')",
                f"zones.to_file({str(tmp_path / 'num.gpkg')!r}, driver='GPKG', engine='pyogrio', layer='zones')",
                "pyogrio.write_dataframe(pd.DataFrame({'code': [2, 3, 4], 'pop': [20, 30, 40]}),",
                f"    {str(tmp_path / 'num.gpkg')!r}, driver='GPKG', layer='stats')",
                "set_result({'ok': True})",
            ]
        ),
        timeout=180,
    )
    assert made.ok, made.error
    numeric = _join(
        gdal_runner,
        tmp_path,
        "out/num_join.gpkg",
        target_path=str(tmp_path / "num.gpkg") + "/zones",
        join_table=str(tmp_path / "num.gpkg") + "/stats",
        join_fields="pop",
    )
    assert numeric.ok, numeric.summary
    rows = _read_rows(
        gdal_runner, numeric.data["result"]["output"], ["code", "pop"], arcpy=False, layer="num_join"
    )
    _assert_rows_equal(rows, [[1, None], [2, 20], [3, 30]], "GDAL 数值键")


def test_gdal_join_on_oid_uses_the_real_oid_name(gdal_runner, tmp_path: Path):
    """不写死 OBJECTID：.gpkg 的 OID 叫 fid 且从 1 开始，写 OBJECTID 也按真名取（适老化链就这么挂回去）。"""
    _make_gpkg(gdal_runner, tmp_path)
    # 统计表按 OID 存（zonal_statistics 的 GDAL 版空 zone_field 时写的就是 OBJECTID 列，1..N）
    made = gdal_runner.run(
        "\n".join(
            [
                "import pandas as pd",
                "import pyogrio",
                "pyogrio.write_dataframe(pd.DataFrame({'OBJECTID': [1, 3], 'SUM': [1.5, 3.5]}),",
                f"    {str(tmp_path / 'zonal.gpkg')!r}, driver='GPKG', layer='zonal')",
                "set_result({'ok': True})",
            ]
        ),
        timeout=180,
    )
    assert made.ok, made.error

    obs = _join(
        gdal_runner,
        tmp_path,
        "out/by_oid.gpkg",
        target_key="OBJECTID",
        join_table=str(tmp_path / "zonal.gpkg") + "/zonal",
        join_key="OBJECTID",
        join_fields="SUM",
    )
    assert obs.ok, obs.summary
    assert "按目标要素的 OID 处理（真名 fid）" in (obs.data.get("stdout") or ""), obs.data.get("stdout")
    rows = _read_rows(gdal_runner, obs.data["result"]["output"], ["code", "SUM"], arcpy=False, layer="by_oid")
    _assert_rows_equal(rows, [["a", 1.5], ["b", None], ["c", 3.5], ["d", None]], "GDAL OID 挂接")

    # 临时列不能跟着写进产物（它就是 OID 本身，不是用户的数据列）
    assert "__dair_oid__" not in (obs.data["result"]["added_fields"] or []), obs.data["result"]
    info = gdal_runner.run(
        "\n".join(
            [
                "import pyogrio",
                f"i = pyogrio.read_info({obs.data['result']['output']!r}, layer='by_oid')",
                "set_result({'fields': [str(f) for f in i['fields']]})",
            ]
        ),
        timeout=120,
    )
    assert info.ok, info.error
    assert "__dair_oid__" not in info.result["fields"], info.result


def test_gdal_join_writes_shapefile_with_honest_field_names(gdal_runner, tmp_path: Path):
    """输出 .shp 也支持：字段名被 GDAL 截断到 10 字符时，added_fields 报**实际**名字。"""
    _make_gpkg(gdal_runner, tmp_path)
    made = gdal_runner.run(
        "\n".join(
            [
                "import pandas as pd",
                "import pyogrio",
                "pyogrio.write_dataframe(pd.DataFrame({'code': ['a', 'b'], 'very_long_field': [1, 2]}),",
                f"    {str(tmp_path / 'long.gpkg')!r}, driver='GPKG', layer='long')",
                "set_result({'ok': True})",
            ]
        ),
        timeout=180,
    )
    assert made.ok, made.error
    obs = _join(
        gdal_runner,
        tmp_path,
        "out/shp_join.shp",
        join_table=str(tmp_path / "long.gpkg") + "/long",
        join_fields="very_long_field",
    )
    assert obs.ok, obs.summary
    # 报原名等于骗人：.shp 里它就叫 very_long_
    assert obs.data["result"]["added_fields"] == ["very_long_"], obs.data["result"]
    rows = _read_rows(gdal_runner, obs.data["result"]["output"], ["code", "very_long_"], arcpy=False)
    _assert_rows_equal(rows, [["a", 1], ["b", 2], ["c", None], ["d", None]], "GDAL .shp")


def test_gdal_rejects_unknown_join_field(gdal_runner, tmp_path: Path):
    """arcpy 实测：fields 里写了不存在的字段会**静默忽略**（等于什么都没挂上）；GDAL 报错。"""
    _make_gpkg(gdal_runner, tmp_path)
    obs = _join(gdal_runner, tmp_path, "out/typo.gpkg", join_fields="pop,pup")

    assert not obs.ok
    assert "挂接表没有字段 pup" in obs.summary
    assert "pop" in obs.summary, "报错要带上现有字段名，否则用户还得自己再去查一遍"
    assert not (tmp_path / "out" / "typo.gpkg").exists()


# --------------------------------------------------------------- 拒绝分支
def test_gdal_rejects_gdb_output_and_multi_layer_container(gdal_runner, tmp_path: Path):
    _make_gpkg(gdal_runner, tmp_path)

    gdb_out = _join(gdal_runner, tmp_path, "out/joined.gdb/joined")
    assert not gdb_out.ok
    assert "GDAL 版输出只支持 .gpkg / .shp" in gdb_out.summary
    assert "FileGDB 写驱动" in gdb_out.summary and "arcpy 后端" in gdb_out.summary
    assert not (tmp_path / "out" / "joined.gdb").exists(), "报错时不该留下半成品"

    # 多层容器（面 + 统计表在同一个 .gpkg）→ 人话错误，不自己挑一层
    made = gdal_runner.run(
        "\n".join(
            [
                "import geopandas as gpd",
                "import pandas as pd",
                "import pyogrio",
                f"g = gpd.read_file({str(tmp_path / 'zones.gpkg')!r}, layer='zones', engine='pyogrio')",
                f"g.to_file({str(tmp_path / 'multi.gpkg')!r}, driver='GPKG', engine='pyogrio', layer='zones')",
                f"pyogrio.write_dataframe(pd.DataFrame({{'code': ['a'], 'pop': [1]}}),",
                f"    {str(tmp_path / 'multi.gpkg')!r}, driver='GPKG', layer='stats')",
                "set_result({'ok': True})",
            ]
        ),
        timeout=180,
    )
    assert made.ok, made.error
    multi = _join(gdal_runner, tmp_path, "out/multi.gpkg", target_path=str(tmp_path / "multi.gpkg"))
    assert not multi.ok
    assert "多层容器" in multi.summary
    assert "zones" in multi.summary and "stats" in multi.summary
    assert "请指明图层" in multi.summary


def test_gdal_rejects_text_key_against_numeric_key(gdal_runner, tmp_path: Path):
    """arcpy 实测：文本键对数值键不报错，静默匹配 0 行、字段全空；GDAL 直接拒绝。"""
    _make_gpkg(gdal_runner, tmp_path)
    made = gdal_runner.run(
        "\n".join(
            [
                "import pandas as pd",
                "import pyogrio",
                "pyogrio.write_dataframe(pd.DataFrame({'code': [1, 2], 'pop': [10, 20]}),",
                f"    {str(tmp_path / 'numstats.gpkg')!r}, driver='GPKG', layer='numstats')",
                "set_result({'ok': True})",
            ]
        ),
        timeout=180,
    )
    assert made.ok, made.error

    obs = _join(gdal_runner, tmp_path, "out/type.gpkg", join_table=str(tmp_path / "numstats.gpkg") + "/numstats")
    assert not obs.ok
    assert "关键字段类型不一致" in obs.summary
    assert "静默匹配 0 行" in obs.summary, "报错要说清 arcpy 那边是什么行为，用户才知道为什么两端不一样"
    assert not (tmp_path / "out" / "type.gpkg").exists()


# ------------------------------------------------------------- 容器输入
@pytest.mark.parametrize("backend", ["arcpy", "gdal"])
def test_container_target_is_resolved_or_rejected_with_a_human_error(
    backend, arcpy_runner, gdal_runner, tmp_path: Path
):
    """单层容器 → 自动取用并跑通；多层容器 → 人话错误（列图层名），不是 Describe 裸错。"""
    if backend == "arcpy":
        gdb = tmp_path / "container.gdb"
        made = arcpy_runner.run(
            _MAKE_ARCPY_INPUTS.replace("__GDB__", repr(str(gdb)))
            .replace("__PARENT__", repr(str(tmp_path)))
            .replace("__TARGET__", repr(TARGET_ROWS))
            .replace("__STATS__", repr(STATS_ROWS)),
            timeout=600,
        )
        assert made.ok, made.error
        runner = arcpy_runner
        container = str(gdb)  # .gdb 里只有 zones 一个要素类（stats 是表，不算图层）
        stats_path = made.result["table"]
        extra = "\n".join(
            [
                "import arcpy",
                "arcpy.management.CreateFeatureclass(" + repr(str(gdb)) + ", 'towns', 'POLYGON',",
                "    spatial_reference=arcpy.SpatialReference(32650))",
                "set_result({'ok': True})",
            ]
        )
        # 输出另开一个 .gdb：写回 container.gdb 会多出一层，把「单层容器」变成多层
        out_gdb = tmp_path / "outputs.gdb"
        assert arcpy_runner.run(
            "import arcpy\narcpy.management.CreateFileGDB(" + repr(str(tmp_path)) + ", 'outputs.gdb')\n"
            "set_result({'ok': True})",
            timeout=300,
        ).ok
        out_path = lambda name: os.path.join(str(out_gdb), name)  # noqa: E731
        out_layer = None
        kwargs = {"arcpy": True}
    else:
        _make_gpkg(gdal_runner, tmp_path)
        made = gdal_runner.run(
            "\n".join(
                [
                    "import geopandas as gpd",
                    f"g = gpd.read_file({str(tmp_path / 'zones.gpkg')!r}, layer='zones', engine='pyogrio')",
                    f"g.to_file({str(tmp_path / 'container.gpkg')!r}, driver='GPKG', engine='pyogrio', layer='zones')",
                    "set_result({'ok': True})",
                ]
            ),
            timeout=180,
        )
        assert made.ok, made.error
        runner = gdal_runner
        container = str(tmp_path / "container.gpkg")
        stats_path = str(tmp_path / "stats.gpkg") + "/stats"
        extra = "\n".join(
            [
                "import geopandas as gpd",
                "from shapely.geometry import Polygon",
                "g = gpd.GeoDataFrame({'code': ['t']},",
                "    geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs='EPSG:32650')",
                f"g.to_file({container!r}, driver='GPKG', engine='pyogrio', layer='towns')",
                "set_result({'ok': True})",
            ]
        )
        out_path = lambda name: os.path.join(str(tmp_path), "out", f"{name}.gpkg")  # noqa: E731
        out_layer = lambda name: name  # noqa: E731
        kwargs = {"arcpy": False}

    params = {
        "target_path": container,
        "target_key": "code",
        "join_table": stats_path,
        "join_key": "code",
        "join_fields": "pop",
    }
    single = _run(
        RecipeLibrary(code_runner=runner),
        runner,
        tmp_path,
        output_path=out_path("single_out"),
        **params,
    )
    assert single.ok, single.summary
    assert "自动使用其中唯一的图层: zones" in (single.data.get("stdout") or ""), single.data.get("stdout")
    rows = _read_rows(
        runner,
        single.data["result"]["output"],
        ["code", "pop"],
        layer=out_layer("single_out") if out_layer else None,
        **kwargs,
    )
    _assert_rows_equal(rows, [["a", 10], ["b", 20], ["c", None], ["d", None]], backend)

    # 再加一个图层：容器变成多层，配方必须报人话错误而不是自己挑一个
    assert runner.run(extra, timeout=300).ok
    multi = _run(
        RecipeLibrary(code_runner=runner),
        runner,
        tmp_path,
        output_path=out_path("multi_out"),
        **params,
    )
    assert not multi.ok, multi.summary
    assert "多层容器" in multi.summary, multi.summary
    assert "zones" in multi.summary and "towns" in multi.summary, multi.summary
    assert "AttributeError" not in multi.summary and "DescribeData" not in multi.summary, multi.summary


def test_join_table_multi_layer_container_is_rejected_on_both_backends(
    arcpy_runner, gdal_runner, tmp_path: Path
):
    """挂接表给的是**多层**容器（库/文件夹里不止一张表）：两端都报人话错误，不替用户猜一张表。

    单层容器不是这条：arcpy 侧仍然一律拒绝，GDAL 侧自动取用（见下面两条）。
    """
    gdb = tmp_path / "jt.gdb"
    made = arcpy_runner.run(
        _MAKE_ARCPY_INPUTS.replace("__GDB__", repr(str(gdb)))
        .replace("__PARENT__", repr(str(tmp_path)))
        .replace("__TARGET__", repr(TARGET_ROWS))
        .replace("__STATS__", repr(STATS_ROWS)),
        timeout=600,
    )
    assert made.ok, made.error
    arcpy_obs = _run(
        RecipeLibrary(code_runner=arcpy_runner),
        arcpy_runner,
        tmp_path,
        target_path=made.result["fc"],
        target_key="code",
        join_table=str(gdb),
        join_key="code",
        output_path=os.path.join(str(gdb), "bad"),
    )
    assert not arcpy_obs.ok
    assert "挂接表给的是容器" in arcpy_obs.summary, arcpy_obs.summary
    assert "请指到表" in arcpy_obs.summary, arcpy_obs.summary

    _make_gpkg(gdal_runner, tmp_path)
    made = gdal_runner.run(
        "\n".join(
            [
                "import geopandas as gpd",
                "import pandas as pd",
                "import pyogrio",
                f"g = gpd.read_file({str(tmp_path / 'zones.gpkg')!r}, layer='zones', engine='pyogrio')",
                f"g.to_file({str(tmp_path / 'jt.gpkg')!r}, driver='GPKG', engine='pyogrio', layer='zones')",
                f"pyogrio.write_dataframe(pd.DataFrame({{'code': ['a'], 'pop': [1]}}),",
                f"    {str(tmp_path / 'jt.gpkg')!r}, driver='GPKG', layer='stats')",
                "set_result({'ok': True})",
            ]
        ),
        timeout=180,
    )
    assert made.ok, made.error
    gdal_obs = _join(gdal_runner, tmp_path, "out/jt.gpkg", join_table=str(tmp_path / "jt.gpkg"))
    assert not gdal_obs.ok
    assert "挂接表是多层容器" in gdal_obs.summary, gdal_obs.summary


def test_gdal_single_layer_container_join_table_is_auto_used(gdal_runner, tmp_path: Path):
    """挂接表给的是**单层**容器：GDAL 侧自动取用并在 stdout 说明用了哪张表（arcpy 侧一律拒绝）。

    pitfalls 这么写就得有这一条：只有一张表时谈不上「替用户挑一张」。
    """
    _make_gpkg(gdal_runner, tmp_path)
    only_stats = tmp_path / "only_stats.gpkg"
    made = gdal_runner.run(
        "\n".join(
            [
                "import pandas as pd",
                "import pyogrio",
                "pyogrio.write_dataframe(pd.DataFrame({'code': ['a', 'b'], 'pop': [10, 20]}),",
                f"    {str(only_stats)!r}, driver='GPKG', layer='stats')",
                "set_result({'ok': True})",
            ]
        ),
        timeout=180,
    )
    assert made.ok, made.error

    obs = _join(
        gdal_runner, tmp_path, "out/single_layer.gpkg", join_table=str(only_stats), join_fields="pop"
    )
    assert obs.ok, obs.summary
    stdout = obs.data.get("stdout") or ""
    assert "挂接表是工作空间容器" in stdout, stdout
    assert "自动使用其中唯一的图层: stats" in stdout, stdout
    rows = _read_rows(
        gdal_runner,
        obs.data["result"]["output"],
        ["code", "pop"],
        arcpy=False,
        layer="single_layer",
    )
    _assert_rows_equal(rows, [["a", 10], ["b", 20], ["c", None], ["d", None]], "GDAL 单层挂接表容器")


def test_gdal_reads_csv_join_table_as_text(gdal_runner, tmp_path: Path):
    """GDAL 的 .csv 一律按文本读（OGR CSV 默认不推断类型）：文本键能挂，数值键直接报类型不一致。"""
    _make_gpkg(gdal_runner, tmp_path)
    csv = tmp_path / "stats.csv"
    csv.write_text("code,pop\na,10\nb,20\n", encoding="utf-8")

    ok = _join(gdal_runner, tmp_path, "out/from_csv.gpkg", join_table=str(csv), join_fields="pop")
    assert ok.ok, ok.summary
    rows = _read_rows(
        gdal_runner, ok.data["result"]["output"], ["code", "pop"], arcpy=False, layer="from_csv"
    )
    _assert_rows_equal(rows, [["a", 10], ["b", 20], ["c", None], ["d", None]], "GDAL 读 CSV 挂接表")

    # 同一个 .csv 的数值键按文本读成 "1"/"2"，跟目标 .gpkg 的整型 OID 对不上 → 报类型不一致，
    # 而不是静默 0 行（arcpy 侧会静默全空，所以必须失败得响亮）
    num_csv = tmp_path / "numstats.csv"
    num_csv.write_text("OBJECTID,pop\n1,10\n2,20\n", encoding="utf-8")
    bad = _join(
        gdal_runner,
        tmp_path,
        "out/from_csv_num.gpkg",
        target_key="OBJECTID",
        join_table=str(num_csv),
        join_key="OBJECTID",
        join_fields="pop",
    )
    assert not bad.ok
    assert "关键字段类型不一致" in bad.summary, bad.summary


def test_gdal_container_inputs_resolve_or_reject(gdal_runner, tmp_path: Path):
    """.gpkg 单层 + shp 文件夹各跑通一次；多层容器给人话错误。"""
    _make_gpkg(gdal_runner, tmp_path)
    single = tmp_path / "single.gpkg"
    shpdir = tmp_path / "shpdir"
    made = gdal_runner.run(
        "\n".join(
            [
                "import os",
                "import geopandas as gpd",
                f"g = gpd.read_file({str(tmp_path / 'zones.gpkg')!r}, layer='zones', engine='pyogrio')",
                f"g.to_file({str(single)!r}, driver='GPKG', engine='pyogrio', layer='zones')",
                f"os.makedirs({str(shpdir)!r}, exist_ok=True)",
                f"g.to_file({str(shpdir / 'zones.shp')!r}, driver='ESRI Shapefile', engine='pyogrio')",
                "set_result({'ok': True})",
            ]
        ),
        timeout=180,
    )
    assert made.ok, made.error

    for target, layer_name, out_name in (
        (str(single), "zones", "from_gpkg.gpkg"),
        (str(shpdir), "zones", "from_shpdir.gpkg"),
    ):
        obs = _join(
            gdal_runner,
            tmp_path,
            f"out/{out_name}",
            target_path=target,
            join_fields="pop",
        )
        assert obs.ok, f"{target}: {obs.summary}"
        assert f"自动使用其中唯一的图层: {layer_name}" in (obs.data.get("stdout") or ""), (
            obs.data.get("stdout")
        )
        rows = _read_rows(
            gdal_runner,
            obs.data["result"]["output"],
            ["code", "pop"],
            arcpy=False,
            layer=os.path.splitext(out_name)[0],
        )
        _assert_rows_equal(rows, [["a", 10], ["b", 20], ["c", None], ["d", None]], f"GDAL {target}")


# ------------------------------------------------------- 双端逐值一致（真 arcpy）
def test_both_backends_produce_the_same_table(arcpy_runner, gdal_runner, tmp_path: Path):
    """同一份逻辑数据（.gdb 与 .gpkg 各造一份）：行数、行序、每个字段值都一致。"""
    gdb = tmp_path / "inputs.gdb"
    made = arcpy_runner.run(
        _MAKE_ARCPY_INPUTS.replace("__GDB__", repr(str(gdb)))
        .replace("__PARENT__", repr(str(tmp_path)))
        .replace("__TARGET__", repr(TARGET_ROWS))
        .replace("__STATS__", repr(STATS_ROWS)),
        timeout=600,
    )
    assert made.ok, made.error

    arcpy_obs = _run(
        RecipeLibrary(code_runner=arcpy_runner),
        arcpy_runner,
        tmp_path,
        target_path=os.path.join(str(gdb), "zones"),
        target_key="code",
        join_table=os.path.join(str(gdb), "stats"),
        join_key="code",
        output_path=os.path.join(str(gdb), "joined"),
    )
    assert arcpy_obs.ok, arcpy_obs.summary

    _make_gpkg(gdal_runner, tmp_path)
    gdal_obs = _join(gdal_runner, tmp_path, "out/joined.gpkg")
    assert gdal_obs.ok, gdal_obs.summary

    cols = ["code", "code_1", "pop", "name_1"]
    arcpy_rows = _read_rows(arcpy_runner, arcpy_obs.data["result"]["output"], cols, arcpy=True)
    gdal_rows = _read_rows(
        gdal_runner, gdal_obs.data["result"]["output"], cols, arcpy=False, layer="joined"
    )
    _assert_rows_equal(arcpy_rows, EXPECTED_ROWS, "arcpy")
    _assert_rows_equal(gdal_rows, EXPECTED_ROWS, "GDAL")
    _assert_rows_equal(gdal_rows, arcpy_rows, "GDAL vs arcpy")

    assert arcpy_obs.data["result"]["feature_count"] == gdal_obs.data["result"]["feature_count"] == 4
    assert arcpy_obs.data["result"]["added_fields"] == gdal_obs.data["result"]["added_fields"] == EXPECTED_ADDED
    assert arcpy_obs.data["result"]["key_match_count"] == gdal_obs.data["result"]["key_match_count"] == 4


def test_arcpy_silently_zero_matches_where_gdal_refuses(arcpy_runner, gdal_runner, tmp_path: Path):
    """类型不一致时两端的**行为差异**是已知且写进 pitfalls 的：arcpy 静默 0 行，GDAL 报错。"""
    gdb = tmp_path / "type.gdb"
    made = arcpy_runner.run(
        "\n".join(
            [
                "import arcpy, os",
                "arcpy.env.overwriteOutput = True",
                "gdb = " + repr(str(gdb)),
                "if not arcpy.Exists(gdb):",
                "    arcpy.management.CreateFileGDB(" + repr(str(tmp_path)) + ", 'type.gdb')",
                "fc = os.path.join(gdb, 'zones')",
                "tbl = os.path.join(gdb, 'stats')",
                "arcpy.management.CreateFeatureclass(gdb, 'zones', 'POLYGON',",
                "    spatial_reference=arcpy.SpatialReference(32650))",
                "arcpy.management.AddField(fc, 'code', 'TEXT', field_length=10)",
                "with arcpy.da.InsertCursor(fc, ['SHAPE@', 'code']) as cur:",
                "    cur.insertRow([arcpy.Polygon(arcpy.Array([arcpy.Point(0, 0), arcpy.Point(50, 0),",
                "                                      arcpy.Point(50, 50), arcpy.Point(0, 50)]),",
                "                                arcpy.SpatialReference(32650)), '1'])",
                "arcpy.management.CreateTable(gdb, 'stats')",
                "arcpy.management.AddField(tbl, 'code', 'LONG')",
                "arcpy.management.AddField(tbl, 'pop', 'LONG')",
                "with arcpy.da.InsertCursor(tbl, ['code', 'pop']) as cur:",
                "    cur.insertRow([1, 10])",
                "set_result({'fc': fc, 'table': tbl})",
            ]
        ),
        timeout=600,
    )
    assert made.ok, made.error

    arcpy_obs = _run(
        RecipeLibrary(code_runner=arcpy_runner),
        arcpy_runner,
        tmp_path,
        target_path=made.result["fc"],
        target_key="code",
        join_table=made.result["table"],
        join_key="code",
        join_fields="pop",
        output_path=os.path.join(str(gdb), "joined"),
    )
    assert arcpy_obs.ok, arcpy_obs.summary
    rows = _read_rows(arcpy_runner, arcpy_obs.data["result"]["output"], ["code", "pop"], arcpy=True)
    assert rows == [["1", None]], "arcpy 实测：文本键对数值键不报错，静默匹配 0 行、字段全空"

    # 同一份数据搬到 GDAL 侧：直接拒绝，不产出「跑成功但全是空」的错数据
    gdal_runner.run(
        "\n".join(
            [
                "import geopandas as gpd",
                "import pandas as pd",
                "import pyogrio",
                "from shapely.geometry import Polygon",
                f"g = gpd.GeoDataFrame({{'code': ['1']}}, geometry=[Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])],",
                "    crs='EPSG:32650')",
                f"g.to_file({str(tmp_path / 'type.gpkg')!r}, driver='GPKG', engine='pyogrio', layer='zones')",
                "pyogrio.write_dataframe(pd.DataFrame({'code': [1], 'pop': [10]}),",
                f"    {str(tmp_path / 'type.gpkg')!r}, driver='GPKG', layer='stats')",
                "set_result({'ok': True})",
            ]
        ),
        timeout=180,
    )
    gdal_obs = _join(
        gdal_runner,
        tmp_path,
        "out/type.gpkg",
        target_path=str(tmp_path / "type.gpkg") + "/zones",
        join_table=str(tmp_path / "type.gpkg") + "/stats",
        join_fields="pop",
    )
    assert not gdal_obs.ok
    assert "关键字段类型不一致" in gdal_obs.summary

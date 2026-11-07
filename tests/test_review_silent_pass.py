# -*- coding: utf-8 -*-
"""DAIR-191：五处「独立形态仍然漏、且静默判通过」的回归用例。

复现代码来自 DAIR-182 的独立造错复测（`benchmark/reports/dair182_w5_bench_20261105/`）：
这些错误形态下 `checked` 非空、summary 写「断言全部通过」，复核者会当 pass 上报。
每条用例都是「改前红、改后绿」——改前那条 assert 拿到的是一份自称全过的报告。
"""

from __future__ import annotations

import io
import json
import sys
import types
from contextlib import redirect_stdout

from gis_cli.review import ReviewClaim, run_assertions
from gis_cli.review import facts as facts_mod

POP = {"Z01": 1_200_000, "Z02": 980_000, "Z03": 1_050_000, "Z04": 1_120_000,
       "Z05": 1_000_000, "Z06": 1_030_000, "Z07": 960_000, "Z08": 1_080_000}
AREA_KM2 = {"Z01": 20.0, "Z02": 18.0, "Z03": 22.0, "Z04": 19.0,
            "Z05": 17.0, "Z06": 21.0, "Z07": 15.0, "Z08": 20.34}
STUDY_AREA_M2 = 152_340_000.0


def _facts(**datasets):
    return {"datasets": datasets, "paths": list(datasets), "collector": ["arcpy"]}


def _vector(**extra):
    base = {
        "exists": True,
        "kind": "vector",
        "collector": "arcpy",
        "count": 8,
        "crs": {"wkid": 32610, "name": "UTM 10N", "is_projected": True, "linear_unit": "Meter"},
        "geometry_types": ["Polygon"],
        "area_total": STUDY_AREA_M2,
        "area_unit": "m2",
        "invalid_count": 0,
        "overlap_ratio": 0.0,
    }
    base.update(extra)
    return base


def _codes(report):
    return [item.code for item in report.findings]


DENSITY = {
    "type": "dimension.density",
    "density": "density_per_km2",
    "count": "pop",
    "area": "area_km2",
    "per_area_unit": "km2",
}


# ------------------------------------------------------------ 缺陷 1：密度分子
def test_density_numerator_must_match_source_population():
    """人口按「十人」为单位：claim 三个数自洽（32000÷12.8=2500），源数据是 32 万。"""
    facts = _facts(
        **{
            "output/green_equity.shp": _vector(
                count=1, area_total=12_800_000.0, group_sums={"POP": {"G01": 320_000}}
            )
        }
    )
    claim = ReviewClaim(
        metrics={"pop": 320_000, "area_km2": 12.8, "density_per_km2": 25_000.0},
        units={"area": "km2"},
        data_paths=["output/green_equity.shp"],
    )
    assert run_assertions([DENSITY], facts, claim).ok

    claim.metrics.update({"pop": 32_000, "density_per_km2": 2_500.0})  # 三数依然自洽
    report = run_assertions([DENSITY], facts, claim)
    assert _codes(report) == ["dimension.density_mismatch"]
    assert not report.ok and "分子" in report.findings[0].detail
    assert report.findings[0].claimed == 32_000 and report.findings[0].measured == 320_000

    # 显式给 count_source_path 时按它比（不依赖同名自动解析）
    explicit = dict(DENSITY, count_source_path=["datasets", "output/green_equity.shp", "group_sums", "POP"])
    report = run_assertions([explicit], facts, claim)
    assert _codes(report) == ["dimension.density_mismatch"]

    # 源数据里没有同名量测 → 不猜（自洽检查仍然算通过）
    bare = _facts(**{"output/green_equity.shp": _vector(count=1, area_total=12_800_000.0)})
    assert run_assertions([DENSITY], bare, claim).ok


# ------------------------------------------------------------ 缺陷 2：精度截断
def test_density_denominator_truncation_is_not_silent():
    """面积 152.34 km² 截断写成 152.0，密度按 152 算：0.22% 落在 2% 容差里。"""
    facts = _facts(**{"output/study_area.shp": _vector(count=1)})
    clean = ReviewClaim(
        metrics={"area_km2": 152.34, "pop": 8_420_000, "density_per_km2": 8_420_000 / 152.34},
        units={"area": "km2"},
        data_paths=["output/study_area.shp"],
    )
    specs = [
        {"type": "dimension.area_unit", "metric": "area_km2",
         "source_path": ["datasets", "output/study_area.shp", "area_total"]},
        DENSITY,
    ]
    assert run_assertions(specs, facts, clean).ok

    clean.metrics.update({"area_km2": 152.0, "density_per_km2": 8_420_000 / 152.0})
    report = run_assertions(specs, facts, clean)
    assert "dimension.density_mismatch" in _codes(report), report.to_dict()
    assert not report.ok and "分母" in report.findings[0].detail


def test_dimension_precision_checks_declared_digits():
    """「声明精度」单列一条断言：写了几位小数就只能要求精确到那一位。"""
    facts = _facts(**{"output/study_area.shp": _vector(count=1)})
    spec = [
        {"type": "dimension.precision", "metric": "area_km2",
         "source_path": ["datasets", "output/study_area.shp", "area_total"],
         "source_unit": "m2", "unit": "km2"},
    ]

    def claim(value):
        return ReviewClaim(metrics={"area_km2": value}, units={"area": "km2"})

    assert run_assertions(spec, facts, claim(152.34)).ok
    assert run_assertions(spec, facts, claim(152.3)).ok        # 152.34 四舍五入到 0.1
    assert run_assertions(spec, facts, claim(152)).ok          # 整数写法是合法取整（±0.5）
    report = run_assertions(spec, facts, claim(152.0))         # 声称精确到 0.1，实际差 0.34
    assert _codes(report) == ["dimension.precision_mismatch"]
    assert "152.0" in report.findings[0].detail


def test_integer_declaration_has_no_relative_floor():
    """整数声明声称精确到个位：0.1% 地板必须拆掉（DAIR-195）。

    改前 pop=8_412_000 vs 量测 8_420_000（0.095%）、area_m2 差 40_000 m²（0.026%）都绿。
    """
    facts = _facts(**{
        "output/study_area.shp": _vector(count=1, fields={"POP": {"count": 1, "sum": 8_420_000.0}})
    })
    claim = ReviewClaim(
        metrics={"pop": 8_412_000, "area_km2": 152.34, "density_per_km2": 8_412_000 / 152.34,
                 "area_m2": 152_380_000},
        units={"area": "km2"},
        data_paths=["output/study_area.shp"],
    )
    precision = [
        {"type": "dimension.precision", "metric": "area_m2",
         "source_path": ["datasets", "output/study_area.shp", "area_total"],
         "source_unit": "m2", "unit": "m2"},
    ]
    assert _codes(run_assertions(precision, facts, claim)) == ["dimension.precision_mismatch"]
    report = run_assertions([DENSITY], facts, claim)
    assert _codes(report) == ["dimension.density_mismatch"], report.to_dict()
    assert report.findings[0].severity == "error" and not report.ok

    # 写一位小数就不再是「精确到个位」的声明，0.1% 地板照旧（同一偏差不报）
    claim.metrics.update({"pop": 8_412_000.0, "density_per_km2": 8_412_000.0 / 152.34,
                          "area_m2": 152_380_000.0})
    assert run_assertions(precision, facts, claim).ok
    assert run_assertions([DENSITY], facts, claim).ok


# ------------------------------------------------------------ 缺陷 3：errors 不计入 ok
def test_assertion_errors_flip_ok_and_are_readable():
    """matches_source 被喂分组字典：以前只进 errors，ok 仍 True、summary 写「全部通过」。"""
    facts = _facts(**{"output/zones.shp": _vector(group_sums={"POP": dict(POP)})})
    claim = ReviewClaim(
        metrics={"total_pop": sum(POP.values()), "feature_count": 8},
        data_paths=["output/zones.shp"],
    )
    report = run_assertions(
        [
            {"type": "conservation.matches_source", "metric": "total_pop",
             "source_path": ["datasets", "output/zones.shp", "group_sums", "POP"]},
            {"type": "conservation.count_matches", "metric": "feature_count",
             "source_path": ["datasets", "output/zones.shp", "count"]},
        ],
        facts,
        claim,
    )
    assert report.ok is False
    assert report.checked == ["conservation.count_matches"]
    assert "断言库未跑成" in report.summary and "断言全部通过" not in report.summary
    assert any("分区字典" in item for item in report.errors), report.errors
    assert any("partition_matches_source" in item for item in report.errors), report.errors


# ------------------------------------------------------------ 缺陷 4：关系方向
def test_relation_direction_reversed_is_caught():
    """实测「3 个结果面被研究区包含」，结论写「3 个结果面包含研究区」。"""
    facts = _facts(
        **{
            "output/parcels.shp": _vector(
                count=12, area_total=4_800_000.0, relations={"within": {"features": 3, "pairs": 3}}
            )
        }
    )
    contains_spec = [
        {"type": "topology.relation_count", "relation": "contains", "metric": "contains_count",
         "source_path": ["datasets", "output/parcels.shp", "relations", "contains"]},
    ]
    report = run_assertions(contains_spec, facts, ReviewClaim(metrics={"contains_count": 3}))
    assert _codes(report) == ["topology.relation_direction_mismatch"]
    assert not report.ok and "within" in report.findings[0].detail

    # 同一个方向比得上时照旧判通过
    within_spec = [
        {"type": "topology.relation_count", "relation": "within", "metric": "within_count",
         "source_path": ["datasets", "output/parcels.shp", "relations", "within"]},
    ]
    assert run_assertions(within_spec, facts, ReviewClaim(metrics={"within_count": 3})).ok


def test_measure_code_emits_both_relation_directions(monkeypatch):
    """ArcPy 量测脚本要同时给 contains / within（否则断言只能 skipped）。"""

    class _Geom:
        def __init__(self, idx, area):
            self.idx, self.area = idx, area

        def touches(self, other):
            return False

        def overlaps(self, other):
            return False

        def within(self, other):
            return (self.idx, other.idx) == (1, 0)

        def contains(self, other):
            return (self.idx, other.idx) == (0, 1)

    class _Field:
        def __init__(self, name, kind):
            self.name, self.type = name, kind

    class _Desc:
        dataType, shapeType = "ShapeFile", "Polygon"
        spatialReference = type("SR", (), {
            "type": "Projected", "linearUnitName": "Meter", "factoryCode": 32650, "name": "UTM 50N"
        })()

    class _Cursor:
        def __enter__(self):
            return iter(rows)

        def __exit__(self, *args):
            return False

    geoms = [_Geom(0, 100.0), _Geom(1, 200.0), _Geom(2, 300.0)]
    rows = [
        (geoms[0], 10, "A"),
        (geoms[1], 20, "B"),
        (geoms[2], 30, "C"),
    ]
    fake = types.ModuleType("arcpy")
    fake.Describe = lambda p: _Desc()
    fake.ListFields = lambda p: [_Field("POP", "Integer"), _Field("ZONE", "String")]
    fake.env = types.SimpleNamespace(scratchGDB=None, workspace=None)
    fake.da = types.SimpleNamespace(SearchCursor=lambda p, fields: _Cursor())
    monkeypatch.setitem(sys.modules, "arcpy", fake)

    code = (
        facts_mod._MEASURE_CODE.replace("__PATHS__", '["fake.shp"]')
        .replace("__GROUPS__", json.dumps({"fake.shp": {"POP": "ZONE"}}))
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exec(compile(code, "<review-measure>", "exec"), {"__name__": "__measure__"})
    entry = json.loads(buffer.getvalue())["fake.shp"]
    assert entry["relations"]["within"] == {"features": 1, "pairs": 1}
    assert entry["relations"]["contains"] == {"features": 1, "pairs": 1}
    # 分区面积汇总要默认产出（area_partition 靠它逐分区核对）
    assert entry["group_sums"]["AREA_M2"] == {"A": 100.0, "B": 200.0, "C": 300.0}
    assert entry["group_sums"]["POP"] == {"A": 10.0, "B": 20.0, "C": 30.0}


# ------------------------------------------------------------ 缺陷 5：分区面积交换
def test_area_partition_catches_swap_when_per_part_area_measured():
    """Z03 +4 km²、Z08 −4 km²（合计不变）：只校验合计永远抓不到。"""
    area_m2 = {key: value * 1_000_000.0 for key, value in AREA_KM2.items()}
    facts = _facts(
        **{"output/zones.shp": _vector(group_sums={"AREA_M2": area_m2})}
    )
    specs = [
        {"type": "conservation.area_partition", "parts": "zone_area",
         "total_path": ["datasets", "output/zones.shp", "area_total"],
         "unit": "km2", "total_unit": "m2"},
    ]
    claim = ReviewClaim(metrics={"zone_area": dict(AREA_KM2)}, units={"area": "km2"})
    assert run_assertions(specs, facts, claim).ok

    claim.metrics["zone_area"] = dict(AREA_KM2, Z03=26.0, Z08=16.34)
    report = run_assertions(specs, facts, claim)
    assert _codes(report) == ["conservation.partition_mismatch"]
    assert not report.ok and "Z03" in report.findings[0].detail

    # 只有总面积事实时不再假装「分区也对了」：给一条 warning 提示
    legacy = _facts(**{"output/zones.shp": _vector()})
    report = run_assertions(specs, legacy, ReviewClaim(metrics={"zone_area": dict(AREA_KM2)},
                                                       units={"area": "km2"}))
    assert report.findings and report.findings[0].code == "conservation.partition_unverified"
    assert report.findings[0].severity == "warning"
    assert "提示" in report.summary and "全部通过" not in report.summary


def test_area_partition_recognizes_multi_group_area_keys():
    """两个分组字段 → group_sums 键是 ``AREA_M2_BY_<cat>``（DAIR-195）。

    以前 ``_AREA_KEY_UNITS`` 只认 ``AREA_M2``，多分组数据集永远只给 partition_unverified，
    逐区核对实际上没生效。
    """
    facts = _facts(
        **{
            "output/zones.shp": _vector(
                area_total=100_000_000.0,
                group_sums={
                    "AREA_M2_BY_ZONE": {"A": 30_000_000.0, "B": 30_000_000.0, "C": 40_000_000.0},
                    "AREA_M2_BY_KIND": {"已建": 60_000_000.0, "未建": 40_000_000.0},
                },
            )
        }
    )
    specs = [
        {"type": "conservation.area_partition", "parts": "zone_area",
         "total_path": ["datasets", "output/zones.shp", "area_total"],
         "unit": "km2", "total_unit": "m2"},
    ]
    claim = ReviewClaim(metrics={"zone_area": {"A": 30.0, "B": 30.0, "C": 40.0}}, units={"area": "km2"})
    report = run_assertions(specs, facts, claim)
    assert report.ok and not report.findings, report.to_dict()

    # 合计不变、分区互换：要靠 AREA_M2_BY_ZONE 逐区抓出来，而不是拿 KIND 的分组乱比
    claim.metrics["zone_area"] = {"A": 60.0, "B": 30.0, "C": 10.0}
    report = run_assertions(specs, facts, claim)
    assert _codes(report) == ["conservation.partition_mismatch"], report.to_dict()
    assert "AREA_M2_BY_ZONE" in report.findings[0].detail
    assert "KIND" not in report.findings[0].detail


def test_area_partition_ambiguous_group_keys_prefers_the_key_claim_matches():
    """两个分组字段的分区标签撞名时，按 claim 的分区值挑键，不猜第一个（DAIR-197 / DAIR-201）。

    ZONE 和 KIND 都用 ``A/B/C``：claim 按 KIND 报、与 ``AREA_M2_BY_KIND`` 精确相等。
    以前 ``_measured_parts`` 命中第一个键就 return（ZONE），把正确的 claim 判成 error（差 60%）；
    DAIR-197 降级成 ``partition_unverified`` warning 保住了 ok，但连「claim 分区全错」也一起放过了。
    """
    facts = _facts(
        **{
            "output/zones.shp": _vector(
                area_total=100_000_000.0,
                group_sums={
                    "AREA_M2_BY_ZONE": {"A": 50_000_000.0, "B": 50_000_000.0},
                    "AREA_M2_BY_KIND": {"A": 20_000_000.0, "B": 80_000_000.0},
                },
            )
        }
    )
    specs = [
        {"type": "conservation.area_partition", "parts": "kind_area",
         "total_path": ["datasets", "output/zones.shp", "area_total"],
         "unit": "m2", "total_unit": "m2"},
    ]
    claim = ReviewClaim(
        metrics={"kind_area": {"A": 20_000_000.0, "B": 80_000_000.0}}, units={"area": "m2"}
    )
    report = run_assertions(specs, facts, claim)
    assert report.ok, report.to_dict()
    # 逐区对得上的那个键就是答案：不该再有一条「分区未核对」的 warning 噪音
    assert not report.findings, report.to_dict()


def test_area_partition_reports_error_when_no_group_key_matches_claim():
    """撞名不等于「claim 的分区对得上」：一个键都对不上时必须照实报错（DAIR-201）。

    ``5f80406c`` 只看标签集合有没有交集：两个候选 → 降级 warning，于是
    claim 分区全错、甚至整段对调也会是 ``ok=True``（``98498bff`` 是能抓到的）。
    """
    facts = _facts(
        **{
            "output/zones.shp": _vector(
                area_total=100_000_000.0,
                group_sums={
                    "AREA_M2_BY_ZONE": {"A": 50_000_000.0, "B": 50_000_000.0},
                    "AREA_M2_BY_KIND": {"A": 20_000_000.0, "B": 80_000_000.0},
                },
            )
        }
    )
    specs = [
        {"type": "conservation.area_partition", "parts": "kind_area",
         "total_path": ["datasets", "output/zones.shp", "area_total"],
         "unit": "m2", "total_unit": "m2"},
    ]
    # 合计 100 km² 对得上、分区全错（A 写 10 km²、B 写 90 km²）
    claim = ReviewClaim(
        metrics={"kind_area": {"A": 10_000_000.0, "B": 90_000_000.0}}, units={"area": "m2"}
    )
    report = run_assertions(specs, facts, claim)
    assert not report.ok, report.to_dict()
    assert _codes(report) == ["conservation.partition_mismatch"], report.to_dict()
    detail = report.findings[0].detail
    assert report.findings[0].severity == "error"
    # 挂最贴合的候选键，并把两个候选键都列出来
    assert "AREA_M2_BY_KIND" in detail and "AREA_M2_BY_ZONE" in detail, detail


def test_area_partition_array_parts_with_single_group_key_is_unverified():
    """claim 的 ``parts`` 写成数组、量测只有一个面积分组键时不能静默通过（DAIR-203）。

    ``_measured_parts`` 命中唯一键、``chosen`` 非空，``_area_partition`` 进 if 后因
    ``claimed_parts`` 为空什么也不做，而 ``elif not findings`` 是同一个 if 的 elif——
    一条 finding 都没有，``ok=True`` 自称「断言全部通过」。
    """
    facts = _facts(
        **{
            "output/z.shp": _vector(
                area_total=100_000_000.0,
                group_sums={"AREA_M2_BY_ZONE": {"A": 60_000_000.0, "B": 40_000_000.0}},
            )
        }
    )
    specs = [
        {"type": "conservation.area_partition", "parts": "parts",
         "total_path": ["datasets", "output/z.shp", "area_total"],
         "unit": "m2", "total_unit": "m2"},
    ]
    claim = ReviewClaim(metrics={"parts": [60_000_000.0, 40_000_000.0]}, units={"area": "m2"})
    report = run_assertions(specs, facts, claim)
    assert report.ok, report.to_dict()
    assert _codes(report) == ["conservation.partition_unverified"], report.to_dict()
    assert report.findings[0].severity == "warning"
    detail = report.findings[0].detail
    # 只有一个候选键：「分不清该比哪个」不成立，只说没标签、未逐区核对
    assert "AREA_M2_BY_ZONE" in detail, detail
    assert "分不清该比哪个" not in detail, detail
    assert "2 个分区面积未逐区核对" in detail, detail


def test_area_partition_source_path_label_is_readable_not_list_repr():
    """spec 显式给 ``source_path`` 时 label 是路径、不是分组键：文案不能泄漏 list repr（DAIR-205）。"""
    facts = _facts(
        **{
            "output/z.shp": _vector(
                area_total=100_000_000.0,
                group_sums={"AREA_M2_BY_ZONE": {"A": 60_000_000.0, "B": 40_000_000.0}},
            )
        }
    )
    specs = [
        {"type": "conservation.area_partition", "parts": "parts",
         "total_path": ["datasets", "output/z.shp", "area_total"],
         "source_path": ["datasets", "output/z.shp", "group_sums", "AREA_M2_BY_ZONE"],
         "unit": "m2", "total_unit": "m2"},
    ]
    # 数组 parts：没标签可比，只能给 warning；来源是 source_path，不能叫「面积分组键」
    claim = ReviewClaim(metrics={"parts": [60_000_000.0, 40_000_000.0]}, units={"area": "m2"})
    report = run_assertions(specs, facts, claim)
    assert _codes(report) == ["conservation.partition_unverified"], report.to_dict()
    detail = report.findings[0].detail
    assert "source_path 量测（datasets.output/z.shp.group_sums.AREA_M2_BY_ZONE）" in detail, detail
    assert "面积分组键" not in detail, detail
    assert "['" not in detail, detail

    # 字典 parts：逐区对不上时同样不能把路径 str() 成 list repr（改前就有）
    claim = ReviewClaim(
        metrics={"parts": {"A": 30_000_000.0, "B": 70_000_000.0}}, units={"area": "m2"}
    )
    report = run_assertions(specs, facts, claim)
    assert _codes(report) == ["conservation.partition_mismatch"], report.to_dict()
    detail = report.findings[0].detail
    assert "分区面积逐区与 datasets.output/z.shp.group_sums.AREA_M2_BY_ZONE 对不上" in detail, detail
    assert "['" not in detail, detail


def test_area_partition_one_coincidental_label_does_not_skip_per_part_check():
    """KIND 里恰好有一个标签也叫 ``A``，不代表 ZONE 的分区可以不用核（DAIR-201）。

    ``5f80406c`` 下 F/G 两种情形都只剩一条「只有总面积…让 collect_facts 声明 group_fields」
    的 warning——group_sums 明明在、group_fields 也声明了，claim 分区全错反而报 ok。
    """
    facts = _facts(
        **{
            "output/zones.shp": _vector(
                area_total=100_000_000.0,
                group_sums={
                    "AREA_M2_BY_ZONE": {"A": 30_000_000.0, "B": 30_000_000.0, "C": 40_000_000.0},
                    "AREA_M2_BY_KIND": {"A": 999.0, "已建": 60_000_000.0},
                },
            )
        }
    )
    specs = [
        {"type": "conservation.area_partition", "parts": "zone_area",
         "total_path": ["datasets", "output/zones.shp", "area_total"],
         "unit": "m2", "total_unit": "m2"},
    ]
    # F：claim 分区 A/C 对调了一半，合计仍 100 km²
    claim = ReviewClaim(
        metrics={"zone_area": {"A": 60_000_000.0, "B": 30_000_000.0, "C": 10_000_000.0}},
        units={"area": "m2"},
    )
    report = run_assertions(specs, facts, claim)
    assert not report.ok, report.to_dict()
    assert _codes(report) == ["conservation.partition_mismatch"], report.to_dict()
    assert "AREA_M2_BY_ZONE" in report.findings[0].detail

    # G：claim 写对了，逐区核对就该干净通过，不该再有一条误导 warning
    claim = ReviewClaim(
        metrics={"zone_area": {"A": 30_000_000.0, "B": 30_000_000.0, "C": 40_000_000.0}},
        units={"area": "m2"},
    )
    report = run_assertions(specs, facts, claim)
    assert report.ok, report.to_dict()
    assert not report.findings, report.to_dict()


def test_area_partition_array_parts_with_multiple_group_keys_lists_candidates():
    """claim 的 ``parts`` 是数组、量测有两个撞名分组键：warning 要列候选（DAIR-199 项 1）。

    DAIR-197 之前 ``_measured_parts`` 多候选返回 ``None``，走的是「只有总面积…让 collect_facts
    声明 group_fields」那句——group_sums 明明在、group_fields 也声明了，第一条建议是空操作。
    """
    facts = _facts(
        **{
            "output/z.shp": _vector(
                area_total=100_000_000.0,
                group_sums={
                    "AREA_M2_BY_ZONE": {"A": 50_000_000.0, "B": 50_000_000.0},
                    "AREA_M2_BY_KIND": {"A": 20_000_000.0, "B": 80_000_000.0},
                },
            )
        }
    )
    specs = [
        {"type": "conservation.area_partition", "parts": "parts",
         "total_path": ["datasets", "output/z.shp", "area_total"],
         "unit": "m2", "total_unit": "m2"},
    ]
    claim = ReviewClaim(metrics={"parts": [60_000_000.0, 40_000_000.0]}, units={"area": "m2"})
    report = run_assertions(specs, facts, claim)
    assert report.ok, report.to_dict()
    assert _codes(report) == ["conservation.partition_unverified"], report.to_dict()
    detail = report.findings[0].detail
    assert "AREA_M2_BY_ZONE" in detail and "AREA_M2_BY_KIND" in detail, detail
    assert "分不清该比哪个" in detail, detail
    assert "只有总面积" not in detail, detail
    assert "声明 group_fields" not in detail, detail


def test_area_partition_declared_group_field_disambiguates_swapped_partitions():
    """spec 声明 ``group_field`` 后按它比，撞名时的分区互换才抓得到（DAIR-199 项 3）。

    没声明时只能按 claim 的值挑键：claim 与 ZONE 的数值一致、与真正要报的 KIND 对调，
    会被当成「对上了」静默通过。声明 KIND 后必须拿 ``AREA_M2_BY_KIND`` 逐区比。
    """
    facts = _facts(
        **{
            "output/z.shp": _vector(
                area_total=100_000_000.0,
                group_sums={
                    "AREA_M2_BY_ZONE": {"A": 50_000_000.0, "B": 50_000_000.0},
                    "AREA_M2_BY_KIND": {"A": 20_000_000.0, "B": 80_000_000.0},
                },
            )
        }
    )
    base_spec = {
        "type": "conservation.area_partition", "parts": "kind_area",
        "total_path": ["datasets", "output/z.shp", "area_total"],
        "unit": "m2", "total_unit": "m2",
    }
    # 实际 KIND 是 A:20M / B:80M，claim 整段对调（合计仍 100M）
    claim = ReviewClaim(
        metrics={"kind_area": {"A": 50_000_000.0, "B": 50_000_000.0}}, units={"area": "m2"}
    )
    # 没声明 group_field：按值挑到 ZONE → 静默通过（已知限制，docs/review.md 有记）
    assert run_assertions([base_spec], facts, claim).ok

    report = run_assertions([dict(base_spec, group_field="KIND")], facts, claim)
    assert not report.ok, report.to_dict()
    assert _codes(report) == ["conservation.partition_mismatch"], report.to_dict()
    assert "AREA_M2_BY_KIND" in report.findings[0].detail, report.findings[0].detail

    # 声明的分组字段没量到：不拿别的键凑数，给一条说得清的 warning
    zone_only = _facts(
        **{
            "output/z.shp": _vector(
                area_total=100_000_000.0,
                group_sums={"AREA_M2_BY_ZONE": {"A": 50_000_000.0, "B": 50_000_000.0}},
            )
        }
    )
    report = run_assertions([dict(base_spec, group_field="KIND")], zone_only, claim)
    assert report.ok, report.to_dict()
    assert _codes(report) == ["conservation.partition_unverified"], report.to_dict()
    detail = report.findings[0].detail
    assert "按 KIND 分组" in detail and "AREA_M2_BY_ZONE" in detail, detail
    assert "只有总面积" not in detail, detail


def test_area_partition_declared_group_field_present_but_labels_disjoint():
    """声明的键存在、只是分区标签与 claim 无交集：文案要说「键在、标签对不上」（DAIR-209）。

    ``AREA_M2_BY_KIND`` 因为在标签过滤阶段被摘掉，声明的字段看起来像没量到；
    这时候回落成「量测里没有对应的面积分组键」会把人支到错方向。
    """
    facts = _facts(
        **{
            "output/z.shp": _vector(
                area_total=100_000_000.0,
                group_sums={
                    "AREA_M2_BY_ZONE": {"A": 50_000_000.0, "B": 50_000_000.0},
                    "AREA_M2_BY_KIND": {"X": 20_000_000.0, "Y": 80_000_000.0},
                },
            )
        }
    )
    base_spec = {
        "type": "conservation.area_partition", "parts": "kind_area",
        "total_path": ["datasets", "output/z.shp", "area_total"],
        "unit": "m2", "total_unit": "m2",
    }
    claim = ReviewClaim(
        metrics={"kind_area": {"A": 50_000_000.0, "B": 50_000_000.0}}, units={"area": "m2"}
    )
    report = run_assertions([dict(base_spec, group_field="KIND")], facts, claim)
    assert report.ok, report.to_dict()
    assert _codes(report) == ["conservation.partition_unverified"], report.to_dict()
    detail = report.findings[0].detail
    assert "AREA_M2_BY_KIND" in detail and "没有交集" in detail, detail
    assert "没有对应的面积分组键" not in detail, detail


def test_area_partition_declared_group_field_only_key_labels_disjoint():
    """只有声明的那个键、标签又与 claim 无交集：文案仍要说「没有交集」（DAIR-214）。

    DAIR-209 只覆盖了「还有另一个能匹配 claim 标签的键」的情形；声明的键一旦被标签过滤
    摘干净，``matches`` 先空掉，就回落成「只有总面积…让 collect_facts 声明 group_fields」
    ——可那时量测确实按 KIND 分过组，这条建议是空的。
    """
    facts = _facts(
        **{
            "output/z.shp": _vector(
                area_total=100_000_000.0,
                group_sums={"AREA_M2_BY_KIND": {"X": 20_000_000.0, "Y": 80_000_000.0}},
            )
        }
    )
    spec = {
        "type": "conservation.area_partition", "parts": "kind_area",
        "total_path": ["datasets", "output/z.shp", "area_total"],
        "unit": "m2", "total_unit": "m2", "group_field": "KIND",
    }
    claim = ReviewClaim(
        metrics={"kind_area": {"A": 50_000_000.0, "B": 50_000_000.0}}, units={"area": "m2"}
    )
    report = run_assertions([spec], facts, claim)
    assert report.ok, report.to_dict()
    assert _codes(report) == ["conservation.partition_unverified"], report.to_dict()
    detail = report.findings[0].detail
    assert "AREA_M2_BY_KIND" in detail and "没有交集" in detail, detail
    assert "只有总面积" not in detail, detail


def test_measure_code_keeps_zero_area_group(monkeypatch):
    """量测阶段不能把 0 面积分区丢掉：丢了之后 claim 写 0 反被判「多出分区」（DAIR-195）。"""

    class _Geom:
        def __init__(self, area):
            self.area = area

    class _Field:
        def __init__(self, name, kind):
            self.name, self.type = name, kind

    class _Desc:
        dataType, shapeType = "ShapeFile", "Polygon"
        spatialReference = type("SR", (), {
            "type": "Projected", "linearUnitName": "Meter", "factoryCode": 32650, "name": "UTM 50N"
        })()

    class _Cursor:
        def __enter__(self):
            return iter([(_Geom(100.0), 10, "A"), (_Geom(0.0), 20, "B")])

        def __exit__(self, *args):
            return False

    fake = types.ModuleType("arcpy")
    fake.Describe = lambda p: _Desc()
    fake.ListFields = lambda p: [_Field("POP", "Integer"), _Field("ZONE", "String")]
    fake.env = types.SimpleNamespace(scratchGDB=None, workspace=None)
    fake.da = types.SimpleNamespace(SearchCursor=lambda p, fields: _Cursor())
    monkeypatch.setitem(sys.modules, "arcpy", fake)

    code = (
        facts_mod._MEASURE_CODE.replace("__PATHS__", '["fake.shp"]')
        .replace("__GROUPS__", json.dumps({"fake.shp": {"POP": "ZONE"}}))
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exec(compile(code, "<review-measure>", "exec"), {"__name__": "__measure__"})
    entry = json.loads(buffer.getvalue())["fake.shp"]
    assert entry["group_sums"]["AREA_M2"] == {"A": 100.0, "B": 0.0}

    facts = {"datasets": {"fake.shp": entry}, "paths": ["fake.shp"], "collector": ["arcpy"]}
    specs = [
        {"type": "conservation.area_partition", "parts": "zone_area",
         "total_path": ["datasets", "fake.shp", "area_total"],
         "unit": "m2", "total_unit": "m2"},
    ]
    claim = ReviewClaim(metrics={"zone_area": {"A": 100.0, "B": 0.0}}, units={"area": "m2"})
    report = run_assertions(specs, facts, claim)
    assert report.ok and not report.findings, report.to_dict()


# ------------------------------------------------- DAIR-193：密度自动勾稽误杀
def test_density_denominator_accepts_any_declared_dataset_area():
    """两个数据集都有 area_total：结论的 12.8 km² 是第二个，不能拿第一个 48 km² 判死。"""
    facts = _facts(
        **{
            "output/result.shp": _vector(area_total=48_000_000.0),
            "output/green_equity.shp": _vector(area_total=12_800_000.0),
        }
    )
    claim = ReviewClaim(
        metrics={"pop": 320_000, "area_km2": 12.8, "density_per_km2": 25_000.0},
        units={"area": "km2"},
        data_paths=["output/result.shp", "output/green_equity.shp"],
    )
    report = run_assertions([DENSITY], facts, claim)
    assert report.ok and not report.error_findings, report.to_dict()
    assert report.checked == ["dimension.density"]

    # 全都不符时用最接近的候选报告（12.8 而不是遍历顺序上的 48）
    claim.metrics.update({"area_km2": 20.0, "density_per_km2": 16_000.0})
    report = run_assertions([DENSITY], facts, claim)
    assert _codes(report) == ["dimension.density_mismatch"]
    assert report.findings[0].measured == 12.8
    assert "green_equity.shp" in report.findings[0].detail

    # 显式 area_source_path 才按 error 判死
    explicit = dict(DENSITY, area_source_path=["datasets", "output/green_equity.shp", "area_total"])
    report = run_assertions([explicit], facts, claim)
    assert report.findings[0].severity == "error" and not report.ok


def test_density_numerator_ignores_unclaimed_datasets_and_downgrades_same_name():
    """人口来自外部表格：未声明的数据集里同名 POP 不参与对账，但要提示而不是静默通过；
    声明了的子集字段同名不同量，只能给 warning。"""
    facts = _facts(
        **{
            "output/study_area.shp": _vector(count=1),
            "output/green_blocks.shp": _vector(count=8, fields={"POP": {"count": 8, "sum": 1_200_000.0}}),
        }
    )
    claim = ReviewClaim(
        metrics={"pop": 8_420_000, "area_km2": 152.34, "density_per_km2": 8_420_000 / 152.34},
        units={"area": "km2"},
        data_paths=["output/study_area.shp"],
    )
    report = run_assertions([DENSITY], facts, claim)
    # 域内 0 候选、域外有同名量测：不静默通过，也不报错（DAIR-195）
    assert report.ok and _codes(report) == ["dimension.density_unverified"], report.to_dict()
    assert report.findings[0].severity == "warning"
    assert "green_blocks.shp" in report.findings[0].detail
    assert "未参与对账" in report.findings[0].detail

    # 声明的路径写法对不上 facts 的键（绝对/相对路径不同）：退回扫全部，但只给 warning，不静默跳过
    claim.data_paths = ["./output/study_area.shp"]
    report = run_assertions([DENSITY], facts, claim)
    assert report.ok and report.findings[0].severity == "warning"

    # 同一数据集内的子集字段（绿地服务范围内人口 vs 研究区人口）：同名不同量，
    # 启发式分不清，只能给 warning——不能把干净样本判死，也不能静默不报。
    claim.data_paths = ["output/study_area.shp", "output/green_blocks.shp"]
    report = run_assertions([DENSITY], facts, claim)
    assert report.ok and not report.error_findings, report.to_dict()
    assert _codes(report) == ["dimension.density_mismatch"]
    assert report.findings[0].severity == "warning"

    # 显式指定口径（或 cross_check="strict"）才按 error 判死
    strict = dict(DENSITY, cross_check="strict")
    report = run_assertions([strict], facts, claim)
    assert report.findings[0].severity == "error" and not report.ok


def test_density_denominator_out_of_scope_area_is_reported():
    """分母（面积）侧域内 0 候选、域外有量测时不能静默通过（DAIR-199 项 2）。"""
    facts = _facts(
        **{
            "output/study_area.shp": _vector(count=1, area_total=None),
            "output/reference.shp": _vector(count=1, area_total=STUDY_AREA_M2),
        }
    )
    claim = ReviewClaim(
        metrics={"pop": 8_420_000, "area_km2": 152.34, "density_per_km2": 8_420_000 / 152.34},
        units={"area": "km2"},
        data_paths=["output/study_area.shp"],
    )
    report = run_assertions([DENSITY], facts, claim)
    assert report.ok and not report.error_findings, report.to_dict()
    assert _codes(report) == ["dimension.density_unverified"], report.to_dict()
    assert report.findings[0].severity == "warning"
    assert "reference.shp" in report.findings[0].detail
    assert "未参与对账" in report.findings[0].detail

    # 把该数据集加进 claim 就正常对账，不再有提示
    claim.data_paths = ["output/study_area.shp", "output/reference.shp"]
    report = run_assertions([DENSITY], facts, claim)
    assert not report.findings, report.to_dict()

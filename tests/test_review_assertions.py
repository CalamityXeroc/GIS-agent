# -*- coding: utf-8 -*-
"""断言库 + 量测事实 + W5 注入基准（全部纯 Python，不需要 ArcGIS）。"""

from __future__ import annotations

import json
import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

from gis_cli.review import (
    ASSERTIONS,
    ReviewClaim,
    collect_facts,
    crs_facts_from_epsg,
    run_assertions,
)
from gis_cli.review import facts as facts_mod
from gis_cli.review.eval import build_cases, run_eval


def _facts(**datasets):
    return {"datasets": datasets, "paths": list(datasets), "collector": ["arcpy"]}


def _vector(**extra):
    base = {
        "exists": True,
        "kind": "vector",
        "collector": "arcpy",
        "count": 4,
        "crs": {"wkid": 32610, "name": "UTM 10N", "is_projected": True, "linear_unit": "Meter"},
        "area_total": 100_000_000.0,
        "area_unit": "m2",
        "bbox": [500000, 4000000, 510000, 4010000],
        "invalid_count": 0,
        "overlap_ratio": 0.0,
    }
    base.update(extra)
    return base


def _codes(report):
    return [item.code for item in report.findings]


# ------------------------------------------------------------------ 目录
def test_catalog_covers_three_families_and_is_recipe_free():
    families = {item.family for item in ASSERTIONS.values()}
    assert families == {"conservation", "topology", "dimension"}
    for family in families:
        assert sum(1 for item in ASSERTIONS.values() if item.family == family) >= 4
    # “独立调用、不依赖配方”的硬约束：断言模块只 import 标准库 + claims
    source = Path(facts_mod.__file__).with_name("assertions.py").read_text(encoding="utf-8")
    assert "from ..recipes" not in source and "import recipes" not in source
    assert "import arcpy" not in source


# ------------------------------------------------------------------ 守恒
def test_conservation_sum_catches_parts_not_adding_up():
    facts = _facts(**{"output/x.shp": _vector()})
    claim = ReviewClaim(metrics={"total": 100, "parts": [30, 30, 40]})
    clean = run_assertions(
        [{"type": "conservation.sum_equals", "total": "total", "parts": "parts"}], facts, claim
    )
    assert clean.ok and clean.checked == ["conservation.sum_equals"]

    claim = ReviewClaim(metrics={"total": 100, "parts": [30, 35, 40]})
    report = run_assertions(
        [{"type": "conservation.sum_equals", "total": "total", "parts": "parts"}], facts, claim
    )
    assert _codes(report) == ["conservation.sum_mismatch"]
    assert not report.ok


def test_conservation_partition_catches_moved_population_that_keeps_total():
    source = {"A": 100.0, "B": 150.0, "C": 50.0}
    facts = _facts(**{"output/x.shp": _vector(group_sums={"POP": source})})
    specs = [
        {
            "type": "conservation.partition_matches_source",
            "metric": "zone_pop",
            "source_path": ["datasets", "output/x.shp", "group_sums", "POP"],
        }
    ]
    moved = ReviewClaim(metrics={"zone_pop": {"A": 120.0, "B": 130.0, "C": 50.0}})
    report = run_assertions(specs, facts, moved)
    assert _codes(report) == ["conservation.partition_mismatch"]
    assert "A" in report.findings[0].detail and "B" in report.findings[0].detail

    missing = ReviewClaim(metrics={"zone_pop": {"A": 100.0, "B": 150.0}})
    assert _codes(run_assertions(specs, facts, missing)) == ["conservation.partition_mismatch"]


def test_conservation_area_partition_and_count():
    facts = _facts(**{"output/z.shp": _vector(area_total=100_000_000.0, count=4)})
    specs = [
        {
            "type": "conservation.area_partition",
            "parts": "zone_area",
            "total_path": ["datasets", "output/z.shp", "area_total"],
        },
        {
            "type": "conservation.count_matches",
            "metric": "feature_count",
            "source_path": ["datasets", "output/z.shp", "count"],
        },
    ]
    ok = ReviewClaim(
        metrics={"zone_area": {"A": 30.0, "B": 30.0, "C": 40.0}, "feature_count": 4},
        units={"area": "km2"},
    )
    assert run_assertions(specs, facts, ok).ok
    bad = ReviewClaim(
        metrics={"zone_area": {"A": 60.0, "B": 30.0, "C": 40.0}, "feature_count": 5},
        units={"area": "km2"},
    )
    report = run_assertions(specs, facts, bad)
    assert set(_codes(report)) == {"conservation.area_mismatch", "conservation.count_mismatch"}


def test_conservation_matches_source_uses_scale_and_tolerance():
    facts = _facts(**{"output/t.csv": {"count": 2, "fields": {"POP": {"sum": 1000.0}}}})
    specs = [
        {
            "type": "conservation.matches_source",
            "metric": "total_pop",
            "source_path": ["datasets", "output/t.csv", "fields", "POP", "sum"],
        }
    ]
    assert run_assertions(specs, facts, ReviewClaim(metrics={"total_pop": 1005})).ok  # 0.5% 内
    assert _codes(run_assertions(specs, facts, ReviewClaim(metrics={"total_pop": 1200}))) == [
        "conservation.source_mismatch"
    ]


# ------------------------------------------------------------------ 拓扑
def test_topology_invalid_overlap_gap():
    facts = _facts(**{"output/p.shp": _vector(invalid_count=3, overlap_ratio=0.06, coverage={"gap_ratio": 0.08})})
    specs = [
        {"type": "topology.geometry_valid", "source_path": ["datasets", "output/p.shp", "invalid_count"]},
        {"type": "topology.no_overlap", "source_path": ["datasets", "output/p.shp", "overlap_ratio"]},
        {"type": "topology.coverage", "source_path": ["datasets", "output/p.shp", "coverage"], "max_ratio": 0.02},
    ]
    report = run_assertions(specs, facts, ReviewClaim())
    assert set(_codes(report)) == {"topology.invalid_geometry", "topology.overlap", "topology.coverage_gap"}

    clean = _facts(**{"output/p.shp": _vector(invalid_count=0, overlap_ratio=0.0, coverage={"gap_ratio": 0.0})})
    assert run_assertions(specs, clean, ReviewClaim()).ok


def test_topology_within_extent_and_count_change():
    facts = _facts(
        **{
            "output/out.shp": _vector(bbox=[500000, 4000000, 525000, 4010000], count=1),
            "output/in.shp": _vector(count=118),
            "output/study.shp": _vector(bbox=[500000, 4000000, 520000, 4020000]),
        }
    )
    specs = [
        {
            "type": "topology.within_extent",
            "result_path": ["datasets", "output/out.shp", "bbox"],
            "extent_path": ["datasets", "output/study.shp", "bbox"],
        },
        {
            "type": "topology.count_change",
            "result_path": ["datasets", "output/out.shp", "count"],
            "source_path": ["datasets", "output/in.shp", "count"],
        },
    ]
    report = run_assertions(specs, facts, ReviewClaim())
    assert set(_codes(report)) == {"topology.out_of_extent", "topology.unexplained_count_change"}

    # 给出解释后，要素数变化不再报错（溶解/合并是合法的）
    claim = ReviewClaim(expected={"count_change": "118 个缓冲区溶解为 1 个面"})
    assert _codes(run_assertions(specs, facts, claim)) == ["topology.out_of_extent"]


def test_topology_count_change_requires_quantified_reason():
    """「已按字段去重」这种空口解释不能算解释（DAIR-185）。"""
    facts = _facts(**{"output/raw.shp": _vector(count=118), "output/dedup.shp": _vector(count=113)})
    specs = [
        {
            "type": "topology.count_change",
            "result_path": ["datasets", "output/dedup.shp", "count"],
            "source_path": ["datasets", "output/raw.shp", "count"],
        },
    ]
    ok = ReviewClaim(expected={"count_change": "按 NAME 字段去重，删除 5 条重复记录（118→113）"})
    assert run_assertions(specs, facts, ok).ok

    report = run_assertions(specs, facts, ReviewClaim(expected={"count_change": "已按字段去重"}))
    assert _codes(report) == ["topology.unexplained_count_change"]
    assert "没有给出" in report.findings[0].detail

    # 写了数字但对不上变化（实际差 5，解释写 8）同样不算解释
    wrong = ReviewClaim(expected={"count_change": "按 NAME 去重，删除 8 条"})
    assert _codes(run_assertions(specs, facts, wrong)) == ["topology.unexplained_count_change"]

    assert _codes(run_assertions(specs, facts, ReviewClaim())) == ["topology.unexplained_count_change"]


def test_topology_relation_count_catches_claim_of_full_adjacency():
    """12 个地块只有 3 个相邻，结论写“全部相邻”——库里有相邻数量断言才抓得到（DAIR-185）。"""
    facts = _facts(
        **{"output/parcels.shp": _vector(count=12, relations={"adjacent": {"features": 3, "pairs": 2}})}
    )
    specs = [
        {
            "type": "topology.relation_count",
            "relation": "adjacent",
            "metric": "adjacent_count",
            "source_path": ["datasets", "output/parcels.shp", "relations", "adjacent"],
        },
    ]
    report = run_assertions(specs, facts, ReviewClaim(metrics={"adjacent_count": 3}))
    assert report.checked == ["topology.relation_count"] and report.ok

    report = run_assertions(specs, facts, ReviewClaim(metrics={"adjacent_count": 12}))
    assert _codes(report) == ["topology.relation_mismatch"]
    assert report.findings[0].claimed == 12 and report.findings[0].measured == 3

    # claim 没写数字 → skipped（“没检查”不等于“通过”）
    skipped = run_assertions(specs, facts, ReviewClaim())
    assert skipped.checked == [] and skipped.skipped

    # 纯 Python 回退不量相邻关系 → skipped，不猜
    no_rel = _facts(**{"output/parcels.shp": _vector(count=12)})
    assert run_assertions(specs, no_rel, ReviewClaim(metrics={"adjacent_count": 3})).skipped


# ------------------------------------------------------------------ 量纲
def test_dimension_area_unit_catches_order_of_magnitude():
    facts = _facts(**{"output/a.shp": _vector(area_total=123_400_000.0)})
    specs = [
        {"type": "dimension.area_unit", "metric": "area_km2",
         "source_path": ["datasets", "output/a.shp", "area_total"]},
    ]
    assert run_assertions(specs, facts, ReviewClaim(metrics={"area_km2": 123.4}, units={"area": "km2"})).ok
    report = run_assertions(specs, facts, ReviewClaim(metrics={"area_km2": 1234.0}, units={"area": "km2"}))
    assert _codes(report) == ["dimension.area_unit_mismatch"]


def test_topology_no_overlap_compares_claim_ratio():
    """实测 0.42%、结论写 0.0%：只查上限会放行，必须比对自报比率（DAIR-185）。"""
    facts = _facts(**{"output/service.shp": _vector(overlap_ratio=0.0042)})
    specs = [
        {
            "type": "topology.no_overlap",
            "source_path": ["datasets", "output/service.shp", "overlap_ratio"],
            "max_ratio": 0.01,
        },
    ]
    assert run_assertions(specs, facts, ReviewClaim(metrics={"overlap_ratio": 0.0042})).ok

    report = run_assertions(specs, facts, ReviewClaim(metrics={"overlap_ratio": 0.0}))
    assert _codes(report) == ["topology.overlap_claim_mismatch"]
    assert report.findings[0].claimed == 0.0
    assert abs(report.findings[0].measured - 0.0042) < 1e-9

    # claim 没写比率 → skipped，不默认通过
    skipped = run_assertions(specs, facts, ReviewClaim())
    assert skipped.checked == [] and skipped.skipped

    # 超过上限时仍按量测报错（claim 没写比率也一样）
    over = _facts(**{"output/service.shp": _vector(overlap_ratio=0.062)})
    assert _codes(run_assertions(specs, over, ReviewClaim())) == ["topology.overlap"]


def test_dimension_distance_unit_catches_meter_kilometer():
    """声明 500 米、实测 500 公里——距离量纲断言（DAIR-185）。"""
    facts = _facts(**{"output/buffer.shp": _vector(measured_distance=500_000.0)})
    specs = [
        {
            "type": "dimension.distance_unit",
            "metric": "buffer_distance",
            "source_path": ["datasets", "output/buffer.shp", "measured_distance"],
            "source_unit": "m",
        },
    ]
    assert run_assertions(
        specs, facts, ReviewClaim(metrics={"buffer_distance": 500_000.0}, units={"distance": "m"})
    ).ok

    report = run_assertions(
        specs, facts, ReviewClaim(metrics={"buffer_distance": 500.0}, units={"distance": "m"})
    )
    assert _codes(report) == ["dimension.distance_unit_mismatch"]
    assert report.findings[0].claimed == 500.0 and report.findings[0].measured == 500_000.0

    # 声明 500 km 与量测 500000 m 等价，不能误杀
    assert run_assertions(
        specs, facts, ReviewClaim(metrics={"buffer_distance": 500.0}, units={"distance": "km"})
    ).ok

    # 缺量测距离 / 缺声明单位 → skipped
    no_fact = _facts(**{"output/buffer.shp": _vector()})
    assert run_assertions(
        specs, no_fact, ReviewClaim(metrics={"buffer_distance": 500.0}, units={"distance": "m"})
    ).skipped
    assert run_assertions(specs, facts, ReviewClaim(metrics={"buffer_distance": 500.0})).skipped


def test_dimension_density_uses_same_denominator_unit():
    specs = [
        {
            "type": "dimension.density",
            "density": "density_per_km2",
            "count": "pop",
            "area": "area_km2",
            "per_area_unit": "km2",
        }
    ]
    claim = ReviewClaim(
        metrics={"pop": 320_000, "area_km2": 12.8, "density_per_km2": 25_000.0},
        units={"area": "km2"},
    )
    assert run_assertions(specs, {}, claim).ok
    # 按人/m² 算完贴了 /km² 的标签（大 100 万倍）
    claim.metrics["density_per_km2"] = 25_000_000_000.0
    assert _codes(run_assertions(specs, {}, claim)) == ["dimension.density_mismatch"]


def test_dimension_crs_units_degree_vs_meter():
    projected = _facts(**{"output/p.shp": _vector()})
    geographic = _facts(
        **{"output/g.shp": _vector(crs={"wkid": 4326, "name": "WGS84", "is_projected": False, "linear_unit": "Degree"})}
    )
    spec = [{"type": "dimension.crs_units", "source_path": ["datasets", "output/p.shp", "crs"], "quantity": "area"}]
    assert run_assertions(spec, projected, ReviewClaim(units={"area": "km2"})).ok
    assert _codes(run_assertions(spec, projected, ReviewClaim(units={"area": "degree2"}))) == [
        "dimension.crs_unit_mismatch"
    ]

    spec_geo = [{"type": "dimension.crs_units", "source_path": ["datasets", "output/g.shp", "crs"], "quantity": "area"}]
    assert _codes(run_assertions(spec_geo, geographic, ReviewClaim(units={"area": "km2"}))) == [
        "dimension.crs_unit_mismatch"
    ]


def test_dimension_value_range_and_raster_cell_area():
    facts = _facts(**{"output/r.tif": {"exists": True, "kind": "raster", "collector": "arcpy",
                                       "count": 12_340, "cell_size": 10.0, "stats": {"min": -1.0, "max": 1.0}}})
    specs = [
        {"type": "dimension.value_range", "metric": "max_value",
         "source_path": ["datasets", "output/r.tif", "stats"]},
        {"type": "dimension.raster_cell_area", "metric": "area_km2",
         "count_path": ["datasets", "output/r.tif", "count"],
         "cell_size_path": ["datasets", "output/r.tif", "cell_size"]},
    ]
    claim = ReviewClaim(metrics={"max_value": 1.0, "area_km2": 1.234}, units={"area": "km2"})
    assert run_assertions(specs, facts, claim).ok
    claim.metrics.update({"max_value": 1.8, "area_km2": 12.34})
    assert set(_codes(run_assertions(specs, facts, claim))) == {
        "dimension.value_out_of_range",
        "dimension.cell_area_mismatch",
    }


# ------------------------------------------------------------------ 缺事实/写错
def test_missing_facts_are_skipped_not_passed_and_bad_specs_are_errors():
    report = run_assertions(
        [
            {"type": "dimension.area_unit", "metric": "area_km2",
             "source_path": ["datasets", "missing.shp", "area_total"]},
            {"type": "conservation.sum_equals", "total": "total", "parts": "parts"},
            {"type": "no.such.assertion"},
            {"type": "conservation.count_matches", "metric": "feature_count",
             "source_path": ["datasets", "x", "count"]},
        ],
        _facts(**{"output/x.shp": _vector(count=4)}),
        ReviewClaim(metrics={"total": 10, "parts": [5, 4], "feature_count": "not-a-number", "area_km2": 1.0}),
    )
    # 缺面积事实 → skipped；分项不守恒 → finding；未知断言/类型错误 → errors
    assert report.findings and report.findings[0].code == "conservation.sum_mismatch"
    assert any("missing.shp" in item for item in report.skipped)
    assert any("未知断言类型" in item for item in report.errors)
    assert any("count_matches" in item for item in report.errors)


def test_all_skipped_report_is_not_ok():
    """facts 全缺时「0 条 checked」不能算通过（否则复核者会报 pass）。"""
    report = run_assertions(
        [{"type": "topology.no_overlap", "source_path": ["datasets", "a.shp", "overlap_ratio"]}],
        {"datasets": {"a.shp": {"collector": "unavailable"}}},
        ReviewClaim(goal="g", summary="s"),
    )
    assert report.checked == []
    assert report.ok is False
    assert "未检查" in report.summary and "断言全部通过" not in report.summary


# ------------------------------------------------------------------ 事实采集
def test_collect_facts_from_geojson_and_csv(tmp_path: Path):
    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32610"}},
        "features": [
            {
                "type": "Feature",
                "properties": {"ZONE": "A", "POP": 100},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[500000, 4000000], [501000, 4000000], [501000, 4001000], [500000, 4001000], [500000, 4000000]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"ZONE": "B", "POP": 150},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[501000, 4000000], [502000, 4000000], [502000, 4001000], [501000, 4001000], [501000, 4000000]]],
                },
            },
        ],
    }
    (tmp_path / "zones.geojson").write_text(json.dumps(geojson), encoding="utf-8")
    (tmp_path / "zones.csv").write_text("ZONE,POP\nA,100\nB,150\n", encoding="utf-8")

    facts = collect_facts(["zones.geojson", "zones.csv"], workspace=tmp_path, group_fields={"POP": "ZONE"})
    geo = facts["datasets"]["zones.geojson"]
    assert geo["count"] == 2
    assert geo["crs"]["is_projected"] is True and geo["area_unit"] == "m2"
    assert abs(geo["area_total"] - 2_000_000.0) < 1e-6
    assert geo["group_sums"]["POP"] == {"A": 100.0, "B": 150.0}
    assert facts["datasets"]["zones.csv"]["fields"]["POP"]["sum"] == 250.0

    claim = ReviewClaim(
        metrics={"total_pop": 250, "zone_pop": {"A": 100, "B": 150}, "area_km2": 2.0},
        units={"area": "km2"},
    )
    report = run_assertions(
        [
            {"type": "conservation.sum_equals", "total": "total_pop", "parts": "zone_pop"},
            {"type": "conservation.partition_matches_source", "metric": "zone_pop",
             "source_path": ["datasets", "zones.geojson", "group_sums", "POP"]},
            {"type": "dimension.area_unit", "metric": "area_km2",
             "source_path": ["datasets", "zones.geojson", "area_total"]},
        ],
        facts,
        claim,
    )
    assert report.ok, report.to_dict()


def test_geojson_duplicate_vertices_and_pinch_are_not_self_intersection(tmp_path: Path):
    """真实数据里的「连续重复顶点」和「捏点」不能被当成自相交（否则复核误杀）。

    实测：`中国_省.geojson`（42 个要素）里海南省有 5 个子环带连续重复顶点，
    旧实现把它们全部报成非法几何——干净数据上的假阳性正是 W5 最贵的错。
    """
    duplicate = [[0, 0], [1, 0], [1, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    pinch = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2], [0, 0]]
    bowtie = [[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]
    assert facts_mod._ring_self_intersects(duplicate) is False
    assert facts_mod._ring_self_intersects(pinch) is False
    assert facts_mod._ring_self_intersects(bowtie) is True

    def _feature(ring):
        return {
            "type": "Feature",
            "properties": {"ZONE": "A"},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }

    geojson = {"type": "FeatureCollection", "features": [_feature(duplicate), _feature(pinch), _feature(bowtie)]}
    (tmp_path / "real_shape.geojson").write_text(json.dumps(geojson), encoding="utf-8")
    facts = collect_facts(["real_shape.geojson"], workspace=tmp_path)
    entry = facts["datasets"]["real_shape.geojson"]
    assert entry["count"] == 3
    assert entry["invalid_count"] == 1, entry

    claim = ReviewClaim(metrics={"feature_count": 3}, summary="3 个面")
    report = run_assertions(
        [{"type": "topology.geometry_valid", "source_path": ["datasets", "real_shape.geojson", "invalid_count"]}],
        facts,
        claim,
    )
    assert [item.code for item in report.findings] == ["topology.invalid_geometry"]


def test_collect_facts_marks_missing_and_unmeasured(tmp_path: Path):
    (tmp_path / "x.shp").write_text("not a real shapefile", encoding="utf-8")
    facts = collect_facts(["missing.shp", "x.shp"], workspace=tmp_path)
    assert facts["datasets"]["missing.shp"]["exists"] is False
    # 没有内核时不假装量过 shp：断言看到缺事实会 skipped
    assert facts["datasets"]["x.shp"]["collector"] == "unavailable"
    assert crs_facts_from_epsg(4326)["is_projected"] is False
    assert crs_facts_from_epsg(32610)["linear_unit"] == "Meter"
    # 量测脚本必须至少能编译（ArcPy 端只在真机跑，语法错误要在这里拦住）
    compile(facts_mod._MEASURE_CODE, "<review-measure>", "exec")


def test_measure_code_counts_adjacent_features(monkeypatch):
    """ArcPy 量测脚本里的相邻关系：12 个地块只有 3 个相邻（0-1、1-2），其余孤立。"""

    class _Geom:
        def __init__(self, idx, chain):
            self.idx, self.chain, self.area = idx, chain, 100.0

        def touches(self, other):
            return self.chain and other.chain and abs(self.idx - other.idx) == 1

        def overlaps(self, other):
            return False

        def intersect(self, other, dim):
            raise AssertionError("相邻链不该走到 intersect")

    class _Desc:
        # 真机（ArcGIS Pro 3.6）对 .shp 返回的就是 "ShapeFile"（大写 F），别写成历史拼写
        dataType, shapeType = "ShapeFile", "Polygon"
        spatialReference = type("SR", (), {
            "type": "Projected", "linearUnitName": "Meter", "factoryCode": 32650, "name": "UTM 50N"
        })()

    class _Cursor:
        def __enter__(self):
            return iter([(g,) for g in geoms])

        def __exit__(self, *args):
            return False

    geoms = [_Geom(0, True), _Geom(1, True), _Geom(2, True)] + [_Geom(i, False) for i in range(3, 12)]
    fake = types.ModuleType("arcpy")
    fake.Describe = lambda p: _Desc()
    fake.ListFields = lambda p: []
    fake.env = types.SimpleNamespace(scratchGDB=None, workspace=None)
    fake.da = types.SimpleNamespace(SearchCursor=lambda p, fields: _Cursor())
    monkeypatch.setitem(sys.modules, "arcpy", fake)

    code = facts_mod._MEASURE_CODE.replace("__PATHS__", '["fake.shp"]').replace("__GROUPS__", "{}")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exec(compile(code, "<review-measure>", "exec"), {"__name__": "__measure__"})
    entry = json.loads(buffer.getvalue())["fake.shp"]
    assert entry["count"] == 12
    assert entry["relations"]["adjacent"] == {"features": 3, "pairs": 2}


def test_measure_code_accepts_datatype_spellings(monkeypatch):
    """dataType 的大小写变体都要认（DAIR-188：只认 "Shapefile" 时真机 shp 全落 unknown）。"""

    class _Geom:
        area = 100.0

        def touches(self, other):
            return False

        def overlaps(self, other):
            return False

    class _Cursor:
        def __enter__(self):
            return iter([(_Geom(),)])

        def __exit__(self, *args):
            return False

    for spelling in ("ShapeFile", "Shapefile", "shapefile"):
        desc = type("D", (), {
            "dataType": spelling,
            "shapeType": "Polygon",
            "spatialReference": type("SR", (), {
                "type": "Projected", "linearUnitName": "Meter", "factoryCode": 32650, "name": "UTM 50N"
            })(),
        })()
        fake = types.ModuleType("arcpy")
        fake.Describe = lambda p, _d=desc: _d
        fake.ListFields = lambda p: []
        fake.env = types.SimpleNamespace(scratchGDB=None, workspace=None)
        fake.da = types.SimpleNamespace(SearchCursor=lambda p, fields: _Cursor())
        monkeypatch.setitem(sys.modules, "arcpy", fake)

        code = facts_mod._MEASURE_CODE.replace("__PATHS__", '["fake.shp"]').replace("__GROUPS__", "{}")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exec(compile(code, "<review-measure>", "exec"), {"__name__": "__measure__"})
        entry = json.loads(buffer.getvalue())["fake.shp"]
        assert entry["kind"] == "vector", (spelling, entry)
        assert entry["count"] == 1, (spelling, entry)


# ------------------------------------------------------------------ 注入基准
def test_injection_benchmark_meets_acceptance_thresholds():
    report = run_eval()
    assert report["tasks"] == 32
    assert report["injected"] == 26
    assert report["catch_rate"] >= 0.5, report["misses"]
    assert report["false_kill_rate"] <= 0.1, report["false_positive_cases"]
    # 0 误杀必须是“真检查过”的结果：干净样本至少跑了一条断言
    assert report["unchecked"] == [], report["unchecked"]
    # 同源 20 条 + 整数声明 1 条（DAIR-199）必须全抓；非同源 5 条（w5x_*）至少抓 3 条（DAIR-185）
    legacy = [c for c in report["cases"] if c["injected"] and not c["task"].startswith("w5x_")]
    unseen = [c for c in report["cases"] if c["injected"] and c["task"].startswith("w5x_")]
    assert len(legacy) == 21 and all(c["caught"] for c in legacy), legacy
    assert len(unseen) == 5 and sum(1 for c in unseen if c["caught"]) >= 3, unseen
    for family in ("conservation", "topology", "dimension"):
        assert report["families"].get(family, {}).get("injected", 0) >= 4, report["families"]
    # 每条用例都不该出现“断言自己坏掉”（那会把跑分变成假绿）
    for case in report["cases"]:
        assert not case["clean_errors"], case
        assert not case["faulty_errors"], case
        assert case["clean_findings"] == [], case
    # warning 单列计数，不进 ok 判据（DAIR-195）：干净样本上只出 warning、没出 error 的用例
    hits = [c for c in report["cases"] if c["clean_warnings"] and not c["clean_findings"]]
    assert report["warning_hits"] == len(hits)
    assert {c["task"] for c in report["warning_hit_cases"]} == {c["task"] for c in hits}
    assert all(item["warnings"] for item in report["warning_hit_cases"])
    # warning 不算误杀：warning_hits 改了也不该动误杀名单
    assert all(not item["findings"] for item in report["false_positive_cases"])


def test_eval_samples_are_plausible_not_zeroed():
    """注入的是“看起来合理但违反守恒/拓扑/量纲”的错误，不是把结果改成 0。"""
    cases = build_cases()
    assert len(cases) == 32
    for case in cases:
        if not case.injected:
            continue
        assert case.faulty_claim is not None
        # 注入后的 claim 仍有非零结论数字（“全改成 0”闭眼都能发现，不算有效注入）
        metrics = case.faulty_claim.get("metrics") or {}
        numeric: list[float] = []

        def _walk(value):
            if isinstance(value, dict):
                for item in value.values():
                    _walk(item)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric.append(float(value))

        _walk(metrics)
        assert numeric and any(abs(v) > 0 for v in numeric), case.task_id
        assert case.expected, case.task_id

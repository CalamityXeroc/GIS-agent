# -*- coding: utf-8 -*-
"""W5 验收口径：32 个任务 / 26 个人为注入错误 / 抓出率与误杀率。

为什么这个文件在生产包里：验收指标（抓出 ≥50%、误杀 ≤10%）必须能**随时重跑**，
不能只活在某次 run 的日志里。它不依赖 ArcGIS、不调模型、不读盘——事实与结论都是
构造好的，跑一遍是毫秒级。

口径（与 ``docs/review.md`` 的「验收口径（W5）」一致）：

- **32 个任务**：取 ``benchmark/tasks/`` 里的任务形态（含 1 个 fake-gdb 守卫任务），
  26 个各注入 1 个错误、另 6 个只有干净样本作对照；每个任务都有一份**干净样本**
  （结论与数据一致）；
- **26 个注入错误**：前 20 个各注入 1 个错误（不是把结果改成 0，而是"看起来合理
  但违反守恒/拓扑/量纲"：把 A 区人口搬到 B 区、面积差一个数量级、米/度混用、
  结果面重叠 6%、要素数少了 2 个却没有解释……）；后 5 个（``w5x_*``）是审查侧
  另造的**非同源**形态（分区整段互换、相邻关系、自报重叠率、空口解释、距离量纲），
  用来量断言库对"没见过的错误形态"的抓出率（见 docs/review.md「同源偏倚」）；
  另 1 个是整数人口声明少写 8000（DAIR-199 项 4）；
- **抓出** = 注入样本至少命中一条预期断言；**误杀** = 干净样本出现任何 error 级发现。

注意：这里量的是**断言层**的抓出/误杀，不是 LLM 复核者的端到端表现。复核 subagent
调用的就是这套断言；端到端 LLM 基准需要真实模型预算，不在本文件口径内（见 docs/review.md）。
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping

from .claims import ReviewClaim, ReviewReport
from .assertions import run_assertions

# ------------------------------------------------------------------ 构造器
def _crs(projected: bool = True) -> dict[str, Any]:
    return {
        "wkid": 32610 if projected else 4326,
        "name": "WGS 1984 UTM Zone 10N" if projected else "WGS 1984",
        "is_projected": projected,
        "linear_unit": "Meter" if projected else "Degree",
    }


def _vector(
    count: int,
    *,
    area_m2: float | None = None,
    projected: bool = True,
    invalid: int = 0,
    overlap_ratio: float | None = 0.0,
    bbox: list[float] | None = None,
    fields: Mapping[str, Mapping[str, Any]] | None = None,
    group_sums: Mapping[str, Mapping[str, float]] | None = None,
    relations: Mapping[str, Mapping[str, Any]] | None = None,
    measured_distance: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "exists": True,
        "kind": "vector",
        "collector": "arcpy",
        "count": count,
        "crs": _crs(projected),
        "geometry_types": ["Polygon"],
        "area_unit": "m2" if projected else "deg2",
        "invalid_count": invalid,
        "overlap_ratio": overlap_ratio,
    }
    if area_m2 is not None:
        payload["area_total"] = float(area_m2)
    if bbox is not None:
        payload["bbox"] = bbox
    if fields:
        payload["fields"] = {name: dict(stats) for name, stats in fields.items()}
    if group_sums:
        payload["group_sums"] = {name: dict(values) for name, values in group_sums.items()}
    if relations:
        payload["relations"] = {name: dict(values) for name, values in relations.items()}
    if measured_distance is not None:
        payload["measured_distance"] = float(measured_distance)
    return payload


def _raster(
    count: int,
    cell_size: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    projected: bool = True,
) -> dict[str, Any]:
    return {
        "exists": True,
        "kind": "raster",
        "collector": "arcpy",
        "count": int(count),
        "cell_size": float(cell_size),
        "crs": _crs(projected),
        "stats": {"min": min_value, "max": max_value},
    }


def _table(
    count: int,
    *,
    fields: Mapping[str, Mapping[str, Any]] | None = None,
    group_sums: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "exists": True,
        "kind": "table",
        "collector": "arcpy",
        "count": int(count),
        "crs": {"wkid": None, "name": "", "is_projected": None, "linear_unit": None},
    }
    if fields:
        payload["fields"] = {name: dict(stats) for name, stats in fields.items()}
    if group_sums:
        payload["group_sums"] = {name: dict(values) for name, values in group_sums.items()}
    return payload


def _facts(*items: tuple[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "datasets": {str(path): dict(payload) for path, payload in items},
        "paths": [str(path) for path, _ in items],
        "collector": ["arcpy"],
    }


def _claim(
    goal: str,
    summary: str,
    metrics: Mapping[str, Any],
    *,
    paths: tuple[str, ...] = (),
    area_unit: str = "km2",
    distance_unit: str | None = None,
    expected: Mapping[str, Any] | None = None,
    assertions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    units = {"area": area_unit}
    if distance_unit:
        units["distance"] = distance_unit
    return {
        "goal": goal,
        "summary": summary,
        "data_paths": list(paths),
        "metrics": dict(metrics),
        "units": units,
        "expected": dict(expected or {}),
        "assertions": [dict(item) for item in (assertions or [])],
    }


@dataclass
class EvalCase:
    task_id: str
    title: str
    facts: dict[str, Any]
    claim: dict[str, Any]
    assertions: list[dict[str, Any]]
    #: 注入的错误描述；空 = 这个任务不做注入（只有干净样本）
    injected: str = ""
    family: str = ""
    expected: tuple[str, ...] = ()
    faulty_facts: dict[str, Any] | None = None
    faulty_claim: dict[str, Any] | None = None


def _case(
    task_id: str,
    title: str,
    facts: dict[str, Any],
    claim: dict[str, Any],
    assertions: list[dict[str, Any]],
    *,
    injected: str = "",
    family: str = "",
    expected: tuple[str, ...] = (),
    mutate: Any = None,
) -> EvalCase:
    """``mutate(facts, claim)`` 只改**错误样本**（deepcopy 后）。"""
    claim = dict(claim)
    claim["assertions"] = [dict(item) for item in assertions]
    faulty_facts = faulty_claim = None
    if mutate is not None:
        faulty_facts = copy.deepcopy(facts)
        faulty_claim = copy.deepcopy(claim)
        mutate(faulty_facts, faulty_claim)
    return EvalCase(
        task_id=task_id,
        title=title,
        facts=facts,
        claim=claim,
        assertions=assertions,
        injected=injected,
        family=family,
        expected=expected,
        faulty_facts=faulty_facts,
        faulty_claim=faulty_claim,
    )


# ------------------------------------------------------------------ 32 个任务
def build_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []

    # 1 投影转换（量纲）：面积声明差 1000 倍（m² 当 km²）
    def _mut_projection(facts, claim):
        claim["metrics"]["area_km2"] = 152_340.0

    cases.append(
        _case(
            "am_task1a_projection",
            "投影转换后面积核对",
            _facts(
                (
                    "output/study_area_prj.shp",
                    _vector(1, area_m2=152_340_000.0, bbox=[500000, 4000000, 520000, 4020000]),
                )
            ),
            _claim(
                "把研究区投影到 UTM 10N 并统计面积",
                "投影后研究区面积 152.34 km²，要素 1 个",
                {"area_km2": 152.34, "feature_count": 1},
                paths=("output/study_area_prj.shp",),
            ),
            [
                {"type": "dimension.area_unit", "metric": "area_km2",
                 "source_path": ["datasets", "output/study_area_prj.shp", "area_total"]},
                {"type": "conservation.count_matches", "metric": "feature_count",
                 "source_path": ["datasets", "output/study_area_prj.shp", "count"]},
            ],
            injected="面积声明差 1000 倍：按 m² 的数直接写成了 km²",
            family="dimension",
            expected=("dimension.area_unit_mismatch",),
            mutate=_mut_projection,
        )
    )

    # 2 拓扑检查：3 个自相交面被当成合格
    def _mut_topology(facts, claim):
        facts["datasets"]["output/topology_result.shp"]["invalid_count"] = 3

    cases.append(
        _case(
            "am_task1c_topology",
            "面拓扑检查",
            _facts(("output/topology_result.shp", _vector(12, area_m2=8_000_000.0))),
            _claim(
                "检查结果的拓扑合法性",
                "12 个面全部合法，无自相交",
                {"feature_count": 12},
                paths=("output/topology_result.shp",),
            ),
            [
                {"type": "topology.geometry_valid",
                 "source_path": ["datasets", "output/topology_result.shp", "invalid_count"]},
                {"type": "topology.no_overlap",
                 "source_path": ["datasets", "output/topology_result.shp", "overlap_ratio"]},
            ],
            injected="结果里有 3 个自相交面，结论却写“全部合法”",
            family="topology",
            expected=("topology.invalid_geometry",),
            mutate=_mut_topology,
        )
    )

    # 3 属性连接：结果要素数少 2 个（未解释）
    def _mut_join(facts, claim):
        claim["metrics"]["feature_count"] = 120

    cases.append(
        _case(
            "am_task2a_join_trend",
            "属性连接后要素数核对",
            _facts(
                (
                    "output/join_result.shp",
                    _vector(118, area_m2=4_500_000.0, bbox=[500000, 4000000, 510000, 4010000]),
                )
            ),
            _claim(
                "把人口表连接到街道面并检查丢失",
                "连接后 118 个街道，全部匹配",
                {"feature_count": 118},
                paths=("output/join_result.shp",),
            ),
            [
                {"type": "conservation.count_matches", "metric": "feature_count",
                 "source_path": ["datasets", "output/join_result.shp", "count"]},
            ],
            injected="结果实际 118 个，结论写 120（少掉的 2 个没有解释）",
            family="conservation",
            expected=("conservation.count_mismatch",),
            mutate=_mut_join,
        )
    )

    # 4 邻近筛选：分类计数被搬家（合计仍是 48）
    def _mut_neighbor(facts, claim):
        claim["metrics"]["class_count"] = {"A": 22, "B": 26}

    cases.append(
        _case(
            "am_task2b_neighbor_highlight",
            "邻近筛选后的分类计数",
            _facts(
                (
                    "output/neighbor_highlight.shp",
                    _vector(48, area_m2=2_000_000.0, group_sums={"CLASS": {"A": 20.0, "B": 28.0}}),
                )
            ),
            _claim(
                "统计邻近道路的 POI 分类",
                "共 48 个 POI，A 类 20 个、B 类 28 个",
                {"total": 48, "class_count": {"A": 20, "B": 28}},
                paths=("output/neighbor_highlight.shp",),
            ),
            [
                {"type": "conservation.sum_equals", "total": "total", "parts": "class_count"},
                {"type": "conservation.partition_matches_source", "metric": "class_count",
                 "source_path": ["datasets", "output/neighbor_highlight.shp", "group_sums", "CLASS"]},
            ],
            injected="A 类 2 个被搬到 B 类（合计仍 48，只查总量查不出来）",
            family="conservation",
            expected=("conservation.partition_mismatch",),
            mutate=_mut_neighbor,
        )
    )

    # 5 城市汇总：12 万人口从 Z02 搬到 Z04（总量不变）
    zone_pop = {"Z01": 2_100_000, "Z02": 2_300_000, "Z03": 1_900_000, "Z04": 2_120_000}

    def _mut_city(facts, claim):
        claim["metrics"]["zone_pop"]["Z02"] = 2_180_000
        claim["metrics"]["zone_pop"]["Z04"] = 2_240_000

    cases.append(
        _case(
            "am_task3a_city_aggregate_hotspot",
            "按城市汇总人口",
            _facts(
                (
                    "output/city_summary.shp",
                    _vector(
                        4,
                        area_m2=152_340_000.0,
                        group_sums={"POP": {key: float(value) for key, value in zone_pop.items()}},
                    ),
                )
            ),
            _claim(
                "按城市汇总人口",
                "4 个城市合计 842 万人",
                {"total_pop": 8_420_000, "zone_pop": dict(zone_pop)},
                paths=("output/city_summary.shp",),
            ),
            [
                {"type": "conservation.sum_equals", "total": "total_pop", "parts": "zone_pop"},
                {"type": "conservation.partition_matches_source", "metric": "zone_pop",
                 "source_path": ["datasets", "output/city_summary.shp", "group_sums", "POP"]},
            ],
            injected="Z02 的 12 万人被搬到 Z04，合计仍是 842 万",
            family="conservation",
            expected=("conservation.partition_mismatch",),
            mutate=_mut_city,
        )
    )

    # 6 相似性指标：最大值超出 [0,1]
    def _mut_similarity(facts, claim):
        claim["metrics"]["max_value"] = 1.87

    cases.append(
        _case(
            "am_task3b_census_indicators_similarity",
            "普查指标相似性",
            _facts(
                (
                    "output/similarity.shp",
                    _vector(120, area_m2=9_000_000.0, fields={"SIM": {"kind": "numeric", "min": 0.12, "max": 0.87, "sum": 52.0, "mean": 0.43, "count": 120}}),
                )
            ),
            _claim(
                "计算各社区与基准的相似性",
                "相似性最大值 0.87",
                {"max_value": 0.87, "feature_count": 120},
                paths=("output/similarity.shp",),
            ),
            [
                {"type": "dimension.value_range", "metric": "max_value",
                 "source_path": ["datasets", "output/similarity.shp", "fields", "SIM"]},
            ],
            injected="相似性最大值写成 1.87（超出量测上限 0.87，指标本应在 0~1）",
            family="dimension",
            expected=("dimension.value_out_of_range",),
            mutate=_mut_similarity,
        )
    )

    # 7 全流程：分区面积合计大于量测总面积（重复计入）
    def _mut_full(facts, claim):
        claim["metrics"]["zone_area"] = {"A": 60.0, "B": 60.0, "C": 40.0}

    cases.append(
        _case(
            "am_task1_full",
            "全流程分区面积汇总",
            _facts(("output/analysis_zones.shp", _vector(3, area_m2=100_000_000.0))),
            _claim(
                "完成投影、分区与面积汇总",
                "三个分区面积 30/30/40 km²，合计 100 km²",
                {"zone_area": {"A": 30.0, "B": 30.0, "C": 40.0}, "total_area_km2": 100.0},
                paths=("output/analysis_zones.shp",),
            ),
            [
                {"type": "conservation.area_partition", "parts": "zone_area",
                 "total_path": ["datasets", "output/analysis_zones.shp", "area_total"]},
            ],
            injected="分区 A 的面积被重复计入一次，合计 160 km² 而实测总面积只有 100 km²",
            family="conservation",
            expected=("conservation.area_mismatch",),
            mutate=_mut_full,
        )
    )

    # 8 质量检查：面积口径差一个数量级
    def _mut_quality(facts, claim):
        claim["metrics"]["area_km2"] = 1234.0

    cases.append(
        _case(
            "am_task1_quality",
            "成果质量检查中的面积核对",
            _facts(("output/quality_check.shp", _vector(1, area_m2=123_400_000.0))),
            _claim(
                "检查成果面积与字段完整性",
                "成果面积 123.4 km²",
                {"area_km2": 123.4, "feature_count": 1},
                paths=("output/quality_check.shp",),
            ),
            [
                {"type": "dimension.area_unit", "metric": "area_km2",
                 "source_path": ["datasets", "output/quality_check.shp", "area_total"]},
            ],
            injected="面积写成 1234 km²（真实 123.4 km²，差 10 倍）",
            family="dimension",
            expected=("dimension.area_unit_mismatch",),
            mutate=_mut_quality,
        )
    )

    # 9 空间连接：分项合计比总量多 5%（重复匹配）
    def _mut_spatial_join(facts, claim):
        claim["metrics"]["zone_pop"] = {"Z01": 30000, "Z02": 35000, "Z03": 40000}

    cases.append(
        _case(
            "am_task2_join_trend",
            "空间连接后的人口汇总",
            _facts(("output/join_pop.shp", _vector(90, area_m2=50_000_000.0))),
            _claim(
                "把人口点空间连接到街道",
                "三个街道合计 10 万人",
                {"total_pop": 100000, "zone_pop": {"Z01": 30000, "Z02": 35000, "Z03": 35000}},
                paths=("output/join_pop.shp",),
            ),
            [
                {"type": "conservation.sum_equals", "total": "total_pop", "parts": "zone_pop"},
            ],
            injected="Z03 多算 5000 人（同一批点被两个街道重复匹配），合计 105000",
            family="conservation",
            expected=("conservation.sum_mismatch",),
            mutate=_mut_spatial_join,
        )
    )

    # 10 组合任务：土地覆盖分区面积合计对不上
    def _mut_combo(facts, claim):
        claim["metrics"]["landcover_area"] = {"林地": 820.0, "草地": 310.0, "耕地": 240.0}

    cases.append(
        _case(
            "gis14_am_combo_density_landcover",
            "密度图 + 土地覆盖面积组合任务",
            _facts(
                ("output/landcover_area.gdb", _vector(1, area_m2=1_300_000_000.0)),
                ("output/result_data.gdb/kde_1982_1986", _raster(4_000_000, 100.0, min_value=0.0, max_value=6.01)),
            ),
            _claim(
                "组合任务：4 期核密度 + 土地覆盖面积",
                "林地 800、草地 300、耕地 200 km²，合计 1300 km²",
                {
                    "landcover_area": {"林地": 800.0, "草地": 300.0, "耕地": 200.0},
                    "max_value": 6.01,
                },
                paths=("output/landcover_area.gdb", "output/result_data.gdb/kde_1982_1986"),
            ),
            [
                {"type": "conservation.area_partition", "parts": "landcover_area",
                 "total_path": ["datasets", "output/landcover_area.gdb", "area_total"]},
                {"type": "dimension.value_range", "metric": "max_value",
                 "source_path": ["datasets", "output/result_data.gdb/kde_1982_1986", "stats"]},
            ],
            injected="林地面积被多算 20 km²、草地多算 10、耕地多算 40，合计 1370 而实测 1300",
            family="conservation",
            expected=("conservation.area_mismatch",),
            mutate=_mut_combo,
        )
    )

    # 11 动物点位：要素数少 5 个没有解释
    def _mut_animal(facts, claim):
        claim["metrics"]["feature_count"] = 1205

    cases.append(
        _case(
            "gis14_am_task1_animal_points",
            "野生动物点位清洗与统计",
            _facts(("output/animal_points.shp", _vector(1200, area_m2=9_000_000.0))),
            _claim(
                "清洗后的动物点位",
                "共 1200 个有效点位",
                {"feature_count": 1200, "total_count": 1200},
                paths=("output/animal_points.shp",),
            ),
            [
                {"type": "conservation.count_matches", "metric": "feature_count",
                 "source_path": ["datasets", "output/animal_points.shp", "count"]},
            ],
            injected="结论写 1205 个点，结果实际 1200（少掉的 5 个没有说明）",
            family="conservation",
            expected=("conservation.count_mismatch",),
            mutate=_mut_animal,
        )
    )

    # 12 气象插值：最大值超出实测范围
    def _mut_meteo(facts, claim):
        claim["metrics"]["max_value"] = 52.0

    cases.append(
        _case(
            "gis14_am_task2a_meteo_interp",
            "气象要素插值",
            _facts(("output/temp_interp.tif", _raster(2_500_000, 30.0, min_value=3.2, max_value=45.6))),
            _claim(
                "对气温站点做插值",
                "插值结果 3.2~45.6 ℃",
                {"min_value": 3.2, "max_value": 45.6},
                paths=("output/temp_interp.tif",),
            ),
            [
                {"type": "dimension.value_range", "metric": "max_value",
                 "source_path": ["datasets", "output/temp_interp.tif", "stats"]},
            ],
            injected="最大温写成 52.0 ℃（实测上限 45.6 ℃，站点里没有这个值）",
            family="dimension",
            expected=("dimension.value_out_of_range",),
            mutate=_mut_meteo,
        )
    )

    # 13 DEM + 土地覆盖：面积合计与量测总面积差 12%
    def _mut_dem(facts, claim):
        claim["metrics"]["zone_area"] = {"A": 560.0, "B": 500.0, "C": 460.0}

    cases.append(
        _case(
            "gis14_am_task2b_dem_landcover",
            "DEM 与土地覆盖叠加",
            _facts(("output/overlay_area.shp", _vector(3, area_m2=1_200_000_000.0))),
            _claim(
                "按地形分区统计土地覆盖面积",
                "三区面积 500/400/300 km²",
                {"zone_area": {"A": 500.0, "B": 400.0, "C": 300.0}},
                paths=("output/overlay_area.shp",),
            ),
            [
                {"type": "conservation.area_partition", "parts": "zone_area",
                 "total_path": ["datasets", "output/overlay_area.shp", "area_total"]},
            ],
            injected="三区面积被写成 560/500/460，合计 1520 km²，比实测 1200 多 27%",
            family="conservation",
            expected=("conservation.area_mismatch",),
            mutate=_mut_dem,
        )
    )

    # 14 云区修补：像元面积口径差 10 倍
    def _mut_cloud(facts, claim):
        claim["metrics"]["area_km2"] = 18000.0

    cases.append(
        _case(
            "gis14_am_task3a_landcover_cloud_fix",
            "土地覆盖云区修补",
            _facts(("output/landcover_fixed.tif", _raster(2_000_000, 30.0, min_value=1.0, max_value=11.0))),
            _claim(
                "修补云区并统计面积",
                "修补后有效面积 1800 km²",
                {"area_km2": 1800.0, "count": 2_000_000, "cell_size": 30.0},
                paths=("output/landcover_fixed.tif",),
            ),
            [
                {"type": "dimension.raster_cell_area", "metric": "area_km2",
                 "count_path": ["datasets", "output/landcover_fixed.tif", "count"],
                 "cell_size_path": ["datasets", "output/landcover_fixed.tif", "cell_size"]},
            ],
            injected="面积按 300 m 像元算（真实 30 m），1800 km² 变成 18000 km²",
            family="dimension",
            expected=("dimension.cell_area_mismatch",),
            mutate=_mut_cloud,
        )
    )

    # 15 Landsat 更新：总量与源数据差 8%
    def _mut_landsat(facts, claim):
        claim["metrics"]["total_pop"] = 1_120_000

    cases.append(
        _case(
            "gis14_am_task3b_landsat_update",
            "Landsat 影像更新与人口统计",
            _facts(("output/pop_update.dbf", _table(200, fields={"POP": {"kind": "numeric", "sum": 1_200_000.0, "min": 0.0, "max": 30_000.0, "mean": 6000.0, "count": 200}}))),
            _claim(
                "用最新影像更新人口统计",
                "更新后人口 120 万",
                {"total_pop": 1_200_000},
                paths=("output/pop_update.dbf",),
            ),
            [
                {"type": "conservation.matches_source", "metric": "total_pop",
                 "source_path": ["datasets", "output/pop_update.dbf", "fields", "POP", "sum"]},
            ],
            injected="人口合计写成 112 万（源表实际 120 万，少 8%）",
            family="conservation",
            expected=("conservation.source_mismatch",),
            mutate=_mut_landsat,
        )
    )

    # 16 四期密度图：研究区覆盖有 8% 缺口
    def _mut_density_maps(facts, claim):
        facts["datasets"]["output/density_study_area.shp"]["coverage"] = {"gap_ratio": 0.08}

    cases.append(
        _case(
            "gis14_am_task4a1_density_maps",
            "四期密度图的研究区覆盖检查",
            _facts(
                ("output/density_study_area.shp", {**_vector(1, area_m2=150_000_000.0), "coverage": {"gap_ratio": 0.01}}),
                ("output/result_data.gdb/kde_1982_1986", _raster(4_000_000, 100.0, min_value=0.0, max_value=6.01)),
            ),
            _claim(
                "四期核密度图",
                "研究区覆盖完整，四期最大值 6.01",
                {"max_value": 6.01, "coverage_ratio": 0.99},
                paths=("output/density_study_area.shp", "output/result_data.gdb/kde_1982_1986"),
            ),
            [
                {"type": "topology.coverage",
                 "source_path": ["datasets", "output/density_study_area.shp", "coverage"],
                 "metric": "gap_ratio", "max_ratio": 0.02},
                {"type": "dimension.value_range", "metric": "max_value",
                 "source_path": ["datasets", "output/result_data.gdb/kde_1982_1986", "stats"]},
            ],
            injected="研究区覆盖实际有 8% 缺口（上限 2%），结论写“覆盖完整”",
            family="topology",
            expected=("topology.coverage_gap",),
            mutate=_mut_density_maps,
        )
    )

    # 17 中心点迁移：要素数变化没有解释
    def _mut_migration(facts, claim):
        claim["expected"].pop("count_change", None)

    cases.append(
        _case(
            "gis14_am_task4a2_centers_migration",
            "中心点提取与迁移",
            _facts(
                ("output/centers_2020.shp", _vector(118, area_m2=1_200_000.0)),
                ("output/centers_2020_merged.shp", _vector(1, area_m2=1_200_000.0)),
            ),
            _claim(
                "提取 118 个城市中心点并合并",
                "118 个中心点按 1 类合并为 1 个面",
                {"feature_count": 1},
                paths=("output/centers_2020_merged.shp",),
                expected={"count_change": "118 个缓冲区按 1 类溶解，要素数从 118 变 1"},
            ),
            [
                {"type": "topology.count_change",
                 "result_path": ["datasets", "output/centers_2020_merged.shp", "count"],
                 "source_path": ["datasets", "output/centers_2020.shp", "count"]},
            ],
            injected="要素数 118→1 却没有给出溶解/合并解释（claim.expected 里删掉了原因）",
            family="topology",
            expected=("topology.unexplained_count_change",),
            mutate=_mut_migration,
        )
    )

    # 18 密度 + 迁移：密度分母用 m² 当 km²
    def _mut_density(facts, claim):
        claim["metrics"]["density_per_km2"] = 2_500_000_000.0

    cases.append(
        _case(
            "gis14_am_task4a_density_migration",
            "密度分析与迁移图",
            _facts(("output/study_area.shp", _vector(1, area_m2=12_800_000.0))),
            _claim(
                "人口密度与迁移",
                "研究区 32 万人，面积 12.8 km²，密度 25000 人/km²",
                {"pop": 320_000, "area_km2": 12.8, "density_per_km2": 25_000.0},
                paths=("output/study_area.shp",),
            ),
            [
                {"type": "dimension.density", "density": "density_per_km2",
                 "count": "pop", "area": "area_km2", "per_area_unit": "km2"},
                {"type": "dimension.area_unit", "metric": "area_km2",
                 "source_path": ["datasets", "output/study_area.shp", "area_total"]},
            ],
            injected="密度按“人/m²”算后直接贴了 /km² 的标签（大 100 万倍）",
            family="dimension",
            expected=("dimension.density_mismatch",),
            mutate=_mut_density,
        )
    )

    # 19 羚羊土地覆盖：面积被写成 10 倍
    def _mut_gazelle(facts, claim):
        claim["metrics"]["area_km2"] = 12_300.0

    cases.append(
        _case(
            "gis14_am_task4b_gazelle_landcover",
            "羚羊分布区土地覆盖统计",
            _facts(("output/gazelle_landcover.shp", _vector(1, area_m2=1_230_000_000.0))),
            _claim(
                "统计保护区土地覆盖",
                "保护区面积 1230 km²",
                {"area_km2": 1230.0},
                paths=("output/gazelle_landcover.shp",),
            ),
            [
                {"type": "dimension.area_unit", "metric": "area_km2",
                 "source_path": ["datasets", "output/gazelle_landcover.shp", "area_total"]},
            ],
            injected="面积写成 1230 km²（真实 12300 km²，差 10 倍）",
            family="dimension",
            expected=("dimension.area_unit_mismatch",),
            mutate=_mut_gazelle,
        )
    )

    # 20 服务值排名：服务区重叠 6.2%
    def _mut_service(facts, claim):
        facts["datasets"]["output/service_areas.shp"]["overlap_ratio"] = 0.062

    cases.append(
        _case(
            "pm_task2b_service_value_rank",
            "公共服务设施服务区排名",
            _facts(("output/service_areas.shp", _vector(36, area_m2=64_000_000.0, overlap_ratio=0.002))),
            _claim(
                "计算设施服务区并排名",
                "36 个服务区，重叠率 0.2%",
                {"feature_count": 36, "overlap_ratio": 0.002},
                paths=("output/service_areas.shp",),
            ),
            [
                {"type": "topology.no_overlap",
                 "source_path": ["datasets", "output/service_areas.shp", "overlap_ratio"],
                 "max_ratio": 0.01},
                {"type": "conservation.count_matches", "metric": "feature_count",
                 "source_path": ["datasets", "output/service_areas.shp", "count"]},
            ],
            injected="服务区实际重叠 6.2%（上限 1%），结论只写 0.2%",
            family="topology",
            expected=("topology.overlap",),
            mutate=_mut_service,
        )
    )

    # 21 适老化（只有干净样本）：分区人口守恒 + 覆盖率
    elderly_pop = {"东区": 32000, "西区": 28000, "南区": 24000, "北区": 16000}
    cases.append(
        _case(
            "pm_task1_elderly_raster",
            "适老化设施覆盖（干净对照）",
            _facts(
                (
                    "output/elderly_coverage.shp",
                    _vector(
                        4,
                        area_m2=40_000_000.0,
                        group_sums={"POP": {key: float(value) for key, value in elderly_pop.items()}},
                    ),
                )
            ),
            _claim(
                "统计各街道老年人口与设施覆盖",
                "四个街道老年人口合计 10 万，覆盖率 78%",
                {"total_pop": 100000, "zone_pop": dict(elderly_pop), "coverage_ratio": 0.78},
                paths=("output/elderly_coverage.shp",),
            ),
            [
                {"type": "conservation.sum_equals", "total": "total_pop", "parts": "zone_pop"},
                {"type": "conservation.partition_matches_source", "metric": "zone_pop",
                 "source_path": ["datasets", "output/elderly_coverage.shp", "group_sums", "POP"]},
            ],
        )
    )

    # 22 缓冲合并（干净对照）
    cases.append(
        _case(
            "buffer_merge_probe",
            "缓冲区合并（干净对照）",
            _facts(("output/buffer_merged.shp", _vector(1, area_m2=50_200_000.0, bbox=[500000, 4000000, 520000, 4020000]))),
            _claim(
                "合并道路缓冲区",
                "合并后面积 50.2 km²",
                {"area_km2": 50.2, "feature_count": 1},
                paths=("output/buffer_merged.shp",),
            ),
            [
                {"type": "dimension.area_unit", "metric": "area_km2",
                 "source_path": ["datasets", "output/buffer_merged.shp", "area_total"]},
                {"type": "topology.within_extent",
                 "result_path": ["datasets", "output/buffer_merged.shp", "bbox"],
                 "extent_path": ["datasets", "output/buffer_merged.shp", "bbox"]},
            ],
        )
    )

    # 23 绿地公平性（干净对照）
    cases.append(
        _case(
            "pm_task2a_green_area_index",
            "绿地公平性指数（干净对照）",
            _facts(("output/green_equity.shp", _vector(1, area_m2=12_800_000.0))),
            _claim(
                "计算人均绿地面积",
                "绿地面积 12.8 km²，人口 32 万，密度 25000 人/km²",
                {"pop": 320_000, "area_km2": 12.8, "density_per_km2": 25_000.0},
                paths=("output/green_equity.shp",),
            ),
            [
                {"type": "dimension.density", "density": "density_per_km2",
                 "count": "pop", "area": "area_km2", "per_area_unit": "km2"},
                {"type": "dimension.area_unit", "metric": "area_km2",
                 "source_path": ["datasets", "output/green_equity.shp", "area_total"]},
            ],
        )
    )

    # 24 社区地图评价（干净对照）：值域 + 要素数
    cases.append(
        _case(
            "pm_task2c_map_eval_communities",
            "社区地图评价（干净对照）",
            _facts(
                (
                    "output/community_score.shp",
                    _vector(86, area_m2=6_400_000.0, fields={"SCORE": {"kind": "numeric", "min": 0.0, "max": 100.0, "sum": 5200.0, "mean": 60.5, "count": 86}}),
                )
            ),
            _claim(
                "评价社区地图完整性",
                "86 个社区，评分 0~100，平均 60.5",
                {"feature_count": 86, "max_value": 100.0, "min_value": 0.0},
                paths=("output/community_score.shp",),
            ),
            [
                {"type": "conservation.count_matches", "metric": "feature_count",
                 "source_path": ["datasets", "output/community_score.shp", "count"]},
                {"type": "dimension.value_range", "metric": "max_value",
                 "source_path": ["datasets", "output/community_score.shp", "fields", "SCORE"]},
                {"type": "dimension.value_range", "metric": "min_value",
                 "source_path": ["datasets", "output/community_score.shp", "fields", "SCORE"]},
            ],
        )
    )

    # 25 导入检查（干净对照）
    cases.append(
        _case(
            "am_task1b_import",
            "数据导入与字段检查（干净对照）",
            _facts(("output/imported.shp", _vector(118, area_m2=4_500_000.0, fields={"POP": {"kind": "numeric", "min": 0.0, "max": 30000.0, "sum": 8_420_000.0, "mean": 71356.0, "count": 118}}))),
            _claim(
                "导入数据并核对字段",
                "导入 118 个要素，人口合计 842 万",
                {"feature_count": 118, "total_pop": 8_420_000},
                paths=("output/imported.shp",),
            ),
            [
                {"type": "conservation.count_matches", "metric": "feature_count",
                 "source_path": ["datasets", "output/imported.shp", "count"]},
                {"type": "conservation.matches_source", "metric": "total_pop",
                 "source_path": ["datasets", "output/imported.shp", "fields", "POP", "sum"]},
            ],
        )
    )

    # 26 分区人口 A/B 整段互换（合计守恒）——清单只给 sum_equals 时漏，补 partition 断言即抓到
    swap_pop = {"Z01": 2_100_000, "Z02": 2_300_000, "Z03": 1_900_000, "Z04": 2_120_000}

    def _mut_zone_swap(facts, claim):
        zone = claim["metrics"]["zone_pop"]
        zone["Z01"], zone["Z02"] = zone["Z02"], zone["Z01"]
        claim["summary"] = "4 个城市合计 842 万人（Z01 230 万、Z02 210 万）"

    cases.append(
        _case(
            "w5x_task1_zone_pop_swap",
            "按城市汇总人口（A/B 分区整段互换）",
            _facts(
                (
                    "output/city_pop_summary.shp",
                    _vector(
                        4,
                        area_m2=152_340_000.0,
                        group_sums={"POP": {key: float(value) for key, value in swap_pop.items()}},
                    ),
                )
            ),
            _claim(
                "按城市汇总人口",
                "4 个城市合计 842 万人：Z01 210 万、Z02 230 万",
                {"total_pop": 8_420_000, "zone_pop": dict(swap_pop)},
                paths=("output/city_pop_summary.shp",),
            ),
            [
                {"type": "conservation.sum_equals", "total": "total_pop", "parts": "zone_pop"},
                {"type": "conservation.partition_matches_source", "metric": "zone_pop",
                 "source_path": ["datasets", "output/city_pop_summary.shp", "group_sums", "POP"]},
            ],
            injected="Z01/Z02 的人口整段互换（合计仍 842 万，只查总量查不出来）",
            family="conservation",
            expected=("conservation.partition_mismatch",),
            mutate=_mut_zone_swap,
        )
    )

    # 27 12 个地块只有 3 个相邻，结论写“全部相邻”
    def _mut_adjacency(facts, claim):
        claim["metrics"]["adjacent_count"] = 12
        claim["summary"] = "12 个地块全部相邻，连成一整片"

    cases.append(
        _case(
            "w5x_task2_parcel_adjacency",
            "地块相邻关系核对",
            _facts(
                (
                    "output/parcels.shp",
                    _vector(
                        12,
                        area_m2=4_800_000.0,
                        relations={"adjacent": {"features": 3, "pairs": 2}},
                    ),
                )
            ),
            _claim(
                "核对 12 个地块的相邻关系",
                "12 个地块里有 3 个与其他地块相邻，其余 9 个孤立",
                {"feature_count": 12, "adjacent_count": 3},
                paths=("output/parcels.shp",),
            ),
            [
                {"type": "topology.relation_count", "relation": "adjacent",
                 "metric": "adjacent_count", "key": "features",
                 "source_path": ["datasets", "output/parcels.shp", "relations", "adjacent"]},
                {"type": "conservation.count_matches", "metric": "feature_count",
                 "source_path": ["datasets", "output/parcels.shp", "count"]},
            ],
            injected="12 个地块里只有 3 个相邻，结论写“全部相邻”（自报 12 个）",
            family="topology",
            expected=("topology.relation_mismatch",),
            mutate=_mut_adjacency,
        )
    )

    # 28 重叠率实测 0.42%、结论写 0.0%（实测值本身在上限内，只有比对自报值才抓得到）
    def _mut_overlap_claim(facts, claim):
        claim["metrics"]["overlap_ratio"] = 0.0
        claim["summary"] = "36 个服务区，重叠率 0.0%"

    cases.append(
        _case(
            "w5x_task3_overlap_claim_ratio",
            "服务区重叠率自报值核对",
            _facts(
                ("output/service_areas.shp", _vector(36, area_m2=64_000_000.0, overlap_ratio=0.0042))
            ),
            _claim(
                "计算设施服务区重叠率",
                "36 个服务区，重叠率 0.42%",
                {"feature_count": 36, "overlap_ratio": 0.0042},
                paths=("output/service_areas.shp",),
            ),
            [
                {"type": "topology.no_overlap",
                 "source_path": ["datasets", "output/service_areas.shp", "overlap_ratio"],
                 "max_ratio": 0.01, "claim_metric": "overlap_ratio"},
                {"type": "conservation.count_matches", "metric": "feature_count",
                 "source_path": ["datasets", "output/service_areas.shp", "count"]},
            ],
            injected="实测重叠率 0.42%，结论写 0.0%（0.42% 在上限 1% 内，只查上限查不出来）",
            family="topology",
            expected=("topology.overlap_claim_mismatch",),
            mutate=_mut_overlap_claim,
        )
    )

    # 29 要素数 118→113，用“已按字段去重”空口解释
    def _mut_dedup(facts, claim):
        claim["expected"]["count_change"] = "已按字段去重"
        claim["summary"] = "118 个要素按字段去重后剩 113 个"

    cases.append(
        _case(
            "w5x_task4_dedup_count_change",
            "去重后要素数变化的解释",
            _facts(
                ("output/join_raw.shp", _vector(118, area_m2=4_500_000.0)),
                ("output/join_dedup.shp", _vector(113, area_m2=4_400_000.0)),
            ),
            _claim(
                "按名称字段去重并核对要素数",
                "118 个要素按 NAME 去重后剩 113 个（删除 5 条重复记录）",
                {"feature_count": 113},
                paths=("output/join_dedup.shp",),
                expected={"count_change": "按 NAME 字段去重，删除 5 条重复记录（118→113）"},
            ),
            [
                {"type": "topology.count_change",
                 "result_path": ["datasets", "output/join_dedup.shp", "count"],
                 "source_path": ["datasets", "output/join_raw.shp", "count"]},
            ],
            injected="要素数 118→113，解释只写“已按字段去重”（没给出删掉几条，和变化对不上）",
            family="topology",
            expected=("topology.unexplained_count_change",),
            mutate=_mut_dedup,
        )
    )

    # 30 缓冲距离声明 500 米、实测 500 公里（米/公里混用）
    def _mut_buffer_distance(facts, claim):
        claim["metrics"]["buffer_distance"] = 500.0
        claim["summary"] = "按 500 米缓冲道路，合并后 1 个面"

    cases.append(
        _case(
            "w5x_task5_buffer_distance",
            "缓冲区距离口径核对",
            _facts(
                ("output/road_buffer.shp", _vector(1, area_m2=50_200_000.0, measured_distance=500_000.0))
            ),
            _claim(
                "按 500 公里缓冲道路并合并",
                "按 500 公里缓冲，合并后 1 个面",
                {"feature_count": 1, "buffer_distance": 500_000.0},
                paths=("output/road_buffer.shp",),
                distance_unit="m",
            ),
            [
                {"type": "dimension.distance_unit", "metric": "buffer_distance",
                 "source_path": ["datasets", "output/road_buffer.shp", "measured_distance"],
                 "source_unit": "m"},
                {"type": "conservation.count_matches", "metric": "feature_count",
                 "source_path": ["datasets", "output/road_buffer.shp", "count"]},
            ],
            injected="缓冲距离声明 500 米、实测 500 公里（差 1000 倍）",
            family="dimension",
            expected=("dimension.distance_unit_mismatch",),
            mutate=_mut_buffer_distance,
        )
    )

    # 31 整数人口声明（干净对照）：``decimals == 0`` 按 ±0.5 末位判，精确相等必须过
    cases.append(
        _case(
            "gis14_am_task4c_integer_population",
            "整数人口声明（干净对照）",
            _facts(
                (
                    "output/study_area.shp",
                    _vector(
                        1,
                        area_m2=152_340_000.0,
                        fields={"POP": {"count": 1, "sum": 8_420_000.0}},
                    ),
                )
            ),
            _claim(
                "人口密度（整数声明）",
                "研究区 8420000 人，面积 152.34 km²，密度 55271 人/km²",
                {"pop": 8_420_000, "area_km2": 152.34, "density_per_km2": 8_420_000 / 152.34},
                paths=("output/study_area.shp",),
            ),
            [
                {"type": "dimension.density", "density": "density_per_km2",
                 "count": "pop", "area": "area_km2", "per_area_unit": "km2"},
            ],
            family="dimension",
        )
    )

    # 32 整数人口少写 8000（0.095%）：0.1% 地板会放过，±0.5 末位必须抓（DAIR-195 / DAIR-199）
    def _mut_integer_population(facts, claim):
        claim["metrics"]["pop"] = 8_412_000
        claim["metrics"]["density_per_km2"] = 8_412_000 / 152.34

    cases.append(
        _case(
            "gis14_am_task4d_integer_population_truncated",
            "整数人口声明被少写 8000",
            _facts(
                (
                    "output/study_area.shp",
                    _vector(
                        1,
                        area_m2=152_340_000.0,
                        fields={"POP": {"count": 1, "sum": 8_420_000.0}},
                    ),
                )
            ),
            _claim(
                "人口密度（整数声明）",
                "研究区 8420000 人，面积 152.34 km²，密度 55271 人/km²",
                {"pop": 8_420_000, "area_km2": 152.34, "density_per_km2": 8_420_000 / 152.34},
                paths=("output/study_area.shp",),
            ),
            [
                {"type": "dimension.density", "density": "density_per_km2",
                 "count": "pop", "area": "area_km2", "per_area_unit": "km2"},
            ],
            injected="整数声明 8420000 写成 8412000（0.095%）：0.1% 地板放过、整数声明的 ±0.5 末位必须抓",
            family="dimension",
            expected=("dimension.density_mismatch",),
            mutate=_mut_integer_population,
        )
    )

    # 任务数对齐 32：26 个任务各注入 1 个错误（前 20 个同源 + 5 个非同源 + 整数声明 1 个）
    # + 6 个只有干净样本的对照；多写了就报错，免得以后加/删任务时验收口径悄悄漂移。
    if len(cases) != 32:
        raise AssertionError(f"W5 基准应有 32 个任务，实际 {len(cases)} 个")
    return cases


# ------------------------------------------------------------------ 跑分
@dataclass
class CaseResult:
    task_id: str
    title: str
    injected: str = ""
    family: str = ""
    expected: tuple[str, ...] = ()
    clean_codes: list[str] = field(default_factory=list)
    clean_warnings: list[str] = field(default_factory=list)
    faulty_codes: list[str] = field(default_factory=list)
    clean_errors: list[str] = field(default_factory=list)
    faulty_errors: list[str] = field(default_factory=list)
    clean_checked: list[str] = field(default_factory=list)
    faulty_checked: list[str] = field(default_factory=list)

    @property
    def caught(self) -> bool:
        return bool(self.injected) and bool(set(self.expected) & set(self.faulty_codes))

    @property
    def false_kill(self) -> bool:
        return bool(self.clean_codes) or bool(self.clean_errors)


def _run_one(case: EvalCase) -> CaseResult:
    result = CaseResult(
        task_id=case.task_id,
        title=case.title,
        injected=case.injected,
        family=case.family,
        expected=case.expected,
    )
    clean_report: ReviewReport = run_assertions(case.assertions, case.facts, ReviewClaim.from_dict(case.claim))
    # 误杀只看 error 级发现（与 docs/review.md 的口径一致）：warning 是“只校验了一部分”的提示，
    # 不应该算成干净样本被判死（DAIR-191 缺陷 5 引入 conservation.partition_unverified）。
    result.clean_codes = [item.code for item in clean_report.error_findings]
    result.clean_warnings = [item.code for item in clean_report.warning_findings]
    result.clean_errors = list(clean_report.errors)
    result.clean_checked = list(clean_report.checked)
    if case.faulty_facts is not None and case.faulty_claim is not None:
        faulty_report: ReviewReport = run_assertions(
            case.assertions, case.faulty_facts, ReviewClaim.from_dict(case.faulty_claim)
        )
        result.faulty_codes = [item.code for item in faulty_report.findings]
        result.faulty_errors = list(faulty_report.errors)
        result.faulty_checked = list(faulty_report.checked)
    return result


def run_eval() -> dict[str, Any]:
    """跑 32 个任务 / 26 个注入错误，返回可断言/可打印的报告。"""
    cases = build_cases()
    results = [_run_one(case) for case in cases]
    injected = [item for item in results if item.injected]
    caught = [item for item in injected if item.caught]
    false_kills = [item for item in results if item.false_kill]
    misses = [item for item in injected if not item.caught]
    # 干净样本里一条断言都没真跑（全 skipped）= 这条「没误杀」是空话：分母必须是真检查过的。
    unchecked = [item for item in results if not item.clean_checked]
    families: dict[str, dict[str, int]] = {}
    for item in injected:
        bucket = families.setdefault(item.family or "unknown", {"injected": 0, "caught": 0})
        bucket["injected"] += 1
        bucket["caught"] += 1 if item.caught else 0
    # 干净样本上只出 warning、没出 error 的用例数（DAIR-195）：warning 既不算抓出也不算误杀，
    # 但「一条 error 都没有、全靠 warning」的样本对账强度是打折的，要能看见。不改 ok 判据。
    warning_only = [item for item in results if item.clean_warnings and not item.clean_codes]
    return {
        "tasks": len(results),
        "injected": len(injected),
        "caught": len(caught),
        "catch_rate": len(caught) / len(injected) if injected else 0.0,
        "false_kills": len(false_kills),
        "false_kill_rate": len(false_kills) / len(results) if results else 0.0,
        "unchecked": [
            {"task": item.task_id, "injected": item.injected} for item in unchecked
        ],
        "families": families,
        "misses": [
            {"task": item.task_id, "injected": item.injected, "expected": list(item.expected),
             "got": item.faulty_codes}
            for item in misses
        ],
        "false_positive_cases": [
            {"task": item.task_id, "findings": item.clean_codes, "errors": item.clean_errors}
            for item in false_kills
        ],
        "warning_hits": len(warning_only),
        "warning_hit_cases": [
            {"task": item.task_id, "warnings": item.clean_warnings}
            for item in warning_only
        ],
        "cases": [
            {
                "task": item.task_id,
                "injected": item.injected,
                "family": item.family,
                "caught": item.caught,
                "expected": list(item.expected),
                "clean_findings": item.clean_codes,
                "clean_warnings": item.clean_warnings,
                "faulty_findings": item.faulty_codes,
                "clean_errors": item.clean_errors,
                "faulty_errors": item.faulty_errors,
            }
            for item in results
        ],
    }


def _print_report(report: Mapping[str, Any]) -> None:
    print("W5 双复核断言层基准")
    print(f"  任务 {report['tasks']} 个，注入 {report['injected']} 个错误")
    print(
        f"  抓出 {report['caught']}/{report['injected']} = {report['catch_rate'] * 100:.1f}%"
        "（验收 ≥50%）"
    )
    print(
        f"  误杀 {report['false_kills']}/{report['tasks']} = {report['false_kill_rate'] * 100:.1f}%"
        "（验收 ≤10%）"
    )
    print(f"  仅 warning（未判错，不算抓出/误杀） {report['warning_hits']} 条")
    for family, stats in sorted((report.get("families") or {}).items()):
        print(f"    {family}: {stats['caught']}/{stats['injected']}")
    for item in report["misses"]:
        print(f"  [漏] {item['task']}: {item['injected']} → got {item['got']}")
    for item in report["false_positive_cases"]:
        print(f"  [误杀] {item['task']}: {item['findings']} errors={item['errors']}")
    for item in report.get("warning_hit_cases") or []:
        print(f"  [仅 warning] {item['task']}: {item['warnings']}")
    for item in report.get("unchecked") or []:
        print(f"  [未检] {item['task']}: 干净样本一条断言都没跑，0 误杀不能算数")


def main(argv: list[str] | None = None) -> int:
    try:  # Windows 控制台默认 GBK：中文报告不能因为一个生僻字就崩
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    argv = list(sys.argv[1:] if argv is None else argv)
    report = run_eval()
    if "--json" in argv:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    ok = report["catch_rate"] >= 0.5 and report["false_kill_rate"] <= 0.1 and not report["unchecked"]
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover - 手工跑分入口
    raise SystemExit(main())

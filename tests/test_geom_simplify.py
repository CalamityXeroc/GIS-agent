# -*- coding: utf-8 -*-
"""保拓扑简化（DAIR-368）：共享边界不许裂缝、弦不许跨要素、环不许抽空。

这组用例是「预览/报告几何交付」的契约：原来的 ``points[::stride]`` 抽稀会让
相邻要素的共享边界对不上（缝隙）、让环退化成横穿要素的直弦（三角形空洞）。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

from gis_cli.report import geom_simplify, html_pack

#: 真机素材（不入库，缺了就跳过；基准单/本机验收时才有）。
REAL_DATA = (
    Path(__file__).resolve().parents[1]
    / "workspace"
    / "input"
    / "中国标准行政区划数据GS（2024）0650号"
    / "geojson格式"
)


# ------------------------------------------------------------------ 合成数据
def _grid_collection(columns: int = 3, rows: int = 2, detail: int = 40):
    """相邻矩形网格：每个单元格的环都带 ``detail`` 个波浪顶点，边界严格共享。"""
    features = []
    step = 1.0
    for cx in range(columns):
        for cy in range(rows):
            ring = []
            # 下边（左→右）
            for k in range(detail):
                ring.append([cx + step * k / detail, cy])
            # 右边（下→上）：带垂直波浪，模拟真实边界的小抖动
            for k in range(detail):
                ring.append([cx + step, cy + step * k / detail + (0.002 if k % 2 else 0.0)])
            # 上边（右→左）
            for k in range(detail, 0, -1):
                ring.append([cx + step * k / detail, cy + step])
            # 左边（上→下）
            for k in range(detail, 0, -1):
                ring.append([cx, cy + step * k / detail])
            ring.append(list(ring[0]))
            features.append(
                {
                    "type": "Feature",
                    "properties": {"cell": f"{cx}-{cy}"},
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                }
            )
    return {"type": "FeatureCollection", "features": features}


def _rings(geometry):
    kind = geometry.get("type")
    coords = geometry.get("coordinates")
    if kind == "Polygon":
        return list(coords)
    if kind == "MultiPolygon":
        return [ring for polygon in coords for ring in polygon]
    if kind == "LineString":
        return [coords]
    if kind == "MultiLineString":
        return list(coords)
    return []


def _point_segment_distance(point, a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    seg2 = dx * dx + dy * dy
    if seg2 <= 0:
        return math.hypot(point[0] - a[0], point[1] - a[1])
    t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / seg2
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return math.hypot(point[0] - (a[0] + t * dx), point[1] - (a[1] + t * dy))


def _max_deviation(original, simplified):
    """原始顶点到简化折线的最远距离（单向 Hausdorff）。"""
    worst = 0.0
    for f_orig, f_simp in zip(original["features"], simplified["features"]):
        for ring_orig, ring_simp in zip(
            _rings(f_orig["geometry"]), _rings(f_simp["geometry"])
        ):
            segments = list(zip(ring_simp, ring_simp[1:])) or [(ring_simp[0], ring_simp[0])]
            for point in ring_orig:
                best = min(_point_segment_distance(point, a, b) for a, b in segments)
                worst = max(worst, best)
    return worst


def _shared_edge_length(collection, *, exact_pairs: bool = True):
    """两侧端点完全相同的边总长（逐点重合 = 无缝）。"""
    users: dict[tuple, dict[int, float]] = {}
    for index, feature in enumerate(collection["features"]):
        for ring in _rings(feature["geometry"]):
            for a, b in zip(ring, ring[1:]):
                key = (tuple(a), tuple(b))
                reverse = (tuple(b), tuple(a))
                slot = key if key <= reverse else reverse
                length = math.hypot(b[0] - a[0], b[1] - a[1])
                users.setdefault(slot, {})[index] = length
    total = 0.0
    for slot, by_feature in users.items():
        if len(by_feature) >= 2:
            total += min(by_feature.values())
    return total


def _sawtooth_collection(features: int = 60, vertices: int = 2000):
    """锯齿条带：DP 的 O(n²) 杀手（每刀只能切掉一个点）。

    真实行政区划边界不会这么尖，但测绘数据里高密度抖动折线是常见的，所以
    「锯齿不许把预览拖死」是硬要求（DAIR-368：100 万点实测要 8 分钟）。
    """
    items = []
    for f in range(features):
        y0 = f * 2.0
        ring = [[0.0, y0]]
        for i in range(vertices):
            ring.append([i * 0.001, y0 + (0.5 if i % 2 else 0.0)])
        ring.append([vertices * 0.001, y0 + 1.0])
        ring.append([0.0, y0 + 1.0])
        ring.append([0.0, y0])
        items.append(
            {
                "type": "Feature",
                "properties": {"band": f},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return {"type": "FeatureCollection", "features": items}


# ------------------------------------------------------------------ 逐点重合
def test_shared_boundary_stays_identical_after_simplify():
    """相邻面的共享边界必须是**同一份**顶点：两侧逐点相同才算无缝。"""
    collection = _grid_collection(columns=3, rows=2, detail=60)

    simplified, stats = geom_simplify.simplify_feature_collection(
        collection, max_points=400
    )

    assert stats["output_points"] < stats["input_points"], "测试前提：确实做了简化"
    shared_before = _shared_edge_length(collection)
    shared_after = _shared_edge_length(simplified)
    # 简化会裁掉一些拐点（线段略短），但共享边必须仍然**存在且逐点重合**；
    # 一旦两侧各自取样，共享边长度会直接掉到 0（每条边都只剩一个使用者）。
    assert shared_before > 0
    assert shared_after > 0
    assert shared_after >= shared_before * 0.9


def _collapsing_neighbour_collection(detail: int = 60):
    """小环 + 大邻居共享一条锯齿边界：容差一大，小环整圈被简化没了。

    小环只宽 0.02，共用边界上下幅 0.004：容差 0.5 时它剩下的顶点撑不起一个面，
    而大邻居的四个角点都远超过容差、能活下来。
    """
    boundary = [
        [(0.004 if 0 < k < detail and k % 2 else 0.0), k / detail]
        for k in range(detail + 1)
    ]
    small = boundary + [[-0.02, 1.0], [-0.02, 0.0], list(boundary[0])]
    big = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 1.0]] + [
        list(point) for point in reversed(boundary[1:-1])
    ] + [[0.0, 0.0]]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"n": "小"},
                "geometry": {"type": "Polygon", "coordinates": [small]},
            },
            {
                "type": "Feature",
                "properties": {"n": "大"},
                "geometry": {"type": "Polygon", "coordinates": [big]},
            },
        ],
    }


def test_degenerate_ring_rolls_back_its_shared_arcs_too():
    """环被简化到画不出来时，要连它的弧一起回退（DAIR-373）。

    只回退「自己的原环」会让共享边两侧对不上：小环拿的是原始顶点，邻居拿的
    是简化后的弧 → 又出现缝隙。回退弧才能两边逐点相同。
    """
    collection = _collapsing_neighbour_collection()
    shared_before = _shared_edge_length(collection)

    simplified, _ = geom_simplify.simplify_feature_collection(collection, tolerance=0.5)

    assert shared_before > 0
    shared_after = _shared_edge_length(simplified)
    assert shared_after >= shared_before * 0.99, (
        f"退化环的共享边必须两侧逐点相同：原始 {shared_before:.6f} → 简化后 {shared_after:.6f}"
    )
    for feature in simplified["features"]:
        for ring in _rings(feature["geometry"]):
            assert len(ring) >= 4, "面被简化成一条线就画不出来了"
            assert ring[0] == ring[-1], "环必须闭合"


def test_simplify_keeps_ring_closed_and_at_least_four_points():
    collection = _grid_collection(columns=2, rows=2, detail=30)

    simplified, _ = geom_simplify.simplify_feature_collection(collection, max_points=60)

    for feature in simplified["features"]:
        for ring in _rings(feature["geometry"]):
            assert len(ring) >= 4, "面被简化成一条线就画不出来了"
            assert ring[0] == ring[-1], "环必须闭合"


def test_simplify_never_moves_geometry_beyond_tolerance():
    """弦不许跨要素：原始顶点到简化折线的偏差不得超过容差（+ 格网精度）。"""
    collection = _grid_collection(columns=3, rows=2, detail=50)

    for budget in (2000, 400, 120):
        simplified, stats = geom_simplify.simplify_feature_collection(
            collection, max_points=budget
        )
        deviation = _max_deviation(collection, simplified)
        # 吸附到格网本身也会挪一点，所以上限是「容差 + 精度」。
        assert deviation <= stats["tolerance"] + stats["precision"] + 1e-12, (
            f"budget={budget} 偏差 {deviation} 超过容差 {stats['tolerance']}"
        )


def test_vertex_budget_is_respected_or_close():
    collection = _grid_collection(columns=4, rows=3, detail=80)

    simplified, stats = geom_simplify.simplify_feature_collection(
        collection, max_points=900
    )

    assert stats["output_points"] <= 900 * 1.35, (
        "预算允许小超（环有最少点数下限），但不能回到原样"
    )


def test_attributes_and_part_structure_are_untouched():
    collection = _grid_collection(columns=2, rows=1, detail=20)
    # 加一个洞 + 一个多面，确认结构（环数/部件数）不被简化改写
    ring_outer = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    ring_hole = [[4, 4], [4, 6], [6, 6], [6, 4], [4, 4]]
    collection["features"].append(
        {
            "type": "Feature",
            "properties": {"cell": "带洞"},
            "geometry": {"type": "Polygon", "coordinates": [ring_outer, ring_hole]},
        }
    )
    collection["features"].append(
        {
            "type": "Feature",
            "properties": {"cell": "多面"},
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [[ring_outer], [[[20, 0], [22, 0], [22, 2], [20, 0]]]],
            },
        }
    )
    collection["features"].append(
        {
            "type": "Feature",
            "properties": {"cell": "点"},
            "geometry": {"type": "Point", "coordinates": [1.5, 1.5]},
        }
    )

    simplified, _ = geom_simplify.simplify_feature_collection(
        collection, max_points=500
    )

    assert [f["properties"] for f in simplified["features"]] == [
        f["properties"] for f in collection["features"]
    ]
    hole_feature = simplified["features"][-3]
    assert hole_feature["geometry"]["type"] == "Polygon"
    assert len(hole_feature["geometry"]["coordinates"]) == 2, "洞不能被合并或丢掉"
    multi = simplified["features"][-2]
    assert multi["geometry"]["type"] == "MultiPolygon"
    assert len(multi["geometry"]["coordinates"]) == 2
    point = simplified["features"][-1]
    assert point["geometry"] == {"type": "Point", "coordinates": [1.5, 1.5]}, (
        "点类几何原样保留"
    )


def test_explicit_tolerance_skips_budget_search():
    collection = _grid_collection(columns=2, rows=2, detail=40)

    simplified, stats = geom_simplify.simplify_feature_collection(
        collection, tolerance=0.05
    )

    assert stats["passes"] == 1
    assert stats["tolerance"] == pytest.approx(0.05)
    assert _max_deviation(collection, simplified) <= 0.05 + stats["precision"] + 1e-12


def test_esri_geometries_get_simplified_and_lose_spatial_reference():
    """内核侧入口：rings/paths 简化 + 丢掉 spatialReference（少传冗余字节）。"""
    rings = [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]
    geoms = [
        {"rings": rings, "spatialReference": {"wkid": 4326}},
        {"paths": [[[0, 0], [1, 1]]], "spatialReference": {"wkid": 4326}},
        {"x": 3, "y": 4},
        "not-a-geometry",
    ]

    out, stats = geom_simplify.simplify_esri_geometries(geoms, max_points=100)

    assert "spatialReference" not in out[0] and "rings" in out[0]
    assert "spatialReference" not in out[1] and "paths" in out[1]
    assert out[2] == {"x": 3, "y": 4}, "点类几何原样保留"
    assert out[3] == "not-a-geometry", "认不出的输入不能被吞掉"
    assert stats["arcs"] >= 1


def _edges_on(ring, *, axis: int, value: float) -> set:
    """环上落在 ``axis == value`` 一侧的边（无序对）。

    环从哪个顶点起画由弧序列决定（可能被旋转），所以这里按边集合比，
    而不是拿顶点串首尾对齐。
    """
    edges = set()
    for a, b in zip(ring, ring[1:]):
        if a[axis] == value and b[axis] == value:
            edges.add(frozenset((tuple(a), tuple(b))))
    return edges


def test_esri_shared_rings_stay_identical():
    """内核侧也一样：两个相邻面共享的边逐点相同。"""
    left = [[0, 0], [1, 0], [1, 0.5], [1, 1], [0, 1], [0, 0]]
    # 起止点避开共享边（x=1），否则环的闭合点也会被 x==1 过滤进共享串。
    right = [[2, 0], [2, 1], [1, 1], [1, 0.5], [1, 0], [2, 0]]
    geoms = [{"rings": [left]}, {"rings": [right]}]

    out, _ = geom_simplify.simplify_esri_geometries(geoms, max_points=8)

    left_shared = _edges_on(out[0]["rings"][0], axis=0, value=1.0)
    right_shared = _edges_on(out[1]["rings"][0], axis=0, value=1.0)
    assert left_shared, "测试前提：这两环确实共享 x=1 这条边"
    assert left_shared == right_shared, "共享边两侧必须是同一串坐标"


# ------------------------------------------------------------------ 工作量上限
def test_sawtooth_arcs_do_not_blow_up():
    """锯齿顶点会把 DP 拖成 O(n²)：必须被工作量上限叫停，而不是硬算到天荒地老。"""
    collection = _sawtooth_collection(features=60, vertices=2000)  # 12 万点
    assert html_pack.count_points(collection) > 100_000

    started = time.perf_counter()
    simplified, stats = geom_simplify.simplify_feature_collection(
        collection, max_points=20_000
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 10.0, f"锯齿图层简化 {elapsed:.1f}s：工作量上限没拦住 O(n²)"
    assert stats["stalled"] is False, "最终交付的这一轮不该是停滞的那一轮"
    assert html_pack.count_points(simplified) <= 20_000
    for feature in simplified["features"]:
        for ring in _rings(feature["geometry"]):
            assert len(ring) >= 4, "面不能被压成一条线"
            assert ring[0] == ring[-1], "环必须闭合"


def test_explicit_tolerance_that_cannot_finish_delivers_original():
    """显式容差压不动时宁可原样交付，绝不交空几何。"""
    collection = _sawtooth_collection(features=5, vertices=2000)

    simplified, stats = geom_simplify.simplify_feature_collection(
        collection, tolerance=1e-9
    )

    assert stats["stalled"] is True
    assert html_pack.count_points(simplified) == html_pack.count_points(collection)


# ------------------------------------------------------------------ 报告链路
def test_thin_features_uses_topology_not_stride():
    """报告侧同一个函数也不再按 stride 抽：共享边界两侧仍然对得上。"""
    collection = _grid_collection(columns=2, rows=2, detail=50)

    thinned = html_pack.thin_features(collection, max_coords=200)

    assert html_pack.count_points(thinned) < html_pack.count_points(collection)
    assert _shared_edge_length(thinned) > 0, "stride 抽稀会让共享边长度掉到 0"
    assert _max_deviation(collection, thinned) <= 0.05  # 容差 ≈ 半像素量级


# ------------------------------------------------------------------ 内核模板
def test_kernel_probe_code_embeds_simplifier_and_runs(monkeypatch: pytest.MonkeyPatch):
    """内核先减点：生成的代码在同一进程里用假 arcpy 真跑一遍。"""
    import sys
    import types

    ring = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]
    rows = [
        [json.dumps({"rings": [ring], "spatialReference": {"wkid": 4326}}), "甲"],
        [json.dumps({"rings": [ring], "spatialReference": {"wkid": 4326}}), "乙"],
    ]

    arcpy = types.ModuleType("arcpy")
    da = types.ModuleType("arcpy.da")

    class _Cursor:
        def __init__(self, path, cols):  # noqa: ANN001
            self.cols = cols

        def __enter__(self):
            return iter(rows)

        def __exit__(self, *exc):  # noqa: ANN002
            return False

    da.SearchCursor = _Cursor
    arcpy.da = da
    monkeypatch.setitem(sys.modules, "arcpy", arcpy)
    monkeypatch.setitem(sys.modules, "arcpy.da", da)

    code = html_pack.layer_probe_code(
        "output/结果.gdb/街道", limit=10, fields=["名称"], max_points=50
    )
    assert "simplify_esri_geometries" in code

    box: dict[str, object] = {}
    exec(compile(code, "<probe>", "exec"), {"set_result": lambda value: box.update(result=value)})

    result = box["result"]
    assert result["error"] == ""
    assert len(result["rows"]) == 2
    geometry, attrs = result["rows"][0]
    assert attrs == ["甲"]
    assert "spatialReference" not in geometry, "内核回传不再带冗余坐标系块"
    assert result["simplify"]["output_points"] <= 50


def test_kernel_probe_code_without_budget_keeps_old_template():
    """不给顶点预算时报告链路行为不变（模板逐字节还是老的）。"""
    code = html_pack.layer_probe_code("output/结果.gdb/街道")
    assert "simplify_esri_geometries" not in code
    assert "_geoms.append(_row[0])" in code


# ------------------------------------------------------------------ 真机素材
@pytest.mark.skipif(
    not (REAL_DATA / "中国_省.geojson").is_file(),
    reason="需要 workspace 里的中国行政区划真机数据（不入库）",
)
def test_real_provinces_keep_shared_boundaries():
    """真机素材：42 个省的共享边界简化后仍逐点重合，且偏差 ≤ 容差。"""
    data = json.loads((REAL_DATA / "中国_省.geojson").read_text(encoding="utf-8"))

    simplified, stats = geom_simplify.simplify_feature_collection(
        data, max_points=20_000
    )

    assert stats["output_points"] < stats["input_points"] / 2
    shared_before = _shared_edge_length(data)
    shared_after = _shared_edge_length(simplified)
    assert shared_before > 0
    assert shared_after >= shared_before * 0.75, (
        "共享边界长度不该大幅掉水（掉下去就是两侧各自取样了）"
    )
    assert _max_deviation(data, simplified) <= stats["tolerance"] + stats["precision"] + 1e-9

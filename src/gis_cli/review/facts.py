# -*- coding: utf-8 -*-
"""量测事实采集：把「数据路径」变成断言库能吃的数字。

复核者不能只看主代理抄来的数字，必须有一条**独立量测**的通道。这里提供两层：

1. **ArcPy 内核**（有 ``code_runner`` 时）：shp / .gdb / .tif 的要素数、面积、
   CRS、非法几何、重叠率、字段统计、分组汇总，全部在持久内核里算，结果是一段 JSON；
2. **纯 Python 回退**（没有 ArcPy 时）：CSV / JSON / GeoJSON 的计数、面积（Shoelace）、
   自相交、字段统计、分组汇总。shp/.gdb 会明确标 ``collector=unavailable``，断言看到
   缺事实会记 skipped，而不是瞎判。

事实结构（路径用**列表**寻址）::

    {"datasets": {"output/zones.shp": {
        "exists": True, "kind": "vector", "collector": "arcpy",
        "count": 12, "crs": {"is_projected": True, "linear_unit": "Meter", "wkid": 32610},
        "geometry_types": ["Polygon"], "area_total": 1234.5, "area_unit": "m2",
        "bbox": [xmin, ymin, xmax, ymax], "invalid_count": 0, "overlap_ratio": 0.0,
        "fields": {"POP": {"kind": "numeric", "sum": 1000.0, "min": 1, "max": 200, "mean": 83.3}},
        "group_sums": {"POP": {"Z01": 500.0, "Z02": 500.0}}
    }}}

注意：``area_total`` 的单位随 CRS 走（投影=m²，地理=度²），事实里用 ``area_unit``
明确写出来；断言不会把度² 当 m² 比。
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

_VECTOR_SUFFIXES = {".shp", ".gdb"}
_RASTER_SUFFIXES = {".tif", ".tiff", ".img", ".jp2", ".png", ".jpg", ".jpeg"}
_TABLE_SUFFIXES = {".dbf", ".csv", ".xlsx", ".xls"}
_JSON_SUFFIXES = {".geojson", ".json"}
#: GeoJSON 超过这个大小就不在进程内解析（避免一次复核吃光内存）。
_MAX_JSON_BYTES = 32 * 1024 * 1024
#: 纯 Python 自相交检测的点数上限（O(n²)）。
_MAX_RING_POINTS = 2000


class FactCollectionError(RuntimeError):
    """量测通道本身不可用（与“数据不存在”区分开）。"""


# ------------------------------------------------------------------ 路径
def _resolve(raw: str, workspace: Path) -> Path | None:
    """按工作区语义解析路径（与 engine.tools._resolve_path 保持同一顺序）。"""
    text = str(raw or "").strip().strip('"').strip("'").replace("\\", "/")
    if not text:
        return None
    first = Path(text)
    candidates: list[Path] = []
    if first.is_absolute():
        candidates.append(first)
    else:
        parts = list(first.parts)
        if parts and parts[0] == workspace.name and len(parts) > 1:
            candidates.append(workspace.parent / text)
            candidates.append(workspace / Path(*parts[1:]))
        else:
            candidates.append(workspace / text)
            candidates.append(Path.cwd() / text)
            candidates.append(workspace.parent / text)
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def _rel(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _kind_for(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix in _VECTOR_SUFFIXES:
        return "vector"
    if suffix in _RASTER_SUFFIXES:
        return "raster"
    if suffix in _TABLE_SUFFIXES:
        return "table"
    if suffix in _JSON_SUFFIXES:
        return "vector"
    return "file"


# ------------------------------------------------------------------ CRS
def crs_facts_from_epsg(code: int | None, name: str = "") -> dict[str, Any]:
    """从 EPSG 编号/名称给出够断言用的 CRS 事实（不求全，未知就留 None）。"""
    if not code:
        return {"wkid": None, "name": name, "is_projected": None, "linear_unit": None}
    code = int(code)
    text = (name or "").lower()
    if code in (4326, 4269, 4490, 4610, 4258, 4214) or "longlat" in text or "geographic" in text:
        return {"wkid": code, "name": name or f"EPSG:{code}", "is_projected": False, "linear_unit": "Degree"}
    if 2000 <= code < 4000 or 32601 <= code <= 32760 or code in (3857, 3395, 4547, 3031, 3035):
        unit = "Foot" if code in (2226, 2227, 2228, 2229, 2230) else "Meter"
        return {"wkid": code, "name": name or f"EPSG:{code}", "is_projected": True, "linear_unit": unit}
    return {"wkid": code, "name": name or f"EPSG:{code}", "is_projected": None, "linear_unit": None}


def _crs_from_text(text: str) -> dict[str, Any]:
    """从 ``urn:ogc:def:crs:EPSG::32610`` / ``EPSG:32610`` / 名称里取 CRS。"""
    raw = str(text or "")
    match = re.search(r"(?:EPSG|epsg)\D{0,4}(\d{4,5})", raw)
    if match is None:
        match = re.search(r"(\d{4,5})", raw)
    code = int(match.group(1)) if match else None
    return crs_facts_from_epsg(code, raw)


# ------------------------------------------------------------------ 纯 Python
def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _empty_stats() -> dict[str, Any]:
    return {"kind": "numeric", "count": 0, "sum": 0.0, "min": None, "max": None, "mean": None}


def _push(stats: dict[str, Any], value: float) -> None:
    stats["count"] += 1
    stats["sum"] += value
    stats["min"] = value if stats["min"] is None else min(stats["min"], value)
    stats["max"] = value if stats["max"] is None else max(stats["max"], value)
    stats["mean"] = stats["sum"] / stats["count"]


def _group_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _records_facts(
    records: Iterable[Mapping[str, Any]],
    *,
    groups: Mapping[str, str] | None = None,
    geometry_areas: list[float] | None = None,
) -> dict[str, Any]:
    """对一批属性记录做字段统计与分组汇总（CSV / GeoJSON 共用）。"""
    fields: dict[str, dict[str, Any]] = {}
    group_sums: dict[str, dict[str, float]] = {}
    count = 0
    for record in records:
        count += 1
        for name, value in record.items():
            number = _to_number(value)
            if number is None:
                continue
            stats = fields.setdefault(str(name), _empty_stats())
            _push(stats, number)
    for number_field, category_field in (groups or {}).items():
        bucket: dict[str, float] = {}
        for record in records:
            number = _to_number(record.get(number_field))
            if number is None:
                continue
            key = _group_key(record.get(category_field))
            bucket[key] = bucket.get(key, 0.0) + number
        if bucket:
            group_sums[str(number_field)] = bucket
    payload: dict[str, Any] = {"count": count, "fields": fields}
    if group_sums:
        payload["group_sums"] = group_sums
    if geometry_areas:
        payload["area_total"] = float(sum(geometry_areas))
    return payload


def _ring_area(ring: list[list[float]]) -> float:
    """Shoelace；返回带符号面积（外环正、内环负由调用方处理）。"""
    total = 0.0
    for index in range(len(ring) - 1):
        x1, y1 = float(ring[index][0]), float(ring[index][1])
        x2, y2 = float(ring[index + 1][0]), float(ring[index + 1][1])
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _segments_cross(a, b, c, d) -> bool:
    """两段是否**真交叉**（交点在两段内部）。

    四个方向值只要有一个为 0（端点相接、共线、零长边），就按“没交叉”处理：
    通用位置测试在这些退化情形下会把相邻/相接的边判成交叉，而真实数据里
    这种退化是常态（见 :func:`_ring_self_intersects`）。
    """

    def orient(p, q, r):
        value = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(value) < 1e-12:
            return 0
        return 1 if value > 0 else 2

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if 0 in (o1, o2, o3, o4):
        return False
    return o1 != o2 and o3 != o4


def _ring_self_intersects(ring: list[list[float]]) -> bool:
    """环是否**真**自相交（两条边在彼此内部交叉）。

    只认真交叉，不去抓退化相接：真实矢量数据里「连续重复顶点」（导出工具常见）
    和「捏点」（子环在一个顶点相接）大量存在，通用位置测试会把它们判成交叉——
    实测省级 GeoJSON 42 个要素里被误判 5 个。它们不影响面积口径，判成非法几何
    只会让复核误杀；真正会把面积算两遍的是边与边的内部交叉。
    """
    if not isinstance(ring, list) or len(ring) < 5:
        return False
    points = [(float(p[0]), float(p[1])) for p in ring if isinstance(p, (list, tuple)) and len(p) >= 2]
    if len(points) > _MAX_RING_POINTS:
        return False
    # 去掉连续重复点：零长边会让方向值恒为 0，进而把隔着一个零长边的两条边误判成交叉
    clean = [points[0]]
    for point in points[1:]:
        if point != clean[-1]:
            clean.append(point)
    points = clean
    if len(points) < 4:
        return False
    if points[0] != points[-1]:
        points.append(points[0])
    segments = list(zip(points[:-1], points[1:]))
    count = len(segments)
    for i in range(count):
        for j in range(i + 2, count):
            # 相邻边共享端点，不算自相交
            if i == 0 and j == count - 1:
                continue
            if _segments_cross(segments[i][0], segments[i][1], segments[j][0], segments[j][1]):
                return True
    return False


def _polygon_parts(geometry: Mapping[str, Any]) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    """返回 (外环列表, 内环列表)。只处理 Polygon / MultiPolygon。"""
    gtype = str(geometry.get("type", "") or "")
    coords = geometry.get("coordinates")
    if gtype == "Polygon" and isinstance(coords, list):
        rings = [ring for ring in coords if isinstance(ring, list)]
        return (rings[:1], rings[1:])
    if gtype == "MultiPolygon" and isinstance(coords, list):
        outers: list[list[list[float]]] = []
        holes: list[list[list[float]]] = []
        for polygon in coords:
            if isinstance(polygon, list) and polygon:
                outers.append(polygon[0])
                holes.extend(polygon[1:])
        return outers, holes
    return [], []


def _geojson_facts(path: Path, *, groups: Mapping[str, str] | None) -> dict[str, Any]:
    size = path.stat().st_size
    if size > _MAX_JSON_BYTES:
        raise FactCollectionError(f"GeoJSON 过大（{size} bytes），跳过进程内解析")
    data = json.loads(_read_text(path))
    features = data.get("features") if isinstance(data, dict) else None
    if not isinstance(features, list):
        raise FactCollectionError("不是 GeoJSON FeatureCollection")
    crs = _crs_from_text(json.dumps(data.get("crs") or ""))
    if crs.get("wkid") is None:
        crs = {"wkid": None, "name": "", "is_projected": None, "linear_unit": None}

    records: list[dict[str, Any]] = []
    areas: list[float] = []
    invalid = 0
    geometry_types: set[str] = set()
    bbox: list[float] | None = None
    for feature in features:
        if not isinstance(feature, Mapping):
            continue
        properties = feature.get("properties")
        records.append(dict(properties) if isinstance(properties, Mapping) else {})
        geometry = feature.get("geometry")
        if not isinstance(geometry, Mapping):
            continue
        geometry_types.add(str(geometry.get("type", "") or ""))
        outers, holes = _polygon_parts(geometry)
        if not outers:
            continue
        for ring in outers + holes:
            if _ring_self_intersects(ring):
                invalid += 1
        area = 0.0
        for ring in outers:
            area += abs(_ring_area(ring))
        for ring in holes:
            area -= abs(_ring_area(ring))
        areas.append(area)
        for ring in outers:
            for point in ring:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                x, y = float(point[0]), float(point[1])
                if bbox is None:
                    bbox = [x, y, x, y]
                else:
                    bbox = [min(bbox[0], x), min(bbox[1], y), max(bbox[2], x), max(bbox[3], y)]

    payload = _records_facts(records, groups=groups, geometry_areas=areas)
    projected = crs.get("is_projected")
    area_unit = "m2" if projected else ("deg2" if projected is False else None)
    payload.update(
        {
            "kind": "vector",
            "format": "geojson",
            "collector": "python",
            "crs": crs,
            "geometry_types": sorted(item for item in geometry_types if item),
            "area_unit": area_unit,
            "bbox": bbox,
            "invalid_count": invalid,
            # 没有多边形叠加能力：重叠率不猜，留 None（断言会 skipped）
            "overlap_ratio": None,
        }
    )
    return payload


def _csv_facts(path: Path, *, groups: Mapping[str, str] | None) -> dict[str, Any]:
    text = _read_text(path)
    reader = csv.DictReader(text.splitlines())
    records = [dict(row) for row in reader]
    payload = _records_facts(records, groups=groups)
    payload.update(
        {
            "kind": "table",
            "format": "csv",
            "collector": "python",
            "crs": {"wkid": None, "name": "", "is_projected": None, "linear_unit": None},
        }
    )
    return payload


# ------------------------------------------------------------------ ArcPy 内核
_MEASURE_CODE = r'''
import json as _json
import os as _os
import arcpy as _arcpy

_PATHS = __PATHS__
_GROUPS = __GROUPS__


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _crs(desc):
    sr = _safe(lambda: desc.spatialReference)
    if sr is None:
        return {"wkid": None, "name": "", "is_projected": None, "linear_unit": None}
    return {
        "wkid": _safe(lambda: int(sr.factoryCode), None),
        "name": _safe(lambda: str(sr.name), ""),
        "is_projected": _safe(lambda: bool(sr.type == "Projected"), None),
        "linear_unit": _safe(lambda: str(sr.linearUnitName), ""),
    }


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"kind": "numeric", "count": 0, "sum": 0.0, "min": None, "max": None, "mean": None}
    return {
        "kind": "numeric",
        "count": len(values),
        "sum": float(sum(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _field_kind(field):
    return str(getattr(field, "type", "") or "")


_NUMERIC_TYPES = {"Integer", "SmallInteger", "Double", "Single", "BigInteger", "OID"}
_SKIP_FIELDS = {"objectid", "shape_length", "shape_area", "shape"}


def _vector(p, groups):
    desc = _arcpy.Describe(p)
    shape_type = _safe(lambda: str(desc.shapeType), "")
    fields = list(_arcpy.ListFields(p) or [])
    numeric = [
        f.name for f in fields
        if _field_kind(f) in _NUMERIC_TYPES and f.name.lower() not in _SKIP_FIELDS
    ]
    all_names = [f.name for f in fields]
    group_fields = [g for g in (groups or {}).values() if g in all_names]
    by_name = {f.name: f for f in fields}
    group_num_fields = [n for n in (groups or {}) if n in numeric and n in by_name]
    cat_names = [n for n in dict.fromkeys(group_fields) if n in all_names]
    cursor_fields = (["SHAPE@"] if shape_type in ("Polygon", "Polyline") else []) + numeric + cat_names
    cursor_fields = list(dict.fromkeys(cursor_fields))

    sums = {n: 0.0 for n in numeric}
    mins = {n: None for n in numeric}
    maxs = {n: None for n in numeric}
    counts = {n: 0 for n in numeric}
    groups_out = {n: {} for n in group_num_fields}
    #: 按分组字段汇总的几何面积（供 area_partition 逐分区核对）
    area_by_group = {n: {} for n in cat_names}
    total_area = 0.0
    invalid = 0
    geometries = []
    bbox = None
    count = 0

    with _arcpy.da.SearchCursor(p, cursor_fields) as cursor:
        for row in cursor:
            count += 1
            record = dict(zip(cursor_fields, row))
            for name in numeric:
                value = _safe(lambda n=name: float(record.get(n)))
                if value is None:
                    continue
                sums[name] += value
                counts[name] += 1
                mins[name] = value if mins[name] is None else min(mins[name], value)
                maxs[name] = value if maxs[name] is None else max(maxs[name], value)
            for name in group_num_fields:
                value = _safe(lambda n=name: float(record.get(n)))
                if value is None:
                    continue
                category = str(record.get((groups or {}).get(name), "") or "")
                groups_out[name][category] = groups_out[name].get(category, 0.0) + value
            geom = record.get("SHAPE@")
            if geom is not None:
                area = _safe(lambda g=geom: float(g.area), None)
                if area is not None:
                    total_area += area
                for cat in cat_names:
                    category = str(record.get(cat, "") or "")
                    area_by_group[cat][category] = area_by_group[cat].get(category, 0.0) + (area or 0.0)
                extent = _safe(lambda g=geom: g.extent, None)
                if extent is not None:
                    box = [float(extent.XMin), float(extent.YMin), float(extent.XMax), float(extent.YMax)]
                    if bbox is None:
                        bbox = box
                    else:
                        bbox = [min(bbox[0], box[0]), min(bbox[1], box[1]),
                                max(bbox[2], box[2]), max(bbox[3], box[3])]
                if shape_type == "Polygon" and len(geometries) < 150:
                    geometries.append(geom)

    # 非法几何：CheckGeometry 会写表；失败就当没量到
    if shape_type == "Polygon" and count <= 5000:
        scratch = _safe(lambda: _arcpy.env.scratchGDB, None)
        if scratch:
            out_table = _os.path.join(scratch, "review_check_geom")
            _safe(lambda: _arcpy.management.Delete(out_table) if _arcpy.Exists(out_table) else None)
            ok = _safe(lambda: _arcpy.management.CheckGeometry(p, out_table) and True, False)
            if ok:
                invalid = int(_safe(lambda: _arcpy.management.GetCount(out_table)[0], 0) or 0)

    # 重叠率：小数据集才做两两求交（triple 重叠会重复计入，作为下限用）
    overlap_area = 0.0
    if shape_type == "Polygon" and len(geometries) >= 2:
        for i in range(len(geometries)):
            for j in range(i + 1, len(geometries)):
                a, b = geometries[i], geometries[j]
                if _safe(lambda x=a, y=b: x.overlaps(y), False):
                    overlap_area += float(_safe(lambda x=a, y=b: x.intersect(y, 4).area, 0.0) or 0.0)

    # 关系：相邻 + 包含/被包含两个方向（“方向写反”的结论靠这两项判定）。
    # 只量测**数据集内部**的关系（结果面 vs 研究区这种跨数据集的不猜）；
    # 只在几何全量收集（≤150）时给，样本被截断时不猜，让断言 skipped。
    relations = {}
    if shape_type == "Polygon" and count >= 2 and len(geometries) == count:
        linked = [False] * len(geometries)
        contained = [False] * len(geometries)   # i 落在 j 里
        container = [False] * len(geometries)   # i 包含 j
        pairs = 0
        within_pairs = 0
        for i in range(len(geometries)):
            for j in range(i + 1, len(geometries)):
                a, b = geometries[i], geometries[j]
                if _safe(lambda x=a, y=b: x.touches(y) or x.overlaps(y), False):
                    linked[i] = linked[j] = True
                    pairs += 1
                if _safe(lambda x=a, y=b: x.within(y), False):
                    contained[i] = True
                    container[j] = True
                    within_pairs += 1
                elif _safe(lambda x=a, y=b: y.within(x), False):
                    contained[j] = True
                    container[i] = True
                    within_pairs += 1
        relations["adjacent"] = {"features": sum(1 for flag in linked if flag), "pairs": pairs}
        # 计数为 0 的关系不写：0 会被当成“量测到了 0”，跨数据集的包含关系会被误判
        if within_pairs:
            relations["within"] = {"features": sum(1 for flag in contained if flag), "pairs": within_pairs}
            relations["contains"] = {"features": sum(1 for flag in container if flag), "pairs": within_pairs}

    is_projected = _safe(lambda: desc.spatialReference.type == "Projected", False)
    area_key = "AREA_M2" if is_projected else "AREA_DEG2"
    for cat in cat_names:
        key = area_key if len(cat_names) == 1 else area_key + "_BY_" + cat
        # 0 面积的分区要留着：claim 写 0 时，丢了它反而会报「多出分区」（DAIR-195）
        groups_out[key] = dict(area_by_group[cat])

    return {
        "kind": "vector",
        "format": "arcpy",
        "collector": "arcpy",
        "count": count,
        "crs": _crs(desc),
        "geometry_types": [shape_type] if shape_type else [],
        "area_total": total_area if shape_type == "Polygon" else None,
        "area_unit": "m2" if is_projected else "deg2",
        "bbox": bbox,
        "invalid_count": invalid,
        "overlap_ratio": (overlap_area / total_area) if total_area else None,
        "relations": relations,
        "fields": {n: _stats([None]) if counts[n] == 0 else {
            "kind": "numeric", "count": counts[n], "sum": sums[n],
            "min": mins[n], "max": maxs[n], "mean": sums[n] / counts[n],
        } for n in numeric},
        "group_sums": {n: v for n, v in groups_out.items() if v},
    }


def _raster(p, groups):
    desc = _arcpy.Describe(p)
    cell = _safe(lambda: float(desc.meanCellWidth), None)
    width = _safe(lambda: int(desc.width), None)
    height = _safe(lambda: int(desc.height), None)
    count = (width * height) if (width and height) else None
    stats = {}
    for label, prop in (("min", "MINIMUM"), ("max", "MAXIMUM"), ("mean", "MEAN")):
        stats[label] = _safe(
            lambda pr=prop: float(_arcpy.management.GetRasterProperties(p, pr).getOutput(0)), None
        )
    return {
        "kind": "raster",
        "format": "arcpy",
        "collector": "arcpy",
        "count": count,
        "cell_size": cell,
        "width": width,
        "height": height,
        "crs": _crs(desc),
        "stats": stats,
        "band_count": _safe(lambda: int(desc.bandCount), None),
    }


def _table(p, groups):
    fields = list(_arcpy.ListFields(p) or [])
    numeric = [f.name for f in fields if _field_kind(f) in _NUMERIC_TYPES and f.name.lower() not in _SKIP_FIELDS]
    all_names = [f.name for f in fields]
    group_num_fields = [n for n in (groups or {}) if n in numeric]
    cat_names = [n for n in dict.fromkeys((groups or {}).values()) if n in all_names]
    cursor_fields = numeric + cat_names
    sums = {n: 0.0 for n in numeric}
    mins = {n: None for n in numeric}
    maxs = {n: None for n in numeric}
    counts = {n: 0 for n in numeric}
    groups_out = {n: {} for n in group_num_fields}
    total = 0
    with _arcpy.da.SearchCursor(p, cursor_fields) as cursor:
        for row in cursor:
            total += 1
            record = dict(zip(cursor_fields, row))
            for name in numeric:
                value = _safe(lambda n=name: float(record.get(n)))
                if value is None:
                    continue
                sums[name] += value
                counts[name] += 1
                mins[name] = value if mins[name] is None else min(mins[name], value)
                maxs[name] = value if maxs[name] is None else max(maxs[name], value)
            for name in group_num_fields:
                value = _safe(lambda n=name: float(record.get(n)))
                if value is None:
                    continue
                category = str(record.get((groups or {}).get(name), "") or "")
                groups_out[name][category] = groups_out[name].get(category, 0.0) + value
    return {
        "kind": "table",
        "format": "arcpy",
        "collector": "arcpy",
        "count": total,
        "crs": {"wkid": None, "name": "", "is_projected": None, "linear_unit": None},
        "fields": {n: {
            "kind": "numeric", "count": counts[n], "sum": sums[n],
            "min": mins[n], "max": maxs[n], "mean": sums[n] / counts[n] if counts[n] else None,
        } for n in numeric},
        "group_sums": {n: v for n, v in groups_out.items() if v},
    }


_result = {}
for _p in _PATHS:
    _groups = _GROUPS.get(_p) or {}
    try:
        _desc = _arcpy.Describe(_p)
        _dtype = str(_safe(lambda: _desc.dataType, "") or "")
        # 大小写不敏感比较：真机 ArcGIS Pro 3.6 对 .shp 返回 "ShapeFile"（大写 F），
        # 旧版本/假模块里是 "Shapefile"——只认一种拼写会让整类数据落到 unknown（DAIR-188）。
        _dtype_key = _dtype.lower()
        if _dtype_key in ("shapefile", "featureclass"):
            _result[_p] = _vector(_p, _groups)
        elif _dtype_key in ("rasterdataset", "raster"):
            _result[_p] = _raster(_p, _groups)
        elif _dtype_key in ("table", "dbasetable"):
            _result[_p] = _table(_p, _groups)
        elif _dtype_key in ("workspace", "filegdbworkspace"):
            _contents = []
            _old = _arcpy.env.workspace
            try:
                _arcpy.env.workspace = _p
                for _name in (_arcpy.ListFeatureClasses() or [])[:30]:
                    _contents.append({"name": _name, "kind": "vector"})
                for _name in (_arcpy.ListRasters() or [])[:30]:
                    _contents.append({"name": _name, "kind": "raster"})
                for _name in (_arcpy.ListTables() or [])[:30]:
                    _contents.append({"name": _name, "kind": "table"})
            finally:
                _arcpy.env.workspace = _old
            _result[_p] = {"kind": "workspace", "format": "arcpy", "collector": "arcpy",
                           "contents": _contents, "count": len(_contents)}
        else:
            _result[_p] = {"kind": "unknown", "collector": "arcpy", "data_type": _dtype, "count": None}
    except Exception as _exc:
        _result[_p] = {"kind": "error", "collector": "arcpy", "error": str(_exc)[:300]}

print(_json.dumps(_result, ensure_ascii=False))
'''


def _kernel_facts(
    paths: list[Path],
    *,
    workspace: Path,
    runner: Any,
    groups: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """在 ArcPy 内核里量测；任何异常都折成单数据集 error，不拖垮复核。"""
    payload_paths = [p.as_posix() for p in paths]
    payload_groups = {p.as_posix(): dict(groups.get(_rel(p, workspace), {})) for p in paths}
    code = (
        _MEASURE_CODE.replace("__PATHS__", json.dumps(payload_paths, ensure_ascii=False))
        .replace("__GROUPS__", json.dumps(payload_groups, ensure_ascii=False))
    )
    try:
        result = runner.run(code, timeout=120)
    except Exception as exc:
        raise FactCollectionError(f"ArcPy 量测执行失败：{exc}") from exc
    stdout = str(getattr(result, "stdout", "") or "").strip()
    stderr = str(getattr(result, "stderr", "") or "").strip()
    if not bool(getattr(result, "ok", False)) and not stdout:
        raise FactCollectionError(f"ArcPy 量测失败：{stderr[:300]}")
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise FactCollectionError("ArcPy 量测没有返回结果")
    try:
        data = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise FactCollectionError(f"ArcPy 量测返回的不是 JSON：{lines[-1][:200]}") from exc
    if not isinstance(data, dict):
        raise FactCollectionError("ArcPy 量测返回结构不对")
    return data


# ------------------------------------------------------------------ 入口
def _normalize_groups(raw: Mapping[str, Any] | None) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """支持两种写法：全局 ``{"POP": "ZONE"}`` 或按路径 ``{"a.shp": {"POP": "ZONE"}}``。"""
    raw = dict(raw or {})
    if not raw:
        return {}, {}
    if all(isinstance(value, str) for value in raw.values()):
        return {str(k): str(v) for k, v in raw.items()}, {}
    per_path: dict[str, dict[str, str]] = {}
    for key, value in raw.items():
        if isinstance(value, Mapping):
            per_path[str(key)] = {str(k): str(v) for k, v in value.items() if isinstance(v, str)}
    return {}, per_path


def collect_facts(
    paths: Iterable[str] | None,
    *,
    workspace: str | Path,
    runner: Any = None,
    group_fields: Mapping[str, Any] | None = None,
    max_paths: int = 40,
) -> dict[str, Any]:
    """量测一组数据/产物，返回断言库能吃的 ``facts``。

    Args:
        paths: 相对工作区或绝对路径；重复/空值自动去掉。
        workspace: 工作区（路径解析基准）。
        runner: ``CodeRunner``（有则用 ArcPy 内核量测 shp/.gdb/.tif）。
        group_fields: 分组汇总声明：``{"POP": "ZONE"}`` 或按路径的
            ``{"output/zones.shp": {"POP": "ZONE"}}``。
        max_paths: 一次最多量测几个路径（复核 prompt 塞不下更多）。

    Returns:
        ``{"datasets": {路径: 事实...}, "paths": [...], "collector": [...]}``。
        单个数据集量测失败只写 ``error``，不会抛异常。
    """
    workspace_path = Path(workspace)
    global_groups, per_path_groups = _normalize_groups(group_fields)
    seen: list[str] = []
    for raw in paths or []:
        text = str(raw or "").strip()
        if text and text not in seen:
            seen.append(text)
    seen = seen[: max(1, int(max_paths))]

    facts: dict[str, Any] = {"datasets": {}, "paths": [], "collector": []}
    kernel_paths: list[Path] = []
    kernel_groups: dict[str, dict[str, str]] = {}
    for raw in seen:
        resolved = _resolve(raw, workspace_path)
        rel = _rel(resolved, workspace_path) if resolved is not None else raw.replace("\\", "/")
        facts["paths"].append(rel)
        if resolved is None or not resolved.exists():
            facts["datasets"][rel] = {"path": rel, "exists": False, "kind": _kind_for(Path(raw).suffix)}
            continue
        suffix = resolved.suffix.lower()
        kind = _kind_for(suffix)
        groups = per_path_groups.get(rel) or per_path_groups.get(raw) or global_groups
        entry: dict[str, Any]
        if suffix in _JSON_SUFFIXES:
            try:
                entry = _geojson_facts(resolved, groups=groups)
            except Exception as exc:
                entry = {"kind": kind, "collector": "python", "error": f"{type(exc).__name__}: {exc}"}
        elif suffix == ".csv":
            try:
                entry = _csv_facts(resolved, groups=groups)
            except Exception as exc:
                entry = {"kind": kind, "collector": "python", "error": f"{type(exc).__name__}: {exc}"}
        elif kind in ("vector", "raster", "table") and runner is not None:
            kernel_paths.append(resolved)
            kernel_groups[rel] = dict(groups)
            entry = None  # 由内核统一补
        else:
            entry = {
                "kind": kind,
                "format": suffix.lstrip("."),
                "collector": "unavailable",
                "note": "没有 ArcPy 内核，shp/.gdb/.tif 未量测",
            }
        entry = dict(entry or {})
        entry.setdefault("path", rel)
        entry.setdefault("exists", True)
        try:
            entry.setdefault("size", resolved.stat().st_size if resolved.is_file() else None)
        except OSError:
            pass
        facts["datasets"][rel] = entry

    if kernel_paths and runner is not None:
        try:
            measured = _kernel_facts(
                kernel_paths,
                workspace=workspace_path,
                runner=runner,
                groups=kernel_groups,
            )
        except Exception as exc:
            logger.warning("kernel fact collection failed: %s", exc)
            measured = {}
            for path in kernel_paths:
                rel = _rel(path, workspace_path)
                facts["datasets"][rel]["error"] = str(exc)[:300]
                facts["datasets"][rel]["collector"] = "unavailable"
        for path in kernel_paths:
            key = path.as_posix()
            rel = _rel(path, workspace_path)
            entry = measured.get(key)
            if isinstance(entry, Mapping):
                merged = dict(facts["datasets"].get(rel) or {})
                merged.update(entry)
                merged.setdefault("path", rel)
                merged.setdefault("exists", True)
                facts["datasets"][rel] = merged

    facts["collector"] = sorted(
        {
            str(entry.get("collector") or "")
            for entry in facts["datasets"].values()
            if isinstance(entry, Mapping) and entry.get("collector")
        }
    )
    return facts


def one_dataset(facts: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    """按路径取单个数据集事实（容忍路径写法差异）。"""
    datasets = facts.get("datasets") if isinstance(facts, Mapping) else None
    if not isinstance(datasets, Mapping):
        return {}
    if path in datasets:
        return datasets[path]
    normalized = str(path or "").replace("\\", "/")
    for key, value in datasets.items():
        if str(key).replace("\\", "/") == normalized and isinstance(value, Mapping):
            return value
    return {}


__all__ = [
    "FactCollectionError",
    "collect_facts",
    "crs_facts_from_epsg",
    "one_dataset",
]

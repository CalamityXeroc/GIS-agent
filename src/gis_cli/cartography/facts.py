# -*- coding: utf-8 -*-
"""制图事实采集（facts）：把"数据里影响版式的事实"取出来。

管线：``facts.py``（取事实）→ ``design.py``（规则引擎出 LayoutSpec）→ ``render.py``（落地到 ArcGIS）。

本模块**只采集事实，不做任何版式判断**——所有版式决策都在 :mod:`design` 里，
这样规则引擎可以脱离 ArcGIS 单测（用假的 facts 直接测）。

采集内容：
- 图层级：几何类型、要素数、范围（投影单位宽高）、坐标系（是否投影/线性单位）、是否栅格
- 字段级：类型、min/max/分位数、空值率、唯一值及计数（用于图例标签与类数决策）
- 版面级：**九宫格占用度**（每个格子里有多少"图形墨迹"）——用来决定图例/指北针放哪个角
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

GRID = 3  # 九宫格
_MAX_SAMPLES = 4000  # 矢量要素重心采样上限
_MAX_FIELD_SAMPLES = 5000  # 字段值采样上限
_MAX_RASTER_CELLS = 400_000  # 栅格采样像元上限

CORNER_KEYS = ("top_left", "top_right", "bottom_left", "bottom_right")

#: 九宫格 → 四角映射（行 0 在顶部）；每角取该角的 2×2 象限
def _corner_cells(row: int, col: int) -> list[tuple[int, int]]:
    rows = (0, 1) if row == 0 else (1, 2)
    cols = (0, 1) if col == 0 else (1, 2)
    return [(r, c) for r in rows for c in cols]


_CORNER_BOXES = {
    "top_left": (0, 0),
    "top_right": (0, 2),
    "bottom_left": (2, 0),
    "bottom_right": (2, 2),
}


_PROBE_CODE = r'''
import json
import arcpy

_ITEMS = __ITEMS__
_RESULT = {"layers": [], "errors": []}
_GRID = 3
_MAX_SAMPLES = __MAX_SAMPLES__
_MAX_FIELD = __MAX_FIELD__
_MAX_RASTER = __MAX_RASTER__


def _q(values, frac):
    if not values:
        return None
    idx = max(0, min(len(values) - 1, int(round(frac * (len(values) - 1)))))
    return float(values[idx])


def _field_stats(path, field):
    stats = {"field": field}
    values = []
    nulls = 0
    total = 0
    counts = {}
    categorical = False
    ftype = ""
    for f in arcpy.ListFields(path):
        if f.name == field:
            ftype = f.type
            break
    stats["type"] = ftype
    step = 1
    try:
        count = int(arcpy.management.GetCount(path)[0])
    except Exception:
        count = 0
    if count > _MAX_FIELD:
        step = max(1, count // _MAX_FIELD)
    with arcpy.da.SearchCursor(path, [field]) as cur:
        for index, (value,) in enumerate(cur):
            total += 1
            if value is None:
                nulls += 1
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if index % step == 0 and len(values) < _MAX_FIELD:
                    values.append(float(value))
            else:
                categorical = True
                key = str(value)
                counts[key] = counts.get(key, 0) + 1
    stats["sample_count"] = len(values)
    stats["null_rate"] = round(nulls / total, 4) if total else 0.0
    if values:
        values.sort()
        stats["min"] = values[0]
        stats["max"] = values[-1]
        stats["p25"] = _q(values, 0.25)
        stats["p50"] = _q(values, 0.50)
        stats["p75"] = _q(values, 0.75)
        stats["mean"] = round(sum(values) / len(values), 6)
        # 偏度指示：中位数偏离均值多少（决定用分位数还是自然断点）
        span = (stats["max"] - stats["min"]) or 1.0
        stats["skew"] = round(abs((stats["p50"] or 0) - stats["mean"]) / span, 4)
    if categorical or counts:
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
        stats["unique_count"] = len(counts)
        stats["values_top"] = [{"value": k, "count": v} for k, v in top]
    return stats


def _vector_occupancy(path, extent, geom_type, count):
    grid = [[0.0] * _GRID for _ in range(_GRID)]
    width = extent.width or 1.0
    height = extent.height or 1.0
    step = max(1, count // _MAX_SAMPLES) if count else 1
    fields = ["SHAPE@"]
    weighted = "polygon" in (geom_type or "").lower() or "line" in (geom_type or "").lower()
    if weighted:
        fields = ["SHAPE@AREA", "SHAPE@XY"] if "polygon" in geom_type.lower() else ["SHAPE@LENGTH", "SHAPE@XY"]
    else:
        fields = ["SHAPE@XY"]
    with arcpy.da.SearchCursor(path, fields) as cur:
        for index, row in enumerate(cur):
            if index % step:
                continue
            if weighted:
                weight = float(row[0] or 0.0)
                xy = row[1]
            else:
                weight = 1.0
                xy = row[0]
            if xy is None:
                continue
            gx = min(_GRID - 1, max(0, int((xy[0] - extent.XMin) / width * _GRID)))
            gy = min(_GRID - 1, max(0, int((xy[1] - extent.YMin) / height * _GRID)))
            grid[_GRID - 1 - gy][gx] += max(weight, 1e-9)
    return grid


def _raster_occupancy(path, extent):
    import numpy as np
    desc = arcpy.Describe(path)
    ncols, nrows = int(desc.width), int(desc.height)
    step_r = max(1, int((ncols * nrows / float(_MAX_RASTER)) ** 0.5) + 1)
    arr = arcpy.RasterToNumPyArray(path, nodata_to_value=-1)
    valid = arr != -1
    grid = [[0.0] * _GRID for _ in range(_GRID)]
    rows, cols = valid.shape
    for r in range(0, rows, step_r):
        for c in range(0, cols, step_r):
            if not valid[r, c]:
                continue
            gx = min(_GRID - 1, int(c / cols * _GRID))
            gy = min(_GRID - 1, int(r / rows * _GRID))
            grid[gy][gx] += 1.0
    return grid


for _item in _ITEMS:
    _path = _item.get("path")
    _field = _item.get("field") or ""
    _rec = {"path": _path, "role": _item.get("role", "primary")}
    try:
        if not arcpy.Exists(_path):
            _rec["exists"] = False
            _RESULT["layers"].append(_rec)
            continue
        _rec["exists"] = True
        _desc = arcpy.Describe(_path)
        _dtype = getattr(_desc, "dataType", "") or ""
        _rec["data_type"] = _dtype
        _rec["is_raster"] = bool(getattr(_desc, "isRaster", False)) or "Raster" in _dtype
        _rec["geom_type"] = getattr(_desc, "shapeType", "") or ""
        _sr = getattr(_desc, "spatialReference", None)
        _rec["crs"] = {
            "wkid": getattr(_sr, "factoryCode", None),
            "name": getattr(_sr, "name", ""),
            "is_projected": getattr(_sr, "type", "") == "Projected",
            "unit": getattr(getattr(_sr, "linearUnit", None), "name", ""),
        }
        _ext = _desc.extent
        _rec["extent"] = {
            "xmin": _ext.XMin, "ymin": _ext.YMin, "xmax": _ext.XMax, "ymax": _ext.YMax,
            "width": abs(_ext.width), "height": abs(_ext.height),
        }
        if _rec["is_raster"]:
            _rec["cell_size"] = float(getattr(_desc, "meanCellWidth", 0) or 0)
            _rec["feature_count"] = None
            _rec["occupancy"] = _raster_occupancy(_path, _ext)
        else:
            _rec["feature_count"] = int(arcpy.management.GetCount(_path)[0])
            _rec["fields"] = [
                {"name": f.name, "type": f.type, "alias": getattr(f, "aliasName", "")}
                for f in arcpy.ListFields(_path) if f.type not in ("Geometry",)
            ]
            _rec["occupancy"] = _vector_occupancy(_path, _ext, _rec["geom_type"], _rec["feature_count"])
        if _field:
            _rec.setdefault("field_stats", {})[_field] = _field_stats(_path, _field)
    except Exception as _exc:
        _rec["error"] = str(_exc)[:200]
        _RESULT["errors"].append(f"{_path}: {str(_exc)[:160]}")
    _RESULT["layers"].append(_rec)

print("MAP_FACTS_JSON=" + json.dumps(_RESULT, ensure_ascii=False))
'''


def build_probe_code(items: list[dict[str, Any]]) -> str:
    """生成采集代码（可在持久内核或进程内 arcpy 中执行）。"""
    return (
        _PROBE_CODE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
        .replace("__MAX_SAMPLES__", str(_MAX_SAMPLES))
        .replace("__MAX_FIELD__", str(_MAX_FIELD_SAMPLES))
        .replace("__MAX_RASTER__", str(_MAX_RASTER_CELLS))
    )


def _parse_probe_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed((stdout or "").splitlines()):
        if line.startswith("MAP_FACTS_JSON="):
            try:
                return json.loads(line.split("=", 1)[1])
            except Exception as exc:  # pragma: no cover
                logger.warning("解析 facts 失败: %s", exc)
    return {"layers": [], "errors": ["facts 采集无输出"]}


def probe_in_process(items: list[dict[str, Any]]) -> dict[str, Any]:
    """在当前进程内用 arcpy 采集（无内核时的通路）。"""
    import contextlib
    import io

    import arcpy  # type: ignore

    code = build_probe_code(items)
    buffer = io.StringIO()
    namespace: dict[str, Any] = {"arcpy": arcpy}
    with contextlib.redirect_stdout(buffer):
        exec(compile(code, "<map_facts>", "exec"), namespace, namespace)
    return _parse_probe_stdout(buffer.getvalue())


def probe_with_runner(code_runner: Any, items: list[dict[str, Any]], *, timeout: float = 300.0) -> dict[str, Any]:
    """通过 CodeRunner（持久内核）采集，失败时退回进程内。"""
    try:
        outcome = code_runner.run(build_probe_code(items), timeout=timeout)
        if getattr(outcome, "ok", False):
            facts = _parse_probe_stdout(getattr(outcome, "stdout", "") or "")
            if facts.get("layers"):
                return facts
    except Exception as exc:
        logger.warning("内核采集 facts 失败，退回进程内: %s", exc)
    return probe_in_process(items)


def collect_map_facts(
    layers: list[dict[str, Any] | str],
    *,
    field: str = "",
    code_runner: Any = None,
    probe: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """采集制图事实。

    Args:
        layers: 图层列表，元素可以是路径字符串或 ``{"path":..., "field":..., "role":"primary|context"}``
        field: 主图层的分级/分类字段（不传就采不到字段分布，渲染模式会退化成 single）
        code_runner: 可选的持久内核执行器（优先使用，失败退回进程内 arcpy）
        probe: 自定义采集函数（单测注入用）
    """
    items: list[dict[str, Any]] = []
    for layer in layers:
        if isinstance(layer, str):
            items.append({"path": layer, "field": field, "role": "primary"})
        else:
            items.append(
                {
                    "path": str(layer.get("path", "")),
                    "field": str(layer.get("field", "") or field or ""),
                    "role": str(layer.get("role", "primary")),
                }
            )
    if probe is not None:
        raw = probe(items)
    elif code_runner is not None:
        raw = probe_with_runner(code_runner, items)
    else:
        raw = probe_in_process(items)
    return normalize_facts(raw)


def normalize_facts(raw: dict[str, Any]) -> dict[str, Any]:
    """补齐派生字段（联合范围、长宽比、九宫格占用度、角区占用比例）。"""
    layers = list(raw.get("layers") or [])
    extent = union_extent(layers)
    facts: dict[str, Any] = {
        "layers": layers,
        "extent": extent,
        "aspect": aspect_ratio(extent),
        "occupancy": occupancy_grid(layers),
        "corner_usage": corner_usage(layers),
        "primary": next((l for l in layers if l.get("role") == "primary"), layers[0] if layers else {}),
        "errors": raw.get("errors") or [],
        "field": str(raw.get("field") or "") or next(
            (name for layer in layers for name in (layer.get("field_stats") or {})), ""),
    }
    facts["layer_count"] = len([l for l in layers if l.get("exists")])
    # 显式传入的 render_mode 优先（调用方比自动推测更清楚自己在画什么）
    facts["render_mode"] = str(raw.get("render_mode") or "") or guess_render_mode(facts)
    return facts


# --------------------------------------------------------------------- 纯函数

def union_extent(layers: list[dict[str, Any]]) -> dict[str, float]:
    """所有图层范围的并集（用于纸张/朝向决策）。"""
    boxes = [l["extent"] for l in layers if l.get("exists") and l.get("extent")]
    if not boxes:
        return {"xmin": 0.0, "ymin": 0.0, "xmax": 0.0, "ymax": 0.0, "width": 0.0, "height": 0.0}
    xmin = min(b["xmin"] for b in boxes)
    ymin = min(b["ymin"] for b in boxes)
    xmax = max(b["xmax"] for b in boxes)
    ymax = max(b["ymax"] for b in boxes)
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "width": abs(xmax - xmin), "height": abs(ymax - ymin)}


def aspect_ratio(extent: dict[str, float]) -> float:
    """数据长宽比 W/H（无数据时按 1.0）。"""
    height = float(extent.get("height") or 0.0)
    width = float(extent.get("width") or 0.0)
    if height <= 0 or width <= 0:
        return 1.0
    return width / height


def occupancy_grid(layers: list[dict[str, Any]]) -> list[list[float]]:
    """九宫格占用度（各图层相加）。"""
    grid = [[0.0] * GRID for _ in range(GRID)]
    for layer in layers:
        layer_grid = layer.get("occupancy")
        if not layer_grid:
            continue
        for r in range(GRID):
            for c in range(GRID):
                try:
                    grid[r][c] += float(layer_grid[r][c])
                except (IndexError, TypeError, ValueError):
                    continue
    return grid


def corner_usage(layers: list[dict[str, Any]]) -> dict[str, float]:
    """四个**象限**的墨迹占比（0~1，相对最大值归一；用于选择图例/指北针位置）。"""
    grid = occupancy_grid(layers)
    raw: dict[str, float] = {}
    for key, (row, col) in _CORNER_BOXES.items():
        raw[key] = float(sum(grid[r][c] for r, c in _corner_cells(row, col)))
    top = max(raw.values()) if raw else 0.0
    if top <= 0:
        return {key: 0.0 for key in raw}
    return {key: round(value / top, 4) for key, value in raw.items()}


def emptiest_corners(facts: dict[str, Any], *, exclude: tuple[str, ...] = ()) -> list[str]:
    """按"越空越靠前"返回角序列（图例优先放在最空的角）。"""
    usage = facts.get("corner_usage") or {}
    ranked = sorted(
        (k for k in CORNER_KEYS if k not in exclude),
        key=lambda key: (usage.get(key, 1.0), CORNER_KEYS.index(key)),
    )
    return ranked


def field_profile(facts: dict[str, Any], field: str, *, primary: bool = True) -> dict[str, Any]:
    """取某字段的画像（优先主图层）。"""
    layers = facts.get("layers") or []
    ordered = sorted(layers, key=lambda l: 0 if (l.get("role") == "primary") == primary else 1)
    for layer in ordered:
        stats = (layer.get("field_stats") or {}).get(field)
        if stats:
            return stats
    return {}


def guess_render_mode(facts: dict[str, Any]) -> str:
    """推测渲染模式：single（单符号）/ unique（分类）/ graduated（分级）。"""
    layers = [l for l in (facts.get("layers") or []) if l.get("exists")]
    if len(layers) > 1:
        return "multi_layer"
    profile = field_profile(facts, str(facts.get("field") or "")) if facts.get("field") else {}
    if profile:
        if profile.get("values_top"):
            return "unique"
        if profile.get("max") is not None:
            return "graduated"
    return "single"


__all__ = [
    "GRID",
    "CORNER_KEYS",
    "collect_map_facts",
    "normalize_facts",
    "build_probe_code",
    "probe_in_process",
    "probe_with_runner",
    "union_extent",
    "aspect_ratio",
    "occupancy_grid",
    "corner_usage",
    "emptiest_corners",
    "field_profile",
    "guess_render_mode",
]
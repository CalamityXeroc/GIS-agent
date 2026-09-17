# -*- coding: utf-8 -*-
"""Filesystem discovery + ArcPy profiling for the catalog.

Discovery runs in the agent process (cheap filesystem walk). Profiling runs
inside the persistent ArcPy kernel, one call for the whole batch.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .store import Catalog, DatasetRecord, record_from_scan

_SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "node_modules", ".gis_agent",
    ".gis_agent_memory", ".gis_agent_state", ".codegraph", ".claude", ".codex",
    ".pytest_cache", "temp", "tmp",
}
_VECTOR_EXT = {".shp", ".geojson", ".gpkg"}
_RASTER_EXT = {".tif", ".tiff", ".img", ".dem", ".asc"}
_TABLE_EXT = {".csv", ".xlsx", ".dbf"}
_PROJECT_EXT = {".aprx", ".lyrx", ".mxd", ".mapx"}


@dataclass
class Discovery:
    """A candidate dataset found on disk."""

    path: str
    kind: str
    container: str = ""
    mtime: float = 0.0
    size: int = 0


def discover(roots: Iterable[str | Path], *, max_items: int = 2000, include_gdb: bool = True) -> list[Discovery]:
    """Walk roots and return candidate datasets.

    GDB containers are reported as one ``gdb`` item; their inner feature
    classes are expanded during profiling.
    """
    found: list[Discovery] = []
    seen: set[str] = set()

    def _add(path: Path, kind: str, container: str = "") -> None:
        key = str(path).lower()
        if key in seen:
            return
        seen.add(key)
        try:
            stat = path.stat()
            mtime, size = stat.st_mtime, stat.st_size
        except OSError:
            mtime, size = 0.0, 0
        found.append(Discovery(str(path), kind, container, mtime, size))

    for root in roots:
        root_path = Path(root)
        if root_path.is_file():
            kind = _kind_for(root_path)
            if kind:
                _add(root_path, kind)
            continue
        if not root_path.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [
                d for d in dirnames
                if d.lower() not in _SKIP_DIRS and not d.startswith(".")
            ]
            current = Path(dirpath)
            if current.suffix.lower() == ".gdb" and include_gdb:
                _add(current, "gdb")
                dirnames[:] = []  # do not descend into gdb internals
                continue
            for filename in filenames:
                if filename.startswith("~$") or filename.startswith("."):
                    continue
                path = current / filename
                kind = _kind_for(path)
                if kind:
                    _add(path, kind)
            if len(found) >= max_items:
                break
        if len(found) >= max_items:
            break
    return found


def _kind_for(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in _VECTOR_EXT:
        return "vector"
    if ext in _RASTER_EXT:
        return "raster"
    if ext in _TABLE_EXT:
        return "table"
    if ext in _PROJECT_EXT:
        return "project"
    return None


# --------------------------------------------------------------------- arcpy
_SCAN_CODE = r'''
import json as _json
import os as _os
import arcpy as _arcpy

_TARGETS = {targets}
_SAMPLE_ROWS = {sample_rows}
_SAMPLE_VALUES = {sample_values}
_MAX_FIELDS = {max_fields}

_arcpy.env.overwriteOutput = True


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _samples_for(path, field_names):
    samples = {{name: [] for name in field_names}}
    nulls = {{name: 0 for name in field_names}}
    total = 0
    try:
        with _arcpy.da.SearchCursor(path, field_names) as cur:
            for row in cur:
                total += 1
                for name, value in zip(field_names, row):
                    if value is None:
                        nulls[name] += 1
                    elif len(samples[name]) < _SAMPLE_VALUES:
                        text = str(value).strip()
                        if text:
                            samples[name].append(text[:80])
                if total >= _SAMPLE_ROWS:
                    break
    except Exception:
        pass
    ratios = {{name: (nulls[name] / total if total else None) for name in field_names}}
    return samples, ratios


def _vector_record(path, kind, container=""):
    desc = _safe(lambda: _arcpy.Describe(path))
    if desc is None:
        return None
    sr = getattr(desc, "spatialReference", None)
    fields = []
    try:
        all_fields = _arcpy.ListFields(path)
    except Exception:
        all_fields = []
    for f in all_fields:
        if f.type in ("Geometry", "Blob", "Raster"):
            continue
        fields.append({{
            "name": f.name,
            "alias": getattr(f, "aliasName", "") or "",
            "type": f.type,
            "length": int(getattr(f, "length", 0) or 0),
            "nullable": bool(getattr(f, "isNullable", False)),
        }})
    sample_names = [f["name"] for f in fields[:_MAX_FIELDS]]
    samples, ratios = _samples_for(path, sample_names) if sample_names else ({{}}, {{}})
    for f in fields:
        if f["name"] in samples:
            f["samples"] = samples[f["name"]]
            f["null_ratio"] = ratios.get(f["name"])
    extent = None
    ext = getattr(desc, "extent", None)
    if ext is not None:
        extent = [ext.XMin, ext.YMin, ext.XMax, ext.YMax]
    count = _safe(lambda: int(_arcpy.management.GetCount(path)[0]))
    return {{
        "path": path,
        "kind": kind,
        "container": container,
        "name": getattr(desc, "baseName", _os.path.basename(path)),
        "crs_name": getattr(sr, "name", "") if sr else "",
        "wkid": getattr(sr, "factoryCode", None) if sr else None,
        "geom_type": getattr(desc, "shapeType", "") or "",
        "feature_count": count,
        "extent": extent,
        "fields": fields,
    }}


def _raster_record(path):
    desc = _safe(lambda: _arcpy.Describe(path))
    if desc is None:
        return None
    sr = getattr(desc, "spatialReference", None)
    stats = {{}}
    for prop in ("MINIMUM", "MAXIMUM", "MEAN", "STD", "NOData_VALUE"):
        val = _safe(lambda p=prop: _arcpy.management.GetRasterProperties(path, p).getOutput(0))
        try:
            stats[prop.lower()] = float(val) if val not in (None, "") else None
        except Exception:
            stats[prop.lower()] = None
    return {{
        "path": path,
        "kind": "raster",
        "container": "",
        "name": _os.path.basename(path),
        "crs_name": getattr(sr, "name", "") if sr else "",
        "wkid": getattr(sr, "factoryCode", None) if sr else None,
        "geom_type": "Raster",
        "feature_count": None,
        "extent": None,
        "fields": [],
        "raster": {{
            "band_count": _safe(lambda: int(desc.bandCount)),
            "cell_size": _safe(lambda: float(desc.meanCellWidth)),
            "width": _safe(lambda: int(desc.width)),
            "height": _safe(lambda: int(desc.height)),
            "min": stats.get("minimum"),
            "max": stats.get("maximum"),
            "mean": stats.get("mean"),
            "nodata": stats.get("nodata_value"),
        }},
    }}


def _project_record(path):
    records = []
    try:
        aprx = _arcpy.mp.ArcGISProject(path)
    except Exception as exc:
        return [{{
            "path": path, "kind": "project", "container": "", "name": _os.path.basename(path),
            "fields": [], "extra": {{"error": str(exc)}},
        }}]
    maps = []
    for m in _safe(lambda: aprx.listMaps(), []) or []:
        layers = [layer.name for layer in (_safe(lambda mm=m: mm.listLayers(), []) or [])]
        maps.append({{"name": m.name, "layers": layers[:50]}})
    layouts = []
    for layout in _safe(lambda: aprx.listLayouts(), []) or []:
        elements = []
        for el in _safe(lambda l=layout: l.listElements(), []) or []:
            elements.append({{"name": el.name, "type": getattr(el, "type", "")}})
        layouts.append({{"name": layout.name, "elements": elements[:50]}})
    records.append({{
        "path": path, "kind": "project", "container": "", "name": _os.path.basename(path),
        "fields": [], "extra": {{"maps": maps, "layouts": layouts}},
    }})
    return records


def _scan_target(target):
    kind = target.get("kind")
    path = target.get("path")
    container = target.get("container", "") or ""
    if kind == "gdb":
        records = []
        try:
            walk = _arcpy.da.Walk(path, datatype=["FeatureClass", "Table"])
            for dirpath, dirnames, filenames in walk:
                for filename in filenames:
                    full = _os.path.join(dirpath, filename)
                    rec = _safe(lambda p=full, c=path: _vector_record(p, "feature_class", c))
                    if rec:
                        records.append(rec)
        except Exception as exc:
            return [{{"path": path, "kind": "gdb", "container": "", "name": _os.path.basename(path),
                      "fields": [], "extra": {{"error": str(exc)}}}}]
        return records
    if kind == "vector":
        rec = _safe(lambda: _vector_record(path, "vector"))
        return [rec] if rec else []
    if kind == "raster":
        rec = _safe(lambda: _raster_record(path))
        return [rec] if rec else []
    if kind == "project":
        return _safe(lambda: _project_record(path), []) or []
    return []


_records = []
for _target in _TARGETS:
    _records.extend(_scan_target(_target) or [])

set_result({{"records": _records, "count": len(_records)}})
'''


def build_scan_code(items: list[Discovery], *, sample_rows: int = 1000, sample_values: int = 3, max_fields: int = 40) -> str:
    """Build the ArcPy scan code for a batch of discovered items."""
    targets = [
        {"path": d.path, "kind": d.kind, "container": d.container}
        for d in items
    ]
    return _SCAN_CODE.format(
        targets=json.dumps(targets, ensure_ascii=False),
        sample_rows=int(sample_rows),
        sample_values=int(sample_values),
        max_fields=int(max_fields),
    )


def profile_table(path: Path, *, max_rows: int = 5000) -> dict[str, Any]:
    """Profile a CSV/XLSX without ArcPy (agent-side, best effort)."""
    import csv

    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader, [])
                samples: list[list[str]] = []
                count = 0
                for row in reader:
                    count += 1
                    if len(samples) < 3:
                        samples.append([str(v)[:80] for v in row[: len(header)]])
                    if count >= max_rows:
                        break
            return {
                "path": str(path),
                "kind": "table",
                "name": path.name,
                "feature_count": count,
                "fields": [
                    {"name": h, "type": "String", "samples": [s[i] for s in samples if i < len(s)][:3]}
                    for i, h in enumerate(header)
                ],
            }
        except Exception as exc:
            return {"path": str(path), "kind": "table", "name": path.name, "extra": {"error": str(exc)}}
    return {"path": str(path), "kind": "table", "name": path.name}


@dataclass
class RefreshStats:
    """Summary of a catalog refresh."""

    discovered: int = 0
    scanned: int = 0
    skipped: int = 0
    removed: int = 0
    errors: list[str] = field(default_factory=list)


def refresh_catalog(
    catalog: Catalog,
    roots: Iterable[str | Path],
    kernel: Any,
    *,
    force: bool = False,
    max_items: int = 2000,
) -> RefreshStats:
    """Incrementally refresh the catalog from roots using an ArcPyKernel."""
    stats = RefreshStats()
    items = discover(roots, max_items=max_items)
    stats.discovered = len(items)

    # Drop entries whose files disappeared.
    stats.removed = catalog.remove_missing(
        {d.path for d in items},
        {d.path for d in items if d.kind == "gdb"},
    )

    to_scan: list[Discovery] = []
    table_items: list[Discovery] = []
    for item in items:
        if not force and catalog.is_fresh(item.path, item.mtime, item.size):
            stats.skipped += 1
            continue
        if item.kind == "table":
            table_items.append(item)
        else:
            to_scan.append(item)

    # Tables are profiled locally.
    for item in table_items:
        try:
            payload = profile_table(Path(item.path))
            payload.update({"mtime": item.mtime, "size": item.size, "container": item.container})
            catalog.upsert(record_from_scan(payload))
            stats.scanned += 1
        except Exception as exc:
            stats.errors.append(f"{item.path}: {exc}")

    if not to_scan:
        return stats

    # ArcPy-backed items are profiled in one kernel call.
    code = build_scan_code(to_scan)
    result = kernel.execute(code, timeout=900)
    if not result.ok:
        stats.errors.append(f"kernel scan failed: {result.error}")
        return stats
    payload = result.result or {}
    records = payload.get("records", []) if isinstance(payload, dict) else []
    by_container = {d.path: d for d in to_scan if d.kind == "gdb"}
    by_path = {d.path: d for d in to_scan}

    for raw in records:
        if not isinstance(raw, dict):
            continue
        path = str(raw.get("path", ""))
        container = str(raw.get("container", "") or "")
        source = by_path.get(path) or by_container.get(container)
        if source is None:
            source = next(
                (d for d in to_scan if container and Path(path).is_relative_to(Path(container))),
                None,
            )
        raw["mtime"] = source.mtime if source else 0.0
        raw["size"] = source.size if source else 0
        catalog.upsert(record_from_scan(raw))
        stats.scanned += 1
    return stats

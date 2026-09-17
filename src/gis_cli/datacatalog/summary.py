# -*- coding: utf-8 -*-
"""Compact catalog digests for LLM prompts and structured query results."""

from __future__ import annotations

from typing import Any

from .store import Catalog, DatasetRecord

_KIND_LABEL = {
    "vector": "矢量",
    "feature_class": "要素类",
    "raster": "栅格",
    "table": "表",
    "project": "工程",
    "gdb": "地理数据库",
    "other": "其他",
}

# ArcGIS solver scratch layers and reference-solution artifacts are noise.
_NOISE_PATTERNS = (
    "barriers",
    "cfroutes",
    "demandpoints",
    "facilities",
    "locationallocationsolver",
    "closestfacilitysolver",
    "solver",
    "route",
    "sastatus",
    "sapolygons",
)
_KIND_ORDER = {"vector": 0, "feature_class": 1, "raster": 2, "table": 3, "project": 4, "gdb": 5, "other": 6}


def _is_noise(record: DatasetRecord) -> bool:
    name = (record.name or "").lower()
    if any(pattern in name for pattern in _NOISE_PATTERNS):
        return True
    return record.feature_count == 0 and record.kind in {"feature_class", "vector"}


def _fmt_count(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def _fmt_extent(extent: list[float] | None) -> str:
    if not extent or len(extent) < 4:
        return ""
    try:
        return f"[{extent[0]:.1f},{extent[1]:.1f},{extent[2]:.1f},{extent[3]:.1f}]"
    except Exception:
        return ""


def record_to_line(record: DatasetRecord, *, max_fields: int = 24, with_samples: bool = False) -> str:
    """Render one dataset as a single compact line."""
    label = _KIND_LABEL.get(record.kind, record.kind)
    parts = [f"[{label}] {record.name}"]
    if record.geom_type and record.geom_type not in {"Raster"}:
        parts.append(f"geom={record.geom_type}")
    if record.crs_name:
        wkid = f"/{record.wkid}" if record.wkid else ""
        parts.append(f"crs={record.crs_name}{wkid}")
    if record.feature_count is not None:
        parts.append(f"n={_fmt_count(record.feature_count)}")
    if record.kind == "raster" and record.raster:
        raster = record.raster
        bits = []
        if raster.get("band_count"):
            bits.append(f"bands={raster['band_count']}")
        if raster.get("cell_size"):
            bits.append(f"cell={raster['cell_size']:g}")
        if raster.get("min") is not None and raster.get("max") is not None:
            bits.append(f"range=[{raster['min']:g},{raster['max']:g}]")
        if bits:
            parts.append(" ".join(bits))
    if record.extent:
        parts.append(f"ext={_fmt_extent(record.extent)}")
    parts.append(f"path={record.path}")

    lines = ["  " + ", ".join(parts)]
    if record.fields:
        field_bits = []
        for f in record.fields[:max_fields]:
            name = f.get("name", "")
            ftype = f.get("type", "")
            bit = f"{name}({ftype})"
            if with_samples and f.get("samples"):
                bit += f"~{f['samples'][:2]}"
            field_bits.append(bit)
        more = len(record.fields) - max_fields
        suffix = f", (+{more} more)" if more > 0 else ""
        lines.append("    fields: " + ", ".join(field_bits) + suffix)
    if record.kind == "project" and record.extra:
        maps = record.extra.get("maps") or []
        layouts = record.extra.get("layouts") or []
        if maps:
            lines.append("    maps: " + ", ".join(str(m.get("name", "")) for m in maps[:10]))
        if layouts:
            lines.append("    layouts: " + ", ".join(str(l.get("name", "")) for l in layouts[:10]))
    return "\n".join(lines)


def build_digest(
    catalog: Catalog,
    *,
    max_datasets: int = 40,
    max_fields: int = 24,
    with_samples: bool = False,
    kinds: list[str] | None = None,
    include_noise: bool = False,
) -> str:
    """Build the compact catalog digest injected into prompts."""
    records = catalog.list_datasets(limit=max_datasets * 6)
    if kinds:
        records = [r for r in records if r.kind in kinds]
    if not include_noise:
        filtered = [r for r in records if not _is_noise(r)]
        # Keep noise only if filtering would hide everything.
        records = filtered or records
    records.sort(key=lambda r: (_KIND_ORDER.get(r.kind, 9), -(r.feature_count or 0), r.name or ""))
    records = records[:max_datasets]
    if not records:
        return "（数据目录为空，请先扫描工作区数据）"
    lines = []
    for record in records:
        lines.append(record_to_line(record, max_fields=max_fields, with_samples=with_samples))
    total = catalog.counts()
    summary = ", ".join(f"{_KIND_LABEL.get(k, k)}={v}" for k, v in sorted(total.items()))
    header = f"共 {sum(total.values())} 个数据集（{summary}）" if total else ""
    return (header + "\n" if header else "") + "\n".join(lines)


def build_focused_digest(
    catalog: Catalog,
    keywords: list[str],
    *,
    max_datasets: int = 6,
    max_fields: int = 15,
) -> str:
    """Digest limited to datasets whose name/path matches the given keywords.

    Used by the code-repair loop so the fix prompt carries only the schema
    information relevant to the failing code instead of the whole catalog.
    """
    seen: dict[str, DatasetRecord] = {}
    for keyword in keywords:
        keyword = (keyword or "").strip()
        if len(keyword) < 2:
            continue
        for record in catalog.list_datasets(name_like=keyword, limit=4):
            seen.setdefault(record.path, record)
            if len(seen) >= max_datasets:
                break
        if len(seen) >= max_datasets:
            break
    if not seen:
        return ""
    lines = [record_to_line(r, max_fields=max_fields, with_samples=False) for r in seen.values()]
    return "\n".join(lines)


def search_catalog(
    catalog: Catalog,
    *,
    query: str = "",
    kind: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Structured catalog search used by the catalog_query tool."""
    records = catalog.list_datasets(kind=kind, name_like=query or None, limit=limit)
    if not records and query:
        records = catalog.list_datasets(limit=limit)
    return [record_to_dict(r) for r in records]


def record_to_dict(record: DatasetRecord) -> dict[str, Any]:
    """Full structured representation (for tool results)."""
    return {
        "path": record.path,
        "kind": record.kind,
        "container": record.container,
        "name": record.name,
        "crs_name": record.crs_name,
        "wkid": record.wkid,
        "geom_type": record.geom_type,
        "feature_count": record.feature_count,
        "extent": record.extent,
        "fields": record.fields,
        "raster": record.raster,
        "extra": record.extra,
    }

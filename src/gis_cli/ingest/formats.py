# -*- coding: utf-8 -*-
"""KML / GPKG / GeoJSON 直读（P0-2）。

这三个格式不需要 ArcGIS 也能读：

- **KML** 是 XML，``xml.etree`` 就能把 Placemark、坐标、范围读出来
- **GPKG** 是 SQLite 库，``sqlite3`` 直接查 ``gpkg_contents`` / ``gpkg_geometry_columns``
- **GeoJSON** 就是 JSON

所以扫描器把 ``.kml`` / ``.gpkg`` 交给这里在 agent 进程里做，不占 ArcPy 内核
（内核调用一次要十几秒，而读元数据是毫秒级）。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree

from .csv_points import IngestError, infer_field_type

#: KML 里的标准属性（Placemark 的 name/description），其余字段按出现顺序补。
_KML_STD_FIELDS = ("Name", "Description")

_GPKG_TYPE_MAP = {
    "INTEGER": "Integer",
    "INT": "Integer",
    "SMALLINT": "SmallInteger",
    "BIGINT": "Integer",
    "REAL": "Double",
    "DOUBLE": "Double",
    "FLOAT": "Single",
    "TEXT": "String",
    "VARCHAR": "String",
    "BOOLEAN": "Boolean",
    "DATE": "Date",
    "DATETIME": "Date",
}


def _local(tag: str) -> str:
    """去掉 XML 命名空间（KML 2.2 带 ns，2.0 有的不带）。"""
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ElementTree.Element, name: str) -> str:
    for child in element:
        if _local(child.tag) == name:
            return (child.text or "").strip()
    return ""


def _coords_in(element: ElementTree.Element) -> list[list[float]]:
    """取出元素下所有 ``coordinates`` 里的坐标元组。"""
    points: list[list[float]] = []
    for node in element.iter():
        if _local(node.tag) != "coordinates":
            continue
        for chunk in (node.text or "").split():
            parts = chunk.split(",")
            if len(parts) < 2:
                continue
            try:
                lon, lat = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            points.append([lon, lat])
    return points


def _geometry_kind(element: ElementTree.Element) -> str:
    kinds = {
        _local(child.tag)
        for child in element.iter()
        if _local(child.tag) in {"Point", "LineString", "LinearRing", "Polygon", "MultiGeometry"}
    }
    kinds.discard("MultiGeometry")
    if "Polygon" in kinds:
        return "Polygon"
    if "LineString" in kinds:
        return "LineString"
    if "Point" in kinds:
        return "Point"
    return ""


def _placemarks(root: ElementTree.Element) -> list[ElementTree.Element]:
    return [node for node in root.iter() if _local(node.tag) == "Placemark"]


def _extent_of(points: Iterable[Sequence[float]]) -> list[float] | None:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if not xs:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def describe_kml(path: str | Path) -> dict[str, Any]:
    """读 KML 的要素数 / 几何类型 / 范围 / 属性名。"""
    file_path = Path(path)
    try:
        root = ElementTree.fromstring(file_path.read_bytes())
    except ElementTree.ParseError as exc:
        raise IngestError(f"KML 解析失败（不是合法 XML）：{file_path}：{exc}") from exc
    marks = _placemarks(root)
    if not marks:
        raise IngestError(f"KML 里没有 Placemark（空文件或只有底图样式）：{file_path}")
    counts: dict[str, int] = {}
    all_points: list[list[float]] = []
    names: set[str] = set()
    for mark in marks:
        kind = _geometry_kind(mark) or "None"
        counts[kind] = counts.get(kind, 0) + 1
        all_points.extend(_coords_in(mark))
        if _child_text(mark, "name"):
            names.add("Name")
        if _child_text(mark, "description"):
            names.add("Description")
    geom_type = next(iter(counts)) if len(counts) == 1 else "Mixed"
    document_name = ""
    for node in root.iter():
        if _local(node.tag) == "Document":
            document_name = _child_text(node, "name")
            break
    return {
        "path": str(file_path),
        "kind": "vector",
        "container": "",
        "name": document_name or file_path.stem,
        "crs_name": "WGS 84",
        "wkid": 4326,
        "geom_type": geom_type,
        "feature_count": len(marks),
        "extent": _extent_of(all_points),
        "fields": [
            {"name": name, "alias": "", "type": "String", "length": 0, "nullable": True, "samples": []}
            for name in _KML_STD_FIELDS
            if name in names
        ],
        "extra": {"format": "kml", "geometry_types": counts},
    }


def _kml_geometry(element: ElementTree.Element) -> dict[str, Any] | None:
    kind = _local(element.tag)
    points = _coords_in(element)
    if kind == "Point":
        return {"type": "Point", "coordinates": points[0]} if points else None
    if kind == "LineString":
        return {"type": "LineString", "coordinates": points} if len(points) >= 2 else None
    if kind == "Polygon":
        rings: list[list[list[float]]] = []
        for boundary in element:
            if _local(boundary.tag) not in {"outerBoundaryIs", "innerBoundaryIs"}:
                continue
            ring = _coords_in(boundary)
            if len(ring) >= 3:
                rings.append(ring)
        return {"type": "Polygon", "coordinates": rings} if rings else None
    return None


def kml_to_geojson(path: str | Path, out_geojson: str | Path | None = None) -> dict[str, Any]:
    """KML → GeoJSON（点/线/面；MultiGeometry 变成 GeometryCollection）。"""
    file_path = Path(path)
    try:
        root = ElementTree.fromstring(file_path.read_bytes())
    except ElementTree.ParseError as exc:
        raise IngestError(f"KML 解析失败（不是合法 XML）：{file_path}：{exc}") from exc
    features: list[dict[str, Any]] = []
    skipped = 0
    points: list[list[float]] = []
    for mark in _placemarks(root):
        geometry: dict[str, Any] | None = None
        for child in mark:
            tag = _local(child.tag)
            if tag == "MultiGeometry":
                parts = [g for g in (_kml_geometry(c) for c in child) if g]
                if parts:
                    geometry = {"type": "GeometryCollection", "geometries": parts}
                    break
            elif tag in {"Point", "LineString", "Polygon"}:
                geometry = _kml_geometry(child)
                break
        if geometry is None:
            skipped += 1
            continue
        points.extend(_coords_in(mark))
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "Name": _child_text(mark, "name"),
                    "Description": _child_text(mark, "description"),
                },
            }
        )
    if not features:
        raise IngestError(f"KML 里没有可转换的几何（共 {skipped} 个 Placemark）：{file_path}")
    target = Path(out_geojson) if out_geojson else file_path.with_suffix(".geojson")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "name": target.stem,
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": features,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "output": str(target),
        "feature_count": len(features),
        "skipped": skipped,
        "extent": _extent_of(points),
    }


def describe_geojson(path: str | Path) -> dict[str, Any]:
    """GeoJSON 的要素数 / 几何类型 / 范围 / 属性名（不依赖 arcpy）。"""
    file_path = Path(path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IngestError(f"GeoJSON 读取失败：{file_path}：{exc}") from exc
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        raise IngestError(f"不是 FeatureCollection：{file_path}")
    kinds: dict[str, int] = {}
    points: list[list[float]] = []
    props: dict[str, list[Any]] = {}

    def _coords(node: Any) -> None:
        if isinstance(node, list) and node and isinstance(node[0], (int, float)):
            if len(node) >= 2:
                points.append([float(node[0]), float(node[1])])
            return
        if isinstance(node, list):
            for item in node:
                _coords(item)

    for feature in features:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if isinstance(geometry, dict):
            kind = str(geometry.get("type") or "Unknown")
            kinds[kind] = kinds.get(kind, 0) + 1
            _coords(geometry.get("coordinates"))
            for part in geometry.get("geometries") or []:
                if isinstance(part, dict):
                    _coords(part.get("coordinates"))
        for key, value in (feature.get("properties") or {}).items():
            props.setdefault(str(key), []).append(value)
    geom_type = next(iter(kinds)) if len(kinds) == 1 else ("Mixed" if kinds else "")
    return {
        "path": str(file_path),
        "kind": "vector",
        "container": "",
        "name": (payload.get("name") if isinstance(payload, dict) else "") or file_path.stem,
        "crs_name": "WGS 84",
        "wkid": 4326,
        "geom_type": geom_type,
        "feature_count": len(features),
        "extent": _extent_of(points),
        "fields": [
            {"name": name, "alias": "", "type": infer_field_type([v for v in values if v is not None][:50]),
             "length": 0, "nullable": True, "samples": [str(v) for v in values if v is not None][:3]}
            for name, values in props.items()
        ],
        "extra": {"format": "geojson", "geometry_types": kinds},
    }


def _gpkg_fields(conn: sqlite3.Connection, table: str, geometry_column: str) -> list[dict[str, Any]]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    fields = []
    for row in rows:
        name = str(row[1])
        if name == geometry_column:
            continue
        declared = str(row[2] or "").upper()
        base = declared.split("(")[0].strip()
        fields.append(
            {
                "name": name,
                "alias": "",
                "type": _GPKG_TYPE_MAP.get(base, "String"),
                "length": 0,
                "nullable": not bool(row[3]),
                "samples": [],
            }
        )
    return fields


def describe_gpkg(path: str | Path) -> list[dict[str, Any]]:
    """读 GeoPackage 的每个要素图层（一个图层一条记录，container 指向 .gpkg）。"""
    file_path = Path(path)
    if not file_path.is_file():
        raise IngestError(f"文件不存在：{file_path}")
    try:
        conn = sqlite3.connect(str(file_path))
    except sqlite3.Error as exc:
        raise IngestError(f"打不开 GeoPackage：{file_path}：{exc}") from exc
    try:
        try:
            rows = conn.execute(
                """
                SELECT c.table_name, c.srs_id, c.min_x, c.min_y, c.max_x, c.max_y,
                       g.column_name, g.geometry_type_name
                FROM gpkg_contents c
                LEFT JOIN gpkg_geometry_columns g ON c.table_name = g.table_name
                WHERE c.data_type = 'features'
                ORDER BY c.table_name
                """
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise IngestError(f"不是 GeoPackage（缺 gpkg_contents 表）：{file_path}：{exc}") from exc
        if not rows:
            raise IngestError(f"GeoPackage 里没有要素图层：{file_path}")
        records = []
        for table, srs_id, min_x, min_y, max_x, max_y, geom_col, geom_type in rows:
            srs_name, wkid = "", None
            if srs_id is not None:
                srs = conn.execute(
                    "SELECT organization, organization_coordsys_id FROM gpkg_spatial_ref_sys WHERE srs_id = ?",
                    (srs_id,),
                ).fetchone()
                if srs and str(srs[0] or "").upper() == "EPSG":
                    wkid = int(srs[1])
                    srs_name = f"EPSG:{wkid}"
                else:
                    srs_name = f"srs_id={srs_id}"
            try:
                count = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            except sqlite3.DatabaseError:
                count = None
            extent = None
            if None not in (min_x, min_y, max_x, max_y):
                extent = [float(min_x), float(min_y), float(max_x), float(max_y)]
            records.append(
                {
                    "path": str(file_path / table),
                    "kind": "feature_class",
                    "container": str(file_path),
                    "name": table,
                    "crs_name": srs_name,
                    "wkid": wkid,
                    "geom_type": str(geom_type or ""),
                    "feature_count": count,
                    "extent": extent,
                    "fields": _gpkg_fields(conn, table, str(geom_col or "")),
                    "extra": {"format": "gpkg", "table": table},
                }
            )
        return records
    finally:
        conn.close()


def describe_vector(path: str | Path) -> dict[str, Any] | list[dict[str, Any]]:
    """按后缀分派：``.kml`` / ``.gpkg`` / ``.geojson``。"""
    suffix = Path(path).suffix.lower()
    if suffix == ".kml":
        return describe_kml(path)
    if suffix == ".gpkg":
        return describe_gpkg(path)
    if suffix in {".geojson", ".json"}:
        return describe_geojson(path)
    raise IngestError(f"不支持直读的矢量格式：{suffix or '(无后缀)'}（只认 .kml / .gpkg / .geojson）")

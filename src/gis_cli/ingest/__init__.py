# -*- coding: utf-8 -*-
"""数据接入层（P0-2）：让外行的数据能进得来。

四件事，各自独立可回滚：

- :mod:`~gis_cli.ingest.csv_points` —— CSV/Excel 带经纬度 → 点图层（GeoJSON → 要素类）
- :mod:`~gis_cli.ingest.formats` —— KML / GPKG / GeoJSON 直读（不需要 arcpy）
- :mod:`~gis_cli.ingest.boundaries` —— 内置省/市/县边界（按需下载 + 缓存 + 许可登记）
- :mod:`~gis_cli.ingest.basemap` —— 天地图 / OSM 底图（出图用）

统一约定：**接入即登记** —— 数据进 catalog 时同时写 ``source`` / ``license`` /
``acquired_at``。不知道许可就写 ``unknown``，不替用户认领。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..datacatalog.store import LICENSES, SOURCES, DatasetRecord
from .basemap import (
    BasemapKeyError,
    BasemapResult,
    BasemapSpec,
    lonlat_to_tile,
    osm_spec,
    render_basemap,
    spec_for,
    tianditu_spec,
    tile_bounds,
)
from .boundaries import BoundaryResult, BoundarySource, build_source, default_cache_dir, ensure_boundary
from .csv_points import (
    CRS_NOTE,
    CoordinateColumnError,
    CoordColumns,
    CsvPointsResult,
    IngestError,
    ProjectedCoordinateError,
    build_arcpy_code,
    convert_csv_points,
    detect_coord_columns,
    parse_coord,
    read_table,
    to_feature_class,
)
from .formats import describe_geojson, describe_gpkg, describe_kml, describe_vector, kml_to_geojson


def register_dataset(
    catalog: Any,
    record: DatasetRecord,
    *,
    source: str,
    license: str = "unknown",
    acquired_at: str | None = None,
    extra: dict[str, Any] | None = None,
) -> DatasetRecord:
    """把一份数据集登记进 catalog，并写清来源 / 许可 / 采集时间。

    ``source`` / ``license`` 的取值是 catalog 的契约（见
    ``datacatalog.store.SOURCES`` / ``LICENSES``），写错直接报错而不是默默存进去。
    """
    if source not in SOURCES:
        raise IngestError(f"source 只认 {'/'.join(SOURCES)}，收到：{source or '(空)'}")
    if license not in LICENSES:
        raise IngestError(f"license 只认 {'/'.join(LICENSES)}，收到：{license or '(空)'}")
    record.source = source
    record.license = license
    record.acquired_at = acquired_at or datetime.now(timezone.utc).isoformat()
    if extra:
        record.extra = {**(record.extra or {}), **extra}
    catalog.upsert(record)
    return record


__all__ = [
    "CRS_NOTE",
    "LICENSES",
    "SOURCES",
    "BasemapKeyError",
    "BasemapResult",
    "BasemapSpec",
    "BoundaryResult",
    "BoundarySource",
    "CoordColumns",
    "CoordinateColumnError",
    "CsvPointsResult",
    "IngestError",
    "ProjectedCoordinateError",
    "build_arcpy_code",
    "build_source",
    "convert_csv_points",
    "default_cache_dir",
    "describe_geojson",
    "describe_gpkg",
    "describe_kml",
    "describe_vector",
    "detect_coord_columns",
    "ensure_boundary",
    "kml_to_geojson",
    "lonlat_to_tile",
    "osm_spec",
    "parse_coord",
    "read_table",
    "register_dataset",
    "render_basemap",
    "spec_for",
    "tianditu_spec",
    "tile_bounds",
    "to_feature_class",
]

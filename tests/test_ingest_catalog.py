# -*- coding: utf-8 -*-
"""catalog 的来源/许可登记 + 扫描器接入 KML/GPKG（P0-2）。

两个重点：

1. **老库要能升级**：用户机器上的 catalog.db 是 P0-2 之前建的，
   没有 source/license/acquired_at 三列，必须自动补列且不丢数据。
2. **重扫不能冲掉来源**：扫描器每次都 upsert 全量记录，但它不知道
   "这份 CSV 是用户上传的" —— 一次重扫就把 csv_upload 打回 local 是不可接受的。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from gis_cli.datacatalog.store import Catalog, DatasetRecord
from gis_cli.datacatalog.summary import record_to_dict, record_to_line
from gis_cli.ingest import IngestError, register_dataset
from gis_cli.ingest.csv_points import convert_csv_points

#: P0-2 之前的 datasets 表（没有 source/license/acquired_at）。
_OLD_SCHEMA = """
CREATE TABLE datasets (
    path TEXT PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'other',
    container TEXT,
    name TEXT,
    mtime REAL,
    size INTEGER,
    crs_name TEXT,
    wkid INTEGER,
    geom_type TEXT,
    feature_count INTEGER,
    extent TEXT,
    extra TEXT,
    updated_at TEXT
);
CREATE TABLE fields (
    path TEXT NOT NULL, name TEXT NOT NULL, alias TEXT, ftype TEXT,
    length INTEGER, nullable INTEGER, samples TEXT, null_ratio REAL,
    PRIMARY KEY (path, name)
);
CREATE TABLE rasters (
    path TEXT PRIMARY KEY, band_count INTEGER, cell_size REAL, min REAL,
    max REAL, mean REAL, nodata REAL, width INTEGER, height INTEGER,
    crs_name TEXT, wkid INTEGER, extra TEXT
);
"""


def _old_catalog(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(_OLD_SCHEMA)
    conn.execute(
        "INSERT INTO datasets (path, kind, name, mtime, size, feature_count, updated_at) "
        "VALUES ('E:/old/roads.shp', 'vector', 'roads', 1.0, 10, 5, '2026-10-01T00:00:00Z')"
    )
    conn.commit()
    conn.close()


def _csv(tmp_path: Path) -> Path:
    source = tmp_path / "input" / "points.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("名称,经度,纬度\n甲,116.4,39.9\n乙,121.5,31.2\n", encoding="utf-8")
    return source


# ---------------------------------------------------------------- 迁移


def test_old_catalog_is_migrated_without_losing_rows(tmp_path):
    db_path = tmp_path / "catalog.db"
    _old_catalog(db_path)

    catalog = Catalog(db_path)
    columns = {row["name"] for row in catalog._conn.execute("PRAGMA table_info(datasets)")}
    assert {"source", "license", "acquired_at"} <= columns

    record = catalog.get("E:/old/roads.shp")
    assert record is not None, "老记录不能因为迁移丢了"
    assert record.feature_count == 5
    assert (record.source, record.license, record.acquired_at) == ("local", "unknown", "")
    catalog.close()


def test_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "catalog.db"
    Catalog(db_path).close()
    Catalog(db_path).close()  # 再开一次不能因为重复 ALTER 报错


# ------------------------------------------------------- 登记 + 重扫不冲掉


def test_register_csv_points_as_csv_upload(tmp_path):
    """验收项：catalog 里能看到 source=csv_upload, license=用户自有。"""
    catalog = Catalog(tmp_path / "catalog.db")
    result = convert_csv_points(_csv(tmp_path), out_geojson=tmp_path / "out" / "points.geojson")
    register_dataset(catalog, result.to_record(), source="csv_upload", license="用户自有")

    stored = catalog.get(result.geojson_path)
    assert stored is not None
    assert stored.source == "csv_upload"
    assert stored.license == "用户自有"
    assert stored.acquired_at, "采集时间要写"
    assert stored.kind == "vector"
    assert stored.feature_count == 2
    catalog.close()


def test_rescan_does_not_reset_source_or_license(tmp_path):
    catalog = Catalog(tmp_path / "catalog.db")
    result = convert_csv_points(_csv(tmp_path), out_geojson=tmp_path / "out" / "points.geojson")
    register_dataset(catalog, result.to_record(), source="csv_upload", license="用户自有")

    # 模拟扫描器重扫：同路径、同内容，但不知道来源（空串）
    scanner_record = DatasetRecord(
        path=result.geojson_path,
        kind="vector",
        name=Path(result.geojson_path).name,
        feature_count=result.feature_count,
        mtime=1.0,
        size=1,
    )
    catalog.upsert(scanner_record)

    stored = catalog.get(result.geojson_path)
    assert stored.source == "csv_upload", "重扫不能把上传来源打回 local"
    assert stored.license == "用户自有"
    catalog.close()


def test_rescan_keeps_extra_from_registration(tmp_path):
    """重扫不能把登记时写进 extra 的元信息抹掉（DAIR-26 B1）。

    扫描器（``scanner._vector_record``）产出的记录**没有 extra 键**，而 upsert
    以前无条件 ``extra=excluded.extra`` —— 默认 prewarm 就扫 ``<workspace>/input``，
    而 CSV 产物正好落那儿：登记完一次 prewarm 就清空。
    """
    catalog = Catalog(tmp_path / "catalog.db")
    result = convert_csv_points(_csv(tmp_path), out_geojson=tmp_path / "out" / "points.geojson")
    register_dataset(
        catalog,
        result.to_record(),
        source="csv_upload",
        license="用户自有",
        extra={"original_name": "点位.xlsx", "uploader": "cheng"},
    )

    # 模拟扫描器重扫：同路径、不带 extra（scanner._vector_record 就没有这个键）
    catalog.upsert(
        DatasetRecord(
            path=result.geojson_path,
            kind="vector",
            name=Path(result.geojson_path).name,
            feature_count=result.feature_count,
            mtime=1.0,
            size=1,
        )
    )

    stored = catalog.get(result.geojson_path)
    assert stored.extra.get("original_name") == "点位.xlsx", f"登记时写的 extra 被重扫抹掉了: {stored.extra}"
    assert stored.extra.get("uploader") == "cheng"
    assert stored.extra.get("skipped_rows") == 0, "接入自带的其他 extra 也要留"
    catalog.close()


def test_rescan_drops_transient_error_key(tmp_path):
    """``error`` 是上一次扫描的瞬时结论：新扫描成功（不带 error）就该清掉。"""
    catalog = Catalog(tmp_path / "catalog.db")
    catalog.upsert(DatasetRecord(path="E:/x/a.aprx", kind="project", name="a", extra={"error": "打不开"}))
    catalog.upsert(DatasetRecord(path="E:/x/a.aprx", kind="project", name="a", extra={"maps": ["m1"]}))

    stored = catalog.get("E:/x/a.aprx")
    assert "error" not in stored.extra, f"瞬时错误不该被继承: {stored.extra}"
    assert stored.extra.get("maps") == ["m1"]
    catalog.close()


def test_new_record_defaults_to_local_unknown(tmp_path):
    catalog = Catalog(tmp_path / "catalog.db")
    catalog.upsert(DatasetRecord(path="E:/x/a.shp", kind="vector", name="a"))
    stored = catalog.get("E:/x/a.shp")
    assert (stored.source, stored.license) == ("local", "unknown")
    catalog.close()


def test_register_dataset_validates_vocabulary(tmp_path):
    catalog = Catalog(tmp_path / "catalog.db")
    record = DatasetRecord(path="E:/x/a.shp", kind="vector", name="a")
    with pytest.raises(IngestError) as excinfo:
        register_dataset(catalog, record, source="wechat")
    assert "source 只认" in str(excinfo.value)
    with pytest.raises(IngestError) as excinfo:
        register_dataset(catalog, record, source="local", license="随便写的")
    assert "license 只认" in str(excinfo.value)
    catalog.close()


def test_summary_line_and_dict_expose_provenance(tmp_path):
    catalog = Catalog(tmp_path / "catalog.db")
    result = convert_csv_points(_csv(tmp_path), out_geojson=tmp_path / "out" / "points.geojson")
    register_dataset(catalog, result.to_record(), source="csv_upload", license="用户自有")
    stored = catalog.get(result.geojson_path)

    line = record_to_line(stored)
    assert "src=csv_upload" in line
    assert "lic=用户自有" in line

    payload = record_to_dict(stored)
    assert payload["source"] == "csv_upload"
    assert payload["license"] == "用户自有"

    # 默认值不占篇幅（老数据的 digest 行不能因为加字段变长）
    plain = DatasetRecord(path="E:/x/a.shp", kind="vector", name="a", source="local", license="unknown")
    assert "src=" not in record_to_line(plain)
    assert "lic=" not in record_to_line(plain)
    catalog.close()


# ------------------------------------------------- 扫描器：KML/GPKG 不用 arcpy


class _KernelMustNotRun:
    def __init__(self):
        self.calls = 0

    def execute(self, code, timeout=None, workspace=None):
        self.calls += 1
        raise AssertionError("KML/GPKG 不该交给 ArcPy 内核（要十几秒，而且 KML 根本不是要素类）")


def _kml(path: Path) -> Path:
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>点位</name>
<Placemark><name>甲</name><Point><coordinates>116.4,39.9,0</coordinates></Point></Placemark>
</Document></kml>
""",
        encoding="utf-8",
    )
    return path


def _gpkg(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT, srs_id INTEGER PRIMARY KEY,
            organization TEXT, organization_coordsys_id INTEGER, definition TEXT, description TEXT);
        CREATE TABLE gpkg_contents (table_name TEXT PRIMARY KEY, data_type TEXT, identifier TEXT,
            description TEXT, last_change TEXT, min_x DOUBLE, min_y DOUBLE, max_x DOUBLE,
            max_y DOUBLE, srs_id INTEGER);
        CREATE TABLE gpkg_geometry_columns (table_name TEXT, column_name TEXT,
            geometry_type_name TEXT, srs_id INTEGER, z TINYINT, m TINYINT);
        """
    )
    conn.execute("INSERT INTO gpkg_spatial_ref_sys VALUES ('WGS 84', 4326, 'EPSG', 4326, '', '')")
    conn.execute(
        "INSERT INTO gpkg_contents VALUES ('county', 'features', 'county', '', '', 113, 22, 115, 24, 4326)"
    )
    conn.execute("INSERT INTO gpkg_geometry_columns VALUES ('county', 'geom', 'POLYGON', 4326, 0, 0)")
    conn.execute("CREATE TABLE county (fid INTEGER PRIMARY KEY, geom BLOB, name TEXT)")
    conn.execute("INSERT INTO county (name) VALUES ('甲')")
    conn.commit()
    conn.close()
    return path


def test_discover_treats_kml_as_vector(tmp_path):
    from gis_cli.datacatalog.scanner import discover

    root = tmp_path / "input"
    root.mkdir()
    _kml(root / "points.kml")
    items = discover([root])
    assert [(Path(d.path).name, d.kind) for d in items] == [("points.kml", "vector")]


def test_refresh_catalog_profiles_kml_and_gpkg_without_kernel(tmp_path):
    from gis_cli.datacatalog.scanner import refresh_catalog

    root = tmp_path / "input"
    root.mkdir()
    _kml(root / "points.kml")
    _gpkg(root / "data.gpkg")
    catalog = Catalog(tmp_path / "catalog.db")
    kernel = _KernelMustNotRun()

    stats = refresh_catalog(catalog, [root], kernel)

    assert kernel.calls == 0, "KML/GPKG 必须本地读，不能起内核"
    assert stats.errors == []
    assert stats.scanned == 2

    kml = catalog.get(str(root / "points.kml"))
    assert kml is not None and kml.feature_count == 1
    assert kml.kind == "vector"
    assert kml.extra["format"] == "kml"

    gpkg_layer = catalog.get(str(root / "data.gpkg" / "county"))
    assert gpkg_layer is not None, "GPKG 每个图层一条记录"
    assert gpkg_layer.container == str(root / "data.gpkg")
    assert gpkg_layer.feature_count == 1
    assert gpkg_layer.geom_type == "POLYGON"
    catalog.close()


def test_second_refresh_skips_unchanged_kml(tmp_path):
    from gis_cli.datacatalog.scanner import refresh_catalog

    root = tmp_path / "input"
    root.mkdir()
    _kml(root / "points.kml")
    catalog = Catalog(tmp_path / "catalog.db")
    kernel = _KernelMustNotRun()

    refresh_catalog(catalog, [root], kernel)
    stats = refresh_catalog(catalog, [root], kernel)
    assert stats.skipped == 1
    assert stats.scanned == 0
    catalog.close()


def test_second_refresh_skips_unchanged_gpkg(tmp_path):
    """.gpkg 是“容器”，不能被当成已消失每轮删一遍（否则永远重扫）。"""
    from gis_cli.datacatalog.scanner import refresh_catalog

    root = tmp_path / "input"
    root.mkdir()
    _gpkg(root / "data.gpkg")
    catalog = Catalog(tmp_path / "catalog.db")
    kernel = _KernelMustNotRun()

    refresh_catalog(catalog, [root], kernel)
    stats = refresh_catalog(catalog, [root], kernel)
    assert stats.removed == 0, "容器还在，里面的图层记录不能删"
    assert stats.skipped == 1
    assert stats.scanned == 0
    assert catalog.get(str(root / "data.gpkg" / "county")) is not None
    catalog.close()


def test_broken_kml_is_reported_not_crashed(tmp_path):
    from gis_cli.datacatalog.scanner import refresh_catalog

    root = tmp_path / "input"
    root.mkdir()
    (root / "broken.kml").write_text("<kml><Document>", encoding="utf-8")
    catalog = Catalog(tmp_path / "catalog.db")
    stats = refresh_catalog(catalog, [root], _KernelMustNotRun())
    assert stats.errors and "broken.kml" in stats.errors[0]
    catalog.close()

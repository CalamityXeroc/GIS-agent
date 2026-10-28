# -*- coding: utf-8 -*-
"""KML / GPKG / GeoJSON 直读（P0-2）。

三种格式都能用标准库读，所以这些用例**不碰 arcpy**：KML 是 XML、GPKG 是
SQLite（用例里现造一个真库）、GeoJSON 是 JSON。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from gis_cli.ingest.formats import (
    describe_geojson,
    describe_gpkg,
    describe_kml,
    describe_vector,
    kml_to_geojson,
)
from gis_cli.ingest.csv_points import IngestError

KML_POINTS = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>监测点位</name>
    <Placemark><name>甲</name><description>一号点</description>
      <Point><coordinates>116.40,39.90,0</coordinates></Point>
    </Placemark>
    <Placemark><name>乙</name>
      <Point><coordinates>121.47,31.23,0</coordinates></Point>
    </Placemark>
  </Document>
</kml>
"""

KML_MIXED = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>范围</name>
    <Placemark><name>线</name>
      <LineString><coordinates>116.0,39.0 117.0,40.0</coordinates></LineString>
    </Placemark>
    <Placemark><name>面</name>
      <Polygon>
        <outerBoundaryIs><LinearRing>
          <coordinates>116.0,39.0 117.0,39.0 117.0,40.0 116.0,39.0</coordinates>
        </LinearRing></outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _make_gpkg(path: Path) -> Path:
    """现造一个最小但结构正确的 GeoPackage。"""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE gpkg_spatial_ref_sys (
            srs_name TEXT NOT NULL, srs_id INTEGER PRIMARY KEY,
            organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL,
            definition TEXT NOT NULL, description TEXT
        );
        CREATE TABLE gpkg_contents (
            table_name TEXT PRIMARY KEY, data_type TEXT NOT NULL, identifier TEXT,
            description TEXT, last_change TEXT, min_x DOUBLE, min_y DOUBLE,
            max_x DOUBLE, max_y DOUBLE, srs_id INTEGER
        );
        CREATE TABLE gpkg_geometry_columns (
            table_name TEXT NOT NULL, column_name TEXT NOT NULL,
            geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL,
            z TINYINT NOT NULL, m TINYINT NOT NULL,
            PRIMARY KEY (table_name, column_name)
        );
        """
    )
    conn.execute(
        "INSERT INTO gpkg_spatial_ref_sys VALUES ('WGS 84', 4326, 'EPSG', 4326, 'GEOGCS[...]', '')"
    )
    conn.execute(
        "INSERT INTO gpkg_contents VALUES "
        "('cities', 'features', 'cities', '', '2026-10-27T00:00:00Z', 113.0, 22.0, 115.0, 24.0, 4326)"
    )
    conn.execute("INSERT INTO gpkg_geometry_columns VALUES ('cities', 'geom', 'POINT', 4326, 0, 0)")
    conn.execute("CREATE TABLE cities (fid INTEGER PRIMARY KEY, geom BLOB, name TEXT, pop REAL)")
    conn.executemany(
        "INSERT INTO cities (name, pop) VALUES (?, ?)",
        [("甲", 100.5), ("乙", 200.5), ("丙", None)],
    )
    conn.commit()
    conn.close()
    return path


# -------------------------------------------------------------------- KML


def test_describe_kml_reads_name_count_extent(tmp_path):
    record = describe_kml(_write(tmp_path / "points.kml", KML_POINTS))
    assert record["kind"] == "vector"
    assert record["name"] == "监测点位"
    assert record["feature_count"] == 2
    assert record["geom_type"] == "Point"
    assert record["extent"] == [116.4, 31.23, 121.47, 39.9]
    assert [f["name"] for f in record["fields"]] == ["Name", "Description"]
    assert record["extra"]["format"] == "kml"


def test_describe_kml_handles_mixed_geometry(tmp_path):
    record = describe_kml(_write(tmp_path / "mixed.kml", KML_MIXED))
    assert record["feature_count"] == 2
    assert record["geom_type"] == "Mixed"
    assert record["extra"]["geometry_types"] == {"LineString": 1, "Polygon": 1}


def test_describe_kml_rejects_empty_or_broken(tmp_path):
    with pytest.raises(IngestError):
        describe_kml(_write(tmp_path / "empty.kml", "<kml xmlns='http://www.opengis.net/kml/2.2'/>"))
    with pytest.raises(IngestError) as excinfo:
        describe_kml(_write(tmp_path / "broken.kml", "<kml><Document>"))
    assert "KML 解析失败" in str(excinfo.value)


def test_kml_to_geojson_converts_points_and_polygons(tmp_path):
    result = kml_to_geojson(_write(tmp_path / "points.kml", KML_POINTS), tmp_path / "points.geojson")
    payload = json.loads(Path(result["output"]).read_text(encoding="utf-8"))
    assert result["feature_count"] == 2
    assert payload["features"][0]["geometry"] == {"type": "Point", "coordinates": [116.4, 39.9]}
    assert payload["features"][0]["properties"]["Name"] == "甲"

    polygon = kml_to_geojson(_write(tmp_path / "mixed.kml", KML_MIXED), tmp_path / "mixed.geojson")
    mixed = json.loads(Path(polygon["output"]).read_text(encoding="utf-8"))
    kinds = [f["geometry"]["type"] for f in mixed["features"]]
    assert kinds == ["LineString", "Polygon"]
    assert len(mixed["features"][1]["geometry"]["coordinates"][0]) == 4


# ------------------------------------------------------------------- GPKG


def test_describe_gpkg_lists_layers_with_fields_and_extent(tmp_path):
    records = describe_gpkg(_make_gpkg(tmp_path / "data.gpkg"))
    assert len(records) == 1
    record = records[0]
    assert record["name"] == "cities"
    assert record["kind"] == "feature_class"
    assert record["container"] == str(tmp_path / "data.gpkg"), "container 指向 .gpkg，与 .gdb 记账方式一致"
    assert record["feature_count"] == 3
    assert record["geom_type"] == "POINT"
    assert record["crs_name"] == "EPSG:4326"
    assert record["wkid"] == 4326
    assert record["extent"] == [113.0, 22.0, 115.0, 24.0]
    fields = {f["name"]: f["type"] for f in record["fields"]}
    assert fields == {"fid": "Integer", "name": "String", "pop": "Double"}
    assert "geom" not in fields, "几何列不算属性字段"


def test_describe_gpkg_rejects_plain_sqlite(tmp_path):
    plain = tmp_path / "plain.gpkg"
    conn = sqlite3.connect(plain)
    conn.execute("CREATE TABLE t (a INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(IngestError) as excinfo:
        describe_gpkg(plain)
    assert "不是 GeoPackage" in str(excinfo.value)


# ---------------------------------------------------------------- GeoJSON


def test_describe_geojson_counts_and_bbox(tmp_path):
    source = _write(
        tmp_path / "x.geojson",
        json.dumps(
            {
                "type": "FeatureCollection",
                "name": "点位",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [116.4, 39.9]},
                        "properties": {"名称": "甲", "值": 1.5},
                    },
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [121.5, 31.2]},
                        "properties": {"名称": "乙", "值": 2},
                    },
                ],
            },
            ensure_ascii=False,
        ),
    )
    record = describe_geojson(source)
    assert record["feature_count"] == 2
    assert record["geom_type"] == "Point"
    assert record["extent"] == [116.4, 31.2, 121.5, 39.9]
    fields = {f["name"]: f["type"] for f in record["fields"]}
    assert fields["值"] == "Double"
    assert fields["名称"] == "String"


def test_describe_vector_dispatches_by_suffix(tmp_path):
    assert describe_vector(_write(tmp_path / "a.kml", KML_POINTS))["feature_count"] == 2
    assert describe_vector(_make_gpkg(tmp_path / "b.gpkg"))[0]["name"] == "cities"
    with pytest.raises(IngestError) as excinfo:
        describe_vector(_write(tmp_path / "c.txt", "x"))
    assert "不支持直读" in str(excinfo.value)

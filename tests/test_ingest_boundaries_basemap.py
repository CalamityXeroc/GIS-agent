# -*- coding: utf-8 -*-
"""行政边界（按需下载 + 缓存 + 许可登记）与底图（天地图 / OSM）。

网络全部注入：``fetch(url, timeout) -> bytes``。CI 不联网，也不该联网。
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.error import URLError

import pytest

from gis_cli.datacatalog.store import Catalog
from gis_cli.ingest import IngestError, register_dataset
from gis_cli.ingest.basemap import (
    BasemapKeyError,
    lonlat_to_tile,
    osm_spec,
    render_basemap,
    spec_for,
    tianditu_spec,
    tile_bounds,
)
from gis_cli.ingest.boundaries import build_source, ensure_boundary

BOUNDARY_GEOJSON = json.dumps(
    {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[113.0, 22.0], [115.0, 22.0], [115.0, 24.0], [113.0, 22.0]]],
                },
                "properties": {"name": "甲市", "adcode": 440100},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[115.0, 22.0], [117.0, 22.0], [117.0, 24.0], [115.0, 22.0]]],
                },
                "properties": {"name": "乙市", "adcode": 440200},
            },
        ],
    },
    ensure_ascii=False,
).encode("utf-8")


# ------------------------------------------------------------------ 边界


def test_ensure_boundary_downloads_then_uses_cache(tmp_path):
    calls: list[str] = []

    def fetch(url, timeout):
        calls.append(url)
        return BOUNDARY_GEOJSON

    cache = tmp_path / "cache"
    first = ensure_boundary("province", cache_dir=cache, fetch=fetch)
    assert first.from_cache is False
    assert first.feature_count == 2
    assert Path(first.path).is_file()
    assert calls == ["https://geojson.cn/api/china/1.6.2/100000.json"], "主源变了要跟着改"
    assert first.license == "unknown", "数据源没声明许可，不能替用户认领"
    assert first.acquired_at

    meta = json.loads((cache / "province_100000.meta.json").read_text(encoding="utf-8"))
    assert meta["source_url"].endswith("100000.json")
    assert meta["attribution"] == "geojson.cn"
    assert meta["license"] == "unknown"

    second = ensure_boundary("province", cache_dir=cache, fetch=fetch)
    assert second.from_cache is True
    assert len(calls) == 1, "第二次必须走缓存，不能再下载"
    assert second.feature_count == 2


def test_ensure_boundary_falls_back_to_next_source(tmp_path):
    """主源挂了（实测 DataV 老接口 404）时要接着试下一个，并把真用的源记下来。"""
    calls: list[str] = []

    def fetch(url, timeout):
        calls.append(url)
        if "geojson.cn" in url:
            raise URLError("HTTP Error 404: Not Found")
        return BOUNDARY_GEOJSON

    result = ensure_boundary("province", cache_dir=tmp_path, fetch=fetch)
    assert len(calls) == 2
    assert "datav.aliyun.com" in result.source_url
    assert result.attribution == "阿里云 DataV.GeoAtlas"
    assert result.feature_count == 2


def test_ensure_boundary_force_redownloads(tmp_path):
    calls: list[int] = []

    def fetch(url, timeout):
        calls.append(1)
        return BOUNDARY_GEOJSON

    ensure_boundary("province", cache_dir=tmp_path, fetch=fetch)
    ensure_boundary("province", cache_dir=tmp_path, fetch=fetch, force=True)
    assert len(calls) == 2


def test_city_level_requires_parent_adcode():
    """市级边界不给 adcode 就明确报错 —— 全国文件里装的是省，不是市。"""
    with pytest.raises(IngestError) as excinfo:
        build_source("city")
    message = str(excinfo.value)
    assert "adcode" in message
    assert "440000" in message, "报错要给出可照抄的例子"


def test_city_level_uses_given_adcode():
    source = build_source("city", "440000")
    assert source.urls[0].endswith("/440000.json"), "市级拿的是该省的下一级（各地市）"
    with pytest.raises(IngestError):
        build_source("county", "粤44")  # 非数字也不行
    with pytest.raises(IngestError) as excinfo:
        build_source("town")
    assert "行政级别只认" in str(excinfo.value)


def test_download_failure_message_tells_where_to_put_the_file(tmp_path):
    def fetch(url, timeout):
        raise URLError("getaddrinfo failed")

    with pytest.raises(IngestError) as excinfo:
        ensure_boundary("province", cache_dir=tmp_path, fetch=fetch)
    message = str(excinfo.value)
    assert "下载失败" in message
    assert "province_100000.geojson" in message, "要告诉用户手动下载后放哪"


def test_bad_boundary_payload_is_rejected(tmp_path):
    with pytest.raises(IngestError) as excinfo:
        ensure_boundary("province", cache_dir=tmp_path, fetch=lambda url, timeout: b"<html>403</html>")
    message = str(excinfo.value)
    assert "下载失败" in message
    assert "不是合法 JSON" in message
    with pytest.raises(IngestError) as excinfo:
        ensure_boundary(
            "province", cache_dir=tmp_path, fetch=lambda url, timeout: b'{"type": "FeatureCollection"}'
        )
    assert "没有要素" in str(excinfo.value)


def test_boundary_is_registered_as_source_boundary(tmp_path):
    catalog = Catalog(tmp_path / "catalog.db")
    result = ensure_boundary("province", cache_dir=tmp_path / "cache", fetch=lambda u, t: BOUNDARY_GEOJSON)
    register_dataset(catalog, result.to_record(), source="boundary", license=result.license)

    stored = catalog.get(result.path)
    assert stored.source == "boundary"
    assert stored.license == "unknown"
    assert stored.acquired_at
    assert stored.feature_count == 2
    assert stored.crs_name == "", "数据源没声明坐标系，不能写成 WGS 84"
    assert "GCJ-02" in stored.extra["crs_note"]
    catalog.close()


# ------------------------------------------------------------------ 底图


def test_tianditu_without_key_explains_how_to_get_one(monkeypatch):
    monkeypatch.delenv("TIANDITU_KEY", raising=False)
    monkeypatch.delenv("TIANDITU_TK", raising=False)
    with pytest.raises(BasemapKeyError) as excinfo:
        tianditu_spec(env={})
    message = str(excinfo.value)
    assert "TIANDITU_KEY" in message
    assert "console.tianditu.gov.cn" in message
    assert "osm" in message, "要给出不需要 key 的替代方案"


def test_tianditu_key_from_env_builds_wmts_url():
    spec = tianditu_spec("img", env={"TIANDITU_KEY": "abc123"})
    url = spec.tile_url(168, 96, 8)
    assert "tk=abc123" in url
    assert "LAYER=img" in url
    assert "TILEMATRIX=8" in url
    assert "TILEROW=96" in url
    assert "TILECOL=168" in url
    assert spec.license == "天地图条款"


def test_osm_needs_no_key_and_is_odbl():
    spec = osm_spec()
    assert spec.license == "ODbL"
    assert "OpenStreetMap" in spec.attribution
    assert spec.tile_url(1, 2, 3).endswith("/3/1/2.png")
    assert "tk=" not in spec.tile_url(1, 2, 3)


def test_spec_for_dispatches_and_rejects_unknown():
    assert spec_for("osm").provider == "osm"
    assert spec_for("tianditu", key="k").provider == "tianditu"
    with pytest.raises(IngestError) as excinfo:
        spec_for("gaode")
    assert "只认 tianditu / osm" in str(excinfo.value)


def test_tile_math_roundtrip():
    x, y = lonlat_to_tile(116.4, 39.9, 10)
    west, south, east, north = tile_bounds(x, y, 10)
    assert west <= 116.4 <= east
    assert south <= 39.9 <= north
    # 赤道/本初子午线在第 0 块瓦片
    assert lonlat_to_tile(0.0, 0.0, 1) == (1, 1)


def _png(color=(210, 225, 240)) -> bytes:
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (256, 256), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_render_basemap_writes_png_world_file_and_meta(tmp_path):
    result = render_basemap(
        [116.3, 39.8, 116.5, 40.0],
        10,
        osm_spec(),
        tmp_path / "basemap" / "osm.png",
        fetch=lambda url, timeout: _png(),
    )
    png = Path(result.png_path)
    assert png.is_file()
    assert png.stat().st_size > 0
    assert result.tiles_fetched >= 1
    assert result.tiles_failed == 0

    world = Path(result.world_file)
    lines = world.read_text(encoding="ascii").split()
    assert len(lines) == 6
    assert float(lines[0]) > 0 and float(lines[3]) < 0, "像素大小要正/负成对"

    meta = json.loads(Path(result.meta_path).read_text(encoding="utf-8"))
    assert meta["license"] == "ODbL"
    assert meta["attribution"].startswith("© OpenStreetMap")
    assert meta["extent"] == [116.3, 39.8, 116.5, 40.0]
    assert meta["size"][0] >= 2 and meta["size"][1] >= 2
    assert "ODbL" in result.summary()


def test_render_basemap_caches_tiles(tmp_path):
    calls: list[str] = []

    def fetch(url, timeout):
        calls.append(url)
        return _png()

    kwargs = dict(cache_dir=tmp_path / "tiles", fetch=fetch)
    render_basemap([116.3, 39.8, 116.5, 40.0], 10, osm_spec(), tmp_path / "a.png", **kwargs)
    first = len(calls)
    render_basemap([116.3, 39.8, 116.5, 40.0], 10, osm_spec(), tmp_path / "b.png", **kwargs)
    assert len(calls) == first, "同一块瓦片第二次必须读磁盘缓存"


def test_render_basemap_rejects_oversized_request(tmp_path):
    with pytest.raises(IngestError) as excinfo:
        render_basemap([-179, -80, 179, 80], 18, osm_spec(), tmp_path / "x.png", fetch=lambda u, t: _png())
    assert "范围太大" in str(excinfo.value)


def test_render_basemap_rejects_bad_extent(tmp_path):
    with pytest.raises(IngestError) as excinfo:
        render_basemap([116.5, 39.8, 116.3, 40.0], 10, osm_spec(), tmp_path / "x.png", fetch=lambda u, t: _png())
    assert "范围不合法" in str(excinfo.value)
    with pytest.raises(IngestError) as excinfo:
        render_basemap([500000, 3400000, 500100, 3400100], 10, osm_spec(), tmp_path / "x.png", fetch=lambda u, t: _png())
    assert "投影坐标" in str(excinfo.value)


def test_render_basemap_reports_invalid_key_xml(tmp_path):
    """天地图 key 无效时返回 XML 而不是图片 —— 要说清是 key 的问题，不是网络。"""

    def fetch(url, timeout):
        return b"<ExceptionReport>Invalid key</ExceptionReport>"

    with pytest.raises(IngestError) as excinfo:
        render_basemap(
            [116.3, 39.8, 116.5, 40.0], 10, tianditu_spec(key="bad"), tmp_path / "x.png", fetch=fetch
        )
    assert "不是图片" in str(excinfo.value)
    assert "key" in str(excinfo.value)


def test_bad_tiles_are_not_cached(tmp_path):
    """key 无效返回的 XML 不能写进瓦片缓存（否则下次永远读出一份坏数据）。"""

    def fetch(url, timeout):
        return b"<ExceptionReport>Invalid key</ExceptionReport>"

    cache = tmp_path / "tiles"
    with pytest.raises(IngestError):
        render_basemap(
            [116.3, 39.8, 116.5, 40.0], 10, tianditu_spec(key="bad"), tmp_path / "x.png",
            cache_dir=cache, fetch=fetch,
        )
    assert not list(cache.rglob("*.png")), "坏响应不能进缓存"

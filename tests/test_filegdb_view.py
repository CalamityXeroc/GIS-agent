# -*- coding: utf-8 -*-
"""`.gdb` 视口读的两条路（DAIR-393）。

- 有 ``osgeo``：GDAL ``SetSpatialFilterRect`` 只读视口内的要素，计数精确；
- 没有 ``osgeo``：退回原来的 ``arcpy.da.SearchCursor`` 顺序扫描，``scan_cap``
  扫到就停、``view_count_exact=false``——兜底行为要与修复前逐字相同。

这里两个依赖都用假的：真 GDAL 只在 ArcGIS Pro 的 Python 里有，真 arcpy 只在
Windows 上有。几何转换本身是纯 Python（WKB→Esri JSON），直接拿手工 WKB 钉。
"""

from __future__ import annotations

import struct
import sys
import types

import pytest

from gis_cli.report import filegdb_view, html_pack


# ---------------------------------------------------------------- WKB 构造
def _wkb_point(x: float, y: float, *, z: float | None = None) -> bytes:
    if z is None:
        return struct.pack("<BIdd", 1, 1, x, y)
    return struct.pack("<BIddd", 1, 1001, x, y, z)


def _wkb_line(points) -> bytes:
    body = struct.pack("<BI", 1, 2) + struct.pack("<I", len(points))
    for x, y in points:
        body += struct.pack("<dd", x, y)
    return body


def _wkb_polygon(rings) -> bytes:
    body = struct.pack("<BI", 1, 3) + struct.pack("<I", len(rings))
    for ring in rings:
        body += struct.pack("<I", len(ring))
        for x, y in ring:
            body += struct.pack("<dd", x, y)
    return body


def _wkb_multipolygon(polygons) -> bytes:
    body = struct.pack("<BI", 1, 6) + struct.pack("<I", len(polygons))
    for rings in polygons:
        body += _wkb_polygon(rings)  # 子几何各自带 byteorder + type 头
    return body


def _signed_area(points) -> float:
    total = 0.0
    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        total += x1 * y2 - x2 * y1
    return total / 2.0


#: 逆时针外环（GeoJSON/OGC 方向）——转换后必须被摆成 Esri 的顺时针。
OUTER_CCW = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 0.0]]
#: 顺时针洞（GeoJSON/OGC 方向）——转换后必须被摆成 Esri 的逆时针。
HOLE_CW = [[2.0, 2.0], [2.0, 6.0], [6.0, 6.0], [2.0, 2.0]]


# ---------------------------------------------------------------- 几何转换
def test_wkb_point_and_z_variant_keep_xy_only():
    assert filegdb_view.esri_geometry_from_wkb(_wkb_point(1.5, 2.5)) == {"x": 1.5, "y": 2.5}
    assert filegdb_view.esri_geometry_from_wkb(_wkb_point(1.5, 2.5, z=9.0)) == {
        "x": 1.5,
        "y": 2.5,
    }


def test_wkb_line_becomes_paths():
    geometry = filegdb_view.esri_geometry_from_wkb(_wkb_line([(0, 0), (1, 1), (2, 0)]))
    assert geometry == {"paths": [[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]]}


def test_wkb_polygon_rings_are_oriented_for_esri():
    """外环 → 顺时针（负面积）、洞 → 逆时针（正面积）：``_rings_to_polygons`` 靠这个判洞。"""
    geometry = filegdb_view.esri_geometry_from_wkb(_wkb_polygon([OUTER_CCW, HOLE_CW]))
    outer, hole = geometry["rings"]
    assert _signed_area(outer) < 0, "外环要摆成顺时针"
    assert _signed_area(hole) > 0, "洞要摆成逆时针"


def test_wkb_multipolygon_flattens_parts_with_own_outer_rings():
    second = [[[20.0, 20.0], [30.0, 20.0], [30.0, 30.0], [20.0, 20.0]]]
    geometry = filegdb_view.esri_geometry_from_wkb(
        _wkb_multipolygon([[OUTER_CCW], second])
    )
    rings = geometry["rings"]
    assert len(rings) == 2, "两个部件各留一个外环"
    assert all(_signed_area(ring) < 0 for ring in rings)


def test_wkb_unknown_geometry_raises_instead_of_inventing_one():
    with pytest.raises(ValueError):
        filegdb_view.esri_geometry_from_wkb(struct.pack("<BII", 1, 7, 0))


def test_container_split_handles_feature_dataset_and_plain_file():
    container, member = filegdb_view._split_container(r"E:\ws\input\数据.gdb\数据集\街道")
    assert container == r"E:\ws\input\数据.gdb"
    assert member == "数据集/街道"
    assert filegdb_view._split_container(r"E:\ws\input\x.gpkg") == (
        r"E:\ws\input\x.gpkg",
        "",
    )


# ---------------------------------------------------------------- GDAL 快路
class _FakeGeometry:
    def __init__(self, wkb: bytes, envelope=(0.0, 1.0, 0.0, 1.0)) -> None:
        self._wkb = wkb
        # GDAL 的 GetEnvelope() 顺序：(minx, maxx, miny, maxy)
        self._envelope = tuple(envelope)

    def ExportToWkb(self) -> bytes:
        return self._wkb

    def GetEnvelope(self):  # noqa: ANN201
        return self._envelope


class _FakeFeature:
    def __init__(self, wkb: bytes, values: dict[str, object], *, fid: int = 1,
                 envelope=(0.0, 1.0, 0.0, 1.0)) -> None:
        self._geometry = _FakeGeometry(wkb, envelope)
        self._values = values
        self._fid = int(fid)

    def GetGeometryRef(self):
        return self._geometry

    def GetFID(self):
        return self._fid

    def GetField(self, index: int):
        return self._values[str(index)]


class _FakeLayer:
    def __init__(self, features, *, fields=("NAME", "LEN")) -> None:
        self.features = list(features)
        self.fields = list(fields)
        self.filter_calls: list[tuple[float, ...]] = []

    def GetName(self) -> str:
        return "city"

    def GetFIDColumn(self) -> str:
        return "OBJECTID"

    def GetLayerDefn(self):
        fields = self.fields

        class _Defn:
            def GetFieldIndex(self, name):  # noqa: ANN001
                return fields.index(name) if name in fields else -1

        return _Defn()

    def SetSpatialFilterRect(self, *box):  # noqa: ANN002
        self.filter_calls.append(tuple(box))

    def GetFeature(self, oid):  # noqa: ANN001
        for feature in self.features:
            if feature.GetFID() == oid:
                return feature
        return None

    def __iter__(self):
        return iter(self.features)


class _FakeOgr:
    def __init__(self, layer: _FakeLayer) -> None:
        self._layer = layer
        self.opened: list[str] = []

    def DontUseExceptions(self) -> None:
        return None

    def Open(self, path, update=0):  # noqa: ANN001
        self.opened.append(str(path))
        layer = self._layer

        class _Dataset:
            def GetLayerByName(self, name):  # noqa: ANN001
                return layer if name in ("city", "街道") else None

            def GetLayerCount(self):
                return 1

            def GetLayerByIndex(self, index):  # noqa: ANN001
                return layer

        return _Dataset()


def _install_fake_osgeo(monkeypatch: pytest.MonkeyPatch, layer: _FakeLayer) -> _FakeOgr:
    fake = _FakeOgr(layer)
    module = types.ModuleType("osgeo")
    module.ogr = fake
    monkeypatch.setitem(sys.modules, "osgeo", module)
    return fake


def test_read_view_uses_gdal_spatial_filter_and_exact_count(monkeypatch: pytest.MonkeyPatch):
    features = [
        _FakeFeature(
            _wkb_point(100.0 + index, 30.0),
            {"0": f"路{index}", "1": index},
            fid=index + 1,
            envelope=(100.0, 100.0, 30.0, 30.0),
        )
        for index in range(3)
    ]
    layer = _FakeLayer(features)
    fake = _install_fake_osgeo(monkeypatch, layer)

    payload = filegdb_view.read_view(
        r"E:\ws\input\数据.gdb\city",
        (99.0, 29.0, 101.0, 31.0),
        limit=10,
        fields=["NAME", "LEN"],
    )

    assert payload["engine"] == "gdal" and payload.get("fallback") is None
    assert fake.opened == [r"E:\ws\input\数据.gdb"]
    assert layer.filter_calls == [(99.0, 29.0, 101.0, 31.0)], "视口必须交给 GDAL 的索引"
    assert payload["view_total"] == 3 and payload["view_count_exact"] is True
    assert [row[1] for row in payload["rows"]] == [["路0", 0], ["路1", 1], ["路2", 2]]


def test_read_view_keeps_counting_after_collect_limit(monkeypatch: pytest.MonkeyPatch):
    """收满 ``limit`` 条还要接着数：计数精确是这条快路的核心收益（不再有「≥N」）。"""
    features = [
        _FakeFeature(
            _wkb_line([(100.0 + index, 30.0), (100.0 + index, 31.0)]),
            {},
            fid=index + 1,
            envelope=(100.0, 100.0, 30.0, 31.0),
        )
        for index in range(5)
    ]
    _install_fake_osgeo(monkeypatch, _FakeLayer(features))

    payload = filegdb_view.read_view(r"E:\ws\input\数据.gdb\city", (99.0, 29.0, 101.0, 31.0), limit=2)

    assert len(payload["geoms"]) == 2
    assert payload["view_total"] == 5
    assert payload["view_count_exact"] is True


def test_read_view_injects_simplify_and_rebuilds_rows(monkeypatch: pytest.MonkeyPatch):
    features = [
        _FakeFeature(
            _wkb_point(100.0, 30.0), {"0": "甲", "1": 1.0}, envelope=(100.0, 100.0, 30.0, 30.0)
        )
    ]
    _install_fake_osgeo(monkeypatch, _FakeLayer(features))
    calls: list[dict] = []

    def fake_simplify(geoms, *, max_points=None):
        calls.append({"count": len(geoms), "max_points": max_points})
        return geoms, {"output_points": 1}

    payload = filegdb_view.read_view(
        r"E:\ws\input\数据.gdb\city",
        (99.0, 29.0, 101.0, 31.0),
        limit=10,
        fields=["NAME", "LEN"],
        max_points=1234,
        simplify=fake_simplify,
    )

    assert calls == [{"count": 1, "max_points": 1234}]
    assert payload["simplify"] == {"output_points": 1}
    assert payload["rows"][0][1] == ["甲", 1.0]


# ---------------------------------------------------------------- arcpy 兜底
class _Cursor:
    def __init__(self, rows) -> None:
        self._rows = list(rows)

    def __iter__(self):
        return iter(self._rows)

    def __enter__(self):
        return iter(self._rows)

    def __exit__(self, *exc):  # noqa: ANN002
        return False


class _Extent:
    def __init__(self, box) -> None:
        self.XMin, self.YMin, self.XMax, self.YMax = box


class _Geometry:
    def __init__(self, box, json_text: str) -> None:
        self.extent = _Extent(box)
        self.JSON = json_text


def _install_fake_arcpy(monkeypatch: pytest.MonkeyPatch, rows) -> list[str]:
    seen: list[str] = []
    arcpy = types.ModuleType("arcpy")
    da = types.ModuleType("arcpy.da")

    class _Search:
        def __init__(self, path, columns):  # noqa: ANN001
            seen.append(str(path))

        def __enter__(self):
            return _Cursor(rows)

        def __exit__(self, *exc):  # noqa: ANN002
            return False

    da.SearchCursor = _Search
    arcpy.da = da
    monkeypatch.setitem(sys.modules, "arcpy", arcpy)
    monkeypatch.setitem(sys.modules, "arcpy.da", da)
    # ``None`` 让 ``from osgeo import ogr`` 直接 ImportError：模拟没装 GDAL 的旧环境。
    monkeypatch.setitem(sys.modules, "osgeo", None)
    return seen


def _in_view():
    return _Geometry((100.0, 30.0, 101.0, 31.0), '{"x": 100.5, "y": 30.5}')


def _out_of_view():
    return _Geometry((120.0, 40.0, 121.0, 41.0), '{"x": 120.5, "y": 40.5}')


def test_read_view_falls_back_to_arcpy_scan_and_keeps_legacy_inexact(monkeypatch: pytest.MonkeyPatch):
    """没有 osgeo：与修复前同一口径——extent 判交、扫到 ``scan_cap`` 停下说「≥N」。"""
    rows = [(_in_view(),), (_in_view(),), (_out_of_view(),), (_in_view(),), (_in_view(),)]
    seen = _install_fake_arcpy(monkeypatch, rows)

    payload = filegdb_view.read_view(
        r"E:\ws\input\数据.gdb\city",
        (99.0, 29.0, 101.0, 31.0),
        limit=10,
        scan_cap=2,
    )

    assert seen == [r"E:\ws\input\数据.gdb\city"]
    assert payload["engine"] == "arcpy"
    assert payload.get("fallback"), "要能看出为什么走了慢路"
    assert payload["view_total"] == 2, "扫到第 3 行就停：只数前两行（都是视口内的）"
    assert payload["view_count_exact"] is False


def test_read_view_fallback_counts_exactly_when_table_is_short(monkeypatch: pytest.MonkeyPatch):
    rows = [(_in_view(),), (_out_of_view(),), (_in_view(),)]
    _install_fake_arcpy(monkeypatch, rows)

    payload = filegdb_view.read_view(
        r"E:\ws\input\数据.gdb\city",
        (99.0, 29.0, 101.0, 31.0),
        limit=10,
        scan_cap=20_000,
    )

    assert payload["view_total"] == 2
    assert payload["view_count_exact"] is True


# ---------------------------------------------------------------- 内核代码
def test_view_probe_code_uses_gdal_hook_and_keeps_safe_fallback():
    code = html_pack.layer_probe_code(
        "output/结果.gdb/街道", limit=10, bbox=(0.0, 0.0, 1.0, 1.0), scan_cap=60000
    )
    assert "SetSpatialFilterRect" in code, "快路要靠 GDAL 的空间索引"
    assert "SearchCursor" in code, "兜底还是原来的顺序扫描"
    assert "_scan_cap = 60000" in code and "_limit = 10" in code
    assert "MakeFeatureLayer" not in code and "SelectLayerByAttribute" not in code
    assert "esri_geometry_from_wkb" in code


# ---------------------------------------------------------------- 包络索引
def _write_sample_index(tmp_path, key="k"):
    target = tmp_path / "idx" / "a.idx"
    filegdb_view.write_index(
        target,
        key,
        [10, 11, 12],
        [
            (100.0, 30.0, 100.2, 30.2),
            (120.0, 40.0, 120.2, 40.2),
            (100.1, 30.1, 100.3, 30.3),
        ],
    )
    return target


def test_index_round_trip_queries_bbox(tmp_path):
    target = _write_sample_index(tmp_path)
    loaded = filegdb_view.load_index(target, "k")
    assert loaded is not None
    assert filegdb_view.query_index(loaded, (99.0, 29.0, 101.0, 31.0)) == [10, 12]
    assert filegdb_view.query_index(loaded, (119.0, 39.0, 121.0, 41.0)) == [11]
    assert filegdb_view.query_index(loaded, (0.0, 0.0, 1.0, 1.0)) == []


def test_index_must_match_data_key(tmp_path):
    target = _write_sample_index(tmp_path, key="old")
    assert filegdb_view.load_index(target, "new") is None, "数据变了就要重扫，不能拿旧索引"


def test_index_ignores_corrupt_file(tmp_path):
    target = tmp_path / "bad.idx"
    target.write_bytes(b"not an index")
    assert filegdb_view.load_index(target, "k") is None


def test_read_view_dumps_index_while_scanning(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """第一次视口：整层扫的同时把包络写出去，下一次就不用再扫（DAIR-393）。"""
    features = [
        _FakeFeature(_wkb_point(100.0, 30.0), {}, fid=7, envelope=(100.0, 100.0, 30.0, 30.0)),
        _FakeFeature(_wkb_point(120.0, 40.0), {}, fid=8, envelope=(120.0, 120.0, 40.0, 40.0)),
    ]
    _install_fake_osgeo(monkeypatch, _FakeLayer(features))
    target = tmp_path / "idx" / "a.idx"

    payload = filegdb_view.read_view(
        r"E:\ws\input\数据.gdb\city",
        (99.0, 29.0, 101.0, 31.0),
        limit=10,
        index_path=str(target),
        index_key="k",
    )

    assert payload["engine"] == "gdal"
    loaded = filegdb_view.load_index(target, "k")
    assert loaded is not None and loaded[0].tolist() == [7, 8], "整层都要进索引，不只是视口内的"
    assert filegdb_view.query_index(loaded, (99.0, 29.0, 101.0, 31.0)) == [7]


def test_build_index_skips_when_index_is_already_fresh(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """预览先扫出索引后，预热再让内核建同一层不重扫（DAIR-397 第 3 条）。

    预热与预览共用内核串行锁：主进程判「没索引」到真正拿到锁之间，预览那次整层扫
    已经把索引写出来了。真机实测同一层被建两遍（多花 3.7 s），所以判据放在内核
    侧——持锁后再按指纹查一次。
    """

    class _CountingLayer(_FakeLayer):
        scans = 0

        def __iter__(self):
            type(self).scans += 1
            return super().__iter__()

    features = [
        _FakeFeature(_wkb_point(100.0, 30.0), {}, fid=7, envelope=(100.0, 100.0, 30.0, 30.0)),
        _FakeFeature(_wkb_point(120.0, 40.0), {}, fid=8, envelope=(120.0, 120.0, 40.0, 40.0)),
    ]
    layer = _CountingLayer(features)
    fake = _install_fake_osgeo(monkeypatch, layer)
    target = tmp_path / "idx" / "a.idx"

    # 1) 预览那次整层扫把索引写出来（带 index_target 的视口读）。
    filegdb_view.read_view(
        r"E:\ws\input\数据.gdb\city",
        (99.0, 29.0, 101.0, 31.0),
        limit=10,
        index_path=str(target),
        index_key="k",
    )
    assert layer.scans == 1 and fake.opened == [r"E:\ws\input\数据.gdb"]

    # 2) 预热随后让内核建同一层：按指纹跳过，不重开数据、不重扫整层。
    payload = filegdb_view.build_index(r"E:\ws\input\数据.gdb\city", str(target), "k")
    assert payload["skipped"] is True and payload["error"] == ""
    assert payload["count"] == 2, "跳过时也要能报出索引里的要素数"
    assert layer.scans == 1, "内层不得再扫一遍"
    assert len(fake.opened) == 1, "连数据源都不该重新打开"

    # 3) 数据指纹变了仍然要重建（不能把「已有」当成「新鲜」）。
    payload = filegdb_view.build_index(r"E:\ws\input\数据.gdb\city", str(target), "k2")
    assert not payload.get("skipped") and payload["count"] == 2
    assert layer.scans == 2


def test_read_by_oids_reads_only_the_requested_features(monkeypatch: pytest.MonkeyPatch):
    features = [
        _FakeFeature(
            _wkb_point(100.0 + index, 30.0),
            {"0": f"路{index}", "1": index},
            fid=index + 20,
            envelope=(100.0 + index, 100.0 + index, 30.0, 30.0),
        )
        for index in range(3)
    ]
    _install_fake_osgeo(monkeypatch, _FakeLayer(features))

    payload = filegdb_view.read_by_oids(
        r"E:\ws\input\数据.gdb\city",
        [21, 22],
        limit=10,
        fields=["NAME", "LEN"],
    )

    assert payload["engine"] == "gdal_index"
    assert payload["view_total"] == 2 and payload["view_count_exact"] is True
    assert [row[1][0] for row in payload["rows"]] == ["路1", "路2"]


def test_read_by_oids_supports_fid_column_as_attribute(monkeypatch: pytest.MonkeyPatch):
    """``fields=OBJECTID``（属性表同步要 OID）不能把整层变空：``OBJECTID`` 是
    FID 列，不算普通字段（2026-11-24 真机复现：图例「已给 0」、地图空白）。"""
    features = [
        _FakeFeature(
            _wkb_point(100.0 + index, 30.0),
            {"0": f"路{index}", "1": index},
            fid=index + 5,
            envelope=(100.0 + index, 100.0 + index, 30.0, 30.0),
        )
        for index in range(2)
    ]
    _install_fake_osgeo(monkeypatch, _FakeLayer(features))

    payload = filegdb_view.read_by_oids(
        r"E:\ws\input\数据.gdb\city",
        [5, 6],
        limit=10,
        fields=["OBJECTID", "NAME"],
    )

    # 带 fields 的请求只回 rows（geoms 清空，DAIR-394 非阻塞项 1）：
    # 几何不能因属性列对不上而变空，属性值也不能错位。
    assert payload["geoms"] == []
    assert [row[1] for row in payload["rows"]] == [[5, "路0"], [6, "路1"]]
    assert all(row[0] for row in payload["rows"]), "几何不能被属性列问题连累"


def test_missing_field_keeps_geometry_with_null_attribute(monkeypatch: pytest.MonkeyPatch):
    _install_fake_osgeo(
        monkeypatch,
        _FakeLayer([_FakeFeature(_wkb_point(100.0, 30.0), {}, envelope=(100.0, 100.0, 30.0, 30.0))]),
    )

    payload = filegdb_view.read_by_oids(
        r"E:\ws\input\数据.gdb\city", [1], limit=10, fields=["不存在"]
    )

    assert len(payload["rows"]) == 1 and payload["rows"][0][0], "几何不能被属性列问题连累"
    assert payload["rows"][0][1] == [None]

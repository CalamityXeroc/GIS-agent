# -*- coding: utf-8 -*-
"""``gis_cli.report.shapefile_view`` 的单测（DAIR-386）。

视口裁剪的理由与实测数字见模块文档：内置 arcpy 全表扫描 3,614ms / 按 FID 白名单
729ms / 本模块 55ms（134,819 条道路）。这里测的是「读得对」：命中判定用记录头
envelope、多部件与环结构原样保留、DBF 只按记录号取被选中的行。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _shapefiles import SHP_POINT, SHP_POLYGON, SHP_POLYLINE, write_shapefile
from gis_cli.report import shapefile_view as sv


def _points(tmp_path: Path, coords) -> Path:
    return write_shapefile(tmp_path / "点.shp", shape_type=SHP_POINT, shapes=list(coords))


def test_points_are_filtered_by_view_bbox(tmp_path: Path):
    path = _points(tmp_path, [[100.0, 30.0], [120.0, 40.0], [100.5, 30.5]])

    hits, total = sv.view_hits(path, (99.0, 29.0, 101.0, 31.0))

    assert total == 3
    assert hits == [0, 2], "只留视口内的点，顺序按记录号"
    assert sv.geometry_family(path) == "point"


def test_polyline_parts_and_points_are_preserved(tmp_path: Path):
    """一段路由多个 part 拼成时不能合并——那正是「线被打断」的另一种写法。"""
    path = write_shapefile(
        tmp_path / "线.shp",
        shape_type=SHP_POLYLINE,
        shapes=[
            [[[100.0, 30.0], [101.0, 31.0]], [[102.0, 30.0], [103.0, 31.0], [104.0, 30.5]]],
            [[[200.0, 60.0], [201.0, 61.0]]],
        ],
    )

    hits, _total = sv.view_hits(path, (99.0, 29.0, 105.0, 32.0))
    assert hits == [0], "跨视口的长线按 envelope 相交算命中；第二条在境外"

    geoms = sv.read_geometries(path, [0])
    paths = geoms[0]["paths"]
    assert [len(part) for part in paths] == [2, 3]
    assert paths[1][2] == [104.0, 30.5]


def test_polygon_rings_keep_outer_and_hole(tmp_path: Path):
    outer = [[100.0, 30.0], [104.0, 30.0], [104.0, 34.0], [100.0, 34.0], [100.0, 30.0]]
    hole = [[101.0, 31.0], [101.0, 32.0], [102.0, 32.0], [102.0, 31.0], [101.0, 31.0]]
    path = write_shapefile(tmp_path / "面.shp", shape_type=SHP_POLYGON, shapes=[[outer, hole]])

    geoms = sv.read_geometries(path, [0])

    rings = geoms[0]["rings"]
    assert len(rings) == 2, "洞是独立一环，不能被并进外环"
    assert rings[0] == outer and rings[1] == hole


def test_view_hits_uses_record_envelope_for_polygons(tmp_path: Path):
    """整块盖住视口的面也算命中（envelope 相交，不是包含关系）。"""
    outer = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]
    path = write_shapefile(tmp_path / "大面.shp", shape_type=SHP_POLYGON, shapes=[[outer]])

    hits, total = sv.view_hits(path, (4.0, 4.0, 5.0, 5.0))

    assert total == 1 and hits == [0]


def test_load_view_counts_in_view_but_returns_limited_rows(tmp_path: Path):
    coords = [[100.0 + index * 0.01, 30.0] for index in range(30)]
    path = _points(tmp_path, coords)

    view = sv.load_view(path, bbox=(99.0, 29.0, 101.0, 31.0), limit=10)

    assert view["total_in_view"] == 30, "视口内真实要素数不受 limit 影响"
    assert len(view["geoms"]) == 10
    # 等间隔挑：铺满视口，而不是只给最西边一小片。
    assert view["geoms"][0]["x"] == pytest.approx(100.0)
    assert view["geoms"][-1]["x"] > 100.2


def test_spread_returns_everything_under_limit():
    assert sv.spread([1, 2, 3], 5) == [1, 2, 3]
    assert sv.spread([1, 2, 3], 0) == [1, 2, 3]
    assert len(sv.spread(list(range(100)), 10)) == 10
    assert sv.spread(list(range(100)), 10)[0] == 0


def test_dbf_attributes_are_read_by_record_number(tmp_path: Path):
    """只取被选中的行（定长记录直接定位），中文按 GBK 老库的 ldid 解码。"""
    path = write_shapefile(
        tmp_path / "属性.shp",
        shape_type=SHP_POINT,
        shapes=[[100.0, 30.0], [100.1, 30.1], [100.2, 30.2]],
        fields=[("NAME", "C", 20, 0), ("LEN", "N", 10, 2)],
        rows=[("辛房路", 12.5), ("永定路", 3.0), ("garbage", None)],
        encoding="gbk",
        ldid=0x4D,
    )

    attrs = sv.read_attributes(path, [1, 0, 2], ["name", "LEN", "MISSING"])

    assert attrs is not None
    assert attrs[0] == {"NAME": "永定路", "LEN": 3.0}, "字段名大小写不敏感，缺列直接丢"
    assert attrs[1] == {"NAME": "辛房路", "LEN": 12.5}
    assert attrs[2] == {"NAME": "garbage", "LEN": None}


def test_attributes_without_dbf_return_none(tmp_path: Path):
    path = _points(tmp_path, [[100.0, 30.0]])
    assert sv.read_attributes(path, [0], ["NAME"]) is None


def test_broken_shapefile_raises_shapefile_error(tmp_path: Path):
    broken = tmp_path / "坏.shp"
    broken.write_bytes(b"not a shapefile at all")
    broken.with_suffix(".shx").write_bytes(b"\x00" * 100)

    with pytest.raises(sv.ShapefileError):
        sv.view_hits(broken, (0.0, 0.0, 1.0, 1.0))


def test_missing_shx_raises_shapefile_error(tmp_path: Path):
    path = _points(tmp_path, [[100.0, 30.0]])
    path.with_suffix(".shx").unlink()

    with pytest.raises(sv.ShapefileError):
        sv.geometry_family(path)

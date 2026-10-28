# -*- coding: utf-8 -*-
"""CSV/Excel → 点图层（P0-2 验收项）。

全部用真文件（临时目录里现造 CSV/XLSX），不联网、不需要 arcpy ——
验收标准就写在 ``tests/`` 里：列识别、跳过行统计、投影坐标不猜。
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from gis_cli.ingest import csv_points
from gis_cli.ingest.csv_points import (
    CoordinateColumnError,
    IngestError,
    ProjectedCoordinateError,
    build_arcpy_code,
    convert_csv_points,
    detect_coord_columns,
    parse_coord,
    to_feature_class,
)


# --------------------------------------------------------------- 造测试文件


def _write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_bytes(text.encode(encoding))
    return path


def _write_xlsx(path: Path, rows: list[list[object]], sheet_name: str = "Sheet1") -> Path:
    """写一个真 .xlsx（标准库 zipfile）：共享字符串 + 数字单元格。"""
    shared: list[str] = []

    def sid(text: str) -> int:
        if text not in shared:
            shared.append(text)
        return shared.index(text)

    body = []
    for r, row in enumerate(rows, start=1):
        cells = []
        for c, value in enumerate(row):
            ref = f"{chr(ord('A') + c)}{r}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="s"><v>{sid(str(value))}</v></c>')
        body.append(f'<row r="{r}">{"".join(cells)}</row>')
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{ns}" xmlns:r="{rel_ns}"><sheets>'
            f'<sheet name="{sheet_name}" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="{pkg_ns}">'
            f'<Relationship Id="rId1" Type="{rel_ns}/worksheet" Target="worksheets/sheet1.xml"/>'
            f"</Relationships>",
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            f'<sst xmlns="{ns}">' + "".join(f"<si><t>{s}</t></si>" for s in shared) + "</sst>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{ns}"><sheetData>{"".join(body)}</sheetData></worksheet>',
        )
    return path


# --------------------------------------------------------------- 列名识别


@pytest.mark.parametrize(
    "header,expect",
    [
        (["名称", "经度", "纬度"], ("经度", "纬度")),
        (["Name", "Longitude", "Latitude"], ("Longitude", "Latitude")),
        (["id", "lon", "lat"], ("lon", "lat")),
        (["id", "lng", "lat"], ("lng", "lat")),
        (["东经", "北纬"], ("东经", "北纬")),
        (["center_lon", "center_lat"], ("center_lon", "center_lat")),
        (["X", "Y"], ("X", "Y")),
    ],
)
def test_detect_coord_columns_recognises_common_aliases(header, expect):
    columns = detect_coord_columns(header)
    assert (columns.lon, columns.lat) == expect


def test_detect_coord_columns_prefers_longitude_over_x():
    """lon 比 x 可信：两个都在时用 lon（x/y 是投影坐标的重灾区）。"""
    columns = detect_coord_columns(["id", "x", "y", "lon", "lat"])
    assert (columns.lon, columns.lat) == ("lon", "lat")


def test_detect_coord_columns_does_not_treat_index_as_x():
    """单字母别名只认精确匹配，否则 index 会“以 x 结尾”被当成经度。"""
    with pytest.raises(CoordinateColumnError) as excinfo:
        detect_coord_columns(["index", "value"])
    assert "没找到经度列" in str(excinfo.value)
    assert "index/value" in str(excinfo.value)


def test_detect_coord_columns_reports_candidates_when_missing():
    with pytest.raises(CoordinateColumnError) as excinfo:
        detect_coord_columns(["城市", "人口", "年份"])
    message = str(excinfo.value)
    assert "候选列是：城市/人口/年份" in message
    assert "lon_field" in message, "报错要给可执行的出路，不能只说没找到"


def test_detect_coord_columns_rejects_same_column_for_both_axes():
    with pytest.raises(CoordinateColumnError) as excinfo:
        detect_coord_columns(["x"])
    assert "没找到纬度列" in str(excinfo.value)


# --------------------------------------------------------------- 坐标解析


@pytest.mark.parametrize(
    "value,expect",
    [
        ("116.4", 116.4),
        (" 116.4 ", 116.4),
        ("１１６．４", 116.4),  # 全角
        ("1,234.5", 1234.5),
        ("abc", None),
        ("", None),
        ("nan", None),
        ("—", None),
        (None, None),
        (float("inf"), None),
    ],
)
def test_parse_coord(value, expect):
    assert parse_coord(value) == expect


# ------------------------------------------------------- CSV → GeoJSON（验收）


def test_convert_csv_skips_three_bad_rows_and_writes_geojson(tmp_path):
    """验收项：只有“名称 + 经纬度”的 CSV（含 3 行坏数据）能直接出图并报“跳过 3 行”。"""
    source = _write(
        tmp_path / "sample.csv",
        "名称,经度,纬度\n"
        "甲,116.40,39.90\n"
        "乙,121.47,31.23\n"
        "丙,113.26,23.13\n"
        "坏1,abc,39.9\n"
        "坏2,,31.2\n"
        "坏3,113.2,\n"
        "丁,114.30,30.59\n",
    )
    result = convert_csv_points(source, out_geojson=tmp_path / "out" / "sample_points.geojson")

    assert result.feature_count == 4
    assert result.skipped == 3
    assert result.skipped_non_numeric == 3
    assert result.skipped_out_of_range == 0
    assert result.extent == [113.26, 23.13, 121.47, 39.9]

    summary = result.summary()
    assert "跳过 3 行" in summary
    assert "WGS84" in summary, "必须明说已按 WGS84 处理"
    assert "经度列=经度" in summary

    payload = json.loads(Path(result.geojson_path).read_text(encoding="utf-8"))
    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) == 4
    first = payload["features"][0]
    assert first["geometry"] == {"type": "Point", "coordinates": [116.4, 39.9]}
    assert first["properties"] == {"名称": "甲"}
    assert payload["crs"]["properties"]["name"].endswith("CRS84")

    record = result.to_record()
    assert record.kind == "vector"
    assert record.geom_type == "Point"
    assert record.wkid == 4326
    assert record.feature_count == 4
    assert record.extra["skipped_rows"] == 3


def test_convert_csv_with_bom_and_gbk_headers(tmp_path):
    """中文 CSV 从 Excel 另存常是 GBK + BOM —— 表头不能读成乱码。"""
    source = _write(
        tmp_path / "gbk.csv",
        "\ufeff名称,经度,纬度\r\n甲,116.40,39.90\r\n乙,121.47,31.23\r\n",
        encoding="gb18030",
    )
    result = convert_csv_points(source)
    assert result.feature_count == 2
    assert result.lon_field == "经度"


def test_convert_csv_reports_projected_coordinates_instead_of_guessing(tmp_path):
    """验收项（风险）：X/Y 是投影坐标时明确报“需要 EPSG 码”，不猜。"""
    source = _write(
        tmp_path / "projected.csv",
        "name,X,Y\n甲,500000.0,3400000.0\n乙,500100.0,3400100.0\n丙,500200.0,3400200.0\n",
    )
    with pytest.raises(ProjectedCoordinateError) as excinfo:
        convert_csv_points(source)
    message = str(excinfo.value)
    assert "超出经纬度范围" in message
    assert "EPSG" in message
    assert not (tmp_path / "projected_points.geojson").exists(), "报错时不能偷偷写出半份数据"


def test_convert_csv_skips_few_outliers_but_keeps_layer(tmp_path):
    """个别录错的坐标只跳过并 warn，不整批拒绝（否则一份表错一行就白干）。"""
    source = _write(
        tmp_path / "mixed.csv",
        "名称,经度,纬度\n甲,116.4,39.9\n乙,121.5,31.2\n丙,500.0,39.9\n丁,113.3,23.1\n",
    )
    result = convert_csv_points(source)
    assert result.feature_count == 3
    assert result.skipped_out_of_range == 1
    assert result.skipped == 1
    assert any("投影坐标" in w for w in result.warnings)


def test_convert_csv_explicit_fields_override_detection(tmp_path):
    source = _write(tmp_path / "explicit.csv", "名称,甲,乙\n丙,116.4,39.9\n")
    result = convert_csv_points(source, lon_field="甲", lat_field="乙")
    assert (result.lon_field, result.lat_field) == ("甲", "乙")
    assert result.feature_count == 1


def test_convert_csv_missing_explicit_field_lists_candidates(tmp_path):
    source = _write(tmp_path / "explicit.csv", "名称,甲,乙\n丙,116.4,39.9\n")
    with pytest.raises(CoordinateColumnError) as excinfo:
        convert_csv_points(source, lon_field="经度", lat_field="纬度")
    assert "候选列是：名称/甲/乙" in str(excinfo.value)


def test_convert_csv_without_coordinate_columns_fails_loudly(tmp_path):
    """验收项：没有经纬度列的 CSV 要报错清楚，不静默失败。"""
    source = _write(tmp_path / "no_coords.csv", "城市,人口\n北京,2189\n上海,2487\n")
    with pytest.raises(CoordinateColumnError) as excinfo:
        convert_csv_points(source)
    assert "没找到经度列" in str(excinfo.value)
    assert "城市/人口" in str(excinfo.value)


def test_convert_csv_all_rows_bad_reports_counts(tmp_path):
    source = _write(tmp_path / "bad.csv", "名称,经度,纬度\n甲,abc,def\n")
    with pytest.raises(IngestError) as excinfo:
        convert_csv_points(source)
    assert "没有任何可用坐标行" in str(excinfo.value)


# ------------------------------------------------------- Excel（验收项）


def test_convert_xlsx_with_dongjing_beijing_headers(tmp_path):
    """验收项：列名是“东经/北纬”的 Excel 能识别。"""
    source = _write_xlsx(
        tmp_path / "点位.xlsx",
        [["名称", "东经", "北纬"], ["甲", 116.4, 39.9], ["乙", 121.47, 31.23]],
    )
    result = convert_csv_points(source, out_geojson=tmp_path / "xlsx_points.geojson")
    assert (result.lon_field, result.lat_field) == ("东经", "北纬")
    assert result.feature_count == 2
    assert result.extent == [116.4, 31.23, 121.47, 39.9]


def test_convert_xlsx_skips_bad_rows(tmp_path):
    source = _write_xlsx(
        tmp_path / "bad.xlsx",
        [["名称", "lon", "lat"], ["甲", 116.4, 39.9], ["乙", "abc", 31.2]],
    )
    result = convert_csv_points(source)
    assert result.feature_count == 1
    assert result.skipped == 1


def test_convert_xls_asks_for_resave(tmp_path):
    source = tmp_path / "old.xls"
    source.write_bytes(b"\xd0\xcf\x11\xe0")
    with pytest.raises(IngestError) as excinfo:
        convert_csv_points(source)
    assert "另存为" in str(excinfo.value)


# ------------------------------------------------------------ 转要素类


def test_build_arcpy_code_is_valid_python_and_uses_create_file_gdb(tmp_path):
    code = build_arcpy_code(tmp_path / "a.geojson", tmp_path / "out" / "result.gdb" / "points")
    compile(code, "<arcpy>", "exec")  # 语法必须是对的
    assert "CreateFileGDB" in code, "不能 os.makedirs 造假 .gdb"
    assert "JSONToFeatures" in code
    assert "SpatialReference(4326)" in code
    assert json.dumps(str(tmp_path / "a.geojson")) in code


def test_arcpy_code_creates_gdb_and_defines_wgs84(tmp_path, monkeypatch):
    """用假 arcpy 真跑一遍生成的代码（CI 没有 ArcGIS，只能这样验证流程）。"""
    import sys
    import types

    calls: list[tuple] = []

    class _SR:
        def __init__(self, code):
            self.factoryCode = code

    arcpy = types.ModuleType("arcpy")
    arcpy.env = types.SimpleNamespace(overwriteOutput=False)
    arcpy.SpatialReference = _SR
    arcpy.Exists = lambda path: False
    arcpy.Describe = lambda path: types.SimpleNamespace(spatialReference=_SR(4490))
    arcpy.management = types.SimpleNamespace(
        CreateFileGDB=lambda folder, name: calls.append(("CreateFileGDB", folder, name)),
        Delete=lambda path: calls.append(("Delete", path)),
        DefineProjection=lambda path, sr: calls.append(("DefineProjection", path, sr.factoryCode)),
        GetCount=lambda path: [str(7)],
    )
    arcpy.conversion = types.SimpleNamespace(
        JSONToFeatures=lambda geojson, out, kind: calls.append(("JSONToFeatures", geojson, out, kind))
    )
    monkeypatch.setitem(sys.modules, "arcpy", arcpy)

    geojson = tmp_path / "a.geojson"
    out_fc = tmp_path / "out" / "result.gdb" / "points"
    result: dict = {}
    exec(build_arcpy_code(geojson, out_fc), {"set_result": lambda payload: result.update(payload)})

    assert ("JSONToFeatures", str(geojson), str(out_fc), "POINT") in calls
    assert ("CreateFileGDB", str(out_fc.parent.parent), "result") in calls
    assert ("DefineProjection", str(out_fc), 4326) in calls, "源数据是 WGS84，坐标系必须是 4326"
    assert result["feature_count"] == 7


class _FakeExecuted:
    def __init__(self, ok: bool, result=None, error=None, stderr: str = ""):
        self.ok = ok
        self.result = result
        self.error = error
        self.stderr = stderr


class _FakeRunner:
    def __init__(self, executed: _FakeExecuted):
        self.executed = executed
        self.calls: list[str] = []

    def execute(self, code: str, *, timeout: float = 600.0, workspace=None):
        self.calls.append(code)
        return self.executed


def test_to_feature_class_returns_kernel_result(tmp_path):
    runner = _FakeRunner(_FakeExecuted(True, {"output": "x.gdb/points", "feature_count": 4}))
    result = to_feature_class(runner, tmp_path / "a.geojson", tmp_path / "x.gdb" / "points")
    assert result["feature_count"] == 4
    assert len(runner.calls) == 1


def test_to_feature_class_explains_common_failures(tmp_path):
    runner = _FakeRunner(
        _FakeExecuted(False, error={"type": "ExecuteError", "message": "ERROR 000258: 输出为假库"})
    )
    with pytest.raises(IngestError) as excinfo:
        to_feature_class(runner, tmp_path / "a.geojson", tmp_path / "x.gdb" / "points")
    message = str(excinfo.value)
    assert "000258" in message
    assert "CreateFileGDB" in message


def test_numeric_columns_are_written_as_json_numbers(tmp_path):
    """数字列必须写 JSON 数字，不能一律 str()（DAIR-26 B2）。

    以前 properties 全写字符串，而 ``infer_field_type`` 把人口/面积判成 Double：
    catalog 登记 Double、``JSONToFeatures`` 建出来的字段却是 Text，自相矛盾。
    实测（真 arcpy）：``人口`` 在要素类里是 String。
    """
    source = _write(
        tmp_path / "num.csv",
        "名称,经度,纬度,人口,面积\n"
        "兰州,103.80,36.06,4380000,13085.5\n"
        "白银,104.14,36.55,1730000,21200.0\n",
    )
    result = convert_csv_points(source, out_geojson=tmp_path / "num.geojson")

    types = {f["name"]: f["type"] for f in result.fields}
    assert types["人口"] == "Double" and types["面积"] == "Double"

    payload = json.loads(Path(result.geojson_path).read_text(encoding="utf-8"))
    props = payload["features"][0]["properties"]
    assert props["人口"] == 4380000.0 and isinstance(props["人口"], float)
    assert props["面积"] == 13085.5
    assert props["名称"] == "兰州", "文本列保持字符串"
    # samples 仍保留原始字符串（catalog 里给人看的样本）
    samples = {f["name"]: f["samples"] for f in result.fields}["人口"]
    assert samples[0] == "4380000" and all(isinstance(v, str) for v in samples)


def test_zero_padded_codes_stay_text(tmp_path):
    """``007`` 这种 0 开头纯数字串是编码，不能转成 7（否则有损）。"""
    source = _write(
        tmp_path / "code.csv",
        "名称,经度,纬度,行政区代码\n"
        "甲,116.40,39.90,007\n"
        "乙,121.47,31.23,012\n",
    )
    result = convert_csv_points(source, out_geojson=tmp_path / "code.geojson")

    assert {f["name"]: f["type"] for f in result.fields}["行政区代码"] == "String"
    payload = json.loads(Path(result.geojson_path).read_text(encoding="utf-8"))
    assert payload["features"][0]["properties"]["行政区代码"] == "007"


def test_empty_cell_in_numeric_column_becomes_null(tmp_path):
    """Double 列里的空值写 null —— 塞空字符串会让 JSONToFeatures 的类型推断崩掉。"""
    source = _write(
        tmp_path / "hole.csv",
        "名称,经度,纬度,人口\n"
        "甲,116.40,39.90,100\n"
        "乙,121.47,31.23,\n",
    )
    result = convert_csv_points(source, out_geojson=tmp_path / "hole.geojson")
    payload = json.loads(Path(result.geojson_path).read_text(encoding="utf-8"))
    assert payload["features"][1]["properties"]["人口"] is None

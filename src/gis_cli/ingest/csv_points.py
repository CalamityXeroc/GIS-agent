# -*- coding: utf-8 -*-
"""CSV / Excel（带经纬度）→ 点图层（P0-2）。

外行的数据长这样：``名称,经度,纬度`` 或 ``Name,Longitude,Latitude``，没有坐标系、
没有字段说明，还夹着几行坏数据。这里只做一件事：把它变成一份坐标能对上地图的
点数据，并把"丢了什么"说清楚。

规则（照 P0 计划 §2.3，**不猜**）：

1. 列名按别名表识别（``lon/lng/long/longitude/x/东经/经度`` 等），识别不到就
   报"没找到经度列，候选列是：A/B/C"，不拿别的列凑数
2. 坐标一律按 **WGS84（EPSG:4326）** 处理，报告里明说
3. 中间产物先写 GeoJSON（没有 arcpy 也能做、也能直接看），再转要素类
4. 坐标不是数字的行 → 跳过并计数；数值超出经纬度范围 → 跳过并单独计数，
   全都不在范围内时直接报"可能是投影坐标，需要 EPSG 码"
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree

#: 经度列别名（按可信度排序：``x`` 垫底 —— 投影坐标冒充经纬度的重灾区）。
LON_ALIASES: tuple[str, ...] = ("lon", "lng", "long", "longitude", "东经", "经度", "x")
#: 纬度列别名（同上）。
LAT_ALIASES: tuple[str, ...] = ("lat", "latitude", "北纬", "纬度", "y")

#: 输出里固定写明的话（验收要求"明说已按 WGS84 处理"）。
CRS_NOTE = "已按 WGS84（EPSG:4326）处理"

_EMPTY_TOKENS = {"", "nan", "none", "null", "na", "n/a", "-", "--", "—", "/"}
_SEP_RE = re.compile(r"[\s_\-./\\()（）\[\]【】:：]+")


class IngestError(ValueError):
    """接入失败（消息直接给用户看，要能照着改）。"""


class CoordinateColumnError(IngestError):
    """识别不到经纬度列（或识别结果自相矛盾）。"""


class ProjectedCoordinateError(IngestError):
    """数值看着像投影坐标（X/Y），不猜坐标系。"""


# --------------------------------------------------------------------- 表头


@dataclass(frozen=True)
class CoordColumns:
    """识别出的经纬度列。"""

    lon: str
    lat: str
    lon_index: int
    lat_index: int
    lon_alias: str = ""
    lat_alias: str = ""

    def describe(self) -> str:
        return f"经度列={self.lon}（别名 {self.lon_alias}），纬度列={self.lat}（别名 {self.lat_alias}）"


def _norm(text: Any) -> str:
    """去 BOM、全角转半角、去首尾空白、转小写。"""
    return unicodedata.normalize("NFKC", str(text or "")).lstrip("\ufeff").strip().lower()


def _compact(text: Any) -> str:
    return _SEP_RE.sub("", _norm(text))


def _match_rank(name: str, aliases: Sequence[str]) -> tuple[int, int] | None:
    """返回 (匹配类型, 别名序号)，越小越可信；None 表示不是这一轴。

    精确匹配 > 前后缀匹配；单字母别名（x/y）只认精确匹配 ——
    否则 ``index`` 会因为"以 x 结尾"被当成经度列（实测过的坑）。
    """
    norm = _norm(name)
    for idx, alias in enumerate(aliases):
        if norm == alias:
            return (0, idx)
    compact = _compact(name)
    if not compact:
        return None
    for idx, alias in enumerate(aliases):
        if len(alias) < 3:
            continue
        if compact.startswith(alias) or compact.endswith(alias):
            return (1, idx)
    return None


def _candidates(header: Sequence[str], limit: int = 10) -> str:
    names = [str(h) for h in header[:limit] if str(h).strip()]
    more = len([h for h in header if str(h).strip()]) - len(names)
    text = "/".join(names) if names else "（空表头）"
    return text + (f" 等 {len(names) + more} 列" if more > 0 else "")


def _pick(header: Sequence[str], aliases: Sequence[str], axis: str) -> tuple[int, str, str]:
    best: tuple[int, int] | None = None
    chosen: tuple[int, str, str] | None = None
    for index, name in enumerate(header):
        rank = _match_rank(name, aliases)
        if rank is None:
            continue
        if best is None or rank < best:
            best = rank
            chosen = (index, str(name), aliases[rank[1]])
    if chosen is None:
        raise CoordinateColumnError(
            f"没找到{axis}列（可用别名：{'/'.join(aliases)}）。"
            f"候选列是：{_candidates(header)}。"
            f"确实有这一列但不叫这些名字时，显式指定 lon_field/lat_field。"
        )
    return chosen


def detect_coord_columns(header: Sequence[str]) -> CoordColumns:
    """从表头识别经纬度列；识别不到就抛 ``CoordinateColumnError``。"""
    if not header:
        raise CoordinateColumnError("表是空的：连表头都没有，无法识别经纬度列。")
    lon_index, lon_name, lon_alias = _pick(header, LON_ALIASES, "经度")
    lat_index, lat_name, lat_alias = _pick(header, LAT_ALIASES, "纬度")
    if lon_index == lat_index:
        raise CoordinateColumnError(
            f"经纬度识别到同一列：{lon_name}。请显式指定 lon_field/lat_field。"
        )
    return CoordColumns(
        lon=lon_name,
        lat=lat_name,
        lon_index=lon_index,
        lat_index=lat_index,
        lon_alias=lon_alias,
        lat_alias=lat_alias,
    )


# --------------------------------------------------------------------- 读表


@dataclass
class Table:
    header: list[str]
    rows: list[list[str]]


def _decode_text(raw: bytes) -> str:
    """按 UTF-8-SIG → GB18030 → UTF-8(replace) 解。

    中文 CSV 从 Excel 另存出来经常是 GBK；只按 UTF-8 读会把表头读成乱码，
    于是"经度"识别不到 —— 用户看到的就是"没找到经度列"，很难自己想明白。
    """
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _sniff_delimiter(first_line: str) -> str:
    if ";" in first_line and "," not in first_line:
        return ";"
    if "\t" in first_line and "," not in first_line:
        return "\t"
    return ","


def _read_csv(path: Path) -> Table:
    text = _decode_text(path.read_bytes())
    first_line = text.splitlines()[0] if text.splitlines() else ""
    # 用 StringIO 而不是 splitlines：带换行的引号字段不能按行拆（会把一行拆成两行）
    reader = csv.reader(io.StringIO(text), delimiter=_sniff_delimiter(first_line))
    raw_rows = [row for row in reader if any(str(cell).strip() for cell in row)]
    if not raw_rows:
        raise IngestError(f"表是空的：{path}")
    header = [str(cell).lstrip("\ufeff").strip() for cell in raw_rows[0]]
    return Table(header=header, rows=[[str(c) for c in row] for row in raw_rows[1:]])


_XLSX_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_XLSX_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _col_index(ref: str) -> int:
    """``B12`` → 1（0 基列号）。"""
    letters = "".join(ch for ch in ref if ch.isalpha()).upper()
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch) - ord("A") + 1)
    return max(index - 1, 0)


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except (KeyError, ElementTree.ParseError):
        return []
    out: list[str] = []
    for si in root.findall(f"{_XLSX_MAIN}si"):
        out.append("".join(node.text or "" for node in si.iter(f"{_XLSX_MAIN}t")))
    return out


def _sheet_target(archive: zipfile.ZipFile, sheet: str | int | None) -> str:
    try:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    except (KeyError, ElementTree.ParseError) as exc:
        raise IngestError(f"不是有效的 .xlsx（缺少 workbook 部件）：{exc}") from exc
    sheets = workbook.findall(f".//{_XLSX_MAIN}sheet")
    if not sheets:
        raise IngestError("这个 .xlsx 里一个工作表都没有。")
    chosen = sheets[0]
    if isinstance(sheet, int):
        if not 1 <= sheet <= len(sheets):
            raise IngestError(f"工作表序号 {sheet} 超出范围（共 {len(sheets)} 个）。")
        chosen = sheets[sheet - 1]
    elif isinstance(sheet, str) and sheet.strip():
        matched = [s for s in sheets if (s.get("name") or "") == sheet.strip()]
        if not matched:
            names = "/".join(s.get("name") or "" for s in sheets)
            raise IngestError(f"没找到工作表 {sheet}，现有：{names}")
        chosen = matched[0]
    rel_id = chosen.get(f"{_XLSX_REL}id") or ""
    target = ""
    for rel in rels.findall(f"{_PKG_REL}Relationship"):
        if rel.get("Id") == rel_id:
            target = rel.get("Target") or ""
            break
    if not target:
        raise IngestError(f"工作表 {chosen.get('name')} 指向的部件找不到。")
    target = target.lstrip("/")
    return target if target.startswith("xl/") else f"xl/{target}"


def _read_xlsx(path: Path, sheet: str | int | None = None) -> Table:
    """标准库读 .xlsx（不引 openpyxl / pandas）。"""
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise IngestError(f"打不开 .xlsx（{exc}）。老式 .xls 请先另存为 .xlsx 或 .csv。") from exc
    with archive:
        shared = _shared_strings(archive)
        target = _sheet_target(archive, sheet)
        try:
            root = ElementTree.fromstring(archive.read(target))
        except KeyError as exc:
            raise IngestError(f"工作表部件缺失：{target}") from exc
    raw_rows: list[list[str]] = []
    for row in root.iter(f"{_XLSX_MAIN}row"):
        values: dict[int, str] = {}
        for cell in row.iter(f"{_XLSX_MAIN}c"):
            index = _col_index(cell.get("r") or "")
            ctype = cell.get("t") or ""
            if ctype == "inlineStr":
                node = cell.find(f"{_XLSX_MAIN}is")
                text = "".join(t.text or "" for t in node.iter(f"{_XLSX_MAIN}t")) if node is not None else ""
            else:
                node = cell.find(f"{_XLSX_MAIN}v")
                raw = node.text if node is not None and node.text is not None else ""
                if ctype == "s":
                    try:
                        text = shared[int(raw)]
                    except (ValueError, IndexError):
                        text = ""
                else:
                    text = raw
            values[index] = text
        if values:
            width = max(values) + 1
            raw_rows.append([values.get(i, "") for i in range(width)])
    raw_rows = [row for row in raw_rows if any(str(cell).strip() for cell in row)]
    if not raw_rows:
        raise IngestError(f"工作表是空的：{path}")
    header = [str(cell).lstrip("\ufeff").strip() for cell in raw_rows[0]]
    return Table(header=header, rows=[[str(c) for c in row] for row in raw_rows[1:]])


def read_table(path: str | Path, *, sheet: str | int | None = None) -> Table:
    """读 CSV / XLSX 成 (表头, 行)。"""
    file_path = Path(path)
    if not file_path.is_file():
        raise IngestError(f"文件不存在：{file_path}")
    suffix = file_path.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        return _read_csv(file_path)
    if suffix == ".xlsx":
        return _read_xlsx(file_path, sheet)
    if suffix == ".xls":
        raise IngestError("不支持老式 .xls：请用 Excel 另存为 .xlsx 或 .csv 后再接入。")
    raise IngestError(f"不支持的表格式：{suffix or '(无后缀)'}（只认 .csv / .xlsx）")


# ------------------------------------------------------------------ 坐标解析


def parse_coord(value: Any) -> float | None:
    """解析一个坐标值；空值/非数字返回 None（不抛异常）。"""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    text = _norm(value)
    if text in _EMPTY_TOKENS:
        return None
    # 只认标准千分位（1,234.5）；“41,15”这种欧式小数逗号不猜，按非数字跳过
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", text):
        text = text.replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _in_range(lon: float, lat: float) -> bool:
    return -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0


# --------------------------------------------------------------------- 结果


@dataclass
class CsvPointsResult:
    """一次 CSV/Excel → 点图层的接入结果（报告就是 ``summary()``）。"""

    source_path: str
    geojson_path: str
    lon_field: str
    lat_field: str
    feature_count: int = 0
    total_rows: int = 0
    skipped_non_numeric: int = 0
    skipped_out_of_range: int = 0
    warnings: list[str] = field(default_factory=list)
    fields: list[dict[str, Any]] = field(default_factory=list)
    extent: list[float] | None = None
    crs_note: str = CRS_NOTE

    @property
    def skipped(self) -> int:
        return self.skipped_non_numeric + self.skipped_out_of_range

    def summary(self) -> str:
        bits = [
            f"{self.crs_note}：{self.source_path} → {self.geojson_path}",
            f"{self.feature_count} 个点（经度列={self.lon_field}，纬度列={self.lat_field}，共 {self.total_rows} 行）",
        ]
        if self.skipped:
            detail = []
            if self.skipped_non_numeric:
                detail.append(f"坐标不是数字 {self.skipped_non_numeric} 行")
            if self.skipped_out_of_range:
                detail.append(f"超出经纬度范围 {self.skipped_out_of_range} 行")
            bits.append(f"跳过 {self.skipped} 行（{'、'.join(detail)}）")
        else:
            bits.append("没有跳过任何行")
        text = "；".join(bits) + "。"
        for warning in self.warnings:
            text += f"\n注意：{warning}"
        return text

    def to_record(self):
        """转成 catalog 记录（GeoJSON 本身也是可直读的矢量数据）。"""
        from ..datacatalog.store import DatasetRecord

        path = Path(self.geojson_path)
        stat = path.stat() if path.exists() else None
        return DatasetRecord(
            path=str(path),
            kind="vector",
            container="",
            name=path.name,
            mtime=stat.st_mtime if stat else 0.0,
            size=stat.st_size if stat else 0,
            crs_name="WGS 84",
            wkid=4326,
            geom_type="Point",
            feature_count=self.feature_count,
            extent=self.extent,
            fields=self.fields,
            raster={},
            extra={
                "format": "geojson",
                "ingested_from": self.source_path,
                "lon_field": self.lon_field,
                "lat_field": self.lat_field,
                "skipped_rows": self.skipped,
            },
        )


#: ``007`` / ``012`` 这种 0 开头的纯数字串：是编码（行政区代码、邮编…），不是数值。
_CODE_LIKE = re.compile(r"^0\d+$")


def numeric_value(value: Any) -> float | None:
    """按“数值列”解析一个单元格：0 开头的纯数字串当编码，不转数字。

    为什么单独一个函数：``infer_field_type`` 和写 GeoJSON properties 必须用同一套
    判断，否则又会出现“catalog 说 Double、要素类落 Text”。另外 ``007`` 直接转
    float 会变成 7——那是有损的，编码列不能这么干。
    """
    if _CODE_LIKE.fullmatch(_norm(value)):
        return None
    return parse_coord(value)


def infer_field_type(values: Iterable[str]) -> str:
    """根据样本值猜字段类型（只区分 Double / String）。"""
    seen = [v for v in values if str(v).strip()]
    if not seen:
        return "String"
    if all(numeric_value(v) is not None for v in seen):
        return "Double"
    return "String"


def _properties(
    row: Sequence[str],
    header: Sequence[str],
    skip: set[int],
    types: dict[str, str] | None = None,
) -> dict[str, Any]:
    """一行 CSV → GeoJSON properties。

    ``types`` 是每列**推断出来的**类型（``infer_field_type``）：Double 列必须写
    JSON 数字，不能一律 ``str()``。以前一律写字符串，而 catalog 里登记的是
    Double、``JSONToFeatures`` 建出来的字段也是 Text——声明与实际自相矛盾
    （实测：``人口`` 在要素类里是 String）。
    """
    out: dict[str, Any] = {}
    types = types or {}
    for index, name in enumerate(header):
        if index in skip:
            continue
        key = str(name).strip() or f"col{index + 1}"
        raw = row[index] if index < len(row) else ""
        if types.get(key) == "Double":
            # 该行不是数字（空/脏值）就写 null：Double 列里塞字符串会让
            # JSONToFeatures 的类型推断崩掉。
            out[key] = numeric_value(raw)
        else:
            out[key] = str(raw)
    return out


def convert_csv_points(
    path: str | Path,
    *,
    out_geojson: str | Path | None = None,
    lon_field: str | None = None,
    lat_field: str | None = None,
    sheet: str | int | None = None,
) -> CsvPointsResult:
    """CSV/Excel → GeoJSON 点图层，返回接入报告。"""
    source = Path(path)
    table = read_table(source, sheet=sheet)
    header = table.header

    if lon_field or lat_field:
        if not (lon_field and lat_field):
            raise CoordinateColumnError("lon_field 与 lat_field 必须同时给（只给一个没法定位点）。")
        lookup = {str(h).strip().lower(): i for i, h in enumerate(header)}
        missing = [f for f in (lon_field, lat_field) if f.strip().lower() not in lookup]
        if missing:
            raise CoordinateColumnError(
                f"指定的列不存在：{'/'.join(missing)}。候选列是：{_candidates(header)}"
            )
        lon_index = lookup[lon_field.strip().lower()]
        lat_index = lookup[lat_field.strip().lower()]
        columns = CoordColumns(
            lon=str(header[lon_index]),
            lat=str(header[lat_index]),
            lon_index=lon_index,
            lat_index=lat_index,
            lon_alias="显式指定",
            lat_alias="显式指定",
        )
    else:
        columns = detect_coord_columns(header)

    # 列类型只算一次：既给 properties 决定“写数字还是写字符串”，也给下面的
    # fields 复用——两边必须用同一份判断，否则又会回到“catalog 说 Double、
    # 要素类落 Text”的自相矛盾。
    column_types: dict[str, str] = {}
    column_samples: dict[str, list[str]] = {}
    for index, column in enumerate(header):
        if index in {columns.lon_index, columns.lat_index}:
            continue
        column_name = str(column).strip() or f"col{index + 1}"
        samples = [
            row[index]
            for row in table.rows[:50]
            if index < len(row) and str(row[index]).strip()
        ]
        column_samples[column_name] = samples
        column_types[column_name] = infer_field_type(samples)

    features: list[dict[str, Any]] = []
    extent = [math.inf, math.inf, -math.inf, -math.inf]
    non_numeric = 0
    out_of_range = 0
    numeric = 0
    for row in table.rows:
        lon_cell = row[columns.lon_index] if columns.lon_index < len(row) else ""
        lat_cell = row[columns.lat_index] if columns.lat_index < len(row) else ""
        lon = parse_coord(lon_cell)
        lat = parse_coord(lat_cell)
        if lon is None or lat is None:
            non_numeric += 1
            continue
        numeric += 1
        if not _in_range(lon, lat):
            out_of_range += 1
            continue
        lon, lat = round(lon, 9), round(lat, 9)
        extent[0] = min(extent[0], lon)
        extent[1] = min(extent[1], lat)
        extent[2] = max(extent[2], lon)
        extent[3] = max(extent[3], lat)
        properties = _properties(
            row, header, {columns.lon_index, columns.lat_index}, column_types
        )
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": properties,
            }
        )

    if numeric and out_of_range and out_of_range * 2 >= numeric:
        raise ProjectedCoordinateError(
            f"检测到 {out_of_range}/{numeric} 行的数值超出经纬度范围"
            f"（经度应在 -180~180、纬度应在 -90~90），可能是投影坐标（如 CGCS2000 / 3 度带），"
            f"需要 EPSG 码才能转换；本工具不猜坐标系。"
            f"如果列选错了，可显式指定 lon_field/lat_field（当前：{columns.describe()}）。"
        )
    if not features:
        raise IngestError(
            f"没有任何可用坐标行（共 {len(table.rows)} 行：坐标不是数字 {non_numeric} 行、"
            f"超出经纬度范围 {out_of_range} 行）。{columns.describe()}。"
        )

    warnings: list[str] = []
    if out_of_range:
        warnings.append(
            f"{out_of_range} 行坐标超出经纬度范围已跳过，可能是投影坐标或录错，建议核对原表。"
        )

    target = Path(out_geojson) if out_geojson else source.with_name(f"{source.stem}_points.geojson")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "name": target.stem,
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    fields: list[dict[str, Any]] = []
    for index, column in enumerate(header):
        if index in {columns.lon_index, columns.lat_index}:
            continue
        column_name = str(column).strip() or f"col{index + 1}"
        samples = column_samples.get(column_name, [])
        fields.append(
            {
                "name": column_name,
                "alias": "",
                "type": column_types.get(column_name) or infer_field_type(samples),
                "length": 0,
                "nullable": True,
                "samples": samples[:3],
            }
        )

    return CsvPointsResult(
        source_path=str(source),
        geojson_path=str(target),
        lon_field=columns.lon,
        lat_field=columns.lat,
        feature_count=len(features),
        total_rows=len(table.rows),
        skipped_non_numeric=non_numeric,
        skipped_out_of_range=out_of_range,
        warnings=warnings,
        fields=fields,
        extent=[extent[0], extent[1], extent[2], extent[3]],
    )


# ------------------------------------------------------------------- 落 GDB

#: 转要素类的代码模板。走 ``arcpy.conversion.JSONToFeatures`` 而不是
#: ``XYTableToPoint``：GeoJSON 已经把"坏行跳过、坐标是 WGS84"落定了，
#: 再回读原始 CSV 等于把 UTF-8 BOM / GBK 那套坑重踩一遍（XYTableToPoint
#: 遇到带 BOM 的 CSV 直接报 000840 找不到字段，配方里已经踩过）。
_ARCPY_TEMPLATE = '''\
import os
import arcpy

arcpy.env.overwriteOutput = True
geojson = {geojson}
out_fc = {out_fc}

parent = os.path.dirname(out_fc)
if parent.lower().endswith(".gdb"):
    gdb, folder = parent, os.path.dirname(parent)
else:
    folder = parent
    gdb = os.path.join(folder, "result_data.gdb")
if folder and not os.path.isdir(folder):
    os.makedirs(folder, exist_ok=True)
# 注意：不能用 os.makedirs 造 .gdb，必须是 CreateFileGDB（否则是个假库）
if not arcpy.Exists(gdb):
    arcpy.management.CreateFileGDB(folder, os.path.basename(gdb)[:-4])
if arcpy.Exists(out_fc):
    arcpy.management.Delete(out_fc)

arcpy.conversion.JSONToFeatures(geojson, out_fc, "POINT")
sr = arcpy.Describe(out_fc).spatialReference
if not sr or sr.factoryCode != 4326:
    arcpy.management.DefineProjection(out_fc, arcpy.SpatialReference(4326))
count = int(arcpy.management.GetCount(out_fc)[0])
set_result({{"output": out_fc, "feature_count": count, "spatial_reference": "WGS 84 (EPSG:4326)"}})
print("CSV points -> feature class: %d" % count)
'''


def build_arcpy_code(geojson_path: str | Path, out_feature_class: str | Path) -> str:
    """生成"GeoJSON → 要素类"的 ArcPy 代码（无 arcpy 也能生成/校验）。"""
    return _ARCPY_TEMPLATE.format(
        geojson=json.dumps(str(geojson_path), ensure_ascii=False),
        out_fc=json.dumps(str(out_feature_class), ensure_ascii=False),
    )


def to_feature_class(
    runner: Any,
    result: CsvPointsResult | str | Path,
    out_feature_class: str | Path,
    *,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """用 ``CodeRunner``（持久内核）把 GeoJSON 落成要素类。

    失败时把内核报错原文带出来 —— 最常见的两种是"输出 GDB 是假库"
    和"同名要素类被占用"，照着报错就能改。
    """
    geojson = result.geojson_path if isinstance(result, CsvPointsResult) else str(result)
    code = build_arcpy_code(geojson, out_feature_class)
    executed = runner.execute(code, timeout=timeout)
    if not executed.ok:
        error = executed.error or {}
        detail = ": ".join(str(p) for p in (error.get("type"), error.get("message")) if p).strip()
        if not detail:
            lines = (executed.stderr or "").strip().splitlines()
            detail = lines[-1] if lines else "未知错误"
        raise IngestError(
            f"转要素类失败：{detail}。常见原因：输出 .gdb 不是真的文件地理数据库"
            f"（要用 CreateFileGDB，不能直接建目录）、同名要素类被占用、或 GeoJSON 不是点要素。"
        )
    return executed.result if isinstance(executed.result, dict) else {"raw": executed.result}

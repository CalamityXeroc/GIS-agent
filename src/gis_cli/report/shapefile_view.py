# -*- coding: utf-8 -*-
"""视口裁剪用的 shapefile 直读（DAIR-386）。

**为什么不用内核**（都是本机真数据实测，``workspace/input/beijing/北京其他道路``
134,819 条折线，街区级视口内 26,673 条）：

| 口径 | 耗时 |
| --- | --- |
| ``arcpy.da.SearchCursor(path, ["SHAPE@"])`` 全表 + extent 相交 | 3,614 ms |
| 同一内核按 ``FID IN (3000 个)`` 白名单读 | 729 ms |
| GDAL ``SetSpatialFilterRect`` + ``GetFeatureCount`` | 1,719 ms |
| 本模块：``.shx`` 偏移 + 记录头 bbox 扫描（**精确计数**） | 55 ms |
| 本模块：读 + 解析命中的 3,000 条记录 | 8 ms |

shapefile 的每条记录在几何之前就有一段 32 字节 envelope（点是 X/Y 两个
double），所以视口筛选和计数只需要读记录头，不用把几何反序列化出来。预览要求
「出图 ≤1.5s」，只有这条路走得过。

返回的几何是 **Esri JSON 形状**（``{"x","y"}`` / ``{"paths"}`` / ``{"rings"}`` /
``{"points"}``），交给 :func:`gis_cli.report.html_pack.esri_geometry_to_geojson`
转换：环方向（外环顺时针、洞逆时针）、多部件拆合、坐标净化都复用同一份实现，
预览和报告不会各写一套。
"""

from __future__ import annotations

import mmap
import struct
from pathlib import Path
from typing import Any, Iterator, Sequence

#: shapeType → 几何族。Z/M 变体只少读尾部数组，XY 布局与基础类型一致。
_GEOM_FAMILY = {
    1: "point",
    3: "polyline",
    5: "polygon",
    8: "multipoint",
    11: "point",
    13: "polyline",
    15: "polygon",
    18: "multipoint",
    21: "point",
    23: "polyline",
    25: "polygon",
    28: "multipoint",
}

#: shapeType 0 = Null Shape（属性行存在但没有几何），直接跳过。
_NULL_SHAPE = 0


class ShapefileError(ValueError):
    """不是能直读的 shapefile（损坏、被加密、或几何类型不支持）。"""


def geometry_family(path: str | Path) -> str:
    """图层主几何类型 → ``point`` / ``polyline`` / ``polygon`` / ``multipoint``。"""
    with _open_pair(path) as (shp, _shx):
        return _family_of(shp)


def _family_of(shp: mmap.mmap) -> str:
    if len(shp) < 100:
        raise ShapefileError("文件头不足 100 字节，不是 shapefile")
    code = struct.unpack_from("<i", shp, 32)[0]
    family = _GEOM_FAMILY.get(code)
    if family is None:
        raise ShapefileError(f"不支持的 shapeType={code}")
    return family


class _open_pair:
    """同时打开 ``.shp``/``.shx`` 的只读 mmap（一起关，别让文件句柄漏掉）。"""

    def __init__(self, path: str | Path) -> None:
        shp = Path(path)
        self._shp_path = shp
        self._shx_path = shp.with_suffix(".shx")
        self.shp: mmap.mmap | None = None
        self.shx: mmap.mmap | None = None

    def __enter__(self) -> tuple[mmap.mmap, mmap.mmap]:
        try:
            self.shp = _map(self._shp_path)
            self.shx = _map(self._shx_path)
        except OSError as exc:
            self.close()
            raise ShapefileError(f"读不出 shapefile：{exc}") from exc
        if len(self.shx) < 100:
            self.close()
            raise ShapefileError(".shx 索引文件不完整")
        return self.shp, self.shx

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False

    def close(self) -> None:
        for mapped in (self.shp, self.shx):
            if mapped is not None:
                mapped.close()
        self.shp = None
        self.shx = None


def _map(path: Path) -> mmap.mmap:
    handle = open(path, "rb")
    try:
        if path.stat().st_size == 0:
            raise ShapefileError(f"{path.name} 是空文件")
        return mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    finally:
        # mmap 已持有自己的引用，句柄可以立刻关（Windows 上也不会锁死文件）。
        handle.close()


def record_count(shp: mmap.mmap, shx: mmap.mmap) -> int:
    return max(0, (len(shx) - 100) // 8)


def _record_offset(shx: mmap.mmap, index: int) -> int:
    """``.shx`` 里的第 index 条偏移量（单位是 2 字节，shapefile 的规矩）。"""
    return struct.unpack_from(">i", shx, 100 + index * 8)[0] * 2


def _extent_of(shp: mmap.mmap, offset: int, family: str) -> tuple[float, float, float, float] | None:
    """记录头里的 envelope；点没有 envelope，用 X/Y 当退化矩形。"""
    content = offset + 8
    code = struct.unpack_from("<i", shp, content)[0]
    if code == _NULL_SHAPE:
        return None
    if family == "point":
        x, y = struct.unpack_from("<2d", shp, content + 4)
        return (x, y, x, y)
    return struct.unpack_from("<4d", shp, content + 4)


def view_hits(path: str | Path, bbox: Sequence[float]) -> tuple[list[int], int]:
    """记录 envelope 与 ``bbox`` 相交的 0 基记录序号（= shapefile 的 FID）。

    返回 ``(命中序号, 图层总记录数)``。相交判定用 envelope 是**保守**的：命中的
    要素可能整体落在视口外一点（大头针/长线跨视口），但绝不会漏掉视口内的要素。
    """
    minx, miny, maxx, maxy = (float(v) for v in bbox)
    with _open_pair(path) as (shp, shx):
        family = _family_of(shp)
        total = record_count(shp, shx)
        # 一条记录至少要有：8 字节记录头 + shapeType 4 + envelope（点是 X/Y 16）。
        need = 28 if family == "point" else 44
        hits: list[int] = []
        for index in range(total):
            offset = _record_offset(shx, index)
            if offset + need > len(shp):
                break  # 记录被截断：能读多少算多少，别抛
            extent = _extent_of(shp, offset, family)
            if extent is None:
                continue
            if (
                extent[0] > maxx
                or extent[1] > maxy
                or extent[2] < minx
                or extent[3] < miny
            ):
                continue
            hits.append(index)
        return hits, total


def spread(ids: list[int], limit: int) -> list[int]:
    """从命中里等间隔挑 ``limit`` 个，尽量铺满视口而不是只给左上角一片。"""
    if limit <= 0 or len(ids) <= limit:
        return list(ids)
    step = len(ids) / float(limit)
    return [ids[min(len(ids) - 1, int(index * step))] for index in range(limit)]


def _points(shp: mmap.mmap, base: int, count: int) -> Iterator[list[float]]:
    for index in range(count):
        x, y = struct.unpack_from("<2d", shp, base + index * 16)
        yield [x, y]


def read_geometry(shp: mmap.mmap, offset: int, family: str) -> dict[str, Any] | None:
    """一条记录 → Esri JSON 形状（解析不了返回 ``None``，不抛）。"""
    content = offset + 8
    try:
        code = struct.unpack_from("<i", shp, content)[0]
        if code == _NULL_SHAPE:
            return None
        if family == "point":
            x, y = struct.unpack_from("<2d", shp, content + 4)
            return {"x": x, "y": y}
        if family == "multipoint":
            (count,) = struct.unpack_from("<i", shp, content + 36)
            return {"points": list(_points(shp, content + 40, _safe_count(shp, count, content + 40)))}
        num_parts, num_points = struct.unpack_from("<2i", shp, content + 36)
        parts_start = content + 44
        points_start = parts_start + 4 * num_parts
        if num_parts < 0 or num_points < 0:
            return None
        num_points = _safe_count(shp, num_points, points_start)
        if num_parts == 0:
            return None
        index = list(struct.unpack_from("<%di" % num_parts, shp, parts_start))
        index.append(num_points)
        parts: list[list[list[float]]] = []
        for part in range(num_parts):
            start = max(0, min(num_points, index[part]))
            end = max(start, min(num_points, index[part + 1]))
            if end - start < 2:
                continue
            parts.append(list(_points(shp, points_start + start * 16, end - start)))
    except (struct.error, ValueError):
        return None
    if not parts:
        return None
    return {"paths": parts} if family == "polyline" else {"rings": parts}


def _safe_count(shp: mmap.mmap, count: int, start: int) -> int:
    """别信文件里的 numPoints：按剩余字节数夹一下，坏文件不该被读爆。"""
    if count <= 0:
        return 0
    room = max(0, (len(shp) - start) // 16)
    return min(count, room)


def read_geometries(path: str | Path, ids: Sequence[int]) -> list[dict[str, Any]]:
    with _open_pair(path) as (shp, shx):
        family = _family_of(shp)
        total = record_count(shp, shx)
        out: list[dict[str, Any]] = []
        for index in ids:
            if not 0 <= index < total:
                continue
            geom = read_geometry(shp, _record_offset(shx, index), family)
            if geom is not None:
                out.append(geom)
        return out


# ------------------------------------------------------------------ 属性表
#: DBF 语言驱动 ID → 代码页。``0x4D`` 是国内最常见的 GBK 老库（本机实测
#: ``北京其他道路.dbf`` 就是它）。认不出的按「先 utf-8 后 gbk」猜。
_DBF_CODEPAGE = {
    0x03: "cp1252",
    0x4D: "gbk",
    0x4E: "cp949",
    0x57: "cp1252",
    0x59: "cp1252",
    0x7A: "gbk",
    0x7B: "cp932",
    0xC8: "cp1250",
}


def _dbf_fields(dbf: mmap.mmap) -> list[tuple[str, str, int, int, int]]:
    """→ ``[(name, type, offset_in_record, length, decimals)]``。"""
    if len(dbf) < 32:
        raise ShapefileError(".dbf 头不足 32 字节")
    header_len, record_len = struct.unpack_from("<HH", dbf, 8)
    fields: list[tuple[str, str, int, int, int]] = []
    position = 32
    offset = 1  # 第 0 字节是删除标记
    while position + 32 <= min(header_len, len(dbf)):
        descriptor = dbf[position : position + 32]
        if descriptor[0] == 0x0D:
            break
        name = descriptor[:11].split(b"\x00")[0].decode("latin-1", "replace").strip()
        kind = chr(descriptor[11])
        length = descriptor[16]
        decimals = descriptor[17]
        if name:
            fields.append((name, kind, offset, length, decimals))
        offset += length
        position += 32
    if record_len <= 0:
        raise ShapefileError(".dbf 记录长度非法")
    return fields


def _decode(raw: bytes, charset: str | None) -> str:
    if charset is None:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("gbk", "replace")
    return raw.decode(charset, "replace")


def _cell(raw: bytes, kind: str, decimals: int, charset: str | None) -> Any:
    if kind in {"N", "F"}:
        text = raw.strip().strip(b"\x00").decode("latin-1", "replace")
        if not text or not text.strip("*"):  # 整列 '*' 填充 = 空值
            return None
        try:
            return float(text) if decimals or "." in text else int(text)
        except ValueError:
            return None
    if kind == "L":
        text = raw.strip()[:1].decode("latin-1", "replace").lower()
        return True if text in {"t", "y"} else False if text in {"f", "n"} else None
    if kind == "D":
        text = raw.strip().decode("latin-1", "replace")
        if len(text) == 8 and text.isdigit():
            return f"{text[:4]}-{text[4:6]}-{text[6:]}"
        return None
    text = _decode(raw, charset).rstrip("\x00").strip()
    return text or None


def read_attributes(
    path: str | Path, ids: Sequence[int], fields: Sequence[str]
) -> list[dict[str, Any]] | None:
    """读 ``.dbf`` 里这些记录的若干列；读不了返回 ``None``（调用方别据此报错）。

    只按记录号定位（固定长度记录 + 直接 seek），不扫全表——属性列是给符号化
    着色用的，缺了不该让整张图拉不出来。字段名大小写不敏感（arcpy 报的是 DBF
    里的原样大写，前端传的可能是小写）。
    """
    wanted = [str(name) for name in fields if str(name)]
    if not wanted:
        return None
    dbf_path = Path(path).with_suffix(".dbf")
    if not dbf_path.is_file():
        return None
    try:
        with open(dbf_path, "rb") as handle:
            size = dbf_path.stat().st_size
            if size < 32:
                return None
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    except OSError:
        return None
    try:
        header_len, record_len = struct.unpack_from("<HH", mapped, 8)
        declared = struct.unpack_from("<I", mapped, 4)[0]
        layout = {
            name.upper(): (name, kind, offset, length, decimals)
            for name, kind, offset, length, decimals in _dbf_fields(mapped)
        }
        charset = _DBF_CODEPAGE.get(mapped[29])
        cpg = dbf_path.with_suffix(".cpg")
        if cpg.is_file():
            try:
                charset = cpg.read_text(encoding="utf-8", errors="replace").strip() or charset
            except OSError:
                pass
        total = min(declared, max(0, (size - header_len) // record_len))
        out: list[dict[str, Any]] = []
        for index in ids:
            if not 0 <= index < total:
                out.append({})
                continue
            base = header_len + index * record_len
            if mapped[base : base + 1] == b"*":  # 删除标记
                out.append({})
                continue
            values: dict[str, Any] = {}
            for name in wanted:
                hit = layout.get(name.upper())
                if hit is None:
                    continue
                key, kind, offset, length, decimals = hit
                values[key] = _cell(
                    mapped[base + offset : base + offset + length], kind, decimals, charset
                )
            out.append(values)
        return out
    except (struct.error, ValueError):
        return None
    finally:
        mapped.close()


def load_view(
    path: str | Path,
    *,
    bbox: Sequence[float] | None = None,
    limit: int | None = None,
    fields: Sequence[str] = (),
) -> dict[str, Any]:
    """视口预览的主体：命中计数 + 取 ``limit`` 条命中要素（含可选属性列）。

    ``bbox`` 为 ``None`` 表示不裁剪（全量计数）；``limit`` 为 ``None`` 表示不截。
    """
    if bbox is None:
        with _open_pair(path) as (_shp, shx):
            ids = list(range(record_count(_shp, shx)))
        total = len(ids)
    else:
        hits, _layer_total = view_hits(path, bbox)
        total = len(hits)  # 视口内真实要素数（不是图层总数）
        ids = spread(hits, limit) if limit else hits
    geoms = read_geometries(path, ids)
    attrs = read_attributes(path, ids, fields) if fields else None
    return {
        "geoms": geoms,
        "attrs": attrs,
        "total_in_view": total,
        "picked": len(ids),
    }

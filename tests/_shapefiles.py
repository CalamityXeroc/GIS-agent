# -*- coding: utf-8 -*-
"""测试用的最小 shapefile 写入器（DAIR-386）。

真数据（``workspace/input``、``E:\\GIS_documents``）不入库，而视口裁剪必须用带
envelope 的真 ``.shp/.shx/.dbf`` 验证（记录头 bbox、环方向、DBF 定长记录）。
这里按 ESRI 规范写一份最小实现：点 / 折线 / 多边形 + 定长 DBF，够测
:mod:`gis_cli.report.shapefile_view` 的筛选与读数即可。

故意不支持：Z/M 变体、MultiPatch。那些由调用方的容错兜住（读不懂就退回内核），
不需要在测试里造。
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

SHP_POINT = 1
SHP_POLYLINE = 3
SHP_POLYGON = 5

_FILE_CODE = 9994
_VERSION = 1000


def _envelope(points: Iterable[Sequence[float]]) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _header(shape_type: int, length_words: int, box: tuple[float, float, float, float]) -> bytes:
    return (
        struct.pack(">i", _FILE_CODE)
        + b"\x00" * 20
        + struct.pack(">i", length_words)
        + struct.pack("<ii", _VERSION, shape_type)
        + struct.pack("<4d", *box)
        + struct.pack("<4d", 0.0, 0.0, 0.0, 0.0)
    )


def _content(shape_type: int, shape: Any) -> bytes:
    if shape_type == SHP_POINT:
        return struct.pack("<i2d", SHP_POINT, float(shape[0]), float(shape[1]))
    parts = [list(part) for part in shape]
    points = [point for part in parts for point in part]
    box = _envelope(points)
    indices: list[int] = []
    running = 0
    for part in parts:
        indices.append(running)
        running += len(part)
    body = struct.pack("<i4d", shape_type, *box)
    body += struct.pack("<ii", len(parts), len(points))
    body += struct.pack("<%di" % len(indices), *indices)
    body += struct.pack("<%dd" % (2 * len(points)), *[value for point in points for value in point[:2]])
    return body


def write_shapefile(
    path: str | Path,
    *,
    shape_type: int,
    shapes: Sequence[Any],
    fields: Sequence[tuple[str, str, int, int]] = (),
    rows: Sequence[Sequence[Any]] = (),
    encoding: str = "utf-8",
    ldid: int = 0x57,
) -> Path:
    """写一组 ``.shp/.shx/.dbf/.prj``（``.prj`` 交给调用方按需写）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if shapes:
        flat = (
            [shapes[0]]
            if shape_type == SHP_POINT
            else [point for shape in shapes for part in shape for point in part]
        )
        box = _envelope(flat)
    else:
        box = (0.0, 0.0, 0.0, 0.0)

    contents = [_content(shape_type, shape) for shape in shapes]
    # 用 list + join 拼：视口测试要写 2.9 万条记录的 fixture，`bytes +=` 在循环里
    # 是 O(n²)（实测 4.7s），join 只要 0.3s。
    shp_parts: list[bytes] = []
    shx_parts: list[bytes] = []
    offset = 50  # 头 100 字节 = 50 个 16 位字
    for index, content in enumerate(contents, start=1):
        shp_parts.append(struct.pack(">ii", index, len(content) // 2))
        shp_parts.append(content)
        shx_parts.append(struct.pack(">ii", offset, len(content) // 2))
        offset += 4 + len(content) // 2
    shp_body = b"".join(shp_parts)
    shx_body = b"".join(shx_parts)

    total = 50 + len(shp_body) // 2
    target.write_bytes(_header(shape_type, total, box) + shp_body)
    target.with_suffix(".shx").write_bytes(
        _header(shape_type, 50 + len(shx_body) // 2, box) + shx_body
    )
    if fields:
        target.with_suffix(".dbf").write_bytes(
            _dbf(fields, rows, encoding=encoding, ldid=ldid)
        )
    return target


def _dbf(
    fields: Sequence[tuple[str, str, int, int]],
    rows: Sequence[Sequence[Any]],
    *,
    encoding: str,
    ldid: int,
) -> bytes:
    header_len = 32 + 32 * len(fields) + 1
    record_len = 1 + sum(item[2] for item in fields)
    head = bytearray(32)
    head[0] = 0x03
    head[1], head[2], head[3] = 124, 1, 1  # 2024-01-01，dBase 只关心字段长度
    struct.pack_into("<IHH", head, 4, len(rows), header_len, record_len)
    head[29] = ldid
    blob = bytes(head)
    for name, kind, length, decimals in fields:
        descriptor = bytearray(32)
        encoded = name.encode("latin-1")[:11]
        descriptor[: len(encoded)] = encoded
        descriptor[11] = ord(kind)
        descriptor[16] = length
        descriptor[17] = decimals
        blob += bytes(descriptor)
    blob += b"\x0D"
    for row in rows:
        record = bytearray(b" ")
        for (name, kind, length, decimals), value in zip(fields, row):
            if value is None:
                cell = b" " * length
            elif kind in {"N", "F"}:
                text = f"{value:.{decimals}f}" if decimals else str(int(value))
                cell = text.encode("latin-1").rjust(length, b" ")
            else:
                cell = str(value).encode(encoding)[:length].ljust(length, b" ")
            record += cell
        blob += bytes(record[:record_len])
    return blob + b"\x1A"

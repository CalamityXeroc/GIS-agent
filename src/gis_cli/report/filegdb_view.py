# -*- coding: utf-8 -*-
"""视口裁剪用的 `.gdb` 要素类读取（DAIR-393）。

`.gdb` 没有 `.shx` 那样的偏移索引，``SearchCursor`` 只能从头往下扫、Python 侧按
``extent`` 判交，扫到 ``_scan_cap`` 行才停：13 万行的 z11 视口要扫 13 万行、12.5 s，
而且停下来时计数只是下限（「≥N」）。这一版把读盘换成两层：

1. **GDAL 快路**（``osgeo``，ArcGIS Pro 自带）：OpenFileGDB 每行解码比
   ``arcpy.da.SearchCursor`` 便宜 2.4 倍（实测 34 µs/行 vs 82 µs/行），视口计数
   直接是精确值。**注意**：GDAL 3.9/3.12 都**不用**库里的 ``.spx`` 空间索引——
   120 条要素的小视口同样要整表扫（实测 3.6 s），所以光靠它只快 1.3–1.4 倍。
2. **包络索引**（:func:`build_index`，落在 ``<workspace>/.gis_agent/gdb_index/``）：
   第一次读整层时顺手把每个要素的 OID + 包络写成定长二进制，之后视口查询先用
   ``numpy`` 掩码挑 OID（13 万行 ~2 ms），再用 ``GetFeature(oid)`` 只读命中的那些
   要素（实测 28,921 条 0.9 s）——不再整表扫。计数天然精确。

取不到 ``osgeo``（旧版 Pro / 精简环境）时退回原来的顺序扫描，行为逐字保持不变：
``scan_cap`` 扫到就停、``view_count_exact=false``。

不许改走「建图层再按位置选择」那条路（内核会挂死且 interrupt 无效）。

这条模块**不 import gis_cli**：内核（ArcGIS Pro 的 Python）不一定 import 得到本包，
``html_pack`` 把源码嵌进内核代码里 ``exec``（与 ``geom_simplify`` 同一套做法）。
几何输出是 **Esri JSON 形状**，与顺序扫描那条路逐字同一口径，环方向（外环顺时针、
洞逆时针）也按 Esri 约定归一，交给 ``esri_geometry_to_geojson`` 转 GeoJSON。
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any, Sequence


class GdalUnavailable(RuntimeError):
    """这台机器（这个内核）没有可用的 ``osgeo``，或者 GDAL 打不开这个库。"""


#: 视口扫描窗口的旧口径（与 ``html_pack.view_scan_cap`` 一致）：GDAL 不可用时
#: 顺序扫描最多看多少行。放在这里是为了让兜底行为与原模板逐字相同。
VIEW_SCAN_FACTOR = 20
VIEW_SCAN_FLOOR = 20_000

#: WKB 基础类型 → 名字。7（GeometryCollection）与 Esri 的圆弧类型不转：一条
#: 要素转不动就跳过它，不能让整层预览报错。
_WKB_POINT = 1
_WKB_LINESTRING = 2
_WKB_POLYGON = 3
_WKB_MULTIPOINT = 4
_WKB_MULTILINESTRING = 5
_WKB_MULTIPOLYGON = 6


def _norm(value: Any) -> Any:
    """属性值 → JSON 能直接带走的标量（与内核模板里的 ``_norm`` 同一口径）。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return "<binary %d bytes>" % len(value)
    try:
        return value.isoformat()
    except Exception:
        return str(value)


# ---------------------------------------------------------------- WKB → Esri JSON
def _signed_area(points: Sequence[Sequence[float]]) -> float:
    """鞋带公式（带符号）：负=顺时针（Esri 外环），正=逆时针（Esri 洞）。"""
    total = 0.0
    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _orient_ring(points: list[list[float]], outer: bool) -> list[list[float]]:
    """按 Esri 约定摆正环方向（外环顺时针、洞逆时针）；退化环不动。"""
    area = _signed_area(points)
    if area == 0.0:
        return points
    if (area < 0.0) != outer:
        points.reverse()
    return points


def _read_points(
    raw: bytes, offset: int, count: int, dims: int, endian: str
) -> tuple[list[list[float]], int]:
    """一次 ``unpack`` 取 count 个点（dims=2/3/4，Z/M 丢掉只留 XY）。"""
    values = struct.unpack_from("%s%dd" % (endian, count * dims), raw, offset)
    if dims == 2:
        return [
            [values[index], values[index + 1]] for index in range(0, len(values), 2)
        ], offset + 16 * count
    points = [
        [values[index], values[index + 1]] for index in range(0, len(values), dims)
    ]
    return points, offset + 8 * dims * count


def _wkb_header(raw: bytes, offset: int) -> tuple[int, int, int, str]:
    """→ (基础类型, 维数, 新偏移, 字节序)。兼容 ISO（1001+）与 Esri（0x80000000）写法。"""
    if offset + 5 > len(raw):
        raise ValueError("WKB 截断")
    endian = "<" if raw[offset] == 1 else ">"
    code = struct.unpack_from(endian + "I", raw, offset + 1)[0]
    offset += 5
    esri_z = bool(code & 0x80000000)
    code &= 0x7FFFFFFF
    if code >= 1000:
        family = (code // 1000) * 1000
        base = code % 1000
        dims = 4 if family == 3000 else 3
    else:
        base = code
        dims = 3 if esri_z else 2
    return base, dims, offset, endian


def esri_geometry_from_wkb(raw: bytes) -> dict[str, Any]:
    """WKB / EWKB → Esri JSON 形状（``x,y`` / ``paths`` / ``rings`` / ``points``）。

    转不动（GeometryCollection、圆环、截断）抛 :class:`ValueError`，调用方按
    「这条要素没有可画几何」跳过——不能让一条坏记录毁掉整层预览。
    """
    base, dims, offset, endian = _wkb_header(raw, 0)
    if base == _WKB_POINT:
        values = struct.unpack_from("%s%dd" % (endian, dims), raw, offset)
        return {"x": values[0], "y": values[1]}
    if base == _WKB_MULTIPOINT:
        count = struct.unpack_from(endian + "I", raw, offset)[0]
        offset += 4
        points: list[list[float]] = []
        for _ in range(count):
            base2, dims2, offset2, endian2 = _wkb_header(raw, offset)
            if base2 != _WKB_POINT:
                raise ValueError("MultiPoint 里混了非点几何")
            values = struct.unpack_from("%s%dd" % (endian2, dims2), raw, offset2)
            points.append([values[0], values[1]])
            offset = offset2 + 8 * dims2
        return {"points": points}
    if base == _WKB_LINESTRING:
        count = struct.unpack_from(endian + "I", raw, offset)[0]
        points, offset = _read_points(raw, offset + 4, count, dims, endian)
        return {"paths": [points]}
    if base == _WKB_MULTILINESTRING:
        count = struct.unpack_from(endian + "I", raw, offset)[0]
        offset += 4
        paths = []
        for _ in range(count):
            base2, dims2, offset2, endian2 = _wkb_header(raw, offset)
            if base2 != _WKB_LINESTRING:
                raise ValueError("MultiLineString 里混了非线几何")
            points, offset2 = _read_points(
                raw, offset2 + 4, struct.unpack_from(endian2 + "I", raw, offset2)[0], dims2, endian2
            )
            paths.append(points)
            offset = offset2
        return {"paths": paths}
    if base == _WKB_POLYGON:
        count = struct.unpack_from(endian + "I", raw, offset)[0]
        offset += 4
        rings = []
        for index in range(count):
            points, offset = _read_points(
                raw, offset + 4, struct.unpack_from(endian + "I", raw, offset)[0], dims, endian
            )
            rings.append(_orient_ring(points, outer=(index == 0)))
        return {"rings": rings}
    if base == _WKB_MULTIPOLYGON:
        count = struct.unpack_from(endian + "I", raw, offset)[0]
        offset += 4
        rings = []
        for _ in range(count):
            base2, dims2, offset2, endian2 = _wkb_header(raw, offset)
            if base2 != _WKB_POLYGON:
                raise ValueError("MultiPolygon 里混了非面几何")
            ring_count = struct.unpack_from(endian2 + "I", raw, offset2)[0]
            offset2 += 4
            for index in range(ring_count):
                points, offset2 = _read_points(
                    raw,
                    offset2 + 4,
                    struct.unpack_from(endian2 + "I", raw, offset2)[0],
                    dims2,
                    endian2,
                )
                rings.append(_orient_ring(points, outer=(index == 0)))
            offset = offset2
        return {"rings": rings}
    raise ValueError("不支持的 WKB 几何类型 %d" % base)


# ---------------------------------------------------------------- GDAL 快路
def _split_container(path: str) -> tuple[str, str]:
    """``…/x.gdb/数据集/要素类`` → (``…/x.gdb``, ``数据集/要素类``)；不是库内成员时成员为空。"""
    lowered = path.lower().replace("\\", "/")
    index = lowered.find(".gdb/")
    if index < 0:
        return path, ""
    member = path[index + 5 :].replace("\\", "/").strip("/")
    return path[: index + 4], member


def _open_layer(path: str):
    """→ (dataset, layer)。缺 ``osgeo`` / 打不开 / 找不到图层一律抛 :class:`GdalUnavailable`。"""
    try:
        from osgeo import ogr
    except Exception as exc:  # 没有 GDAL Python 绑定
        raise GdalUnavailable("没有 osgeo：%s" % (exc,))
    try:  # 静音 GDAL 4.0 的过渡警告；下面一律按 None / 返回值判错
        ogr.DontUseExceptions()
    except Exception:
        pass
    container, member = _split_container(str(path))
    dataset = ogr.Open(container, 0)
    if dataset is None:
        raise GdalUnavailable("GDAL 打不开 %s" % (container,))
    layer = None
    if member:
        layer = dataset.GetLayerByName(member)
        if layer is None:
            # 要素数据集里的要素类在 GDAL 侧只暴露叶名（``数据集/要素类`` → ``要素类``）。
            leaf = member.rsplit("/", 1)[-1]
            for index in range(dataset.GetLayerCount()):
                candidate = dataset.GetLayerByIndex(index)
                name = candidate.GetName()
                if name == leaf or name.endswith("/" + leaf) or name.endswith("." + leaf):
                    layer = candidate
                    break
    if layer is None and dataset.GetLayerCount() == 1:
        layer = dataset.GetLayerByIndex(0)
    if layer is None:
        raise GdalUnavailable("GDAL 找不到图层 %s" % (member or path,))
    return dataset, layer


#: 取值计划里的两个特殊标记：``@fid`` 是要素 ID 列（GDAL 不把它算普通字段，
#: 但前端属性表要它），``@null`` 是库里没有这一列（给 null，不让整层变成空）。
_FID = "@fid"
_NULL = "@null"


def _field_indexes(layer, names: Sequence[str]) -> list[Any]:
    """字段白名单 → 取值计划（普通字段下标 / ``@fid`` / ``@null``）。

    以前「任一列对不上就整组不带属性」，结果前端带 ``fields=OBJECTID``（属性表
    同步要 OID）时 ``OBJECTID`` 不是普通字段、计划为空，``rows`` 被截成 0 条，
    整层预览变成空白 —— 2026-11-24 真机复现（DAIR-393）。
    """
    definition = layer.GetLayerDefn()
    fid = str(layer.GetFIDColumn() or "")
    plan: list[Any] = []
    for name in names:
        index = definition.GetFieldIndex(str(name))
        if index >= 0:
            plan.append(index)
        elif fid and str(name).upper() == fid.upper():
            plan.append(_FID)
        else:
            plan.append(_NULL)
    return plan


def _row_values(feature, plan: Sequence[Any]) -> list[Any]:
    """按取值计划读一行属性（``_norm`` 收口成 JSON 标量）。"""
    values = []
    for item in plan:
        if item == _FID:
            values.append(feature.GetFID())
        elif item == _NULL:
            values.append(None)
        else:
            values.append(_norm(feature.GetField(item)))
    return values


def _read_with_gdal(
    path: str,
    bbox: Sequence[float],
    *,
    limit: int,
    fields: Sequence[str],
    index_target: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """整层扫一遍 → 视口内的要素；``index_target`` 给定时顺手把包络索引落盘。

    GDAL 的 ``SetSpatialFilterRect`` 并不用 ``.spx``（3.9/3.12 实测都是整表扫），
    所以这里在 Python 侧做相交判断——反正要按 OID 顺序把每行过一遍。要建索引时
    必须看**整层**（不能带 filter，否则索引只有视口那一块），这时也正是这个循环。
    """
    dataset, layer = _open_layer(str(path))
    indexes = _field_indexes(layer, fields)
    if index_target is None:
        layer.SetSpatialFilterRect(bbox[0], bbox[1], bbox[2], bbox[3])
    geoms: list[Any] = []
    attrs: list[list[Any]] = []
    oids: list[int] = []
    boxes: list[tuple[float, float, float, float]] = []
    total = 0
    try:
        for feature in layer:
            geometry = feature.GetGeometryRef()
            if geometry is None:
                continue
            envelope = geometry.GetEnvelope()  # (minx, maxx, miny, maxy)
            if index_target is not None:
                boxes.append((envelope[0], envelope[2], envelope[1], envelope[3]))
                oids.append(feature.GetFID())
            if (
                envelope[0] > bbox[2]
                or envelope[2] > bbox[3]
                or envelope[1] < bbox[0]
                or envelope[3] < bbox[1]
            ):
                continue
            total += 1
            if len(geoms) >= limit:
                continue
            try:
                esri = esri_geometry_from_wkb(geometry.ExportToWkb())
            except Exception:
                continue
            geoms.append(esri)
            if indexes:
                attrs.append(_row_values(feature, indexes))
    finally:
        dataset = None
    if index_target is not None and oids:
        # 整层扫完才落盘：中途出错时不能留下「只扫了一半」的索引。
        index_file, index_key_text = index_target
        try:
            write_index(index_file, index_key_text, oids, boxes)
        except Exception:
            pass  # 索引只是加速项，落盘失败不影响这次预览
    return {
        "error": "",
        "geoms": geoms,
        "attrs": attrs,
        "simplify": {},
        "view_total": total,
        "view_count_exact": True,
        "engine": "gdal",
    }


# ---------------------------------------------------------------- arcpy 兜底
def _read_with_arcpy(
    path: str, bbox: Sequence[float], *, limit: int, fields: Sequence[str], scan_cap: int | None
) -> dict[str, Any]:
    """原来的顺序扫描（DAIR-386 的模板逐字搬过来）：没有 osgeo 时的唯一出路。"""
    import arcpy  # 只有 ArcGIS Pro 的解释器里有

    cap = int(scan_cap or max(int(limit) * VIEW_SCAN_FACTOR, VIEW_SCAN_FLOOR))
    columns = ["SHAPE@"] + [str(name) for name in fields]
    geoms: list[Any] = []
    attrs: list[list[Any]] = []
    viewed = 0
    scanned = 0
    stopped = False
    try:
        with arcpy.da.SearchCursor(str(path), columns) as cursor:
            for row in cursor:
                scanned += 1
                if scanned > cap:
                    stopped = True
                    break
                geometry = row[0]
                if geometry is None:
                    continue
                extent = geometry.extent
                if (
                    extent.XMin > bbox[2]
                    or extent.YMin > bbox[3]
                    or extent.XMax < bbox[0]
                    or extent.YMax < bbox[1]
                ):
                    continue
                viewed += 1
                if len(geoms) < limit:
                    try:
                        geoms.append(json.loads(geometry.JSON))
                    except Exception:
                        geoms.append(geometry.JSON)
                    attrs.append([_norm(value) for value in row[1:]])
    except Exception as exc:
        return {
            "error": str(exc)[:300],
            "geoms": [],
            "attrs": [],
            "simplify": {},
            "view_total": viewed,
            "view_count_exact": False,
            "engine": "arcpy",
        }
    return {
        "error": "",
        "geoms": geoms,
        "attrs": attrs,
        "simplify": {},
        "view_total": viewed,
        "view_count_exact": not stopped,
        "engine": "arcpy",
    }


def read_view(
    path: str,
    bbox: Sequence[float],
    *,
    limit: int,
    fields: Sequence[str] = (),
    max_points: int | None = None,
    scan_cap: int | None = None,
    simplify: Any = None,
    index_path: str | None = None,
    index_key: str = "",
) -> dict[str, Any]:
    """读一个 `.gdb` 要素类的视口（GDAL 优先，缺 osgeo 退回顺序扫描）→ 内核回传体。

    返回体与 ``html_pack`` 的内核模板同一口径：``geoms`` / ``rows`` / ``error`` /
    ``simplify`` / ``view_total`` / ``view_count_exact``，外加 ``engine`` 说明这次
    走的是哪条路（``gdal`` / ``arcpy``），兜底时带 ``fallback`` 原因。

    ``simplify`` 是 ``simplify_esri_geometries``（内核侧由模板注入）：简化必须
    在几何还没跨进程搬运之前做完，这是 5MB 闸门的前提。

    ``index_path`` / ``index_key`` 给定时，这次整层扫的同时把包络索引写出去
    （:func:`write_index`）：第一次等得久一点，之后视口全走
    :func:`read_by_oids`，不再整表扫。
    """
    names = [str(name) for name in (fields or ())]
    box = [float(value) for value in bbox]
    limit = max(1, int(limit))
    fallback = ""
    target = (str(index_path), str(index_key)) if index_path else None
    try:
        payload = _read_with_gdal(
            str(path), box, limit=limit, fields=names, index_target=target
        )
    except GdalUnavailable as exc:
        payload = None
        fallback = str(exc)
    except Exception as exc:  # 库坏了 / 几何解码炸了：退回内核，宁可慢也别报「读不出」
        payload = None
        fallback = "GDAL 读取失败：%s" % (exc,)
    if payload is None:
        payload = _read_with_arcpy(
            str(path), box, limit=limit, fields=names, scan_cap=scan_cap
        )
        payload["fallback"] = fallback
    return _finish(payload, names, max_points=max_points, simplify=simplify)


def _finish(
    payload: dict[str, Any],
    names: Sequence[str],
    *,
    max_points: int | None,
    simplify: Any,
) -> dict[str, Any]:
    """简化 + 组装 ``rows``（GDAL 快路 / 索引快路 / 顺序扫描三条路共用）。"""
    if payload.get("geoms") and max_points and simplify is not None:
        try:
            geoms, stats = simplify(payload["geoms"], max_points=int(max_points))
        except Exception as exc:
            stats = {"error": str(exc)[:200]}
        else:
            payload["geoms"] = geoms
        payload["simplify"] = stats
    geoms = payload["geoms"]
    attrs = payload.get("attrs")
    if names and isinstance(attrs, list) and len(attrs) == len(geoms):
        payload["rows"] = [[geoms[index], attrs[index]] for index in range(len(geoms))]
    elif names and geoms:
        # 属性没取全（列名对不上之类）：几何照画，属性给 null，别把整层弄成空的。
        payload["rows"] = [[geom, [None] * len(names)] for geom in geoms]
    else:
        payload["rows"] = []
    # 带 ``fields`` 的请求，调用方只读 ``rows``：``geoms`` 再留一份等于 IPC 体积
    # 翻倍（整包过一遍内核哨兵），按旧模板口径清掉（DAIR-394 非阻塞项 1）。
    payload["geoms"] = [] if names else geoms
    payload.pop("attrs", None)
    return payload


# ---------------------------------------------------------------- 包络索引
#: 索引文件格式（定长、可 mmap、跨进程只读）：
#:
#: - ``0..7``   magic ``GDIDX1\0\0``
#: - ``8..11``  count（uint32 LE）
#: - ``12..15`` key_len（uint32 LE）
#: - ``16..``   key（utf-8 JSON，数据指纹；与文件对不上就当索引过期）
#: - 对齐到 8 字节后：OID（int64 LE × count）
#: - 再后：包络（float64 LE × 4 × count，顺序 minx, miny, maxx, maxy）
INDEX_MAGIC = b"GDIDX1\0\0"
INDEX_HEADER_BYTES = 16

#: 索引目录（相对 workspace）：与其它派生产物放一起，删掉就是重扫一次。
INDEX_DIR = Path(".gis_agent") / "gdb_index"


def index_path(workspace: str | Path, layer_path: str) -> Path:
    """一个要素类一个索引文件（路径小写后取 sha1，Windows 大小写不敏感）。"""
    digest = hashlib.sha1(str(layer_path).lower().encode("utf-8")).hexdigest()[:20]
    return Path(workspace) / INDEX_DIR / ("%s.idx" % digest)


def index_key(layer_path: str, feature_count: int | None = None) -> str:
    """数据指纹：库内所有 ``.gdbtable`` 的最新 mtime + 总大小 + 要素数。

    ``.gdb`` 目录自身的 mtime 只在增删文件时变，改要素不会变；逐个 ``.gdbtable``
    取 mtime/size 才能发现「有人编辑了这个图层」。取整库而不是单表：主进程不知道
    要素类对应哪个 ``a0000000N.gdbtable``，整库保守失效只会多扫一次。
    """
    container, member = _split_container(str(layer_path))
    latest = 0.0
    total = 0
    try:
        with os.scandir(container) as entries:
            for entry in entries:
                if not entry.name.lower().endswith(".gdbtable"):
                    continue
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                latest = max(latest, stat.st_mtime)
                total += stat.st_size
    except OSError:
        return ""
    return json.dumps(
        {"m": member, "t": round(latest, 3), "s": total, "n": feature_count},
        ensure_ascii=False,
        sort_keys=True,
    )


def write_index(
    path: str | Path,
    key: str,
    oids: Sequence[int],
    boxes: Sequence[Sequence[float]],
) -> None:
    """写索引（先写临时文件再 ``os.replace``：读的一方永远看不到半截文件）。"""
    count = len(oids)
    key_bytes = str(key).encode("utf-8")
    head = INDEX_MAGIC + struct.pack("<II", count, len(key_bytes))
    blob = bytearray(head)
    blob += key_bytes
    blob += b"\0" * ((-len(blob)) % 8)
    blob += struct.pack("<%dq" % count, *[int(value) for value in oids]) if count else b""
    if count:
        blob += struct.pack(
            "<%dd" % (count * 4),
            *[float(value) for box in boxes for value in box],
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(bytes(blob))
    os.replace(temporary, target)


def load_index(path: str | Path, key: str) -> tuple[Any, Any] | None:
    """读索引 → (OID 数组, 包络数组 count×4)；文件缺失 / 过期 / 损坏一律 None。

    用 ``mmap`` + ``numpy.frombuffer``：13 万行索引是 5.4 MB，读一次 ~2 ms，
    视口查询是一次四列比较（~2 ms）。
    """
    import mmap

    try:
        import numpy as np
    except ImportError:  # 索引只是加速项：没有 numpy 就整层扫，别让预览 500
        return None

    target = Path(path)
    try:
        handle = open(target, "rb")
    except OSError:
        return None
    try:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            if len(mapped) < INDEX_HEADER_BYTES or mapped[:8] != INDEX_MAGIC:
                return None
            count, key_len = struct.unpack_from("<II", mapped, 8)
            stored = bytes(mapped[INDEX_HEADER_BYTES : INDEX_HEADER_BYTES + key_len])
            if stored.decode("utf-8", "replace") != str(key):
                return None
            offset = INDEX_HEADER_BYTES + key_len
            offset += (-offset) % 8
            if len(mapped) < offset + count * 8 + count * 32:
                return None
            # 复制一份再关 mmap（``frombuffer`` 的视图会让 mmap.close() 抛 BufferError）。
            # 13 万行 5.4 MB 拷一次 ~10 ms，换掉整层扫的 4–9 s 很划算。
            oids = np.frombuffer(mapped, dtype="<i8", count=count, offset=offset).copy()
            boxes = np.frombuffer(
                mapped, dtype="<f8", count=count * 4, offset=offset + count * 8
            ).reshape(count, 4).copy()
            return oids, boxes
    except (OSError, ValueError):
        return None
    finally:
        handle.close()


def query_index(loaded: tuple[Any, Any], bbox: Sequence[float]) -> list[int]:
    """索引里与视口包络相交的 OID（与内核那条路的 ``extent`` 判交同一口径）。"""
    oids, boxes = loaded
    if not len(oids):
        return []
    minx, miny, maxx, maxy = (float(value) for value in bbox)
    hit = (
        (boxes[:, 0] <= maxx)
        & (boxes[:, 1] <= maxy)
        & (boxes[:, 2] >= minx)
        & (boxes[:, 3] >= miny)
    )
    return oids[hit].tolist()


def build_index(path: str, index_file: str, key: str) -> dict[str, Any]:
    """整层扫一遍把包络写进索引（每层只做一次；之后视口请求不再整表扫）。

    开头先按指纹查一次：没有就扫，有就返回 ``skipped``。这个判据必须在**内核里**
    做——预热与预览共用内核串行锁，主进程判「这层没索引」到真正拿到锁之间，
    预览那次整层扫可能已经把同一层的索引写出来了（DAIR-397 实测 green 层被建两
    遍，比无竞争时多花 3.7 s）。
    """
    existing = load_index(index_file, key)
    if existing is not None:
        return {"error": "", "count": int(len(existing[0])), "engine": "gdal", "skipped": True}
    dataset, layer = _open_layer(str(path))
    oids: list[int] = []
    boxes: list[tuple[float, float, float, float]] = []
    try:
        for feature in layer:
            geometry = feature.GetGeometryRef()
            if geometry is None:
                continue
            envelope = geometry.GetEnvelope()
            boxes.append((envelope[0], envelope[2], envelope[1], envelope[3]))
            oids.append(feature.GetFID())
        if oids:
            write_index(index_file, key, oids, boxes)
    except Exception as exc:
        return {"error": str(exc)[:300], "count": 0, "engine": "gdal"}
    finally:
        dataset = None
    return {"error": "", "count": len(oids), "engine": "gdal"}


def read_by_oids(
    path: str,
    oids: Sequence[int],
    *,
    limit: int,
    fields: Sequence[str] = (),
    max_points: int | None = None,
    simplify: Any = None,
) -> dict[str, Any]:
    """按索引挑出的 OID 读要素（``GetFeature`` 随机读，不再整表扫）。

    计数就是 ``len(oids)``——索引是整层扫出来的，视口里有多少条是确定的，
    所以 ``view_count_exact=true``，前端不用再说「≥N」。
    """
    names = [str(name) for name in (fields or ())]
    limit = max(1, int(limit))
    wanted = [int(value) for value in oids]
    payload: dict[str, Any] = {
        "error": "",
        "geoms": [],
        "attrs": [],
        "simplify": {},
        "view_total": len(wanted),
        "view_count_exact": True,
        "engine": "gdal_index",
    }
    try:
        dataset, layer = _open_layer(str(path))
    except GdalUnavailable as exc:
        return {
            "error": "",
            "geoms": [],
            "attrs": [],
            "simplify": {},
            "view_total": 0,
            "view_count_exact": False,
            "engine": "gdal_index",
            "fallback": str(exc),
        }
    except Exception as exc:
        payload["error"] = str(exc)[:300]
        return payload
    indexes = _field_indexes(layer, names)
    try:
        for oid in wanted:
            if len(payload["geoms"]) >= limit:
                break
            feature = layer.GetFeature(oid)
            if feature is None:
                continue
            geometry = feature.GetGeometryRef()
            if geometry is None:
                continue
            try:
                esri = esri_geometry_from_wkb(geometry.ExportToWkb())
            except Exception:
                continue
            payload["geoms"].append(esri)
            if indexes:
                payload["attrs"].append(_row_values(feature, indexes))
    except Exception as exc:
        payload["error"] = str(exc)[:300]
    finally:
        dataset = None
    return _finish(payload, names, max_points=max_points, simplify=simplify)

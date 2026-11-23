# -*- coding: utf-8 -*-
"""Filesystem discovery + ArcPy profiling for the catalog.

Discovery runs in the agent process (cheap filesystem walk). Profiling runs
inside the persistent ArcPy kernel, one call for the whole batch.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .kernel_health import KernelHealth
from .store import (
    Catalog,
    DatasetRecord,
    normalize_catalog_path,
    record_from_scan,
    sanitize_data,
    sanitize_text,
)

_SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "node_modules", ".gis_agent",
    ".gis_agent_memory", ".gis_agent_state", ".codegraph", ".claude", ".codex",
    ".pytest_cache", "temp", "tmp",
}
_VECTOR_EXT = {".shp", ".geojson", ".gpkg", ".kml"}
_RASTER_EXT = {".tif", ".tiff", ".img", ".dem", ".asc"}
_TABLE_EXT = {".csv", ".xlsx", ".dbf"}
_PROJECT_EXT = {".aprx", ".lyrx", ".mxd", ".mapx"}

#: 不靠 arcpy 也能读的矢量格式（在 agent 进程里读，毫秒级）。
#: .kml 是 XML、.gpkg 是 SQLite；交给内核只会白等十几秒，而且 .kml 根本
#: 不是 arcpy 能直接 Describe 的数据集（要先 KMLToLayer）。
_LOCAL_VECTOR_EXT = {".kml", ".gpkg"}


@dataclass
class Discovery:
    """A candidate dataset found on disk."""

    path: str
    kind: str
    container: str = ""
    mtime: float = 0.0
    size: int = 0


def discover(roots: Iterable[str | Path], *, max_items: int = 2000, include_gdb: bool = True) -> list[Discovery]:
    """Walk roots and return candidate datasets.

    GDB containers are reported as one ``gdb`` item; their inner feature
    classes are expanded during profiling.
    """
    found: list[Discovery] = []
    seen: set[str] = set()

    def _add(path: Path, kind: str, container: str = "") -> None:
        key = str(path).lower()
        if key in seen:
            return
        seen.add(key)
        try:
            stat = path.stat()
            mtime, size = stat.st_mtime, stat.st_size
        except OSError:
            mtime, size = 0.0, 0
        if kind == "gdb":
            # GDB 的“新鲜度”不能看目录 mtime：**读一下 GDB 就会新建 .sr.lock**，
            # 目录 mtime 因此立刻变化 → 缓存永远失效（实测每次启动都重扫 133 个
            # 要素类，耗时 158s）。改用“容器内非锁文件的最新 mtime”，读操作
            # 不会再把它推新。
            mtime = gdb_content_mtime(path)
        found.append(Discovery(str(path), kind, container, mtime, size))

    for root in roots:
        root_path = Path(root)
        if root_path.is_file():
            kind = _kind_for(root_path)
            if kind:
                _add(root_path, kind)
            continue
        if not root_path.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [
                d for d in dirnames
                if d.lower() not in _SKIP_DIRS and not d.startswith(".")
            ]
            current = Path(dirpath)
            if current.suffix.lower() == ".gdb" and include_gdb:
                _add(current, "gdb")
                dirnames[:] = []  # do not descend into gdb internals
                continue
            for filename in filenames:
                if filename.startswith("~$") or filename.startswith("."):
                    continue
                path = current / filename
                kind = _kind_for(path)
                if kind:
                    _add(path, kind)
            if len(found) >= max_items:
                break
        if len(found) >= max_items:
            break
    return found


#: GDB 内属于“临时/锁”的文件（不能参与新鲜度判定）
_GDB_IGNORE_SUFFIX = (".lock",)
_GDB_IGNORE_PREFIX = ("_gdb.",)


def gdb_content_mtime(path: str | Path) -> float:
    """GDB 的“内容时间”：容器内非锁、非临时文件的最新 mtime。

    为什么要这样算（实测，AM/PM 两套真题里的 GDB）：

    - ArcGIS **只要读过**一个 GDB 就会在里面新建 ``*.sr.lock``，
      目录自身的 mtime 随即被刷新；
    - 于是“目录 mtime 变了 → 缓存失效 → 重扫”形成死循环，
      每次启动都重扫全部 133 个要素类（实测 158.5s），而数据其实一动不动。

    只用真实数据文件的 mtime，读写探测就不会再把缓存推新；
    真正改数据（新增/重写要素类）仍能被检测到。
    目录不可读时退回目录自身 mtime（宁可能多扫一次，不能漏扫）。
    """
    try:
        newest = 0.0
        for entry in os.scandir(path):
            name = entry.name
            if name.endswith(_GDB_IGNORE_SUFFIX) or name.startswith(_GDB_IGNORE_PREFIX):
                continue
            try:
                stat = entry.stat()
            except OSError:
                continue
            if stat.st_mtime > newest:
                newest = stat.st_mtime
        if newest:
            return newest
    except OSError:
        pass
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def _kind_for(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in _VECTOR_EXT:
        return "vector"
    if ext in _RASTER_EXT:
        return "raster"
    if ext in _TABLE_EXT:
        return "table"
    if ext in _PROJECT_EXT:
        return "project"
    return None


# --------------------------------------------------------------------- arcpy
_SCAN_CODE = r'''
import json as _json
import os as _os
import arcpy as _arcpy

_TARGETS = {targets}
_SAMPLE_ROWS = {sample_rows}
_SAMPLE_VALUES = {sample_values}
_MAX_FIELDS = {max_fields}

_arcpy.env.overwriteOutput = True


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _samples_for(path, field_names):
    samples = {{name: [] for name in field_names}}
    nulls = {{name: 0 for name in field_names}}
    total = 0
    try:
        with _arcpy.da.SearchCursor(path, field_names) as cur:
            for row in cur:
                total += 1
                for name, value in zip(field_names, row):
                    if value is None:
                        nulls[name] += 1
                    elif len(samples[name]) < _SAMPLE_VALUES:
                        text = str(value).strip()
                        if text:
                            samples[name].append(text[:80])
                if total >= _SAMPLE_ROWS:
                    break
    except Exception:
        pass
    ratios = {{name: (nulls[name] / total if total else None) for name in field_names}}
    return samples, ratios


def _vector_record(path, kind, container=""):
    desc = _safe(lambda: _arcpy.Describe(path))
    if desc is None:
        return None
    sr = getattr(desc, "spatialReference", None)
    shape_type = getattr(desc, "shapeType", "") or ""
    # .gdb 里同时有要素类和非空间表，``Walk(datatype=["FeatureClass", "Table"])``
    # 不告诉我们命中的是哪种。以 dataType 为准（.gdb 表就是 "Table"），shapeType
    # 只在拿不到 dataType 时兜底：shapeType 拿不到只说明不知道几何种类，不代表
    # 没有几何；把真图层静默判成表，前端会标「仅属性表」且不给预览。
    data_type = str(getattr(desc, "dataType", "") or "").strip().lower()
    is_table = data_type == "table" or (
        not data_type and shape_type.strip().lower() in ("", "table")
    )
    if kind == "feature_class" and is_table:
        kind = "table"
        # 表没有几何：别把 shapeType 的 "Table" 当几何类型写进库——否则
        # ``kind='table' + geom_type<>''`` 会永久命中 store 的自愈谓词，
        # 每次 refresh 都重扫整个 .gdb（DAIR-348 审查）。
        shape_type = ""
    fields = []
    try:
        all_fields = _arcpy.ListFields(path)
    except Exception:
        all_fields = []
    for f in all_fields:
        if f.type in ("Geometry", "Blob", "Raster"):
            continue
        fields.append({{
            "name": f.name,
            "alias": getattr(f, "aliasName", "") or "",
            "type": f.type,
            "length": int(getattr(f, "length", 0) or 0),
            "nullable": bool(getattr(f, "isNullable", False)),
        }})
    sample_names = [f["name"] for f in fields[:_MAX_FIELDS]]
    samples, ratios = _samples_for(path, sample_names) if sample_names else ({{}}, {{}})
    for f in fields:
        if f["name"] in samples:
            f["samples"] = samples[f["name"]]
            f["null_ratio"] = ratios.get(f["name"])
    extent = None
    ext = getattr(desc, "extent", None)
    if ext is not None:
        extent = [ext.XMin, ext.YMin, ext.XMax, ext.YMax]
    count = _safe(lambda: int(_arcpy.management.GetCount(path)[0]))
    return {{
        "path": path,
        "kind": kind,
        "container": container,
        "name": getattr(desc, "baseName", _os.path.basename(path)),
        "crs_name": getattr(sr, "name", "") if sr else "",
        "wkid": getattr(sr, "factoryCode", None) if sr else None,
        "geom_type": shape_type,
        "feature_count": count,
        "extent": extent,
        "fields": fields,
    }}


def _raster_record(path):
    desc = _safe(lambda: _arcpy.Describe(path))
    if desc is None:
        return None
    sr = getattr(desc, "spatialReference", None)
    stats = {{}}
    for prop in ("MINIMUM", "MAXIMUM", "MEAN", "STD", "NOData_VALUE"):
        val = _safe(lambda p=prop: _arcpy.management.GetRasterProperties(path, p).getOutput(0))
        try:
            stats[prop.lower()] = float(val) if val not in (None, "") else None
        except Exception:
            stats[prop.lower()] = None
    return {{
        "path": path,
        "kind": "raster",
        "container": "",
        "name": _os.path.basename(path),
        "crs_name": getattr(sr, "name", "") if sr else "",
        "wkid": getattr(sr, "factoryCode", None) if sr else None,
        "geom_type": "Raster",
        "feature_count": None,
        "extent": None,
        "fields": [],
        "raster": {{
            "band_count": _safe(lambda: int(desc.bandCount)),
            "cell_size": _safe(lambda: float(desc.meanCellWidth)),
            "width": _safe(lambda: int(desc.width)),
            "height": _safe(lambda: int(desc.height)),
            "min": stats.get("minimum"),
            "max": stats.get("maximum"),
            "mean": stats.get("mean"),
            "nodata": stats.get("nodata_value"),
        }},
    }}


def _project_record(path):
    records = []
    try:
        aprx = _arcpy.mp.ArcGISProject(path)
    except Exception as exc:
        return [{{
            "path": path, "kind": "project", "container": "", "name": _os.path.basename(path),
            "fields": [], "extra": {{"error": str(exc)}},
        }}]
    maps = []
    for m in _safe(lambda: aprx.listMaps(), []) or []:
        layers = [layer.name for layer in (_safe(lambda mm=m: mm.listLayers(), []) or [])]
        maps.append({{"name": m.name, "layers": layers[:50]}})
    layouts = []
    for layout in _safe(lambda: aprx.listLayouts(), []) or []:
        elements = []
        for el in _safe(lambda l=layout: l.listElements(), []) or []:
            elements.append({{"name": el.name, "type": getattr(el, "type", "")}})
        layouts.append({{"name": layout.name, "elements": elements[:50]}})
    records.append({{
        "path": path, "kind": "project", "container": "", "name": _os.path.basename(path),
        "fields": [], "extra": {{"maps": maps, "layouts": layouts}},
    }})
    return records


def _scan_target(target):
    kind = target.get("kind")
    path = target.get("path")
    container = target.get("container", "") or ""
    if kind == "gdb":
        records = []
        try:
            walk = _arcpy.da.Walk(path, datatype=["FeatureClass", "Table"])
            for dirpath, dirnames, filenames in walk:
                for filename in filenames:
                    full = _os.path.join(dirpath, filename)
                    rec = _safe(lambda p=full, c=path: _vector_record(p, "feature_class", c))
                    if rec:
                        records.append(rec)
        except Exception as exc:
            return [{{"path": path, "kind": "gdb", "container": "", "name": _os.path.basename(path),
                      "fields": [], "extra": {{"error": str(exc)}}}}]
        return records
    if kind == "vector":
        rec = _safe(lambda: _vector_record(path, "vector"))
        return [rec] if rec else []
    if kind == "raster":
        rec = _safe(lambda: _raster_record(path))
        return [rec] if rec else []
    if kind == "project":
        return _safe(lambda: _project_record(path), []) or []
    return []


_records = []
for _target in _TARGETS:
    _records.extend(_scan_target(_target) or [])

set_result({{"records": _records, "count": len(_records)}})
'''


def build_scan_code(items: list[Discovery], *, sample_rows: int = 1000, sample_values: int = 3, max_fields: int = 40) -> str:
    """Build the ArcPy scan code for a batch of discovered items.

    ``ensure_ascii=True`` 是必要的（DAIR-369）：工作区里可能有文件名含孤立代理项
    （WTF-8 破损名），非 ASCII 的路径嵌进代码文本后要过 JSON/websocket 传输，
    原样编码会抛 ``UnicodeEncodeError``，把整批扫描带掉。转义成 ``\\uddd7``
    后 Python 字面量解析回来还是同一个字符串，内核照样能用它访问文件系统。
    """
    targets = [
        {"path": d.path, "kind": d.kind, "container": d.container}
        for d in items
    ]
    return _SCAN_CODE.format(
        targets=json.dumps(targets, ensure_ascii=True),
        sample_rows=int(sample_rows),
        sample_values=int(sample_values),
        max_fields=int(max_fields),
    )


def profile_table(path: Path, *, max_rows: int = 5000) -> dict[str, Any]:
    """Profile a CSV/XLSX without ArcPy (agent-side, best effort)."""
    import csv

    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader, [])
                samples: list[list[str]] = []
                count = 0
                for row in reader:
                    count += 1
                    if len(samples) < 3:
                        samples.append([str(v)[:80] for v in row[: len(header)]])
                    if count >= max_rows:
                        break
            return {
                "path": str(path),
                "kind": "table",
                "name": path.name,
                "feature_count": count,
                "fields": [
                    {"name": h, "type": "String", "samples": [s[i] for s in samples if i < len(s)][:3]}
                    for i, h in enumerate(header)
                ],
            }
        except Exception as exc:
            return {"path": str(path), "kind": "table", "name": path.name, "extra": {"error": str(exc)}}
    return {"path": str(path), "kind": "table", "name": path.name}


def profile_vector(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    """Profile a KML/GPKG without ArcPy (agent-side).

    KML 一个文件一条记录；GPKG 一个图层一条记录（container 指向 .gpkg，
    与 .gdb 的记账方式一致，新鲜度判断也能复用 container 分支）。
    """
    from ..ingest.formats import describe_vector

    return describe_vector(path)


@dataclass
class RefreshStats:
    """Summary of a catalog refresh."""

    discovered: int = 0
    scanned: int = 0
    skipped: int = 0
    removed: int = 0
    #: 归一路径时合并掉的重复记录数（DAIR-335）。
    merged: int = 0
    errors: list[str] = field(default_factory=list)
    #: 内核根本没起来（``kernel is None``）时的可读原因；非空就代表这一轮的
    #: 内核侧条目（.shp/.gdb/.tif…）一个都没扫成（DAIR-374）。
    kernel_unavailable: str = ""


#: 名字含孤立代理项（WTF-8 破损名）时的降级说明。落库前必须清洗成合法 UTF-8，
#: 但只标“不可读”不够——用户要知道为什么打不开。
BAD_NAME_NOTE = "名称含非法字符（WTF-8 破损），已用替换字符显示；原文件名无法用于读取"

#: 内核没起来（arcpy 缺失 / 内核健康度退避）时的可读原因：文案与预览接口 503 的
#: ``kernel_unavailable`` 保持一致，用户不要在目录扫描里看到 ``AttributeError``（DAIR-374）。
KERNEL_UNAVAILABLE_NOTE = "ArcGIS 内核不可用：未检测到 arcpy（当前环境没有 ArcGIS Pro Python）"


def refresh_catalog(
    catalog: Catalog,
    roots: Iterable[str | Path],
    kernel: Any,
    *,
    workspace: str | Path | None = None,
    force: bool = False,
    max_items: int = 2000,
    health: KernelHealth | None = None,
) -> RefreshStats:
    """Incrementally refresh the catalog from roots using an ArcPyKernel.

    ``workspace``：落库前把路径归一成「相对 workspace」的一条
    （``normalize_catalog_path``），并先合并库里已有的重复写法。不传时
    保持旧行为（原样写），一次性脚本与测试可以不动。
    扫描本身仍用绝对路径（内核进程的 cwd 不可控）。

    ``health``：内核熔断器（DAIR-365）。传了就只在**内核级**失败 / 成功后
    记账，由调用方决定退避期间是否还起这一轮；不传则完全不记（预热、一次性
    脚本保持旧行为）。
    """
    stats = RefreshStats()
    roots = list(roots)
    # 内核没起来就不要走到 ``kernel.execute``（DAIR-374）：None.execute 抛的
    # ``AttributeError`` 对用户毫无意义，而且 gdb 容器会变成“0 成员、无原因”。
    # 注意：只有**真有条目要交给内核**时才算这一轮的失败原因——纯 csv 目录
    # / 空目录本来就不靠内核，不该报「内核不可用」。
    kernel_unavailable = "" if kernel is not None else KERNEL_UNAVAILABLE_NOTE
    if workspace is not None:
        # 必须先合并历史重复：否则归一后的 path 在库里找不到旧记录，
        # 除了全量重扫一遍，还会多出一套新写法与旧写法并存的重复。
        stats.merged = catalog.renormalize_paths(workspace)

    def _key(path: str | Path) -> str:
        return normalize_catalog_path(path, workspace) if workspace else str(path)

    def _store(payload: dict[str, Any]) -> None:
        # 落库前统一清洗字符串（DAIR-369）：坏名（孤立代理项）流到 sqlite 绑定 /
        # JSON 响应里就是 UnicodeEncodeError，一条就能让整轮扫描失败。这里
        # 只清洗“写进库/回给界面”的字段；扫描期访问文件系统用的原始路径在
        # 内核那一侧（``build_scan_code`` 用 ASCII 转义保住它）。
        raw_path = str(payload.get("path", "") or "")
        clean = sanitize_data(payload)
        if str(clean.get("path", "")) != raw_path:
            extra = dict(clean.get("extra") or {})
            extra.setdefault("error", BAD_NAME_NOTE)
            clean["extra"] = extra
        if str(clean.get("kind", "")) == "gdb":
            # 内核报“这个库读不出来”的占位记录（`_scan_target` 的 except 分支）：
            # 清掉 members 标记，否则 mtime 被这一轮更新成新的、下轮 is_fresh
            # 会判成“没变化”，失败的 gdb 就永远修不回来了。
            extra = dict(clean.get("extra") or {})
            extra["members"] = None
            clean["extra"] = extra
        if str(clean.get("kind", "")) == "table" and clean.get("geom_type"):
            # 表没有几何：内核怎么报都不能把它当几何类型落库（旧版会写成
            # "Table"）。留着就永久命中 kind_mismatch_keys()，每轮 refresh
            # 都要重扫整个 .gdb（DAIR-348 审查）。
            clean = {**clean, "geom_type": ""}
        record: DatasetRecord = record_from_scan(clean)
        if workspace is not None:
            record.path = _key(record.path)
            if record.container:
                record.container = _key(record.container)
        catalog.upsert(record)

    def _store_gdb_containers(reason: str = "") -> None:
        """每个 .gdb 都落一条容器记录（``kind="gdb"``）。

        为什么必须单独记账（DAIR-369）：成员记录只在**内核**扫成功后才存在。
        内核不可用 / 某个 gdb 读不出来（`_scan_target` 返回带 ``extra.error``
        的占位记录）时，目录树里就只剩“什么都没有”，用户看到的就是“gdb
        打不开”。容器记录让 .gdb 始终以文件夹形态出现，成员数 0 时也不消失。

        不写 ``extra.members``：那是 ``_mark_gdb_members`` 成功枚举后才写的，
        ``is_fresh`` 靠它区分“空 gdb”与“还没扫成功过的 gdb”。

        ``reason``：内核不可用的原因（DAIR-374）。带上它而不是留白，用户才能
        知道“这个库里有东西但现在打不开”与“这个库是空的”不是一回事。
        """
        for item in items:
            if item.kind != "gdb":
                continue
            key = sanitize_text(_key(item.path))
            if catalog.contains(key):
                # 已存在就**不动**：mtime/size/成员数只由成功枚举那一步
                # （``_mark_gdb_members``）更新，否则“旧数据 + 新 mtime”
                # 会被 is_fresh 当成已扫过，改过的 gdb 永远不重扫。
                continue
            catalog.upsert(
                DatasetRecord(
                    path=key,
                    kind="gdb",
                    name=sanitize_text(Path(item.path).name),
                    mtime=item.mtime,
                    size=item.size,
                    extra={"error": reason} if reason else {},
                )
            )

    def _mark_gdb_members(stored: list[dict[str, Any]]) -> None:
        """内核枚举成功后，把成员数写回容器记录（``extra.members``）。

        一方面是给界面一个“这个库里真有 N 个图层”的口径，另一方面是
        ``is_fresh`` 的判据：没有这个标记就说明这轮没成功枚举过，下次要重扫。

        ``stored`` 只含**真正落库成功**的成员（DAIR-373）：内核报了但随后写库
        失败的成员不能算进成员数，否则容器会被当成“已枚举完整”。
        """
        counts: dict[str, int] = {}
        failed: set[str] = set()
        for raw in stored:
            if not isinstance(raw, dict):
                continue
            container = str(raw.get("container", "") or "")
            if container:
                key = _key(container)
                counts[key] = counts.get(key, 0) + 1
            elif str(raw.get("kind", "")) == "gdb":
                # 内核报“这个库读不出来”的占位记录（path 就是容器自己）：
                # ``_store`` 已经写了可读原因 + members=None，这里不能再把它
                # 改成“已枚举”（否则失败的 gdb 从此不再重扫）。
                failed.add(_key(raw.get("path", "")))
        for item in by_container.values():
            key = _key(item.path)
            if key in failed:
                continue
            catalog.upsert(
                DatasetRecord(
                    path=sanitize_text(key),
                    kind="gdb",
                    name=sanitize_text(Path(item.path).name),
                    mtime=item.mtime,
                    size=item.size,
                    extra={"members": counts.get(key, 0)},
                )
            )

    items = discover(roots, max_items=max_items)
    stats.discovered = len(items)

    # 清理“已消失”的记录时**限定在本次扫描的根目录内**（实测坑）：
    # 默认只扫 input/ 时，旧逻辑把 output/ 里的记录（自己刚生成的成果）
    # 当成“文件已消失”删掉了（实测 removed=19）。数据目录应只增不误删。
    root_prefixes = []
    covers_all = False
    for r in roots:
        key = _key(Path(r).resolve())
        if key in (os.curdir, ""):
            # 根就是 workspace 本身（没有 input/data/output 时的兜底根）→
            # 归一后是 "."，拿它当前缀永远匹配不上 input\\output 记录，
            # 已删除的数据集就永远清不掉。这种扫描的意义就是「全库对账」。
            covers_all = True
            continue
        if not key.endswith(os.sep):
            key += os.sep  # 带分隔符：否则 input 会把 input2 也算进清理范围
        root_prefixes.append(key.lower())
    # 容器：.gdb 目录与 .gpkg 文件。它们的子记录 path 不在 items 里，
    # 只看 path 会把它们当“已消失”每轮删一遍（GPKG 于是永远重扫）。
    containers = {
        _key(d.path) for d in items
        if d.kind == "gdb" or Path(d.path).suffix.lower() == ".gpkg"
    }
    stats.removed = catalog.remove_missing(
        {_key(d.path) for d in items},
        containers,
        within=None if covers_all else root_prefixes,
        # 全库对账只对工作区内负责：外部数据源的绝对写法记录不归这次扫描管。
        workspace=workspace if covers_all else None,
    )

    to_scan: list[Discovery] = []
    table_items: list[Discovery] = []
    local_vector_items: list[Discovery] = []
    # 旧版本把 .gdb 里的非空间表标成了 feature_class（geom_type 为空）。
    # 这些记录即使 mtime 没变也必须重扫，否则错分类永远自愈不了（DAIR-343）。
    stale_kinds = catalog.kind_mismatch_keys()
    for item in items:
        key = _key(item.path)
        if (
            not force
            and key not in stale_kinds
            and catalog.is_fresh(key, item.mtime, item.size)
        ):
            stats.skipped += 1
            continue
        if item.kind == "table":
            table_items.append(item)
        elif item.kind == "vector" and Path(item.path).suffix.lower() in _LOCAL_VECTOR_EXT:
            local_vector_items.append(item)
        else:
            to_scan.append(item)

    # 容器记录要在**新鲜度判断之后**落：先落的话 ``is_fresh(container)`` 会直接
    # 命中刚写进去的行，gdb 第一轮就被判成“没变化”，成员永远扫不出来。
    # 真有 gdb 要展开、而内核又没起来时，容器就带上原因（否则目录里只是个
    # 0 成员的空文件夹，用户不知道是空的还是打不开）。
    _store_gdb_containers(kernel_unavailable if to_scan else "")

    # Tables are profiled locally.
    for item in table_items:
        try:
            payload = profile_table(Path(item.path))
            payload.update({"mtime": item.mtime, "size": item.size, "container": item.container})
            _store(payload)
            stats.scanned += 1
        except Exception as exc:
            stats.errors.append(sanitize_text(f"{item.path}: {exc}")[:300])

    # KML / GPKG 也本地读（不用起内核，不用 arcpy）。
    for item in local_vector_items:
        try:
            payload = profile_vector(Path(item.path))
            records = payload if isinstance(payload, list) else [payload]
            for raw in records:
                raw["mtime"] = item.mtime
                raw["size"] = item.size
                raw.setdefault("container", item.container)
                _store(raw)
                stats.scanned += 1
        except Exception as exc:
            stats.errors.append(sanitize_text(f"{item.path}: {exc}")[:300])

    if not to_scan:
        return stats

    if kernel is None:
        # 内核压根没起来（arcpy 缺失 / 健康度退避）：能把本地条目扫的扫了，
        # 内核侧条目给一句人能读的原因，而不是让 ``None.execute`` 抛
        # ``AttributeError``（DAIR-374）。按内核级失败记账，退避/重试口径与
        # ``ok=False`` 一致。
        if health is not None:
            health.record_failure(kernel_unavailable)
        stats.kernel_unavailable = kernel_unavailable
        stats.errors.append(kernel_unavailable)
        return stats

    # ArcPy-backed items are profiled in one kernel call.
    code = build_scan_code(to_scan)
    try:
        result = kernel.execute(code, timeout=900)
    except Exception as exc:
        # 内核起不来 / 被中断：与 ``ok=False`` 同等对待（整轮没扫成）。
        if health is not None:
            health.record_failure(sanitize_text(f"kernel scan failed: {exc}"))
        raise
    if not result.ok:
        if health is not None:
            health.record_failure(sanitize_text(f"kernel scan failed: {result.error}"))
        stats.errors.append(sanitize_text(f"kernel scan failed: {result.error}"))
        return stats
    if health is not None:
        health.record_success()
    payload = result.result or {}
    records = payload.get("records", []) if isinstance(payload, dict) else []
    by_container = {d.path: d for d in to_scan if d.kind == "gdb"}
    by_path = {d.path: d for d in to_scan}
    stored: list[dict[str, Any]] = []

    for raw in records:
        if not isinstance(raw, dict):
            continue
        path = str(raw.get("path", ""))
        container = str(raw.get("container", "") or "")
        source = by_path.get(path) or by_container.get(container)
        if source is None:
            source = next(
                (d for d in to_scan if container and Path(path).is_relative_to(Path(container))),
                None,
            )
        raw["mtime"] = source.mtime if source else 0.0
        raw["size"] = source.size if source else 0
        try:
            _store(raw)
        except Exception as exc:
            # 单条记录写不进去（坏名、字段异常）只降级它自己：这个 batch 里
            # 其它 gdb / shp 的记录必须能落库（DAIR-369 打回点）。
            stats.errors.append(sanitize_text(f"{path or container}: {exc}")[:300])
            continue
        stored.append(raw)
        stats.scanned += 1
    # 成员数必须在写库**之后**算：先算的话，随后写失败的成员也被计入成员数，
    # 容器就被判成“已成功枚举”，缺失的那条以后不再重扫（DAIR-373）。
    _mark_gdb_members(stored)
    return stats

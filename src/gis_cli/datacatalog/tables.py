# -*- coding: utf-8 -*-
"""属性表读取：把真实数据文件读成「字段清单 + 行」，交给 catalog 行缓存。

为什么先落缓存再查询（DAIR-328）：属性表要服务端分页/排序/筛选，而 ArcPy 的
``SearchCursor`` 只有 ``where_clause`` 字符串、没有参数化，也没有排序/分页。
所以这里**只做一次全表读取**（受 :data:`store.MAX_CACHE_ROWS` 限制），之后
所有查询都在 catalog 的 SQLite 行缓存上用绑定参数完成——用户输入永远不会
被拼进 SQL 或 ``where_clause``。

读取路径按格式分流：

- ``.geojson/.json``：标准库直读；
- ``.csv``：标准库直读（UTF-8 / GBK 自动回退）；
- ``.gpkg`` 内的图层：SQLite 只读连接；
- ``.shp`` / ``.gdb`` 要素类：ArcPy 内核（一次 ``SearchCursor``）；无 arcpy 时
  抛 :class:`TableReadError`（503 + 原因），而不是崩。

所有上限都写在这里，调用方不需要重复判断：行数、字段数、单元格长度、
文件大小（``.geojson`` 直读时的内存保护）。
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from .store import MAX_CACHE_ROWS, MAX_CELL_CHARS
from .scanner import gdb_content_mtime

#: 字段类型里能按数值比较/排序的那些（Esri Field.type 与 SQLite 声明类型混用）。
NUMERIC_TYPES = {
    "Double",
    "Single",
    "Integer",
    "SmallInteger",
    "BigInteger",
    "OID",
    "Float",
    "Float32",
    "Float64",
    "Int16",
    "Int32",
    "Int64",
    "Decimal",
    "REAL",
    "NUMERIC",
    "INTEGER",
    "FLOAT",
}

#: 直读格式的体积上限：超过就不进内存（提示改用内核/GDAL 或抽稀）。
MAX_DIRECT_BYTES = 64 * 1024 * 1024
#: 字段数上限：属性表宽到 120 列已经没有可用性，缓存也不再涨。
MAX_FIELDS = 120
#: GeoJSON 推断字段类型时最多看多少个要素（避免为推断类型扫全表）。
TYPE_SAMPLE_ROWS = 500

#: 能直读的属性表格式。
DIRECT_CSV = {".csv"}
DIRECT_JSON = {".geojson", ".json"}


class TableReadError(RuntimeError):
    """读不出属性表。``code`` 给 HTTP 层映射状态码用，``reason`` 给人看。"""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


def _cell(value: Any) -> Any:
    """单元格归一：只保留 JSON 原生类型，超长文本截断（并在 notes 里记账）。"""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return f"<binary {len(value)} bytes>"
    text = value if isinstance(value, str) else str(value)
    return text[:MAX_CELL_CHARS]


def _infer_type(values: list[Any]) -> str:
    """按样本推断字段类型：全是整数 → Integer，全是数字 → Double，否则 String。"""
    seen_number = False
    seen_int = False
    for value in values:
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            return "String"
        if isinstance(value, int):
            seen_int = True
            seen_number = True
            continue
        if isinstance(value, float):
            seen_number = True
            continue
        if isinstance(value, str):
            text = value.strip()
            try:
                number = float(text)
            except ValueError:
                return "String"
            seen_number = True
            if number.is_integer() and "." not in text and "e" not in text.lower():
                seen_int = True
    if not seen_number:
        return "String"
    return "Integer" if seen_int else "Double"


def _fields_from_rows(
    names: list[str], rows: list[dict[str, Any]], *, sample: int = TYPE_SAMPLE_ROWS
) -> list[dict[str, Any]]:
    fields = []
    for name in names[:MAX_FIELDS]:
        values = [row.get(name) for row in rows[:sample]]
        fields.append(
            {
                "name": name,
                "alias": "",
                "type": _infer_type(values),
                "length": 0,
            }
        )
    return fields


def _coerce_csv_rows(
    rows: list[dict[str, Any]], fields: list[dict[str, Any]]
) -> list[str]:
    """CSV 没有 NULL 语义，所以按推断出的字段类型把单元格改写成原生类型。

    空单元格 → ``None``（与 ArcGIS/QGIS 属性表的空值一致）；数值列转 int/float，
    否则排序会变成字符串比较（实测：``"100" < "8.0"``），分位断点也会错。
    转不动的单元格保留原文本并记一条 note，不静默丢数据。
    """
    notes: list[str] = []
    for field in fields:
        name = field["name"]
        numeric = field["type"] in NUMERIC_TYPES
        bad = 0
        for row in rows:
            value = row.get(name)
            if value is None or value == "":
                row[name] = None
                continue
            if not numeric:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                continue
            try:
                number = float(str(value).strip())
            except (TypeError, ValueError):
                bad += 1
                continue
            row[name] = int(number) if field["type"] == "Integer" and number.is_integer() else number
        if bad:
            notes.append(f"字段 {name} 有 {bad} 个值不是数字，保留原文本")
    return notes


# --------------------------------------------------------------------- 直读
def _read_json(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size > MAX_DIRECT_BYTES:
        raise TableReadError(
            "too_large",
            f"GeoJSON 有 {size / 1048576:.1f}MB，超过直读上限 "
            f"{MAX_DIRECT_BYTES // 1048576}MB；请改用要素类或先抽稀",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError) as exc:
        raise TableReadError("unreadable", f"GeoJSON 解析失败：{exc}") from exc

    features = payload.get("features") if isinstance(payload, dict) else None
    if features is None and isinstance(payload, dict) and payload.get("type") == "Feature":
        features = [payload]
    if not isinstance(features, list):
        raise TableReadError("unsupported", "该 JSON 不是 GeoJSON（没有 features 数组）")

    names: list[str] = []
    rows: list[dict[str, Any]] = []
    truncated = False
    for feature in features:
        if len(rows) >= MAX_CACHE_ROWS:
            truncated = True
            break
        properties = feature.get("properties") if isinstance(feature, dict) else None
        properties = properties if isinstance(properties, dict) else {}
        row = {}
        for key, value in properties.items():
            if len(names) < MAX_FIELDS and key not in names:
                names.append(key)
            if key in names:
                row[key] = _cell(value)
        rows.append(row)
    return {
        "fields": _fields_from_rows(names, rows),
        "rows": rows,
        "truncated": truncated,
        "source": "geojson",
        "notes": [],
    }


def _read_csv(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size > MAX_DIRECT_BYTES:
        raise TableReadError(
            "too_large",
            f"CSV 有 {size / 1048576:.1f}MB，超过直读上限 "
            f"{MAX_DIRECT_BYTES // 1048576}MB；请先抽稀",
        )
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gbk"):
        try:
            with open(path, "r", encoding=encoding, newline="", errors="strict") as handle:
                reader = csv.DictReader(handle)
                names = list(reader.fieldnames or [])[:MAX_FIELDS]
                rows: list[dict[str, Any]] = []
                truncated = False
                for record in reader:
                    if len(rows) >= MAX_CACHE_ROWS:
                        truncated = True
                        break
                    rows.append({name: _cell(record.get(name)) for name in names})
            fields = _fields_from_rows(names, rows)
            notes = _coerce_csv_rows(rows, fields)
            if encoding != "utf-8-sig":
                notes.append(f"按 {encoding} 解码")
            return {
                "fields": fields,
                "rows": rows,
                "truncated": truncated,
                "source": "csv",
                "notes": notes,
            }
        except UnicodeDecodeError as exc:  # 换下一个编码再试
            last_error = exc
            continue
        except (OSError, csv.Error) as exc:
            raise TableReadError("unreadable", f"CSV 读取失败：{exc}") from exc
    raise TableReadError("unreadable", f"CSV 编码无法识别（UTF-8/GBK 都失败）：{last_error}")


def stat_key(path: Path) -> tuple[float, int]:
    """(mtime, size)；``.gdb`` 内的要素类看容器内容时间（读一次 GDB 会改目录 mtime）。"""
    try:
        stat = path.stat()
        return stat.st_mtime, stat.st_size
    except OSError:
        pass
    lowered = str(path).lower().replace("\\", "/")
    index = lowered.find(".gdb/")
    if index >= 0:
        try:
            return gdb_content_mtime(Path(str(path)[: index + 4])), 0
        except OSError:  # pragma: no cover - 目录都读不到
            return 0.0, 0
    return 0.0, 0


def split_gpkg(path: Path) -> tuple[Path, str] | None:
    """``…/x.gpkg/图层`` → (``…/x.gpkg``, 图层)；不是 gpkg 成员返回 None。"""
    text = str(path)
    lowered = text.lower().replace("\\", "/")
    index = lowered.find(".gpkg/")
    if index < 0:
        return None
    member = text[index + 6 :].strip("/\\")
    if not member:
        return None
    return Path(text[: index + 5]), member


def _quote_ident(name: str) -> str:
    """SQLite 标识符引用（名字来自库内 PRAGMA，不是用户输入；双写引号兜底）。"""
    return '"' + str(name).replace('"', '""') + '"'


def _read_gpkg(path: Path) -> dict[str, Any]:
    split = split_gpkg(path)
    if split is None:
        raise TableReadError("unsupported", "这是 .gpkg 文件本身，请选择其中的图层再打开属性表")
    container, layer = split
    if not container.is_file():
        raise TableReadError("not_found", "数据文件不存在")
    try:
        conn = sqlite3.connect(f"file:{container.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise TableReadError("unreadable", f"打不开 GeoPackage：{exc}") from exc
    conn.row_factory = sqlite3.Row
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT table_name FROM gpkg_contents").fetchall()
        }
        if layer not in tables:
            raise TableReadError(
                "not_found", f"GeoPackage 里没有图层 {layer}（可用：{', '.join(sorted(tables))[:200]}）"
            )
        try:
            geometry_column = conn.execute(
                "SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?",
                (layer,),
            ).fetchone()
        except sqlite3.Error:
            geometry_column = None
        skip = {geometry_column[0]} if geometry_column else set()
        columns = [
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({_quote_ident(layer)})").fetchall()
            if row["name"] not in skip
        ][:MAX_FIELDS]
        if not columns:
            raise TableReadError("unsupported", "该图层没有属性字段")
        select = ", ".join(_quote_ident(name) for name in columns)
        fetched = conn.execute(
            f"SELECT {select} FROM {_quote_ident(layer)} LIMIT ?", (MAX_CACHE_ROWS + 1,)
        ).fetchall()
    except sqlite3.Error as exc:
        raise TableReadError("unreadable", f"GeoPackage 读取失败：{exc}") from exc
    finally:
        conn.close()

    truncated = len(fetched) > MAX_CACHE_ROWS
    rows = [
        {name: _cell(row[name]) for name in columns} for row in fetched[:MAX_CACHE_ROWS]
    ]
    return {
        "fields": _fields_from_rows(columns, rows),
        "rows": rows,
        "truncated": truncated,
        "source": "gpkg",
        "notes": [],
    }


# --------------------------------------------------------------------- 内核
#: 内核取行模板。``__PATH__`` 等占位符一律用 ``json.dumps`` 注入，用户文本
#: 只能成为字符串字面量，进不了代码语法（这里也没有 where_clause）。
_KERNEL_TEMPLATE = r'''
import arcpy as _arcpy

_path = __PATH__
_limit = __LIMIT__
_max_fields = __MAX_FIELDS__


def _norm(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return "<binary %d bytes>" % len(value)
    try:
        return value.isoformat()
    except Exception:
        return str(value)


_error = None
_fields = []
_rows = []
_total = None
try:
    _fields = [
        {
            "name": f.name,
            "alias": getattr(f, "aliasName", "") or "",
            "type": f.type,
            "length": int(getattr(f, "length", 0) or 0),
        }
        for f in _arcpy.ListFields(_path)
        if f.type not in ("Geometry", "Blob", "Raster")
    ][:_max_fields]
    _names = [f["name"] for f in _fields]
    try:
        _total = int(_arcpy.management.GetCount(_path)[0])
    except Exception:
        _total = None
    with _arcpy.da.SearchCursor(_path, _names) as _cursor:
        for _row in _cursor:
            if len(_rows) > _limit:
                break
            _rows.append([_norm(value) for value in _row])
except Exception as _exc:
    _error = str(_exc)

set_result({"fields": _fields, "rows": _rows, "total": _total, "error": _error})
'''


def build_kernel_code(path: str, *, limit: int = MAX_CACHE_ROWS, max_fields: int = MAX_FIELDS) -> str:
    """生成内核取行代码（路径/上限都是字面量，不含任何用户拼接的语法）。"""
    return (
        _KERNEL_TEMPLATE.replace("__PATH__", json.dumps(str(path), ensure_ascii=False))
        .replace("__LIMIT__", str(int(limit)))
        .replace("__MAX_FIELDS__", str(int(max_fields)))
    )


def _read_kernel(
    path: Path, *, code_runner: Any, timeout: float
) -> dict[str, Any]:
    if code_runner is None:
        raise TableReadError(
            "kernel_unavailable",
            "该格式（.gdb 要素类 / .shp）需要 ArcGIS 内核，当前环境没有 arcpy",
        )
    code = build_kernel_code(str(path))
    try:
        outcome = code_runner.run(code, timeout=timeout)
    except Exception as exc:  # 内核进程崩了
        raise TableReadError("kernel_unavailable", f"ArcGIS 内核调用失败：{exc}") from exc
    if not getattr(outcome, "ok", False):
        error = getattr(outcome, "error", None)
        if isinstance(error, dict):
            reason = str(error.get("message") or error.get("type") or "")
            probe = f"{error.get('type', '')} {error.get('message', '')}".lower()
        else:
            reason = str(error or "")
            probe = reason.lower()
        code_name = (
            "kernel_timeout"
            if "timeout" in probe or "timed out" in probe or "exceeded" in probe
            else "kernel_unavailable"
        )
        raise TableReadError(code_name, f"ArcGIS 内核读表失败：{reason or '未知原因'}")
    payload = getattr(outcome, "result", None)
    if not isinstance(payload, dict):
        raise TableReadError("unreadable", "ArcGIS 内核没有返回属性表")
    if payload.get("error"):
        raise TableReadError("unreadable", f"内核读不出该图层：{payload['error']}")

    fields = [item for item in payload.get("fields") or [] if item.get("name")]
    names = [item["name"] for item in fields]
    raw_rows = payload.get("rows") or []
    truncated = len(raw_rows) > MAX_CACHE_ROWS
    rows = [
        {name: _cell(value) for name, value in zip(names, row)}
        for row in raw_rows[:MAX_CACHE_ROWS]
    ]
    notes = []
    total = payload.get("total")
    if isinstance(total, int) and total > len(rows):
        notes.append(f"表共 {total:,} 行，已缓存前 {len(rows):,} 行")
    return {
        "fields": fields[:MAX_FIELDS],
        "rows": rows,
        "truncated": truncated,
        "source": "arcpy",
        "notes": notes,
        "source_total": total if isinstance(total, int) else None,
    }


# --------------------------------------------------------------------- 入口
def load_table(
    path: Path,
    *,
    code_runner: Any = None,
    timeout: float = 60.0,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """读一个数据集的属性表 → ``{fields, rows, truncated, source, notes}``。

    失败抛 :class:`TableReadError`（带 ``code`` 与中文 ``reason``），绝不返回
    半截数据冒充完整表。``max_rows`` 只用于测试/调试，生产走 ``MAX_CACHE_ROWS``。
    """
    max_rows = MAX_CACHE_ROWS if max_rows is None else int(max_rows)
    suffix = path.suffix.lower()
    if suffix in DIRECT_JSON:
        result = _read_json(path)
    elif suffix in DIRECT_CSV:
        result = _read_csv(path)
    elif suffix == ".gpkg" or split_gpkg(path) is not None:
        result = _read_gpkg(path)
    elif suffix == ".kml":
        raise TableReadError(
            "unsupported", "KML 是 XML 结构，属性表暂不支持直读；请先转为 GeoJSON 或 GeoPackage"
        )
    elif suffix in {".xlsx", ".xls", ".dbf"}:
        raise TableReadError("unsupported", f"{suffix} 表格暂不支持服务端分页")
    elif suffix == ".shp" or ".gdb" in str(path).lower():
        result = _read_kernel(path, code_runner=code_runner, timeout=timeout)
    else:
        raise TableReadError("unsupported", f"该格式（{suffix or '未知'}）不支持属性表分页")
    if max_rows < len(result["rows"]):
        result["rows"] = result["rows"][:max_rows]
        result["truncated"] = True
    return result


__all__ = [
    "MAX_CACHE_ROWS",
    "MAX_FIELDS",
    "NUMERIC_TYPES",
    "TableReadError",
    "build_kernel_code",
    "load_table",
    "split_gpkg",
    "stat_key",
]

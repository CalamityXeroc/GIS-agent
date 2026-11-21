# -*- coding: utf-8 -*-
"""只读数据目录 API：数据集列表 / 画像 / 唯一值 / 属性表分页 / 栅格逐波段元数据。

数据来自引擎已经维护的 ``<workspace>/.gis_agent/catalog.db``，这里不触发扫描、
不写库（库不存在时返回空页，而不是新建一个空库）。

DAIR-328 新增的两组接口（「唯一值符号化」与「属性表分页」的后端）：

- ``GET /api/catalog/distinct``：字段唯一值 + 计数，可选分位断点；
- ``GET /api/catalog/rows``：服务端分页 / 排序 / 单条件筛选。

两者都不直接读数据文件：第一次请求把整表读进 catalog 的行缓存
（``table_cache_*`` 表，见 :mod:`gis_cli.datacatalog.tables`），之后的
分页/排序/筛选全是 **参数化 SQL**——字段名先过真实字段白名单，值一律走绑定
参数，运算符按白名单映射。用户输入永远进不了 SQL 语法。
"""

from __future__ import annotations

import bisect
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..datacatalog import bands as bands_meta
from ..datacatalog import tables
from ..datacatalog.store import Catalog, DatasetRecord, normalize_catalog_path
from ..datacatalog.tables import NUMERIC_TYPES, TableReadError
from .security import safe_path

#: catalog 库位置（相对 workspace），与 ``engine/runner.py`` 保持一致。
CATALOG_DB = Path(".gis_agent") / "catalog.db"

#: 单页上限：本地控制台不需要一次拉几千行。
MAX_LIMIT = 200

#: 属性表单页上限（DAIR-328 明确要求 500 封顶）。
MAX_ROWS_LIMIT = 500
#: 唯一值清单上限：唯一值符号化最多也就几十类，200 够用，1000 是安全阀。
MAX_DISTINCT_LIMIT = 1000
#: 分级符号化的级数范围。
MIN_CLASSES = 2
MAX_CLASSES = 12
#: 首次把整表读进缓存的超时（秒）；可用 ``GIS_TABLE_CACHE_TIMEOUT`` 覆盖。
DEFAULT_CACHE_TIMEOUT_S = 60.0
#: ``where`` 文本上限：单条件表达式不该长于这个。
MAX_WHERE_CHARS = 300

#: 允许 ``where`` 的运算符 → SQL 运算符。左列是用户写法，右列是**字面量**。
OPERATORS: tuple[tuple[str, str], ...] = (
    (">=", ">="),
    ("<=", "<="),
    ("!=", "<>"),
    ("<>", "<>"),
    ("≠", "<>"),
    ("=", "="),
    (">", ">"),
    ("<", "<"),
    ("LIKE", "LIKE"),
    ("like", "LIKE"),
)
#: 未加引号的值里不允许出现的字符：出现就当成「想混 SQL 片段」，直接 422。
#: （值本身是绑定参数，即便放行也注入不了；这里拒绝是为了让语义清晰可审计。）
FORBIDDEN_VALUE_CHARS = ("'", '"', ";", "`", "\\", "\n", "\r", "\x00")

#: 有属性表的数据集类型（栅格/工程/库目录不算）。
TABLE_KINDS = {"vector", "feature_class", "table"}


# ------------------------------------------------------------------ 基础查询
def _open_catalog(workspace: str | Path) -> Catalog | None:
    db_path = Path(workspace) / CATALOG_DB
    if not db_path.is_file():
        return None
    return Catalog(db_path)


def _brief(record: DatasetRecord) -> dict[str, Any]:
    """列表页字段：不带 fields/samples（那可能几百个字段）。"""
    return {
        "path": record.path,
        "kind": record.kind,
        "container": record.container,
        "name": record.name,
        "size": record.size,
        "mtime": record.mtime,
        "crs_name": record.crs_name,
        "wkid": record.wkid,
        "geom_type": record.geom_type,
        "feature_count": record.feature_count,
        "extent": record.extent,
        "field_count": len(record.fields),
        "source": record.source,
        "license": record.license,
        "acquired_at": record.acquired_at,
    }


def _full(record: DatasetRecord) -> dict[str, Any]:
    data = _brief(record)
    data["fields"] = record.fields
    data["raster"] = record.raster
    data["extra"] = record.extra
    return data


def _find_record(
    catalog: Catalog, workspace: str | Path, resolved: Path, raw_path: str
) -> DatasetRecord | None:
    """按 resolve 后的绝对路径找记录，找不到再按工作区相对写法 / 文件名回退。

    catalog 的路径口径是「相对 workspace」（DAIR-335），而调用方手里的入参
    可能是绝对路径（旧链接、抓包重放、浏览器缓存里的旧树）：三种写法都要能命中。
    """
    record = (
        catalog.get(str(resolved))
        or catalog.get(raw_path)
        or catalog.get(normalize_catalog_path(resolved, workspace))
    )
    if record is not None:
        return record
    for candidate in catalog.list_datasets(name_like=resolved.name, limit=50):
        try:
            # 候选记录的 path 是「相对 workspace」写法，不能拿进程 cwd 当基准。
            if _resolve(workspace, candidate.path) == resolved:
                return candidate
        except Exception:  # pragma: no cover - 坏路径 / 越界直接跳过
            continue
    return None


def _resolve(workspace: str | Path, path: str) -> Path:
    """入参 → workspace 内的绝对路径（先过 ``safe_path`` 的越界闸）。

    兼容 catalog 里「扫描时 cwd = 仓库根」留下的相对写法（``workspace/input/x``）：
    首段与工作区同名时去掉再试一次，边界仍是 workspace 内部。

    ``.gdb`` 内的要素类在文件系统上**不是文件**，所以不能拿 ``exists()`` 当判据
    （否则 ``workspace/input/x.gdb/fc`` 会被拼成 ``workspace/workspace/...``，
    DAIR-330 在真实 catalog 上踩到）：库目录在就算解析成功。
    """
    resolved = safe_path(workspace, path)
    if resolved.exists():
        return resolved
    raw = Path(path)
    if not raw.is_absolute() and len(raw.parts) > 1:
        head, *rest = raw.parts
        if head.lower() == Path(workspace).name.lower():
            candidate = Path(workspace).joinpath(*rest)
            container = candidate.parent
            if candidate.exists() or (
                container.is_dir() and container.suffix.lower() == ".gdb"
            ):
                return candidate
    return resolved


def _cache_timeout() -> float:
    import os

    try:
        value = float(os.environ.get("GIS_TABLE_CACHE_TIMEOUT", ""))
    except (TypeError, ValueError):
        return DEFAULT_CACHE_TIMEOUT_S
    return value if value > 0 else DEFAULT_CACHE_TIMEOUT_S


def _bad_request(code: str, reason: str, **extra: Any) -> HTTPException:
    return HTTPException(
        status_code=422, detail={"code": code, "reason": reason, **extra}
    )


# ------------------------------------------------------------------ where 解析
def parse_where(text: str, fields: list[dict[str, Any]]) -> tuple[str, str, Any] | None:
    """把 ``字段 op 值`` 解析成 ``(字段名, SQL 运算符, 绑定值)``；空串返回 None。

    只接受**单条件**（不做 AND/OR——那需要表达式解析器，且现在没需求）。
    字段名必须与真实字段完全一致（大小写敏感，与属性表显示一致）；运算符只
    认 :data:`OPERATORS` 里的写法；值按字段类型重写（数值字段走 ``float()``）。
    任何不合法都抛 422，绝不"尽力猜"。
    """
    text = (text or "").strip()
    if not text:
        return None
    if len(text) > MAX_WHERE_CHARS:
        raise _bad_request(
            "bad_where", f"where 过长（上限 {MAX_WHERE_CHARS} 字符，只支持单条件）"
        )

    matched: dict[str, Any] | None = None
    for field in sorted(fields, key=lambda item: len(str(item.get("name") or "")), reverse=True):
        name = str(field.get("name") or "")
        if not name or not text.startswith(name):
            continue
        rest = text[len(name) :].lstrip()
        if any(rest.startswith(token) for token, _ in OPERATORS):
            matched = field
            break
    if matched is None:
        raise _bad_request(
            "bad_where",
            "where 必须是「字段 运算符 值」，且字段必须在真实字段白名单里",
            allowed_fields=[str(f.get("name")) for f in fields][:60],
            allowed_operators=[token for token, _ in OPERATORS],
        )

    name = str(matched["name"])
    rest = text[len(name) :].lstrip()
    sql_op = ""
    for token, mapped in OPERATORS:
        if rest.startswith(token):
            sql_op = mapped
            rest = rest[len(token) :]
            break
    if not sql_op:  # pragma: no cover - 上面的白名单匹配已经保证了
        raise _bad_request("bad_where", "缺少运算符")

    raw_value = rest.strip()
    if not raw_value:
        raise _bad_request("bad_where", f"字段 {name} 后面缺少值")

    if raw_value[0] in "'\"":
        quote = raw_value[0]
        if len(raw_value) < 2 or raw_value[-1] != quote:
            raise _bad_request(
                "bad_where", "引号未闭合：值里不能出现未转义的引号或 SQL 片段"
            )
        value_text = raw_value[1:-1].replace(quote * 2, quote)
    else:
        for char in FORBIDDEN_VALUE_CHARS:
            if char in raw_value:
                shown = "换行" if char in "\n\r" else char
                raise _bad_request(
                    "bad_where", f"值里不允许出现「{shown}」（只接受字面量）"
                )
        value_text = raw_value

    field_type = str(matched.get("type") or "")
    # LIKE 是文本匹配：数值字段也按原文本比（``AREA LIKE 1%`` 有实际用途）。
    if field_type in NUMERIC_TYPES and sql_op != "LIKE":
        try:
            number = float(value_text)
        except ValueError:
            raise _bad_request(
                "bad_where", f"字段 {name} 是数值类型（{field_type}），值必须是数字"
            ) from None
        value: Any = int(number) if number.is_integer() and "." not in value_text else number
    else:
        value = value_text
    return name, sql_op, value


# ------------------------------------------------------------------ 分位断点
def quantile_breaks(values: list[float], classes: int) -> tuple[list[float], list[int]]:
    """等频（分位）断点 + 每级计数；``breaks`` 长度 ``classes+1``。

    线性插值取分位点，首尾用真实 min/max（前端分级符号化需要闭区间）。
    值重复导致断点重合时如实保留（零宽级别计数为 0），不做"美化"。
    """
    ordered = sorted(values)
    total = len(ordered)
    if total == 0:
        return [], []
    classes = max(MIN_CLASSES, min(int(classes), MAX_CLASSES))
    breaks = [ordered[0]]
    for index in range(1, classes):
        position = (total - 1) * index / classes
        low = int(position)
        high = min(low + 1, total - 1)
        frac = position - low
        breaks.append(ordered[low] + (ordered[high] - ordered[low]) * frac)
    breaks.append(ordered[-1])
    counts = [0] * classes
    for value in ordered:
        slot = bisect.bisect_right(breaks, value) - 1
        counts[min(max(slot, 0), classes - 1)] += 1
    return [round(value, 6) for value in breaks], counts


# ------------------------------------------------------------------ 行缓存
def _ensure_cache(
    catalog: Catalog,
    resolved: Path,
    *,
    code_runner: Any,
    refresh: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """确保行缓存是最新的；返回 ``(缓存元信息, 本次构建信息)``。"""
    mtime, size = tables.stat_key(resolved)
    meta = catalog.table_meta(str(resolved))
    if meta and not refresh and abs(float(meta["mtime"]) - mtime) < 1e-6 and int(
        meta["size"]
    ) == int(size):
        return meta, {"hit": True, "build_ms": 0.0}

    started = time.perf_counter()
    try:
        payload = tables.load_table(
            resolved, code_runner=code_runner, timeout=_cache_timeout()
        )
    except TableReadError as exc:
        raise _map_read_error(exc, resolved) from exc

    if not payload.get("fields"):
        raise _bad_request(
            "no_fields", "该数据集没有可展示的属性字段（空表或只有几何字段）"
        )
    catalog.replace_table_cache(
        str(resolved),
        mtime=mtime,
        size=size,
        fields=payload["fields"],
        rows=payload["rows"],
        truncated=bool(payload.get("truncated")),
        source=str(payload.get("source") or ""),
        notes=list(payload.get("notes") or []),
    )
    meta = catalog.table_meta(str(resolved)) or {}
    return meta, {
        "hit": False,
        "build_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def _map_read_error(exc: TableReadError, resolved: Path) -> HTTPException:
    if exc.code == "kernel_unavailable":
        return HTTPException(
            status_code=503,
            detail={
                "code": "kernel_unavailable",
                "reason": exc.reason,
                "suggestion": "装 ArcGIS Pro 后重试，或把图层导出为 GeoJSON / GeoPackage 再打开属性表",
                "path": str(resolved),
            },
        )
    if exc.code == "kernel_timeout":
        return HTTPException(
            status_code=504,
            detail={
                "code": "table_read_timeout",
                "reason": exc.reason,
                "suggestion": "表太大或内核忙；可先用 .geojson 预览，或等内核恢复后重试",
                "path": str(resolved),
            },
        )
    if exc.code == "not_found":
        return HTTPException(status_code=404, detail=exc.reason)
    return HTTPException(
        status_code=422, detail={"code": exc.code, "reason": exc.reason, "path": str(resolved)}
    )


def _table_record(
    workspace: str | Path, path: str, *, kind_error: str
) -> tuple[Catalog, Path, DatasetRecord]:
    """取有属性表的数据集记录（打开 catalog 由调用方 close）。"""
    catalog = _open_catalog(workspace)
    if catalog is None:
        raise HTTPException(status_code=404, detail="catalog not found")
    try:
        resolved = _resolve(workspace, path)
        record = _find_record(catalog, workspace, resolved, path)
    except Exception:
        catalog.close()
        raise
    if record is None:
        catalog.close()
        raise HTTPException(status_code=404, detail="dataset not found")
    if record.kind not in TABLE_KINDS:
        catalog.close()
        raise _bad_request(
            "not_tabular",
            kind_error,
            kind=record.kind,
        )
    return catalog, resolved, record


# ------------------------------------------------------------------ 公共逻辑
def dataset_rows(
    workspace: str | Path,
    path: str,
    *,
    offset: int = 0,
    limit: int = 100,
    order: str = "",
    direction: str = "asc",
    where: str = "",
    code_runner: Any = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """属性表服务端分页（参数、上限与失败语义见路由 docstring）。"""
    started = time.perf_counter()
    catalog, resolved, record = _table_record(
        workspace,
        path,
        kind_error="该数据集不是矢量/表格，没有属性表（栅格请用 /api/catalog/raster/bands）",
    )
    try:
        meta, build = _ensure_cache(
            catalog, resolved, code_runner=code_runner, refresh=refresh
        )
        fields = list(meta.get("fields") or [])
        parsed = parse_where(where, fields)
        direction_norm = "desc" if str(direction).lower() == "desc" else "asc"
        order_name = (order or "").strip()
        if order_name:
            allowed = {str(field.get("name")) for field in fields}
            if order_name not in allowed:
                raise _bad_request(
                    "bad_order",
                    f"排序字段 {order_name} 不在真实字段白名单里",
                    allowed_fields=sorted(allowed)[:60],
                )
        rows, total = catalog.query_rows(
            str(resolved),
            offset=offset,
            limit=limit,
            order=order_name,
            direction=direction_norm,
            where=parsed,
        )
    finally:
        catalog.close()

    return {
        "path": str(resolved),
        "kind": record.kind,
        "fields": fields,
        "rows": rows,
        "offset": offset,
        "limit": limit,
        "total": total,
        "returned": len(rows),
        "truncated": total > offset + len(rows),
        "order": order_name,
        "dir": direction_norm,
        "where": (where or "").strip(),
        "cache": {
            "hit": build["hit"],
            "build_ms": build["build_ms"],
            "rows": int(meta.get("row_count") or 0),
            "truncated": bool(meta.get("truncated")),
            "source": meta.get("source") or "",
            "notes": list(meta.get("notes") or []),
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def dataset_distinct(
    workspace: str | Path,
    path: str,
    *,
    field: str,
    limit: int = 200,
    breaks: str = "",
    classes: int = 5,
    code_runner: Any = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """字段唯一值清单（可选分位断点）。参数与失败语义见路由 docstring。"""
    started = time.perf_counter()
    catalog, resolved, record = _table_record(
        workspace,
        path,
        kind_error="该数据集不是矢量/表格，没有字段唯一值（栅格请用逐波段元数据）",
    )
    try:
        meta, build = _ensure_cache(
            catalog, resolved, code_runner=code_runner, refresh=refresh
        )
        fields = list(meta.get("fields") or [])
        by_name = {str(item.get("name")): item for item in fields}
        if field not in by_name:
            raise _bad_request(
                "bad_field",
                f"字段 {field} 不在真实字段白名单里",
                allowed_fields=list(by_name)[:60],
            )
        summary = catalog.distinct_values(str(resolved), field, limit=limit)
        wants_breaks = str(breaks).strip().lower() == "quantile"
        payload: dict[str, Any] = {
            "path": str(resolved),
            "kind": record.kind,
            "field": field,
            "field_type": by_name[field].get("type") or "",
            **summary,
        }
        if wants_breaks:
            values = catalog.numeric_values(str(resolved), field)
            if values:
                break_points, counts = quantile_breaks(values, classes)
                payload["breaks"] = break_points
                payload["counts"] = counts
                payload["classes"] = len(counts)
                payload["break_basis"] = {
                    "method": "quantile",
                    "values": len(values),
                    "sampled": len(values) < int(meta.get("row_count") or 0),
                }
            else:
                payload["breaks"] = []
                payload["counts"] = []
                payload["classes"] = 0
                payload["break_basis"] = {
                    "method": "quantile",
                    "values": 0,
                    "reason": "该字段没有数值（无法分级）",
                }
    finally:
        catalog.close()

    payload["cache"] = {
        "hit": build["hit"],
        "build_ms": build["build_ms"],
        "rows": int(meta.get("row_count") or 0),
        "truncated": bool(meta.get("truncated")),
        "source": meta.get("source") or "",
    }
    payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return payload


def raster_bands(
    workspace: str | Path,
    path: str,
    *,
    refresh: bool = False,
    code_runner: Any = None,
    budget_s: float | None = None,
) -> dict[str, Any]:
    """栅格逐波段元数据（缓存优先；读不到给 unavailable 原因）。"""
    catalog = _open_catalog(workspace)
    if catalog is None:
        raise HTTPException(status_code=404, detail="catalog not found")
    try:
        resolved = _resolve(workspace, path)
        record = _find_record(catalog, workspace, resolved, path)
    finally:
        catalog.close()
    if record is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    if record.kind != "raster":
        raise _bad_request(
            "not_raster", "该数据集不是栅格", kind=record.kind
        )
    raster = record.raster or {}
    result = bands_meta.read_bands(
        workspace,
        resolved,
        code_runner=code_runner,
        budget_s=budget_s,
        refresh=refresh,
        fallback_band_count=raster.get("band_count"),
    )
    result["path"] = str(resolved)
    result["kind"] = record.kind
    # catalog 侧已有的整栅格汇总（扫描时算过），逐波段缺失时前端仍能显示口径。
    result["raster"] = raster
    return result


# ------------------------------------------------------------------ 路由
def build_router(workspace: str | Path, *, runtime: Any = None) -> APIRouter:
    """装配 catalog 路由。

    ``runtime``：预览/内核持有者（``PreviewRuntime``，鸭子类型：只要有 ``get()``）。
    没传时所有需要 ArcPy 的请求走"明确降级"分支（503 + 原因），不会崩。
    """

    def runner() -> Any:
        if runtime is None:
            return None
        try:
            return runtime.get()
        except Exception:  # pragma: no cover - 内核起不来时降级
            return None

    router = APIRouter()

    @router.get("/api/catalog/datasets")
    def get_datasets(
        kind: str = Query("", description="数据集类型，精确匹配"),
        q: str = Query("", description="按名称或路径模糊搜索"),
        limit: int = Query(50, ge=1, le=MAX_LIMIT),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        return list_datasets(workspace, kind=kind, q=q, limit=limit, offset=offset)

    @router.get("/api/catalog/datasets/profile")
    def get_profile(path: str = Query(..., description="数据集路径（必须在 workspace 内）")):
        return dataset_profile(workspace, path)

    @router.get("/api/catalog/distinct")
    def get_distinct(
        path: str = Query(..., description="数据集路径（必须在 workspace 内）"),
        field: str = Query(..., description="字段名（必须在真实字段白名单里）"),
        limit: int = Query(
            200, ge=1, le=MAX_DISTINCT_LIMIT, description="返回的唯一值个数上限"
        ),
        breaks: str = Query(
            "", description="传 quantile 时额外返回分位断点（breaks/counts）"
        ),
        classes: int = Query(
            5, ge=MIN_CLASSES, le=MAX_CLASSES, description="分位分级数（breaks=quantile 时生效）"
        ),
        refresh: bool = Query(False, description="忽略行缓存，重读数据文件"),
    ) -> dict[str, Any]:
        """字段唯一值清单（唯一值符号化的数据源）。

        - ``values`` 按出现次数降序，``{value, count}``；``total_distinct`` 是
          **真实基数**（不是本页条数），``truncated`` 表示被 ``limit`` 截断；
        - ``null_count`` 是空值行数（符号化时单列一类）；
        - ``breaks=quantile`` 时加 ``breaks``（长度 ``classes+1``，首尾是真实
          min/max）与 ``counts``（每级要素数）；字段没有数值时返回空数组 + 原因；
        - 失败语义：403 越界 / 404 无此数据集 / 422 字段不在白名单 /
          503 无 arcpy 且需要内核 / 504 首次读表超时。
        """
        return dataset_distinct(
            workspace,
            path,
            field=field,
            limit=limit,
            breaks=breaks,
            classes=classes,
            code_runner=runner(),
            refresh=refresh,
        )

    @router.get("/api/catalog/rows")
    def get_rows(
        path: str = Query(..., description="数据集路径（必须在 workspace 内）"),
        offset: int = Query(0, ge=0, description="起始行号（从 0 开始）"),
        limit: int = Query(100, ge=1, le=MAX_ROWS_LIMIT, description="单页行数，上限 500"),
        order: str = Query("", description="排序字段（必须在真实字段白名单里）"),
        dir: str = Query("asc", pattern="^(asc|desc)$", description="排序方向 asc/desc"),
        where: str = Query(
            "", description="单条件筛选：字段 op 值（op ∈ =,!=,<>,>,<,>=,<=,LIKE）"
        ),
        refresh: bool = Query(False, description="忽略行缓存，重读数据文件"),
    ) -> dict[str, Any]:
        """属性表服务端分页 / 排序 / 筛选。

        - ``rows`` 按 ``fields`` 顺序给出 ``{字段: 值}``；``total`` 是**筛选后**
          的总行数（不是本页条数），``truncated`` 表示后面还有页；
        - ``cache.truncated`` 为真表示表大于行缓存上限（默认 5 万行），此时
          ``total`` 只覆盖已缓存部分——前端要如实标注，不许写"共 N 行"；
        - ``where`` 只支持单条件，字段必须命中真实字段白名单；值走绑定参数，
          引号/分号等一律拒绝（422 ``bad_where``）；
        - ``limit`` 上限 500，``offset``/``limit`` 越界由 FastAPI 校验成 422；
        - 失败语义：403 越界 / 404 无此数据集 / 422 非矢量或参数不合法 /
          503 无 arcpy 且需要内核 / 504 首次读表超时。
        """
        return dataset_rows(
            workspace,
            path,
            offset=offset,
            limit=limit,
            order=order,
            direction=dir,
            where=where,
            code_runner=runner(),
            refresh=refresh,
        )

    @router.get("/api/catalog/raster/bands")
    def get_raster_bands(
        path: str = Query(..., description="栅格路径（必须在 workspace 内）"),
        refresh: bool = Query(False, description="忽略逐波段缓存，重读栅格"),
    ) -> dict[str, Any]:
        """栅格逐波段元数据（dtype / min / max / mean / nodata / 尺寸 / 色彩解释）。

        - 命中 ``catalog.db`` 的 ``raster_bands`` 缓存时 0 次内核调用；
        - 未命中时在预算内（默认 1.4s，``GIS_BAND_BUDGET`` 可调）读一次，
          超预算或没算统计值时对应波段给 ``unavailable`` **原因**，不编数值；
        - 无 arcpy 时返回 200 + ``unavailable``（不崩、不 500）；
        - 失败语义：403 越界 / 404 无此数据集 / 422 不是栅格。
        """
        return raster_bands(
            workspace,
            path,
            refresh=refresh,
            code_runner=runner(),
            budget_s=None,
        )

    return router


def list_datasets(
    workspace: str | Path,
    *,
    kind: str = "",
    q: str = "",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """分页 + kind 精确过滤 + q 模糊匹配（名称或路径）。"""
    limit = max(1, min(int(limit), MAX_LIMIT))
    offset = max(0, int(offset))

    # 多取一行用来判断还有没有下一页，避免额外的 COUNT 查询。
    fetch = offset + limit + 1
    catalog = _open_catalog(workspace)
    if catalog is None:
        return {
            "items": [],
            "count": 0,
            "limit": limit,
            "offset": offset,
            "has_more": False,
            "counts": {},
        }

    try:
        records = catalog.list_datasets(
            kind=kind or None,
            name_like=q or None,
            limit=fetch,
        )
        counts = catalog.counts()
    finally:
        catalog.close()

    page = records[offset : offset + limit]
    return {
        "items": [_brief(record) for record in page],
        "count": len(page),
        "limit": limit,
        "offset": offset,
        "has_more": len(records) > offset + limit,
        "counts": counts,
    }


def dataset_profile(workspace: str | Path, path: str) -> dict[str, Any]:
    """单个数据集的字段画像（只读 catalog，不打开数据文件）。

    解析走 ``_resolve``（与 rows / bands 一致）：老写法 ``workspace/input/x``
    在这里也能命中，而不是只有属性表能打开、元数据 404（DAIR-336 审查指出）。
    """
    resolved = _resolve(workspace, path)
    catalog = _open_catalog(workspace)
    if catalog is None:
        raise HTTPException(status_code=404, detail="catalog not found")

    try:
        record = _find_record(catalog, workspace, resolved, path)
    finally:
        catalog.close()

    if record is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    return _full(record)

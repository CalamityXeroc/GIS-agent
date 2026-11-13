# -*- coding: utf-8 -*-
"""数据资产预览 API：字段画像 / 离线图层预览 / 成果包沙箱预览。

三条边界（见 ``docs/p2-console-plan.md`` §6、§9）：

- **只读**：不写 catalog、不跑分析、不生成成果包；
- **不越界**：任何路径先 ``safe_path()`` 落到 workspace 内，报表还额外限定在
  ``output/`` 下、只回 ``text/html``；
- **不隐式投影**：坐标系不是 4326/3857 时照原样画，但响应里必带
  ``crs_warning``，让页面显式告诉用户「位置只是相对形状，不是真实经纬度」。

几何读取复用 ``report/html_pack.py``：``.geojson`` 直读，GDB/shp 走内核一次
``arcpy.da.SearchCursor`` 取 Esri JSON，再转 GeoJSON、``thin_features()`` 抽稀。

两类路径分开判存在性（DAIR-248）：文件型（``.geojson/.json/.shp/.gpkg/.kml``）
看 ``is_file()``；``*.gdb/数据集/要素类`` 在文件系统上**不是文件**，只能看
「父级库目录在不在 + 成员名非空」，否则 catalog 里所有 .gdb 要素类都会被误判
成「数据文件不存在」。几何读取上限 :data:`DEFAULT_READ_TIMEOUT_S`（10s，可用
``GIS_PREVIEW_READ_TIMEOUT`` 覆盖），超时 422 + 原因 + 建议 + ``elapsed_ms``；
内核本身跑不起来（arcpy 挂死/未安装）则 503 + 明确原因。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from ..datacatalog.store import Catalog, DatasetRecord
from ..report import html_pack
from . import catalog_api
from .security import safe_path

logger = logging.getLogger(__name__)

#: 预览响应硬上限（计划 §6）：超过就继续抽稀，绝不把几十 MB 的几何塞给浏览器。
MAX_PREVIEW_BYTES = 5 * 1024 * 1024

#: 初始要素上限。与 ``html_pack.thin_features`` 的默认值保持一致。
DEFAULT_MAX_FEATURES = html_pack.MAX_FEATURES
#: 调用方最多能要多少要素（再大也还是会被 5MB 上限压回来）。
MAX_MAX_FEATURES = 5000

#: 预览缓存目录（相对 workspace）。
CACHE_DIR = Path(".gis_agent") / "preview_cache"

#: 成果包只认这两个文件名（``report/`` 的产物契约）。
REPORT_NAMES = ("成果包.html", "溯源卡.html")
#: 报表根目录（相对 workspace）。
REPORT_ROOT = "output"

#: 可以直接按经纬度/Web 墨卡托理解的数据；其余一律要告警。
KNOWN_CRS = {4326, 3857}

#: 字段类型里能算 min/max 的那些（arcpy Field.type 的写法）。
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
}

CRS_WARNING = "未做投影变换"
CRS_UNKNOWN_WARNING = "坐标系未知，未做投影变换"

#: 能直接读的格式；其余走 ArcGIS 内核。
DIRECT_SUFFIXES = {".geojson", ".json"}

#: 几何读取上限（秒）：超过就 422 + 原因 + 建议，绝不让浏览器干等。
#: 可用环境变量 ``GIS_PREVIEW_READ_TIMEOUT`` 覆盖（运维可调，不必改代码）。
DEFAULT_READ_TIMEOUT_S = 10.0

#: 内核冷启动是**一次性**成本（实测 ``import arcpy`` 10s / 内核就绪 8.3s），
#: 单独记账，不占几何读取的 10s 预算——否则第一个 .gdb 预览必然超时。
KERNEL_BOOT_TIMEOUT_S = 25.0

#: 路径形态：普通文件 / .gdb 目录本身 / .gdb 内的要素类 / 不存在。
PATH_FILE = "file"
PATH_GDB = "gdb"
PATH_GDB_FEATURE_CLASS = "gdb_feature_class"
PATH_MISSING = "missing"

#: 允许离线预览的 catalog kind（.gdb 内的要素类在 catalog 里记作 feature_class）。
PREVIEWABLE_KINDS = {"vector", "feature_class"}

TIMEOUT_SUGGESTION = "要素类太大或内核不可用；可先用 .geojson 预览，或等内核恢复后重试"

#: 报表页面的 CSP：脚本/样式可以跑（成果包是内联 canvas + 内联 CSS），但拿不到
#: 同源 cookie、不能发请求、不能读父页面。iframe 的 ``sandbox`` 是第二道闸。
REPORT_CSP = (
    "sandbox allow-scripts; default-src 'none'; "
    "style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
    "img-src data: blob:; font-src data:"
)


# ------------------------------------------------------------------ 字段画像
def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _field_stats(field: dict[str, Any]) -> dict[str, Any]:
    """在 catalog 已有的 ``samples`` 上算 min/max，**不重扫数据表**。

    catalog 抽样本来就只有几十个值（扫描时 ``sample_values`` 限制），所以这里给
    的是「样本范围」，前端标注为「按样本」。要看精确范围得另开单做全表统计，
    那正是本单明确不做的。
    """
    stats: dict[str, Any] = {"numeric": False, "sample_count": 0}
    samples = field.get("samples") or []
    stats["sample_count"] = len(samples)
    ftype = str(field.get("type") or "")
    numbers = [n for n in (_numeric(v) for v in samples) if n is not None]
    if ftype in NUMERIC_TYPES and numbers:
        stats.update(
            {
                "numeric": True,
                "min": min(numbers),
                "max": max(numbers),
                "range_basis": "samples",
            }
        )
    return stats


def profile_dataset(workspace: str | Path, path: str) -> dict[str, Any]:
    """catalog 里那条记录的完整画像 + 逐字段样本统计。"""
    data = catalog_api.dataset_profile(workspace, path)
    fields = []
    for field in data.get("fields") or []:
        item = dict(field)
        item.update(_field_stats(field))
        fields.append(item)
    data["fields"] = fields
    data["field_count"] = len(fields)
    # 数值列清单：前端在字段表上方做一句摘要，不用自己再过滤一遍。
    data["numeric_fields"] = [f["name"] for f in fields if f.get("numeric")]
    return data


# ------------------------------------------------------------------ 路径分类
def _gdb_container(path: Path) -> tuple[Path, str] | None:
    """``…/x.gdb/数据集/要素类`` → (``…/x.gdb``, ``数据集/要素类``)；不在库里返回 None。

    ``.gdb`` 内的要素类在文件系统上**不是文件**：``exists()/is_file()`` 一律 False。
    判断存在性只能看「父级库目录在不在 + 成员名非空」（DAIR-243 实测：catalog 里
    200 条全是这种，用 ``is_file()`` 判存在会让预览对所有真实数据返回 404）。
    """
    text = str(path)
    lowered = text.lower().replace("\\", "/")
    index = lowered.find(".gdb/")
    if index < 0:
        return None
    member = text[index + 5 :].strip("/\\")
    if not member:
        return None
    return Path(text[: index + 4]), member


def _path_kind(path: Path) -> str:
    """file / gdb（库目录本身）/ gdb_feature_class（库内成员）/ missing。"""
    container = _gdb_container(path)
    if container is not None:
        return PATH_GDB_FEATURE_CLASS if container[0].is_dir() else PATH_MISSING
    if path.suffix.lower() == ".gdb":
        return PATH_GDB if path.is_dir() else PATH_MISSING
    return PATH_FILE if path.is_file() else PATH_MISSING


def _resolve_target(workspace: Path, path: str) -> Path:
    """入参 → workspace 内的路径（先过 ``safe_path`` 的越界闸）。

    兼容 catalog 里「扫描时 cwd = 仓库根」留下的相对写法（``workspace/input/x``）：
    首段与工作区同名时去掉它再试一次。结果仍在 workspace 内，越界边界不变。
    """
    resolved = safe_path(workspace, path)
    if _path_kind(resolved) != PATH_MISSING:
        return resolved
    raw = Path(path)
    if not raw.is_absolute() and len(raw.parts) > 1:
        head, *rest = raw.parts
        if head.lower() == workspace.name.lower():
            candidate = workspace.joinpath(*rest)
            if _path_kind(candidate) != PATH_MISSING:
                return candidate
    return resolved


# ------------------------------------------------------------------ 几何读取
def _stat_key(path: Path) -> tuple[float, int]:
    """缓存键用的 (mtime, size)。

    GDB 里的要素类在文件系统上不存在，退回「容器内容时间」（忽略 ``*.sr.lock``，
    否则读一次就把缓存全废掉，见 ``scanner.gdb_content_mtime``）。
    """
    try:
        stat = path.stat()
        return stat.st_mtime, stat.st_size
    except OSError:
        pass
    container = _gdb_container(path)
    if container is not None:
        from ..datacatalog.scanner import gdb_content_mtime

        try:
            return gdb_content_mtime(container[0]), 0
        except OSError:  # pragma: no cover - 目录都读不到
            return 0.0, 0
    return 0.0, 0


def _crs_warning(wkid: int | None) -> str | None:
    if wkid in KNOWN_CRS:
        return None
    if wkid:
        return CRS_WARNING
    return CRS_UNKNOWN_WARNING


def _cache_key(path: Path, max_features: int) -> tuple[str, float, int]:
    mtime, size = _stat_key(path)
    digest = hashlib.sha1(
        f"{path}|{mtime}|{size}|{max_features}".encode("utf-8", errors="replace")
    ).hexdigest()
    return digest, mtime, size


def _cache_path(workspace: Path, digest: str) -> Path:
    return workspace / CACHE_DIR / f"{digest}.json"


def _encoded_size(payload: Any) -> int:
    return len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _read_cached(cache_file: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    try:
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    # mtime/size 都在键里，这里再核一次是给「键算对了但文件被换掉」兜底。
    for key in ("mtime", "size"):
        if raw.get(key) != expected.get(key):
            return None
    payload = raw.get("payload")
    return payload if isinstance(payload, dict) else None


def _write_cached(cache_file: Path, envelope: dict[str, Any]) -> None:
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(envelope, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    except OSError as exc:  # 缓存写不进去只影响下次速度，不影响本次响应
        logger.debug("预览缓存写入失败 %s: %s", cache_file, exc)


def _empty_geojson() -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": []}


def _thin_under_limit(
    collection: dict[str, Any],
    *,
    max_features: int,
    base: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """抽稀到响应 ≤5MB：超了就减半重抽，最多 8 轮。

    返回 ``(最终 payload, 抽稀后的 collection, 实际 max_features)``。
    """
    current = max(1, int(max_features))
    for _ in range(8):
        thinned = html_pack.thin_features(collection, max_features=current)
        payload = dict(base)
        payload["geojson"] = thinned
        payload["returned_features"] = len(thinned.get("features") or [])
        payload["point_count"] = html_pack.count_points(thinned)
        payload["max_features"] = current
        payload["truncated"] = bool(
            payload.get("feature_count")
            and payload["feature_count"] > payload["returned_features"]
        )
        if _encoded_size(payload) <= MAX_PREVIEW_BYTES:
            return payload, thinned, current
        current = max(1, current // 2)
    # 极端情况：单要素的坐标就超 5MB（比如一个上百万点的多边形）。宁可只回
    # bbox 也不回一个打不开的响应；前端有 bbox 仍能说明范围。
    thinned = _empty_geojson()
    payload = dict(base)
    payload.update(
        {
            "geojson": thinned,
            "returned_features": 0,
            "point_count": 0,
            "max_features": 0,
            "truncated": True,
            "oversized": True,
        }
    )
    return payload, thinned, 0


def _dataset_record(
    workspace: str | Path, resolved: Path, raw_path: str
) -> DatasetRecord | None:
    catalog: Catalog | None = catalog_api._open_catalog(workspace)
    if catalog is None:
        return None
    try:
        record = catalog.get(str(resolved)) or catalog.get(raw_path)
        if record is None:
            for candidate in catalog.list_datasets(name_like=resolved.name, limit=50):
                if Path(candidate.path).resolve() == resolved:
                    record = candidate
                    break
        return record
    finally:
        catalog.close()


class PreviewTimeout(RuntimeError):
    """几何读取超时。不是 HTTPException：调用方要带上 elapsed_ms 再组装响应。"""

    def __init__(self, elapsed_ms: float, timeout_s: float) -> None:
        super().__init__(f"几何读取超过 {timeout_s:g}s")
        self.elapsed_ms = elapsed_ms
        self.timeout_s = timeout_s


def _read_timeout() -> float:
    """几何读取上限（秒）；``GIS_PREVIEW_READ_TIMEOUT`` 可覆盖。"""
    try:
        value = float(os.environ.get("GIS_PREVIEW_READ_TIMEOUT", ""))
    except (TypeError, ValueError):
        return DEFAULT_READ_TIMEOUT_S
    return value if value > 0 else DEFAULT_READ_TIMEOUT_S


def _timeout_detail(elapsed_ms: float, timeout_s: float) -> dict[str, Any]:
    return {
        "code": "preview_timeout",
        "reason": f"读取几何超时：超过 {timeout_s:g}s 仍未读回",
        "suggestion": TIMEOUT_SUGGESTION,
        "elapsed_ms": elapsed_ms,
        "timeout_s": timeout_s,
    }


def _kernel_detail(reason: str, elapsed_ms: float) -> dict[str, Any]:
    return {
        "code": "kernel_unavailable",
        "reason": f"ArcGIS 内核不可用：{reason}" if reason else "ArcGIS 内核不可用",
        "suggestion": "内核恢复前可先用 .geojson 预览；或稍后重试",
        "elapsed_ms": elapsed_ms,
    }


def _looks_like_timeout(error: Any) -> bool:
    if isinstance(error, dict):
        text = f"{error.get('type', '')} {error.get('message', '')}"
    else:
        text = str(error or "")
    lowered = text.lower()
    return "timeout" in lowered or "timed out" in lowered


def _wait_kernel_ready(code_runner: Any) -> None:
    """先等内核预热（一次性成本），再把 10s 预算交给几何读取。

    不分开记账的话，第一个 .gdb 预览必然 422：``import arcpy`` 就要 10s，正好
    吃掉整个读取预算。测试注入的假 runner 没有 prewarm/wait_prewarm，直接跳过。
    """
    prewarm = getattr(code_runner, "prewarm", None)
    if callable(prewarm):
        try:
            prewarm()
        except Exception as exc:  # pragma: no cover - 预热失败留给读取阶段报错
            logger.debug("预览内核预热失败: %s", exc)
    wait = getattr(code_runner, "wait_prewarm", None)
    if callable(wait):
        try:
            wait(KERNEL_BOOT_TIMEOUT_S)
        except Exception as exc:  # pragma: no cover
            logger.debug("预览内核预热等待失败: %s", exc)


def _load_via_kernel(
    resolved: Path,
    *,
    code_runner: Any,
    limit: int,
    read_timeout: float,
) -> dict[str, Any]:
    """.gdb 要素类 / .shp 走内核：一次 ``arcpy.da.SearchCursor`` 取 Esri JSON。

    与 ``html_pack.load_layer`` 的内核分支同一段代码模板（``layer_probe_code``），
    差别只在超时：报告导出愿意等 180s，预览只给 ``read_timeout``（10s），
    由 runner 真中止内核调用，而不是只把 HTTP 请求断掉。
    """
    started = time.perf_counter()

    def elapsed() -> float:
        return round((time.perf_counter() - started) * 1000, 2)

    if code_runner is None:
        raise HTTPException(
            status_code=503,
            detail=_kernel_detail(
                "未检测到 arcpy（当前环境没有 ArcGIS Pro Python）", elapsed()
            ),
        )
    code = html_pack.layer_probe_code(str(resolved), limit=limit)
    outcome = code_runner.run(code, timeout=read_timeout)
    if not getattr(outcome, "ok", False):
        error = getattr(outcome, "error", None)
        if _looks_like_timeout(error):
            raise PreviewTimeout(elapsed(), read_timeout)
        reason = (
            str(error.get("message") or error.get("type") or "")
            if isinstance(error, dict)
            else str(error or "")
        )
        raise HTTPException(status_code=503, detail=_kernel_detail(reason, elapsed()))
    payload = getattr(outcome, "result", None)
    if isinstance(payload, dict) and payload.get("error"):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "dataset_unreadable",
                "reason": f"内核读不出该图层：{payload['error']}",
                "suggestion": "确认要素类名与 .gdb 是否对应；可先用 .geojson 预览",
                "elapsed_ms": elapsed(),
            },
        )
    geoms = payload.get("geoms") if isinstance(payload, dict) else None
    return html_pack.esri_geometries_to_feature_collection(geoms or [])


def preview_layer(
    workspace: str | Path,
    path: str,
    *,
    max_features: int = DEFAULT_MAX_FEATURES,
    code_runner: Any = None,
    read_timeout: float | None = None,
) -> dict[str, Any]:
    """读一个矢量图层 → GeoJSON → 抽稀 → 落盘缓存（键 = path + mtime + 上限）。"""
    ws = Path(workspace).resolve()
    resolved = _resolve_target(ws, path)
    max_features = max(1, min(int(max_features), MAX_MAX_FEATURES))
    timeout = float(read_timeout or _read_timeout())

    target_kind = _path_kind(resolved)
    if target_kind == PATH_GDB:
        raise HTTPException(
            status_code=422,
            detail="这是文件数据库（.gdb）目录，请选择其中的要素类再预览",
        )
    if target_kind == PATH_MISSING:
        raise HTTPException(status_code=404, detail="数据文件不存在")

    record = _dataset_record(ws, resolved, path)
    if record is not None and record.kind and record.kind not in PREVIEWABLE_KINDS:
        raise HTTPException(
            status_code=422,
            detail=f"该类型（{record.kind}）不支持离线图层预览，仅支持矢量",
        )

    started = time.perf_counter()
    digest, mtime, size = _cache_key(resolved, max_features)
    cache_file = _cache_path(ws, digest)
    cached = _read_cached(cache_file, {"mtime": mtime, "size": size})
    if cached is not None:
        cached["cache"] = {
            "hit": True,
            "key": digest[:12],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        return cached

    direct = resolved.suffix.lower() in DIRECT_SUFFIXES
    if not direct:
        # 内核冷启动是一次性成本，先等它（不计入 10s 几何读取预算）；
        # 内核根本不可用时由 _load_via_kernel 转成 503 + 原因。
        _wait_kernel_ready(code_runner)
    # read_ms 只算几何读取本身：不含内核冷启动/预热，否则冷启动的十几秒会
    # 被读成「读一次几何要 12s」，看不出真实瓶颈。
    read_started = time.perf_counter()
    try:
        if direct:
            collection = _call_with_timeout(
                lambda: html_pack.load_layer(
                    resolved, code_runner=None, limit=max_features, workspace=ws
                ),
                timeout,
            )
        else:
            collection = _call_with_timeout(
                lambda: _load_via_kernel(
                    resolved,
                    code_runner=code_runner,
                    limit=max_features,
                    read_timeout=timeout,
                ),
                timeout,
            )
    except PreviewTimeout as exc:
        raise HTTPException(
            status_code=422, detail=_timeout_detail(exc.elapsed_ms, exc.timeout_s)
        ) from exc
    except HTTPException:
        raise
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"无权限读取该文件：{exc}") from exc
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"读取失败：{exc}") from exc
    except Exception as exc:  # pragma: no cover - 内核异常兜底
        logger.warning("图层读取异常 %s: %s", resolved, exc)
        raise HTTPException(status_code=422, detail=f"读取失败：{exc}") from exc

    read_ms = round((time.perf_counter() - read_started) * 1000, 2)
    if not collection or not collection.get("features"):
        raise HTTPException(
            status_code=422, detail="图层没有可预览的要素（空图层或几何读取失败）"
        )

    bbox = (record.extent if record is not None else None) or html_pack.bbox_of(collection)
    if bbox is None:
        raise HTTPException(status_code=422, detail="图层没有可用的坐标范围")
    wkid = record.wkid if record is not None else None
    feature_count = (
        record.feature_count if record is not None else len(collection["features"])
    )

    base = {
        "path": str(resolved),
        "name": (record.name if record is not None and record.name else resolved.name),
        "kind": (
            record.kind
            if record is not None
            else (
                "feature_class"
                if target_kind == PATH_GDB_FEATURE_CLASS
                else "vector"
            )
        ),
        "wkid": wkid,
        "crs_name": record.crs_name if record is not None else "",
        "crs_warning": _crs_warning(wkid),
        "bbox": [float(v) for v in bbox],
        "feature_count": feature_count,
        "field_count": len(record.fields) if record is not None else 0,
    }
    payload, _, effective = _thin_under_limit(
        collection, max_features=max_features, base=base
    )
    payload["cache"] = {
        "hit": False,
        "key": digest[:12],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "read_ms": read_ms,
    }

    _write_cached(
        cache_file,
        {
            "mtime": mtime,
            "size": size,
            "max_features": effective,
            "created_at": time.time(),
            "payload": payload,
        },
    )
    return payload


# ------------------------------------------------------------------ 成果包
def _report_root(workspace: str | Path) -> Path:
    return (Path(workspace).resolve() / REPORT_ROOT).resolve()


def list_reports(workspace: str | Path) -> dict[str, Any]:
    """``output/`` 下的成果包 / 溯源卡（只认这两个文件名，不做全盘递归匹配）。"""
    root = _report_root(workspace)
    items: list[dict[str, Any]] = []
    if root.is_dir():
        try:
            candidates = sorted(root.rglob("*.html"))
        except OSError:
            candidates = []
        for path in candidates:
            if path.name not in REPORT_NAMES:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:  # pragma: no cover - rglob 不会越界
                continue
            items.append(
                {
                    "name": relative,
                    "dir": path.parent.relative_to(root).as_posix() or ".",
                    "kind": "package" if path.name == REPORT_NAMES[0] else "provenance",
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "url": f"/api/reports/{relative}",
                }
            )
    items.sort(key=lambda item: item["mtime"], reverse=True)
    return {
        "items": items,
        "count": len(items),
        "root": REPORT_ROOT,
        "names": list(REPORT_NAMES),
    }


def read_report(workspace: str | Path, name: str) -> Path:
    """把报表名解析成 ``output/`` 内的真实文件；任何越界/非 HTML 都拒绝。"""
    relative = str(name).replace("\\", "/").lstrip("/")
    if not relative.lower().endswith(".html"):
        raise HTTPException(status_code=415, detail="只允许预览 text/html 报表")
    root = _report_root(workspace)
    resolved = safe_path(workspace, Path(REPORT_ROOT) / relative)
    if resolved != root and root not in resolved.parents:
        raise HTTPException(status_code=403, detail="报表路径超出 output/ 目录")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="报表不存在")
    return resolved


# ------------------------------------------------------------------ 路由
def _default_runner_loader(workspace: Path) -> Callable[[], Any]:
    def load() -> Any:
        # 先做一次解释器发现：没有 ArcGIS Pro 就别去起内核（否则要白等
        # 冷启动超时）。发现结果有 lru_cache，重复调用很便宜。
        from ..arcpy_bridge import discover_arcgis_pro_python
        from ..runtime.code_runner import CodeRunner

        info = discover_arcgis_pro_python()
        return CodeRunner(
            workspace=workspace,
            python_executable=info.python_executable,
            state_dir=workspace / ".gis_agent",
            exec_timeout=120.0,
        )

    return load


def _call_with_timeout(fn: Callable[[], Any], timeout: float) -> Any:
    """在守护线程里执行并限时等待，超时抛 :class:`PreviewTimeout`。

    为什么还要这层：内核可能卡在冷启动 / 许可等待上（DAIR-243 实测一次 120s），
    而预览是同步 HTTP 请求。内核自己的超时只在真正执行代码时生效，这层是墙钟
    兼底：超时后请求立刻返回，后台线程自行收尾（daemon，不阻塞进程退出）。
    """
    import threading

    box: dict[str, Any] = {}
    started = time.perf_counter()

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - 原样抛给请求线程
            box["error"] = exc

    thread = threading.Thread(target=target, name="gis-agent-preview-geom", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise PreviewTimeout(round((time.perf_counter() - started) * 1000, 2), timeout)
    if "error" in box:
        raise box["error"]
    return box.get("value")


class PreviewRuntime:
    """预览用的内核持有者：懒加载 + 关服务时能一起收掉。"""

    def __init__(
        self,
        workspace: Path,
        *,
        code_runner: Any = None,
        runner_loader: Callable[[], Any] | None = None,
    ) -> None:
        self.workspace = workspace
        self._runner = code_runner
        self._loader = runner_loader or _default_runner_loader(workspace)
        self._loaded = code_runner is not None

    def get(self) -> Any:
        if not self._loaded:
            self._loaded = True
            try:
                self._runner = self._loader()
            except Exception as exc:  # pragma: no cover - 无 ArcGIS 的机器
                logger.warning("预览内核不可用: %s", exc)
                self._runner = None
        return self._runner

    def close(self) -> None:
        runner = self._runner
        self._runner = None
        shutdown = getattr(runner, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:  # pragma: no cover - 关停失败不影响已返回的响应
                logger.debug("预览内核关停失败: %s", exc)


def _quote(text: str) -> str:
    from urllib.parse import quote

    return quote(text, safe="")


def build_router(
    workspace: str | Path,
    *,
    code_runner: Any = None,
    runner_loader: Callable[[], Any] | None = None,
    runtime: PreviewRuntime | None = None,
) -> APIRouter:
    """装配预览路由。

    ``code_runner`` 直接注入（测试用）；``runner_loader`` 给生产用的懒加载
    （第一次预览 GDB/shp 时才起内核，不用为一个 .geojson 白等 10s 的 arcpy 导入）。
    ``runtime`` 传进来时由调用方负责 ``close()``（``app.py`` 在 lifespan 里收）。
    """
    ws = Path(workspace).resolve()
    preview_runtime = runtime or PreviewRuntime(
        ws, code_runner=code_runner, runner_loader=runner_loader
    )

    router = APIRouter()

    @router.get("/api/catalog/datasets/{path:path}/profile")
    def get_dataset_profile(path: str) -> dict[str, Any]:
        return profile_dataset(ws, path)

    @router.get("/api/catalog/preview")
    def get_preview(
        path: str = Query(..., description="图层路径（必须在 workspace 内）"),
        max_features: int = Query(
            DEFAULT_MAX_FEATURES, ge=1, le=MAX_MAX_FEATURES, description="要素上限"
        ),
    ) -> dict[str, Any]:
        return preview_layer(
            ws, path, max_features=max_features, code_runner=preview_runtime.get()
        )

    @router.get("/api/reports")
    def get_reports() -> dict[str, Any]:
        return list_reports(ws)

    @router.get("/api/reports/{name:path}")
    def get_report(name: str, download: bool = False) -> HTMLResponse:
        target = read_report(ws, name)
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=f"无权限读取报表：{exc}") from exc
        except OSError as exc:
            raise HTTPException(status_code=422, detail=f"报表读取失败：{exc}") from exc
        headers = {
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": REPORT_CSP,
        }
        if download:
            headers["Content-Disposition"] = (
                f"attachment; filename*=UTF-8''{_quote(target.name)}"
            )
        return HTMLResponse(text, headers=headers)

    return router

# -*- coding: utf-8 -*-
"""栅格逐波段元数据：dtype / min / max / mean / nodata / 尺寸 / 色彩解释。

三条硬约束（DAIR-328）：

1. **不伪造数值**：读不到就 ``unavailable: "<原因>"``，绝不把 min/max 写成
   0、null 冒充"算过了"。
2. **单请求有上限**：大栅格（4096×3072）首次打开要 1.3s，所以逐波段结果落
   ``catalog.db`` 的 ``raster_bands`` 表；命中缓存 0 次内核调用，未命中才在
   :data:`DEFAULT_BUDGET_S` 预算内读一次，超预算直接返回原因而不是干等。
3. **无 arcpy 不崩**：没有内核时返回 ``unavailable`` 说明，profile 照常 200。

数据来源：``arcpy.Raster(path).bands()`` 的波段对象。实测（4096×3072 三段
TIFF）打开 ~1.28s、每个波段取 8 个属性 ~1ms；栅格带统计值时 min/max/mean
直接可读，没算统计值时属性为 None（**不会**触发十几秒的全量统计）。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .store import Catalog
from .tables import stat_key

#: 单请求内允许花在逐波段读取上的墙钟预算（秒）。可用 ``GIS_BAND_BUDGET`` 覆盖。
DEFAULT_BUDGET_S = 1.4
#: 内核侧超时比墙钟预算多留的余量：让内核自己收尾，而不是被线程硬切。
KERNEL_SLACK_S = 1.0
#: 内核冷启动（``import arcpy``）的等待上限，与预览一致。
KERNEL_BOOT_TIMEOUT_S = 25.0

BAND_CACHE_DB = Path(".gis_agent") / "catalog.db"

STATS_HINT = "栅格未计算统计值（在 ArcGIS 中执行「计算统计值」后可显示）"
COLOR_HINT = "arcpy 未暴露色彩解释（需 GDAL/Rasterio 路径）"

_BAND_TEMPLATE = r'''
import arcpy as _arcpy

_path = __PATH__

_error = None
_bands = []
_band_count = None
try:
    _raster = _arcpy.Raster(_path)
    _band_count = int(_raster.bandCount)
    for _index, _band in enumerate(_raster.bands() or [], start=1):
        _item = {"band": _index, "name": getattr(_band, "name", "") or ("Band_%d" % _index)}
        for _key, _attr in (
            ("dtype", "pixelType"),
            ("min", "minimum"),
            ("max", "maximum"),
            ("mean", "mean"),
            ("nodata", "noDataValue"),
            ("width", "width"),
            ("height", "height"),
            ("cell_size", "meanCellWidth"),
        ):
            try:
                _item[_key] = getattr(_band, _attr, None)
            except Exception:
                _item[_key] = None
        try:
            _item["color_interpretation"] = _band.getBandProperty("ColorInterpretation") or None
        except Exception:
            _item["color_interpretation"] = None
        _bands.append(_item)
except Exception as _exc:
    _error = str(_exc)

set_result({"bands": _bands, "band_count": _band_count, "error": _error})
'''


def build_band_code(path: str) -> str:
    """生成逐波段读取代码（路径是 JSON 字符串字面量，无任何拼接语法）。"""
    return _BAND_TEMPLATE.replace("__PATH__", json.dumps(str(path), ensure_ascii=False))


def _budget() -> float:
    try:
        value = float(os.environ.get("GIS_BAND_BUDGET", ""))
    except (TypeError, ValueError):
        return DEFAULT_BUDGET_S
    return value if value > 0 else DEFAULT_BUDGET_S


def _open_catalog(workspace: str | Path) -> Catalog | None:
    db_path = Path(workspace) / BAND_CACHE_DB
    if not db_path.is_file():
        return None
    return Catalog(db_path)


def _norm_band(row: dict[str, Any]) -> dict[str, Any]:
    """缓存行 → 响应条目；统计值缺失时补 ``unavailable`` 原因。"""
    band = {
        "band": int(row.get("band") or 0),
        "name": row.get("name") or "",
        "dtype": row.get("dtype"),
        "min": row.get("min"),
        "max": row.get("max"),
        "mean": row.get("mean"),
        "nodata": row.get("nodata"),
        "width": row.get("width"),
        "height": row.get("height"),
        "cell_size": row.get("cell_size"),
        "color_interpretation": row.get("color_interpretation"),
        "unavailable": row.get("unavailable"),
    }
    if band["min"] is None and band["max"] is None and band["mean"] is None:
        band["unavailable"] = band["unavailable"] or STATS_HINT
    return band


def _call_with_timeout(fn: Any, timeout: float) -> Any:
    """在守护线程里执行并限时等待（同预览的做法：内核挂住时请求立刻返回）。"""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - 原样抛给请求线程
            box["error"] = exc

    thread = threading.Thread(target=target, name="gis-agent-band-meta", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"逐波段元数据读取超过 {timeout:g}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _wait_kernel_ready(code_runner: Any) -> None:
    """先等内核预热（一次性成本，~13s），不占逐波段读取的预算。

    与预览同一套口径：``import arcpy`` 与内核就绪是服务级的一次性开销，
    把它算进 1.4s 预算会让「服务刚起来第一次打开栅格元数据」永远超预算。
    """
    prewarm = getattr(code_runner, "prewarm", None)
    if callable(prewarm):
        try:
            prewarm()
        except Exception:  # pragma: no cover - 预热失败留给读取阶段报错
            pass
    wait = getattr(code_runner, "wait_prewarm", None)
    if callable(wait):
        try:
            wait(KERNEL_BOOT_TIMEOUT_S)
        except Exception:  # pragma: no cover
            pass


def _read_via_kernel(path: Path, code_runner: Any, budget_s: float) -> dict[str, Any]:
    outcome = _call_with_timeout(
        lambda: code_runner.run(build_band_code(str(path)), timeout=budget_s + KERNEL_SLACK_S),
        budget_s,
    )
    if not getattr(outcome, "ok", False):
        error = getattr(outcome, "error", None)
        reason = (
            str(error.get("message") or error.get("type") or "")
            if isinstance(error, dict)
            else str(error or "")
        )
        raise RuntimeError(reason or "ArcGIS 内核读波段失败")
    payload = getattr(outcome, "result", None)
    if not isinstance(payload, dict):
        raise RuntimeError("ArcGIS 内核没有返回波段信息")
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload


def read_bands(
    workspace: str | Path,
    path: str | Path,
    *,
    code_runner: Any = None,
    budget_s: float | None = None,
    refresh: bool = False,
    fallback_band_count: int | None = None,
) -> dict[str, Any]:
    """读栅格逐波段元数据（缓存优先，超预算给原因）。

    参数：
        ``workspace``：工作区根（缓存库在 ``<workspace>/.gis_agent/catalog.db``）。
        ``path``：栅格路径（**已经过 safe_path 的绝对路径**）。
        ``code_runner``：ArcPy 内核；``None`` 表示当前环境没有 arcpy。
        ``budget_s``：本次允许花在读取上的墙钟秒数（默认 1.4s，环境变量可覆盖）。
        ``refresh``：忽略缓存重读。
        ``fallback_band_count``：catalog 里记的波段数（无内核时仍能告诉前端"几段"）。

    返回：``{band_count, bands, cached, elapsed_ms, budget_ms, unavailable, notes}``。
    ``bands`` 里每段要么有真实统计值，要么带 ``unavailable`` 原因；不编造。
    """
    started = time.perf_counter()
    resolved = Path(path)
    budget = float(budget_s or _budget())
    mtime, size = stat_key(resolved)

    result: dict[str, Any] = {
        "band_count": fallback_band_count,
        "bands": [],
        "cached": False,
        "elapsed_ms": 0.0,
        "budget_ms": round(budget * 1000),
        "unavailable": None,
        "notes": [],
    }

    catalog = _open_catalog(workspace)
    try:
        if catalog is not None and not refresh:
            cached_rows = catalog.raster_band_rows(str(resolved), mtime, size)
            if cached_rows:
                result["bands"] = [_norm_band(row) for row in cached_rows]
                result["band_count"] = len(result["bands"]) or fallback_band_count
                result["cached"] = True
                result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
                if result["bands"] and not any(
                    band["color_interpretation"] for band in result["bands"]
                ):
                    result["notes"].append(COLOR_HINT)
                return result

        if code_runner is None:
            result["unavailable"] = (
                "无 ArcGIS 内核（当前环境没有 arcpy），逐波段元数据不可读"
            )
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return result

        try:
            _wait_kernel_ready(code_runner)
            payload = _read_via_kernel(resolved, code_runner, budget)
        except TimeoutError as exc:
            result["unavailable"] = (
                f"逐波段读取超过预算（{budget * 1000:.0f}ms）：{exc}；"
                "大栅格首次打开较慢，稍后重试会命中缓存"
            )
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return result
        except Exception as exc:
            result["unavailable"] = f"逐波段元数据读取失败：{exc}"
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return result

        bands = [_norm_band(band) for band in payload.get("bands") or []]
        result["bands"] = bands
        result["band_count"] = payload.get("band_count") or len(bands) or fallback_band_count
        if bands and not any(band["color_interpretation"] for band in bands):
            result["notes"].append(COLOR_HINT)
        if catalog is not None and bands:
            catalog.save_raster_bands(str(resolved), mtime=mtime, size=size, bands=bands)
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result
    finally:
        if catalog is not None:
            catalog.close()


__all__ = [
    "BAND_CACHE_DB",
    "COLOR_HINT",
    "DEFAULT_BUDGET_S",
    "STATS_HINT",
    "build_band_code",
    "read_bands",
]

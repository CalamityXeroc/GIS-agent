# -*- coding: utf-8 -*-
"""GDAL 后端的断言：用 .venv-gdal 自己读产物，不借 arcpy。

为什么需要：``feature_count_gt`` / ``raster_not_constant`` 这类断言在 store 里是走
``arcpy.management.GetCount`` / ``arcpy.Describe`` 的。GDAL 配方跑在没有 arcpy 的机器上
（甚至本机 CLI 的 venv 里也没 arcpy），这些断言只能报“数不出来”——产物其实好好的。
配方声明了 backend，断言就必须用**同一个后端**去验，否则“GDAL 写出来的东西拿 arcpy 判”
全是假阴性。

实现只做两件事：让执行器（``GdalCodeRunner``）在 GDAL 解释器里跑一段
``pyogrio`` / ``rasterio`` 代码，把要素数/坐标系/字段/字段合计、栅格极值回传。
返回 ``None`` 表示“这个 kind 我不接管”，调用方会退到内置实现（``file_exists`` 这类
与后端无关的断言照旧）。
"""

from __future__ import annotations

from typing import Any

#: 本模块接管的断言类型（都要知道“这个数据集里有多少要素”）
_COUNT_KINDS = (
    "feature_count_gt",
    "feature_count_equal",
    "feature_count_lt",
    "feature_count_between",
)

#: 要读字段/字段统计的断言（arcpy 口径下走 SearchCursor，GDAL 这边走 pyogrio）
_FIELD_KINDS = ("fields_present", "field_sum_gt")

#: 要比 EPSG 码的断言。为什么必须由 GDAL 侧接管：内置实现走 ``arcpy.Describe(path)``，
#: 而 GDAL 后端的产物常常是 ``x.gpkg`` 这个**容器路径**——Describe 拿到的 dataType 是
#: Workspace、spatialReference 是 None，于是 wkid=None、断言判失败（DAIR-129 实测：
#: project_to_crs 的 GDAL 版产物明明写对了 4548，断言仍报 wkid=None）。
_CRS_KINDS = ("crs_wkid", "crs_wkid_in")


def _info_code(path: str) -> str:
    """读数据集信息的代码（pyogrio 自带 libgdal，不需要 osgeo 模块）。"""
    return "\n".join(
        [
            "import os",
            "import json",
            f"_p = {path!r}",
            "_rec = {'exists': False, 'reason': '文件不存在'}",
            "if os.path.exists(_p):",
            "    try:",
            "        import pyogrio",
            "        _i = pyogrio.read_info(_p)",
            "        _fl = _i.get('fields')",
            "        _rec = {'exists': True,",
            "                'feature_count': int(_i['features']),",
            "                'crs': str(_i.get('crs') or ''),",
            "                'fields': [str(f) for f in (_fl if _fl is not None else [])]}",
            "    except Exception as _e:",
            "        _rec = {'exists': False, 'reason': str(_e)[:200]}",
            "set_result(_rec)",
        ]
    )


def _field_sum_code(path: str, field: str) -> str:
    """读一个数值字段合计的代码（只读这一列，不拉几何）。"""
    return "\n".join(
        [
            "import os",
            f"_p = {path!r}",
            f"_f = {field!r}",
            "_rec = {'exists': False, 'found': False, 'reason': '文件不存在'}",
            "if os.path.exists(_p):",
            "    try:",
            "        import pyogrio",
            "        _fl = pyogrio.read_info(_p).get('fields')",
            "        _fields = [str(f) for f in (_fl if _fl is not None else [])]",
            "        if _f not in _fields:",
            "            _rec = {'exists': True, 'found': False,",
            "                    'reason': '字段不存在', 'fields': _fields}",
            "        else:",
            "            _df = pyogrio.read_dataframe(_p, columns=[_f], read_geometry=False)",
            "            _s = _df[_f][_df[_f].notna()]",
            "            _rec = {'exists': True, 'found': True,",
            "                    'sum': float(_s.sum()) if len(_s) else None}",
            "    except Exception as _e:",
            "        _rec = {'exists': True, 'found': False, 'reason': str(_e)[:200]}",
            "set_result(_rec)",
        ]
    )


def _crs_code(path: str) -> str:
    """读图层坐标系的 EPSG 码（pyogrio 读 WKT → pyproj 认码）。

    两层置信度与 zonal_statistics 的 CRS 判等同一口径：先默认置信度，认不出再退到
    ``min_confidence=20``——ESRI 写法的 WKT 常常只能这样认出来（实测）。
    """
    return "\n".join(
        [
            "import os",
            "import pyogrio",
            f"_p = {path!r}",
            "_rec = {'exists': False, 'reason': '文件不存在'}",
            "if os.path.exists(_p):",
            "    try:",
            "        _crs = pyogrio.read_info(_p).get('crs')",
            "        _rec = {'exists': True, 'wkid': None, 'low': None, 'name': ''}",
            "        if _crs:",
            "            from pyproj import CRS",
            "            _c = CRS.from_user_input(_crs)",
            "            _rec['wkid'] = _c.to_epsg()",
            "            _rec['low'] = _c.to_epsg(min_confidence=20)",
            "            _rec['name'] = _c.name",
            "    except Exception as _e:",
            "        _rec = {'exists': True, 'wkid': None, 'low': None, 'name': '',",
            "                'reason': str(_e)[:200]}",
            "set_result(_rec)",
        ]
    )


def _check_crs(
    runner: Any, kind: str, target: str, expected: Any
) -> tuple[bool, str]:
    """``crs_wkid`` / ``crs_wkid_in`` 的 GDAL 实现（与内置 arcpy 口径同义）。"""
    payload, error = _payload(runner, _crs_code(target))
    if payload is None:
        return False, error
    if not payload.get("exists"):
        return False, f"要素类不存在（pyogrio 也读不出）: {payload.get('reason') or ''}"
    allowed = expected if isinstance(expected, list) else [expected]
    try:
        allowed_codes = {int(a) for a in allowed}
    except Exception:
        return False, f"断言里的 EPSG 码不是整数: {expected!r}"
    got, low = payload.get("wkid"), payload.get("low")
    ok = (got is not None and int(got) in allowed_codes) or (
        low is not None and int(low) in allowed_codes
    )
    return ok, f"wkid={got} low={low} allowed={sorted(allowed_codes)}"


def _raster_range_code(path: str) -> str:
    """读栅格有效像元 (min, max) 的代码：按块读，不把整幅图拉进内存。

    nodata / NaN 先剔掉：``raster_not_constant`` 看的是**有效值**是否全相同，
    不剔的话“整幅只有 nodata”会被当成“非常量”。
    """
    return "\n".join(
        [
            "import os",
            "import numpy as np",
            f"_p = {path!r}",
            "_rec = {'exists': False, 'reason': '文件不存在'}",
            "if os.path.exists(_p):",
            "    try:",
            "        import rasterio",
            "        from rasterio.windows import Window",
            "        with rasterio.open(_p) as _ds:",
            "            _lo = None; _hi = None; _n = 0",
            "            _step = 2048",
            "            for _r0 in range(0, _ds.height, _step):",
            "                for _c0 in range(0, _ds.width, _step):",
            "                    _win = Window(_c0, _r0, min(_step, _ds.width - _c0),",
            "                                  min(_step, _ds.height - _r0))",
            "                    _a = _ds.read(1, window=_win, masked=True).astype('float64')",
            "                    _v = _a.compressed()",
            "                    _v = _v[np.isfinite(_v)]",
            "                    if _v.size:",
            "                        _n += int(_v.size)",
            "                        _blo = float(_v.min()); _bhi = float(_v.max())",
            "                        _lo = _blo if _lo is None else min(_lo, _blo)",
            "                        _hi = _bhi if _hi is None else max(_hi, _bhi)",
            "            _rec = {'exists': True, 'valid': _n, 'min': _lo, 'max': _hi}",
            "    except Exception as _e:",
            "        _rec = {'exists': False, 'reason': str(_e)[:200]}",
            "set_result(_rec)",
        ]
    )


def _payload(runner: Any, code: str, *, timeout: int = 120) -> tuple[dict[str, Any] | None, str]:
    """在 GDAL 执行器里跑一小段代码，拿回 ``set_result`` 的字典。"""
    outcome = runner.run(code, timeout=timeout)
    payload = outcome.result if outcome.ok and isinstance(outcome.result, dict) else None
    if payload:
        return payload, ""
    reason = ""
    if isinstance(outcome.error, dict):
        reason = str(outcome.error.get("message") or outcome.error.get("type") or "")
    return None, f"GDAL 侧读取失败: {reason[:160] or '没有返回结果'}"


def _check_field(
    runner: Any, kind: str, target: str, expected: Any, raw: dict[str, Any]
) -> tuple[bool, str]:
    """``fields_present`` / ``field_sum_gt`` 的 GDAL 实现（与内置 arcpy 口径同义）。"""
    if kind == "fields_present":
        payload, error = _payload(runner, _info_code(target))
        if payload is None:
            return False, error
        fields = [str(f) for f in payload.get("fields") or []]
        required = expected if isinstance(expected, list) else [expected]
        missing = [f for f in required if f not in fields]
        return not missing, f"missing={missing}"

    field = str(raw.get("field", "") or "")
    if not field:
        return False, "断言缺少 field"
    payload, error = _payload(runner, _field_sum_code(target, field))
    if payload is None:
        return False, error
    if not payload.get("exists"):
        return False, f"要素类不存在（pyogrio 也读不出）: {payload.get('reason') or ''}"
    if not payload.get("found"):
        detail = str(payload.get("reason") or field)
        if payload.get("fields") is not None:
            detail += f"；现有 {payload.get('fields')}"
        return False, detail
    total = payload.get("sum")
    ok = total is not None and total > float(expected or 0)
    return ok, f"sum({field})={total} expected>{expected}"


def gdal_raster_check(
    runner: Any, kind: str, target: str, expected: Any, raw: dict[str, Any] | None = None
) -> tuple[bool, str] | None:
    """按 GDAL 口径判断一条栅格断言；不接管的 kind 返回 ``None``。

    ``raster_not_constant`` 的内置实现走 arcpy Describe（``_raster_range``）。GDAL 后端
    跑在没有 arcpy 的 .venv-gdal 里，那条路只能得到 range=None —— 产物其实好好的，
    整个 run 却被判成“断言未通过”（DAIR-91 实测：zonal_statistics_raster 的 GDAL 跑
    输出 min=450/max=2450，却因为这条断言被标失败）。栅格产物与向量产物同理，
    得让 GDAL 自己读。
    """
    if kind != "raster_not_constant":
        return None
    if runner is None:  # 没有执行器就没有 GDAL 侧答案，交给内置实现
        return None
    payload, error = _payload(runner, _raster_range_code(target))
    if payload is None:
        return False, error
    if not payload.get("exists"):
        return False, f"栅格读不出来（rasterio 也读不出）: {payload.get('reason') or ''}"
    low, high = payload.get("min"), payload.get("max")
    ok = low is not None and high is not None and float(low) != float(high)
    return ok, f"range=({low}, {high})"


def gdal_vector_check(
    runner: Any, kind: str, target: str, expected: Any, raw: dict[str, Any] | None = None
) -> tuple[bool, str] | None:
    """按 GDAL 口径判断一条向量断言；不接管的 kind 返回 ``None``。"""
    if (
        kind != "vector_exists"
        and kind not in _COUNT_KINDS
        and kind not in _FIELD_KINDS
        and kind not in _CRS_KINDS
    ):
        return None
    if runner is None:  # 没有执行器就没有 GDAL 侧答案，交给内置实现
        return None
    if kind in _FIELD_KINDS:
        return _check_field(runner, kind, target, expected, raw or {})
    if kind in _CRS_KINDS:
        return _check_crs(runner, kind, target, expected)
    outcome = runner.run(_info_code(target), timeout=120)
    payload = outcome.result if outcome.ok and isinstance(outcome.result, dict) else None
    if not payload:
        reason = ""
        if isinstance(outcome.error, dict):
            reason = str(outcome.error.get("message") or outcome.error.get("type") or "")
        return False, f"GDAL 侧读取失败: {reason[:160] or '没有返回结果'}"

    exists = bool(payload.get("exists"))
    if kind == "vector_exists":
        return exists, "" if exists else f"要素类不存在（pyogrio 也读不出）: {payload.get('reason') or ''}"
    if not exists:
        return False, f"要素类不存在（pyogrio 也读不出）: {payload.get('reason') or ''}"

    count = int(payload.get("feature_count") or 0)
    if kind == "feature_count_gt":
        return count > int(expected or 0), f"count={count}"
    if kind == "feature_count_equal":
        return int(expected) == count, f"count={count} expected={expected}"
    if kind == "feature_count_lt":
        return count < int(expected), f"count={count} expected<{expected}"
    try:
        lo, hi = int(expected[0]), int(expected[1])
    except Exception:
        return False, f"非法范围: {expected}"
    return lo <= count <= hi, f"count={count} expected in [{lo},{hi}]"

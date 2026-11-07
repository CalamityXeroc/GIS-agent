# -*- coding: utf-8 -*-
"""守恒 / 拓扑 / 量纲断言库（P1-5 / W5）。

三类断言，**不依赖任何具体配方**：每条断言只吃「主代理自报的数字（claim）」+
「系统量测到的事实（facts）」，返回 0..n 条 :class:`~gis_cli.review.claims.Finding`。

为什么做成声明式清单而不是写死检查：

- 复核 subagent 可以在 ``assertion_check`` 里按任务挑选/组合断言；
- 单测与注入基准可以直接喂事实，不需要 ArcGIS；
- 新增一类分析时只加断言，不改复核流程。

约定：

- ``facts`` 是 ``collect_facts`` 的产物（或测试造的同构字典），路径用**列表**表示，
  例如 ``["datasets", "output/zones.shp", "fields", "POP", "sum"]``；
- 缺事实/缺声明数字时该条断言记为 **skipped**，不硬判失败（"没检查" ≠ "检查通过"）；
- 断言写错（未知类型/参数类型不对）记 **error**，交给复核者当问题上报。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Iterable, Mapping

from .claims import Finding, ReviewClaim, ReviewReport

# ------------------------------------------------------------------ 单位
#: 面积单位 → 平方米。``None`` = 与米制面积不可比（经纬度平方）。
_AREA_TO_M2: dict[str, float | None] = {
    "m2": 1.0,
    "sqm": 1.0,
    "平方米": 1.0,
    "km2": 1_000_000.0,
    "平方公里": 1_000_000.0,
    "ha": 10_000.0,
    "公顷": 10_000.0,
    "mu": 2000.0 / 3.0,
    "亩": 2000.0 / 3.0,
    "degree2": None,
    "deg2": None,
    "平方度": None,
}
#: 距离单位 → 米。
_DIST_TO_M: dict[str, float | None] = {
    "m": 1.0,
    "meter": 1.0,
    "metre": 1.0,
    "米": 1.0,
    "km": 1000.0,
    "千米": 1000.0,
    "公里": 1000.0,
    "ft": 0.3048,
    "英尺": 0.3048,
    "degree": None,
    "deg": None,
    "度": None,
}
_GEOGRAPHIC_UNITS = {"degree", "deg", "度", "degree2", "deg2", "平方度"}


def _norm_unit(unit: Any) -> str:
    text = str(unit or "").strip().lower().replace("²", "2").replace("^2", "2")
    return text.replace(" ", "")


def _area_factor(unit: Any) -> float | None:
    return _AREA_TO_M2.get(_norm_unit(unit))


def _dist_factor(unit: Any) -> float | None:
    return _DIST_TO_M.get(_norm_unit(unit))


def _is_area_unit(unit: Any) -> bool:
    return _norm_unit(unit) in _AREA_TO_M2


def _is_dist_unit(unit: Any) -> bool:
    return _norm_unit(unit) in _DIST_TO_M


class AssertionSkipped(Exception):
    """缺少可判定所需的事实/声明数字（不是失败）。"""


# ------------------------------------------------------------------ 取值
_MISSING = object()


def _lookup(data: Any, path: Any) -> Any:
    """按列表路径取嵌套值；路径同时接受单键字符串（不按点切分）。"""
    keys = list(path) if isinstance(path, (list, tuple)) else ([path] if path not in (None, "") else [])
    current = data
    for key in keys:
        if isinstance(current, Mapping) and key in current:
            current = current[key]
        elif isinstance(current, (list, tuple)) and isinstance(key, int) and -len(current) <= key < len(current):
            current = current[key]
        else:
            return _MISSING
    return current


def _fact(facts: Any, path: Any, label: str = "") -> Any:
    value = _lookup(facts, path)
    if value is _MISSING or value is None:
        where = f"{label}={path}" if label else str(path)
        raise AssertionSkipped(f"缺少量测事实 {where}")
    return value


def _metric(claim: ReviewClaim, name: str) -> Any:
    if not name:
        raise AssertionSkipped("未指定声明数字的键名")
    value = claim.metrics.get(name)
    if value is None:
        raise AssertionSkipped(f"claim.metrics 缺少 {name}")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, Mapping):
        keys = "、".join(str(k) for k in list(value)[:6])
        raise ValueError(
            f"{label} 是分区字典（{len(value)} 项：{keys}…），不是单个数字——"
            "请改传标量路径（如 [...fields, POP, sum]），或改用按分区比的断言"
            "（conservation.partition_matches_source / sum_equals）"
        )
    if isinstance(value, (list, tuple)):
        raise ValueError(f"{label} 是数组（{len(value)} 项），不是单个数字")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} 不是数字：{value!r}")


def _decimals(value: Any) -> int | None:
    """声明值写出来的小数位数（``152.0``→1、``152``→0、``1.2e-3``→4）。

    这是「声明精度」的唯一来源：值写成几位小数，就只能要求它精确到那一位。
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return 0
    text = repr(value) if isinstance(value, float) else str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        exponent = Decimal(text).as_tuple().exponent
    except (InvalidOperation, ValueError):
        return None
    return max(0, -int(exponent))


def _precision_tolerance(raw_claimed: Any, measured: float, min_rel: float = 0.001) -> float:
    """按声明值的末位算允许误差（半个末位），小数声明不比 ``min_rel`` 更严。

    为什么不直接拿 ``rel_tol``：2% 的容差会把「152.34 写成 152.0」（0.22%）放过去，
    而 152.0 这个写法声称精确到 0.1（±0.05）——它自己的位数就不支持这么大的误差。
    0.1% 地板是留给「同一份数据、不同算法/投影」的正常噪声，避免误杀（DAIR-191）。

    整数声明（``decimals == 0``）不放地板（DAIR-195）：8_412_000 这种写法声称精确到个位，
    0.1% 地板能容下 8_400 人的偏差，``pop=8_412_000`` vs 量测 8_420_000 就会被放过。
    要表达粗粒度估计，写成带小数的形式（``8412000.0``，仍受 0.1% 地板保护）或显式放宽 ``min_rel``。
    """
    decimals = _decimals(raw_claimed)
    if decimals is None:
        return min_rel
    half = 0.5 * 10 ** (-decimals)
    base = abs(measured) or 1.0
    if decimals == 0:
        return half / base
    return max(half / base, min_rel)


def _to_unit(value: float, from_unit: Any, to_unit: Any, quantity: str) -> float | None:
    """面积/距离按单位换算；单位不可换算（度 vs 米）返回 ``None``。"""
    table = _AREA_TO_M2 if quantity == "area" else _DIST_TO_M
    factor_from = table.get(_norm_unit(from_unit))
    factor_to = table.get(_norm_unit(to_unit))
    if factor_from is None or factor_to is None:
        return None
    return value * factor_from / factor_to


def _closeness(claimed: float, measured: float, rel_tol: float, abs_tol: float = 0.0) -> tuple[bool, float]:
    diff = abs(claimed - measured)
    if diff <= abs_tol:
        return True, diff
    base = abs(measured) if measured else abs(claimed)
    if base == 0:
        return claimed == measured, 0.0
    rel = diff / base
    return rel <= rel_tol, rel


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) >= 1_000_000 or (0 < abs(value) < 0.001):
            return f"{value:.4g}"
        return f"{value:,.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _fmt_precise(value: float) -> str:
    """精度类消息要看到位数（``_fmt`` 的 4 位有效数字会把 152.34 显示成 152.3）。"""
    return f"{value:,.10g}"


def _fmt_declared(value: Any) -> str:
    """声明值按它写出来的样子显示（152.0 不能显示成 152）。"""
    return repr(value) if isinstance(value, float) else str(value)


def _mismatch(
    code: str,
    family: str,
    detail: str,
    *,
    claimed: Any = None,
    measured: Any = None,
    severity: str = "error",
) -> Finding:
    return Finding(
        code=code,
        family=family,
        detail=detail,
        severity=severity,
        claimed=claimed,
        measured=measured,
    )


# ---------------------------------------------------------------- 源数据勾稽
#: 声明数字的键名 → 量测事实里同名数值字段/分组汇总的候选名。
_NAME_PREFIXES = ("total_", "sum_", "zone_", "group_", "sum_of_")
_NAME_SUFFIXES = ("_total", "_sum", "_count")

#: 只有这些「就是数据集总面积」的键名才允许自动去比 area_total，
#: 避免把「绿地面积」这种子集数字错当成整层面积（误杀）。
_CANONICAL_AREA_METRICS = {
    "area": None,
    "total_area": None,
    "area_km2": "km2",
    "area_m2": "m2",
    "area_ha": "ha",
    "total_area_km2": "km2",
    "total_area_m2": "m2",
    "total_area_ha": "ha",
}


def _name_candidates(name: str) -> list[str]:
    """``total_pop`` → ``[total_pop, pop]``；用来找同名的源数据字段。"""
    base = str(name or "").strip().lower()
    if not base:
        return []
    out = [base]
    for prefix in _NAME_PREFIXES:
        if base.startswith(prefix) and len(base) > len(prefix):
            out.append(base[len(prefix) :])
    for item in list(out):
        for suffix in _NAME_SUFFIXES:
            if item.endswith(suffix) and len(item) > len(suffix):
                out.append(item[: -len(suffix)])
    return list(dict.fromkeys(out))


def _dataset_scope(
    facts: Mapping[str, Any], claim: ReviewClaim
) -> tuple[list[tuple[str, Mapping[str, Any]]], bool]:
    """（要扫的数据集, 口径是否明确）。

    DAIR-193：只扫 claim 自己声明的数据集（以前把其余数据集也排在后面一起扫，同名不同量的
    数据集——结果层 vs 参考层、研究区 vs 子集——会把干净样本判死），与 ``tools.py`` 里
    review_result 的既有口径对齐。``data_paths`` 为空、或声明的路径一条都没对上 facts 的键
    （绝对/相对写法不同）时退回扫全部，但这时口径不明（``False``），调用方只报 warning。
    """
    datasets = facts.get("datasets") if isinstance(facts, Mapping) else None
    if not isinstance(datasets, Mapping):
        return [], False
    normalized = {str(key).replace("\\", "/"): value for key, value in datasets.items()}
    if claim.data_paths:
        ordered: list[tuple[str, Mapping[str, Any]]] = []
        for raw in claim.data_paths:
            key = str(raw).replace("\\", "/")
            value = normalized.get(key)
            if isinstance(value, Mapping):
                ordered.append((key, value))
        if ordered:
            return ordered, len(claim.data_paths) == 1
    return [(key, value) for key, value in normalized.items() if isinstance(value, Mapping)], False


def _ordered_datasets(facts: Mapping[str, Any], claim: ReviewClaim) -> list[tuple[str, Mapping[str, Any]]]:
    """要参与同名勾稽的数据集（见 ``_dataset_scope``）。"""
    return _dataset_scope(facts, claim)[0]


def _out_of_scope_datasets(
    facts: Mapping[str, Any], claim: ReviewClaim
) -> list[tuple[str, Mapping[str, Any]]]:
    """声明域之外的数据集（只在「域内没量到、域外量到了」时用来补提示）。"""
    datasets = facts.get("datasets") if isinstance(facts, Mapping) else None
    if not isinstance(datasets, Mapping):
        return []
    scoped = {key for key, _ in _ordered_datasets(facts, claim)}
    normalized = {str(key).replace("\\", "/"): value for key, value in datasets.items()}
    return [
        (key, value)
        for key, value in normalized.items()
        if isinstance(value, Mapping) and key not in scoped
    ]


def _source_measures(
    facts: Mapping[str, Any],
    claim: ReviewClaim,
    names: list[str],
    *,
    datasets: list[tuple[str, Mapping[str, Any]]] | None = None,
) -> list[tuple[str, float]]:
    """找 facts 里与 ``names`` 同名的数值量（字段 sum 优先，其次分组汇总之和）。"""
    wanted = {item.lower() for item in names if item}
    if not wanted:
        return []
    found: list[tuple[str, float]] = []
    for path, entry in (datasets if datasets is not None else _ordered_datasets(facts, claim)):
        for field, stats in (entry.get("fields") or {}).items():
            if str(field).lower() in wanted and isinstance(stats, Mapping) and stats.get("sum") is not None:
                try:
                    found.append((f"{path}.fields.{field}.sum", float(stats["sum"])))
                except (TypeError, ValueError):
                    continue
        for field, values in (entry.get("group_sums") or {}).items():
            if str(field).lower() not in wanted or not isinstance(values, Mapping):
                continue
            numbers: list[float] = []
            for value in values.values():
                try:
                    numbers.append(float(value))
                except (TypeError, ValueError):
                    continue
            if numbers:
                found.append((f"{path}.group_sums.{field}（{len(numbers)} 组之和）", float(sum(numbers))))
    return found


def _measured_area(
    spec: Mapping[str, Any],
    facts: Mapping[str, Any],
    claim: ReviewClaim,
    area_name: str,
    claim_unit: str,
    *,
    datasets: list[tuple[str, Mapping[str, Any]]] | None = None,
) -> list[tuple[float, str]]:
    """量测到的面积候选（换算到 claim 的面积单位），由调用方择近。

    DAIR-193：以前取第一个有 ``area_total`` 的数据集就 return，与分子侧「任一命中即通过」
    不对称——多数据集时声明的是哪个面积全看遍历顺序。现在收全部候选，任一在容差内即通过。
    """
    explicit = spec.get("area_source_path")
    if explicit is not None:
        value = _number(_fact(facts, explicit, "area_source_path"), "量测面积")
        source_unit = spec.get("area_source_unit") or "m2"
        converted = _to_unit(value, source_unit, claim_unit, "area")
        if converted is None:
            raise AssertionSkipped(f"area_source_unit={source_unit} 与 {claim_unit} 不可换算")
        return [(converted, str(explicit))]
    if str(area_name or "").lower() not in _CANONICAL_AREA_METRICS:
        return []
    found: list[tuple[float, str]] = []
    for path, entry in (datasets if datasets is not None else _ordered_datasets(facts, claim)):
        total = entry.get("area_total")
        if total is None:
            continue
        source_unit = entry.get("area_unit") or "m2"
        try:
            converted = _to_unit(float(total), source_unit, claim_unit, "area")
        except (TypeError, ValueError):
            continue
        if converted is not None:
            found.append((converted, f"{path}.area_total"))
    return found


#: ``group_sums`` 里看起来是面积的键 → 单位（供 area_partition 逐分区核对）。
_AREA_KEY_UNITS = {
    "area": "m2",
    "area_m2": "m2",
    "area_km2": "km2",
    "area_ha": "ha",
    "area_deg2": "deg2",
    "shape_area": "m2",
    "shape_area_m2": "m2",
    "面积": "m2",
}


def _area_key_unit(key: str) -> str | None:
    """面积键 → 单位；也认 ``facts.py`` 多分组场景的 ``AREA_M2_BY_<cat>`` 形式（DAIR-195）。"""
    name = str(key or "").strip().lower()
    unit = _AREA_KEY_UNITS.get(name)
    if unit is not None:
        return unit
    base, sep, _ = name.partition("_by_")
    if sep:
        return _AREA_KEY_UNITS.get(base)
    return None


def _part_problems(
    claimed: Mapping[str, Any],
    measured: Mapping[str, Any],
    tol: float,
    abs_tol: float,
) -> list[str]:
    """逐分区比对，返回人读得懂的问题列表（分区多了/少了/数值对不上）。"""
    problems: list[str] = []
    for key in sorted(set(measured) - set(claimed)):
        problems.append(f"缺少分区 {key}（量测 {_fmt(measured[key])}）")
    for key in sorted(set(claimed) - set(measured)):
        problems.append(f"多出分区 {key}（声明 {_fmt(claimed[key])}）")
    for key in sorted(set(claimed) & set(measured)):
        try:
            ok, rel = _closeness(
                _number(claimed[key], key),
                _number(measured[key], key),
                tol,
                abs_tol,
            )
        except ValueError:
            problems.append(f"分区 {key} 的数值无法比较")
            continue
        if not ok:
            problems.append(f"分区 {key} 声明 {_fmt(claimed[key])} ≠ 量测 {_fmt(measured[key])}（差 {_pct(rel)}）")
    return problems


# ------------------------------------------------------------------ 断言注册
@dataclass(frozen=True)
class Assertion:
    name: str
    family: str
    summary: str
    params: str
    run: Callable[[Mapping[str, Any], Mapping[str, Any], ReviewClaim], Iterable[Finding]]


def _rel(spec: Mapping[str, Any], default: float) -> float:
    try:
        return abs(float(spec.get("rel_tol", default)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- 守恒
def _sum_equals(spec, facts, claim) -> list[Finding]:
    total = _number(_metric(claim, str(spec.get("total", "total") or "total")), "总量")
    raw_parts = _metric(claim, str(spec.get("parts", "parts") or "parts"))
    if isinstance(raw_parts, Mapping):
        values = list(raw_parts.values())
    elif isinstance(raw_parts, (list, tuple)):
        values = list(raw_parts)
    else:
        raise ValueError("parts 必须是字典或数组")
    if not values:
        raise AssertionSkipped("分项为空")
    part_sum = sum(_number(item, "分项") for item in values)
    tol = _rel(spec, 0.01)
    abs_tol = float(spec.get("abs_tol", 0.0) or 0.0)
    ok, rel = _closeness(part_sum, total, tol, abs_tol)
    if ok:
        return []
    return [
        _mismatch(
            "conservation.sum_mismatch",
            "conservation",
            f"分项合计 {_fmt(part_sum)} 与声明总量 {_fmt(total)} 差 {_pct(rel)}（容差 {_pct(tol)}）",
            claimed=total,
            measured=part_sum,
        )
    ]


def _matches_source(spec, facts, claim) -> list[Finding]:
    claimed = _number(_metric(claim, str(spec.get("metric", "") or "")), "声明值")
    measured = _number(_fact(facts, spec.get("source_path"), "source_path"), "量测值")
    scale = float(spec.get("scale", 1.0) or 1.0)
    measured *= scale
    tol = _rel(spec, 0.01)
    ok, rel = _closeness(claimed, measured, tol, float(spec.get("abs_tol", 0.0) or 0.0))
    if ok:
        return []
    return [
        _mismatch(
            "conservation.source_mismatch",
            "conservation",
            f"声明 {_fmt(claimed)} 与数据量测 {_fmt(measured)} 差 {_pct(rel)}（容差 {_pct(tol)}）",
            claimed=claimed,
            measured=measured,
        )
    ]


def _partition_matches_source(spec, facts, claim) -> list[Finding]:
    raw_claimed = _metric(claim, str(spec.get("metric", "") or ""))
    if not isinstance(raw_claimed, Mapping):
        raise ValueError("metric 必须是分区字典 {分区: 数值}")
    raw_measured = _fact(facts, spec.get("source_path"), "source_path")
    if not isinstance(raw_measured, Mapping):
        raise ValueError("source_path 必须指向分区字典")
    tol = _rel(spec, 0.01)
    ignore = {str(x) for x in (spec.get("ignore_keys") or [])}
    claimed = {str(k): v for k, v in raw_claimed.items() if str(k) not in ignore}
    measured = {str(k): v for k, v in raw_measured.items() if str(k) not in ignore}
    problems = _part_problems(claimed, measured, tol, float(spec.get("abs_tol", 0.0) or 0.0))
    if not problems:
        return []
    return [
        _mismatch(
            "conservation.partition_mismatch",
            "conservation",
            "分区明细与数据对不上：" + "；".join(problems[:4]) + ("…" if len(problems) > 4 else ""),
            claimed=raw_claimed,
            measured=raw_measured,
        )
    ]


#: 逐分区量测的来源标签：``group_sums`` 键是 ``dataset.group_sums.KEY`` 字符串，
#: spec 显式给的 ``source_path`` 保留路径本身（渲染时拼成点号路径，不是 list repr）。
_PartLabel = str | tuple[str, ...]


def _short_key(label: _PartLabel) -> str:
    """``output/z.shp.group_sums.AREA_M2_BY_ZONE`` → ``AREA_M2_BY_ZONE``（消息里只留键名）。

    ``source_path`` 的路径标签是 tuple：拼成 ``datasets.output/z.shp.group_sums.AREA_M2_BY_ZONE``，
    不能 ``str()`` 成 ``['datasets', ...]`` 这种 list repr（DAIR-205）。
    """
    if isinstance(label, tuple):
        return ".".join(str(item) for item in label)
    return str(label).rsplit(".group_sums.", 1)[-1]


def _group_key_field(label: _PartLabel) -> str | None:
    """``AREA_M2_BY_KIND`` → ``kind``；没有 ``_BY_`` 后缀（单分组数据集）返回 None。"""
    name = _short_key(label).lower()
    _, sep, field = name.partition("_by_")
    if not sep:
        return None
    return field.strip() or None


@dataclass(frozen=True)
class _PartCandidates:
    """逐分区量测的候选键；``chosen`` 为 None 表示分不清该比哪个。"""

    labels: tuple[_PartLabel, ...]
    chosen: tuple[dict[str, float], _PartLabel] | None = None
    #: 分不清的原因（spec 声明的 ``group_field`` 没量到）；用于回落 warning 的文案
    note: str | None = None


def _measured_parts(
    spec: Mapping[str, Any],
    facts: Mapping[str, Any],
    claim: ReviewClaim,
    claim_unit: str,
    claimed: Mapping[str, Any],
    tol: float,
    abs_tol: float,
) -> _PartCandidates | None:
    """逐分区量测面积（换算到 claim 单位）：显式 source_path 优先，否则找 area 类分组汇总。"""
    explicit = spec.get("source_path")
    if explicit is not None:
        raw = _fact(facts, explicit, "source_path")
        if not isinstance(raw, Mapping):
            raise ValueError("source_path 必须指向分区面积字典")
        source_unit = spec.get("source_unit") or claim_unit
        converted: dict[str, float] = {}
        for key, value in raw.items():
            item = _to_unit(_number(value, str(key)), source_unit, claim_unit, "area")
            if item is None:
                raise AssertionSkipped(f"source_unit={source_unit} 与 {claim_unit} 不可换算")
            converted[str(key)] = item
        # label 保留路径本身：它不是分组键，``str()`` 成 list repr 也没法读（DAIR-205）
        label: _PartLabel = (
            tuple(str(item) for item in explicit) if isinstance(explicit, (list, tuple)) else str(explicit)
        )
        return _PartCandidates((label,), (converted, label))

    claimed_keys = set(claimed)
    total_path = spec.get("total_path")
    dataset_key = str(total_path[-2]) if isinstance(total_path, (list, tuple)) and len(total_path) >= 2 else ""
    datasets = facts.get("datasets") if isinstance(facts, Mapping) else None
    entry = datasets.get(dataset_key) if isinstance(datasets, Mapping) and dataset_key else None
    if not isinstance(entry, Mapping):
        return None
    matches: list[tuple[dict[str, float], str]] = []
    for key, values in (entry.get("group_sums") or {}).items():
        source_unit = _area_key_unit(key)
        if source_unit is None or not isinstance(values, Mapping):
            continue
        # 分组字段跟 claim 的分区键对不上（比如按 ZONE 量、claim 按 DISTRICT 报）时
        # 不能当成“分区少/多”报错：这是没量到同一个分区，不是分区错了。
        if claimed_keys and not (claimed_keys & {str(item) for item in values}):
            continue
        converted = {}
        for part, value in values.items():
            item = _to_unit(_number(value, str(part)), source_unit, claim_unit, "area")
            if item is None:
                continue
            converted[str(part)] = item
        if converted:
            matches.append((converted, f"{dataset_key}.group_sums.{key}"))
    # spec 声明了分组字段时先按声明过滤（DAIR-199）：撞名的两个键都用 A/B/C 时，
    # 按 claim 的值挑键会把「分区互换」当成对得上的那个键，必须让 spec 说话。
    declared_group = str(spec.get("group_field") or "").strip()
    wanted = declared_group.lower()

    def _disjoint_labels_note() -> str | None:
        """声明的分组键在、只是标签与 claim 无交集时的文案；否则 ``None``。

        这个键可能早在 ``claimed_keys`` 过滤时就被摘掉了，所以得重新扫一遍原始
        ``group_sums``，不能只看 ``matches``——否则会回落成「没有这个键」或
        「只有总面积」（DAIR-214）。
        """
        if not wanted:
            return None
        keys = [
            str(key)
            for key, values in (entry.get("group_sums") or {}).items()
            if _area_key_unit(key) is not None
            and isinstance(values, Mapping)
            and _group_key_field(str(key)) == wanted
        ]
        if not keys:
            return None
        return (
            "spec 声明按 {} 分组，量测里有 {}，但它的分区标签与 claim 的分区（{}）没有交集".format(
                declared_group,
                " / ".join(keys),
                " / ".join(str(item) for item in claimed),
            )
        )

    if not matches:
        note = _disjoint_labels_note()
        if note is not None:
            return _PartCandidates((), None, note)
        return None
    if declared_group:
        scoped = [item for item in matches if _group_key_field(item[1]) == wanted]
        if scoped:
            matches = scoped
        else:
            # 单分组数据集的键没有 ``_BY_`` 后缀，无法用名字核对声明，按唯一候选处理
            unsuffixed = [item for item in matches if _group_key_field(item[1]) is None]
            if unsuffixed:
                matches = unsuffixed
            else:
                # 声明的字段未必真的没量到：标签与 claim 无交集时，这个键在上面的
                # ``claimed_keys`` 过滤就被摘掉了，再说「没有这个键」会把人支到错方向（DAIR-209）。
                note = _disjoint_labels_note()
                if note is not None:
                    return _PartCandidates(tuple(label for _, label in matches), None, note)
                return _PartCandidates(
                    tuple(label for _, label in matches),
                    None,
                    "spec 声明按 {} 分组，但量测里没有对应的面积分组键（量测到 {}）".format(
                        declared_group,
                        " / ".join(_short_key(label) for _, label in matches),
                    ),
                )
    labels = tuple(label for _, label in matches)
    if len(matches) == 1:
        return _PartCandidates(labels, matches[0])
    # claim 只报了分区数组、没有标签可比：多个分组键时分不清该比哪个
    if not claimed:
        return _PartCandidates(labels)
    # 多个分组键都能对上 claim 的分区标签（都叫 A/B/C、都用行政区代码）时不猜第一个，
    # 先用 claim 的分区值判定该比哪个（DAIR-201）：
    # 有一个键能逐区对上就是它（DAIR-197 的「拿第一个键误判 error」由此消失）；
    # 全都对不上才挂最贴合的候选并列出全部候选键——这不是「没量到同一分区」，是真的分区错了。
    for parts, label in matches:
        if not _part_problems(claimed, parts, tol, abs_tol):
            return _PartCandidates(labels, (parts, label))
    closest = min(
        matches,
        key=lambda item: (len(_part_problems(claimed, item[0], tol, abs_tol)), item[1]),
    )
    return _PartCandidates(labels, closest)


def _area_partition(spec, facts, claim) -> list[Finding]:
    raw_parts = _metric(claim, str(spec.get("parts", "parts") or "parts"))
    if isinstance(raw_parts, Mapping):
        values = list(raw_parts.values())
    elif isinstance(raw_parts, (list, tuple)):
        values = list(raw_parts)
    else:
        raise ValueError("parts 必须是字典或数组")
    if not values:
        raise AssertionSkipped("分区面积为空")
    unit = _norm_unit(spec.get("unit") or claim.units.get("area") or "m2")
    total_raw = _number(_fact(facts, spec.get("total_path"), "total_path"), "量测总面积")
    total_unit = _norm_unit(spec.get("total_unit") or "m2")
    if total_unit not in _AREA_TO_M2:
        raise AssertionSkipped(f"未知总面积单位 {total_unit}")
    total_m2 = total_raw * (_area_factor(total_unit) or 0.0)
    part_factor = _area_factor(unit)
    if part_factor is None:
        raise AssertionSkipped(f"分区面积单位 {unit} 无法换算成平方米")
    part_sum_m2 = sum(_number(item, "分区面积") for item in values) * part_factor
    tol = _rel(spec, 0.02)
    abs_tol = float(spec.get("abs_tol", 0.0) or 0.0)
    findings: list[Finding] = []
    ok, rel = _closeness(part_sum_m2, total_m2, tol, abs_tol)
    if not ok:
        findings.append(
            _mismatch(
                "conservation.area_mismatch",
                "conservation",
                f"分区面积合计 {_fmt(part_sum_m2)} m² 与量测总面积 {_fmt(total_m2)} m² 差 {_pct(rel)}"
                f"（容差 {_pct(tol)}）",
                claimed=part_sum_m2,
                measured=total_m2,
            )
        )
    # 合计相等不代表分区之间没换：有逐分区量测就逐区比（DAIR-191 缺陷 5）
    claimed_parts = {str(k): v for k, v in raw_parts.items()} if isinstance(raw_parts, Mapping) else {}
    candidates = _measured_parts(spec, facts, claim, unit, claimed_parts, tol, abs_tol)
    # claim 只给了分区数组（没标签）时逐区无从比：必须落到下面的 warning，不能静默通过（DAIR-203）
    if candidates is not None and candidates.chosen is not None and claimed_parts:
        measured, label = candidates.chosen
        problems = _part_problems(claimed_parts, measured, tol, abs_tol)
        if problems:
            detail = (
                f"分区面积逐区与 {_short_key(label)} 对不上："
                + "；".join(problems[:4])
                + ("…" if len(problems) > 4 else "")
            )
            if len(candidates.labels) > 1:
                detail += (
                    "；{} 个分组键都能对上 claim 的分区标签（{}），逐区都比过、全都对不上，"
                    "这里挂最贴合的 {}".format(
                        len(candidates.labels),
                        " / ".join(_short_key(item) for item in candidates.labels),
                        _short_key(label),
                    )
                )
            findings.append(
                _mismatch(
                    "conservation.partition_mismatch",
                    "conservation",
                    detail,
                    claimed=claimed_parts,
                    measured=measured,
                )
            )
    elif not findings:
        # 「合计对」不能当成「分区对」：写进 findings（warning）而不是静默通过
        if candidates is not None and candidates.note:
            # spec 声明了 group_field、但量测里没有那个分组键：别拿别的键凑数
            unverified = "{}，{} 个分区面积未逐区核对".format(candidates.note, len(values))
        elif candidates is not None and len(candidates.labels) == 1:
            # 只有一个候选键时「分不清该比哪个」不成立（DAIR-203）；
            # 来源要分清：``group_sums`` 分组键 vs spec 显式给的 ``source_path`` 量测（DAIR-205）
            label = candidates.labels[0]
            if isinstance(label, tuple):
                measured_from = "source_path 量测（{}）已量到".format(_short_key(label))
            else:
                measured_from = "1 个面积分组键（{}）已量到".format(_short_key(label))
            unverified = (
                "{}，claim 的分区没给标签，{} 个分区面积未逐区核对".format(measured_from, len(values))
            )
        elif candidates is not None:
            # 歧义回落单独一句：group_sums 明明在、group_fields 也声明了，不能再讲「只有总面积」（DAIR-201）
            unverified = (
                "{} 个面积分组键（{}）都已量到，claim 的分区没给标签，分不清该比哪个，"
                "{} 个分区面积未逐区核对".format(
                    len(candidates.labels),
                    " / ".join(_short_key(item) for item in candidates.labels),
                    len(values),
                )
            )
        else:
            unverified = (
                f"只有总面积 {_fmt(total_m2)} m² 的量测，{len(values)} 个分区面积未逐区核对"
                "（让 collect_facts 声明 group_fields 输出分区面积汇总，或把分区面积汇总路径传给 source_path）"
            )
        findings.append(
            _mismatch(
                "conservation.partition_unverified",
                "conservation",
                unverified,
                severity="warning",
            )
        )
    return findings


def _count_matches(spec, facts, claim) -> list[Finding]:
    claimed = _number(_metric(claim, str(spec.get("metric", "feature_count") or "feature_count")), "声明要素数")
    if spec.get("expected") is not None:
        measured = _number(spec.get("expected"), "expected")
        source = "任务声明的期望值"
    else:
        measured = _number(_fact(facts, spec.get("source_path"), "source_path"), "量测要素数")
        source = "数据量测"
    if int(round(claimed)) == int(round(measured)):
        return []
    return [
        _mismatch(
            "conservation.count_mismatch",
            "conservation",
            f"声明要素数 {_fmt(claimed)} ≠ {source} {_fmt(measured)}",
            claimed=claimed,
            measured=measured,
        )
    ]


# ---------------------------------------------------------------- 拓扑
def _geometry_valid(spec, facts, claim) -> list[Finding]:
    raw = _fact(facts, spec.get("source_path"), "source_path")
    if isinstance(raw, Mapping):
        raw = raw.get("invalid_count", raw.get("invalid", _MISSING))
        if raw is _MISSING:
            raise AssertionSkipped("拓扑事实里没有 invalid_count")
    invalid = _number(raw, "非法几何数")
    limit = float(spec.get("max_invalid", 0) or 0)
    if invalid <= limit:
        return []
    return [
        _mismatch(
            "topology.invalid_geometry",
            "topology",
            f"结果中有 {int(invalid)} 个自相交/非法几何（上限 {int(limit)}）",
            measured=invalid,
        )
    ]


def _no_overlap(spec, facts, claim) -> list[Finding]:
    raw = _fact(facts, spec.get("source_path"), "source_path")
    if isinstance(raw, Mapping):
        key = str(spec.get("metric", "overlap_ratio") or "overlap_ratio")
        ratio = raw.get(key)
        if ratio is None and key == "overlap_ratio":
            overlap = raw.get("overlap_area")
            total = raw.get("area_total")
            if overlap is not None and total:
                ratio = _number(overlap, "overlap_area") / _number(total, "area_total")
        if ratio is None:
            raise AssertionSkipped("拓扑事实里没有 overlap_ratio")
    else:
        ratio = raw
    ratio = _number(ratio, "重叠率")
    limit = float(spec.get("max_ratio", 0.001) or 0.0)
    findings: list[Finding] = []
    if ratio > limit:
        findings.append(
            _mismatch(
                "topology.overlap",
                "topology",
                f"结果面重叠 {_pct(ratio)}（上限 {_pct(limit)}）——同一块面积被算了不止一次",
                measured=ratio,
            )
        )
    # 自报重叠率必须与量测一致：只查上限的话，「实测 0.42%、结论写 0.0%」会漏
    claim_key = str(spec.get("claim_metric", "overlap_ratio") or "overlap_ratio")
    declared = claim.metrics.get(claim_key)
    if declared is None:
        if findings:
            return findings
        raise AssertionSkipped(f"claim.metrics 缺少 {claim_key}，无法比对自报重叠率")
    declared = _number(declared, "自报重叠率")
    abs_tol = float(spec.get("claim_abs_tol", 0.0005) or 0.0)
    ok, _ = _closeness(declared, ratio, _rel(spec, 0.1), abs_tol)
    if not ok:
        findings.append(
            _mismatch(
                "topology.overlap_claim_mismatch",
                "topology",
                f"结论写重叠率 {_pct(declared)}，量测是 {_pct(ratio)}"
                f"（容差 {_pct(abs_tol)}）——自报值与数据对不上",
                claimed=declared,
                measured=ratio,
            )
        )
    return findings


def _coverage(spec, facts, claim) -> list[Finding]:
    raw = _fact(facts, spec.get("source_path"), "source_path")
    if not isinstance(raw, Mapping):
        raise ValueError("source_path 必须指向含覆盖率事实的字典")
    metric = str(spec.get("metric", "gap_ratio") or "gap_ratio")
    value = raw.get(metric)
    if value is None:
        if metric == "gap_ratio" and raw.get("coverage_ratio") is not None:
            value = 1.0 - _number(raw["coverage_ratio"], "coverage_ratio")
        else:
            raise AssertionSkipped(f"拓扑事实里没有 {metric}")
    value = _number(value, metric)
    limit = float(spec.get("max_ratio", 0.01) or 0.0)
    if value <= limit:
        return []
    return [
        _mismatch(
            "topology.coverage_gap",
            "topology",
            f"结果面覆盖存在 {_pct(value)} 的缺口/未覆盖（上限 {_pct(limit)}）",
            measured=value,
        )
    ]


def _within_extent(spec, facts, claim) -> list[Finding]:
    result = _fact(facts, spec.get("result_path"), "result_path")
    extent = _fact(facts, spec.get("extent_path"), "extent_path")
    if not (isinstance(result, (list, tuple)) and isinstance(extent, (list, tuple))):
        raise ValueError("result_path / extent_path 必须指向 [minx,miny,maxx,maxy]")
    if len(result) != 4 or len(extent) != 4:
        raise ValueError("bbox 必须是 4 个数")
    r = [_number(x, "bbox") for x in result]
    e = [_number(x, "bbox") for x in extent]
    tol = _rel(spec, 0.001)
    width = max(abs(e[2] - e[0]), abs(e[3] - e[1]), 1e-9)
    pad = width * tol
    if r[0] >= e[0] - pad and r[1] >= e[1] - pad and r[2] <= e[2] + pad and r[3] <= e[3] + pad:
        return []
    return [
        _mismatch(
            "topology.out_of_extent",
            "topology",
            f"结果范围 {r} 超出研究区 {e}（容差 {_pct(tol)}）",
            claimed=r,
            measured=e,
        )
    ]


_COUNT_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _reason_numbers(text: str) -> list[float]:
    """取解释文字里的数字（``8,420`` / ``118→113`` / ``5 条`` 都能取到）。"""
    values: list[float] = []
    for token in _COUNT_NUMBER_RE.findall(str(text or "")):
        try:
            values.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return values


def _count_change(spec, facts, claim) -> list[Finding]:
    result = _number(_fact(facts, spec.get("result_path"), "result_path"), "结果要素数")
    source = _number(_fact(facts, spec.get("source_path"), "source_path"), "源要素数")
    delta = result - source
    allowed = float(spec.get("allowed_delta", 0) or 0)
    if abs(delta) <= allowed:
        return []
    reason_key = str(spec.get("reason_key", "count_change") or "count_change")
    reason = claim.expected.get(reason_key)
    reason_text = reason.strip() if isinstance(reason, str) else ""
    if reason_text and not spec.get("require_quantified", True):
        return []
    if reason_text:
        # 「空口解释」不算解释：给出的数字必须和这次变化对得上（源数/结果数/差值）
        targets = (source, result, abs(delta))
        if any(
            abs(value - target) <= 1e-6 * max(1.0, abs(target))
            for value in _reason_numbers(reason_text)
            for target in targets
        ):
            return []
        detail = (
            f"要素数从 {_fmt(source)} 变为 {_fmt(result)}（差 {_fmt(delta)}），"
            f"解释「{reason_text}」没有给出与这次变化对得上的数量"
        )
    else:
        detail = (
            f"要素数从 {_fmt(source)} 变为 {_fmt(result)}（差 {_fmt(delta)}），"
            f"但 claim.expected.{reason_key} 没有给出可解释的原因"
        )
    return [
        _mismatch(
            "topology.unexplained_count_change",
            "topology",
            detail,
            claimed=result,
            measured=source,
        )
    ]


#: 方向相反的一对关系（断言能拿另一个方向的事实判定“写反了”）。
_INVERSE_RELATION = {"contains": "within", "within": "contains"}


def _relation_features(raw: Any, key: str) -> float | None:
    if isinstance(raw, Mapping):
        value = raw.get(key)
        return None if value is None else _number(value, "量测要素数")
    return _number(raw, "量测要素数")


def _relation_count(spec, facts, claim) -> list[Finding]:
    relation = str(spec.get("relation", "") or "").strip()
    if not relation:
        raise ValueError("relation 不能为空（adjacent / contains / within / intersects…）")
    metric_name = str(spec.get("metric", "") or f"{relation}_count")
    claimed = _number(_metric(claim, metric_name), f"声明 {relation} 要素数")
    key = str(spec.get("key", "features") or "features")
    path = spec.get("source_path")
    raw = _lookup(facts, path)
    if raw is _MISSING or raw is None:
        # 方向反：同一处事实里只有反方向（如量测 within、结论写 contains）。
        # 数量相同也抓得到，这才是“关系方向”这一类错误的形态（DAIR-191 缺陷 4）。
        inverse = _INVERSE_RELATION.get(relation.lower())
        swapped = (
            list(path[:-1]) + [inverse]
            if inverse and isinstance(path, (list, tuple)) and path and str(path[-1]).lower() == relation.lower()
            else None
        )
        sibling = _lookup(facts, swapped) if swapped else _MISSING
        if sibling is not _MISSING and sibling is not None:
            measured = _relation_features(sibling, key)
            if measured is not None:
                return [
                    _mismatch(
                        "topology.relation_direction_mismatch",
                        "topology",
                        f"量测到的关系是反方向「{inverse}」（{_fmt(measured)} 个要素），"
                        f"声明的是「{relation}」（{_fmt(claimed)} 个）——方向写反，或该方向没有量测支撑",
                        claimed=claimed,
                        measured=measured,
                    )
                ]
        raise AssertionSkipped(f"缺少量测事实 {path}")
    measured = _relation_features(raw, key)
    if measured is None:
        raise AssertionSkipped(f"{relation} 事实里没有 {key}")
    ok, _ = _closeness(claimed, measured, _rel(spec, 0.01), float(spec.get("abs_tol", 0.5) or 0.0))
    if ok:
        return []
    return [
        _mismatch(
            "topology.relation_mismatch",
            "topology",
            f"声明 {_fmt(claimed)} 个要素存在「{relation}」关系，量测只有 {_fmt(measured)} 个",
            claimed=claimed,
            measured=measured,
        )
    ]


# ---------------------------------------------------------------- 量纲
def _crs_units(spec, facts, claim) -> list[Finding]:
    raw = _fact(facts, spec.get("source_path"), "source_path")
    if not isinstance(raw, Mapping):
        raise ValueError("source_path 必须指向 crs 字典")
    projected = raw.get("is_projected")
    if projected is None:
        raise AssertionSkipped("crs 事实里没有 is_projected")
    quantity = str(spec.get("quantity", "area") or "area")
    unit = spec.get("unit") or claim.units.get(quantity)
    if not unit:
        raise AssertionSkipped(f"claim.units 没有声明 {quantity} 单位")
    unit = _norm_unit(unit)
    if quantity == "area":
        if unit not in _AREA_TO_M2:
            raise AssertionSkipped(f"无法识别的面积单位 {unit}")
        metric = _area_factor(unit) is not None
    elif quantity == "distance":
        if unit not in _DIST_TO_M:
            raise AssertionSkipped(f"无法识别的距离单位 {unit}")
        metric = _dist_factor(unit) is not None
    else:
        raise ValueError(f"quantity 只能是 area/distance，收到 {quantity}")

    if bool(projected) and not metric:
        return [
            _mismatch(
                "dimension.crs_unit_mismatch",
                "dimension",
                f"数据是投影坐标系，{quantity} 却声明为 {unit}（度）——米/度混用会把量级算错",
                claimed=unit,
                measured=raw,
            )
        ]
    if not bool(projected) and metric:
        return [
            _mismatch(
                "dimension.crs_unit_mismatch",
                "dimension",
                f"数据是地理坐标系（度），{quantity} 却声明为 {unit}（米制）——"
                "面积/距离没有做投影就直接按米算",
                claimed=unit,
                measured=raw,
            )
        ]
    return []


def _area_unit(spec, facts, claim) -> list[Finding]:
    claimed = _number(_metric(claim, str(spec.get("metric", "") or "")), "声明面积")
    measured = _number(_fact(facts, spec.get("source_path"), "source_path"), "量测面积")
    source_unit = _norm_unit(spec.get("source_unit") or "m2")
    claim_unit = _norm_unit(spec.get("unit") or claim.units.get("area") or "")
    if not claim_unit:
        raise AssertionSkipped("claim.units.area 未声明")
    if claim_unit not in _AREA_TO_M2:
        raise AssertionSkipped(f"无法识别的面积单位 {claim_unit}")
    if source_unit not in _AREA_TO_M2:
        raise AssertionSkipped(f"无法识别的量测面积单位 {source_unit}")
    measured_factor = _area_factor(source_unit)
    claim_factor = _area_factor(claim_unit)
    if measured_factor is None or claim_factor is None:
        raise AssertionSkipped("面积量测与声明单位不可比（度 vs 米）")
    measured_converted = measured * measured_factor / claim_factor
    tol = _rel(spec, 0.02)
    ok, rel = _closeness(claimed, measured_converted, tol, float(spec.get("abs_tol", 0.0) or 0.0))
    if ok:
        return []
    return [
        _mismatch(
            "dimension.area_unit_mismatch",
            "dimension",
            f"声明面积 {_fmt(claimed)} {claim_unit} 与量测 {_fmt(measured)} {source_unit}"
            f"（= {_fmt(measured_converted)} {claim_unit}）差 {_pct(rel)}——单位/换算系数对不上",
            claimed=claimed,
            measured=measured_converted,
        )
    ]


def _distance_unit(spec, facts, claim) -> list[Finding]:
    claimed = _number(_metric(claim, str(spec.get("metric", "") or "")), "声明距离")
    measured = _number(_fact(facts, spec.get("source_path"), "source_path"), "量测距离")
    source_unit = _norm_unit(spec.get("source_unit") or "m")
    claim_unit = _norm_unit(spec.get("unit") or claim.units.get("distance") or "")
    if not claim_unit:
        raise AssertionSkipped("claim.units.distance 未声明")
    if claim_unit not in _DIST_TO_M:
        raise AssertionSkipped(f"无法识别的距离单位 {claim_unit}")
    if source_unit not in _DIST_TO_M:
        raise AssertionSkipped(f"无法识别的量测距离单位 {source_unit}")
    measured_factor = _dist_factor(source_unit)
    claim_factor = _dist_factor(claim_unit)
    if measured_factor is None or claim_factor is None:
        raise AssertionSkipped("距离量测与声明单位不可比（度 vs 米）")
    measured_converted = measured * measured_factor / claim_factor
    tol = _rel(spec, 0.02)
    ok, rel = _closeness(claimed, measured_converted, tol, float(spec.get("abs_tol", 0.0) or 0.0))
    if ok:
        return []
    return [
        _mismatch(
            "dimension.distance_unit_mismatch",
            "dimension",
            f"声明距离 {_fmt(claimed)} {claim_unit} 与量测 {_fmt(measured)} {source_unit}"
            f"（= {_fmt(measured_converted)} {claim_unit}）差 {_pct(rel)}——距离单位/换算系数对不上",
            claimed=claimed,
            measured=measured_converted,
        )
    ]


def _density_cross_checks(
    spec: Mapping[str, Any],
    facts: Mapping[str, Any],
    claim: ReviewClaim,
    *,
    count_name: str,
    raw_count: Any,
    count: float,
    area_name: str,
    raw_area: Any,
    area: float,
    area_unit: str,
) -> list[Finding]:
    """把密度的分子/分母拉回源数据对账（DAIR-191 缺陷 1、缺陷 2）。

    只用 claim 内部三个数复算，口径错（人口按“十人”为单位、面积被截断）时三个数
    依然自洽，永远抓不到；必须回到量测值比。容差用“声明位数”而不是 2%：
    ``152.0`` 声称精确到 0.1，就不能容下 0.34 的偏差（带 0.1% 地板防误杀）。
    """
    min_rel = float(spec.get("cross_rel_tol", 0.001) or 0.0)
    findings: list[Finding] = []
    # DAIR-193：启发式分不清「分子/分母写错了」和「同名但不同量」（研究区人口 vs 绿地服务
    # 范围内人口、结果层 vs 参考层）。口径明确（只声明了一个数据集且它对上了 facts）时同名
    # 量测无歧义，照旧报 error；声明了多个或路径对不上时无法确定这个数出自哪层 → 降为 warning。
    # 显式给 count_source_path / area_source_path 或 cross_check="strict" 一律按 error 报。
    scope_exact = _dataset_scope(facts, claim)[1]
    auto_severity = (
        "error" if scope_exact or str(spec.get("cross_check", "")) == "strict" else "warning"
    )
    auto_hint = "（按名字自动匹配，声明了多个数据集时分不清同名不同量；确认口径请显式给 source_path）"

    explicit_count = spec.get("count_source_path")
    if explicit_count is not None:
        raw_measured_count = _fact(facts, explicit_count, "count_source_path")
        if isinstance(raw_measured_count, Mapping):
            numbers = [_number(item, str(key)) for key, item in raw_measured_count.items()]
            candidates = [(f"{explicit_count}（{len(numbers)} 组之和）", float(sum(numbers)))]
        else:
            candidates = [(str(explicit_count), _number(raw_measured_count, "量测计数"))]
        severity = "error"
    elif count_name and raw_count is not None:
        candidates = _source_measures(facts, claim, _name_candidates(count_name))
        severity = auto_severity
    else:
        candidates = []
        severity = auto_severity
    if candidates and raw_count is not None:
        tol = _precision_tolerance(raw_count, count, min_rel)
        if not any(_closeness(count, value, tol)[0] for _, value in candidates):
            label, value = candidates[0]
            rel = abs(count - value) / (abs(value) or 1.0)
            extra = f"（共 {len(candidates)} 处同名量测，全部不符）" if len(candidates) > 1 else ""
            findings.append(
                _mismatch(
                    "dimension.density_mismatch",
                    "dimension",
                    f"密度的分子「{count_name or 'count'}」与源数据不符：声明 {_fmt(count)}，"
                    f"{label} 量测 {_fmt(value)}（差 {_pct(rel)}，容差 {_pct(tol)}）{extra}"
                    "——数字之间自洽也可能分子口径就错了"
                    + (auto_hint if severity != "error" else ""),
                    claimed=count,
                    measured=value,
                    severity=severity,
                )
            )
    elif count_name and raw_count is not None:
        # 声明域内 0 候选、域外有同名量测：不静默通过，也不报错（没对过账，但不是错了）。
        # 与「路径漂移退回全部 + warning」对称（DAIR-195）。
        outside = _source_measures(
            facts, claim, _name_candidates(count_name), datasets=_out_of_scope_datasets(facts, claim)
        )
        if outside:
            shown = "、".join(label for label, _ in outside[:3]) + ("…" if len(outside) > 3 else "")
            findings.append(
                _mismatch(
                    "dimension.density_unverified",
                    "dimension",
                    f"声明域内无同名量测：「{count_name}」只在 {len(outside)} 处未声明的数据集里量到"
                    f"（{shown}），本次未参与对账——确认口径请把该数据集加进 claim.data_paths"
                    "或显式给 count_source_path",
                    severity="warning",
                )
            )

    if area_name and raw_area is not None:
        area_candidates = _measured_area(spec, facts, claim, area_name, area_unit)
        if area_candidates:
            evaluated = [
                (value, label, *_closeness(area, value, _precision_tolerance(raw_area, value, min_rel)))
                for value, label in area_candidates
            ]
            if not any(ok for _, _, ok, _ in evaluated):
                value, label, _, rel = min(evaluated, key=lambda item: item[3])
                tol = _precision_tolerance(raw_area, value, min_rel)
                extra = (
                    f"（共 {len(area_candidates)} 处面积量测，全部不符）"
                    if len(area_candidates) > 1 else ""
                )
                area_severity = "error" if spec.get("area_source_path") is not None else auto_severity
                findings.append(
                    _mismatch(
                        "dimension.density_mismatch",
                        "dimension",
                        f"密度的分母（{area_name}）与量测面积不符：声明 {_fmt(area)} {area_unit}，"
                        f"{label} 量测 {_fmt(value)} {area_unit}（差 {_pct(rel)}，容差 {_pct(tol)}）{extra}"
                        "——面积被截断/取整后仍拿它算密度"
                        + (auto_hint if area_severity != "error" else ""),
                        claimed=area,
                        measured=value,
                        severity=area_severity,
                    )
                )
        else:
            # 与分子侧对称（DAIR-199）：声明域内一个面积量测都没有、域外有时不静默通过。
            # 没对过账不等于错了，所以是 warning，不动 ok 判据。
            outside = _measured_area(
                spec,
                facts,
                claim,
                area_name,
                area_unit,
                datasets=_out_of_scope_datasets(facts, claim),
            )
            if outside:
                shown = "、".join(label for _, label in outside[:3]) + ("…" if len(outside) > 3 else "")
                findings.append(
                    _mismatch(
                        "dimension.density_unverified",
                        "dimension",
                        f"声明域内无量测面积：「{area_name}」只在 {len(outside)} 处未声明的数据集里量到"
                        f"（{shown}），本次未参与对账——确认口径请把该数据集加进 claim.data_paths"
                        "或显式给 area_source_path",
                        severity="warning",
                    )
                )
    return findings


def _density(spec, facts, claim) -> list[Finding]:
    density = _number(_metric(claim, str(spec.get("density", "density") or "density")), "声明密度")
    count_name = str(spec.get("count", "") or "")
    raw_count: Any = None
    if count_name:
        raw_count = _metric(claim, count_name)
        count = _number(raw_count, "计数")
    else:
        count = _number(_fact(facts, spec.get("count_path"), "count_path"), "量测计数")
    per_unit = _norm_unit(spec.get("per_area_unit") or "km2")
    per_factor = _area_factor(per_unit)
    if per_factor is None:
        raise AssertionSkipped(f"per_area_unit={per_unit} 无法换算")

    area_name = str(spec.get("area", "") or "")
    raw_area: Any = None
    if area_name:
        raw_area = _metric(claim, area_name)
        area = _number(raw_area, "面积")
        area_unit = _norm_unit(spec.get("area_unit") or claim.units.get("area") or "m2")
    else:
        area = _number(_fact(facts, spec.get("area_path"), "area_path"), "量测面积")
        area_unit = _norm_unit(spec.get("area_source_unit") or "m2")
    area_factor = _area_factor(area_unit)
    if area_factor is None:
        raise AssertionSkipped(f"面积单位 {area_unit} 不可换算成米制")
    area_in_per_unit = area * area_factor / per_factor
    if area_in_per_unit == 0:
        raise AssertionSkipped("面积为 0，无法算密度")
    expected = count / area_in_per_unit
    tol = _rel(spec, 0.02)
    findings: list[Finding] = []
    ok, rel = _closeness(density, expected, tol)
    if not ok:
        findings.append(
            _mismatch(
                "dimension.density_mismatch",
                "dimension",
                f"声明密度 {_fmt(density)}/ {per_unit} 与「{_fmt(count)} ÷ {_fmt(area_in_per_unit)} {per_unit}」"
                f"= {_fmt(expected)} 差 {_pct(rel)}——分子/分母口径不一致",
                claimed=density,
                measured=expected,
            )
        )
    if bool(spec.get("cross_check", True)):
        findings.extend(
            _density_cross_checks(
                spec,
                facts,
                claim,
                count_name=count_name,
                raw_count=raw_count,
                count=count,
                area_name=area_name,
                raw_area=raw_area,
                area=area,
                area_unit=area_unit,
            )
        )
    return findings


def _precision(spec, facts, claim) -> list[Finding]:
    """声明精度 vs 量测值：写了几位小数，就只能要求精确到那一位（带 0.1% 地板）。

    ``dimension.area_unit`` 的 2% 容差是给单位换算/投影差异的，会把
    「152.34 写成 152.0」（0.22%）放过去；这条断言专管“位数写高了”（DAIR-191 缺陷 2）。
    0.1% 地板与 ``dimension.density`` 的分子/分母勾稽同一口径。
    """
    metric_name = str(spec.get("metric", "") or "")
    raw_claimed = _metric(claim, metric_name)
    claimed = _number(raw_claimed, "声明值")
    measured_raw = _number(_fact(facts, spec.get("source_path"), "source_path"), "量测值")
    quantity = str(spec.get("quantity", "area") or "area")
    if quantity not in ("area", "distance"):
        raise ValueError(f"quantity 只能是 area/distance，收到 {quantity}")
    claim_unit = _norm_unit(spec.get("unit") or claim.units.get(quantity) or ("m2" if quantity == "area" else "m"))
    source_unit = _norm_unit(spec.get("source_unit") or ("m2" if quantity == "area" else "m"))
    measured = _to_unit(measured_raw, source_unit, claim_unit, quantity)
    if measured is None:
        raise AssertionSkipped(f"量测单位 {source_unit} 与声明单位 {claim_unit} 不可换算")
    min_rel = float(spec.get("min_rel", 0.001) or 0.0)
    tol = _precision_tolerance(raw_claimed, measured, min_rel)
    ok, rel = _closeness(claimed, measured, tol)
    if ok:
        return []
    decimals = _decimals(raw_claimed)
    resolution = f"{0.5 * 10 ** (-decimals):g}" if decimals is not None else "未知"
    digits = f"{decimals} 位小数" if decimals is not None else "位数未知"
    return [
        _mismatch(
            "dimension.precision_mismatch",
            "dimension",
            f"声明 {_fmt_declared(raw_claimed)} {claim_unit} 写到 {digits}（允许 ±{resolution}），"
            f"量测 {_fmt_precise(measured)} {claim_unit} 差 {_fmt_precise(abs(claimed - measured))}（{_pct(rel)}）"
            "——精度声明与量测不符（截断/四舍五入后仍按更高位数上报）",
            claimed=claimed,
            measured=measured,
        )
    ]


def _value_range(spec, facts, claim) -> list[Finding]:
    value = _number(_metric(claim, str(spec.get("metric", "") or "")), "声明值")
    if spec.get("min_path") is not None or spec.get("max_path") is not None:
        low = _number(_fact(facts, spec.get("min_path"), "min_path"), "量测最小值")
        high = _number(_fact(facts, spec.get("max_path"), "max_path"), "量测最大值")
    else:
        raw = _fact(facts, spec.get("source_path"), "source_path")
        if not isinstance(raw, Mapping) or raw.get("min") is None or raw.get("max") is None:
            raise AssertionSkipped("range 事实里没有 min/max")
        low = _number(raw["min"], "min")
        high = _number(raw["max"], "max")
    if low > high:
        low, high = high, low
    slack = _rel(spec, 0.01)
    pad = max(abs(low), abs(high), 1.0) * slack
    if low - pad <= value <= high + pad:
        return []
    return [
        _mismatch(
            "dimension.value_out_of_range",
            "dimension",
            f"声明值 {_fmt(value)} 超出量测范围 [{_fmt(low)}, {_fmt(high)}]（容差 {_pct(slack)}）",
            claimed=value,
            measured=[low, high],
        )
    ]


def _raster_cell_area(spec, facts, claim) -> list[Finding]:
    claimed = _number(_metric(claim, str(spec.get("metric", "") or "")), "声明面积")
    count = _number(_fact(facts, spec.get("count_path"), "count_path"), "像元数")
    cell = _number(_fact(facts, spec.get("cell_size_path"), "cell_size_path"), "像元大小")
    measured_m2 = count * cell * cell
    claim_unit = _norm_unit(spec.get("unit") or claim.units.get("area") or "m2")
    factor = _area_factor(claim_unit)
    if factor is None:
        raise AssertionSkipped(f"面积单位 {claim_unit} 无法换算成平方米")
    measured_converted = measured_m2 / factor
    tol = _rel(spec, 0.02)
    ok, rel = _closeness(claimed, measured_converted, tol)
    if ok:
        return []
    return [
        _mismatch(
            "dimension.cell_area_mismatch",
            "dimension",
            f"声明面积 {_fmt(claimed)} {claim_unit} 与「{_fmt(count)} 像元 × ({_fmt(cell)} m)²」"
            f"= {_fmt(measured_converted)} {claim_unit} 差 {_pct(rel)}——像元尺寸/面积口径对不上",
            claimed=claimed,
            measured=measured_converted,
        )
    ]


#: 断言清单（顺序即 prompt 里的展示顺序）。新增断言只在这里注册。
ASSERTIONS: dict[str, Assertion] = {
    item.name: item
    for item in [
        Assertion(
            "conservation.sum_equals",
            "conservation",
            "分项合计必须等于声明的总量（人口/面积/计数不凭空多出或消失）",
            'total="total_pop", parts="zone_pop", rel_tol=0.01',
            _sum_equals,
        ),
        Assertion(
            "conservation.matches_source",
            "conservation",
            "声明总量必须等于从源数据量测到的总量",
            'metric="total_pop", source_path=[...], scale=1.0, rel_tol=0.01',
            _matches_source,
        ),
        Assertion(
            "conservation.partition_matches_source",
            "conservation",
            "每个分区的值必须与源数据分组汇总一致（抓“把 A 区人口搬到 B 区”）",
            'metric="zone_pop", source_path=[...], rel_tol=0.01, ignore_keys=[]',
            _partition_matches_source,
        ),
        Assertion(
            "conservation.area_partition",
            "conservation",
            "分区面积合计必须等于量测到的总面积（口径/重叠/遗漏）；有逐分区量测时同时逐区核对"
            "（合计相等但分区之间互换也能抓）",
            'parts="zone_area", total_path=[...], unit="km2", total_unit="m2", rel_tol=0.02, '
            'source_path=[... group_sums.AREA_M2]（可选，逐分区量测）',
            _area_partition,
        ),
        Assertion(
            "conservation.count_matches",
            "conservation",
            "声明要素数必须等于源数据/期望的要素数",
            'metric="feature_count", source_path=[...]（或 expected=118）',
            _count_matches,
        ),
        Assertion(
            "topology.geometry_valid",
            "topology",
            "结果面不得自相交/非法",
            'source_path=[... invalid_count], max_invalid=0',
            _geometry_valid,
        ),
        Assertion(
            "topology.no_overlap",
            "topology",
            "结果面之间不得重叠（除业务允许），且结论自报的重叠率必须与量测一致",
            'source_path=[... overlap_ratio], metric="overlap_ratio", max_ratio=0.001, '
            'claim_metric="overlap_ratio", claim_abs_tol=0.0005',
            _no_overlap,
        ),
        Assertion(
            "topology.coverage",
            "topology",
            "结果面覆盖研究区时不得有大块缺口",
            'source_path=[...], metric="gap_ratio", max_ratio=0.01',
            _coverage,
        ),
        Assertion(
            "topology.within_extent",
            "topology",
            "结果范围必须落在研究区范围内",
            'result_path=[... bbox], extent_path=[... bbox]',
            _within_extent,
        ),
        Assertion(
            "topology.count_change",
            "topology",
            "要素数变化必须有带数量的解释（溶解/合并/筛选），空口解释按异常处理",
            'result_path=[... count], source_path=[... count], allowed_delta=0, '
            'reason_key="count_change", require_quantified=true',
            _count_change,
        ),
        Assertion(
            "topology.relation_count",
            "topology",
            "声明「有多少要素与其他要素相邻/包含/被包含」必须等于量测到的数量；方向写反（量测 within、"
            "结论写 contains）也算不符",
            'relation="adjacent", metric="adjacent_count", '
            'source_path=[... relations.adjacent], key="features"',
            _relation_count,
        ),
        Assertion(
            "dimension.crs_units",
            "dimension",
            "CRS 单位必须与声明的量纲匹配（米 vs 度）",
            'source_path=[... crs], quantity="area", unit="km2"',
            _crs_units,
        ),
        Assertion(
            "dimension.area_unit",
            "dimension",
            "声明面积必须与量测面积按正确单位换算后一致（抓“差一个数量级”）",
            'metric="area_km2", source_path=[... area], unit="km2", source_unit="m2", rel_tol=0.02',
            _area_unit,
        ),
        Assertion(
            "dimension.distance_unit",
            "dimension",
            "声明的距离必须与量测距离按正确单位换算后一致（抓「500 米写成 500 公里」）",
            'metric="buffer_distance", source_path=[... distance], unit="m", source_unit="m", rel_tol=0.02',
            _distance_unit,
        ),
        Assertion(
            "dimension.density",
            "dimension",
            "密度必须等于计数 ÷ 面积、分母单位一致，且分子/分母都要与源数据量测对得上"
            "（防“人口按十人为单位/面积被截断”这类自洽的错误口径）",
            'density="density_per_km2", count="pop", area="area_km2", per_area_unit="km2", '
            'count_source_path=[...]（可选）, area_source_path=[...]（可选）, cross_rel_tol=0.001',
            _density,
        ),
        Assertion(
            "dimension.precision",
            "dimension",
            "声明值写了几位小数就只能要求精确到那一位（抓 152.34 写成 152.0 这类截断后仍按高位报）",
            'metric="area_km2", source_path=[... area_total], quantity="area", unit="km2", '
            'source_unit="m2", min_rel=0.001',
            _precision,
        ),
        Assertion(
            "dimension.value_range",
            "dimension",
            "声明的最小/最大值必须落在量测值域内",
            'metric="max_value", source_path=[... {min,max}], slack=0.01',
            _value_range,
        ),
        Assertion(
            "dimension.raster_cell_area",
            "dimension",
            "栅格面积必须与「像元数 × 像元尺寸²」一致",
            'metric="area_km2", count_path=[...], cell_size_path=[...], unit="km2"',
            _raster_cell_area,
        ),
    ]
}


def assertion_catalog() -> list[dict[str, str]]:
    """给复核 prompt / 文档用的断言清单。"""
    return [
        {
            "type": item.name,
            "family": item.family,
            "summary": item.summary,
            "params": item.params,
        }
        for item in ASSERTIONS.values()
    ]


def run_assertions(
    specs: Iterable[Mapping[str, Any]] | None,
    facts: Mapping[str, Any] | None,
    claim: ReviewClaim | Mapping[str, Any] | None = None,
) -> ReviewReport:
    """独立跑一组断言（纯函数：不读盘、不联网、不碰 arcpy）。

    Args:
        specs: 断言声明列表，每条至少含 ``type``；空/None 时用 ``claim.assertions``。
        facts: 量测事实（``collect_facts`` 的产物，或同构字典）。
        claim: :class:`ReviewClaim` 或它的 ``to_dict()``。

    Returns:
        :class:`ReviewReport`。断言的 ``skipped``（缺事实）与 ``errors``（写错）
        分开记：前者是“没检查”，后者是“检查本身坏了”，都不能当成通过。
    """
    claim_obj = claim if isinstance(claim, ReviewClaim) else ReviewClaim.from_dict(claim)
    raw_specs = list(specs) if specs is not None else list(claim_obj.assertions)
    facts_obj: Mapping[str, Any] = facts if isinstance(facts, Mapping) else {}
    report = ReviewReport()
    for raw in raw_specs:
        if not isinstance(raw, Mapping):
            report.errors.append(f"断言不是对象：{raw!r}")
            continue
        name = str(raw.get("type", "") or "").strip()
        spec = dict(raw)
        spec["type"] = name
        definition = ASSERTIONS.get(name)
        if definition is None:
            report.errors.append(f"未知断言类型：{name or '(空)'}")
            continue
        try:
            findings = list(definition.run(spec, facts_obj, claim_obj) or [])
        except AssertionSkipped as exc:
            report.skipped.append(f"{name}: {exc}")
            continue
        except Exception as exc:  # 断言自己坏掉要留痕，不能静默放行
            report.errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        report.checked.append(name)
        for finding in findings:
            if not finding.family:
                finding.family = definition.family
            if finding.assertion is None:
                finding.assertion = spec
            report.findings.append(finding)
    return report


__all__ = [
    "ASSERTIONS",
    "Assertion",
    "AssertionSkipped",
    "assertion_catalog",
    "run_assertions",
]

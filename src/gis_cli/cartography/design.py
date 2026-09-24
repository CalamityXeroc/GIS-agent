# -*- coding: utf-8 -*-
"""制图设计规则引擎：``facts`` + ``intent`` → ``LayoutSpec``。

职责分工：
- 本模块做**决策**（纸张/朝向、图名格式与位置、图例要不要/放哪/多大、比例尺样式与整数刻度、
  指北针样式、配色、标注开关），并给出**每个决定的理由**（decisions）；
- :mod:`render` 只负责把 spec 落到 ArcGIS，不含任何判断；
- 纯 Python（不 import arcpy），因此可以用假 facts 做矩阵单测。

设计原则（来自用户参考图与制图惯例）：
1. 数据长宽比决定纸张朝向；近方形数据用竖版（图例/图名需要纵向空间）。
2. 图名顶部居中、字号按"一行放得下"反算并夹在 min/max 之间；太长则分两行。
3. 图例位置由**九宫格占用度**决定（放最空的角，避免压住数据）；数据满覆盖时移到图框外，
   但保证地图主体仍可见（用户要求：能看见图即可）。
4. 比例尺刻度必须是整数（1/2/4×10^k），单位随长度自动在「米/千米」间切换。
5. 图例是否加标题、是否放图例，遵循图种惯例：单一符号不放图例；多图层符号清单不加标题。
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any

from . import facts as facts_mod
from .profiles import DEFAULT_PROFILE_NAME, load_profile

logger = logging.getLogger(__name__)

#: 纸张（竖版基准尺寸，mm）
PAPERS: dict[str, tuple[float, float]] = {"A4": (210.0, 297.0), "A3": (297.0, 420.0), "16:9": (338.0, 190.0)}

_SEMANTIC_LABELS: dict[int, list[str]] = {
    2: ["低", "高"],
    3: ["低", "中", "高"],
    4: ["低", "较低", "较高", "高"],
    5: ["少", "较少", "中", "较多", "多"],
    6: ["很少", "少", "较少", "较多", "多", "很多"],
    7: ["很低", "低", "较低", "中", "较高", "高", "很高"],
}

_MM_PER_PT = 25.4 / 72.0


# --------------------------------------------------------------------- intent

def normalize_intent(intent: dict[str, Any] | None) -> dict[str, Any]:
    """规整意图参数（缺省即自动）。

    注意：这里必须**保留调用方传入的自定义键**（如 ``class_bounds`` 统一分级、
    ``category_names`` 类别显示名）——以前是白名单重建，会把这类键静默丢掉，
    导致“传了参数却看不出效果”（实测：category_names 没进图例、class_bounds 不生效）。
    """
    data = dict(intent or {})
    normalized = dict(data)
    normalized.update(
        {
            "theme": str(data.get("theme", "") or "").strip(),
            "purpose": str(data.get("purpose", "分布") or "分布").strip(),
            "medium": str(data.get("medium", "") or "").strip(),          # print_A4 / screen_16_9 / ""（自动）
            "orientation": str(data.get("orientation", "auto") or "auto").strip(),
            "legend_labels": str(data.get("legend_labels", "") or "").strip(),  # range / semantic / both
            "style_profile": str(data.get("style_profile", "") or "").strip(),
            "title": str(data.get("title", "") or "").strip(),
            "region": str(data.get("region", "") or "").strip(),
            "scale": str(data.get("scale", "") or "").strip(),
            "map_kind": str(data.get("map_kind", "") or "").strip(),
            "classification": str(data.get("classification", "") or "").strip(),
            "field": str(data.get("field", "") or "").strip(),
            "legend_title": str(data.get("legend_title", "") or "").strip(),
            "color_ramp": str(data.get("color_ramp", "") or "").strip(),
            "category_colors": str(data.get("category_colors", "") or "").strip(),
            "category_names": str(data.get("category_names", "") or "").strip(),
            "class_bounds": data.get("class_bounds") or data.get("explicit_bounds") or "",
            "category_order": list(data.get("category_order") or []),
            "overrides": dict(data.get("overrides") or {}),
            "note": str(data.get("note", "") or "").strip(),
        }
    )
    return normalized


# ------------------------------------------------------------------ 长度取整

def nice_length(value: float, *, mantissas: tuple[float, ...] = (1.0, 2.0, 4.0, 5.0)) -> float:
    """把长度吸附到 m×10^k（m∈{1,2,4,5}，含向上跨一档的 10×10^k），用于比例尺整数刻度。"""
    if value <= 0:
        return 0.0
    exponent = math.floor(math.log10(value))
    base = 10.0 ** exponent
    candidates = [mantissa * base for mantissa in mantissas]
    candidates.append(10.0 * base)  # 跨档：5000 与 10000 之间要能选到 10000
    best = min(candidates, key=lambda candidate: abs(value - candidate))
    return float(round(best, 6))


def nice_division(length_m: float) -> tuple[int, float]:
    """给定比例尺总长，选一个能让"每格刻度"也是整数的分割方式。"""
    for divisions in (4, 5, 3, 2):
        division = length_m / divisions
        snapped = nice_length(division, mantissas=(1.0, 2.0, 2.5, 5.0))
        if abs(snapped - division) <= division * 0.02 + 1e-9:
            return divisions, float(division)
    divisions = 4
    return divisions, float(length_m / divisions)


def format_class_labels(bounds: list[tuple[float, float]], mode: str = "semantic") -> list[str]:
    """把分级区间变成图例标签：语义化 / 取整区间 / 两者结合。

    ``bounds`` 为 ``[(下界, 上界), ...]``（由实际分级断点得到）。
    """
    mode = (mode or "semantic").lower()
    count = len(bounds)
    words = _SEMANTIC_LABELS.get(count) or _SEMANTIC_LABELS.get(
        min(_SEMANTIC_LABELS), ["低", "高"]
    )
    labels: list[str] = []
    for index, (low, high) in enumerate(bounds):
        word = words[min(index, len(words) - 1)] if index < len(words) else f"第{index + 1}级"
        rng = f"{_round_num(low)} – {_round_num(high)}"
        if mode == "range":
            labels.append(rng)
        elif mode == "both":
            labels.append(f"{word}（{rng}）")
        else:
            labels.append(word)
    return labels


def _round_num(value: float) -> str:
    """数值取整到 3 位有效数字，去掉多余小数（避免 11516.196000 这类噪声）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(number)))
    if magnitude >= 4:
        return f"{round(number / 1000.0):g} 千" if False else f"{int(round(number)):,}"
    digits = max(0, 2 - magnitude)
    text = f"{round(number, digits):.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


# ------------------------------------------------------------------- 主入口

def design_layout(
    facts: dict[str, Any],
    intent: dict[str, Any] | None = None,
    *,
    profile: dict[str, Any] | None = None,
    hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """产出布局规格（LayoutSpec）。

    Args:
        facts: :func:`facts.collect_map_facts` 的结果（或等价结构的假数据）
        intent: 语义意图（主题/用途/介质/朝向/图例标签…）
        profile: 风格档位（缺省按 intent.style_profile 加载）
        hints: 版面修复提示（由 :func:`suggest_fixes` 产出：强制朝向/图例角位/比例尺长度…）
    """
    intent = normalize_intent(intent)
    profile = profile or load_profile(intent.get("style_profile"))
    hints = dict(hints or {})
    decisions: list[dict[str, Any]] = []

    renderer = _plan_renderer(facts, intent, profile, decisions)
    title_text = intent.get("title") or _compose_title(intent, facts, renderer)
    page = _choose_page(facts, intent, profile, renderer, title_text, decisions, hints)
    extent = facts.get("extent") or {}
    occupancy = facts.get("corner_usage") or {}

    legend_needed, legend_reason = _legend_needed(facts, intent, renderer)
    legend = _plan_legend(facts, intent, profile, renderer, legend_needed, legend_reason, occupancy, hints, decisions)

    geometry = _solve_geometry(page, profile, title_text, legend, occupancy, decisions, hints)
    title_box = _plan_title(title_text, page, geometry, profile, decisions, hints)

    scale_bar = _plan_scale_bar(geometry, extent, profile, intent, decisions, hints, facts)
    if hints.get("scale_bar_element_width_mm"):
        scale_bar["element_width_mm"] = float(hints["scale_bar_element_width_mm"])
    north_arrow = _plan_north_arrow(geometry, facts, profile, intent, decisions, hints)
    labels = _plan_labels(facts, profile, decisions)

    spec: dict[str, Any] = {
        "profile": profile.get("name", DEFAULT_PROFILE_NAME),
        "intent": intent,
        "page": page,
        "neatline": geometry["neatline"],
        "title": {**title_box, "text": title_text},
        "map_frame": geometry["map_frame"],
        "legend": {**legend, **geometry["legend_box"]},
        "scale_bar": {**scale_bar, **geometry["scale_bar_box"]},
        "north_arrow": {**north_arrow, **geometry["north_arrow_box"]},
        "renderer": renderer,
        "labels": labels,
        # 叠加层元素（迁移图的箭头/年份标注）：数据坐标，渲染后按实测地图框范围换算像素
        "overlay": dict(intent.get("overlay") or {}),
        "extent": {"padding": float(profile.get("extent_padding", 0.08)), "data": extent},
        "dpi": int(profile.get("dpi", 250)),
        "title_style": str(intent.get("title_style") or "overlay"),
        "decisions": decisions,
    }
    spec["design_note"] = _design_note(spec)
    return _apply_overrides(spec, intent.get("overrides") or {})


# ------------------------------------------------------------------ 纸张朝向

def _choose_page(
    facts: dict[str, Any],
    intent: dict[str, Any],
    profile: dict[str, Any],
    renderer: dict[str, Any],
    title_text: str,
    decisions: list[dict[str, Any]],
    hints: dict[str, Any],
) -> dict[str, Any]:
    aspect = float(facts.get("aspect") or 1.0)
    medium = intent.get("medium") or profile.get("medium", "print")
    want = str(hints.get("orientation") or intent.get("orientation") or profile.get("prefer_orientation") or "auto")
    candidates = _page_candidates(profile, medium)
    # 介质里点名了纸型就不再让"更小纸"规则插手（尊重用户/Agent 的明确选择）
    explicit_medium = str(intent.get("medium") or "").strip()
    if explicit_medium:
        for token in ("16:9", "A4", "A3"):
            if token in explicit_medium:
                filtered = [c for c in candidates if c["paper"] == token]
                candidates = filtered or candidates
                break
    if hints.get("paper"):
        candidates = [c for c in candidates if c["paper"] == hints["paper"]] or candidates
    if want in ("portrait", "landscape"):
        filtered = [c for c in candidates if c["orientation"] == want]
        candidates = filtered or candidates

    legend_box = _estimate_legend_size(renderer, profile, intent)
    rows = max(1, int(renderer.get("class_count") or 1))
    is_screen = str(profile.get("medium")) == "screen" or "screen" in str(medium)
    scored: list[tuple[float, dict[str, Any], str]] = []
    for page in candidates:
        neat = _neatline_box(page, profile)
        title_band = _title_band_height(title_text, page, profile)
        content = {
            "x": neat["x"] + profile.get("inner_margin_mm", 6.0),
            "y": neat["y"] + profile.get("inner_margin_mm", 6.0),
            "w": neat["w"] - 2 * profile.get("inner_margin_mm", 6.0),
            "h": neat["h"] - title_band - 2 * profile.get("inner_margin_mm", 6.0),
        }
        outside = bool(hints.get("legend_outside")) or (
            not hints.get("legend_outside") and _legend_outside_recommended(facts, profile, legend_box)
        )
        map_w = content["w"] - (legend_box[0] + 4.0 if outside else 0.0)
        map_box_aspect = map_w / max(content["h"], 1.0)
        fit = -abs(math.log(max(map_box_aspect, 1e-6) / max(aspect, 1e-6)))
        room = map_w * content["h"] - (0 if outside else legend_box[0] * legend_box[1])
        room_score = min(1.0, max(0.0, room / max(content["w"] * content["h"], 1.0)))
        score = fit * 2.0 + room_score * 0.5
        if is_screen and page["paper"] == "16:9":
            score += 0.6            # 屏幕汇报默认宽屏
        if renderer.get("mode") == "multi_layer" and page["orientation"] == "landscape":
            score += 0.1
        why = (
            f"数据长宽比 {aspect:.2f}，{page['paper']} {page['orientation']} 下地图框长宽比 "
            f"{map_box_aspect:.2f} 匹配最好（评分 {score:.2f}）"
        )
        scored.append((score, page, why))

    scored.sort(key=lambda item: -item[0])
    best_score, page, best_why = scored[0]
    # 制图惯例：能装下就用小纸（大纸会把字号相对拉小、也浪费纸）；
    # 同样能装下时优先档位里排在前面的纸张。
    preferred = [str(p) for p in (profile.get("preferred_papers") or [])]
    tolerance = 0.15
    allow_tiebreak = not explicit_medium
    for score, candidate, _why in (scored[1:] if allow_tiebreak else []):
        if best_score - score > tolerance:
            continue
        smaller = candidate["width_mm"] * candidate["height_mm"] < page["width_mm"] * page["height_mm"]
        better_order = (
            preferred.index(candidate["paper"]) < preferred.index(page["paper"])
            if (candidate["paper"] in preferred and page["paper"] in preferred)
            else False
        )
        if smaller or better_order:
            previous = f"{page['paper']} {page['orientation']}"
            page, best_why = candidate, (
                f"数据长宽比 {aspect:.2f}；{candidate['paper']} {candidate['orientation']} 与 {previous} "
                f"用地面积/匹配度接近，按惯例优先更小或更靠前的纸型"
            )
    decisions.append({"key": "page", "value": f"{page['paper']} {page['orientation']}",
                      "why": best_why or "唯一候选", "source": "rule"})
    if rows > 6 and page["orientation"] == "portrait":
        decisions.append({"key": "legend_rows", "value": rows,
                          "why": "图例条目较多，竖版仍有纵向空间；如过挤可切换 screen_report 档", "source": "rule"})
    return page


def _page_candidates(profile: dict[str, Any], medium: str) -> list[dict[str, Any]]:
    papers = list(profile.get("preferred_papers") or ["A4"])
    if medium in ("screen", "screen_16_9") and "16:9" not in papers:
        papers.append("16:9")
    out: list[dict[str, Any]] = []
    for paper in papers:
        if paper not in PAPERS:
            continue
        width, height = PAPERS[paper]
        if paper == "16:9":
            out.append({"paper": paper, "orientation": "landscape", "width_mm": width, "height_mm": height})
            continue
        out.append({"paper": paper, "orientation": "portrait", "width_mm": min(width, height),
                    "height_mm": max(width, height)})
        out.append({"paper": paper, "orientation": "landscape", "width_mm": max(width, height),
                    "height_mm": min(width, height)})
    if medium in ("print_A4",):
        out = [p for p in out if p["paper"] == "A4"] or out
    if medium in ("print_A3",):
        out = [p for p in out if p["paper"] == "A3"] or out
    return out


def _neatline_box(page: dict[str, Any], profile: dict[str, Any]) -> dict[str, float]:
    margin = float(profile.get("outer_margin_mm", 4.0))
    return {
        "x": margin,
        "y": margin,
        "w": page["width_mm"] - 2 * margin,
        "h": page["height_mm"] - 2 * margin,
        "line_width": 0.5,
    }


def _title_band_height(title_text: str, page: dict[str, Any], profile: dict[str, Any]) -> float:
    cfg = profile.get("title", {})
    pt = _title_font_pt(title_text, page, profile)
    lines = _title_lines(title_text, page, profile, pt)
    factor = float(cfg.get("two_line_factor", 1.25)) if lines > 1 else 1.0
    return pt * _MM_PER_PT * factor + float(cfg.get("band_gap_mm", 3.0)) * 2


def _title_box_width(page: dict[str, Any], profile: dict[str, Any]) -> float:
    neat = _neatline_box(page, profile)
    return neat["w"] - 2 * max(2.0, float(profile.get("inner_margin_mm", 6.0)) * 0.5)


def _title_font_pt(text: str, page: dict[str, Any], profile: dict[str, Any], *, chars: int | None = None) -> float:
    cfg = profile.get("title", {})
    length = max(1, chars if chars is not None else len(text))
    available_mm = _title_box_width(page, profile)
    per_char_mm = available_mm / length * 0.92
    pt = per_char_mm / _MM_PER_PT
    return float(min(max(pt, float(cfg.get("min_pt", 12.0))), float(cfg.get("max_pt", 20.0))))


def _title_lines(text: str, page: dict[str, Any], profile: dict[str, Any], pt: float) -> int:
    cfg = profile.get("title", {})
    limit = int(cfg.get("max_chars_per_line", 22))
    if len(text) <= limit:
        return 1
    # 一行放不下：先看缩到 min_pt 能否放下，否则分两行
    min_pt = float(cfg.get("min_pt", 12.0))
    width_mm = len(text) * min_pt * _MM_PER_PT / 0.92
    if width_mm <= _title_box_width(page, profile) and len(text) <= limit * 1.2:
        return 1
    return 2 if len(text) <= limit * 2 else 3


def _plan_title(
    text: str,
    page: dict[str, Any],
    geometry: dict[str, Any],
    profile: dict[str, Any],
    decisions: list[dict[str, Any]],
    hints: dict[str, Any],
) -> dict[str, Any]:
    cfg = profile.get("title", {})
    neat = geometry["neatline"]
    pt = float(hints.get("title_pt") or _title_font_pt(text, page, profile))
    lines = int(hints.get("title_lines") or _title_lines(text, page, profile, pt))
    row_mm = pt * _MM_PER_PT * (float(cfg.get("two_line_factor", 1.25)) if lines > 1 else 1.0)
    height = row_mm * lines
    gap = float(cfg.get("band_gap_mm", 3.0))
    box = {
        "x": neat["x"] + float(profile.get("inner_margin_mm", 6.0)) * 0.5,
        "y": neat["y"] + neat["h"] - gap - height,
        "w": neat["w"] - float(profile.get("inner_margin_mm", 6.0)),
        "h": height,
    }
    decisions.append({
        "key": "title",
        "value": f"{text}｜{pt:.1f}pt×{lines}行，顶部居中",
        "why": f"按可用宽度 {box['w']:.0f}mm 与 {len(text)} 字反算字号并夹在 "
               f"{cfg.get('min_pt', 12)}–{cfg.get('max_pt', 20)}pt 之间（参考图惯例：图名居中最醒目）",
        "source": "rule",
    })
    return {
        "font": cfg.get("font", "宋体"),
        "height_pt": round(pt, 2),
        "bold": bool(cfg.get("bold", True)),
        "lines": lines,
        "align": "center",
        "box": box,
    }


# ------------------------------------------------------------------- 标题文本

def _compose_title(intent: dict[str, Any], facts: dict[str, Any], renderer: dict[str, Any]) -> str:
    """``{区域}{尺度}{主题}{图种}``——缺哪段就略过，并去重（主题里已含尺度/图种时不重复拼接）。"""
    region = intent.get("region") or ""
    scale = intent.get("scale") or ""
    theme = intent.get("theme") or (renderer.get("field") or "专题")
    kind = intent.get("map_kind") or _default_map_kind(renderer)
    if scale and (theme.startswith(scale) or scale in theme):
        scale = ""
    if kind:
        core = kind.rstrip("图")
        if theme.endswith("图"):
            kind = ""
        elif core and core in theme:
            kind = "图"          # 主题已含“类型/分布”，只补一个“图”字
    return f"{region}{scale}{theme}{kind}".strip()


def _default_map_kind(renderer: dict[str, Any]) -> str:
    mode = renderer.get("mode")
    if mode in ("graduated", "unique", "multi_layer", "raster_stretch"):
        return "分布图"
    return "示意图"


# ---------------------------------------------------------------------- 渲染器

def _plan_renderer(
    facts: dict[str, Any],
    intent: dict[str, Any],
    profile: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    layers = [l for l in (facts.get("layers") or []) if l.get("exists")]
    explicit = str(intent.get("mode") or "").strip()
    facts_mode = str(facts.get("render_mode") or "").strip()
    primary_is_raster = bool((facts.get("primary") or {}).get("is_raster"))
    field = intent.get("field") or ""
    profile_stats = facts_mod.field_profile(facts, field) if field else {}
    if not field:
        for layer in layers:
            stats = layer.get("field_stats") or {}
            if stats:
                field = next(iter(stats))
                profile_stats = stats[field]
                break

    continuous = bool(profile_stats.get("max") is not None and not profile_stats.get("values_top"))
    class_count = 5
    method = "DefinedInterval"
    interval = None
    if continuous:
        skew = float(profile_stats.get("skew") or 0.0)
        highest = float(profile_stats.get("max") or 0.0)
        lowest = float(profile_stats.get("min") or 0.0)
        want_method = str(intent.get("classification") or "").strip().lower()
        if want_method in ("quantile", "natural", "interval"):
            method = {"quantile": "Quantile", "natural": "NaturalBreaks", "interval": "DefinedInterval"}[want_method]
        elif skew > 0.45:
            method = "Quantile"
        if method == "DefinedInterval":
            # 整数等间距：分界值整齐，图例数值不会出现 903.483788 这种噪声
            span = max(highest - lowest, 1e-9)
            interval = nice_length(max(span / max(class_count, 1), 1e-9), mantissas=(1.0, 2.0, 2.5, 5.0))
            if interval and interval > 0:
                class_count = max(2, min(9, int(round(span / interval)) or class_count))
        decisions.append({
            "key": "classification",
            "value": {"DefinedInterval": "等间距（整数分界）", "Quantile": "分位数",
                      "NaturalBreaks": "自然断点"}.get(method, method),
            "why": (f"数据偏斜 {skew:.2f}；等间距分界值取整数（间隔 {interval:g}），图例数值更规范"
                    if method == "DefinedInterval" else f"数据偏斜 {skew:.2f}"),
            "source": "rule",
        })
    if primary_is_raster and not explicit:
        mode = "raster_stretch"
    elif explicit:
        mode = explicit
    elif facts_mode in ("single", "multi_layer"):
        mode = facts_mode
    elif profile_stats.get("values_top"):
        mode = "unique"
    elif continuous:
        mode = "graduated"
    else:
        mode = facts_mode or "single"
    if mode == "unique" and profile_stats.get("values_top"):
        class_count = min(8, max(2, int(profile_stats.get("unique_count") or 2)))

    theme_text = " ".join([intent.get("theme", ""), field, intent.get("note", "")])
    ramp = intent.get("color_ramp") or _pick_ramp(theme_text, profile)
    colors = profile.get("colors", {}) if profile else {}
    palette = list(colors.get("categorical") or [])
    category_colors = intent.get("category_colors") or _auto_category_colors(profile_stats, palette)
    # 显式分级（统一图例）：多期/多图要能直接对比，必须用同一组分级上界。
    # DefinedInterval 每张图从各自最小值起算，做不到跨图统一。
    explicit_bounds = _parse_class_bounds(intent.get("class_bounds") or intent.get("explicit_bounds"))
    if explicit_bounds:
        # 语义：给定的是**分级边界**（n 个边界 → n-1 级）。这样 "0,50,100" 读起来自然，
        # 也避免出现“首级区间倒置”（若把边界当上界，第一级会变成 数据最小值–第一个边界 ✗）。
        class_count = max(1, len(explicit_bounds) - 1)
        method = "Manual"
        interval = None
        decisions.append({
            "key": "explicit_bounds",
            "value": explicit_bounds,
            "why": f"指定 {class_count} 级显式分级（多图共用同一图例才能横向对比）",
            "source": "intent",
        })
    labels_mode = intent.get("legend_labels") or (profile.get("legend", {}) or {}).get("labels_mode", "semantic")
    resolved_mode = "unique" if mode == "unique" else ("graduated" if continuous and not primary_is_raster else mode)
    if primary_is_raster:
        resolved_mode = "raster_stretch"
    if resolved_mode == "unique":
        labels_mode = "value"  # 分类图例直接用类别名，不用语义词/区间
    elif resolved_mode == "raster_stretch":
        labels_mode = "range"

    if resolved_mode == "graduated" and ramp:
        decisions.append({"key": "color_ramp", "value": ramp,
                          "why": f"按主题关键词选择连续色带（主题：{theme_text.strip() or '未指定'}）", "source": "rule"})
    return {
        "mode": resolved_mode,
        "field": field,
        "continuous": continuous,
        "class_count": class_count,
        "classification_method": method,
        "interval_size": interval,
        "explicit_bounds": explicit_bounds,
        "color_ramp": ramp,
        "category_colors": category_colors,
        "category_names": intent.get("category_names") or "",
        "category_order": list(intent.get("category_order") or []),
        "labels_mode": labels_mode,
        "nodata_color": colors.get("nodata_color", "#E8E8E8"),
        "field_stats": profile_stats,
        "layer_paths": [l.get("path") for l in layers],
    }


def _parse_class_bounds(raw: Any) -> list[float]:
    """解析显式分级上界（"0,50,100" 或 "[0,50,100]" 或列表）→ 升序去重 float 列表。

    用于“统一图例”：多张图传入同一组上界，图例才能横向对比（核密度多期图实测需求）。
    """
    values: list[float] = []
    if raw is None or raw == "":
        return values
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        text = str(raw).strip().strip("[]()")
        items = [piece for piece in text.replace(";", ",").replace(" ｜ ", ",").split(",") if piece.strip()]
    for item in items:
        try:
            values.append(float(str(item).strip()))
        except (TypeError, ValueError):
            continue
    unique = sorted(set(values))
    return unique


def _pick_ramp(text: str, profile: dict[str, Any]) -> str:
    colors = (profile or {}).get("colors", {}) or {}
    keywords = colors.get("continuous_keywords") or {}
    lowered = (text or "").lower()
    for keyword, ramp in keywords.items():
        if keyword.lower() in lowered:
            return ramp
    return colors.get("default_ramp", "YlOrRd")


def _auto_category_colors(stats: dict[str, Any], palette: list[str]) -> str:
    """按值出现次数给分类配色（次数多的用靠前颜色）。"""
    values = [item.get("value") for item in (stats.get("values_top") or [])]
    if not values or not palette:
        return ""
    parts = []
    for index, value in enumerate(values):
        parts.append(f"{value}:{palette[index % len(palette)]}")
    return ";".join(parts)


# ---------------------------------------------------------------------- 图例

def _legend_needed(
    facts: dict[str, Any], intent: dict[str, Any], renderer: dict[str, Any]
) -> tuple[bool, str]:
    mode = renderer.get("mode")
    if mode == "raster_stretch":
        return True, "栅格拉伸渲染必须给出色带图例（标注数值范围）"
    if mode == "single":
        return False, "图层为单一符号，图上无需图例（参考图惯例：服务圈单一符号图不放图例）"
    if mode == "graduated":
        return True, "分级设色必须给出图例，否则读者无法解读颜色"
    if mode == "unique":
        count = int(renderer.get("class_count") or 0)
        if count <= 1:
            return False, "唯一值只有 1 类，等价于单一符号"
        return True, f"唯一值 {count} 类需要图例区分"
    if mode == "multi_layer":
        return True, "多图层叠加需要用图例说明各图层符号"
    return False, "无需图例"


def _estimate_legend_size(
    renderer: dict[str, Any], profile: dict[str, Any], intent: dict[str, Any]
) -> tuple[float, float]:
    legend_cfg = profile.get("legend", {})
    font_pt = float(legend_cfg.get("font_pt", 9.0))
    title_pt = float(legend_cfg.get("title_pt", 11.0))
    swatch_w, swatch_h = legend_cfg.get("swatch_mm", [7.0, 5.0])
    pad = float(legend_cfg.get("pad_mm", 2.5))
    rows = max(1, int(renderer.get("class_count") or 1))
    labels = _estimate_labels(renderer, rows)
    longest = max(len(text) for text in labels) if labels else 6
    width = pad * 2 + float(swatch_w) + 3.0 + longest * font_pt * _MM_PER_PT * 0.95
    title = intent.get("legend_title") or intent.get("theme") or renderer.get("field") or ""
    height = pad * 2 + (title_pt * _MM_PER_PT * 1.7 if title else 0.0) + rows * max(
        float(swatch_h) + 1.2, font_pt * _MM_PER_PT * float(legend_cfg.get("row_factor", 1.5))
    )
    return float(math.ceil(width)), float(math.ceil(height))


def _estimate_labels(renderer: dict[str, Any], rows: int) -> list[str]:
    mode = str(renderer.get("labels_mode") or "semantic")
    if mode == "range":
        return ["0000 – 0000"] * rows
    if mode == "both":
        return ["很多（0000 – 0000）"] * rows
    return _SEMANTIC_LABELS.get(rows, ["较低", "中", "较高"])[:rows] or ["中"]


def _legend_outside_recommended(
    facts: dict[str, Any], profile: dict[str, Any], legend_box: tuple[float, float]
) -> bool:
    """判断数据是否"铺满整框"——是则把图例移到图框外，保证地图主体可见。

    物理判据（用九宫格的**绝对**占用度，不用归一值）：**四个角都有墨迹**，且角上平均密度
    与中心同量级——这正是铺满整幅的栅格；城市型稀疏数据四角接近空白，不成立，图例留在框内。
    """
    grid = facts.get("occupancy") or []
    if len(grid) < 2 or not grid[0]:
        return False
    try:
        center = float(grid[1][1])
        corners = [float(grid[0][0]), float(grid[0][-1]), float(grid[-1][0]), float(grid[-1][-1])]
    except (IndexError, TypeError, ValueError):
        return False
    if center <= 0 or any(value <= 0 for value in corners):
        return False
    # 只要还有一个角明显比中心空，就能把图例放进图框（更省纸、也更符合参考图做法）
    return bool(min(corners) / center > 0.5)


def _plan_legend(
    facts: dict[str, Any],
    intent: dict[str, Any],
    profile: dict[str, Any],
    renderer: dict[str, Any],
    needed: bool,
    reason: str,
    occupancy: dict[str, float],
    hints: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    cfg = profile.get("legend", {})
    width, height = _estimate_legend_size(renderer, profile, intent)
    mode = renderer.get("mode")
    title = intent.get("legend_title") or ""
    if needed and not title and mode in ("graduated", "unique", "raster_stretch") and len(renderer.get("layer_paths") or []) <= 1:
        title = intent.get("theme") or ""
        if not title:
            decisions.append({"key": "legend_title", "value": "(无)",
                              "why": "主题未知：变量名缺失，暂不加图例标题以免误导", "source": "rule"})
    if mode == "multi_layer":
        title = ""
        decisions.append({"key": "legend_title", "value": "(无)",
                          "why": "多图层符号清单按惯例不加标题（参考图：医院点位/服务圈）", "source": "rule"})

    order = list(cfg.get("corner_order") or facts_mod.CORNER_KEYS)
    # 占用度优先（不遮挡数据），但**差距不大时回归档位惯例**（参考图：图例稳定放右下）
    usage_map = occupancy or {}
    preferred = next((candidate for candidate in order if candidate in facts_mod.CORNER_KEYS), "bottom_right")
    preferred_usage = float(usage_map.get(preferred, 0.0))

    def _corner_key(candidate: str) -> tuple[float, int]:
        return (float(usage_map.get(candidate, 0.0)), order.index(candidate) if candidate in order else 99)

    corner = str(hints.get("legend_corner") or "").strip() or min(facts_mod.CORNER_KEYS, key=_corner_key)
    emptied = float(usage_map.get(corner, 0.0)) <= preferred_usage * 0.6
    if corner != preferred and not emptied:
        decisions.append({
            "key": "legend_corner_pref", "value": f"改用惯例角位 {preferred}",
            "why": (f"最空角位 {corner} 的占用度 {usage_map.get(corner, 0.0):.2f} 与惯例角位 {preferred} "
                    f"{preferred_usage:.2f} 差距不大（未低于其 60%），按参考图惯例仍放 {preferred}"),
            "source": "rule",
        })
        corner = preferred
    outside = bool(hints.get("legend_outside")) or (
        bool(cfg.get("allow_outside", True)) and _legend_outside_recommended(facts, profile, (width, height))
    )
    if outside:
        decisions.append({"key": "legend_position", "value": "图框外右侧",
                          "why": "数据在四角/中心都很满（满覆盖），图例压上去会遮住内容；移到图框外保证地图主体完整",
                          "source": "rule"})
    else:
        decisions.append({"key": "legend_position", "value": corner,
                          "why": f"按九宫格占用度选择最空的角（{corner} 占用 {occupancy.get(corner, 0):.2f}）",
                          "source": "rule"})
    decisions.append({"key": "legend_needed", "value": "是" if needed else "否", "why": reason, "source": "rule"})
    return {
        "needed": bool(needed),
        "title": title,
        "font_pt": float(cfg.get("font_pt", 9.0)),
        "title_pt": float(cfg.get("title_pt", 11.0)),
        "border": bool(cfg.get("border", False)),
        "show_layer_name": bool(cfg.get("show_layer_name", False)),
        "width_mm": width,
        "height_mm": height,
        "corner": corner if not outside else "outside",
        "outside": outside,
        "swatch_mm": list(cfg.get("swatch_mm", [7.0, 5.0])),
    }


# ------------------------------------------------------------------ 版面求解

def _solve_geometry(
    page: dict[str, Any],
    profile: dict[str, Any],
    title_text: str,
    legend: dict[str, Any],
    occupancy: dict[str, float],
    decisions: list[dict[str, Any]],
    hints: dict[str, Any],
) -> dict[str, Any]:
    """先定标题带与图例占位，再给地图框取最大可用矩形，最后摆比例尺/指北针（不重叠）。"""
    neat = _neatline_box(page, profile)
    inner = float(profile.get("inner_margin_mm", 6.0))
    title_band = _title_band_height(title_text, page, profile)
    # 地图框直接占满图廓（外层细框由地图框自带边框承担）：
    # 这样"只有一层边框"，且边框同时存在于导出的 JPG 与 .aprx 里
    content = {
        "x": neat["x"],
        "y": neat["y"],
        "w": neat["w"],
        "h": neat["h"] - title_band,
    }
    name_cfg = profile.get("north_arrow", {})
    na_size = float(min(max(page["width_mm"] * float(name_cfg.get("size_frac", 0.055)),
                            float(name_cfg.get("min_mm", 12.0))), float(name_cfg.get("max_mm", 20.0))))
    sb_w = float(min(content["w"] * 0.5, content["w"] * float((profile.get("scale_bar", {}) or {}).get("target_width_frac", 0.24)) + 12.0))

    legend_box: dict[str, float] = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0, "visible": bool(legend.get("needed"))}
    map_box = dict(content)
    if legend.get("needed") and legend.get("outside"):
        col_w = float(legend["width_mm"]) + 4.0
        map_box["w"] = max(40.0, content["w"] - col_w)
        legend_box.update({
            "x": map_box["x"] + map_box["w"] + 4.0,
            "y": content["y"] + max(0.0, (content["h"] - float(legend["height_mm"])) / 2.0),
            "w": float(legend["width_mm"]),
            "h": float(legend["height_mm"]),
        })
    elif legend.get("needed"):
        corner = str(legend.get("corner") or "bottom_right")
        # 图例必须完全落在图框**内部**，且与图框线留出安全边距（用户反馈：不能压图框）
        margin = float((profile.get("legend", {}) or {}).get("frame_margin_mm", 6.0))
        left = corner.endswith("left")
        top = corner.startswith("top")
        legend_box.update({
            "x": map_box["x"] + (margin if left else map_box["w"] - float(legend["width_mm"]) - margin),
            "y": map_box["y"] + (map_box["h"] - float(legend["height_mm"]) - margin if top else margin),
            "w": float(legend["width_mm"]),
            "h": float(legend["height_mm"]),
        })
        decisions.append({
            "key": "legend_size",
            "value": f"{legend_box['w']:.0f}×{legend_box['h']:.0f} mm",
            "why": f"按 {int(legend.get('font_pt', 9))}pt 字号与 {int(legend.get('class_count') or 0) or 'n'} 条目的行高估算",
            "source": "rule",
        })

    # 比例尺：默认左下；被图例占了就移到中下
    sb_h = float(profile.get("scale_bar", {}).get("font_pt", 8.0)) * _MM_PER_PT * 2.6 + 4.0
    sb_x = map_box["x"] + 2.0
    sb_y = map_box["y"] + 2.0
    if legend.get("needed") and not legend.get("outside") and str(legend.get("corner", "")).startswith("bottom_left"):
        sb_x = map_box["x"] + (map_box["w"] - sb_w) / 2.0
    scale_bar_box = {"x": sb_x, "y": sb_y, "w": sb_w, "h": sb_h, "visible": True}

    # 指北针：默认右上；被图例占了就左上
    na_x = map_box["x"] + map_box["w"] - na_size - 3.0
    na_y = map_box["y"] + map_box["h"] - na_size - 3.0
    if legend.get("needed") and not legend.get("outside") and str(legend.get("corner", "")).startswith("top_right"):
        na_x = map_box["x"] + 3.0
    north_box = {"x": na_x, "y": na_y, "w": na_size, "h": na_size, "visible": True}
    return {"neatline": neat, "content": content, "map_frame": map_box,
            "legend_box": legend_box, "scale_bar_box": scale_bar_box, "north_arrow_box": north_box}


def _plan_scale_bar(
    geometry: dict[str, Any],
    extent: dict[str, float],
    profile: dict[str, Any],
    intent: dict[str, Any],
    decisions: list[dict[str, Any]],
    hints: dict[str, Any],
    facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = profile.get("scale_bar", {})
    crs = ((facts or {}).get("primary") or {}).get("crs") or {}
    if facts and crs and not crs.get("is_projected"):
        # 制图惯例：未投影（度）的地图上放米制比例尺是错误信息，宁可不放
        decisions.append({"key": "scale_bar", "value": "省略",
                          "why": "数据为地理坐标系（度），米制比例尺无意义；如需比例尺请先投影到投影坐标系",
                          "source": "rule"})
        return {"visible": False, "style_keywords": [], "divisions": 0, "subdivisions": 0,
                "units": "METERS", "unit_label": "", "length_m": 0.0, "division_m": 0.0,
                "bar_mm": 0.0, "map_scale": 0.0, "font_pt": float(cfg.get("font_pt", 8.0))}
    frame = geometry["map_frame"]
    pad = float(profile.get("extent_padding", 0.08))
    data_w = float(extent.get("width") or 0.0)
    data_h = float(extent.get("height") or 0.0)
    map_scale = 1.0
    if data_w > 0 and data_h > 0:
        padded_w = data_w * (1 + 2 * pad)
        padded_h = data_h * (1 + 2 * pad)
        map_scale = max(padded_w * 1000.0 / frame["w"], padded_h * 1000.0 / frame["h"])
    target_mm = frame["w"] * float(cfg.get("target_width_frac", 0.24))
    length_m = float(hints.get("scale_bar_length_m") or nice_length(target_mm * map_scale / 1000.0))
    divisions, division_m = nice_division(length_m)
    units, unit_label = ("KILOMETERS", "千米") if length_m >= 10000 else ("METERS", cfg.get("unit_label", "米"))
    bar_mm = length_m * 1000.0 / map_scale if map_scale else 0.0
    decisions.append({
        "key": "scale_bar",
        "value": f"{length_m:g} m / {divisions} 段，单位 {unit_label}（约 {bar_mm:.0f} mm）",
        "why": f"按地图比例尺 1:{map_scale:,.0f} 反算：取 m×10^k 中最接近图框宽 "
               f"{float(cfg.get('target_width_frac', 0.24)) * 100:.0f}% 的整数长度，保证刻度是整数",
        "source": "rule",
    })
    return {
        "visible": True,
        "style_keywords": list(cfg.get("style_keywords") or []),
        "divisions": int(divisions),
        "subdivisions": 1,
        "units": units,
        "unit_label": unit_label,
        "length_m": length_m,
        "division_m": division_m,
        "bar_mm": bar_mm,
        "map_scale": round(map_scale, 2),
        "font_pt": float(cfg.get("font_pt", 8.0)),
    }


def _plan_north_arrow(
    geometry: dict[str, Any],
    facts: dict[str, Any],
    profile: dict[str, Any],
    intent: dict[str, Any],
    decisions: list[dict[str, Any]],
    hints: dict[str, Any],
) -> dict[str, Any]:
    cfg = profile.get("north_arrow", {})
    needed = cfg.get("needed", "auto")
    if str(needed).lower() == "false":
        return {"needed": False, "style_keywords": [], "size_mm": 0.0}
    crs = (facts.get("primary") or {}).get("crs") or {}
    projected = bool(crs.get("is_projected"))
    decisions.append({
        "key": "north_arrow",
        "value": "是" if needed != "false" else "否",
        "why": ("无经纬网/方里网标注时按惯例保留指北针；区域全图用八芒罗盘玫瑰"
                if projected else "未检出投影坐标系，方向参考意义有限，仍保留指北针"),
        "source": "rule",
    })
    return {
        "needed": True,
        "style_keywords": list(hints.get("north_arrow_keywords") or cfg.get("style_keywords") or []),
        "simple_keywords": list(cfg.get("simple_keywords") or []),
        "size_mm": float(geometry["north_arrow_box"]["w"]),
    }


def _plan_labels(
    facts: dict[str, Any], profile: dict[str, Any], decisions: list[dict[str, Any]]
) -> dict[str, Any]:
    cfg = profile.get("labels", {})
    primary = facts.get("primary") or {}
    count = int(primary.get("feature_count") or 0)
    threshold = int(cfg.get("enable_below_features", 80))
    enabled = bool(count and count <= threshold)
    if not enabled and count:
        decisions.append({"key": "labels", "value": "关闭",
                          "why": f"要素数 {count} 超过阈值 {threshold}，逐个标注会糊成一片", "source": "rule"})
    return {"enabled": enabled, "field": ""}


def _design_note(spec: dict[str, Any]) -> str:
    page = spec["page"]
    bits = [
        f"{page['paper']} {page['orientation']}",
        f"图名 {spec['title']['height_pt']:.0f}pt",
    ]
    if spec["legend"].get("needed"):
        bits.append(f"图例{spec['legend'].get('corner')}")
    else:
        bits.append("不放图例")
    bits.append(f"比例尺 {spec['scale_bar']['length_m']:g}m/{spec['scale_bar']['unit_label']}")
    if spec["north_arrow"].get("needed"):
        bits.append("指北针")
    return "｜".join(bits)


def _apply_overrides(spec: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """按点路径强制覆盖（如 ``{"legend.outside": True}``）。"""
    for path, value in (overrides or {}).items():
        parts = str(path).split(".")
        node = spec
        ok = True
        for part in parts[:-1]:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                ok = False
                break
        if ok and isinstance(node, dict):
            node[parts[-1]] = value
    return spec


# ------------------------------------------------------------------ 版面修复

def suggest_fixes(checks: list[dict[str, Any]]) -> dict[str, Any]:
    """把 QC 不通过项翻译成设计提示（供 :func:`design_layout` 的 ``hints`` 用）。"""
    hints: dict[str, Any] = {}
    for check in checks or []:
        if check.get("ok"):
            continue
        name = str(check.get("name", ""))
        if name == "title_too_small":
            hints["title_pt"] = float(check.get("suggest_pt") or 18.0)
        elif name == "title_overflow":
            hints["title_lines"] = 2
        elif name in ("legend_covers_data", "legend_over_data"):
            hints["legend_outside"] = True
        elif name == "element_outside_neatline":
            hints["legend_outside"] = hints.get("legend_outside", False)
        elif name == "scale_bar_too_long":
            shorter = check.get("suggest_length_m")
            if shorter:
                hints["scale_bar_length_m"] = float(shorter)
        elif name == "scale_bar_not_round":
            width_hint = check.get("suggest_element_width_mm")
            if width_hint:
                hints["scale_bar_element_width_mm"] = float(width_hint)
        elif name == "font_too_small":
            hints["legend_font_pt"] = float(check.get("suggest_pt") or 8.0)
    return hints


def empty_facts() -> dict[str, Any]:
    """无数据时的兜底 facts（便于单测与容错）。"""
    return {
        "layers": [],
        "extent": {"xmin": 0.0, "ymin": 0.0, "xmax": 0.0, "ymax": 0.0, "width": 0.0, "height": 0.0},
        "aspect": 1.0,
        "occupancy": [[0.0] * 3 for _ in range(3)],
        "corner_usage": {key: 0.0 for key in facts_mod.CORNER_KEYS},
        "primary": {},
        "layer_count": 0,
        "render_mode": "single",
    }


__all__ = [
    "design_layout",
    "normalize_intent",
    "nice_length",
    "nice_division",
    "format_class_labels",
    "suggest_fixes",
    "empty_facts",
    "PAPERS",
]
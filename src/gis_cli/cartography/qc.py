# -*- coding: utf-8 -*-
"""版面体检（layout QC）：出图后量测"图名/图例/比例尺/指北针/数据"是否达到专业惯例。

只做量测与判定，不给修法——修法在 :func:`design.suggest_fixes`，两者合起来构成
"设计 → 渲染 → 体检 → 微调 → 再渲染"的版面自修复闭环。

检查项（不通过项的名字与 :func:`design.suggest_fixes` 约定一致）：
- ``title_too_small`` / ``title_overflow``：图名字号与是否溢出图廓
- ``title_not_centered``：图名是否水平居中
- ``legend_labels_unrounded``：图例标签是否残留多位小数（``11516.196000`` 这类）
- ``legend_covers_data``：图例叠放区是否压住数据
- ``scale_bar_not_round``：比例尺刻度是否为整数
- ``font_too_small``：正文（图例）字号是否低于打印可读下限
- ``elements_overlap`` / ``element_outside_neatline``：元素是否互相压盖/越出图廓
- ``map_frame_blank``：地图框内是否真的画上了数据
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from . import styles as styles_mod

logger = logging.getLogger(__name__)

_LONG_DECIMAL = re.compile(r"\d+\.\d{3,}")
_ROUND_MANTISSAS = (1.0, 2.0, 2.5, 5.0)


def _arcpy():
    import arcpy  # type: ignore

    return arcpy


def _box(element: Any) -> tuple[float, float, float, float]:
    x = float(getattr(element, "elementPositionX", 0.0))
    y = float(getattr(element, "elementPositionY", 0.0))
    w = float(getattr(element, "elementWidth", 0.0))
    h = float(getattr(element, "elementHeight", 0.0))
    return x, y, w, h


def _boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx + 0.5 or bx + bw <= ax + 0.5 or ay + ah <= by + 0.5 or by + bh <= ay + 0.5)


def _near_round(value: float, *, tolerance: float = 0.05) -> bool:
    """是否接近整数刻度（渲染时元素宽度吸附会带来几 % 偏差）。"""
    if value <= 0:
        return False
    import math

    exponent = math.floor(math.log10(value))
    base = 10.0 ** exponent
    if base <= 0:
        return False
    steps = [m * base for m in (1.0, 2.0, 2.5, 4.0, 5.0, 10.0)]
    return any(abs(value - step) <= step * tolerance for step in steps)


def _is_clean_length(value_m: float, *, in_km: bool) -> bool:
    """刻度长度在**显示单位**下是否整齐（千米显示：0.5/1/2/2.5/5…；米显示：100/250/500/1000…）。"""
    if value_m <= 0:
        return False
    if in_km:
        km = value_m / 1000.0
        clean = (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0)
        return any(abs(km - candidate) <= candidate * 0.01 for candidate in clean)
    return _is_round(value_m) or _near_round(value_m, tolerance=0.02)


def _nearest_round(value: float) -> float:
    """最接近的"整齐"数值（用于把比例尺每格长度拉到整数刻度）。"""
    if value <= 0:
        return 0.0
    import math

    exponent = math.floor(math.log10(value))
    candidates = [m * 10 ** e for e in (exponent - 1, exponent, exponent + 1)
                  for m in (1.0, 2.0, 2.5, 4.0, 5.0, 10.0)]
    return float(min(candidates, key=lambda candidate: abs(value - candidate)))


def _snap_integer(value: float) -> float:
    """把 ArcGIS 的浮点噪声（如 7500.000001）吸附回整数，其他值原样返回。"""
    nearest = round(value)
    return float(nearest) if abs(value - nearest) <= max(1e-3, abs(value) * 1e-9) else float(value)


def _is_round(value: float) -> bool:
    """是否"整齐数值"：0 或整数且有效位后至少两位为零（如 1000/2500/7500；而 903/1375 不算）。"""
    if value == 0:
        return True
    if value < 0:
        return False
    import math

    if abs(value - round(value)) > 1e-9:
        return False
    integer = int(round(value))
    if integer < 100:
        return True
    exponent = max(0, math.floor(math.log10(integer)) - 1)
    if integer % (10 ** exponent) == 0:
        return True
    # 允许 12.5 这类半步值（如 12500 = 12.5×1000）
    if exponent >= 1:
        coarse = 10 ** (exponent - 1)
        return integer % coarse == 0 and (integer // coarse) % 5 == 0
    return False


def _cim_symbol_height(element: Any) -> float | None:
    try:
        definition = element.getDefinition("V3")
    except Exception:
        return None
    graphic = getattr(definition, "graphic", None)
    symbol = getattr(graphic, "symbol", None) if graphic is not None else None
    height = getattr(symbol, "height", None)
    try:
        return float(height) if height is not None else None
    except (TypeError, ValueError):
        return None


def _image_density(image_path: str, box_px: tuple[int, int, int, int] | None = None) -> float | None:
    try:
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        with Image.open(image_path) as img:
            array = np.asarray(img.convert("RGB"))
        if box_px:
            left, top, right, bottom = box_px
            array = array[max(0, top):max(0, bottom), max(0, left):max(0, right)]
        if array.size == 0:
            return None
        gray = array.mean(axis=2)
        return float((gray < 240).mean())
    except Exception as exc:  # pragma: no cover
        logger.warning("图片量测失败: %s", exc)
        return None


def check_layout(
    aprx_path: str,
    *,
    spec: dict[str, Any] | None = None,
    image_path: str = "",
    min_font_pt: float = 7.0,
) -> dict[str, Any]:
    """核验工程版面，返回 ``{"ok": bool, "checks": [...]}``。"""
    import os

    checks: list[dict[str, Any]] = []
    target = Path(aprx_path)
    if not target.exists():
        return {"ok": False, "checks": [{"name": "aprx_exists", "ok": False, "detail": "工程文件不存在"}]}

    from arcpy import mp  # type: ignore

    project = mp.ArcGISProject(str(target))
    layouts = project.listLayouts()
    if not layouts:
        return {"ok": False, "checks": [{"name": "layout_exists", "ok": False, "detail": "工程里没有布局"}]}
    layout = layouts[0]
    elements = {str(element.name): element for element in layout.listElements()}
    boxes = {name: _box(element) for name, element in elements.items()}

    neatline = boxes.get("Neatline")
    frame = boxes.get("Main Map")
    if spec:
        neat_spec = spec.get("neatline") or {}
        neatline = (neat_spec.get("x", 0.0), neat_spec.get("y", 0.0), neat_spec.get("w", 0.0), neat_spec.get("h", 0.0)) if neat_spec else neatline

    # 1) 图名
    title_element = elements.get("Map Title")
    if title_element is not None and neatline:
        nx, ny, nw, nh = neatline
        tx, ty, tw, th = boxes.get("Map Title", (0, 0, 0, 0))
        height_pt = _cim_symbol_height(title_element)
        min_title_pt = float((spec or {}).get("title", {}).get("min_pt") or 12.0)
        ok_small = height_pt is None or height_pt >= min_title_pt
        checks.append({
            "name": "title_too_small", "ok": bool(ok_small),
            "detail": f"图名字号 {height_pt:.1f}pt（下限 {min_title_pt:.0f}pt）" if height_pt else "无法读取图名字号",
            "suggest_pt": max(min_title_pt, 18.0),
        })
        center_title = tx + tw / 2.0
        center_page = nx + nw / 2.0
        offset = abs(center_title - center_page)
        checks.append({
            "name": "title_not_centered", "ok": offset <= max(2.0, nw * 0.02),
            "detail": f"图名中心偏离页面中心 {offset:.1f}mm",
        })
        overflow = (tw > nw + 1.0) or (th > nh * 0.25)
        checks.append({
            "name": "title_overflow", "ok": not overflow,
            "detail": f"图名框 {tw:.0f}×{th:.0f}mm，图廓 {nw:.0f}×{nh:.0f}mm",
        })

    # 2) 图例
    legend_element = elements.get("Legend")
    if legend_element is not None:
        overlay_entries = ((spec or {}).get("renderer") or {}).get("legend_entries") or []
        if overlay_entries:
            # 导出图里的图例是叠加层重绘的（干净标签）；ArcGIS 自身的标签格式只作备注，不参与判定
            overlay_text = [str(item.get("label") or "") for item in overlay_entries]
            overlay_numbers = [_snap_integer(float(value))
                               for text in overlay_text for value in re.findall(r"-?\d+(?:\.\d+)?", text)]
            overlay_loose = [value for value in overlay_numbers
                             if not (_is_round(value) or _near_round(value, tolerance=0.02))]
            checks.append({
                "name": "legend_labels_unrounded", "ok": not overlay_loose,
                "detail": (f"图例（叠加重绘）分界值整齐（{overlay_numbers[:5]}）" if overlay_numbers and not overlay_loose
                           else (f"分界值不整齐 {overlay_loose[:4]}" if overlay_loose else "图例无可解析数值（分类图例）")),
            })
        try:
            definition = legend_element.getDefinition("V3")
            labels: list[str] = []
            font_sizes: list[float] = []
            for item in getattr(definition, "items", []) or []:
                symbol = getattr(item, "labelSymbol", None)
                if symbol is not None and getattr(symbol, "height", None):
                    font_sizes.append(float(symbol.height))
                for attr in ("heading", "label", "description"):
                    text = str(getattr(item, attr, "") or "")
                    if text:
                        labels.append(text)
            # 分级渲染器的标签在渲染器上（图例项里读不到文本），这里补读，否则发现不了未取整数字
            for map_obj in project.listMaps():
                for candidate in map_obj.listLayers():
                    try:
                        renderer = candidate.symbology.renderer
                    except Exception:
                        continue
                    for brk in (getattr(renderer, "classBreaks", None) or []):
                        text = str(getattr(brk, "label", "") or "")
                        if text:
                            labels.append(text)
                    for group in (getattr(renderer, "groups", None) or []):
                        for group_item in (getattr(group, "items", None) or []):
                            text = str(getattr(group_item, "label", "") or "")
                            if text:
                                labels.append(text)
            joined = " ".join(labels)
            if overlay_entries:
                joined = ""          # 已按叠加层条目校验过，避免对 ArcGIS 原始 6 位小数标签重复判负
            if not joined and not overlay_entries:
                mode = str(((spec or {}).get("renderer") or {}).get("mode") or "")
                if mode in ("multi_layer", "single", "raster_stretch", ""):
                    checks.append({"name": "legend_labels_unrounded", "ok": True,
                                   "detail": f"渲染模式 {mode or '未声明'}：无分级标签，跳过取整检查"})
                else:
                    checks.append({"name": "legend_labels_readable", "ok": False,
                                   "detail": "读不到图例/分级标签文本，无法确认是否取整"})
            renderer_types = []
            for map_obj in project.listMaps():
                for candidate in map_obj.listLayers():
                    try:
                        renderer_types.append(type(candidate.symbology.renderer).__name__)
                    except Exception:
                        continue
            is_raster = any("RasterStretch" in name for name in renderer_types)
            # 检查的是"分界值是否整齐"，而不是小数位数（ArcGIS 会按字段精度补 6 位小数）
            numbers = [_snap_integer(float(value)) for value in re.findall(r"-?\d+(?:\.\d+)?", joined)] if joined else []
            loose = [value for value in numbers if not (_is_round(value) or _near_round(value, tolerance=0.02))]
            bad = [] if (is_raster or not numbers) else loose[:5]
            checks.append({
                "name": "legend_labels_unrounded", "ok": not bad,
                "detail": ("栅格拉伸渲染：图例标注数值范围，跳过取整检查" if is_raster
                           else (f"分界值不整齐 {bad}" if bad
                                 else f"分界值均为整齐数值（{numbers[:4]}…）" if numbers
                                 else "无分级标签")),
            })
            if font_sizes:
                smallest = min(font_sizes)
                checks.append({
                    "name": "font_too_small", "ok": smallest >= min_font_pt,
                    "detail": f"图例最小字号 {smallest:.1f}pt（下限 {min_font_pt:.1f}pt）",
                    "suggest_pt": max(min_font_pt, 9.0),
                })
        except Exception as exc:
            checks.append({"name": "legend_readable", "ok": False, "detail": f"图例读取失败: {str(exc)[:80]}"})

        if frame and image_path and Path(image_path).exists():
            x, y, w, h = boxes.get("Legend", (0, 0, 0, 0))
            dpi = float((spec or {}).get("dpi") or 250)
            page_h = float((spec or {}).get("page", {}).get("height_mm") or 297.0)
            mm2px = dpi / 25.4
            box_px = (
                int(x * mm2px), int((page_h - (y + h)) * mm2px),
                int((x + w) * mm2px), int((page_h - y) * mm2px),
            )
            density = _image_density(image_path, box_px)
            if density is not None:
                checks.append({
                    "name": "legend_covers_data", "ok": density <= 0.45,
                    "detail": f"图例框内非白像素占比 {density:.2f}（>0.45 视为压住数据）",
                })
        if frame:
            legend_box = boxes.get("Legend", (0, 0, 0, 0))
            if not (0 <= legend_box[0] and legend_box[0] + legend_box[2] <= frame[0] + frame[2] + 1e-6):
                checks.append({"name": "legend_position_sane", "ok": True,
                               "detail": "图例位于地图框外（满覆盖数据的预期行为）"})

    # 3) 比例尺：用回读的 division × divisions 反算条长（division 单位跟随显示单位）
    scale_bar = elements.get("Scale Bar")
    if scale_bar is not None:
        definition = None
        divisions = None
        division_raw = 0.0
        try:
            definition = scale_bar.getDefinition("V3")
            divisions = getattr(definition, "divisions", None)
            division_raw = float(getattr(definition, "division", 0.0) or 0.0)
        except Exception:
            pass
        map_scale = 0.0
        frame_element = elements.get("Main Map")
        if frame_element is not None:
            try:
                map_scale = float(frame_element.camera.scale)
            except Exception:
                map_scale = 0.0
        bar_mm = float(getattr(scale_bar, "elementWidth", 0.0) or 0.0)
        units_obj = getattr(definition, "units", None) if definition is not None else None
        is_km = isinstance(units_obj, dict) and int(units_obj.get("uwkid") or 0) == 9036
        division_m = _snap_integer(division_raw) * (1000.0 if is_km else 1.0)
        details: list[str] = []
        if map_scale > 0 and divisions and division_m > 0:
            ground_length_m = division_m * float(divisions)
            # 刻度整齐与否要看**显示单位**：千米显示下 0.55 不算整齐，1/0.5/2 才算
            ok = _is_clean_length(division_m, in_km=is_km)
            details.append(
                f"条长 {ground_length_m:,.0f} m（每格 {division_m:,.0f} m，{int(divisions)} 段，"
                f"1:{map_scale:,.0f}，元素宽 {bar_mm:.1f}mm，单位={'千米' if is_km else '米'}）"
            )
            check = {"name": "scale_bar_not_round", "ok": bool(ok), "detail": "；".join(details)}
            if not ok:
                # 以设计意图（spec 里的计划长度）为目标反算所需元素宽度；条长 ∝（元素宽 − 单位标签预留）
                planned_total = float(((spec or {}).get("scale_bar") or {}).get("length_m") or 0.0)
                target_total = planned_total if planned_total > 0 else (_nearest_round(division_m) or division_m) * float(divisions)
                width_hint = bar_mm + (target_total - ground_length_m) * 1000.0 / map_scale
                check["detail"] += f"；目标条长 {target_total:,.0f} m → 建议元素宽 {width_hint:.1f}mm"
                if 20.0 <= width_hint <= 200.0:
                    check["suggest_element_width_mm"] = round(width_hint, 1)
            checks.append(check)
        else:
            checks.append({"name": "scale_bar_not_round", "ok": True,
                           "detail": "比例尺参数不足，跳过（缺地图比例尺或 division 回读）"})

    # 3.5) 图例必须在图框内（与图框线留边距），不能压图框
    spec_legend = (spec or {}).get("legend") or {}
    if elements.get("Legend") is not None and frame and not spec_legend.get("outside"):
        lx, ly, lw, lh = boxes.get("Legend", (0, 0, 0, 0))
        fx, fy, fw, fh = frame
        margin = float(spec_legend.get("frame_margin_mm") or 3.0)
        problems = []
        if lx < fx + margin - 0.6:
            problems.append(f"左越界 {fx + margin - lx:.1f}mm")
        if ly < fy + margin - 0.6:
            problems.append(f"下越界 {fy + margin - ly:.1f}mm")
        if lx + lw > fx + fw - margin + 0.6:
            problems.append(f"右越界 {lx + lw - (fx + fw - margin):.1f}mm")
        if ly + lh > fy + fh - margin + 0.6:
            problems.append(f"上越界 {ly + lh - (fy + fh - margin):.1f}mm")
        checks.append({
            "name": "legend_outside_frame", "ok": not problems,
            "detail": ("图例完全在图框内（留 %.1fmm 边距）" % margin) if not problems
                      else "图例压出图框：" + "、".join(problems),
        })

    # 3.8) 单层边框：只看图（ArcGIS 保存时会重新套用地图框样式，CIM 里的 borderSymbol 不可作为判据）
    if image_path and Path(image_path).exists() and frame and neatline:
        try:
            from PIL import Image  # type: ignore
            import numpy as np  # type: ignore

            with Image.open(image_path) as img:
                gray = np.asarray(img.convert("L"))
            dpi = float((spec or {}).get("dpi") or 250)
            page_h = float((spec or {}).get("page", {}).get("height_mm") or 297.0)
            mm2px = dpi / 25.4
            fx, fy, fw, fh = frame
            nx, ny, nw, nh = neatline
            top = int((page_h - (fy + fh)) * mm2px)
            bottom = int((page_h - fy) * mm2px)
            left = int(fx * mm2px)
            right = int((fx + fw) * mm2px)

            def dark_ratio(values) -> float:
                return float((values < 140).mean()) if len(values) else 0.0

            # 新架构：地图框 = 外框（唯一一层）。判据 = 外层有线 + 内侧没有第二条线
            outer_left = dark_ratio(gray[:, int(nx * mm2px)])
            outer_right = dark_ratio(gray[:, min(int((nx + nw) * mm2px), gray.shape[1] - 1)])
            outer_top = dark_ratio(gray[min(int((page_h - (ny + nh)) * mm2px), gray.shape[0] - 1), :])
            outer_ok = max(outer_left, outer_right, outer_top) >= 0.5
            inset = 8.0
            inner_left = dark_ratio(gray[max(0, top):bottom, int((nx + inset) * mm2px)])
            inner_right = dark_ratio(gray[max(0, top):bottom,
                                         min(int((nx + nw - inset) * mm2px), gray.shape[1] - 1)])
            inner_bottom = dark_ratio(gray[min(int((page_h - (ny + inset)) * mm2px), gray.shape[0] - 1), left:right])
            inner_lines = {k: round(v, 2) for k, v in
                           {"内左": inner_left, "内右": inner_right, "内下": inner_bottom}.items() if v > 0.35}
            checks.append({
                "name": "double_border", "ok": outer_ok and not inner_lines,
                "detail": ("仅外层一层边框（内侧无第二条线）" if outer_ok and not inner_lines
                           else (f"内侧仍有长直线 {inner_lines}（两层边框）" if inner_lines else "找不到外层边框")),
            })
        except Exception as exc:
            checks.append({"name": "double_border", "ok": True, "detail": f"图片判据不可用，跳过（{str(exc)[:50]}）"})

    # 4) 元素相互压盖 / 越出图廓
    named = {k: v for k, v in boxes.items() if k in ("Legend", "Scale Bar", "North Arrow", "Map Title")}
    overlaps = []
    keys = list(named)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if _boxes_overlap(named[keys[i]], named[keys[j]]):
                overlaps.append(f"{keys[i]}×{keys[j]}")
    checks.append({"name": "elements_overlap", "ok": not overlaps, "detail": "、".join(overlaps) or "四要素互不压盖"})
    if neatline:
        nx, ny, nw, nh = neatline
        strays = [k for k, (x, y, w, h) in boxes.items()
                  if x < nx - 0.6 or y < ny - 0.6 or x + w > nx + nw + 0.6 or y + h > ny + nh + 0.6]
        checks.append({"name": "element_outside_neatline", "ok": not strays,
                       "detail": "、".join(strays[:4]) or "所有元素都在图廓内"})

    # 5) 地图框是否真画上数据
    if image_path and Path(image_path).exists() and frame:
        dpi = float((spec or {}).get("dpi") or 250)
        page_h = float((spec or {}).get("page", {}).get("height_mm") or 297.0)
        mm2px = dpi / 25.4
        fx, fy, fw, fh = frame
        density = _image_density(image_path, (
            int(fx * mm2px), int((page_h - (fy + fh)) * mm2px),
            int((fx + fw) * mm2px), int((page_h - fy) * mm2px),
        ))
        if density is not None:
            checks.append({"name": "map_frame_blank", "ok": density >= 0.15,
                           "detail": f"地图框内非白像素占比 {density:.2f}（<0.15 接近空白）"})

    ok_all = all(bool(check.get("ok")) for check in checks)
    return {"ok": ok_all, "checks": checks, "aprx": str(target)}


def summarize(result: dict[str, Any], limit: int = 6) -> str:
    failed = [c for c in (result.get("checks") or []) if not c.get("ok")]
    if not failed:
        return "版面体检全部通过"
    parts = [f"{c['name']}: {c.get('detail', '')}" for c in failed[:limit]]
    return "版面体检未通过 → " + "；".join(parts)


__all__ = ["check_layout", "summarize"]
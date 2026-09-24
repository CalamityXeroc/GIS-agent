# -*- coding: utf-8 -*-
"""把 LayoutSpec 落地到 ArcGIS（渲染层：只执行，不做任何设计判断）。

实测得到的关键 API 事实（写在注释里避免再踩）：
- 图名等文本只能用 CIM 建（``layout`` 没有 ``createTextElement``），字号属性是
  ``CIMTextSymbol.height``（**不是** ``fontSize``——旧代码用错属性，导致图名一直是默认小字号）。
- 图例的 ``showHeading`` / ``labelSymbol`` 等**不在 arcpy.mp 的 LegendItem 上**，
  必须走 ``legend.getDefinition("V3")`` 的 CIM 定义；直接对 LegendItem 赋值会静默无效。
- 比例尺的 ``division``（每格地面长度）决定刻度数字，配合 ``divisions``/``units``/``unitLabel`` 才能得到整数刻度。
- 样式项名称随安装语言变化（本机中文），因此按名字从目录里取，取不到就报出来而不是静默用第一项。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import design as design_mod
from . import styles as styles_mod


def _arcpy():
    import arcpy  # type: ignore

    return arcpy


def _blank_template() -> str:
    arcpy = _arcpy()
    install = arcpy.GetInstallInfo().get("InstallDir", "") or ""
    candidate = os.path.join(
        install, "Resources", "ArcToolBox", "Services", "routingservices", "data", "Blank.aprx"
    )
    if os.path.exists(candidate):
        return candidate
    import glob

    found = glob.glob(os.path.join(install, "Resources", "**", "Blank.aprx"), recursive=True)
    if not found:
        raise RuntimeError("未找到 ArcGIS 空白工程模板 Blank.aprx")
    return found[0]


def _set(obj: Any, **props: Any) -> dict[str, str]:
    """逐个尝试设置属性，记录成功/失败（便于排查 API 名字差异）。"""
    result: dict[str, str] = {}
    for key, value in props.items():
        if value is None:
            continue
        try:
            setattr(obj, key, value)
            result[key] = "ok"
        except Exception as exc:  # pragma: no cover - 依赖 ArcGIS 版本
            result[key] = f"fail:{type(exc).__name__}"
    return result


#: 常见色带的 5 级配色（ArcGIS 的 classBreak 符号颜色有时读不到，用于图例色块兜底）
_RAMP_PALETTES: dict[str, list[str]] = {
    "ylorrd": ["#FFFFB2", "#FECC5C", "#FD8D3C", "#F03B20", "#BD0026"],
    "greens": ["#E8F6E1", "#BDE4B0", "#8ED180", "#56AD50", "#237D32"],
    "blues": ["#DEEBF7", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"],
    "reds": ["#FEE0D2", "#FC9272", "#FB6A4A", "#DE2D26", "#A50F15"],
    "purples": ["#EFEDF5", "#BCBDDC", "#9E9AC8", "#756BB1", "#54278F"],
    "oranges": ["#FEEDDE", "#FDBE85", "#FD8D3C", "#E6550D", "#A63603"],
    "greys": ["#F0F0F0", "#BDBDBD", "#969696", "#636363", "#252525"],
    "gray": ["#F0F0F0", "#BDBDBD", "#969696", "#636363", "#252525"],
    "rdylgn": ["#A50026", "#F46D43", "#FFFFBF", "#66BD63", "#1A9850"],
    "rdbu": ["#B2182B", "#EF8A62", "#F7F7F7", "#67A9CF", "#2166AC"],
}


def _ramp_colors(ramp_name: str, count: int) -> list[str]:
    """按色带名取 count 个颜色（找不到就返回空列表）。"""
    key = str(ramp_name or "").strip().lower().replace(" ", "")
    palette = _RAMP_PALETTES.get(key)
    if not palette:
        for name, colors in _RAMP_PALETTES.items():
            if name and name in key:
                palette = colors
                break
    if not palette:
        return []
    if count <= len(palette):
        # 均匀取样，保证首尾都用上
        step = (len(palette) - 1) / max(1, count - 1) if count > 1 else 0
        return [palette[min(len(palette) - 1, int(round(index * step)))] for index in range(count)]
    return [palette[index % len(palette)] for index in range(count)]


def _symbol_hex(symbol: Any) -> str:
    """取符号颜色的 #RRGGBB（ArcGIS 返回 Color 对象或 dict 两种形态）。"""
    color = getattr(symbol, "color", None)
    if color is None:
        return ""
    rgb = getattr(color, "RGB", None)
    if rgb is None and isinstance(color, dict):
        rgb = color.get("RGB")
    if not rgb:
        return ""
    try:
        return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    except Exception:
        return ""


def _hex_to_rgb(value: str) -> list[int]:
    text = str(value).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    return [int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16), 100]


def _parse_color_map(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for chunk in str(text or "").split(";"):
        if ":" in chunk:
            key, value = chunk.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def _style_item(project: Any, catalog: dict[str, list[str]], kind: str, keywords: list[str], notes: list[str]) -> Any:
    name = styles_mod.pick_style_name(catalog, kind, keywords or None, fallback_index=None)
    if not name:
        notes.append(f"{kind}: 未匹配到样式名（关键字 {keywords}），使用 ArcGIS 默认样式")
        return None
    try:
        for item in project.listStyleItems("ArcGIS 2D", kind):
            if item.name == name:
                notes.append(f"{kind}: 使用样式「{name}」")
                return item
    except Exception as exc:  # pragma: no cover
        notes.append(f"{kind}: 取样式失败 {str(exc)[:60]}")
    return None


def _ensure_project(aprx_path: str, *, append: bool = False) -> tuple[Any, str, list[str]]:
    """准备工程：默认从空白模板重新出图（同名工程直接重建，避免布局越堆越多）。

    ``append=True`` 时才复用已有工程（往里面追加一张图），且**不删除旧布局**
    （实测 `project.deleteItem(layout)` 会让 ArcGIS 直接崩掉）。
    """
    from arcpy import mp  # type: ignore

    notes: list[str] = []
    target = Path(aprx_path).resolve() if aprx_path else Path(tempfile.gettempdir()) / "map_render.aprx"
    target.parent.mkdir(parents=True, exist_ok=True)
    template = _blank_template()
    if append and target.exists():
        shutil.copy2(template, target.with_suffix(target.suffix + ".incoming"))
        # 复用：直接打开已有工程（不删任何东西）
        project = mp.ArcGISProject(str(target))
        try:
            os.remove(str(target.with_suffix(target.suffix + ".incoming")))
        except Exception:
            pass
        notes.append(f"复用已有工程（追加布局）: {target}")
        return project, str(target), notes
    if target.exists():
        target.unlink()
    shutil.copy2(template, target)
    project = mp.ArcGISProject(str(target))
    notes.append(f"从空白模板重新出图: {target}")
    return project, str(target), notes


def _hide_frame_border(map_frame: Any, notes: list[str]) -> None:
    """把地图框自带边框改成**白色**，等效于"无边框"，只保留外层图廓线（避免两层边框）。

    实测要点：
    - 边框符号在 CIM 的 ``CIMMapFrame.graphicFrame.borderSymbol``（不在 arcpy.mp 属性上）；
    - 置 ``None`` 只对当次会话有效，**保存/重开工程后 ArcGIS 会重新套用地图框样式**把黑边带回来；
    - 改成白色 CIMSymbolReference 则能持久（重开后导出图框边缘暗像素 0）。
    """
    try:
        from arcpy import cim  # type: ignore

        definition = map_frame.getDefinition("V3")
        graphic_frame = getattr(definition, "graphicFrame", None)
        if graphic_frame is None or not hasattr(graphic_frame, "borderSymbol"):
            return
        reference = cim.CIMSymbolReference()
        stroke = cim.CIMSolidStroke()
        stroke.width = 0.5
        stroke.color = cim.CIMRGBColor()
        stroke.color.red, stroke.color.green, stroke.color.blue = 0, 0, 0
        reference.symbol = stroke
        graphic_frame.borderSymbol = reference
        map_frame.setDefinition(definition)
        notes.append("map_frame: 边框已统一为 0.5pt 黑色细线（地图框即外框，只有一层）")
    except Exception as exc:  # pragma: no cover
        notes.append(f"map_frame 边框处理失败: {str(exc)[:80]}")


def _apply_text_element_options(layout: Any, element_name: str, *, font: str, height_pt: float,
                                bold: bool, notes: list[str]) -> None:
    """用 arcpy.mp 的 TEXT_ELEMENT 属性调字号/字体。

    实测：`CIMTextGraphic` 会让 ArcGIS **静默崩溃**；`CIMParagraphTextSymbol` 本版本不存在；
    而建出来的 `TEXT_ELEMENT` 自身有 `textSize` / `fontFamilyName` / `fontStyleName` 可写——
    这是唯一可靠且不崩的控字号方式。
    """
    try:
        for element in layout.listElements():
            if str(getattr(element, "name", "")) != element_name:
                continue
            applied = _set(
                element,
                textSize=float(height_pt),
                fontFamilyName=font,
                fontStyleName="Bold" if bold else "Regular",
            )
            notes.append(f"title(元素API {element_name}): {applied}")
    except Exception as exc:  # pragma: no cover
        notes.append(f"title 元素属性设置失败: {str(exc)[:80]}")


def _add_text(layout: Any, box: dict[str, float], text: str, *, font: str, height_pt: float,
              bold: bool, align: str = "center") -> None:
    """用 CIM 加文本。

    实测踩坑：`CIMParagraphTextGraphic` 必须配 `CIMParagraphTextSymbol`，配 `CIMTextSymbol` 会被忽略
    （图名一直是默认小字号就是这个原因）。所以这里用单行文本图形 `CIMTextGraphic` + `CIMTextSymbol`，
    多行时按行拆成多个元素堆叠。
    """
    from arcpy import cim  # type: ignore

    arcpy = _arcpy()
    lines = [chunk for chunk in str(text).split("\n") if chunk.strip()] or [str(text)]
    line_h = float(box["h"]) / max(1, len(lines))
    definition = layout.getDefinition("V3")
    for index, line in enumerate(lines):
        graphic = cim.CIMParagraphTextGraphic()   # 注：CIMTextGraphic 会崩，必须用段落图形
        graphic.text = line
        symbol = cim.CIMTextSymbol()
        symbol.fontFamilyName = font
        symbol.height = float(height_pt)          # ← 关键：CIM 字号属性是 height
        symbol.bold = bool(bold)
        symbol.horizontalAlignment = align
        symbol.verticalAlignment = "Center"
        symbol.color = cim.CIMRGBColor()
        symbol.color.red, symbol.color.green, symbol.color.blue = 0, 0, 0
        graphic.symbol = symbol
        top = float(box["y"]) + float(box["h"]) - (index + 1) * line_h
        graphic.shape = arcpy.Polygon(arcpy.Array([
            arcpy.Point(box["x"], top),
            arcpy.Point(box["x"] + box["w"], top),
            arcpy.Point(box["x"] + box["w"], top + line_h),
            arcpy.Point(box["x"], top + line_h),
            arcpy.Point(box["x"], top),
        ]))
        element = cim.CIMGraphicElement()
        element.name = "Map Title" if index == 0 else f"Map Title {index + 1}"
        element.graphic = graphic
        definition.elements.append(element)
    layout.setDefinition(definition)


def _add_neatline(layout: Any, box: dict[str, float]) -> None:
    """图廓线：用多边形描边（实测 CIMLineGraphic 配多边形 shape 会得到 NaN 几何画不出来）。"""
    from arcpy import cim  # type: ignore

    arcpy = _arcpy()
    definition = layout.getDefinition("V3")
    graphic = cim.CIMPolygonGraphic()
    symbol = cim.CIMPolygonSymbol()
    stroke = cim.CIMSolidStroke()
    stroke.color = cim.CIMRGBColor()
    stroke.color.red, stroke.color.green, stroke.color.blue = 0, 0, 0
    stroke.width = float(box.get("line_width", 0.5))
    symbol.symbolLayers = [stroke]
    graphic.symbol = symbol
    graphic.shape = arcpy.Polygon(arcpy.Array([
        arcpy.Point(box["x"], box["y"]),
        arcpy.Point(box["x"] + box["w"], box["y"]),
        arcpy.Point(box["x"] + box["w"], box["y"] + box["h"]),
        arcpy.Point(box["x"], box["y"] + box["h"]),
        arcpy.Point(box["x"], box["y"]),
    ]))
    element = cim.CIMGraphicElement()
    element.name = "Neatline"
    element.graphic = graphic
    definition.elements.append(element)
    layout.setDefinition(definition)


def _apply_graduated(layer: Any, spec: dict[str, Any], project: Any, notes: list[str]) -> None:
    renderer_spec = spec["renderer"]
    sym = layer.symbology
    sym.updateRenderer("GraduatedColorsRenderer")
    renderer = sym.renderer
    method = str(renderer_spec.get("classification_method") or "DefinedInterval")
    interval = renderer_spec.get("interval_size")
    applied = _set(
        renderer,
        classificationField=renderer_spec.get("field") or None,
        breakCount=int(renderer_spec.get("class_count") or 5),
        classificationMethod=method,
    )
    if interval:
        applied.update(_set(renderer, intervalSize=float(interval)))
    notes.append("renderer: " + str(applied))
    ramp_name = str(renderer_spec.get("color_ramp") or "")
    if ramp_name:
        try:
            for ramp in project.listColorRamps():
                if ramp_name.lower() in ramp.name.lower():
                    renderer.colorRamp = ramp
                    notes.append(f"color_ramp: {ramp.name}")
                    break
        except Exception as exc:  # pragma: no cover
            notes.append(f"color_ramp 设置失败: {str(exc)[:60]}")
    layer.symbology = sym

    # 显式分级（统一图例）：把给定上界写进 CIM classBreaks.upperBound。
    # 为什么必须走 CIM：元素 API 的 upperBound 写不进去；而 DefinedInterval 会让每张图
    # 从各自最小值起算，多期图无法共用图例（4 期核密度图对比的硬需求）。
    explicit = sorted(float(value) for value in (renderer_spec.get("explicit_bounds") or []))
    if explicit:
        # 语义：给定分级边界（n 个边界 → n-1 级），所以只把 boundaries[1:] 写成各级上界
        upper_bounds = explicit[1:]
        try:
            layer_def_bounds = layer.getDefinition("V3")
            cim_renderer = getattr(layer_def_bounds, "renderer", None)
            # 实测（探针 probe_vector_bounds2）：矢量 graduated 的 CIM 分级挂在
            # ``renderer.breaks`` 上；``classificationMethod`` 也必须走 CIM 才生效
            # （元素 API 上赋值会被忽略，仍是 StandardDeviation）。老版本用 classBreaks 命名。
            cim_breaks = getattr(cim_renderer, "breaks", None)
            if cim_breaks is None:
                cim_breaks = getattr(cim_renderer, "classBreaks", None)
            cim_breaks = list(cim_breaks or [])
            if upper_bounds and len(cim_breaks) == len(upper_bounds):
                try:
                    cim_renderer.classificationMethod = "Manual"
                except Exception:
                    pass
                for brk, upper in zip(cim_breaks, upper_bounds):
                    brk.upperBound = float(upper)
                layer.setDefinition(layer_def_bounds)
                notes.append(f"explicit_bounds(统一图例): 边界={explicit} 上界={upper_bounds}")
            else:
                notes.append(
                    f"explicit_bounds 给 {len(explicit)} 个边界（{len(upper_bounds)} 级），"
                    f"但渲染器有 {len(cim_breaks)} 级，已忽略"
                )
        except Exception as exc:  # pragma: no cover
            notes.append(f"explicit_bounds 写入失败: {str(exc)[:80]}")

    # 用实际断点重写分级标签（语义化 / 取整区间）。必须走 CIM：元素 API 上设了图例不认。
    try:
        layer_def = layer.getDefinition("V3")
        cim_renderer = getattr(layer_def, "renderer", None)
        cim_breaks = getattr(cim_renderer, "breaks", None)
        if cim_breaks is None:
            cim_breaks = getattr(cim_renderer, "classBreaks", None)
        cim_breaks = list(cim_breaks or [])
        bound_source = [float(brk.upperBound) for brk in cim_breaks]
        if not bound_source:
            bound_source = [float(brk.upperBound) for brk in (layer.symbology.renderer.classBreaks or [])]
        bounds: list[tuple[float, float]] = []
        low = float(renderer_spec.get("field_stats", {}).get("min") or 0.0)
        for upper in bound_source:
            bounds.append((low, upper))
            low = upper
        if bound_source:
            # 回写“实际生效”的分级上界：断言（如 shared_legend_bounds）应比对实测值而非假设值
            spec["renderer"]["applied_bounds"] = bound_source
        labels = design_mod.format_class_labels(bounds, str(renderer_spec.get("labels_mode") or "semantic"))
        layer_def = layer.getDefinition("V3")
        cim_breaks = getattr(getattr(layer_def, "renderer", None), "classBreaks", None) or []
        updated = 0
        for brk, label in zip(cim_breaks, labels):
            if hasattr(brk, "label"):
                brk.label = label
                updated += 1
        # 采集"图例条目（标签+颜色）"：ArcGIS 的标签格式无法定制（6 位小数），
        # 导出后用叠加层按我们的格式重绘（见 image_overlay.stamp_legend）
        entries = []
        try:
            breaks = list(layer.symbology.renderer.classBreaks or [])
            ramp_fallback = _ramp_colors(str(renderer_spec.get("color_ramp") or ""), len(breaks) or 5)
            for index, (brk, label) in enumerate(zip(breaks, labels)):
                color = _symbol_hex(getattr(brk, "symbol", None))
                if not color and isinstance(getattr(brk, "symbol", None), dict):
                    color = _symbol_hex(brk.symbol)
                if not color and ramp_fallback:
                    color = ramp_fallback[index % len(ramp_fallback)]
                entries.append({"label": label, "color": color or "#CCCCCC"})
        except Exception:
            entries = []
        if entries:
            spec["renderer"]["legend_entries"] = entries
        if updated:
            layer.setDefinition(layer_def)
            notes.append(f"class_labels(CIM): {labels}")
        else:
            renderer_type = type(getattr(layer_def, "renderer", None)).__name__
            # 元素 API 兜底：部分版本 CIM 挂在 renderer.classBreaks 上读不到，但元素 API 能生效
            fallback = 0
            for brk, label in zip(layer.symbology.renderer.classBreaks or [], labels):
                try:
                    brk.label = label
                    fallback += 1
                except Exception:
                    continue
            if fallback:
                layer.symbology = layer.symbology
                notes.append(f"class_labels(元素API 兜底): {labels}")
            else:
                notes.append(f"class_labels: 无法写标签（CIM={renderer_type}）")
    except Exception as exc:  # pragma: no cover
        notes.append(f"class_labels 设置失败: {str(exc)[:80]}")


def _parse_name_map(raw: Any) -> dict[str, str]:
    """解析“值→显示名”映射，支持 "1:水域;2:林地" 字符串或 {"1": "水域"} 字典。

    用途：竞赛/行业规范里类别往往用编码存储（如土地覆盖 1=水域 2=林地 5=耕地），
    图例必须显示名称而不是编码。
    """
    mapping: dict[str, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            mapping[str(key).strip()] = str(value).strip()
        return mapping
    for pair in str(raw or "").split(";"):
        if ":" in pair or "=" in pair:
            sep = ":" if ":" in pair else "="
            key, value = pair.split(sep, 1)
            if key.strip():
                mapping[key.strip()] = value.strip()
    return mapping


def _apply_unique(layer: Any, spec: dict[str, Any], notes: list[str]) -> None:
    renderer_spec = spec["renderer"]
    sym = layer.symbology
    sym.updateRenderer("UniqueValueRenderer")
    field = renderer_spec.get("field")
    if field:
        try:
            sym.renderer.fields = [field]
        except Exception as exc:  # pragma: no cover
            notes.append(f"unique.fields 设置失败: {str(exc)[:60]}")
    color_map = _parse_color_map(renderer_spec.get("category_colors") or "")
    name_map = _parse_name_map(renderer_spec.get("category_names") or "")
    applied: dict[str, str] = {}
    for group in sym.renderer.groups:
        for item in getattr(group, "items", []) or []:
            values = list(item.values) if item.values else []
            key = ""
            if values:
                first = values[0]
                key = str(first[0]) if isinstance(first, (list, tuple)) else str(first)
            hex_color = color_map.get(key, renderer_spec.get("nodata_color") or "#D9D9D9")
            try:
                item.symbol.color = {"RGB": _hex_to_rgb(hex_color)}  # type: ignore[assignment]
            except Exception:
                pass
            display = name_map.get(key, key)
            try:
                item.label = display or str(getattr(item, "label", ""))
            except Exception:
                pass
            applied[key] = hex_color
    layer.symbology = sym
    if applied:
        notes.append(f"unique_colors: {applied}")
        if name_map:
            notes.append(f"category_names: {name_map}")
        spec["renderer"]["legend_entries"] = [
            {"label": name_map.get(key, key), "color": color} for key, color in applied.items()
        ]


def _apply_raster_classify(layer: Any, spec: dict[str, Any], project: Any, notes: list[str]) -> bool:
    """栅格分类渲染 + 显式分级（统一图例的栅格路径）。

    为什么必须有这条路径：核密度这类成果图是**栅格**，多期图要能横向对比就必须共用
    同一套分级。实测：栅格图层的 ``symbology`` **没有** ``updateRenderer``，色带外的
    分级完全改不动；只能走 CIM，把 colorizer 换成 ``CIMRasterClassifyColorizer`` 并写
    ``classBreaks.upperBound``（探针确认的属性：classBreaks/classificationMethod/
    colorRamp/field/minimumBreak）。

    返回是否成功；失败时调用方退回拉伸渲染并如实记录（绝不因样式让整张图失败）。
    """
    renderer_spec = spec["renderer"]
    bounds = sorted(float(value) for value in (renderer_spec.get("explicit_bounds") or []))
    if not bounds:
        return False
    try:
        from arcpy import cim  # type: ignore
    except Exception as exc:  # pragma: no cover
        notes.append(f"raster_classify: 无法导入 CIM({str(exc)[:40]})")
        return False

    labels_mode = str(renderer_spec.get("labels_mode") or "range")
    # 给定的是分级边界（n 个边界 → n-1 级）：第一个边界即首级下界
    low = float(bounds[0])
    upper_bounds = bounds[1:]
    if not upper_bounds:
        return False
    labels = design_mod.format_class_labels(_bounds_pairs(upper_bounds, low), labels_mode)
    ramp_colors = _ramp_colors(str(renderer_spec.get("color_ramp") or ""), len(upper_bounds))
    if not ramp_colors:
        # 兜底色带：没有色带名时也必须给每级颜色，否则分类栅格会渲染成空白
        # （实测：不设 color 的 CIMRasterClassifyColorizer 出图后看不到栅格）
        ramp_colors = _ramp_colors("YlOrRd", len(upper_bounds)) or [
            "#FFFFCC", "#FFEDA0", "#FEB24C", "#F03B20", "#BD0026",
        ]
    try:
        definition = layer.getDefinition("V3")
        colorizer = cim.CIMRasterClassifyColorizer()
        colorizer.classificationMethod = "Manual"
        colorizer.minimumBreak = low
        # 字段：**默认不设**。单波段栅格（核密度/高程这类）没有 "Value" 字段，
        # 设了会让分类渲染失效、出图后图面空白（实测）。只有多波段/带属性表的栅格
        # 才需要显式指定 raster_field。
        field = str(renderer_spec.get("raster_field") or "")
        if field:
            colorizer.field = field
        breaks = []
        for index, upper in enumerate(upper_bounds):
            brk = cim.CIMRasterClassBreak()
            brk.upperBound = float(upper)
            brk.label = labels[index] if index < len(labels) else str(upper)
            if ramp_colors:
                # 注意 _hex_to_rgb 返回 [r, g, b, alpha]，不能直接 unpack 成 3 个
                rgb = _hex_to_rgb(ramp_colors[index % len(ramp_colors)])
                color = cim.CIMRGBColor()
                color.red, color.green, color.blue = rgb[0], rgb[1], rgb[2]
                brk.color = color
            breaks.append(brk)
        colorizer.classBreaks = breaks
        definition.colorizer = colorizer
        layer.setDefinition(definition)

        # 读回实测值（断言比对的是实测分级，不是我们以为的分级）
        back = layer.getDefinition("V3").colorizer
        applied = [float(brk.upperBound) for brk in (getattr(back, "classBreaks", None) or [])]
        if not applied:
            notes.append("raster_classify: 写完后读不到 classBreaks，视为失败")
            return False
        spec["renderer"]["applied_bounds"] = applied
        entries = []
        for index, brk in enumerate(getattr(back, "classBreaks", None) or []):
            hex_color = _symbol_hex(getattr(brk, "color", None)) or (
                ramp_colors[index % len(ramp_colors)] if ramp_colors else "#CCCCCC"
            )
            label = str(getattr(brk, "label", "") or (labels[index] if index < len(labels) else ""))
            entries.append({"label": label, "color": hex_color})
        if entries:
            spec["renderer"]["legend_entries"] = entries
        notes.append(f"raster_classify(统一图例): {applied}")
        return True
    except Exception as exc:  # pragma: no cover
        notes.append(f"raster_classify 失败: {str(exc)[:90]}")
        return False


def _bounds_pairs(bounds: list[float], low: float) -> list[tuple[float, float]]:
    """把上界列表转成 (下界, 上界) 区间列表（供标签格式化）。"""
    pairs: list[tuple[float, float]] = []
    for upper in bounds:
        pairs.append((low, float(upper)))
        low = float(upper)
    return pairs


def _apply_raster_stretch(layer: Any, spec: dict[str, Any], project: Any, notes: list[str]) -> None:
    """栅格：尽量设置拉伸渲染的色带。

    实测：栅格图层的 ``symbology`` **没有** ``updateRenderer``（与矢量不同），
    因此这里只做"能设就设"的色带替换；设不上就沿用 ArcGIS 默认渲染，并如实记录，
    绝不因为渲染样式把整张图搞失败。
    """
    renderer_spec = spec["renderer"]
    sym = layer.symbology
    ramp_name = str(renderer_spec.get("color_ramp") or "")
    renderer = getattr(sym, "renderer", None)
    applied = False
    if ramp_name and renderer is not None and hasattr(renderer, "colorRamp"):
        try:
            for ramp in project.listColorRamps():
                if ramp_name.lower() in ramp.name.lower():
                    renderer.colorRamp = ramp
                    notes.append(f"raster_ramp: {ramp.name}")
                    applied = True
                    break
        except Exception as exc:  # pragma: no cover
            notes.append(f"raster_ramp 设置失败: {str(exc)[:60]}")
    if applied:
        try:
            layer.symbology = sym
        except Exception:
            pass
    else:
        notes.append("raster_ramp: 栅格 symbology 不支持直接改渲染器，沿用 ArcGIS 默认渲染")


def _apply_single(layer: Any, spec: dict[str, Any], notes: list[str]) -> None:
    color_map = _parse_color_map(spec["renderer"].get("category_colors") or "")
    hex_color = next(iter(color_map.values()), "") if color_map else ""
    if not hex_color:
        return
    try:
        sym = layer.symbology
        sym.renderer.symbol.color = {"RGB": _hex_to_rgb(hex_color)}  # type: ignore[assignment]
        layer.symbology = sym
        notes.append(f"single_color: {hex_color}")
    except Exception as exc:  # pragma: no cover
        notes.append(f"single_color 失败: {str(exc)[:60]}")


def _configure_legend(legend: Any, spec: dict[str, Any], notes: list[str]) -> None:
    legend_spec = spec["legend"]
    applied = _set(
        legend,
        title=legend_spec.get("title") or "",
        autoAdd=False,
        elementPositionX=legend_spec["x"],
        elementPositionY=legend_spec["y"],
        elementWidth=max(15.0, float(legend_spec.get("w") or 40.0)),
        elementHeight=max(10.0, float(legend_spec.get("h") or 20.0)),
    )
    notes.append("legend: " + str(applied))
    try:
        definition = legend.getDefinition("V3")
        definition.showTitle = bool(legend_spec.get("title"))
        if hasattr(definition, "titleSymbol"):
            definition.titleSymbol.height = float(legend_spec.get("title_pt") or 11.0)
            definition.titleSymbol.fontFamilyName = "宋体"
        if hasattr(definition, "frame") and definition.frame is not None:
            try:
                definition.frame.visible = bool(legend_spec.get("border", False))
            except Exception:
                pass
        for item in getattr(definition, "items", []) or []:
            for attr, value in (
                ("showHeading", False),
                ("showLayerName", bool(legend_spec.get("show_layer_name", False))),
                ("showLabels", True),
                ("showVisibleFeatures", None),
                ("showDescription", False),
                ("showCounts", False),
            ):
                if value is None or not hasattr(item, attr):
                    continue
                try:
                    setattr(item, attr, value)
                except Exception:
                    continue
            for sym_attr in ("labelSymbol", "headingSymbol", "layerNameSymbol", "descriptionSymbol"):
                symbol = getattr(item, sym_attr, None)
                if symbol is not None and hasattr(symbol, "height"):
                    try:
                        symbol.height = float(legend_spec.get("font_pt") or 9.0)
                        symbol.fontFamilyName = "宋体"
                    except Exception:
                        continue
            try:
                item.patchWidth = float((legend_spec.get("swatch_mm") or [7.0, 5.0])[0])
                item.patchHeight = float((legend_spec.get("swatch_mm") or [7.0, 5.0])[1])
            except Exception:
                pass
        legend.setDefinition(definition)
        notes.append("legend_cim: showHeading/showLayerName/labelSymbol 已设置")
        # 先消除内容溢出：装不下时 ArcGIS 会把内容画到元素框之外（实测会跑到页面底部）
        try:
            for _ in range(4):
                if not bool(getattr(legend, "isOverflowing", False)):
                    break
                _set(legend,
                     elementWidth=float(legend.elementWidth) + 4.0,
                     elementHeight=float(legend.elementHeight) + 7.0)
            if bool(getattr(legend, "isOverflowing", False)):
                notes.append("legend: 仍标记为溢出（已尝试放大 4 次）")
            else:
                notes.append("legend: 内容未溢出")
        except Exception:
            pass
        # 回读实际框：ArcGIS 会按内容自动放大图例 → ①必须按实际框覆盖 ②越出图框就拉回内部
        try:
            actual_x = float(legend.elementPositionX)
            actual_y = float(legend.elementPositionY)
            actual_w = float(legend.elementWidth)
            actual_h = float(legend.elementHeight)
            frame_box = spec.get("map_frame") or {}
            margin = float((spec.get("legend") or {}).get("frame_margin_mm") or 4.0)
            if frame_box and not spec["legend"].get("outside"):
                fx, fy, fw, fh = (float(frame_box["x"]), float(frame_box["y"]),
                                  float(frame_box["w"]), float(frame_box["h"]))
                clamped_x = min(max(actual_x, fx + margin), fx + fw - margin - actual_w)
                clamped_y = min(max(actual_y, fy + margin), fy + fh - margin - actual_h)
                if abs(clamped_x - actual_x) > 0.01 or abs(clamped_y - actual_y) > 0.01:
                    _set(legend, elementPositionX=clamped_x, elementPositionY=clamped_y)
                    actual_x, actual_y = clamped_x, clamped_y
                    notes.append("legend: 实际框越出图框，已拉回图框内（留 %.1fmm 边距）" % margin)
            # 叠加层按**设计框**绘制（ArcGIS 图例随后隐藏，不参与成像）
            design_x = float((spec.get("legend") or {}).get("x") or actual_x)
            design_y = float((spec.get("legend") or {}).get("y") or actual_y)
            design_w = float((spec.get("legend") or {}).get("w") or actual_w)
            design_h = float((spec.get("legend") or {}).get("h") or actual_h)
            spec["legend"]["overlay_box"] = [design_x, design_y, design_w, design_h]
            # ArcGIS 图例的实际绘制位置不受元素坐标可靠控制（锚点行为），会留下框外残留；
            # 因此把它隐藏，由叠加层出图例（工程里仍保留该元素，可一键显示后自行调整）
            _set(legend, visible=False)
            notes.append("legend: 工程内图例已隐藏（避免 ArcGIS 重绘残留），导出图由叠加层绘制")
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover
        notes.append(f"legend_cim 失败: {str(exc)[:80]}")


def _configure_scale_bar(scale_bar: Any, spec: dict[str, Any], notes: list[str]) -> None:
    cfg = spec["scale_bar"]
    bar_mm = float(cfg.get("bar_mm") or 0.0)
    # 实测：比例尺地面长度 ≈ 元素宽度 × 地图比例尺；所以元素宽度直接取"计划条长"（不带留白），
    # 这样 ArcGIS 按元素宽度自适出来的总长就是计划中的整数长度（刻度才能是整数）。
    width = max(25.0, min(float(spec["scale_bar"].get("w") or 60.0), bar_mm))
    if cfg.get("element_width_mm"):
        width = float(cfg["element_width_mm"])   # 修复轮给出的校正宽度（把刻度拉到整数）
    # 先把 CIM 里的单位/字号设好，再用元素 API 收尾（反过来会被 setDefinition 重置）
    try:
        definition = scale_bar.getDefinition("V3")
        done = []
        for key, value in (("numberFormat", "#,##0"), ("labelPosition", "BELOW")):
            if hasattr(definition, key):
                try:
                    setattr(definition, key, value)
                    done.append(key)
                except Exception:
                    continue
        for symbol_name in ("labelSymbol", "unitLabelSymbol"):
            symbol = getattr(definition, symbol_name, None)
            if symbol is not None and hasattr(symbol, "height"):
                try:
                    symbol.height = float(cfg.get("font_pt") or 8.0)
                    symbol.fontFamilyName = "宋体"
                    done.append(symbol_name)
                except Exception:
                    continue
        scale_bar.setDefinition(definition)
        notes.append("scale_bar(CIM 字号/格式): " + ",".join(done))
    except Exception as exc:  # pragma: no cover
        notes.append(f"scale_bar(CIM) 失败: {str(exc)[:80]}")
    applied = _set(
        scale_bar,
        divisions=int(cfg.get("divisions") or 4),
        subdivisions=int(cfg.get("subdivisions") or 1),
        elementPositionX=cfg["x"],
        elementPositionY=cfg["y"],
        elementWidth=width,
        elementHeight=max(8.0, float(cfg.get("h") or 10.0)),
    )
    # 先设 CIM 里的字号/格式（units 是单位对象，本版本 cim.CIMLinearUnit 不存在，只能沿用样式默认的千米），
    # 再用元素 API 收尾设 divisions / 单位标签（实测必须放在 setDefinition 之后才不会被重置）。
    try:
        definition = scale_bar.getDefinition("V3")
        done = []
        for key, value in (("numberFormat", "#,##0"), ("labelPosition", "BELOW")):
            if hasattr(definition, key):
                try:
                    setattr(definition, key, value)
                    done.append(key)
                except Exception:
                    continue
        for symbol_name in ("labelSymbol", "unitLabelSymbol"):
            symbol = getattr(definition, symbol_name, None)
            if symbol is not None and hasattr(symbol, "height"):
                try:
                    symbol.height = float(cfg.get("font_pt") or 8.0)
                    symbol.fontFamilyName = "宋体"
                    done.append(symbol_name)
                except Exception:
                    continue
        scale_bar.setDefinition(definition)
        notes.append("scale_bar(CIM 字号/格式): " + ",".join(done))
        _set(scale_bar, divisions=int(cfg.get("divisions") or 4),
             subdivisions=int(cfg.get("subdivisions") or 1),
             unitLabel=cfg.get("unit_label") or "米")
        back = scale_bar.getDefinition("V3")
        # 渲染单位由样式决定（本机为千米，uwkid 9036）→ 单位标签必须与之一致，否则 0.5/1/2 配"米"是错的
        units_obj = getattr(back, "units", None)
        if isinstance(units_obj, dict) and int(units_obj.get("uwkid") or 0) == 9036:
            cfg["unit_label"] = "千米"
        elif isinstance(units_obj, dict) and int(units_obj.get("uwkid") or 0) == 9001:
            cfg["unit_label"] = "米"
        notes.append(
            "scale_bar: " + str(applied)
            + f"；回读 division={getattr(back, 'division', None)} divisions={getattr(back, 'divisions', None)} "
            + f"unitLabel={getattr(back, 'unitLabel', None)!r} elementWidth={getattr(scale_bar, 'elementWidth', None):.1f}mm "
            + f"（实际比例尺 1:{float(spec['scale_bar'].get('map_scale') or 0):,.0f}）"
        )
    except Exception as exc:  # pragma: no cover
        notes.append(f"scale_bar 收尾失败: {str(exc)[:100]}")


def _detect_arcgis_pro(notes: list[str]) -> None:
    """检测 ArcGIS Pro 是否在运行并提醒。

    实测：Pro 与我们的 arcpy 脚本同时对同一安装做 GP 调用会互相干扰（Pro 会弹严重错误甚至崩溃）。
    这里只提醒不阻断，由调用方决定。
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return
    try:
        running = [p.info.get("name", "") for p in psutil.process_iter(["name"])
                   if "arcgispro" in str(p.info.get("name", "")).lower()]
    except Exception:
        return
    if running:
        notes.append("警告：检测到 ArcGIS Pro 正在运行，与我们出图会互相干扰，建议先完全退出 Pro")


def render(
    spec: dict[str, Any],
    output_path: str,
    *,
    aprx_path: str = "",
    catalog: dict[str, list[str]] | None = None,
    append: bool = False,
) -> dict[str, Any]:
    """按 spec 渲染出图；返回结果字典（含 spec 与 notes，便于追溯决策）。"""
    arcpy = _arcpy()
    from arcpy import mp  # noqa: F401  # 触发 arcpy.mp 初始化

    arcpy.env.overwriteOutput = True
    notes: list[str] = []
    _detect_arcgis_pro(notes)

    target_aprx = aprx_path or str(Path(output_path).with_suffix(".aprx"))
    project, aprx_file, boot_notes = _ensure_project(target_aprx, append=append)
    notes.extend(boot_notes)
    catalog = catalog if catalog is not None else styles_mod.load_catalog()

    page = spec["page"]
    layout = project.createLayout(float(page["width_mm"]), float(page["height_mm"]), "MILLIMETER")
    layout.name = (spec["title"]["text"] or "Map")[:50]

    map_obj = project.listMaps()[0]
    for existing in list(map_obj.listLayers()):
        map_obj.removeLayer(existing)
    layer_paths = [p for p in (spec["renderer"].get("layer_paths") or []) if p]
    layers = [map_obj.addDataFromPath(path) for path in layer_paths]
    if not layers:
        raise RuntimeError("spec 里没有图层路径，无法出图")
    primary = layers[0]
    mode = spec["renderer"].get("mode")
    if mode == "graduated":
        _apply_graduated(primary, spec, project, notes)
    elif mode == "unique":
        _apply_unique(primary, spec, notes)
    elif mode == "raster_stretch":
        _apply_raster_stretch(primary, spec, project, notes)
        # 统一图例：栅格也要能共用分级（核密度多期对比）——显式分级存在时改用分类渲染
        if spec["renderer"].get("explicit_bounds"):
            if _apply_raster_classify(primary, spec, project, notes):
                pass
            else:
                notes.append("栅格显式分级未生效，已退拉伸渲染（统一图例可能不成立）")
    elif mode == "single":
        _apply_single(primary, spec, notes)

    frame = spec["map_frame"]
    map_frame = layout.createMapFrame(
        arcpy.Polygon(arcpy.Array([
            arcpy.Point(frame["x"], frame["y"]),
            arcpy.Point(frame["x"] + frame["w"], frame["y"]),
            arcpy.Point(frame["x"] + frame["w"], frame["y"] + frame["h"]),
            arcpy.Point(frame["x"], frame["y"] + frame["h"]),
            arcpy.Point(frame["x"], frame["y"]),
        ])),
        map_obj,
        "Main Map",
    )
    extent = spec.get("extent", {}).get("data") or {}
    padding = float(spec.get("extent", {}).get("padding") or 0.0)
    if extent.get("width"):
        pad_x = float(extent["width"]) * padding
        pad_y = float(extent["height"]) * padding
        map_frame.camera.setExtent(
            arcpy.Extent(
                float(extent["xmin"]) - pad_x, float(extent["ymin"]) - pad_y,
                float(extent["xmax"]) + pad_x, float(extent["ymax"]) + pad_y,
            )
        )
    else:
        map_frame.camera.setExtent(map_frame.getLayerExtent(primary))

    # 回写**实测**的地图框范围：叠加层（箭头/标注）要用它做数据坐标→像素换算。
    # 用设计值会有偏差（相机 setExtent 后会按屏幕纵横比微调），这类偏差会直接体现为箭头错位。
    try:
        # 实测：ArcGIS Pro 3.6 的 MapFrame **没有** getExtent()，范围挂在 camera 上
        # （与已知坑“Map 无 .camera”同源：这个版本的 API 与文档不一致）。
        camera = getattr(map_frame, "camera", None)
        frame_extent = camera.getExtent() if camera is not None else map_frame.getExtent()
        spec["render_frame_extent"] = {
            "xmin": float(frame_extent.XMin),
            "ymin": float(frame_extent.YMin),
            "xmax": float(frame_extent.XMax),
            "ymax": float(frame_extent.YMax),
            "width": float(frame_extent.XMax - frame_extent.XMin),
            "height": float(frame_extent.YMax - frame_extent.YMin),
        }
        notes.append(
            "render_frame_extent: "
            f"({spec['render_frame_extent']['xmin']:.0f},{spec['render_frame_extent']['ymin']:.0f})~"
            f"({spec['render_frame_extent']['xmax']:.0f},{spec['render_frame_extent']['ymax']:.0f})"
        )
    except Exception as exc:  # pragma: no cover
        notes.append(f"render_frame_extent 读取失败: {str(exc)[:60]}")

    if spec["legend"].get("needed"):
        legend_style = _style_item(project, catalog, "LEGEND", [], notes)
        legend = layout.createMapSurroundElement(
            arcpy.Polygon(arcpy.Array([
                arcpy.Point(spec["legend"]["x"], spec["legend"]["y"]),
                arcpy.Point(spec["legend"]["x"] + spec["legend"]["w"], spec["legend"]["y"]),
                arcpy.Point(spec["legend"]["x"] + spec["legend"]["w"], spec["legend"]["y"] + spec["legend"]["h"]),
                arcpy.Point(spec["legend"]["x"], spec["legend"]["y"] + spec["legend"]["h"]),
                arcpy.Point(spec["legend"]["x"], spec["legend"]["y"]),
            ])),
            "LEGEND", map_frame, legend_style,
        )
        _configure_legend(legend, spec, notes)
    else:
        notes.append("legend: 按设计不放图例（单一符号）")

    if spec["scale_bar"].get("visible", True):
        # 实测：元素宽度决定条长，而实际地图比例尺要等相机设定后才能读到——
        # 所以先读相机比例尺，再按“计划条长”反算元素宽度，这样刻度才能落到整数。
        actual_scale = 0.0
        try:
            actual_scale = float(map_frame.camera.scale)
        except Exception:
            actual_scale = 0.0
        if actual_scale > 0:
            spec["scale_bar"]["map_scale"] = round(actual_scale, 2)
            spec["scale_bar"]["bar_mm"] = float(spec["scale_bar"].get("length_m") or 0.0) * 1000.0 / actual_scale
        sb_style = _style_item(project, catalog, "SCALE_BAR", spec["scale_bar"].get("style_keywords") or [], notes)
        scale_bar = layout.createMapSurroundElement(
            arcpy.Polygon(arcpy.Array([
                arcpy.Point(spec["scale_bar"]["x"], spec["scale_bar"]["y"]),
                arcpy.Point(spec["scale_bar"]["x"] + spec["scale_bar"]["w"], spec["scale_bar"]["y"]),
                arcpy.Point(spec["scale_bar"]["x"] + spec["scale_bar"]["w"], spec["scale_bar"]["y"] + spec["scale_bar"]["h"]),
                arcpy.Point(spec["scale_bar"]["x"], spec["scale_bar"]["y"] + spec["scale_bar"]["h"]),
                arcpy.Point(spec["scale_bar"]["x"], spec["scale_bar"]["y"]),
            ])),
            "SCALE_BAR", map_frame, sb_style,
        )
        _configure_scale_bar(scale_bar, spec, notes)

    if spec["north_arrow"].get("needed"):
        na_style = _style_item(project, catalog, "NORTH_ARROW", spec["north_arrow"].get("style_keywords") or [], notes)
        north = layout.createMapSurroundElement(
            arcpy.Point(spec["north_arrow"]["x"], spec["north_arrow"]["y"]), "NORTH_ARROW", map_frame, na_style,
        )
        size = float(spec["north_arrow"].get("size_mm") or 16.0)
        _set(
            north,
            elementPositionX=spec["north_arrow"]["x"],
            elementPositionY=spec["north_arrow"]["y"],
            elementWidth=size,
            elementHeight=size,
        )
        notes.append(f"north_arrow: size={size}mm")

    # 图名必须在地图框**之后**添加（先加会被 createMapFrame 重置）
    # 图廓线不再用 CIM 画（实测 CIMPolygonGraphic 的几何不可靠），改为导出后按规格描边
    title = spec["title"]
    _add_text(
        layout,
        {"x": title["box"]["x"], "y": title["box"]["y"], "w": title["box"]["w"], "h": title["box"]["h"]},
        title["text"],
        font=title.get("font", "宋体"),
        height_pt=float(title.get("height_pt") or 18.0),
        bold=bool(title.get("bold", True)),
        align=str(title.get("align") or "center"),
    )
    _apply_text_element_options(
        layout, "Map Title",
        font=title.get("font", "宋体"),
        height_pt=float(title.get("height_pt") or 18.0),
        bold=bool(title.get("bold", True)),
        notes=notes,
    )
    notes.append(f"title: {title['text']} @ {title.get('height_pt')}pt")

    if spec.get("labels", {}).get("enabled"):
        try:
            primary.showLabels = True
        except Exception:
            pass

    # 边框清理必须放在最后：设置相机范围/其他元素会重建地图框定义，提前清会被覆盖
    _hide_frame_border(map_frame, notes)

    project.save()
    aprx_size = os.path.getsize(aprx_file)

    out = str(output_path)
    base, ext = os.path.splitext(out)
    if ext.lower() not in (".jpg", ".jpeg", ".pdf"):
        out = base + ".jpg"
    layout.exportToJPEG(out if out.lower().endswith((".jpg", ".jpeg")) else base + ".jpg",
                        resolution=int(spec.get("dpi") or 250))
    produced = next((p for p in (out, out + ".jpg", base + ".jpg") if os.path.exists(p)), "")
    if not produced:
        raise RuntimeError(f"JPG 未生成: {out}")

    # 标题按设计规格在导出图上叠加：ArcGIS 3.6 无法控制布局文本元素字号/对齐（见 image_overlay 注释）
    overlay = {}
    if str(spec.get("title_style", "overlay")) == "overlay":
        from .image_overlay import stamp_title

        tbox = spec["title"]["box"]
        overlay = stamp_title(
            produced,
            spec["title"]["text"],
            box_mm=(tbox["x"], tbox["y"], tbox["w"], tbox["h"]),
            page_height_mm=float(spec["page"]["height_mm"]),
            dpi=float(spec.get("dpi") or 250),
            font_name=str(spec["title"].get("font") or "宋体"),
            size_pt=float(spec["title"].get("height_pt") or 20.0),
            bold=bool(spec["title"].get("bold", True)),
        )
        # 比例尺单位标签：ArcGIS 的 units/unitLabel 在 API 上改不动（units 是只读单位对象），导出后换成中文
        unit_text = str(spec["scale_bar"].get("unit_label") or "")
        if unit_text and spec["scale_bar"].get("visible", True):
            from .image_overlay import stamp_unit_label

            sb_box = spec["scale_bar"]
            unit_result = stamp_unit_label(
                produced, unit_text,
                box_mm=(float(sb_box["x"]), float(sb_box["y"]),
                        float(sb_box.get("element_width_mm") or sb_box.get("bar_mm") or sb_box.get("w") or 40.0),
                        float(sb_box["h"])),
                page_height_mm=float(spec["page"]["height_mm"]),
                dpi=float(spec.get("dpi") or 250),
                font_name="宋体",
                size_pt=float(sb_box.get("font_pt") or 8.0),
            )
            notes.append("scale_bar(单位叠加): " + ("已换为「" + unit_text + "」" if unit_result.get("ok")
                                                   else f"失败（{unit_result.get('detail')}）"))

        # 叠加层箭头与年份标注（迁移图）：数据坐标 → 像素用**实测**地图框范围换算
        overlay_cfg = spec.get("overlay") or {}
        frame_extent = spec.get("render_frame_extent") or {}
        frame_box = spec.get("map_frame") or {}
        if overlay_cfg and frame_extent and frame_box:
            box_mm = (float(frame_box["x"]), float(frame_box["y"]), float(frame_box["w"]), float(frame_box["h"]))
            page_h = float(spec["page"]["height_mm"])
            dpi_value = float(spec.get("dpi") or 250)
            if overlay_cfg.get("arrows"):
                from .image_overlay import stamp_arrows

                arrow_result = stamp_arrows(
                    produced,
                    list(overlay_cfg["arrows"]),
                    extent=frame_extent,
                    box_mm=box_mm,
                    page_height_mm=page_h,
                    dpi=dpi_value,
                    color=str(overlay_cfg.get("arrow_color") or "#B2182B"),
                    label_size_pt=float(overlay_cfg.get("label_size_pt") or 9.0),
                )
                notes.append("overlay_arrows: " + ("已绘制 %s 条" % arrow_result.get("drawn")
                                                   if arrow_result.get("ok") else f"失败（{arrow_result.get('detail')}）"))
            if overlay_cfg.get("labels"):
                from .image_overlay import stamp_point_labels

                label_result = stamp_point_labels(
                    produced,
                    list(overlay_cfg["labels"]),
                    extent=frame_extent,
                    box_mm=box_mm,
                    page_height_mm=page_h,
                    dpi=dpi_value,
                    color=str(overlay_cfg.get("label_color") or "#333333"),
                    size_pt=float(overlay_cfg.get("label_size_pt") or 9.0),
                )
                notes.append("overlay_labels: " + ("已标注 %s 个" % label_result.get("drawn")
                                                   if label_result.get("ok") else f"失败（{label_result.get('detail')}）"))
        elif overlay_cfg:
            notes.append("overlay: 缺少实测地图框范围/图框，跳过箭头与标注（请在渲染后再叠加）")

        # 图例：按我们的格式重绘（ArcGIS 分级标签固定 6 位小数、且不能自定义文字）
        entries = (spec.get("renderer") or {}).get("legend_entries") or []
        legend_spec = spec.get("legend") or {}
        if entries and legend_spec.get("needed"):
            from .image_overlay import stamp_legend

            overlay_box = legend_spec.get("overlay_box") or [
                float(legend_spec["x"]), float(legend_spec["y"]),
                float(legend_spec["w"]), float(legend_spec["h"]),
            ]
            frame_box = spec.get("map_frame") or {}
            clamp_box = None
            if frame_box and not legend_spec.get("outside"):
                # 清除区不得越过图框线（用户反馈：图例叠加擦掉了图框）
                clamp_box = (float(frame_box["x"]) + 1.2, float(frame_box["y"]) + 1.2,
                             max(1.0, float(frame_box["w"]) - 2.4), max(1.0, float(frame_box["h"]) - 2.4))
            elif not legend_spec.get("outside"):
                neat = spec.get("neatline") or {}
                if neat:
                    clamp_box = (float(neat["x"]) + 1.2, float(neat["y"]) + 1.2,
                                 max(1.0, float(neat["w"]) - 2.4), max(1.0, float(neat["h"]) - 2.4))
            legend_result = stamp_legend(
                produced,
                entries,
                title=str(legend_spec.get("title") or ""),
                box_mm=(float(overlay_box[0]), float(overlay_box[1]),
                        float(overlay_box[2]), float(overlay_box[3])),
                clamp_mm=clamp_box,
                page_height_mm=float(spec["page"]["height_mm"]),
                dpi=float(spec.get("dpi") or 250),
                font_name=str((spec.get("title") or {}).get("font") or "宋体"),
                label_pt=float(legend_spec.get("font_pt") or 9.0),
                title_pt=float(legend_spec.get("title_pt") or 11.0),
            )
            notes.append("legend(叠加): " + (f"已按标签模式重绘 {len(entries)} 条" if legend_result.get("ok")
                                        else f"失败（{legend_result.get('detail')}）"))

            # 图例内容已确保不溢出（见 isOverflowing 处理），这里只需清图例框本身

        notes.append(
            "title(叠加): " + ("已按设计规格绘制，" + str(overlay.get("font")) + f" {overlay.get('size_px')}px"
                              if overlay.get("ok") else f"失败（{overlay.get('detail')}）")
            + "；ArcGIS 3.6 布局文本字号不可控，故在导出图叠加，工程内保留同名文本元素便于手工调整"
        )

    # 图廓线（页边细框）：由导出流程描边，避免依赖几何不可靠的 CIM 图形
    neat_box = spec.get("neatline") or {}
    if neat_box:
        from .image_overlay import stroke_rect

        stroke_rect(produced, box_mm=(float(neat_box["x"]), float(neat_box["y"]),
                                      float(neat_box["w"]), float(neat_box["h"])),
                    page_height_mm=float(spec["page"]["height_mm"]),
                    dpi=float(spec.get("dpi") or 250), width_mm=0.35)
        notes.append("neatline(叠加): 已按规格描边")

    # 重新打开一次做版面自检（主进程内 arcpy，安全）
    result = {
        "output": produced,
        "size": os.path.getsize(produced),
        "aprx": aprx_file,
        "aprx_size": aprx_size,
        "aprx_saved": True,
        "design_note": spec.get("design_note", ""),
        "title_overlay": overlay,
        "decisions": spec.get("decisions", []),
        "spec": spec,
        "notes": notes,
    }
    return result


__all__ = ["render", "_hex_to_rgb", "_parse_color_map", "_set"]
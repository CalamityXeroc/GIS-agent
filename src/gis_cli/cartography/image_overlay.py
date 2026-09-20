# -*- coding: utf-8 -*-
"""导出图叠加层：把"按设计规格应该呈现、但 ArcGIS 3.6 布局文本控件做不到"的元素补上。

为什么需要它（实测结论，别删注释）：
- ArcGIS Pro 3.6 的布局独立文本元素（TEXT_ELEMENT）**字号与对齐无法通过 API 控制**：
  CIM 里 `CIMTextSymbol.height` 被忽略（8/20/30pt 都渲染成默认约 8pt），
  `element.textSize` 只读（回读恒为 0），`CIMTextGraphic` 会直接让进程静默崩溃。
- 因此标题按设计规格（字号/居中/字体）在**导出后**用 Pillow 绘制到 JPG 上；
  布局里仍保留同名文本元素（默认字号），用户若要在 ArcGIS Pro 里手工调整也找得到。

依赖 Pillow；字体缺失时自动回退，不影响出图。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

#: 中文常见字体 → 字体文件候选（Windows）。
#: 注意：**不要**用 `simsunb.ttf`——那是 SimSun-ExtB（只有生僻扩展字），常见汉字会渲染成方框。
_FONT_CANDIDATES: dict[str, list[str]] = {
    "宋体": [r"C:\Windows\Fonts\simsun.ttc"],
    "黑体": [r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyhbd.ttc"],
    "微软雅黑": [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc"],
    "仿宋": [r"C:\Windows\Fonts\simfang.ttf", r"C:\Windows\Fonts\simsun.ttc"],
    "楷体": [r"C:\Windows\Fonts\simkai.ttf", r"C:\Windows\Fonts\simsun.ttc"],
}
_FALLBACK_FONTS = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
]


def resolve_font(font_name: str = "宋体", *, desired_bold: bool = True) -> str:
    """按字体名找可用字体文件（请求粗体时优先用自带粗体面的字体，否则用偏移绘制合成粗体）。"""
    candidates = list(_FONT_CANDIDATES.get(str(font_name).strip(), []))
    if desired_bold and str(font_name).strip() in ("微软雅黑", "黑体"):
        bold_faces = [p for p in candidates if "bd" in os.path.basename(p).lower()]
        candidates = bold_faces + [p for p in candidates if p not in bold_faces]
    for path in candidates + _FALLBACK_FONTS:
        if path and os.path.exists(path):
            return path
    return ""


def stamp_title(
    image_path: str,
    text: str,
    *,
    box_mm: tuple[float, float, float, float],
    page_height_mm: float,
    dpi: float,
    font_name: str = "宋体",
    size_pt: float = 20.0,
    color: str = "#000000",
    bold: bool = True,
    clear_box: bool = True,
) -> dict:
    """在导出图上按 mm 坐标绘制标题（居中）。

    ``box_mm`` 为 ``(x, y, w, h)``，其中 y 以页面**左下角**为原点（与 LayoutSpec 一致）。
    粗体在无粗体字体面时用**偏移重绘**合成，避免选到 SimSun-ExtB 这类只有生僻字的字体。
    返回 ``{"ok": bool, "font": 字体文件, "size_px": 像素字号, "detail": ...}``。
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as exc:  # pragma: no cover - 无 Pillow 环境
        return {"ok": False, "detail": f"Pillow 不可用: {exc}"}

    target = Path(image_path)
    if not target.exists():
        return {"ok": False, "detail": "图片不存在"}
    font_file = resolve_font(font_name)
    if not font_file:
        return {"ok": False, "detail": "找不到可用中文字体"}

    mm_to_px = float(dpi) / 25.4
    size_px = max(6, int(round(float(size_pt) * mm_to_px / (72.0 / 25.4))))
    try:
        font = ImageFont.truetype(font_file, size_px)
    except Exception as exc:
        return {"ok": False, "detail": f"字体加载失败: {exc}"}

    with Image.open(target) as img:
        canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    x, y, w, h = (float(v) for v in box_mm)
    left = x * mm_to_px
    top = (float(page_height_mm) - (y + h)) * mm_to_px
    right = (x + w) * mm_to_px
    bottom = (float(page_height_mm) - y) * mm_to_px
    if clear_box:
        draw.rectangle([left, top, right, bottom], fill="#FFFFFF")
    text = str(text or "").strip()
    if not text:
        canvas.save(target, quality=95)
        return {"ok": True, "font": font_file, "size_px": size_px, "detail": "空标题"}
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw_x = left + (right - left - text_w) / 2.0 - bbox[0]
    draw_y = top + (bottom - top - text_h) / 2.0 - bbox[1]
    draw.text((draw_x, draw_y), text, font=font, fill=color)
    if bold:
        # 合成粗体：偏移重绘（SimSun 无粗体面，用 ExtB 会变方框）
        draw.text((draw_x + 1.0, draw_y), text, font=font, fill=color)
        draw.text((draw_x + 0.5, draw_y + 0.5), text, font=font, fill=color)
    canvas.save(target, quality=95)
    return {"ok": True, "font": os.path.basename(font_file), "size_px": size_px,
            "text_width_mm": round(text_w / mm_to_px, 1), "box_w_mm": round(right - left, 1)}


def stamp_unit_label(
    image_path: str,
    text: str,
    *,
    box_mm: tuple[float, float, float, float],
    page_height_mm: float,
    dpi: float,
    font_name: str = "宋体",
    size_pt: float = 8.0,
    color: str = "#000000",
    reserve_mm: float = 22.0,
) -> dict:
    """把比例尺右端的单位标签换成中文（ArcGIS 的 units/unitLabel 在 API 上改不动）。

    ``box_mm`` 为比例尺元素框 ``(x, y, w, h)``（y 以页面左下角为原点）。
    只清掉右端 ``reserve_mm`` 宽的区域再画文字，避免碰到黑白条。
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "detail": f"Pillow 不可用: {exc}"}
    target = Path(image_path)
    if not target.exists():
        return {"ok": False, "detail": "图片不存在"}
    font_file = resolve_font(font_name)
    if not font_file:
        return {"ok": False, "detail": "找不到可用中文字体"}
    mm_to_px = float(dpi) / 25.4
    size_px = max(6, int(round(float(size_pt) * mm_to_px / (72.0 / 25.4))))
    font = ImageFont.truetype(font_file, size_px)
    x, y, w, h = (float(v) for v in box_mm)
    right = (x + w) * mm_to_px
    left = max(x * mm_to_px, right - reserve_mm * mm_to_px)
    top = (float(page_height_mm) - (y + h)) * mm_to_px
    bottom = (float(page_height_mm) - y) * mm_to_px
    with Image.open(target) as img:
        canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([left, top, right, bottom], fill="#FFFFFF")
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (left + 1.0, (top + bottom) / 2.0 - (bbox[3] - bbox[1]) / 2.0 - bbox[1]),
        text, font=font, fill=color,
    )
    canvas.save(target, quality=95)
    return {"ok": True, "font": os.path.basename(font_file), "text": text}


def stamp_legend(
    image_path: str,
    entries: list[dict[str, str]],
    *,
    title: str = "",
    box_mm: tuple[float, float, float, float],
    page_height_mm: float,
    dpi: float,
    font_name: str = "宋体",
    label_pt: float = 9.0,
    title_pt: float = 11.0,
    swatch_mm: tuple[float, float] = (7.0, 5.0),
    clear_margin_mm: tuple[float, float, float, float] = (4.0, 6.0, 10.0, 10.0),
    clamp_mm: tuple[float, float, float, float] | None = None,
) -> dict:
    """按设计规格重绘图例（无边框、标题+色块+标签）。

    为什么重绘：ArcGIS 分级图例的标签是自动格式化的（`0.000000 - 2500.000000` 这种），
    而且 classBreak.label 无法通过 API 自定义（详见 docs/cartography.md 的 API 踩坑清单）。
    这里用我们自己的标签文本（如「少」「0 – 2,500」）重绘，导出图才是干净版本。
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "detail": f"Pillow 不可用: {exc}"}
    target = Path(image_path)
    if not target.exists() or not entries:
        return {"ok": False, "detail": "图片不存在或无图例条目"}
    font_file = resolve_font(font_name)
    if not font_file:
        return {"ok": False, "detail": "找不到可用中文字体"}

    mm_to_px = float(dpi) / 25.4
    label_px = max(6, int(round(float(label_pt) * mm_to_px / (72.0 / 25.4))))
    title_px = max(7, int(round(float(title_pt) * mm_to_px / (72.0 / 25.4))))
    label_font = ImageFont.truetype(font_file, label_px)
    title_font = ImageFont.truetype(font_file, title_px)

    x, y, w, h = (float(v) for v in box_mm)
    left = x * mm_to_px
    right = (x + w) * mm_to_px
    top = (float(page_height_mm) - (y + h)) * mm_to_px
    bottom = (float(page_height_mm) - y) * mm_to_px

    with Image.open(target) as img:
        canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    # 清掉 ArcGIS 原图例：它的实际绘制范围会略大于回读框（自动排版），因此按边距扩大清除区
    mx0, my0, mx1, my1 = (float(v) for v in clear_margin_mm)
    clear_x0, clear_y0 = x - mx0, y - my1
    clear_x1, clear_y1 = x + w + mx1, y + h + my0
    if clamp_mm:
        cx, cy, cw, ch = (float(v) for v in clamp_mm)
        clear_x0 = max(clear_x0, cx)
        clear_y0 = max(clear_y0, cy)
        clear_x1 = min(clear_x1, cx + cw)
        clear_y1 = min(clear_y1, cy + ch)
    draw.rectangle([
        clear_x0 * mm_to_px,
        (float(page_height_mm) - clear_y1) * mm_to_px,
        clear_x1 * mm_to_px,
        (float(page_height_mm) - clear_y0) * mm_to_px,
    ], fill="#FFFFFF")

    pad = 1.2 * mm_to_px
    cursor_y = top + pad
    if title:
        tbox = draw.textbbox((0, 0), title, font=title_font)
        draw.text((left + pad, cursor_y - tbox[1]), title, font=title_font, fill="#000000")
        cursor_y += (tbox[3] - tbox[1]) + 1.6 * mm_to_px
    swatch_w, swatch_h = float(swatch_mm[0]) * mm_to_px, float(swatch_mm[1]) * mm_to_px
    row_h = max(swatch_h + 1.0 * mm_to_px, label_px * 1.45)
    drawn = 0
    for entry in entries:
        if cursor_y + row_h > bottom:
            break
        colour = str(entry.get("color") or "#CCCCCC")
        draw.rectangle([left + pad, cursor_y + (row_h - swatch_h) / 2,
                        left + pad + swatch_w, cursor_y + (row_h + swatch_h) / 2],
                       fill=colour, outline="#666666")
        text = str(entry.get("label") or "")
        tbox = draw.textbbox((0, 0), text, font=label_font)
        draw.text((left + pad + swatch_w + 1.5 * mm_to_px, cursor_y + (row_h - (tbox[3] - tbox[1])) / 2 - tbox[1]),
                  text, font=label_font, fill="#000000")
        cursor_y += row_h
        drawn += 1
    canvas.save(target, quality=95)
    return {"ok": True, "drawn": drawn, "font": os.path.basename(font_file)}


def _symbol_hex_placeholder() -> None:  # pragma: no cover - 占位，避免误删
    return None


def clear_rect(
    image_path: str,
    *,
    box_mm: tuple[float, float, float, float],
    page_height_mm: float,
    dpi: float,
) -> dict:
    """把指定矩形涂白（用于擦掉 ArcGIS 自动排版越出图框的图例残留）。"""
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "detail": f"Pillow 不可用: {exc}"}
    target = Path(image_path)
    if not target.exists():
        return {"ok": False, "detail": "图片不存在"}
    mm_to_px = float(dpi) / 25.4
    x, y, w, h = (float(v) for v in box_mm)
    if w <= 0 or h <= 0:
        return {"ok": False, "detail": "空矩形"}
    with Image.open(target) as img:
        canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([
        x * mm_to_px,
        (float(page_height_mm) - (y + h)) * mm_to_px,
        (x + w) * mm_to_px,
        (float(page_height_mm) - y) * mm_to_px,
    ], fill="#FFFFFF")
    canvas.save(target, quality=95)
    return {"ok": True, "cleared_mm": [x, y, w, h]}


def stroke_rect(
    image_path: str,
    *,
    box_mm: tuple[float, float, float, float],
    page_height_mm: float,
    dpi: float,
    width_mm: float = 0.35,
    color: str = "#000000",
) -> dict:
    """按 mm 框重描一条矩形边框（清理残留后修复图框线/图廓线用）。"""
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "detail": f"Pillow 不可用: {exc}"}
    target = Path(image_path)
    if not target.exists():
        return {"ok": False, "detail": "图片不存在"}
    mm_to_px = float(dpi) / 25.4
    x, y, w, h = (float(v) for v in box_mm)
    left, right = x * mm_to_px, (x + w) * mm_to_px
    bottom = (float(page_height_mm) - y) * mm_to_px
    top = (float(page_height_mm) - (y + h)) * mm_to_px
    line_px = max(1, int(round(float(width_mm) * mm_to_px)))
    with Image.open(target) as img:
        canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([left, top, right, bottom], outline=color, width=line_px)
    canvas.save(target, quality=95)
    return {"ok": True}


__all__ = ["stamp_title", "stamp_unit_label", "stamp_legend", "clear_rect", "stroke_rect", "resolve_font"]
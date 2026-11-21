# -*- coding: utf-8 -*-
"""针对性探针：搞清 ①标题文本怎么才能按点数渲染 ②图例标签如何生效 ③比例尺刻度/单位如何生效。

做法：在同一张布局上放多种写法，导出后用像素测量文字高度（客观判断，不靠肉眼）。
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import arcpy  # noqa: E402
from arcpy import mp, cim  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = Path(tempfile.gettempdir()) / "map_probe3"


def blank() -> str:
    install = arcpy.GetInstallInfo().get("InstallDir", "") or ""
    cand = os.path.join(install, "Resources", "ArcToolBox", "Services", "routingservices", "data", "Blank.aprx")
    if os.path.exists(cand):
        return cand
    import glob

    return glob.glob(os.path.join(install, "Resources", "**", "Blank.aprx"), recursive=True)[0]


def rect(x, y, w, h):
    return arcpy.Polygon(arcpy.Array([
        arcpy.Point(x, y), arcpy.Point(x + w, y), arcpy.Point(x + w, y + h), arcpy.Point(x, y + h), arcpy.Point(x, y),
    ]))


def add_text_variant(layout, box, text, height_pt, mode):
    """三种写法：paragraph+CIMTextSymbol / text+CIMTextSymbol / paragraph+先建后改。"""
    d = layout.getDefinition("V3")
    if mode == "paragraph":
        graphic = cim.CIMParagraphTextGraphic()
    else:
        graphic = cim.CIMTextGraphic()
    graphic.text = text
    symbol = cim.CIMTextSymbol()
    symbol.fontFamilyName = "宋体"
    symbol.height = height_pt
    symbol.bold = True
    symbol.horizontalAlignment = "Center"
    graphic.symbol = symbol
    graphic.shape = rect(*box)
    element = cim.CIMGraphicElement()
    element.name = f"{mode}_{height_pt}"
    element.graphic = graphic
    d.elements.append(element)
    layout.setDefinition(d)


def ink_height(image_path: str, box_mm, page_h_mm: float, dpi: float = 200) -> int:
    """量某矩形区域内的墨迹高度（像素）。"""
    from PIL import Image
    import numpy as np

    mm2px = dpi / 25.4
    x, y, w, h = box_mm
    with Image.open(image_path) as img:
        arr = np.asarray(img.convert("L"))
    top = int((page_h_mm - (y + h)) * mm2px)
    bottom = int((page_h_mm - y) * mm2px)
    left = int(x * mm2px)
    right = int((x + w) * mm2px)
    crop = arr[max(0, top):max(0, bottom), max(0, left):max(0, right)]
    dark = crop < 128
    rows = np.where(dark.any(axis=1))[0]
    return int(rows.max() - rows.min() + 1) if rows.size else 0


def main() -> int:
    layer_path = None
    for candidate in sorted(REPO.glob("benchmark/runs/*/pm_task2b_*/workspace/output/result_data.gdb")):
        if arcpy.Exists(str(candidate / "community_service")):
            layer_path = str(candidate / "community_service")
            break

    OUT.mkdir(parents=True, exist_ok=True)
    aprx_path = OUT / "probe3.aprx"
    if aprx_path.exists():
        aprx_path.unlink()
    shutil.copy2(blank(), aprx_path)
    aprx = mp.ArcGISProject(str(aprx_path))
    layout = aprx.createLayout(210, 297, "MILLIMETER")
    m = aprx.listMaps()[0]
    layer = m.addDataFromPath(layer_path)
    mf = layout.createMapFrame(rect(10, 60, 190, 200), m, "Main Map")
    mf.camera.setExtent(mf.getLayerExtent(layer))

    # ① 三种标题写法（同样 20pt）
    variants = [
        ("paragraph", 20, (5, 250, 200, 20)),
        ("text", 20, (5, 225, 200, 20)),
        ("paragraph", 10, (5, 200, 200, 20)),
    ]
    for mode, pt, box in variants:
        add_text_variant(layout, (box[0], box[1], box[2], box[3]), f"{mode} {pt}pt 测试标题", pt, mode)

    # ② 分级渲染 + 图例标签（元素 API 与 CIM 两种写法）
    sym = layer.symbology
    sym.updateRenderer("GraduatedColorsRenderer")
    sym.renderer.classificationField = "ELDER_COMM"
    sym.renderer.breakCount = 5
    layer.symbology = sym
    labels = ["少", "较少", "中", "较多", "多"]
    applied_el = []
    try:
        for brk, text in zip(layer.symbology.renderer.classBreaks, labels):
            brk.label = text
            applied_el.append((text, brk.label))
    except Exception as exc:
        print("元素 API 设置 classBreak.label 失败:", str(exc)[:80])
    cim_applied = []
    try:
        layer_def = layer.getDefinition("V3")
        breaks = getattr(getattr(layer_def, "renderer", None), "classBreaks", None) or []
        for brk, text in zip(breaks, labels):
            if hasattr(brk, "label"):
                brk.label = text
                cim_applied.append(text)
        layer.setDefinition(layer_def)
        print("CIM 写法设置标签数:", len(cim_applied))
    except Exception as exc:
        print("CIM 设置 classBreak.label 失败:", str(exc)[:100])
    print("元素 API 回读:", applied_el[:2])

    legend = layout.createMapSurroundElement(rect(120, 15, 70, 40), "LEGEND", mf,
                                             aprx.listStyleItems("ArcGIS 2D", "LEGEND")[0])
    legend.title = "老年人口"
    ldef = legend.getDefinition("V3")
    for item in getattr(ldef, "items", []) or []:
        item.showHeading = False
        item.showLayerName = False
    legend.setDefinition(ldef)

    # ③ 比例尺：元素 API + CIM 双管齐下
    sb = layout.createMapSurroundElement(rect(15, 15, 60, 20), "SCALE_BAR", mf,
                                        aprx.listStyleItems("ArcGIS 2D", "SCALE_BAR")[0])
    element_set = {}
    for key, value in (("divisions", 4), ("subdivisions", 1), ("units", "METERS"), ("unitLabel", "米")):
        try:
            setattr(sb, key, value)
            element_set[key] = getattr(sb, key)
        except Exception as exc:
            element_set[key] = f"fail:{str(exc)[:30]}"
    print("比例尺元素 API:", element_set)
    try:
        sdef = sb.getDefinition("V3")
        for key, value in (("division", 1000.0), ("divisions", 4), ("units", "METERS"), ("unitLabel", "米")):
            if hasattr(sdef, key):
                setattr(sdef, key, value)
        sb.setDefinition(sdef)
        sdef2 = sb.getDefinition("V3")
        print("比例尺 CIM 回读: division=", getattr(sdef2, "division", None),
              "divisions=", getattr(sdef2, "divisions", None),
              "units=", getattr(sdef2, "units", None),
              "unitLabel=", getattr(sdef2, "unitLabel", None))
    except Exception as exc:
        print("比例尺 CIM 设置失败:", str(exc)[:120])

    aprx.save()
    image = str(OUT / "probe3.jpg")
    layout.exportToJPEG(image, resolution=200)
    print("导出:", image)

    for mode, pt, box in variants:
        px = ink_height(image, box, 297.0, 200)
        mm = px * 25.4 / 200
        print(f"  标题写法 {mode} {pt}pt -> 墨迹高 {px}px ≈ {mm:.1f}mm（期望约 {pt * 25.4 / 72:.1f}mm）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
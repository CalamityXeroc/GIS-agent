# -*- coding: utf-8 -*-
"""检查生成的 .aprx：有几个布局、各元素的实际框与字号（定位"图名不生效"的真因）。"""
import sys, io, glob
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import arcpy  # noqa: E402
from arcpy import mp  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
aprx_path = sys.argv[1] if len(sys.argv) > 1 else str(
    sorted(REPO.glob("workspace/成果展示/制图设计示例/图1*.aprx"))[-1])
print("工程:", aprx_path)
project = mp.ArcGISProject(aprx_path)
layouts = project.listLayouts()
print(f"布局数: {len(layouts)}")
for index, layout in enumerate(layouts):
    page_w = layout.pageWidth
    page_h = layout.pageHeight
    print(f"\n--- 布局{index}: {layout.name} ({page_w:.0f}×{page_h:.0f} {layout.pageUnits}) ---")
    for element in layout.listElements():
        info = f"  {element.type:<22} {str(element.name)[:28]:<30}"
        try:
            info += f" x={element.elementPositionX:6.1f} y={element.elementPositionY:6.1f} w={element.elementWidth:6.1f} h={element.elementHeight:6.1f}"
        except Exception as exc:
            info += f" (位置读取失败 {str(exc)[:30]})"
        print(info)
        if element.type == "GRAPHIC_ELEMENT":
            try:
                definition = element.getDefinition("V3")
                graphic = getattr(definition, "graphic", None)
                symbol = getattr(graphic, "symbol", None)
                text = getattr(graphic, "text", "")
                height = getattr(symbol, "height", None)
                align = getattr(symbol, "horizontalAlignment", None)
                print(f"      文本={str(text)[:24]!r} 字号height={height} 对齐={align} 字体={getattr(symbol, 'fontFamilyName', None)}")
            except Exception as exc:
                print(f"      CIM 读取失败: {str(exc)[:60]}")
    try:
        for element in layout.listElements("MAPSURROUND_ELEMENT"):
            if element.type != "MAPSURROUND_ELEMENT":
                continue
            definition = element.getDefinition("V3")
            name = type(definition).__name__
            if "Scale" in name:
                print(f"  比例尺元素: divisions={getattr(definition, 'divisions', None)} "
                      f"subdivisions={getattr(definition, 'subdivisions', None)} "
                      f"units={getattr(definition, 'units', None)} unitLabel={getattr(definition, 'unitLabel', None)!r} "
                      f"division={getattr(definition, 'division', None)}")
    except Exception as exc:
        print("  比例尺读取失败:", str(exc)[:60])
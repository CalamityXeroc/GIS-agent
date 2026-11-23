# -*- coding: utf-8 -*-
"""探针：栅格图层的 CIM colorizer 到底长什么样，能否改成"分类渲染 + 显式分级"。

目的（统一图例的栅格路径）：4 期核密度是栅格，要让它们共用同一套分级，
必须把 colorizer 换成 CIMRasterClassifyColorizer 并写入 classBreaks.upperBound。
这里只探测 API 事实，不改任何工程。
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = Path(r"E:\BST_lab\GIS_CLI")
sys.path.insert(0, str(REPO / "src"))

import arcpy  # noqa: E402

APRX = REPO / "workspace/展示_统一图例冒烟/raster_probe.aprx"
print("工程存在:", APRX.exists())

project = arcpy.mp.ArcGISProject(str(APRX))
layer = None
for m in project.listMaps():
    for item in m.listLayers():
        print("  图层:", item.name, "| isRaster:", item.isRasterLayer, "| isFeature:", item.isFeatureLayer)
        if item.isRasterLayer:
            layer = item
            break
    if layer:
        break

if layer is None:
    raise SystemExit("没找到栅格图层")

definition = layer.getDefinition("V3")
colorizer = getattr(definition, "colorizer", None)
print("\ncolorizer 类型:", type(colorizer).__name__ if colorizer is not None else None)
if colorizer is not None:
    attrs = [a for a in dir(colorizer) if not a.startswith("_")]
    print("可用属性(部分):", [a for a in attrs if any(k in a.lower() for k in ("break", "ramp", "class", "stretch", "field", "type"))])
    breaks = getattr(colorizer, "classBreaks", None)
    print("classBreaks:", len(breaks) if breaks else breaks)
    if breaks:
        first = breaks[0]
        print("  一个 break 的属性:", [a for a in dir(first) if not a.startswith("_")][:14])

# 看看能不能造一个分类 colorizer
try:
    from arcpy.cim import cim as CIM  # type: ignore
    print("\nCIM 模块可用:", hasattr(CIM, "CIMRasterClassifyColorizer"))
    candidate = CIM.CIMRasterClassifyColorizer()
    print("新建 CIMRasterClassifyColorizer 属性:", [a for a in dir(candidate) if not a.startswith("_")][:16])
    brk = CIM.CIMRasterClassBreak()
    print("CIMRasterClassBreak 属性:", [a for a in dir(brk) if not a.startswith("_")][:12])
except Exception as exc:
    print("\nCIM 模块探测失败:", type(exc).__name__, str(exc)[:200])

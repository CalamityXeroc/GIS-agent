# -*- coding: utf-8 -*-
"""探针：矢量图层的显式分级为什么没生效（写 CIM 后读回是自动分级）。

对比：栅格路径是"整只替换 colorizer"（生效 ✓）；矢量路径是"改现有 classBreaks"（疑被重算）。
逐个尝试不同写入策略，看哪种能稳定落地。
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = Path(r"E:\BST_lab\GIS_CLI")
sys.path.insert(0, str(REPO / "src"))

import arcpy  # noqa: E402

arcpy.env.overwriteOutput = True
APRX = REPO / "workspace/展示_统一图例冒烟/矢量D_统一分级.aprx"
TARGET = [20.0, 40.0, 60.0, 100.0]

project = arcpy.mp.ArcGISProject(str(APRX))
layer = None
for m in project.listMaps():
    for item in m.listLayers():
        if item.isFeatureLayer:
            layer = item
            break
    if layer:
        break
print("图层:", layer.name)

renderer_el = layer.symbology.renderer
print("元素 API method:", getattr(renderer_el, "classificationMethod", None))
print("元素 API breakCount:", getattr(renderer_el, "breakCount", None))
print("元素 API 上界:", [float(b.upperBound) for b in renderer_el.classBreaks])

definition = layer.getDefinition("V3")
cim_breaks = definition.renderer.classBreaks
print("CIM method:", getattr(definition.renderer, "classificationMethod", None))
print("CIM 上界:", [float(b.upperBound) for b in cim_breaks])

print("\n--- 策略1：只改 upperBound + setDefinition ---")
try:
    for brk, upper in zip(cim_breaks, TARGET):
        brk.upperBound = float(upper)
    layer.setDefinition(definition)
    back = layer.getDefinition("V3").renderer.classBreaks
    print("读回:", [float(b.upperBound) for b in back])
except Exception as exc:
    print("失败:", type(exc).__name__, str(exc)[:120])

print("\n--- 策略2：先置 Manual 再写 upperBound ---")
try:
    definition = layer.getDefinition("V3")
    definition.renderer.classificationMethod = "Manual"
    for brk, upper in zip(definition.renderer.classBreaks, TARGET):
        brk.upperBound = float(upper)
    layer.setDefinition(definition)
    back = layer.getDefinition("V3").renderer.classBreaks
    print("读回:", [float(b.upperBound) for b in back])
    print("method:", getattr(layer.getDefinition("V3").renderer, "classificationMethod", None))
except Exception as exc:
    print("失败:", type(exc).__name__, str(exc)[:120])

print("\n--- 策略3：重建整个 renderer（CIMGraduatedColorsRenderer + classBreaks + colorRamp） ---")
try:
    from arcpy import cim  # type: ignore

    definition = layer.getDefinition("V3")
    old = definition.renderer
    new = cim.CIMGraduatedColorsRenderer()
    new.classificationField = getattr(old, "classificationField", "")
    new.classificationMethod = "Manual"
    new.breakCount = len(TARGET)
    new.colorRamp = getattr(old, "colorRamp", None)
    breaks = []
    for upper in TARGET:
        brk = cim.CIMClassBreak()
        brk.upperBound = float(upper)
        brk.label = str(upper)
        breaks.append(brk)
    new.classBreaks = breaks
    definition.renderer = new
    layer.setDefinition(definition)
    back = layer.getDefinition("V3").renderer.classBreaks
    print("读回:", [float(b.upperBound) for b in back])
    print("method:", getattr(layer.getDefinition("V3").renderer, "classificationMethod", None))
except Exception as exc:
    print("失败:", type(exc).__name__, str(exc)[:160])

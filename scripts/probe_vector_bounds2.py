# -*- coding: utf-8 -*-
"""探针 v2：矢量显式分级的可行写入路径。

已知：元素 API 的 renderer.method 是 StandardDeviation；CIM 侧是 CIMClassBreaksRenderer
且没有 classBreaks 属性。逐个试：① 元素 API 改 upperBound ② 元素 API 改 method+breaks
③ CIM 属性 dump（找 breaks 到底挂在哪）。
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
layer = next(item for m in project.listMaps() for item in m.listLayers() if item.isFeatureLayer)
print("图层:", layer.name)

definition = layer.getDefinition("V3")
renderer = definition.renderer
print("CIM renderer 类型:", type(renderer).__name__)
print("CIM renderer 属性:", [a for a in dir(renderer) if not a.startswith("_")][:30])

print("\n--- 策略A：元素 API 直接改 upperBound ---")
try:
    breaks = list(layer.symbology.renderer.classBreaks)
    for brk, upper in zip(breaks, TARGET):
        brk.upperBound = float(upper)
    layer.symbology = layer.symbology
    back = [float(b.upperBound) for b in layer.symbology.renderer.classBreaks]
    print("读回(元素API):", back)
    print("CIM 读回:", [float(b.upperBound) for b in layer.getDefinition("V3").renderer.classBreaks]
          if hasattr(layer.getDefinition("V3").renderer, "classBreaks") else "CIM 无 classBreaks")
except Exception as exc:
    print("失败:", type(exc).__name__, str(exc)[:140])

print("\n--- 策略B：元素 API 先置 Manual 再改 upperBound ---")
try:
    renderer_el = layer.symbology.renderer
    renderer_el.classificationMethod = "Manual"
    layer.symbology = layer.symbology
    breaks = list(layer.symbology.renderer.classBreaks)
    for brk, upper in zip(breaks, TARGET):
        brk.upperBound = float(upper)
    layer.symbology = layer.symbology
    print("method:", getattr(layer.symbology.renderer, "classificationMethod", None))
    print("读回:", [float(b.upperBound) for b in layer.symbology.renderer.classBreaks])
except Exception as exc:
    print("失败:", type(exc).__name__, str(exc)[:140])

print("\n--- 策略C：元素 API method=Manual + breakCount 重设后再写 ---")
try:
    renderer_el = layer.symbology.renderer
    renderer_el.classificationMethod = "Manual"
    renderer_el.breakCount = len(TARGET)
    layer.symbology = layer.symbology
    breaks = list(layer.symbology.renderer.classBreaks)
    print("重设后条数:", len(breaks), "上界:", [float(b.upperBound) for b in breaks])
    for brk, upper in zip(breaks, TARGET):
        brk.upperBound = float(upper)
    layer.symbology = layer.symbology
    print("读回:", [float(b.upperBound) for b in layer.symbology.renderer.classBreaks])
except Exception as exc:
    print("失败:", type(exc).__name__, str(exc)[:140])

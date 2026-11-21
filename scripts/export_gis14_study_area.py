# -*- coding: utf-8 -*-
"""把任务一的 study_area 导出为独立 shapefile，作为后续任务的显式输入。

背景：基准任务原先靠 carry_from 承接上一步的工作区（含 study_area）。
一旦运行目录被清理（或换机器重跑），任务就"从零开始"、缺研究区而卡住——
基准任务必须自洽：把它需要的前序成果做成显式输入。

产出：workspace/input/第14届真题数据/study_area.shp（UTM 10N，单个矩形面）
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import arcpy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(r"E:\BST_lab\GIS_CLI")
SRC = REPO / "workspace/成果展示/第14届真题实测/任务一_清洗投影与研究区/result_data.gdb/study_area"
DST_DIR = REPO / "workspace/input/第14届真题数据"
DST_DIR.mkdir(parents=True, exist_ok=True)

print("源:", SRC, "存在:", arcpy.Exists(str(SRC)))
out = str(DST_DIR / "study_area.shp")
arcpy.env.overwriteOutput = True
arcpy.management.CopyFeatures(str(SRC), out)
desc = arcpy.Describe(out)
ext = desc.extent
print("已导出:", out)
print(f"  CRS={desc.spatialReference.factoryCode} 要素数={arcpy.management.GetCount(out)[0]}")
print(f"  范围: ({ext.XMin:.1f}, {ext.YMin:.1f}) ~ ({ext.XMax:.1f}, {ext.YMax:.1f})")

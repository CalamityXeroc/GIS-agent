# -*- coding: utf-8 -*-
"""盘点第 14 届四个任务交付库（temp_data.gdb / result_data.gdb）的内容。

考卷要求：temp_data.gdb 存过程数据、result_data.gdb 存结果数据，
提交时应删除非必要的过程数据。这里检查是否有残留。
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import arcpy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(r"E:\BST_lab\GIS_CLI\workspace\成果展示\第14届真题实测")
TASKS = [
    "任务一_清洗投影与研究区",
    "任务二a_气象插值与精度检验",
    "任务二b_DEM拼接与土地覆盖",
    "任务三a_云区修补与分类面积",
]


def inventory(gdb: Path) -> None:
    if not arcpy.Exists(str(gdb)):
        print("   （不存在）")
        return
    arcpy.env.workspace = str(gdb)
    for label, items in (
        ("要素类", arcpy.ListFeatureClasses()),
        ("栅格", arcpy.ListRasters()),
        ("表", arcpy.ListTables()),
        ("要素数据集", arcpy.ListDatasets()),
    ):
        if items:
            print(f"   {label}: {items}")
    for ds in arcpy.ListDatasets() or []:
        try:
            print(f"   [数据集 {ds}] 要素类: {arcpy.ListFeatureClasses(feature_dataset=ds)}")
        except Exception:
            pass


for task in TASKS:
    print("=" * 74)
    print(task)
    for name in ("temp_data.gdb", "result_data.gdb"):
        gdb = BASE / task / name
        print(f" -- {name}")
        inventory(gdb)

# 输出目录里的非成果文件（考卷要求交付时清理过程文件）
print("=" * 74)
print("输出目录里的过程文件（非 GDB/非报告）")
for task in TASKS:
    d = BASE / task
    for item in sorted(d.iterdir()):
        if item.suffix in {".pkl", ".json", ".txt"}:
            print(f"   {task}/{item.name}  ({item.stat().st_size} 字节)")

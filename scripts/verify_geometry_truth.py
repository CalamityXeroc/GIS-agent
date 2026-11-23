# -*- coding: utf-8 -*-
"""Verify whether the agent ACTUALLY transformed geometry (not just relabeled CRS)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import arcpy

base = (sys.argv[1] if len(sys.argv) > 1
        else r"E:\BST_lab\GIS_CLI\benchmark\runs\20260909_095936\am_task1_quality\workspace")

def show(label, path):
    try:
        desc = arcpy.Describe(path)
        sr = getattr(desc, "spatialReference", None)
        ext = desc.extent
        n = int(arcpy.management.GetCount(path)[0])
        print(f"{label:<44} n={n:<5} wkid={getattr(sr,'factoryCode',None)} "
              f"X[{ext.XMin:,.1f}, {ext.XMax:,.1f}] Y[{ext.YMin:,.1f}, {ext.YMax:,.1f}]")
    except Exception as e:
        print(f"{label:<44} ERR {e}")

print("=== 原始输入（坐标应该是原始单位）===")
show("input/county.shp", base + r"\input\county.shp")
show("input/city.shp",   base + r"\input\city.shp")

print()
print("=== agent 产出 result_data.gdb ===")
show("result_data.gdb/county",   base + r"\output\result_data.gdb\county")
show("result_data.gdb/city",     base + r"\output\result_data.gdb\city")
show("result_data.gdb/county_1", base + r"\output\result_data.gdb\county_1")

print()
print("=== agent 产出 temp_data.gdb（过程库）===")
for dirpath, dirnames, filenames in arcpy.da.Walk(base + r"\output\temp_data.gdb", datatype=["FeatureClass", "Table"]):
    for fn in filenames:
        show("temp_data.gdb/" + fn, dirpath + "\\" + fn)

print()
print("=== 参考答案（正确投影后的量级）===")
show(r"AM参考 result_data.gdb\city", r"E:\BST_lab\GIS_CLI\workspace\input\13第十三届全国大学生GIS应用技能大赛（兰州交通大学）\AM\AM\1-Process\result_data.gdb\city")

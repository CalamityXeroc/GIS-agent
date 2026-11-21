# -*- coding: utf-8 -*-
"""Inspect the agent's benchmark workspace output (result_data.gdb)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import arcpy

gdb = r"E:\BST_lab\GIS_CLI\benchmark\runs\20260909_095936\am_task1_quality\workspace\output\result_data.gdb"
print("=== result_data.gdb ===")
for dirpath, dirnames, filenames in arcpy.da.Walk(gdb, datatype=["FeatureClass", "Table"]):
    for fn in filenames:
        p = dirpath + "\\" + fn
        try:
            desc = arcpy.Describe(p)
            sr = getattr(desc, "spatialReference", None)
            n = int(arcpy.management.GetCount(p)[0])
            stype = getattr(desc, "shapeType", "") or "Table"
            print(f"{fn:<28} {stype:<10} n={n:<5} crs={getattr(sr, 'name', '')} wkid={getattr(sr, 'factoryCode', '')}")
            if fn == "county_1":
                fields = [f.name for f in arcpy.ListFields(p)]
                print("   county_1 fields:", fields)
        except Exception as e:
            print(fn, "ERR", e)

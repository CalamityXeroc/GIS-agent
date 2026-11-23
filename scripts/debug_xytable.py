# -*- coding: utf-8 -*-
"""Debug XYTableToPoint on the ASCII-path CSV copy."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import arcpy

arcpy.env.overwriteOutput = True
base = r"C:\Users\cheng\AppData\Local\Temp\bench2b_offline"
csv = os.path.join(base, "input", "population.CSV")
gdb = os.path.join(base, "output", "result_data.gdb")
print("csv exists:", os.path.exists(csv))
print("csv fields:", [f.name for f in arcpy.ListFields(csv)][:4])
print("gdb exists:", arcpy.Exists(gdb))
if not arcpy.Exists(gdb):
    arcpy.management.CreateFileGDB(os.path.join(base, "output"), "result_data")
out = os.path.join(gdb, "pts_test")
try:
    arcpy.management.XYTableToPoint(csv, out, "CenterLon", "CenterLat", arcpy.SpatialReference(4326))
    print("OK count=", int(arcpy.management.GetCount(out)[0]))
except Exception as e:
    print("ERR:", str(e)[:500])

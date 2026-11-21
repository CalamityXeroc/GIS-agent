# -*- coding: utf-8 -*-
"""Test MakeXYEventLayer -> CopyFeatures in a clean temp GDB."""
import sys, io, os, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import arcpy

base = r"C:\Users\cheng\AppData\Local\Temp\xytest_clean"
if os.path.exists(base):
    shutil.rmtree(base, ignore_errors=True)
os.makedirs(os.path.join(base, "output"))
csv = r"C:\Users\cheng\AppData\Local\Temp\bench2b_offline\input\population.CSV"
gdb = os.path.join(base, "output", "result_data.gdb")
arcpy.env.overwriteOutput = True

arcpy.management.CreateFileGDB(os.path.join(base, "output"), "result_data")
print("gdb created:", arcpy.Exists(gdb))

arcpy.management.MakeXYEventLayer(csv, "CenterLon", "CenterLat", "xy_lyr", arcpy.SpatialReference(4326))
print("event layer count:", int(arcpy.management.GetCount("xy_lyr")[0]))

out = os.path.join(gdb, "points_tmp")
try:
    arcpy.management.CopyFeatures("xy_lyr", out)
    print("CopyFeatures OK:", int(arcpy.management.GetCount(out)[0]))
except Exception as e:
    print("CopyFeatures ERR:", str(e)[:300])

# 再试另一种：直接对事件图层做 Project
try:
    out2 = os.path.join(gdb, "points_prj")
    arcpy.management.Project("xy_lyr", out2, arcpy.SpatialReference(4508))
    print("Project(layer) OK:", int(arcpy.management.GetCount(out2)[0]))
except Exception as e:
    print("Project(layer) ERR:", str(e)[:300])

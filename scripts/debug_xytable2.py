# -*- coding: utf-8 -*-
"""Test XYTableToPoint workaround: import CSV to GDB table first."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import arcpy

arcpy.env.overwriteOutput = True
base = r"C:\Users\cheng\AppData\Local\Temp\bench2b_offline"
csv = os.path.join(base, "input", "population.CSV")
gdb = os.path.join(base, "output", "result_data.gdb")

# 1) CSV -> GDB table
tbl = os.path.join(gdb, "population_tbl")
try:
    arcpy.conversion.TableToTable(csv, gdb, "population_tbl")
    print("TableToTable OK, fields:", [f.name for f in arcpy.ListFields(tbl)][:4])
except Exception as e:
    print("TableToTable ERR:", str(e)[:300])

# 2) GDB table -> points
out = os.path.join(gdb, "pts_from_tbl")
try:
    arcpy.management.XYTableToPoint(tbl, out, "CenterLon", "CenterLat", arcpy.SpatialReference(4326))
    print("XYTableToPoint(GDB table) OK count=", int(arcpy.management.GetCount(out)[0]))
except Exception as e:
    print("XYTableToPoint(GDB table) ERR:", str(e)[:300])

# 3) MakeXYEventLayer on raw CSV
try:
    arcpy.management.MakeXYEventLayer(csv, "CenterLon", "CenterLat", "xylyr", arcpy.SpatialReference(4326))
    print("MakeXYEventLayer OK count=", int(arcpy.management.GetCount("xylyr")[0]))
except Exception as e:
    print("MakeXYEventLayer ERR:", str(e)[:300])

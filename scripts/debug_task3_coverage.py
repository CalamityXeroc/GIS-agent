# -*- coding: utf-8 -*-
import sys, io, os, glob, tempfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import arcpy
arcpy.env.overwriteOutput = True

ws = sorted(glob.glob(r"C:\Users\cheng\AppData\Local\Temp\bench3a_*"))[-1]
gdb = os.path.join(ws, "output", "result_data.gdb")
city = os.path.join(gdb, "city_4508")
county_pop = os.path.join(gdb, "county_pop")
print("ws:", ws)

tmp_dir = tempfile.mkdtemp(prefix="t3cov_")
arcpy.management.CreateFileGDB(tmp_dir, "c.gdb")
cg = os.path.join(tmp_dir, "c.gdb")

z = os.path.join(cg, "z")
arcpy.management.CopyFeatures(city, z)
arcpy.management.AddField(z, "ZONE_KEY", "TEXT", field_length=255)
arcpy.management.CalculateField(z, "ZONE_KEY", "!name!", "PYTHON3")
sj = os.path.join(cg, "sj")
arcpy.analysis.SpatialJoin(county_pop, z, sj, "JOIN_ONE_TO_ONE", "KEEP_ALL", match_option="INTERSECT")
n_null = 0
zone_counts = {}
with arcpy.da.SearchCursor(sj, ["ZONE_KEY"]) as cur:
    for r in cur:
        if r[0] is None:
            n_null += 1
        else:
            zone_counts[r[0]] = zone_counts.get(r[0], 0) + 1
print("区县总数:", int(arcpy.management.GetCount(sj)[0]), "未落入任何市的区县:", n_null)
print("有区县的市数:", len(zone_counts))
all_cities = set()
with arcpy.da.SearchCursor(city, ["name"]) as cur:
    for r in cur:
        all_cities.add(r[0])
print("没有区县落入的市:", sorted(all_cities - set(zone_counts)))

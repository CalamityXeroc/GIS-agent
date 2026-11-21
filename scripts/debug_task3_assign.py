# -*- coding: utf-8 -*-
import sys, io, os, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import arcpy

ws = sorted(glob.glob(r"C:\Users\cheng\AppData\Local\Temp\bench3a_*"))[-1]
gdb = os.path.join(ws, "output", "result_data.gdb")
county_pop = os.path.join(gdb, "county_pop")
city = os.path.join(gdb, "city_4508")

# 1) 直接几何判断: 娄底市面 是否包含 4313xx 区县
loudi = None
with arcpy.da.SearchCursor(city, ["SHAPE@", "name"]) as cur:
    for g, n in cur:
        if n == "娄底市":
            loudi = g
counties = []
with arcpy.da.SearchCursor(county_pop, ["SHAPE@", "code", "name"]) as cur:
    for g, code, name in cur:
        if str(code).startswith("4313"):
            counties.append((name, code, g))
print("娄底市区县数:", len(counties))
for name, code, g in counties:
    print("  ", name, code, "within=", g.within(loudi), "intersects=", g.overlaps(loudi) or g.touches(loudi) or g.contains(loudi))

# 2) 空间连接(带 ZONE_KEY)对照
tmp = os.path.join(os.environ.get("TEMP", "."), "t3assign")
os.makedirs(tmp, exist_ok=True)
cg = os.path.join(tmp, "c.gdb")
if not arcpy.Exists(cg):
    arcpy.management.CreateFileGDB(tmp, "c.gdb")
z = os.path.join(cg, "z")
arcpy.management.CopyFeatures(city, z)
arcpy.management.AddField(z, "ZONE_KEY", "TEXT", field_length=255)
arcpy.management.CalculateField(z, "ZONE_KEY", "!name!", "PYTHON3")
sj = os.path.join(cg, "sj3")
arcpy.analysis.SpatialJoin(county_pop, z, sj, "JOIN_ONE_TO_ONE", "KEEP_ALL", match_option="INTERSECT")
print("\n空间连接中 4313xx 的归属:")
with arcpy.da.SearchCursor(sj, ["name", "code", "ZONE_KEY", "Join_Count"]) as cur:
    for name, code, zk, jc in cur:
        if str(code).startswith("4313"):
            print("  ", name, code, "->", zk, "join_count=", jc)

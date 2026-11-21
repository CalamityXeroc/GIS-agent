# -*- coding: utf-8 -*-
"""Debug: self spatial join on v6 county_pop."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import arcpy

arcpy.env.overwriteOutput = True
cp = sys.argv[1]
print("input:", cp)
print("count:", int(arcpy.management.GetCount(cp)[0]))

sj = r"in_memory\_dbg_sj"
if arcpy.Exists(sj):
    arcpy.management.Delete(sj)
try:
    arcpy.analysis.SpatialJoin(cp, cp, sj, "JOIN_ONE_TO_MANY", "KEEP_COMMON", match_option="INTERSECT")
    print("sj count:", int(arcpy.management.GetCount(sj)[0]))
    print("sj fields:", [f.name for f in arcpy.ListFields(sj)][:8])
    cols = [c for c in ["TARGET_FID", "JOIN_FID", "TARGET_FID_1", "Join_Count_1"] if c in [f.name for f in arcpy.ListFields(sj)]]
    print("reading cols:", cols)
    with arcpy.da.SearchCursor(sj, cols) as cur:
        rows = [r for r in cur]
    print("rows:", len(rows), "first:", rows[:6])
    pairs = [r for r in rows if len(r) >= 3 and r[0] != r[2]]
    print("non-self pairs:", len(pairs))
except Exception as e:
    print("ERR:", str(e)[:400])

# 直接几何邻接测试
rows = [(o, s) for o, s in arcpy.da.SearchCursor(cp, ["OID@", "SHAPE@"])]
touching = 0
for i in range(min(len(rows), 60)):
    for j in range(i + 1, min(len(rows), 60)):
        if rows[i][1].touches(rows[j][1]) or rows[i][1].overlaps(rows[j][1]):
            touching += 1
print("geometry touching/overlaps among first 60:", touching)

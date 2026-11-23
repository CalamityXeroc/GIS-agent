# -*- coding: utf-8 -*-
import sys, io, os, glob, tempfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import arcpy
arcpy.env.overwriteOutput = True

runs = sorted(glob.glob(r"E:\BST_lab\GIS_CLI\benchmark\runs\2026*"))
county_pop = os.path.join(runs[-1], "am_task2b_neighbor_highlight", "workspace", "output", "result_data.gdb", "county_pop")
base = glob.glob(r"E:\BST_lab\GIS_CLI\workspace\input\**\B上午\数据", recursive=True)[0]

tmp_dir = tempfile.mkdtemp(prefix="t3sum_")
arcpy.management.CreateFileGDB(tmp_dir, "p.gdb")
gdb = os.path.join(tmp_dir, "p.gdb")
city = os.path.join(gdb, "city")
arcpy.management.Project(os.path.join(base, "city.shp"), city, arcpy.SpatialReference(4508))

city_pop = os.path.join(gdb, "city_pop")
arcpy.analysis.SpatialJoin(
    target_features=city, join_features=county_pop, out_feature_class=city_pop,
    join_operation="JOIN_ONE_TO_ONE", join_type="KEEP_ALL", match_option="INTERSECT",
    field_mapping="Y2011 SUM;Y2012 SUM;Y2020 SUM",
)
print("fields:", [f.name for f in arcpy.ListFields(city_pop) if f.type not in ("Geometry", "OID")])
vals = []
with arcpy.da.SearchCursor(city_pop, ["Y2011", "Y2012", "Y2020", "Join_Count"]) as cur:
    for r in cur:
        vals.append(r)
print("n rows:", len(vals))
print("first 5:", vals[:5])
v2011 = [v[0] for v in vals if v[0] is not None]
print("Y2011: n=%d min=%s max=%s distinct=%d" % (len(v2011), min(v2011) if v2011 else None, max(v2011) if v2011 else None, len(set(v2011))))

# 对照: 用 summarize_stats 求和 county_pop 比较
out_tbl = os.path.join(gdb, "sum_by_name")
try:
    arcpy.analysis.Statistics(county_pop, out_tbl, [["Y2011", "SUM"]], "name")
    tvals = {}
    with arcpy.da.SearchCursor(out_tbl, ["name", "SUM_Y2011"]) as cur:
        for r in cur:
            tvals[r[0]] = r[1]
    print("stats table rows:", len(tvals), "sample:", list(tvals.items())[:3])
except Exception as e:
    print("stats failed:", str(e)[:200])

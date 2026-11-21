# -*- coding: utf-8 -*-
import sys, io, os, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import arcpy

ws = sorted(glob.glob(r"C:\Users\cheng\AppData\Local\Temp\bench3a_*"))[-1]
gdb = os.path.join(ws, "output", "result_data.gdb")
county_pop = os.path.join(gdb, "county_pop")
city = os.path.join(gdb, "city_4508")

from collections import Counter
pref = Counter()
with arcpy.da.SearchCursor(county_pop, ["code", "name"]) as cur:
    for code, name in cur:
        pref[str(code)[:4]] += 1
print("区县 code 前4位（地市）分布:", sorted(pref.items())[:60])
print("地市数(按code):", len(pref))

# 空间连接得到的市 -> 区县数
tmp_dir = os.path.join(os.environ.get("TEMP", "."), "t3codes")
os.makedirs(tmp_dir, exist_ok=True)
cg = os.path.join(tmp_dir, "c.gdb")
if not arcpy.Exists(cg):
    arcpy.management.CreateFileGDB(tmp_dir, "c.gdb")
z = os.path.join(cg, "z")
arcpy.management.CopyFeatures(city, z)
arcpy.management.AddField(z, "ZONE_KEY", "TEXT", field_length=255)
arcpy.management.CalculateField(z, "ZONE_KEY", "!name!", "PYTHON3")
sj = os.path.join(cg, "sj2")
arcpy.analysis.SpatialJoin(county_pop, z, sj, "JOIN_ONE_TO_ONE", "KEEP_ALL", match_option="INTERSECT")
cnt = Counter()
with arcpy.da.SearchCursor(sj, ["ZONE_KEY"]) as cur:
    for r in cur:
        cnt[r[0]] += 1
print("空间连接: 市数 =", len(cnt), "；高计数市(>15):", [(k, v) for k, v in cnt.most_common(6)])
# 检查缺失市的 code（娄底4313 益阳4309 随州4213 鹤壁4106 济源4190 仙桃4290 天门4290 潜江4290 神农架4290）
for code in ("4313", "4309", "4213", "4106", "4190", "4290"):
    print(" code", code, "->", pref.get(code, 0), "个区县")

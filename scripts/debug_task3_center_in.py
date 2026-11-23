# -*- coding: utf-8 -*-
import sys, io, os, glob, tempfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import arcpy
from collections import Counter

ws = sorted(glob.glob(r"C:\Users\cheng\AppData\Local\Temp\bench3a_*"))[-1]
gdb = os.path.join(ws, "output", "result_data.gdb")
county_pop = os.path.join(gdb, "county_pop")
city = os.path.join(gdb, "city_4508")

tmp = tempfile.mkdtemp(prefix="t3ci_")
arcpy.management.CreateFileGDB(tmp, "c.gdb")
cg = os.path.join(tmp, "c.gdb")
z = os.path.join(cg, "z")
arcpy.management.CopyFeatures(city, z)
arcpy.management.AddField(z, "ZONE_KEY", "TEXT", field_length=255)
arcpy.management.CalculateField(z, "ZONE_KEY", "!name!", "PYTHON3")

for opt in ("HAVE_THEIR_CENTER_IN", "WITHIN"):
    sj = os.path.join(cg, "sj_" + opt)
    try:
        arcpy.analysis.SpatialJoin(county_pop, z, sj, "JOIN_ONE_TO_ONE", "KEEP_ALL", match_option=opt)
        cnt = Counter()
        n_null = 0
        with arcpy.da.SearchCursor(sj, ["ZONE_KEY"]) as cur:
            for r in cur:
                if r[0] is None:
                    n_null += 1
                else:
                    cnt[r[0]] += 1
        print(f"[{opt}] 匹配市数={len(cnt)} 未匹配区县={n_null} 最大市区县数={cnt.most_common(3)}")
    except Exception as e:
        print(f"[{opt}] 失败: {str(e)[:150]}")

# 郑州/娄底 校验
sj = os.path.join(cg, "sj_HAVE_THEIR_CENTER_IN")
mapping = {}
with arcpy.da.SearchCursor(sj, ["name", "code", "ZONE_KEY"]) as cur:
    for name, code, zk in cur:
        if str(code).startswith(("4101", "4313")):
            mapping.setdefault(str(code)[:4], Counter())[zk] += 1
for pref, label in (("4101", "郑州市"), ("4313", "娄底市")):
    print(f" {label}({pref}) 归属分布:", dict(mapping.get(pref, {})))

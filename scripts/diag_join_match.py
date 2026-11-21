# -*- coding: utf-8 -*-
"""诊断：缓冲区统计表与社区图层的 join 匹配情况。"""
import sys, io, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")
from pathlib import Path
import arcpy
arcpy.env.overwriteOutput = True

WS = Path(sorted(glob.glob(r"C:\Users\cheng\AppData\Local\Temp\pm2b_*"))[-1])
GDB = str(WS / "output" / "result_data.gdb")
tbl = GDB + "\\buffer_elder_tbl"
svc = GDB + "\\community_service"
comm = GDB + "\\community_elder"

print("tbl 行数:", arcpy.management.GetCount(tbl)[0])
print("tbl 字段:", [(f.name, f.type) for f in arcpy.ListFields(tbl) if f.type not in ("Geometry",)])
print("community_service SUM_VALUE 空值:", end=" ")
n = 0
with arcpy.da.SearchCursor(svc, ["SUM_VALUE"]) as cur:
    for (v,) in cur:
        n += 1 if v is None else 0
print(n)

# BSM 值域对比
def stats(path, field, label):
    vals = []
    with arcpy.da.SearchCursor(path, [field]) as cur:
        for (v,) in cur:
            if v is not None:
                vals.append(float(v))
    print(f"  {label}: n={len(vals)} min={min(vals)} max={max(vals)} 整数率={sum(1 for v in vals if abs(v-round(v))<1e-9)/len(vals):.3f}")

stats(tbl, "BSM", "buffer_elder_tbl.BSM")
stats(comm, "BSM", "community_elder.BSM")

# 实际匹配数
tbl_keys = set()
with arcpy.da.SearchCursor(tbl, ["BSM"]) as cur:
    for (v,) in cur:
        if v is not None:
            tbl_keys.add(int(round(float(v))))
comm_keys = []
with arcpy.da.SearchCursor(comm, ["BSM"]) as cur:
    for (v,) in cur:
        comm_keys.append(None if v is None else int(round(float(v))))
matched = sum(1 for k in comm_keys if k in tbl_keys)
print(f"键匹配: 表 {len(tbl_keys)} 个键；社区 {len(comm_keys)} 条；能匹配 {matched}")
print(f"社区层 SUM_VALUE 非空 {len(comm_keys)-n} vs 匹配 {matched}")
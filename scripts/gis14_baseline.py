# -*- coding: utf-8 -*-
"""为第 14 届上午题任务一计算参考基线（只读输入，产出写到 temp 目录）。

产出：
- 清洗后点数的合理区间（去重/去空值/处理半球符号错误后的整群）
- 研究区矩形（清洗点包络 + 2 km）在 UTM 10N 下的范围与尺寸
- 各类脏数据的条目数（供 prompt 说明与断言选取）

用 ArcGIS Pro 的 python 跑（需要 arcpy 做投影）。
"""
from __future__ import annotations

import io
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import arcpy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SRC = Path(
    r"E:\GIS_documents\小火的竞赛资料\14第十四届全国大学生GIS应用技能大赛"
    r"\14第十四届全国大学生GIS应用技能大赛\A上午\A上午\data"
)
OUT = Path(tempfile.gettempdir()) / "gis14_baseline"
OUT.mkdir(parents=True, exist_ok=True)

DMS = re.compile(r"^\s*(\d+)[^\d]*(\d+)['\u2032]?[^\d]*([\d.]+)?[^\dA-Za-z]*([NSEW])")


def parse_dms(text: str):
    """解析 41°2'15.00″N / 124°47'8.20″W → (值, 半球)；失败返回 (None, None)。"""
    m = DMS.match(str(text).strip().strip('"'))
    if not m:
        return None, None
    deg, minute, sec, hemi = m.group(1), m.group(2), m.group(3) or 0.0, m.group(4).upper()
    val = float(deg) + float(minute) / 60 + float(sec) / 3600
    return (-val if hemi in ("S", "W") else val), hemi

def read_rows(path: Path):
    return path.read_text(encoding="gbk").splitlines()


def _explode(values: list[str], n: int) -> list[str]:
    """把一行里的逗号配对值拆成 n 条（长度不足时复用最后一个）。"""
    if n <= 1:
        return [values[0]]
    out = []
    for i in range(n):
        out.append(values[i] if i < len(values) else values[-1])
    return out


# ---------- 1. 动物点位清洗 ----------
lines = read_rows(SRC / "动物点位" / "animal.txt")
rows = [ln for ln in lines[1:] if ln.strip()]
stats = {"raw": len(rows)}
uniq = list(dict.fromkeys(rows))            # 去整行重复
stats["dropped_exact_dup"] = len(rows) - len(uniq)

kept, dropped_null, dropped_sign, dropped_bad = [], 0, 0, 0
exploded_rows = 0
species_seen = Counter()
for row in uniq:
    parts = row.split("\t")
    if len(parts) < 5:
        dropped_bad += 1
        continue
    cells = [cell.strip().strip('"') for cell in parts[:5]]
    year = cells[0]
    counts = cells[1].split(",")
    names = cells[2].split(",")
    lat_cells = cells[3].split(",")
    lon_cells = cells[4].split(",")
    n = max(len(counts), len(names), len(lat_cells), len(lon_cells))
    if n > 1:
        exploded_rows += 1
    for i in range(n):
        count = _explode(counts, n)[i].strip()
        name = _explode(names, n)[i].strip()
        lat_raw = _explode(lat_cells, n)[i]
        lon_raw = _explode(lon_cells, n)[i]
        lat, lat_h = parse_dms(lat_raw)
        lon, lon_h = parse_dms(lon_raw)
        if not year.strip():
            stats_year_empty = stats.get("dropped_empty_year", 0) + 1
            stats["dropped_empty_year"] = stats_year_empty
            continue
        if lat is None or lon is None or not str(count).strip():
            dropped_null += 1
            continue
        try:
            count_val = float(count)
        except ValueError:
            dropped_bad += 1
            continue
        # 主体群是 41°N / 124°W；南半球（~85°W）整块为错误区域记录 → 剔除
        if lat_h == "S":
            dropped_sign += 1
            continue
        # 經度写成 E 的记录，量级与主体群一致（124.8°），属半球符号笔误 → 修正为 W
        if lon_h == "E":
            lon = -abs(lon)
        kept.append((int(year), count_val, name, lat, lon))
        species_seen[name] += 1

stats["multi_record_rows"] = exploded_rows
stats["kept_points"] = len(kept)
stats["dropped_null"] = dropped_null
stats["dropped_sign_error"] = dropped_sign
stats["dropped_malformed"] = dropped_bad
stats["species_rows"] = dict(species_seen)
print("=== 动物点位清洗统计 ===")
for k, v in stats.items():
    print(f"  {k}: {v}")

# ---------- 2. 投影到 UTM 10N 求包络 + 2 km ----------
wgs84 = arcpy.SpatialReference(4326)
utm10 = arcpy.SpatialReference(32610)
pts = [arcpy.PointGeometry(arcpy.Point(lon, lat), wgs84).projectAs(utm10) for _, _, _, lat, lon in kept]
xs = [g.firstPoint.X for g in pts]
ys = [g.firstPoint.Y for g in pts]
print("\n=== 清洗点 UTM10N 范围 ===")
print(f"  X: {min(xs):.1f} ~ {max(xs):.1f}  (跨度 {(max(xs)-min(xs))/1000:.2f} km)")
print(f"  Y: {min(ys):.1f} ~ {max(ys):.1f}  (跨度 {(max(ys)-min(ys))/1000:.2f} km)")

BUF = 2000.0
x0, x1 = min(xs) - BUF, max(xs) + BUF
y0, y1 = min(ys) - BUF, max(ys) + BUF
print(f"  研究区(外扩2km): X {x0:.1f}~{x1:.1f} Y {y0:.1f}~{y1:.1f} "
      f"尺寸 {(x1-x0)/1000:.2f} x {(y1-y0)/1000:.2f} km 面积 {(x1-x0)*(y1-y0)/1e6:.1f} km²")

# 反算到经纬度，便于提示词核对
envelope_wgs = arcpy.Polygon(
    arcpy.Array([arcpy.Point(x0, y0), arcpy.Point(x1, y0), arcpy.Point(x1, y1), arcpy.Point(x0, y1)]),
    utm10,
).projectAs(wgs84)
ext = envelope_wgs.extent
print(f"  经纬度范围: {ext.XMin:.4f}~{ext.XMax:.4f} ℃?, {ext.YMin:.4f}~{ext.YMax:.4f}")

# ---------- 3. 气象站点清洗 ----------
mlines = read_rows(SRC / "气象数据" / "Meteorological_sampling.txt")
mrows = [ln for ln in mlines[1:] if ln.strip()]
muniq = list(dict.fromkeys(mrows))
mstats = {"raw": len(mrows), "dropped_exact_dup": len(mrows) - len(muniq)}
temps, pres, bad_temp, bad_coord, sign_err = [], [], 0, 0, 0
for row in muniq:
    parts = row.split("\t")
    if len(parts) < 3:
        continue
    try:
        pre, tmp = float(parts[0]), float(parts[1])
    except Exception:
        continue
    lon, lat = parts[2].strip().strip('"').split(",")
    lon_v, lon_h = parse_dms(lon)
    lat_v, lat_h = parse_dms(lat)
    if lon_v is None or lat_v is None:
        bad_coord += 1
        continue
    if lat_h != "N" or lon_h != "W":
        sign_err += 1
        continue
    if not (200 <= tmp <= 330):      # 合理的开尔文范围
        bad_temp += 1
        continue
    temps.append(tmp)
    pres.append(pre)
mstats.update(
    kept=len(temps), dropped_out_of_range_temp=bad_temp, dropped_bad_coord=bad_coord, dropped_sign=sign_err,
    temp_k=(min(temps), max(temps)), temp_c=(min(temps) - 273.15, max(temps) - 273.15),
    pre=(min(pres), max(pres)), pre_mean=sum(pres) / len(pres),
)
print("\n=== 气象站点清洗统计 ===")
for k, v in mstats.items():
    print(f"  {k}: {v}")

(OUT / "baseline.json").write_text(
    json.dumps({"animal": stats, "meteo": mstats,
                "study_area_utm10": {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                                     "size_km": [(x1 - x0) / 1000, (y1 - y0) / 1000]}},
               ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"\n基线已写入: {OUT / 'baseline.json'}")

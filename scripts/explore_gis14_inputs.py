# -*- coding: utf-8 -*-
"""勘查第 14 届上午题输入数据（只读，不依赖 arcpy）。

目的：为基准任务的断言确定"可验证的基线"——
清洗后应剩多少点、坐标范围、重复/空值/离群各多少、气象站温度分布等。
"""
from __future__ import annotations

import io
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA = Path(
    r"E:\GIS_documents\小火的竞赛资料\14第十四届全国大学生GIS应用技能大赛"
    r"\14第十四届全国大学生GIS应用技能大赛\A上午\A上午\data"
)


def sniff(path: Path) -> None:
    raw = path.read_bytes()[:400]
    print(f"--- {path.name} 头 240 字节 ---")
    print(raw[:240])
    for enc in ("utf-8", "gbk", "utf-8-sig", "latin-1"):
        try:
            txt = raw.decode(enc)
            print(f"  [{enc}] 可解码: {txt[:70]!r}")
        except Exception as exc:
            print(f"  [{enc}] 失败: {exc}")


def parse_dms(text: str) -> float | None:
    """把 41°2'15.00"N 这类（含乱码分隔符）解析为十进制度。"""
    text = str(text).strip().strip('"')
    m = re.match(r"^\s*(\d+)[^\d]*(\d+)['\u2032]?[^\d]*([\d.]+)?[^\dA-Za-z]*([NSEW])", text)
    if not m:
        return None
    deg, minute, sec, hemi = m.group(1), m.group(2), m.group(3) or 0.0, m.group(4).upper()
    val = float(deg) + float(minute) / 60 + float(sec) / 3600
    return -val if hemi in ("S", "W") else val


animal = DATA / "动物点位" / "animal.txt"
meteo = DATA / "气象数据" / "Meteorological_sampling.txt"
for p in (animal, meteo):
    print("=" * 78)
    print(p)
    sniff(p)

print("=" * 78)
print("### animal.txt 结构分析")
lines = None
for enc in ("utf-8", "gbk", "latin-1"):
    try:
        lines = animal.read_text(encoding=enc, errors="strict").splitlines()
        print(f"用 {enc} 读取成功，{len(lines)} 行")
        break
    except Exception as exc:
        print(f"{enc} 失败: {exc}")
assert lines
header, rows = lines[0], [ln for ln in lines[1:] if ln.strip()]
print("表头:", header)
print("数据行:", len(rows))
cols = [r.split("\t") for r in rows]
print("字段数分布:", Counter(len(c) for c in cols))
name_col = Counter(c[2] for c in cols if len(c) >= 5)
print("动物种类:", dict(name_col))
year_col = Counter(c[0] for c in cols if len(c) >= 5)
print("年份范围:", min(year_col), "-", max(year_col), "共", len(year_col), "年")
dup_all = len(rows) - len(set(rows))
dup_key = len(rows) - len({(c[0], c[2], c[3], c[4]) for c in cols if len(c) >= 5})
print("整行重复:", dup_all, "| (年+种+纬度+经度) 重复:", dup_key)
empty = Counter()
bad_parse = []
values_w, values_e, values_s, values_n = 0, 0, 0, 0
lons, lats = [], []
for c in cols:
    if len(c) < 5:
        empty["字段数不足"] += 1
        continue
    for idx, label in ((1, "Count"), (3, "Latitude"), (4, "Longitude")):
        if not str(c[idx]).strip():
            empty[label] += 1
    lat = parse_dms(c[3])
    lon = parse_dms(c[4])
    if lat is None or lon is None:
        bad_parse.append(tuple(c))
        continue
    if "S" in c[3].upper():
        values_s += 1
    else:
        values_n += 1
    if "W" in c[4].upper():
        values_w += 1
    else:
        values_e += 1
    lats.append(lat)
    lons.append(lon)
print("空值统计:", dict(empty))
print("解析失败:", len(bad_parse))
for row in bad_parse[:5]:
    print("   ", row)
print(f"半球: N={values_n} S={values_s} | E={values_e} W={values_w}")
if lons:
    print(f"经度范围: {min(lons):.4f} ~ {max(lons):.4f}")
    print(f"纬度范围: {min(lats):.4f} ~ {max(lats):.4f}")
    in_range = sum(1 for lo, la in zip(lons, lats) if 110 <= lo <= 135 and 20 <= la <= 50)
    print(f"落在东亚合理区间的点数: {in_range} / {len(lons)}")

print("=" * 78)
print("### Meteorological_sampling.txt 结构分析")
mlines = None
for enc in ("utf-8", "gbk", "latin-1"):
    try:
        mlines = meteo.read_text(encoding=enc, errors="strict").splitlines()
        print(f"用 {enc} 读取成功，{len(mlines)} 行")
        break
    except Exception as exc:
        print(f"{enc} 失败: {exc}")
assert mlines
print("表头:", mlines[0])
mrows = [ln for ln in mlines[1:] if ln.strip()]
print("数据行:", len(mrows))
mcols = [r.split("\t") for r in mrows]
print("字段数分布:", Counter(len(c) for c in mcols))
print("整行重复:", len(mrows) - len(set(mrows)))
temp_vals, pre_vals = [], []
bad = []
for c in mcols:
    if len(c) < 3:
        bad.append(c)
        continue
    try:
        pre_vals.append(float(c[0]))
        temp_vals.append(float(c[1]))
    except Exception:
        bad.append(c)
print("数值解析失败:", len(bad))
for row in bad[:5]:
    print("   ", row)
if temp_vals:
    print(f"降雨(m): {min(pre_vals):.2f} ~ {max(pre_vals):.2f} 均值 {sum(pre_vals)/len(pre_vals):.2f}")
    print(f"温度(K): {min(temp_vals):.2f} ~ {max(temp_vals):.2f} 均值 {sum(temp_vals)/len(temp_vals):.2f}")
    print(f"温度(℃): {min(temp_vals)-273.15:.2f} ~ {max(temp_vals)-273.15:.2f}")
mlat, mlon = [], []
for c in mcols:
    if len(c) < 3:
        continue
    parts = c[2].split(",")
    if len(parts) != 2:
        continue
    lon, lat = parse_dms(parts[0]), parse_dms(parts[1])
    if lon is not None and lat is not None:
        mlon.append(lon)
        mlat.append(lat)
print(f"气象站解析出坐标数: {len(mlon)}")
if mlon:
    print(f"经度 {min(mlon):.4f} ~ {max(mlon):.4f} | 纬度 {min(mlat):.4f} ~ {max(mlat):.4f}")

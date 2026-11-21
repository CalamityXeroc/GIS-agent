# -*- coding: utf-8 -*-
"""预览第 14 届（2025）上午题实测产出（四联图，用 PIL 直接渲染）。

面板：DEM 裁切 / 土地覆盖（云已修补）/ 年均温度插值 / 野生动物点位（按物种着色）
数据来源：benchmark/runs 下四个任务的 workspace/output（agent 实际产出）。
"""
from __future__ import annotations

import io
import sys
from collections import defaultdict
from pathlib import Path

import arcpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(r"E:\BST_lab\GIS_CLI")
RUNS = {
    "task1": REPO / "benchmark/runs/20260922_152144/gis14_am_task1_animal_points/workspace/output/result_data.gdb",
    "task2a": REPO / "benchmark/runs/20260922_192854/gis14_am_task2a_meteo_interp/workspace/output/result_data.gdb",
    "task2b": REPO / "benchmark/runs/20260922_195702/gis14_am_task2b_dem_landcover/workspace/output/result_data.gdb",
    "task3a": REPO / "benchmark/runs/20260922_204234/gis14_am_task3a_landcover_cloud_fix/workspace/output/result_data.gdb",
}
OUT_DIR = REPO / "workspace/成果展示/第14届真题实测"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL = 900          # 每个面板的像素边长
FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 34)
FONT_S = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)


def to_image(arr: np.ndarray, rgb: np.ndarray, nodata_mask: np.ndarray) -> Image.Image:
    """把 (H,W) 数值数组经调色板映射成 RGB 图像。"""
    img = np.zeros((*arr.shape, 3), dtype="uint8")
    img[nodata_mask] = (255, 255, 255)
    ok = ~nodata_mask
    img[ok] = rgb[ok]
    return Image.fromarray(img)


def norm01(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = a.astype("float64")
    lo, hi = np.nanmin(v[mask]), np.nanmax(v[mask])
    out = np.zeros_like(v)
    out[mask] = (v[mask] - lo) / (hi - lo if hi > lo else 1.0)
    return out


def dem_colors(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    t = norm01(a, mask)
    # 地形色带：低处深绿 → 中棕 → 高处白
    r = np.clip(60 + t * 195, 0, 255)
    g = np.clip(110 + t * 120, 0, 255)
    b = np.clip(60 + t * 195, 0, 255)
    return np.stack([r, g, b], axis=-1).astype("uint8")


LC_COLORS = {
    1: (30, 100, 220), 2: (20, 130, 40), 4: (60, 160, 150), 5: (220, 200, 60),
    7: (200, 60, 60), 8: (190, 170, 140), 9: (240, 240, 250), 10: (255, 0, 255), 11: (170, 200, 90),
}


def lc_colors(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img = np.zeros((*a.shape, 3), dtype="uint8")
    for code, rgb in LC_COLORS.items():
        img[a == code] = rgb
    return img


def temp_colors(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    t = norm01(a, mask)
    r = np.clip(255 * (0.2 + 0.8 * t), 0, 255)
    g = np.clip(255 * (0.9 - 0.7 * t), 0, 255)
    b = np.clip(255 * (0.9 - 1.1 * t), 0, 255)
    return np.stack([r, g, b], axis=-1).astype("uint8")


def read_raster(path: Path):
    """读取栅格为数组 + 有效像元掩膜（整型栅格不支持 NaN 作 nodata，需退到哨兵值）。"""
    desc = arcpy.Describe(str(path))
    try:
        arr = arcpy.RasterToNumPyArray(str(path), nodata_to_value=np.nan).astype("float64")
        mask = ~np.isnan(arr)
    except ValueError:
        arr = arcpy.RasterToNumPyArray(str(path), nodata_to_value=-9999).astype("float64")
        mask = arr != -9999
    return arr, mask, desc


def fit(img: Image.Image, size: int) -> Image.Image:
    scale = size / max(img.size)
    return img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.NEAREST)


def main() -> None:
    panels: list[tuple[str, Image.Image, str]] = []

    # 1. DEM
    arr, mask, desc = read_raster(RUNS["task2b"] / "dem_clip")
    lo, hi = float(np.min(arr[mask])), float(np.max(arr[mask]))
    panels.append(("DEM 裁切（拼接+投影+裁切，30 m）", fit(to_image(arr, dem_colors(arr, mask), ~mask), PANEL),
                   f"高程 {lo:.0f} ~ {hi:.0f} m"))

    # 2. 土地覆盖（云修补后）
    arr2, mask2, _ = read_raster(RUNS["task3a"] / "landcover_fixed")
    panels.append(("土地覆盖（云 261 像元已修补）", fit(to_image(arr2, lc_colors(arr2, mask2), ~mask2), PANEL),
                   "林地/牧场/裸地为主，10 m"))

    # 3. 温度插值
    arr3, mask3, _ = read_raster(RUNS["task2a"] / "tmp_raster")
    panels.append(("年均温度插值（IDW，500 m，中误差 0.33℃）", fit(to_image(arr3, temp_colors(arr3, mask3), ~mask3), PANEL),
                   f"{np.nanmin(arr3):.1f} ~ {np.nanmax(arr3):.1f} ℃"))

    # 4. 动物点位（按物种着色）
    fc = str(RUNS["task1"] / "animal_points")
    d = arcpy.Describe(fc)
    ext = d.extent
    species_colors = {
        "Giraffe": (230, 120, 30), "Gazelle": (200, 40, 130), "Impala": (30, 90, 220),
        "Bison": (140, 80, 40), "Zebra": (40, 160, 140),
    }
    base = Image.new("RGB", (PANEL, PANEL), (255, 255, 255))
    draw = ImageDraw.Draw(base)
    w, h = ext.width, ext.height
    scale = min(PANEL / w, PANEL / h)
    off_x = (PANEL - w * scale) / 2
    off_y = (PANEL - h * scale) / 2
    counts: dict[str, int] = defaultdict(int)
    with arcpy.da.SearchCursor(fc, ["NAME", "SHAPE@X", "SHAPE@Y"]) as cur:
        for name, x, y in cur:
            counts[str(name)] += 1
            px = off_x + (x - ext.XMin) * scale
            py = off_y + (ext.YMax - y) * scale
            rgb = species_colors.get(str(name), (120, 120, 120))
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=rgb)
    legend = "  ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    panels.append(("野生动物点位（清洗后 3464 个，按物种）", base, legend))

    # 合成 2x2
    pad, title_h, note_h = 18, 52, 40
    total_w = pad + (PANEL + pad) * 2
    total_h = pad + (PANEL + title_h + note_h + pad) * 2 + 70
    canvas = Image.new("RGB", (total_w, total_h), (250, 250, 248))
    dc = ImageDraw.Draw(canvas)
    dc.text((pad, 14), "第14届（2025）全国大学生GIS应用技能大赛 · 上午题 · Agent 实测产出",
            font=FONT, fill=(20, 20, 20))
    for idx, (title, img, note) in enumerate(panels):
        cx = pad + (idx % 2) * (PANEL + pad)
        cy = 70 + (idx // 2) * (PANEL + title_h + note_h + pad)
        dc.text((cx, cy + 8), title, font=FONT, fill=(25, 25, 25))
        canvas.paste(img, (cx, cy + title_h))
        dc.rectangle([cx, cy + title_h, cx + img.width, cy + title_h + img.height], outline=(120, 120, 120))
        dc.text((cx, cy + title_h + img.height + 8), note[:64], font=FONT_S, fill=(70, 70, 70))

    out = OUT_DIR / "第14届上午题_实测产出预览.png"
    canvas.save(out)
    print("预览已保存:", out)
    print("尺寸:", canvas.size)


if __name__ == "__main__":
    main()
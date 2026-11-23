# -*- coding: utf-8 -*-
"""冒烟验证：迁移图的箭头与年份标注是否真的画在正确位置。

做法（可判定，不靠肉眼）：
1. 用研究区范围的合成路径渲染一张带箭头的图（箭头 + 年份点标注，两者不重复）；
2. 按 map_to_px 算出箭头端点的期望像素，采样该处颜色，应命中箭头色；
3. 对照采样图框中心，应不是箭头色；
4. 核对底图栅格确实渲染出来了（图像非空白）；
5. 让识图模型看一眼，确认"能看见箭头与年份"。
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = Path(r"E:\BST_lab\GIS_CLI")
sys.path.insert(0, str(REPO / "src"))

import arcpy  # noqa: E402
from PIL import Image  # noqa: E402

from gis_cli.cartography import create_map  # noqa: E402
from gis_cli.cartography.image_overlay import map_to_px  # noqa: E402

arcpy.env.overwriteOutput = True
GDB = REPO / "workspace/成果展示/第14届真题实测/任务二b_DEM拼接与土地覆盖/result_data.gdb"
DEM = str(GDB / "dem_clip")
OUT = REPO / "workspace/展示_迁移图冒烟"
OUT.mkdir(parents=True, exist_ok=True)

# 研究区范围（UTM 10N）：合成一条自西南向东北再折向东南的迁移路径
PATH_POINTS = [
    (331000, 4542000),
    (339000, 4560000),
    (347000, 4576000),
    (353000, 4564000),
    (356000, 4546000),
]
YEARS = ["1982", "1987", "1992", "1997", "2001"]
ARROW_COLOR = "#B2182B"

# 年份统一由点标注给出（箭头标签会与点标注重复，实际使用时二选一）
arrows = [
    {"from": list(PATH_POINTS[i]), "to": list(PATH_POINTS[i + 1])}
    for i in range(len(PATH_POINTS) - 1)
]
labels = [{"at": list(point), "text": YEARS[i]} for i, point in enumerate(PATH_POINTS)]

image = OUT / "迁移路径_冒烟.jpg"
result = create_map(
    [DEM],
    {
        "theme": "分布中心迁移",
        "title": "野生动物分布中心逐五年迁移路径（冒烟）",
        "legend_labels": "range",
        "mode": "graduated",
        "field": "Value",
        "class_bounds": "1200,1500,1800,2100,2500,3000",
        "overlay": {"arrows": arrows, "labels": labels, "arrow_color": ARROW_COLOR},
    },
    output_path=str(image),
)
spec = result.get("spec") or {}
print("叠加笔记:")
for note in (result.get("notes") or []):
    if str(note).startswith("overlay"):
        print("   ", str(note)[:110])

frame_extent = spec.get("render_frame_extent") or {}
frame_box = spec.get("map_frame") or {}
print("实测地图框范围:", {k: round(v, 1) for k, v in frame_extent.items()} if frame_extent else "缺失")
if not frame_extent or not frame_box:
    raise SystemExit("缺少实测地图框范围，无法做像素校验")

box_mm = (float(frame_box["x"]), float(frame_box["y"]), float(frame_box["w"]), float(frame_box["h"]))
probe_points = [tuple(arrows[0]["from"]), tuple(arrows[0]["to"]), tuple(arrows[2]["to"])]
probe_px = map_to_px(
    probe_points, frame_extent, box_mm,
    page_height_mm=float(spec["page"]["height_mm"]), dpi=float(spec["dpi"]),
)

with Image.open(image) as img:
    rgb = img.convert("RGB")
    gray_stats = img.convert("L")


def sample(px: tuple[float, float], radius: int = 4) -> tuple[int, int, int]:
    """取一小片区域内最"红"的像素（箭头是红色，抗锯齿会淡化边上像素）。"""
    best = (255, 255, 255)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x, y = int(px[0]) + dx, int(px[1]) + dy
            if 0 <= x < rgb.width and 0 <= y < rgb.height:
                pixel = rgb.getpixel((x, y))
                if pixel[0] - max(pixel[1], pixel[2]) > best[0] - max(best[1], best[2]):
                    best = pixel
    return best


def is_arrow_color(pixel: tuple[int, int, int]) -> bool:
    return pixel[0] > 120 and pixel[0] - max(pixel[1], pixel[2]) > 30


print("")
print("像素校验（箭头应为 #B2182B 红）:")
hits = 0
for name, px in zip(("起点", "第一段终点", "第三段终点"), probe_px):
    pixel = sample(px)
    ok = is_arrow_color(pixel)
    hits += 1 if ok else 0
    print("   %s 期望像素=(%.0f,%.0f) 采样=%s 命中=%s" % (name, px[0], px[1], pixel, ok))

center_px = (
    (box_mm[0] + box_mm[2] / 2) * float(spec["dpi"]) / 25.4,
    (float(spec["page"]["height_mm"]) - (box_mm[1] + box_mm[3] / 2)) * float(spec["dpi"]) / 25.4,
)
center_pixel = sample(center_px, radius=2)
print("   对照（图框中心）采样=%s 命中箭头色=%s" % (center_pixel, is_arrow_color(center_pixel)))
print("命中 %d/3 个箭头采样点" % hits)

# 底图栅格是否渲染出来（空白图 std 接近 0）
import numpy as np  # noqa: E402

arr = np.asarray(gray_stats, dtype="float32")
print("底图校验: 整图 std=%.1f（空白会接近 0）" % float(arr.std()))

# ---- 识图质检：模型能不能看到箭头与年份 ----
try:
    from gis_cli.engine.llm_client import EngineLLMClient, EngineLLMConfig  # noqa: E402
    from gis_cli.engine.verifier import Verifier  # noqa: E402

    cfg = EngineLLMConfig.from_file(REPO / "config/llm_config.json")
    if cfg.vision:
        verifier = Verifier(workspace=REPO, llm_client=EngineLLMClient(cfg), vision_mode="warn")
        verdict = verifier.vision_check_image(image)
        print("")
        print("识图质检:", "通过" if verdict.get("ok") else "有问题")
        print("  结论:", str(verdict.get("summary"))[:160])
        if verdict.get("issues"):
            print("  问题:", "；".join(verdict["issues"])[:220])
    else:
        print("")
        print("（配置未开识图，跳过识图质检）")
except Exception as exc:
    print("")
    print("识图质检失败:", type(exc).__name__, str(exc)[:120])

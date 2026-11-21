# -*- coding: utf-8 -*-
"""冒烟验证：显式分级（统一图例）在真实渲染里是否生效（栅格 + 矢量两条路径）。

栅格（4 期核密度那种场景）：
- 图A / 图B：同一个 DEM，传同一组分级边界 → 读回 aprx 的 classBreaks 必须完全一致
- 图C：不传边界（自动分级）→ 分级应与 A/B 不同
矢量（行政区/统计数据那种场景）：
- 图D：用 animal_points 的 COUNT 字段 + 显式边界 → 读回上界应与给定一致

最后用 shared_legend_bounds 断言比对 spec 文件（A/B 一致应通过；掺入 C 应失败）。
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = Path(r"E:\BST_lab\GIS_CLI")
sys.path.insert(0, str(REPO / "src"))

import arcpy  # noqa: E402

from gis_cli.cartography import create_map  # noqa: E402
from gis_cli.recipes.store import RecipeLibrary  # noqa: E402

arcpy.env.overwriteOutput = True
GDB = REPO / "workspace/成果展示/第14届真题实测/任务二b_DEM拼接与土地覆盖/result_data.gdb"
DEM = str(GDB / "dem_clip")
POINTS = str(REPO / "workspace/成果展示/第14届真题实测/任务一_清洗投影与研究区/result_data.gdb/animal_points")
OUT = REPO / "workspace/展示_统一图例冒烟"
OUT.mkdir(parents=True, exist_ok=True)

R_BOUNDS = [1200, 1500, 1800, 2100, 2500, 3000]   # DEM 实测 1201~3019 m
V_BOUNDS = [10, 20, 40, 60, 100]                    # COUNT 字段


def read_bounds(aprx_path: Path) -> list[float]:
    """读回工程里第一个栅格/矢量图层实际生效的分级上界（实测值）。"""
    project = arcpy.mp.ArcGISProject(str(aprx_path))
    for m in project.listMaps():
        for layer in m.listLayers():
            values: list[float] = []
            # 矢量：CIM renderer.breaks（实测挂在这里）；栅格：CIM colorizer.classBreaks
            try:
                definition = layer.getDefinition("V3")
                renderer = getattr(definition, "renderer", None)
                breaks = list(getattr(renderer, "breaks", None) or getattr(renderer, "classBreaks", None) or [])
                values = [float(getattr(brk, "upperBound", 0.0)) for brk in breaks]
            except Exception:
                values = []
            if not values:
                try:
                    colorizer = layer.getDefinition("V3").colorizer
                    values = [float(brk.upperBound) for brk in (getattr(colorizer, "classBreaks", None) or [])]
                except Exception:
                    values = []
            if values:
                return values
    return []


def render(tag: str, bounds, *, layer_path: str = DEM, field: str = "Value") -> tuple[Path, Path, dict]:
    intent = {
        "theme": "高程" if layer_path == DEM else "观测数量",
        "title": f"分级图（{tag}）",
        "legend_labels": "range",
        "mode": "graduated",
        "field": field,
    }
    if bounds:
        intent["class_bounds"] = ",".join(str(b) for b in bounds)
    image = OUT / f"{tag}.jpg"
    result = create_map([layer_path], intent, output_path=str(image))
    aprx = Path(str(result.get("aprx") or image.with_suffix(".aprx")))
    spec_path = OUT / f"{tag}.spec.json"
    spec_path.write_text(json.dumps(result.get("spec") or {}, ensure_ascii=False, indent=2), encoding="utf-8")
    return image, aprx, (result.get("spec") or {})


print(f"栅格: {DEM}")
print(f"矢量: {POINTS}")
print(f"栅格分级边界: {R_BOUNDS}\n")

img_a, aprx_a, spec_a = render("栅格A_统一分级", R_BOUNDS)
img_b, aprx_b, spec_b = render("栅格B_统一分级", R_BOUNDS)
img_c, aprx_c, spec_c = render("栅格C_自动分级", [])
img_d, aprx_d, spec_d = render("矢量D_统一分级", V_BOUNDS, layer_path=POINTS, field="COUNT")

bounds_a, bounds_b, bounds_c, bounds_d = (
    read_bounds(aprx_a), read_bounds(aprx_b), read_bounds(aprx_c), read_bounds(aprx_d)
)
print("读回 栅格A:", bounds_a)
print("读回 栅格B:", bounds_b)
print("读回 栅格C(自动):", bounds_c)
print("读回 矢量D:", bounds_d)

expect_raster = [float(x) for x in R_BOUNDS[1:]]
expect_vector = [float(x) for x in V_BOUNDS[1:]]
print()
print(f"[1] 栅格 A/B 分级完全一致: {bool(bounds_a) and bounds_a == bounds_b}")
print(f"[2] 栅格 A 分级 == 给定边界(去首): {bounds_a == expect_raster}  ({bounds_a})")
print(f"[3] 栅格 C 自动分级与 A 不同: {bounds_c != bounds_a}  ({bounds_c})")
print(f"[4] 矢量 D 分级 == 给定边界(去首): {bounds_d == expect_vector}  ({bounds_d})")

lib = RecipeLibrary()
ab = OUT / "specs_raster_AB"
ab.mkdir(exist_ok=True)
for tag, spec in (("A", spec_a), ("B", spec_b)):
    (ab / f"{tag}.spec.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
res_ab = lib.run_assertions(
    [{"type": "shared_legend_bounds", "path": str(ab), "value": {"min_files": 2}}]
)[0]
print(f"\n[5] shared_legend_bounds(栅格A,B): {res_ab.ok} | {res_ab.detail}")

abc = OUT / "specs_raster_ABC"
abc.mkdir(exist_ok=True)
for tag, spec in (("A", spec_a), ("B", spec_b), ("C", spec_c)):
    (abc / f"{tag}.spec.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
res_abc = lib.run_assertions(
    [{"type": "shared_legend_bounds", "path": str(abc), "value": {"min_files": 3}}]
)[0]
print(f"[6] shared_legend_bounds(A,B,C) 应失败: {not res_abc.ok} | {res_abc.detail[:110]}")

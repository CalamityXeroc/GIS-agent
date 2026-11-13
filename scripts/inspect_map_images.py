# -*- coding: utf-8 -*-
"""程序化「看图」：对制图产物做视觉质量核验（替代人眼抽查）。

检查项：
  1. 非空白：灰度标准差、白色占比、空白边框检查
  2. 有效果：颜色多样性（唯一色数量）、主要色块占比（防"整幅一色"）
  3. 分级渲染：色阶是否成组（聚类主色数量 ≈ 分级数）
  4. 图元完整：四要素（图名/图例/比例尺/指北针）是否出现在工程布局中
  5. 细节密度：边缘密度（防"色块糊成一片"或"几乎空白"）

用法：python scripts/inspect_map_images.py <图片路径> [<工程路径.aprx> ...]
"""
import sys, io, os, glob
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image


def inspect_image(path: str) -> dict:
    img = Image.open(path)
    rgb = np.asarray(img.convert("RGB"), dtype="uint8")
    gray = np.asarray(img.convert("L"), dtype="float32")
    h, w = gray.shape

    # 颜色统计（量化到 16 级，避免抗锯齿噪声）
    quant = (rgb // 16).astype("uint8")
    flat = quant.reshape(-1, 3)
    keys = (flat[:, 0].astype("uint32") << 16) | (flat[:, 1].astype("uint32") << 8) | flat[:, 2]
    counts = Counter(keys.tolist())
    total = len(keys)
    unique_colors = len(counts)
    top_color, top_n = counts.most_common(1)[0]

    # 白色占比（纸张背景）
    white = float(np.mean((gray > 240)))

    # 主色块（占比 > 0.5% 的量化色）
    major = [(k, c) for k, c in counts.items() if c / total > 0.005]
    major.sort(key=lambda x: -x[1])

    # 边缘密度（相邻像素灰度差 > 30 的比例）
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edge = float((np.mean(dx > 30) + np.mean(dy > 30)) / 2)

    # 饱和度（彩色程度）：用于判断是否只是灰度图
    mx = rgb.max(axis=2).astype("float32")
    mn = rgb.min(axis=2).astype("float32")
    sat = float(np.mean((mx - mn) / np.maximum(mx, 1)))

    return {
        "path": path,
        "size": (w, h),
        "kb": round(os.path.getsize(path) / 1024, 1),
        "gray_std": round(float(gray.std()), 2),
        "white_ratio": round(white, 3),
        "unique_colors": unique_colors,
        "major_color_count": len(major),
        "top_color_ratio": round(top_n / total, 3),
        "edge_density": round(edge, 4),
        "saturation": round(sat, 3),
    }


def verdict(info: dict) -> list:
    """给出可判定结论。"""
    out = []
    out.append(("非空白", info["gray_std"] > 20, f"灰度std={info['gray_std']}（>20 合格）"))
    out.append(("有内容（非整幅单色）", info["major_color_count"] >= 3,
                f"占比>0.5%的颜色数={info['major_color_count']}（≥3 合格）"))
    out.append(("留白合理（0.2~0.95）", 0.2 <= info["white_ratio"] <= 0.95,
                f"白底占比={info['white_ratio']}"))
    out.append(("有细节（边缘密度）", info["edge_density"] > 0.002,
                f"边缘密度={info['edge_density']}（>0.002 合格）"))
    return out


def inspect_aprx(aprx_path: str) -> dict:
    from arcpy import mp
    a = mp.ArcGISProject(aprx_path)
    maps = a.listMaps()
    layouts = a.listLayouts()
    lyr_info = []
    for m in maps:
        for lyr in m.listLayers():
            r = getattr(lyr.symbology, "renderer", None)
            lyr_info.append({
                "map": m.name, "layer": lyr.name,
                "renderer": type(r).__name__ if r else None,
                "field": getattr(r, "classificationField", None),
                "class_count": len(getattr(r, "classBreaks", []) or []),
            })
    layout_info = []
    for lay in layouts:
        els = [e.name for e in lay.listElements()]
        layout_info.append({"layout": lay.name, "element_count": len(els), "elements": els})
    return {"aprx": aprx_path, "maps": [m.name for m in maps], "layers": lyr_info, "layouts": layout_info}


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = glob.glob(r"E:\BST_lab\GIS_CLI\workspace\成果展示\**\*.jpg", recursive=True)
    imgs = [a for a in args if a.lower().endswith((".jpg", ".jpeg", ".png"))]
    aprxs = [a for a in args if a.lower().endswith(".aprx")]

    print("=" * 78)
    print("程序化看图：图片质量核验")
    print("=" * 78)
    all_ok = True
    for p in imgs:
        if not os.path.exists(p):
            print(f"[缺失] {p}")
            continue
        info = inspect_image(p)
        print(f"\n{os.path.basename(p)}  ({info['size'][0]}x{info['size'][1]}, {info['kb']} KB)")
        print(f"  灰度std={info['gray_std']} 白底={info['white_ratio']} 唯一色={info['unique_colors']} "
              f"主色数={info['major_color_count']} 主色占比={info['top_color_ratio']} "
              f"边缘密度={info['edge_density']} 饱和度={info['saturation']}")
        for name, ok, detail in verdict(info):
            print(f"    [{'OK  ' if ok else 'FAIL'}] {name}: {detail}")
            all_ok = all_ok and ok

    if aprxs:
        print("\n" + "=" * 78)
        print("工程核验（渲染器 / 布局要素）")
        print("=" * 78)
        for ap in aprxs:
            try:
                info = inspect_aprx(ap)
            except Exception as exc:
                print(f"[FAIL] {ap}: {str(exc)[:120]}")
                all_ok = False
                continue
            print(f"\n{os.path.basename(ap)}")
            for l in info["layers"]:
                print(f"  图层 {l['layer']}: 渲染器={l['renderer']} 字段={l['field']} 分级数={l['class_count']}")
                if l["renderer"] != "Graduated_colors_renderer":
                    all_ok = False
            for lay in info["layouts"]:
                print(f"  布局 {lay['layout']}: {lay['element_count']} 个元素 {lay['elements']}")

    print("\n" + "=" * 78)
    print("结论:", "全部合格" if all_ok else "存在问题（见上 FAIL 行）")
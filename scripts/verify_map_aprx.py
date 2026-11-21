# -*- coding: utf-8 -*-
"""核验 .aprx 工程：图层/渲染器/类别配色/布局四要素/数据源是否都正确落盘。

用法: python scripts/verify_map_aprx.py <工程.aprx> [更多工程...]
"""
import sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import arcpy  # noqa: E402
from arcpy import mp  # noqa: E402


def fmt_color(symbol) -> str:
    """把符号颜色转成 #RRGGBB（API 返回 dict 或 Color 对象两种形态）。"""
    color = getattr(symbol, "color", None)
    if color is None:
        return "(无色)"
    rgb = getattr(color, "RGB", None)
    if rgb is None and isinstance(color, dict):
        rgb = color.get("RGB")
    if not rgb:
        return "(无色)"
    try:
        return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    except Exception:
        return str(rgb)


def describe_renderer(layer) -> list[str]:
    lines = []
    renderer = layer.symbology.renderer
    rtype = type(renderer).__name__
    lines.append(f"  渲染器: {rtype}")
    if rtype == "UniqueValueRenderer":
        lines.append(f"  分类字段: {list(renderer.fields)}")
        for group in renderer.groups:
            for item in group.items:
                vals = list(item.values) if item.values else []
                key = ""
                if vals:
                    v0 = vals[0]
                    key = str(v0[0]) if isinstance(v0, (list, tuple)) else str(v0)
                lines.append(f"    - {key}: {fmt_color(item.symbol)}")
    elif rtype == "GraduatedColorsRenderer":
        lines.append(
            f"  分级字段: {renderer.classificationField} | 级数: {renderer.breakCount}"
            f" | 方法: {renderer.classificationMethod}"
        )
        for index, cb in enumerate(renderer.classBreaks or []):
            lines.append(
                f"    - 第{index + 1}级 {cb.upperBound:g} 以下: {fmt_color(cb.symbol)}"
            )
    return lines


def verify(aprx_path: Path) -> bool:
    print("=" * 78)
    print(f"工程: {aprx_path}")
    if not aprx_path.exists():
        print("  [FAIL] 文件不存在")
        return False
    print(f"  大小: {aprx_path.stat().st_size / 1024:.0f} KB")
    project = mp.ArcGISProject(str(aprx_path))
    maps = project.listMaps()
    layouts = project.listLayouts()
    print(f"  地图数: {len(maps)} | 布局数: {len(layouts)}")
    ok = True
    for layer in [ly for m in maps for ly in m.listLayers() if not ly.isGroupLayer]:
        print(f"  图层: {layer.name}")
        try:
            for line in describe_renderer(layer):
                print(line)
        except Exception as exc:
            ok = False
            print(f"    [FAIL] 渲染器读取失败: {str(exc)[:100]}")
        source = layer.dataSource
        exists = arcpy.Exists(source)
        print(f"  数据源: {source} -> {'存在' if exists else '【缺失】'}")
        ok = ok and exists
    for layout in layouts:
        elements = list(layout.listElements())
        names = [element.name for element in elements]
        kinds = sorted({element.type for element in elements})
        joined = " ".join(names).lower()
        surround = [e for e in elements if "MAPSURROUND" in str(e.type).upper()]
        has = {
            "图名": any("TEXT" in str(e.type).upper() for e in elements),
            "图例": any("LEGEND" in str(e.type).upper() for e in elements),
            "比例尺": ("scale" in joined or "比例尺" in " ".join(names)) or len(surround) >= 2,
            "指北针": ("north" in joined or "指北针" in " ".join(names)) or len(surround) >= 2,
            "地图框": any("MAPFRAME" in str(e.type).upper() for e in elements),
        }
        print(f"  布局: {layout.name} | 元素 {len(names)} 个: {kinds}")
        print(f"        元素名: {names}")
        for key, present in has.items():
            mark = "OK  " if present else "缺失"
            print(f"    [{mark}] {key}")
        ok = ok and all(has.values())
    return ok


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("用法: python scripts/verify_map_aprx.py <工程.aprx> ...")
        return 2
    results = [verify(p) for p in paths]
    print("=" * 78)
    print(f"结论: {sum(results)}/{len(results)} 个工程核验通过")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
# -*- coding: utf-8 -*-
"""探明制图 API 与样式清单（为 cartography/apply.py 提供事实依据）。

一次性脚本，输出：
1. Legend / LegendItem / ScaleBar / NorthArrow 的可用属性，以及我们想设置的属性能否设置成功
2. ArcGIS 2D 样式项清单（NORTH_ARROW / SCALE_BAR / LEGEND），写入 config/cartography_styles.json
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import arcpy  # noqa: E402
from arcpy import mp, cim  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CATALOG = REPO / "config" / "cartography_styles.json"


def blank_template() -> str:
    install = arcpy.GetInstallInfo().get("InstallDir", "") or ""
    cand = os.path.join(install, "Resources", "ArcToolBox", "Services", "routingservices", "data", "Blank.aprx")
    if os.path.exists(cand):
        return cand
    import glob

    found = glob.glob(os.path.join(install, "Resources", "**", "Blank.aprx"), recursive=True)
    if not found:
        raise RuntimeError("未找到 Blank.aprx 模板")
    return found[0]


def props(obj) -> list[str]:
    return [a for a in dir(obj) if not a.startswith("_")]


def try_set(obj, **kwargs) -> dict:
    out = {}
    for key, value in kwargs.items():
        try:
            setattr(obj, key, value)
            out[key] = f"OK({value!r})"
        except Exception as exc:
            out[key] = f"FAIL({type(exc).__name__}: {str(exc)[:60]})"
    return out


def main() -> int:
    ws = REPO / "workspace" / "成果展示" / "PM能力测试"
    gdb_layer = None
    for candidate in sorted(REPO.glob("benchmark/runs/*/pm_task2c_*/workspace/output/result_data.gdb")):
        layer_path = candidate / "community_eval"
        if arcpy.Exists(str(layer_path)):
            gdb_layer = str(layer_path)
            break
    if gdb_layer is None:
        print("找不到测试图层，退化为只列样式清单")
    print("测试图层:", gdb_layer)

    tmp = Path(tempfile.gettempdir()) / "map_api_probe"
    tmp.mkdir(parents=True, exist_ok=True)
    aprx_path = tmp / "probe.aprx"
    if aprx_path.exists():
        aprx_path.unlink()
    shutil.copy2(blank_template(), aprx_path)
    aprx = mp.ArcGISProject(str(aprx_path))

    # ---- 样式清单
    catalog: dict[str, list[str]] = {}
    for kind in ("NORTH_ARROW", "SCALE_BAR", "LEGEND"):
        try:
            catalog[kind] = [item.name for item in aprx.listStyleItems("ArcGIS 2D", kind)]
        except Exception as exc:
            catalog[kind] = []
            print(f"枚举 {kind} 失败: {str(exc)[:100]}")
    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    for kind, names in catalog.items():
        print(f"\n=== {kind}（{len(names)}）===")
        print("  " + " | ".join(names[:40]))

    if gdb_layer is None:
        return 0

    # ---- 布局与元素
    layout = aprx.createLayout(210, 297, "MILLIMETER")
    layout.name = "probe"
    m = aprx.listMaps()[0]
    layer = m.addDataFromPath(gdb_layer)
    mf = layout.createMapFrame(
        arcpy.Polygon(arcpy.Array([
            arcpy.Point(10, 40), arcpy.Point(200, 40), arcpy.Point(200, 250), arcpy.Point(10, 250), arcpy.Point(10, 40),
        ])),
        m,
        "Main Map",
    )
    mf.camera.setExtent(mf.getLayerExtent(layer))

    na_style = aprx.listStyleItems("ArcGIS 2D", "NORTH_ARROW")[0]
    sb_style = aprx.listStyleItems("ArcGIS 2D", "SCALE_BAR")[0]
    lg_style = aprx.listStyleItems("ArcGIS 2D", "LEGEND")[0]
    na = layout.createMapSurroundElement(arcpy.Point(190, 240), "NORTH_ARROW", mf, na_style)
    legend = layout.createMapSurroundElement(
        arcpy.Polygon(arcpy.Array([
            arcpy.Point(120, 10), arcpy.Point(195, 10), arcpy.Point(195, 35), arcpy.Point(120, 35), arcpy.Point(120, 10),
        ])),
        "LEGEND", mf, lg_style,
    )
    sb = layout.createMapSurroundElement(
        arcpy.Polygon(arcpy.Array([
            arcpy.Point(10, 10), arcpy.Point(100, 10), arcpy.Point(100, 35), arcpy.Point(10, 35), arcpy.Point(10, 10),
        ])),
        "SCALE_BAR", mf, sb_style,
    )

    print("\n=== 属性清单 ===")
    print("legend:", props(legend))
    print("legend.items[0]:", props(legend.items[0]) if getattr(legend, "items", None) else "(无 items)")
    print("scale_bar:", props(sb))
    print("north_arrow:", props(na))
    print("mapframe:", [p for p in props(mf) if "element" in p.lower() or "camera" in p.lower()])

    print("\n=== 尝试设置（legend）===")
    print(try_set(legend, title="社区类型", autoAdd=False, elementPositionX=120, elementPositionY=10,
                  elementWidth=75, elementHeight=25))
    if getattr(legend, "items", None):
        item = legend.items[0]
        print("legendItem:", try_set(item, showHeading=False, showLabels=True, showVisibleCheckboxes=False,
                                     patchWidth=8, patchHeight=5))
        for attr in ("labelSymbol", "symbolHeight", "labelGap"):
            if hasattr(item, attr):
                print(f"  legendItem.{attr}:", type(getattr(item, attr)).__name__)
    if hasattr(legend, "titleSymbol"):
        print("titleSymbol:", try_set(legend.titleSymbol, height=11, fontFamilyName="宋体", bold=True))
    if hasattr(legend, "labelSymbol"):
        print("legend.labelSymbol:", try_set(legend.labelSymbol, height=9, fontFamilyName="宋体"))

    print("\n=== 尝试设置（scale bar）===")
    print(try_set(sb, divisions=4, subdivisions=1, units="METERS", unitLabel="米",
                  elementPositionX=10, elementPositionY=12, elementWidth=60, elementHeight=10))
    for attr in ("labelSymbol", "divisionHeight", "labelPosition", "numberFormat", "labelGap", "fitted"):
        if hasattr(sb, attr):
            print(f"  scaleBar.{attr} =", getattr(sb, attr))

    print("\n=== 尝试设置（north arrow）===")
    print(try_set(na, elementPositionX=175, elementPositionY=225, elementWidth=16, elementHeight=16))
    print("northArrow class:", type(na).__name__)

    print("\n=== 尝试设置（CIM 文本字号）===")
    d = layout.getDefinition("V3")
    tg = cim.CIMParagraphTextGraphic()
    tg.text = "测试标题"
    sym = cim.CIMTextSymbol()
    sym.fontFamilyName = "宋体"
    sym.height = 20
    sym.bold = True
    sym.horizontalAlignment = "Center"
    tg.symbol = sym
    el = cim.CIMGraphicElement()
    el.name = "Map Title"
    el.graphic = tg
    d.elements.append(el)
    layout.setDefinition(d)
    print("CIM 文本 height 设置完成（若下方无异常即属性名正确）")
    try:
        aprx.save()
        print("aprx 保存成功")
    except Exception as exc:
        print("aprx 保存失败:", str(exc)[:120])
    out = str(tmp / "probe.jpg")
    layout.exportToJPEG(out, resolution=120)
    print("导出:", out, os.path.exists(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
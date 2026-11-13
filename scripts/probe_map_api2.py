# -*- coding: utf-8 -*-
"""精确探查：布局 create* 方法、文本元素 API、图例/比例尺的 CIM 定义结构。"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import arcpy  # noqa: E402
from arcpy import mp, cim  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def blank_template() -> str:
    install = arcpy.GetInstallInfo().get("InstallDir", "") or ""
    cand = os.path.join(install, "Resources", "ArcToolBox", "Services", "routingservices", "data", "Blank.aprx")
    if os.path.exists(cand):
        return cand
    import glob

    return glob.glob(os.path.join(install, "Resources", "**", "Blank.aprx"), recursive=True)[0]


def props(obj) -> list[str]:
    return [a for a in dir(obj) if not a.startswith("_")]


def main() -> int:
    layer_path = None
    for candidate in sorted(REPO.glob("benchmark/runs/*/pm_task2c_*/workspace/output/result_data.gdb")):
        if arcpy.Exists(str(candidate / "community_eval")):
            layer_path = str(candidate / "community_eval")
            break

    tmp = Path(tempfile.gettempdir()) / "map_api_probe2"
    tmp.mkdir(parents=True, exist_ok=True)
    aprx_path = tmp / "probe2.aprx"
    if aprx_path.exists():
        aprx_path.unlink()
    shutil.copy2(blank_template(), aprx_path)
    aprx = mp.ArcGISProject(str(aprx_path))
    layout = aprx.createLayout(297, 210, "MILLIMETER")
    print("layout.create* :", [p for p in props(layout) if p.startswith("create")])
    m = aprx.listMaps()[0]
    layer = m.addDataFromPath(layer_path) if layer_path else None
    mf = layout.createMapFrame(
        arcpy.Polygon(arcpy.Array([
            arcpy.Point(10, 40), arcpy.Point(200, 40), arcpy.Point(200, 190), arcpy.Point(10, 190), arcpy.Point(10, 40),
        ])),
        m, "Main Map",
    )
    if layer is not None:
        mf.camera.setExtent(mf.getLayerExtent(layer))

    # ---- 文本元素
    if hasattr(layout, "createTextElement"):
        el = layout.createTextElement(
            arcpy.Polygon(arcpy.Array([
                arcpy.Point(20, 195), arcpy.Point(280, 195), arcpy.Point(280, 205), arcpy.Point(20, 205), arcpy.Point(20, 195),
            ])),
            "测试标题 ABC",
        )
        print("\ntext element 属性:", props(el))
        for key, value in (("fontSize", 24), ("fontFamily", "宋体"), ("bold", True), ("textSize", 24), ("fontStyle", "Bold")):
            try:
                setattr(el, key, value)
                print(f"  set {key}={value!r} -> OK, 回读 {getattr(el, key)!r}")
            except Exception as exc:
                print(f"  set {key}={value!r} -> FAIL {str(exc)[:70]}")
        if hasattr(el, "getDefinition"):
            cim_def = el.getDefinition("V3")
            print("  CIM 类:", type(cim_def).__name__, "| 序列化字段示例:",
                  [k for k in ("name", "graphic") if hasattr(cim_def, k)])
    else:
        print("\nlayout 无 createTextElement（需用 CIM 创建文本）")

    # ---- 图例 CIM 结构
    legend = layout.createMapSurroundElement(
        arcpy.Polygon(arcpy.Array([
            arcpy.Point(210, 45), arcpy.Point(288, 45), arcpy.Point(288, 80), arcpy.Point(210, 80), arcpy.Point(210, 45),
        ])),
        "LEGEND", mf, aprx.listStyleItems("ArcGIS 2D", "LEGEND")[0],
    )
    ldef = legend.getDefinition("V3")
    print("\nlegend CIM 顶层:", props(ldef))
    items = getattr(ldef, "items", None) or []
    if items:
        print("legendItem CIM 属性:", props(items[0]))
    print("legend 可设置 title:", hasattr(legend, "title"), "| showTitle:", hasattr(legend, "showTitle"))

    # 直接改 CIM 图例项
    if items:
        try:
            items[0].showHeading = False
            items[0].showLabels = True
            if hasattr(items[0], "labelSymbol"):
                items[0].labelSymbol.height = 10
                items[0].labelSymbol.fontFamilyName = "宋体"
            legend.setDefinition(ldef)
            print("CIM 图例项设置成功（showHeading/showLabels/labelSymbol.height）")
        except Exception as exc:
            print("CIM 图例项设置失败:", str(exc)[:150])

    # ---- 比例尺 CIM 结构
    sb = layout.createMapSurroundElement(
        arcpy.Polygon(arcpy.Array([
            arcpy.Point(10, 10), arcpy.Point(100, 10), arcpy.Point(100, 35), arcpy.Point(10, 35), arcpy.Point(10, 10),
        ])),
        "SCALE_BAR", mf, aprx.listStyleItems("ArcGIS 2D", "SCALE_BAR")[0],
    )
    sdef = sb.getDefinition("V3")
    print("\nscale bar CIM:", props(sdef))
    try:
        sb.divisions = 4
        sb.subdivisions = 1
        sb.units = "METERS"
        sb.unitLabel = "米"
        aprx.save()
        print("比例尺 divisions/subdivisions/units/unitLabel 设置 + 保存 OK")
    except Exception as exc:
        print("比例尺设置失败:", str(exc)[:120])

    # ---- 指北针
    na = layout.createMapSurroundElement(arcpy.Point(280, 190), "NORTH_ARROW", mf,
                                         aprx.listStyleItems("ArcGIS 2D", "NORTH_ARROW")[0])
    na.elementWidth = 16
    na.elementHeight = 16
    print("\n指北针 elementWidth/Height OK")

    aprx.save()
    layout.exportToJPEG(str(tmp / "probe2.jpg"), resolution=100)
    print("导出完成:", (tmp / "probe2.jpg").exists())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# -*- coding: utf-8 -*-
"""生成样式样张：把候选指北针 / 比例尺铺成网格导出图片，用于人工（多模态）挑选。"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import arcpy  # noqa: E402
from arcpy import mp, cim  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = Path(tempfile.gettempdir()) / "style_sheet"

NORTH_PREFS = ["罗盘", "compass", "正北", "格网北", "地形", "简单实心", "ArcGIS 指北针"]
SB_PREFS = ["公制黑白相间", "黑白相间", "公制比例线", "比例线", "公制双重", "双重黑白", "公制空心", "空心", "公制单一"]


def blank_template() -> str:
    install = arcpy.GetInstallInfo().get("InstallDir", "") or ""
    cand = os.path.join(install, "Resources", "ArcToolBox", "Services", "routingservices", "data", "Blank.aprx")
    if os.path.exists(cand):
        return cand
    import glob

    return glob.glob(os.path.join(install, "Resources", "**", "Blank.aprx"), recursive=True)[0]


def pick_names(all_names: list[str], prefs: list[str], limit: int = 12) -> list[str]:
    picked: list[str] = []
    for pref in prefs:
        for name in all_names:
            if pref.lower() in name.lower() and name not in picked:
                picked.append(name)
                if len(picked) >= limit:
                    return picked
    for name in all_names:
        if name not in picked:
            picked.append(name)
            if len(picked) >= limit:
                break
    return picked


def add_label(layout, x: float, y: float, text: str, size: int = 7) -> None:
    d = layout.getDefinition("V3")
    pg = cim.CIMParagraphTextGraphic()
    pg.text = text
    sym = cim.CIMTextSymbol()
    sym.fontFamilyName = "宋体"
    sym.height = size
    pg.symbol = sym
    pg.shape = arcpy.Polygon(arcpy.Array([
        arcpy.Point(x, y), arcpy.Point(x + 90, y), arcpy.Point(x + 90, y + 8), arcpy.Point(x, y + 8), arcpy.Point(x, y),
    ]))
    el = cim.CIMGraphicElement()
    el.name = f"lbl_{text[:12]}"
    el.graphic = pg
    d.elements.append(el)
    layout.setDefinition(d)


def main() -> int:
    layer_path = None
    for candidate in sorted(REPO.glob("benchmark/runs/*/pm_task2c_*/workspace/output/result_data.gdb")):
        if arcpy.Exists(str(candidate / "community_eval")):
            layer_path = str(candidate / "community_eval")
            break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / "styles.aprx"
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(blank_template(), tmp)
    aprx = mp.ArcGISProject(str(tmp))
    layout = aprx.createLayout(420, 297, "MILLIMETER")
    m = aprx.listMaps()[0]
    layer = m.addDataFromPath(layer_path)
    mf = layout.createMapFrame(
        arcpy.Polygon(arcpy.Array([
            arcpy.Point(10, 10), arcpy.Point(410, 10), arcpy.Point(410, 287), arcpy.Point(10, 287), arcpy.Point(10, 10),
        ])),
        m, "Main Map",
    )
    mf.camera.setExtent(mf.getLayerExtent(layer))
    mf.camera.scale = 60000  # 固定比例尺，让比例尺刻度有可比性

    # 指北针样张
    north_names = pick_names([i.name for i in aprx.listStyleItems("ArcGIS 2D", "NORTH_ARROW")], NORTH_PREFS, 12)
    cols, cell_w, cell_h = 6, 60, 60
    for index, name in enumerate(north_names):
        style = next(i for i in aprx.listStyleItems("ArcGIS 2D", "NORTH_ARROW") if i.name == name)
        r, c = divmod(index, cols)
        x = 20 + c * cell_w
        y = 230 - r * cell_h
        na = layout.createMapSurroundElement(arcpy.Point(x, y), "NORTH_ARROW", mf, style)
        na.elementWidth = 22
        na.elementHeight = 22
        add_label(layout, x - 8, y - 26, f"{index + 1}.{name}", 7)
    layout.exportToJPEG(str(OUT_DIR / "north_arrows.jpg"), resolution=150)
    print("指北针样张:", OUT_DIR / "north_arrows.jpg")
    for index, name in enumerate(north_names):
        print(f"  {index + 1}. {name}")

    # 清掉指北针再排比例尺（保留地图框）
    for element in list(layout.listElements("MAPSURROUND_ELEMENT")):
        try:
            element.delete()
        except Exception:
            pass
    for element in list(layout.listElements("GRAPHIC_ELEMENT")):
        try:
            if str(getattr(element, "name", "")).startswith("lbl_"):
                element.delete()
        except Exception:
            pass

    sb_names = pick_names([i.name for i in aprx.listStyleItems("ArcGIS 2D", "SCALE_BAR")], SB_PREFS, 12)
    cols, cell_w, cell_h = 4, 105, 55
    for index, name in enumerate(sb_names):
        style = next(i for i in aprx.listStyleItems("ArcGIS 2D", "SCALE_BAR") if i.name == name)
        r, c = divmod(index, cols)
        x = 15 + c * cell_w
        y = 235 - r * cell_h
        sb = layout.createMapSurroundElement(
            arcpy.Polygon(arcpy.Array([
                arcpy.Point(x, y), arcpy.Point(x + 85, y), arcpy.Point(x + 85, y + 25), arcpy.Point(x, y + 25), arcpy.Point(x, y),
            ])),
            "SCALE_BAR", mf, style,
        )
        try:
            sb.divisions = 4
            sb.subdivisions = 1
            sb.units = "METERS"
            sb.unitLabel = "米"
        except Exception as exc:
            print("  设置失败", name, str(exc)[:60])
        add_label(layout, x, y + 26, f"{index + 1}.{name}", 7)
    aprx.save()
    layout.exportToJPEG(str(OUT_DIR / "scale_bars.jpg"), resolution=150)
    print("比例尺样张:", OUT_DIR / "scale_bars.jpg")
    for index, name in enumerate(sb_names):
        print(f"  {index + 1}. {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
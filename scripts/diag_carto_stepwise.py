# -*- coding: utf-8 -*-
"""分步排查：哪一步 CIM 操作让 ArcGIS 静默崩溃。每步立刻 flush 输出。"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import arcpy  # noqa: E402
from arcpy import mp, cim  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def say(message: str) -> None:
    print(message, flush=True)


def blank() -> str:
    install = arcpy.GetInstallInfo().get("InstallDir", "") or ""
    cand = os.path.join(install, "Resources", "ArcToolBox", "Services", "routingservices", "data", "Blank.aprx")
    if os.path.exists(cand):
        return cand
    import glob

    return glob.glob(os.path.join(install, "Resources", "**", "Blank.aprx"), recursive=True)[0]


def rect(x, y, w, h):
    return arcpy.Polygon(arcpy.Array([
        arcpy.Point(x, y), arcpy.Point(x + w, y), arcpy.Point(x + w, y + h), arcpy.Point(x, y + h), arcpy.Point(x, y),
    ]))


def main() -> int:
    layer_path = None
    for candidate in sorted(REPO.glob("benchmark/runs/*/pm_task2b_*/workspace/output/result_data.gdb")):
        if arcpy.Exists(str(candidate / "community_service")):
            layer_path = str(candidate / "community_service")
            break
    work = Path(tempfile.gettempdir()) / "carto_stepwise"
    work.mkdir(parents=True, exist_ok=True)
    aprx_path = work / "step.aprx"
    if aprx_path.exists():
        aprx_path.unlink()
    shutil.copy2(blank(), aprx_path)
    say("1 模板复制完成")
    project = mp.ArcGISProject(str(aprx_path))
    say("2 工程打开")
    layout = project.createLayout(210, 297, "MILLIMETER")
    say("3 布局创建")
    m = project.listMaps()[0]
    layer = m.addDataFromPath(layer_path)
    say("4 图层添加")
    frame = layout.createMapFrame(rect(10, 40, 190, 220), m, "Main Map")
    frame.camera.setExtent(frame.getLayerExtent(layer))
    say("5 地图框创建 + 相机设置")

    for mode in ("CIMPolygonGraphic", "CIMPolygonSymbol", "CIMSolidStroke", "CIMTextGraphic", "CIMTextSymbol",
                 "CIMParagraphTextGraphic", "CIMParagraphTextSymbol", "CIMLinearUnit", "CIMClassBreaksRenderer",
                 "CIMColorRamp"):
        say(f"   检查 cim.{mode}: {hasattr(cim, mode)}")

    # A. 图廓（多边形描边）
    try:
        definition = layout.getDefinition("V3")
        say("6a 取布局定义 OK")
        polygon_graphic = cim.CIMPolygonGraphic()
        polygon_symbol = cim.CIMPolygonSymbol()
        stroke = cim.CIMSolidStroke()
        stroke.color = cim.CIMRGBColor()
        stroke.color.red, stroke.color.green, stroke.color.blue = 0, 0, 0
        stroke.width = 0.5
        polygon_symbol.symbolLayers = [stroke]
        polygon_graphic.symbol = polygon_symbol
        polygon_graphic.shape = rect(4, 4, 202, 289)
        element = cim.CIMGraphicElement()
        element.name = "Neatline"
        element.graphic = polygon_graphic
        definition.elements.append(element)
        layout.setDefinition(definition)
        say("6b 图廓（CIMPolygonGraphic）添加 OK")
    except Exception as exc:
        say(f"6b 图廓失败: {type(exc).__name__}: {str(exc)[:120]}")

    # B. 标题：分别试三种写法，看哪种不崩且能控字号
    for label, use_paragraph, symbol_name in (
        ("B1 段落+CIMParagraphTextSymbol", True, "CIMParagraphTextSymbol"),
        ("B2 段落+CIMTextSymbol", True, "CIMTextSymbol"),
    ):
        if not hasattr(cim, symbol_name):
            say(f"{label}: 类不存在，跳过")
            continue
        try:
            definition = layout.getDefinition("V3")
            graphic = cim.CIMParagraphTextGraphic()
            graphic.text = f"{label} 社区尺度老年人口分布图"
            symbol = getattr(cim, symbol_name)()
            symbol.fontFamilyName = "宋体"
            symbol.height = 20
            try:
                symbol.bold = True
            except Exception:
                pass
            try:
                symbol.horizontalAlignment = "Center"
            except Exception:
                pass
            symbol.color = cim.CIMRGBColor()
            symbol.color.red, symbol.color.green, symbol.color.blue = 0, 0, 0
            graphic.symbol = symbol
            graphic.shape = rect(7, 265 if use_paragraph else 250, 196, 12)
            element = cim.CIMGraphicElement()
            element.name = f"Title_{label[:2]}"
            element.graphic = graphic
            definition.elements.append(element)
            layout.setDefinition(definition)
            say(f"{label}: 添加 OK")
        except Exception as exc:
            say(f"{label} 失败: {type(exc).__name__}: {str(exc)[:120]}")

    # C. 比例尺（CIMLinearUnit）
    try:
        scale_bar = layout.createMapSurroundElement(rect(12, 12, 57, 12), "SCALE_BAR", frame,
                                                   project.listStyleItems("ArcGIS 2D", "SCALE_BAR")[0])
        say("8a 比例尺元素创建 OK")
        definition = scale_bar.getDefinition("V3")
        if hasattr(cim, "CIMLinearUnit"):
            unit = cim.CIMLinearUnit()
            unit.uwkid = 9001
            definition.units = unit
        definition.unitLabel = "米"
        definition.division = 1000.0
        definition.divisions = 4
        scale_bar.setDefinition(definition)
        back = scale_bar.getDefinition("V3")
        say(f"8b 比例尺 CIM 设置 OK: division={getattr(back,'division',None)} units={getattr(back,'units',None)} unitLabel={getattr(back,'unitLabel',None)!r}")
    except Exception as exc:
        say(f"8b 比例尺失败: {type(exc).__name__}: {str(exc)[:120]}")

    # D. 图例 + 分级标签
    try:
        sym = layer.symbology
        sym.updateRenderer("GraduatedColorsRenderer")
        sym.renderer.classificationField = "ELDER_COMM"
        sym.renderer.breakCount = 5
        layer.symbology = sym
        say("9a 分级渲染 OK")
        layer_def = layer.getDefinition("V3")
        renderer = getattr(layer_def, "renderer", None)
        say(f"9b 图层 CIM 渲染器类型: {type(renderer).__name__} 有 classBreaks: {hasattr(renderer, 'classBreaks')}")
        breaks = getattr(renderer, "classBreaks", None) or []
        for index, brk in enumerate(breaks):
            if hasattr(brk, "label"):
                brk.label = f"级别{index + 1}"
        layer.setDefinition(layer_def)
        say(f"9c CIM 分级标签设置 OK（{len(breaks)} 级）")
    except Exception as exc:
        say(f"9c 分级标签失败: {type(exc).__name__}: {str(exc)[:120]}")

    project.save()
    say("10 工程保存")
    layout.exportToJPEG(str(work / "step.jpg"), resolution=200)
    say(f"11 导出完成: {(work / 'step.jpg').exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
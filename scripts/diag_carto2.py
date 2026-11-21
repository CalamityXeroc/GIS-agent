# -*- coding: utf-8 -*-
"""第二轮排查：标题字号怎么才能真正生效；图例标签/分级断点怎么设；比例尺单位。

只测不崩的路径：
- 标题：①TEXT_ELEMENT 自身属性 ②两遍 setDefinition（先建元素后改 symbol）
- 图例：①直接改 classBreaks[i].upperBound 成整数 ②DefinedInterval + intervalSize
- 比例尺：units 字符串写法对比 + division 反算
"""
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
    work = Path(tempfile.gettempdir()) / "carto_diag2"
    work.mkdir(parents=True, exist_ok=True)
    aprx_path = work / "diag2.aprx"
    if aprx_path.exists():
        aprx_path.unlink()
    shutil.copy2(blank(), aprx_path)
    project = mp.ArcGISProject(str(aprx_path))
    layout = project.createLayout(210, 297, "MILLIMETER")
    m = project.listMaps()[0]
    layer = m.addDataFromPath(layer_path)
    frame = layout.createMapFrame(rect(10, 40, 190, 220), m, "Main Map")
    frame.camera.setExtent(frame.getLayerExtent(layer))

    # ---- 标题：先建（段落+CIMTextSymbol），看元素属性，再二遍改 symbol
    definition = layout.getDefinition("V3")
    graphic = cim.CIMParagraphTextGraphic()
    graphic.text = "标题字号测试 社区尺度老年人口分布图"
    symbol = cim.CIMTextSymbol()
    symbol.fontFamilyName = "宋体"
    symbol.height = 8
    graphic.symbol = symbol
    graphic.shape = rect(7, 270, 196, 12)
    element = cim.CIMGraphicElement()
    element.name = "Map Title"
    element.graphic = graphic
    definition.elements.append(element)
    layout.setDefinition(definition)
    say("标题元素已建（初始 height=8）")

    for candidate in layout.listElements():
        if str(candidate.name) == "Map Title":
            say(f"TEXT_ELEMENT 属性: {[a for a in dir(candidate) if not a.startswith('_')]}")
            break

    # 二遍：把 symbol.height 改成 20
    try:
        definition = layout.getDefinition("V3")
        for item in definition.elements:
            if str(getattr(item, "name", "")) == "Map Title":
                target = getattr(item, "graphic", item)
                target_symbol = getattr(target, "symbol", None)
                if target_symbol is not None:
                    target_symbol.height = 20
                    try:
                        target_symbol.bold = True
                    except Exception:
                        pass
                    try:
                        target_symbol.horizontalAlignment = "Center"
                    except Exception:
                        pass
                    say("二遍改 symbol.height=20 已写入")
        layout.setDefinition(definition)
        say("二遍 setDefinition 完成")
    except Exception as exc:
        say(f"二遍改字号失败: {type(exc).__name__}: {str(exc)[:120]}")

    # ---- 图例：分级断点改整数 + DefinedInterval
    sym = layer.symbology
    sym.updateRenderer("GraduatedColorsRenderer")
    sym.renderer.classificationField = "ELDER_COMM"
    sym.renderer.breakCount = 5
    layer.symbology = sym
    renderer = layer.symbology.renderer
    say(f"分级渲染器: breakCount={getattr(renderer,'breakCount',None)} "
        f"method={getattr(renderer,'classificationMethod',None)} "
        f"有 intervalSize={hasattr(renderer,'intervalSize')}")
    breaks = renderer.classBreaks or []
    say(f"初始断点上界: {[getattr(b,'upperBound',None) for b in breaks]}")
    try:
        for index, brk in enumerate(breaks):
            brk.upperBound = 2000.0 * (index + 1)
        layer.symbology = layer.symbology
        say(f"改成整数上界后: {[getattr(b,'upperBound',None) for b in layer.symbology.renderer.classBreaks]}")
    except Exception as exc:
        say(f"改 upperBound 失败: {type(exc).__name__}: {str(exc)[:100]}")
    try:
        if hasattr(renderer, "intervalSize"):
            renderer.classificationMethod = "DefinedInterval"
            renderer.intervalSize = 2500.0
            layer.symbology = layer.symbology
            say(f"DefinedInterval 后断点: {[getattr(b,'upperBound',None) for b in layer.symbology.renderer.classBreaks]}")
            say(f"   标签: {[getattr(b,'label','') for b in layer.symbology.renderer.classBreaks]}")
    except Exception as exc:
        say(f"DefinedInterval 失败: {type(exc).__name__}: {str(exc)[:100]}")

    legend = layout.createMapSurroundElement(rect(150, 60, 45, 45), "LEGEND", frame,
                                             project.listStyleItems("ArcGIS 2D", "LEGEND")[0])
    legend.title = "老年人口"
    ldef = legend.getDefinition("V3")
    for item in getattr(ldef, "items", []) or []:
        item.showHeading = False
        item.showLayerName = False
    legend.setDefinition(ldef)

    # ---- 比例尺：元素 API 的 units 字符串 vs CIM division 反算
    sb = layout.createMapSurroundElement(rect(12, 12, 57, 12), "SCALE_BAR", frame,
                                        project.listStyleItems("ArcGIS 2D", "SCALE_BAR")[0])
    for units_value in ("METERS", "Meters", "METER"):
        try:
            sb.units = units_value
            back = sb.getDefinition("V3")
            say(f"units={units_value!r} -> 回读 {getattr(back, 'units', None)} unitLabel={getattr(back,'unitLabel',None)!r}")
        except Exception as exc:
            say(f"units={units_value!r} 失败: {type(exc).__name__}")
    try:
        sb.unitLabel = "米"
        sb.elementWidth = 57.6
        sb.elementHeight = 11.0
        back = sb.getDefinition("V3")
        say(f"最终比例尺: division={getattr(back,'division',None)} divisions={getattr(back,'divisions',None)} "
            f"elementWidth={getattr(sb,'elementWidth',None)} mapScale={frame.camera.scale:.0f}")
    except Exception as exc:
        say(f"比例尺收尾失败: {type(exc).__name__}: {str(exc)[:100]}")

    project.save()
    layout.exportToJPEG(str(work / "diag2.jpg"), resolution=200)
    say(f"导出完成: {(work / 'diag2.jpg').exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
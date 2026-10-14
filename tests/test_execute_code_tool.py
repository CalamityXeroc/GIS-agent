"""Unit tests for execute_code ArcPy code auto-fixes."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gis_cli.tools.execute_code import ExecuteCodeTool


def test_fix_common_api_mistakes_rewrites_map_camera_and_layout_mapframe():
    code = """
aprx = arcpy.mp.ArcGISProject(aprx_path)
m = aprx.listMaps()[0]
layer = m.addDataFromPath(input_fc)
layout = aprx.listLayouts()[0]
camera = m.camera
camera.setExtent(layout.mapFrame.elementExtent)
layout.exportToJPEG(output_path, 300)
"""
    fixed = ExecuteCodeTool._fix_common_api_mistakes(code)

    assert "m.camera" not in fixed
    assert "layout.mapFrame.elementExtent" not in fixed
    assert 'layout.listElements("MAPFRAME_ELEMENT")' in fixed
    assert "mf.map = m" in fixed


def test_fix_common_api_mistakes_injects_mapframe_binding_before_layout_export():
    code = """
aprx = arcpy.mp.ArcGISProject(aprx_path)
m = aprx.listMaps()[0]
layer = m.addDataFromPath(input_fc)
layout = aprx.listLayouts()[0]
layout.exportToJPEG(output_path, 300)
"""
    fixed = ExecuteCodeTool._fix_common_api_mistakes(code)

    assert "mf.map = m" in fixed
    assert "mf.camera.setExtent" in fixed


def test_check_known_bad_patterns_detects_non_m_map_camera_usage():
    code = """
aprx = arcpy.mp.ArcGISProject(aprx_path)
main_map = aprx.listMaps()[0]
camera = main_map.camera
"""
    errors = ExecuteCodeTool._check_known_bad_patterns(code)
    assert errors
    assert any("Map 对象没有 .camera 属性" in msg for msg in errors)

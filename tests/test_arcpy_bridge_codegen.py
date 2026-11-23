from gis_cli.arcpy_bridge import build_graduated_colors_code


def test_build_graduated_colors_code_uses_cim_graphic_elements() -> None:
    code = build_graduated_colors_code(
        input_path=r"E:\BST_lab\GIS CLI\workspace\output\province_forest_coverage_albers.shp",
        field_name="ForestCov",
        output_path=r"E:\BST_lab\GIS CLI\workspace\output\china_vegetation_coverage.jpg",
        title="China Vegetation Coverage",
        color_ramp_name="Greens",
    )

    assert "cim.CIMGraphicElement()" in code
    assert "_title_pg = cim.CIMParagraphTextGraphic()" in code
    assert "_title_el.graphic = _title_pg" in code
    assert "_nl_el.graphic = _nl_line" in code


def test_build_graduated_colors_code_keeps_mapframe_after_set_definition() -> None:
    code = build_graduated_colors_code(
        input_path=r"E:\BST_lab\GIS CLI\workspace\output\province_forest_coverage_albers.shp",
        field_name="ForestCov",
        output_path=r"E:\BST_lab\GIS CLI\workspace\output\china_vegetation_coverage.jpg",
    )

    set_def_index = code.index("layout.setDefinition(_nl_d)")
    create_mf_index = code.index('mf = layout.createMapFrame(mf_geom, m, "Main Map")')
    assert set_def_index < create_mf_index


def test_build_graduated_colors_code_contains_green_force_fallback() -> None:
    code = build_graduated_colors_code(
        input_path=r"E:\BST_lab\GIS CLI\workspace\output\province_forest_coverage.shp",
        field_name="forest_cov",
        output_path=r"E:\BST_lab\GIS CLI\workspace\output\china_vegetation_coverage.jpg",
        color_ramp_name="Greens",
    )

    assert "_infer_hue" in code
    assert "_force_hue" in code
    assert "_palettes = {" in code
    assert "_cb.symbol.color = _palette" in code

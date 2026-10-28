# -*- coding: utf-8 -*-
"""制图规则引擎单测（纯 Python，不需要 ArcGIS）。

覆盖：长度取整、图例标签格式、纸张朝向矩阵、图例需求与标题、比例尺单位与整数刻度、
图名拼装与字号、决策理由、版面修复建议、样式选择、档位合并、facts 纯函数、渲染层小工具。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from gis_cli.cartography import describe_spec, design_only, suggest_fixes
from gis_cli.cartography.design import (
    format_class_labels,
    nice_division,
    nice_length,
)
from gis_cli.cartography.facts import (
    aspect_ratio,
    corner_usage,
    emptiest_corners,
    guess_render_mode,
    normalize_facts,
    union_extent,
)
from gis_cli.cartography.profiles import builtin_profiles, load_profile
from gis_cli.cartography.styles import pick_style_name


# ------------------------------------------------------------------ 构造假事实

def grid_from_corners(usage: dict[str, float]) -> list[list[float]]:
    """把"四角占用度"写成九宫格（用于驱动图例/指北针的角位选择）。"""
    tl, tr = usage.get("top_left", 0.0), usage.get("top_right", 0.0)
    bl, br = usage.get("bottom_left", 0.0), usage.get("bottom_right", 0.0)
    return [
        [tl + 1.0, (tl + tr) / 2 + 1.0, tr + 1.0],
        [(tl + bl) / 2 + 1.0, 1.0, (tr + br) / 2 + 1.0],
        [bl + 1.0, (bl + br) / 2 + 1.0, br + 1.0],
    ]


def make_facts(
    *,
    aspect: float = 1.02,
    mode: str = "graduated",
    width_m: float = 30000.0,
    feature_count: int = 2341,
    grid: list[list[float]] | None = None,
    count_usage: dict[str, float] | None = None,
    field_stats: dict | None = None,
    layers: int = 1,
) -> dict:
    height_m = width_m / max(aspect, 1e-6)
    # 默认：城市型稀疏数据（中心密、四角近空）→ 图例留在图框内
    default_usage = {"top_left": 0.2, "top_right": 0.2, "bottom_left": 0.2, "bottom_right": 0.2}
    occupancy = grid or [
        [default_usage["top_left"], 0.8, default_usage["top_right"]],
        [0.8, 5.0, 0.8],
        [default_usage["bottom_left"], 0.8, default_usage["bottom_right"]],
    ]
    layer_list = []
    for index in range(layers):
        layer_list.append({
            "path": f"data_{index}.gdb/layer",
            "exists": True,
            "geom_type": "Polygon",
            "feature_count": feature_count,
            "extent": {"xmin": 0, "ymin": 0, "xmax": width_m, "ymax": height_m, "width": width_m, "height": height_m},
            "crs": {"wkid": 4548, "name": "CGCS2000", "is_projected": True},
            "occupancy": occupancy,
            "field_stats": {"V": field_stats} if (field_stats and index == 0) else {},
        })
    raw = {
        "layers": layer_list,
        "extent": {"xmin": 0, "ymin": 0, "xmax": width_m, "ymax": height_m, "width": width_m, "height": height_m},
        "aspect": aspect,
        "occupancy": occupancy,
        "corner_usage": count_usage or {"top_left": 0.0, "top_right": 0.0, "bottom_left": 0.0, "bottom_right": 0.0},
        "primary": {"feature_count": feature_count, "crs": {"is_projected": True}, "data_type": "FeatureClass"},
        "layer_count": layers,
        "render_mode": mode,
        "field": "V",
    }
    return normalize_facts(raw)


CONTINUOUS = {"type": "Double", "min": 0.0, "max": 100.0, "p25": 20.0, "p50": 40.0, "p75": 70.0, "mean": 45.0,
              "skew": 0.05, "null_rate": 0.01, "sample_count": 100}
CATEGORICAL = {
    "type": "String", "unique_count": 3, "null_rate": 0.0,
    "values_top": [
        {"value": "普通社区", "count": 1809},
        {"value": "优化社区", "count": 500},
        {"value": "标杆社区", "count": 32},
    ],
}


# ---------------------------------------------------------------- 长度与标签

def test_nice_length_prefers_round_values():
    assert nice_length(8345) == 10000.0     # 能跨档选到 10×10^k
    assert nice_length(1200) == 1000.0
    assert nice_length(330) == 400.0
    assert nice_length(0) == 0.0


def test_nice_division_gives_integer_steps():
    divisions, division = nice_division(10000.0)
    assert (divisions, division) == (4, 2500.0)
    divisions, division = nice_division(5000.0)
    assert division * divisions == 5000.0
    assert abs(division - round(division)) < 1e-6


def test_format_class_labels_modes_and_no_long_decimals():
    bounds = [(0.0, 183.0), (183.0, 723.0), (723.0, 1667.0), (1667.0, 2828.0), (2828.0, 14112.0)]
    semantic = format_class_labels(bounds, "semantic")
    assert semantic == ["少", "较少", "中", "较多", "多"]
    ranged = format_class_labels(bounds, "range")
    assert ranged[0] == "0 – 183"
    assert all("." not in text.split("–")[0].strip() for text in ranged)  # 上下界取整
    both = format_class_labels(bounds[:3], "both")
    assert both[0].startswith("低（") and both[0].endswith("）")
    assert "." not in ranged[0]      # 整数上下界不应出现小数点


# -------------------------------------------------------------------- 纸张朝向

def test_near_square_data_uses_portrait():
    spec = design_only([], {"theme": "老年人口"}, facts=make_facts(aspect=1.02, field_stats=CONTINUOUS))["spec"]
    assert spec["page"]["paper"] == "A4"
    assert spec["page"]["orientation"] == "portrait"
    assert spec["page"]["width_mm"] < spec["page"]["height_mm"]


def test_wide_data_uses_landscape():
    spec = design_only([], {"theme": "绿地服务价值"}, facts=make_facts(aspect=2.6, field_stats=CONTINUOUS))["spec"]
    assert spec["page"]["orientation"] == "landscape"


def test_screen_medium_prefers_wide_page():
    spec = design_only(
        [], {"theme": "汇报用图", "medium": "screen_16_9"},
        facts=make_facts(aspect=1.5, field_stats=CONTINUOUS),
    )["spec"]
    assert spec["page"]["paper"] == "16:9"
    assert spec["page"]["orientation"] == "landscape"


def test_orientation_can_be_forced_by_intent():
    spec = design_only([], {"theme": "地图", "orientation": "portrait"},
                       facts=make_facts(aspect=2.6, field_stats=CONTINUOUS))["spec"]
    assert spec["page"]["orientation"] == "portrait"


# ---------------------------------------------------------------------- 图名

def test_title_composition_fills_region_and_scale_without_duplication():
    facts = make_facts(field_stats=CONTINUOUS)
    spec = design_only([], {"theme": "老年人口", "region": "郑州市", "scale": "社区尺度"}, facts=facts)["spec"]
    assert spec["title"]["text"] == "郑州市社区尺度老年人口分布图"

    spec2 = design_only([], {"theme": "社区尺度老年人口", "scale": "社区尺度"}, facts=facts)["spec"]
    assert spec2["title"]["text"].count("社区尺度") == 1

    spec3 = design_only([], {"theme": "社区类型"}, facts=make_facts(mode="unique", field_stats=CATEGORICAL))["spec"]
    assert spec3["title"]["text"] == "社区类型分布图"   # 分类图按惯例也用"分布图"（参考图：标杆社区和待优化社区分布图）


def test_title_font_size_clamped_and_two_line_for_long_text():
    long_theme = "全国典型城市群多尺度社区人口老龄化空间格局与绿地服务耦合协调评价"
    spec = design_only([], {"theme": long_theme}, facts=make_facts(field_stats=CONTINUOUS))["spec"]
    assert 12.0 <= spec["title"]["height_pt"] <= 20.0
    assert spec["title"]["lines"] >= 2


def test_title_box_centered_in_neatline():
    spec = design_only([], {"theme": "老年人口"}, facts=make_facts(field_stats=CONTINUOUS))["spec"]
    neat, box = spec["neatline"], spec["title"]["box"]
    assert box["x"] >= neat["x"] - 1e-6
    assert box["x"] + box["w"] <= neat["x"] + neat["w"] + 1e-6
    assert abs((box["y"] + box["h"] / 2) - (neat["y"] + neat["h"] - (neat["y"] + neat["h"] - box["y"]) / 2)) < 40


# ---------------------------------------------------------------------- 图例

def test_single_symbol_map_has_no_legend():
    spec = design_only([], {"theme": "医疗服务圈"}, facts=make_facts(mode="single"))["spec"]
    assert spec["legend"]["needed"] is False
    assert any("单一符号" in d["why"] for d in spec["decisions"])


def test_graduated_map_legend_uses_theme_title_and_semantic_labels():
    spec = design_only([], {"theme": "老年人口"}, facts=make_facts(field_stats=CONTINUOUS))["spec"]
    assert spec["legend"]["needed"] is True
    assert spec["legend"]["title"] == "老年人口"
    assert spec["renderer"]["labels_mode"] == "semantic"


def test_unique_map_legend_uses_category_values():
    spec = design_only([], {"theme": "社区类型"}, facts=make_facts(mode="unique", field_stats=CATEGORICAL))["spec"]
    assert spec["renderer"]["mode"] == "unique"
    assert spec["renderer"]["labels_mode"] == "value"
    assert spec["renderer"]["class_count"] == 3
    assert "标杆社区" in spec["renderer"]["category_colors"]


def test_multi_layer_legend_has_no_title():
    spec = design_only([], {"theme": "医院点位"}, facts=make_facts(mode="multi_layer", layers=2))["spec"]
    assert spec["legend"]["needed"] is True
    assert spec["legend"]["title"] == ""
    assert any("多图层" in d["why"] for d in spec["decisions"])


def test_legend_moves_outside_when_data_covers_everything():
    facts = make_facts(
        field_stats=CONTINUOUS,
        grid=[[5, 5, 5], [5, 5, 5], [5, 5, 5]],
        count_usage={"top_left": 1.0, "top_right": 1.0, "bottom_left": 1.0, "bottom_right": 1.0},
    )
    spec = design_only([], {"theme": "人口密度"}, facts=facts)["spec"]
    assert spec["legend"]["outside"] is True
    assert spec["map_frame"]["w"] < spec["neatline"]["w"]


def test_legend_picks_emptiest_corner():
    facts = make_facts(
        field_stats=CONTINUOUS,
        grid=[[4.0, 0.2, 0.1], [0.2, 5.0, 0.2], [0.1, 0.2, 0.02]],
    )
    spec = design_only([], {"theme": "老年人口"}, facts=facts)["spec"]
    assert spec["legend"]["corner"] == "bottom_right"
    assert spec["legend"]["x"] > spec["map_frame"]["x"] + spec["map_frame"]["w"] / 2


# -------------------------------------------------------------------- 比例尺

def test_scale_bar_units_switch_and_integer_divisions():
    small = design_only([], {"theme": "小区域"}, facts=make_facts(width_m=3000.0, field_stats=CONTINUOUS))["spec"]
    assert small["scale_bar"]["unit_label"] == "米"
    assert abs(small["scale_bar"]["division_m"] - round(small["scale_bar"]["division_m"])) < 1e-6

    big = design_only([], {"theme": "大区域"}, facts=make_facts(width_m=300000.0, field_stats=CONTINUOUS))["spec"]
    assert big["scale_bar"]["unit_label"] == "千米"


def test_scale_bar_length_targets_quarter_of_frame():
    spec = design_only([], {"theme": "老年人口"}, facts=make_facts(field_stats=CONTINUOUS))["spec"]
    bar_mm = spec["scale_bar"]["bar_mm"]
    frame_w = spec["map_frame"]["w"]
    assert 0.1 * frame_w <= bar_mm <= 0.45 * frame_w


def test_scale_bar_reason_mentions_round_numbers():
    spec = design_only([], {"theme": "老年人口"}, facts=make_facts(field_stats=CONTINUOUS))["spec"]
    decisions = {d["key"]: d for d in spec["decisions"]}
    assert "整数" in decisions["scale_bar"]["why"]


# -------------------------------------------------------------------- 指北针

def test_north_arrow_present_with_reason_and_size():
    spec = design_only([], {"theme": "老年人口"}, facts=make_facts(field_stats=CONTINUOUS))["spec"]
    assert spec["north_arrow"]["needed"] is True
    assert 12.0 <= spec["north_arrow"]["size_mm"] <= 20.0
    assert any(d["key"] == "north_arrow" for d in spec["decisions"])
    # 图例不在右上时，指北针在右上角
    assert spec["north_arrow"]["x"] > spec["map_frame"]["x"] + spec["map_frame"]["w"] / 2


# ---------------------------------------------------------------- 版式完整性

def test_all_elements_inside_neatline_and_not_overlapping():
    spec = design_only([], {"theme": "老年人口"}, facts=make_facts(field_stats=CONTINUOUS))["spec"]
    neat = spec["neatline"]
    for key in ("map_frame", "legend", "scale_bar", "north_arrow"):
        box = spec[key]
        assert box["x"] >= neat["x"] - 1e-6, key
        assert box["y"] >= neat["y"] - 1e-6, key
        assert box["x"] + box["w"] <= neat["x"] + neat["w"] + 1e-6, key
        assert box["y"] + box["h"] <= neat["y"] + neat["h"] + 1e-6, key
    legend, bar = spec["legend"], spec["scale_bar"]
    overlap = not (legend["x"] + legend["w"] <= bar["x"] or bar["x"] + bar["w"] <= legend["x"]
                   or legend["y"] + legend["h"] <= bar["y"] or bar["y"] + bar["h"] <= legend["y"])
    assert not overlap


def test_describe_spec_is_human_readable():
    spec = design_only([], {"theme": "老年人口", "region": "郑州市", "scale": "社区尺度"},
                       facts=make_facts(field_stats=CONTINUOUS))["spec"]
    text = describe_spec(spec)
    assert "纸张" in text and "图名" in text and "比例尺" in text and "决策理由" in text
    assert "A4" in text


def test_overrides_can_force_decisions():
    spec = design_only(
        [], {"theme": "老年人口", "overrides": {"legend.outside": True, "title.height_pt": 16}},
        facts=make_facts(field_stats=CONTINUOUS),
    )["spec"]
    assert spec["legend"]["outside"] is True
    assert spec["title"]["height_pt"] == 16


def test_suggest_fixes_maps_qc_failures_to_hints():
    hints = suggest_fixes([
        {"name": "title_too_small", "ok": False, "suggest_pt": 18},
        {"name": "legend_covers_data", "ok": False},
        {"name": "scale_bar_too_long", "ok": False, "suggest_length_m": 5000},
        {"name": "map_frame_blank", "ok": True},
    ])
    assert hints["title_pt"] == 18.0
    assert hints["legend_outside"] is True
    assert hints["scale_bar_length_m"] == 5000.0


def test_labels_disabled_for_dense_data():
    facts = make_facts(feature_count=5000, field_stats=CONTINUOUS)
    spec = design_only([], {"theme": "老年人口"}, facts=facts)["spec"]
    assert spec["labels"]["enabled"] is False


# ------------------------------------------------------------------- 样式档位

def test_pick_style_name_respects_keyword_priority():
    catalog = {"SCALE_BAR": ["比例线 1", "公制黑白相间比例尺 1", "空心比例尺 1"]}
    assert pick_style_name(catalog, "SCALE_BAR", ["公制黑白相间", "比例线"]) == "公制黑白相间比例尺 1"
    assert pick_style_name(catalog, "SCALE_BAR", ["不存在的关键字"], fallback_index=None) == ""
    assert pick_style_name(catalog, "NORTH_ARROW", ["罗盘"]) == ""
    assert pick_style_name({}, "SCALE_BAR", ["任意"]) == ""


def test_profile_loading_and_override(tmp_path: Path):
    profiles = builtin_profiles()
    assert "competition_standard" in profiles
    standard = load_profile("competition_standard")
    assert standard["legend"]["font_pt"] == 9.0
    override = tmp_path / "profiles.json"
    override.write_text(json.dumps({"competition_standard": {"legend": {"font_pt": 8.0}}}), encoding="utf-8")
    merged = load_profile("competition_standard", override_path=override)
    assert merged["legend"]["font_pt"] == 8.0
    assert merged["legend"]["show_layer_name"] is False          # 其余字段保持默认
    fallback = load_profile("不存在的档位")
    assert fallback["name"] == "competition_standard"


# ------------------------------------------------------------------ facts 纯函数

def test_facts_pure_helpers():
    assert aspect_ratio({"width": 300.0, "height": 150.0}) == 2.0
    assert aspect_ratio({"width": 0.0, "height": 0.0}) == 1.0
    layers = [
        {"exists": True, "extent": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 5, "width": 10, "height": 5},
         "occupancy": [[1, 0, 0], [0, 0, 0], [0, 0, 4]]},
        {"exists": True, "extent": {"xmin": -5, "ymin": -5, "xmax": 5, "ymax": 5, "width": 10, "height": 10},
         "occupancy": [[0, 0, 0], [0, 2, 0], [0, 0, 0]]},
    ]
    extent = union_extent(layers)
    assert extent["width"] == 15 and extent["height"] == 10
    usage = corner_usage(layers)
    assert usage["top_left"] > usage["top_right"]
    ordered = emptiest_corners({"corner_usage": usage})
    assert ordered[0] in ("bottom_left", "bottom_right", "top_right")
    assert guess_render_mode({"layers": [{"exists": True}], "field": "V",
                              "layers_placeholder": None}) in ("single", "graduated", "unique")


def test_normalize_facts_derives_aspect_and_usage():
    facts = make_facts(aspect=2.0, field_stats=CONTINUOUS)
    assert facts["aspect"] == 2.0
    assert set(facts["corner_usage"]) == {"top_left", "top_right", "bottom_left", "bottom_right"}
    assert facts["primary"]["feature_count"] == 2341


# ------------------------------------------------------------------ 渲染层工具

def test_render_helpers_parse_colors():
    from gis_cli.cartography.apply import _hex_to_rgb, _parse_color_map

    assert _hex_to_rgb("#1F77B4") == [31, 119, 180, 100]
    assert _hex_to_rgb("2CA02C") == [44, 160, 44, 100]
    mapping = _parse_color_map("标杆社区:#1F77B4;需整改社区:#2CA02C")
    assert mapping == {"标杆社区": "#1F77B4", "需整改社区": "#2CA02C"}


def test_render_module_uses_cim_height_not_fontsize():
    """回归防护：CIM 文本字号属性是 height，旧代码用 fontSize 被静默忽略（图名一直是小字）。"""
    source = Path("src/gis_cli/cartography/apply.py").read_text(encoding="utf-8")
    assert "symbol.height = float(height_pt)" in source
    assert "fontSize =" not in source      # 不能再用 fontSize 赋值（会被静默忽略）


def test_layout_math_is_consistent():
    spec = design_only([], {"theme": "老年人口"}, facts=make_facts(field_stats=CONTINUOUS))["spec"]
    page, neat = spec["page"], spec["neatline"]
    assert math.isclose(neat["w"], page["width_mm"] - 2 * neat["x"], abs_tol=1e-6)
    assert spec["title"]["height_pt"] > 0
    assert spec["map_frame"]["h"] > 0


def test_publish_creates_missing_output_directory(tmp_path):
    """目标目录不存在时必须自己建，不能当成“文件被 ArcGIS Pro 占用”。

    实测组合任务：模型把图写到 ``output/成果图/`` 而该目录还不存在，
    ``copy2`` 抛 FileNotFoundError（WinError 3），被 ``_publish`` 重试 4 次后
    报成「该文件可能正在 ArcGIS Pro 中打开」——连续 4 次 create_map 全失败，
    最后模型自己 os.makedirs 才出图。
    """
    from gis_cli.cartography import _publish

    source = tmp_path / "src.jpg"
    source.write_bytes(b"jpg")
    target = tmp_path / "output" / "成果图" / "密度_1982_1986.jpg"

    assert _publish(str(source), str(target)) == str(target)
    assert target.read_bytes() == b"jpg"


def test_unique_class_labels_drop_raw_floats_and_match_legend_entries():
    """回归：unique 模式的类名不能把字段原始浮点写进图例。

    实测（DAIR-32 基准 4a）：``create_map(mode="unique", field="FROM_YEAR")`` 的
    唯一值渲染器把 ``DIST_M`` 的原始浮点当类名，交付的 .aprx 图例里是
    ``193833.858800001``，``map_layout_ok`` 断言（legend_labels_unrounded）判不通过。
    """
    from gis_cli.cartography.apply import _apply_unique

    class _Symbol:
        color = None

    class _Item:
        def __init__(self, value):
            self.values = [(value, str(value))]
            self.label = str(value)
            self.symbol = _Symbol()

    class _Group:
        def __init__(self, items):
            self.items = items

    class _Renderer:
        def __init__(self, items):
            self.groups = [_Group(items)]
            self.fields = None

    class _Symbology:
        def __init__(self, items):
            self.renderer = _Renderer(items)

        def updateRenderer(self, name):
            self.updated = name

    class _Layer:
        def __init__(self, items):
            self.symbology = _Symbology(items)

    layer = _Layer([_Item(193833.858800001), _Item(1982), _Item("水田")])
    spec = {"renderer": {
        "mode": "unique", "field": "FROM_YEAR", "nodata_color": "#E8E8E8",
        "category_colors": "193833.858800001:#E8E8E8", "legend_entries": [],
    }}
    notes: list[str] = []
    _apply_unique(layer, spec, notes)

    items = layer.symbology.renderer.groups[0].items
    labels = [item.label for item in items]
    assert labels == ["193,834", "1982", "水田"]
    # 类名与图例条目走同一个显示文本（不然 ArcGIS 原生图例与叠加重绘又是两个结论）
    assert [entry["label"] for entry in spec["renderer"]["legend_entries"]] == labels
    # 原始浮点不许残留在图例条目 / 类名 / notes 里
    assert "193833.858800001" not in json.dumps(spec["renderer"]["legend_entries"], ensure_ascii=False)
    assert "193833.858800001" not in " ".join(notes)
    # 配色仍按**原始值**查表（模型给的 category_colors 是按原始值写的）
    assert items[0].symbol.color == {"RGB": [232, 232, 232, 100]}


def test_unique_class_labels_respect_category_names():
    """给了 category_names 就用它（编码→名称），不再走数值格式化。"""
    from gis_cli.cartography.apply import _apply_unique

    class _Symbol:
        color = None

    class _Item:
        def __init__(self, value):
            self.values = [(value, str(value))]
            self.label = str(value)
            self.symbol = _Symbol()

    class _Group:
        def __init__(self, items):
            self.items = items

    class _Renderer:
        def __init__(self, items):
            self.groups = [_Group(items)]
            self.fields = None

    class _Symbology:
        def __init__(self, items):
            self.renderer = _Renderer(items)

        def updateRenderer(self, name):
            self.updated = name

    class _Layer:
        def __init__(self, items):
            self.symbology = _Symbology(items)

    layer = _Layer([_Item(1), _Item(2)])
    spec = {"renderer": {
        "mode": "unique", "field": "LC", "category_names": "1:水域;2:林地",
        "category_colors": "1:#1F77B4;2:#2CA02C", "legend_entries": [],
    }}
    _apply_unique(layer, spec, [])
    assert [item.label for item in layer.symbology.renderer.groups[0].items] == ["水域", "林地"]
    assert [entry["label"] for entry in spec["renderer"]["legend_entries"]] == ["水域", "林地"]


def test_class_label_keeps_codes_and_rounds_measurements():
    """类名格式化：编码/年份不动，测量值去噪声（不能把 110101 取成 110,000）。"""
    from gis_cli.cartography.apply import _category_label

    assert _category_label("110101") == "110101"
    assert _category_label("1982") == "1982"
    assert _category_label("193833.858800001") == "193,834"
    assert _category_label("0.123456") == "0.12"
    assert _category_label("1.5") == "1.5"
    assert _category_label("水田") == "水田"
    assert _category_label("") == ""


def test_qc_categorical_labels_only_reject_raw_floats():
    """分类图例（unique）只判“有没有原始浮点”，不判“是不是整齐数值”。

    DAIR-32 实测：图例里 ``193833.858800001`` 必须判负；而编码 110101、取值 1.2
    这类**合法的类别名**不能被“整齐数值”口径误杀（否则体检不阻断后会卡死交付）。
    """
    from gis_cli.cartography.qc import _categorical_labels_detail

    ok, detail = _categorical_labels_detail(["193833.858800001"])
    assert ok is False and "193833.858800001" in detail
    for labels in (["1982", "1993"], ["110101", "110102"], ["1.2", "2.5"], ["水域", "林地"]):
        assert _categorical_labels_detail(labels)[0] is True, labels

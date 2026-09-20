# -*- coding: utf-8 -*-
"""风格档位（style profile）：把"一套制图习惯"参数化，供规则引擎取默认值。

档位来源：用户手绘的 17 张竞赛成果图（A4 竖版、图名顶部居中约 20pt 加粗宋体、
图例右下无边框、比例尺左下黑白交替分段带整数刻度并以「米」为单位、
指北针右上角八芒罗盘玫瑰、数据居中给四角留白）。实测确认的样式名：
- 指北针：``ArcGIS 指北针 13``（黑八芒罗盘玫瑰，含 N/E/S/W）
- 比例尺：``公制黑白相间比例尺 1``（公制、黑白交替分段，可做整数刻度）

``config/cartography_profiles.json`` 是**可选覆盖**（只需要写想改的字段），
不要把整份档位复制进去——避免两份默认值不同步。
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OVERRIDE_PATH = REPO_ROOT / "config" / "cartography_profiles.json"

_BUILTIN: dict[str, dict[str, Any]] = {
    "competition_standard": {
        "label": "竞赛标准（A4 竖版，宋体大标题，图例右下）",
        "medium": "print",
        "preferred_papers": ["A4", "A3"],
        "prefer_orientation": "auto",
        "outer_margin_mm": 4.0,
        "inner_margin_mm": 6.0,
        "dpi": 250,
        "extent_padding": 0.08,
        "min_font_pt": 7.0,
        "title": {
            "font": "宋体",
            "bold": True,
            "min_pt": 12.0,
            "max_pt": 20.0,
            "max_chars_per_line": 22,
            "band_gap_mm": 3.0,
            "two_line_factor": 1.25,
        },
        "legend": {
            "border": False,
            "show_layer_name": False,
            "font_pt": 9.0,
            "title_pt": 11.0,
            "swatch_mm": [7.0, 5.0],
            "row_factor": 1.5,
            "pad_mm": 2.5,
            "corner_order": ["bottom_right", "bottom_left", "top_right", "top_left"],
            "frame_margin_mm": 6.0,
            "dense_threshold": 0.5,
            "allow_outside": True,
            "labels_mode": "semantic",
        },
        "scale_bar": {
            "style_keywords": [
                "公制黑白相间比例尺 1",
                "公制黑白相间",
                "黑白相间",
                "alternating",
                "公制比例线",
                "比例线",
            ],
            "unit_label": "米",
            "target_width_frac": 0.24,
            "font_pt": 8.0,
            "label_gap_mm": 1.2,
            "position": "bottom_left",
        },
        "north_arrow": {
            "style_keywords": ["指北针 13", "罗盘北", "正北", "罗盘", "compass", "north arrow", "指北针"],
            "simple_keywords": ["简单实心指北针", "简单空心指北针", "ArcGIS 指北针 1", "arrow", "箭头"],
            "size_frac": 0.09,
            "min_mm": 18.0,
            "max_mm": 28.0,
            "position": "top_right",
            "needed": "auto",
        },
        "colors": {
            "continuous_keywords": {
                "人口": "YlOrRd",
                "老年": "YlOrRd",
                "热": "YlOrRd",
                "绿地": "Greens",
                "植被": "Greens",
                "水": "Blues",
                "密度": "Purples",
                "可达": "RdYlGn",
                "指数": "Greens",
                "高度": "Oranges",
            },
            "default_ramp": "YlOrRd",
            "diverging_ramps": ["RdBu", "RdYlGn"],
            "categorical": ["#7FB2E5", "#2E8B2E", "#E5A33C", "#B07FB2", "#E06666", "#8C8C8C"],
            "nodata_color": "#E8E8E8",
        },
        "labels": {"enable_below_features": 80, "density_threshold": 0.18},
    },
    "screen_report": {
        "label": "屏幕汇报（16:9 横版，字号放大）",
        "medium": "screen",
        "preferred_papers": ["16:9"],
        "prefer_orientation": "landscape",
        "outer_margin_mm": 3.0,
        "inner_margin_mm": 5.0,
        "dpi": 150,
        "extent_padding": 0.06,
        "min_font_pt": 10.0,
        "title": {
            "font": "微软雅黑",
            "bold": True,
            "min_pt": 16.0,
            "max_pt": 28.0,
            "max_chars_per_line": 26,
            "band_gap_mm": 4.0,
            "two_line_factor": 1.25,
        },
        "legend": {
            "border": False,
            "show_layer_name": False,
            "font_pt": 11.0,
            "title_pt": 13.0,
            "swatch_mm": [8.0, 6.0],
            "row_factor": 1.5,
            "pad_mm": 3.0,
            "corner_order": ["top_right", "bottom_right", "top_left", "bottom_left"],
            "dense_threshold": 0.7,
            "allow_outside": True,
            "labels_mode": "semantic",
        },
        "scale_bar": {
            "style_keywords": ["公制黑白相间比例尺 1", "公制黑白相间", "黑白相间", "alternating"],
            "unit_label": "米",
            "target_width_frac": 0.2,
            "font_pt": 10.0,
            "label_gap_mm": 1.5,
            "position": "bottom_left",
        },
        "north_arrow": {
            "style_keywords": ["指北针 13", "罗盘北", "正北", "罗盘", "compass"],
            "simple_keywords": ["简单实心指北针", "箭头"],
            "size_frac": 0.07,
            "min_mm": 18.0,
            "max_mm": 30.0,
            "position": "top_right",
            "needed": "auto",
        },
        "colors": {},
        "labels": {"enable_below_features": 40, "density_threshold": 0.15},
    },
    "dense_raster": {
        "label": "满覆盖栅格（图例强制外置，保证地图看得见）",
        "medium": "print",
        "preferred_papers": ["A4", "A3"],
        "prefer_orientation": "auto",
        "outer_margin_mm": 4.0,
        "inner_margin_mm": 6.0,
        "dpi": 250,
        "extent_padding": 0.03,
        "min_font_pt": 7.0,
        "title": {"font": "宋体", "bold": True, "min_pt": 12.0, "max_pt": 20.0, "max_chars_per_line": 22,
                  "band_gap_mm": 3.0, "two_line_factor": 1.25},
        "legend": {"border": False, "show_layer_name": False, "font_pt": 9.0, "title_pt": 11.0,
                   "swatch_mm": [7.0, 5.0], "row_factor": 1.5, "pad_mm": 2.5,
                   "corner_order": ["bottom_right", "top_right", "bottom_left", "top_left"],
                   "dense_threshold": 0.2, "allow_outside": True, "labels_mode": "semantic"},
        "scale_bar": {"style_keywords": ["公制黑白相间比例尺 1", "公制黑白相间", "黑白相间", "alternating"],
                      "unit_label": "米", "target_width_frac": 0.24, "font_pt": 8.0,
                      "label_gap_mm": 1.2, "position": "bottom_left"},
        "north_arrow": {"style_keywords": ["指北针 13", "罗盘北", "正北", "罗盘", "compass"],
                        "simple_keywords": ["简单实心指北针", "箭头"], "size_frac": 0.08,
                        "min_mm": 18.0, "max_mm": 26.0, "position": "top_right", "needed": "auto"},
        "colors": {},
        "labels": {"enable_below_features": 0, "density_threshold": 0.05},
    },
}

DEFAULT_PROFILE_NAME = "competition_standard"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def builtin_profiles() -> dict[str, dict[str, Any]]:
    return copy.deepcopy(_BUILTIN)


def load_profile(name: str | None = None, *, override_path: str | Path | None = None) -> dict[str, Any]:
    """取档位（内置 + 用户覆盖）。未知名字回退默认档并保留 name 字段便于排查。"""
    profile_name = (name or DEFAULT_PROFILE_NAME).strip() or DEFAULT_PROFILE_NAME
    base = _BUILTIN.get(profile_name)
    if base is None:
        logger.warning("未知风格档位 %s，回退 %s", profile_name, DEFAULT_PROFILE_NAME)
        profile_name, base = DEFAULT_PROFILE_NAME, _BUILTIN[DEFAULT_PROFILE_NAME]
    profile = copy.deepcopy(base)
    profile["name"] = profile_name

    path = Path(override_path) if override_path else OVERRIDE_PATH
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8")) or {}
            section = data.get(profile_name) or {}
            profile = _deep_merge(profile, section)
        except Exception as exc:
            logger.warning("风格档位覆盖文件读取失败: %s", exc)
    return profile


def profile_names() -> list[str]:
    return sorted(_BUILTIN)


__all__ = ["DEFAULT_PROFILE_NAME", "builtin_profiles", "load_profile", "profile_names", "OVERRIDE_PATH"]
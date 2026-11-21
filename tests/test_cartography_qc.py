# -*- coding: utf-8 -*-
"""版面体检的判定口径与文案单测（纯 Python，不需要 ArcGIS）。

对应 DAIR-61 的两条：

- B3 ``title_too_small``：工程里读不到图名字号时**不能算通过**，要标 ``unmeasured``；
- B6 ``legend_covers_data``：通过/不通过两条 detail 的阈值方向要读得通（占比高才是压住数据）。
"""
from __future__ import annotations

from gis_cli.cartography.qc import _legend_cover_check, _title_size_check


# ------------------------------------------------------- B3 图名字号不可评测

def test_title_size_unreadable_is_marked_unmeasured_not_passed():
    check = _title_size_check(None, 12.0)
    assert check["name"] == "title_too_small"
    assert check["unmeasured"] is True
    assert check["ok"] is True                     # 不是缺陷：不挡交付闸门
    assert "不可评测" in check["detail"]
    assert "suggest_pt" not in check


def test_title_size_small_title_fails_with_suggestion():
    check = _title_size_check(8.0, 12.0)
    assert check["ok"] is False
    assert not check.get("unmeasured")
    assert "8.0pt" in check["detail"] and "12pt" in check["detail"]
    assert check["suggest_pt"] == 18.0


def test_title_size_big_enough_passes():
    check = _title_size_check(18.0, 12.0)
    assert check["ok"] is True
    assert not check.get("unmeasured")


# --------------------------------------------------- B6 图例压数据的文案方向

def test_legend_cover_detail_wording_for_pass():
    check = _legend_cover_check(0.18)
    assert check["name"] == "legend_covers_data"
    assert check["ok"] is True
    assert "0.18" in check["detail"]
    assert "未压住数据" in check["detail"]
    assert "超过" not in check["detail"]


def test_legend_cover_detail_wording_for_fail():
    check = _legend_cover_check(0.72)
    assert check["ok"] is False
    assert "0.72" in check["detail"]
    assert "判为压住数据" in check["detail"]
    assert "超过阈值 0.45" in check["detail"]


def test_legend_cover_threshold_is_exclusive_above():
    """阈值 0.45 本身算通过（>0.45 才算压住），与旧判定逻辑一致。"""
    assert _legend_cover_check(0.45)["ok"] is True
    assert _legend_cover_check(0.46)["ok"] is False

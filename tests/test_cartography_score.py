# -*- coding: utf-8 -*-
"""成果质量分单测（纯 Python，不需要 ArcGIS）。

覆盖 P1-3 的验收口径：

1. 扣分单调：同一缺陷重复叠加分数单调不增；无缺陷时 ``total == 100``；
2. 每类缺陷各一条用例（qc 现有检查名全覆盖），并断言"该维度扣分 + 其他维度不动"；
3. 退化输入不抛异常（缺 ``checks`` / 未知检查名 / 元素缺 ``name`` / 根本不是 dict）；
4. 纯函数：``score.py`` 不 import arcpy、不 import 重包、不碰 qc.py。

为什么值得这些用例：分数是给模型与用户看的**信息**，一旦算错（比如把"读不到图例"
也扣分，或对同一缺陷重复计权），模型会为了分数去改本来没问题的版式。
"""
from __future__ import annotations

import ast
import itertools
import json
from pathlib import Path

import pytest

from gis_cli.cartography.score import (
    CORRECTNESS,
    DEDUCTIONS,
    DIMENSIONS,
    NORMATIVITY,
    READABILITY,
    Deduction,
    QualityScore,
    brief_quality,
    score_layout,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "gis_cli"
SCORE_FILE = SRC / "cartography" / "score.py"

#: 本单要求覆盖的 qc 检查名（分维度列出，与单据给的归属表一致）。
REQUIRED = {
    NORMATIVITY: [
        "title_too_small",
        "title_overflow",
        "title_not_centered",
        "legend_labels_unrounded",
        "scale_bar_not_round",
        "font_too_small",
        "element_outside_neatline",
    ],
    READABILITY: ["legend_covers_data", "elements_overlap", "map_frame_blank"],
}

#: qc.py 里"环境读不到就跳过"的检查名：结论不可判定，不该进分数。
UNSCORED_QC_NAMES = [
    "legend_labels_readable",
    "legend_readable",
    "legend_position_sane",
    "legend_outside_frame",
    "double_border",
    "aprx_exists",
    "layout_exists",
]

ALL_DEFECTS = [name for names in REQUIRED.values() for name in names]


def qc_with(*checks: dict) -> dict:
    return {"ok": all(bool(item.get("ok")) for item in checks), "checks": list(checks)}


def failed(name: str, detail: str = "") -> dict:
    return {"name": name, "ok": False, "detail": detail or f"{name} 未通过"}


def dimension_of(score: QualityScore, dimension: str) -> float:
    return float(getattr(score, dimension))


# --------------------------------------------------------------- 无缺陷时满分

def test_clean_layout_scores_full_marks():
    """无缺陷时 total == 100，三维度都是 100，明细为空。"""
    score = score_layout({"ok": True, "checks": [{"name": "title_too_small", "ok": True}]})
    assert score.total == 100.0
    assert (score.normativity, score.readability, score.correctness) == (100.0, 100.0, 100.0)
    assert score.deductions == []


# ------------------------------------------- 每类缺陷：本维度扣分、其他维度不动

@pytest.mark.parametrize("name", ALL_DEFECTS)
def test_each_defect_deducts_only_its_own_dimension(name: str):
    """每条 qc 检查各扣自己那个维度，其他两个维度保持 100。"""
    points, dimension = DEDUCTIONS[name]
    score = score_layout(qc_with(failed(name, f"实测：{name} 未通过")))

    assert score.total == pytest.approx(100.0 - points, abs=0.05)
    assert dimension_of(score, dimension) == pytest.approx(100.0 - points, abs=0.05)
    for other in DIMENSIONS:
        if other != dimension:
            assert dimension_of(score, other) == 100.0, f"{name} 不该动 {other}"

    assert len(score.deductions) == 1
    item = score.deductions[0]
    assert isinstance(item, Deduction)
    assert (item.name, item.points, item.dimension) == (name, points, dimension)
    assert item.reason == f"实测：{name} 未通过"


def test_deduction_table_covers_every_required_qc_check():
    """扣分表必须恰好覆盖单据点名的检查名（≥8 条），维度归属一致。"""
    assert set(DEDUCTIONS) == set(ALL_DEFECTS)
    assert len(DEDUCTIONS) >= 8
    for dimension, names in REQUIRED.items():
        for name in names:
            assert DEDUCTIONS[name][1] == dimension
    # 权重都在 (0, 100] 且总和不超过 100：单张图扣光分数必须是"多重缺陷"的结果
    assert all(0 < points <= 100 for points, _ in DEDUCTIONS.values())
    for dimension in DIMENSIONS:
        assert sum(points for points, dim in DEDUCTIONS.values() if dim == dimension) <= 100


# --------------------------------------------------------- 扣分单调（属性式）

@pytest.mark.parametrize("name", ALL_DEFECTS)
def test_repeated_defect_never_raises_score(name: str):
    """同一缺陷重复叠加：分数单调不增，且正好扣 n 份（扣到 0 为止）。"""
    points, _ = DEDUCTIONS[name]
    totals = []
    for repeat in range(6):
        score = score_layout(qc_with(*[failed(name) for _ in range(repeat)]))
        totals.append(score.total)
        assert len(score.deductions) == repeat
        assert score.total == pytest.approx(max(0.0, 100.0 - points * repeat), abs=0.05)
    assert totals == sorted(totals, reverse=True)


def test_more_distinct_defects_never_score_higher():
    """任意两种缺陷组合都不比只留其中一种更高（覆盖所有两两组合）。"""
    for first, second in itertools.combinations(ALL_DEFECTS, 2):
        one = score_layout(qc_with(failed(first))).total
        two = score_layout(qc_with(failed(first), failed(second))).total
        assert two <= one, f"{first} + {second} 反而比只有 {first} 高"
        assert two == pytest.approx(one - DEDUCTIONS[second][0], abs=0.05)


def test_score_floor_is_zero():
    """全类缺陷叠加扣分超过 100 时分数夹在 0，不出现负数。"""
    checks = [failed(name) for name in ALL_DEFECTS] * 3
    score = score_layout(qc_with(*checks))
    assert score.total == 0.0
    assert (score.normativity, score.readability) == (0.0, 0.0)


# ------------------------------------------------------------ 正确性 = 断言通过率

def test_correctness_is_assertion_pass_rate():
    """正确性维度就是断言通过率：4 条断言挂 1 条 → 75 分。"""
    assertions = [
        {"name": "总量守恒", "ok": True},
        {"name": "坐标系一致", "ok": True},
        {"name": "面积不为负", "ok": False, "detail": "3 个面面积为负"},
        {"name": "字段齐全", "ok": True},
    ]
    score = score_layout({"ok": True, "checks": []}, assertions=assertions)
    assert score.correctness == pytest.approx(75.0)
    assert score.total == pytest.approx(75.0)
    assert (score.normativity, score.readability) == (100.0, 100.0)
    assert [item.name for item in score.deductions] == ["面积不为负"]
    assert score.deductions[0].dimension == CORRECTNESS
    assert score.deductions[0].points == pytest.approx(25.0)


@pytest.mark.parametrize("assertions", [None, []])
def test_no_assertions_leaves_correctness_at_full_marks(assertions):
    """没有断言时正确性给**满分**（docstring 里明写的口径）：缺证据不等于错。"""
    score = score_layout({"ok": True, "checks": []}, assertions=assertions)
    assert score.correctness == 100.0
    assert score.total == 100.0
    assert score.deductions == []


def test_all_assertions_failing_zeroes_correctness():
    score = score_layout(
        {"ok": True, "checks": []},
        assertions=[{"name": "a", "ok": False}, {"name": "b", "ok": False}],
    )
    assert score.correctness == 0.0
    assert score.total == 0.0


def test_missing_assertion_ok_counts_as_failure():
    """断言条目缺 ``ok`` 时按未通过算（不能因为字段漏了就默默满分）。"""
    score = score_layout({"ok": True, "checks": []}, assertions=[{"name": "漏了 ok"}])
    assert score.correctness == 0.0


def test_dimensions_combine_without_leaking():
    """跨维度同时扣分：各维度各扣各的，总分是全部扣分之和。"""
    score = score_layout(
        qc_with(failed("title_not_centered"), failed("elements_overlap")),
        assertions=[{"name": "a", "ok": False}, {"name": "b", "ok": True}],
    )
    assert score.normativity == pytest.approx(96.0)
    assert score.readability == pytest.approx(92.0)
    assert score.correctness == pytest.approx(50.0)
    assert score.total == pytest.approx(100.0 - 4.0 - 8.0 - 50.0)


# --------------------------------------------------- 读不到量（不可评测）

def test_unmeasured_check_is_listed_not_counted_as_pass():
    """qc 读不到量的项（``unmeasured``）：不扣分，但必须列出来——那是"没测"，不是"满分"。"""
    score = score_layout({"ok": True, "checks": [
        {"name": "title_too_small", "ok": True, "unmeasured": True,
         "detail": "图名字号读不到——该维度不可评测"},
        {"name": "title_not_centered", "ok": True, "detail": "图名中心偏离 0.0mm"},
    ]})
    assert score.total == 100.0
    assert score.deductions == []
    assert [item.name for item in score.unmeasured] == ["title_too_small"]
    item = score.unmeasured[0]
    assert (item.points, item.dimension) == (DEDUCTIONS["title_too_small"][0], NORMATIVITY)
    assert "不可评测" in item.reason


def test_unmeasured_survives_to_dict_and_brief():
    """不可评测标志要能走到报告与 trace：读分数的人不会把它当成满分证据。"""
    score = score_layout({"ok": True, "checks": [
        {"name": "title_too_small", "ok": True, "unmeasured": True, "detail": "读不到"},
    ]})
    payload = json.loads(json.dumps(score.to_dict(), ensure_ascii=False))
    assert payload["unmeasured"][0]["name"] == "title_too_small"
    assert payload["unmeasured"][0]["points"] == DEDUCTIONS["title_too_small"][0]
    assert brief_quality(payload)["unmeasured"] == ["title_too_small"]


def test_unmeasured_does_not_swallow_other_defects():
    score = score_layout(qc_with(
        {"name": "title_too_small", "ok": True, "unmeasured": True},
        failed("scale_bar_not_round", "刻度不整齐"),
    ))
    assert score.total == pytest.approx(100.0 - DEDUCTIONS["scale_bar_not_round"][0])
    assert [item.name for item in score.deductions] == ["scale_bar_not_round"]
    assert [item.name for item in score.unmeasured] == ["title_too_small"]


def test_unknown_check_name_unmeasured_is_ignored():
    """不认识的名字照样不扣分、也不进 unmeasured（不猜维度）。"""
    score = score_layout(qc_with({"name": "legend_readable", "ok": True, "unmeasured": True}))
    assert score.total == 100.0
    assert score.unmeasured == []


# ----------------------------------------------------------------- 退化输入

@pytest.mark.parametrize(
    "qc",
    [
        {"ok": True, "checks": []},
        {},                                   # 缺 checks 键
        {"ok": True},                         # 只有 ok
        {"checks": None},                     # checks 不是 list
        {"checks": {"name": "title_too_small", "ok": False}},   # dict 而不是 list
        {"checks": [{}]},                     # 元素缺 name
        {"checks": [{"ok": False}]},          # 有 ok 没 name
        {"checks": ["junk", None, 42, ["x"]]},  # 元素根本不是 dict
        {"checks": []},
        None,                                 # 整个 qc 都没有
        "not a dict",                         # 类型完全不对
    ],
)
def test_degenerate_inputs_do_not_raise(qc):
    """退化输入一律不抛异常：读不懂的东西不扣分（也就不会误伤分数）。"""
    score = score_layout(qc)
    assert score.total == 100.0
    assert score.deductions == []


@pytest.mark.parametrize("name", UNSCORED_QC_NAMES)
def test_unscored_qc_checks_are_ignored(name: str):
    """qc.py 里"环境读不到就跳过"的项不参与打分（不算做错了）。"""
    assert name not in DEDUCTIONS
    score = score_layout(qc_with(failed(name)))
    assert score.total == 100.0 and score.deductions == []


def test_unknown_check_name_is_ignored_not_guessed():
    score = score_layout(qc_with(failed("some_future_check")))
    assert score.total == 100.0 and score.deductions == []


# ------------------------------------------------------------------- 输出形状

def test_to_dict_is_json_ready():
    score = score_layout(
        {**qc_with(failed("legend_covers_data", "图例框内非白像素占比 0.72")),
         "aprx": "x.aprx"},
        assertions=[{"name": "总量守恒", "ok": False, "detail": "差 3"}, {"name": "字段齐全", "ok": True}],
    )
    payload = json.loads(json.dumps(score.to_dict(), ensure_ascii=False))
    assert set(payload) == {"total", "normativity", "readability", "correctness",
                            "deductions", "unmeasured"}
    assert set(payload["deductions"][0]) == {"name", "points", "dimension", "reason"}
    assert payload["total"] == pytest.approx(100.0 - 10.0 - 50.0)
    assert payload["deductions"][0]["reason"] == "图例框内非白像素占比 0.72"


# --------------------------------------------------------------- 纯函数边界

def test_score_module_is_pure():
    """``score.py`` 只能 import 标准库：不碰 arcpy / 重包 / 本包的 qc。"""
    tree = ast.parse(SCORE_FILE.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            roots.add(node.module.split(".")[0])

    banned = {"arcpy", "PIL", "numpy", "gis_cli", "os", "pathlib", "requests", "yaml"}
    assert not (roots & banned), f"score.py 混进了不该有的依赖: {sorted(roots & banned)}"
    assert roots <= {"__future__", "dataclasses", "typing"}, sorted(roots)


def test_score_module_does_not_touch_files_or_network():
    """纯函数：不读写文件、不联网（只看 AST 里的 **调用**，docstring 里提到 arcpy 不算）。"""
    tree = ast.parse(SCORE_FILE.read_text(encoding="utf-8"))
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "open" not in calls, "score.py 不能读写文件"
    called_attrs = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not (called_attrs & {"urlopen", "urlretrieve"}), f"score.py 不能联网: {called_attrs}"


# --------------------------------------------------------------- trace 紧凑态

def test_brief_quality_keeps_totals_and_deduction_names_only():
    """trace 里只留 total + 三维度 + 扣分项名字（reason 全文不进 trace）。"""
    score = score_layout(
        {"ok": False, "checks": [{"name": "legend_covers_data", "ok": False, "detail": "非白像素 0.72"}]},
        assertions=[{"name": "aprx_map_check", "ok": True}],
    )
    brief = brief_quality(score.to_dict())
    assert set(brief) == {"total", "normativity", "readability", "correctness",
                          "deductions", "unmeasured"}
    assert brief["unmeasured"] == []
    assert brief["total"] == 90.0 and brief["readability"] == 90.0
    assert brief["deductions"] == ["legend_covers_data"]
    assert "0.72" not in json.dumps(brief, ensure_ascii=False)


def test_brief_quality_returns_none_without_score():
    """工具没给质量分时不能凭空造一个满分的 brief（没测 ≠ 满分）。"""
    assert brief_quality(None) is None
    assert brief_quality("100") is None

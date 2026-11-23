# -*- coding: utf-8 -*-
"""成果质量分（quality score）：把版面体检的"是/否"折算成 0–100 的可解释分数。

**只做打分，不做判定**：入参就是 :func:`qc.check_layout` 的返回值，判定逻辑、
阈值、修法全在 ``qc.py`` / ``design.py`` 里。本模块不新增任何判据、不读文件、
不 import arcpy、不联网，因此可以在没有 ArcGIS 的机器上单测。

分数是**信息而不是闸门**：``create_map`` / ``check_map_project`` 的 ``ok`` 语义、
``finish`` 前的 ``map_layout`` 体检都不看它——"敢不敢发"仍由现有体检决定。

口径（三条，都能单测）：

1. 总分 = 100 − Σ扣分（下限 0）；每个维度 = 100 − 该维度扣分（下限 0）。
   同一缺陷重复出现就重复扣，所以分数对缺陷数**单调不增**。
2. 扣分表 :data:`DEDUCTIONS` 是模块级常量：一项一档，权重与维度归属都写在表里，
   不在 if 里藏逻辑。
3. ``checks`` 里**不认识**的检查名一律忽略（不猜维度、不扣分）。qc.py 还会产出
   ``legend_labels_readable`` / ``double_border`` / ``legend_outside_frame`` 这类
   "环境读不到就跳过"的项，硬算进分数等于把"读不到"当成"做错了"。
4. 但"读不到"也**不能当满分**：qc 把读不到量的项标成 ``{"unmeasured": True}``
   （目前只有 ``title_too_small``——工程里读不到图名字号）。这类项不扣分，但记进
   :attr:`QualityScore.unmeasured`（名字 + 该项分值 + 维度 + 原因）。所以
   ``normativity=100`` 的含义是"**测到的**项没有缺陷"；要确认某维度真的满分，
   得同时看 ``unmeasured`` 为空——否则这 8 分就是永远悬空的假满分。

正确性维度只有一个输入：``assertions`` 的通过率（每条失败扣 ``100/N``）。
**没有断言时该维度给满分 100**，理由：分数用于解释"已知证据里有多少问题"，
没有断言是"没有可核验的证据"，不是"断言失败"；留空（0 分或 null）会让
"没跑断言"看起来比"跑了但全错"更糟，反而诱导模型为了分数去少写断言。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: 三个维度的字段名（也是 :class:`QualityScore` 的字段名与扣分表的归属值）。
NORMATIVITY = "normativity"
READABILITY = "readability"
CORRECTNESS = "correctness"
DIMENSIONS = (NORMATIVITY, READABILITY, CORRECTNESS)

#: 扣分表：``qc.check_layout`` 的检查项 → (扣分, 归属维度)。
#:
#: 权重按"这条缺陷有多挡数据/多不像专业图"排：
#: - 图面空白 20：等于没有成果；
#: - 图例压数据 / 元素越出图廓 10：直接挡掉图面信息；
#: - 图名过小、图例未取整、比例尺非整数刻度、四要素压盖 8：一眼就不专业；
#: - 图名溢出、字号过小 6：可读但明显欠打磨；
#: - 图名不居中 4：观感问题。
#:
#: 维度归属只认 qc.py 现有的检查名，不新造判定。表里没有的名字（包括 qc.py
#: 那些"环境不可判定"的项）不扣分，见模块 docstring 第 3 条。
DEDUCTIONS: dict[str, tuple[float, str]] = {
    # 规范性：图名 / 图例 / 比例尺 / 字号 / 元素位置
    "title_too_small": (8.0, NORMATIVITY),
    "title_overflow": (6.0, NORMATIVITY),
    "title_not_centered": (4.0, NORMATIVITY),
    "legend_labels_unrounded": (8.0, NORMATIVITY),
    "scale_bar_not_round": (8.0, NORMATIVITY),
    "font_too_small": (6.0, NORMATIVITY),
    "element_outside_neatline": (10.0, NORMATIVITY),
    # 可读性：图面是否被压住 / 是否根本没画上东西
    "legend_covers_data": (10.0, READABILITY),
    "elements_overlap": (8.0, READABILITY),
    "map_frame_blank": (20.0, READABILITY),
}

#: 满分。断言失败按条数均分这 100 分，维度分也从它往下扣。
_FULL_SCORE = 100.0


@dataclass(frozen=True)
class Deduction:
    """一条扣分：检查项名 + 扣多少分 + 归属维度 + 人读的理由（qc 的 detail）。"""

    name: str
    points: float
    dimension: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "points": self.points,
            "dimension": self.dimension,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class QualityScore:
    """0–100 总分 + 三维度分 + 扣分明细（明细加起来就是总分的减法）。"""

    total: float
    normativity: float
    readability: float
    correctness: float
    deductions: list[Deduction] = field(default_factory=list)
    unmeasured: list[Deduction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "normativity": self.normativity,
            "readability": self.readability,
            "correctness": self.correctness,
            "deductions": [item.to_dict() for item in self.deductions],
            "unmeasured": [item.to_dict() for item in self.unmeasured],
        }


def _clamp(value: float) -> float:
    """夹到 0–100，并保留 1 位小数（避免 66.66666 这类浮点噪声进报告）。"""
    return round(max(0.0, min(100.0, value)), 1)


def score_layout(
    qc: Mapping[str, Any] | None,
    *,
    assertions: Sequence[Mapping[str, Any]] | None = None,
) -> QualityScore:
    """把一次版面体检折算成 :class:`QualityScore`（纯函数，不抛异常）。

    参数
    ----
    qc:
        ``qc.check_layout()`` 的返回值 ``{"ok": bool, "checks": [{"name", "ok", "detail"}]}``。
        只读 ``checks``；``ok`` 不参与打分（总分反映的是缺陷总量，不是那一个布尔）。
        缺 ``checks`` 键、``checks`` 不是 list、元素缺 ``name`` / 不是 dict：
        一律跳过，**不抛异常**。
    assertions:
        断言结果，每条形如 ``{"name": str, "ok": bool, "detail": str}``（``ok`` 缺省视为
        未通过）。只用来算正确性维度：每条失败扣 ``100/条数``，所以 ``correctness``
        就是断言通过率。传 ``None`` 或空序列时正确性给 **满分 100**（见模块 docstring）。

    返回
    ----
    :class:`QualityScore`：``total = 100 − Σ扣分``，每个维度 ``= 100 − 该维度扣分``，
    全部夹在 0–100。同一缺陷重复出现会重复扣分（单调不增）。带 ``unmeasured`` 标记的
    检查（qc 读不到量）不扣分，但会列进 ``QualityScore.unmeasured``，不能当成通过。
    """
    deductions: list[Deduction] = []
    unmeasured: list[Deduction] = []

    raw_checks = qc.get("checks") if isinstance(qc, Mapping) else None
    for check in raw_checks if isinstance(raw_checks, (list, tuple)) else ():
        if not isinstance(check, Mapping):
            continue
        name = str(check.get("name") or "").strip()
        rule = DEDUCTIONS.get(name)
        # 未知检查名（含 qc 那些"读不到就跳过"的项）不扣分；通过项自然不扣分
        if rule is None:
            continue
        points, dimension = rule
        reason = str(check.get("detail") or "")
        if check.get("unmeasured"):
            unmeasured.append(Deduction(name=name, points=float(points), dimension=dimension,
                                        reason=reason))
            continue
        if check.get("ok"):
            continue
        deductions.append(
            Deduction(
                name=name,
                points=float(points),
                dimension=dimension,
                reason=reason,
            )
        )

    items = [item for item in (assertions or ()) if isinstance(item, Mapping)]
    if items:
        per = round(_FULL_SCORE / len(items), 2)
        for item in items:
            if item.get("ok"):
                continue
            deductions.append(
                Deduction(
                    name=str(item.get("name") or "assertion"),
                    points=per,
                    dimension=CORRECTNESS,
                    reason=str(item.get("detail") or "断言未通过"),
                )
            )

    per_dimension = {dimension: _FULL_SCORE for dimension in DIMENSIONS}
    for item in deductions:
        if item.dimension in per_dimension:
            per_dimension[item.dimension] -= item.points
    return QualityScore(
        total=_clamp(_FULL_SCORE - sum(item.points for item in deductions)),
        normativity=_clamp(per_dimension[NORMATIVITY]),
        readability=_clamp(per_dimension[READABILITY]),
        correctness=_clamp(per_dimension[CORRECTNESS]),
        deductions=deductions,
        unmeasured=unmeasured,
    )


def brief_quality(quality: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """trace 用的紧凑质量分：``total`` + 三维度 + 扣分项**名字** + 未评测项名字。

    为什么要缩：每条 ``tool_result`` 都写进 trace，而一条扣分的 ``reason`` 常有上百字；
    "trace 每天都要扫"与"复盘要看到扣了哪几项"这两件事只需这么点信息。
    完整明细仍留在工具返回值（``data["quality"]``）里，必要时从那里取。

    传进来的不是 dict（比如工具没给质量分）→ 返回 ``None``，调用方不写这个键：
    没测 ≠ 满分。
    """
    if not isinstance(quality, Mapping):
        return None
    out: dict[str, Any] = {key: quality.get(key) for key in ("total", *DIMENSIONS)}
    raw = quality.get("deductions")
    out["deductions"] = [
        str(item.get("name") or "")
        for item in (raw if isinstance(raw, (list, tuple)) else ())
        if isinstance(item, Mapping)
    ]
    # 读不到量的项要一眼看得出"没测"，而不是以为它通过了
    raw_unmeasured = quality.get("unmeasured")
    out["unmeasured"] = [
        str(item.get("name") or "")
        for item in (raw_unmeasured if isinstance(raw_unmeasured, (list, tuple)) else ())
        if isinstance(item, Mapping)
    ]
    return out


__all__ = [
    "CORRECTNESS",
    "DEDUCTIONS",
    "DIMENSIONS",
    "Deduction",
    "NORMATIVITY",
    "QualityScore",
    "READABILITY",
    "brief_quality",
    "score_layout",
]

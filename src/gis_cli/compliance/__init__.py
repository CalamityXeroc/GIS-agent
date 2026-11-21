# -*- coding: utf-8 -*-
"""地图合规闸门（P1-2 / W3）：YAML 规则库 + 事实采集 + 分级输出。

**只做提示 + 阻断，不做合规判定。** 这里是"提醒你少东西了"：图面有没有审图号、
边界数据的来源登记了没有、未送审的图有没有打水印。是不是合法、能不能对外发布，
以自然资源主管部门的地图审核结论为准 —— 这句话同时出现在规则库、卡片与 UI 文案里。

分级（``rules.LEVELS``）：

- ``block`` 阻断：必须处理，交付闸门（``verify_outputs`` / ``finish``）不通过；
- ``warn``  警告：照常交付，但要让人看见；
- ``info``  提示：只提一句，不影响任何判定。

对外入口：

- :func:`check_map`：纯函数，输入 :class:`MapContext`，输出 :class:`ComplianceReport`；
- :func:`gis_cli.compliance.facts.context_from_spec`：从 ``create_map`` 的 LayoutSpec 造上下文；
- :mod:`gis_cli.compliance.store`：边车读写与多图汇总（``create_map`` 写、验收读）。

每条规则**独立开关、单独给结论**：:class:`ComplianceReport.results` 里每条规则都有
``hit`` / ``pass`` / ``off`` / ``error`` 之一，``enabled={"rule_id": False}`` 可临时关掉单条。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .facts import MapContext, collect_facts
from .rules import (
    LEVEL_LABELS,
    LEVELS,
    LEVEL_ORDER,
    Rule,
    RuleError,
    RuleSet,
    evaluate,
    format_text,
    load_rules,
    worst_level,
)

logger = logging.getLogger(__name__)

#: 责任边界文案（与 ``rules.yaml`` 的 notice 保持一致，有测试守着不漂移）。
NOTICE = (
    "合规闸门只提醒你图面与数据里可能缺了什么，并可按严重程度阻断交付；它不做合规判定："
    "是否合法、能否对外发布，以自然资源主管部门的地图审核结论为准。"
)

#: 规则结论的四种状态：命中 / 未命中 / 已关闭 / 求值出错。
STATUSES: tuple[str, ...] = ("hit", "pass", "off", "error")


@dataclass(frozen=True)
class RuleResult:
    """一条规则的结论（``status`` 就是这条规则单独的判定）。"""

    rule_id: str
    title: str
    level: str
    status: str
    message: str = ""
    advice: str = ""
    category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "title": self.title,
            "category": self.category,
            "level": self.level,
            "status": self.status,
            "message": self.message,
            "advice": self.advice,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RuleResult":
        data = data if isinstance(data, Mapping) else {}
        return cls(
            rule_id=str(data.get("rule_id") or ""),
            title=str(data.get("title") or ""),
            level=str(data.get("level") or "info"),
            status=str(data.get("status") or "pass"),
            message=str(data.get("message") or ""),
            advice=str(data.get("advice") or ""),
            category=str(data.get("category") or ""),
        )


@dataclass(frozen=True)
class ComplianceReport:
    """一次自检的结论：逐条规则结论 + 事实快照 + 总体级别。"""

    results: list[RuleResult] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    notice: str = NOTICE
    error: str = ""
    elapsed_ms: float = 0.0

    # ---------------------------------------------------------- 派生结论
    @property
    def hits(self) -> list[RuleResult]:
        return [item for item in self.results if item.status == "hit"]

    @property
    def level(self) -> str:
        """总体级别：命中里最重的一级；没命中就是 ``pass``。"""
        return worst_level(item.level for item in self.hits)

    @property
    def blocked(self) -> bool:
        return self.level == "block"

    def count(self, level: str) -> int:
        return sum(1 for item in self.hits if item.level == level)

    @property
    def summary(self) -> str:
        if self.error:
            return f"合规闸门未完成：{self.error}"
        if not self.results:
            return "合规闸门：规则库为空，未做检查"
        head = "合规闸门："
        counts = [f"{LEVEL_LABELS[level]} {self.count(level)} 项" for level in LEVELS if self.count(level)]
        if not counts:
            return f"{head}未发现问题（{len(self.results)} 条规则）"
        detail = "；".join(item.message for item in self.hits[:2] if item.message)
        return f"{head}{'、'.join(counts)}" + (f"。{detail}" if detail else "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "blocked": self.blocked,
            "summary": self.summary,
            "notice": self.notice,
            "findings": [item.to_dict() for item in self.hits],
            "rules": [item.to_dict() for item in self.results],
            "facts": dict(self.facts),
            "error": self.error,
            "elapsed_ms": round(float(self.elapsed_ms), 2),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ComplianceReport":
        """从存档的 ``to_dict()`` 还原（边车里没有上下文时的退路）。"""
        data = data if isinstance(data, Mapping) else {}
        raw_rules = data.get("rules")
        results = [
            RuleResult.from_dict(item)
            for item in (raw_rules if isinstance(raw_rules, list) else [])
            if isinstance(item, Mapping)
        ]
        return cls(
            results=results,
            facts=dict(data.get("facts") or {}),
            notice=str(data.get("notice") or NOTICE),
            error=str(data.get("error") or ""),
            elapsed_ms=float(data.get("elapsed_ms") or 0.0),
        )


# ------------------------------------------------------------------ 入口
def _resolve_ruleset(rules: RuleSet | Iterable[Rule] | None, path: str | None) -> RuleSet:
    if isinstance(rules, RuleSet):
        return rules
    if rules is not None:
        return RuleSet(version=1, notice=NOTICE, rules=list(rules))
    return load_rules(path)


def check_map(
    context: MapContext,
    *,
    rules: RuleSet | Iterable[Rule] | None = None,
    rules_path: str | None = None,
    enabled: Mapping[str, bool] | None = None,
) -> ComplianceReport:
    """对一次出图跑一遍合规闸门（纯函数：不写盘、不联网、不碰 arcpy）。

    Args:
        context: 事实来源（见 :class:`gis_cli.compliance.facts.MapContext`）。
        rules: 已装载的规则集（测试/自定义库用）；不给就按 ``rules_path`` 读，
            再不给就读内置 ``rules.yaml``。
        enabled: 单条规则开关覆盖（``{"rule_id": False}``），未列出的按 YAML 里的
            ``enabled`` 走。关掉的规则结论是 ``off``，不是 ``pass`` ——
            "没检查"和"检查了没问题"必须分得开。
        rules_path: 自定义规则库路径。

    Returns:
        :class:`ComplianceReport`。规则库装载失败**不抛异常**：返回 ``error`` 非空的报告
        （不命中任何规则，但状态会被验收汇进 ``problems``、写进摘要）。
        闸门自己坏掉时不许静默放行，也不许把出图带崩。
    """
    started = time.perf_counter()
    facts = collect_facts(context)
    try:
        ruleset = _resolve_ruleset(rules, rules_path)
    except RuleError as exc:
        logger.warning("合规规则库装载失败: %s", exc)
        return ComplianceReport(
            results=[],
            facts=facts,
            notice=NOTICE,
            error=str(exc),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    overrides = dict(enabled or {})
    results: list[RuleResult] = []
    for rule in ruleset.rules:
        on = bool(overrides.get(rule.id, rule.enabled))
        base = dict(
            rule_id=rule.id,
            title=rule.title,
            level=rule.level,
            category=rule.category,
        )
        if not on:
            results.append(RuleResult(**base, status="off"))
            continue
        try:
            hit = evaluate(rule.when, facts)
        except Exception as exc:  # 单条规则坏掉不能拖垮整张图，但也必须留痕
            logger.warning("合规规则 %s 求值失败: %s", rule.id, exc)
            results.append(RuleResult(**base, status="error", message=str(exc)[:160]))
            continue
        if hit:
            results.append(
                RuleResult(
                    **base,
                    status="hit",
                    message=format_text(rule.message, facts),
                    advice=format_text(rule.advice, facts),
                )
            )
        else:
            results.append(RuleResult(**base, status="pass"))

    return ComplianceReport(
        results=results,
        facts=facts,
        notice=ruleset.notice or NOTICE,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


__all__ = [
    "LEVEL_LABELS",
    "LEVEL_ORDER",
    "LEVELS",
    "NOTICE",
    "STATUSES",
    "ComplianceReport",
    "MapContext",
    "RuleError",
    "RuleResult",
    "check_map",
    "collect_facts",
    "load_rules",
]

# -*- coding: utf-8 -*-
"""合规规则库：YAML 装载 + 条件求值 + 文案格式化（纯 Python，不碰 arcpy）。

一条规则 = **事实（facts）上的一个布尔条件** + 分级 + 一句话结论 + 修法建议。
事实由 :mod:`gis_cli.compliance.facts` 从地图上下文里采集；规则本身不读文件、
不联网、不 import arcpy —— 所以规则库可以整份替换、按 id 单独开关，
也能在没有 ArcGIS 的机器上逐条单测。

条件 DSL（只有四种形状，够用且能被校验）::

    when:
      all:                      # 全部成立（列表）
        - fact: sensitive_count
          op: gt
          value: 0
        - any:                  # 其一成立
            - fact: approval
              op: eq
              value: absent
        - not:                  # 取反
            fact: watermark
            op: eq
            value: present

叶子算子是 :data:`OPS`；写错（未知 fact、未知算子、未知级别、重复 id、
文案引用了不存在的事实）在**装载时**就抛 :class:`RuleError`，
不要等到出图时才静默通过 —— 闸门静默失效比误报更糟。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml  # type: ignore

#: 分级（由重到轻）。``pass`` 不是规则级别，只用于"没命中"的结论。
LEVELS: tuple[str, ...] = ("block", "warn", "info")
LEVEL_LABELS: dict[str, str] = {"block": "阻断", "warn": "警告", "info": "提示"}
#: 比较用权重：最大者即报告的总体级别。
LEVEL_ORDER: dict[str, int] = {"pass": 0, "info": 1, "warn": 2, "block": 3}

#: 条件叶子支持的算子。
OPS: tuple[str, ...] = ("eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "truthy", "falsy")

#: 事实名（:mod:`gis_cli.compliance.facts` 的 ``collect_facts`` 必须提供这些键）。
#: 规则里引用别的一律在装载时报错 —— 拼错一个键就会让规则恒假，属于最隐蔽的失效。
FACT_NAMES: frozenset[str] = frozenset(
    {
        "layer_count",
        "sensitive_count",
        "sensitive_layers",
        "sensitive_in_text",
        "sensitive_hint",
        "boundary_count",
        "boundary_layers",
        "unregistered_count",
        "unregistered_layers",
        "approval",
        "approval_note",
        "watermark",
        "boundary_source",
        "catalog",
        "audience",
    }
)

_RULE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")
#: 随包分发的默认规则库（``pyproject.toml`` 里有 package-data 保证打进 wheel）。
PACKAGED_RULES = Path(__file__).with_name("rules.yaml")
#: 用户覆盖：仓库根的 ``config/compliance_rules.yaml``（与风格档位/样式目录同一约定）。
#: 用户改规则不必改源码；不存在就用内置那份。
OVERRIDE_RULES = Path(__file__).resolve().parents[3] / "config" / "compliance_rules.yaml"


class RuleError(ValueError):
    """规则库写错（YAML 结构/未知事实/未知算子/文案占位符）时抛给人看的错。"""

    kind = "ComplianceRuleError"


@dataclass(frozen=True)
class Rule:
    """一条合规规则（不可变：装载后不再改，运行时只按 id 开关）。"""

    id: str
    title: str
    level: str
    when: dict[str, Any]
    message: str
    advice: str = ""
    category: str = ""
    rationale: str = ""
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "level": self.level,
            "enabled": self.enabled,
            "rationale": self.rationale,
            "when": dict(self.when),
            "message": self.message,
            "advice": self.advice,
        }


@dataclass(frozen=True)
class RuleSet:
    """整份规则库：版本 + 责任边界文案 + 规则列表（顺序即输出顺序）。"""

    version: int
    notice: str
    rules: list[Rule] = field(default_factory=list)

    def by_id(self, rule_id: str) -> Rule | None:
        return next((rule for rule in self.rules if rule.id == rule_id), None)

    @property
    def ids(self) -> list[str]:
        return [rule.id for rule in self.rules]


# ------------------------------------------------------------------ 装载
def default_rules_path() -> Path:
    """本次真正生效的规则库：``config/compliance_rules.yaml``（若存在）否则内置 ``rules.yaml``。"""
    return OVERRIDE_RULES if OVERRIDE_RULES.is_file() else PACKAGED_RULES


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuleError(f"{where} 必须是对象（YAML mapping）")
    return dict(value)


def _check_text(text: str, where: str) -> str:
    """文案里的 ``{xxx}`` 必须是 facts 的键；返回整理成单行的文案。"""
    flat = " ".join(str(text or "").split())
    for name in _PLACEHOLDER.findall(flat):
        if name not in FACT_NAMES:
            raise RuleError(f"{where} 引用了不存在的事实 {{{name}}}；可用事实：{', '.join(sorted(FACT_NAMES))}")
    return flat


def _validate_condition(node: Any, where: str) -> dict[str, Any]:
    """递归校验条件树（形状 + 事实名 + 算子 + value 类型）。"""
    data = _require_mapping(node, where)
    keys = set(data) & {"all", "any", "not", "fact"}
    if len(keys) != 1:
        raise RuleError(f"{where} 必须恰好是 all / any / not / fact 之一，收到：{sorted(data)}")
    key = keys.pop()
    if key in ("all", "any"):
        items = data[key]
        if not isinstance(items, list) or not items:
            raise RuleError(f"{where}.{key} 必须是非空列表")
        return {key: [_validate_condition(item, f"{where}.{key}[{index}]") for index, item in enumerate(items)]}
    if key == "not":
        return {"not": _validate_condition(data[key], f"{where}.not")}

    fact = str(data.get("fact") or "").strip()
    if fact not in FACT_NAMES:
        raise RuleError(f"{where} 的 fact「{fact}」不存在；可用事实：{', '.join(sorted(FACT_NAMES))}")
    op = str(data.get("op") or "").strip()
    if op not in OPS:
        raise RuleError(f"{where} 的算子「{op}」不支持；可用算子：{', '.join(OPS)}")
    if "value" not in data and op not in ("truthy", "falsy"):
        raise RuleError(f"{where} 缺少 value（算子 {op} 需要它）")
    if op in ("in", "not_in") and not isinstance(data.get("value"), (list, tuple)):
        raise RuleError(f"{where} 的 {op} 需要列表 value")
    return {"fact": fact, "op": op, "value": data.get("value")}


def parse_ruleset(payload: Mapping[str, Any]) -> RuleSet:
    """把 YAML 内容解析成 :class:`RuleSet`（含全部校验；写错即抛 RuleError）。"""
    data = _require_mapping(payload, "规则库")
    version = int(data.get("version") or 0)
    if version != 1:
        raise RuleError(f"规则库 version 只支持 1，收到：{data.get('version')!r}")
    notice = _check_text(str(data.get("notice") or ""), "notice")
    if not notice:
        raise RuleError("规则库缺少 notice（责任边界文案）")

    levels = data.get("levels") or {}
    if isinstance(levels, Mapping):
        for name in levels:
            if str(name) not in LEVELS:
                raise RuleError(f"levels 里出现未知级别「{name}」；只认 {', '.join(LEVELS)}")

    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise RuleError("规则库 rules 必须是非空列表")

    rules: list[Rule] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rules):
        where = f"rules[{index}]"
        item = _require_mapping(raw, where)
        rule_id = str(item.get("id") or "").strip()
        if not _RULE_ID.match(rule_id):
            raise RuleError(f"{where}.id「{rule_id}」不合法（小写蛇形，如 approval_missing）")
        if rule_id in seen:
            raise RuleError(f"规则 id 重复：{rule_id}")
        seen.add(rule_id)
        level = str(item.get("level") or "").strip()
        if level not in LEVELS:
            raise RuleError(f"{where}.level「{level}」不合法；只认 {', '.join(LEVELS)}")
        title = " ".join(str(item.get("title") or "").split())
        if not title:
            raise RuleError(f"{where}.title 不能为空")
        rules.append(
            Rule(
                id=rule_id,
                title=title,
                level=level,
                when=_validate_condition(item.get("when"), f"{where}.when"),
                message=_check_text(str(item.get("message") or ""), f"{where}.message"),
                advice=_check_text(str(item.get("advice") or ""), f"{where}.advice"),
                category=" ".join(str(item.get("category") or "").split()),
                rationale=" ".join(str(item.get("rationale") or "").split()),
                enabled=bool(item.get("enabled", True)),
            )
        )
    return RuleSet(version=version, notice=notice, rules=rules)


def load_rules(path: str | Path | None = None) -> RuleSet:
    """读一份 YAML 规则库；文件缺失/语法错/结构错都抛 :class:`RuleError`。"""
    target = Path(path) if path else default_rules_path()
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuleError(f"读不到规则库 {target}：{exc}") from exc
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover - 依赖 PyYAML 的报错文本
        raise RuleError(f"规则库 {target.name} 不是合法 YAML：{exc}") from exc
    return parse_ruleset(payload or {})


# ------------------------------------------------------------------ 求值
def _compare(fact_value: Any, op: str, expected: Any) -> bool:
    if op == "truthy":
        return bool(fact_value)
    if op == "falsy":
        return not bool(fact_value)
    if op == "eq":
        return fact_value == expected
    if op == "ne":
        return fact_value != expected
    if op == "in":
        return fact_value in tuple(expected or ())
    if op == "not_in":
        return fact_value not in tuple(expected or ())
    # 数值比较：拿不到数就是"不成立"，不抛异常（规则写错已在装载时拦住；
    # 运行时缺事实只会让规则不命中，不该把整张图的闸门带崩）
    try:
        left, right = float(fact_value), float(expected)
    except (TypeError, ValueError):
        return False
    if op == "gt":
        return left > right
    if op == "gte":
        return left >= right
    if op == "lt":
        return left < right
    if op == "lte":
        return left <= right
    return False  # pragma: no cover - OPS 已校验，走不到这里


def evaluate(condition: Mapping[str, Any], facts: Mapping[str, Any]) -> bool:
    """在事实集合上求值一棵条件树（未知形状直接抛错，不静默放行）。"""
    if "all" in condition:
        return all(evaluate(item, facts) for item in condition["all"])
    if "any" in condition:
        return any(evaluate(item, facts) for item in condition["any"])
    if "not" in condition:
        return not evaluate(condition["not"], facts)
    return _compare(facts.get(str(condition.get("fact"))), str(condition.get("op")), condition.get("value"))


def format_text(template: str, facts: Mapping[str, Any]) -> str:
    """把 ``{fact}`` 换成事实值（缺失的键原样保留，装载时已校验过不会发生）。"""
    def _sub(match: re.Match[str]) -> str:
        value = facts.get(match.group(1))
        if value is None:
            return match.group(0)
        return str(value)

    return " ".join(_PLACEHOLDER.sub(_sub, str(template or "")).split())


def worst_level(levels: Iterable[str]) -> str:
    """取最重的一级；空集合返回 ``pass``。"""
    best = "pass"
    for level in levels:
        if LEVEL_ORDER.get(str(level), 0) > LEVEL_ORDER[best]:
            best = str(level)
    return best


__all__ = [
    "FACT_NAMES",
    "LEVELS",
    "LEVEL_LABELS",
    "LEVEL_ORDER",
    "OPS",
    "OVERRIDE_RULES",
    "PACKAGED_RULES",
    "Rule",
    "RuleError",
    "RuleSet",
    "default_rules_path",
    "evaluate",
    "format_text",
    "load_rules",
    "parse_ruleset",
    "worst_level",
]

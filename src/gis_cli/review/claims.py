# -*- coding: utf-8 -*-
"""复核的输入/输出契约：主代理只交「结论」，复核者只回「判定」。

为什么单独定义契约：W5 的核心约束是**复核者不看主轨迹**。只要主代理仍能
通过 ``summary``/``methodology`` 把“我是怎么一步步做出来的”带进去，复核就
退化成复述。所以复核的输入被收窄成三块：

1. :class:`ReviewClaim` —— 任务目标 + 主代理的结论与自报数字（不含方法步骤）；
2. 数据摘要（``review.facts.collect_facts``，系统直接从数据量测）；
3. 断言库（``review.assertions``）。

``methodology`` 故意**不在**契约里：它是“怎么做”的叙述，属于主轨迹。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

#: 断言失败的默认严重度。``error`` 会让复核判定为 fail；``warning`` 只提示。
SEVERITIES = ("error", "warning")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_paths(value: Any) -> list[str]:
    out: list[str] = []
    for item in _as_list(value):
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


@dataclass
class ReviewClaim:
    """主代理交给复核者的结论（**不含方法步骤/日志/中间推理**）。

    ``metrics`` 是自报数字的平铺字典，键名由主代理/任务决定，例如::

        {"total_pop": 8420000, "zone_pop": {"Z01": 120000, ...},
         "area_km2": 152.34, "density_per_km2": 5528.1, "feature_count": 118}

    复核者与断言库只读这份数字 + 量测事实；``expected`` 用来解释**可接受的**
    变化（例如溶解后要素数从 118 变 1），没有解释的变化按拓扑断言报错。
    """

    goal: str = ""
    summary: str = ""
    data_paths: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    crs: str = ""
    expected: dict[str, Any] = field(default_factory=dict)
    #: 任务自带的断言清单（主代理可原样透传；复核者也可以通过 assertion_check 传）
    assertions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.goal:
            payload["goal"] = self.goal
        if self.summary:
            payload["summary"] = self.summary
        if self.data_paths:
            payload["data_paths"] = list(self.data_paths)
        if self.metrics:
            payload["metrics"] = dict(self.metrics)
        if self.units:
            payload["units"] = dict(self.units)
        if self.crs:
            payload["crs"] = self.crs
        if self.expected:
            payload["expected"] = dict(self.expected)
        if self.assertions:
            payload["assertions"] = [dict(item) for item in self.assertions if isinstance(item, Mapping)]
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ReviewClaim":
        data = data if isinstance(data, Mapping) else {}
        return cls(
            goal=str(data.get("goal", "") or ""),
            summary=str(data.get("summary", "") or ""),
            data_paths=_as_paths(data.get("data_paths")),
            metrics=_as_dict(data.get("metrics")),
            units={str(k): str(v) for k, v in _as_dict(data.get("units")).items()},
            crs=str(data.get("crs", "") or ""),
            expected=_as_dict(data.get("expected")),
            assertions=[
                dict(item) for item in _as_list(data.get("assertions")) if isinstance(item, Mapping)
            ],
        )


@dataclass
class Finding:
    """复核发现的一条问题。"""

    code: str
    detail: str
    family: str = ""
    severity: str = "error"
    assertion: dict[str, Any] | None = None
    claimed: Any = None
    measured: Any = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "detail": self.detail,
        }
        if self.family:
            payload["family"] = self.family
        if self.assertion is not None:
            payload["assertion"] = self.assertion
        if self.claimed is not None:
            payload["claimed"] = self.claimed
        if self.measured is not None:
            payload["measured"] = self.measured
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Finding":
        data = data if isinstance(data, Mapping) else {}
        severity = str(data.get("severity", "error") or "error").lower()
        return cls(
            code=str(data.get("code", "") or "review.finding"),
            detail=str(data.get("detail", "") or ""),
            family=str(data.get("family", "") or ""),
            severity=severity if severity in SEVERITIES else "error",
            assertion=_as_dict(data.get("assertion")) or None,
            claimed=data.get("claimed"),
            measured=data.get("measured"),
        )


@dataclass
class ReviewReport:
    """断言库跑完的结论（纯函数产物，可独立于 LLM 调用）。"""

    findings: list[Finding] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def error_findings(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == "error"]

    @property
    def warning_findings(self) -> list[Finding]:
        return [item for item in self.findings if item.severity != "error"]

    @property
    def ok(self) -> bool:
        # 没进 checked 就是「没检查」：全 skipped 时不能算通过（文档自己写的口径）。
        # 断言自己跑坏（errors）同样是「没检查」——以前只记 errors 不改 ok，
        # 复核报告会写成「断言全部通过」（DAIR-191 缺陷 3）。
        return bool(self.checked) and not self.error_findings and not self.errors

    def _summary_body(self) -> str:
        if not self.checked:
            return f"未检查（{len(self.skipped)} 条断言缺事实）"
        if not self.findings:
            extra = f"（跳过 {len(self.skipped)} 条缺事实的断言）" if self.skipped else ""
            return f"断言全部通过（{len(self.checked)} 条）{extra}"
        parts: list[str] = []
        errors = self.error_findings
        warnings = self.warning_findings
        if errors:
            codes = "、".join(sorted({item.code for item in errors})[:4])
            parts.append(f"发现 {len(errors)} 个问题（{codes}）")
        if warnings:
            codes = "、".join(sorted({item.code for item in warnings})[:4])
            parts.append(f"{len(warnings)} 条提示（{codes}）")
        return "断言" + "；".join(parts)

    @property
    def summary(self) -> str:
        # 断言库自己坏掉时必须先说，不能藏在「其余 N 条通过」后面
        if self.errors:
            head = "断言库未跑成：" + "；".join(self.errors[:2])
            if not self.checked:
                return head
            tail = f"另有 {len(self.checked)} 条断言跑完"
            if self.error_findings:
                tail += f"，发现 {len(self.error_findings)} 个问题"
            return f"{head}（{tail}）"
        return self._summary_body()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary,
            "findings": [item.to_dict() for item in self.findings],
            "checked": list(self.checked),
            "skipped": list(self.skipped),
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ReviewReport":
        data = data if isinstance(data, Mapping) else {}
        return cls(
            findings=[
                Finding.from_dict(item) for item in _as_list(data.get("findings")) if isinstance(item, Mapping)
            ],
            checked=[str(x) for x in _as_list(data.get("checked"))],
            skipped=[str(x) for x in _as_list(data.get("skipped"))],
            errors=[str(x) for x in _as_list(data.get("errors"))],
        )


@dataclass
class ReviewVerdict:
    """一次复核的最终判定（会进 trace / 验收观察 / 任务状态）。"""

    #: pass / fail / unparsed / refused / error / max_turns / stopped / deadline_exceeded …
    status: str = "unparsed"
    findings: list[Finding] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    notes: str = ""
    subtask_id: str = ""
    turns: int = 0
    tokens: int = 0
    elapsed_s: float = 0.0
    trace: str = ""
    raw: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "pass" and not any(item.severity == "error" for item in self.findings)

    @property
    def error_findings(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == "error"]

    def summary(self, limit: int = 600) -> str:
        if self.status == "pass":
            extra = f"（检查 {len(self.checked)} 条断言）" if self.checked else ""
            return f"双复核通过{extra}"
        if self.status == "unparsed":
            return "双复核未给出可解析的判定（按未通过处理，不阻断交付）"
        if self.status == "fail":
            head = "、".join(sorted({item.code for item in self.findings})[:4]) or "发现不一致"
            return f"双复核未通过（{len(self.error_findings)} 个问题：{head}）"
        return f"双复核未完成（{self.status}）：{self.error or self.notes}"[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ok": self.ok,
            "summary": self.summary(),
            "findings": [item.to_dict() for item in self.findings],
            "checked": list(self.checked),
            "notes": self.notes[:2000],
            "subtask_id": self.subtask_id,
            "turns": self.turns,
            "tokens": self.tokens,
            "elapsed_s": round(float(self.elapsed_s), 1),
            "trace": self.trace,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ReviewVerdict":
        data = data if isinstance(data, Mapping) else {}
        return cls(
            status=str(data.get("status", "unparsed") or "unparsed"),
            findings=[
                Finding.from_dict(item) for item in _as_list(data.get("findings")) if isinstance(item, Mapping)
            ],
            checked=[str(x) for x in _as_list(data.get("checked"))],
            notes=str(data.get("notes", "") or ""),
            subtask_id=str(data.get("subtask_id", "") or ""),
            turns=int(data.get("turns") or 0),
            tokens=int(data.get("tokens") or 0),
            elapsed_s=float(data.get("elapsed_s") or 0.0),
            trace=str(data.get("trace", "") or ""),
            error=str(data.get("error", "") or ""),
        )


def parse_review_payload(payload: Mapping[str, Any] | None, *, raw: str = "") -> ReviewVerdict:
    """把复核者 ``finish`` 的 JSON 转成 :class:`ReviewVerdict`。

    约定（写进复核 prompt）：``verdict`` ∈ ``pass`` / ``fail``，``findings`` 是
    数组。缺字段、类型不对都按 ``unparsed`` 处理——**不能**把“没读懂”当成通过。
    但结论异常的 fail（说 fail 却没给 findings）补一条通用 finding，避免调用方
    拿到一个空 findings 的失败。
    """
    payload = payload if isinstance(payload, Mapping) else {}
    raw_verdict = str(payload.get("verdict", "") or "").strip().lower()
    findings = [
        Finding.from_dict(item) for item in _as_list(payload.get("findings")) if isinstance(item, Mapping)
    ]
    checked = [str(x) for x in _as_list(payload.get("checked"))]
    notes = str(payload.get("notes", "") or "")
    if raw_verdict in ("pass", "ok", "通过"):
        status = "pass" if not any(item.severity == "error" for item in findings) else "fail"
    elif raw_verdict in ("fail", "failed", "不通过", "未通过"):
        status = "fail"
        if not findings:
            findings = [
                Finding(
                    code="review.reported_fail",
                    detail=notes or "复核者判定不通过，但没有给出具体发现",
                    severity="error",
                )
            ]
    else:
        status = "unparsed"
    return ReviewVerdict(
        status=status,
        findings=findings,
        checked=checked,
        notes=notes,
        raw=raw[:8000],
    )


__all__ = [
    "SEVERITIES",
    "Finding",
    "ReviewClaim",
    "ReviewReport",
    "ReviewVerdict",
    "parse_review_payload",
]

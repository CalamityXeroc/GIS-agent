from __future__ import annotations

from .models import RiskLevel

DANGEROUS_KEYWORDS = ["删除", "覆盖", "清空", "drop", "truncate", "delete"]


def evaluate_risk(prompt: str, default: RiskLevel) -> RiskLevel:
    prompt_lc = prompt.lower()
    if any(k in prompt_lc for k in DANGEROUS_KEYWORDS):
        return RiskLevel.HIGH
    return default


def requires_confirmation(risk: RiskLevel) -> bool:
    return risk == RiskLevel.HIGH

# -*- coding: utf-8 -*-
"""Recipe schema: versioned, verifiable GIS methodology units."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore

_SLOT = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_RAW_SLOT = re.compile(r"@@\s*([A-Za-z_][A-Za-z0-9_]*)\s*@@")


@dataclass
class RecipeParam:
    """A parameter accepted by a recipe."""

    name: str
    type: str = "string"
    required: bool = False
    default: Any = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "required": self.required,
            "default": self.default,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "RecipeParam":
        return cls(
            name=name,
            type=str(data.get("type", "string")),
            required=bool(data.get("required", False)),
            default=data.get("default"),
            description=str(data.get("description", "") or ""),
        )


@dataclass
class Recipe:
    """A reusable, validated GIS methodology."""

    id: str
    name: str
    description: str = ""
    domain: str = "general"
    triggers: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    params: dict[str, RecipeParam] = field(default_factory=dict)
    steps: list[str] = field(default_factory=list)
    code_template: str = ""
    validation: list[dict[str, Any]] = field(default_factory=list)
    pitfalls: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    version: str = "1.0"
    provenance: str = ""
    source_path: str = ""

    # ------------------------------------------------------------------ render
    def render(self, params: dict[str, Any]) -> str:
        """Render the code template.

        ``{{param}}`` is replaced with a Python literal (quoted strings,
        numbers, JSON for containers); ``@@param@@`` is replaced with the raw
        string value for embedding inside code/expressions.
        """
        resolved = self.resolve_params(params)

        def _literal_sub(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in resolved:
                return match.group(0)
            return _py_literal(resolved[key])

        def _raw_sub(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in resolved:
                return match.group(0)
            return "" if resolved[key] is None else str(resolved[key])

        text = _SLOT.sub(_literal_sub, self.code_template)
        return _RAW_SLOT.sub(_raw_sub, text)

    def resolve_params(self, params: dict[str, Any]) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for name, spec in self.params.items():
            if name in params and params[name] is not None:
                resolved[name] = params[name]
            elif spec.default is not None:
                resolved[name] = spec.default
            elif spec.required:
                raise ValueError(f"配方 {self.id} 缺少必填参数: {name}")
        for name, value in (params or {}).items():
            if name not in resolved:
                resolved[name] = value
        return resolved

    def to_dict(self, *, with_code: bool = False) -> dict[str, Any]:
        data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "triggers": self.triggers,
            "preconditions": self.preconditions,
            "params": {k: v.to_dict() for k, v in self.params.items()},
            "steps": self.steps,
            "validation": self.validation,
            "pitfalls": self.pitfalls,
            "examples": self.examples,
            "version": self.version,
            "provenance": self.provenance,
        }
        if with_code:
            data["code_template"] = self.code_template
        return data

    # ------------------------------------------------------------------- load
    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_path: str = "") -> "Recipe":
        raw_params = data.get("params") or {}
        params = {
            str(name): RecipeParam.from_dict(str(name), spec or {})
            for name, spec in raw_params.items()
        }
        return cls(
            id=str(data.get("id", "") or ""),
            name=str(data.get("name", "") or ""),
            description=str(data.get("description", "") or ""),
            domain=str(data.get("domain", "general") or "general"),
            triggers=[str(t) for t in (data.get("triggers") or [])],
            preconditions=[str(t) for t in (data.get("preconditions") or [])],
            params=params,
            steps=[str(s) for s in (data.get("steps") or [])],
            code_template=str(data.get("code_template", "") or ""),
            validation=[v for v in (data.get("validation") or []) if isinstance(v, dict)],
            pitfalls=[str(p) for p in (data.get("pitfalls") or [])],
            examples=[str(e) for e in (data.get("examples") or [])],
            version=str(data.get("version", "1.0") or "1.0"),
            provenance=str(data.get("provenance", "") or ""),
            source_path=source_path,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Recipe":
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Invalid recipe file: {p}")
        return cls.from_dict(data, source_path=str(p))

    def to_yaml(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            yaml.safe_dump(self.to_dict(with_code=True), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def match_score(self, query: str) -> int:
        """Simple keyword score for retrieval."""
        text = (query or "").strip().lower()
        if not text:
            return 1
        score = 0
        tokens = [t for t in re.split(r"[\s,，。;；/]+", text) if t]
        for token in tokens:
            if token in self.id.lower() or token in self.name.lower():
                score += 6
            if token in self.description.lower():
                score += 3
            for trigger in self.triggers:
                trigger_lower = trigger.lower()
                if token and (token in trigger_lower or trigger_lower in token):
                    score += 4
        return score


def _py_literal(value: Any) -> str:
    """Render a Python-safe literal for template substitution."""
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)

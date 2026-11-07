# -*- coding: utf-8 -*-
"""Recipe schema: versioned, verifiable GIS methodology units."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore

from .backends import DEFAULT_BACKEND, normalize_backends
from .fragments import FRAGMENTS, FRAGMENT_SLOT

#: 参数默认值写成它 → 运行时自动填成当前工作区路径。
#: 蒸馏出来的配方常常需要“以工作区为根”的路径（``ws = {{workspace}}``），
#: 但把源机器路径当默认值不可移植，写 ``.`` 又不对（内核 CWD 不是工作区）。
WORKSPACE_SENTINEL = "@workspace@"

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
    category: str = ""
    #: 这个配方要哪个执行后端跑（``[arcpy]`` / ``[gdal]`` / ``[arcpy, gdal]``）。
    #: 缺省 ``[arcpy]``：47 个老配方 manifest 一行不改也必须照样跑。
    backend: list[str] = field(default_factory=lambda: [DEFAULT_BACKEND])
    triggers: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    params: dict[str, RecipeParam] = field(default_factory=dict)
    steps: list[str] = field(default_factory=list)
    code_template: str = ""
    #: 同一配方的 GDAL 版代码（``backend`` 同时声明了 gdal 才有意义）。
    #: 空串 = **没有** GDAL 实现；按 gdal 后端渲染时明确报错，不回落 arcpy 代码
    #: —— 那会在 .venv-gdal 里撞 ``import arcpy`` 而报看不懂的错。
    code_template_gdal: str = ""
    validation: list[dict[str, Any]] = field(default_factory=list)
    pitfalls: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    version: str = "1.0"
    provenance: str = ""
    source_path: str = ""

    # ------------------------------------------------------------------ render
    def template_for(self, backend: str = DEFAULT_BACKEND) -> str:
        """取该后端的代码模板；这个后端没有模板就报错，**绝不回落**。

        为什么不能回落：GDAL 版不是“arcpy 代码换了个执行器”，两者 API 完全不同。
        回落的结果是 .venv-gdal 里 ``import arcpy`` 报 ``No module named 'arcpy'``
        —— 一个与配方内容无关的错，查起来还得先猜是后端路由错了。
        """
        name = str(backend or DEFAULT_BACKEND).strip().lower()
        if name == "arcpy":
            return self.code_template
        if name == "gdal":
            if not self.code_template_gdal.strip():
                raise ValueError(
                    f"配方 {self.id or '(未命名)'} 没有 GDAL 版代码（code_template_gdal 为空），"
                    f"不能用 gdal 后端跑；声明后端为 {list(self.backend)}。"
                )
            return self.code_template_gdal
        raise ValueError(f"未知执行后端: {backend!r}（支持 arcpy / gdal）")

    def render(self, params: dict[str, Any], *, backend: str = DEFAULT_BACKEND) -> str:
        """Render the code template of ``backend``.

        ``{{param}}`` is replaced with a Python literal (quoted strings,
        numbers, JSON for containers); ``@@param@@`` is replaced with the raw
        string value for embedding inside code/expressions.
        """
        template = self.template_for(backend)
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

        text = _SLOT.sub(_literal_sub, template)
        text = _RAW_SLOT.sub(_raw_sub, text)
        # 片段只在真用到时展开：老配方没有占位符，渲染输出必须逐字不变（快照盯着）
        return _expand_fragments(text) if FRAGMENT_SLOT.search(text) else text

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
        # 按声明类型收敛：蒸馏出的配方常把 ``decimals`` 声明成 integer 却给
        # 默认值 ``"2"``，渲染后变成 ``round(x, '2')`` → TypeError。类型是配方的
        # 契约，就在渲染前生效，代码与断言看到的是同一个值。
        for name, value in list(resolved.items()):
            spec = self.params.get(name)
            if spec is not None:
                resolved[name] = _coerce_param(value, spec.type)
        return resolved

    def to_dict(self, *, with_code: bool = False) -> dict[str, Any]:
        data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "category": self.category,
            "backend": list(self.backend),
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
            # 只在真写过 GDAL 版时出现：老配方 to_dict 的形态一行不变
            if self.code_template_gdal:
                data["code_template_gdal"] = self.code_template_gdal
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
            category=str(data.get("category", "") or data.get("domain", "general") or "general"),
            backend=normalize_backends(data.get("backend")),
            triggers=[str(t) for t in (data.get("triggers") or [])],
            preconditions=[str(t) for t in (data.get("preconditions") or [])],
            params=params,
            steps=[str(s) for s in (data.get("steps") or [])],
            code_template=str(data.get("code_template", "") or ""),
            code_template_gdal=str(data.get("code_template_gdal", "") or ""),
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


def _expand_fragments(text: str) -> str:
    """把 ``{{@name@}}`` 展开成 fragments.py 里的源码，按占位所在的缩进对齐。

    两条硬约束，违反就报人话错误，不留到执行器里变成 SyntaxError / 静默残留：

    1. 占位符必须**独占一行**（``x = {{@zone_layer@}}`` 会拼出 ``x = def ...``）；
    2. 一行**只能有一个**占位符（以前只展开第一个，第二个原样留在代码里）。

    片段名拼错同样当场报错：静默留一行 ``{{@...@}}`` 会变成语法错，报错点在执行器里，
    离真正的原因（片段名拼错）很远。
    """
    lines: list[str] = []
    for line in text.split("\n"):
        match = FRAGMENT_SLOT.fullmatch(line.strip())
        if match is None:
            slots = FRAGMENT_SLOT.findall(line)
            if not slots:
                lines.append(line)
                continue
            if len(slots) > 1:
                raise ValueError(
                    f"代码片段占位符必须独占一行，这行写了 {len(slots)} 个"
                    f"（{', '.join('{{@' + s + '@}}' for s in slots)}）：{line.strip()}"
                    f"—— 片段是整块语句，多段请各占一行"
                )
            raise ValueError(
                f"代码片段占位符必须独占一行，不能和内联代码混写：{line.strip()}"
                f"（{{@{slots[0]}@}} 会整段展开成 def 定义，混写拼出的代码是语法错）"
            )
        name = match.group(1)
        if name not in FRAGMENTS:
            raise ValueError(f"未知的代码片段: {name}（可用: {sorted(FRAGMENTS)}）")
        pad = line[: len(line) - len(line.lstrip())]
        block = [pad + item if item else item for item in FRAGMENTS[name].splitlines()]
        lines.append("\n".join(block))
    return "\n".join(lines)


def _coerce_param(value: Any, declared: str) -> Any:
    """把“声明了非字符串类型、实际传了字符串”的参数转成真值。

    只处理字符串输入（真 int/bool 原样保留）；转换不了就原样返回，交给代码
    自己报错——静默改成别的值比报错更难查。
    """
    if not isinstance(value, str):
        return value
    kind = (declared or "").lower()
    text = value.strip()
    try:
        if kind in {"integer", "int"}:
            return int(text)
        if kind in {"number", "float", "double"}:
            return float(text)
    except ValueError:
        return value
    if kind in {"boolean", "bool"}:
        lowered = text.lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return value


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

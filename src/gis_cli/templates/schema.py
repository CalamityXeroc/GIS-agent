# -*- coding: utf-8 -*-
"""民生模板 manifest 的 schema：数据清单 + 配方链 + 参数绑定 + 断言集。

模板不是"新配方"：它把已有配方按顺序串起来，并把上一步的产物喂给下一步。
**绑定语法是这个模块的核心契约**（文档：``docs/templates.md``），只有三种引用：

- ``$data.<输入名>``            —— 取 ``run_template`` 传进来的输入（或 manifest 里的默认值）
- ``$steps.<步骤id>.<结果键>`` —— 取前序步骤的结果（最常用的是 ``$steps.project.output``）
- ``$out/<相对路径>``          —— 本次运行的输出目录下的文件

绑定只在**整串字符串**上生效；参数里的嵌套 dict/list 会逐项递归解析。
以 ``$`` 开头的字符串一律当绑定解析：写错前缀会当场报错，而不是把
``$dat.zones`` 当普通路径丢给 arcpy 埋到深处才炸。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml  # type: ignore

from ..recipes.backends import DEFAULT_BACKEND, normalize_backends

#: 模板 id / 步骤 id / 输入名的合法形态：小写蛇形（``$steps.x.y`` 才拼得干净）。
_IDENT = re.compile(r"^[a-z][a-z0-9_]*$")

#: ``data`` 里声明的期望类型。只是给人/模型看的契约，执行时不做格式强校验
#: （``.gdb/要素类`` 没有扩展名，按扩展名猜类型会误杀）。
DATA_TYPES = ("vector", "raster", "table", "file", "string", "integer", "number", "boolean")


class TemplateError(ValueError):
    """manifest 或参数绑定写错时抛给人看的错：``kind`` 进 Observation.error.type。"""

    kind = "TemplateError"


class TemplateManifestError(TemplateError):
    kind = "TemplateManifestError"


class TemplateBindingError(TemplateError):
    kind = "TemplateBindingError"


class TemplateDataError(TemplateError):
    kind = "TemplateDataError"


class TemplateBackendError(TemplateError):
    """模板/某一步在本机缺后端。类型与配方层的 ``BackendUnavailable`` 对齐。"""

    kind = "BackendUnavailable"

    def __init__(self, message: str, *, required: list[str] | None = None, available: set[str] | None = None):
        super().__init__(message)
        self.required = list(required or [])
        self.available = sorted(available or [])


@dataclass(frozen=True)
class Binding:
    """一条解析过的 ``$`` 引用。"""

    kind: str  # data / step / out
    name: str = ""  # $data.<name>
    step: str = ""  # $steps.<step>.<key>
    key: str = ""
    path: str = ""  # $out/<path>


def parse_binding(value: Any) -> Binding | None:
    """把字符串解析成绑定；不是以 ``$`` 开头 → ``None``（按字面量处理）。

    ``$`` 开头的字符串认不出时**抛错**而不是回落到字面量：模板里出现
    ``$dat.zones`` 这种笔误时，回落的后果是把 "$dat.zones" 当路径传给配方，
    报错位置离真正的错很远。
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text.startswith("$"):
        return None
    if text == "$out":
        return Binding("out", path="")
    if text.startswith("$out/"):
        return Binding("out", path=text[len("$out/"):])
    if text.startswith("$data."):
        name = text[len("$data."):]
        if _IDENT.match(name):
            return Binding("data", name=name)
    if text.startswith("$steps."):
        parts = text[len("$steps."):].split(".")
        if len(parts) == 2 and all(_IDENT.match(part) for part in parts):
            return Binding("step", step=parts[0], key=parts[1])
    raise TemplateBindingError(
        f"无法识别的绑定 {value!r}：只支持 $data.<输入名>、$steps.<步骤id>.<结果键>"
        f"（如 $steps.project.output）、$out/<文件名> 三种写法。"
    )


def resolve_bindings(
    value: Any,
    resolver: Callable[[Binding, str], Any],
    *,
    path: str = "",
) -> Any:
    """递归解析参数/断言里的绑定；``path`` 是给人看的位置（如 ``zone_path``）。"""
    if isinstance(value, str):
        binding = parse_binding(value)
        if binding is None:
            return value
        return resolver(binding, path)
    if isinstance(value, list):
        return [
            resolve_bindings(item, resolver, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: resolve_bindings(item, resolver, path=f"{path}.{key}" if path else str(key))
            for key, item in value.items()
        }
    return value


@dataclass
class TemplateInput:
    """数据清单里的一项：这个模板需要什么、干什么用、能不能不给。"""

    name: str
    purpose: str = ""
    type: str = "file"
    required: bool = True
    default: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "type": self.type,
            "required": self.required,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, name: str, raw: Any) -> "TemplateInput":
        data = raw or {}
        if not isinstance(data, dict):
            raise TemplateManifestError(f"数据输入 {name} 的声明必须是对象，实际是 {type(data).__name__}")
        kind = str(data.get("type", "file") or "file").strip().lower()
        if kind not in DATA_TYPES:
            raise TemplateManifestError(
                f"数据输入 {name} 的类型 {kind!r} 不认识（可用: {'、'.join(DATA_TYPES)}）"
            )
        return cls(
            name=name,
            purpose=str(data.get("purpose", "") or ""),
            type=kind,
            required=bool(data.get("required", True)),
            default=data.get("default"),
        )


@dataclass
class TemplateStep:
    """配方链的一步：跑哪个配方、参数怎么绑。"""

    id: str
    recipe_id: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "recipe_id": self.recipe_id, "params": dict(self.params)}


@dataclass
class Template:
    """一个民生模板：数据清单 + 配方链 + 断言集 + 后端要求。"""

    id: str
    name: str
    description: str = ""
    version: str = "1.0"
    #: 模板可用的执行后端（``[arcpy]`` / ``[arcpy, gdal]``）。缺省 arcpy。
    #: 只在"一个都跑不了"时拒绝；具体每步仍按配方自己的 backend 路由。
    backend: list[str] = field(default_factory=lambda: [DEFAULT_BACKEND])
    data: dict[str, TemplateInput] = field(default_factory=dict)
    pipeline: list[TemplateStep] = field(default_factory=list)
    assertions: list[dict[str, Any]] = field(default_factory=list)
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "backend": list(self.backend),
            "data": {name: spec.to_dict() for name, spec in self.data.items()},
            "pipeline": [step.to_dict() for step in self.pipeline],
            "assertions": [dict(item) for item in self.assertions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_path: str = "") -> "Template":
        if not isinstance(data, dict):
            raise TemplateManifestError("manifest 顶层必须是映射")
        template_id = str(data.get("id", "") or "").strip()
        if not _IDENT.match(template_id):
            raise TemplateManifestError(
                f"模板 id {template_id!r} 不合法：只允许小写字母/数字/下划线，且以字母开头"
            )

        raw_data = data.get("data") or {}
        if not isinstance(raw_data, dict):
            raise TemplateManifestError("data 必须是「输入名: 声明」的映射")
        inputs: dict[str, TemplateInput] = {}
        for name, raw in raw_data.items():
            key = str(name)
            if not _IDENT.match(key):
                raise TemplateManifestError(f"数据输入名 {key!r} 不合法（同模板 id 规则）")
            inputs[key] = TemplateInput.from_dict(key, raw)

        raw_pipeline = data.get("pipeline")
        if not isinstance(raw_pipeline, list) or not raw_pipeline:
            raise TemplateManifestError("pipeline 必须是非空列表（模板至少要有一步配方）")
        pipeline: list[TemplateStep] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_pipeline, start=1):
            if not isinstance(raw, dict):
                raise TemplateManifestError(f"pipeline 第 {index} 步必须是对象")
            step_id = str(raw.get("id", "") or "").strip()
            recipe_id = str(raw.get("recipe_id", "") or "").strip()
            if not _IDENT.match(step_id):
                raise TemplateManifestError(
                    f"pipeline 第 {index} 步的 id {step_id!r} 不合法（小写蛇形，如 project）"
                )
            if step_id in seen:
                raise TemplateManifestError(f"pipeline 里步骤 id 重复: {step_id}")
            if not recipe_id:
                raise TemplateManifestError(f"pipeline 步骤 {step_id} 缺少 recipe_id")
            params = raw.get("params") or {}
            if not isinstance(params, dict):
                raise TemplateManifestError(f"pipeline 步骤 {step_id} 的 params 必须是对象")
            seen.add(step_id)
            pipeline.append(TemplateStep(step_id, recipe_id, dict(params)))

        raw_assertions = data.get("assertions") or []
        if not isinstance(raw_assertions, list):
            raise TemplateManifestError("assertions 必须是列表")
        assertions: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_assertions, start=1):
            if not isinstance(raw, dict) or not str(raw.get("type", "") or "").strip():
                raise TemplateManifestError(f"assertions 第 {index} 条必须是带 type 的对象")
            assertions.append(dict(raw))

        return cls(
            id=template_id,
            name=str(data.get("name", "") or template_id),
            description=str(data.get("description", "") or ""),
            version=str(data.get("version", "1.0") or "1.0"),
            backend=normalize_backends(data.get("backend")),
            data=inputs,
            pipeline=pipeline,
            assertions=assertions,
            source_path=source_path,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Template":
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise TemplateManifestError(f"manifest 顶层必须是映射: {p}")
        return cls.from_dict(data, source_path=str(p))

    def match(self, query: str) -> bool:
        text = (query or "").strip().lower()
        if not text:
            return True
        return text in self.id.lower() or text in self.name.lower() or text in self.description.lower()

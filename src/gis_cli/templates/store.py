# -*- coding: utf-8 -*-
"""模板库：加载 ``templates/<id>/manifest.yaml``，按 manifest 跑配方链。

执行分成两段，纪律与 P0-1a 一致：

1. **预检（一行配方代码都不执行）**：模板后端、每一步的配方与配方后端、
   必填参数是否齐、所有 ``$`` 绑定是否指得着（输入在、步骤在前）。
2. **执行**：逐步调 ``RecipeLibrary.run`` 并按绑定解析参数；每步记录
   ``recipe_id / params / ok / output / assertions``。某步失败即停链。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..recipes.backends import (
    backend_hint,
    pick_backend,
    runner_backends,
    unavailable_message,
)
from .schema import (
    Binding,
    Template,
    TemplateBackendError,
    TemplateBindingError,
    TemplateDataError,
    TemplateError,
    resolve_bindings,
)

logger = logging.getLogger(__name__)

#: 仓库根下的模板目录（``templates/<id>/manifest.yaml``）。
#: store.py -> gis_cli/templates -> gis_cli -> src -> 仓库根。
DEFAULT_TEMPLATES_DIR = Path(__file__).resolve().parents[3] / "templates"


def _out_path_problem(path: str) -> str | None:
    """``$out/<path>`` 只允许留在输出目录里：绝对路径与 ``..`` 都不行。

    Windows 上 ``/etc/x.shp`` 是「根锚定但没有盘符」的路径：
    ``Path('/etc/x.shp').is_absolute()`` 是 ``False``，但 ``out_dir / '/etc/x.shp'``
    会丢掉左边整段落到当前盘根，所以无盘符的根锚定也要拦。
    """
    normalized = path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if (
        Path(path).is_absolute()
        or Path(path).drive
        or normalized.startswith("/")
        or ".." in parts
    ):
        return "$out 下面只允许相对路径，不能跳出输出目录。"
    return None


class TemplateLibrary:
    """加载模板 manifest，并按 manifest 跑配方链。"""

    def __init__(
        self,
        *,
        templates_dir: str | Path | None = None,
        recipes: Any = None,
        code_runner: Any = None,
    ):
        self.templates_dir = Path(templates_dir) if templates_dir else DEFAULT_TEMPLATES_DIR
        self.recipes = recipes
        self.code_runner = code_runner
        self._templates: dict[str, Template] = {}
        self._loaded = False

    # ------------------------------------------------------------------- load
    def load(self) -> int:
        """（重新）加载 ``templates/*/manifest.yaml``；坏 manifest 只记日志，不拖垮整库。"""
        self._templates = {}
        self._loaded = True
        if not self.templates_dir.exists():
            return 0
        for path in sorted(self.templates_dir.glob("*/manifest.y*ml")):
            try:
                template = Template.from_yaml(path)
            except Exception as exc:
                logger.warning("failed to load template %s: %s", path, exc)
                continue
            self._templates[template.id] = template
        return len(self._templates)

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------ query
    def get(self, template_id: str) -> Template | None:
        self.ensure_loaded()
        return self._templates.get(template_id)

    def all(self) -> list[Template]:
        self.ensure_loaded()
        return sorted(self._templates.values(), key=lambda item: item.id)

    def search(self, query: str = "", limit: int = 20) -> list[Template]:
        self.ensure_loaded()
        return [item for item in self.all() if item.match(query)][:limit]

    # -------------------------------------------------------------------- run
    def run(
        self,
        template_id: str,
        data: dict[str, Any],
        ctx: Any,
        *,
        output_dir: str | Path | None = None,
        timeout: float | None = None,
    ) -> Any:
        """按 manifest 跑配方链，返回带逐步记录的 ``Observation``。"""
        from ..engine.state import Observation

        template = self.get(template_id)
        if template is None:
            known = ", ".join(item.id for item in self.all()) or "（空）"
            return Observation(
                ok=False,
                summary=f"模板不存在: {template_id}",
                error={"type": "TemplateNotFound", "message": f"可用模板: {known}"},
            )
        recipes = getattr(ctx, "recipes", None) or self.recipes
        if recipes is None:
            return Observation(ok=False, summary="模板库没有可用的配方库", error={"type": "NoRecipes"})
        runner = getattr(ctx, "code_runner", None) or self.code_runner
        available = runner_backends(runner)

        # 模板级后端门禁：先于一切，缺后端时一行代码都不执行。
        backend = pick_backend(template.backend, available)
        if backend is None:
            message = unavailable_message(template.backend, available)
            return Observation(
                ok=False,
                summary=f"模板 {template.id} 在本机跑不了: {message}",
                error={
                    "type": "BackendUnavailable",
                    "message": message,
                    "template": template.id,
                    "required": list(template.backend),
                    "available": sorted(available or []),
                },
                hint=backend_hint(template.backend),
            )

        try:
            inputs = self._resolve_inputs(template, data)
            out_dir = self._output_dir(ctx, template, output_dir)
            plan = self._preflight(template, recipes, inputs, available)
        except TemplateError as exc:
            return self._error(exc)

        out_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, dict[str, Any]] = {}
        step_records: list[dict[str, Any]] = []

        def resolve(binding: Binding, path: str) -> Any:
            if binding.kind == "data":
                if binding.name not in inputs:
                    raise TemplateDataError(
                        f"模板 {template.id} 的参数 {path} 引用了 $data.{binding.name}，"
                        "但这个可选输入这次没有提供。"
                    )
                return inputs[binding.name]
            if binding.kind == "out":
                if not binding.path:
                    return str(out_dir)
                return str(out_dir / binding.path)
            record = results.get(binding.step)
            if record is None:
                raise TemplateBindingError(
                    f"模板 {template.id} 的参数 {path} 引用了 $steps.{binding.step}.{binding.key}，"
                    f"但步骤 {binding.step} 还没有结果（步骤顺序写错了？）。"
                )
            payload = record["payload"] if isinstance(record["payload"], dict) else {}
            if binding.key in payload:
                return payload[binding.key]
            if binding.key == "output" and record["artifacts"]:
                return record["artifacts"][0]
            known = ", ".join(sorted(payload)) or "（无）"
            raise TemplateBindingError(
                f"模板 {template.id} 的参数 {path} 引用了 $steps.{binding.step}.{binding.key}，"
                f"但步骤 {binding.step} 的结果里没有这个键（可用: {known}）。"
            )

        for index, (step, _recipe) in enumerate(plan, start=1):
            try:
                params = resolve_bindings(step.params, resolve)
            except TemplateError as exc:
                return self._error(exc, template, step_records)
            observation = recipes.run(step.recipe_id, params, ctx, timeout=timeout)
            payload = observation.data.get("result") if isinstance(observation.data, dict) else None
            step_backend = observation.data.get("backend") if isinstance(observation.data, dict) else None
            artifacts = [str(item) for item in (observation.artifacts or [])]
            output = None
            if isinstance(payload, dict) and payload.get("output"):
                output = str(payload["output"])
            elif artifacts:
                output = artifacts[0]
            record = {
                "index": index,
                "step": step.id,
                "recipe_id": step.recipe_id,
                "params": params,
                "ok": bool(observation.ok),
                "backend": step_backend,
                "output": output,
                "artifacts": artifacts,
                "assertions": (
                    observation.data.get("assertions", []) if isinstance(observation.data, dict) else []
                ),
                "summary": observation.summary,
            }
            if observation.error:
                record["error"] = observation.error
            step_records.append(record)
            results[step.id] = {"payload": payload, "artifacts": artifacts}
            if not observation.ok:
                return Observation(
                    ok=False,
                    summary=(
                        f"模板 {template.id} 在第 {index} 步 ({step.id}/{step.recipe_id}) 失败: "
                        f"{observation.summary}"
                    ),
                    data=self._payload(template, backend, out_dir, step_records, []),
                    artifacts=self._artifacts(step_records),
                    error=observation.error or {"type": "StepFailed", "message": observation.summary},
                    hint=observation.hint,
                )

        # 模板级断言：全部步骤跑完后统一判定，用模板自己的后端选断言实现。
        try:
            raw_assertions = [resolve_bindings(item, resolve) for item in template.assertions]
        except TemplateError as exc:
            return self._error(exc, template, step_records)
        checks = recipes.run_assertions(raw_assertions, backend=backend)
        failed = [check for check in checks if not check.ok]
        summary = f"模板 {template.id} 跑通：{len(step_records)} 步"
        if checks:
            summary += f"，断言 {len(checks) - len(failed)}/{len(checks)} 通过"
        if failed:
            summary += "；未通过: " + "；".join(
                f"{check.kind}({check.target}) {check.detail}" for check in failed[:3]
            )
        return Observation(
            ok=not failed,
            summary=summary,
            data=self._payload(
                template, backend, out_dir, step_records, [check.to_dict() for check in checks]
            ),
            artifacts=self._artifacts(step_records),
            hint="" if not failed else "按未通过的模板断言检查上游参数与数据，再重跑模板。",
        )

    # ---------------------------------------------------------------- helpers
    def _payload(
        self,
        template: Template,
        backend: str,
        out_dir: Path,
        steps: list[dict[str, Any]],
        assertions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "template": template.to_dict(),
            "backend": backend,
            "output_dir": str(out_dir),
            "steps": steps,
            "assertions": assertions,
        }

    @staticmethod
    def _artifacts(steps: list[dict[str, Any]]) -> list[str]:
        found: list[str] = []
        for step in steps:
            for item in step.get("artifacts") or []:
                if item not in found:
                    found.append(item)
        return found

    @staticmethod
    def _error(
        exc: TemplateError,
        template: Template | None = None,
        steps: list[dict[str, Any]] | None = None,
    ) -> Any:
        from ..engine.state import Observation

        payload = None
        if template is not None:
            payload = {"template": template.to_dict(), "steps": list(steps or [])}
        error: dict[str, Any] = {"type": exc.kind, "message": str(exc)}
        required = getattr(exc, "required", None)
        if required:
            error["required"] = required
            error["available"] = getattr(exc, "available", [])
        return Observation(
            ok=False,
            summary=str(exc),
            data=payload,
            error=error,
            hint=backend_hint(required) if required else "",
        )

    @staticmethod
    def _output_dir(ctx: Any, template: Template, raw: str | Path | None) -> Path:
        workspace = Path(getattr(ctx, "workspace", "."))
        if raw is None or str(raw).strip() == "":
            return workspace / "output" / template.id
        path = Path(raw)
        return path if path.is_absolute() else workspace / path

    def _resolve_inputs(self, template: Template, data: dict[str, Any]) -> dict[str, Any]:
        """核对输入清单：多给的、少给的都报人话，不把错误拖到配方参数里。"""
        provided = dict(data or {})
        unknown = [name for name in provided if name not in template.data]
        if unknown:
            known = "、".join(template.data) or "（无）"
            raise TemplateDataError(
                f"模板 {template.id} 没有这些输入: {'、'.join(unknown)}（可用: {known}）"
            )
        values: dict[str, Any] = {}
        missing: list[str] = []
        for name, spec in template.data.items():
            if name in provided and provided[name] is not None:
                value = provided[name]
                values[name] = str(value) if isinstance(value, Path) else value
            elif spec.default is not None:
                values[name] = spec.default
            elif spec.required:
                missing.append(f"{name}（{spec.purpose or spec.type}）")
        if missing:
            raise TemplateDataError(
                f"模板 {template.id} 缺少必需输入: {'、'.join(missing)}。"
                "先用 list_templates 看数据清单，再通过 data 参数传入。"
            )
        return values

    def _preflight(
        self,
        template: Template,
        recipes: Any,
        inputs: dict[str, Any],
        available: set[str] | None,
    ) -> list[tuple[Any, Any]]:
        """执行前的全部静态检查：配方在、后端在、必填参数齐、绑定指得着。"""
        plan: list[tuple[Any, Any]] = []
        earlier: list[str] = []
        for index, step in enumerate(template.pipeline, start=1):
            recipe = recipes.get(step.recipe_id)
            if recipe is None:
                raise TemplateBindingError(
                    f"模板 {template.id} 的第 {index} 步 ({step.id}) 引用了不存在的配方: "
                    f"{step.recipe_id}"
                )
            step_backend = pick_backend(recipe.backend, available)
            if step_backend is None:
                message = unavailable_message(recipe.backend, available)
                raise TemplateBackendError(
                    f"模板 {template.id} 的第 {index} 步 ({step.id}/{step.recipe_id}) 在本机跑不了: "
                    f"{message}",
                    required=recipe.backend,
                    available=available,
                )
            try:
                recipe.template_for(step_backend)
            except ValueError as exc:
                raise TemplateBindingError(
                    f"模板 {template.id} 的第 {index} 步 ({step.id}) 配方不可用: {exc}"
                ) from exc

            missing = [
                name
                for name, spec in recipe.params.items()
                if spec.required and name not in step.params and spec.default is None
            ]
            if missing:
                raise TemplateBindingError(
                    f"模板 {template.id} 的第 {index} 步 ({step.id}) 没给配方 {step.recipe_id} 的必填参数: "
                    f"{'、'.join(missing)}"
                )
            unknown = [name for name in step.params if name not in recipe.params]
            if unknown:
                raise TemplateBindingError(
                    f"模板 {template.id} 的第 {index} 步 ({step.id}) 给配方 {step.recipe_id} 传了不认识的参数: "
                    f"{'、'.join(unknown)}（可用: {'、'.join(recipe.params) or '无'}）"
                )

            def check(binding: Binding, path: str, _step=step, _index=index, _earlier=earlier) -> None:
                where = f"模板 {template.id} 的第 {_index} 步 ({_step.id})"
                if binding.kind == "data":
                    if binding.name not in template.data:
                        raise TemplateDataError(
                            f"{where} 的参数 {path} 引用了 $data.{binding.name}，"
                            f"但 manifest 的 data 里没有这个输入（已声明: {'、'.join(template.data) or '无'}）。"
                        )
                    if binding.name not in inputs:
                        raise TemplateDataError(
                            f"{where} 的参数 {path} 引用了 $data.{binding.name}，"
                            "但这个可选输入这次没有提供。"
                        )
                elif binding.kind == "step":
                    if binding.step not in _earlier:
                        raise TemplateBindingError(
                            f"{where} 的参数 {path} 引用了 $steps.{binding.step}.{binding.key}，"
                            f"但这一步之前没有名为 '{binding.step}' 的步骤"
                            f"（可用: {'、'.join(_earlier) or '无'}）。"
                            "模板只能引用**前面**的步骤。"
                        )
                elif binding.kind == "out":
                    problem = _out_path_problem(binding.path)
                    if problem:
                        raise TemplateBindingError(
                            f"{where} 的参数 {path} 用了 $out/{binding.path}：{problem}"
                        )

            resolve_bindings(step.params, check)
            plan.append((step, recipe))
            earlier.append(step.id)

        # 模板断言的绑定同样先查一遍（此时各步骤都已在前，任何 $steps 引用都合法）。
        for index, assertion in enumerate(template.assertions, start=1):
            def check_assertion(binding: Binding, path: str) -> None:
                where = f"模板 {template.id} 的第 {index} 条断言（{path}）"
                if binding.kind == "data":
                    if binding.name not in template.data:
                        raise TemplateDataError(
                            f"{where}引用了 $data.{binding.name}，manifest 的 data 里没有这个输入"
                            f"（已声明: {'、'.join(template.data) or '无'}）。"
                        )
                    if binding.name not in inputs:
                        raise TemplateDataError(
                            f"{where}引用了 $data.{binding.name}，但这个可选输入这次没有提供。"
                        )
                if binding.kind == "step" and binding.step not in earlier:
                    raise TemplateBindingError(
                        f"{where}引用了 $steps.{binding.step}.{binding.key}，"
                        f"但模板里没有名为 '{binding.step}' 的步骤（可用: {'、'.join(earlier) or '无'}）。"
                    )
                if binding.kind == "out":
                    problem = _out_path_problem(binding.path)
                    if problem:
                        raise TemplateBindingError(f"{where}的 $out/{binding.path} 不合法：{problem}")

            resolve_bindings(assertion, check_assertion)
        return plan

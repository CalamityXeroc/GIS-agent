# -*- coding: utf-8 -*-
"""Engine tool registry and the built-in tool set.

The engine exposes a small, high-leverage tool surface. GIS work itself is
done through ``execute_code`` (CodeAct); the other tools handle control flow,
data awareness, methodology reuse, and verification.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .state import Observation

logger = logging.getLogger(__name__)

GIS_OUTPUT_EXT = {
    ".shp", ".gdb", ".pdf", ".png", ".tif", ".tiff", ".csv", ".xlsx",
    ".dbf", ".gpkg", ".lyrx", ".mapx", ".jpg", ".jpeg", ".bmp", ".svg", ".aprx",
}


#: 自写代码标志 → 可闭环该操作的已验证配方（代码反复失败时的引导）
_CODE_MARKER_RECIPES = (
    ("exporttojpeg", "graduated_colors_map"),
    ("arccgisproject(", "graduated_colors_map"),
    ("sym.renderer", "graduated_colors_map"),
)


def _recipe_hint(ctx: EngineContext, description: str, code: str) -> str:
    """自写代码失败时提示可替代的已验证配方，避免无限修补自写代码。"""
    if getattr(ctx, "recipes", None) is None:
        return ""
    norm = (code or "").lower().replace(" ", "")
    for marker, rid in _CODE_MARKER_RECIPES:
        if marker in norm and ctx.recipes.get(rid) is not None:
            return (
                "已有验证过的配方「%s」可覆盖此操作（自写地图/符号化代码易反复出错）。"
                "强烈建议改用 run_recipe(recipe_id=%s) 完成，参数见 list_recipes。" % (rid, rid)
            )
    if description and description.strip():
        best, best_score = None, 0
        try:
            for recipe in ctx.recipes.all():
                score = recipe.match_score(description)
                if score > best_score:
                    best, best_score = recipe, score
        except Exception:
            return ""
        if best is not None and best_score >= 6:
            return (
                "已有验证过的配方「%s」与此操作描述高度匹配。"
                "若自写代码持续失败，建议改用 run_recipe(recipe_id=%s) 完成。" % (best.id, best.id)
            )
    return ""


def extract_outputs(value: Any, workspace: str | Path, *, limit: int = 30) -> list[str]:
    """Walk a set_result payload and collect GIS output paths."""
    workspace_path = Path(workspace)
    outputs: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        text = candidate.strip()
        if not text or text in seen:
            return
        lowered = text.lower().replace("\\", "/")
        is_gis = any(lowered.endswith(ext) or f"{ext}/" in lowered for ext in GIS_OUTPUT_EXT)
        if ".gdb/" in lowered:
            is_gis = True
        if not is_gis:
            return
        seen.add(text)
        outputs.append(text)

    def _walk(node: Any) -> None:
        if len(outputs) >= limit:
            return
        if isinstance(node, str):
            _add(node)
        elif isinstance(node, dict):
            for item in node.values():
                _walk(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)

    _walk(value)
    return outputs


# ------------------------------------------------------------------ registry
@dataclass
class EngineTool:
    """A tool the engine loop can call."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], "EngineContext"], Observation]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class EngineToolRegistry:
    """Registry with OpenAI-format schema export."""

    def __init__(self) -> None:
        self._tools: dict[str, EngineTool] = {}

    def register(self, tool: EngineTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> EngineTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self._tools.values()]

    def execute(self, name: str, args: dict[str, Any], ctx: "EngineContext") -> Observation:
        tool = self._tools.get(name)
        if tool is None:
            return Observation(
                ok=False,
                summary=f"未知工具: {name}",
                error={"type": "UnknownTool", "message": f"Available: {', '.join(self.names())}"},
            )
        try:
            return tool.handler(args or {}, ctx)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("tool %s crashed", name)
            return Observation(
                ok=False,
                summary=f"工具 {name} 内部异常: {exc}",
                error={"type": type(exc).__name__, "message": str(exc)},
            )


@dataclass
class EngineContext:
    """Everything a tool handler may need."""

    workspace: Path
    catalog: Any = None
    code_runner: Any = None
    recipes: Any = None
    verifier: Any = None
    state: Any = None
    kernel: Any = None
    llm: Any = None
    output_dir: Path = field(default_factory=lambda: Path("workspace/output"))
    backup_dir: Path = field(default_factory=lambda: Path("workspace/.gis_agent/backups"))
    auto_backup: bool = True
    exec_timeout: float = 600.0
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- tools
def build_default_registry() -> EngineToolRegistry:
    registry = EngineToolRegistry()

    # ---------------------------------------------------------- catalog_query
    def _catalog_query(args: dict[str, Any], ctx: EngineContext) -> Observation:
        from ..datacatalog.summary import build_digest, search_catalog

        if ctx.catalog is None:
            return Observation(ok=False, summary="数据目录不可用", error={"type": "NoCatalog"})
        query = str(args.get("query", "") or "")
        kind = args.get("kind")
        limit = int(args.get("limit", 20) or 20)
        records = search_catalog(ctx.catalog, query=query, kind=kind, limit=limit)
        digest = build_digest(ctx.catalog, max_datasets=30, with_samples=False)
        return Observation(
            ok=True,
            summary=f"数据目录查询：{len(records)} 条匹配" + (f"（关键词: {query}）" if query else ""),
            data={"matches": records, "digest": digest},
        )

    registry.register(
        EngineTool(
            name="catalog_query",
            description=(
                "查询已扫描的数据目录，获取数据集的路径、几何类型、坐标系、要素数、字段名/类型/样本。"
                "在写任何 ArcPy 代码之前，先用它确认真实字段名和坐标系，禁止猜测字段名。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "按名称/路径关键词过滤，可空"},
                    "kind": {"type": "string", "description": "vector / feature_class / raster / table / project"},
                    "limit": {"type": "integer", "description": "最多返回条数，默认 20"},
                },
            },
            handler=_catalog_query,
        )
    )

    # ------------------------------------------------------------ execute_code
    def _execute_code(args: dict[str, Any], ctx: EngineContext) -> Observation:
        from ..runtime.code_runner import backup_paths, looks_destructive

        code = str(args.get("code", "") or "")
        if not code.strip():
            return Observation(ok=False, summary="代码为空", error={"type": "EmptyCode"})
        description = str(args.get("description", "") or "")
        timeout = float(args.get("timeout_seconds", ctx.exec_timeout) or ctx.exec_timeout)

        if ctx.auto_backup and looks_destructive(code):
            candidates = _referenced_existing_paths(code, ctx.workspace)
            if candidates:
                created = backup_paths(candidates, ctx.backup_dir)
                if created:
                    logger.info("backed up before destructive code: %s", created)

        result = ctx.code_runner.run(code, timeout=timeout, workspace=str(ctx.workspace))
        artifacts = extract_outputs(result.result, ctx.workspace) if result.ok else []
        if result.ok:
            summary = f"执行成功（{result.duration_ms} ms）" + (f"：{description}" if description else "")
            if artifacts:
                summary += f"；产出 {len(artifacts)} 个文件"
            for path in artifacts:
                if ctx.state is not None:
                    ctx.state.add_artifact(path)
            return Observation(
                ok=True,
                summary=summary,
                data={"stdout": result.stdout[-4000:], "result": result.result},
                artifacts=artifacts,
                images=result.display_images[:3],
            )
        error = result.error or {"type": "Unknown", "message": "execution failed"}
        hint = "请根据 traceback 修正代码后重试；优先检查字段名/坐标系/路径是否存在。"
        recipe_hint = _recipe_hint(ctx, description, code)
        summary = f"代码执行失败: [{error.get('type')}] {error.get('message')}"
        if recipe_hint:
            hint = recipe_hint + " 其次：" + hint
            summary += f"；{recipe_hint}"
        return Observation(
            ok=False,
            summary=summary,
            data={"stdout": (result.stdout or "")[-3000:], "stderr": (result.stderr or "")[-2000:]},
            error=error,
            hint=hint,
        )

    registry.register(
        EngineTool(
            name="execute_code",
            description=(
                "在 ArcGIS Pro 的持久 Python 内核中执行 ArcPy 代码。代码里可直接 import arcpy，"
                "并用 set_result(字典) 返回结果（建议包含 output 路径）。失败会返回完整错误与 traceback，"
                "应据此修正代码重试。这是完成 GIS 分析/制图的主要手段。"
                "所有产出必须写入 workspace/output 目录（可用工具上下文里的 workspace 拼出）。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的完整 Python/ArcPy 代码"},
                    "description": {"type": "string", "description": "这一步做什么（用于日志与报告）"},
                    "timeout_seconds": {"type": "integer", "description": "超时秒数，默认 600"},
                },
                "required": ["code"],
            },
            handler=_execute_code,
        )
    )

    # ---------------------------------------------------------- read_document
    def _read_document(args: dict[str, Any], ctx: EngineContext) -> Observation:
        from ..requirements.extractor import read_document_text

        path = str(args.get("path", "") or "")
        max_chars = int(args.get("max_chars", 20000) or 20000)
        target = Path(path)
        if not target.is_absolute():
            target = ctx.workspace / path
        if not target.exists():
            return Observation(
                ok=False,
                summary=f"文档不存在: {target}",
                error={"type": "FileNotFound", "message": str(target)},
            )
        try:
            text = read_document_text(target, max_chars)
        except Exception as exc:
            return Observation(
                ok=False,
                summary=f"文档解析失败: {exc}",
                error={"type": type(exc).__name__, "message": str(exc)},
            )
        return Observation(
            ok=True,
            summary=f"已读取 {target.name}（{len(text)} 字符）",
            data={"path": str(target), "text": text, "total_chars": len(text)},
        )

    registry.register(
        EngineTool(
            name="read_document",
            description="读取 docx/pdf/txt 文档正文，用于解析任务书、需求文档。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文档路径（可相对 workspace）"},
                    "max_chars": {"type": "integer", "description": "最多返回字符数，默认 20000"},
                },
                "required": ["path"],
            },
            handler=_read_document,
        )
    )

    # -------------------------------------------------------------- run_recipe
    def _run_recipe(args: dict[str, Any], ctx: EngineContext) -> Observation:
        if ctx.recipes is None:
            return Observation(ok=False, summary="配方库不可用", error={"type": "NoRecipes"})
        recipe_id = str(args.get("recipe_id", "") or "")
        params = args.get("params") or {}
        if not isinstance(params, dict):
            return Observation(ok=False, summary="params 必须是对象", error={"type": "BadParams"})
        return ctx.recipes.run(recipe_id, params, ctx)

    registry.register(
        EngineTool(
            name="run_recipe",
            description=(
                "执行已验证的 GIS 方法论配方（如投影、分级设色、缓冲区、2SFCA 可达性）。"
                "先用 list_recipes 查看可用配方与参数；配方执行后会自动跑其自带的验证断言。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "recipe_id": {"type": "string", "description": "配方 id"},
                    "params": {"type": "object", "description": "配方参数键值对"},
                },
                "required": ["recipe_id", "params"],
            },
            handler=_run_recipe,
        )
    )

    # ------------------------------------------------------------ list_recipes
    def _list_recipes(args: dict[str, Any], ctx: EngineContext) -> Observation:
        if ctx.recipes is None:
            return Observation(ok=False, summary="配方库不可用", error={"type": "NoRecipes"})
        query = str(args.get("query", "") or "")
        items = ctx.recipes.search(query)
        return Observation(
            ok=True,
            summary=f"找到 {len(items)} 个配方",
            data={"recipes": items},
        )

    registry.register(
        EngineTool(
            name="list_recipes",
            description="按关键词检索 GIS 方法论配方，返回配方 id、用途、参数与前置条件。",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "关键词，可空表示全部"}},
            },
            handler=_list_recipes,
        )
    )

    # --------------------------------------------------------- verify_outputs
    def _verify_outputs(args: dict[str, Any], ctx: EngineContext) -> Observation:
        if ctx.verifier is None:
            return Observation(ok=False, summary="验证器不可用", error={"type": "NoVerifier"})
        verdict = ctx.verifier.verify(ctx.state)
        return Observation(
            ok=bool(verdict.get("pass")),
            summary=verdict.get("summary", ""),
            data=verdict,
            hint=verdict.get("repair_hint", ""),
        )

    registry.register(
        EngineTool(
            name="verify_outputs",
            description="对当前产出做结构化验收（文件存在/要素数/坐标系/字段/栅格/图片非空白），返回逐条结论。",
            parameters={"type": "object", "properties": {}},
            handler=_verify_outputs,
        )
    )

    # --------------------------------------------------------------- ask_user
    def _ask_user(args: dict[str, Any], ctx: EngineContext) -> Observation:
        question = str(args.get("question", "") or "").strip()
        options = args.get("options") or []
        if not question:
            return Observation(ok=False, summary="question 不能为空", error={"type": "BadParams"})
        return Observation(
            ok=True,
            summary=f"等待用户回答: {question}",
            data={"awaiting_user": True, "question": question, "options": options},
        )

    registry.register(
        EngineTool(
            name="ask_user",
            description=(
                "向用户提问以澄清需求（数据位置、目标坐标系、输出格式等）。"
                "只在信息缺失且无法从数据目录推断时使用；能自己查就不要问。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要问的问题"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选项（可选）",
                    },
                },
                "required": ["question"],
            },
            handler=_ask_user,
        )
    )

    # --------------------------------------------------------- update_tasklist
    def _update_tasklist(args: dict[str, Any], ctx: EngineContext) -> Observation:
        items = args.get("items") or []
        if ctx.state is None:
            return Observation(ok=False, summary="状态不可用", error={"type": "NoState"})
        ctx.state.apply_tasklist(items)
        return Observation(
            ok=True,
            summary="任务清单已更新",
            data={"tasklist": [item.to_dict() for item in ctx.state.tasklist]},
        )

    registry.register(
        EngineTool(
            name="update_tasklist",
            description=(
                "建立/更新可见任务清单。开始任务时先列出计划，之后每完成一步就更新状态。"
                "status 取值：pending/running/done/failed/blocked/skipped。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "status": {"type": "string"},
                                "evidence": {"type": "string"},
                                "detail": {"type": "string"},
                            },
                            "required": ["id", "title"],
                        },
                    }
                },
                "required": ["items"],
            },
            handler=_update_tasklist,
        )
    )

    # ----------------------------------------------------------------- finish
    def _finish(args: dict[str, Any], ctx: EngineContext) -> Observation:
        summary = str(args.get("summary", "") or "").strip()
        methodology = str(args.get("methodology", "") or "").strip()
        artifacts = args.get("artifacts") or []
        if ctx.state is not None:
            ctx.state.final_summary = summary
            ctx.state.methodology = methodology
            for path in artifacts:
                ctx.state.add_artifact(str(path))
        return Observation(
            ok=True,
            summary=f"准备交付：{summary[:200]}",
            data={"finished": True, "summary": summary, "methodology": methodology, "artifacts": artifacts},
        )

    registry.register(
        EngineTool(
            name="finish",
            description=(
                "宣布任务完成并提交交付说明。调用后系统会自动运行验收；若验收不通过，会带着缺口让你继续修复。"
                "summary 用中文说明做了什么、产出在哪里；methodology 说明采用的 GIS 方法论。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "结果说明"},
                    "methodology": {"type": "string", "description": "采用的 GIS 方法论与关键参数"},
                    "artifacts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "交付文件路径列表",
                    },
                },
                "required": ["summary"],
            },
            handler=_finish,
        )
    )

    return registry


_QUOTED_PATH = re.compile(r"""["']([A-Za-z]:[\\/][^"'<>|*?\n]+|\.{0,2}[\\/][^"'<>|*?\n]+)["']""")


def _referenced_existing_paths(code: str, workspace: Path) -> list[str]:
    """Best-effort: existing paths referenced by destructive code."""
    found: list[str] = []
    for match in _QUOTED_PATH.finditer(code):
        raw = match.group(1)
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = workspace / raw
        if candidate.exists() and str(candidate) not in found:
            found.append(str(candidate))
    return found

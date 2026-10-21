# -*- coding: utf-8 -*-
"""Engine tool registry and the built-in tool set.

The engine exposes a small, high-leverage tool surface. GIS work itself is
done through ``execute_code`` (CodeAct); the other tools handle control flow,
data awareness, methodology reuse, and verification.
"""

from __future__ import annotations

import json
import time
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .state import Observation

logger = logging.getLogger(__name__)

#: 产出文件的扩展名（统一一处，避免 tools 与 verifier 各写一份而漏格式）。
#: 实测坑：以前两份都只列 GIS/图像格式，**没有 .docx/.xlsx/.pptx**，
#: 于是“写个 Word 文档”这类非 GIS 产出无法被登记也无法被发现，
#: 验收永远报“没有任何产出文件”。
GIS_OUTPUT_EXT = {
    # 矢量 / 栅格 / 表格
    ".shp", ".gdb", ".tif", ".tiff", ".img", ".dbf", ".gpkg", ".csv", ".xlsx", ".xls",
    # 地图与工程
    ".lyrx", ".mapx", ".aprx", ".mxd",
    # 图片
    ".png", ".jpg", ".jpeg", ".bmp", ".svg", ".gif", ".webp",
    # 文档（技能产出：写报告 / Word / PDF / PPT）
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".md", ".txt", ".html",
    # 制图设计规格（graduated_colors_map 等配方回写，验收要读它核对分级一致）；
    # 以前漏了 .json，4 份 .spec.json 明明落在 output/成果图/ 下却从未进产物清单，
    # 语义验收看不到就说“缺失 .spec.json”，把 agent 逼到反复重做（14 届 4a-1 实测）。
    ".json",
}

#: 归档/中间产物后缀：即使名字里带上面某个扩展名也不算产出
_NON_OUTPUT_EXT = {".tmp", ".bak", ".log", ".pkl", ".pyc"}


#: 自写代码标志 → 可闭环该操作的已验证配方（代码反复失败时的引导）
_CODE_MARKER_RECIPES = (
    ("exporttojpeg", "graduated_colors_map"),
    ("arccgisproject(", "graduated_colors_map"),
    ("sym.renderer", "graduated_colors_map"),
)


def _stderr_warning_hint(stderr_text: str) -> str:
    """把成功执行时的 stderr 告警转成修复提示（复用错误记忆规则）。

    典型场景：matplotlib 缺中文字形（图上中文变方框），这类告警只写 stderr，
    若不回喂模型，交付的图再难也无人发现。无命中时返回空串。
    """
    if not stderr_text:
        return ""
    try:
        from ..runtime.error_memory import enrich_hint

        return enrich_hint("", stderr_text)
    except Exception as exc:  # pragma: no cover - 记忆模块不影响主流程
        logger.debug("stderr warning hint failed: %s", exc)
        return ""


def _recipe_soft_hint(ctx: EngineContext, description: str, code: str) -> str:
    """代码执行成功，但该操作其实有自带断言的配方时，提一句（不阻断）。"""
    if getattr(ctx, "recipes", None) is None:
        return ""
    norm = (code or "").lower().replace(" ", "")
    markers = (
        ("analysis.intersect(", "intersect_layers"),
        ("dissolve(", "dissolve_with_stats"),
        ("zonalstatisticsastable(", "zonal_statistics"),
        ("zonalstatistics(", "zonal_statistics_raster"),
        ("polygontoraster(", "polygon_to_raster"),
        ("analysis.buffer(", "buffer_dissolve"),
        ("slope(", "terrain_derivatives"),
        ("idw(", "interpolate_surface"),
    )
    for marker, rid in markers:
        if marker in norm and ctx.recipes.get(rid) is not None:
            recipe = ctx.recipes.get(rid)
            return (
                f"提示（不阻断）：此操作对应已验证配方「{rid}」（{recipe.name}），"
                f"配方自带数据断言与踩坑说明；同类操作下次可直接 run_recipe(recipe_id={rid})，更稳。"
            )
    return ""


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
    """Walk a set_result payload and collect GIS output paths.

    只收**产物**：workspace 之外的路径、以及 ``input/`` ``temp/`` 里的路径都不算。

    实测两个坑：
    1. 模型习惯把读过的输入也写进 set_result，若当成产物登记，验收会报
       “产出不在 output 目录内”而判定失败（benchmark 的 4b 就栽在这）。
    2. 模型会把工具回参里的**外部**工程当成产物，例如
       ``E:\\ArcGISPro3.6\\...\\Blank.aprx``、用户 Documents 下的 MyProject.aprx。
       这些是 ArcGIS 自带的模板工程，一登记就让验收永远报“产出不在 output 目录内”，
       把 agent 逼到 130 轮上限（14 届 4a-1 实测：断言 7/7 全过却 status=max_turns）。
       产物必须在 workspace 内，所以这里直接以 workspace 为界拒掉外部路径。
    """
    workspace_path = Path(workspace).resolve()
    outputs: list[str] = []
    seen: set[str] = set()

    def _is_input_path(rel: str) -> bool:
        """workspace 相对路径是否落在输入/中间目录里。

        只看目录段（不看文件名）：input/animal_points.shp 是输入，而
        output/animal_points.shp 是产物。
        """
        dirs = [seg for seg in rel.split("/") if seg][:-1]  # 去掉文件名
        return any(seg in ("input", "temp") for seg in dirs)

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
        # 产物必须落在 workspace 内：外部路径（ArcGIS 模板、用户工程）不是本次产出。
        #
        # 注意：模型给的常常是**相对路径**（如 output\result_data.gdb\kde_1982_1986），
        # 而 Path.resolve() 会把相对路径按**当前工作目录**展开——不是 workspace，
        # 于是 relative_to 抛异常、产物被当成外部路径丢掉。实测（14 届 4a-1）：
        # finish 里列的 4 个 GDB 栅格就是这么没的，验收只能说“缺失栅格元数据”，
        # agent 空转到轮次上限。所以相对路径先拼 workspace 再解析。
        raw_path = Path(text)
        absolute = raw_path if raw_path.is_absolute() else workspace_path / raw_path
        try:
            rel = str(absolute.resolve().relative_to(workspace_path)).lower()
        except Exception:
            return
        rel = rel.replace("\\", "/")
        if _is_input_path(rel):
            return
        # 登记绝对路径：后面验收（可能跑在别的进程/内核里）拿到相对路径会解析错目录。
        seen.add(text)
        outputs.append(str(absolute))

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
    """A tool the engine loop can call.

    ``readonly`` 标记 = 这个工具**不产生副作用、也不碰内核**，可以与其他
    readonly 工具并发执行。判定标准要严：

    - 只读文件/目录/数据库；
    - **不跑 arcpy 代码**（内核已串行化，并发进去只会排队，徒增复杂度）；
    - **不调 LLM**（会打满并发配额，反而拖慢主链路）；
    - 不改任务状态（tasklist/finish/ask_user）。

    默认 False（保守）：新加工具忘了标记只是退回串行，不会错。
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], "EngineContext"], Observation]
    readonly: bool = False

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

    def is_readonly(self, name: str) -> bool:
        """该工具能否与其他只读工具并发（未注册的工具一律当 False）。"""
        tool = self._tools.get(name)
        return bool(tool is not None and tool.readonly)

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
def _resolve_path(ctx: Any, raw: str) -> tuple[Path | None, list[Path]]:
    """把模型给的路径解析成真实文件（返回 (命中路径, 尝试过的候选)）。

    实测坑：模型常把工作区名写进路径（如 ``workspace/input/x.pdf``），
    旧的 ``ctx.workspace / path`` 会拼成 ``workspace/workspace/input/x.pdf``，
    于是报"文档不存在"，并把错误路径回喂给模型，越修越偏。

    策略：绝对路径优先；相对路径依次尝试
    ① 工作区下 ② 当前目录 ③ 工作区父目录下（覆盖"已含工作区名"）
    ④ 去掉开头的工作区目录名后再拼工作区。
    """
    text = str(raw or "").strip().strip('"').strip("'").replace("\\", "/")
    if not text:
        return None, []
    ws = Path(getattr(ctx, "workspace", Path.cwd()))
    first = Path(text)
    candidates: list[Path] = []
    if first.is_absolute():
        candidates.append(first)
    else:
        parts = list(first.parts)
        starts_with_ws_name = bool(parts) and parts[0] == ws.name and len(parts) > 1
        if starts_with_ws_name:
            # 路径里已带工作区名：直接拼父目录 / 去掉该目录名，避免产出 workspace/workspace/...
            candidates.append(ws.parent / text)
            candidates.append(ws / Path(*parts[1:]))
        else:
            candidates.append(ws / text)
        try:
            candidates.append(Path.cwd() / text)
        except Exception:
            pass
        if not starts_with_ws_name:
            candidates.append(ws.parent / text)
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate, candidates
        except OSError:
            continue
    return None, candidates


def _missing_path_message(raw: str, tried: list[Path]) -> str:
    """把"到底找了哪些位置"讲清楚，避免模型在错误路径上反复试。"""
    unique: list[str] = []
    for item in tried:
        text = str(item)
        if text not in unique:
            unique.append(text)
    head = unique[0] if unique else str(raw)
    rest = "；".join(unique[1:4])
    return f"路径不存在: {head}" + (f"（也已尝试: {rest}）" if rest else "")


def build_default_registry() -> EngineToolRegistry:
    registry = EngineToolRegistry()

    # ---------------------------------------------------------- catalog_query
    def _catalog_query(args: dict[str, Any], ctx: EngineContext) -> Observation:
        from ..datacatalog.summary import build_digest, search_catalog

        if ctx.catalog is None:
            return Observation(ok=False, summary="数据目录不可用", error={"type": "NoCatalog"})
        # 后台扫描可能还在跑（启动不阻塞）：首次用到目录时等一下，
        # 保证返回的永远是真实字段名/坐标系，而不是“空目录”。
        prewarmer = ctx.extra.get("prewarmer") if isinstance(ctx.extra, dict) else None
        if prewarmer is not None and getattr(prewarmer, "busy", False):
            prewarmer.wait_ready()
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
            readonly=True,
        )
    )

    # ------------------------------------------------------------ execute_code
    def _execute_code(args: dict[str, Any], ctx: EngineContext) -> Observation:
        from ..runtime.code_runner import backup_paths, backup_targets

        code = str(args.get("code", "") or "")
        if not code.strip():
            return Observation(ok=False, summary="代码为空", error={"type": "EmptyCode"})
        description = str(args.get("description", "") or "")
        timeout = float(args.get("timeout_seconds", ctx.exec_timeout) or ctx.exec_timeout)

        # 备份判据分两类（详见 backup_targets）：
        # 1) 显式删除（Delete/rmtree/remove）；
        # 2) **写工具的输出已存在** —— 内核里 overwriteOutput=True 是全局开着的，
        #    正常写产出也会静默覆盖原数据，以前这类完全不备份。
        backed_up: list[str] = []
        if ctx.auto_backup:
            targets = backup_targets(code, ctx.workspace)
            # GDB 里的要素类不是文件系统路径：``Path.exists()`` 对它们返回 False，
            # 只有 ``arcpy.Exists()`` 为真。而这里跑在**主进程**（venv，通常没有
            # arcpy），所以纯靠文件系统判断会漏掉整个 GDB 内的覆盖 —— 而 GDB
            # 正是本项目最常见的产物形态。没找到目标时再问一次内核。
            if not targets and ".gdb" in code.lower() and ctx.code_runner is not None:
                targets = _gdb_targets_via_kernel(code, ctx)
            if targets:
                backed_up = _backup_via_kernel(targets, ctx) or backup_paths(targets, ctx.backup_dir)
                if backed_up:
                    logger.info("backed up before overwriting: %s", backed_up)

        result = ctx.code_runner.run(code, timeout=timeout, workspace=str(ctx.workspace))
        artifacts = extract_outputs(result.result, ctx.workspace) if result.ok else []
        if result.ok:
            summary = f"执行成功（{result.duration_ms} ms）" + (f"：{description}" if description else "")
            if artifacts:
                summary += f"；产出 {len(artifacts)} 个文件"
            for path in artifacts:
                if ctx.state is not None:
                    ctx.state.add_artifact(path)
            soft_hint = _recipe_soft_hint(ctx, description, code)
            data: dict[str, Any] = {"stdout": result.stdout[-4000:], "result": result.result}
            # 覆盖前做了备份要让模型知道：它才知道自己刚改掉了什么、
            # 以及需要时可以怎么回滚。
            if backed_up:
                data["backed_up"] = backed_up
            # 预警也要能看见：matplotlib 缺字形、UserWarning 之类只写 stderr，
            # 之前成功路径只回传 stdout，导致“中文全变方框”的图一路通过验收。
            warning_hint = ""
            stderr_text = str(result.stderr or "").strip()
            if stderr_text:
                data["stderr"] = stderr_text[-1500:]
                warning_hint = _stderr_warning_hint(stderr_text)
                if warning_hint:
                    summary += "；⚠️ 有执行告警，见 hint"
            hint = "\n".join(part for part in (warning_hint, soft_hint) if part)
            return Observation(
                ok=True,
                summary=summary,
                data=data,
                artifacts=artifacts,
                images=result.display_images[:3],
                hint=hint,
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
        target, tried = _resolve_path(ctx, path)
        if target is None:
            message = _missing_path_message(path, tried)
            return Observation(
                ok=False,
                summary=f"文档不存在: {message}",
                error={"type": "FileNotFound", "message": message},
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
            readonly=True,
        )
    )

    # ---------------------------------------------------------------- read_file
    def _read_file(args: dict[str, Any], ctx: EngineContext) -> Observation:
        """读任意文本文件（技能正文、脚本、配置等）。

        技能（``SKILL.md``）和它自带的脚本都是普通文本，而 ``read_document``
        只认 docx/pdf/txt——没有这个工具，即便告诉模型“有 skill 可用”，它也读不到。
        """
        path = str(args.get("path", "") or "")
        max_chars = int(args.get("max_chars", 40000) or 40000)
        target, tried = _resolve_path(ctx, path)
        if target is None or not target.is_file():
            return Observation(
                ok=False,
                summary=f"文件不存在: {_missing_path_message(path, tried)}",
                error={"type": "FileNotFound"},
            )
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return Observation(
                ok=False,
                summary=f"读取失败: {exc}",
                error={"type": type(exc).__name__, "message": str(exc)},
            )
        total = len(text)
        truncated = total > max_chars
        if truncated:
            text = text[:max_chars] + f"\n\n…（已截断，原文 {total} 字符）"
        return Observation(
            ok=True,
            summary=f"已读取 {target.name}（{total} 字符{'，已截断' if truncated else ''}）",
            data={"path": str(target), "text": text, "total_chars": total, "truncated": truncated},
        )

    registry.register(
        EngineTool(
            name="read_file",
            description=(
                "读取任意文本文件（.md/.txt/.py/.json 等），用于查看技能正文（SKILL.md）、"
                "技能自带脚本、配置或说明文档。二进制文档（docx/pdf）改用 read_document。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径（可相对 workspace）"},
                    "max_chars": {"type": "integer", "description": "最多返回字符数，默认 40000"},
                },
                "required": ["path"],
            },
            handler=_read_file,
            readonly=True,
        )
    )

    # ---------------------------------------------------------------- run_skill
    def _run_skill(args: dict[str, Any], ctx: EngineContext) -> Observation:
        """把一个技能的 SKILL.md 全文读进对话（progressive disclosure 的第二步）。

        常驻上下文里只有技能名+描述（便宜）；真正要用时才把全文拉进来。
        """
        from ..skills.catalog import discover_skills, read_skill_text

        name = str(args.get("name", "") or "").strip()
        if not name:
            return Observation(ok=False, summary="必须给技能名", error={"type": "BadArgs"})
        skills = discover_skills(ctx.workspace)
        chosen = None
        for item in skills:
            if item.name.lower() == name.lower().lstrip("/"):
                chosen = item
                break
        if chosen is None:
            available = ", ".join(item.name for item in skills[:20]) or "（无）"
            return Observation(
                ok=False,
                summary=f"没找到技能 {name}；可用技能：{available}",
                error={"type": "SkillNotFound"},
            )
        text = read_skill_text(chosen)
        return Observation(
            ok=True,
            summary=f"已加载技能 {chosen.name}（{len(text)} 字符，脚本目录 {chosen.root}）",
            data={
                "name": chosen.name,
                "description": chosen.description,
                "root": str(chosen.root),
                "skill_file": str(chosen.skill_file),
                "text": text,
            },
        )

    registry.register(
        EngineTool(
            name="run_skill",
            description=(
                "加载某个技能的完整说明（SKILL.md 正文）。当任务与“可用技能”列表中某个技能的"
                "描述吻合时（例如用户要写 Word 文档 / 读 PDF / 做幻灯片），必须先调用它读到全文，"
                "再严格按其中的步骤执行；技能自带的脚本用 execute_code 运行。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "技能名（与技能列表一致）"},
                },
                "required": ["name"],
            },
            handler=_run_skill,
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
        raw_timeout = args.get("timeout_seconds")
        try:
            timeout = float(raw_timeout) if raw_timeout else None
        except (TypeError, ValueError):
            timeout = None
        return ctx.recipes.run(recipe_id, params, ctx, timeout=timeout)

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
                    "timeout_seconds": {
                        "type": "number",
                        "description": "可选：覆盖执行超时秒数（缓冲区分区统计等重活建议 1800 以上）",
                    },
                },
                "required": ["recipe_id", "params"],
            },
            handler=_run_recipe,
        )
    )

    # -------------------------------------------------------------- create_map
    def _create_map(args: dict[str, Any], ctx: EngineContext) -> Observation:
        """直接调制图引擎出图（不需要写代码，也不要去读 cartography 源码）。

        实测（14 届 4a-2）：任务要求“迁移方向箭头 + 年份标注”，模型找不到
        overlay 的用法，只好反复用 execute_code 读我们自己的源码（80+ 轮），
        而 `overlay` 本来就支持——只是没有工具面把它暴露出来。
        """
        output_path = str(args.get("output_path", "") or "").strip()
        if not output_path:
            return Observation(ok=False, summary="缺少 output_path", error={"type": "BadArgs"})
        layers = [str(item) for item in (args.get("layers") or []) if str(item).strip()]
        if not layers:
            return Observation(ok=False, summary="缺少 layers（至少一个图层路径）", error={"type": "BadArgs"})
        # 相对路径先按 workspace 解析成**绝对路径**：内核的 cwd 不一定是 workspace，
        # 而 addDataFromPath 碰到相对路径会把本地数据当成 Web 服务去解析，报
        # “AUTOMATIC 失败…web_service_type”（已复现）。create_map 内部也会再解析一次，
        # 这里做是为了不依赖内核 cwd。
        layers = [str(_resolve_path(ctx, item)[0] or (Path(ctx.workspace) / item)) for item in layers]

        intent: dict[str, Any] = {}
        for key in (
            "title", "theme", "region", "scale", "purpose", "medium", "orientation",
            "legend_labels", "style_profile", "field", "name",
            # 渲染与分级：以前这些只存在于“直接写代码”的路径里，调 create_map 的人
            # 只能去读 cartography 源码猜（实测 214 次读源码里最多的就是这几类）。
            "mode", "explicit_bounds", "class_bounds", "classification", "color_ramp",
            "category_names", "category_colors", "category_order", "legend_title",
            "map_kind", "note",
        ):
            if args.get(key):
                intent[key] = args[key]
        if isinstance(args.get("overlay"), dict):
            intent["overlay"] = args["overlay"]
        if isinstance(args.get("overrides"), dict):
            intent["overrides"] = args["overrides"]
        if args.get("fields"):
            intent["fields"] = args["fields"]

        aprx_path = str(args.get("aprx_path", "") or "")
        code = (
            "from gis_cli.cartography import create_map\n"
            f"_r = create_map({layers!r}, {intent!r}, output_path={output_path!r}, "
            f"aprx_path={aprx_path!r}, code_runner=None)\n"
            # 设计规格边车（同名 .spec.json）：制图配方里有这一步，create_map 没有。
            # 后果是同一个任务两条出图路径证据不一致：模型用 create_map 出图时
            # 验收找不到 .spec.json，判“未统一图例”并把 agent 逼去反复重做
            # （14 届任务四a 实测：89 轮、10 次 create_map、一个 spec 也没留下）。
            "import json as _json, os as _os\n"
            "try:\n"
            "    _out = str(_r.get('output') or '')\n"
            "    if _out:\n"
            "        _sp = _os.path.splitext(_out)[0] + '.spec.json'\n"
            "        with open(_sp, 'w', encoding='utf-8') as _fh:\n"
            "            _json.dump({k: v for k, v in _r.items() if k != 'spec'}, _fh, ensure_ascii=False, indent=2)\n"
            "        _r['spec_path'] = _sp\n"
            "except Exception as _e:\n"
            "    _r['spec_error'] = str(_e)[:200]\n"
            "set_result(_r)\n"
        )
        # 制图引擎自己要在 ArcGIS 解释器（内核）里跑，所以固定走 code_runner；
        # 不要在这里直接用 create_map（benchmark/CLI 的 venv 没有 arcpy）。
        result = ctx.code_runner.run(code, timeout=float(args.get("timeout_seconds") or 900), workspace=str(ctx.workspace))
        if not result.ok:
            detail = ""
            if isinstance(result.error, dict):
                detail = str(result.error.get("message") or result.error.get("type") or "")
            return Observation(
                ok=False,
                summary=f"出图失败: {detail[:300]}",
                error=result.error,
                data={"stdout": result.stdout[-2000:], "stderr": result.stderr[-1500:]},
                hint="看 traceback 修正参数；overlay 的坐标系必须与图层一致（数据坐标，不是像素）。",
            )
        payload = result.result if isinstance(result.result, dict) else {}
        artifacts = extract_outputs(
            [output_path, aprx_path, payload.get("spec"), payload.get("output"), payload.get("aprx")],
            ctx.workspace,
        )
        if ctx.state is not None:
            for path in artifacts:
                ctx.state.add_artifact(path)
        qc_ok = payload.get("qc_ok")
        summary = f"出图完成：{output_path}"
        if qc_ok is not None:
            summary += f"（版面体检 {'通过' if qc_ok else '未通过'}）"
        return Observation(
            ok=True,
            summary=summary,
            data=payload,
            artifacts=artifacts,
            hint="" if qc_ok is not False else f"版面问题：{str(payload.get('qc_summary'))[:220]}",
        )

    registry.register(
        EngineTool(
            name="create_map",
            description=(
                "出地图（含版面自修复）：直接传图层与设计意图，就能得到 JPG + 同名可编辑 .aprx。"
                "版面（图名/图例/比例尺/指北针）由规则引擎自动决定，不传也能出图。\n"
                "**这些参数就是全部接口，不要去读 cartography 源码找别的开关。**\n"
                "【分级渲染】用 field 指定字段；mode 强制渲染器（graduated 分级 / unique 分类 / "
                "raster_stretch 栅格拉伸 / single 单色）；explicit_bounds 或 class_bounds 给**显式分级上界**"
                '（如 "0,2,4,6,8"），多张图传同一串就能共用一套图例；classification 选分级方法'
                "（quantile/natural/interval）；color_ramp 指定色带名（如 Purples/Reds/Greens）。\n"
                "【叠加层】迁移图这类需要“方向箭头/年份标注”的用 overlay 参数，"
                "**不要**自己用 PIL 画（坐标系易错）。坐标用**图层的数据坐标系**（不是像素）：\n"
                '{"arrows": [{"from": [起点X, 起点Y], "to": [终点X, 终点Y], "label": "1982→1983"}], '
                '"labels": [{"at": [X, Y], "text": "1982"}]}\n'
                "（from/to 也可写成 x0/y0/x1/y1，两种写法都认）\n"
                "arrows 画带箭头的迁移线，labels 在点位旁标文字；两者不要重复标同一信息。\n"
                "【版式微调】要覆盖自动设计（如比例尺长度）用 overrides，点路径写法："
                '{"scale_bar.element_width_mm": 55}；出图后看 qc_summary 的提示即可。'
            ),
            parameters={
                "type": "object",
                "properties": {
                    "layers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "图层路径，按绘制顺序（先画的在下层）",
                    },
                    "output_path": {"type": "string", "description": "输出 JPG 路径"},
                    "aprx_path": {"type": "string", "description": "可选：输出 .aprx 路径（默认同名）"},
                    "title": {"type": "string", "description": "图名"},
                    "theme": {"type": "string", "description": "图主题材（写进图例标题）"},
                    "region": {"type": "string", "description": "区域名"},
                    "scale": {"type": "string", "description": "尺度词（如 1982-2001 年）"},
                    "purpose": {"type": "string", "description": "用途：distribution/evaluation/competition"},
                    "field": {"type": "string", "description": "分级/分类字段"},
                    "legend_labels": {"type": "string", "description": "图例标签：semantic/range/both"},
                    "style_profile": {"type": "string", "description": "风格档位：competition_standard/screen_report/dense_raster"},
                    "mode": {
                        "type": "string",
                        "description": "渲染器：graduated 分级 / unique 分类 / raster_stretch 栅格拉伸 / single 单色（不传则自动判断）",
                    },
                    "explicit_bounds": {
                        "type": "string",
                        "description": "显式分级上界，逗号分隔（如 0,2,4,6,8）。多图共用一套图例时传同一个值",
                    },
                    "class_bounds": {"type": "string", "description": "explicit_bounds 的别名"},
                    "classification": {
                        "type": "string",
                        "description": "自动分级方法：quantile/natural/interval（未给显式分级时生效）",
                    },
                    "color_ramp": {"type": "string", "description": "色带名（如 Purples/Reds/Greens/Blues）"},
                    "category_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "unique 模式的类别名（按代码顺序）",
                    },
                    "category_colors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "unique 模式的类别配色 #RRGGBB（与 category_names 对应）",
                    },
                    "map_kind": {"type": "string", "description": "图种词，拼进图名（如 分布图/评价图）"},
                    "note": {"type": "string", "description": "附注，写进设计记录"},
                    "overlay": {
                        "type": "object",
                        "description": (
                            '叠加层：{"arrows":[{"from":[x,y],"to":[x,y],"label":"1982→1983"}], '
                            '"labels":[{"at":[x,y],"text":"1982"}]}，坐标为数据坐标'
                        ),
                    },
                    "overrides": {"type": "object", "description": "强制覆盖设计（点路径，如 scale_bar.element_width_mm）"},
                    "timeout_seconds": {"type": "number", "description": "可选：执行超时秒数"},
                },
                "required": ["layers", "output_path"],
            },
            handler=_create_map,
        )
    )

    # ------------------------------------------------------------ list_recipes
    def _list_recipes(args: dict[str, Any], ctx: EngineContext) -> Observation:
        if ctx.recipes is None:
            return Observation(ok=False, summary="配方库不可用", error={"type": "NoRecipes"})
        query = str(args.get("query", "") or "").strip()
        if not query and hasattr(ctx.recipes, "catalog_digest"):
            # 空查询＝看总目录：避免逐个关键词试探
            return Observation(
                ok=True,
                summary=f"配方总目录（{len(ctx.recipes.all())} 个）",
                data={"catalog": ctx.recipes.catalog_digest()},
            )
        items = ctx.recipes.search(query)
        return Observation(
            ok=True,
            summary=f"找到 {len(items)} 个配方",
            data={"recipes": items},
        )

    registry.register(
        EngineTool(
            name="list_recipes",
            description="检索配方：给关键词返回配方 id/用途/参数/前置条件；query 留空返回按类别分组的全部配方目录。",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "关键词；留空表示全部"}},
            },
            handler=_list_recipes,
            readonly=True,
        )
    )

    # --------------------------------------------------------- view_image
    def _view_image(args: dict[str, Any], ctx: EngineContext) -> Observation:
        """让模型看一眼产出的图片（地图/统计图）。模型需支持识图。"""
        from ..runtime.image_input import is_image_path

        raw = str(args.get("path", "") or "").strip()
        if not raw:
            return Observation(ok=False, summary="path 为空", error={"type": "BadRequest"})
        resolved, _tried = _resolve_path(ctx, raw)
        path = resolved if resolved is not None else (Path(raw) if Path(raw).is_absolute() else (ctx.workspace / raw))
        if not path.exists():
            return Observation(
                ok=False, summary=f"图片不存在: {path}", error={"type": "FileNotFound"}
            )
        if not is_image_path(path):
            return Observation(
                ok=False, summary=f"不是图片文件: {path.name}", error={"type": "NotImage"}
            )
        size_kb = path.stat().st_size / 1024
        return Observation(
            ok=True,
            summary=(
                f"已加载图片供你查看（{path.name}，{size_kb:.0f} KB）。"
                "请核对图面：图名/图例/比例尺/指北针是否齐全且可读，中文有无方框乱码，"
                "图幅有无被裁切或大面积空白；有问题则修正后重新导出。"
            ),
            data={"path": str(path)},
            images=[str(path)],
        )

    registry.register(
        EngineTool(
            name="view_image",
            description=(
                "查看一张产出图片（地图/统计图/截图），用于核对图面质量：图名/图例/比例尺是否齐全、"
                "中文是否可读（有无方框）、地图是否被裁切或空白。模型支持识图时才看得见内容。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "图片路径（可相对 workspace）"},
                },
                "required": ["path"],
            },
            handler=_view_image,
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

    # ------------------------------------------------------------ web_search
    def _web_search(args: dict[str, Any], ctx: EngineContext) -> Observation:
        """联网搜索（查 ArcPy API / WKID / 报错信息）。

        没网时不报错也不崩：返回一条明确的失败观察，模型据此改用已有知识。
        竞赛现场常常无网，所以这是**常规路径**而不是异常路径。
        """
        from . import web_search as ws

        query = str(args.get("query", "") or "").strip()
        if not query:
            return Observation(ok=False, summary="query 为空", error={"type": "BadRequest"})
        try:
            limit = int(args.get("max_results", 5) or 5)
        except (TypeError, ValueError):
            limit = 5
        payload = ws.search(query, limit=limit)
        results = payload.get("results") or []
        if not results:
            reason = payload.get("error") or "无结果"
            return Observation(
                ok=False,
                summary=f"联网搜索未成功（{reason}）。可改用已有知识继续，或换关键词再试。",
                data={"query": query, "backend": payload.get("backend", "")},
                hint=(
                    "搜索不可用不等于任务做不下去：先看数据目录、配方库和文档，"
                    "不要因为搜索失败而停下来反复重试。"
                ),
            )
        body = ws.format_results(payload)
        return Observation(
            ok=True,
            summary=(
                f"搜索到 {len(results)} 条结果（{payload.get('backend', '')}）。"
                "结果只作参考，具体参数仍以本地数据和文档为准。"
            ),
            data={"query": query, "backend": payload.get("backend", ""), "text": body},
        )

    registry.register(
        EngineTool(
            name="web_search",
            description=(
                "联网搜索 GIS 资料（ArcPy/ArcGIS Pro 最新 API、坐标系 WKID、报错信息、ESRI 文档）。"
                "仅在本地数据目录、配方、文档都查不到时使用；没网时会明确告诉你不可用，"
                "请改用已有知识继续，不要反复重试。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "max_results": {
                        "type": "integer",
                        "description": "最多返回几条（默认 5，上限 8）",
                    },
                },
                "required": ["query"],
            },
            handler=_web_search,
            # 只读且无副作用，可与其他只读工具并发
            readonly=True,
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

    # --------------------------------------------------------- map_design
    def _map_design(args: dict[str, Any], ctx: EngineContext) -> Observation:
        """先出"设计说明"（纸张/图名/图例/比例尺/指北针 + 理由），再照它出图。

        让模型只负责语义选择（主题/用途/介质/是否覆盖），几何与合法性由规则引擎算。
        """
        layers: list[Any] = []
        if isinstance(args.get("layers"), list):
            layers.extend(args["layers"])
        if args.get("data_path"):
            layers.insert(0, str(args["data_path"]))
        if not layers:
            return Observation(ok=False, summary="缺少 data_path 或 layers", error={"type": "BadArgs"})
        try:
            from ..cartography import design_only

            intent = {
                key: args[key]
                for key in (
                    "theme", "purpose", "medium", "orientation", "legend_labels", "style_profile",
                    "title", "region", "scale", "map_kind", "field", "legend_title", "color_ramp",
                )
                if args.get(key)
            }
            if args.get("overrides"):
                intent["overrides"] = args["overrides"]
            result = design_only(layers, intent, code_runner=getattr(ctx, "code_runner", None))
        except Exception as exc:
            return Observation(ok=False, summary=f"设计失败: {str(exc)[:200]}",
                               error={"type": type(exc).__name__, "message": str(exc)[:400]})
        facts = result.get("facts") or {}
        brief = {
            "数据长宽比": round(float(facts.get("aspect") or 1.0), 2),
            "要素数": (facts.get("primary") or {}).get("feature_count"),
            "推测渲染模式": facts.get("render_mode"),
            "角区占用": facts.get("corner_usage"),
        }
        return Observation(
            ok=True,
            summary="设计完成：\n" + str(result.get("summary") or "")[:1500],
            data={"design": result.get("summary"), "spec": result.get("spec"), "facts_brief": brief,
                  "design_note": (result.get("spec") or {}).get("design_note", "")},
        )

    registry.register(
        EngineTool(
            name="map_design",
            description=(
                "制图设计（不渲染）：根据数据事实与主题，给出纸张朝向、图名格式与字号、图例内容与位置、"
                "比例尺刻度与单位、指北针样式，并附每个决定的理由。出图前先调用它，可先向用户说明版式选择。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "主数据路径"},
                    "layers": {"type": "array", "items": {"type": "string"}, "description": "多图层叠加时的图层路径"},
                    "field": {"type": "string", "description": "分级/分类字段"},
                    "theme": {"type": "string", "description": "图主题材（如 老年人口/绿地服务价值）"},
                    "region": {"type": "string", "description": "区域名（如 郑州市）"},
                    "scale": {"type": "string", "description": "尺度词（如 社区尺度）"},
                    "purpose": {"type": "string", "description": "用途：分布/评价/规划/对比/汇报"},
                    "medium": {"type": "string", "description": "介质：print_A4/print_A3/screen_16_9"},
                    "orientation": {"type": "string", "description": "朝向：auto/portrait/landscape"},
                    "legend_labels": {"type": "string", "description": "图例标签：semantic/range/both"},
                    "style_profile": {"type": "string", "description": "风格档位：competition_standard/screen_report/dense_raster"},
                    "overrides": {"type": "object", "description": "强制覆盖设计（点路径）"},
                },
                "required": ["data_path"],
            },
            handler=_map_design,
        )
    )

    # --------------------------------------------------------- check_map_project
    def _check_map_project(args: dict[str, Any], ctx: EngineContext) -> Observation:
        """核验地图工程（.aprx）内部的渲染器/配色/布局四要素。

        不要用 execute_code 在内核里 mp.ArcGISProject 打开工程：实测第二次打开会挂死。
        """
        aprx_path = str(args.get("aprx_path", "") or "").strip()
        if not aprx_path:
            return Observation(ok=False, summary="缺少 aprx_path", error={"type": "BadArgs"})
        if ctx.recipes is None:
            return Observation(ok=False, summary="断言引擎不可用", error={"type": "NoRecipes"})
        raw: dict[str, Any] = {"type": "aprx_map_check", "path": aprx_path}
        for key in ("renderer", "colors"):
            if args.get(key):
                raw[key] = args[key]
        if args.get("elements"):
            raw["elements"] = list(args["elements"])
        result = ctx.recipes.run_assertions([raw])[0]
        return Observation(
            ok=bool(result.ok),
            summary=("工程核验通过：" if result.ok else "工程核验未通过：") + result.detail[:400],
            data=result.to_dict(),
            hint="" if result.ok else "按提示修正渲染器/配色/布局后重新出图（用 category_map / graduated_colors_map）。",
        )

    registry.register(
        EngineTool(
            name="check_map_project",
            description=(
                "核验 .aprx 地图工程：唯一值/分级渲染器、各类别配色（#RRGGBB）、布局四要素（图名/图例/比例尺/指北针）"
                "是否都落盘。交付制图成果前用它自检；不要在 execute_code 里用 mp.ArcGISProject 打开工程（会挂死）。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "aprx_path": {"type": "string", "description": "工程文件路径"},
                    "renderer": {
                        "type": "string",
                        "description": "期望渲染器类型（UniqueValueRenderer / GraduatedColorsRenderer），可空",
                    },
                    "colors": {
                        "type": "string",
                        "description": '期望配色，如 "标杆社区=#1F77B4;需整改社区=#2CA02C"，可空',
                    },
                    "elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "必须存在的布局要素（图名/图例/比例尺/指北针），可空",
                    },
                },
                "required": ["aprx_path"],
            },
            handler=_check_map_project,
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
                text = str(path).strip()
                # finish 传的路径要过与 set_result 同一套“是不是产物”的判据：
                # 模型会把自己的工程/模板 aprx、以及读过的 input/*.shp 一并写上，
                # 登记进去会让验收永远报“产出不在 output 目录内”（详见 extract_outputs）。
                if not text:
                    continue
                if not extract_outputs([text], ctx.workspace):
                    continue
                ctx.state.add_artifact(text)
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



def _gdb_targets_via_kernel(code: str, ctx: Any) -> list[str]:
    """在**内核里**用 ``arcpy.Exists`` 确认哪些 GDB 路径会被覆盖。

    为什么必须走内核：``Path.exists('x.gdb/study_area')`` 返回 **False**，
    只有 ``arcpy.Exists()`` 为真。而 ``backup_targets`` 跑在主进程（venv，
    通常没有 arcpy），纯文件系统判断会漏掉整个 GDB 内的覆盖 —— 而 GDB 正是
    本项目最常见的产物形态。

    实测教训：覆盖 ``study_area`` 后旧数据静默消失、零备份，就是因为
    ``_gdb_exists`` 在 venv 里恒为 False。

    代价：一次极轻量的内核往返（只做 exists 判断）。只在代码里出现 ``.gdb``
    且文件系统没找到目标时才触发。
    """
    from ..runtime.code_runner import backup_targets as _candidates

    candidates = _candidates(code, ctx.workspace, ignore_exists=True)
    if not candidates:
        return []
    payload = ", ".join(repr(str(item)) for item in candidates)
    probe = (
        "import arcpy as _a\n"
        f"_c = [{payload}]\n"
        "set_result([p for p in _c if _a.Exists(p)])\n"
    )
    try:
        outcome = ctx.code_runner.run(probe, timeout=60, workspace=str(ctx.workspace))
    except Exception as exc:  # pragma: no cover - 探测失败不能阻断执行
        logger.warning("gdb existence probe failed: %s", exc)
        return []
    if not outcome.ok or not isinstance(outcome.result, list):
        return []
    out_root = str((Path(ctx.workspace) / "output").resolve()).lower()
    kept: list[str] = []
    for item in outcome.result:
        text = str(item)
        try:
            resolved = str(Path(text).resolve())
        except OSError:
            resolved = text
        if resolved.lower().startswith(out_root):
            kept.append(resolved)
    return kept


def _backup_via_kernel(targets: list[str], ctx: Any) -> list[str]:
    """在**内核里**做备份（GDB 成员必须走 arcpy，主进程没有 arcpy）。

    普通文件/目录也能处理：能 copy 的走 shutil，GDB 成员走 CopyFeatures/
    CopyRows/CopyRaster 导成独立备份库。

    返回创建的备份路径；失败/无内核时返回空列表（由调用方退回主进程备份）。
    """
    if ctx.code_runner is None or not targets:
        return []
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = str(Path(ctx.backup_dir) / stamp)
    payload = ", ".join(repr(str(item)) for item in targets)
    probe = f'''
import os as _os, shutil as _sh
import arcpy as _a
_root = r"{root}"
_os.makedirs(_root, exist_ok=True)
_created = []
for _src in [{payload}]:
    try:
        _desc = _a.Describe(_src)
        _kind = getattr(_desc, "dataType", "") or ""
        _name = _os.path.basename(_src.rstrip("\\/"))
        if _kind in ("FeatureClass", "Table", "RasterDataset", "Raster"):
            _gdb = _os.path.join(_root, _name + "_backup.gdb")
            if not _a.Exists(_gdb):
                _a.management.CreateFileGDB(_root, _name + "_backup.gdb")
            _dst = _os.path.join(_gdb, _name)
            if _kind in ("FeatureClass",):
                _a.management.CopyFeatures(_src, _dst)
            elif _kind == "Table":
                _a.management.CopyRows(_src, _dst)
            else:
                _a.management.CopyRaster(_src, _dst)
            _created.append(_gdb)
        elif _os.path.isdir(_src):
            _dst = _os.path.join(_root, _name)
            if _os.path.isdir(_dst):
                _sh.rmtree(_dst, ignore_errors=True)
            _sh.copytree(_src, _dst, ignore=_sh.ignore_patterns("*.lock", "*.lockfile"))
            _created.append(_dst)
        elif _os.path.isfile(_src):
            _dst = _os.path.join(_root, _name)
            _sh.copy2(_src, _dst)
            _created.append(_dst)
    except Exception as _exc:
        pass
set_result(_created)
'''
    try:
        outcome = ctx.code_runner.run(probe, timeout=300, workspace=str(ctx.workspace))
    except Exception as exc:  # pragma: no cover - 备份失败不能阻断执行
        logger.warning("kernel backup failed: %s", exc)
        return []
    if not outcome.ok or not isinstance(outcome.result, list):
        return []
    return [str(item) for item in outcome.result]

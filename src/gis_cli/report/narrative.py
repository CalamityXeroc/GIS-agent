# -*- coding: utf-8 -*-
"""成果包内容（P0-4 步 b）：把验收结论、配方调用翻成人话。

证据只有两份，都是引擎已有的结论，这里**不新增理解能力、不调模型**：

- ``verdict``：``Verifier.verify()`` 的返回，与 trace 里的 ``verification``
  事件是同一份对象；
- trace JSONL：只取 ``run_recipe`` 的调用（哪个配方、什么参数、是否成功）。

不编造：拿不到证据的区块留空，渲染层写"本次没有…"，绝不补一句"效果良好"。
术语（要素数、断言、feature_count）一律换成人能读的说法，正文里不出现。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

#: 固定免责声明（合规要求，不允许按场景改写）。
DISCLAIMER = "本成果由 AI 生成，未经专业审图，不得直接用于法定报批。"

#: 可以内联进 HTML / PDF 的图片后缀。
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")
#: 矢量成果后缀（GDB 内要素类靠路径里的 ``.gdb/`` 识别）。
VECTOR_EXTS = (".shp", ".geojson")
RASTER_EXTS = (".tif", ".tiff", ".img", ".jp2")
TABLE_EXTS = (".csv", ".xlsx", ".xls", ".dbf")

#: 需求状态 → 人话。
_STATUS_PLAIN = {"PASS": "已满足", "PARTIAL": "基本满足", "FAIL": "未满足"}

#: 数据来源代码 → 人话（与 ``datacatalog.store.SOURCES`` 对齐）。
_SOURCE_PLAIN = {
    "local": "本机已有数据",
    "csv_upload": "用户上传的表格/点位",
    "boundary": "在线获取的行政边界",
    "download": "下载的数据",
}

#: 值里超过这个长度就截断（报告不是日志）。
_TEXT_LIMIT = 120


# ------------------------------------------------------------------ 数据结构
@dataclass
class KeyNumber:
    """结论区的一个关键数字（标签 + 值 + 可选补充）。"""

    label: str
    value: str
    note: str = ""


@dataclass
class SourceRow:
    """数据来源表的一行（全部来自 catalog 登记值）。"""

    name: str
    path: str
    source: str
    license: str
    acquired_at: str


@dataclass
class RecipeCall:
    """本次运行真正调用过的配方。"""

    recipe_id: str
    title: str
    params: dict[str, Any] = field(default_factory=dict)
    ok: bool | None = None
    count: int = 1

    def describe(self) -> str:
        bits = []
        for key, value in list(self.params.items())[:4]:
            text = str(value)
            if len(text) > 40:
                text = text[:39] + "…"
            bits.append(f"{key}={text}")
        if len(self.params) > 4:
            bits.append(f"等 {len(self.params)} 项")
        status = {True: "执行成功", False: "执行失败", None: "未记录结果"}[self.ok]
        tail = f"（{status}"
        tail += f"，共 {self.count} 次）" if self.count > 1 else "）"
        return f"配方「{self.title}」" + (f"：{'，'.join(bits)}" if bits else "") + tail


@dataclass
class ImageRef:
    caption: str
    path: str


@dataclass
class LayerRef:
    name: str
    path: str
    note: str = ""


@dataclass
class ReportContent:
    """一份成果包的全部文字内容（渲染层只负责排版）。"""

    title: str = ""
    conclusion: str = ""
    key_numbers: list[KeyNumber] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    sources: list[SourceRow] = field(default_factory=list)
    verifications: list[str] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)
    layers: list[LayerRef] = field(default_factory=list)
    generated_at: str = ""
    disclaimer: str = DISCLAIMER


# ------------------------------------------------------------------ 小工具
def _short(text: Any, limit: int = _TEXT_LIMIT) -> str:
    """压成一行并截断（报告里的长句一律先过这里）。"""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _suffix(path: str) -> str:
    return Path(str(path)).suffix.lower()


def is_gdb_member(path: str) -> bool:
    """GDB 内的要素类/栅格在文件系统里没有后缀，只能看路径里有没有 ``.gdb/``。"""
    return ".gdb/" in str(path).replace("\\", "/").lower()


def is_image(path: str) -> bool:
    return _suffix(path) in IMAGE_EXTS


def is_vector(path: str) -> bool:
    """能不能当图层导出来（GDB 成员 / .shp / .geojson）。

    注意 ``.gdb`` 目录本身不算：它是容器，没有单独的几何可画。
    ``.gpkg`` 也不在这里——多图层容器要么问 arcpy 要图层名，要么就不做。
    """
    text = str(path)
    if is_gdb_member(text):
        return True
    return _suffix(text) in VECTOR_EXTS


def is_raster(path: str) -> bool:
    return _suffix(path) in RASTER_EXTS


# ------------------------------------------------------------------ 结论
def key_numbers(artifacts: Iterable[str], structural: Iterable[dict[str, Any]] | None) -> list[KeyNumber]:
    """关键数字：优先用验收实测的要素数/栅格值域，没有就退回按后缀计数。"""
    records = [r for r in (structural or []) if isinstance(r, dict) and r.get("exists")]
    numbers: list[KeyNumber] = []

    vectors = [r for r in records if isinstance(r.get("feature_count"), int)]
    if vectors:
        total = sum(int(r["feature_count"]) for r in vectors)
        numbers.append(KeyNumber("矢量图层", f"{len(vectors)} 个", f"合计 {total:,} 个要素"))
    rasters = [r for r in records if r.get("kind") == "RasterDataset"]
    if rasters:
        first = rasters[0]
        low, high = first.get("raster_min"), first.get("raster_max")
        note = f"取值 {low:g} ~ {high:g}" if isinstance(low, (int, float)) and isinstance(high, (int, float)) else ""
        numbers.append(KeyNumber("栅格成果", f"{len(rasters)} 个", note))
    images = [r for r in records if isinstance(r.get("image"), dict)]
    if images:
        numbers.append(KeyNumber("成果图", f"{len(images)} 张"))

    if numbers:
        return numbers

    # 没有内核体检记录（例如无 arcpy 环境）时按后缀计数，数值只是"有几个文件"。
    paths = [str(p) for p in artifacts]
    fallback = [
        ("矢量图层", sum(1 for p in paths if is_vector(p))),
        ("栅格成果", sum(1 for p in paths if is_raster(p))),
        ("成果图", sum(1 for p in paths if is_image(p))),
        ("数据表", sum(1 for p in paths if _suffix(p) in TABLE_EXTS)),
    ]
    return [KeyNumber(label, f"{count} 个") for label, count in fallback if count]


# ------------------------------------------------------------------ 验证
def plain_verifications(
    verdict: dict[str, Any] | None,
    requirements: Iterable[Any] = (),
) -> list[str]:
    """把验收结论翻成"检查了什么、结论如何"，不出现断言术语。"""
    verdict = verdict or {}
    requirements = list(requirements or [])
    lines: list[str] = []
    records = [r for r in (verdict.get("structural") or []) if isinstance(r, dict)]
    alive = [r for r in records if r.get("exists")]

    vectors = [r for r in alive if isinstance(r.get("feature_count"), int)]
    if vectors:
        empty = [r for r in vectors if int(r["feature_count"]) == 0]
        if empty:
            names = "、".join(_short(Path(str(r.get("path", ""))).name, 30) for r in empty[:3])
            lines.append(f"{len(empty)} 个图层是空的（0 个要素）：{names}")
        else:
            total = sum(int(r["feature_count"]) for r in vectors)
            lines.append(f"{len(vectors)} 个矢量图层都读得出来，合计 {total:,} 个要素")

    rasters = [r for r in alive if r.get("kind") == "RasterDataset"]
    if rasters:
        lows = [r.get("raster_min") for r in rasters if isinstance(r.get("raster_min"), (int, float))]
        highs = [r.get("raster_max") for r in rasters if isinstance(r.get("raster_max"), (int, float))]
        extra = f"（取值 {min(lows):g} ~ {max(highs):g}）" if lows and highs else ""
        lines.append(f"{len(rasters)} 个栅格都能读出数值{extra}")

    images = [r for r in alive if isinstance(r.get("image"), dict)]
    if images:
        blank = [r for r in images if not r["image"].get("ok")]
        if blank:
            names = "、".join(_short(Path(str(r.get("path", ""))).name, 30) for r in blank[:3])
            lines.append(f"{len(blank)} 张成果图可能空白：{names}")
        else:
            lines.append(f"{len(images)} 张成果图检查过，不是空白图")

    texts = {str(getattr(r, "id", "")): str(getattr(r, "text", "")) for r in requirements}
    per_req = (verdict.get("semantic") or {}).get("per_requirement") or []
    for index, item in enumerate(per_req[:10], start=1):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).upper()
        label = _STATUS_PLAIN.get(status)
        if label is None:
            continue
        text = _short(texts.get(str(item.get("id", "")), str(item.get("id", ""))), 40)
        lines.append(f"第 {index} 条要求「{text}」{label}")

    if verdict.get("degraded"):
        lines.append("同一处问题重复出现且成果没有变化，系统降级放行，请人工复核。")
    for item in (verdict.get("missing") or [])[:3]:
        lines.append(f"仍然缺少：{_short(item, 60)}")
    for item in (verdict.get("problems") or [])[:3]:
        lines.append(f"仍然要注意：{_short(item, 60)}")
    hygiene = verdict.get("hygiene") or {}
    if isinstance(hygiene, dict) and hygiene.get("summary"):
        lines.append(f"交付目录整理情况：{_short(hygiene['summary'], 80)}")
    return lines


# ------------------------------------------------------------------ 方法
def recipes_from_trace(trace_path: str | Path | None) -> list[RecipeCall]:
    """从 run 的 JSONL 里取出 ``run_recipe`` 的调用（配方 + 参数 + 成败）。

    只读 ``tool_call`` / ``tool_result`` 两种事件，且一一配对；解析失败的行
    直接跳过——trace 是崩溃安全写的，末行可能是半行。
    """
    if not trace_path:
        return []
    path = Path(str(trace_path))
    if not path.exists():
        return []
    calls: list[RecipeCall] = []
    pending: RecipeCall | None = None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                event = evt.get("event")
                payload = evt.get("payload") or {}
                if event == "tool_call" and payload.get("tool") == "run_recipe":
                    args = payload.get("args") or {}
                    params = args.get("params") if isinstance(args.get("params"), dict) else {}
                    pending = RecipeCall(recipe_id=str(args.get("recipe_id", "") or ""), title="", params=params)
                    calls.append(pending)
                elif event == "tool_result" and payload.get("tool") == "run_recipe" and pending is not None:
                    pending.ok = bool(payload.get("ok"))
                    pending = None
    except OSError as exc:  # pragma: no cover - 权限/占用
        logger.debug("读 trace 失败: %s", exc)
    # 同一配方同一参数重复调用（重试）合并成一次，别让报告变成日志。
    merged: dict[tuple[str, str], RecipeCall] = {}
    for call in calls:
        key = (call.recipe_id, json.dumps(call.params, ensure_ascii=False, sort_keys=True, default=str))
        if key in merged:
            merged[key].count += 1
            if call.ok is not None:
                merged[key].ok = call.ok
        else:
            merged[key] = call
    return [
        RecipeCall(
            recipe_id=c.recipe_id,
            title=c.recipe_id or "未知配方",
            params=c.params,
            ok=c.ok,
            count=c.count,
        )
        for c in merged.values()
    ]


def _recipe_title(library: Any, recipe_id: str) -> str:
    try:
        recipe = library.get(recipe_id) if library is not None else None
    except Exception:  # pragma: no cover - 库实现异常不影响报告
        recipe = None
    return str(getattr(recipe, "name", "") or recipe_id or "未知配方")


def plain_methods(methodology: str, recipes: Iterable[RecipeCall]) -> list[str]:
    """方法区：先用模型自己写的方法论，再补"哪个配方、什么参数"。"""
    lines: list[str] = []
    text = str(methodology or "").strip()
    if text:
        for block in text.split("\n"):
            block = block.strip().lstrip("-*• ").strip()
            if block:
                lines.append(_short(block, 200))
    for call in recipes:
        lines.append(call.describe())
    return lines


# ------------------------------------------------------------------ 数据来源
def source_rows(catalog: Any, workspace: str | Path | None = None, *, limit: int = 12) -> list[SourceRow]:
    """catalog 登记的输入数据（来源/许可/采集时间），跳过 ``output/`` 下的成果。"""
    if catalog is None:
        return []
    try:
        records = catalog.list_datasets(limit=200)
    except Exception as exc:  # pragma: no cover - 老库/锁
        logger.debug("读 catalog 失败: %s", exc)
        return []
    root = Path(str(workspace)).resolve() if workspace else None
    # 统一成反斜杠前的正斜杠：catalog 里的路径是 Windows 反斜杠，不归一会漏过滤
    out_dir = str((root / "output").resolve()).replace("\\", "/").lower() if root else ""
    rows: list[SourceRow] = []
    for record in records:
        raw = str(getattr(record, "path", "") or "")
        if not raw:
            continue
        normalized = raw.replace("\\", "/").lower()
        if out_dir and normalized.startswith(out_dir):
            continue
        if "/.gis_agent/" in normalized:
            continue
        name = str(getattr(record, "name", "") or Path(raw).name)
        rows.append(
            SourceRow(
                name=name,
                path=_relative(raw, root),
                source=_SOURCE_PLAIN.get(str(getattr(record, "source", "") or ""), "来源未标注"),
                license=str(getattr(record, "license", "") or "未标注"),
                acquired_at=str(getattr(record, "acquired_at", "") or "未记录"),
            )
        )
    return rows[:limit]


def _relative(path: str, workspace: Path | None) -> str:
    raw = str(path or "").replace("\\", "/")
    if workspace is None:
        return raw
    try:
        return str(Path(path).relative_to(workspace)).replace("\\", "/")
    except (ValueError, OSError):
        return raw


# ------------------------------------------------------------------ 组装
def _split_artifacts(artifacts: Iterable[str]) -> tuple[list[ImageRef], list[LayerRef]]:
    images: list[ImageRef] = []
    layers: list[LayerRef] = []
    seen: set[str] = set()
    for raw in artifacts:
        path = str(raw)
        if not path or path in seen:
            continue
        seen.add(path)
        if is_image(path):
            images.append(ImageRef(caption=Path(path).stem, path=path))
        elif is_vector(path):
            # 图层名给人看：去掉 .shp/.geojson 尾巴（GDB 内的要素类本来就没后缀）
            layers.append(LayerRef(name=Path(path).stem or Path(path).name, path=path))
    return images, layers


def build_content(
    *,
    goal: str = "",
    summary: str = "",
    methodology: str = "",
    artifacts: Iterable[str] = (),
    verdict: dict[str, Any] | None = None,
    requirements: Iterable[Any] = (),
    catalog: Any = None,
    workspace: str | Path | None = None,
    trace_path: str | Path | None = None,
    library: Any = None,
) -> ReportContent:
    """把一次运行的材料整理成成果包内容（纯函数，不写文件）。"""
    paths = [str(p) for p in artifacts if str(p).strip()]
    images, layers = _split_artifacts(paths)
    verdict = verdict or {}
    recipes = recipes_from_trace(trace_path)
    if library is not None:
        for call in recipes:
            call.title = _recipe_title(library, call.recipe_id)
    return ReportContent(
        title=_short(_first_line(goal), 80) or "GIS 分析成果",
        conclusion=_short(summary, 400) or "本次没有留下文字结论。",
        key_numbers=key_numbers(paths, verdict.get("structural")),
        methods=plain_methods(methodology, recipes),
        sources=source_rows(catalog, workspace),
        verifications=plain_verifications(verdict, requirements),
        images=images,
        layers=layers,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def _first_line(text: Any) -> str:
    for line in str(text or "").splitlines():
        line = " ".join(line.split())
        if line:
            return line
    return ""


__all__ = [
    "DISCLAIMER",
    "IMAGE_EXTS",
    "KeyNumber",
    "LayerRef",
    "ImageRef",
    "RecipeCall",
    "ReportContent",
    "SourceRow",
    "build_content",
    "is_image",
    "is_raster",
    "is_vector",
    "key_numbers",
    "plain_methods",
    "plain_verifications",
    "recipes_from_trace",
    "source_rows",
]

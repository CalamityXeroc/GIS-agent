# -*- coding: utf-8 -*-
"""成果包（P0-4）：交付时生成**单文件** HTML + PDF。

验收口径（P0-4 §4.4）：断网能打开、单文件 ≤20MB、不需要 ArcGIS、不联网。
所以这里只做编排，不新增能力：

1. ``narrative`` 把验收结论/配方调用翻成人话；
2. ``html_pack`` 内嵌图片（超 1600px 先缩）与 GeoJSON，模板零外部请求；
3. ``pdf_pack`` 用 matplotlib ``PdfPages`` 拼结论页/图页/方法页（可不装）。

任何一步失败都只写进 ``notes``，绝不阻断交付：成果包是交付的**附加物**，
不是交付本身。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import narrative, html_pack, pdf_pack

logger = logging.getLogger(__name__)

#: 单文件硬上限（验收条件）。
MAX_HTML_BYTES = 20 * 1024 * 1024
#: 最多内联几个图层/几张图（报告不是数据交付）。
MAX_LAYERS = 6
MAX_IMAGES = 8
#: 体积超标时的降档梯度：(图的宽上限, JPEG 质量)。第一档就是验收要求的 1600px。
IMAGE_PROFILES = ((1600, 82), (1280, 78), (1000, 74), (800, 70))

HTML_NAME = "成果包.html"
PDF_NAME = "成果包.pdf"


@dataclass
class Deliverables:
    """一次成果包生成的结果（路径 + 做了什么 + 没做成什么）。"""

    html: Path | None = None
    pdf: Path | None = None
    layers: int = 0
    images: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def paths(self) -> list[str]:
        return [str(path) for path in (self.html, self.pdf) if path is not None]


def build_deliverables(
    *,
    workspace: str | Path,
    state: Any,
    verdict: dict[str, Any] | None = None,
    catalog: Any = None,
    code_runner: Any = None,
    trace_path: str | Path | None = None,
    recipes: Any = None,
    out_dir: str | Path | None = None,
    want_pdf: bool = True,
    max_bytes: int = MAX_HTML_BYTES,
) -> Deliverables:
    """生成成果包；没有产物时直接返回空结果（不写任何文件）。"""
    ws = Path(workspace)
    artifacts = [str(item) for item in (getattr(state, "artifacts", []) or []) if str(item).strip()]
    result = Deliverables()
    if not artifacts:
        result.notes.append("本次没有任何产物，未生成成果包")
        return result

    content = narrative.build_content(
        goal=str(getattr(state, "goal", "") or ""),
        summary=str(getattr(state, "final_summary", "") or ""),
        methodology=str(getattr(state, "methodology", "") or ""),
        artifacts=artifacts,
        verdict=verdict,
        requirements=getattr(state, "requirements", []) or [],
        catalog=catalog,
        workspace=ws,
        trace_path=trace_path,
        library=recipes,
    )

    layers, layer_notes = _collect_layers(content.layers, code_runner=code_runner, workspace=ws)
    image_refs, image_notes = _collect_images(content.images)
    result.notes.extend(layer_notes + image_notes)

    kept_images, html_text = _fit_html(content, image_refs, layers, max_bytes, result.notes)
    target_dir = Path(out_dir) if out_dir else ws / "output"
    target_dir.mkdir(parents=True, exist_ok=True)
    # 写的就是刚才量过体积的那份字符串：重渲染一次虽然结果相同，但“量过的”与
    # “写出去的”不是同一份字节，验收口径就只是巧合了。``newline="\n"`` 同理：
    # 默认会把 \n 翻成 \r\n（Windows），磁盘上就比量过的多出几 KB。
    result.html = target_dir / HTML_NAME
    result.html.write_text(html_text, encoding="utf-8", newline="\n")
    # ``_fit_html`` 会就地裁掉放不下的图层，这里数的就是实际内联的层数。
    result.layers = len(layers)
    result.images = len(kept_images)

    if want_pdf:
        pdf_path = pdf_pack.write_pdf(content, out_path=target_dir / PDF_NAME, images=kept_images)
        if pdf_path is None:
            result.notes.append("未生成 PDF（本机没有可用的 matplotlib）")
        else:
            result.pdf = pdf_path
    return result


# ------------------------------------------------------------------ 收集
def _collect_layers(
    refs: Iterable[narrative.LayerRef],
    *,
    code_runner: Any,
    workspace: Path,
) -> tuple[list[tuple[narrative.LayerRef, dict[str, Any]]], list[str]]:
    refs = list(refs)
    layers: list[tuple[narrative.LayerRef, dict[str, Any]]] = []
    notes: list[str] = []
    for ref in refs[:MAX_LAYERS]:
        collection = html_pack.load_layer(ref.path, code_runner=code_runner, workspace=workspace)
        if not collection or not collection.get("features"):
            notes.append(f"图层「{ref.name}」导不出几何，未内联")
            continue
        layers.append((ref, html_pack.thin_features(collection)))
    if len(refs) > MAX_LAYERS:
        notes.append(f"矢量成果较多，只内联了前 {MAX_LAYERS} 个图层")
    return layers, notes


def _collect_images(refs: Iterable[narrative.ImageRef]) -> tuple[list[narrative.ImageRef], list[str]]:
    """只限额和去重；真正的读图/缩图在 :func:`_fit_html` 里按体积梯度做一次。"""
    refs = list(refs)
    notes: list[str] = []
    if len(refs) > MAX_IMAGES:
        notes.append(f"成果图较多，只内嵌了前 {MAX_IMAGES} 张")
    return refs[:MAX_IMAGES], notes


def _fit_html(
    content: narrative.ReportContent,
    refs: list[narrative.ImageRef],
    layers: list[tuple[narrative.LayerRef, dict[str, Any]]],
    max_bytes: int,
    notes: list[str],
) -> tuple[list[tuple[narrative.ImageRef, str, bytes]], str]:
    """把 HTML 压进 ``max_bytes``：先降图档，再少放几张，最后连图层一起丢。

    ``notes`` 是调用方已经攒下的"没内联什么"清单（图层导出失败等），会被原样
    写进页面；这里只往里追记裁剪相关的说明。

    ``layers`` 是**就地裁剪**的：返回后调用方手里那份列表就是实际内联的层，
    图层清单与"内联了几个图层"因此不会跟页面上画出来的对不上。
    """
    kept: list[tuple[narrative.ImageRef, str, bytes]] = []
    failed: set[str] = set()
    text = ""
    for max_width, quality in IMAGE_PROFILES:
        kept = []
        for ref in refs:
            if ref.path in failed:
                continue
            got = html_pack.prepare_image(ref.path, max_width=max_width, quality=quality)
            if got is None:
                failed.add(ref.path)
                notes.append(f"图「{ref.caption}」读不出来，未内嵌")
                continue
            kept.append((ref, got[0], got[1]))
        text = html_pack.render_html(content, images=kept, layers=layers, notes=notes)
        if len(text.encode("utf-8")) <= max_bytes:
            return kept, text

    # 图已经压到最小还超限：逐张减少，宁可少放图也不能突破体积上限。
    while kept and len(text.encode("utf-8")) > max_bytes:
        dropped = kept.pop()
        notes.append(f"为保证单文件体积上限，图「{dropped[0].caption}」未内嵌")
        text = html_pack.render_html(content, images=kept, layers=layers, notes=notes)

    # 图丢光了还超限：再逐层丢。多环面图层抽稀压不下去，只能少画一层。
    if len(text.encode("utf-8")) > max_bytes:
        while layers and len(text.encode("utf-8")) > max_bytes:
            dropped_layer = layers.pop()
            notes.append(f"为保证单文件体积上限，图层「{dropped_layer[0].name}」未内联")
            text = html_pack.render_html(content, images=kept, layers=layers, notes=notes)
        notes.append(f"成果包超过 {round(max_bytes / (1024 * 1024), 1):g}MB，已按体积上限裁剪")
        # 兜底说明也要出现在页面上：重渲染一次，才算真写进去了。
        text = html_pack.render_html(content, images=kept, layers=layers, notes=notes)
    return kept, text


__all__ = [
    "Deliverables",
    "HTML_NAME",
    "IMAGE_PROFILES",
    "MAX_HTML_BYTES",
    "MAX_IMAGES",
    "MAX_LAYERS",
    "PDF_NAME",
    "build_deliverables",
]

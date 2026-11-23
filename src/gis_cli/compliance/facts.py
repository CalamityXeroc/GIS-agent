# -*- coding: utf-8 -*-
"""合规事实采集：把"这次出图到底画了什么"压成一组可判定的事实。

**分层**：本模块只做**事实**（画了哪些图层、图面写了什么、数据目录登记了什么），
不做任何"该不该报"的判断 —— 判断在 ``rules.yaml`` 里，是数据不是代码。
这样新增/调整规则不用改这里，也就能在没有 ArcGIS 的机器上单测规则。

事实的来源只有三处，全部由调用方显式传入：

- 图层清单（``MapContext.layers``）：名称/路径来自本次出图的图层，来源与许可
  来自数据目录（catalog）的登记记录；
- 图面文字（``MapContext.texts``）：图名、图例标题、叠加标注、注记等**看得见的字**；
- 调用方声明（``approval_number`` / ``watermark`` / ``audience``）：审图号、
  未送审水印、对外还是内部使用。

⚠️ 边界：本模块**不读像素、不识别图片**。"声明了审图号"与"图上真的画了审图号"
是两件事，闸门只核前者 —— 图面是否真画上去属于制图质量，由 ``qc`` 与人工负责。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

#: 国界/敏感要素：命中即按"须送审"处理（分级阻断）。
#: 只收**国界级/军事级**词，绝不收"敏感"这种泛词 —— "生态敏感区"是常用图层名，
#: 收进来就会把普通生态图误判成阻断（误报会直接让用户关掉闸门）。
SENSITIVE_PATTERNS: tuple[str, ...] = (
    "国界",
    "国境线",
    "边境线",
    "疆界",
    "南海诸岛",
    "九段线",
    "曾母暗沙",
    "钓鱼岛",
    "赤尾屿",
    "藏南",
    "阿克赛钦",
    "领海基线",
    "领海线",
    "军事禁区",
    "军事设施",
    "national_border",
    "national_boundary",
    "country_border",
    "international_boundary",
    "nine_dash",
    "south_china_sea",
)

#: 边界类图层（含国界/省界/县界/海岸线等）：命中才判"来源未登记"。
BOUNDARY_PATTERNS: tuple[str, ...] = SENSITIVE_PATTERNS + (
    "边界",
    "界线",
    "行政界",
    "省界",
    "市界",
    "县界",
    "区界",
    "乡界",
    "镇界",
    "海岸线",
    "岸线",
    "boundary",
    "border",
    "coastline",
    "admin",
)

#: 审图号的标准形态：``GS(2023)1234号`` / ``GS（2019）第1822号``。
APPROVAL_VALID = re.compile(r"GS\s*[（(]\s*((?:19|20)\d{2})\s*[)）]\s*第?\s*\d{1,5}\s*号", re.IGNORECASE)
#: "像审图号但格式不对"的线索（``审图号：2023-1234``、``GS2023-1234``）。
APPROVAL_HINT = re.compile(r"审图号|审图编号|GS\s*[（(]?\s*(?:19|20)\d{2}", re.IGNORECASE)

#: "未送审"水印的写法（任一命中即算已标）。
WATERMARK_PATTERNS: tuple[str, ...] = (
    "未送审",
    "送审用图",
    "内部使用",
    "内部资料",
    "内部传阅",
    "不得对外发布",
    "禁止外传",
    "not for publication",
    "internal use",
    "draft",
)

#: 数据目录里"等于没登记来源"的取值（``local`` 是扫描器给本机文件的默认值）。
UNREGISTERED_SOURCES: frozenset[str] = frozenset({"", "local", "unknown"})

#: 单个事实文本的最大长度（进报告/边车，别把整条路径铺进去）。
TEXT_LIMIT = 120


@dataclass(frozen=True)
class LayerRef:
    """一个参与出图的图层：名字/路径，以及数据目录里登记到的来源。"""

    name: str = ""
    path: str = ""
    source: str = ""
    license: str = ""
    #: 数据目录里是否有这条数据的登记记录（False = 没登记）。
    registered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "source": self.source,
            "license": self.license,
            "registered": self.registered,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "LayerRef":
        return cls(
            name=str(data.get("name") or ""),
            path=str(data.get("path") or ""),
            source=str(data.get("source") or ""),
            license=str(data.get("license") or ""),
            registered=bool(data.get("registered")),
        )

    @property
    def match_text(self) -> str:
        """关键词匹配用的文本：图层名 + 路径（目录名也可能带"国界数据"这类线索）。"""
        return f"{self.name} {self.path}".lower()


@dataclass
class MapContext:
    """一次出图的合规上下文（可序列化，随边车落盘，验收时按当前规则重算）。"""

    texts: list[str] = field(default_factory=list)
    layers: list[LayerRef] = field(default_factory=list)
    #: public（对外，默认）| internal（内部/送审前）
    audience: str = "public"
    #: 调用方声明的审图号（图面文字里扫不到时的补充证据）
    approval_number: str = ""
    #: 调用方声明的未送审水印文字
    watermark: str = ""
    #: 是否拿到了数据目录。False = 来源无法核验，规则②不下结论（不许猜）。
    catalog_available: bool = False
    #: 这张图的名字（写进边车，汇总时能指回是哪张图）
    image: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "texts": [_clip(text) for text in self.texts if str(text).strip()],
            "layers": [layer.to_dict() for layer in self.layers],
            "audience": self.audience,
            "approval_number": self.approval_number,
            "watermark": self.watermark,
            "catalog_available": bool(self.catalog_available),
            "image": self.image,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MapContext | None":
        if not isinstance(data, Mapping):
            return None
        layers = data.get("layers")
        return cls(
            texts=[str(item) for item in (data.get("texts") or []) if str(item).strip()],
            layers=[LayerRef.from_dict(item) for item in (layers if isinstance(layers, list) else []) if isinstance(item, Mapping)],
            audience=str(data.get("audience") or "public"),
            approval_number=str(data.get("approval_number") or ""),
            watermark=str(data.get("watermark") or ""),
            catalog_available=bool(data.get("catalog_available")),
            image=str(data.get("image") or ""),
        )


# ------------------------------------------------------------------ 小工具
def _clip(value: Any, limit: int = TEXT_LIMIT) -> str:
    flat = " ".join(str(value if value is not None else "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _join_names(items: Iterable[str], limit: int = TEXT_LIMIT) -> str:
    seen: list[str] = []
    for item in items:
        text = _clip(item, 40)
        if text and text not in seen:
            seen.append(text)
    return _clip("、".join(seen), limit)


#: 关键词 → 编译好的正则（模块级缓存：规则库是常量，重复编译没意义）。
_PATTERN_RE: dict[str, re.Pattern[str]] = {}


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    """把关键词编译成匹配器。

    ASCII 关键词要求**左右不是字母/数字**，否则会误命中无关的路径片段：
    实测 CI 跑在 GitHub Actions 上，临时目录是
    ``C:/Users/runneradmin/AppData/...``，里面的 ``runneradmin`` 含子串
    ``admin``，于是任何图层都被判成"边界类图层"（2026-11-13 CI 红）。
    中文关键词仍按子串匹配（中文没有词边界概念）。
    """
    cached = _PATTERN_RE.get(pattern)
    if cached is not None:
        return cached
    if re.fullmatch(r"[a-z0-9_]+", pattern):
        compiled = re.compile(rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])")
    else:
        compiled = re.compile(re.escape(pattern))
    _PATTERN_RE[pattern] = compiled
    return compiled


def _matches(layer: LayerRef, patterns: Iterable[str]) -> bool:
    text = layer.match_text
    return any(_pattern_regex(pattern).search(text) for pattern in patterns)


def normalize_audience(value: Any) -> str:
    """只认 public / internal；其余（含空）一律按 public（对外），因为"漏标"比"误标内部"更危险。"""
    text = str(value or "").strip().lower()
    if text in ("internal", "inside", "private", "内部", "未送审", "送审", "review"):
        return "internal"
    return "public"


# ------------------------------------------------------------------ 事实
def _approval_status(texts: Iterable[str]) -> tuple[str, str]:
    """审图号状态：``valid`` / ``malformed`` / ``absent`` + 一句人读说明。"""
    ordered = [str(text) for text in texts if str(text).strip()]
    for text in ordered:
        found = APPROVAL_VALID.search(text)
        if found:
            return "valid", f"审图号 {found.group(0).strip()}"
    for text in ordered:
        found = APPROVAL_HINT.search(text)
        if found:
            return "malformed", f"审图号「{_clip(found.group(0), 40)}」格式无法识别（应为 GS(年份)编号号）"
    return "absent", "未见审图号"


def _has_watermark(texts: Iterable[str]) -> bool:
    for text in texts:
        lowered = str(text or "").lower()
        if any(pattern in lowered for pattern in WATERMARK_PATTERNS):
            return True
    return False


def _layer_source_status(layer: LayerRef) -> str:
    """单个边界图层的来源状态：registered / local（只登记成本机文件）/ unregistered。"""
    if not layer.registered:
        return "unregistered"
    if str(layer.source or "").strip().lower() in UNREGISTERED_SOURCES:
        return "local"
    return "registered"


def collect_facts(context: MapContext) -> dict[str, Any]:
    """把 :class:`MapContext` 折算成规则可判定的事实（纯函数，不抛异常）。"""
    context = context if isinstance(context, MapContext) else MapContext()
    layers = list(context.layers or [])
    sensitive = [layer for layer in layers if _matches(layer, SENSITIVE_PATTERNS)]
    boundary = [layer for layer in layers if _matches(layer, BOUNDARY_PATTERNS)]
    # 数据目录不可用时，来源状态记 unknown：拿不到证据就不下“未登记”的结论
    if context.catalog_available:
        unregistered = [layer for layer in boundary if _layer_source_status(layer) != "registered"]
        if any(_layer_source_status(layer) == "unregistered" for layer in boundary):
            boundary_source = "unregistered"
        elif unregistered:
            boundary_source = "partial"        # 有登记、但只登记成本机文件/未知来源
        else:
            boundary_source = "registered"
    else:
        unregistered = []
        boundary_source = "unknown"

    texts = [str(text) for text in (context.texts or []) if str(text).strip()]
    approval, approval_note = _approval_status([*texts, context.approval_number])
    watermark = "present" if _has_watermark([context.watermark, *texts]) else "absent"
    # 图名/注记里直接写了国界/敏感词也算“涉及”：真实图件的图层名常常是
    # boundary.shp 这类泛名，线索全在图名上（加这条是为了少漏报，不是放宽阻断）。
    lowered_texts = [text.lower() for text in texts]
    text_hits = [
        pattern for pattern in SENSITIVE_PATTERNS
        if any(pattern in text for text in lowered_texts)
    ]
    hint_parts: list[str] = []
    if sensitive:
        hint_parts.append(f"图面含 {len(sensitive)} 个国界/敏感要素图层（{_join_names(layer.name or layer.path for layer in sensitive)}）")
    if text_hits:
        mention = f"图名/注记提到「{_join_names(text_hits, 40)}」"
        hint_parts.append(f"图名/注记也提到「{_join_names(text_hits, 40)}」" if sensitive else mention)

    return {
        "layer_count": len(layers),
        "sensitive_count": len(sensitive),
        "sensitive_layers": _join_names(layer.name or layer.path for layer in sensitive),
        "sensitive_in_text": "present" if text_hits else "absent",
        "sensitive_hint": "，".join(hint_parts),
        "boundary_count": len(boundary),
        "boundary_layers": _join_names(layer.name or layer.path for layer in boundary),
        "unregistered_count": len(unregistered),
        "unregistered_layers": _join_names(layer.name or layer.path for layer in unregistered),
        "approval": approval,
        "approval_note": approval_note,
        "watermark": watermark,
        "boundary_source": boundary_source,
        "catalog": "available" if context.catalog_available else "unavailable",
        "audience": normalize_audience(context.audience),
    }


# ------------------------------------------------------------------ 上下文构造
def _iter_catalog_rows(rows: Iterable[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows or []:
        if isinstance(row, Mapping):
            out.append(dict(row))
            continue
        out.append(
            {
                "path": str(getattr(row, "path", "") or ""),
                "name": str(getattr(row, "name", "") or ""),
                "source": str(getattr(row, "source", "") or ""),
                "license": str(getattr(row, "license", "") or ""),
            }
        )
    return out


def _normalize_path(raw: Any) -> str:
    return str(raw or "").replace("\\", "/").strip().lower()


def _find_catalog_row(path: str, rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """图层路径 ↔ 目录记录：先精确匹配，再退一步按"一方是另一方的后缀/前缀"匹配（GDB 成员）。"""
    target = _normalize_path(path)
    if not target:
        return None
    candidates = [row for row in rows if isinstance(row, Mapping)]
    for row in candidates:
        if _normalize_path(row.get("path")) == target:
            return row
    for row in candidates:
        other = _normalize_path(row.get("path"))
        if not other:
            continue
        if other.endswith("/" + target) or target.endswith("/" + other):
            return row
    return None


def _layer_name(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    # GDB 成员是 ``x.gdb/要素类``：取最后一段，别把整个库名当图层名
    last = text.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    stem = Path(last).stem if "." in last else last
    return stem or last


def catalog_rows_for(
    catalog: Any,
    layers: Iterable[Any],
    *,
    limit: int = 500,
) -> list[dict[str, Any]] | None:
    """从数据目录里挑出与本次图层相关的登记记录，喂给合规上下文。

    只留四个字段（path/name/source/license）：整个目录可能有几百条，全部塞进
    内核调用的代码字符串既慢又没必要。``catalog is None`` → 返回 ``None``
    （= 拿不到目录，rule② 不下结论），与"目录里没有这条记录"（= 未登记）区分开。
    """
    if catalog is None:
        return None
    try:
        records = catalog.list_datasets(limit=limit)
    except Exception as exc:  # 老库/锁/实现异常都不该让自检崩
        logger.debug("合规自检读数据目录失败: %s", exc)
        return None
    wanted = [_normalize_path(getattr(item, "path", "") or item) for item in layers]
    wanted = [item for item in wanted if item]
    rows: list[dict[str, Any]] = []
    for record in records:
        path = str(getattr(record, "path", "") or "")
        norm = _normalize_path(path)
        if not norm:
            continue
        base = norm.rsplit("/", 1)[-1]
        if not any(
            norm == item or norm.endswith("/" + item) or item.endswith("/" + norm)
            or base == item.rsplit("/", 1)[-1]
            for item in wanted
        ):
            continue
        rows.append(
            {
                "path": path,
                "name": str(getattr(record, "name", "") or ""),
                "source": str(getattr(record, "source", "") or ""),
                "license": str(getattr(record, "license", "") or ""),
            }
        )
    return rows


def context_from_spec(
    spec: Mapping[str, Any] | None,
    *,
    layers: Iterable[Any] | None = None,
    catalog_rows: Iterable[Any] | None = None,
    intent: Mapping[str, Any] | None = None,
    image: str | Path = "",
) -> MapContext:
    """从 ``LayoutSpec``（create_map 的设计规格）构造上下文。

    - 图层：优先用调用方传的 ``layers``（可能是 Layer 对象），否则读
      ``spec["renderer"]["layer_paths"]``；
    - 文字：图名、图例标题、叠加层标注 —— 这些是**真会落到图面上**的字；
    - 声明：``approval_number`` / ``watermark`` / ``audience`` 从 intent 取。

    ``catalog_rows is None`` 表示没有拿到数据目录，来源类规则不下结论。
    """
    spec = spec if isinstance(spec, Mapping) else {}
    payload = intent if isinstance(intent, Mapping) else (spec.get("intent") or {})
    payload = payload if isinstance(payload, Mapping) else {}

    texts: list[str] = []
    title = spec.get("title")
    if isinstance(title, Mapping) and title.get("text"):
        texts.append(str(title["text"]))
    legend = spec.get("legend")
    if isinstance(legend, Mapping) and legend.get("title"):
        texts.append(str(legend["title"]))
    overlay = spec.get("overlay") if isinstance(spec.get("overlay"), Mapping) else {}
    for item in overlay.get("labels") or []:
        if isinstance(item, Mapping) and item.get("text"):
            texts.append(str(item["text"]))
    for item in overlay.get("arrows") or []:
        if isinstance(item, Mapping) and item.get("label"):
            texts.append(str(item["label"]))
    for key in ("title", "note", "legend_title", "map_kind", "region", "scale", "theme"):
        if payload.get(key):
            texts.append(str(payload[key]))

    raw_layers = [item for item in (layers if layers is not None else (spec.get("renderer") or {}).get("layer_paths") or []) if item]
    rows = _iter_catalog_rows(catalog_rows)
    refs: list[LayerRef] = []
    for item in raw_layers:
        path = str(getattr(item, "path", "") or item)
        row = _find_catalog_row(path, rows)
        refs.append(
            LayerRef(
                name=_layer_name(path),
                path=path,
                source=str(row.get("source") or "") if row else "",
                license=str(row.get("license") or "") if row else "",
                registered=row is not None,
            )
        )

    watermark = payload.get("watermark")
    if isinstance(watermark, bool):
        watermark = "未送审" if watermark else ""
    return MapContext(
        texts=texts,
        layers=refs,
        audience=normalize_audience(payload.get("audience")),
        approval_number=str(payload.get("approval_number") or ""),
        watermark=str(watermark or ""),
        catalog_available=catalog_rows is not None,
        image=str(image or ""),
    )


__all__ = [
    "APPROVAL_HINT",
    "APPROVAL_VALID",
    "BOUNDARY_PATTERNS",
    "LayerRef",
    "MapContext",
    "SENSITIVE_PATTERNS",
    "UNREGISTERED_SOURCES",
    "WATERMARK_PATTERNS",
    "collect_facts",
    "context_from_spec",
    "catalog_rows_for",
    "normalize_audience",
]

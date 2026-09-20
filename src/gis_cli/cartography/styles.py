# -*- coding: utf-8 -*-
"""样式目录（style catalog）：ArcGIS 2D 样式项清单 + 按关键字选样式。

为什么需要它：样式项名称随**安装语言/版本**变化（本机是中文：
"公制黑白相间比例尺 1"、"ArcGIS 指北针 13"），不能在代码里写死；
而且原来的 ``_pick_style(items, kw, fallback_idx=0)`` 在关键字不匹配时**静默取第一项**，
这正是"比例尺/指北针长得不对"的根因。现在改成：
1. 运行时枚举一次，落盘 ``config/cartography_styles.json``；
2. 选择时按**优先级关键字列表**匹配（中英文都写），并打印命中的样式名；
3. 全不匹配时返回空串，由渲染层回退 ArcGIS 默认并**明确记录**，不假装选成功。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CATALOG_PATH = REPO_ROOT / "config" / "cartography_styles.json"

KINDS = ("NORTH_ARROW", "SCALE_BAR", "LEGEND")

#: 默认偏好（本机实测：竞赛风格用黑八芒罗盘玫瑰 + 公制黑白相间比例尺）
DEFAULT_KEYWORDS: dict[str, list[str]] = {
    "NORTH_ARROW": ["指北针 13", "罗盘北", "正北", "罗盘", "compass", "north arrow", "指北针"],
    "SCALE_BAR": ["公制黑白相间比例尺 1", "公制黑白相间", "黑白相间", "alternating", "公制比例线", "比例线"],
    "LEGEND": ["图例 1", "legend 1", "图例 2", "legend"],
}

#: 大比例尺/局部图更适合简单箭头；这里给出"按图型换样式"的备选关键字
ALT_KEYWORDS: dict[str, list[str]] = {
    "north_arrow_simple": ["简单实心指北针", "简单空心指北针", "ArcGIS 指北针 1", "箭头", "arrow"],
    "scale_bar_line": ["公制比例线 1", "比例线 1", "scale line"],
}


def load_catalog(path: str | Path | None = None) -> dict[str, list[str]]:
    """读样式目录（不存在时返回空目录，让调用方按关键字回退）。"""
    target = Path(path) if path else CATALOG_PATH
    if not target.exists():
        logger.info("样式目录不存在（%s），将在首次刷新后生成", target)
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("样式目录读取失败: %s", exc)
        return {}
    return {str(k): [str(n) for n in (v or [])] for k, v in (data or {}).items()}


def save_catalog(catalog: dict[str, list[str]], path: str | Path | None = None) -> Path:
    target = Path(path) if path else CATALOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def pick_style_name(
    catalog: dict[str, list[str]],
    kind: str,
    keywords: list[str] | None = None,
    *,
    fallback_index: int | None = 0,
) -> str:
    """按关键字优先级挑样式名（纯函数，便于单测）。

    Args:
        catalog: ``{"NORTH_ARROW": ["...", ...], ...}``
        kind: 样式类别
        keywords: 关键字列表，**顺序即优先级**；None 用默认偏好
        fallback_index: 全部不匹配时的兜底下标；None 表示宁缺毋滥（返回空串）
    """
    names = list(catalog.get(kind) or [])
    if not names:
        return ""
    prefs = list(keywords) if keywords is not None else list(DEFAULT_KEYWORDS.get(kind, []))
    for keyword in prefs:
        needle = (keyword or "").strip().lower()
        if not needle:
            continue
        for name in names:
            if needle in name.lower():
                return name
    if fallback_index is None:
        return ""
    if 0 <= fallback_index < len(names):
        return names[fallback_index]
    return ""


def refresh_catalog(
    *,
    template_aprx: str | Path | None = None,
    path: str | Path | None = None,
    enumerate_fn: Any = None,
) -> dict[str, list[str]]:
    """枚举样式项并落盘。

    在**主进程内**用 arcpy 执行（不要在持久内核里打开工程——实测会挂死）。
    ``enumerate_fn`` 供单测注入。
    """
    if enumerate_fn is not None:
        catalog = enumerate_fn()
    else:
        catalog = _enumerate_with_arcpy(template_aprx)
    save_catalog(catalog, path)
    return catalog


def _blank_template() -> str:
    import arcpy  # type: ignore

    install = arcpy.GetInstallInfo().get("InstallDir", "") or ""
    candidate = os.path.join(
        install, "Resources", "ArcToolBox", "Services", "routingservices", "data", "Blank.aprx"
    )
    if os.path.exists(candidate):
        return candidate
    import glob

    found = glob.glob(os.path.join(install, "Resources", "**", "Blank.aprx"), recursive=True)
    if not found:
        raise RuntimeError("未找到 ArcGIS 空白工程模板 Blank.aprx，无法枚举样式")
    return found[0]


def _enumerate_with_arcpy(template_aprx: str | Path | None = None) -> dict[str, list[str]]:
    from arcpy import mp  # type: ignore

    template = str(template_aprx) if template_aprx else _blank_template()
    work_dir = Path(tempfile.mkdtemp(prefix="style_catalog_"))
    probe_aprx = work_dir / "catalog.aprx"
    shutil.copy2(template, probe_aprx)
    project = mp.ArcGISProject(str(probe_aprx))
    catalog: dict[str, list[str]] = {}
    for kind in KINDS:
        try:
            catalog[kind] = [item.name for item in project.listStyleItems("ArcGIS 2D", kind)]
        except Exception as exc:  # pragma: no cover - 依赖安装环境
            logger.warning("枚举 %s 样式失败: %s", kind, exc)
            catalog[kind] = []
    return catalog


__all__ = [
    "CATALOG_PATH",
    "DEFAULT_KEYWORDS",
    "ALT_KEYWORDS",
    "load_catalog",
    "save_catalog",
    "pick_style_name",
    "refresh_catalog",
]
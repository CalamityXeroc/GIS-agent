# -*- coding: utf-8 -*-
"""统一的中文字体与绘图环境配置（matplotlib 与 Pillow 共用常量）。

为什么需要这个模块
------------------
实测（第 14 届基准 任务三a）：agent 用 matplotlib 自绘的对比图，标题、图例、
坐标轴**全部渲染成方框**（tofu）——matplotlib 默认字体 DejaVu Sans 没有中文字形，
而内核里没有任何字体配置。更糟的是这类问题只写 stderr，成功路径不会回喂给模型，
于是"乱码图"一路通过验收。

因此这里做三件事：
1. 给出中文字体文件与家族名的统一候选（Pillow 侧 image_overlay 复用同一份）；
2. ``configure()`` 把 matplotlib 配成"中文能看"的状态（注册字体文件 + 设 rcParams）；
3. 对缺失依赖完全静默降级，绝不影响主流程。

注意：**不要**使用 ``simsunb.ttf``（SimSun-ExtB 只有生僻扩展字，常见汉字会变方框）。
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

#: Windows 中文字体文件候选（按优先级）。ttf/ttc 都能被 matplotlib 注册。
CJK_FONT_FILES: tuple[str, ...] = (
    r"C:\Windows\Fonts\msyh.ttc",   # 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",  # 黑体
    r"C:\Windows\Fonts\simsun.ttc",  # 宋体
    r"C:\Windows\Fonts\msjh.ttc",    # 微软正黑（繁体）
)

#: 兜底家族名：即使字体文件注册失败，这些名字也排在 DejaVu Sans 前面。
CJK_FAMILIES: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Microsoft JhengHei",
)


def available_font_files() -> list[str]:
    """返回本机实际存在的中文字体文件（找不到时为空列表）。"""
    return [path for path in CJK_FONT_FILES if os.path.exists(path)]


def resolve_cjk_font() -> str:
    """返回第一个可用的中文字体文件路径（找不到返回空串）。"""
    files = available_font_files()
    return files[0] if files else ""


def configure(*, dpi: int = 120) -> dict:
    """把 matplotlib 配成"中文能看"的状态（best-effort，绝不抛异常）。

    返回 ``{"available", "font", "families", "backend"}``：
    - ``available``: matplotlib 是否可用且配置成功
    - ``font``: 实际注册并解析到的中文字体文件
    - ``families``: 写入 ``font.sans-serif`` 的家族名（按优先级）
    - ``backend``: 当前 matplotlib 后端

    要点：
    - 只在 **还没导入 pyplot** 时切到 Agg（后台/服务端无窗口；pyplot 已导入后再
      ``use()`` 会告警甚至报错）。
    - 按 **路径** 注册字体文件，然后读取该文件的真实家族名再设进 rcParams——
      硬编码家族名在不同 Windows 版本上可能对不上（如 ``SimHei`` 与 ``SimHei Regular``）。
    """
    info: dict = {"available": False, "font": "", "families": [], "backend": ""}
    try:
        import matplotlib  # type: ignore
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        logger.debug("matplotlib 不可用，跳过中文字体配置: %s", exc)
        return info

    try:
        if "matplotlib.pyplot" not in sys.modules:
            try:
                matplotlib.use("Agg")
            except Exception as exc:  # pragma: no cover
                logger.debug("切换 Agg 后端失败: %s", exc)

        from matplotlib import font_manager  # type: ignore

        families: list[str] = []
        for path in available_font_files():
            try:
                font_manager.fontManager.addfont(path)
                name = font_manager.FontProperties(fname=path).get_name()
                if name and name not in families:
                    families.append(name)
            except Exception as exc:  # pragma: no cover - 字体文件异常
                logger.debug("注册字体失败 %s: %s", path, exc)

        for name in CJK_FAMILIES:
            if name not in families:
                families.append(name)

        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = families
        matplotlib.rcParams["axes.unicode_minus"] = False
        if dpi:
            matplotlib.rcParams["figure.dpi"] = int(dpi)

        info.update(
            available=True,
            font=resolve_cjk_font(),
            families=families,
            backend=str(matplotlib.get_backend()),
        )
    except Exception as exc:  # pragma: no cover - 配置失败不影响主流程
        logger.debug("matplotlib 配置失败: %s", exc)
    return info


__all__ = [
    "CJK_FONT_FILES",
    "CJK_FAMILIES",
    "available_font_files",
    "resolve_cjk_font",
    "configure",
]

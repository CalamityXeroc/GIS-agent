# -*- coding: utf-8 -*-
"""PDF 成果包（P0-4 步 c）：结论页 + 图页 + 方法页。

只用 matplotlib 的 ``PdfPages`` 拼页——它已经在 ``pyproject`` 的 extras 里
（内核配中文字体也要用），不再引新依赖。没装 matplotlib 时不报错，交给调用方
在成果包里写一句"未生成 PDF"，HTML 仍然照常交付。

中文字体复用 ``runtime.mpl_setup``：默认字体没有中文字形，不配置的话整份
PDF 的中文都是方框（第 14 届基准实测过）。
"""

from __future__ import annotations

import io
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

from .narrative import DISCLAIMER, ImageRef, ReportContent

logger = logging.getLogger(__name__)

#: A4 纵向（英寸），与 matplotlib figure 单位一致。
A4 = (8.27, 11.69)

#: 正文每行最多多少个**显示宽度**（CJK 算 2），10.5pt 下 92 大约等于版心宽度。
_LINE_WIDTH = 92

#: 折行时当成一个整体不拆开的片段：空白串 / 英文数字串 / 任意单字符（汉字自然成段）。
_TOKEN = re.compile(r"\s+|[A-Za-z0-9_.,%:/+\-]+|.", re.S)


def _import_pyplot() -> Any:
    """拿到配好中文字体的 pyplot；matplotlib 缺失时抛 ``ImportError``。"""
    from ..runtime.mpl_setup import configure

    configure()
    import matplotlib.pyplot as plt  # type: ignore

    return plt


# ------------------------------------------------------------------ 排版小工具
def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in str(text))


def wrap_text(text: str, limit: int = _LINE_WIDTH) -> list[str]:
    """按显示宽度折行，但**不在数字/英文词中间断**（中文一字一格可以随便断）。

    为什么不能逐字符断：实测把「不足 10%。」断成「不足 1」+「0%。」，
    读者会以为数字写错了。
    """
    lines: list[str] = []
    for raw_line in str(text or "").split("\n"):
        current = ""
        for token in _TOKEN.findall(raw_line):
            if token.isspace():
                if current:
                    current += " "
                continue
            if display_width(current + token) <= limit:
                current += token
                continue
            if current.strip():
                lines.append(current.rstrip())
            current = ""
            # 单词比整行还长（长 URL / 长路径）：只能硬断
            while display_width(token) > limit:
                piece = ""
                for ch in token:
                    if display_width(piece + ch) > limit:
                        break
                    piece += ch
                lines.append(piece)
                token = token[len(piece):]
            current = token
        lines.append(current.rstrip())
    return lines or [""]


class _Writer:
    """一页一页往下写（版心写满就自动翻页，标题带"（续）"）。"""

    def __init__(self, pdf: Any, plt: Any, *, title: str) -> None:
        self.pdf = pdf
        self.plt = plt
        self.title = title
        self.fig: Any = None
        self.y = 0.0
        self.heading = ""
        self._pages = 0

    # -------------------------------------------------------------- 生命周期
    def page(self, heading: str = "") -> None:
        self.close()
        self.fig = self.plt.figure(figsize=A4)
        self._pages += 1
        self.y = 0.92
        self.heading = heading
        if self._pages == 1:
            self._put(self.title, size=16, weight="bold", color="#155e9b")
            self.y -= 0.01
        if heading:
            self._put(heading, size=13, weight="bold", color="#155e9b")
            self.y -= 0.008

    def close(self) -> None:
        if self.fig is None:
            return
        self.fig.text(0.5, 0.045, f"第 {self._pages} 页", ha="center", fontsize=8, color="#7a8797")
        self.pdf.savefig(self.fig)
        self.plt.close(self.fig)
        self.fig = None

    # -------------------------------------------------------------- 文本
    def _put(self, text: str, *, size: float = 10.5, weight: str = "normal", color: str = "#1b2430") -> None:
        if self.fig is None:
            self.page(self.heading)
        self.fig.text(0.09, self.y, text, fontsize=size, weight=weight, color=color, va="top")
        self.y -= (size * 1.95) / 72.0 / A4[1]

    def _ensure(self, size: float = 10.5) -> None:
        if self.fig is None:
            self.page(self.heading)
        if self.y < 0.09 + (size * 1.95) / 72.0 / A4[1]:
            self.page((self.heading + "（续）") if self.heading else "")

    def paragraph(self, text: str, *, size: float = 11.0, indent: str = "", color: str = "#1b2430") -> None:
        for line in wrap_text(text):
            self._ensure(size)
            self._put(indent + line, size=size, color=color)

    def bullet(self, text: str, *, size: float = 11.0) -> None:
        lines = wrap_text(text)
        for index, line in enumerate(lines):
            self._ensure(size)
            self._put(("· " if index == 0 else "  ") + line, size=size)

    def section(self, heading: str) -> None:
        self._ensure(12)
        self.y -= 0.012
        self._put(heading, size=12, weight="bold", color="#155e9b")
        self.y -= 0.004

    def note(self, text: str, *, color: str = "#7a8797") -> None:
        for line in wrap_text(text):
            self._ensure(9)
            self._put(line, size=9, color=color)

    def image(self, data: bytes, caption: str) -> None:
        """整页放一张图（短边留白，等比缩放）。"""
        from PIL import Image

        self.close()
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception as exc:  # pragma: no cover - 坏图
            logger.debug("PDF 嵌入图片失败: %s", exc)
            return
        fig = self.plt.figure(figsize=A4)
        self._pages += 1
        width, height = img.size
        # 等比缩放：图的像素宽高比要等于它在纸上的英寸宽高比。
        aspect = (height / float(width)) if width else 0.6
        draw_w, draw_h = 0.88, 0.88 * aspect * (A4[0] / A4[1])
        if draw_h > 0.74:
            draw_h = 0.74
            draw_w = 0.74 / (aspect * (A4[0] / A4[1])) if aspect else 0.88
        # 竖直居中（宽扁的图贴顶会留下半页白）
        top = min(0.92, 0.5 + draw_h / 2 + 0.03)
        ax = fig.add_axes([(1 - draw_w) / 2, top - draw_h, draw_w, draw_h])
        ax.imshow(img)
        ax.set_axis_off()
        fig.text(0.5, max(0.05, top - draw_h - 0.06), caption, ha="center", fontsize=10, color="#44505f")
        self.pdf.savefig(fig)
        self.plt.close(fig)
        # 不在这里预先开新页：下一段文字自己会在 `_ensure` 里按需开页（带“（续）”），
        # 否则最后一张图之后会多出一张只剩页脚的白页。
        self.fig = None
        self.y = 0.0


# ------------------------------------------------------------------ 组装
def write_pdf(
    content: ReportContent,
    *,
    out_path: str | Path,
    images: Sequence[tuple[ImageRef, str, bytes]] = (),
) -> Path | None:
    """写 PDF；matplotlib 缺失或写失败时返回 ``None``（不抛异常）。"""
    try:
        plt = _import_pyplot()
    except Exception as exc:
        logger.info("未生成 PDF（matplotlib 不可用）: %s", exc)
        return None
    try:
        from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
    except Exception as exc:  # pragma: no cover - 残缺安装
        logger.info("未生成 PDF（PdfPages 不可用）: %s", exc)
        return None

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with PdfPages(str(path)) as pdf:
            writer = _Writer(pdf, plt, title=content.title)
            writer.page("结论")
            writer.paragraph(content.conclusion, size=12.0)
            if content.key_numbers:
                writer.section("关键数字")
                for item in content.key_numbers:
                    detail = f"（{item.note}）" if item.note else ""
                    writer.bullet(f"{item.label}：{item.value}{detail}")
            writer.note(f"生成时间 {content.generated_at}")

            for ref, _mime, data in images:
                writer.image(data, ref.caption)

            writer.page("方法")
            if content.methods:
                for line in content.methods:
                    writer.bullet(line)
            else:
                writer.paragraph("本次没有记录方法与配方。")

            writer.section("数据来源")
            if content.sources:
                for row in content.sources:
                    acquired = "" if row.acquired_at in ("", "未记录") else f"，取得时间 {row.acquired_at}"
                    writer.bullet(f"{row.name}（{row.path}）：{row.source}，许可 {row.license}{acquired}")
            else:
                writer.paragraph("数据目录里没有登记到本次用到的数据。")

            writer.section("验证")
            if content.verifications:
                for line in content.verifications:
                    writer.bullet(line)
            else:
                writer.paragraph("本次没有留下验收记录。")

            writer.y -= 0.02
            writer.paragraph(DISCLAIMER, size=10, color="#8a5a00")
            writer.close()
    except Exception as exc:  # pragma: no cover - 磁盘/字体异常
        logger.warning("PDF 生成失败: %s", exc)
        return None
    return path


__all__ = ["A4", "display_width", "wrap_text", "write_pdf"]

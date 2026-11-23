# -*- coding: utf-8 -*-
"""把 CLI 输出渲染成 PNG，便于"用眼睛验收"（含颜色）。

用法（必须用带 PIL 的解释器，例如 ArcGIS Pro 的 python）::

    python scripts/shot.py --cmd "python -m gis_cli.agent.cli chat ..." --out shot.png
    python scripts/shot.py --text-file out.txt --out shot.png

实现：解析 ANSI SGR 序列（38;5;N / 1 / 0 等）→ 用 PIL 逐字符绘制，中文用雅黑、
西文用 Consolas，背景按 256 色表近似。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CJK_FONT = r"C:\Windows\Fonts\msyh.ttc"
MONO_FONT = r"C:\Windows\Fonts\consola.ttf"

SGR = re.compile(r"\x1b\[([0-9;]*)m")


def _palette_256(index: int) -> tuple[int, int, int]:
    """xterm 256 色近似（够看清观感即可）。"""
    basic = [
        (0, 0, 0), (205, 49, 49), (13, 188, 121), (229, 229, 16),
        (36, 114, 200), (188, 63, 188), (17, 168, 205), (229, 229, 229),
        (102, 102, 102), (241, 76, 76), (35, 209, 139), (245, 245, 67),
        (59, 142, 234), (214, 112, 214), (41, 184, 219), (255, 255, 255),
    ]
    if index < 16:
        return basic[index]
    if index < 232:
        index -= 16
        r, g, b = index // 36, (index % 36) // 6, index % 6
        levels = [0, 95, 135, 175, 215, 255]
        return levels[r], levels[g], levels[b]
    grey = 8 + (index - 232) * 10
    return grey, grey, grey


DEFAULT_FG = (222, 226, 230)
DEFAULT_BG = (24, 26, 30)


def parse_ansi(text: str) -> list[list[tuple[str, tuple[int, int, int], bool]]]:
    """把带 SGR 的文本拆成``每行 [(字符, 颜色, 是否加粗)]``。"""
    lines: list[list[tuple[str, tuple[int, int, int], bool]]] = [[]]
    fg = DEFAULT_FG
    bold = False
    for chunk in re.split(r"(\x1b\[[0-9;]*m)", text):
        if not chunk:
            continue
        if SGR.fullmatch(chunk):
            codes = [int(x) for x in SGR.fullmatch(chunk).group(1).split(";") if x != ""]  # type: ignore[union-attr]
            if not codes or 0 in codes:
                fg, bold = DEFAULT_FG, False
            i = 0
            while i < len(codes):
                code = codes[i]
                if code == 1:
                    bold = True
                elif code == 38 and i + 2 < len(codes) and codes[i + 1] == 5:
                    fg = _palette_256(codes[i + 2])
                    i += 2
                elif code == 39:
                    fg = DEFAULT_FG
                i += 1
            continue
        for char in chunk:
            if char == "\n":
                lines.append([])
                continue
            if char == "\r":
                continue
            lines[-1].append((char, fg, bold))
    return lines


def render(text: str, out: Path, *, font_size: int = 17, pad: int = 16) -> Path:
    lines = parse_ansi(text)
    try:
        font = ImageFont.truetype(MONO_FONT, font_size)
        cjk = ImageFont.truetype(CJK_FONT, font_size)
    except Exception:
        font = cjk = ImageFont.load_default()

    char_w = font.getlength("M")
    line_h = int(font_size * 1.45)
    def _w(ch: str) -> float:
        return char_w if not (ord(ch) > 0x2E80 or 0x2000 <= ord(ch) <= 0x2BFF) else char_w * 2

    width = max((sum(_w(c) for c, _, _ in line) for line in lines), default=0)
    img_w = int(min(max(width + pad * 2, 640), 2400))
    img_h = int(line_h * max(len(lines), 1) + pad * 2)

    image = Image.new("RGB", (img_w, img_h), DEFAULT_BG)
    draw = ImageDraw.Draw(image)
    y = pad
    for line in lines:
        x = pad
        for char, color, bold in line:
            # 字形选择：CJK 与常见符号（◆✓●▸│└─╭╮╯╰）用雅黑，西文等宽用 Consolas
            wide = ord(char) > 0x2E80 or 0x2000 <= ord(char) <= 0x2BFF
            use = cjk if wide else font
            draw.text((x, y), char, font=use, fill=color, stroke_width=1 if bold else 0, stroke_fill=color)
            x += char_w if not wide else char_w * 2
        y += line_h
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", default="", help="要执行的命令（捕获其输出）")
    parser.add_argument("--text-file", default="", help="或直接渲染一个文本文件")
    parser.add_argument("--out", default="workspace/_shots/shot.png")
    parser.add_argument("--width", type=int, default=118)
    parser.add_argument("--font-size", type=int, default=17)
    args = parser.parse_args()

    if args.cmd:
        env = dict(**__import__("os").environ)
        env["COLUMNS"] = str(args.width)
        proc = subprocess.run(
            args.cmd,
            shell=True,
            capture_output=True,
            text=False,
            env=env,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        raw = (proc.stdout or b"") + (proc.stderr or b"")
        text = raw.decode("utf-8", errors="replace")
    else:
        text = Path(args.text_file).read_text(encoding="utf-8", errors="replace")

    path = render(text, Path(args.out), font_size=args.font_size)
    print(f"已生成 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""HTML → PDF：用系统自带的 Edge / Chrome 无头打印，零 Python 依赖。

为什么不引 reportlab/weasyprint：宿主 venv 里没有，而浏览器已经装在机器上，
且中文、内联图片、CSS 排版全都天然正确。实测 1.4MB 的单文件成果包约 3 秒出 14 页 PDF。

用法::

    python html_to_pdf.py --html output/成果包.html --pdf output/成果包.pdf
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

#: 常见浏览器安装位置（Windows 优先，其次类 Unix）
_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)
_NAMES = ("msedge", "microsoft-edge", "google-chrome", "chrome", "chromium", "chromium-browser")


def find_browser(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit if Path(explicit).exists() else None
    env = os.environ.get("GIS_PDF_BROWSER")
    if env and Path(env).exists():
        return env
    for candidate in _CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for name in _NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def _file_url(path: Path) -> str:
    # 中文路径必须走 file:/// 且做百分号编码，否则浏览器打不开
    return path.resolve().as_uri()


def html_to_pdf(html: Path, pdf: Path, *, browser: str | None = None, budget_ms: int = 15000) -> Path:
    exe = find_browser(browser)
    if exe is None:
        raise RuntimeError(
            "找不到 Edge/Chrome。设置 GIS_PDF_BROWSER=<浏览器 exe 路径>，"
            "或改用 Pro python + matplotlib 的兜底方案（见 SKILL.md）"
        )
    html = html.resolve()
    if not html.is_file():
        raise FileNotFoundError(f"HTML 不存在: {html}")
    pdf = pdf.resolve()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    if pdf.exists():
        pdf.unlink()  # 浏览器对已存在的目标会拒绝写

    # 临时 user-data-dir：并发跑多个转换时不抢同一个浏览器配置
    with tempfile.TemporaryDirectory(prefix="gis_pdf_") as profile:
        cmd = [
            exe,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--user-data-dir={profile}",
            f"--virtual-time-budget={budget_ms}",
            f"--print-to-pdf={pdf}",
            _file_url(html),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    # 浏览器进程退出 ≠ 文件已写完：实测 exit=0 时 PDF 可能还在刷盘（几秒）。
    # 不等就会误报「没有产出」，所以这里轮询到文件稳定（两次大小一致）为止。
    deadline = time.monotonic() + 20.0
    last_size = -1
    while time.monotonic() < deadline:
        if pdf.exists():
            size = pdf.stat().st_size
            if size > 0 and size == last_size:
                break
            last_size = size
        time.sleep(0.2)
    if not pdf.exists():
        detail = (proc.stderr or proc.stdout or "").strip()[:400]
        raise RuntimeError(f"浏览器没有产出 PDF（exit={proc.returncode}）: {detail}")

    # 校验：头 + 页数（浏览器偶尔会写 0 字节文件）
    raw = pdf.read_bytes()
    if not raw.startswith(b"%PDF-"):
        raise RuntimeError(f"产出的不是 PDF（头 {raw[:8]!r}）")
    pages = re.findall(rb"/Count\s+(\d+)", raw)
    print(f"[report-pdf] 浏览器: {exe}")
    print(f"[report-pdf] 输出: {pdf}  ({len(raw) / 1048576:.2f} MB, {pages[-1].decode() if pages else '?'} 页)")
    return pdf


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HTML → PDF（浏览器无头打印，零依赖）")
    ap.add_argument("--html", required=True, help="输入 HTML（通常是 output/成果包.html）")
    ap.add_argument("--pdf", help="输出 PDF（默认与 HTML 同名）")
    ap.add_argument("--browser", help="浏览器 exe 路径（默认自动发现）")
    ap.add_argument("--budget-ms", type=int, default=15000, help="渲染等待预算，内联图多时调大")
    args = ap.parse_args(argv)

    html = Path(args.html)
    pdf = Path(args.pdf) if args.pdf else html.with_suffix(".pdf")
    started = time.monotonic()
    try:
        html_to_pdf(html, pdf, browser=args.browser, budget_ms=args.budget_ms)
    except Exception as exc:
        print(f"[report-pdf] 失败: {exc}", file=sys.stderr)
        return 1
    print(f"[report-pdf] 用时 {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

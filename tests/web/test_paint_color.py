# -*- coding: utf-8 -*-
"""MapLibre 预览配色契约（DAIR-301）：oklch token 必须先转成 rgba 再进 paint。

背景：`theme.css` 的 `--chart-1/--chart-2` 是 `oklch(...)`，MapLibre 的 color
parser 不认。`addLayer` 校验失败**只 `fire(error)` 不抛**，错误边界接不到、
也不会回退 `MapCanvas`，用户看到的是只剩控件的空图。

这里的断言用两件真实之物，而不是重写一遍转换逻辑：

1. **真 MapLibre 校验器**（`@maplibre/maplibre-gl-style-spec`）：
   `oklch` 字面值必须被判 ``color expected``，转换结果必须通过；
2. **真 Chromium**：`toPaintColor` 的 canvas 2d 像素回读（不需要 WebGL）
   必须产出 `rgba(...)`，且色值与 oklch 对得上。

前置（node / tsc / Chrome）缺失时 skip，并在 skip 原因里写明缺了什么。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[2] / "web"
PAINT_COLOR_TS = WEB / "src" / "lib" / "paint-color.ts"
MAP_CANVAS_TSX = WEB / "src" / "components" / "gis" / "map-canvas.tsx"
STYLE_SPEC = WEB / "node_modules" / "@maplibre" / "maplibre-gl-style-spec" / "dist" / "index.mjs"
TS_COMPILER = WEB / "node_modules" / "typescript" / "bin" / "tsc"

#: theme.css 里浅色主题的 `--chart-1`（oklch），以及它等价的可解析色值。
CHART_1_OKLCH = "oklch(0.72 0.18 250)"
CHART_1_RGBA = "rgba(53, 169, 255, 1.000)"

WINDOWS_CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)


def _node() -> str | None:
    return shutil.which("node")


def _chrome() -> str | None:
    configured = os.environ.get("GIS_CHROME")
    if configured and Path(configured).is_file():
        return configured
    for candidate in WINDOWS_CHROME_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return shutil.which("chrome") or shutil.which("chromium") or shutil.which("chromium-browser")


def _run(cmd: list[str], *, cwd: Path, timeout: int) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    assert proc.returncode == 0, f"{cmd} 失败（rc={proc.returncode}）：{proc.stdout}{proc.stderr}"
    return proc.stdout


# ---------------------------------------------------------------- 真校验器
def test_maplibre_rejects_oklch_paint_and_accepts_converted(tmp_path: Path) -> None:
    """复现门禁症状：oklch 进 paint 被判非法，转换后的 rgba 通过。"""
    node = _node()
    if not node:
        pytest.skip("没有 node：跳过 MapLibre 校验器用例")
    if not STYLE_SPEC.is_file():
        pytest.skip("缺少 @maplibre/maplibre-gl-style-spec：先 cd web && npm ci")

    script = tmp_path / "validate.mjs"
    script.write_text(
        f"""
const spec = await import('file:///{STYLE_SPEC.as_posix()}');

const styleWith = (color) => ({{
  version: 8,
  sources: {{ preview: {{ type: 'geojson', data: {{ type: 'FeatureCollection', features: [] }} }} }},
  layers: [
    {{ id: 'preview-fill', type: 'fill', source: 'preview',
       paint: {{ 'fill-color': color, 'fill-opacity': 0.3 }} }},
    {{ id: 'preview-line', type: 'line', source: 'preview',
       paint: {{ 'line-color': color, 'line-width': 1.4 }} }},
    {{ id: 'preview-point', type: 'circle', source: 'preview',
       paint: {{ 'circle-color': color, 'circle-radius': 3.5 }} }},
  ],
}});

const result = {{}};
for (const color of [{json.dumps(CHART_1_OKLCH)}, {json.dumps(CHART_1_RGBA)}]) {{
  result[color] = spec.validateStyleMin(styleWith(color)).map((e) => e.message);
}}
console.log(JSON.stringify(result));
""",
        encoding="utf-8",
    )
    raw = _run([node, str(script)], cwd=tmp_path, timeout=180)
    errors = json.loads(raw.strip().splitlines()[-1])

    assert errors[CHART_1_OKLCH], "MapLibre 竟然接受了 oklch —— 这条用例的假设不成立"
    assert any("color expected" in msg for msg in errors[CHART_1_OKLCH])
    assert errors[CHART_1_RGBA] == [], f"转换结果仍被 MapLibre 拒绝：{errors[CHART_1_RGBA]}"


# ---------------------------------------------------------------- 真浏览器
def test_to_paint_color_outputs_rgba_in_chromium(tmp_path: Path) -> None:
    """canvas 2d 像素回读：oklch → `rgba(...)`，非法值回落 hex。只需无 WebGL 的 2d。"""
    node = _node()
    if not node:
        pytest.skip("没有 node：跳过 toPaintColor 用例")
    if not TS_COMPILER.is_file():
        pytest.skip("缺少 typescript：先 cd web && npm ci")
    chrome = _chrome()
    if not chrome:
        pytest.skip("没有 Chrome/Chromium：跳过真实像素回读用例")

    # 编成 ESM 再内联进页面（内联 module script 不受 file:// 的跨源限制）。
    _run(
        [
            node,
            str(TS_COMPILER),
            PAINT_COLOR_TS.relative_to(WEB).as_posix(),
            "--ignoreConfig",
            "--outDir",
            str(tmp_path),
            "--target",
            "es2022",
            "--module",
            "esnext",
            "--skipLibCheck",
        ],
        cwd=WEB,
        timeout=240,
    )
    compiled = (tmp_path / "paint-color.js").read_text(encoding="utf-8")
    page = tmp_path / "probe.html"
    page.write_text(
        "<!doctype html><meta charset=utf-8><body><pre id=out>pending</pre>\n"
        "<script type=module>\n"
        + compiled.replace("export function", "function")
        + f"""
const values = {{
  oklch: toPaintColor({json.dumps(CHART_1_OKLCH)}, '#2563eb'),
  oklchAlt: toPaintColor({json.dumps("oklch(0.65 0.15 200)")}, '#0ea5e9'),
  hex: toPaintColor('#2563eb', '#2563eb'),
  bogus: toPaintColor('not-a-color', '#2563eb'),
  empty: toPaintColor('   ', '#0ea5e9'),
}};
document.getElementById('out').textContent = JSON.stringify(values);
</script>""",
        encoding="utf-8",
    )

    dom = _run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",  # 不需要 WebGL：只跑 canvas 2d
            "--no-sandbox",
            "--no-first-run",
            "--virtual-time-budget=8000",
            "--dump-dom",
            page.as_uri(),
        ],
        cwd=tmp_path,
        timeout=240,
    )
    match = re.search(r'<pre id="out">([^<]*)</pre>', dom)
    assert match and match.group(1) != "pending", f"页面脚本没有跑出结果：\n{dom[:2000]}"
    values = json.loads(match.group(1).replace("&quot;", '"'))

    for key in ("oklch", "oklchAlt", "hex"):
        assert re.fullmatch(r"rgba\(\d+, \d+, \d+, \d\.\d{3}\)", values[key]), values[key]
    assert values["oklch"] == CHART_1_RGBA
    # 两条一组的主题色不能撞成一个值（否则 --chart-2 根本没读到）
    assert values["oklch"] != values["oklchAlt"]
    # 转换失败/变量缺失时回落调用方给的 hex —— MapLibre 认十六进制。
    assert values["bogus"] == "#2563eb"
    assert values["empty"] == "#0ea5e9"


# ---------------------------------------------------------------- 接线
def test_map_canvas_paints_only_converted_colors() -> None:
    """CSS 变量里的色值必须过 toPaintColor 才能进 paint。

    DAIR-334：旧 `components/layers/map-view.tsx` 已随死代码清理删除，同样的
    契约挂在现役的 `components/gis/map-canvas.tsx` 上：色源只有 `chartPalette()`
    一处读 CSS 变量，且每个 token 都过转换；主题变化时同一个函数会再跑一遍。
    """
    text = MAP_CANVAS_TSX.read_text(encoding="utf-8")
    assert "toPaintColor" in text
    # 唯一读 CSS 变量的地方是 chartPalette()，逐 token 转换，不漏一支色。
    assert re.search(r"toPaintColor\(\s*styles\.getPropertyValue\(token\)\.trim\(\),", text)
    # 每次同步都现读一次色（不是模块级常量），主题变化后本次 effect 重跑即取到新色。
    assert re.search(r"const colors = chartPalette\(\)", text)

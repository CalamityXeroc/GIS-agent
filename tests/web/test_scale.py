# -*- coding: utf-8 -*-
"""状态栏比例尺契约（DAIR-335）：MapLibre 的 zoom 是 512px 瓦片口径。

回归的症状：状态栏写着 `1:176,565`，同一屏幕上 MapLibre 自带 `ScaleControl`
画的是 2 km —— 恒差 2.00×。根因是用了 256px 瓦片口径的 156543.03392。

这里不重写一遍换算逻辑，而是拿两件真东西对表：

1. **装好的 maplibre-gl 源码**：`src/geo/transform_helper.ts` 里
   `this._tileSize = 512` —— 这是「zoom 口径」的来源。配合外部事实
   （WGS84 赤道周长 40075016.686 m），每像素米数只能是
   `周长 / 512 * cos(lat) / 2^zoom`。测的是 512 这个口径，不是抄
   MapLibre 某个版本的实现细节（5.24 的 `ScaleControl` 已经改成取 100px
   两端 unproject 再算球面距离，不再有 78271.484 这个常数）。
2. **DAIR-333 的实测数字**：z=11.6 / φ=22.6° 时状态栏报 176,565（256 口径），
   正确值约一半。

`scale.ts` 用仓库里的 tsc 编成 ESM 后在 node 里跑，不是把公式抄一遍。
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[2] / "web"
SCALE_TS = WEB / "src" / "lib" / "scale.ts"
MAP_CANVAS_TSX = WEB / "src" / "components" / "gis" / "map-canvas.tsx"
TS_COMPILER = WEB / "node_modules" / "typescript" / "bin" / "tsc"
MAPLIBRE_TRANSFORM = WEB / "node_modules" / "maplibre-gl" / "src" / "geo" / "transform_helper.ts"

#: WGS84 赤道周长（米）—— 外部事实，不从被测代码里读。
EARTH_CIRCUMFERENCE_M = 40075016.686
#: MapLibre 的 zoom 口径：z=0 时整个世界宽这么多 CSS 像素。
EXPECTED_TILE_PX = 512

#: DAIR-333 在真实数据上量到的场景：z=11.6、纬度 22.6°，256 口径给 1:176,565。
REPORTED_ZOOM = 11.6
REPORTED_LAT = 22.6
REPORTED_WRONG_SCALE = 176565


def _node() -> str | None:
    return shutil.which("node")


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


def _maplibre_tile_size() -> int:
    """从装好的 maplibre-gl 源码里读 zoom 口径（不是从我们的代码里读）。"""
    if not MAPLIBRE_TRANSFORM.is_file():
        pytest.skip("缺少 maplibre-gl 源码：先 cd web && npm ci")
    match = re.search(
        r"_tileSize\s*=\s*(\d+)", MAPLIBRE_TRANSFORM.read_text(encoding="utf-8")
    )
    assert match, "maplibre-gl 源码里找不到 _tileSize 赋值：口径变了，先核对再改这条用例"
    return int(match.group(1))


def test_scale_denominator_uses_512px_tile_convention(tmp_path: Path) -> None:
    tile = _maplibre_tile_size()
    assert tile == EXPECTED_TILE_PX, f"maplibre-gl 的瓦片口径变成 {tile}px 了"

    node = _node()
    if not node:
        pytest.skip("没有 node：跳过 scale.ts 用例")
    if not TS_COMPILER.is_file():
        pytest.skip("缺少 typescript：先 cd web && npm ci")

    _run(
        [
            node,
            str(TS_COMPILER),
            "src/lib/scale.ts",
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

    harness = tmp_path / "probe.mjs"
    harness.write_text(
        "import { metersPerPixel, scaleDenominator, TILE_SIZE } from './scale.js';\n"
        "const cases = [[0,0],[4,35],[8,0],[11.6,22.6],[14,45],[17,-33.9]];\n"
        "console.log(JSON.stringify({\n"
        "  tile: TILE_SIZE,\n"
        "  rows: cases.map(([z, lat]) => ({z, lat,\n"
        "    mpp: metersPerPixel(z, lat),\n"
        "    scale: scaleDenominator(z, lat)})),\n"
        "}));\n",
        encoding="utf-8",
    )
    payload = json.loads(_run([node, str(harness)], cwd=tmp_path, timeout=120).strip().splitlines()[-1])

    assert payload["tile"] == tile, "scale.ts 的瓦片口径必须与 MapLibre 一致"

    for row in payload["rows"]:
        expected_mpp = (
            EARTH_CIRCUMFERENCE_M / tile
            * math.cos(row["lat"] * math.pi / 180)
            / (2 ** row["z"])
        )
        assert row["mpp"] == pytest.approx(expected_mpp, rel=1e-6), row
        # 96dpi 的 1:N 换算：m/px * 96 / 0.0254
        assert row["scale"] == round(row["mpp"] * 96 / 0.0254), row

    # z=0 赤道：512 口径 1:295,829,355；256 口径会翻倍成 1:591,658,710。
    assert payload["rows"][0]["scale"] == 295_829_355, payload["rows"][0]

    # 真实场景对表：报告里的 1:176,565 是 256 口径算出来的，正确值约一半。
    at_report = next(row for row in payload["rows"] if row["z"] == REPORTED_ZOOM)
    legacy = round(
        156543.03392
        * math.cos(REPORTED_LAT * math.pi / 180)
        / (2 ** REPORTED_ZOOM)
        * 96
        / 0.0254
    )
    assert abs(legacy - REPORTED_WRONG_SCALE) / REPORTED_WRONG_SCALE < 0.005, f"基准数字对不上：{legacy}"
    assert at_report["scale"] * 2 == pytest.approx(legacy, rel=2e-4)


def test_map_canvas_does_not_reinvent_the_256px_constant() -> None:
    """地图组件必须调用 lib/scale.ts，不能自己再写一个 256 口径的常数。"""
    source = MAP_CANVAS_TSX.read_text(encoding="utf-8")
    assert "156543.03392" not in source, "256px 瓦片口径的常数又回到了 map-canvas"
    assert "from '@/lib/scale'" in source, "map-canvas 应使用 lib/scale.ts 的 scaleDenominator"

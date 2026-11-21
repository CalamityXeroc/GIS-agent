# -*- coding: utf-8 -*-
"""天地图 / OSM 底图（P0-2，出图用）。

做法：把 XYZ 瓦片拼成一张**带地理参考的静态底图**（PNG + ``.pgw`` 世界文件 +
``.json`` 元数据）。这样它既能被 matplotlib 直接 ``imshow``，也能被 ArcGIS Pro
当栅格图层加载，不依赖任何在线底图插件，出图后离线可复现。

天地图需要 key（免费申请）：设环境变量 ``TIANDITU_KEY``，或写
``config/basemap.json`` 的 ``{"tianditu_key": "..."}``。没 key 时用 OSM。

许可如实登记：天地图按《天地图服务条款》，OSM 是 ODbL（且瓦片服务只适合轻量
使用，生产环境请自建或换商业服务）。
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from .csv_points import IngestError

try:  # Pillow 是核心依赖；真缺了也只影响底图这一项能力
    from PIL import Image
except Exception:  # pragma: no cover - 环境缺 Pillow 时的降级
    Image = None  # type: ignore[assignment]

#: 天地图图层：矢量底图 / 矢量注记 / 影像底图 / 影像注记 / 地形晕渲
TIANDITU_LAYERS = {
    "vec": "矢量底图",
    "cva": "矢量注记",
    "img": "影像底图",
    "cia": "影像注记",
    "ter": "地形晕渲",
}

TIANDITU_TILE = (
    "https://t{s}.tianditu.gov.cn/{layer}_w/wmts?SERVICE=WMTS&REQUEST=GetTile"
    "&VERSION=1.0.0&LAYER={layer}&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles"
    "&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk={key}"
)
OSM_TILE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

#: 单次拼图最多取多少张瓦片（防止手滑把 zoom 写成 18 拉几千张）。
MAX_TILES = 64
#: 底图输出最长边（像素）。
MAX_SIZE = 2048
#: 瓦片边长（天地图/OSM 都是 256）。
TILE_SIZE = 256
#: Web Mercator 的纬度上限。
MAX_LATITUDE = 85.0511287798066

_USER_AGENT = "gis-agent/0.8 (basemap render)"


class BasemapKeyError(IngestError):
    """天地图没配 key。"""


@dataclass(frozen=True)
class BasemapSpec:
    """一种底图源。"""

    provider: str
    label: str
    url_template: str
    attribution: str
    license: str
    max_zoom: int = 18
    min_zoom: int = 0
    subdomains: tuple[str, ...] = ("",)
    requires_key: bool = False
    layer: str = ""
    key: str = field(default="", repr=False)

    def tile_url(self, x: int, y: int, z: int) -> str:
        sub = self.subdomains[(x + y) % len(self.subdomains)] if self.subdomains else ""
        return self.url_template.format(z=z, x=x, y=y, s=sub, layer=self.layer, key=self.key)


@dataclass
class BasemapResult:
    """拼好的底图。"""

    provider: str
    png_path: str
    world_file: str
    meta_path: str
    extent: tuple[float, float, float, float]
    zoom: int
    tiles_fetched: int
    tiles_failed: int
    attribution: str
    license: str
    acquired_at: str
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        west, south, east, north = self.extent
        text = (
            f"{self.provider} 底图（zoom={self.zoom}）：{self.png_path}，"
            f"范围 [{west:.4f},{south:.4f},{east:.4f},{north:.4f}]，"
            f"{self.tiles_fetched} 张瓦片；许可={self.license}；{self.attribution}"
        )
        for warning in self.warnings:
            text += f"\n注意：{warning}"
        return text


# ------------------------------------------------------------------- 配置


def load_config(path: str | Path | None = None, *, workspace: str | Path | None = None) -> dict[str, Any]:
    """读底图配置：显式 path > 工作区 config/basemap.json > 当前目录 config/basemap.json。"""
    candidates = []
    if path:
        candidates.append(Path(path))
    else:
        if workspace:
            candidates.append(Path(workspace) / "config" / "basemap.json")
        candidates.append(Path("config") / "basemap.json")
    for candidate in candidates:
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError) as exc:
                raise IngestError(f"底图配置读不了：{candidate}：{exc}") from exc
            return payload if isinstance(payload, dict) else {}
    return {}


def load_tianditu_key(
    path: str | Path | None = None,
    *,
    workspace: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """天地图 key：环境变量 ``TIANDITU_KEY`` 优先，其次配置文件。"""
    import os

    environ = env if env is not None else os.environ
    for name in ("TIANDITU_KEY", "TIANDITU_TK"):
        value = (environ.get(name) or "").strip()
        if value:
            return value
    config = load_config(path, workspace=workspace)
    return str(config.get("tianditu_key") or config.get("TIANDITU_KEY") or "").strip()


def tianditu_spec(layer: str = "vec", *, key: str | None = None, **config_kw: Any) -> BasemapSpec:
    """天地图底图（必须给 key）。"""
    layer = (layer or "vec").strip().lower()
    if layer not in TIANDITU_LAYERS:
        raise IngestError(f"天地图图层只认 {'/'.join(TIANDITU_LAYERS)}，收到：{layer}")
    token = (key or "").strip() or load_tianditu_key(**config_kw)
    if not token:
        raise BasemapKeyError(
            "天地图底图需要 key：到 https://console.tianditu.gov.cn 申请后，"
            "设置环境变量 TIANDITU_KEY，或在 config/basemap.json 写 "
            '{"tianditu_key": "..."}；不想申请就把 provider 换成 osm。'
        )
    return BasemapSpec(
        provider="tianditu",
        label=f"天地图·{TIANDITU_LAYERS[layer]}",
        url_template=TIANDITU_TILE,
        attribution="© 天地图",
        license="天地图条款",
        max_zoom=18,
        subdomains=tuple(str(i) for i in range(8)),
        requires_key=True,
        layer=layer,
        key=token,
    )


def osm_spec() -> BasemapSpec:
    """OpenStreetMap 底图（不需要 key，但只适合轻量使用）。"""
    return BasemapSpec(
        provider="osm",
        label="OpenStreetMap",
        url_template=OSM_TILE,
        attribution="© OpenStreetMap contributors",
        license="ODbL",
        max_zoom=19,
        subdomains=("",),
    )


def spec_for(provider: str = "tianditu", *, layer: str = "vec", key: str | None = None, **config_kw: Any) -> BasemapSpec:
    provider = (provider or "").strip().lower()
    if provider in {"tianditu", "tdt"}:
        return tianditu_spec(layer, key=key, **config_kw)
    if provider in {"osm", "openstreetmap"}:
        return osm_spec()
    raise IngestError(f"底图只认 tianditu / osm，收到：{provider or '(空)'}")


# ----------------------------------------------------------------- 瓦片数学


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """经纬度 → 瓦片行列号（Web Mercator / XYZ）。"""
    n = 2 ** int(zoom)
    lat = max(min(float(lat), MAX_LATITUDE), -MAX_LATITUDE)
    x = int((float(lon) + 180.0) / 360.0 * n)
    rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(rad) + 1.0 / math.cos(rad)) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def tile_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    """瓦片 → (west, south, east, north)。"""
    n = 2 ** int(zoom)

    def _lon(px: float) -> float:
        return px / n * 360.0 - 180.0

    def _lat(py: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * py / n))))

    return _lon(x), _lat(y + 1), _lon(x + 1), _lat(y)


def _http_get(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (固定 https 源)
        return response.read()


def _check_extent(extent: tuple[float, float, float, float] | list[float]) -> tuple[float, float, float, float]:
    if len(extent) != 4:
        raise IngestError("范围要写成 [西, 南, 东, 北]（经度/纬度）。")
    west, south, east, north = (float(v) for v in extent)
    if not (west < east and south < north):
        raise IngestError(f"范围不合法：西 {west} 应小于东 {east}，南 {south} 应小于北 {north}。")
    if not (-180 <= west <= 180 and -180 <= east <= 180 and -90 <= south <= 90 and -90 <= north <= 90):
        raise IngestError(
            f"范围超出经纬度：{extent}。底图只接受 WGS84 经纬度范围"
            f"（经度 -180~180、纬度 -90~90），投影坐标请先转换。"
        )
    return west, south, east, north


def render_basemap(
    extent: tuple[float, float, float, float] | list[float],
    zoom: int,
    spec: BasemapSpec,
    out_png: str | Path,
    *,
    cache_dir: str | Path | None = None,
    fetch: Callable[[str, float], bytes] | None = None,
    timeout: float = 30.0,
    max_tiles: int = MAX_TILES,
    max_size: int = MAX_SIZE,
) -> BasemapResult:
    """把范围内瓦片拼成一张带地理参考的底图。"""
    if Image is None:  # pragma: no cover
        raise IngestError("底图拼图需要 Pillow（pip install pillow）。")
    west, south, east, north = _check_extent(extent)
    zoom = int(zoom)
    if not spec.min_zoom <= zoom <= spec.max_zoom:
        raise IngestError(f"zoom 要在 {spec.min_zoom}~{spec.max_zoom} 之间，收到：{zoom}")

    x0, y0 = lonlat_to_tile(west, north, zoom)
    x1, y1 = lonlat_to_tile(east, south, zoom)
    cols, rows = x1 - x0 + 1, y1 - y0 + 1
    if cols * rows > max_tiles:
        raise IngestError(
            f"范围太大：zoom={zoom} 需要 {cols * rows} 张瓦片（上限 {max_tiles}）。"
            f"把 zoom 调小或缩小范围。"
        )

    getter = fetch or _http_get
    directory = Path(cache_dir) if cache_dir else None
    mosaic = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE), (255, 255, 255))
    fetched = failed = 0
    warnings: list[str] = []
    first_error = ""
    for row in range(rows):
        for col in range(cols):
            x, y = x0 + col, y0 + row
            url = spec.tile_url(x, y, zoom)
            raw: bytes | None = None
            cached = directory / spec.provider / str(zoom) / f"{x}_{y}.png" if directory else None
            from_cache = cached is not None and cached.is_file()
            if from_cache:
                raw = cached.read_bytes()
            else:
                try:
                    raw = getter(url, timeout)
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    failed += 1
                    first_error = first_error or f"{url}（{exc}）"
                    continue
            try:
                tile = Image.open(BytesIO(raw)).convert("RGB")
            except Exception as exc:  # 返回的不是图片（常见：key 无效时返回 XML 错误页）
                failed += 1
                first_error = first_error or f"{url}（不是图片：{exc}）"
                continue
            # 只缓存真的能打开的图片：key 无效时服务端返回 XML，
            # 若先存盘再校验，这份 XML 会永久堵在缓存里（实测坑）。
            if cached is not None and not from_cache:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(raw)
            if tile.size != (TILE_SIZE, TILE_SIZE):
                tile = tile.resize((TILE_SIZE, TILE_SIZE))
            mosaic.paste(tile, (col * TILE_SIZE, row * TILE_SIZE))
            fetched += 1

    if fetched == 0:
        hint = (
            "检查网络、天地图 key 是否有效（key 无效时服务端返回 XML 而不是图片）"
            if spec.provider == "tianditu"
            else "OSM 瓦片服务在境内经常不可达（实测超时），建议换天地图或自建/商业瓦片源"
        )
        raise IngestError(f"一张瓦片都没取到：{first_error or '未知原因'}。{hint}。")
    if failed:
        warnings.append(f"{failed} 张瓦片没取到（已留白）：{first_error}")

    m_west, m_south, _, _ = tile_bounds(x0, y1, zoom)
    _, _, m_east, m_north = tile_bounds(x1, y0, zoom)
    px_per_lon = mosaic.width / (m_east - m_west)
    px_per_lat = mosaic.height / (m_north - m_south)
    left = max(int(round((west - m_west) * px_per_lon)), 0)
    right = min(int(round((east - m_west) * px_per_lon)), mosaic.width)
    top = max(int(round((m_north - north) * px_per_lat)), 0)
    bottom = min(int(round((m_north - south) * px_per_lat)), mosaic.height)
    if right - left < 2 or bottom - top < 2:
        raise IngestError("范围太小：裁剪后不到 2 像素，把 zoom 调大或扩大范围。")
    image = mosaic.crop((left, top, right, bottom))
    if max(image.size) > max_size:
        scale = max_size / max(image.size)
        image = image.resize((max(int(image.width * scale), 1), max(int(image.height * scale), 1)))

    target = Path(out_png)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG")

    # 世界文件：左上角像素中心 + 像素大小（ArcGIS/ArcMap 都认）
    px_x = (east - west) / image.width
    px_y = (north - south) / image.height
    world = target.with_suffix(".pgw")
    world.write_text(
        "\n".join(
            [
                f"{px_x:.12f}",
                "0.000000000000",
                "0.000000000000",
                f"{-px_y:.12f}",
                f"{west + px_x / 2:.12f}",
                f"{north - px_y / 2:.12f}",
                "",
            ]
        ),
        encoding="ascii",
    )

    acquired_at = datetime.now(timezone.utc).isoformat()
    meta = {
        "provider": spec.provider,
        "label": spec.label,
        "extent": [west, south, east, north],
        "zoom": zoom,
        "pixel_size": [px_x, px_y],
        "size": [image.width, image.height],
        "attribution": spec.attribution,
        "license": spec.license,
        "acquired_at": acquired_at,
        "tiles_fetched": fetched,
        "tiles_failed": failed,
        "world_file": world.name,
    }
    meta_path = target.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return BasemapResult(
        provider=spec.provider,
        png_path=str(target),
        world_file=str(world),
        meta_path=str(meta_path),
        extent=(west, south, east, north),
        zoom=zoom,
        tiles_fetched=fetched,
        tiles_failed=failed,
        attribution=spec.attribution,
        license=spec.license,
        acquired_at=acquired_at,
        warnings=warnings,
    )

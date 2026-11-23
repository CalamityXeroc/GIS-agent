# -*- coding: utf-8 -*-
"""内置行政边界（省/市/县）：按需下载 + 缓存 + 许可登记（P0-2）。

**不打包进安装包**：全国边界几十 MB，装一次只为偶尔用一次不值当。首次用到时
下载，落到缓存目录，之后再读缓存；每个缓存文件旁边写一份 ``.meta.json``
记录来源 URL、许可、采集时间 —— catalog 的 ``source/license/acquired_at``
就是从这里来的。

坐标系**不猜**：这几个数据源都没有声明坐标系，业界普遍认为是 GCJ-02
（与 WGS84 有几十到几百米偏移）。这里如实写"未知"并给出提示，需要精确套合
时必须由用户确认后再转换。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .csv_points import IngestError
from .formats import describe_geojson

#: 行政级别 → 中文名。
LEVELS = {"province": "省", "city": "市", "county": "县"}

#: 数据源候选表：**按顺序试，谁先给出合法 FeatureCollection 就用谁**。
#:
#: 语义统一是"父级 adcode → 它的下一级"：
#: ``100000`` → 34 个省级；``440000`` → 广东各地市；``440100`` → 广州各区。
#:
#: 为什么要多个：阿里云 DataV 的老接口（``areas_v3/boundary/{adcode}_full.json``）
#: 实测已返回 403/404（2026-10 复核），不能只押一家。
BOUNDARY_URL_TEMPLATES: dict[str, tuple[str, ...]] = {
    "province": (
        "https://geojson.cn/api/china/1.6.2/{adcode}.json",
        "https://geo.datav.aliyun.com/areas_v3/boundary/{adcode}_full.json",
    ),
    "city": (
        "https://geojson.cn/api/china/1.6.2/{adcode}.json",
        "https://geo.datav.aliyun.com/areas_v3/boundary/{adcode}_full.json",
    ),
    "county": (
        "https://geo.datav.aliyun.com/areas_v3/boundary/{adcode}_full.json",
        "https://geojson.cn/api/china/1.6.2/{adcode}.json",
        "https://raw.githubusercontent.com/longwosion/geojson-map-china/master/geometryCouties/{adcode}.json",
    ),
}

#: 全国（省级）区划代码。
NATIONAL_ADCODE = "100000"

#: 数据源未声明许可 —— 默认 unknown，绝不替用户认领。
DEFAULT_LICENSE = "unknown"

#: 实际用的源写进 meta（多源候选时不能写死一家）。
ATTRIBUTIONS = (
    ("geojson.cn", "geojson.cn"),
    ("datav.aliyun.com", "阿里云 DataV.GeoAtlas"),
    ("raw.githubusercontent.com", "longwosion/geojson-map-china"),
)
ATTRIBUTION = "geojson.cn / 阿里云 DataV.GeoAtlas"

#: 数据源坐标系未声明（常见说法是 GCJ-02）。
CRS_NOTE = "未知（数据源未声明坐标系，常见说法为 GCJ-02，与 WGS84 有偏移）"

_FETCH_TIMEOUT = 60.0
_USER_AGENT = "gis-agent/0.8 (boundary cache)"


@dataclass(frozen=True)
class BoundarySource:
    """一次边界数据请求的描述（``urls`` 是候选列表，按序尝试）。"""

    level: str
    adcode: str
    urls: tuple[str, ...]
    attribution: str = ATTRIBUTION
    license: str = DEFAULT_LICENSE
    crs_note: str = CRS_NOTE

    @property
    def url(self) -> str:
        return self.urls[0] if self.urls else ""


@dataclass
class BoundaryResult:
    """下载/命中缓存的结果。"""

    level: str
    adcode: str
    path: str
    from_cache: bool
    feature_count: int
    license: str
    acquired_at: str
    source_url: str
    attribution: str
    crs_note: str = CRS_NOTE
    extent: list[float] | None = None

    def summary(self) -> str:
        where = "缓存" if self.from_cache else "刚下载"
        text = (
            f"{LEVELS.get(self.level, self.level)}级行政边界（adcode={self.adcode}）{where}："
            f"{self.path}，{self.feature_count} 个面；许可={self.license}；来源={self.attribution}"
        )
        if self.crs_note:
            text += f"；坐标系：{self.crs_note}"
        return text

    def to_record(self):
        """转 catalog 记录：source=boundary，许可与采集时间照 meta 登记。"""
        from ..datacatalog.store import DatasetRecord

        path = Path(self.path)
        stat = path.stat() if path.exists() else None
        try:
            described = describe_geojson(path)
        except IngestError:
            described = {}
        return DatasetRecord(
            path=str(path),
            kind="vector",
            container="",
            name=f"{LEVELS.get(self.level, self.level)}级行政边界_{self.adcode}",
            mtime=stat.st_mtime if stat else 0.0,
            size=stat.st_size if stat else 0,
            # 坐标系统一留空：数据源没声明，不能拿 GeoJSON 的 RFC 7946 默认值
            # 冒充“这就是 WGS84” —— 对不上图时用户得知道该怀疑哪一步。
            crs_name="",
            wkid=None,
            geom_type=str(described.get("geom_type") or ""),
            feature_count=self.feature_count,
            extent=described.get("extent") or self.extent,
            fields=list(described.get("fields") or []),
            raster={},
            extra={
                "format": "geojson",
                "boundary_level": self.level,
                "adcode": self.adcode,
                "source_url": self.source_url,
                "attribution": self.attribution,
                "crs_note": self.crs_note,
            },
            source="boundary",
            license=self.license,
            acquired_at=self.acquired_at,
        )


def default_cache_dir(workspace: str | Path | None = None) -> Path:
    """缓存目录：工作区里的 ``.gis_cache/boundaries``，没给工作区就用用户目录。"""
    if workspace:
        return Path(workspace) / ".gis_cache" / "boundaries"
    return Path.home() / ".gis_cli" / "cache" / "boundaries"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attribution_for(url: str) -> str:
    for token, label in ATTRIBUTIONS:
        if token in url:
            return label
    return url


def _http_get(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (固定 https 源)
        return response.read()


def build_source(level: str, adcode: str | None = None, *, license: str = DEFAULT_LICENSE) -> BoundarySource:
    """拼出请求描述；``city``/``county`` 必须给上级区划代码。"""
    level = (level or "").strip().lower()
    if level not in LEVELS:
        raise IngestError(f"行政级别只认 {'/'.join(LEVELS)}，收到：{level or '(空)'}")
    adcode = (adcode or "").strip()
    if not adcode:
        if level == "province":
            adcode = NATIONAL_ADCODE
        else:
            raise IngestError(
                f"{LEVELS[level]}级边界需要上级区划代码（adcode），例如 "
                f"adcode='440000'（广东省）拿到的就是广东各地市；"
                f"不带 adcode 无法确定范围，本工具不猜。"
            )
    if not adcode.isdigit():
        raise IngestError(f"adcode 必须是数字区划代码（如 100000 / 440000），收到：{adcode}")
    templates = BOUNDARY_URL_TEMPLATES.get(level) or ()
    return BoundarySource(
        level=level,
        adcode=adcode,
        urls=tuple(template.format(adcode=adcode) for template in templates),
        license=license,
    )


def ensure_boundary(
    level: str = "province",
    *,
    adcode: str | None = None,
    cache_dir: str | Path | None = None,
    fetch: Callable[[str, float], bytes] | None = None,
    timeout: float = _FETCH_TIMEOUT,
    force: bool = False,
    license: str = DEFAULT_LICENSE,
) -> BoundaryResult:
    """拿到指定级别的行政边界（有缓存读缓存，没有才下载）。

    ``fetch(url, timeout) -> bytes`` 可注入，测试不联网。
    """
    source = build_source(level, adcode, license=license)
    directory = Path(cache_dir) if cache_dir else default_cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{source.level}_{source.adcode}.geojson"
    meta_path = target.with_suffix(".meta.json")

    if target.is_file() and not force:
        meta = _read_meta(meta_path)
        return BoundaryResult(
            level=source.level,
            adcode=source.adcode,
            path=str(target),
            from_cache=True,
            feature_count=int(meta.get("feature_count") or 0),
            license=str(meta.get("license") or source.license),
            acquired_at=str(meta.get("acquired_at") or ""),
            source_url=str(meta.get("source_url") or source.url),
            attribution=str(meta.get("attribution") or source.attribution),
            crs_note=str(meta.get("crs_note") or source.crs_note),
            extent=meta.get("extent"),
        )

    getter = fetch or _http_get
    payload: dict[str, Any] | None = None
    raw = b""
    used_url = ""
    attribution = source.attribution
    failures: list[str] = []
    for url in source.urls:
        try:
            candidate = getter(url, timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            failures.append(f"{url}（{exc}）")
            continue
        try:
            decoded = json.loads(candidate.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            failures.append(f"{url}（不是合法 JSON：{exc}）")
            continue
        features = decoded.get("features") if isinstance(decoded, dict) else None
        if not isinstance(features, list) or not features:
            failures.append(f"{url}（没有要素）")
            continue
        payload, raw, used_url = decoded, candidate, url
        attribution = _attribution_for(url)
        break
    if payload is None:
        raise IngestError(
            f"边界数据下载失败（试了 {len(source.urls)} 个源）：" + "；".join(failures) + "。"
            f"可以手动下载一份该区划的 GeoJSON，存成 {target}（文件名必须一致），再重跑就会走缓存。"
        )
    features = payload.get("features") or []

    tmp = target.with_suffix(".geojson.tmp")
    tmp.write_bytes(raw)
    tmp.replace(target)

    described: dict[str, Any] = {}
    try:
        described = describe_geojson(target)
    except IngestError:
        described = {}
    meta = {
        "level": source.level,
        "adcode": source.adcode,
        "source_url": used_url,
        "attribution": attribution,
        "license": source.license,
        "crs_note": source.crs_note,
        "acquired_at": _now(),
        "feature_count": len(features),
        "extent": described.get("extent"),
        "bytes": len(raw),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return BoundaryResult(
        level=source.level,
        adcode=source.adcode,
        path=str(target),
        from_cache=False,
        feature_count=len(features),
        license=source.license,
        acquired_at=str(meta["acquired_at"]),
        source_url=used_url,
        attribution=attribution,
        crs_note=source.crs_note,
        extent=described.get("extent"),
    )


def _read_meta(meta_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}

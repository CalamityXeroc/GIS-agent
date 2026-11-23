# -*- coding: utf-8 -*-
"""数据资产预览 API：字段画像 / 离线图层预览 / 成果包沙箱预览。

三条边界（见 ``docs/p2-console-plan.md`` §6、§9）：

- **只读**：不写 catalog、不跑分析、不生成成果包；
- **不越界**：任何路径先 ``safe_path()`` 落到 workspace 内，报表还额外限定在
  ``output/`` 下、只回 ``text/html``；
- **CRS 归一（DAIR-299）**：预览统一输出 EPSG:4326——非 4326 输入用纯
  Python ``pyproj`` 实际重投影（3857 也转）；pyproj 不可用或 wkid 不认识时
  降级回「照原样画 + ``crs_warning``」，此时 ``crs`` 为 null。

几何读取复用 ``report/html_pack.py``：``.geojson`` 直读，GDB/shp 走内核一次
``arcpy.da.SearchCursor`` 取 Esri JSON，再转 GeoJSON；顶点压缩交给
``report/geom_simplify.py`` 做**保拓扑**简化（DAIR-368：原来的等间隔 stride
抽稀会让相邻要素的共享边界对不上、环退化成直弦）。

两类路径分开判存在性（DAIR-248）：文件型（``.geojson/.json/.shp/.gpkg/.kml``）
看 ``is_file()``；``*.gdb/数据集/要素类`` 在文件系统上**不是文件**，只能看
「父级库目录在不在 + 成员名非空」，否则 catalog 里所有 .gdb 要素类都会被误判
成「数据文件不存在」。

读取预算（DAIR-368）不再是一根 10s 硬线：:func:`_read_timeout` 按要读的要素数
放大（保底 :data:`DEFAULT_READ_TIMEOUT_S`、上限 :data:`MAX_READ_TIMEOUT_S`，两个
环境变量可调）。真超时了先**降级**（少读要素 + 更粗的顶点预算，返回
``truncated`` + ``degraded`` 让人知道不是全集），降级也超时才 422；前端没有
自己的短超时，两边看到的是同一个预算。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

try:  # 纯 Python 重投影（DAIR-299）；缺 pyproj 时降级回「照原样画 + 告警」。
    from pyproj import Transformer

    _HAS_PYPROJ = True
except ImportError:  # pragma: no cover - 环境312/313都已装 pyproj
    _HAS_PYPROJ = False

from ..datacatalog import bands as bands_meta
from ..datacatalog.store import Catalog, DatasetRecord
from ..report import geom_simplify, html_pack
from . import catalog_api
from .security import safe_path

logger = logging.getLogger(__name__)

#: 预览响应硬上限（计划 §6）：超过就继续抽稀，绝不把几十 MB 的几何塞给浏览器。
MAX_PREVIEW_BYTES = 5 * 1024 * 1024

#: 初始要素上限。与 ``html_pack.thin_features`` 的默认值保持一致。
DEFAULT_MAX_FEATURES = html_pack.MAX_FEATURES
#: 调用方最多能要多少要素（再大也还是会被 5MB 上限压回来）。
MAX_MAX_FEATURES = 5000
#: 预览的顶点预算（DAIR-368）。报告侧另有 ``html_pack.MAX_COORDS = 20k`` 的口径；
#: 网页预览要经得住缩放，给得宽一些：5MB 硬上限才是真正兼底的那道闸。
PREVIEW_POINT_BUDGET = 60_000

#: 预览缓存目录（相对 workspace）。
CACHE_DIR = Path(".gis_agent") / "preview_cache"

#: 成果包只认这两个文件名（``report/`` 的产物契约）。
REPORT_NAMES = ("成果包.html", "溯源卡.html")
#: 报表根目录（相对 workspace）。
REPORT_ROOT = "output"

#: 可以直接按经纬度/Web 墨卡托理解的数据；其余一律要告警。
KNOWN_CRS = {4326, 3857}

#: 字段类型里能算 min/max 的那些（arcpy Field.type 的写法）。
NUMERIC_TYPES = {
    "Double",
    "Single",
    "Integer",
    "SmallInteger",
    "BigInteger",
    "OID",
    "Float",
    "Float32",
    "Float64",
    "Int16",
    "Int32",
    "Int64",
    "Decimal",
}

CRS_WARNING = "未做投影变换"
CRS_UNKNOWN_WARNING = "坐标系未知，未做投影变换"

#: 预览响应统一输出的坐标系（MapLibre 只吃 4326）。
TARGET_CRS = "EPSG:4326"
#: 响应结构版本：改动字段契约时 +1，让旧预览缓存整体失效。
#: 3：3857 + 无 pyproj 的 ``crs`` 由谎报的 4326 改为 null（DAIR-301）。
#: 4：几何改成保拓扑简化（旧缓存里还是 stride 抽稀的碎边界，必须失效）。
#: 5：``feature_count`` 恢复成全量（读盘不再截要素），旧缓存里 .geojson 的
#:    ``truncated`` 是误报的 false（DAIR-368）。
#: 6：catalog 没画像时 ``feature_count`` 回退直读全量（旧缓存的 .geojson 会
#:    继续谎报 truncated=false）+ Esri rings 按方向判洞（旧缓存里小岛还被
#:    并成越界洞）（DAIR-375）。
PREVIEW_SCHEMA = 6

_transformers: dict[int, "Transformer"] = {}


def _transformer_for(wkid: int) -> "Transformer":
    """wkid → 4326 的 Transformer，按 wkid 复用（构造有解析开销）。"""
    cached = _transformers.get(wkid)
    if cached is None:
        cached = Transformer.from_crs(f"EPSG:{wkid}", TARGET_CRS, always_xy=True)
        _transformers[wkid] = cached
    return cached


def _reproject_coords(coords: Any, transformer: "Transformer") -> Any:
    """递归返回重投影后的新坐标树（不修改入参）。

    在副本上投影，成功后再整体替换：半途失败时原数据必须仍是**单一**坐标系，
    否则会得到一半米、一半度的混合几何（DAIR-301）。
    """
    if (
        isinstance(coords, list)
        and len(coords) >= 2
        and isinstance(coords[0], (int, float))
        and isinstance(coords[1], (int, float))
    ):
        x, y = transformer.transform(coords[0], coords[1])
        return [x, y, *coords[2:]]
    if isinstance(coords, list):
        return [_reproject_coords(item, transformer) for item in coords]
    return coords


def _reproject_collection(collection: dict[str, Any], wkid: int) -> bool:
    """把 GeoJSON FeatureCollection 重投影到 :data:`TARGET_CRS`。

    pyproj 不认识这个 wkid / 单个要素出错时返回 False，且**不改动** ``collection``
    （调用方降级为原样返回 + 告警）。
    """
    if not _HAS_PYPROJ:
        return False
    try:
        transformer = _transformer_for(wkid)
        features: list[Any] = []
        for feature in collection.get("features") or []:
            geometry = feature.get("geometry")
            if geometry and geometry.get("coordinates") is not None:
                feature = {
                    **feature,
                    "geometry": {
                        **geometry,
                        "coordinates": _reproject_coords(
                            geometry["coordinates"], transformer
                        ),
                    },
                }
            features.append(feature)
        collection["features"] = features
        return True
    except Exception as exc:
        logger.warning("重投影到 %s 失败（wkid=%s）：%s", TARGET_CRS, wkid, exc)
        return False

#: 能直接读的格式；其余走 ArcGIS 内核。
DIRECT_SUFFIXES = {".geojson", ".json"}

#: 几何读取保底预算（秒）：超过就降级，降级也超时才 422。
#: 环境变量 ``GIS_PREVIEW_READ_TIMEOUT`` 改保底值，``GIS_PREVIEW_MAX_READ_TIMEOUT`` 改上限。
DEFAULT_READ_TIMEOUT_S = 10.0
#: 自适应上限（秒）：要素再多也不过这个数。
MAX_READ_TIMEOUT_S = 180.0
#: 每多读一个要素多给多少秒（400 要素 ≈ +24s）。
READ_TIMEOUT_PER_FEATURE_S = 0.06

#: 内核冷启动是**一次性**成本（实测 ``import arcpy`` 10s / 内核就绪 8.3s），
#: 单独记账，不占几何读取的 10s 预算——否则第一个 .gdb 预览必然超时。
KERNEL_BOOT_TIMEOUT_S = 25.0

#: 路径形态：普通文件 / .gdb 目录本身 / .gdb 内的要素类 / 不存在。
PATH_FILE = "file"
PATH_GDB = "gdb"
PATH_GDB_FEATURE_CLASS = "gdb_feature_class"
PATH_MISSING = "missing"

#: 允许离线预览的 catalog kind（.gdb 内的要素类在 catalog 里记作 feature_class）。
PREVIEWABLE_KINDS = {"vector", "feature_class"}

TIMEOUT_SUGGESTION = "要素类太大或内核不可用；已降级只读部分要素仍未读完，可先用 .geojson 预览，或等内核恢复后重试"

#: 报表页面的 CSP：脚本/样式可以跑（成果包是内联 canvas + 内联 CSS），但拿不到
#: 同源 cookie、不能发请求、不能读父页面。iframe 的 ``sandbox`` 是第二道闸。
REPORT_CSP = (
    "sandbox allow-scripts; default-src 'none'; "
    "style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
    "img-src data: blob:; font-src data:"
)


# ------------------------------------------------------------------ 字段画像
def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _field_stats(field: dict[str, Any]) -> dict[str, Any]:
    """在 catalog 已有的 ``samples`` 上算 min/max，**不重扫数据表**。

    catalog 抽样本来就只有几十个值（扫描时 ``sample_values`` 限制），所以这里给
    的是「样本范围」，前端标注为「按样本」。要看精确范围得另开单做全表统计，
    那正是本单明确不做的。
    """
    stats: dict[str, Any] = {"numeric": False, "sample_count": 0}
    samples = field.get("samples") or []
    stats["sample_count"] = len(samples)
    ftype = str(field.get("type") or "")
    numbers = [n for n in (_numeric(v) for v in samples) if n is not None]
    if ftype in NUMERIC_TYPES and numbers:
        stats.update(
            {
                "numeric": True,
                "min": min(numbers),
                "max": max(numbers),
                "range_basis": "samples",
            }
        )
    return stats


def profile_dataset(
    workspace: str | Path,
    path: str,
    *,
    code_runner: Any = None,
    band_budget: float | None = None,
) -> dict[str, Any]:
    """catalog 里那条记录的完整画像 + 逐字段样本统计 + 栅格逐波段元数据。

    栅格额外给 ``bands``（逐波段 dtype/min/max/mean/nodata/尺寸）与
    ``bands_info``（缓存是否命中、耗时、不可用原因）。无 arcpy 或未算统计值时
    ``bands`` 为空且 ``bands_info.unavailable`` 给出原因——**不伪造数值**。
    """
    data = catalog_api.dataset_profile(workspace, path)
    fields = []
    for field in data.get("fields") or []:
        item = dict(field)
        item.update(_field_stats(field))
        fields.append(item)
    data["fields"] = fields
    data["field_count"] = len(fields)
    # 数值列清单：前端在字段表上方做一句摘要，不用自己再过滤一遍。
    data["numeric_fields"] = [f["name"] for f in fields if f.get("numeric")]
    if data.get("kind") == "raster":
        raster = data.get("raster") or {}
        try:
            resolved = _profile_target(workspace, path)
            band_info = bands_meta.read_bands(
                workspace,
                resolved,
                code_runner=code_runner,
                budget_s=band_budget,
                fallback_band_count=raster.get("band_count"),
            )
        except Exception as exc:  # 画像不能因为波段读不到就 500
            logger.warning("逐波段元数据读取失败 %s: %s", path, exc)
            band_info = {
                "band_count": raster.get("band_count"),
                "bands": [],
                "cached": False,
                "elapsed_ms": 0.0,
                "unavailable": f"逐波段元数据读取失败：{exc}",
                "notes": [],
            }
        data["bands"] = band_info.get("bands") or []
        data["bands_info"] = {k: v for k, v in band_info.items() if k != "bands"}
    return data


def _profile_target(workspace: str | Path, path: str) -> Path:
    """画像里的数据集路径 → workspace 内绝对路径（与预览同一套解析规则）。"""
    return _resolve_target(Path(workspace).resolve(), path)


# ------------------------------------------------------------------ 路径分类
def _gdb_container(path: Path) -> tuple[Path, str] | None:
    """``…/x.gdb/数据集/要素类`` → (``…/x.gdb``, ``数据集/要素类``)；不在库里返回 None。

    ``.gdb`` 内的要素类在文件系统上**不是文件**：``exists()/is_file()`` 一律 False。
    判断存在性只能看「父级库目录在不在 + 成员名非空」（DAIR-243 实测：catalog 里
    200 条全是这种，用 ``is_file()`` 判存在会让预览对所有真实数据返回 404）。
    """
    text = str(path)
    lowered = text.lower().replace("\\", "/")
    index = lowered.find(".gdb/")
    if index < 0:
        return None
    member = text[index + 5 :].strip("/\\")
    if not member:
        return None
    return Path(text[: index + 4]), member


def _path_kind(path: Path) -> str:
    """file / gdb（库目录本身）/ gdb_feature_class（库内成员）/ missing。"""
    container = _gdb_container(path)
    if container is not None:
        return PATH_GDB_FEATURE_CLASS if container[0].is_dir() else PATH_MISSING
    if path.suffix.lower() == ".gdb":
        return PATH_GDB if path.is_dir() else PATH_MISSING
    return PATH_FILE if path.is_file() else PATH_MISSING


def _resolve_target(workspace: Path, path: str) -> Path:
    """入参 → workspace 内的路径（先过 ``safe_path`` 的越界闸）。

    兼容 catalog 里「扫描时 cwd = 仓库根」留下的相对写法（``workspace/input/x``）：
    首段与工作区同名时去掉它再试一次。结果仍在 workspace 内，越界边界不变。
    """
    resolved = safe_path(workspace, path)
    if _path_kind(resolved) != PATH_MISSING:
        return resolved
    raw = Path(path)
    if not raw.is_absolute() and len(raw.parts) > 1:
        head, *rest = raw.parts
        if head.lower() == workspace.name.lower():
            candidate = workspace.joinpath(*rest)
            if _path_kind(candidate) != PATH_MISSING:
                return candidate
    return resolved


# ------------------------------------------------------------------ 几何读取
def _stat_key(path: Path) -> tuple[float, int]:
    """缓存键用的 (mtime, size)。

    GDB 里的要素类在文件系统上不存在，退回「容器内容时间」（忽略 ``*.sr.lock``，
    否则读一次就把缓存全废掉，见 ``scanner.gdb_content_mtime``）。
    """
    try:
        stat = path.stat()
        return stat.st_mtime, stat.st_size
    except OSError:
        pass
    container = _gdb_container(path)
    if container is not None:
        from ..datacatalog.scanner import gdb_content_mtime

        try:
            return gdb_content_mtime(container[0]), 0
        except OSError:  # pragma: no cover - 目录都读不到
            return 0.0, 0
    return 0.0, 0


def _crs_warning(wkid: int | None) -> str | None:
    if wkid in KNOWN_CRS:
        return None
    if wkid:
        return CRS_WARNING
    return CRS_UNKNOWN_WARNING


#: 单次预览最多带回的属性列（DAIR-330 唯一值/分级符号化）。
MAX_PREVIEW_FIELDS = 3


def _preview_fields(record: Any, raw: str) -> list[str]:
    """``fields=`` 入参 → 真实字段白名单里的字段名（顺序去重，≤3）。

    字段名会被注入内核代码（``repr`` 成列表字面量），所以先过 catalog 的真实
    字段白名单：认不出的直接丢掉，不猜、不报错。没有 catalog 记录时一律不放行
    （宁可着色回退到单色，也不把未知列名拼进内核）。
    """
    names = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if not names or record is None:
        return []
    allowed = {str(field.get("name")) for field in (record.fields or [])}
    picked: list[str] = []
    for name in names:
        if name in allowed and name not in picked:
            picked.append(name)
    return picked[:MAX_PREVIEW_FIELDS]


def _cache_key(path: Path, max_features: int, fields: Sequence[str] = ()) -> tuple[str, float, int]:
    mtime, size = _stat_key(path)
    # ``pyproj`` 可用性进键（DAIR-303 D4）：同一工作区换 venv / 补装 pyproj 后，
    # 旧的「未重投影」payload 不能再命中，否则 crs=null + 米制 bbox 会继续下发。
    digest = hashlib.sha1(
        f"{path}|{mtime}|{size}|{max_features}|schema={PREVIEW_SCHEMA}"
        f"|pyproj={int(_HAS_PYPROJ)}|fields={','.join(fields)}".encode("utf-8", errors="replace")
    ).hexdigest()
    return digest, mtime, size


def _cache_path(workspace: Path, digest: str) -> Path:
    return workspace / CACHE_DIR / f"{digest}.json"


def _encoded_size(payload: Any) -> int:
    return len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _read_cached(cache_file: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    try:
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    # mtime/size 都在键里，这里再核一次是给「键算对了但文件被换掉」兜底。
    for key in ("mtime", "size"):
        if raw.get(key) != expected.get(key):
            return None
    payload = raw.get("payload")
    return payload if isinstance(payload, dict) else None


def _write_cached(cache_file: Path, envelope: dict[str, Any]) -> None:
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(envelope, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    except OSError as exc:  # 缓存写不进去只影响下次速度，不影响本次响应
        logger.debug("预览缓存写入失败 %s: %s", cache_file, exc)


def _empty_geojson() -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": []}


def _fit_payload(
    collection: dict[str, Any],
    *,
    base: dict[str, Any],
    point_budget: int,
    max_features: int | None = None,
    upstream: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """把几何压到响应 ≤5MB：**先加容差**（少顶点、保留全部要素），再只说 bbox。

    原来这里是「减半重抽要素」：要素一少，地图上直接缺块。顶点才是真正占地
    方的，所以先用 :func:`geom_simplify.simplify_feature_collection` 按预算缩小
    顶点（仍是保拓扑的），实在压不进去才退回只给 bbox。

    ``max_features`` 只在这里（简化之前）截要素：读盘时不能截，否则
    ``feature_count`` 拿到的就是截断后的数字，``truncated`` 会谎报成 false
    （DAIR-368 回归）。

    返回 ``(最终 payload, 简化后的 collection, 实际顶点预算)``。
    """
    if max_features:
        limit = max(1, int(max_features))
        features = collection.get("features") or []
        if len(features) > limit:
            collection = {**collection, "features": features[:limit]}
    budget = max(200, int(point_budget))
    last: dict[str, Any] = {}
    for _ in range(8):
        simplified, stats = geom_simplify.simplify_feature_collection(
            collection, max_points=budget
        )
        payload = dict(base)
        payload["geojson"] = simplified
        payload["returned_features"] = len(simplified.get("features") or [])
        payload["point_count"] = html_pack.count_points(simplified)
        payload["point_budget"] = budget
        # 内核先减过点时，「原始顶点」要取内核看到的那份，否则 payload 会说
        # 「input == output，没简化」——100 万点被压到 1 千点的图层会被标成没动过。
        own_input = int(stats.get("input_points") or 0)
        own_output = int(stats.get("output_points") or 0)
        upstream_input = int((upstream or {}).get("input_points") or 0)
        payload["input_points"] = max(own_input, upstream_input)
        # 主进程这一轮没动几何（内核已经减完）时，报内核用的容差；否则报本轮的。
        effective_tolerance = (
            (upstream or {}).get("tolerance")
            if own_input == own_output
            else stats.get("tolerance")
        )
        if effective_tolerance is None:
            effective_tolerance = stats.get("tolerance") or 0.0
        payload["tolerance"] = round(float(effective_tolerance), 9)
        # 有没有真的丢顶点：前端据此标「简化」，不用自己拿容差猜。
        payload["simplified"] = payload["input_points"] > int(
            stats.get("output_points") or 0
        )
        payload["truncated"] = bool(
            payload.get("feature_count")
            and payload["feature_count"] > payload["returned_features"]
        )
        if _encoded_size(payload) <= MAX_PREVIEW_BYTES:
            return payload, simplified, budget
        last = payload
        budget = max(200, budget // 2)
    # 极端情况：单个要素的坐标就超 5MB（比如一个上百万点的多边形）。宁可只回
    # bbox 也不回一个打不开的响应；前端有 bbox 仍能说明范围。
    payload = dict(base)
    payload.update(
        {
            "geojson": _empty_geojson(),
            "returned_features": 0,
            "point_count": 0,
            "point_budget": 0,
            "tolerance": last.get("tolerance"),
            "input_points": last.get("input_points"),
            "simplified": True,
            "truncated": True,
            "oversized": True,
        }
    )
    return payload, _empty_geojson(), 0


def _dataset_record(
    workspace: str | Path, resolved: Path, raw_path: str
) -> DatasetRecord | None:
    """catalog 里的这条记录（查找口径与属性表/元数据一致，见 ``catalog_api._find_record``）。"""
    catalog: Catalog | None = catalog_api._open_catalog(workspace)
    if catalog is None:
        return None
    try:
        return catalog_api._find_record(catalog, workspace, resolved, raw_path)
    finally:
        catalog.close()


class PreviewTimeout(RuntimeError):
    """几何读取超时。不是 HTTPException：调用方要带上 elapsed_ms 再组装响应。"""

    def __init__(self, elapsed_ms: float, timeout_s: float) -> None:
        super().__init__(f"几何读取超过 {timeout_s:g}s")
        self.elapsed_ms = elapsed_ms
        self.timeout_s = timeout_s


def _read_timeout(features: int | None = None) -> float:
    """几何读取预算（秒）：保底 :data:`DEFAULT_READ_TIMEOUT_S`，按要素数放大。

    为什么不能再用固定 10s：密集图层光是全精度几何的内核序列化 + 进程间回传
    就吃掉十几秒，硬线只能把「慢」变成「错」（用户看到的 ``服务端 10029 ms`` 的
    422）。现在按要读的要素数给预算，并留两个环境变量让运维调：

    - ``GIS_PREVIEW_READ_TIMEOUT``：保底值（默认 10s）；
    - ``GIS_PREVIEW_MAX_READ_TIMEOUT``：上限（默认 180s）。

    只设保底值时上限跟着保底走（运维只改一个数，行为必须确定，不能反而被自适应
    放大）；两个都设就是「保底→上限」的区间。
    """
    base = _env_seconds("GIS_PREVIEW_READ_TIMEOUT", DEFAULT_READ_TIMEOUT_S)
    explicit_cap = _env_seconds("GIS_PREVIEW_MAX_READ_TIMEOUT", None)
    if explicit_cap is None:
        cap = MAX_READ_TIMEOUT_S if base == DEFAULT_READ_TIMEOUT_S else base
    else:
        cap = explicit_cap
    cap = max(base, cap)
    if not features:
        return base
    return round(min(cap, base + READ_TIMEOUT_PER_FEATURE_S * max(1, int(features))), 3)


def _env_seconds(name: str, default: float | None) -> float | None:
    try:
        value = float(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


class _ReadPlan:
    """一档读取预算：要素数 / 顶点预算 / 超时。降级 = 缩一档重试。"""

    __slots__ = ("limit", "point_budget", "timeout")

    def __init__(self, limit: int, point_budget: int, timeout: float) -> None:
        self.limit = max(1, int(limit))
        self.point_budget = max(200, int(point_budget))
        self.timeout = float(timeout)

    def degraded(self) -> "_ReadPlan":
        """阼一级：要素数 / 顶点预算都压到 1/4，预算不延长（总耗时 ≈ 1 倍）。"""
        return _ReadPlan(self.limit // 4, self.point_budget // 4, self.timeout)

    def as_detail(self) -> dict[str, Any]:
        return {
            "feature_limit": self.limit,
            "point_budget": self.point_budget,
            "timeout_s": self.timeout,
        }


def _timeout_detail(
    elapsed_ms: float, timeout_s: float, degraded: dict[str, Any] | None = None
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "code": "preview_timeout",
        "reason": f"读取几何超时：超过 {timeout_s:g}s 仍未读回",
        "suggestion": TIMEOUT_SUGGESTION,
        "elapsed_ms": elapsed_ms,
        "timeout_s": timeout_s,
    }
    if degraded:
        detail["degraded"] = degraded
    return detail


def _no_geometry_detail(elapsed_ms: float) -> dict[str, Any]:
    """非空间表：只有属性、没有几何，预览接口给不出图层。"""
    return {
        "code": "no_geometry",
        "reason": "该数据集是非空间表（没有几何），只能看属性表",
        "suggestion": "双击打开属性表",
        "elapsed_ms": elapsed_ms,
    }


def _kernel_detail(reason: str, elapsed_ms: float) -> dict[str, Any]:
    return {
        "code": "kernel_unavailable",
        "reason": f"ArcGIS 内核不可用：{reason}" if reason else "ArcGIS 内核不可用",
        "suggestion": "内核恢复前可先用 .geojson 预览；或稍后重试",
        "elapsed_ms": elapsed_ms,
    }


def _looks_like_timeout(error: Any) -> bool:
    if isinstance(error, dict):
        text = f"{error.get('type', '')} {error.get('message', '')}"
    else:
        text = str(error or "")
    lowered = text.lower()
    return "timeout" in lowered or "timed out" in lowered


def _wait_kernel_ready(code_runner: Any) -> None:
    """先等内核预热（一次性成本），再把 10s 预算交给几何读取。

    不分开记账的话，第一个 .gdb 预览必然 422：``import arcpy`` 就要 10s，正好
    吃掉整个读取预算。测试注入的假 runner 没有 prewarm/wait_prewarm，直接跳过。
    """
    prewarm = getattr(code_runner, "prewarm", None)
    if callable(prewarm):
        try:
            prewarm()
        except Exception as exc:  # pragma: no cover - 预热失败留给读取阶段报错
            logger.debug("预览内核预热失败: %s", exc)
    wait = getattr(code_runner, "wait_prewarm", None)
    if callable(wait):
        try:
            wait(KERNEL_BOOT_TIMEOUT_S)
        except Exception as exc:  # pragma: no cover
            logger.debug("预览内核预热等待失败: %s", exc)


def _load_via_kernel(
    resolved: Path,
    *,
    code_runner: Any,
    limit: int,
    read_timeout: float,
    fields: Sequence[str] = (),
    point_budget: int | None = None,
    kernel_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """.gdb 要素类 / .shp 走内核：一次 ``arcpy.da.SearchCursor`` 取 Esri JSON。

    ``point_budget`` 给定时顶点在内核里就压完（``layer_probe_code(..., max_points=)``）：
    全精度几何只在 ArcGIS 进程内存在过，进程间通道只走简化后的小 JSON。

    ``kernel_stats`` 是给调用方回填诊断的口子：内核已经减过点时，主进程再算一遍
    「输入顶点」只会看到内核的输出，payload 里的 ``simplified`` 就会谎报成 false。
    """
    started = time.perf_counter()

    def elapsed() -> float:
        return round((time.perf_counter() - started) * 1000, 2)

    if code_runner is None:
        raise HTTPException(
            status_code=503,
            detail=_kernel_detail(
                "未检测到 arcpy（当前环境没有 ArcGIS Pro Python）", elapsed()
            ),
        )
    code = html_pack.layer_probe_code(
        str(resolved), limit=limit, fields=fields, max_points=point_budget
    )
    outcome = code_runner.run(code, timeout=read_timeout)
    if not getattr(outcome, "ok", False):
        error = getattr(outcome, "error", None)
        if _looks_like_timeout(error):
            raise PreviewTimeout(elapsed(), read_timeout)
        reason = (
            str(error.get("message") or error.get("type") or "")
            if isinstance(error, dict)
            else str(error or "")
        )
        raise HTTPException(status_code=503, detail=_kernel_detail(reason, elapsed()))
    payload = getattr(outcome, "result", None)
    if isinstance(payload, dict) and isinstance(payload.get("simplify"), dict):
        logger.debug("内核侧几何简化 %s: %s", resolved, payload.get("simplify"))
        if kernel_stats is not None:
            kernel_stats.update(payload["simplify"])
    if isinstance(payload, dict) and payload.get("error"):
        # 旧 catalog 里没归类的非空间表（kind 还是 feature_class）：内核取几何时
        # 报的正是这句，映射成同一个 code，前端按「只能看属性表」处理。
        if "table has no geometry" in str(payload["error"]).lower():
            raise HTTPException(status_code=422, detail=_no_geometry_detail(elapsed()))
        raise HTTPException(
            status_code=422,
            detail={
                "code": "dataset_unreadable",
                "reason": f"内核读不出该图层：{payload['error']}",
                "suggestion": "确认要素类名与 .gdb 是否对应；可先用 .geojson 预览",
                "elapsed_ms": elapsed(),
            },
        )
    if fields and isinstance(payload, dict):
        return html_pack.esri_rows_to_feature_collection(payload.get("rows") or [], fields)
    geoms = payload.get("geoms") if isinstance(payload, dict) else None
    return html_pack.esri_geometries_to_feature_collection(geoms or [])


def preview_layer(
    workspace: str | Path,
    path: str,
    *,
    max_features: int = DEFAULT_MAX_FEATURES,
    code_runner: Any = None,
    read_timeout: float | None = None,
    fields: Sequence[str] = (),
) -> dict[str, Any]:
    """读一个矢量图层 → GeoJSON → 保拓扑简化 → 落盘缓存（键 = path + mtime + 上限）。

    ``fields``：额外带回的属性列（DAIR-330 符号化着色用，≤3 个，先过 catalog
    真实字段白名单）。空 = 只回几何（与旧行为一致）。``.geojson`` 直读本来就有
    属性，此时 ``fields`` 不参与。

    读取预算按要素数自适应（:func:`_read_timeout`）；真超时先降级读一部分
    （``truncated`` + ``degraded`` 告知），降级也超时才 422。
    """
    ws = Path(workspace).resolve()
    resolved = _resolve_target(ws, path)
    max_features = max(1, min(int(max_features), MAX_MAX_FEATURES))
    started = time.perf_counter()

    target_kind = _path_kind(resolved)
    if target_kind == PATH_GDB:
        raise HTTPException(
            status_code=422,
            detail="这是文件数据库（.gdb）目录，请选择其中的要素类再预览",
        )
    if target_kind == PATH_MISSING:
        raise HTTPException(status_code=404, detail="数据文件不存在")

    record = _dataset_record(ws, resolved, path)
    if record is not None and record.kind == "table":
        # 非空间表：内核取几何必然报 "table has no geometry"，直接短路，
        # 别让浏览器白等一次内核调用（用户看到的是可读的原因，不是 422 噪音）。
        raise HTTPException(
            status_code=422,
            detail=_no_geometry_detail(
                round((time.perf_counter() - started) * 1000, 2)
            ),
        )
    if record is not None and record.kind and record.kind not in PREVIEWABLE_KINDS:
        raise HTTPException(
            status_code=422,
            detail=f"该类型（{record.kind}）不支持离线图层预览，仅支持矢量",
        )

    # 字段白名单必须拿到 catalog 记录；拿不到就当没要属性（不猜列名）。
    wanted = _preview_fields(record, ",".join(fields))
    digest, mtime, size = _cache_key(resolved, max_features, wanted)
    cache_file = _cache_path(ws, digest)
    cached = _read_cached(cache_file, {"mtime": mtime, "size": size})
    if cached is not None:
        cached["cache"] = {
            "hit": True,
            "key": digest[:12],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        return cached

    # 预算：要素数决定时长（catalog 知道真实要素数时按它，否则按本次上限）。
    known = record.feature_count if record is not None else None
    plan = _ReadPlan(
        max_features,
        PREVIEW_POINT_BUDGET,
        float(read_timeout) if read_timeout else _read_timeout(min(known or max_features, max_features)),
    )

    direct = resolved.suffix.lower() in DIRECT_SUFFIXES
    if not direct:
        # 内核冷启动是一次性成本，先等它（不计入几何读取预算）；
        # 内核根本不可用时由 _load_via_kernel 转成 503 + 原因。
        _wait_kernel_ready(code_runner)

    #: 内核侧简化统计（内核先减点时才非空，见 ``_load_via_kernel``）。
    kernel_stats: dict[str, Any] = {}

    def read_once(current: _ReadPlan) -> dict[str, Any]:
        if direct:
            # .geojson 直读不吃 point_budget：文件已整份在内存里，顶点在主进程减。
            # 要素数也不在这里截：要留全量给 feature_count 算 truncated
            # （``_fit_payload`` 会在简化前截）。
            return _call_with_timeout(
                lambda: html_pack.load_layer(
                    resolved, code_runner=None, limit=None, workspace=ws
                ),
                current.timeout,
            )
        return _call_with_timeout(
            lambda: _load_via_kernel(
                resolved,
                code_runner=code_runner,
                limit=current.limit,
                read_timeout=current.timeout,
                fields=wanted,
                point_budget=current.point_budget,
                kernel_stats=kernel_stats,
            ),
            current.timeout,
        )

    # read_ms 只算几何读取本身：不含内核冷启动/预热，否则冷启动的十几秒会
    # 被读成「读一次几何要 12s」，看不出真实瓶颈。
    read_started = time.perf_counter()
    degraded: dict[str, Any] | None = None
    try:
        collection = read_once(plan)
    except PreviewTimeout as exc:
        # 超时先降级：少读 3/4 要素 + 更粗的顶点预算，而不是直接 422。
        # 只拿回「部分要素 + 明确告知」比「什么都没有」有用得多。
        retry = plan.degraded()
        try:
            collection = read_once(retry)
        except PreviewTimeout as exc2:
            raise HTTPException(
                status_code=422,
                detail=_timeout_detail(
                    exc2.elapsed_ms,
                    exc.timeout_s,
                    {"attempt": retry.as_detail(), "first_elapsed_ms": exc.elapsed_ms},
                ),
            ) from exc2
        degraded = {
            "code": "preview_degraded",
            "reason": (
                f"读取超过 {exc.timeout_s:g}s，已降级为只读前 {retry.limit} 个要素"
                f"（顶点预算 {retry.point_budget}），图形已简化"
            ),
            "elapsed_ms": exc.elapsed_ms,
            "first_blocked_ms": exc.elapsed_ms,
        }
        degraded.update(retry.as_detail())
        logger.info("预览降级 %s: %s", resolved, degraded["reason"])
    except HTTPException:
        raise
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"无权限读取该文件：{exc}") from exc
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"读取失败：{exc}") from exc
    except Exception as exc:  # pragma: no cover - 内核异常兜底
        logger.warning("图层读取异常 %s: %s", resolved, exc)
        raise HTTPException(status_code=422, detail=f"读取失败：{exc}") from exc

    read_ms = round((time.perf_counter() - read_started) * 1000, 2)
    if not collection or not collection.get("features"):
        raise HTTPException(
            status_code=422, detail="图层没有可预览的要素（空图层或几何读取失败）"
        )

    wkid = record.wkid if record is not None else None

    # CRS 归一（DAIR-299）：非 4326 的输入实际重投影到 4326 再返回（3857 也转，
    # 否则前端把米当度画）；pyproj 缺失或 wkid 不认识时降级回「照原样画 + crs_warning」。
    crs_warning = _crs_warning(wkid)
    reprojected = False
    if wkid and wkid != 4326:
        reprojected = _reproject_collection(collection, wkid)
        if reprojected:
            crs_warning = None
    # 「实际是 4326」只有两个来源：源数据本来就是 4326，或确实重投影成功了。
    # 3857 在 KNOWN_CRS 里 → _crs_warning 返回 None，但 pyproj 缺失时
    # reprojected=False：不补告警就会把米制数据谎报成 4326（DAIR-301）。
    if wkid not in (None, 4326) and not reprojected:
        crs_warning = crs_warning or CRS_WARNING

    # 重投影后 catalog 的 extent（原 CRS 的米制值）不再可用，从集合重算。
    bbox = (None if reprojected else (record.extent if record is not None else None)) or (
        html_pack.bbox_of(collection)
    )
    if bbox is None:
        raise HTTPException(status_code=422, detail="图层没有可用的坐标范围")

    feature_count = record.feature_count if record is not None else None
    if feature_count is None and (direct or record is None):
        # catalog 对 .geojson 只做轻量画像，记录的 ``feature_count`` 常是 NULL。
        # 直读文件时 ``load_layer`` 拿到的本来就是全量（limit=None），这时必须
        # 回退到内存里的真实条数；否则 ``_fit_payload`` 按 ``feature_count`` 算
        # 的 ``truncated`` 会把「只给了前 max_features 个」谎报成完整（DAIR-375）。
        feature_count = len(collection["features"])

    base = {
        "path": str(resolved),
        "name": (record.name if record is not None and record.name else resolved.name),
        "kind": (
            record.kind
            if record is not None
            else (
                "feature_class"
                if target_kind == PATH_GDB_FEATURE_CLASS
                else "vector"
            )
        ),
        "wkid": wkid,
        "crs_name": record.crs_name if record is not None else "",
        "crs": TARGET_CRS if crs_warning is None else None,
        "crs_warning": crs_warning,
        "bbox": [float(v) for v in bbox],
        "feature_count": feature_count,
        "field_count": len(record.fields) if record is not None else 0,
        # 服务端实际给了多少读取预算：前端不用自己猜 10s（DAIR-368）。
        "budget_s": plan.timeout,
        "max_features": plan.limit,
        "degraded": degraded,
    }
    payload, _, effective = _fit_payload(
        collection,
        base=base,
        point_budget=plan.point_budget,
        max_features=plan.limit,
        upstream=kernel_stats,
    )
    payload["cache"] = {
        "hit": False,
        "key": digest[:12],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "read_ms": read_ms,
    }

    if degraded is None:
        # 降级（超时后只读了部分要素）的结果**不落盘**（DAIR-370 复审阻塞项）：
        # 一次瞬时慢（内核冷启动/排队/许可等待）不能把「前 N 个要素」写成这个
        # 图层的永久答案——缓存键只认 mtime/size，写进去以后用户刷新也没用。
        # 降级只对本次请求有效，下一次预览重新尝试读全量，仍超时再降级。
        _write_cached(
            cache_file,
            {
                "mtime": mtime,
                "size": size,
                "max_features": effective,
                "created_at": time.time(),
                "payload": payload,
            },
        )
    return payload

# ------------------------------------------------------------------ 成果包
def _report_root(workspace: str | Path) -> Path:
    return (Path(workspace).resolve() / REPORT_ROOT).resolve()


def list_reports(workspace: str | Path) -> dict[str, Any]:
    """``output/`` 下的成果包 / 溯源卡（只认这两个文件名，不做全盘递归匹配）。"""
    root = _report_root(workspace)
    items: list[dict[str, Any]] = []
    if root.is_dir():
        try:
            candidates = sorted(root.rglob("*.html"))
        except OSError:
            candidates = []
        for path in candidates:
            if path.name not in REPORT_NAMES:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:  # pragma: no cover - rglob 不会越界
                continue
            items.append(
                {
                    "name": relative,
                    "dir": path.parent.relative_to(root).as_posix() or ".",
                    "kind": "package" if path.name == REPORT_NAMES[0] else "provenance",
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "url": f"/api/reports/{relative}",
                }
            )
    items.sort(key=lambda item: item["mtime"], reverse=True)
    return {
        "items": items,
        "count": len(items),
        "root": REPORT_ROOT,
        "names": list(REPORT_NAMES),
    }


def read_report(workspace: str | Path, name: str) -> Path:
    """把报表名解析成 ``output/`` 内的真实文件；任何越界/非 HTML 都拒绝。"""
    relative = str(name).replace("\\", "/").lstrip("/")
    if not relative.lower().endswith(".html"):
        raise HTTPException(status_code=415, detail="只允许预览 text/html 报表")
    root = _report_root(workspace)
    resolved = safe_path(workspace, Path(REPORT_ROOT) / relative)
    if resolved != root and root not in resolved.parents:
        raise HTTPException(status_code=403, detail="报表路径超出 output/ 目录")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="报表不存在")
    return resolved


# ------------------------------------------------------------------ 路由
def _default_runner_loader(workspace: Path) -> Callable[[], Any]:
    def load() -> Any:
        # 先做一次解释器发现：没有 ArcGIS Pro 就别去起内核（否则要白等
        # 冷启动超时）。发现结果有 lru_cache，重复调用很便宜。
        from ..arcpy_bridge import discover_arcgis_pro_python
        from ..runtime.code_runner import CodeRunner

        info = discover_arcgis_pro_python()
        return CodeRunner(
            workspace=workspace,
            python_executable=info.python_executable,
            state_dir=workspace / ".gis_agent",
            exec_timeout=120.0,
        )

    return load


def _call_with_timeout(fn: Callable[[], Any], timeout: float) -> Any:
    """在守护线程里执行并限时等待，超时抛 :class:`PreviewTimeout`。

    为什么还要这层：内核可能卡在冷启动 / 许可等待上（DAIR-243 实测一次 120s），
    而预览是同步 HTTP 请求。内核自己的超时只在真正执行代码时生效，这层是墙钟
    兼底：超时后请求立刻返回，后台线程自行收尾（daemon，不阻塞进程退出）。
    """
    import threading

    box: dict[str, Any] = {}
    started = time.perf_counter()

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - 原样抛给请求线程
            box["error"] = exc

    thread = threading.Thread(target=target, name="gis-agent-preview-geom", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise PreviewTimeout(round((time.perf_counter() - started) * 1000, 2), timeout)
    if "error" in box:
        raise box["error"]
    return box.get("value")


class PreviewRuntime:
    """预览用的内核持有者：懒加载 + 关服务时能一起收掉。"""

    def __init__(
        self,
        workspace: Path,
        *,
        code_runner: Any = None,
        runner_loader: Callable[[], Any] | None = None,
    ) -> None:
        self.workspace = workspace
        self._runner = code_runner
        self._loader = runner_loader or _default_runner_loader(workspace)
        self._loaded = code_runner is not None

    def get(self) -> Any:
        if not self._loaded:
            self._loaded = True
            try:
                self._runner = self._loader()
            except Exception as exc:  # pragma: no cover - 无 ArcGIS 的机器
                logger.warning("预览内核不可用: %s", exc)
                self._runner = None
        return self._runner

    def close(self) -> None:
        runner = self._runner
        self._runner = None
        shutdown = getattr(runner, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:  # pragma: no cover - 关停失败不影响已返回的响应
                logger.debug("预览内核关停失败: %s", exc)


def _quote(text: str) -> str:
    from urllib.parse import quote

    return quote(text, safe="")


def build_router(
    workspace: str | Path,
    *,
    code_runner: Any = None,
    runner_loader: Callable[[], Any] | None = None,
    runtime: PreviewRuntime | None = None,
) -> APIRouter:
    """装配预览路由。

    ``code_runner`` 直接注入（测试用）；``runner_loader`` 给生产用的懒加载
    （第一次预览 GDB/shp 时才起内核，不用为一个 .geojson 白等 10s 的 arcpy 导入）。
    ``runtime`` 传进来时由调用方负责 ``close()``（``app.py`` 在 lifespan 里收）。
    """
    ws = Path(workspace).resolve()
    preview_runtime = runtime or PreviewRuntime(
        ws, code_runner=code_runner, runner_loader=runner_loader
    )

    router = APIRouter()

    @router.get("/api/catalog/datasets/{path:path}/profile")
    def get_dataset_profile(path: str) -> dict[str, Any]:
        return profile_dataset(ws, path, code_runner=preview_runtime.get())

    @router.get("/api/catalog/preview")
    def get_preview(
        path: str = Query(..., description="图层路径（必须在 workspace 内）"),
        max_features: int = Query(
            DEFAULT_MAX_FEATURES, ge=1, le=MAX_MAX_FEATURES, description="要素上限"
        ),
        fields: str = Query(
            "",
            description=(
                "额外带回的属性列（逗号分隔，≤3 个，先过 catalog 真实字段白名单）；"
                "供唯一值/分级符号化在地图上按字段着色"
            ),
        ),
    ) -> dict[str, Any]:
        return preview_layer(
            ws,
            path,
            max_features=max_features,
            code_runner=preview_runtime.get(),
            fields=[item for item in fields.split(",") if item.strip()],
        )

    @router.get("/api/reports")
    def get_reports() -> dict[str, Any]:
        return list_reports(ws)

    @router.get("/api/reports/{name:path}")
    def get_report(name: str, download: bool = False) -> HTMLResponse:
        target = read_report(ws, name)
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=f"无权限读取报表：{exc}") from exc
        except OSError as exc:
            raise HTTPException(status_code=422, detail=f"报表读取失败：{exc}") from exc
        headers = {
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": REPORT_CSP,
        }
        if download:
            headers["Content-Disposition"] = (
                f"attachment; filename*=UTF-8''{_quote(target.name)}"
            )
        return HTMLResponse(text, headers=headers)

    return router

# -*- coding: utf-8 -*-
"""单文件 HTML 成果包（P0-4 步 a）。

硬约束（验收条件）：**断网打开、不需要 ArcGIS、单文件 ≤20MB**。所以：

- 图 base64 内嵌，原图超过 1600px 宽就先缩图再压（Pillow，已有依赖）；
- 图层是内联的 GeoJSON，用原生 canvas + 少量 JS 画，不引任何 CDN/地图库；
- 模板里没有一条外部请求（没有 ``http://``、没有外链字体/脚本），
  单测直接断言这件事。

空间数据从哪来：``.geojson`` 直接读；GDB 要素类 / ``.shp`` 交给内核里的一次
``arcpy.da.SearchCursor`` 取几何的 Esri JSON，再由 :func:`esri_geometry_to_geojson`
转成 GeoJSON（纯 Python，可在没有 arcpy 的机器上单测）。
"""

from __future__ import annotations

import base64
import html
import io
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .narrative import DISCLAIMER, ImageRef, LayerRef, ReportContent

logger = logging.getLogger(__name__)

#: 内嵌图的默认上限宽度（验收要求：超过就压到 1600px）。
DEFAULT_MAX_WIDTH = 1600

#: 每个图层的要素上限（报告是给人看的，不是数据交付）。
MAX_FEATURES = 400
#: 每个图层的坐标点预算（超出就按等间隔抽稀）。
MAX_COORDS = 20_000
#: 图层配色（按顺序取，多了循环）。
LAYER_COLORS = ("#1f6fb2", "#d1603d", "#2e8b57", "#7b5ea7", "#c9a227", "#3c8d9c")

#: 内核里取几何的代码模板：只做一件事——把几何的 Esri JSON 取出来。
#: 不用 ``FeaturesToJSON``：它要写文件、参数签名跨版本有差异，而这里只要几何。
_PROBE = """\
import json as _json
import arcpy as _arcpy

_path = {path}
_limit = {limit}
_geoms = []
try:
    with _arcpy.da.SearchCursor(_path, ["SHAPE@JSON"]) as _cur:
        for _row in _cur:
            if _row[0]:
                _geoms.append(_row[0])
            if len(_geoms) >= _limit:
                break
except Exception as _exc:
    set_result({{"error": str(_exc)[:300], "geoms": _geoms}})
else:
    set_result({{"error": "", "geoms": _geoms}})
"""


# ------------------------------------------------------------------ 图片
def prepare_image(
    path: str | Path,
    *,
    max_width: int = DEFAULT_MAX_WIDTH,
    quality: int = 82,
) -> tuple[str, bytes] | None:
    """读图 → 缩到 ``max_width`` → 重编码成能内嵌的 (mime, bytes)。

    有 alpha 通道的存 PNG（保留透明），其余存 JPEG（体积可控）。读不出来
    就返回 ``None``——一张坏图不该让整个成果包失败。
    """
    try:
        from PIL import Image
    except Exception:  # pragma: no cover - Pillow 是核心依赖，理论上不会缺
        logger.debug("Pillow 不可用，跳过图片内嵌")
        return None
    try:
        with Image.open(str(path)) as img:
            img.load()
            width, height = img.size
            if width > max_width and width > 0:
                ratio = max_width / float(width)
                img = img.resize((max_width, max(1, int(round(height * ratio)))), Image.LANCZOS)
            has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
            buffer = io.BytesIO()
            if has_alpha:
                img.convert("RGBA").save(buffer, format="PNG", optimize=True)
                return "image/png", buffer.getvalue()
            img.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
            return "image/jpeg", buffer.getvalue()
    except Exception as exc:
        logger.debug("图片内嵌失败 %s: %s", path, exc)
        return None


def _data_url(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# -------------------------------------------------------------- 几何 → GeoJSON
def _clean_point(point: Any) -> list[float] | None:
    """``[x, y, (z)]`` → ``[x, y]``；非有限数直接丢掉（JSON 里不能出现 NaN）。"""
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return [x, y]


def _clean_line(points: Any) -> list[list[float]]:
    out = [p for p in (_clean_point(item) for item in (points or [])) if p]
    return out if len(out) >= 2 else []


def _close_ring(ring: list[list[float]]) -> list[list[float]]:
    if len(ring) >= 3 and ring[0] != ring[-1]:
        ring = ring + [ring[0]]
    return ring


def _ring_area(ring: Sequence[Sequence[float]]) -> float:
    """鞋带公式（绝对值）：只用来比较哪个环当外圈。"""
    total = 0.0
    for i in range(len(ring) - 1):
        total += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
    return abs(total) / 2.0


def _point_in_ring(point: Sequence[float], ring: Sequence[Sequence[float]]) -> bool:
    """射线法。环不闭合也能算。"""
    x, y = point[0], point[1]
    inside = False
    count = len(ring)
    for i in range(count):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % count][0], ring[(i + 1) % count][1]
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
                inside = not inside
    return inside


def _rings_to_polygons(rings: list[list[list[float]]]) -> list[list[list[list[float]]]]:
    """Esri 的 rings → GeoJSON 的 polygons（外圈 + 内洞）。

    Esri 不保证环的先后顺序，也不保证"洞紧跟外圈"，所以用面积排序 + 点在环内
    判定：小环若落在某个大环里就是它的洞，否则自成一区。这样带洞的面不会被
    填成实心（研究区里的空洞一眼能看出来）。
    """
    usable: list[list[list[float]]] = []
    for ring in rings:
        cleaned = _clean_line(ring)
        if len(cleaned) < 3:
            continue
        closed = _close_ring(cleaned)
        if len(closed) >= 4:
            usable.append(closed)
    order = sorted(range(len(usable)), key=lambda i: _ring_area(usable[i]), reverse=True)
    outers: list[dict[str, Any]] = []
    for index in order:
        ring = usable[index]
        host = None
        for outer in reversed(outers):  # 从最小的外圈开始试，取最贴近的那个
            if _point_in_ring(ring[0], outer["ring"]):
                host = outer
                break
        if host is None:
            outers.append({"ring": ring, "holes": []})
        else:
            host["holes"].append(ring)
    return [[outer["ring"], *outer["holes"]] for outer in outers]


def esri_geometry_to_geojson(geom: Any) -> dict[str, Any] | None:
    """Esri JSON 几何（``rings``/``paths``/``points``/``x,y``）→ GeoJSON 几何。

    已经是 GeoJSON 的输入原样返回（``.geojson`` 文件走的就是这条路）。
    """
    if not isinstance(geom, dict):
        return None
    kind = geom.get("type")
    if kind in {"Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"}:
        return geom
    if "rings" in geom:
        polygons = _rings_to_polygons(list(geom.get("rings") or []))
        if not polygons:
            return None
        return {"type": "Polygon", "coordinates": polygons[0]} if len(polygons) == 1 else {
            "type": "MultiPolygon",
            "coordinates": polygons,
        }
    if "paths" in geom:
        lines = [line for line in (_clean_line(path) for path in geom.get("paths") or []) if line]
        if not lines:
            return None
        return {"type": "LineString", "coordinates": lines[0]} if len(lines) == 1 else {
            "type": "MultiLineString",
            "coordinates": lines,
        }
    if "points" in geom:
        points = [p for p in (_clean_point(item) for item in geom.get("points") or []) if p]
        if not points:
            return None
        return {"type": "MultiPoint", "coordinates": points}
    if "x" in geom and "y" in geom:
        point = _clean_point([geom.get("x"), geom.get("y")])
        return {"type": "Point", "coordinates": point} if point else None
    return None


def esri_geometries_to_feature_collection(geoms: Iterable[Any]) -> dict[str, Any]:
    features = []
    for raw in geoms:
        geom = raw
        if isinstance(raw, (str, bytes)):
            try:
                geom = json.loads(raw)
            except Exception:
                continue
        converted = esri_geometry_to_geojson(geom)
        if converted:
            features.append({"type": "Feature", "properties": {}, "geometry": converted})
    return {"type": "FeatureCollection", "features": features}


def _as_feature_collection(data: Any) -> dict[str, Any] | None:
    """把 ``.geojson`` 的内容整成 FeatureCollection（单要素/裸几何也接受）。"""
    if not isinstance(data, dict):
        return None
    if data.get("type") == "FeatureCollection" and isinstance(data.get("features"), list):
        return data
    if data.get("type") == "Feature":
        return {"type": "FeatureCollection", "features": [data]}
    geometry = esri_geometry_to_geojson(data)
    if geometry is not None:
        return {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": geometry}]}
    return None


# ------------------------------------------------------------------ 抽稀
def _iter_points(geometry: dict[str, Any]) -> Iterable[list[float]]:
    kind = geometry.get("type")
    coords = geometry.get("coordinates")
    if kind == "Point":
        yield coords
    elif kind in {"MultiPoint", "LineString"}:
        yield from coords or []
    elif kind in {"MultiLineString", "Polygon"}:
        for ring in coords or []:
            yield from ring or []
    elif kind == "MultiPolygon":
        for polygon in coords or []:
            for ring in polygon or []:
                yield from ring or []


def count_points(feature_collection: dict[str, Any]) -> int:
    total = 0
    for feature in feature_collection.get("features") or []:
        geometry = feature.get("geometry") or {}
        total += sum(1 for _ in _iter_points(geometry))
    return total


def _thin_line(points: list[list[float]], stride: int) -> list[list[float]]:
    if stride <= 1 or len(points) <= 2:
        return points
    kept = points[::stride]
    if kept[-1] != points[-1]:
        kept.append(points[-1])
    return kept if len(kept) >= 2 else points[:2]


def _thin_ring(ring: list[list[float]], stride: int) -> list[list[float]]:
    """抽稀闭合环：抽完不足 4 个点就不抽（否则面会被描成一条线）。"""
    thinned = _thin_line(ring, stride)
    return thinned if len(thinned) >= 4 else ring


def _thin_geometry(geometry: dict[str, Any], stride: int) -> dict[str, Any]:
    kind = geometry.get("type")
    coords = geometry.get("coordinates")
    if stride <= 1:
        return geometry
    if kind == "LineString":
        return {**geometry, "coordinates": _thin_line(coords, stride)}
    if kind == "MultiLineString":
        return {**geometry, "coordinates": [_thin_line(line, stride) for line in coords or []]}
    if kind == "Polygon":
        return {**geometry, "coordinates": [_thin_ring(ring, stride) for ring in coords or []]}
    if kind == "MultiPolygon":
        return {
            **geometry,
            "coordinates": [[_thin_ring(ring, stride) for ring in polygon or []] for polygon in coords or []],
        }
    return geometry


def thin_features(
    feature_collection: dict[str, Any],
    *,
    max_features: int = MAX_FEATURES,
    max_coords: int = MAX_COORDS,
) -> dict[str, Any]:
    """要素数与坐标点双重限额：先截要素，再按比例抽稀顶点。

    50 万个点的图层内联进 HTML 会变成几十 MB、手机上打不开——而报告只需要
    看得出形状。抽稀后形状仍然成立（等间隔保留 + 端点保留）。
    """
    features = list(feature_collection.get("features") or [])[:max_features]
    collection = {"type": "FeatureCollection", "features": [f for f in features if isinstance(f, dict)]}
    total = count_points(collection)
    if total > max_coords:
        stride = int(math.ceil(total / float(max_coords)))
        for feature in collection["features"]:
            geometry = feature.get("geometry")
            if isinstance(geometry, dict):
                feature["geometry"] = _thin_geometry(geometry, stride)
    return collection


def bbox_of(feature_collection: dict[str, Any]) -> list[float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for feature in feature_collection.get("features") or []:
        for point in _iter_points(feature.get("geometry") or {}):
            if point and len(point) >= 2:
                xs.append(point[0])
                ys.append(point[1])
    if not xs:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


# ------------------------------------------------------------------ 图层
def _exists_on_disk(path: Path) -> bool:
    """GDB 内的要素类在文件系统上 ``exists() == False``，要看库目录在不在。"""
    text = str(path)
    lowered = text.lower().replace("\\", "/")
    if ".gdb/" in lowered:
        return Path(text[: lowered.index(".gdb/") + 4]).exists()
    return path.exists()


def _resolve_input_path(target: Path, workspace: Path | None) -> Path:
    """相对路径先按当前目录找、再按工作区找；都没有就按工作区拼（内核要绝对路径）。

    为什么不能直接拼工作区：跑子进程/基准时当前目录里已经带了工作区前缀
    （``ws/output/x.gdb/街道``），再拼一次会变成 ``ws/ws/output/...``，
    内核只会回 "cannot open"（实测踩到，图层全丢）。
    """
    if target.is_absolute():
        return target
    candidates = [Path.cwd() / target]
    if workspace is not None:
        candidates.append(workspace / target)
    for candidate in candidates:
        if _exists_on_disk(candidate):
            return candidate.resolve()
    return candidates[-1].resolve()


def layer_probe_code(path: str, limit: int = MAX_FEATURES) -> str:
    """内核里取几何的代码（单独成函数，便于单测断言它只用安全 API）。"""
    return _PROBE.format(path=repr(str(path)), limit=int(limit))


def load_layer(
    path: str | Path,
    *,
    code_runner: Any = None,
    limit: int = MAX_FEATURES,
    workspace: str | Path | None = None,
) -> dict[str, Any] | None:
    """把一个矢量成果读成 FeatureCollection（读不出来返回 ``None``）。

    相对路径按 ``workspace`` 展开成绝对路径再交给内核：内核的工作目录是它自己的，
    拿相对路径去 ``SearchCursor`` 只会得到 "does not exist"。
    """
    target = _resolve_input_path(Path(str(path)), Path(workspace) if workspace else None)
    if target.suffix.lower() == ".geojson":
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("GeoJSON 读取失败 %s: %s", path, exc)
            return None
        return _as_feature_collection(data)
    if code_runner is None:
        return None
    try:
        outcome = code_runner.run(layer_probe_code(str(target), limit=limit), timeout=180)
    except Exception as exc:  # pragma: no cover - 内核不可用
        logger.debug("内核取几何失败 %s: %s", path, exc)
        return None
    payload = getattr(outcome, "result", None)
    if not getattr(outcome, "ok", False) or not isinstance(payload, dict):
        logger.warning("内核取几何失败 %s: %s", path, getattr(outcome, "error", ""))
        return None
    if payload.get("error"):
        # 内核说得出原因（路径不存在、坐标系损坏…）就别装成功：空图层会让人以为
        # “这层本来就没东西”，而实际上是一个能修的错。
        logger.warning("图层导出失败 %s: %s", path, payload.get("error"))
        return None
    return esri_geometries_to_feature_collection(payload.get("geoms") or [])


# ------------------------------------------------------------------ 渲染
_CSS = """
:root{--ink:#1b2430;--muted:#5c6a7a;--line:#dbe2ea;--accent:#155e9b;--paper:#ffffff}
*{box-sizing:border-box}
body{margin:0;background:#eef1f5;color:var(--ink);
  font:16px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}
main{max-width:1000px;margin:0 auto;padding:32px 24px 64px;background:var(--paper)}
header{padding-bottom:18px;border-bottom:3px solid var(--accent)}
header h1{margin:0 0 6px;font-size:26px;line-height:1.35}
header p{margin:0;color:var(--muted);font-size:14px}
section{margin:36px 0 0}
h2{font-size:19px;margin:0 0 12px;padding-left:10px;border-left:5px solid var(--accent)}
.lead{font-size:17px;margin:0 0 16px}
ul.numbers{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  gap:12px;padding:0;margin:0;list-style:none}
ul.numbers li{background:#f2f6fb;border-left:4px solid var(--accent);padding:10px 14px}
ul.numbers b{display:block;font-size:22px;line-height:1.3}
ul.numbers span{color:var(--muted);font-size:13px}
figure{margin:18px 0}
figure img{display:block;width:100%;height:auto;border:1px solid var(--line);background:#fff}
figcaption{margin-top:6px;color:var(--muted);font-size:14px}
.mapbox{border:1px solid var(--line);background:#fbfcfd;padding:12px}
.mapbox canvas{display:block;width:100%;height:auto;background:#fff}
.controls{display:flex;flex-wrap:wrap;gap:6px 18px;margin:0 0 10px;font-size:14px}
.controls label{display:inline-flex;align-items:center;gap:6px;cursor:pointer}
.swatch{width:12px;height:12px;border-radius:2px;display:inline-block}
.hint{color:var(--muted);font-size:13px;margin:8px 0 0}
ul.plain{margin:0;padding-left:22px}
ul.plain li{margin:4px 0}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;background:#f7f9fc}
td code{font-size:13px;color:var(--muted);word-break:break-all}
.empty{color:var(--muted);margin:0}
.disclaimer{margin-top:40px;padding:14px 18px;background:#fff5e6;border:1px solid #ffd79a;
  border-radius:4px;color:#8a5a00;font-size:14px}
footer{margin-top:18px;color:var(--muted);font-size:13px}
@media (max-width:640px){main{padding:20px 14px 48px}header h1{font-size:21px}}
@media print{body{background:#fff}main{max-width:none;padding:0}section{break-inside:avoid}}
"""

_JS = """
(function () {
  var layers = window.__LAYERS__ || [];
  var canvas = document.getElementById("map");
  if (!canvas || !layers.length || !canvas.getContext) return;
  var ctx = canvas.getContext("2d");
  var W = canvas.width, H = canvas.height, PAD = 26;

  function addCoords(geom, xs, ys) {
    if (!geom) return;
    var t = geom.type, c = geom.coordinates;
    function put(p) { if (p) { xs.push(p[0]); ys.push(p[1]); } }
    if (t === "Point") put(c);
    else if (t === "MultiPoint" || t === "LineString") (c || []).forEach(put);
    else if (t === "MultiLineString" || t === "Polygon") (c || []).forEach(function (r) { (r || []).forEach(put); });
    else if (t === "MultiPolygon") (c || []).forEach(function (poly) {
      (poly || []).forEach(function (r) { (r || []).forEach(put); });
    });
    else if (t === "GeometryCollection") (geom.geometries || []).forEach(function (g) { addCoords(g, xs, ys); });
  }

  var xs = [], ys = [];
  layers.forEach(function (layer) {
    ((layer.geojson || {}).features || []).forEach(function (f) { addCoords(f.geometry, xs, ys); });
  });
  if (!xs.length) return;
  var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
  var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
  var bw = Math.max(x1 - x0, 1e-9), bh = Math.max(y1 - y0, 1e-9);
  var scale = Math.min((W - 2 * PAD) / bw, (H - 2 * PAD) / bh);
  var ox = (W - bw * scale) / 2, oy = (H - bh * scale) / 2;

  function proj(p) {
    return [ox + (p[0] - x0) * scale, H - (oy + (p[1] - y0) * scale)];
  }

  // 一个多边形（外圈 + 洞）一次成路径，用 evenodd 填充，洞才是空的。
  // 逐圈 fill 会把洞又填实（实测：内圈叠色更深，看起来像两块面）。
  function polygonPath(rings) {
    ctx.beginPath();
    for (var i = 0; i < rings.length; i++) {
      var pts = rings[i];
      for (var k = 0; k < pts.length; k++) {
        var q = proj(pts[k]);
        if (k === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
      }
      ctx.closePath();
    }
  }

  function drawGeometry(geom, color) {
    if (!geom) return;
    var t = geom.type, c = geom.coordinates, i;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    if (t === "Polygon") {
      polygonPath(c);
      ctx.globalAlpha = 0.35;
      ctx.fill("evenodd");
      ctx.globalAlpha = 1;
      ctx.lineWidth = 1;
      ctx.stroke();
    } else if (t === "MultiPolygon") {
      for (i = 0; i < c.length; i++) {
        polygonPath(c[i]);
        ctx.globalAlpha = 0.35;
        ctx.fill("evenodd");
        ctx.globalAlpha = 1;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    } else if (t === "LineString" || t === "MultiLineString") {
      var lines = t === "LineString" ? [c] : c;
      ctx.lineWidth = 1.4;
      lines.forEach(function (line) {
        if (!line || line.length < 2) return;
        ctx.beginPath();
        for (var k = 0; k < line.length; k++) {
          var q = proj(line[k]);
          if (k === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
        }
        ctx.stroke();
      });
    } else if (t === "Point" || t === "MultiPoint") {
      var pts = t === "Point" ? [c] : c;
      pts.forEach(function (p) {
        var q = proj(p);
        ctx.beginPath();
        ctx.arc(q[0], q[1], 2.8, 0, Math.PI * 2);
        ctx.fill();
      });
    } else if (t === "GeometryCollection") {
      (geom.geometries || []).forEach(function (g) { drawGeometry(g, color); });
    }
  }

  function draw() {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.strokeRect(1, 1, W - 2, H - 2);
    layers.forEach(function (layer) {
      if (!layer.visible) return;
      ((layer.geojson || {}).features || []).forEach(function (f) {
        drawGeometry(f.geometry, layer.color);
      });
    });
  }

  var boxes = document.querySelectorAll("input[data-layer]");
  for (var n = 0; n < boxes.length; n++) {
    boxes[n].addEventListener("change", function (ev) {
      var index = parseInt(ev.target.getAttribute("data-layer"), 10);
      if (layers[index]) layers[index].visible = ev.target.checked;
      draw();
    });
  }
  draw();
})();
"""


def _format_extent(box: Sequence[float]) -> str:
    """坐标范围写给人看：地理坐标要 4 位小数，投影坐标 1 位就够。"""
    span = max(box[2] - box[0], box[3] - box[1])
    digits = 4 if span < 1 else (2 if span < 100 else 1)
    return f"{box[0]:.{digits}f}, {box[1]:.{digits}f} ~ {box[2]:.{digits}f}, {box[3]:.{digits}f}"


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _json_for_script(value: Any) -> str:
    """JSON 内联进 ``<script>``：把 ``</`` 转义，避免提前闭合标签。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _numbers_block(content: ReportContent) -> str:
    if not content.key_numbers:
        return ""
    items = "".join(
        f"<li><b>{_esc(item.value)}</b>{_esc(item.label)}"
        + (f"<span>{_esc(item.note)}</span>" if item.note else "")
        + "</li>"
        for item in content.key_numbers
    )
    return f'<ul class="numbers">{items}</ul>'


def _images_block(images: Sequence[tuple[ImageRef, str, bytes]]) -> str:
    if not images:
        return '<p class="empty">本次没有可内嵌的成果图。</p>'
    figures = []
    for ref, mime, data in images:
        figures.append(
            f'<figure><img alt="{_esc(ref.caption)}" src="{_data_url(mime, data)}">'
            f"<figcaption>{_esc(ref.caption)}</figcaption></figure>"
        )
    return "".join(figures)


def _layers_block(
    layers: Sequence[tuple[LayerRef, dict[str, Any]]],
    notes: Sequence[str],
) -> tuple[str, list[dict[str, Any]]]:
    if not layers:
        body = '<p class="empty">本次没有可内嵌的矢量图层（没有矢量成果，或图层导出失败）。</p>'
        if notes:
            body += '<p class="hint">' + "；".join(_esc(n) for n in notes) + "</p>"
        return f"<section><h2>图层</h2>{body}</section>", []

    controls = []
    payload = []
    for index, (ref, collection) in enumerate(layers):
        color = LAYER_COLORS[index % len(LAYER_COLORS)]
        count = len(collection.get("features") or [])
        box = bbox_of(collection)
        extent = f"，范围 {_format_extent(box)}" if box else ""
        controls.append(
            f'<label><input type="checkbox" data-layer="{index}" checked>'
            f'<span class="swatch" style="background:{color}"></span>'
            f'{_esc(ref.name)}<span class="hint">（{count} 个要素{_esc(extent)}）</span></label>'
        )
        payload.append(
            {
                "name": ref.name,
                "color": color,
                "visible": True,
                "geojson": collection,
            }
        )
    notes_html = ('<p class="hint">' + "；".join(_esc(n) for n in notes) + "</p>") if notes else ""
    body = (
        f'<div class="mapbox"><div class="controls">{"".join(controls)}</div>'
        '<canvas id="map" width="1200" height="720" role="img" aria-label="成果图层示意"></canvas>'
        '<p class="hint">勾选框开关图层；图面按坐标范围自动缩放，不做底图叠加，断网也能看。</p>'
        f"{notes_html}</div>"
    )
    return f"<section><h2>图层</h2>{body}</section>", payload


def _methods_block(content: ReportContent) -> str:
    if not content.methods:
        return '<section><h2>方法</h2><p class="empty">本次没有记录方法与配方。</p></section>'
    items = "".join(f"<li>{_esc(line)}</li>" for line in content.methods)
    return f'<section><h2>方法</h2><ul class="plain">{items}</ul></section>'


def _sources_block(content: ReportContent) -> str:
    if not content.sources:
        return '<section><h2>数据来源</h2><p class="empty">数据目录里没有登记到本次用到的数据。</p></section>'
    rows = "".join(
        f"<tr><td>{_esc(row.name)}</td><td><code>{_esc(row.path)}</code></td>"
        f"<td>{_esc(row.source)}</td><td>{_esc(row.license)}</td><td>{_esc(row.acquired_at)}</td></tr>"
        for row in content.sources
    )
    return (
        "<section><h2>数据来源</h2><table><thead><tr>"
        "<th>数据</th><th>位置</th><th>来源</th><th>许可</th><th>取得时间</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></section>"
    )


def _verify_block(content: ReportContent) -> str:
    if not content.verifications:
        return '<section><h2>验证</h2><p class="empty">本次没有留下验收记录。</p></section>'
    items = "".join(f"<li>{_esc(line)}</li>" for line in content.verifications)
    return f'<section><h2>验证</h2><ul class="plain">{items}</ul></section>'


def render_html(
    content: ReportContent,
    *,
    images: Sequence[tuple[ImageRef, str, bytes]] = (),
    layers: Sequence[tuple[LayerRef, dict[str, Any]]] = (),
    notes: Sequence[str] = (),
) -> str:
    """拼出完整的单文件 HTML（纯函数：给定内容与内嵌资源即得字符串）。"""
    layers_section, payload = _layers_block(layers, notes)
    script = (
        f"<script>window.__LAYERS__ = {_json_for_script(payload)};</script>\n<script>{_JS}</script>"
        if payload
        else ""
    )
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_esc(content.title)}</title>",
            f"<style>{_CSS}</style>",
            "</head>",
            "<body>",
            "<main>",
            "<header>",
            f"<h1>{_esc(content.title)}</h1>",
            f'<p>生成时间 {_esc(content.generated_at)}　·　单文件成果包（断网可看）</p>',
            "</header>",
            f'<section><h2>结论</h2><p class="lead">{_esc(content.conclusion)}</p>{_numbers_block(content)}</section>',
            f"<section><h2>成果图</h2>{_images_block(images)}</section>",
            layers_section,
            _methods_block(content),
            _sources_block(content),
            _verify_block(content),
            f'<div class="disclaimer">{_esc(DISCLAIMER)}</div>',
            "<footer>本文件由 GIS Agent 自动生成，所需内容全部内嵌，不发起任何网络请求。</footer>",
            "</main>",
            script,
            "</body>",
            "</html>",
            "",
        ]
    )


__all__ = [
    "LAYER_COLORS",
    "MAX_COORDS",
    "MAX_FEATURES",
    "bbox_of",
    "count_points",
    "esri_geometries_to_feature_collection",
    "esri_geometry_to_geojson",
    "layer_probe_code",
    "load_layer",
    "prepare_image",
    "render_html",
    "thin_features",
]

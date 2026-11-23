# -*- coding: utf-8 -*-
"""保拓扑几何简化：共享边界只简化一次（取代等间隔 stride 抽稀）。

背景（DAIR-368）：预览/报告原来用 ``points[::stride]`` 抽稀顶点。每环各自
按相位取样有两个必然后果：

1. 相邻要素的**共享边界被不同相位采样** → 两侧对不上 → 缝隙；
2. 环丢掉 75%~90% 顶点后退化成一串**长直弦** → 横穿要素的直线与三角形空洞。

这里改成 TopoJSON 那一套做法：

- 顶点先吸附到统一格网（同一位置 → 同一个点，吸收浮点噪声）；
- 按「这条边被哪些环共用」把每条环切成**弧**（arc），使用签名变化处就是切点；
- 每条唯一的弧只做**一次** Douglas–Peucker；相邻要素正/反向复用同一份顶点
  → 共享边界逐点相同，天然严丝合缝；弦与原曲线的偏差恒 ≤ 容差；
- 闭合弧至少保留 4 点；环退化到不足 4 点时回退原环（宁可重一点也不抽空几何）。

本模块**只依赖标准库**，因为 ArcGIS 内核对 ``arcpy.da.SearchCursor`` 的几何做
同一套简化时，是用 ``exec`` 直接加载本文件的源码（见
``report/html_pack.layer_probe_code``）：内核先减点再回传，与主进程用的是**同一
份**算法，不会两边行为漂移。
"""

import math

#: 格网精度 = extent 跨度 × 该比例：只用来吸收浮点噪声，不用来简化。
PRECISION_RATIO = 1.0 / 4_000_000.0
#: 没有顶点预算时的起步容差：参考画布的一半像素。
REFERENCE_PIXELS = 1200.0
PIXEL_TOLERANCE = 0.5
#: 预算迭代的轮数上限（每轮全量化简一次，别无限逼近）。
MAX_PASSES = 10
#: 环（闭合路径）至少保留的点数：4 点 = 3 个不同顶点 + 闭合点。
MIN_RING_POINTS = 4
#: 退化环回退弧的最多轮数：回退只会加顶点，第二轮就能确认是否收敛。
MAX_ROLLBACK_ROUNDS = 2
#: 没有使用者的边（只在函数路径里出现，不参与连边）。
_NO_USERS = frozenset()
#: DP 的**工作量**上限：每条弧每顶点允许多少次「点到弦距离」计算。
#:
#: DP 最坏是 O(n²)：锯齿状顶点（等振幅来回摆动）上每一刀只能切掉一个点，
#: 5 千点的锯齿环实测要 2.5s，100 万点整层要 8 分钟——预览预算 34s 必然超时。
#: 正常边界的代价是 O(n log n)：每层扫一遍、深 log2(n)（100 万点也就 20），
#: 所以 24 倍有足够余量；超了就是「这个容差在这条弧上压不动」。
#:
#: 定额按**每条弧自己的长度**算，不按全层总点数：锯齿的二次爆炸只发生在单条弧
#: 内部，按弧定额既能给出自然的 log 余量，又能让锯齿弧早早被叫停（一条 5 千
#: 点的锯齿弧约 12 万次就算停）。
DP_WORK_PER_POINT = 24
#: 工作量下限：小弧（几十个点）给足余量，别被常数抖动误判成停滞。
MIN_DP_WORK = 2_048


# ------------------------------------------------------------------ 几何遍历
def _is_point(coords):
    return (
        isinstance(coords, (list, tuple))
        and len(coords) >= 2
        and isinstance(coords[0], (int, float))
        and isinstance(coords[1], (int, float))
    )


def _clean_path(path):
    return [list(point[:2]) for point in path or () if _is_point(point)]


def _is_closed_path(path):
    """源环首尾是否重合（GeoJSON 要求重合，但真实数据里不一定）。"""
    if not path or len(path) < 3:
        return False
    first, last = path[0], path[-1]
    return _is_point(first) and _is_point(last) and first[0] == last[0] and first[1] == last[1]


def _iter_geometry_paths(geometry):
    """取几何里所有"线/环"坐标序列（点类几何不参与简化）。"""
    if not isinstance(geometry, dict):
        return
    kind = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return
    if kind == "LineString":
        yield coords
    elif kind in ("MultiLineString", "Polygon"):
        for path in coords:
            yield path
    elif kind == "MultiPolygon":
        for polygon in coords:
            for path in polygon or ():
                yield path


def count_geometry_points(geometry):
    total = 0
    for path in _iter_geometry_paths(geometry):
        total += len(path or ())
    return total


def _count_path_points(paths):
    return sum(len(path) for path in paths)


def _bbox(paths):
    xs = []
    ys = []
    for path in paths:
        for point in path:
            xs.append(point[0])
            ys.append(point[1])
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


# ------------------------------------------------------------------ 模板
def _add_path(paths, flags, coords, closed):
    paths.append(_clean_path(coords))
    flags.append(bool(closed))
    return len(paths) - 1


def _geojson_template(geometry, paths, flags):
    """GeoJSON 几何 → 模板（只在叶子处登记 path 下标，结构原样保留）。

    ``("path", i)`` 指向 ``paths[i]``；``("seq", [...])`` 是坐标数组的一层嵌套。
    面的环按闭合路径处理：源数据里有环首尾不重合的情况（实测 中国_省 28 个），
    当折线展开会让共享边一半对不上。
    """
    if not isinstance(geometry, dict):
        return None
    kind = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords or kind in (None, "Point", "MultiPoint", "GeometryCollection"):
        return None
    if kind == "LineString":
        return ("path", _add_path(paths, flags, coords, False))
    if kind == "MultiLineString":
        return ("seq", [("path", _add_path(paths, flags, path, False)) for path in coords])
    if kind == "Polygon":
        return ("seq", [("path", _add_path(paths, flags, ring, True)) for ring in coords])
    if kind == "MultiPolygon":
        return (
            "seq",
            [
                ("seq", [("path", _add_path(paths, flags, ring, True)) for ring in polygon or ()])
                for polygon in coords
            ],
        )
    return None


def _esri_template(geometry, paths, flags):
    """Esri JSON 几何（rings/paths）→ 模板；点类几何返回 None（不简化）。

    不保留 ``spatialReference`` 之类的附属键：主进程的转换只读 rings/paths，
    丢掉它就是本模块要的「少传冗余字节」。
    """
    if not isinstance(geometry, dict):
        return None
    rings = geometry.get("rings")
    if isinstance(rings, (list, tuple)) and rings:
        return ("rings", [("path", _add_path(paths, flags, ring, True)) for ring in rings])
    segments = geometry.get("paths")
    if isinstance(segments, (list, tuple)) and segments:
        return ("paths", [("path", _add_path(paths, flags, segment, False)) for segment in segments])
    return None


def _rebuild_tree(node, out_paths):
    if node[0] == "path":
        return out_paths[node[1]]
    return [_rebuild_tree(child, out_paths) for child in node[1]]


def _rebuild_esri(node, out_paths):
    if node[0] == "rings":
        return {"rings": _rebuild_tree(node, out_paths)}
    return {"paths": _rebuild_tree(node, out_paths)}


# ------------------------------------------------------------------ DP 简化
def _douglas_peucker(points, tolerance, work_limit=None):
    """迭代版 Douglas–Peucker：返回 ``(保留的下标, 是否因工作量上限中断, 用掉的工作量)``。

    ``work_limit`` 是「点到弦距离」计算的次数上限（见 :data:`DP_WORK_PER_POINT`）。
    超了就停下并报 ``stalled=True``：这个容差在这条弧上压不动（锯齿状顶点会把
    DP 拖成 O(n²)），调用方应当放粗容差重来，而不是硬等。
    """
    total = len(points)
    if total <= 2 or tolerance <= 0:
        return list(range(total)), False, 0
    keep = bytearray(total)
    keep[0] = 1
    keep[total - 1] = 1
    stack = [(0, total - 1)]
    tol2 = tolerance * tolerance
    work = 0
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        work += end - start
        if work_limit is not None and work > work_limit:
            return [], True, work
        ax, ay = points[start][0], points[start][1]
        bx, by = points[end][0], points[end][1]
        dx = bx - ax
        dy = by - ay
        seg2 = dx * dx + dy * dy
        best = -1.0
        best_index = -1
        if seg2 <= 0.0:
            # 闭合弧（首尾同一点）：基线退化成一个点，按距离该点的远近取最远点。
            for i in range(start + 1, end):
                px, py = points[i][0], points[i][1]
                d2 = (px - ax) * (px - ax) + (py - ay) * (py - ay)
                if d2 > best:
                    best = d2
                    best_index = i
        else:
            for i in range(start + 1, end):
                px, py = points[i][0], points[i][1]
                t = ((px - ax) * dx + (py - ay) * dy) / seg2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                qx = ax + t * dx
                qy = ay + t * dy
                d2 = (px - qx) * (px - qx) + (py - qy) * (py - qy)
                if d2 > best:
                    best = d2
                    best_index = i
        if best_index >= 0 and best > tol2:
            keep[best_index] = 1
            stack.append((start, best_index))
            stack.append((best_index, end))
    return [i for i in range(total) if keep[i]], False, work


def _simplify_arc(points, tolerance, work_limit=None):
    """简化一条弧：端点必留；闭合弧至少 4 点（不足就把容差减半重试）。

    返回 ``(保留的点, 是否中断, 用掉的工作量)``。中断时保留的点无意义（调用方
    会整轮丢弃），直接回原弧只是为了让返回值还能用。
    """
    closed = len(points) > 1 and points[0] == points[-1]
    limit = tolerance
    indices, stalled, work = _douglas_peucker(points, limit, work_limit)
    if stalled:
        return list(points), True, work
    kept = [points[i] for i in indices]
    if closed:
        for _ in range(4):
            if len(kept) >= MIN_RING_POINTS:
                break
            limit /= 2.0
            indices, stalled, more = _douglas_peucker(points, limit, work_limit)
            work += more
            if stalled:
                return list(points), True, work
            kept = [points[i] for i in indices]
        if len(kept) < MIN_RING_POINTS:
            return list(points), False, work
    elif len(kept) < 2:
        return list(points), False, work
    return kept, False, work


def _slice_indices(count, cuts, closed):
    """按切点把路径切成弧的下标序列。``cuts`` 已升序、互不相同。"""
    if closed:
        if not cuts:
            return [list(range(count)) + [0]]  # 整圈一条闭合弧
        pieces = []
        for position, start in enumerate(cuts):
            end = cuts[(position + 1) % len(cuts)]
            seq = [start]
            index = start
            while True:
                index = (index + 1) % count
                seq.append(index)
                if index == end or len(seq) > count:
                    break
            pieces.append(seq)
        return pieces
    bounds = [0] + list(cuts) + [count - 1]
    unique = []
    for index in sorted(bounds):
        if not unique or unique[-1] != index:
            unique.append(index)
    return [list(range(a, b + 1)) for a, b in zip(unique, unique[1:]) if b > a]


# ------------------------------------------------------------------ 拓扑
class Plan(object):
    """一次拓扑分析的产物：格网吸附后的路径 + 唯一弧表 + 每条路径的弧序列。

    同一条弧只存一份顶点：相邻要素重建时正/反向复用，共享边界逐点相同。
    """

    def __init__(self, paths, closed_flags=None, precision=None):
        self.original = [list(path or ()) for path in paths]
        box = _bbox(self.original)
        self.usable = box is not None
        if not self.usable:
            return
        span = max(box[2] - box[0], box[3] - box[1])
        if precision is None or precision <= 0:
            precision = span * PRECISION_RATIO
        self.precision = float(precision) if precision > 0 else 1e-12
        self.span = span if span > 0 else 1.0
        self.decimals = max(0, min(12, int(math.ceil(-math.log10(self.precision))) + 2))
        self._snap(box[0], box[1], closed_flags)
        self._build_arcs()

    # -- 1. 吸附到格网：同一位置的顶点得到同一个 key（浮点噪声消灭掉）
    def _snap(self, origin_x, origin_y, closed_flags):
        inv = 1.0 / self.precision
        decimals = self.decimals
        self.keys = []
        self.coords = []
        self.ring = []
        self.closed = []
        self.key_coords = {}
        for index, path in enumerate(self.original):
            want_ring = _is_closed_path(path) if closed_flags is None else bool(closed_flags[index])
            keys = []
            coords = []
            last = None
            for point in path:
                if not _is_point(point):
                    continue
                key = (
                    int(round((point[0] - origin_x) * inv)),
                    int(round((point[1] - origin_y) * inv)),
                )
                if key == last:
                    continue
                last = key
                keys.append(key)
                if key not in self.key_coords:
                    self.key_coords[key] = [
                        round(origin_x + key[0] * self.precision, decimals),
                        round(origin_y + key[1] * self.precision, decimals),
                    ]
                coords.append(self.key_coords[key])
            # 环先首尾去重，再按“闭合环”参与连边（源数据里有环首尾不重合）。
            if len(keys) > 1 and keys[0] == keys[-1]:
                keys.pop()
                coords.pop()
            self.keys.append(keys)
            self.coords.append(coords)
            self.ring.append(want_ring)
            # 只对“真的是环”的路径按闭合拓扑处理（撑不到 3 个点就不是面了）。
            self.closed.append(want_ring and len(keys) >= 3)

    # -- 2. 找共享边 → 按“使用签名”切弧
    def _build_arcs(self):
        edge_users = {}
        for index, keys in enumerate(self.keys):
            count = len(keys)
            if count < 2:
                continue
            ring = self.closed[index]
            for i in range(count if ring else count - 1):
                a = keys[i]
                b = keys[(i + 1) % count]
                if a == b:
                    continue
                edge = (a, b) if a < b else (b, a)
                users = edge_users.get(edge)
                if users is None:
                    edge_users[edge] = {index}
                else:
                    users.add(index)

        def signature(keys, index, count):
            """这条边的「使用签名」：被哪几条路径共用。

            直接返回 set（按内容比较）而不是每次 ``frozenset(...)``：100 万条边
            每边新建一个 frozenset 要白烧半秒，而这里只做相邻比较。
            """
            a = keys[index]
            b = keys[(index + 1) % count]
            edge = (a, b) if a < b else (b, a)
            return edge_users.get(edge, _NO_USERS)

        self.arcs = {}
        self.arc_of_path = []
        for index, keys in enumerate(self.keys):
            count = len(keys)
            entry = []
            self.arc_of_path.append(entry)
            if count < 2:
                continue
            ring = self.closed[index]
            if ring:
                signatures = [signature(keys, i, count) for i in range(count)]
                cuts = [i for i in range(count) if signatures[i] != signatures[i - 1]]
            else:
                signatures = [signature(keys, i, count) for i in range(count - 1)]
                cuts = [i for i in range(1, count - 1) if signatures[i] != signatures[i - 1]]
            for piece in _slice_indices(count, cuts, ring):
                seq = tuple(keys[i] for i in piece)
                if len(seq) < 2:
                    continue
                reverse = tuple(reversed(seq)) < seq
                canon = tuple(reversed(seq)) if reverse else seq
                entry.append((canon, reverse))
                if canon not in self.arcs:
                    self.arcs[canon] = [self.key_coords[key] for key in canon]

    def work_limit(self, count):
        """一条弧一轮简化的工作量上限：见 :data:`DP_WORK_PER_POINT`。"""
        return max(MIN_DP_WORK, DP_WORK_PER_POINT * int(count))

    # -- 3. 按容差简化所有弧，重建每条路径
    def simplify(self, tolerance):
        """按容差简化所有弧。返回 ``(路径表, 是否因工作量上限中断)``。

        中断时路径表是空的：这一轮的结果不可用，调用方要放粗容差重来
        （:func:`_fit` 就是干这个的），别拿半成品去重建几何。某条弧一报中断就
        整轮放弃，剩下的弧再算也是白烧。
        """
        simplified = {}
        for canon, points in self.arcs.items():
            piece, stalled, _used = _simplify_arc(
                points, tolerance, self.work_limit(len(points))
            )
            if stalled:
                return [], True
            simplified[canon] = piece
        out = []
        degenerate: list[int] = []
        # 环被简化到画不出来时，不能只回退“这个环自己的顶点”：邻居用的仍是
        # 简化后的共享弧，两侧就对不上（缝隙重现，DAIR-373）。正确做法是把
        # **这个环用到的弧整条回退**，邻居跟着一起退回原弧 → 共享边仍逐点相同。
        # 回退只会让环变密、不会让别的环再退化，所以一两轮就收敛；仍画不出来
        # 才放弃拓扑一致、退回自己的原环（保“有面可画”）。
        for _ in range(MAX_ROLLBACK_ROUNDS):
            out, degenerate = self._rebuild(simplified)
            if not degenerate:
                break
            reverted = False
            for index in degenerate:
                for canon, _reverse in self.arc_of_path[index]:
                    if simplified[canon] is not self.arcs[canon]:
                        simplified[canon] = self.arcs[canon]
                        reverted = True
            if not reverted:
                break
        for index in degenerate:
            if self.ring[index]:
                out[index] = _closed_copy(self.original[index])
            else:
                out[index] = _clean_path(self.original[index])
        return out, False

    def _rebuild(self, simplified):
        """按弧表重建每条路径，返 ``(路径表, 退化路径下标)``。"""
        out = []
        degenerate = []
        for index, entry in enumerate(self.arc_of_path):
            coords = []
            for canon, reverse in entry:
                piece = simplified[canon]
                if reverse:
                    piece = piece[::-1]
                start = 1 if coords and piece and coords[-1] == piece[0] else 0
                coords.extend(piece[start:])
            if self.ring[index]:
                if coords and coords[0] != coords[-1]:
                    coords.append(coords[0])
                if len(coords) < MIN_RING_POINTS:
                    degenerate.append(index)
            elif len(coords) < 2:
                degenerate.append(index)
            out.append(coords)
        return out, degenerate

    def default_tolerance(self):
        return self.span / REFERENCE_PIXELS * PIXEL_TOLERANCE


def _closed_copy(path):
    coords = _clean_path(path)
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


# ------------------------------------------------------------------ 预算
def _fit(plan, tolerance, max_points):
    """选定容差：给了 ``tolerance`` 就一次到位；只给预算就从像素尺度起步。

    返回 ``(容差, 简化后的路径表, 轮数, 是否中断)``。起点是「半个像素」
    （:meth:`Plan.default_tolerance`，由 extent 推导，见 :data:`REFERENCE_PIXELS`）。

    两件事一起决定收敛方式：

    - **超预算** → 按点数比例加粗容差（预算只是上限，不是必须用完的目标）；
    - **压不动**（``stalled``，锯齿状顶点把 DP 拖成 O(n²)）→ 容差直接 ×4，
      因为这时候「点数没降下来」不是容差不够大，而是这一档根本算不完。

    只在**没有任何一轮停滞**时做一次「往回找一档」：停滞说明细容差这条路
    走不通，回头再试细一点只会再撞一次 O(n²)。
    """
    if tolerance is not None and tolerance > 0:
        limit = float(tolerance)
        out, stalled = plan.simplify(limit)
        if stalled:  # 显式容差也压不动：宁可原样交付，也不能交空几何
            return limit, [_clean_path(path) for path in plan.original], 1, True
        return limit, out, 1, False
    base = plan.default_tolerance()
    if not max_points or max_points <= 0:
        out, stalled = plan.simplify(base)
        if stalled:
            return base, [_clean_path(path) for path in plan.original], 1, True
        return base, out, 1, False
    limit = base
    out, stalled = plan.simplify(limit)
    points = 0 if stalled else _count_path_points(out)
    passes = 1
    stalled_ever = stalled
    previous = None
    while (stalled or points > max_points) and passes < MAX_PASSES:
        if stalled:
            limit *= 4.0
        else:
            factor = (points / float(max_points)) ** 0.75
            if previous is not None and points > previous * 0.98:
                # 容差加大了点数却几乎没降（阶梯状折线）：一脚踩深点，别磨。
                factor = max(factor, 4.0)
            limit *= max(1.2, min(8.0, factor))
            previous = points
        out, stalled = plan.simplify(limit)
        stalled_ever = stalled_ever or stalled
        points = 0 if stalled else _count_path_points(out)
        passes += 1
    if stalled:
        # 走到轮数上限还是压不动：原样交付（调用方还有 5MB 那道闸兜底）。
        return limit, [_clean_path(path) for path in plan.original], passes, True
    if not stalled_ever and points < max_points * 0.5 and passes < MAX_PASSES:
        # 一下子压太狠了：往回找一档，形状能多留就多留。
        finer = limit * 0.55
        candidate, candidate_stalled = plan.simplify(finer)
        passes += 1
        if not candidate_stalled and _count_path_points(candidate) <= max_points:
            return finer, candidate, passes, False
    return limit, out, passes, False


def _empty_stats(input_points=0):
    return {
        "tolerance": 0.0,
        "precision": 0.0,
        "input_points": input_points,
        "output_points": input_points,
        "arcs": 0,
        "passes": 0,
        "stalled": False,
    }


def _stats(plan, tolerance, passes, input_points, out_paths, stalled=False):
    return {
        "tolerance": tolerance,
        "precision": plan.precision,
        "input_points": input_points,
        "output_points": _count_path_points(out_paths),
        "arcs": len(plan.arcs),
        "passes": passes,
        "stalled": bool(stalled),
    }


# ------------------------------------------------------------------ 对外入口
def simplify_feature_collection(collection, *, tolerance=None, max_points=None):
    """保拓扑简化一个 GeoJSON FeatureCollection。

    返回 ``(新集合, 统计)``。只改几何的顶点密度：环/部件数量与属性不动；
    点类几何原样保留。共享边界两侧逐点相同（同一条弧只简化一次）。
    """
    features = [feature for feature in collection.get("features") or () if isinstance(feature, dict)]
    paths = []
    flags = []
    templates = [
        _geojson_template(feature.get("geometry") or {}, paths, flags) for feature in features
    ]
    if not paths:
        return collection, _empty_stats()
    input_points = _count_path_points(paths)
    plan = Plan(paths, flags)
    if not plan.usable:
        return collection, _empty_stats(input_points)
    limit, out_paths, passes, stalled = _fit(plan, tolerance, max_points)
    rebuilt = []
    for feature, template in zip(features, templates):
        if template is None:
            rebuilt.append(feature)
            continue
        geometry = dict(feature.get("geometry") or {})
        geometry["coordinates"] = _rebuild_tree(template, out_paths)
        rebuilt.append({**feature, "geometry": geometry})
    result = dict(collection)
    result["features"] = rebuilt
    return result, _stats(plan, limit, passes, input_points, out_paths, stalled)


def simplify_esri_geometries(geometries, *, tolerance=None, max_points=None):
    """保拓扑简化一批 Esri JSON 几何（``rings``/``paths``）。

    内核侧入口：ArcGIS 里读出来的几何先在**内核里**减点再回传，主进程不再收
    全精度大 JSON（DAIR-368 问题 3）。返回 ``(新几何列表, 统计)``；认不出的
    几何（点／已是字符串）原样留在原位。
    """
    items = list(geometries)
    paths = []
    flags = []
    templates = []
    for geometry in items:
        if isinstance(geometry, dict):
            templates.append(_esri_template(geometry, paths, flags))
        else:
            templates.append(None)
    if not paths:
        return items, _empty_stats()
    input_points = _count_path_points(paths)
    plan = Plan(paths, flags)
    if not plan.usable:
        return items, _empty_stats(input_points)
    limit, out_paths, passes, stalled = _fit(plan, tolerance, max_points)
    rebuilt = []
    for geometry, template in zip(items, templates):
        if template is None:
            rebuilt.append(geometry)
        else:
            rebuilt.append(_rebuild_esri(template, out_paths))
    return rebuilt, _stats(plan, limit, passes, input_points, out_paths, stalled)

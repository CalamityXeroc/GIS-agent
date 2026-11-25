/**
 * 地图视口 → 预览查询框（DAIR-386）。
 *
 * 后端 `/api/catalog/preview` 支持 `bbox`：只读视口内的要素，并回
 * `total_in_view` / `coverage`。这里决定「什么时候算换了个视口」。
 */

export type Bbox = [number, number, number, number]

export interface ViewportBounds {
  west: number
  south: number
  east: number
  north: number
}

/**
 * 把视口**外接**到 zoom 的网格上，网格边长 = 360 / 2^(zoom+2)（约 1/4 个瓦片）。
 *
 * 为什么要量化：平移几像素就换一个 key，每次 `moveend` 都会发一轮请求（几个
 * 图层就是几条）。对齐网格后小幅平移落在同一格里，React Query 直接用缓存。
 * 向外取整（floor 西/南、ceil 东/北）保证量化后的框**包住**真实视口——宁可多读
 * 一点，也不能把视口边缘的要素筛掉。
 */
export function quantizeViewport(bounds: ViewportBounds, zoom: number): Bbox {
  const level = Math.max(0, Math.min(22, Math.floor(zoom)))
  const span = 360 / Math.pow(2, level + 2)
  const snap = (value: number, up: boolean) =>
    (up ? Math.ceil(value / span) : Math.floor(value / span)) * span
  const clamp = (value: number, min: number, max: number) =>
    Math.min(max, Math.max(min, value))
  const west = clamp(snap(bounds.west, false), -180, 180)
  const south = clamp(snap(bounds.south, false), -90, 90)
  const east = clamp(snap(bounds.east, true), -180, 180)
  const north = clamp(snap(bounds.north, true), -90, 90)
  // 世界视图下东西各夹到 ±180：量化后框不能塌成一条线。
  return [west, south, east > west ? east : west + span, north > south ? north : south + span]
}

/** 视口指纹：只用来判断「是不是同一个框」，精度到 1e-5 度（约 1 米）。 */
export function bboxKey(bbox: Bbox | null): string {
  if (!bbox) return ''
  return bbox.map((value) => value.toFixed(5)).join(',')
}

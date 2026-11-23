import proj4 from 'proj4'

/**
 * 本项目数据里最常见的两个投影（CGCS2000 3 度带 / 6 度带高斯克吕格）。
 *
 * 注意 ``+k=1``：EPSG 4548 / 4508 是「3 度带 / 6 度带高斯克吕格（CM 带）」，比例因子是 1；
 * 用 0.9996 的是 EPSG 4547 / 4507（zone 版），不要照抄 UTM 的参数。
 * 参数可用 pyproj 核对：``CRS(4548).to_proj4()``。
 */
proj4.defs(
  'EPSG:4548',
  '+proj=tmerc +lat_0=0 +lon_0=117 +k=1 +x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs'
)
proj4.defs(
  'EPSG:4508',
  '+proj=tmerc +lat_0=0 +lon_0=111 +k=1 +x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs'
)

export const LAYER_CRS: Record<number, string> = {
  4326: 'EPSG:4326',
  3857: 'EPSG:3857',
  4548: 'EPSG:4548',
  4508: 'EPSG:4508',
}

export type BBox = [number, number, number, number]

type Position = number[]

/** 坐标数组嵌套层数不固定（Point/Line/Polygon/Multi），按「第一个元素是数字」判到叶子。 */
function mapCoordinates(coords: unknown, fn: (position: Position) => Position): unknown {
  if (!Array.isArray(coords)) return coords
  if (coords.length > 0 && typeof coords[0] === 'number') return fn(coords as Position)
  return coords.map((child) => mapCoordinates(child, fn))
}

export interface GeoJSONFeature {
  type: 'Feature'
  properties: Record<string, unknown>
  geometry: { type: string; coordinates?: unknown } | null
}

export interface GeoJSONCollection {
  type: 'FeatureCollection'
  features: GeoJSONFeature[]
}

/**
 * 把一整个 FeatureCollection 从 ``wkid`` 转到经纬度。
 * 4326 直接返回原对象（避免无谓复制）；其余走 proj4，输出 [lon, lat]。
 */
export function toWGS84(
  collection: GeoJSONCollection,
  wkid: number | null
): { data: GeoJSONCollection; transformed: boolean } {
  const target = wkid ? LAYER_CRS[wkid] : 'EPSG:4326'
  if (!wkid || wkid === 4326 || !target || !proj4.defs(target)) {
    return { data: collection, transformed: false }
  }
  const features = collection.features.map((feature) => {
    if (!feature.geometry) return feature
    const coordinates = mapCoordinates(feature.geometry.coordinates, (position) => {
      const [x, y, ...rest] = position
      const [lon, lat] = proj4(target, 'EPSG:4326', [x, y])
      return [lon, lat, ...rest]
    })
    return { ...feature, geometry: { ...feature.geometry, coordinates } }
  })
  return { data: { type: 'FeatureCollection', features }, transformed: true }
}

export function bboxOf(collection: GeoJSONCollection): BBox | null {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  const visit = (coords: unknown) => {
    if (!Array.isArray(coords)) return
    if (coords.length > 1 && typeof coords[0] === 'number' && typeof coords[1] === 'number') {
      const [x, y] = coords as [number, number]
      if (!Number.isFinite(x) || !Number.isFinite(y)) return
      if (x < minX) minX = x
      if (y < minY) minY = y
      if (x > maxX) maxX = x
      if (y > maxY) maxY = y
      return
    }
    for (const child of coords) visit(child)
  }
  for (const feature of collection.features) {
    if (feature.geometry) visit(feature.geometry.coordinates)
  }
  if (!Number.isFinite(minX) || !Number.isFinite(minY)) return null
  return [minX, minY, maxX, maxY]
}

/** 投影平面上的真实图幅尺寸（米）；仅对等距投影（高斯克吕格/UTM）有意义。 */
export function planarExtentKm(bbox: BBox): { widthKm: number; heightKm: number } {
  return {
    widthKm: Math.round((bbox[2] - bbox[0]) / 100) / 10,
    heightKm: Math.round((bbox[3] - bbox[1]) / 100) / 10,
  }
}
/**
 * 地图比例尺（状态栏 1:N）——MapLibre 的 zoom 是 **512px 瓦片口径**。
 *
 * Web Mercator 的 z=0 是「整个世界落在 512 CSS 像素宽」的方格上，所以赤道处
 * 1 像素 = 40075016.686 / 512 ≈ 78271.5 m。早期实现用了 256px 瓦片口径的
 * 156543.03392，状态栏比例尺因此恒偏 2 倍（DAIR-335 实测：同一屏幕 MapLibre
 * 自带 `ScaleControl` 画 2 km，状态栏报 1:176,565）。
 *
 * 对齐点：MapLibre `ScaleControl` 内部就是 `78271.484 * cos(lat) / 2^zoom`
 * （maplibre-gl/src/ui/control/scale.ts），常量差异 < 1e-6，可忽略。
 */

/** 赤道周长（WGS84，米）。 */
export const EARTH_CIRCUMFERENCE = 40075016.686

/** MapLibre 的瓦片边长（CSS 像素）。zoom 0 时世界宽正好是它。 */
export const TILE_SIZE = 512

/** 每 CSS 像素对应的地面米数（含纬度收敛修正：墨卡托在纬度 φ 处被放大 1/cos φ）。 */
export function metersPerPixel(zoom: number, latitude: number): number {
  return (
    (EARTH_CIRCUMFERENCE / TILE_SIZE) *
    Math.cos((latitude * Math.PI) / 180) /
    Math.pow(2, zoom)
  )
}

/** 屏幕 96dpi（0.0254 m/inch）下 1:N 的 N。 */
export function scaleDenominator(zoom: number, latitude: number): number {
  return Math.round((metersPerPixel(zoom, latitude) * 96) / 0.0254)
}

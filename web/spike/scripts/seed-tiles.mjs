#!/usr/bin/env node
/**
 * 抓一份「本地底图瓦片快照」，让 spike 在没有外网/key 的机器上也能显示真实底图。
 *
 * 用途：把在线 OSM 派生栅格（CARTO light_all）按指定范围/层级抓成静态文件，
 * 落到 public/tiles/{z}/{x}/{y}.png，MapLibre 用 ./tiles/{z}/{x}/{y}.png 直接读。
 *
 * 这是「内网底图」的最小验证；正式方案见 docs/dair298-web-gis-viz.md §3.4
 * （自托管 PMTiles + 全量 OSM 抽取，或天地图/GeoQ 内网瓦片服务）。
 *
 * 用法：node scripts/seed-tiles.mjs [minZoom] [maxZoom]
 * 默认 z11–14，覆盖北京样例范围（并可外扩 1 个瓦片）。
 */
import { existsSync, statSync } from 'node:fs'
import { mkdir, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'

const TEMPLATE = 'https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png'
const BBOX = { west: 116.26, south: 39.8, east: 116.52, north: 40.02 }
const MIN_Z = Number(process.argv[2] ?? 11)
const MAX_Z = Number(process.argv[3] ?? 14)
const OUT = join(process.cwd(), 'public', 'tiles')

const lon2tile = (lon, z) => Math.floor(((lon + 180) / 360) * 2 ** z)
const lat2tile = (lat, z) => {
  const rad = (lat * Math.PI) / 180
  return Math.floor(((1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2) * 2 ** z)
}

let downloaded = 0
let bytes = 0
for (let z = MIN_Z; z <= MAX_Z; z += 1) {
  const x0 = lon2tile(BBOX.west, z) - 1
  const x1 = lon2tile(BBOX.east, z) + 1
  const y0 = lat2tile(BBOX.north, z) - 1
  const y1 = lat2tile(BBOX.south, z) + 1
  for (let x = x0; x <= x1; x += 1) {
    for (let y = y0; y <= y1; y += 1) {
      const url = TEMPLATE.replace('{z}', z).replace('{x}', x).replace('{y}', y)
      const target = join(OUT, String(z), String(x), `${y}.png`)
      try {
        if (existsSync(target) && statSync(target).size > 0) continue
        const response = await fetch(url, {
          headers: { 'user-agent': 'gis-agent-spike/0.0.1 (local basemap snapshot)' },
          signal: AbortSignal.timeout(15_000),
        })
        if (!response.ok) {
          console.warn(`skip ${z}/${x}/${y}: HTTP ${response.status}`)
          continue
        }
        const buffer = Buffer.from(await response.arrayBuffer())
        await mkdir(dirname(target), { recursive: true })
        await writeFile(target, buffer)
        downloaded += 1
        bytes += buffer.length
      } catch (cause) {
        // 单块超时/断连不打断整批；重跑本脚本会补齐缺块。
        console.warn(`retry-later ${z}/${x}/${y}: ${cause.message}`)
      }
    }
  }
}
console.log(`tiles: ${downloaded} files, ${(bytes / 1024 / 1024).toFixed(2)} MB → public/tiles`)
console.log('attribution: © OpenStreetMap contributors · © CARTO')

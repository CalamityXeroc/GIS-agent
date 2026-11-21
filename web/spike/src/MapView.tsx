import {
  AttributionControl,
  Map as MapLibreMap,
  NavigationControl,
  ScaleControl,
  type GeoJSONSource,
} from 'maplibre-gl'
import { useEffect, useRef, useState } from 'react'

import type { BBox, GeoJSONCollection } from './geo'
import type { BasemapMode } from './layers'

/** 本地瓦片：由 scripts/seed-tiles.mjs 抓的快照，覆盖北京样例范围 z11–14；超出范围由 maxzoom 拉伸。 */
const OFFLINE_TILES = ['./tiles/{z}/{x}/{y}.png']
const ONLINE_TILES = ['https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png']
const OFFLINE_ATTRIBUTION = '底图瓦片：© OpenStreetMap contributors · © CARTO（本地快照，内网可复现）'
const ONLINE_ATTRIBUTION = '底图：© OpenStreetMap contributors · © CARTO（在线；内网需换自托管瓦片）'

function baseStyle(basemap: BasemapMode) {
  return {
    version: 8 as const,
    sources: {
      basemap: {
        type: 'raster' as const,
        tiles: basemap === 'online' ? ONLINE_TILES : OFFLINE_TILES,
        tileSize: 256,
        maxzoom: basemap === 'online' ? 19 : 14,
        attribution: basemap === 'online' ? ONLINE_ATTRIBUTION : OFFLINE_ATTRIBUTION,
      },
    },
    layers: [{ id: 'bg', type: 'background' as const, paint: { 'background-color': '#f7f7f5' } }],
  }
}

/** 底图切换不重建 style：拆/装 raster source 即可，叠加图层与视口都保住。 */
function applyBasemap(map: MapLibreMap, basemap: BasemapMode) {
  if (map.getLayer('basemap')) map.removeLayer('basemap')
  if (map.getSource('basemap')) map.removeSource('basemap')
  if (basemap === 'none') return
  map.addSource('basemap', {
    type: 'raster',
    tiles: basemap === 'online' ? ONLINE_TILES : OFFLINE_TILES,
    tileSize: 256,
    maxzoom: basemap === 'online' ? 19 : 14,
    attribution: basemap === 'online' ? ONLINE_ATTRIBUTION : OFFLINE_ATTRIBUTION,
  })
  map.addLayer(
    { id: 'basemap', type: 'raster', source: 'basemap', paint: { 'raster-opacity': 0.95 } },
    map.getLayer('layer-fill') ? 'layer-fill' : undefined
  )
}

interface MapViewProps {
  data: GeoJSONCollection
  bbox: BBox
  basemap: BasemapMode
  zoom: number
}

export function MapView({ data, bbox, basemap, zoom }: MapViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const [ready, setReady] = useState(false)
  const [centerText, setCenterText] = useState('')
  const [zoomText, setZoomText] = useState('')
  const storyRef = useRef({ data, bbox, zoom, basemap })
  storyRef.current = { data, bbox, zoom, basemap }

  useEffect(() => {
    if (!containerRef.current) return
    const map = new MapLibreMap({
      container: containerRef.current,
      style: baseStyle(basemap),
      center: [116.38, 39.91],
      zoom,
      maxZoom: 18,
      attributionControl: false,
    })
    map.addControl(new NavigationControl({ visualizePitch: true }), 'top-right')
    map.addControl(new ScaleControl({ maxWidth: 120, unit: 'metric' }), 'bottom-left')
    map.addControl(new AttributionControl({ compact: true }), 'bottom-right')

    map.on('load', () => {
      applyBasemap(map, storyRef.current.basemap)
      map.addSource('layer', { type: 'geojson', data: storyRef.current.data as never })
      map.addLayer({
        id: 'layer-fill',
        type: 'fill',
        source: 'layer',
        filter: ['==', ['geometry-type'], 'Polygon'],
        paint: { 'fill-color': '#2563eb', 'fill-opacity': 0.35, 'fill-outline-color': '#1d4ed8' },
      })
      map.addLayer({
        id: 'layer-line',
        type: 'line',
        source: 'layer',
        filter: ['in', ['geometry-type'], ['literal', ['LineString', 'MultiLineString']]],
        paint: { 'line-color': '#0ea5e9', 'line-width': 1.1 },
      })
      map.addLayer({
        id: 'layer-outline',
        type: 'line',
        source: 'layer',
        filter: ['==', ['geometry-type'], 'Polygon'],
        paint: { 'line-color': '#1e40af', 'line-width': 0.8 },
      })
      map.addLayer({
        id: 'layer-point',
        type: 'circle',
        source: 'layer',
        filter: ['==', ['geometry-type'], 'Point'],
        paint: {
          'circle-color': '#2563eb',
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 3, 1.6, 10, 3, 16, 6],
          'circle-opacity': 0.75,
          'circle-stroke-width': 0.4,
          'circle-stroke-color': '#ffffff',
        },
      })
      map.fitBounds(
        [
          [storyRef.current.bbox[0], storyRef.current.bbox[1]],
          [storyRef.current.bbox[2], storyRef.current.bbox[3]],
        ],
        { padding: 28, duration: 0 }
      )
      setReady(true)
    })

    const sync = () => {
      const c = map.getCenter()
      setCenterText(`${c.lng.toFixed(4)}, ${c.lat.toFixed(4)}`)
      setZoomText(map.getZoom().toFixed(2))
    }
    map.on('move', sync)
    sync()

    mapRef.current = map
    return () => {
      map.remove()
      mapRef.current = null
    }
    // 只在挂载时建图；底图/数据/缩放走下面的增量更新。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 底图切换：拆/装 raster source，不重建 style（不闪、不丢叠加图层）。
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    applyBasemap(map, basemap)
  }, [basemap, ready])

  // 数据/图层切换：换 source 数据并重新取景。
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const source = map.getSource('layer') as GeoJSONSource | undefined
    source?.setData(data as never)
    map.fitBounds(
      [
        [bbox[0], bbox[1]],
        [bbox[2], bbox[3]],
      ],
      { padding: 28, duration: 0 }
    )
  }, [data, bbox, ready])

  // 深链缩放：?z=16 用于截图对比不同细节层级。
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    if (Math.abs(map.getZoom() - zoom) > 0.01) map.setZoom(zoom)
  }, [zoom, ready])

  return (
    <div className="map-wrap">
      <div ref={containerRef} className="map" />
      <div className="map-badge">
        <span>Z {zoomText || '—'}</span>
        <span>中心 {centerText || '—'}</span>
      </div>
    </div>
  )
}
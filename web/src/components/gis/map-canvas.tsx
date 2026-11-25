import maplibregl from 'maplibre-gl'
import type {
  FilterSpecification,
  GeoJSONSourceSpecification,
  LayerSpecification,
} from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import {
  AlertTriangle,
  ChevronDown,
  Info,
  Layers,
  Loader2,
  Maximize2,
  MousePointer2,
  RefreshCw,
  SquareDashedMousePointer,
  X,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'
import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'

import localStyle from '@/components/layers/map-style.json'
import type { LayerPreview } from '@/lib/api'
import { toPaintColor } from '@/lib/paint-color'
import { scaleDenominator } from '@/lib/scale'
import { useThemeStore } from '@/lib/theme'
import { cn } from '@/lib/utils'

import { describeEmptyView, describePreviewError } from './format'
import {
  LAYER_PALETTE,
  isPreviewable,
  useGisStore,
  type GisLayer,
  type GisLayerStyle,
} from './store'
import type { LayerPreviewState } from './use-layer-previews'
import { bboxKey, quantizeViewport } from './viewport'

export type CatalogStatus = 'loading' | 'error' | 'ready'

export interface MapStageProps {
  layers: GisLayer[]
  previews: Record<string, LayerPreviewState>
  selectedUid: number | null
  catalogStatus: CatalogStatus
  catalogCount: number
  catalogError?: string
  catalogTruncated?: boolean
  onRetryCatalog: () => void
  onRetryPreview: (path: string) => void
  onOpenCatalog: () => void
}

const FALLBACK_COLORS = ['#3b82f6', '#06b6d4', '#8b5cf6', '#ec4899', '#22c55e']

const POLYGON_FILTER = [
  'in',
  ['geometry-type'],
  ['literal', ['Polygon', 'MultiPolygon']],
] as unknown as FilterSpecification

const LINE_FILTER = [
  'in',
  ['geometry-type'],
  ['literal', ['LineString', 'MultiLineString']],
] as unknown as FilterSpecification

const POINT_FILTER = [
  'in',
  ['geometry-type'],
  ['literal', ['Point', 'MultiPoint']],
] as unknown as FilterSpecification

const sourceId = (uid: number) => `gis-src-${uid}`
const fillId = (uid: number) => `gis-fill-${uid}`
const outlineId = (uid: number) => `gis-outline-${uid}`
const lineId = (uid: number) => `gis-line-${uid}`
const pointId = (uid: number) => `gis-point-${uid}`
const selFillId = (uid: number) => `gis-sel-fill-${uid}`
const selLineId = (uid: number) => `gis-sel-line-${uid}`
const selPointId = (uid: number) => `gis-sel-point-${uid}`

/** 要素图层 id → uid：只有这 4 种是数据图层，高亮层与浮层不算。 */
const BASE_LAYER_RE = /^gis-(?:fill|outline|line|point)-(\d+)$/

/** CSS token → MapLibre 认识的 rgba；oklch 直接用会被 addLayer 静默丢弃。 */
function chartPalette(): string[] {
  if (typeof window === 'undefined') return FALLBACK_COLORS
  const styles = window.getComputedStyle(document.documentElement)
  return LAYER_PALETTE.map((token, index) =>
    toPaintColor(styles.getPropertyValue(token).trim(), FALLBACK_COLORS[index])
  )
}

/** 选中高亮色：跟主题主色走，不另起一套颜色语义。 */
function highlightColor(): string {
  if (typeof window === 'undefined') return '#f59e0b'
  return toPaintColor(
    window.getComputedStyle(document.documentElement).getPropertyValue('--primary').trim(),
    '#f59e0b'
  )
}

const countFormat = new Intl.NumberFormat('zh-CN')

/**
 * 预览质量角标（DAIR-368）：要素被截 / 顶点被简化 / 超时降级，必须看得见。
 *
 * 以前只写两个字「抽稀」：看不出是“只给了前 400 个要素”还是“顶点被压过”，
 * 也看不出超时降级。角标给短标签，`title` 给完整原因（含服务端预算）。
 *
 * 「只给了多少」不再由这里承担——DAIR-386 起由 :func:`viewportNote` 直接写出
 * 「视口内 X / 全量 Y」，只写「部分」看不出少在哪儿。
 */
function previewBadge(preview: LayerPreview): { label: string; hint: string } | null {
  const lines: string[] = []
  if (preview.degraded) {
    lines.push(preview.degraded.reason ?? '读取超时，已降级为部分要素')
  }
  if (preview.oversized) {
    lines.push('图层过大，本次只返回坐标范围')
  }
  if (preview.truncated) {
    // 视口内计数不精确（.gdb 内核扫到上限就停，`view_count_exact=false`）时，
    // `total_in_view` 只是下限：拿它当分母会写出「视口内 3000 / 3000，只给了一部分」，
    // 跟图例的「视口内 ≥3000 / 全量 900,000（未给全）」自相矛盾（DAIR-390）。
    // 这时分母只能用全量数；连全量数都没有就不写分母，不编数字。
    const inexact = preview.view_bbox !== null && !preview.view_count_exact
    const inView = `${inexact ? '≥' : ''}${countFormat.format(preview.returned_features)}`
    const total = inexact
      ? preview.feature_count
      : (preview.total_in_view ?? preview.feature_count)
    lines.push(
      total === null
        ? `视口内 ${inView}，只给了一部分`
        : `视口内 ${inView} / ${countFormat.format(total)}，只给了一部分`
    )
  }
  if (preview.simplified) {
    lines.push(
      `顶点已按容差简化：${preview.input_points ?? '?'} → ${preview.point_count} 点` +
        (preview.tolerance ? `（容差 ${preview.tolerance}）` : '')
    )
  }
  if (lines.length === 0) return null
  if (typeof preview.budget_s === 'number') {
    lines.push(`服务端读取预算 ${preview.budget_s}s`)
  }
  const label = preview.degraded
    ? '降级'
    : preview.oversized
      ? '仅范围'
      : preview.truncated
        ? '部分'
        : '简化'
  return { label, hint: lines.join('\n') }
}

/**
 * 图例里的诚实口径（DAIR-386）：`视口内 26,673 / 全量 134,819`。
 *
 * 只写「部分」看不出到底缺多少；只写全量又看不到当前视野里给了多少。覆盖率达到
 * 100% 也把「已给 N（100%）」写出来——用户能直接对照「视口内多少、图上多少」，
 * 不用自己推。这是真数字，不是「大部分」这种含糊词。
 */
function viewportNote(preview: LayerPreview): string | null {
  if (preview.total_in_view === null) return null
  const inView = `${preview.view_count_exact ? '' : '≥'}${countFormat.format(preview.total_in_view)}`
  const total =
    preview.feature_count === null ? null : countFormat.format(preview.feature_count)
  const head = total
    ? `视口内 ${inView} / 全量 ${total}`
    : `视口内 ${inView}`
  const coverage = preview.coverage
  // 视口内真的零要素（拖到数据外）：0 就是 0，不是「没给全」（DAIR-391 D2）。
  if (preview.total_in_view === 0) return head
  // 视口内只数到下限（.gdb 内核扫到上限就停了）：给不出百分比，但必须说「没给全」，
  // 不能落到「看起来全给了」（DAIR-388）。
  if (coverage === null) return `${head}（未给全）`
  return `${head} · 已给 ${countFormat.format(preview.returned_features)}（${Math.round(coverage * 100)}%）`
}

/** 地图上点/框命中的要素 → `图层路径 → 要素 id`（只认已知标识列的图层）。 */
function collectSelection(
  map: maplibregl.Map,
  geometry: maplibregl.PointLike | [maplibregl.PointLike, maplibregl.PointLike],
  single = false
): Map<string, string[]> {
  const store = useGisStore.getState()
  const found = new Map<string, string[]>()
  let hits: maplibregl.MapGeoJSONFeature[] = []
  try {
    hits = map.queryRenderedFeatures(geometry)
  } catch {
    return found
  }
  for (const hit of hits) {
    const match = BASE_LAYER_RE.exec(String(hit.layer?.id ?? ''))
    if (!match) continue
    const layer = store.layers.find((item) => item.uid === Number(match[1]))
    if (!layer) continue
    const field = store.idFields[layer.path]
    if (!field) continue
    const value = hit.properties?.[field]
    if (value === null || value === undefined) continue
    const ids = found.get(layer.path) ?? []
    const id = String(value)
    if (!ids.includes(id)) ids.push(id)
    found.set(layer.path, ids)
    // 点选只认最上面那个要素（queryRenderedFeatures 按绘制顺序自上而下）：
    // 叠了三层多边形时不把三层全选中，那是框选的语义。
    if (single) return found
  }
  return found
}

/** 选择集写入：不加选就是「替换所有图层的选择」，加选就是按图层并集。 */
function applySelection(found: Map<string, string[]>, additive: boolean) {
  const store = useGisStore.getState()
  if (!additive) store.clearSelection()
  for (const [path, ids] of found) {
    if (ids.length === 0) continue
    const current = additive ? (store.selection[path] ?? []) : []
    store.setSelection(path, [...current, ...ids.filter((id) => !current.includes(id))])
  }
}

function fitBbox(map: maplibregl.Map, bbox: [number, number, number, number] | null): boolean {
  // 非 4326 且后端没能归一的数据，bbox 是米（例如 222545,120207）；
  // 直接交给 fitBounds 会抛 Invalid LngLat 并把整个组件卸掉（实测崩溃）。
  if (!bbox || !isLngLatBbox(bbox)) return false
  if (!(bbox[2] > bbox[0]) || !(bbox[3] > bbox[1])) return false
  map.fitBounds(
    [
      [bbox[0], bbox[1]],
      [bbox[2], bbox[3]],
    ],
    { padding: 36, duration: 0, maxZoom: 17 }
  )
  return true
}

/** 只有经度 ±180、纬度 ±90 以内的 bbox 才能交给 MapLibre 当经纬度用。 */
function isLngLatBbox(bbox: [number, number, number, number] | null): bbox is [number, number, number, number] {
  if (!bbox || bbox.length < 4) return false
  const [minX, minY, maxX, maxY] = bbox
  return (
    [minX, minY, maxX, maxY].every(Number.isFinite) &&
    minX >= -180 &&
    maxX <= 180 &&
    minY >= -90 &&
    maxY <= 90
  )
}

function bboxOf(previews: LayerPreview[]): [number, number, number, number] | null {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const preview of previews) {
    const [x1, y1, x2, y2] = preview.bbox
    if (!isLngLatBbox([x1, y1, x2, y2])) continue
    minX = Math.min(minX, x1)
    minY = Math.min(minY, y1)
    maxX = Math.max(maxX, x2)
    maxY = Math.max(maxY, y2)
  }
  return Number.isFinite(minX) ? [minX, minY, maxX, maxY] : null
}

/**
 * 工作台地图（spec §4 ⑥）：MapLibre + 仓库内离线样式，一层一个 source/layer。
 *
 * 沿用 DAIR-299/301 定下的离线做法（原 `components/layers/` 旧组件已删，样式文件保留）：
 * - `map-style.json` 只有透明背景，不含 glyphs/sprite，断网可用；
 * - 颜色读 CSS 变量后必须过 `toPaintColor`，oklch 会被 MapLibre 静默拒收；
 * - 图层增量同步（addSource/setData/moveLayer），不重建 map。
 */
export function MapStage({
  layers,
  previews,
  selectedUid,
  catalogStatus,
  catalogCount,
  catalogError,
  catalogTruncated,
  onRetryCatalog,
  onRetryPreview,
  onOpenCatalog,
}: MapStageProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const mounted = useRef(new Set<number>())
  const dataCache = useRef(new Map<number, unknown>())
  const fitted = useRef(new Set<number>())
  const [ready, setReady] = useState(false)
  const [mapError, setMapError] = useState<string | null>(null)
  const [legendOpen, setLegendOpen] = useState(true)
  const theme = useThemeStore((state) => state.theme)
  const command = useGisStore((state) => state.command)
  const mapTool = useGisStore((state) => state.mapTool)
  const selection = useGisStore((state) => state.selection)
  const idFields = useGisStore((state) => state.idFields)
  const lastCommand = useRef(0)
  /** 框选结束后的那次 click 不要被当成点选（同一次手势只能有一种语义）。 */
  const suppressClick = useRef(0)
  const bandStart = useRef<{ x: number; y: number } | null>(null)
  const [band, setBand] = useState<
    { left: number; top: number; width: number; height: number } | null
  >(null)

  // 建图：只做一次。数据/主题走下面的增量 effect。
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    let frame = 0
    const map = new maplibregl.Map({
      container,
      style: localStyle as unknown as maplibregl.StyleSpecification,
      attributionControl: false,
      center: [104, 35],
      zoom: 4,
      maxZoom: 20,
      minZoom: 0,
      dragRotate: false,
      pitchWithRotate: false,
      touchPitch: false,
    })
    mapRef.current = map
    map.touchZoomRotate.disableRotation()

    map.addControl(new maplibregl.ScaleControl({ maxWidth: 96, unit: 'metric' }), 'bottom-left')

    const pushView = () => {
      const store = useGisStore.getState()
      store.setView({
        zoom: Number(map.getZoom().toFixed(2)),
        scale: scaleDenominator(map.getZoom(), map.getCenter().lat),
        baseZoom: store.view.baseZoom ?? map.getZoom(),
      })
    }

    // 预览按视口取（DAIR-386）：`moveend` 后把量化过的视口写回 store，
    // useLayerPreviews 用它的指纹当 queryKey。量化后小幅平移不换 key，
    // 不会每拖一下就把每个图层重请求一轮。
    let viewportTimer = 0
    const pushViewport = () => {
      const bounds = map.getBounds()
      const zoom = Math.floor(map.getZoom())
      const bbox = quantizeViewport(
        {
          west: bounds.getWest(),
          south: bounds.getSouth(),
          east: bounds.getEast(),
          north: bounds.getNorth(),
        },
        zoom
      )
      const store = useGisStore.getState()
      if (store.viewport.zoom === zoom && bboxKey(store.viewport.bbox) === bboxKey(bbox)) return
      store.setViewport({ bbox, zoom })
    }
    const scheduleViewport = () => {
      window.clearTimeout(viewportTimer)
      viewportTimer = window.setTimeout(pushViewport, 300)
    }

    map.on('load', () => {
      setReady(true)
      const store = useGisStore.getState()
      store.setView({
        ready: true,
        zoom: Number(map.getZoom().toFixed(2)),
        scale: scaleDenominator(map.getZoom(), map.getCenter().lat),
        baseZoom: map.getZoom(),
      })
      pushViewport()
    })

    map.on('moveend', scheduleViewport)

    map.on('mousemove', (event) => {
      if (frame) return
      frame = requestAnimationFrame(() => {
        frame = 0
        useGisStore.getState().setView({
          cursor: [event.lngLat.lng, event.lngLat.lat],
        })
      })
    })

    // 点选：只在「选择」工具下生效；浏览态保持平移/缩放不动。
    map.on('click', (event) => {
      if (useGisStore.getState().mapTool !== 'select') return
      if (performance.now() < suppressClick.current) return
      const original = event.originalEvent as MouseEvent | undefined
      const additive = Boolean(original?.shiftKey || original?.ctrlKey || original?.metaKey)
      applySelection(collectSelection(map, event.point, true), additive)
    })

    map.on('move', pushView)
    map.on('zoom', pushView)
    map.on('rotate', pushView)

    // 不接 tiles/glyphs，正常的样式错误不应出现；真出了就显示成横幅，别刷控制台。
    map.on('error', (event) => {
      const message = event.error?.message
      if (message) setMapError(message)
    })

    const observer = new ResizeObserver(() => map.resize())
    observer.observe(container)

    return () => {
      observer.disconnect()
      if (frame) cancelAnimationFrame(frame)
      window.clearTimeout(viewportTimer)
      map.remove()
      mapRef.current = null
      setReady(false)
      useGisStore.getState().setView({ ready: false })
      useGisStore.getState().setViewport({ bbox: null, zoom: 0 })
    }
  }, [])

  // 图层同步：增删 source/layer、更新数据与显隐、按栈序重排。
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const colors = chartPalette()
    const wanted = new Set(layers.map((layer) => layer.uid))

    for (const uid of [...mounted.current]) {
      if (wanted.has(uid)) continue
      removeLayerSet(map, uid)
      mounted.current.delete(uid)
      dataCache.current.delete(uid)
    }

    for (const layer of layers) {
      const preview = previews[layer.path]?.preview
      // bbox 不在经纬度范围内 = 后端没能把几何归一到 4326（crs=null），
      // 这种情况下图上不画也不能画，避免把米当成度画到太平洋/直接报错。
      if (!preview || !isLngLatBbox(preview.bbox)) {
        if (mounted.current.has(layer.uid)) {
          removeLayerSet(map, layer.uid)
          mounted.current.delete(layer.uid)
          dataCache.current.delete(layer.uid)
        }
        continue
      }
      const color = colors[layer.style.colorIndex % colors.length]
      syncLayer(map, layer, preview, color, dataCache.current)
      mounted.current.add(layer.uid)
    }

    // 栈顶最后绘制：按「栈底 → 栈顶」逐个 moveLayer 到最上层。
    for (const layer of [...layers].reverse()) {
      if (!mounted.current.has(layer.uid)) continue
      for (const id of [fillId(layer.uid), outlineId(layer.uid), lineId(layer.uid), pointId(layer.uid)]) {
        if (map.getLayer(id)) map.moveLayer(id)
      }
    }
  }, [layers, previews, ready, theme])

  // 新图层第一次拿到预览时取景：首层取全部，后续只取新加的那层。
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const readyLayers = layers.filter((layer) => previews[layer.path]?.preview)
    const fresh = readyLayers.filter((layer) => !fitted.current.has(layer.uid))
    if (fresh.length === 0) return
    const hadAny = fitted.current.size > 0
    for (const layer of readyLayers) fitted.current.add(layer.uid)
    const chosen = hadAny
      ? [fresh[0]]
      : readyLayers
    const didFit = fitBbox(
      map,
      bboxOf(
        chosen
          .map((layer) => previews[layer.path]?.preview)
          .filter((preview): preview is LayerPreview => Boolean(preview))
      )
    )
    // 首次取景就是「1.00×」：状态栏的缩放倍率以用户实际看到的第一帧为基准，
    // 而不是建图时的 zoom 4（那会让倍率显示成 30× 这种无意义数字）。
    // 取景被跳过（bbox 不是经纬度）时不能写基线：否则基线停在建图 zoom，
    // 之后每次真取景都按它算倍率（实测状态栏出现 160.90× 这种数字）。
    if (!hadAny && didFit) {
      useGisStore.getState().setView({ baseZoom: Number(map.getZoom().toFixed(2)) })
    }
  }, [layers, previews, ready])

  // 命令栏/图层行内按钮 → 地图动作（store 里只发命令，地图自己执行）。
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready || !command || command.seq === lastCommand.current) return
    lastCommand.current = command.seq
    if (command.kind === 'zoom-in') map.zoomIn({ duration: 160 })
    if (command.kind === 'zoom-out') map.zoomOut({ duration: 160 })
    if (command.kind === 'fit-all') {
      fitBbox(
        map,
        bboxOf(
          layers
            .map((layer) => previews[layer.path]?.preview)
            .filter((preview): preview is LayerPreview => Boolean(preview))
        )
      )
    }
    if (command.kind === 'fit-layer') {
      const layer = layers.find((item) => item.uid === command.uid)
      const preview = layer ? previews[layer.path]?.preview : undefined
      if (preview) fitBbox(map, preview.bbox)
    }
  }, [command, ready, layers, previews])

  // 工具态：框选时关掉拖拽平移（否则 MapLibre 会把拖动吃掉），游标跟着变。
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const canvas = map.getCanvas()
    if (mapTool === 'box') {
      map.dragPan.disable()
      canvas.style.cursor = 'crosshair'
    } else if (mapTool === 'select') {
      map.dragPan.enable()
      canvas.style.cursor = 'pointer'
    } else {
      map.dragPan.enable()
      canvas.style.cursor = ''
    }
  }, [mapTool, ready])

  // 选中高亮：复用图层自己的 source，只加 3 个按 id 过滤的高亮层（不重建数据）。
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const highlight = highlightColor()
    for (const layer of layers) {
      const field = idFields[layer.path]
      const ids = (selection[layer.path] ?? []).map(String)
      const preview = previews[layer.path]?.preview
      const uid = layer.uid
      const wanted = Boolean(field && ids.length > 0 && preview && isLngLatBbox(preview.bbox))
      if (!wanted) {
        for (const id of [selPointId(uid), selLineId(uid), selFillId(uid)]) {
          if (map.getLayer(id)) map.removeLayer(id)
        }
        continue
      }
      const idFilter = [
        'in',
        ['to-string', ['get', field as string]],
        ['literal', ids],
      ] as unknown as FilterSpecification
      const fillFilter = ['all', POLYGON_FILTER, idFilter] as unknown as FilterSpecification
      const lineFilter = [
        'all',
        ['in', ['geometry-type'], ['literal', ['Polygon', 'MultiPolygon', 'LineString', 'MultiLineString']]],
        idFilter,
      ] as unknown as FilterSpecification
      const pointFilter = ['all', POINT_FILTER, idFilter] as unknown as FilterSpecification

      if (!map.getLayer(selFillId(uid))) {
        map.addLayer({
          id: selFillId(uid),
          type: 'fill',
          source: sourceId(uid),
          filter: fillFilter,
          paint: { 'fill-color': highlight, 'fill-opacity': 0.16 },
        })
        map.addLayer({
          id: selLineId(uid),
          type: 'line',
          source: sourceId(uid),
          filter: lineFilter,
          paint: { 'line-color': highlight, 'line-width': 2.4 },
        })
        map.addLayer({
          id: selPointId(uid),
          type: 'circle',
          source: sourceId(uid),
          filter: pointFilter,
          paint: {
            'circle-color': highlight,
            'circle-radius': 6,
            'circle-stroke-width': 1.6,
            'circle-stroke-color': '#ffffff',
          },
        })
      } else {
        map.setFilter(selFillId(uid), fillFilter)
        map.setFilter(selLineId(uid), lineFilter)
        map.setFilter(selPointId(uid), pointFilter)
        map.setPaintProperty(selFillId(uid), 'fill-color', highlight)
        map.setPaintProperty(selLineId(uid), 'line-color', highlight)
        map.setPaintProperty(selPointId(uid), 'circle-color', highlight)
      }
      // 高亮永远在数据层上方：否则新加的图层会把选中要素盖住。
      for (const id of [selFillId(uid), selLineId(uid), selPointId(uid)]) {
        if (map.getLayer(id)) map.moveLayer(id)
      }
      for (const id of [selFillId(uid), selLineId(uid), selPointId(uid)]) {
        if (map.getLayer(id)) {
          map.setLayoutProperty(id, 'visibility', layer.visible ? 'visible' : 'none')
        }
      }
    }
  }, [layers, previews, selection, idFields, ready, theme])

  /** 指针坐标 → 地图容器内坐标（框选矩形与 queryRenderedFeatures 用同一坐标系）。 */
  const localPoint = (event: ReactPointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    return { x: event.clientX - rect.left, y: event.clientY - rect.top }
  }

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (useGisStore.getState().mapTool !== 'box') return
    if ((event.target as HTMLElement | null)?.tagName !== 'CANVAS') return
    const point = localPoint(event)
    bandStart.current = point
    setBand({ left: point.x, top: point.y, width: 0, height: 0 })
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const start = bandStart.current
    if (!start) return
    const point = localPoint(event)
    setBand({
      left: Math.min(start.x, point.x),
      top: Math.min(start.y, point.y),
      width: Math.abs(point.x - start.x),
      height: Math.abs(point.y - start.y),
    })
  }

  const onPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    const start = bandStart.current
    if (!start) return
    bandStart.current = null
    setBand(null)
    // 框选手势之后的 click 不能再当成点选。
    suppressClick.current = performance.now() + 250
    const point = localPoint(event)
    if (Math.abs(point.x - start.x) < 3 && Math.abs(point.y - start.y) < 3) return
    const map = mapRef.current
    if (!map) return
    const additive = event.shiftKey || event.ctrlKey || event.metaKey
    applySelection(
      collectSelection(map, [
        [start.x, start.y],
        [point.x, point.y],
      ]),
      additive
    )
  }

  const cancelBand = () => {
    bandStart.current = null
    setBand(null)
  }

  const visibleLayers = layers.filter((layer) => layer.visible)
  const previewErrors = layers
    .map((layer) => ({ layer, error: previews[layer.path]?.error }))
    .filter((entry): entry is { layer: GisLayer; error: Error } => Boolean(entry.error))
  const previewIssues = previewErrors.map(({ layer, error }) => ({
    layer,
    issue: describePreviewError(error),
  }))
  // 「仅属性表」不是失败：单独计数，不混进红色「预览失败」的条数。
  const failedPreviews = previewIssues.filter((entry) => !entry.issue.info).length
  const crsWarning = [
    ...new Set(
      layers
        .map((layer) => previews[layer.path]?.preview?.crs_warning)
        .filter((value): value is string => Boolean(value))
    ),
  ][0]
  const loadingPreviews = layers.filter((layer) => previews[layer.path]?.loading).length
  const selectionCount = Object.values(selection).reduce((sum, ids) => sum + ids.length, 0)
  const rasterLayers = layers.filter((layer) => layer.kind === 'raster')
  // 有预览但没有可用经纬度坐标（后端未归一）：图上只能给提示，不画错位几何。
  const unmappable = layers.filter((layer) => {
    const preview = previews[layer.path]?.preview
    return Boolean(preview && !isLngLatBbox(preview.bbox))
  })
  const notPreviewable = layers.filter(
    (layer) => layer.kind !== 'raster' && !isPreviewable(layer.item)
  )

  return (
    <div
      className="relative min-h-0 flex-1 overflow-hidden bg-[var(--gis-map-bg)]"
      style={{
        backgroundImage:
          'linear-gradient(to right, var(--gis-map-grid) 1px, transparent 1px), linear-gradient(to bottom, var(--gis-map-grid) 1px, transparent 1px)',
        backgroundSize: '64px 64px',
        backgroundPosition: '-1px -1px',
      }}
      data-testid="map-stage"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={cancelBand}
    >
      <div ref={containerRef} className="h-full w-full" />

      {/* 框选橡皮筋（工具态才出现，不挡鼠标事件） */}
      {band ? (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute z-[9] border border-primary/80 bg-primary/10"
          style={{
            left: band.left,
            top: band.top,
            width: band.width,
            height: band.height,
          }}
        />
      ) : null}

      {/* 工具提示条：选择/框选下告诉用户能做什么、已选多少 */}
      {mapTool !== 'pan' ? (
        <div className="absolute top-2.5 left-1/2 z-10 flex -translate-x-1/2 items-center gap-1.5 rounded-[var(--gis-radius)] border border-primary/40 bg-card/95 px-2 py-1 text-[10.5px] shadow-[var(--gis-shadow-float)]">
          {mapTool === 'select' ? (
            <MousePointer2 className="size-3 text-primary" />
          ) : (
            <SquareDashedMousePointer className="size-3 text-primary" />
          )}
          <span className="font-semibold text-primary">
            {mapTool === 'select' ? '点选' : '框选'}
          </span>
          <span className="text-muted-foreground">
            {mapTool === 'select' ? '单击要素选中' : '拖拽矩形选中'} · Shift 加选
          </span>
          {selectionCount > 0 ? (
            <>
              <span className="tabular-nums text-foreground/80">已选 {selectionCount}</span>
              <button
                type="button"
                onClick={() => useGisStore.getState().clearSelection()}
                className="inline-flex items-center gap-0.5 rounded-[2px] px-1 text-muted-foreground hover:bg-accent hover:text-foreground"
              >
                <X className="size-2.5" /> 清除
              </button>
            </>
          ) : null}
          <button
            type="button"
            onClick={() => useGisStore.getState().setMapTool('pan')}
            title="回到浏览（平移 / 缩放）"
            className="ml-0.5 rounded-[2px] px-1 text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            浏览
          </button>
        </div>
      ) : null}

      {/* 地图控件：缩放 + 全图 + 正北（不旋转，所以只作指示）。 */}
      <div className="absolute top-2.5 left-2.5 z-10 flex flex-col gap-1.5">
        <div className="flex flex-col overflow-hidden rounded-[var(--gis-radius)] border border-border bg-card/95 shadow-[var(--gis-shadow-float)]">
          <button
            type="button"
            title="放大（滚轮以光标为锚点）"
            aria-label="放大"
            onClick={() => mapRef.current?.zoomIn({ duration: 160 })}
            className="grid size-[26px] place-items-center text-foreground/80 hover:bg-accent hover:text-primary"
          >
            <ZoomIn className="size-3.5" />
          </button>
          <button
            type="button"
            title="缩小"
            aria-label="缩小"
            onClick={() => mapRef.current?.zoomOut({ duration: 160 })}
            className="grid size-[26px] place-items-center border-t border-border text-foreground/80 hover:bg-accent hover:text-primary"
          >
            <ZoomOut className="size-3.5" />
          </button>
          <button
            type="button"
            title="全图（缩放到所有已上图图层）"
            aria-label="全图"
            onClick={() => useGisStore.getState().sendCommand('fit-all')}
            className="grid size-[26px] place-items-center border-t border-border text-foreground/80 hover:bg-accent hover:text-primary"
          >
            <Maximize2 className="size-3.5" />
          </button>
        </div>
        <div
          className="grid size-[26px] place-items-center rounded-[var(--gis-radius)] border border-border bg-card/95 text-muted-foreground shadow-[var(--gis-shadow-float)]"
          title="正北（2D 视图，不旋转）"
          aria-hidden="true"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className="size-4">
            <path d="M12 3l4 8-4 10-4-10z" />
          </svg>
        </div>
      </div>

      {/* 顶部条：CRS 警告 / 预览失败 / 目录失败 / 地图渲染错误。 */}
      <div className="pointer-events-none absolute top-11 left-[52px] right-[176px] z-10 flex max-w-[640px] flex-col gap-1.5">
        {crsWarning ? (
          <div
            role="alert"
            className="pointer-events-auto flex items-start gap-1.5 rounded-[var(--gis-radius)] border border-warning/45 bg-warning/10 px-2 py-1.5 text-[11px] leading-relaxed shadow-[var(--gis-shadow-float)]"
          >
            <AlertTriangle className="mt-px size-3.5 shrink-0 text-warning" />
            <div>
              <b className="text-warning">坐标提示</b>
              <span className="text-foreground/80"> {crsWarning}</span>
            </div>
          </div>
        ) : null}

        {catalogStatus === 'error' ? (
          <div
            role="alert"
            className="pointer-events-auto flex items-start gap-1.5 rounded-[var(--gis-radius)] border border-destructive/45 bg-destructive/10 px-2 py-1.5 text-[11px] leading-relaxed shadow-[var(--gis-shadow-float)]"
          >
            <AlertTriangle className="mt-px size-3.5 shrink-0 text-destructive" />
            <div className="min-w-0 flex-1">
              <b className="text-destructive">读取 catalog 失败</b>
              {catalogError ? (
                <span className="text-foreground/80"> {catalogError}</span>
              ) : null}
            </div>
            <button
              type="button"
              onClick={onRetryCatalog}
              className="shrink-0 rounded-[var(--gis-radius)] border border-border bg-card px-1.5 py-0.5 text-[10.5px] hover:bg-accent"
            >
              重试
            </button>
          </div>
        ) : null}

        {previewIssues.slice(0, 2).map(({ layer, issue }) => {
          const info = issue.info === true
          return (
            <div
              key={layer.uid}
              role={info ? 'status' : 'alert'}
              className={cn(
                'pointer-events-auto flex items-start gap-1.5 rounded-[var(--gis-radius)] border px-2 py-1.5 text-[11px] leading-relaxed shadow-[var(--gis-shadow-float)]',
                info
                  ? 'border-warning/40 bg-warning/10'
                  : 'border-destructive/45 bg-destructive/10'
              )}
            >
              {info ? (
                <Info className="mt-px size-3.5 shrink-0 text-warning" />
              ) : (
                <AlertTriangle className="mt-px size-3.5 shrink-0 text-destructive" />
              )}
              <div className="min-w-0 flex-1">
                <b className={info ? 'text-warning' : 'text-destructive'}>
                  {layer.name}：{issue.title}
                </b>
                <span className="text-foreground/80"> {issue.reason}</span>
                {issue.elapsedMs !== undefined ? (
                  <span className="text-muted-foreground">
                    （服务端 {Math.round(issue.elapsedMs)} ms）
                  </span>
                ) : null}
                {info && issue.suggestion ? (
                  <span className="text-muted-foreground"> 建议：{issue.suggestion}</span>
                ) : null}
              </div>
              {info ? null : (
                <button
                  type="button"
                  onClick={() => onRetryPreview(layer.path)}
                  className="shrink-0 rounded-[var(--gis-radius)] border border-border bg-card px-1.5 py-0.5 text-[10.5px] hover:bg-accent"
                >
                  重试
                </button>
              )}
            </div>
          )
        })}
        {failedPreviews > 2 ? (
          <div className="pointer-events-auto rounded-[var(--gis-radius)] border border-destructive/40 bg-destructive/10 px-2 py-1 text-[10.5px] text-destructive">
            另有 {failedPreviews - 2} 个图层预览失败
          </div>
        ) : null}

        {mapError ? (
          <div className="pointer-events-auto flex items-center gap-1.5 rounded-[var(--gis-radius)] border border-destructive/45 bg-destructive/10 px-2 py-1.5 text-[11px] text-destructive">
            <AlertTriangle className="size-3.5 shrink-0" /> 地图渲染错误：{mapError}
          </div>
        ) : null}

        {unmappable.length > 0 ? (
          <div className="pointer-events-auto rounded-[var(--gis-radius)] border border-warning/40 bg-warning/10 px-2 py-1.5 text-[10.5px] leading-relaxed text-foreground/80">
            <b className="text-warning">坐标系未归一，图上不画几何</b>：
            {unmappable.map((layer) => layer.name).join('、')}（服务端未能重投影到 EPSG:4326，
            元数据仍可看）。
          </div>
        ) : null}

        {rasterLayers.length > 0 ? (
          <div className="pointer-events-auto rounded-[var(--gis-radius)] border border-warning/40 bg-warning/10 px-2 py-1.5 text-[10.5px] leading-relaxed text-foreground/80">
            <b className="text-warning">栅格预览待后端补</b>：{rasterLayers.length} 个栅格图层
            只能看元数据（波段/像元），地图上暂不画影像，也不画米制范围框（会被误当成经纬度）。
          </div>
        ) : null}

        {notPreviewable.length > 0 ? (
          <div className="pointer-events-auto rounded-[var(--gis-radius)] border border-border bg-card/95 px-2 py-1.5 text-[10.5px] text-muted-foreground">
            {notPreviewable.length} 个非空间数据（表格/工程）已加入栈，但没有几何可绘制。
          </div>
        ) : null}
      </div>

      {/* 图例：只在有可见图层时出现，颜色与图层栈色块一致。 */}
      {visibleLayers.length > 0 ? (
        <div className="absolute right-2.5 bottom-8 z-10 w-[196px] overflow-hidden rounded-[var(--gis-radius)] border border-border bg-card/95 text-[11px] shadow-[var(--gis-shadow-float)]">
          <div className="flex h-[24px] items-center gap-1 border-b border-border px-2 text-[10.5px] font-semibold text-muted-foreground">
            <Layers className="size-3" />
            图例 · {visibleLayers.length} 层
            <button
              type="button"
              className="ml-auto grid size-4 place-items-center rounded-[2px] text-muted-foreground hover:bg-accent"
              title={legendOpen ? '折叠图例' : '展开图例'}
              aria-label={legendOpen ? '折叠图例' : '展开图例'}
              onClick={() => setLegendOpen((value) => !value)}
            >
              <ChevronDown
                className={cn('size-3 transition-transform', !legendOpen && '-rotate-90')}
              />
            </button>
          </div>
          {legendOpen ? (
            <ul className="flex flex-col gap-1 px-2 py-1.5">
              {visibleLayers.map((layer) => {
                const state = previews[layer.path]
                // 视口里零要素是信息态，不是「失败」：图例不挂红色角标（DAIR-391）。
                const emptyView = state?.preview ? describeEmptyView(state.preview) : null
                const categories =
                  layer.uid === selectedUid && layer.style.mode === 'unique'
                    ? Object.entries(layer.style.categoryColors)
                    : []
                return (
                  <li key={layer.uid} className="flex flex-col gap-0.5">
                    <div className="flex items-center gap-1.5">
                      <span
                        aria-hidden="true"
                        className="size-[11px] shrink-0 rounded-[2px] border border-foreground/20"
                        style={{
                          background:
                            layer.style.fillColor ?? `var(${LAYER_PALETTE[layer.style.colorIndex]})`,
                        }}
                      />
                      <span
                        className={cn(
                          'min-w-0 flex-1 truncate',
                          layer.uid === selectedUid && 'font-semibold'
                        )}
                      >
                        {layer.name}
                      </span>
                      {state?.loading ? (
                        <Loader2 className="size-3 animate-spin text-muted-foreground" />
                      ) : state?.refreshing ? (
                        // 已经有几何在图上，只是在换新视口的数据：数字不动，标一下
                        // 「更新中」就好（DAIR-393：以前这里整行变占位，看起来像
                        // 每次缩放都要重新加载一遍）。
                        <span className="flex items-center gap-1 tabular-nums text-muted-foreground">
                          <Loader2 className="size-2.5 animate-spin" />
                          <span className="text-[9.5px]">
                            {previewBadge(state.preview!)?.label ?? ''}
                          </span>
                        </span>
                      ) : state?.error ? (
                        <span className="text-[9.5px] text-destructive">失败</span>
                      ) : emptyView ? (
                        <span
                          className="text-[9.5px] text-muted-foreground"
                          title={emptyView.reason}
                        >
                          无要素
                        </span>
                      ) : state?.preview && !isLngLatBbox(state.preview.bbox) ? (
                        <span className="text-[9.5px] text-warning">未归一</span>
                      ) : state?.preview ? (
                        <span
                          className="text-[9.5px] tabular-nums text-muted-foreground"
                          title={previewBadge(state.preview)?.hint}
                        >
                          {previewBadge(state.preview)?.label ?? ''}
                        </span>
                      ) : (
                        <span className="text-[9.5px] text-muted-foreground">待补</span>
                      )}
                    </div>
                    {/* 诚实口径（DAIR-386）：视口内真实要素数 / 全量，覆盖率不足时补百分比。 */}
                    {state?.preview && viewportNote(state.preview) ? (
                      <div
                        className="ml-[17px] text-[9.5px] leading-tight tabular-nums text-muted-foreground"
                        title={previewBadge(state.preview)?.hint}
                      >
                        {viewportNote(state.preview)}
                      </div>
                    ) : null}
                    {/* 唯一值图例：逐类改色后跟着变，与右 dock 同一份 categoryColors。 */}
                    {categories.length > 0 ? (
                      <ul className="ml-[17px] flex flex-col gap-[2px]">
                        {categories.slice(0, 6).map(([key, color]) => (
                          <li key={key} className="flex items-center gap-1.5">
                            <span
                              aria-hidden="true"
                              className="size-[9px] shrink-0 rounded-[2px] border border-foreground/20"
                              style={{ background: color }}
                            />
                            <span className="min-w-0 flex-1 truncate text-[10px] text-foreground/80">
                              {key}
                            </span>
                          </li>
                        ))}
                        {categories.length > 6 ? (
                          <li className="text-[9.5px] text-muted-foreground">
                            …还有 {categories.length - 6} 类
                          </li>
                        ) : null}
                      </ul>
                    ) : null}
                  </li>
                )
              })}
            </ul>
          ) : null}
        </div>
      ) : null}

      {/* 空态 / 加载态：地图本体始终占主区，不缩成卡片。 */}
      {layers.length === 0 ? (
        <div className="pointer-events-none absolute inset-0 z-[5] grid place-items-center">
          {catalogStatus === 'loading' ? (
            <div className="flex items-center gap-2 rounded-[var(--gis-radius)] border border-border bg-card/95 px-3 py-2 text-[11.5px] text-muted-foreground shadow-[var(--gis-shadow-float)]">
              <Loader2 className="size-3.5 animate-spin" /> 正在读取 catalog…
            </div>
          ) : (
            <div className="pointer-events-auto w-[min(420px,82%)] rounded-[6px] border border-border bg-card/95 p-5 text-center shadow-[var(--gis-shadow-float)]">
              <div className="mx-auto mb-2.5 grid size-10 place-items-center rounded-[8px] bg-accent text-primary">
                <Layers className="size-5" />
              </div>
              <h3 className="text-[13px] font-semibold text-foreground">地图是空的</h3>
              <p className="mt-1 mb-3 text-[11.5px] leading-relaxed text-muted-foreground">
                从左侧目录双击数据集（或按住展开后点 +），图层会叠到地图上；
                地图始终占主区域，不因空态缩成卡片。
              </p>
              <div className="flex flex-wrap justify-center gap-1.5">
                <button
                  type="button"
                  onClick={onOpenCatalog}
                  className="h-[26px] rounded-[var(--gis-radius)] border border-primary bg-primary px-2.5 text-[11.5px] text-primary-foreground hover:opacity-90"
                >
                  打开目录
                </button>
                <button
                  type="button"
                  onClick={onRetryCatalog}
                  className="inline-flex h-[26px] items-center gap-1 rounded-[var(--gis-radius)] border border-border bg-card px-2.5 text-[11.5px] text-foreground/80 hover:bg-accent"
                >
                  <RefreshCw className="size-3" /> 重新扫描目录
                </button>
              </div>
              {catalogStatus === 'ready' ? (
                <p className="mt-2 text-[10.5px] text-muted-foreground">
                  catalog 共 {catalogCount} 项
                  {catalogTruncated ? '（只加载了前 2000 项）' : ''}
                </p>
              ) : null}
            </div>
          )}
        </div>
      ) : null}

      {layers.length > 0 && loadingPreviews > 0 ? (
        <div className="pointer-events-none absolute bottom-8 left-1/2 z-10 -translate-x-1/2 rounded-full border border-border bg-card/95 px-2.5 py-1 text-[10.5px] text-muted-foreground shadow-[var(--gis-shadow-float)]">
          <Loader2 className="mr-1 inline size-3 animate-spin" />
          正在读取 {loadingPreviews} 个图层的几何…
        </div>
      ) : null}
    </div>
  )
}

/** 一个图层的填充/描边/点色：单一色或数据驱动表达式（唯一值 match / 分级 step）。 */
function paintColor(
  style: GisLayerStyle,
  fallback: string
): string | maplibregl.ExpressionSpecification {
  if (style.mode === 'unique' && style.uniqueField) {
    const entries = Object.entries(style.categoryColors)
    if (entries.length === 0) return fallback
    // 类目值统一转字符串比较：distinct 可能回数字（1/2/3），GeoJSON 属性也可能。
    const expression: unknown[] = ['match', ['to-string', ['get', style.uniqueField]]]
    for (const [key, color] of entries) expression.push(key, color)
    expression.push(fallback)
    return expression as unknown as maplibregl.ExpressionSpecification
  }
  if (
    style.mode === 'graduated' &&
    style.graduatedField &&
    style.graduatedColors.length > 1 &&
    style.graduatedBreaks.length >= style.graduatedColors.length + 1
  ) {
    // step(input, c0, b1, c1, b2, c2, …)：首尾断点是真实 min/max，中间的是分位点。
    const expression: unknown[] = [
      'step',
      ['to-number', ['get', style.graduatedField], 0],
      style.graduatedColors[0],
    ]
    for (let index = 1; index < style.graduatedColors.length; index += 1) {
      expression.push(style.graduatedBreaks[index], style.graduatedColors[index])
    }
    return expression as unknown as maplibregl.ExpressionSpecification
  }
  return style.fillColor ?? fallback
}

/** 加/改一个图层的 source + 4 个子图层；数据没变就不 setData。 */
function syncLayer(
  map: maplibregl.Map,
  layer: GisLayer,
  preview: LayerPreview,
  color: string,
  cache: Map<number, unknown>
) {
  const uid = layer.uid
  const data = preview.geojson as unknown as GeoJSONSourceSpecification['data']
  const source = map.getSource(sourceId(uid)) as maplibregl.GeoJSONSource | undefined
  const style = layer.style
  // 唯一值/分级时描边跟填充同色（除非用户单独指定），否则一个图层里 40 种填充
  // 配一根统一颜色的描边会很难看。
  const fill = paintColor(style, color)
  const stroke =
    style.strokeColor ?? (style.mode === 'single' ? color : fill)

  if (!source) {
    map.addSource(sourceId(uid), { type: 'geojson', data })
    const specs: LayerSpecification[] = [
      {
        id: fillId(uid),
        type: 'fill',
        source: sourceId(uid),
        filter: POLYGON_FILTER,
        paint: {
          'fill-color': fill,
          'fill-opacity': style.opacity,
          // 相邻面的共享边界必须严丝合缝：默认的抗锯齿是在**每个面各自**的边缘上
          // 与透明底混色，两个面各捐一半 alpha，接缝处就透出底色成了“白缝”。
          // 关掉后填充按像素硬边界画，配合下面描边的圆角连接把边界压成一条线。
          'fill-antialias': false,
        },
      },
      {
        id: outlineId(uid),
        type: 'line',
        source: sourceId(uid),
        filter: POLYGON_FILTER,
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
          'line-color': stroke,
          'line-width': style.lineWidth,
          'line-opacity': style.strokeOpacity,
        },
      },
      {
        id: lineId(uid),
        type: 'line',
        source: sourceId(uid),
        filter: LINE_FILTER,
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
          'line-color': stroke,
          'line-width': Math.max(1.4, style.lineWidth),
          'line-opacity': 0.9,
        },
      },
      {
        id: pointId(uid),
        type: 'circle',
        source: sourceId(uid),
        filter: POINT_FILTER,
        paint: {
          'circle-color': fill,
          'circle-radius': style.pointSize,
          'circle-opacity': 0.85,
          'circle-stroke-width': 0.6,
          'circle-stroke-color': '#ffffff',
        },
      },
    ]
    for (const spec of specs) map.addLayer(spec)
    cache.set(uid, preview.geojson)
  } else if (cache.get(uid) !== preview.geojson) {
    source.setData(data)
    cache.set(uid, preview.geojson)
  }

  // 主题切换 / 样式改动：色值与透明度就地更新，不重载图层。
  if (map.getLayer(fillId(uid))) {
    map.setPaintProperty(fillId(uid), 'fill-color', fill)
    map.setPaintProperty(fillId(uid), 'fill-opacity', style.opacity)
    // 旧会话里已建好的图层也要把接缝补上（DAIR-368）。
    map.setPaintProperty(fillId(uid), 'fill-antialias', false)
  }
  if (map.getLayer(outlineId(uid))) {
    map.setPaintProperty(outlineId(uid), 'line-color', stroke)
    map.setPaintProperty(outlineId(uid), 'line-width', style.lineWidth)
    map.setPaintProperty(outlineId(uid), 'line-opacity', style.strokeOpacity)
    map.setLayoutProperty(outlineId(uid), 'line-join', 'round')
    map.setLayoutProperty(outlineId(uid), 'line-cap', 'round')
  }
  if (map.getLayer(lineId(uid))) {
    map.setPaintProperty(lineId(uid), 'line-color', stroke)
    map.setPaintProperty(lineId(uid), 'line-width', Math.max(1.4, style.lineWidth))
    map.setLayoutProperty(lineId(uid), 'line-join', 'round')
    map.setLayoutProperty(lineId(uid), 'line-cap', 'round')
  }
  if (map.getLayer(pointId(uid))) {
    map.setPaintProperty(pointId(uid), 'circle-color', fill)
    map.setPaintProperty(pointId(uid), 'circle-radius', style.pointSize)
  }
  for (const id of [fillId(uid), outlineId(uid), lineId(uid), pointId(uid)]) {
    if (map.getLayer(id)) {
      map.setLayoutProperty(id, 'visibility', layer.visible ? 'visible' : 'none')
    }
  }
}

function removeLayerSet(map: maplibregl.Map, uid: number) {
  // 高亮层也引用同一个 source，必须先于 source 删除，否则 removeSource 会报错。
  for (const id of [
    selPointId(uid),
    selLineId(uid),
    selFillId(uid),
    pointId(uid),
    lineId(uid),
    outlineId(uid),
    fillId(uid),
  ]) {
    if (map.getLayer(id)) map.removeLayer(id)
  }
  if (map.getSource(sourceId(uid))) map.removeSource(sourceId(uid))
}

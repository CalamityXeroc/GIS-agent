import { useEffect, useMemo, useState } from 'react'

import { bboxOf, planarExtentKm, toWGS84, type BBox, type GeoJSONCollection } from './geo'
import { LAYERS, type BasemapMode, type SpikeLayer } from './layers'
import { MapCanvas as LegacyCanvas } from './legacy-canvas'
import { MapView } from './MapView'

const params = new URLSearchParams(window.location.search)
const initialLayer = params.get('layer') ?? LAYERS[0].id
const initialZoomOverride = params.get('z') ? Number(params.get('z')) : null
const initialBasemap = params.get('basemap') as BasemapMode | null
const rawMode = params.get('raw') === '1'

interface Loaded {
  layer: SpikeLayer
  raw: GeoJSONCollection
  rawBbox: BBox | null
  data: GeoJSONCollection
  bbox: BBox | null
  transformed: boolean
}

export function App() {
  const [layerId, setLayerId] = useState(
    LAYERS.some((item) => item.id === initialLayer) ? initialLayer : LAYERS[0].id
  )
  const layer = LAYERS.find((item) => item.id === layerId) ?? LAYERS[0]
  const [basemap, setBasemap] = useState<BasemapMode>(initialBasemap ?? layer.defaultBasemap)
  const [loaded, setLoaded] = useState<Loaded | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showRaw, setShowRaw] = useState(rawMode)

  useEffect(() => {
    let cancelled = false
    setLoaded(null)
    setError(null)
    fetch(layer.file)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        return response.json() as Promise<GeoJSONCollection>
      })
      .then((raw) => {
        if (cancelled) return
        const { data, transformed } = toWGS84(raw, layer.wkid)
        setLoaded({ layer, raw, rawBbox: bboxOf(raw), data, bbox: bboxOf(data), transformed })
      })
      .catch((cause: Error) => {
        if (!cancelled) setError(cause.message)
      })
    return () => {
      cancelled = true
    }
  }, [layer])

  // 切图层时回到该图层默认底图/缩放（URL 参数优先）。
  useEffect(() => {
    if (!initialBasemap) setBasemap(layer.defaultBasemap)
  }, [layer])

  const zoom = initialZoomOverride ?? layer.defaultZoom
  const extent = useMemo(() => (loaded?.rawBbox ? planarExtentKm(loaded.rawBbox) : null), [loaded])

  return (
    <div className="shell">
      <header className="head">
        <div>
          <div className="eyebrow">DAIR-298 · spike（隔离目录，不改主干）</div>
          <h1>Web 端 GIS 级空间数据可视化：现状 vs MapLibre</h1>
        </div>
        <nav className="switches">
          <div className="switch-group">
            <span className="switch-label">图层</span>
            {LAYERS.map((item) => (
              <button
                key={item.id}
                className={item.id === layer.id ? 'chip chip-on' : 'chip'}
                onClick={() => setLayerId(item.id)}
              >
                {item.name}
              </button>
            ))}
          </div>
          <div className="switch-group">
            <span className="switch-label">底图</span>
            {([
              ['offline', '本地快照'],
              ['online', '在线 Carto'],
              ['none', '无底图'],
            ] as Array<[BasemapMode, string]>).map(([mode, label]) => (
              <button
                key={mode}
                className={mode === basemap ? 'chip chip-on' : 'chip'}
                onClick={() => setBasemap(mode)}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="switch-group">
            <span className="switch-label">对照</span>
            <button
              className={showRaw ? 'chip chip-warn chip-on' : 'chip'}
              onClick={() => setShowRaw((value) => !value)}
            >
              停用投影变换（错误示范）
            </button>
          </div>
        </nav>
      </header>

      {error ? <div className="error">样例数据读取失败：{error}</div> : null}

      <main className="panels">
        <section className="panel">
          <div className="panel-head">
            <h2>现状：map-canvas.tsx（逐字复刻）</h2>
            <span className="tag tag-muted">bbox 归一化 · 无底图 · 无缩放</span>
          </div>
          {loaded ? (
            <LegacyCanvas features={loaded.raw.features} bbox={loaded.rawBbox} className="legacy-canvas" />
          ) : (
            <div className="skeleton" />
          )}
          <p className="caption">
            顶点按 bbox 铺满画布：形状能看，但「这块地在哪、多大、相邻什么」全靠猜。源坐标
            wkid={layer.wkid ?? '无'}（{layer.crsLabel}）被当成画布坐标直接使用，后端只给一条
            crs_warning，前端没有任何变换与地理参照。
          </p>
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Spike：MapLibre GL JS + proj4</h2>
            <span className="tag">Web Mercator · 真比例 · 本地底图 · 可缩放</span>
          </div>
          {loaded && loaded.bbox && !showRaw ? (
            <MapView
              key={`${layer.id}-ok`}
              data={loaded.data}
              bbox={loaded.bbox}
              basemap={basemap}
              zoom={zoom}
            />
          ) : null}
          {loaded && showRaw && loaded.rawBbox ? (
            <MapView
              key={`${layer.id}-raw`}
              data={loaded.raw}
              bbox={loaded.rawBbox}
              basemap={basemap}
              zoom={zoom}
            />
          ) : null}
          {!loaded ? <div className="skeleton" /> : null}
          <p className="caption">
            {showRaw
              ? `停用变换后直接喂 MapLibre：坐标是「米」（如 ${layer.wkid ? '447114, 4425294' : '—'}），
                 超出 ±180 经度范围，渲染不出任何地理图形——这正是现状在前端“画得出来、却对不上地图”的原因。`
              : layer.story}
          </p>
        </section>
      </main>

      <section className="facts">
        <Fact label="落点" was="无（米被当经纬度）" now={loaded?.bbox ? formatCenter(loaded.bbox) : '—'} />
        <Fact label="图幅真实尺寸" was="—" now={extent ? `${extent.widthKm} × ${extent.heightKm} km` : '—'} />
        <Fact
          label="底图 / 比例尺"
          was="无 / 无"
          now={basemap === 'none' ? '无底图（仍可量算）/ 有' : basemap === 'offline' ? '本地瓦片 / 有' : '在线 Carto / 有'}
        />
        <Fact label="缩放" was="固定 4×4 网格" now={`z3–18（当前 ${zoom.toFixed(1)}）`} />
        <Fact
          label="要素上限"
          was={`${layer.returnedFeatures} 条（400 / 5MB 抽稀）`}
          now={`GeoJSON ${layer.returnedFeatures} 条 → 大图层走 PMTiles`}
        />
        <Fact label="真实要素数" was="后端 catalog" now={layer.featureCount ? `${layer.featureCount.toLocaleString()} 条` : '未登记'} />
      </section>

      <footer className="foot">
        <span>
          数据：本项目 workspace 图层预览缓存（真实图层，见 README §数据来源）；投影参数用 pyproj
          核对（EPSG:4548 → +proj=tmerc … +lon_0=117 +k=1）。
        </span>
        <span className="mono">npm run dev → http://localhost:4318</span>
      </footer>
    </div>
  )
}

function formatCenter(bbox: BBox): string {
  return `${((bbox[0] + bbox[2]) / 2).toFixed(4)}E, ${((bbox[1] + bbox[3]) / 2).toFixed(4)}N`
}

function Fact({ label, was, now }: { label: string; was: string; now: string }) {
  return (
    <div className="fact">
      <div className="fact-label">{label}</div>
      <div className="fact-was">{was}</div>
      <div className="fact-now">{now}</div>
    </div>
  )
}
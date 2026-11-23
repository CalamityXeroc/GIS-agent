import { create } from 'zustand'

import type { DatasetItem } from '@/lib/api'

/**
 * GIS 工作台前端状态（DAIR-325）。
 *
 * 边界：
 * - 只存「前端会话态」——图层集合与顺序、显隐、dock 尺寸、地图视图。
 *   服务端数据仍走 `@tanstack/react-query`，不在这里缓存。
 * - 本轮没有工程/样式持久化接口，所以刷新即丢；这一点在 UI 上明说。
 * - dock 的**默认值**来自 `theme.css` 的 `--gis-dock-*`（断点也在 CSS 里），
 *   拖拽后的值才落到 store；`touched` 记录哪些被用户改过，窗口 resize 时
 *   只同步没被改过的那几块。
 */

export type DockKey = 'left' | 'right' | 'table' | 'ratio'

/** 右 dock 四个 tab（spec §4 ④⑤）：选中项存 store，切图层不丢。 */
export type RightTab = 'general' | 'symbology' | 'fields' | 'metadata'

/** 矢量渲染器：单一符号 / 唯一值 / 分级色彩。 */
export type RendererMode = 'single' | 'unique' | 'graduated'
/** 地图工具（spec §4 ⑥ 命令栏工具组）：浏览 / 点选 / 框选。 */
export type MapTool = 'pan' | 'select' | 'box'
/** 栅格拉伸方式（真实渲染待后端补，这里只记录用户选择）。 */
export type RasterStretch = 'minmax' | 'std' | 'pct'

/** 图层可见/样式。颜色一律存 `#rrggbb`，地图侧直接用。 */
export interface GisLayerStyle {
  /** `--chart-N` 的下标（0–4），`map-canvas` 读 CSS 变量后转 rgba。 */
  colorIndex: number
  opacity: number
  lineWidth: number
  mode: RendererMode
  /** null = 跟着 `colorIndex` 的色板走；用户改过就是具体色值。 */
  fillColor: string | null
  /** null = 与填充同色。 */
  strokeColor: string | null
  strokeOpacity: number
  /** 点图层的圆点半径（px）。 */
  pointSize: number
  /** 唯一值：类目来自 `/api/catalog/distinct`，颜色按类目名存。 */
  uniqueField: string | null
  categoryColors: Record<string, string>
  /** 分级：断点来自 distinct 的 quantile。 */
  graduatedField: string | null
  graduatedClasses: number
  /** 由色带按级数采样的具体色值（空 = 还没算）。 */
  graduatedColors: string[]
  /** 分位断点（长度 = 级数+1，来自 distinct 的 quantile）；地图侧 step 表达式用。 */
  graduatedBreaks: number[]
  /** 栅格：R/G/B 波段号（1 起）、拉伸、伽马、色带名。 */
  bandR: number
  bandG: number
  bandB: number
  stretch: RasterStretch
  gamma: number
  ramp: string
}

export function defaultLayerStyle(index: number): GisLayerStyle {
  return {
    // 每加一层换一个色，叠在一起还能分清谁是谁。
    colorIndex: index % LAYER_PALETTE.length,
    opacity: 0.32,
    lineWidth: 1.2,
    mode: 'single',
    fillColor: null,
    strokeColor: null,
    strokeOpacity: 0.85,
    pointSize: 3.5,
    uniqueField: null,
    categoryColors: {},
    graduatedField: null,
    graduatedClasses: 5,
    graduatedColors: [],
    graduatedBreaks: [],
    bandR: 1,
    bandG: 2,
    bandB: 3,
    stretch: 'minmax',
    gamma: 1,
    ramp: 'terrain',
  }
}

/** 渲染器实际要随预览带回的字段（最多两个：唯一值 + 分级）。 */
export function styleFields(style: GisLayerStyle): string[] {
  const fields: string[] = []
  if (style.uniqueField) fields.push(style.uniqueField)
  if (style.graduatedField && style.graduatedField !== style.uniqueField) {
    fields.push(style.graduatedField)
  }
  return fields
}

export interface GisLayer {
  /** MapLibre 图层/数据源 ID 用的稳定数字；路径做 id 会带反斜杠与中文。 */
  uid: number
  path: string
  name: string
  kind: string
  item: DatasetItem
  visible: boolean
  style: GisLayerStyle
}

export interface MapViewState {
  zoom: number
  scale: number | null
  cursor: [number, number] | null
  ready: boolean
  /** 首次取景时的缩放级别；状态栏「缩放 1.00×」以它为基准。 */
  baseZoom: number | null
}

export type MapCommandKind = 'zoom-in' | 'zoom-out' | 'fit-all' | 'fit-layer'

export interface MapCommand {
  seq: number
  kind: MapCommandKind
  uid?: number
}

interface DockSizes {
  left: number
  right: number
  table: number
  ratio: number
}

interface GisState {
  layers: GisLayer[]
  selectedLayerId: number | null
  selectedDatasetPath: string | null
  /** 属性表 tab（存路径；图层被移除时一并清掉）。 */
  tableTabs: string[]
  tableTab: string | null
  /** 每个图层的要素标识字段（从 `/api/catalog/rows` 字段头学到的 OID 列）。 */
  idFields: Record<string, string>
  /**
   * 每个图层的地图选择集：存 idField 的**字符串值**（map 表达式按字符串比较）。
   * 地图点选/框选写入，属性表按同一份键高亮/过滤，两边不各存一份。
   */
  selection: Record<string, string[]>
  mapTool: MapTool
  docks: DockSizes & { leftOpen: boolean; rightOpen: boolean; tableOpen: boolean }
  touched: Record<DockKey, boolean>
  view: MapViewState
  command: MapCommand | null
  rightTab: RightTab

  addDataset: (item: DatasetItem) => void
  removeLayer: (uid: number) => void
  toggleLayer: (uid: number) => void
  moveLayer: (uid: number, toIndex: number) => void
  selectLayer: (uid: number | null) => void
  selectDataset: (path: string | null) => void
  openTable: (path: string) => void
  closeTable: (path: string) => void
  setTableTab: (path: string) => void
  setMapTool: (tool: MapTool) => void
  setIdField: (path: string, field: string | null) => void
  setSelection: (path: string, ids: string[]) => void
  toggleFeature: (path: string, id: string, additive: boolean) => void
  clearSelection: (path?: string) => void
  setRightTab: (tab: RightTab) => void
  setLayerStyle: (uid: number, patch: Partial<GisLayerStyle>) => void
  setCategoryColor: (uid: number, key: string, color: string) => void
  resetLayerStyle: (uid: number) => void

  setDock: (key: DockKey, value: number) => void
  resetDock: (key: DockKey) => void
  toggleDock: (key: 'left' | 'right' | 'table') => void
  toggleAllDocks: () => void
  syncDockDefaults: () => void

  setView: (patch: Partial<MapViewState>) => void
  sendCommand: (kind: MapCommandKind, uid?: number) => void
}

/** 可拖范围照 spec §2：左 200–430、右 248–430、属性表 110px–62% 视口。 */
export const DOCK_LIMITS = {
  left: { min: 200, max: 430 },
  right: { min: 248, max: 430 },
  table: { min: 110, max: () => window.innerHeight * 0.62 },
}

function dockLimit(key: 'left' | 'right' | 'table'): { min: number; max: number } {
  const limit = DOCK_LIMITS[key]
  return { min: limit.min, max: typeof limit.max === 'function' ? limit.max() : limit.max }
}

/** 目录 ≥96px、图层 ≥160px（spec §2）；比例 = 图层面板占左 dock 的高度。 */
export const PANE_LIMITS = { catalog: 96, layers: 160 } as const

/** 图层默认色从 `--chart-N` 里取（浅/深主题各有一套值）。 */
export const LAYER_PALETTE = ['--chart-1', '--chart-2', '--chart-3', '--chart-4', '--chart-5']

function cssNumber(name: string, fallback: number): number {
  if (typeof window === 'undefined') return fallback
  const raw = window.getComputedStyle(document.documentElement).getPropertyValue(name)
  const value = Number.parseFloat(raw)
  return Number.isFinite(value) ? value : fallback
}

function cssRatio(name: string, fallback: number): number {
  if (typeof window === 'undefined') return fallback
  const raw = window
    .getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim()
  const value = Number.parseFloat(raw)
  if (!Number.isFinite(value)) return fallback
  return raw.endsWith('%') ? value / 100 : value
}

/** 默认 dock 尺寸只认 CSS token：断点写在 `theme.css`，这里不重复一份。 */
function defaultDocks(): DockSizes {
  return {
    left: cssNumber('--gis-dock-left', 278),
    right: cssNumber('--gis-dock-right', 312),
    table: cssNumber('--gis-dock-table', 236),
    ratio: cssRatio('--gis-layer-ratio', 0.56),
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

/**
 * 目录树/图层预览都只支持带几何的矢量。
 *
 * - `vector`：.shp / .geojson / .gpkg / .kml。**不看 geom_type** —— .geojson 在
 *   catalog 里的 geom_type 本来就是空的，按它判会把整个 geojson 判死。
 * - `feature_class`：.gdb 里的要素类。geom_type 为空说明它是非空间表
 *   （旧 catalog 把 .gdb 里的表也标成了 feature_class），上图必报
 *   "table has no geometry"。
 */
export function isPreviewable(item: Pick<DatasetItem, 'kind' | 'geom_type'>): boolean {
  if (item.kind === 'vector') return true
  return item.kind === 'feature_class' && Boolean(item.geom_type)
}

/** 非空间表：有属性、没有几何 —— 只能看属性表，不进图层栈。 */
export function isNonSpatialTable(item: Pick<DatasetItem, 'kind' | 'geom_type'>): boolean {
  if (item.kind === 'table') return true
  return item.kind === 'feature_class' && !item.geom_type
}

let nextUid = 1

export const useGisStore = create<GisState>((set, get) => {
  const defaults = defaultDocks()

  return {
    layers: [],
    selectedLayerId: null,
    selectedDatasetPath: null,
    tableTabs: [],
    tableTab: null,
    idFields: {},
    selection: {},
    mapTool: 'pan',
    docks: { ...defaults, leftOpen: true, rightOpen: true, tableOpen: true },
    touched: { left: false, right: false, table: false, ratio: false },
    view: { zoom: 0, scale: null, cursor: null, ready: false, baseZoom: null },
    command: null,
    rightTab: 'general',

    addDataset: (item) => {
      const existing = get().layers.find((layer) => layer.path === item.path)
      if (existing) {
        get().selectLayer(existing.uid)
        if (isPreviewable(existing.item)) get().openTable(existing.path)
        return
      }
      const layer: GisLayer = {
        uid: nextUid++,
        path: item.path,
        name: item.name,
        kind: item.kind,
        item,
        visible: true,
        style: defaultLayerStyle(get().layers.length),
      }
      set((state) => ({
        // 栈顶 = 最后绘制：新图层插到列表最上面。
        layers: [layer, ...state.layers],
        selectedLayerId: layer.uid,
        selectedDatasetPath: layer.path,
      }))
      if (isPreviewable(item)) get().openTable(item.path)
    },

    removeLayer: (uid) => {
      const { layers, selectedLayerId, tableTab } = get()
      const target = layers.find((layer) => layer.uid === uid)
      if (!target) return
      const remaining = layers.filter((layer) => layer.uid !== uid)
      const tableTabs = get().tableTabs.filter((path) => path !== target.path)
      // 删图层要连选择集一起清：留下悬空的 id 会在表里高亮到不存在的行。
      const selection = { ...get().selection }
      const idFields = { ...get().idFields }
      if (!remaining.some((layer) => layer.path === target.path)) {
        delete selection[target.path]
        delete idFields[target.path]
      }
      // 删掉的正好是属性表 tab：切到相邻 tab，没有就关掉内容区。
      const nextTab =
        tableTab === target.path ? (tableTabs[0] ?? null) : tableTab
      const nextSelected =
        selectedLayerId === uid ? (remaining[0]?.uid ?? null) : selectedLayerId
      set({
        layers: remaining,
        tableTabs,
        tableTab: nextTab,
        selection,
        idFields,
        selectedLayerId: nextSelected,
        selectedDatasetPath:
          selectedLayerId === uid
            ? (remaining.find((layer) => layer.uid === nextSelected)?.path ?? null)
            : get().selectedDatasetPath,
      })
    },

    toggleLayer: (uid) =>
      set((state) => ({
        layers: state.layers.map((layer) =>
          layer.uid === uid ? { ...layer, visible: !layer.visible } : layer
        ),
      })),

    moveLayer: (uid, toIndex) => {
      const { layers } = get()
      const from = layers.findIndex((layer) => layer.uid === uid)
      if (from < 0) return
      const target = clamp(toIndex, 0, layers.length - 1)
      if (from === target) return
      const next = [...layers]
      const [moved] = next.splice(from, 1)
      next.splice(target, 0, moved)
      set({ layers: next })
    },

    selectLayer: (uid) => {
      if (uid === null) {
        set({ selectedLayerId: null, selectedDatasetPath: null })
        return
      }
      const layer = get().layers.find((item) => item.uid === uid)
      set({ selectedLayerId: uid, selectedDatasetPath: layer?.path ?? null })
    },

    selectDataset: (path) => {
      if (path === null) {
        set({ selectedDatasetPath: null, selectedLayerId: null })
        return
      }
      const layer = get().layers.find((item) => item.path === path)
      set({
        selectedDatasetPath: path,
        selectedLayerId: layer?.uid ?? null,
      })
    },

    openTable: (path) => {
      const { tableTabs } = get()
      set({
        tableTabs: tableTabs.includes(path) ? tableTabs : [...tableTabs, path].slice(-6),
        tableTab: path,
        docks: { ...get().docks, tableOpen: true },
      })
    },

    closeTable: (path) => {
      const { tableTabs, tableTab } = get()
      const index = tableTabs.indexOf(path)
      const remaining = tableTabs.filter((item) => item !== path)
      set({
        tableTabs: remaining,
        tableTab:
          tableTab === path
            ? (remaining[Math.max(0, index - 1)] ?? remaining[0] ?? null)
            : tableTab,
      })
    },

    setTableTab: (path) => set({ tableTab: path }),

    setMapTool: (tool) => set({ mapTool: tool }),

    setIdField: (path, field) => {
      const current = get().idFields[path] ?? null
      if (current === field || (!field && !current)) return
      const idFields = { ...get().idFields }
      if (field) idFields[path] = field
      else delete idFields[path]
      set({ idFields })
    },

    setSelection: (path, ids) =>
      set({ selection: { ...get().selection, [path]: ids } }),

    toggleFeature: (path, id, additive) => {
      const current = get().selection[path] ?? []
      let next: string[]
      if (!additive) {
        next = current.length === 1 && current[0] === id ? [] : [id]
      } else if (current.includes(id)) {
        next = current.filter((item) => item !== id)
      } else {
        next = [...current, id]
      }
      set({ selection: { ...get().selection, [path]: next } })
    },

    clearSelection: (path) => {
      const selection = { ...get().selection }
      if (path) delete selection[path]
      else for (const key of Object.keys(selection)) delete selection[key]
      set({ selection })
    },

    setRightTab: (tab) => set({ rightTab: tab }),

    setLayerStyle: (uid, patch) =>
      set((state) => ({
        layers: state.layers.map((layer) =>
          layer.uid === uid ? { ...layer, style: { ...layer.style, ...patch } } : layer
        ),
      })),

    setCategoryColor: (uid, key, color) =>
      set((state) => ({
        layers: state.layers.map((layer) =>
          layer.uid === uid
            ? {
                ...layer,
                style: {
                  ...layer.style,
                  categoryColors: { ...layer.style.categoryColors, [key]: color },
                },
              }
            : layer
        ),
      })),

    resetLayerStyle: (uid) =>
      set((state) => ({
        layers: state.layers.map((layer) =>
          layer.uid === uid ? { ...layer, style: defaultLayerStyle(layer.style.colorIndex) } : layer
        ),
      })),

    setDock: (key, value) => {
      const { docks } = get()
      if (key === 'ratio') {
        set({
          docks: { ...docks, ratio: clamp(value, 0.2, 0.8) },
          touched: { ...get().touched, ratio: true },
        })
        return
      }
      const limit = dockLimit(key)
      set({
        docks: { ...docks, [key]: clamp(value, limit.min, limit.max) },
        touched: { ...get().touched, [key]: true },
      })
    },

    resetDock: (key) => {
      const fresh = defaultDocks()
      set({
        docks: { ...get().docks, [key]: fresh[key] },
        touched: { ...get().touched, [key]: false },
      })
    },

    toggleDock: (key) => {
      const docks = get().docks
      set({ docks: { ...docks, [`${key}Open`]: !docks[`${key}Open`] } })
    },

    toggleAllDocks: () => {
      const { leftOpen, rightOpen, tableOpen } = get().docks
      const open = !(leftOpen && rightOpen && tableOpen)
      set({
        docks: { ...get().docks, leftOpen: open, rightOpen: open, tableOpen: open },
      })
    },

    syncDockDefaults: () => {
      const fresh = defaultDocks()
      const { docks, touched } = get()
      set({
        docks: {
          ...docks,
          left: touched.left ? docks.left : fresh.left,
          right: touched.right ? docks.right : fresh.right,
          table: touched.table ? docks.table : fresh.table,
          ratio: touched.ratio ? docks.ratio : fresh.ratio,
        },
      })
    },

    setView: (patch) => set({ view: { ...get().view, ...patch } }),

    sendCommand: (kind, uid) =>
      set({
        command: { seq: (get().command?.seq ?? 0) + 1, kind, uid },
      }),
  }
})

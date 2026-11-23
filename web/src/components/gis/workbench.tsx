import {
  BoxSelect,
  ChevronsDownUp,
  Crosshair,
  FolderTree,
  Layers,
  Loader2,
  Map,
  Maximize2,
  Menu,
  MousePointer2,
  Moon,
  PanelLeft,
  Plus,
  RefreshCw,
  Ruler,
  ScanSearch,
  Sun,
  Table2,
  ZoomIn,
  ZoomOut,
  type LucideIcon,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'

import { NavSheet } from '@/components/layout/nav-sheet'
import type { DatasetItem } from '@/lib/api'
import { useThemeStore } from '@/lib/theme'
import { cn } from '@/lib/utils'

import { AttributeTableDock } from './attribute-table'
import { CatalogTree } from './catalog-tree'
import { formatNumber } from './format'
import { LayerStack } from './layer-stack'
import { MapStage, type CatalogStatus } from './map-canvas'
import { RightDock } from './right-dock'
import { Splitter } from './splitter'
import { StatusBar } from './status-bar'
import { DOCK_LIMITS, isNonSpatialTable, isPreviewable, useGisStore } from './store'
import { buildCatalogTree } from './tree'
import { useCatalog } from './use-catalog'
import { useLayerPreviews } from './use-layer-previews'

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function IconButton({
  icon: Icon,
  title,
  onClick,
  disabled,
  spinning,
  danger,
}: {
  icon: LucideIcon
  title: string
  onClick?: () => void
  disabled?: boolean
  spinning?: boolean
  danger?: boolean
}) {
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'grid size-5 place-items-center rounded-[3px] text-muted-foreground hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40',
        danger && 'hover:text-destructive'
      )}
    >
      <Icon className={cn('size-3', spinning && 'animate-spin')} />
    </button>
  )
}

function ToolButton({
  icon: Icon,
  label,
  title,
  onClick,
  active,
  disabled,
}: {
  icon: LucideIcon
  label?: string
  title: string
  onClick?: () => void
  active?: boolean
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      aria-pressed={active}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'inline-flex h-[26px] shrink-0 items-center gap-1 rounded-[var(--gis-radius)] px-1.5 text-[11.5px] transition-colors',
        active
          ? 'bg-accent text-primary shadow-[inset_0_0_0_1px_color-mix(in_oklab,var(--primary)_26%,transparent)]'
          : 'text-foreground/80 hover:bg-accent',
        disabled && 'cursor-not-allowed text-muted-foreground opacity-55 hover:bg-transparent'
      )}
    >
      <Icon className="size-3.5" strokeWidth={1.7} />
      {label ? <span className="hidden sm:inline">{label}</span> : null}
    </button>
  )
}

function Divider() {
  return <span className="mx-0.5 h-[18px] w-px shrink-0 bg-border" aria-hidden="true" />
}

/** 属性表 dock 的开/收：命令栏按钮与 Ctrl+T 必须同一套语义。 */
function toggleTableDock(state: ReturnType<typeof useGisStore.getState>) {
  if (state.docks.tableOpen) {
    state.toggleDock('table')
    return
  }
  const layer =
    state.layers.find((item) => item.uid === state.selectedLayerId) ?? state.layers[0]
  if (layer && isPreviewable(layer.item)) state.openTable(layer.path)
  else state.toggleDock('table')
}

function PanelHead({
  icon: Icon,
  title,
  count,
  children,
}: {
  icon: LucideIcon
  title: string
  count?: number
  children?: ReactNode
}) {
  return (
    <div className="flex h-[var(--gis-panel-head)] shrink-0 items-center gap-1.5 border-b border-border bg-card px-2 text-[11px] font-semibold tracking-wide text-muted-foreground select-none">
      <Icon className="size-3 shrink-0" />
      {title}
      {count !== undefined ? (
        <span className="rounded-full bg-muted px-1.5 text-[9.5px] leading-[15px] font-medium tabular-nums">
          {formatNumber(count)}
        </span>
      ) : null}
      <div className="ml-auto flex items-center gap-0.5">{children}</div>
    </div>
  )
}

/**
 * 「数据资产」GIS 工作台（spec v1）：命令栏 / 视图 tab / 左 dock（目录+图层）/
 * 地图 / 右 dock / 属性表 dock / 状态栏。
 *
 * 布局约束：页面不整页滚动，地图永远是主区；dock 尺寸走 store，默认值与断点
 * 在 `theme.css`。数据只来自既有只读接口（`/api/catalog/*`），本轮不动后端。
 *
 * 沉浸式（DAIR-345）：这一页由 `AppShell` 去掉顶栏与导航侧栏，命令栏即页面顶栏，
 * 目录树贴 x=0；导航改由命令栏左侧的 ☰ 打开抽屉。
 */
export function GisWorkbench() {
  const catalog = useCatalog()
  const layers = useGisStore((state) => state.layers)
  const docks = useGisStore((state) => state.docks)
  const selectedLayerId = useGisStore((state) => state.selectedLayerId)
  const selectedDatasetPath = useGisStore((state) => state.selectedDatasetPath)
  const addDataset = useGisStore((state) => state.addDataset)
  const openTable = useGisStore((state) => state.openTable)
  const selectDataset = useGisStore((state) => state.selectDataset)
  const setDock = useGisStore((state) => state.setDock)
  const resetDock = useGisStore((state) => state.resetDock)
  const toggleDock = useGisStore((state) => state.toggleDock)
  const toggleAllDocks = useGisStore((state) => state.toggleAllDocks)
  const syncDockDefaults = useGisStore((state) => state.syncDockDefaults)
  const sendCommand = useGisStore((state) => state.sendCommand)
  const mapTool = useGisStore((state) => state.mapTool)
  const setMapTool = useGisStore((state) => state.setMapTool)

  const [collapseSeq, setCollapseSeq] = useState(0)
  const [navOpen, setNavOpen] = useState(false)
  const theme = useThemeStore((state) => state.theme)
  const toggleTheme = useThemeStore((state) => state.toggle)
  const leftDockRef = useRef<HTMLElement | null>(null)
  const dragStart = useRef(0)
  const previews = useLayerPreviews(layers)

  const items = catalog.data?.items ?? []
  const catalogStatus: CatalogStatus = catalog.isError
    ? 'error'
    : catalog.isLoading
      ? 'loading'
      : 'ready'

  const tree = useMemo(() => (items.length > 0 ? buildCatalogTree(items) : null), [items])
  const addedPaths = useMemo(() => new Set(layers.map((layer) => layer.path)), [layers])

  const selectedItem = useMemo<DatasetItem | null>(() => {
    if (selectedDatasetPath) {
      const fromCatalog = items.find((item) => item.path === selectedDatasetPath)
      if (fromCatalog) return fromCatalog
    }
    if (selectedLayerId !== null) {
      return layers.find((layer) => layer.uid === selectedLayerId)?.item ?? null
    }
    return null
  }, [items, layers, selectedDatasetPath, selectedLayerId])

  const activeLayer =
    layers.find((layer) => layer.uid === selectedLayerId) ?? layers[0] ?? null

  /** 选中项在图层栈里的那一层（目录里点一个没上图的数据集时为 null）。 */
  const selectedLayer = useMemo(
    () => (selectedItem ? (layers.find((layer) => layer.path === selectedItem.path) ?? null) : null),
    [layers, selectedItem]
  )

  /** 预览首要素的几何类型：catalog 对 .geojson 的 geom_type 是空的。 */
  const selectedGeometry = useMemo(() => {
    if (!selectedItem) return null
    const preview = previews[selectedItem.path]?.preview
    const first = preview?.geojson.features.find((feature) => feature.geometry)
    return first?.geometry?.type ?? selectedItem.geom_type ?? null
  }, [previews, selectedItem])

  // 断点变化时，没被用户拖过的那几块跟着 CSS token 走。
  useEffect(() => {
    const onResize = () => syncDockDefaults()
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [syncDockDefaults])

  const openCatalog = useCallback(() => {
    if (!useGisStore.getState().docks.leftOpen) toggleDock('left')
  }, [toggleDock])

  /** 双击 / 行内 +：非空间表没有几何，直接开属性表，不进图层栈。 */
  const handleAdd = useCallback(
    (item: DatasetItem) => {
      if (isNonSpatialTable(item)) {
        selectDataset(item.path)
        openTable(item.path)
        return
      }
      addDataset(item)
    },
    [addDataset, openTable, selectDataset]
  )

  const toggleTable = useCallback(() => {
    toggleTableDock(useGisStore.getState())
  }, [])

  // 键盘：Esc 取消选择、Ctrl+T 属性表、Ctrl+D 添加数据、F 缩放到选中图层。
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      const typing = target
        ? target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable
        : false
      const state = useGisStore.getState()
      if (event.key === 'Escape' && !typing) {
        // 一次 Esc 只做一件事：先清选择集，再退工具，最后退图层选中。
        const hasSelection = Object.values(state.selection).some((ids) => ids.length > 0)
        if (hasSelection) {
          state.clearSelection()
          return
        }
        if (state.mapTool !== 'pan') {
          state.setMapTool('pan')
          return
        }
        state.selectLayer(null)
        return
      }
      const modifier = event.ctrlKey || event.metaKey
      if (modifier && event.key.toLowerCase() === 't') {
        event.preventDefault()
        // 和命令栏按钮共用一套语义：开着就收，没开就开当前图层。
        toggleTableDock(state)
        return
      }
      if (modifier && event.key.toLowerCase() === 'd') {
        event.preventDefault()
        if (!state.docks.leftOpen) state.toggleDock('left')
        return
      }
      if (!typing && !modifier && event.key.toLowerCase() === 'f' && state.layers.length > 0) {
        const layer =
          state.layers.find((item) => item.uid === state.selectedLayerId) ?? state.layers[0]
        if (layer) state.sendCommand('fit-layer', layer.uid)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const handleZoomAll = () => sendCommand('fit-all')

  const catalogCount = items.length

  return (
    <div
      className="flex h-dvh flex-col overflow-hidden bg-background text-foreground"
      style={{ fontSize: 'var(--gis-font-body)' }}
      data-testid="gis-workbench"
    >
      {/* 命令栏 40px：沉浸式工作台自己就是页面顶栏（DAIR-345）。 */}
      <div
        data-testid="gis-command-bar"
        className="flex h-[var(--gis-command-bar)] shrink-0 items-center gap-0.5 overflow-x-auto border-b border-border bg-card px-2 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
      >
        <button
          type="button"
          aria-label="打开导航"
          title="打开导航"
          onClick={() => setNavOpen(true)}
          className="mr-0.5 grid size-6 shrink-0 place-items-center rounded-[3px] text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          <Menu className="size-4" />
        </button>
        <div className="mr-1 flex shrink-0 items-baseline gap-2 px-1">
          <h1 className="text-[13px] font-semibold tracking-tight">数据资产</h1>
          <span className="hidden text-[10.5px] text-muted-foreground md:inline">
            workspace · {layers.length} 图层 / {formatNumber(catalogCount)} 项
          </span>
        </div>
        <Divider />
        <ToolButton
          icon={MousePointer2}
          title="选择工具：单击要素选中（Shift 加选）；浏览态左键平移、滚轮缩放"
          active={mapTool === 'select'}
          onClick={() => setMapTool(mapTool === 'select' ? 'pan' : 'select')}
        />
        <ToolButton
          icon={BoxSelect}
          title="框选：在地图上拖一个矩形选中要素（Shift 加选）"
          active={mapTool === 'box'}
          onClick={() => setMapTool(mapTool === 'box' ? 'pan' : 'box')}
        />
        <Divider />
        <ToolButton
          icon={ZoomIn}
          title="放大（滚轮以光标为锚点）"
          onClick={() => sendCommand('zoom-in')}
        />
        <ToolButton icon={ZoomOut} title="缩小" onClick={() => sendCommand('zoom-out')} />
        <ToolButton
          icon={Maximize2}
          title="全图（缩放到所有已上图图层）"
          onClick={handleZoomAll}
          disabled={layers.length === 0}
        />
        <Divider />
        <ToolButton icon={Ruler} title="测量距离 / 面积（本期留空）" disabled />
        <ToolButton icon={ScanSearch} title="识别要素（后续单接入）" disabled />
        <ToolButton
          icon={Table2}
          label="属性表"
          title="显示 / 收起属性表（Ctrl+T）"
          onClick={toggleTable}
        />
        <Divider />
        <ToolButton
          icon={Plus}
          label="添加数据"
          title="从目录添加数据（Ctrl+D）"
          onClick={openCatalog}
        />
        <ToolButton
          icon={PanelLeft}
          label="面板"
          title="显示 / 隐藏左、右、底三块面板"
          onClick={toggleAllDocks}
          active={!docks.leftOpen || !docks.rightOpen || !docks.tableOpen}
        />
        <Divider />
        <ToolButton
          icon={Crosshair}
          title="缩放到选中图层（F）"
          disabled={!activeLayer}
          onClick={() => {
            if (activeLayer) sendCommand('fit-layer', activeLayer.uid)
          }}
        />
        <span className="flex-1" />
        <div className="flex shrink-0 items-center gap-1 pl-2">
          <IconButton
            icon={RefreshCw}
            title="刷新 catalog（重新扫描 input/）"
            onClick={() => void catalog.refresh()}
            spinning={catalog.isScanning || catalog.isFetching}
          />
          <span className="rounded-full bg-muted px-1.5 py-0.5 text-[10px] tabular-nums text-muted-foreground">
            catalog {formatNumber(catalogCount)}
          </span>
          <button
            type="button"
            onClick={toggleTheme}
            aria-label={theme === 'dark' ? '切换到浅色模式' : '切换到深色模式'}
            title={theme === 'dark' ? '切换到浅色模式' : '切换到深色模式'}
            className="grid size-6 shrink-0 place-items-center rounded-[3px] text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            {theme === 'dark' ? <Sun className="size-3.5" /> : <Moon className="size-3.5" />}
          </button>
        </div>
      </div>

      {/* 主区：左 dock / 地图 / 右 dock */}
      <div className="flex min-h-0 flex-1">
        {docks.leftOpen ? (
          <aside
            ref={leftDockRef}
            style={{ width: docks.left }}
            className="flex min-w-0 shrink-0 flex-col overflow-hidden border-r border-border bg-background"
            aria-label="目录与图层"
          >
            <div className="flex min-h-0 flex-1 flex-col">
              <PanelHead icon={FolderTree} title="目录" count={tree?.count ?? catalogCount}>
                <IconButton
                  icon={ChevronsDownUp}
                  title="全部折叠（保留根节点）"
                  onClick={() => setCollapseSeq((value) => value + 1)}
                />
                <IconButton
                  icon={RefreshCw}
                  title="刷新目录（重新扫描 input/）"
                  onClick={() => void catalog.refresh()}
                  spinning={catalog.isScanning || catalog.isFetching}
                />
              </PanelHead>
              <div className="min-h-0 flex-1 overflow-auto px-1 pb-1.5">
                {catalog.isScanning ? (
                  <p
                    className="flex items-center gap-1.5 px-2 pt-1.5 text-[10.5px] text-muted-foreground"
                    data-testid="catalog-scanning"
                  >
                    <Loader2 className="size-3 shrink-0 animate-spin" />
                    正在扫描数据目录… 已发现 {formatNumber(catalog.scan?.discovered ?? 0)} 项
                  </p>
                ) : catalog.scanError ? (
                  <p
                    className="px-2 pt-1.5 text-[10.5px] leading-relaxed text-amber-600"
                    data-testid="catalog-scan-error"
                  >
                    扫描失败：{catalog.scanError}
                    {catalog.scan?.errors?.[0] ? `（${catalog.scan.errors[0]}）` : ''}
                    {catalog.stale
                      ? '（下面的目录是上次结果，已标为「已过期」，点刷新重试）'
                      : '（目录暂不可用，修好后会自动重扫）'}
                  </p>
                ) : null}
                {catalogStatus === 'loading' ? (
                  <div className="flex flex-col gap-1 px-2 pt-1.5">
                    {Array.from({ length: 7 }).map((_, index) => (
                      <div
                        key={index}
                        className="h-4 animate-pulse rounded-[2px] bg-muted"
                        style={{ width: `${55 + ((index * 13) % 36)}%` }}
                      />
                    ))}
                  </div>
                ) : catalogStatus === 'error' ? (
                  <div className="px-2 py-4 text-center text-[11.5px] leading-relaxed text-destructive">
                    <p>catalog 读取失败</p>
                    <p className="mt-1 text-muted-foreground">
                      {(catalog.error as Error | null)?.message}
                    </p>
                    <button
                      type="button"
                      onClick={() => void catalog.refetch()}
                      className="mt-2 h-[24px] rounded-[var(--gis-radius)] border border-border bg-card px-2 text-[11px] text-foreground/80 hover:bg-accent"
                    >
                      重试
                    </button>
                  </div>
                ) : catalogCount === 0 ? (
                  catalog.scanError ? (
                    <div className="px-3 py-6 text-center text-[11.5px] leading-relaxed text-muted-foreground">
                      <b className="text-amber-600">目录为空：上一次扫描失败</b>
                      <p className="mt-1">
                        没有可用结果就不会拿旧数据冒充；内核恢复或修好数据后会自动重扫。
                      </p>
                    </div>
                  ) : catalog.isScanning ? (
                    <div className="px-3 py-6 text-center text-[11.5px] leading-relaxed text-muted-foreground">
                      <b className="text-foreground/75">正在扫描数据目录…</b>
                      <p className="mt-1">扫到多少就先显示多少，不用等它跑完。</p>
                    </div>
                  ) : (
                    <div className="px-3 py-6 text-center text-[11.5px] leading-relaxed text-muted-foreground">
                      <b className="text-foreground/75">workspace/input 还是空的</b>
                      <p className="mt-1">
                        把 .gdb / .shp / .tif 放进 input/ 后会自动扫描出现，不用点刷新、也不用先跑任务。
                      </p>
                    </div>
                  )
                ) : tree ? (
                  <div
                    aria-label={catalog.stale ? '上次扫描结果（已过期）' : undefined}
                    className={catalog.stale ? 'opacity-55' : undefined}
                  >
                    {catalog.stale ? (
                      <p
                        className="px-2 pb-1 text-[10px] text-amber-600"
                        data-testid="catalog-stale-note"
                      >
                        上次扫描结果 · 已过期
                      </p>
                    ) : null}
                    <CatalogTree
                      root={tree}
                      selectedPath={selectedItem?.path ?? null}
                      addedPaths={addedPaths}
                      collapseSeq={collapseSeq}
                      onSelect={(item) => selectDataset(item.path)}
                      onAdd={handleAdd}
                    />
                  </div>
                ) : null}
                {catalog.data?.truncated ? (
                  <p className="px-2 pt-2 text-[10px] text-muted-foreground">
                    目录超过 2000 项，只加载了前 2000 项。
                  </p>
                ) : null}
              </div>
            </div>

            <Splitter
              axis="y"
              label="调整目录与图层高度（双击复位）"
              valueNow={docks.ratio * 100}
              valueMin={20}
              valueMax={80}
              invert
              onDragStart={() => {
                dragStart.current = docks.ratio
              }}
              onDrag={(delta, total) => {
                const height = leftDockRef.current?.clientHeight ?? total
                const minRatio = Math.min(0.7, 160 / Math.max(height, 1))
                const maxRatio = Math.max(minRatio + 0.05, 1 - 96 / Math.max(height, 1))
                setDock(
                  'ratio',
                  clamp(dragStart.current + delta / Math.max(total, 1), minRatio, maxRatio)
                )
              }}
              onReset={() => resetDock('ratio')}
              onKeyAdjust={(delta) => setDock('ratio', docks.ratio + delta * 0.002)}
            />

            <div
              className="flex min-h-0 flex-col"
              style={{ flex: `0 0 ${docks.ratio * 100}%`, minHeight: 160 }}
            >
              <PanelHead icon={Layers} title="图层" count={layers.length}>
                <IconButton icon={Plus} title="从目录添加图层（Ctrl+D）" onClick={openCatalog} />
              </PanelHead>
              <LayerStack />
            </div>
          </aside>
        ) : null}

        {docks.leftOpen ? (
          <Splitter
            axis="x"
            label="调整左面板宽度（双击复位）"
            valueNow={docks.left}
            valueMin={DOCK_LIMITS.left.min}
            valueMax={DOCK_LIMITS.left.max}
            onDragStart={() => {
              dragStart.current = docks.left
            }}
            onDrag={(delta) => setDock('left', dragStart.current + delta)}
            onReset={() => resetDock('left')}
            onKeyAdjust={(delta) => setDock('left', docks.left + delta)}
          />
        ) : null}

        <section className="flex min-w-0 flex-1 flex-col">
      <div className="flex h-[var(--gis-view-tab)] shrink-0 items-stretch overflow-hidden border-b border-border bg-background text-[11.5px]">
        <div className="relative flex min-w-0 items-center gap-1.5 border-r border-border bg-card px-2.5 font-semibold">
          <Map className="size-3 shrink-0" />
          <span className="truncate">地图{activeLayer ? ` · ${activeLayer.name}` : ''}</span>
          <span className="text-[10px] font-normal text-muted-foreground">只读</span>
          <span
            aria-hidden="true"
            className="absolute inset-x-0 top-0 h-0.5 bg-primary"
          />
        </div>
        <button
          type="button"
          disabled
          title="布局视图本期留空"
          className="flex cursor-not-allowed items-center border-r border-border px-2.5 text-[11.5px] text-muted-foreground opacity-45"
        >
          布局
        </button>
        <button
          type="button"
          disabled
          title="新建视图本期留空"
          aria-label="新建视图"
          className="grid w-[26px] cursor-not-allowed place-items-center border-r border-border text-muted-foreground opacity-45"
        >
          <Plus className="size-3" />
        </button>
        <div className="ml-auto flex items-center gap-2 px-2 text-[10.5px] text-muted-foreground">
          <span>2D</span>
          <span className="opacity-50">·</span>
          <span>只读</span>
        </div>
      </div>

          <MapStage
            layers={layers}
            previews={previews}
            selectedUid={selectedLayerId}
            catalogStatus={catalogStatus}
            catalogCount={catalogCount}
            catalogError={(catalog.error as Error | null)?.message}
            catalogTruncated={catalog.data?.truncated}
            onRetryCatalog={() => void catalog.refetch()}
            onRetryPreview={(path) => previews[path]?.refetch()}
            onOpenCatalog={openCatalog}
          />
        </section>

        {docks.rightOpen ? (
          <Splitter
            axis="x"
            label="调整右面板宽度（双击复位）"
            valueNow={docks.right}
            valueMin={DOCK_LIMITS.right.min}
            valueMax={DOCK_LIMITS.right.max}
            invert
            onDragStart={() => {
              dragStart.current = docks.right
            }}
            onDrag={(delta) => setDock('right', dragStart.current + delta)}
            onReset={() => resetDock('right')}
            onKeyAdjust={(delta) => setDock('right', docks.right - delta)}
          />
        ) : null}

        {docks.rightOpen ? (
          <RightDock
            selected={selectedItem}
            layer={selectedLayer}
            inLayerStack={Boolean(selectedLayer)}
            geometryType={selectedGeometry}
            width={docks.right}
            onClose={() => toggleDock('right')}
          />
        ) : null}
      </div>

      {/* 属性表 dock 236px（屏高 <940 时 190） */}
      {docks.tableOpen ? (
        <Splitter
          axis="y"
          label="调整属性表高度（双击复位）"
          valueNow={docks.table}
          valueMin={DOCK_LIMITS.table.min}
          valueMax={DOCK_LIMITS.table.max()}
          invert
          onDragStart={() => {
            dragStart.current = docks.table
          }}
          onDrag={(delta) => setDock('table', dragStart.current + delta)}
          onReset={() => resetDock('table')}
          onKeyAdjust={(delta) => setDock('table', docks.table - delta)}
        />
      ) : null}
      {docks.tableOpen ? (
        <AttributeTableDock
          layers={layers}
          items={items}
          height={docks.table}
          onCollapse={() => toggleDock('table')}
        />
      ) : null}

      <StatusBar catalogStatus={catalogStatus} catalogCount={catalogCount} />

      <NavSheet open={navOpen} onOpenChange={setNavOpen} />
    </div>
  )
}

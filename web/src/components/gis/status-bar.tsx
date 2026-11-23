import { AlertTriangle, Loader2, MapPin } from 'lucide-react'

import type { CatalogStatus } from './map-canvas'
import { formatNumber } from './format'
import { useGisStore } from './store'

export interface StatusBarProps {
  catalogStatus: CatalogStatus
  catalogCount: number
}

/**
 * 状态栏（spec §2，24px）：坐标 / 比例尺 / 显示坐标系 / 可见图层 / catalog。
 *
 * 只显示真实值：地图没光标时坐标写「—」，读不到比例尺也写「—」，
 * 不编造「渲染 42 ms」这类数字。
 */
export function StatusBar({ catalogStatus, catalogCount }: StatusBarProps) {
  const view = useGisStore((state) => state.view)
  const layers = useGisStore((state) => state.layers)
  const visible = layers.filter((layer) => layer.visible).length

  const cursor = view.cursor
  const zoomRatio =
    view.baseZoom !== null && view.zoom > 0
      ? Math.pow(2, view.zoom - view.baseZoom)
      : null

  return (
    <footer
      className="flex h-[var(--gis-status-bar)] shrink-0 items-center overflow-hidden border-t border-border bg-background text-[10.5px] text-muted-foreground"
      aria-label="状态栏"
    >
      <span className="flex h-full items-center gap-1 border-r border-border px-2 tabular-nums">
        <MapPin className="size-3" />
        {cursor
          ? `X ${cursor[0].toFixed(4)}°  Y ${cursor[1].toFixed(4)}°`
          : 'X —  Y —'}
      </span>
      <span className="flex h-full items-center gap-1 border-r border-border px-2 tabular-nums">
        比例尺{' '}
        <b className="font-normal text-foreground/80">
          {view.scale ? `1:${formatNumber(view.scale)}` : '—'}
        </b>
      </span>
      <span className="flex h-full items-center gap-1 border-r border-border px-2">
        坐标系 <b className="font-normal text-foreground/80">EPSG:3857</b>
        <span className="opacity-55">(显示)</span>
      </span>
      <span className="flex h-full items-center gap-1 border-r border-border px-2 tabular-nums">
        图层{' '}
        <b className="font-normal text-foreground/80" data-testid="status-layer-visibility">
          {visible} / {layers.length} 可见
        </b>
      </span>
      <span className="flex h-full items-center gap-1 px-2">
        {catalogStatus === 'loading' ? (
          <Loader2 className="size-3 animate-spin text-muted-foreground" />
        ) : catalogStatus === 'error' ? (
          <AlertTriangle className="size-3 text-destructive" />
        ) : (
          <span className="size-1.5 rounded-full bg-success" aria-hidden="true" />
        )}
        catalog {catalogStatus === 'loading' ? '读取中' : catalogStatus === 'error' ? '读取失败' : '已同步'}
        <b className="font-normal text-foreground/80 tabular-nums">
          {catalogStatus === 'ready' ? `${formatNumber(catalogCount)} 项` : ''}
        </b>
      </span>
      <span className="ml-auto flex h-full items-center gap-3 px-2">
        <span className="tabular-nums">
          缩放{' '}
          <b className="font-normal text-foreground/80">
            {zoomRatio === null ? '—' : `${zoomRatio.toFixed(2)}×`}
          </b>
        </span>
        <span className="tabular-nums">
          Z <b className="font-normal text-foreground/80">{view.zoom ? view.zoom.toFixed(2) : '—'}</b>
        </span>
        <span>只读</span>
      </span>
    </footer>
  )
}

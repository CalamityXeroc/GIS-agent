import { Crosshair, GripVertical, Layers, Table2, X } from 'lucide-react'
import { useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'

import { cn } from '@/lib/utils'

import { formatNumber, kindLabel } from './format'
import { isNonSpatialTable, isPreviewable, LAYER_PALETTE, useGisStore } from './store'

/**
 * 图层栈（spec §4 ②）：栈顶 = 最后绘制。
 *
 * 拖拽排序用 pointer 事件（MapLibre 画布会抢 HTML5 DnD 的 drag 事件）；
 * 拖拽中只画插入位置，松手才写 store，避免每帧重排 MapLibre 图层。
 */
export function LayerStack() {
  const layers = useGisStore((state) => state.layers)
  const selectedLayerId = useGisStore((state) => state.selectedLayerId)
  const selectLayer = useGisStore((state) => state.selectLayer)
  const toggleLayer = useGisStore((state) => state.toggleLayer)
  const removeLayer = useGisStore((state) => state.removeLayer)
  const moveLayer = useGisStore((state) => state.moveLayer)
  const openTable = useGisStore((state) => state.openTable)
  const sendCommand = useGisStore((state) => state.sendCommand)

  const listRef = useRef<HTMLDivElement | null>(null)
  const dragRef = useRef<{ uid: number; rowHeight: number; top: number } | null>(null)
  const [drag, setDrag] = useState<{ uid: number; from: number; to: number } | null>(null)

  const visibleCount = layers.filter((layer) => layer.visible).length

  const handleDown = (event: PointerEvent<HTMLButtonElement>, uid: number, index: number) => {
    if (event.button !== 0) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    const row = event.currentTarget.parentElement?.getBoundingClientRect()
    const list = listRef.current?.getBoundingClientRect()
    dragRef.current = {
      uid,
      rowHeight: row?.height || 30,
      // 内容原点 = 列表上沿 + pt-0.5(2px) − scrollTop。列表滚起来以后行在视口里
      // 整体上移，不减 scrollTop 会把落点算低 scrollTop/rowHeight 格（DAIR-334）。
      top: (list?.top ?? 0) + 2 - (listRef.current?.scrollTop ?? 0),
    }
    setDrag({ uid, from: index, to: index })
  }

  const handleMove = (event: PointerEvent<HTMLButtonElement>) => {
    const info = dragRef.current
    if (!info) return
    const offset = event.clientY - info.top
    const to = Math.max(0, Math.min(layers.length - 1, Math.round(offset / info.rowHeight)))
    setDrag((current) => (current && current.to !== to ? { ...current, to } : current))
  }

  const handleUp = (event: PointerEvent<HTMLButtonElement>) => {
    const info = dragRef.current
    dragRef.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    if (info && drag && drag.to !== drag.from) moveLayer(info.uid, drag.to)
    setDrag(null)
  }

  const handleKey = (event: KeyboardEvent<HTMLButtonElement>, uid: number, index: number) => {
    if (event.key === 'ArrowUp' && index > 0) {
      event.preventDefault()
      moveLayer(uid, index - 1)
    } else if (event.key === 'ArrowDown' && index < layers.length - 1) {
      event.preventDefault()
      moveLayer(uid, index + 1)
    }
  }

  if (layers.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-1 px-4 text-center text-[11px] leading-relaxed text-muted-foreground">
        <Layers className="size-4 opacity-50" aria-hidden="true" />
        <span className="font-medium text-foreground/70">图层栈是空的</span>
        <span>双击左栏目录里的数据集即可上图；图层可拖拽排序、勾选显隐。</span>
      </div>
    )
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div ref={listRef} className="min-h-0 flex-1 overflow-auto px-1 pt-0.5 pb-1.5" role="list">
        {layers.map((layer, index) => {
          const selected = layer.uid === selectedLayerId
          const previewable = isPreviewable(layer.item)
          // 非空间表（旧 catalog 里还挂着 feature_class）：只有属性表，不上图。
          const tableOnly = isNonSpatialTable(layer.item)
          const swatch = `var(${LAYER_PALETTE[layer.style.colorIndex]})`
          return (
            <div
              key={layer.uid}
              role="listitem"
              data-layer-row
              onClick={() => selectLayer(layer.uid)}
              title={`${layer.name}\n${layer.path}`}
              className={cn(
                'group flex h-[var(--gis-row-layer)] cursor-default items-center gap-1.5 rounded-[2px] border-t-2 border-transparent pr-1 pl-1 text-[12px] whitespace-nowrap',
                selected && 'border-l-2 border-l-primary bg-accent font-medium',
                'hover:bg-accent/50',
                drag?.uid === layer.uid && 'opacity-50',
                drag && drag.to === index && drag.to !== drag.from && 'border-t-primary'
              )}
            >
              <button
                type="button"
                aria-label={`拖动排序 ${layer.name}（↑↓ 键也可）`}
                className="grid w-[10px] shrink-0 cursor-grab place-items-center text-muted-foreground/40 touch-none active:cursor-grabbing"
                onPointerDown={(event) => handleDown(event, layer.uid, index)}
                onPointerMove={handleMove}
                onPointerUp={handleUp}
                onPointerCancel={handleUp}
                onKeyDown={(event) => handleKey(event, layer.uid, index)}
              >
                <GripVertical className="size-[9px]" />
              </button>
              <input
                type="checkbox"
                checked={layer.visible}
                aria-label={`${layer.name} 可见`}
                onChange={() => toggleLayer(layer.uid)}
                onClick={(event) => event.stopPropagation()}
                className="size-[13px] shrink-0 accent-[var(--primary)]"
              />
              <span
                aria-hidden="true"
                className="size-[13px] shrink-0 rounded-[2px] border border-foreground/20"
                style={{ background: swatch, opacity: layer.visible ? 1 : 0.45 }}
              />
              <span
                className={cn(
                  'min-w-0 flex-1 truncate',
                  !layer.visible && 'text-muted-foreground line-through decoration-1'
                )}
              >
                {layer.name}
              </span>
              <span className="shrink-0 text-[10px] text-muted-foreground">
                {tableOnly ? (
                  <span
                    className="rounded-[2px] bg-muted px-1 text-[9px] leading-[13px]"
                    title="非空间表：没有几何，只能看属性表"
                  >
                    仅属性表
                  </span>
                ) : (
                  kindLabel(layer.kind)
                )}
                {layer.item.feature_count !== null
                  ? ` · ${formatNumber(layer.item.feature_count)}`
                  : ''}
              </span>
              <span className="flex shrink-0 items-center opacity-0 transition-opacity group-hover:opacity-100">
                <button
                  type="button"
                  title="缩放到图层"
                  aria-label={`缩放到 ${layer.name}`}
                  onClick={(event) => {
                    event.stopPropagation()
                    selectLayer(layer.uid)
                    sendCommand('fit-layer', layer.uid)
                  }}
                  className="grid size-[18px] place-items-center rounded-[2px] text-muted-foreground hover:bg-card hover:text-primary"
                >
                  <Crosshair className="size-[11px]" />
                </button>
                <button
                  type="button"
                  title={
                    previewable || tableOnly
                      ? tableOnly
                        ? '打开属性表（非空间表）'
                        : '打开属性表'
                      : '该类型没有属性表'
                  }
                  aria-label={`打开 ${layer.name} 的属性表`}
                  disabled={!previewable && !tableOnly}
                  onClick={(event) => {
                    event.stopPropagation()
                    openTable(layer.path)
                  }}
                  className="grid size-[18px] place-items-center rounded-[2px] text-muted-foreground hover:bg-card hover:text-primary disabled:cursor-not-allowed disabled:opacity-35"
                >
                  <Table2 className="size-[11px]" />
                </button>
                <button
                  type="button"
                  title="从地图移除（不删磁盘文件）"
                  aria-label={`从地图移除 ${layer.name}`}
                  onClick={(event) => {
                    event.stopPropagation()
                    removeLayer(layer.uid)
                  }}
                  className="grid size-[18px] place-items-center rounded-[2px] text-muted-foreground hover:bg-card hover:text-destructive"
                >
                  <X className="size-[11px]" />
                </button>
              </span>
            </div>
          )
        })}
      </div>
      <div className="flex h-[26px] shrink-0 items-center gap-1 border-t border-border bg-card px-2 text-[10.5px] text-muted-foreground">
        <span>绘制顺序：自上而下</span>
        <span className="ml-auto tabular-nums" data-testid="layer-visibility-count">
          {visibleCount} / {layers.length} 可见
        </span>
      </div>
    </div>
  )
}

import { useRef, type KeyboardEvent, type PointerEvent } from 'react'

import { cn } from '@/lib/utils'

/**
 * dock 分隔条（spec §2）：5px 宽/高，拖拽调尺寸，双击复位。
 *
 * 用 pointer 事件 + `setPointerCapture`，不用 HTML5 DnD —— MapLibre 画布会
 * 抢走 drag 事件（spec §5）。键盘 ←→/↑↓ 每次走 `step`（spec 无障碍要求）。
 */
export interface SplitterProps {
  axis: 'x' | 'y'
  label: string
  /** 拖拽方向与「面板变大」相反时置 true（右侧/底部 dock）。 */
  invert?: boolean
  onDragStart?: () => void
  /** delta：面板变大方向为正的像素位移；total：父容器在拖动轴上的尺寸。 */
  onDrag: (delta: number, total: number) => void
  onReset: () => void
  onKeyAdjust: (delta: number) => void
  valueNow?: number
  valueMin?: number
  valueMax?: number
  className?: string
}

export function Splitter({
  axis,
  label,
  invert = false,
  onDragStart,
  onDrag,
  onReset,
  onKeyAdjust,
  valueNow,
  valueMin,
  valueMax,
  className,
}: SplitterProps) {
  const drag = useRef<{ start: number; total: number } | null>(null)

  const coordinate = (event: PointerEvent<HTMLDivElement>) =>
    axis === 'x' ? event.clientX : event.clientY

  const handleDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    const rect = event.currentTarget.parentElement?.getBoundingClientRect()
    drag.current = {
      start: coordinate(event),
      total: axis === 'x' ? (rect?.width ?? 1) : (rect?.height ?? 1),
    }
    onDragStart?.()
  }

  const handleMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!drag.current) return
    const raw = coordinate(event) - drag.current.start
    onDrag(invert ? -raw : raw, drag.current.total)
  }

  const handleUp = (event: PointerEvent<HTMLDivElement>) => {
    if (!drag.current) return
    drag.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }

  const handleKey = (event: KeyboardEvent<HTMLDivElement>) => {
    const negative = axis === 'x' ? 'ArrowLeft' : 'ArrowUp'
    const positive = axis === 'x' ? 'ArrowRight' : 'ArrowDown'
    if (event.key === negative) {
      event.preventDefault()
      onKeyAdjust(-8)
    } else if (event.key === positive) {
      event.preventDefault()
      onKeyAdjust(8)
    } else if (event.key === 'Enter') {
      event.preventDefault()
      onReset()
    }
  }

  return (
    <div
      role="separator"
      aria-label={label}
      aria-orientation={axis === 'x' ? 'vertical' : 'horizontal'}
      aria-valuenow={valueNow === undefined ? undefined : Math.round(valueNow)}
      aria-valuemin={valueMin}
      aria-valuemax={valueMax}
      tabIndex={0}
      data-splitter={axis}
      onPointerDown={handleDown}
      onPointerMove={handleMove}
      onPointerUp={handleUp}
      onPointerCancel={handleUp}
      onDoubleClick={(event) => {
        event.preventDefault()
        onReset()
      }}
      onKeyDown={handleKey}
      className={cn(
        'group relative z-20 shrink-0 touch-none select-none outline-none focus-visible:bg-primary/10',
        axis === 'x'
          ? 'w-[var(--gis-splitter)] cursor-col-resize after:absolute after:inset-y-0 after:left-1/2 after:w-px after:-translate-x-1/2 after:bg-border after:content-[""] group-hover:after:bg-primary/60'
          : 'h-[var(--gis-splitter)] cursor-row-resize after:absolute after:inset-x-0 after:top-1/2 after:h-px after:-translate-y-1/2 after:bg-border after:content-[""] group-hover:after:bg-primary/60',
        className
      )}
    />
  )
}

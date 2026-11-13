import { Check, ChevronDown } from 'lucide-react'
import * as React from 'react'
import { createPortal } from 'react-dom'

import { cn } from '@/lib/utils'

export interface SelectOption {
  value: string
  label: string
}

export interface SelectProps {
  value: string
  onValueChange: (value: string) => void
  options: SelectOption[]
  /** 触发器上的附加类名（高度/圆角/底色按使用处传）。 */
  className?: string
  /** 无可见文字标签时必填，供读屏使用。 */
  'aria-label'?: string
  disabled?: boolean
  /** 触发器里没有选中项时的占位文字。 */
  placeholder?: string
}

const GAP = 4
const EDGE = 8
const MAX_LIST_HEIGHT = 240

/**
 * 主题化下拉：button 触发器 + 挂在 body 上的 `role="listbox"` 弹层。
 *
 * 为什么不用原生 `<select>`：Chromium 的弹出层取 select 自身的 used
 * `background-color`，`bg-transparent` 会回落白底，深色下漏白且无法用 CSS 覆盖。
 * 弹层走 `--popover` / `--border` token，键盘 ↑↓ Home End Enter Esc 可用。
 */
export function Select({
  value,
  onValueChange,
  options,
  className,
  disabled = false,
  placeholder,
  'aria-label': ariaLabel,
}: SelectProps) {
  const [open, setOpen] = React.useState(false)
  const [activeIndex, setActiveIndex] = React.useState(0)
  const [box, setBox] = React.useState<{
    left: number
    top: number
    width: number
    maxHeight: number
    up: boolean
  } | null>(null)
  const triggerRef = React.useRef<HTMLButtonElement>(null)
  const listRef = React.useRef<HTMLDivElement>(null)
  const listId = React.useId()

  const selectedIndex = options.findIndex((option) => option.value === value)
  const selected = selectedIndex >= 0 ? options[selectedIndex] : undefined

  const close = React.useCallback(() => {
    setOpen(false)
    setBox(null)
  }, [])

  const place = React.useCallback(() => {
    const el = triggerRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const spaceBelow = window.innerHeight - rect.bottom - GAP - EDGE
    const spaceAbove = rect.top - GAP - EDGE
    const up = spaceBelow < MAX_LIST_HEIGHT && spaceAbove > spaceBelow
    const width = Math.max(rect.width, 88)
    setBox({
      // 右边界钳制：分页那条触发器贴着视口右侧，窄视口下 `left + width` 会溢出。
      left: Math.max(EDGE, Math.min(rect.left, window.innerWidth - width - EDGE)),
      top: up ? rect.top - GAP : rect.bottom + GAP,
      width,
      maxHeight: Math.min(up ? spaceAbove : spaceBelow, MAX_LIST_HEIGHT),
      up,
    })
  }, [])

  const openList = React.useCallback(() => {
    if (disabled) return
    setActiveIndex(selectedIndex >= 0 ? selectedIndex : 0)
    setOpen(true)
  }, [disabled, selectedIndex])

  // 开合时定位；开着时滚页/改窗口尺寸都要跟着走，否则弹层会飘走。
  React.useLayoutEffect(() => {
    if (!open) return
    place()
    const onScrollOrResize = () => place()
    window.addEventListener('resize', onScrollOrResize)
    window.addEventListener('scroll', onScrollOrResize, true)
    return () => {
      window.removeEventListener('resize', onScrollOrResize)
      window.removeEventListener('scroll', onScrollOrResize, true)
    }
  }, [open, place])

  // 点外部关闭。
  React.useEffect(() => {
    if (!open) return
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node
      if (triggerRef.current?.contains(target) || listRef.current?.contains(target)) return
      close()
    }
    document.addEventListener('pointerdown', onPointerDown)
    return () => document.removeEventListener('pointerdown', onPointerDown)
  }, [open, close])

  // 键盘移动时把高亮项滚进可视区（不用 scrollIntoView，避免带动整页）。
  React.useLayoutEffect(() => {
    if (!open) return
    const list = listRef.current
    const item = list?.querySelector<HTMLElement>(`[data-index="${activeIndex}"]`)
    if (!list || !item) return
    if (item.offsetTop < list.scrollTop) list.scrollTop = item.offsetTop
    else if (item.offsetTop + item.offsetHeight > list.scrollTop + list.clientHeight) {
      list.scrollTop = item.offsetTop + item.offsetHeight - list.clientHeight
    }
  }, [open, activeIndex])

  const commit = (index: number) => {
    const option = options[index]
    if (!option) return
    if (option.value !== value) onValueChange(option.value)
    close()
    triggerRef.current?.focus()
  }

  const onKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (!open) {
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp' || event.key === 'Enter' || event.key === ' ') {
        event.preventDefault()
        openList()
      }
      return
    }
    const last = options.length - 1
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault()
        setActiveIndex((index) => Math.min(index + 1, last))
        break
      case 'ArrowUp':
        event.preventDefault()
        setActiveIndex((index) => Math.max(index - 1, 0))
        break
      case 'Home':
        event.preventDefault()
        setActiveIndex(0)
        break
      case 'End':
        event.preventDefault()
        setActiveIndex(last)
        break
      case 'Enter':
      case ' ':
        event.preventDefault()
        commit(activeIndex)
        break
      case 'Escape':
        event.preventDefault()
        close()
        break
      case 'Tab':
        close()
        break
    }
  }

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        role="combobox"
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listId : undefined}
        // 弹层 portal 到 body，不是触发器的 DOM 后代；ARIA 1.2 要求
        // `aria-activedescendant` 指向「后代或经 aria-owns 声明的逻辑后代」，
        // 所以这里补 aria-owns，否则读屏不播报当前高亮项。
        aria-owns={open ? listId : undefined}
        aria-activedescendant={open ? `${listId}-${activeIndex}` : undefined}
        disabled={disabled}
        className={cn(
          'inline-flex items-center justify-between gap-1 text-left outline-none transition-colors',
          'hover:bg-accent/50 focus-visible:bg-accent/50 disabled:cursor-not-allowed disabled:opacity-50',
          className
        )}
        onClick={() => (open ? close() : openList())}
        onKeyDown={onKeyDown}
      >
        <span className="truncate">{selected?.label ?? placeholder ?? ''}</span>
        <ChevronDown
          aria-hidden="true"
          className="size-3.5 shrink-0 opacity-60 transition-transform data-[open=true]:rotate-180"
          data-open={open}
        />
      </button>
      {open && box
        ? createPortal(
            <div
              ref={listRef}
              id={listId}
              role="listbox"
              aria-label={ariaLabel}
              tabIndex={-1}
              className="fixed z-[60] overflow-y-auto rounded-md border border-border bg-popover p-1 text-popover-foreground outline-none"
              style={{
                left: box.left,
                top: box.top,
                width: box.width,
                maxHeight: box.maxHeight,
                transform: box.up ? 'translateY(-100%)' : undefined,
              }}
            >
              {options.map((option, index) => {
                const isSelected = option.value === value
                return (
                  <div
                    key={option.value}
                    id={`${listId}-${index}`}
                    data-index={index}
                    role="option"
                    aria-selected={isSelected}
                    className={cn(
                      'flex cursor-pointer items-center justify-between gap-2 rounded-sm px-2 py-1.5 text-xs',
                      index === activeIndex
                        ? 'bg-accent text-accent-foreground'
                        : 'text-popover-foreground'
                    )}
                    onMouseEnter={() => setActiveIndex(index)}
                    onMouseDown={(event) => event.preventDefault()}
                    onClick={() => commit(index)}
                  >
                    <span className="truncate">{option.label}</span>
                    {isSelected ? <Check aria-hidden="true" className="size-3.5 shrink-0" /> : null}
                  </div>
                )
              })}
            </div>,
            document.body
          )
        : null}
    </>
  )
}

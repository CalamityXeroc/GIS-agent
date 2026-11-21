import { AlertTriangle, Info } from 'lucide-react'
import type { ReactNode } from 'react'

import { Select as UiSelect, type SelectOption } from '@/components/ui/select'
import { cn } from '@/lib/utils'

/**
 * 右 dock 的原子件（spec §3 密度指标）：区块头 10.5px、字段行 11.5px、
 * 标签列 96px、控件高 24、圆角 4px、层次只靠 1px 描边。
 */

export function Section({
  title,
  badge,
  children,
}: {
  title: string
  badge?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="mb-[13px] last:mb-0.5">
      <h4 className="mb-[5px] flex items-center gap-1.5 text-[10.5px] font-semibold tracking-[0.03em] text-muted-foreground">
        {title}
        {badge}
        <span className="h-px flex-1 bg-border" aria-hidden="true" />
      </h4>
      {children}
    </section>
  )
}

/** 「待补」徽标：接口/字段缺失时挂在区块头上，正文里再说原因。 */
export function Todo({ children = '待补' }: { children?: ReactNode }) {
  return (
    <span className="ml-auto inline-flex h-[15px] shrink-0 items-center rounded-[2px] border border-warning/45 bg-warning/15 px-[5px] text-[9.5px] font-semibold text-warning">
      {children}
    </span>
  )
}

export function Field({
  label,
  value,
  mono,
  wrap,
}: {
  label: string
  value: ReactNode
  mono?: boolean
  wrap?: boolean
}) {
  const text = typeof value === 'string' || typeof value === 'number' ? String(value) : undefined
  return (
    <div className="grid grid-cols-[96px_minmax(0,1fr)] gap-x-2 gap-y-[3px] py-[2px] text-[11.5px]">
      <dt className="truncate text-muted-foreground" title={label}>
        {label}
      </dt>
      <dd
        className={cn(
          'min-w-0 text-foreground',
          mono ? 'font-mono text-[10.5px]' : '',
          wrap ? 'leading-[1.45] break-all whitespace-normal' : 'truncate'
        )}
        title={text}
      >
        {value}
      </dd>
    </div>
  )
}

export function Hint({ children }: { children: ReactNode }) {
  return (
    <p className="mt-[5px] text-[10.5px] leading-[1.55] text-muted-foreground">{children}</p>
  )
}

/** 失败态：接口不可用时给「待补 + 原因」，不留空表冒充数据。 */
export function Unavailable({ title, reason }: { title: string; reason: string }) {
  return (
    <div className="rounded-[var(--gis-radius)] border border-warning/40 bg-warning/10 px-2 py-1.5 text-[10.5px] leading-relaxed">
      <div className="flex items-center gap-1 font-semibold text-warning">
        <AlertTriangle className="size-3 shrink-0" />
        {title}
      </div>
      <p className="mt-0.5 text-foreground/75">{reason}</p>
    </div>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-1.5 px-3 text-center text-[11px] leading-relaxed text-muted-foreground">
      <Info className="size-4 opacity-50" />
      <span className="font-medium text-foreground/70">{title}</span>
      {children}
    </div>
  )
}

export function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2 py-[3px] text-[11.5px]">
      <label className="shrink-0 text-muted-foreground">{label}</label>
      <div className="flex min-w-0 items-center justify-end gap-1.5">{children}</div>
    </div>
  )
}

/**
 * 下拉：统一走 `components/ui/select.tsx`（原生 `<select>` 的弹层在深色主题下
 * 会漏白，`tests/web/test_frontend_select.py` 明确禁止），这里只补工作台尺寸。
 */
export function SelectRow({
  value,
  onChange,
  options,
  title,
  disabled,
}: {
  value: string
  onChange: (value: string) => void
  options: SelectOption[]
  title?: string
  disabled?: boolean
}) {
  return (
    <span title={title} className="min-w-0">
      <UiSelect
        value={value}
        onValueChange={onChange}
        options={options}
        aria-label={title ?? '选择'}
        disabled={disabled}
        className="h-[24px] min-w-0 max-w-[176px] rounded-[3px] border border-border bg-card px-1.5 text-[11.5px] text-foreground"
      />
    </span>
  )
}

/** 原生取色器套在色块里：色块显示当前色，点开就是系统调色板。 */
export function ColorSwatch({
  value,
  onChange,
  title,
  size = 'normal',
}: {
  value: string
  onChange: (value: string) => void
  title?: string
  size?: 'normal' | 'small'
}) {
  return (
    <span
      className={cn(
        'relative inline-block shrink-0 overflow-hidden rounded-[3px] border border-border',
        size === 'small' ? 'size-[14px]' : 'h-[22px] w-[44px]'
      )}
      style={{ background: value }}
      title={title ?? value}
    >
      <input
        type="color"
        value={value}
        aria-label={title ?? '选择颜色'}
        onChange={(event) => onChange(event.target.value)}
        className="absolute -inset-1.5 size-[200%] cursor-pointer border-0 p-0 opacity-0"
      />
    </span>
  )
}

export function RangeRow({
  label,
  min,
  max,
  step,
  value,
  suffix = '',
  onChange,
}: {
  label: string
  min: number
  max: number
  step: number
  value: number
  suffix?: string
  onChange: (value: number) => void
}) {
  return (
    <Row label={label}>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-[16px] w-[112px] accent-primary"
      />
      <output className="w-[38px] shrink-0 text-right text-[10.5px] tabular-nums text-muted-foreground">
        {Number.isInteger(value) ? value : value.toFixed(2)}
        {suffix}
      </output>
    </Row>
  )
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string; title?: string }[]
  value: T
  onChange: (value: T) => void
}) {
  return (
    <div className="flex overflow-hidden rounded-[3px] border border-border">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          title={option.title}
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
          className={cn(
            'h-[22px] px-[7px] text-[11px]',
            value === option.value
              ? 'bg-accent font-semibold text-primary'
              : 'text-muted-foreground hover:bg-accent/60 hover:text-foreground'
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

/** 色带预览条：色带只作用于显示，不改数据。 */
export function RampBar({ colors }: { colors: string[] }) {
  return (
    <div className="mt-1 flex h-[10px] overflow-hidden rounded-[2px] border border-border">
      {colors.map((color, index) => (
        <i key={`${color}-${index}`} className="flex-1" style={{ background: color }} />
      ))}
    </div>
  )
}

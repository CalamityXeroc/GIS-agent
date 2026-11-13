import type * as React from 'react'

import { IconBadge, type IconBadgeProps } from '@/components/ui/icon-badge'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'

type Tone = 'default' | 'muted' | 'success' | 'warning' | 'destructive'

const HINT_TONE: Record<Tone, string> = {
  default: 'text-foreground',
  muted: 'text-muted-foreground',
  success: 'text-success',
  warning: 'text-warning',
  destructive: 'text-destructive',
}

const BAR_TONE: Record<'accent-1' | 'accent-2' | 'accent-3', string> = {
  'accent-1': 'bg-overview-accent-1/70',
  'accent-2': 'bg-overview-accent-2/70',
  'accent-3': 'bg-overview-accent-3/70',
}

export interface StatCellProps {
  label: React.ReactNode
  value: React.ReactNode
  hint?: React.ReactNode
  hintTone?: Tone
}

/** KPI 里的一格：12px 灰标签 + ≥28px 等宽数字 + 12px 副标（§2.5 第 1 条）。 */
export function StatCell({ label, value, hint, hintTone = 'muted' }: StatCellProps) {
  return (
    <div className="px-5 py-4">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-2 text-[30px] leading-none font-semibold tracking-tight tabular-nums text-foreground">
        {value}
      </div>
      {hint ? (
        <div className={cn('mt-2 text-xs', HINT_TONE[hintTone])}>{hint}</div>
      ) : null}
    </div>
  )
}

/**
 * KPI 行：**一张卡片切 5 格**，格间 1px 竖线，不是 5 张卡。
 * 传入的每个子节点应是一个 ``StatCell``。
 */
export function StatRow({
  children,
  className,
}: {
  children: React.ReactNode
  className?: string
}) {
  return (
    <div
      data-slot="stat-row"
      className={cn(
        'grid grid-cols-1 divide-y divide-border overflow-hidden rounded-lg border border-border bg-card sm:grid-cols-2 lg:grid-cols-5 lg:divide-x lg:divide-y-0',
        className
      )}
    >
      {children}
    </div>
  )
}

export interface StatCardProps {
  title: React.ReactNode
  value: React.ReactNode
  description?: React.ReactNode
  icon?: React.ComponentType<{ className?: string }>
  tone?: 'accent-1' | 'accent-2' | 'accent-3'
  iconTone?: IconBadgeProps['tone']
  sparkline?: number[]
  loading?: boolean
  className?: string
}

/** 单张 KPI 卡（带图标与迷你柱状 sparkline），用于非五连排的场景。 */
export function StatCard({
  title,
  value,
  description,
  icon: Icon,
  tone = 'accent-1',
  iconTone = 'chart1',
  sparkline,
  loading = false,
  className,
}: StatCardProps) {
  const max = sparkline?.length ? Math.max(...sparkline, 1) : 1

  return (
    <div
      data-slot="stat-card"
      className={cn(
        'flex flex-col rounded-lg border border-border bg-card p-4',
        className
      )}
    >
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="text-xs text-muted-foreground">{title}</div>
          {loading ? (
            <Skeleton className="mt-3 h-7 w-24" />
          ) : (
            <div className="mt-2 text-[30px] leading-none font-semibold tracking-tight tabular-nums">
              {value}
            </div>
          )}
          {description ? (
            <div className="mt-2 text-xs text-muted-foreground">{description}</div>
          ) : null}
        </div>
        {Icon ? (
          <IconBadge tone={iconTone}>
            <Icon />
          </IconBadge>
        ) : null}
      </div>

      {sparkline?.length ? (
        <div className="mt-4 flex h-9 items-end gap-1" aria-hidden="true">
          {sparkline.map((value, index) => (
            <div
              key={index}
              className={cn('min-h-[2px] flex-1 rounded-sm', BAR_TONE[tone])}
              style={{ height: `${Math.max(4, (value / max) * 100)}%` }}
            />
          ))}
        </div>
      ) : null}
    </div>
  )
}

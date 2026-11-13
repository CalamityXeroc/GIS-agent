import type * as React from 'react'

import { cn } from '@/lib/utils'

type ChipTone = 'default' | 'muted' | 'success' | 'warning' | 'destructive'

const TONE: Record<ChipTone, string> = {
  default: 'text-foreground',
  muted: 'text-muted-foreground',
  success: 'text-success',
  warning: 'text-warning',
  destructive: 'text-destructive',
}

export interface StatChipItem {
  label: React.ReactNode
  value: React.ReactNode
  tone?: ChipTone
}

/**
 * 内联指标 chip 组：健康条那一行「HTML 内核 · 正常 | 队列 0 | 模型 3」
 * 就由它渲染（§2.5 第 2 条）。
 */
export function StatChips({
  items,
  className,
}: {
  items: StatChipItem[]
  className?: string
}) {
  return (
    <div
      data-slot="stat-chips"
      className={cn('flex flex-wrap items-center gap-1.5', className)}
    >
      {items.map((item, index) => (
        <span
          key={index}
          className="inline-flex items-center gap-1 rounded-full border border-border bg-background px-2 py-0.5 text-[11px] leading-5"
        >
          <span className="text-muted-foreground">{item.label}</span>
          <span
            className={cn('font-medium tabular-nums', TONE[item.tone ?? 'default'])}
          >
            {item.value}
          </span>
        </span>
      ))}
    </div>
  )
}

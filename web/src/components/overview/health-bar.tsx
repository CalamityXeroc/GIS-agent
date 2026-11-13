import { cn } from '@/lib/utils'
import type { HealthItem, HealthQueue, ModelUsage } from '@/lib/api'

interface HealthBarProps {
  health: {
    kernel: HealthItem
    arcgis: HealthItem
    model: HealthItem
    queue: HealthQueue
  }
  models: ModelUsage[]
}

/**
 * 健康条：**一张卡片一行**（§2.5 第 2 条）：绿点 + 名称 + 内联 chip，
 * 右侧是模型色点列表（颜色与按天用量图同源，取 --chart-1…5）。
 */
export function HealthBar({ health, models }: HealthBarProps) {
  const items = [health.kernel, health.arcgis, health.model, health.queue]

  return (
    <div
      data-slot="health-bar"
      className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-lg border border-border bg-card px-4 py-3"
    >
      {items.map((item) => (
        <div key={item.label} className="flex min-w-0 items-center gap-2 text-xs">
          <span
            aria-hidden="true"
            className={cn(
              'size-1.5 shrink-0 rounded-full',
              item.status === 'ok' ? 'bg-success' : 'bg-warning'
            )}
          />
          <span className="shrink-0 text-muted-foreground">{item.label}</span>
          <span className="max-w-[22rem] truncate rounded-full border border-border px-2 py-0.5 text-[11px] text-foreground tabular-nums">
            {item.detail}
          </span>
        </div>
      ))}

      <div className="ml-auto flex items-center gap-3 text-xs text-muted-foreground">
        <span className="whitespace-nowrap">在用模型</span>
        <div className="flex items-center gap-2">
          {models.slice(0, 5).map((item, index) => (
            <span key={item.model} className="flex items-center gap-1">
              <span
                aria-hidden="true"
                className="size-2 rounded-full"
                style={{ background: `var(--chart-${index + 1})` }}
              />
              <span className="max-w-[9rem] truncate text-[11px]">{item.model}</span>
            </span>
          ))}
          {models.length === 0 ? <span className="text-[11px]">—</span> : null}
        </div>
      </div>
    </div>
  )
}

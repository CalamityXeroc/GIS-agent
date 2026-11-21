import { useQuery } from '@tanstack/react-query'
import { AreaChart, BarChart3, LayoutDashboard, TriangleAlert } from 'lucide-react'
import { lazy, Suspense, useState } from 'react'

import { PageHeader } from '@/components/layout/page-header'
import { HealthBar } from '@/components/overview/health-bar'
import type { ChartMode } from '@/components/overview/usage-chart'
import { Empty } from '@/components/ui/empty'
import { Select } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { StatCell, StatRow } from '@/components/ui/stat-card'
import { TitledCard } from '@/components/ui/titled-card'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { getMetrics } from '@/lib/api'
import { formatDuration, formatTokens } from '@/lib/run-events'

// VChart 全量包约 2MB（gzip 530KB）：只在总览页按需拉，别把首屏和运行历史拖慢。
const UsageChart = lazy(async () => {
  const module = await import('@/components/overview/usage-chart')
  return { default: module.UsageChart }
})

function percent(value: number): string {
  return `${Math.round(value * 100)}%`
}

function OverviewSkeleton() {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 divide-y divide-border overflow-hidden rounded-lg border border-border bg-card sm:grid-cols-2 lg:grid-cols-5 lg:divide-x lg:divide-y-0">
        {Array.from({ length: 5 }).map((_, index) => (
          <div key={index} className="px-5 py-4">
            <Skeleton className="h-3 w-16" />
            <Skeleton className="mt-3 h-7 w-24" />
            <Skeleton className="mt-3 h-3 w-20" />
          </div>
        ))}
      </div>
      <Skeleton className="h-12 w-full" />
      <Skeleton className="h-[320px] w-full" />
    </div>
  )
}

export function OverviewPage() {
  const [days, setDays] = useState(7)
  const [mode, setMode] = useState<ChartMode>('bar')

  const metricsQuery = useQuery({
    queryKey: ['metrics', days],
    queryFn: () => getMetrics(days),
    refetchInterval: 15000,
  })
  const data = metricsQuery.data

  return (
    <>
      <PageHeader
        title="概览"
        description="运行成功率、耗时、token 与验收缺口的聚合视图，口径全部来自 trace 事实。"
        actions={
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <span>统计窗口</span>
            <Select
              aria-label="统计窗口"
              className="h-8 rounded-full border border-border bg-card px-2.5 text-xs text-foreground"
              value={String(days)}
              onValueChange={(next) => setDays(Number(next))}
              options={[
                { value: '7', label: '近 7 天' },
                { value: '30', label: '近 30 天' },
                { value: '90', label: '近 90 天' },
              ]}
            />
          </div>
        }
      />

      {metricsQuery.isLoading ? (
        <OverviewSkeleton />
      ) : metricsQuery.isError ? (
        <Empty
          icon={LayoutDashboard}
          title="指标加载失败"
          description={(metricsQuery.error as Error)?.message ?? '请确认服务是否在运行。'}
        />
      ) : data ? (
        <div className="space-y-4">
          {data.warning ? (
            <div className="flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2.5 text-xs text-warning">
              <TriangleAlert className="size-3.5 shrink-0" />
              <span>{data.warning}</span>
            </div>
          ) : null}

          {/* KPI：一张卡切 5 格，格间 1px 竖线 */}
          <StatRow>
            <StatCell
              label="运行数"
              value={data.kpis.runs}
              hint={`全部 ${data.total_runs} 个 · 窗口内 ${data.runs_in_window} 个`}
            />
            <StatCell
              label="验收通过率"
              value={data.kpis.runs ? percent(data.kpis.pass_rate) : '—'}
              hint={`${data.kpis.passed} 通过 · ${data.kpis.failed} 失败 · ${data.kpis.unverified} 未验收`}
              hintTone={data.kpis.failed > 0 ? 'destructive' : 'muted'}
            />
            <StatCell
              label="中位耗时"
              value={formatDuration(data.kpis.duration_p50_s)}
              hint={`p90 ${formatDuration(data.kpis.duration_p90_s)} · n=${data.kpis.duration_n}`}
            />
            <StatCell
              label="token 总量"
              value={formatTokens(data.kpis.tokens_total)}
              hint={`缓存命中 ${percent(data.kpis.cache_ratio)} · ${formatTokens(data.kpis.tokens_cached)}`}
            />
            <StatCell
              label="失败工具数"
              value={data.kpis.tool_failures}
              hint={`返工 ${data.kpis.repairs} 次 · ${data.kpis.turns} 轮`}
              hintTone={data.kpis.tool_failures > 0 ? 'warning' : 'muted'}
            />
          </StatRow>

          {/* 健康条：一张卡一行 */}
          <HealthBar health={data.health} models={data.models} />

          <div className="grid gap-4 lg:grid-cols-3">
            <TitledCard
              className="lg:col-span-2"
              title="按天用量"
              meta={<span className="tabular-nums">{data.window.since} → {data.window.until}</span>}
              actions={
                <ToggleGroup
                  type="single"
                  value={mode}
                  aria-label="图表类型"
                  onValueChange={(value) => value && setMode(value as ChartMode)}
                >
                  <ToggleGroupItem value="bar" aria-label="柱状图">
                    <BarChart3 className="size-3.5" />
                  </ToggleGroupItem>
                  <ToggleGroupItem value="area" aria-label="面积图">
                    <AreaChart className="size-3.5" />
                  </ToggleGroupItem>
                </ToggleGroup>
              }
              bodyClassName="px-2 pb-2 pt-0"
            >
              {data.kpis.tokens_total > 0 ? (
                <Suspense fallback={<Skeleton className="h-[268px] w-full" />}>
                  <UsageChart daily={data.daily} models={data.models} mode={mode} />
                </Suspense>
              ) : (
                <Empty
                  title="窗口内没有用量"
                  description="换个更长的统计窗口，或先跑一次任务。"
                  className="min-h-[240px] border-0 bg-transparent"
                />
              )}
            </TitledCard>

            <TitledCard
              title="验收缺口 TOP"
              meta={<span className="tabular-nums">{data.gaps.length} 类</span>}
              bodyClassName="px-4 pb-4 pt-1"
            >
              {data.gaps.length === 0 ? (
                <p className="py-6 text-center text-xs text-muted-foreground">
                  窗口内没有验收缺口记录。
                </p>
              ) : (
                <ul className="space-y-2.5">
                  {data.gaps.map((gap) => {
                    const max = data.gaps[0]?.count || 1
                    return (
                      <li key={gap.label} className="space-y-1">
                        <div className="flex items-baseline justify-between gap-2 text-xs">
                          <span className="min-w-0 truncate text-foreground">{gap.label}</span>
                          <span className="shrink-0 tabular-nums text-muted-foreground">
                            {gap.count} 次 · {gap.runs} run
                          </span>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                          <div
                            className="h-full rounded-full bg-[var(--chart-1)]"
                            style={{ width: `${Math.max(6, (gap.count / max) * 100)}%` }}
                          />
                        </div>
                      </li>
                    )
                  })}
                </ul>
              )}
            </TitledCard>
          </div>
        </div>
      ) : null}
    </>
  )
}

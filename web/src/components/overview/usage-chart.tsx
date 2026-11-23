import { VChart, type ISpec } from '@visactor/react-vchart'
import { useMemo } from 'react'

import { chartPalette, themeColor } from '@/lib/colors'
import { formatTokens } from '@/lib/run-events'
import type { DailyBucket, ModelUsage } from '@/lib/api'

/** 一张图最多几路模型：再多图例挤成两行，且色板只有 5 个（§2.5 第 6 条）。 */
const MAX_SERIES = 5

export type ChartMode = 'bar' | 'area'

interface UsageChartProps {
  daily: DailyBucket[]
  models: ModelUsage[]
  mode: ChartMode
  height?: number
}

function buildRows(daily: DailyBucket[], models: ModelUsage[]) {
  const top = models.slice(0, MAX_SERIES).map((item) => item.model)
  const rest = new Set(models.slice(MAX_SERIES).map((item) => item.model))
  const rows: Array<{ date: string; model: string; tokens: number }> = []
  for (const bucket of daily) {
    for (const model of top) {
      rows.push({ date: bucket.date.slice(5), model, tokens: bucket.by_model[model] ?? 0 })
    }
    if (rest.size) {
      let other = 0
      for (const [model, tokens] of Object.entries(bucket.by_model)) {
        if (rest.has(model)) other += tokens
      }
      rows.push({ date: bucket.date.slice(5), model: '其他', tokens: other })
    }
  }
  return rows
}

/**
 * 按天用量图：默认堆叠柱（按模型分色），右上 segmented 可切面积图。
 * 颜色全部取 `--chart-1…5`，深色模式走同一套 token（§2.5 第 6 条）。
 */
export function UsageChart({ daily, models, mode, height = 268 }: UsageChartProps) {
  const spec = useMemo<ISpec>(() => {
    const palette = chartPalette()
    const series = models.slice(0, MAX_SERIES).map((item) => item.model)
    if (models.length > MAX_SERIES) series.push('其他')
    const colorMap: Record<string, string> = {}
    series.forEach((model, index) => {
      colorMap[model] = index < palette.length ? palette[index] : themeColor('--muted-foreground')
    })

    const rows = buildRows(daily, models)
    const axisLabel = { style: { fill: themeColor('--muted-foreground'), fontSize: 11 } }

    return {
      type: mode === 'bar' ? 'bar' : 'area',
      background: 'transparent',
      animation: false,
      padding: { top: 8, right: 8, bottom: 0, left: 4 },
      data: [{ id: 'usage', values: rows }],
      xField: 'date',
      yField: 'tokens',
      seriesField: 'model',
      stack: true,
      color: series.map((model) => colorMap[model]),
      bar: { style: { cornerRadius: [2, 2, 0, 0] } },
      area: { style: { fillOpacity: 0.6 } },
      line: { style: { lineWidth: 1 } },
      axes: [
        {
          orient: 'left',
          label: { ...axisLabel, formatMethod: (value: number) => formatTokens(value) },
          grid: { style: { stroke: themeColor('--border'), lineDash: [3, 3] } },
          domainLine: { visible: false },
          tick: { visible: false },
        },
        {
          orient: 'bottom',
          label: axisLabel,
          domainLine: { style: { stroke: themeColor('--border') } },
          tick: { visible: false },
        },
      ],
      // 悬停时的列带默认取 VChart 主题的 axisGridColor（浅灰），深色下是一抹白底。
      crosshair: {
        xField: {
          visible: true,
          line: {
            type: 'rect',
            style: {
              fill: themeColor('--border'),
              opacity: 0.6,
              lineWidth: 0,
              stroke: 'transparent',
            },
          },
        },
      },
      legends: {
        visible: true,
        orient: 'bottom',
        position: 'middle',
        padding: { top: 6, bottom: 0, left: 0, right: 0 },
        item: {
          shape: { style: { size: 8 } },
          label: { style: { fill: themeColor('--muted-foreground'), fontSize: 11 } },
        },
      },
      tooltip: {
        mark: {
          content: [
            {
              key: (datum: { model?: string }) => datum?.model ?? '',
              value: (datum: { tokens?: number }) => formatTokens(datum?.tokens ?? 0),
            },
          ],
        },
        // 默认主题把提示框写成白底黑字，深色下必须显式换成 popover token。
        style: {
          panel: {
            backgroundColor: themeColor('--popover'),
            border: { color: themeColor('--border'), width: 1, radius: 8 },
            shadow: { x: 0, y: 0, blur: 0, spread: 0, color: 'transparent' },
          },
          titleLabel: { fill: themeColor('--muted-foreground'), fontSize: 11 },
          keyLabel: { fill: themeColor('--muted-foreground'), fontSize: 12 },
          valueLabel: { fill: themeColor('--popover-foreground'), fontSize: 12 },
        },
      },
    } as ISpec
  }, [daily, models, mode])

  return (
    <div style={{ height }} className="w-full">
      <VChart spec={spec} options={{ mode: 'desktop-browser' }} />
    </div>
  )
}

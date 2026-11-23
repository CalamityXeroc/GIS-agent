import { toHexColor } from '@/lib/paint-color'

import { LAYER_PALETTE } from '../store'

/**
 * 右 dock 用到的两套色：图层的「单一符号」色板（`--chart-*`，跟主题走）和
 * 唯一值/分级用的固定色带。
 *
 * 为什么不在 store 里算：`--chart-N` 是 oklch，转 hex 要过 canvas（浏览器），
 * 而 store 的默认值在模块加载时就要求值。所以色值在组件里算好后**以 hex 存回
 * store**，地图侧只认 hex，主题切换时地图侧再按 `colorIndex` 重取色板色。
 */

const CHART_FALLBACK = ['#3b82f6', '#06b6d4', '#8b5cf6', '#ec4899', '#22c55e']

/** `--chart-N` → `#rrggbb`（MapLibre 与 `<input type="color">` 都认）。 */
export function chartHex(index: number): string {
  const slot = ((index % LAYER_PALETTE.length) + LAYER_PALETTE.length) % LAYER_PALETTE.length
  if (typeof window === 'undefined') return CHART_FALLBACK[slot]
  const raw = window
    .getComputedStyle(document.documentElement)
    .getPropertyValue(LAYER_PALETTE[slot])
    .trim()
  return toHexColor(raw, CHART_FALLBACK[slot])
}

function hslToHex(h: number, s: number, l: number): string {
  const hue = ((h % 360) + 360) % 360
  const sat = s / 100
  const light = l / 100
  const c = (1 - Math.abs(2 * light - 1)) * sat
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1))
  const m = light - c / 2
  const [r, g, b] =
    hue < 60
      ? [c, x, 0]
      : hue < 120
        ? [x, c, 0]
        : hue < 180
          ? [0, c, x]
          : hue < 240
            ? [0, x, c]
            : hue < 300
              ? [x, 0, c]
              : [c, 0, x]
  const hex = (value: number) =>
    Math.round((value + m) * 255)
      .toString(16)
      .padStart(2, '0')
  return `#${hex(r)}${hex(g)}${hex(b)}`
}

/**
 * 唯一值类目的默认配色：前 5 类沿用图层色板（与「单一符号」一致），
 * 超出后按黄金角散开色相——42 个省份也能一眼分开，且每次渲染结果稳定。
 */
export function categoryColor(index: number): string {
  if (index < CHART_FALLBACK.length) return chartHex(index)
  return hslToHex(28 + index * 137.508, 62, 52)
}

/** 逐类补色：已有颜色不动，只给新出现的类目补一个。 */
export function fillCategoryColors(
  current: Record<string, string>,
  keys: string[]
): Record<string, string> {
  const next = { ...current }
  let fresh = Object.keys(next).length
  for (const key of keys) {
    if (next[key]) continue
    next[key] = categoryColor(fresh)
    fresh += 1
  }
  return next
}

export interface Ramp {
  key: string
  label: string
  colors: string[]
}

/** 色带：栅格（terrain/gray/veg/blue）与分级色彩共用，都是「只影响显示」。 */
export const RAMPS: Ramp[] = [
  {
    key: 'terrain',
    label: '地形（绿–棕–白）',
    colors: ['#cfdcc0', '#a8bd8a', '#d8c9a0', '#b99a76', '#efe9e2'],
  },
  { key: 'gray', label: '灰度', colors: ['#f2f2f2', '#bdbdbd', '#7d7d7d', '#3d3d3d'] },
  { key: 'veg', label: '植被增强', colors: ['#eef3e6', '#c9dbb0', '#94bb7c', '#5c9257', '#2f6b3c'] },
  { key: 'blue', label: '蓝调', colors: ['#e8f0f4', '#9dc0d4', '#4f86a6', '#274f66'] },
  {
    key: 'warm',
    label: '暖色（黄–橙–红）',
    colors: ['#fdf0d5', '#f9d08b', '#f2a25c', '#d9683b', '#9e2a2b'],
  },
]

export function rampOf(key: string): Ramp {
  return RAMPS.find((ramp) => ramp.key === key) ?? RAMPS[0]
}

function mix(from: string, to: string, ratio: number): string {
  const parse = (hex: string) => {
    const clean = hex.replace('#', '')
    return [0, 2, 4].map((offset) => Number.parseInt(clean.slice(offset, offset + 2), 16))
  }
  const a = parse(from)
  const b = parse(to)
  const channel = (index: number) =>
    Math.round(a[index] + (b[index] - a[index]) * ratio)
      .toString(16)
      .padStart(2, '0')
  return `#${channel(0)}${channel(1)}${channel(2)}`
}

/** 按级数从色带里等距采样（线性插值），保证 3 级和 7 级都好看。 */
export function sampleRamp(key: string, count: number): string[] {
  const stops = rampOf(key).colors
  const total = Math.max(1, Math.round(count))
  if (total === 1) return [stops[Math.floor(stops.length / 2)]]
  const result: string[] = []
  for (let index = 0; index < total; index += 1) {
    const position = (index / (total - 1)) * (stops.length - 1)
    const low = Math.floor(position)
    const high = Math.min(stops.length - 1, low + 1)
    result.push(mix(stops[low], stops[high], position - low))
  }
  return result
}

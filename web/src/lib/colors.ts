/**
 * 把 theme.css 里的 CSS 变量解析成 canvas 能稳定吃下的色值。
 *
 * 图表库的颜色解析不认 `oklch()`，但浏览器自己的 canvas 认得：把值赋给
 * `fillStyle` 再读回来，Chrome 会规范化成 `#rrggbb`。这样色板仍然只有
 * theme.css 一处真相（§2.2），不在前端复制一份十六进制。
 */

/** 兜底色：canvas 不可用或变量缺失时用（等于浅色主题 --chart-N 的近似值）。 */
const FALLBACK: Record<string, string> = {
  '--chart-1': '#4d8bf0',
  '--chart-2': '#2aa5b0',
  '--chart-3': '#8a7ae0',
  '--chart-4': '#d46bb4',
  '--chart-5': '#3aa76d',
  '--muted-foreground': '#8a8a8a',
  '--border': '#e5e5e5',
  '--foreground': '#1a1a1a',
  '--card': '#ffffff',
  '--popover': '#ffffff',
  '--popover-foreground': '#1a1a1a',
}

let probe: CanvasRenderingContext2D | null = null

function context(): CanvasRenderingContext2D | null {
  if (probe) return probe
  if (typeof document === 'undefined') return null
  probe = document.createElement('canvas').getContext('2d')
  return probe
}

export function themeColor(name: string): string {
  const fallback = FALLBACK[name] ?? '#888888'
  const ctx = context()
  if (!ctx || typeof window === 'undefined') return fallback
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  if (!raw) return fallback
  // 先设一个哨兵：若赋值被忽略（非法色值），读回来仍是哨兵，就走兜底。
  ctx.fillStyle = '#010203'
  ctx.fillStyle = raw
  const normalized = String(ctx.fillStyle)
  return normalized === '#010203' ? fallback : normalized
}

/** `--chart-1…5` 顺序色板（按序列分配，超出用中性灰）。 */
export function chartPalette(): string[] {
  return [1, 2, 3, 4, 5].map((index) => themeColor(`--chart-${index}`))
}

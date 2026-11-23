/**
 * 把 theme.css 里的 CSS 颜色转成 MapLibre 能解析的 `rgba(...)`。
 *
 * MapLibre 的 color parser 只认 rgb/hsl/hex/命名色：`oklch(0.72 0.18 250)`
 * （theme.css 的 `--chart-1`）会让 `addLayer` 校验失败 —— 而 `addLayer` 校验
 * 失败只 `fire(error)` 不抛，错误边界接不到、也不会回退 `MapCanvas`，
 * 用户看到的是只剩控件的空图（DAIR-301）。
 *
 * 浏览器自己认得 oklch，所以借 1×1 canvas 画一个像素再读回来拿规范化结果。
 * 不能复用 `themeColor()`：它靠 `ctx.fillStyle` 归一，而 Chrome 对 oklch 的
 * `fillStyle` 回读仍是 `oklch(...)`（实测 Chrome 130），拿到的值 MapLibre 依旧不认。
 */

let probe: CanvasRenderingContext2D | null = null

function context(): CanvasRenderingContext2D | null {
  if (probe) return probe
  if (typeof document === 'undefined') return null
  const ctx = document.createElement('canvas').getContext('2d', {
    willReadFrequently: true,
  })
  if (!ctx) return null
  ctx.canvas.width = 1
  ctx.canvas.height = 1
  probe = ctx
  return ctx
}

/**
 * `raw` 转成 `rgba(r, g, b, a)`；无法转换时原样返回 `fallback`（调用方保证是
 * MapLibre 认识的十六进制）。
 */
export function toPaintColor(raw: string, fallback: string): string {
  const value = raw.trim()
  if (!value) return fallback
  const ctx = context()
  if (!ctx) return fallback
  // 哨兵：色值非法时浏览器会忽略 fillStyle 赋值，画出来仍是哨兵色，走兜底。
  ctx.fillStyle = '#010203'
  ctx.fillStyle = value
  ctx.clearRect(0, 0, 1, 1)
  ctx.fillRect(0, 0, 1, 1)
  const [r, g, b, a] = ctx.getImageData(0, 0, 1, 1).data
  if (r === 1 && g === 2 && b === 3 && a === 255) return fallback
  return `rgba(${r}, ${g}, ${b}, ${(a / 255).toFixed(3)})`
}

/**
 * 同上，但返回 `#rrggbb`——原生 `<input type="color">` 只认十六进制。
 * 色板 token（`--chart-N`）是 oklch，直接塞进 color 控件会被浏览器丢弃。
 */
export function toHexColor(raw: string, fallback: string): string {
  const value = raw.trim()
  if (!value) return fallback
  const ctx = context()
  if (!ctx) return fallback
  ctx.fillStyle = '#010203'
  ctx.fillStyle = value
  ctx.clearRect(0, 0, 1, 1)
  ctx.fillRect(0, 0, 1, 1)
  const [r, g, b, a] = ctx.getImageData(0, 0, 1, 1).data
  if (r === 1 && g === 2 && b === 3 && a === 255) return fallback
  const hex = (channel: number) => channel.toString(16).padStart(2, '0')
  return `#${hex(r)}${hex(g)}${hex(b)}`
}
import { useEffect, useRef } from 'react'

import type { GeoJSONFeature } from '@/lib/api'
import { useThemeStore } from '@/lib/theme'
import { cn } from '@/lib/utils'

type BBox = [number, number, number, number]

export interface MapCanvasProps {
  features: GeoJSONFeature[]
  bbox?: BBox | null
  className?: string
}

/** 取一个几何的所有顶点（GeoJSON 的坐标是嵌套数组，这里按几何类型拆）。 */
function collectPoints(geometry: GeoJSONFeature['geometry'], out: number[][]) {
  if (!geometry) return
  const pushRing = (ring: unknown) => {
    if (!Array.isArray(ring)) return
    for (const point of ring) {
      if (Array.isArray(point) && point.length >= 2) out.push(point as number[])
    }
  }
  const coordinates = geometry.coordinates as unknown
  switch (geometry.type) {
    case 'Point':
      pushRing([coordinates])
      break
    case 'MultiPoint':
    case 'LineString':
      for (const point of (coordinates as unknown[]) ?? []) pushRing([point])
      break
    case 'MultiLineString':
    case 'Polygon':
      for (const ring of (coordinates as unknown[]) ?? []) pushRing(ring)
      break
    case 'MultiPolygon':
      for (const polygon of (coordinates as unknown[]) ?? []) {
        for (const ring of (polygon as unknown[]) ?? []) pushRing(ring)
      }
      break
    default:
      break
  }
}

function bboxOf(features: GeoJSONFeature[]): BBox | null {
  const points: number[][] = []
  for (const feature of features) collectPoints(feature.geometry, points)
  if (points.length === 0) return null
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const [x, y] of points) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue
    if (x < minX) minX = x
    if (y < minY) minY = y
    if (x > maxX) maxX = x
    if (y > maxY) maxY = y
  }
  if (!Number.isFinite(minX) || !Number.isFinite(minY)) return null
  return [minX, minY, maxX, maxY]
}

function readVar(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback
  const value = window.getComputedStyle(document.documentElement).getPropertyValue(name)
  return value.trim() || fallback
}

function formatTick(value: number, span: number): string {
  const digits = span > 5000 ? 0 : span > 100 ? 1 : span > 1 ? 3 : 5
  return value.toFixed(digits)
}

/**
 * 离线地图：只画几何，不引 Leaflet、不请求瓦片（计划 §6 / §9）。
 *
 * 颜色全部读 CSS 变量，主题切换（``.dark``）后靠 ``useThemeStore`` 触发重绘，
 * 不用两套硬编码色值——这是 §8 清单第 9 条的机器可验点。
 */
export function MapCanvas({ features, bbox, className }: MapCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const theme = useThemeStore((state) => state.theme)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const parent = canvas.parentElement
    if (!parent) return

    const draw = () => {
      const width = Math.max(1, parent.clientWidth)
      const height = Math.max(1, parent.clientHeight)
      const dpr = window.devicePixelRatio || 1
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`

      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, width, height)

      const background = readVar('--card', '#ffffff')
      const border = readVar('--border', '#e5e7eb')
      const muted = readVar('--muted-foreground', '#6b7280')
      const accent = readVar('--chart-1', '#2563eb')
      const accentAlt = readVar('--chart-2', '#0ea5e9')

      ctx.fillStyle = background
      ctx.fillRect(0, 0, width, height)

      const extent = bbox ?? bboxOf(features)
      if (!extent) {
        ctx.fillStyle = muted
        ctx.font = '12px Public Sans, system-ui, sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText('没有可绘制的几何', width / 2, height / 2)
        return
      }

      const [minX, minY, maxX, maxY] = extent
      const spanX = maxX - minX || 1
      const spanY = maxY - minY || 1
      const pad = 22
      const scale = Math.min((width - pad * 2) / spanX, (height - pad * 2) / spanY)
      const offsetX = (width - spanX * scale) / 2
      const offsetY = (height - spanY * scale) / 2

      const project = (x: number, y: number): [number, number] => [
        offsetX + (x - minX) * scale,
        height - offsetY - (y - minY) * scale,
      ]

      // 网格：4×4，读 --border，深色模式下自动变淡。
      ctx.strokeStyle = border
      ctx.lineWidth = 1
      ctx.globalAlpha = 0.9
      for (let index = 0; index <= 4; index += 1) {
        const gx = offsetX + (spanX * scale * index) / 4
        const gy = height - offsetY - (spanY * scale * index) / 4
        ctx.beginPath()
        ctx.moveTo(gx, offsetY - 6)
        ctx.lineTo(gx, height - offsetY + 6)
        ctx.stroke()
        ctx.beginPath()
        ctx.moveTo(offsetX - 6, gy)
        ctx.lineTo(width - offsetX + 6, gy)
        ctx.stroke()
      }
      ctx.globalAlpha = 1

      const drawRing = (ring: unknown, close: boolean) => {
        if (!Array.isArray(ring) || ring.length === 0) return
        ctx.beginPath()
        ;(ring as number[][]).forEach((point, index) => {
          if (!Array.isArray(point) || point.length < 2) return
          const [px, py] = project(point[0], point[1])
          if (index === 0) ctx.moveTo(px, py)
          else ctx.lineTo(px, py)
        })
        if (close) ctx.closePath()
      }

      for (const feature of features) {
        const geometry = feature.geometry
        if (!geometry) continue
        const coordinates = geometry.coordinates as unknown

        if (geometry.type === 'Point' || geometry.type === 'MultiPoint') {
          const points: number[][] = []
          collectPoints(geometry, points)
          ctx.fillStyle = accent
          for (const [x, y] of points) {
            const [px, py] = project(x, y)
            ctx.beginPath()
            ctx.arc(px, py, 2.2, 0, Math.PI * 2)
            ctx.fill()
          }
          continue
        }

        if (geometry.type === 'LineString' || geometry.type === 'MultiLineString') {
          const lines =
            geometry.type === 'LineString'
              ? [coordinates as number[][]]
              : ((coordinates as number[][][]) ?? [])
          ctx.strokeStyle = accent
          ctx.lineWidth = 1.2
          for (const line of lines) {
            drawRing(line, false)
            ctx.stroke()
          }
          continue
        }

        const polygons =
          geometry.type === 'Polygon'
            ? [coordinates as number[][][]]
            : ((coordinates as number[][][][]) ?? [])
        for (const polygon of polygons) {
          ctx.beginPath()
          for (const ring of polygon ?? []) {
            if (!Array.isArray(ring) || ring.length === 0) continue
            ;(ring as number[][]).forEach((point, index) => {
              if (!Array.isArray(point) || point.length < 2) return
              const [px, py] = project(point[0], point[1])
              if (index === 0) ctx.moveTo(px, py)
              else ctx.lineTo(px, py)
            })
            ctx.closePath()
          }
          ctx.fillStyle = accent
          ctx.globalAlpha = 0.18
          // evenodd：内环（洞）自动镂空，不用逐环做裁切。
          ctx.fill('evenodd')
          ctx.globalAlpha = 1
          ctx.strokeStyle = accentAlt
          ctx.lineWidth = 0.9
          ctx.stroke()
        }
      }

      // 范围标注：把 bbox 的两角写出来，肉眼能核对「是不是这块地方」。
      ctx.fillStyle = muted
      ctx.font = '10px Public Sans, system-ui, sans-serif'
      ctx.textAlign = 'left'
      ctx.fillText(
        `${formatTick(minX, spanX)}, ${formatTick(minY, spanY)}`,
        6,
        height - 6
      )
      ctx.textAlign = 'right'
      ctx.fillText(
        `${formatTick(maxX, spanX)}, ${formatTick(maxY, spanY)}`,
        width - 6,
        14
      )
    }

    draw()
    const observer = new ResizeObserver(draw)
    observer.observe(parent)
    return () => observer.disconnect()
    // theme 进依赖：切深浅色时 CSS 变量变了，必须重画一次。
  }, [features, bbox, theme])

  return (
    <div className={cn('relative overflow-hidden rounded-lg border border-border', className)}>
      <canvas ref={canvasRef} role="img" aria-label="图层范围预览（离线 canvas）" />
    </div>
  )
}

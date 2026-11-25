import { ApiError, type DatasetItem, type LayerPreview } from '@/lib/api'

/** 目录/图层/元数据共用的格式化与 CRS 语义（从 DAIR-325 前的 data.tsx 收拢过来）。 */

export const KIND_LABEL: Record<string, string> = {
  vector: '矢量',
  feature_class: '要素类',
  raster: '栅格',
  table: '表格',
  gdb: '文件数据库',
  project: '工程',
  other: '其他',
}

export function kindLabel(kind: string): string {
  return KIND_LABEL[kind] ?? kind
}

export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return value.toLocaleString('zh-CN')
}

export function formatSample(value: number): string {
  if (!Number.isFinite(value)) return String(value)
  if (Math.abs(value) >= 1000) return value.toFixed(0)
  if (Math.abs(value) >= 1) return value.toFixed(2)
  return value.toFixed(4)
}

export function formatExtent(extent: number[] | null | undefined): string {
  if (!extent || extent.length < 4) return '—'
  const [minX, minY, maxX, maxY] = extent
  return `${formatSample(minX)}, ${formatSample(minY)} → ${formatSample(maxX)}, ${formatSample(maxY)}`
}

export function formatDate(value: number | string | null | undefined): string {
  if (value === null || value === undefined || value === '') return '—'
  // 后端 mtime 是秒（float），acquired_at 是 ISO 字符串，两条路都要能走。
  const date = new Date(typeof value === 'number' ? value * 1000 : value)
  return Number.isNaN(date.getTime())
    ? String(value)
    : date.toLocaleString('zh-CN', { hour12: false })
}

export function formatBytes(size: number | null | undefined): string {
  if (size === null || size === undefined || !Number.isFinite(size)) return '—'
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  if (size < 1024 * 1024 * 1024) return `${(size / 1024 / 1024).toFixed(1)} MB`
  return `${(size / 1024 / 1024 / 1024).toFixed(2)} GB`
}

export function crsLabel(item: Pick<DatasetItem, 'crs_name' | 'wkid'>): string {
  if (item.wkid) return `${item.crs_name || 'EPSG'} · ${item.wkid}`
  return item.crs_name || '未知'
}

/** 目录行尾的徽标：有 WKID 写 EPSG 号，否则给坐标系前缀（未知就不挂徽标）。 */
export function crsBadge(item: Pick<DatasetItem, 'crs_name' | 'wkid'>): string | null {
  if (item.wkid) return `EPSG:${item.wkid}`
  const name = item.crs_name?.trim()
  if (!name || /^unknown$/i.test(name)) return null
  return name.slice(0, 18)
}

/** 由 WKID / CRS 名归并出的投影族（spec §4：不放原始 WKT）。 */
export function projectionFamily(item: Pick<DatasetItem, 'crs_name' | 'wkid'>): string {
  const name = item.crs_name || ''
  if (/gauss|_gk_|gauss[-_ ]?kruger/i.test(name)) {
    const cm = /cm[\s_]*([0-9.]+)/i.exec(name)?.[1]
    return cm ? `Gauss-Kruger（3° 带 · CM ${cm}°E）` : 'Gauss-Kruger'
  }
  if (/utm/i.test(name)) {
    const zone = /zone[\s_]*([0-9]+)\s*([NS])?/i.exec(name)
    return zone ? `UTM（6° 带 · Zone ${zone[1]}${zone[2] ?? ''}）` : 'UTM'
  }
  if (/wgs_1984|gcs_|geographic|china_geodetic|4326/i.test(name) || item.wkid === 4326) {
    return '地理坐标（经纬度）'
  }
  return '未知投影'
}

export function datumOf(item: Pick<DatasetItem, 'wkid'>): string {
  const wkid = item.wkid
  if (!wkid) return '—'
  if (wkid === 4326 || wkid === 3857) return 'WGS 1984'
  // UTM 南北半球 60 带：32601–32660 / 32701–32760 都是 WGS 1984 基准。
  if ((wkid >= 32601 && wkid <= 32660) || (wkid >= 32701 && wkid <= 32760)) return 'WGS 1984'
  if ((wkid >= 4490 && wkid <= 4554) || wkid === 4479) return 'CGCS2000'
  if (wkid === 4214) return 'Beijing 1954'
  if (wkid === 4610) return 'Xian 1980'
  return '—'
}

export function unitOf(item: Pick<DatasetItem, 'wkid'>): string {
  if (item.wkid === 4326) return '度'
  return item.wkid ? '米' : '—'
}

export interface PreviewIssue {
  title: string
  reason: string
  suggestion?: string
  elapsedMs?: number
  /** 信息态（不是错误）：例如非空间表只能看属性表。 */
  info?: boolean
}

/**
 * 视口里零要素（DAIR-391）：带 bbox 的预览回 200 + 空 features，不是失败。
 *
 * 「拖到数据外」是正常操作：回 422 会让每个图层弹一条红横幅「预览失败」。这里和
 * `describePreviewError` 的 `no_geometry` 一样走信息态，图例只提示、不报警。
 */
export function describeEmptyView(
  preview: Pick<LayerPreview, 'total_in_view'>
): PreviewIssue | null {
  if (preview.total_in_view !== 0) return null
  return {
    title: '视口内无要素',
    reason: '当前地图视野不在该图层的数据范围内，移回图层范围即可显示',
    info: true,
  }
}

/** 把服务端的结构化 detail（preview_timeout / kernel_unavailable / no_geometry …）解成给人看的一句话。 */
export function describePreviewError(error: Error): PreviewIssue {
  const detail = error instanceof ApiError ? error.detail : undefined
  if (detail && typeof detail === 'object') {
    const record = detail as Record<string, unknown>
    const code = typeof record.code === 'string' ? record.code : ''
    const reason = typeof record.reason === 'string' ? record.reason : error.message
    const suggestion =
      typeof record.suggestion === 'string' ? record.suggestion : undefined
    const elapsedMs = typeof record.elapsed_ms === 'number' ? record.elapsed_ms : undefined
    if (code === 'preview_timeout') {
      const seconds = typeof record.timeout_s === 'number' ? record.timeout_s : 10
      return { title: `预览超时（${seconds}s）`, reason, suggestion, elapsedMs }
    }
    if (code === 'kernel_unavailable') {
      return { title: 'ArcGIS 内核不可用', reason, suggestion, elapsedMs }
    }
    if (code === 'no_geometry') {
      // 非空间表：不是失败，是这类数据集本来就画不出图层。
      return { title: '仅属性表', reason, suggestion, elapsedMs, info: true }
    }
    return { title: '预览失败', reason, suggestion, elapsedMs }
  }
  return { title: '预览失败', reason: error.message }
}

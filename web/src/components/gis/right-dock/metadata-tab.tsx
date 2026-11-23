import { Loader2 } from 'lucide-react'

import type { DatasetItem, DatasetProfile, RasterBand, RasterSummary } from '@/lib/api'

import {
  crsBadge,
  datumOf,
  formatBytes,
  formatDate,
  formatNumber,
  kindLabel,
  projectionFamily,
  unitOf,
} from '../format'
import { describeApiError } from './errors'
import { Field, Hint, Section, Todo, Unavailable } from './parts'

export interface MetadataTabProps {
  item: DatasetItem
  profile?: DatasetProfile
  loading: boolean
  error: Error | null
}

/**
 * 「元数据」tab（spec §4 ④）：空间参考 / 范围 / 要素（或栅格+波段）/ 数据属性。
 *
 * 所有值都来自 `/api/catalog/datasets/{path}/profile`，逐字段对照——面板里
 * 显示什么，接口就返回什么，不在前端二次推算（除了宽高/中心这种四则运算）。
 * catalog 只读、不打开数据文件，所以内核不可用时这里照样能看。
 */
export function MetadataTab({ item, profile, loading, error }: MetadataTabProps) {
  const data = profile ?? item
  const isRaster = item.kind === 'raster'
  const raster = (profile?.raster ?? {}) as RasterSummary
  const bands = profile?.bands ?? []
  const bandsInfo = profile?.bands_info

  return (
    <>
      {error ? (
        <Unavailable
          title={describeApiError(error).title}
          reason={`${describeApiError(error).reason}（下面先用 catalog 列表里的字段）`}
        />
      ) : null}
      {loading ? (
        <div className="mb-2 flex items-center gap-1.5 text-[10.5px] text-muted-foreground">
          <Loader2 className="size-3 animate-spin" /> 正在读取画像…
        </div>
      ) : null}

      <Section title="空间参考">
        <Field label="坐标系" value={data.crs_name || '未知'} wrap />
        <Field label="WKID / EPSG" value={data.wkid ? `${data.wkid} · EPSG:${data.wkid}` : '—'} />
        <Field label="投影" value={projectionFamily(data)} wrap />
        <Field label="基准面" value={datumOf(data)} />
        <Field label="坐标单位" value={unitOf(data)} />
        <Field label="地图显示" value="EPSG:3857（Web Mercator）" wrap />
        <Field label="轴序" value="X = Easting，Y = Northing" />
        {!data.wkid ? (
          <Hint>catalog 没记到 WKID：坐标单位与基准面推不出来，所以留「—」而不是猜一个。</Hint>
        ) : null}
      </Section>

      <Section title="范围（图层 CRS）">
        <ExtentFields extent={data.extent} unit={unitOf(data)} />
      </Section>

      {isRaster ? (
        <Section title="栅格">
          <Field label="波段数" value={formatNumber(bandsInfo?.band_count ?? raster.band_count)} />
          <Field label="像元大小" value={cellSize(raster)} />
          <Field label="行列数" value={rowsCols(raster)} />
          <Field label="NoData" value={formatNumber(raster.nodata as number | null)} />
          <Field label="最小值" value={formatNumber(raster.min as number | null)} />
          <Field label="最大值" value={formatNumber(raster.max as number | null)} />
          <Field label="平均值" value={formatNumber(raster.mean as number | null)} />
          <Hint>
            以上是 catalog 扫描时算的**整幅**汇总；逐波段明细见下表。
          </Hint>
        </Section>
      ) : (
        <Section title="要素">
          <Field label="要素数" value={formatNumber(data.feature_count)} />
          <Field label="几何类型" value={data.geom_type || '—'} />
          <Field label="字段数" value={formatNumber(data.field_count ?? profile?.fields.length)} />
          <Field label="统计口径" value="来自 catalog（不打开数据文件）" wrap />
        </Section>
      )}

      {isRaster ? (
        <Section
          title="波段明细"
          badge={bands.length === 0 || bands.some((band) => band.unavailable) ? <Todo /> : undefined}
        >
          {bands.length > 0 ? (
            <BandTable bands={bands} />
          ) : (
            <Unavailable
              title="逐波段元数据待补"
              reason={
                bandsInfo?.unavailable ||
                '接口没有返回逐波段数据（无 ArcGIS 内核或该栅格读不出波段属性）'
              }
            />
          )}
          {bandsInfo?.notes?.length ? <Hint>{bandsInfo.notes.join('；')}</Hint> : null}
          <Hint>
            读不到的格子标「待补」并在悬停里给原因，不把 min/max 写成 0 冒充算过。
          </Hint>
        </Section>
      ) : null}

      <Section title="数据属性">
        <Field label="文件大小" value={formatBytes(data.size)} />
        <Field label="更新时间" value={formatDate(data.mtime)} />
        <Field label="接入时间" value={formatDate(data.acquired_at)} />
        <Field label="来源" value={data.source || '—'} wrap />
        <Field label="许可" value={data.license || '—'} wrap />
        <Field label="类型" value={kindLabel(data.kind)} />
        <Field label="路径" value={data.path} mono wrap />
        {crsBadge(data) ? <Hint>目录树行尾徽标：{crsBadge(data)}</Hint> : null}
      </Section>
    </>
  )
}

function num(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  if (Math.abs(value) >= 1000) return value.toFixed(2)
  if (Math.abs(value) >= 1) return value.toFixed(4)
  return value.toFixed(6)
}

function ExtentFields({ extent, unit }: { extent: number[] | null | undefined; unit: string }) {
  if (!extent || extent.length < 4) {
    return <Field label="范围" value="—（catalog 没记 bbox）" wrap />
  }
  const [west, south, east, north] = extent
  const width = east - west
  const height = north - south
  const suffix = unit === '—' ? '' : ` ${unit}`
  return (
    <>
      <Field label="西 / 左" value={`${num(west)}${suffix}`} />
      <Field label="南 / 下" value={`${num(south)}${suffix}`} />
      <Field label="东 / 右" value={`${num(east)}${suffix}`} />
      <Field label="北 / 上" value={`${num(north)}${suffix}`} />
      <Field label="宽 × 高" value={`${num(width)} × ${num(height)}${suffix}`} />
      <Field
        label="中心点"
        value={`${num((west + east) / 2)} , ${num((south + north) / 2)}`}
        mono
      />
    </>
  )
}

function cellSize(raster: RasterSummary): string {
  const value = raster.cell_size
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—'
  return `${num(Number(value))} × ${num(Number(value))}`
}

function rowsCols(raster: RasterSummary): string {
  const width = raster.width
  const height = raster.height
  if (width === null || width === undefined || height === null || height === undefined) {
    return '—'
  }
  return `${formatNumber(Number(width))} × ${formatNumber(Number(height))} px`
}

function BandTable({ bands }: { bands: RasterBand[] }) {
  return (
    <table className="w-full border-collapse text-[10.5px]">
      <thead>
        <tr className="text-muted-foreground">
          {['#', '名称', '类型', '最小', '最大', '均值', 'NoData'].map((title) => (
            <th
              key={title}
              className="h-[20px] border-b border-border px-1 text-left font-semibold whitespace-nowrap"
            >
              {title}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {bands.map((band) => (
          <tr key={band.band}>
            <td className="h-[21px] border-b border-border/60 px-1 tabular-nums">{band.band}</td>
            <BandCell value={band.name || `Band_${band.band}`} mono />
            <BandCell value={band.dtype} />
            <BandCell value={band.min} numeric />
            <BandCell value={band.max} numeric />
            <BandCell value={band.mean} numeric />
            <BandCell value={band.nodata} numeric />
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function BandCell({
  value,
  mono,
  numeric,
}: {
  value: string | number | null | undefined
  mono?: boolean
  numeric?: boolean
}) {
  const missing = value === null || value === undefined || value === ''
  return (
    <td
      className={`h-[21px] border-b border-border/60 px-1 whitespace-nowrap ${
        mono ? 'font-mono text-[10px]' : ''
      } ${numeric ? 'tabular-nums' : ''} ${missing ? 'text-muted-foreground italic' : ''}`}
      title={missing ? '待补：该字段读不到（无内核或栅格未算统计值）' : String(value)}
    >
      {missing ? '待补' : typeof value === 'number' ? num(value) : value}
    </td>
  )
}

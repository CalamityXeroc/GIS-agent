import { useQuery } from '@tanstack/react-query'
import {
  AlertTriangle,
  Database,
  FileSpreadsheet,
  Folder,
  FolderOpen,
  Grid2x2,
  HardDrive,
  Layers,
  List,
  RefreshCw,
  Ruler,
  Search,
} from 'lucide-react'
import { useEffect, useMemo, useState, type ReactNode } from 'react'

import { MapCanvas } from '@/components/layers/map-canvas'
import { PageHeader } from '@/components/layout/page-header'
import { Button } from '@/components/ui/button'
import { DataTable, type DataTableColumn } from '@/components/ui/data-table'
import { Empty } from '@/components/ui/empty'
import { IconBadge } from '@/components/ui/icon-badge'
import { Sheet, SheetContent, SheetTitle } from '@/components/ui/sheet'
import { Skeleton } from '@/components/ui/skeleton'
import {
  ApiError,
  getDatasetProfile,
  getLayerPreview,
  listDatasets,
  type DatasetField,
  type DatasetItem,
  type DatasetProfile,
  type LayerPreview,
} from '@/lib/api'
import { cn } from '@/lib/utils'

type ViewMode = 'cards' | 'table'

interface ContainerGroup {
  key: string
  label: string
  isGdb: boolean
  items: DatasetItem[]
}

const KIND_LABEL: Record<string, string> = {
  vector: '矢量',
  feature_class: '要素类',
  raster: '栅格',
  table: '表格',
  gdb: '文件数据库',
  project: '工程',
  other: '其他',
}

function containerOf(item: DatasetItem): string {
  if (item.container) return item.container.replace(/\\/g, '/')
  const normalized = item.path.replace(/\\/g, '/')
  const index = normalized.lastIndexOf('/')
  return index > 0 ? normalized.slice(0, index) : '未分组'
}

function containerLabel(key: string): string {
  if (key === '未分组') return key
  const parts = key.split('/').filter(Boolean)
  return parts[parts.length - 1] || key
}

function groupDatasets(items: DatasetItem[]): ContainerGroup[] {
  const groups = new Map<string, ContainerGroup>()
  for (const item of items) {
    const key = containerOf(item)
    const group = groups.get(key) ?? {
      key,
      label: containerLabel(key),
      isGdb: key.toLowerCase().endsWith('.gdb'),
      items: [],
    }
    group.items.push(item)
    groups.set(key, group)
  }
  for (const group of groups.values()) {
    group.items.sort((a, b) => a.name.localeCompare(b.name, 'zh-CN'))
  }
  return [...groups.values()].sort((a, b) => a.label.localeCompare(b.label, 'zh-CN'))
}

function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return value.toLocaleString('zh-CN')
}

function formatSample(value: number): string {
  if (!Number.isFinite(value)) return String(value)
  if (Math.abs(value) >= 1000) return value.toFixed(0)
  if (Math.abs(value) >= 1) return value.toFixed(2)
  return value.toFixed(4)
}

function formatExtent(extent: number[] | null | undefined): string {
  if (!extent || extent.length < 4) return '—'
  const [minX, minY, maxX, maxY] = extent
  return `${formatSample(minX)}, ${formatSample(minY)} → ${formatSample(maxX)}, ${formatSample(maxY)}`
}

function crsLabel(item: Pick<DatasetItem, 'crs_name' | 'wkid'>): string {
  if (item.wkid) {
    return `${item.crs_name || 'EPSG'} · ${item.wkid}`
  }
  return item.crs_name || '未知'
}

function formatDate(value: number | string | null | undefined): string {
  if (value === null || value === undefined || value === '') return '—'
  // 后端 mtime 是秒（float），acquired_at 是 ISO 字符串，两条路都要能走。
  const date = new Date(typeof value === 'number' ? value * 1000 : value)
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString('zh-CN', { hour12: false })
}

function formatBytes(size: number | null | undefined): string {
  if (size === null || size === undefined || !Number.isFinite(size)) return '—'
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  if (size < 1024 * 1024 * 1024) return `${(size / 1024 / 1024).toFixed(1)} MB`
  return `${(size / 1024 / 1024 / 1024).toFixed(2)} GB`
}

function DatasetIcon({ item, className }: { item: DatasetItem; className?: string }) {
  if (item.kind === 'raster') return <FileSpreadsheet className={className} />
  if (item.kind === 'table') return <FileSpreadsheet className={className} />
  if (item.kind === 'gdb') return <HardDrive className={className} />
  return <Layers className={className} />
}

/** 左树节点：容器分组（.gdb / 目录）→ 图层。 */
function ContainerTree({
  groups,
  activeKey,
  onSelect,
}: {
  groups: ContainerGroup[]
  activeKey: string | null
  onSelect: (key: string) => void
}) {
  return (
    <div className="flex flex-col gap-1">
      {groups.map((group) => {
        const active = group.key === activeKey
        return (
          <details key={group.key} open className="group">
            <summary
              className={cn(
                'flex cursor-pointer list-none items-center gap-2 rounded-md px-2 py-1.5 text-xs transition-colors',
                active
                  ? 'bg-accent font-medium text-accent-foreground'
                  : 'text-foreground/75 hover:bg-accent/60'
              )}
              onClick={() => onSelect(group.key)}
            >
              {group.isGdb ? (
                <HardDrive className="size-3.5 shrink-0 text-muted-foreground" />
              ) : (
                <FolderOpen className="size-3.5 shrink-0 text-muted-foreground" />
              )}
              <span className="min-w-0 flex-1 truncate" title={group.key}>
                {group.label}
              </span>
              <span className="tabular-nums text-muted-foreground/70">{group.items.length}</span>
            </summary>
            <div className="mt-0.5 flex flex-col gap-0.5 pl-4">
              {group.items.map((item) => (
                <div
                  key={item.path}
                  className="flex items-center gap-1.5 rounded-md px-2 py-1 text-[11px] text-muted-foreground"
                  title={item.path}
                >
                  <DatasetIcon item={item} className="size-3 shrink-0" />
                  <span className="min-w-0 flex-1 truncate">{item.name}</span>
                  <span className="shrink-0 tabular-nums text-muted-foreground/60">
                    {formatNumber(item.feature_count)}
                  </span>
                </div>
              ))}
            </div>
          </details>
        )
      })}
    </div>
  )
}

function AttributeCell({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] text-muted-foreground">{label}</dt>
      <dd className="mt-0.5 truncate text-xs text-foreground" title={typeof children === 'string' ? children : undefined}>
        {children}
      </dd>
    </div>
  )
}

function NullRatioBar({ ratio }: { ratio: number | null }) {
  const value = ratio === null || ratio === undefined ? 0 : Math.max(0, Math.min(1, ratio))
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-20 overflow-hidden rounded-full bg-muted">
        <div
          className={cn('h-full rounded-full', value > 0.3 ? 'bg-warning' : 'bg-chart-1/70')}
          style={{ width: `${Math.max(2, value * 100)}%` }}
        />
      </div>
      <span className="tabular-nums text-[11px] text-muted-foreground">
        {ratio === null || ratio === undefined ? '—' : `${Math.round(value * 100)}%`}
      </span>
    </div>
  )
}

function sampleText(field: DatasetField): string {
  const samples = (field.samples || []).slice(0, 3)
  if (samples.length === 0) return '—'
  return samples
    .map((value) => {
      if (value === null || value === undefined) return 'null'
      if (value === '') return '(空)'
      return String(value)
    })
    .join(' · ')
}

function FieldTable({ fields }: { fields: DatasetField[] }) {
  if (fields.length === 0) {
    return <p className="text-xs text-muted-foreground">catalog 里还没有这张表的字段画像。</p>
  }
  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr className="bg-[var(--table-header)] text-[11px] font-medium text-muted-foreground">
            <th className="px-3 py-2 text-left">字段</th>
            <th className="px-3 py-2 text-left">类型</th>
            <th className="px-3 py-2 text-left">样例</th>
            <th className="px-3 py-2 text-left">空值率</th>
            <th className="px-3 py-2 text-right">数值范围</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {fields.map((field) => (
            <tr key={field.name}>
              <td className="px-3 py-2">
                <div className="font-medium text-foreground">{field.name}</div>
                {field.alias ? (
                  <div className="text-[11px] text-muted-foreground">{field.alias}</div>
                ) : null}
              </td>
              <td className="px-3 py-2 text-muted-foreground">{field.type || '—'}</td>
              <td className="max-w-[12rem] truncate px-3 py-2 text-muted-foreground" title={sampleText(field)}>
                {sampleText(field)}
              </td>
              <td className="px-3 py-2">
                <NullRatioBar ratio={field.null_ratio} />
              </td>
              <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                {field.numeric && field.min !== undefined && field.max !== undefined
                  ? `${formatSample(field.min)} ~ ${formatSample(field.max)}`
                  : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function PreviewMeta({ preview }: { preview: LayerPreview }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
      <span>
        返回 <b className="tabular-nums text-foreground">{preview.returned_features}</b> /{' '}
        <span className="tabular-nums">{formatNumber(preview.feature_count)}</span> 要素
      </span>
      <span>
        顶点 <span className="tabular-nums">{formatNumber(preview.point_count)}</span>
      </span>
      {preview.truncated ? (
        <span className="text-warning">已抽稀（上限 {preview.max_features}）</span>
      ) : null}
      <span>
        {preview.cache.hit ? '缓存命中' : '缓存未命中'} ·{' '}
        <span className="tabular-nums">{preview.cache.elapsed_ms} ms</span>
        {preview.cache.read_ms !== undefined ? (
          <span className="text-muted-foreground/70">
            {' '}
            （读几何 {preview.cache.read_ms} ms）
          </span>
        ) : null}
      </span>
    </div>
  )
}

function CrsBanner({ warning }: { warning: string }) {
  return (
    <div
      role="alert"
      className="flex items-start gap-2 rounded-lg border border-warning/40 bg-warning/10 px-3 py-2.5 text-xs leading-relaxed text-foreground"
    >
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warning" />
      <div>
        <span className="font-medium text-warning">{warning}</span>
        <span className="text-muted-foreground">
          {' '}
          —— 图形按数据原始坐标直接绘制，只用来核对形状与范围，不能据此判断真实
          经纬度位置。
        </span>
      </div>
    </div>
  )
}

interface PreviewIssue {
  title: string
  reason: string
  suggestion?: string
  elapsedMs?: number
}

/** 把服务端的结构化 detail（preview_timeout / kernel_unavailable …）解成给人看的一句话。 */
function describePreviewError(error: Error): PreviewIssue {
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
    return { title: '预览失败', reason, suggestion, elapsedMs }
  }
  return { title: '预览失败', reason: error.message }
}

function PreviewError({
  error,
  retrying,
  onRetry,
}: {
  error: Error
  retrying: boolean
  onRetry: () => void
}) {
  const issue = describePreviewError(error)
  return (
    <div
      role="alert"
      className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-3 text-xs"
    >
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 size-4 shrink-0 text-destructive" />
        <div className="min-w-0 flex-1">
          <div className="font-medium text-destructive">{issue.title}</div>
          <p className="mt-1 leading-relaxed text-foreground/80">{issue.reason}</p>
          {issue.suggestion ? (
            <p className="mt-1 leading-relaxed text-muted-foreground">{issue.suggestion}</p>
          ) : null}
          {issue.elapsedMs !== undefined ? (
            <p className="mt-1 text-[11px] tabular-nums text-muted-foreground">
              服务端耗时 {Math.round(issue.elapsedMs)} ms
            </p>
          ) : null}
        </div>
        <Button variant="outline" size="sm" onClick={onRetry} disabled={retrying}>
          <RefreshCw className={cn('size-3.5', retrying && 'animate-spin')} />
          重试
        </Button>
      </div>
    </div>
  )
}

function LayerDetail({ layer }: { layer: DatasetItem }) {
  const profile = useQuery<DatasetProfile>({
    queryKey: ['dataset-profile', layer.path],
    queryFn: () => getDatasetProfile(layer.path),
  })
  const preview = useQuery<LayerPreview>({
    queryKey: ['layer-preview', layer.path],
    queryFn: () => getLayerPreview(layer.path),
    retry: false,
  })

  return (
    <div className="flex flex-col gap-4 pb-4">
      <div>
        <div className="flex items-center gap-2">
          <IconBadge tone={layer.kind === 'raster' ? 'chart3' : 'chart1'}>
            <DatasetIcon item={layer} />
          </IconBadge>
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold text-foreground" title={layer.path}>
              {layer.name}
            </div>
            <div className="text-[11px] text-muted-foreground">
              {KIND_LABEL[layer.kind] ?? layer.kind}
              {layer.geom_type ? ` · ${layer.geom_type}` : ''}
            </div>
          </div>
        </div>
        <div className="mt-2 truncate text-[11px] text-muted-foreground" title={layer.path}>
          {layer.path}
        </div>
      </div>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 rounded-lg border border-border bg-card p-3 sm:grid-cols-3">
        <AttributeCell label="要素数">{formatNumber(layer.feature_count)}</AttributeCell>
        <AttributeCell label="坐标系">{crsLabel(layer)}</AttributeCell>
        <AttributeCell label="字段数">{formatNumber(layer.field_count)}</AttributeCell>
        <AttributeCell label="范围">{formatExtent(layer.extent)}</AttributeCell>
        <AttributeCell label="来源">{layer.source || '—'}</AttributeCell>
        <AttributeCell label="许可">{layer.license || 'unknown'}</AttributeCell>
        <AttributeCell label="文件大小">{formatBytes(layer.size)}</AttributeCell>
        <AttributeCell label="更新时间">{formatDate(layer.mtime)}</AttributeCell>
        <AttributeCell label="接入时间">{formatDate(layer.acquired_at)}</AttributeCell>
      </dl>

      {preview.data?.crs_warning ? <CrsBanner warning={preview.data.crs_warning} /> : null}

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="text-xs font-medium text-foreground">离线范围预览</div>
          {preview.isFetching ? (
            <span className="text-[11px] text-muted-foreground">读取中…</span>
          ) : null}
        </div>
        {preview.isLoading ? (
          <Skeleton className="h-64 w-full rounded-lg" />
        ) : preview.isError ? (
          <PreviewError
            error={preview.error as Error}
            retrying={preview.isFetching}
            onRetry={() => void preview.refetch()}
          />
        ) : preview.data ? (
          <>
            <MapCanvas
              features={preview.data.geojson.features}
              bbox={preview.data.bbox}
              className="h-64 w-full"
            />
            {preview.data.oversized ? (
              <p className="text-[11px] text-warning">
                图层过大，无法在 5MB 内给出几何；这里只展示范围框。
              </p>
            ) : null}
            <PreviewMeta preview={preview.data} />
          </>
        ) : null}
      </div>

      <div className="space-y-2">
        <div className="flex items-baseline justify-between gap-2">
          <div className="text-xs font-medium text-foreground">字段画像</div>
          {profile.data ? (
            <div className="text-[11px] text-muted-foreground">
              共 {profile.data.field_count} 个字段
              {profile.data.numeric_fields.length > 0
                ? ` · 数值列 ${profile.data.numeric_fields.length} 个（范围按 catalog 样本计算）`
                : ''}
            </div>
          ) : null}
        </div>
        {profile.isLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, index) => (
              <Skeleton key={index} className="h-8 w-full" />
            ))}
          </div>
        ) : profile.isError ? (
          <p className="text-xs text-destructive">字段画像读取失败：{(profile.error as Error).message}</p>
        ) : profile.data ? (
          <FieldTable fields={profile.data.fields} />
        ) : null}
      </div>
    </div>
  )
}

function LayerCard({ item, onOpen }: { item: DatasetItem; onOpen: (item: DatasetItem) => void }) {
  return (
    <button
      type="button"
      onClick={() => onOpen(item)}
      className="group flex flex-col rounded-lg border border-border bg-card p-4 text-left transition-colors hover:border-primary/40 hover:bg-accent/40"
    >
      <div className="flex items-start gap-3">
        <IconBadge tone={item.kind === 'raster' ? 'chart3' : 'chart1'}>
          <DatasetIcon item={item} />
        </IconBadge>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-medium text-foreground" title={item.path}>
            {item.name}
          </div>
          <div className="mt-0.5 truncate text-[11px] text-muted-foreground">
            {KIND_LABEL[item.kind] ?? item.kind}
            {item.geom_type ? ` · ${item.geom_type}` : ''}
          </div>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 text-[11px]">
        <div>
          <div className="text-muted-foreground">要素数</div>
          <div className="mt-0.5 tabular-nums text-foreground">
            {formatNumber(item.feature_count)}
          </div>
        </div>
        <div className="min-w-0">
          <div className="text-muted-foreground">坐标系</div>
          <div className="mt-0.5 truncate text-foreground" title={crsLabel(item)}>
            {crsLabel(item)}
          </div>
        </div>
      </div>

      <div className="mt-2 flex items-start gap-1.5 text-[11px] text-muted-foreground">
        <Ruler className="mt-0.5 size-3 shrink-0" />
        <span className="line-clamp-2">{formatExtent(item.extent)}</span>
      </div>

      <div className="mt-2 flex items-center gap-1.5 text-[11px] text-muted-foreground">
        <Folder className="size-3 shrink-0" />
        <span className="truncate">{item.source || 'local'}</span>
        <span className="text-muted-foreground/50">·</span>
        <span className="truncate">{item.license || 'unknown'}</span>
      </div>
    </button>
  )
}

export function DataAssetsPage() {
  const [view, setView] = useState<ViewMode>('cards')
  const [keyword, setKeyword] = useState('')
  const [activeGroup, setActiveGroup] = useState<string | null>(null)
  const [openLayer, setOpenLayer] = useState<DatasetItem | null>(null)

  const datasets = useQuery({
    queryKey: ['datasets'],
    queryFn: () => listDatasets({ limit: 200 }),
  })

  const filtered = useMemo(() => {
    const items = datasets.data?.items ?? []
    const text = keyword.trim().toLowerCase()
    if (!text) return items
    return items.filter(
      (item) =>
        item.name.toLowerCase().includes(text) || item.path.toLowerCase().includes(text)
    )
  }, [datasets.data, keyword])

  const groups = useMemo(() => groupDatasets(filtered), [filtered])

  useEffect(() => {
    if (groups.length === 0) {
      setActiveGroup(null)
      return
    }
    if (!activeGroup || !groups.some((group) => group.key === activeGroup)) {
      setActiveGroup(groups[0].key)
    }
  }, [groups, activeGroup])

  const active = groups.find((group) => group.key === activeGroup) ?? groups[0]

  const columns: DataTableColumn<DatasetItem>[] = [
    {
      id: 'name',
      header: '名称',
      cell: (item) => (
        <div className="flex items-center gap-2">
          <DatasetIcon item={item} className="size-3.5 shrink-0 text-muted-foreground" />
          <span className="max-w-[16rem] truncate font-medium text-foreground" title={item.path}>
            {item.name}
          </span>
        </div>
      ),
    },
    {
      id: 'kind',
      header: '类型',
      cell: (item) => (
        <span className="text-xs text-muted-foreground">
          {KIND_LABEL[item.kind] ?? item.kind}
          {item.geom_type ? ` · ${item.geom_type}` : ''}
        </span>
      ),
    },
    {
      id: 'features',
      header: '要素数',
      align: 'right',
      cell: (item) => formatNumber(item.feature_count),
    },
    {
      id: 'crs',
      header: '坐标系',
      cell: (item) => (
        <span className="text-xs text-muted-foreground">{crsLabel(item)}</span>
      ),
    },
    {
      id: 'extent',
      header: '范围',
      cell: (item) => (
        <span className="text-[11px] text-muted-foreground">{formatExtent(item.extent)}</span>
      ),
    },
    {
      id: 'source',
      header: '来源',
      cell: (item) => <span className="text-xs text-muted-foreground">{item.source || '—'}</span>,
    },
    {
      id: 'license',
      header: '许可',
      cell: (item) => (
        <span className="text-xs text-muted-foreground">{item.license || 'unknown'}</span>
      ),
    },
  ]

  return (
    <>
      <PageHeader
        title="数据资产"
        description="catalog 里登记的矢量/栅格资产：坐标系、范围、字段画像与离线预览。"
        actions={
          <>
            <label className="relative">
              <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
              <input
                value={keyword}
                onChange={(event) => setKeyword(event.target.value)}
                placeholder="搜名称或路径"
                className="h-8 w-56 rounded-full border border-border bg-card pr-3 pl-8 text-xs text-foreground outline-none placeholder:text-muted-foreground focus:border-primary/50"
              />
            </label>
            <Button
              variant="outline"
              size="sm"
              onClick={() => void datasets.refetch()}
              disabled={datasets.isFetching}
            >
              <RefreshCw className={cn('size-3.5', datasets.isFetching && 'animate-spin')} />
              刷新
            </Button>
          </>
        }
      />

      {datasets.isLoading ? (
        <div className="grid gap-4 lg:grid-cols-[17rem_1fr]">
          <Skeleton className="h-80 w-full" />
          <Skeleton className="h-80 w-full" />
        </div>
      ) : datasets.isError ? (
        <Empty
          icon={Database}
          title="读取 catalog 失败"
          description={(datasets.error as Error).message}
        />
      ) : (datasets.data?.items.length ?? 0) === 0 ? (
        <Empty
          icon={Database}
          title="还没有登记数据资产"
          description="catalog 由引擎在任务开始时扫描 workspace/input 生成；先跑一次任务，或把数据放进 input/ 后刷新。"
        />
      ) : groups.length === 0 ? (
        <Empty
          icon={Search}
          title="没有匹配的数据集"
          description="换个关键词，或清空搜索框。"
        />
      ) : (
        <div className="grid gap-4 lg:grid-cols-[17rem_1fr]">
          <aside className="h-fit rounded-lg border border-border bg-card p-2 lg:sticky lg:top-[calc(var(--app-header-height)+1rem)]">
            <div className="flex items-center justify-between px-2 pt-1 pb-2">
              <span className="text-[11px] font-medium text-muted-foreground">容器 / 图层</span>
              <span className="text-[11px] tabular-nums text-muted-foreground/70">
                {filtered.length}
              </span>
            </div>
            <ContainerTree
              groups={groups}
              activeKey={active?.key ?? null}
              onSelect={setActiveGroup}
            />
            {(datasets.data?.has_more ?? false) ? (
              <p className="px-2 pt-2 text-[10px] text-muted-foreground">
                只显示前 {datasets.data?.count} 条，请用搜索缩小范围。
              </p>
            ) : null}
          </aside>

          <section className="min-w-0">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <div className="min-w-0">
                <div className="truncate text-sm font-medium text-foreground" title={active?.key}>
                  {active?.label}
                </div>
                <div className="text-[11px] text-muted-foreground">
                  {active?.items.length ?? 0} 个图层
                  {active?.isGdb ? ' · ArcGIS 文件数据库' : ''}
                </div>
              </div>
              <div className="flex items-center gap-0.5 rounded-full border border-border bg-card p-0.5">
                <button
                  type="button"
                  onClick={() => setView('cards')}
                  aria-pressed={view === 'cards'}
                  aria-label="卡片视图"
                  className={cn(
                    'inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors',
                    view === 'cards' ? 'bg-accent text-accent-foreground' : 'hover:text-foreground'
                  )}
                >
                  <Grid2x2 className="size-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => setView('table')}
                  aria-pressed={view === 'table'}
                  aria-label="表格视图"
                  className={cn(
                    'inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors',
                    view === 'table' ? 'bg-accent text-accent-foreground' : 'hover:text-foreground'
                  )}
                >
                  <List className="size-3.5" />
                </button>
              </div>
            </div>

            {view === 'cards' ? (
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {(active?.items ?? []).map((item) => (
                  <LayerCard key={item.path} item={item} onOpen={setOpenLayer} />
                ))}
              </div>
            ) : (
              <DataTable
                columns={columns}
                rows={active?.items ?? []}
                rowKey={(item) => item.path}
                onRowClick={setOpenLayer}
                initialPageSize={20}
              />
            )}
          </section>
        </div>
      )}

      <Sheet open={openLayer !== null} onOpenChange={(open) => (open ? null : setOpenLayer(null))}>
        <SheetContent side="right" className="w-[min(860px,94vw)] overflow-y-auto">
          <SheetTitle className="pr-8">{openLayer?.name ?? '图层详情'}</SheetTitle>
          {openLayer ? <LayerDetail layer={openLayer} /> : null}
        </SheetContent>
      </Sheet>
    </>
  )
}

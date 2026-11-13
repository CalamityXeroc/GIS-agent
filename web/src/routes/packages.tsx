import { useQuery } from '@tanstack/react-query'
import {
  Download,
  ExternalLink,
  FileBadge,
  FileText,
  Package,
  RefreshCw,
  ShieldCheck,
} from 'lucide-react'
import { useState } from 'react'

import { PageHeader } from '@/components/layout/page-header'
import { Button } from '@/components/ui/button'
import { Empty } from '@/components/ui/empty'
import { IconBadge } from '@/components/ui/icon-badge'
import { Sheet, SheetContent, SheetTitle } from '@/components/ui/sheet'
import { Skeleton } from '@/components/ui/skeleton'
import { listReports, reportUrl, type ReportItem } from '@/lib/api'
import { cn } from '@/lib/utils'

function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / 1024 / 1024).toFixed(1)} MB`
}

function formatTime(mtime: number): string {
  const date = new Date(mtime * 1000)
  return Number.isNaN(date.getTime())
    ? '—'
    : date.toLocaleString('zh-CN', { hour12: false })
}

const KIND_META = {
  provenance: {
    label: '溯源卡',
    icon: ShieldCheck,
    tone: 'chart5' as const,
    description: '数据 / 步骤 / 参数 / 验证依据',
  },
  package: {
    label: '成果包',
    icon: FileBadge,
    tone: 'chart1' as const,
    description: '结论、地图、验证与产物清单',
  },
}

function ReportCard({ item, onPreview }: { item: ReportItem; onPreview: (item: ReportItem) => void }) {
  const meta = KIND_META[item.kind]
  const folder = item.dir === '.' ? 'output/' : `output/${item.dir}/`
  return (
    <div className="flex flex-col rounded-lg border border-border bg-card p-4">
      <div className="flex items-start gap-3">
        <IconBadge tone={meta.tone}>
          <meta.icon />
        </IconBadge>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-medium text-foreground">
              {meta.label}
            </span>
            <span className="shrink-0 rounded-full border border-border px-2 py-0.5 text-[10px] text-muted-foreground">
              HTML
            </span>
          </div>
          <div className="mt-0.5 truncate text-[11px] text-muted-foreground" title={`${folder}${item.name}`}>
            {folder}
            <span className="text-foreground/70">{item.name.split('/').pop()}</span>
          </div>
        </div>
      </div>

      <p className="mt-2 text-[11px] text-muted-foreground">{meta.description}</p>

      <div className="mt-2 flex items-center gap-2 text-[11px] text-muted-foreground">
        <span className="tabular-nums">{formatBytes(item.size)}</span>
        <span className="text-muted-foreground/50">·</span>
        <span className="tabular-nums">{formatTime(item.mtime)}</span>
      </div>

      <div className="mt-3 flex items-center gap-2">
        <Button size="sm" onClick={() => onPreview(item)}>
          <FileText className="size-3.5" />
          预览
        </Button>
        <Button size="sm" variant="outline" asChild>
          <a href={reportUrl(item.name, true)}>
            <Download className="size-3.5" />
            下载
          </a>
        </Button>
        <Button size="sm" variant="ghost" asChild>
          <a href={reportUrl(item.name)} target="_blank" rel="noreferrer">
            <ExternalLink className="size-3.5" />
            新窗口
          </a>
        </Button>
      </div>
    </div>
  )
}

export function PackagesPage() {
  const [preview, setPreview] = useState<ReportItem | null>(null)
  const reports = useQuery({ queryKey: ['reports'], queryFn: listReports })

  return (
    <>
      <PageHeader
        title="成果包"
        description="output/ 下每次运行产出的成果包与溯源卡：沙箱内嵌预览，可下载原件。"
        actions={
          <Button
            variant="outline"
            size="sm"
            onClick={() => void reports.refetch()}
            disabled={reports.isFetching}
          >
            <RefreshCw className={cn('size-3.5', reports.isFetching && 'animate-spin')} />
            刷新
          </Button>
        }
      />

      {reports.isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, index) => (
            <Skeleton key={index} className="h-40 w-full rounded-lg" />
          ))}
        </div>
      ) : reports.isError ? (
        <Empty
          icon={Package}
          title="读取成果包失败"
          description={(reports.error as Error).message}
        />
      ) : (reports.data?.items.length ?? 0) === 0 ? (
        <Empty
          icon={Package}
          title="output/ 下还没有成果包"
          description="跑一次带交付物的任务后，引擎会写 output/<任务>/成果包.html 与 溯源卡.html，这里会自动出现。"
        />
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {reports.data?.items.map((item) => (
            <ReportCard key={item.name} item={item} onPreview={setPreview} />
          ))}
        </div>
      )}

      <Sheet
        open={preview !== null}
        onOpenChange={(open) => (open ? null : setPreview(null))}
      >
        <SheetContent side="right" className="w-[min(1080px,96vw)] overflow-y-auto p-4">
          <SheetTitle className="pr-8">
            {preview ? `${KIND_META[preview.kind].label} · ${preview.dir}` : '成果包预览'}
          </SheetTitle>
          {preview ? (
            <div className="flex min-h-0 flex-col gap-2">
              <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                <span className="truncate" title={preview.name}>
                  {preview.name}
                </span>
                <span className="text-muted-foreground/50">·</span>
                <span className="tabular-nums">{formatBytes(preview.size)}</span>
                <span className="text-muted-foreground/50">·</span>
                <span className="tabular-nums">{formatTime(preview.mtime)}</span>
                <a
                  className="ml-auto inline-flex items-center gap-1 text-primary hover:underline"
                  href={reportUrl(preview.name, true)}
                >
                  <Download className="size-3" />
                  下载原件
                </a>
              </div>
              {/* sandbox：可以跑报告里的内联 canvas 脚本，但拿不到同源权限。 */}
              <iframe
                key={preview.name}
                title={`成果包预览：${preview.name}`}
                src={reportUrl(preview.name)}
                sandbox="allow-scripts"
                className="h-[calc(100dvh-12rem)] w-full rounded-lg border border-border bg-card"
              />
            </div>
          ) : null}
        </SheetContent>
      </Sheet>
    </>
  )
}

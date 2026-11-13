import { useQuery } from '@tanstack/react-query'
import { Download, FileJson, FileText, History, RotateCcw, Search } from 'lucide-react'
import { useMemo, useState } from 'react'

import { PageHeader } from '@/components/layout/page-header'
import { Button } from '@/components/ui/button'
import { DataTable, type DataTableColumn } from '@/components/ui/data-table'
import { Empty } from '@/components/ui/empty'
import { Select } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import {
  getTraceDetail,
  listTraces,
  traceDownloadUrl,
  type TraceRow,
  type VerdictFilter,
} from '@/lib/api'
import { buildTimeline, formatClock, formatDuration, formatTokens, STATUS_LABEL } from '@/lib/run-events'
import { cn } from '@/lib/utils'

const STATUS_OPTIONS = [
  { value: '', label: '全部状态' },
  { value: 'completed', label: '已完成' },
  { value: 'failed', label: '失败' },
  { value: 'stopped', label: '已停止' },
  { value: 'incomplete', label: '未完成' },
  { value: 'awaiting_user', label: '等待补充' },
]

const VERDICT_OPTIONS = [
  { value: '', label: '全部' },
  { value: 'pass', label: '通过' },
  { value: 'fail', label: '未通过' },
  { value: 'none', label: '未验收' },
]

const STATUS_TONE: Record<string, string> = {
  completed: 'text-success',
  failed: 'text-destructive',
  stopped: 'text-warning',
  interrupted: 'text-warning',
  incomplete: 'text-muted-foreground',
  awaiting_user: 'text-primary',
  llm_unavailable: 'text-destructive',
  max_turns: 'text-warning',
  error: 'text-destructive',
}

function StatusPill({ status }: { status: string }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border border-border bg-card px-2.5 py-1 text-xs font-medium whitespace-nowrap',
        STATUS_TONE[status] ?? 'text-muted-foreground'
      )}
    >
      <span className="size-1.5 rounded-full bg-current" />
      {STATUS_LABEL[status] ?? status}
    </span>
  )
}

function VerdictCell({ row }: { row: TraceRow }) {
  if (row.passed === true) {
    return <span className="text-xs font-medium text-success">通过</span>
  }
  if (row.passed === false) {
    const gap = row.problems[0]
    return (
      <div className="min-w-0">
        <div className="text-xs font-medium text-destructive">未通过</div>
        <div className="max-w-[16rem] truncate text-[11px] text-muted-foreground">
          {gap || '无缺口明细'}
        </div>
      </div>
    )
  }
  return <span className="text-xs text-muted-foreground">未验收</span>
}

interface Filters {
  since: string
  until: string
  status: string
  pass: VerdictFilter
  q: string
}

const EMPTY_FILTERS: Filters = { since: '', until: '', status: '', pass: '', q: '' }

/** 行展开：时间线摘要 + 缺口全文 + 原始文件下载。 */
function RunRowDetail({ run }: { run: TraceRow }) {
  const detail = useQuery({
    queryKey: ['trace', run.run_id],
    queryFn: () => getTraceDetail(run.run_id),
    staleTime: 60_000,
  })

  const summary = useMemo(() => {
    if (!detail.data) return []
    const items = buildTimeline(detail.data.events)
    const interesting = items.filter((item) =>
      ['step', 'result', 'verify', 'error', 'warn', 'final', 'plan', 'banner'].includes(item.kind)
    )
    return interesting.slice(-6)
  }, [detail.data])

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <div className="lg:col-span-2">
        <div className="mb-2 text-xs text-muted-foreground">时间线摘要（最近 6 条）</div>
        {detail.isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-4 w-1/2" />
          </div>
        ) : summary.length === 0 ? (
          <p className="text-xs text-muted-foreground">这份 trace 没有可摘要的事件。</p>
        ) : (
          <ol className="space-y-1.5">
            {summary.map((item) => (
              <li key={item.key} className="flex items-baseline gap-2 text-xs">
                <span className="shrink-0 tabular-nums text-muted-foreground/70">
                  {formatClock(item.ts)}
                </span>
                <span className="min-w-0 flex-1 truncate text-foreground">{item.title}</span>
                {item.meta ? (
                  <span className="shrink-0 text-[11px] text-muted-foreground">{item.meta}</span>
                ) : null}
              </li>
            ))}
          </ol>
        )}
      </div>

      <div className="space-y-3">
        <div>
          <div className="mb-1.5 text-xs text-muted-foreground">原始文件</div>
          <div className="flex flex-wrap gap-2">
            <a
              className="inline-flex h-7 items-center gap-1.5 rounded-full border border-border px-2.5 text-xs text-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
              href={traceDownloadUrl(run.run_id, 'jsonl')}
              download
            >
              <FileJson className="size-3.5" />
              .jsonl
            </a>
            <a
              className="inline-flex h-7 items-center gap-1.5 rounded-full border border-border px-2.5 text-xs text-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
              href={traceDownloadUrl(run.run_id, 'md')}
              download
            >
              <FileText className="size-3.5" />
              .md 报告
            </a>
          </div>
        </div>

        {run.problems.length > 0 ? (
          <div>
            <div className="mb-1.5 text-xs text-muted-foreground">验收缺口</div>
            <ul className="space-y-1 text-[11px] text-foreground">
              {run.problems.map((problem, index) => (
                <li key={index} className="line-clamp-2">
                  · {problem}
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
          <span>模型</span>
          {Object.entries(run.models).length === 0 ? (
            <span>—</span>
          ) : (
            Object.entries(run.models).map(([model, tokens]) => (
              <span key={model} className="rounded-full border border-border px-2 py-0.5">
                {model} · {formatTokens(tokens)}
              </span>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

export function RunsPage() {
  const [draft, setDraft] = useState<Filters>(EMPTY_FILTERS)
  const [applied, setApplied] = useState<Filters>(EMPTY_FILTERS)

  const tracesQuery = useQuery({
    queryKey: ['traces', applied],
    queryFn: () => listTraces({ ...applied, limit: 200 }),
    refetchInterval: 20000,
  })

  const update = (patch: Partial<Filters>) => setDraft((current) => ({ ...current, ...patch }))

  const chips = [
    applied.since ? `起 ${applied.since}` : '',
    applied.until ? `止 ${applied.until}` : '',
    applied.status ? STATUS_LABEL[applied.status] ?? applied.status : '',
    applied.pass === 'pass' ? '验收通过' : '',
    applied.pass === 'fail' ? '验收未通过' : '',
    applied.pass === 'none' ? '未验收' : '',
    applied.q ? `关键词 ${applied.q}` : '',
  ].filter(Boolean)

  const columns: DataTableColumn<TraceRow>[] = [
    {
      id: 'time',
      header: '时间',
      cell: (row) => (
        <div className="min-w-[6.5rem]">
          <div className="text-xs tabular-nums text-foreground">{row.started_at.slice(0, 10)}</div>
          <div className="text-[11px] tabular-nums text-muted-foreground">
            {formatClock(row.started_at)}
          </div>
        </div>
      ),
    },
    {
      id: 'status',
      header: '状态',
      cell: (row) => <StatusPill status={row.status} />,
    },
    {
      id: 'goal',
      header: '目标摘要',
      cell: (row) => (
        <div className="min-w-0 max-w-[22rem]">
          <div className="truncate text-xs text-foreground">{row.goal || '（无目标记录）'}</div>
          <div className="truncate text-[11px] text-muted-foreground">{row.run_id}</div>
        </div>
      ),
    },
    {
      id: 'turns',
      header: '轮次',
      align: 'right',
      cell: (row) => <span className="text-xs">{row.turns}</span>,
    },
    {
      id: 'tokens',
      header: 'token',
      align: 'right',
      cell: (row) => (
        <div>
          <div className="text-xs">{formatTokens(row.tokens)}</div>
          <div className="text-[11px] text-muted-foreground">
            缓 {formatTokens(row.cached_tokens)}
          </div>
        </div>
      ),
    },
    {
      id: 'duration',
      header: '耗时',
      align: 'right',
      cell: (row) => <span className="text-xs">{formatDuration(row.duration_s)}</span>,
    },
    {
      id: 'verdict',
      header: '验收缺口',
      cell: (row) => <VerdictCell row={row} />,
    },
  ]

  return (
    <>
      <PageHeader
        title="运行历史"
        description="所有 trace 的筛选与回溯；每行可展开时间线摘要，并下载原始 .jsonl / .md。"
      />

      {/* 筛选卡：第一行控件，第二行 chips + 重置/搜索 */}
      <section className="mb-4 overflow-hidden rounded-lg border border-border bg-card">
        <div className="grid gap-2 border-b border-border px-3 py-3 sm:grid-cols-2 lg:grid-cols-5">
          <label className="flex flex-col gap-1 text-[11px] text-muted-foreground">
            起始日期
            <input
              type="date"
              className="h-8 rounded-md border border-border bg-transparent px-2 text-xs text-foreground outline-none"
              value={draft.since}
              onChange={(event) => update({ since: event.target.value })}
            />
          </label>
          <label className="flex flex-col gap-1 text-[11px] text-muted-foreground">
            结束日期
            <input
              type="date"
              className="h-8 rounded-md border border-border bg-transparent px-2 text-xs text-foreground outline-none"
              value={draft.until}
              onChange={(event) => update({ until: event.target.value })}
            />
          </label>
          <div className="flex flex-col gap-1 text-[11px] text-muted-foreground">
            <span>状态</span>
            <Select
              aria-label="按状态筛选"
              className="h-8 w-full rounded-md border border-border bg-transparent px-2 text-xs text-foreground"
              value={draft.status}
              onValueChange={(next) => update({ status: next })}
              options={STATUS_OPTIONS}
            />
          </div>
          <div className="flex flex-col gap-1 text-[11px] text-muted-foreground">
            <span>验收结果</span>
            <Select
              aria-label="按验收结果筛选"
              className="h-8 w-full rounded-md border border-border bg-transparent px-2 text-xs text-foreground"
              value={draft.pass}
              onValueChange={(next) => update({ pass: next as VerdictFilter })}
              options={VERDICT_OPTIONS}
            />
          </div>
          <label className="flex flex-col gap-1 text-[11px] text-muted-foreground">
            关键词
            <input
              type="search"
              placeholder="目标摘要 / run_id"
              className="h-8 rounded-md border border-border bg-transparent px-2 text-xs text-foreground outline-none placeholder:text-muted-foreground/60"
              value={draft.q}
              onChange={(event) => update({ q: event.target.value })}
              onKeyDown={(event) => {
                if (event.key === 'Enter') setApplied({ ...draft })
              }}
            />
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-2 px-3 py-2">
          {chips.length === 0 ? (
            <span className="text-[11px] text-muted-foreground">未筛选 · 显示全部运行</span>
          ) : (
            chips.map((chip) => (
              <span
                key={chip}
                className="rounded-full border border-border px-2 py-0.5 text-[11px] text-foreground"
              >
                {chip}
              </span>
            ))
          )}
          <div className="ml-auto flex items-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setDraft(EMPTY_FILTERS)
                setApplied(EMPTY_FILTERS)
              }}
            >
              <RotateCcw className="size-3.5" />
              重置
            </Button>
            <Button size="sm" onClick={() => setApplied({ ...draft })}>
              <Search className="size-3.5" />
              搜索
            </Button>
          </div>
        </div>
      </section>

      {tracesQuery.isError ? (
        <Empty
          icon={History}
          title="运行列表加载失败"
          description={(tracesQuery.error as Error)?.message ?? '请确认服务是否在运行。'}
        />
      ) : (
        <DataTable
          columns={columns}
          rows={tracesQuery.data?.items ?? []}
          rowKey={(row) => row.run_id}
          loading={tracesQuery.isLoading}
          total={tracesQuery.data?.total}
          initialPageSize={20}
          renderExpanded={(row) => <RunRowDetail run={row} />}
          toolbar={
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Download className="size-3.5" />
              <span>共 {tracesQuery.data?.total ?? 0} 条 trace（展开行可下载原始文件）</span>
            </div>
          }
          empty={
            <Empty
              icon={History}
              title="没有符合条件的运行"
              description="放宽时间范围或清空筛选条件再看看。"
              className="min-h-[280px] border-0 bg-transparent"
            />
          }
        />
      )}
    </>
  )
}

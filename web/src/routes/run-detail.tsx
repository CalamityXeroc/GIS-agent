import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams } from '@tanstack/react-router'
import { Activity, CircleStop, RotateCcw, Send } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'

import { PageHeader } from '@/components/layout/page-header'
import { PlanDialog } from '@/components/runs/plan-dialog'
import { RunStatusStrip } from '@/components/runs/run-status-strip'
import { RunTimeline } from '@/components/runs/run-timeline'
import { Button } from '@/components/ui/button'
import { Empty } from '@/components/ui/empty'
import { Skeleton } from '@/components/ui/skeleton'
import { TitledCard } from '@/components/ui/titled-card'
import { useRunStream } from '@/hooks/use-run-stream'
import { ApiError, getRun, injectRun, isTerminal, resumeRun, stopRun } from '@/lib/api'
import { buildTimeline, formatDuration, STATUS_LABEL } from '@/lib/run-events'
import { cn } from '@/lib/utils'

const STATUS_TONE: Record<string, string> = {
  queued: 'text-muted-foreground',
  running: 'text-primary',
  awaiting_plan: 'text-warning',
  stopping: 'text-warning',
  succeeded: 'text-success',
  failed: 'text-destructive',
  stopped: 'text-warning',
  interrupted: 'text-warning',
}

function StatusPill({ status }: { status: string }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border border-border bg-card px-2.5 py-1 text-xs font-medium',
        STATUS_TONE[status] ?? 'text-muted-foreground'
      )}
    >
      <span
        className={cn(
          'size-1.5 rounded-full bg-current',
          status === 'running' || status === 'stopping' ? 'animate-pulse' : ''
        )}
      />
      {STATUS_LABEL[status] ?? status}
    </span>
  )
}

function InfoRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5">
      <dt className="shrink-0 text-xs text-muted-foreground">{label}</dt>
      <dd className="min-w-0 truncate text-right text-xs text-foreground">{value || '—'}</dd>
    </div>
  )
}

export function RunDetailPage() {
  const { runId } = useParams({ from: '/runs/$runId' })
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { events, connected, truncated, finished } = useRunStream(runId)
  const [now, setNow] = useState(() => Date.now())
  const [injectText, setInjectText] = useState('')
  const [pending, setPending] = useState('')
  const [planOpen, setPlanOpen] = useState(false)
  const lastAutoPlan = useRef('')

  const runQuery = useQuery({
    queryKey: ['run', runId],
    queryFn: () => getRun(runId),
    refetchInterval: (query) => (isTerminal(query.state.data?.status) ? false : 1500),
  })
  const run = runQuery.data

  const refresh = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: ['run', runId] })
  }, [queryClient, runId])

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [])

  const items = useMemo(() => buildTimeline(events), [events])

  const lastEvent = events[events.length - 1]?.event
  useEffect(() => {
    if (lastEvent === 'run_end') refresh()
  }, [lastEvent, refresh])

  const planSignature = run?.plan ? `${run.plan.round ?? 1}:${run.plan.deadline ?? ''}` : ''
  useEffect(() => {
    if (run?.status !== 'awaiting_plan' || !planSignature) return
    if (lastAutoPlan.current === planSignature) return
    lastAutoPlan.current = planSignature
    setPlanOpen(true)
  }, [run?.status, planSignature])

  const elapsed = useMemo(() => {
    if (!run?.started_at) return 0
    const start = Date.parse(run.started_at)
    if (Number.isNaN(start)) return 0
    if (run.ended_at) return Math.max(0, (Date.parse(run.ended_at) - start) / 1000)
    if (isTerminal(run.status)) return run.duration_s ?? 0
    return Math.max(0, (now - start) / 1000)
  }, [run, now])

  const terminal = isTerminal(run?.status)
  // 历史 run（服务重启前 / 从 trace 链接直接打开）：数据只有 trace 回放，只读。
  const readonly = Boolean(run?.readonly)
  const canInject = run?.status === 'running' && !readonly
  // 停止中：服务端已进 stopping（收到 stop 立即迁），或 stop 请求已写进 job。
  // 两个都要认：SSE 可能比下一次轮询先到。
  const stopping = run?.status === 'stopping' || Boolean(run?.stop_requested)

  const handleStop = async () => {
    setPending('stop')
    try {
      await stopRun(runId)
      toast.success('已请求停止：当前工具返回后不再发起新工具')
      refresh()
      // 成功时**不清** pending：在轮询拿到 stopping/终态之前按钮必须保持禁用，
      // 否则会闪回「停止」并允许重复提交。
    } catch (error) {
      toast.error(error instanceof ApiError ? error.message : '停止失败')
      setPending('')
    }
  }

  const handleResume = async () => {
    setPending('resume')
    try {
      const result = await resumeRun(runId)
      toast.success('已从断点创建续跑 run')
      await navigate({ to: '/runs/$runId', params: { runId: result.run.id } })
    } catch (error) {
      toast.error(error instanceof ApiError ? error.message : '续跑失败')
    } finally {
      setPending('')
    }
  }

  const handleInject = async () => {
    const text = injectText.trim()
    if (!text) return
    setPending('inject')
    try {
      await injectRun(runId, text)
      setInjectText('')
      toast.success('插话已送达，引擎下一轮会读到')
    } catch (error) {
      toast.error(error instanceof ApiError ? error.message : '插话失败')
    } finally {
      setPending('')
    }
  }

  if (runQuery.isLoading) {
    return (
      <>
        <PageHeader title="运行详情" description="正在读取 run 状态…" />
        <Skeleton className="h-24 w-full" />
        <Skeleton className="mt-4 h-64 w-full" />
      </>
    )
  }

  if (runQuery.isError || !run) {
    return (
      <>
        <PageHeader title={`运行详情 · ${runId}`} description="这个 run 不存在，或服务重启后 jobs.json 里没有它。" />
        <Empty
          icon={Activity}
          title="找不到这个 run"
          description={runQuery.error instanceof ApiError ? runQuery.error.message : '请从运行历史里重新进入。'}
        />
      </>
    )
  }

  return (
    <>
      <PageHeader
        title={<span className="break-all">运行详情 · {run.id}</span>}
        description={<span className="line-clamp-2">{run.goal}</span>}
        actions={
          <>
            <StatusPill status={run.status} />
            {readonly ? null : (
              <>
                {run.plan && run.status === 'awaiting_plan' ? (
                  <Button variant="outline" size="sm" onClick={() => setPlanOpen(true)}>
                    查看计划卡
                  </Button>
                ) : null}
                {terminal ? (
                  run.status === 'succeeded' ? null : (
                    <Button variant="outline" size="sm" onClick={handleResume} disabled={pending !== ''}>
                      <RotateCcw /> {pending === 'resume' ? '创建中…' : '从断点续跑'}
                    </Button>
                  )
                ) : (
                  <Button
                    variant="outline"
                    size="sm"
                    className="border-destructive/40 text-destructive hover:bg-destructive/10 hover:text-destructive"
                    onClick={handleStop}
                    disabled={pending !== '' || stopping}
                  >
                    <CircleStop /> {stopping ? '正在停止…' : '停止'}
                  </Button>
                )}
              </>
            )}
          </>
        }
      />

      {readonly ? (
        <div className="mt-4 rounded-lg border border-border bg-card px-4 py-3 text-xs leading-relaxed text-muted-foreground">
          只读：这个 run 不在当前服务的 jobs.json 里（服务重启前 / 历史 trace），详情按 trace
          回放，不能停止、续跑或插话。
        </div>
      ) : null}

      <RunStatusStrip run={run} elapsed={elapsed} />

      {run.error ? (
        <div className="mt-4 rounded-lg border border-destructive/40 bg-destructive/5 px-4 py-3 text-xs leading-relaxed text-destructive">
          {run.error}
        </div>
      ) : null}

      <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
        <div className="space-y-4">
          <RunTimeline
            items={items}
            connected={connected}
            truncated={truncated}
            loading={runQuery.isLoading}
            finished={finished || terminal}
          />

          <TitledCard title="插话" meta={canInject ? '运行中可插话' : '当前不可插话'}>
            <div className="flex flex-col gap-2 sm:flex-row">
              <input
                value={injectText}
                onChange={(event) => setInjectText(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.nativeEvent.isComposing) void handleInject()
                }}
                disabled={!canInject || pending === 'inject'}
                placeholder={
                  readonly
                    ? '历史 run（只读），不支持插话'
                    : canInject
                      ? '补充一句要求，引擎下一轮读到'
                      : 'run 未在运行'
                }
                className="h-9 min-w-0 flex-1 rounded-full border border-border bg-background px-4 text-xs text-foreground outline-none placeholder:text-muted-foreground disabled:opacity-60"
              />
              <Button
                size="sm"
                onClick={handleInject}
                disabled={!canInject || pending === 'inject' || !injectText.trim()}
              >
                <Send /> {pending === 'inject' ? '发送中…' : '发送'}
              </Button>
            </div>
          </TitledCard>
        </div>

        <TitledCard title="运行信息" meta={run.model || undefined}>
          <dl className="divide-y divide-border">
            <div>
              <InfoRow label="登记时间" value={new Date(run.created_at).toLocaleString('zh-CN', { hour12: false })} />
              <InfoRow label="开始时间" value={run.started_at ? new Date(run.started_at).toLocaleString('zh-CN', { hour12: false }) : '—'} />
              <InfoRow label="结束时间" value={run.ended_at ? new Date(run.ended_at).toLocaleString('zh-CN', { hour12: false }) : '—'} />
              <InfoRow label="总耗时" value={terminal ? formatDuration(run.duration_s) : formatDuration(elapsed)} />
              <InfoRow label="验收" value={run.verdict === 'pass' ? '通过' : run.verdict === 'fail' ? '未通过' : '—'} />
              <InfoRow label="自动修复" value={`${run.repairs} 次`} />
              <InfoRow label="引擎 run" value={<span className="font-mono">{run.engine_run_id ?? '—'}</span>} />
              <InfoRow label="trace" value={<span className="font-mono">{run.trace_path ? run.trace_path.split(/[\\/]/).pop() : '—'}</span>} />
              {run.resume_from ? <InfoRow label="续跑自" value={<span className="font-mono">{run.resume_from}</span>} /> : null}
            </div>
          </dl>
        </TitledCard>
      </div>

      <PlanDialog
        runId={run.id}
        plan={run.plan}
        open={planOpen}
        onOpenChange={setPlanOpen}
        onDecided={refresh}
      />
    </>
  )
}

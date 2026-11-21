import { ListTree, Search } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { Empty } from '@/components/ui/empty'
import { TitledCard } from '@/components/ui/titled-card'
import { cn } from '@/lib/utils'
import {
  formatClock,
  matchesQuery,
  type ItemKind,
  type TimelineItem,
} from '@/lib/run-events'

const KIND_STYLE: Record<ItemKind, { text: string; dot: string }> = {
  banner: { text: 'text-primary', dot: 'bg-primary' },
  turn: { text: 'text-muted-foreground', dot: 'bg-border' },
  thought: { text: 'text-foreground/80', dot: 'bg-chart-3' },
  step: { text: 'text-foreground', dot: 'bg-chart-1' },
  result: { text: 'text-foreground', dot: 'bg-chart-2' },
  note: { text: 'text-muted-foreground', dot: 'bg-muted-foreground/50' },
  warn: { text: 'text-warning', dot: 'bg-warning' },
  error: { text: 'text-destructive', dot: 'bg-destructive' },
  verify: { text: 'text-success', dot: 'bg-success' },
  plan: { text: 'text-foreground', dot: 'bg-chart-4' },
  final: { text: 'text-foreground', dot: 'bg-foreground' },
  delta: { text: 'text-muted-foreground', dot: 'bg-chart-5' },
}

function styleFor(item: TimelineItem) {
  const base = KIND_STYLE[item.kind]
  if (item.ok === false && item.kind !== 'warn' && item.kind !== 'error') {
    return { text: 'text-destructive', dot: 'bg-destructive' }
  }
  if (item.ok === true && item.kind === 'verify') {
    return { text: 'text-success', dot: 'bg-success' }
  }
  return base
}

interface RunTimelineProps {
  items: TimelineItem[]
  connected: boolean
  truncated: boolean
  loading: boolean
  /** run 已经结束：连接按设计关闭，不该显示成"离线" */
  finished: boolean
}

/** 时间线：粘底滚动、可折叠附加行、llm_delta 合并、可搜索。 */
export function RunTimeline({ items, connected, truncated, loading, finished }: RunTimelineProps) {
  const [query, setQuery] = useState('')
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const scroller = useRef<HTMLDivElement | null>(null)
  const stickBottom = useRef(true)

  const filtered = useMemo(
    () => items.filter((item) => matchesQuery(item, query)),
    [items, query]
  )

  useEffect(() => {
    const node = scroller.current
    if (!node || !stickBottom.current) return
    // 等一帧再滚：SSE 回放时事件是成批进来的，元素高度还没算完时 scrollTop 会滚不到底
    const frame = requestAnimationFrame(() => {
      node.scrollTop = node.scrollHeight
    })
    return () => cancelAnimationFrame(frame)
  }, [filtered.length])

  const onScroll = () => {
    const node = scroller.current
    if (!node) return
    stickBottom.current = node.scrollHeight - node.scrollTop - node.clientHeight < 48
  }

  const toggle = (key: string) => {
    setExpanded((previous) => {
      const next = new Set(previous)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  return (
    <TitledCard
      title="时间线"
      icon={ListTree}
      meta={
        <span className="flex items-center gap-2">
          {truncated ? <span className="text-warning">历史已截断</span> : null}
          <span className={cn('inline-flex items-center gap-1', connected ? 'text-success' : 'text-muted-foreground')}>
            <span className={cn('size-1.5 rounded-full', connected ? 'bg-success' : 'bg-muted-foreground')} />
            {finished ? '已结束' : connected ? '实时' : '重连中'}
          </span>
          <span className="tabular-nums">{filtered.length} / {items.length}</span>
        </span>
      }
      bodyClassName="p-0"
    >
      <div className="flex items-center gap-2 border-b border-border px-4 py-2">
        <Search className="size-4 shrink-0 text-muted-foreground" />
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="搜索事件（工具名 / 正文 / 状态）…"
          className="h-8 w-full bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground"
        />
      </div>

      <div
        ref={scroller}
        onScroll={onScroll}
        data-slot="run-timeline"
        className="h-[clamp(320px,56vh,680px)] overflow-y-auto px-4 py-4"
      >
        {loading && items.length === 0 ? (
          <p className="text-xs text-muted-foreground">正在连接事件流…</p>
        ) : filtered.length === 0 ? (
          <Empty
            className="min-h-[240px] border-none bg-transparent"
            title={query ? '没有匹配的事件' : '还没有事件'}
            description={query ? '换个关键词试试。' : 'run 开始后，引擎会把每一步推到这里。'}
          />
        ) : (
          <ol className="space-y-0">
            {filtered.map((item) => {
              const tone = styleFor(item)
              const isDelta = item.kind === 'delta'
              const collapsible = isDelta || item.lines.length > 0
              const open = expanded.has(item.key)
              return (
                <li key={item.key} className="relative border-l border-border pl-4 pb-4 last:pb-1">
                  <span
                    className={cn('absolute -left-[4.5px] top-1.5 size-2 rounded-full', tone.dot)}
                    aria-hidden
                  />
                  <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                    <span className={cn('text-xs font-medium', tone.text)}>{item.title}</span>
                    {isDelta ? (
                      <span className="text-[11px] text-muted-foreground">共 {item.count ?? 1} 段增量</span>
                    ) : null}
                    {item.meta ? (
                      <span className="text-[11px] text-muted-foreground">{item.meta}</span>
                    ) : null}
                    <span className="ml-auto text-[11px] text-muted-foreground tabular-nums">
                      {formatClock(item.ts)}
                    </span>
                  </div>

                  {item.rows && item.rows.length > 0 ? (
                    <dl className="mt-2 overflow-hidden rounded-md border border-border">
                      {item.rows.map((row, index) => (
                        <div
                          key={`${item.key}-${row.label}-${index}`}
                          className="grid grid-cols-[4.5rem_minmax(0,1fr)] gap-2 border-b border-border px-3 py-1.5 last:border-b-0"
                        >
                          <dt className="text-[11px] text-muted-foreground">{row.label}</dt>
                          <dd className="text-[11px] leading-relaxed whitespace-pre-wrap break-words text-foreground">
                            {row.value}
                          </dd>
                        </div>
                      ))}
                    </dl>
                  ) : null}

                  {item.body && !isDelta ? (
                    <pre className="mt-1.5 font-sans text-xs leading-relaxed whitespace-pre-wrap break-words text-muted-foreground">
                      {item.body}
                    </pre>
                  ) : null}

                  {collapsible ? (
                    <div className="mt-1">
                      <button
                        type="button"
                        onClick={() => toggle(item.key)}
                        className="text-[11px] text-primary hover:underline"
                      >
                        {open ? '收起' : isDelta ? '展开增量正文' : `展开 ${item.lines.length} 行`}
                      </button>
                      {open ? (
                        <pre className="mt-1.5 max-h-64 overflow-auto rounded-md border border-border bg-background px-3 py-2 text-[11px] leading-relaxed whitespace-pre-wrap break-words text-muted-foreground">
                          {(isDelta ? [item.body] : item.lines).join('\n')}
                        </pre>
                      ) : null}
                    </div>
                  ) : null}
                </li>
              )
            })}
          </ol>
        )}
      </div>
    </TitledCard>
  )
}

/** 引擎事件 → 时间线条目（纯函数，页面只负责画）。 */

export interface RawRunEvent {
  seq: number
  event: string
  ts: string
  payload?: Record<string, unknown>
  duration_ms?: number
}

export type ItemKind =
  | 'banner'
  | 'turn'
  | 'thought'
  | 'step'
  | 'result'
  | 'note'
  | 'warn'
  | 'error'
  | 'verify'
  | 'plan'
  | 'final'
  | 'delta'

export interface TimelineItem {
  key: string
  seq: number
  ts: string
  kind: ItemKind
  title: string
  body: string
  lines: string[]
  ok?: boolean
  meta?: string
  /** 合并后的 llm_delta 段数 */
  count?: number
  /** 计划卡的 5 行结构化内容（优先于终端框线原文） */
  rows?: Array<{ label: string; value: string }>
}

/** 服务端用 `event:` 命名帧，EventSource 必须按名字注册监听。 */
export const EVENT_NAMES = [
  'run_start',
  'requirements_extracted',
  'turn',
  'llm_result',
  'llm_delta',
  'tool_call',
  'tool_result',
  'repair_attempt',
  'repair_error',
  'verification',
  'deliverables',
  'llm_error',
  'deadline_exceeded',
  'run_end',
  'run_error',
  'resumed',
  'plan_card',
  'plan_pending',
  'plan_decided',
  'plan_card_auto_start',
  'plan_cancelled',
  'plan_gate_timeout',
  'stop_requested',
  'user_injection',
  'catalog_scan',
  'note',
] as const

export const STATUS_LABEL: Record<string, string> = {
  queued: '排队中',
  running: '运行中',
  awaiting_plan: '等待确认',
  stopping: '正在停止',
  succeeded: '已成功',
  failed: '失败',
  stopped: '已停止',
  interrupted: '已中断',
  completed: '已完成',
  cancelled: '已取消',
  pending: '待开始',
  plan_pending: '等待确认',
  awaiting_user: '等待补充',
  deadline_exceeded: '超时',
  max_turns: '轮次用尽',
  llm_unavailable: '模型不可用',
  canceled: '已取消',
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

function asText(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function asNumber(value: unknown, fallback = 0): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function asLines(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.map((item) => asText(item)).filter((item) => item.length > 0)
}

function actionThought(payload: Record<string, unknown>): string {
  const actions = Array.isArray(payload.actions) ? payload.actions : []
  for (const action of actions) {
    const record = asRecord(action)
    const thought = asText(record.reasoning) || asText(record.content)
    if (thought) return thought
  }
  return ''
}

function toItem(event: RawRunEvent): TimelineItem {
  const payload = asRecord(event.payload)
  const base = { key: `e${event.seq}`, seq: event.seq, ts: event.ts, body: '', lines: [] as string[] }

  switch (event.event) {
    case 'run_start':
      return {
        ...base,
        kind: 'banner',
        title: '开始',
        body: asText(payload.goal),
        meta: asText(payload.run_id),
      }
    case 'requirements_extracted': {
      const count = asNumber(payload.count)
      return {
        ...base,
        kind: 'note',
        title: `抽取验收需求 ${count} 条`,
        lines: asLines(payload.requirements ?? payload.items),
      }
    }
    case 'turn':
      return {
        ...base,
        kind: 'turn',
        title: `第 ${asNumber(payload.turn, 1)} 轮`,
        lines: asText(payload.tasklist) ? [asText(payload.tasklist)] : [],
      }
    case 'llm_result': {
      const usage = asRecord(payload.usage)
      const actions = Array.isArray(payload.actions) ? payload.actions : []
      const names = actions.map((action) => asText(asRecord(action).tool) || '?')
      const meta = [asText(payload.model), usage.total_tokens ? `${asNumber(usage.total_tokens)} tokens` : '']
        .filter(Boolean)
        .join(' · ')
      const thought = actionThought(payload)
      return {
        ...base,
        kind: 'thought',
        title: thought ? '模型思考' : '模型返回工具调用',
        body: thought,
        lines: names.length ? [`下一步：${names.join('、')}`] : [],
        meta,
      }
    }
    case 'tool_call':
      return {
        ...base,
        kind: 'step',
        title: asText(payload.tool) || '工具调用',
        lines: Object.keys(asRecord(payload.args)).length
          ? [asText(payload.args)]
          : [],
      }
    case 'tool_result': {
      const ok = payload.ok !== false
      const summary = asText(payload.summary)
      return {
        ...base,
        kind: 'result',
        title: `${asText(payload.tool) || '工具'} 返回`,
        body: summary.length > 160 ? '' : summary,
        lines: summary.length > 160 ? [summary] : [],
        ok,
      }
    }
    case 'repair_attempt':
      return {
        ...base,
        kind: 'warn',
        title: `自动修复第 ${asNumber(payload.attempt, 1)} 次`,
        body: asText(payload.error) || asText(payload.reason),
        ok: false,
      }
    case 'repair_error':
      return { ...base, kind: 'error', title: '修复失败', body: asText(payload.error), ok: false }
    case 'verification': {
      const pass = payload.pass === true
      return {
        ...base,
        kind: 'verify',
        title: pass ? '验收通过' : '验收未通过',
        lines: asLines(payload.problems),
        ok: pass,
      }
    }
    case 'deliverables':
      return {
        ...base,
        kind: 'note',
        title: '产物已落盘',
        lines: asLines(payload.artifacts ?? payload.files ?? payload.deliverables),
      }
    case 'llm_error':
      return { ...base, kind: 'error', title: '模型调用失败', body: asText(payload.error), ok: false }
    case 'deadline_exceeded':
      return {
        ...base,
        kind: 'error',
        title: '超过运行时限',
        body: asText(payload.error),
        ok: false,
      }
    case 'run_error':
      return { ...base, kind: 'error', title: '运行出错', body: asText(payload.error), ok: false }
    case 'run_end': {
      const status = asText(payload.status) || 'ended'
      // 实时事件来自 TaskState.to_dict()（turn 单数、无 token 总数），
      // 磁盘 trace 来自 RunRecorder.finish()（turns / total_tokens）——两种都要认。
      const usage = asRecord(payload.usage)
      const turns = asNumber(payload.turns ?? payload.turn)
      const tokens = asNumber(payload.total_tokens ?? usage.total_tokens)
      return {
        ...base,
        kind: 'final',
        title: `运行结束 · ${STATUS_LABEL[status] ?? status}`,
        body: asText(payload.error),
        ok: status === 'completed' || status === 'succeeded',
        meta: `${turns} 轮 · ${formatTokens(tokens)} tokens`,
      }
    }
    case 'resumed':
      return {
        ...base,
        kind: 'note',
        title: '从断点续跑',
        body: `第 ${asNumber(payload.turn)} 轮 / ${asNumber(payload.messages)} 条历史消息`,
      }
    case 'plan_card': {
      const rows = Array.isArray(payload.rows)
        ? payload.rows
            .map((row) => ({ label: asText(asRecord(row).label), value: asText(asRecord(row).value) }))
            .filter((row) => row.label || row.value)
        : []
      return {
        ...base,
        kind: 'plan',
        title: `计划卡（第 ${asNumber(payload.round, 1)} 轮）`,
        // 有结构化行就画表格；没有才退回终端框线原文（框线在比例字体里会错位）
        body: rows.length ? '' : asText(payload.text),
        rows,
      }
    }
    case 'plan_pending':
      return { ...base, kind: 'note', title: '等待你确认计划' }
    case 'plan_decided':
      return { ...base, kind: 'note', title: '计划已确认，继续执行' }
    case 'plan_card_auto_start':
      return { ...base, kind: 'note', title: '计划自动开始', body: asText(payload.reason) }
    case 'plan_cancelled':
      return { ...base, kind: 'warn', title: '计划被取消，未开始执行', ok: false }
    case 'plan_gate_timeout':
      return { ...base, kind: 'warn', title: '计划卡超时自动开始', body: asText(payload.reason) }
    case 'stop_requested':
      return { ...base, kind: 'warn', title: '已请求停止', body: asText(payload.reason), ok: false }
    case 'user_injection':
      return { ...base, kind: 'note', title: '你插话', body: asText(payload.text) }
    case 'catalog_scan':
      return { ...base, kind: 'note', title: `数据目录扫描 · ${asText(payload.state) || '进行中'}` }
    case 'note':
      return { ...base, kind: 'note', title: asText(payload.text) || '提示' }
    default:
      return { ...base, kind: 'note', title: event.event, body: asText(payload) }
  }
}

/**
 * 事件流 → 时间线。连续 `llm_delta` 合并成一条"共 N 段增量"，
 * 展开才看正文 —— 否则一屏全是单字。
 */
export function buildTimeline(events: RawRunEvent[]): TimelineItem[] {
  const items: TimelineItem[] = []
  for (const event of events) {
    if (event.event === 'llm_delta') {
      const payload = asRecord(event.payload)
      const chunk = asText(payload.text)
      const last = items[items.length - 1]
      if (last && last.kind === 'delta') {
        last.count = (last.count ?? 1) + Math.max(1, asNumber(payload.merged, 1))
        last.seq = event.seq
        last.ts = event.ts
        last.body += chunk
        last.lines = [last.body]
        continue
      }
      items.push({
        key: `delta-${event.seq}`,
        seq: event.seq,
        ts: event.ts,
        kind: 'delta',
        title: '模型流式输出',
        body: chunk,
        lines: [chunk],
        count: Math.max(1, asNumber(payload.merged, 1)),
      })
      continue
    }
    items.push(toItem(event))
  }
  return items
}

export function matchesQuery(item: TimelineItem, query: string): boolean {
  if (!query) return true
  const needle = query.toLowerCase()
  const haystack = [item.title, item.body, item.meta ?? '', ...item.lines].join('\n').toLowerCase()
  return haystack.includes(needle)
}

export function formatTokens(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`
  return String(value)
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '—'
  const total = Math.max(0, seconds)
  if (total < 60) return `${total.toFixed(1)}s`
  const minutes = Math.floor(total / 60)
  const rest = Math.round(total % 60)
  if (minutes < 60) return `${minutes}m ${String(rest).padStart(2, '0')}s`
  const hours = Math.floor(minutes / 60)
  return `${hours}h ${String(minutes % 60).padStart(2, '0')}m`
}

export function formatClock(ts: string): string {
  const date = new Date(ts)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleTimeString('zh-CN', { hour12: false })
}

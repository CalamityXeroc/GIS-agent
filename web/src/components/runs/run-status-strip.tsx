import { StatCell, StatRow } from '@/components/ui/stat-card'
import type { RunDetail } from '@/lib/api'
import { formatDuration, formatTokens } from '@/lib/run-events'

interface RunStatusStripProps {
  run: RunDetail | undefined
  /** 已耗时（秒，页面每秒刷新；结束后用端到端耗时） */
  elapsed: number
}

/** 状态条：轮次 / token / 当前工具 / 已耗时 / 熔断，一张卡切 5 格。 */
export function RunStatusStrip({ run, elapsed }: RunStatusStripProps) {
  const repairs = run?.repairs ?? 0
  const stopping = run?.status === 'stopping'
  const tool = run?.current_tool || (stopping ? '正在停止…' : run?.status === 'running' ? '思考中' : '—')

  return (
    <StatRow>
      <StatCell
        label="轮次"
        value={run?.turns ?? 0}
        hint={stopping ? '正在停止' : run?.status === 'running' ? '进行中' : undefined}
      />
      <StatCell
        label="Token"
        value={formatTokens(run?.tokens ?? 0)}
        hint={run?.model || undefined}
      />
      <StatCell label="当前工具" value={<span className="text-lg">{tool}</span>} hint={run?.verdict ? `验收 ${run.verdict === 'pass' ? '通过' : '未通过'}` : undefined} />
      <StatCell
        label="已耗时"
        value={formatDuration(elapsed)}
        hint={stopping ? '正在停止' : run?.status === 'running' ? '持续计时' : '已结束'}
      />
      <StatCell
        label="熔断"
        value={`${repairs} 次`}
        hint={repairs > 0 ? '自动修复已触发' : '未触发'}
        hintTone={repairs > 0 ? 'warning' : 'muted'}
      />
    </StatRow>
  )
}

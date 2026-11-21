import { Loader2 } from 'lucide-react'

import type { DatasetField, DatasetProfile } from '@/lib/api'

import { formatNumber } from '../format'
import { describeApiError } from './errors'
import { EmptyState, Hint, Section, Unavailable } from './parts'

export interface FieldsTabProps {
  profile?: DatasetProfile
  loading: boolean
  error: Error | null
  isRaster: boolean
}

/**
 * 「字段」tab：字段名 / 类型 / 空值率 / 样本。
 *
 * 空值率来自 `/profile` 的 `null_ratio`（catalog 抽样统计），不在这里重算。
 * 栅格没有属性字段——直接说没有，不列空表。
 */
export function FieldsTab({ profile, loading, error, isRaster }: FieldsTabProps) {
  if (isRaster) {
    return (
      <EmptyState title="栅格没有属性字段">
        <span>栅格的像元值在「元数据 → 栅格 / 波段明细」里；属性表只对矢量/表格有意义。</span>
      </EmptyState>
    )
  }
  if (loading) {
    return (
      <div className="flex items-center gap-1.5 py-3 text-[10.5px] text-muted-foreground">
        <Loader2 className="size-3 animate-spin" /> 正在读取字段画像…
      </div>
    )
  }
  if (error) {
    const issue = describeApiError(error)
    return <Unavailable title={issue.title} reason={issue.reason} />
  }
  const fields = profile?.fields ?? []
  if (fields.length === 0) {
    return (
      <Unavailable
        title="没有字段信息"
        reason="catalog 里这条记录没有字段清单（可能还没扫描过，或该格式不提供字段）。"
      />
    )
  }

  return (
    <>
      <Section title={`字段（${fields.length}）`}>
        <div className="mb-1 flex items-center gap-1.5 text-[9.5px] font-semibold text-muted-foreground">
          <span className="min-w-0 flex-1">名称</span>
          <span className="w-[52px] shrink-0">类型</span>
          <span className="w-[46px] shrink-0 text-right">空值率</span>
        </div>
        {fields.map((field) => (
          <FieldRow key={field.name} field={field} />
        ))}
      </Section>
      <Hint>
        空值率来自 catalog 的抽样统计（null_ratio），左侧灰条 = 该列空值占比；
        样本只给前 3 条，只作预览，不是类目清单。
      </Hint>
    </>
  )
}

function FieldRow({ field }: { field: DatasetField }) {
  const ratio = field.null_ratio ?? 0
  const sample = (field.samples ?? [])
    .filter((value) => value !== null && value !== undefined && value !== '')
    .slice(0, 3)
    .join(' · ')
  return (
    <div
      className="flex h-[23px] items-center gap-1.5 border-b border-border/55 text-[11px]"
      title={`${field.name}${sample ? `：${sample}` : ''}`}
    >
      <span className="min-w-0 flex-1 truncate font-mono text-[10.5px]">{field.name}</span>
      <span className="w-[52px] shrink-0 truncate text-[10px] text-muted-foreground">
        {field.type}
      </span>
      <span className="flex w-[46px] shrink-0 items-center justify-end gap-1">
        <span className="h-[4px] w-[18px] overflow-hidden rounded-full bg-muted">
          <i
            className="block h-full bg-warning/70"
            style={{ width: `${Math.min(100, Math.max(0, ratio * 100))}%` }}
          />
        </span>
        <span className="w-[22px] text-right text-[9.5px] tabular-nums text-muted-foreground">
          {ratio > 0 ? `${Math.round(ratio * 100)}%` : '0'}
        </span>
      </span>
    </div>
  )
}

/** 字段 tab 顶部的一句摘要（供右 dock 之外复用，比如状态栏）。 */
export function fieldsSummary(profile?: DatasetProfile): string {
  if (!profile) return '—'
  return `${formatNumber(profile.fields.length)} 个字段 · ${formatNumber(profile.numeric_fields.length)} 个数值列`
}

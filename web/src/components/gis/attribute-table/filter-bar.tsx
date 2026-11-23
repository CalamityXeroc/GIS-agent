import { FilterX, Search } from 'lucide-react'
import { useMemo, useState } from 'react'

import { Select } from '@/components/ui/select'
import type { CatalogField } from '@/lib/api'
import { cn } from '@/lib/utils'

import { buildWhere, isNumericField } from './use-rows'

const TEXT_OPS = [
  { value: '=', label: '=' },
  { value: '!=', label: '≠' },
  { value: 'LIKE', label: 'LIKE' },
  { value: '>', label: '>' },
  { value: '<', label: '<' },
]

const NUMBER_OPS = [
  { value: '=', label: '=' },
  { value: '!=', label: '≠' },
  { value: '>', label: '>' },
  { value: '<', label: '<' },
  { value: '>=', label: '≥' },
  { value: '<=', label: '≤' },
  { value: 'LIKE', label: 'LIKE' },
]

export interface FilterBarProps {
  fields: CatalogField[]
  /** 已应用的条件（空 = 不过滤）。 */
  where: string
  disabled?: boolean
  onApply: (where: string) => void
  onClear: () => void
}

/**
 * 筛选条（spec §4 ③「按属性值筛选」）：字段 + 运算符 + 值 → `where`。
 *
 * 条件是**单条**的：`/api/catalog/rows` 只接受 `字段 op 值`（后端用绑定参数，
 * 运算符与字段名都过白名单）。多条条件需要表达式解析器，本期不做也不假装做。
 * 应用后由父组件把 `where` 发到服务端，分页与 total 都按筛选后口径。
 */
export function FilterBar({ fields, where, disabled, onApply, onClear }: FilterBarProps) {
  const filterable = useMemo(
    () => fields.filter((field) => String(field.type ?? '').toUpperCase() !== 'OID'),
    [fields]
  )
  const [fieldName, setFieldName] = useState(() => filterable[0]?.name ?? '')
  const [op, setOp] = useState('=')
  const [value, setValue] = useState('')

  const field = filterable.find((item) => item.name === fieldName) ?? filterable[0]
  const ops = field && isNumericField(field) ? NUMBER_OPS : TEXT_OPS
  const expression = buildWhere(field, op, value)

  const apply = () => {
    if (!expression) return
    onApply(expression)
  }

  const clear = () => {
    setValue('')
    onClear()
  }

  return (
    <div className="flex min-w-0 items-center gap-1.5">
      <span className="flex items-center gap-1 text-[10.5px] text-muted-foreground">
        <Search className="size-3" /> 筛选
      </span>
      <Select
        value={field?.name ?? ''}
        onValueChange={(next) => {
          setFieldName(next)
          setOp('=')
        }}
        options={filterable.map((item) => ({ value: item.name, label: item.alias || item.name }))}
        aria-label="筛选字段"
        disabled={disabled || filterable.length === 0}
        className="h-[22px] max-w-[132px] rounded-[3px] border border-border bg-card px-1.5 text-[11px] text-foreground"
      />
      <Select
        value={op}
        onValueChange={setOp}
        options={ops}
        aria-label="筛选运算符"
        disabled={disabled}
        className="h-[22px] w-[62px] rounded-[3px] border border-border bg-card px-1.5 text-[11px] text-foreground"
      />
      <input
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter') apply()
        }}
        placeholder={field && isNumericField(field) ? '数值' : '值，如 林地'}
        aria-label="筛选值"
        disabled={disabled}
        className="h-[22px] w-[132px] rounded-[3px] border border-border bg-card px-1.5 text-[11px] text-foreground outline-none placeholder:text-muted-foreground focus:border-primary/55 disabled:opacity-50"
      />
      <button
        type="button"
        onClick={apply}
        disabled={disabled || !expression}
        title={expression ? `按 ${expression} 筛选（服务端）` : '填好字段、运算符和值后再应用'}
        className="h-[22px] rounded-[3px] border border-border bg-card px-2 text-[11px] text-foreground/85 hover:bg-accent disabled:cursor-not-allowed disabled:opacity-45"
      >
        应用
      </button>
      {where ? (
        <button
          type="button"
          onClick={clear}
          title={`清除筛选：${where}`}
          className={cn(
            'inline-flex h-[22px] max-w-[190px] items-center gap-1 rounded-[3px] border border-primary/35 bg-primary/10 px-1.5 text-[10.5px] text-primary'
          )}
        >
          <FilterX className="size-3 shrink-0" />
          <span className="truncate">{where}</span>
        </button>
      ) : null}
    </div>
  )
}

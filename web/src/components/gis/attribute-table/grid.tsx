import { ArrowDown, ArrowUp } from 'lucide-react'
import type { PointerEvent as ReactPointerEvent } from 'react'

import type { CatalogField } from '@/lib/api'
import { cn } from '@/lib/utils'

import { isNumericField } from './use-rows'

/** 列宽：数值列窄一点，字符串列按列名给个下限，最多 240px。 */
export function defaultColumnWidth(field: CatalogField): number {
  if (isNumericField(field)) return 96
  const label = field.alias || field.name
  return Math.min(240, Math.max(80, label.length * 8 + 34))
}

const MIN_COLUMN = 48
const ROW_NO_WIDTH = 46

function cellText(value: unknown): string {
  if (value === null || value === undefined || value === '') return ''
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return String(value)
    if (Number.isInteger(value)) return String(value)
    const abs = Math.abs(value)
    // 不能统一 toFixed(3)：经纬度/度单位的面积动辄 1e-5，截成 0.000 等于把数据
    // 显示成假值；极小/极大用科学计数法，其余保留 4 位小数。
    if (abs < 1e-3 || abs >= 1e7) {
      return value.toExponential(4).replace(/\.?0+e/, 'e')
    }
    return String(Number(value.toFixed(4)))
  }
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value)
    } catch {
      return String(value)
    }
  }
  return String(value)
}

export interface TableGridProps {
  /** 本页第一行的服务端行号（footer/行号列共用，不许从 1 重新数）。 */
  offset: number
  fields: CatalogField[]
  rows: Record<string, unknown>[]
  /** 要素标识列；null 时行点击不能联动地图（如实禁用，不猜）。 */
  idField: string | null
  selectedIds: string[]
  order: string
  dir: 'asc' | 'desc'
  widths: Record<string, number>
  onWidth: (field: string, width: number) => void
  onResetWidth: (field: string) => void
  onSort: (field: string) => void
  onToggleRow: (id: string, additive: boolean) => void
  onReadonlyCell: () => void
}

/**
 * 属性表格（spec §3：行高 23px、正文 11.5px、层次只靠 1px 描边）。
 *
 * - 表头点击 = 服务端排序（由父组件发请求，本组件不碰数据顺序）；
 * - 表头右缘拖动 = 列宽，双击复位；
 * - 行点击 = 选中要素（与地图共用一份选择集）；
 * - 双击单元格 = 本期只读提示（没有写入接口，不假装能编辑）。
 */
export function TableGrid({
  offset,
  fields,
  rows,
  idField,
  selectedIds,
  order,
  dir,
  widths,
  onWidth,
  onResetWidth,
  onSort,
  onToggleRow,
  onReadonlyCell,
}: TableGridProps) {
  const selected = new Set(selectedIds)

  const startResize = (
    event: ReactPointerEvent<HTMLElement>,
    field: CatalogField,
    current: number
  ) => {
    event.preventDefault()
    event.stopPropagation()
    const startX = event.clientX
    const move = (moveEvent: PointerEvent) => {
      onWidth(field.name, Math.max(MIN_COLUMN, current + moveEvent.clientX - startX))
    }
    const up = () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
      document.body.style.cursor = ''
    }
    document.body.style.cursor = 'col-resize'
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }

  return (
    <table
      className="table-fixed border-separate border-spacing-0 text-[11.5px]"
      data-testid="attribute-table"
      style={{ width: ROW_NO_WIDTH + fields.reduce((sum, field) => sum + (widths[field.name] ?? defaultColumnWidth(field)), 0) }}
    >
      <colgroup>
        <col style={{ width: ROW_NO_WIDTH }} />
        {fields.map((field) => (
          <col key={field.name} style={{ width: widths[field.name] ?? defaultColumnWidth(field) }} />
        ))}
      </colgroup>
      <thead>
        <tr>
          <th
            scope="col"
            className="sticky top-0 z-[2] h-6 border-r border-b border-border-strong bg-[var(--table-header)] px-1 text-right text-[10.5px] font-semibold text-muted-foreground"
            title="服务端行号：按当前排序/筛选口径算，不从 1 重新数"
          >
            #
          </th>
          {fields.map((field) => {
            const active = order === field.name
            return (
              <th
                key={field.name}
                scope="col"
                aria-sort={active ? (dir === 'asc' ? 'ascending' : 'descending') : 'none'}
                title={`${field.alias || field.name}${field.type ? ` · ${field.type}` : ''}\n点击排序（服务端）· 拖动右缘改列宽 · 双击复位`}
                onClick={() => onSort(field.name)}
                className={cn(
                  'group relative sticky top-0 z-[2] h-6 cursor-pointer border-r border-b border-border-strong bg-[var(--table-header)] px-2 text-left text-[10.5px] font-semibold whitespace-nowrap select-none',
                  active ? 'text-primary' : 'text-foreground/80 hover:text-foreground'
                )}
              >
                <span className="flex items-center gap-1">
                  <span className="truncate">{field.alias || field.name}</span>
                  {active ? (
                    dir === 'asc' ? (
                      <ArrowUp className="size-2.5 shrink-0" />
                    ) : (
                      <ArrowDown className="size-2.5 shrink-0" />
                    )
                  ) : null}
                </span>
                <span
                  role="separator"
                  aria-orientation="vertical"
                  aria-label={`调整 ${field.name} 列宽（双击复位）`}
                  onPointerDown={(event) =>
                    startResize(
                      event,
                      field,
                      widths[field.name] ?? defaultColumnWidth(field)
                    )
                  }
                  onClick={(event) => event.stopPropagation()}
                  onDoubleClick={(event) => {
                    event.stopPropagation()
                    onResetWidth(field.name)
                  }}
                  className="absolute inset-y-0 right-0 w-[5px] cursor-col-resize opacity-0 group-hover:opacity-100 hover:bg-primary/45"
                />
              </th>
            )
          })}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, index) => {
          const id = idField ? row[idField] : null
          const key = id === null || id === undefined ? `row-${offset + index}` : String(id)
          const isSelected = id !== null && id !== undefined && selected.has(String(id))
          return (
            <tr
              key={key}
              data-testid="attribute-row"
              onClick={(event) =>
                id !== null &&
                id !== undefined &&
                onToggleRow(String(id), event.ctrlKey || event.metaKey || event.shiftKey)
              }
              className={cn(
                isSelected ? 'bg-primary/12' : 'hover:bg-primary/5',
                id !== null && id !== undefined ? 'cursor-pointer' : ''
              )}
            >
              <td className="h-[var(--gis-row-table)] border-r border-b border-border/60 bg-muted/40 px-1 text-right text-[10.5px] tabular-nums text-muted-foreground">
                {offset + index + 1}
              </td>
              {fields.map((field) => {
                const raw = row[field.name]
                const text = cellText(raw)
                const numeric = typeof raw === 'number'
                return (
                  <td
                    key={field.name}
                    title={text || '（空值）'}
                    onDoubleClick={onReadonlyCell}
                    className={cn(
                      'h-[var(--gis-row-table)] max-w-0 truncate border-r border-b border-border/60 px-2 whitespace-nowrap',
                      numeric ? 'text-right tabular-nums text-foreground/85' : 'text-foreground/85',
                      text === '' && 'text-muted-foreground/60'
                    )}
                  >
                    {text || '—'}
                  </td>
                )
              })}
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

export { MIN_COLUMN, ROW_NO_WIDTH }

import { ChevronLeft, ChevronRight, Columns3 } from 'lucide-react'
import { Fragment, useMemo, useState, type ReactNode } from 'react'

import { Empty } from '@/components/ui/empty'
import { Select } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'

export interface DataTableColumn<T> {
  id: string
  header: ReactNode
  cell: (row: T) => ReactNode
  /** 数字列右对齐并走 tabular-nums（§2.5 第 4 条）。 */
  align?: 'left' | 'right'
  /** 允许在列显隐菜单里关掉。 */
  hideable?: boolean
}

export interface DataTableProps<T> {
  columns: DataTableColumn<T>[]
  rows: T[]
  rowKey: (row: T) => string
  /** 左上角筛选/操作区。 */
  toolbar?: ReactNode
  loading?: boolean
  empty?: ReactNode
  initialPageSize?: number
  pageSizeOptions?: number[]
  /** 服务端分页时的总数；不传则用 rows.length。 */
  total?: number
  onRowClick?: (row: T) => void
  /** 传了就启用行展开：首列自动多一个展开按钮，展开内容占满整行。 */
  renderExpanded?: (row: T) => ReactNode
  className?: string
}

/**
 * 表格壳：工具栏 + 列显隐 + 无竖线两行单元格 + 底部「总计 / 每页 / 分页器」。
 *
 * 本单只立壳（S2），行数据由 S4/S6 接真接口；分页状态先在前端切片，
 * 接服务端分页时把 ``total`` 传进来即可。
 */
export function DataTable<T>({
  columns,
  rows,
  rowKey,
  toolbar,
  loading = false,
  empty,
  initialPageSize = 20,
  pageSizeOptions = [10, 20, 50],
  total,
  onRowClick,
  renderExpanded,
  className,
}: DataTableProps<T>) {
  const [hidden, setHidden] = useState<string[]>([])
  const [expanded, setExpanded] = useState<string[]>([])
  const [pageSize, setPageSize] = useState(initialPageSize)
  const [page, setPage] = useState(1)

  const visibleColumns = useMemo(
    () => columns.filter((column) => !hidden.includes(column.id)),
    [columns, hidden]
  )

  const rowCount = total ?? rows.length
  const pageCount = Math.max(1, Math.ceil(rowCount / pageSize))
  const safePage = Math.min(page, pageCount)
  const pageRows = useMemo(
    () => rows.slice((safePage - 1) * pageSize, safePage * pageSize),
    [rows, safePage, pageSize]
  )

  const toggleColumn = (id: string) => {
    setHidden((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id]
    )
  }

  const toggleRow = (key: string) => {
    setExpanded((current) =>
      current.includes(key) ? current.filter((item) => item !== key) : [...current, key]
    )
  }

  return (
    <div
      data-slot="data-table"
      className={cn('overflow-hidden rounded-lg border border-border bg-card', className)}
    >
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-3 py-2">
        <div className="min-w-0 flex-1">{toolbar}</div>
        <details className="relative">
          <summary className="flex h-8 cursor-pointer list-none items-center gap-1.5 rounded-full border border-border px-3 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground">
            <Columns3 className="size-3.5" />
            列
          </summary>
          <div className="absolute right-0 z-20 mt-1 w-44 rounded-md border border-border bg-popover p-1.5">
            {columns.map((column) => (
              <label
                key={column.id}
                className="flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-xs text-foreground hover:bg-accent"
              >
                <input
                  type="checkbox"
                  className="size-3.5 accent-[var(--primary)]"
                  checked={!hidden.includes(column.id)}
                  disabled={column.hideable === false}
                  onChange={() => toggleColumn(column.id)}
                />
                <span className="truncate">{column.header}</span>
              </label>
            ))}
          </div>
        </details>
      </div>

      {loading ? (
        <div className="space-y-3 px-4 py-5">
          {Array.from({ length: 6 }).map((_, index) => (
            <Skeleton key={index} className="h-9 w-full" />
          ))}
        </div>
      ) : rows.length === 0 ? (
        <div className="p-4">
          {empty ?? (
            <Empty
              title="暂无数据"
              description="这里会在有运行记录或数据资产后自动填充。"
              className="min-h-[260px] border-0 bg-transparent"
            />
          )}
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="bg-[var(--table-header)] text-xs font-medium text-muted-foreground">
                {renderExpanded ? <th scope="col" className="w-8 px-2 py-2.5" /> : null}
                {visibleColumns.map((column) => (
                  <th
                    key={column.id}
                    scope="col"
                    className={cn(
                      'px-4 py-2.5 text-left font-medium whitespace-nowrap',
                      column.align === 'right' && 'text-right'
                    )}
                  >
                    {column.header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {pageRows.map((row) => {
                const key = rowKey(row)
                const open = expanded.includes(key)
                return (
                  <Fragment key={key}>
                    <tr
                      onClick={onRowClick ? () => onRowClick(row) : undefined}
                      className={cn(
                        'transition-colors',
                        onRowClick ? 'cursor-pointer hover:bg-[var(--table-header-hover)]' : undefined
                      )}
                    >
                      {renderExpanded ? (
                        <td className="w-8 px-2 py-2.5">
                          <button
                            type="button"
                            aria-label={open ? '收起' : '展开'}
                            aria-expanded={open}
                            className="grid size-6 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
                            onClick={(event) => {
                              event.stopPropagation()
                              toggleRow(key)
                            }}
                          >
                            <ChevronRight
                              className={cn('size-3.5 transition-transform', open && 'rotate-90')}
                            />
                          </button>
                        </td>
                      ) : null}
                      {visibleColumns.map((column) => (
                        <td
                          key={column.id}
                          className={cn(
                            'px-4 py-2.5 align-middle',
                            column.align === 'right' && 'text-right tabular-nums'
                          )}
                        >
                          {column.cell(row)}
                        </td>
                      ))}
                    </tr>
                    {renderExpanded && open ? (
                      <tr className="bg-[var(--table-header)]/60">
                        <td colSpan={visibleColumns.length + 1} className="px-4 py-3">
                          {renderExpanded(row)}
                        </td>
                      </tr>
                    ) : null}
                  </Fragment>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border px-4 py-2.5 text-xs text-muted-foreground">
        <div className="flex items-center gap-1.5">
          总计 <span className="tabular-nums text-foreground">{rowCount}</span> 条
          {rowCount > 0 ? (
            <span className="text-muted-foreground/70">
              · 第 <span className="tabular-nums">{safePage}</span> /{' '}
              <span className="tabular-nums">{pageCount}</span> 页
            </span>
          ) : null}
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5">
            <span>每页</span>
            <Select
              aria-label="每页条数"
              className="h-7 rounded-md border border-border bg-transparent px-1.5 text-xs text-foreground"
              value={String(pageSize)}
              onValueChange={(next) => {
                setPageSize(Number(next))
                setPage(1)
              }}
              options={pageSizeOptions.map((size) => ({ value: String(size), label: String(size) }))}
            />
            <span>条</span>
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              aria-label="上一页"
              className="grid size-7 place-items-center rounded-full border border-border transition-colors hover:bg-accent hover:text-accent-foreground disabled:opacity-40"
              disabled={safePage <= 1}
              onClick={() => setPage(safePage - 1)}
            >
              <ChevronLeft className="size-3.5" />
            </button>
            <button
              type="button"
              aria-label="下一页"
              className="grid size-7 place-items-center rounded-full border border-border transition-colors hover:bg-accent hover:text-accent-foreground disabled:opacity-40"
              disabled={safePage >= pageCount}
              onClick={() => setPage(safePage + 1)}
            >
              <ChevronRight className="size-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

import {
  AlertTriangle,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Loader2,
  Plus,
  RefreshCw,
  SquareDashedMousePointer,
  Table2,
  X,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { Select } from '@/components/ui/select'
import type { DatasetItem } from '@/lib/api'
import { cn } from '@/lib/utils'

import { describeApiError } from '../right-dock/errors'
import { formatNumber, kindLabel } from '../format'
import { useGisStore, type GisLayer } from '../store'
import { FilterBar } from './filter-bar'
import { TableGrid } from './grid'
import {
  DEFAULT_VIEW,
  SELECTION_ROW_CAP,
  idFieldOf,
  isNumericField,
  useSelectionRows,
  useTableRows,
  type TextViewState,
} from './use-rows'

/** 后端 `TABLE_KINDS`：只有矢量/要素类/独立表格有属性表。 */
const TABLE_KINDS = new Set(['vector', 'feature_class', 'table'])

const PAGE_SIZES = [
  { value: '50', label: '50 行/页' },
  { value: '100', label: '100 行/页' },
  { value: '200', label: '200 行/页' },
]

const EMPTY_IDS: string[] = []

export interface AttributeTableDockProps {
  layers: GisLayer[]
  /** catalog 全量：非空间表不进图层栈，属性表 tab 只能靠它解析。 */
  items: DatasetItem[]
  height: number
  onCollapse: () => void
}

/** 属性表 tab 的数据：图层栈里的图层，或目录里只能看表的非空间表。 */
interface TableTab {
  path: string
  name: string
  kind: string
  item: DatasetItem
}

/**
 * 底部属性表 dock（spec §4 ③，DAIR-331）。
 *
 * 数据口径（不许含糊）：
 * - 行、排序、筛选、total 全部来自 `/api/catalog/rows`（服务端分页），
 *   前端不对已加载的 50 行做二次排序/筛选；
 * - footer 只写服务端给的数字：`显示 1–50 / 12,847（服务端分页）`；
 *   行缓存被截断时再挂一条「行缓存仅前 N 行」，绝不写「已加载全量」；
 * - 「仅显示选中要素」是**选择集视图**（点选/框选得到的 id，逐行按 id 取数），
 *   footer 明写它不是分页口径；
 * - 单元格编辑本期只读（没有写入接口），双击给提示，不改数据。
 */
export function AttributeTableDock({ layers, items, height, onCollapse }: AttributeTableDockProps) {
  const tableTabs = useGisStore((state) => state.tableTabs)
  const tableTab = useGisStore((state) => state.tableTab)
  const setTableTab = useGisStore((state) => state.setTableTab)
  const closeTable = useGisStore((state) => state.closeTable)
  const openTable = useGisStore((state) => state.openTable)
  const setIdField = useGisStore((state) => state.setIdField)
  const toggleFeature = useGisStore((state) => state.toggleFeature)
  const selectionMap = useGisStore((state) => state.selection)

  const [views, setViews] = useState<Record<string, TextViewState>>({})
  const [widths, setWidths] = useState<Record<string, Record<string, number>>>({})
  const [hint, setHint] = useState<string | null>(null)
  const hintTimer = useRef<number | null>(null)

  const tabs = tableTabs.flatMap<TableTab>((path) => {
    const layer = layers.find((item) => item.path === path)
    if (layer) {
      return [{ path: layer.path, name: layer.name, kind: layer.kind, item: layer.item }]
    }
    // 非空间表只开属性表、不进图层栈（DAIR-343）：它的 tab 从 catalog 解析。
    const item = items.find((entry) => entry.path === path)
    return item ? [{ path: item.path, name: item.name, kind: item.kind, item }] : []
  })
  const active = tabs.find((layer) => layer.path === tableTab) ?? tabs[0] ?? null
  const path = active?.path ?? null
  const view = (path ? views[path] : undefined) ?? DEFAULT_VIEW
  const tabular = Boolean(active && TABLE_KINDS.has(active.kind))

  const rowsQuery = useTableRows(path, view, tabular && !view.selectionOnly)
  const data = rowsQuery.data
  const fields = useMemo(() => data?.fields ?? [], [data])
  const idField = useMemo(() => (fields.length > 0 ? idFieldOf(fields) : null), [fields])
  const numericId = isNumericField(fields.find((field) => field.name === idField))
  const selectedIds = (path ? selectionMap[path] : undefined) ?? EMPTY_IDS
  const selectionRows = useSelectionRows(
    path,
    idField,
    numericId,
    view.selectionOnly ? selectedIds : EMPTY_IDS
  )

  // 表格一旦拿到字段头，就把要素标识列登记给地图：预览会带上这一列，
  // 点选/框选才能把要素对回表格行。
  useEffect(() => {
    if (path && idField) setIdField(path, idField)
  }, [path, idField, setIdField])

  useEffect(
    () => () => {
      if (hintTimer.current) window.clearTimeout(hintTimer.current)
    },
    []
  )

  const flash = (message: string) => {
    setHint(message)
    if (hintTimer.current) window.clearTimeout(hintTimer.current)
    hintTimer.current = window.setTimeout(() => setHint(null), 4200)
  }

  const patchView = (patch: Partial<TextViewState>) => {
    if (!path) return
    const current = views[path] ?? DEFAULT_VIEW
    setViews({ ...views, [path]: { ...current, ...patch } })
  }

  const setWidth = (field: string, width: number) => {
    if (!path) return
    setWidths({ ...widths, [path]: { ...(widths[path] ?? {}), [field]: width } })
  }

  const resetWidth = (field: string) => {
    if (!path) return
    const next = { ...(widths[path] ?? {}) }
    delete next[field]
    setWidths({ ...widths, [path]: next })
  }

  const sortBy = (field: string) => {
    if (view.order === field) {
      patchView({ dir: view.dir === 'asc' ? 'desc' : 'asc', offset: 0 })
    } else {
      patchView({ order: field, dir: 'asc', offset: 0 })
    }
  }

  const openNextTable = () => {
    const candidate = layers.find(
      (layer) => TABLE_KINDS.has(layer.kind) && !tableTabs.includes(layer.path)
    )
    if (candidate) openTable(candidate.path)
  }

  const total = data?.total ?? 0
  const offset = data?.offset ?? 0
  const returned = data?.returned ?? 0
  const pageCount = Math.max(1, Math.ceil(total / view.limit))
  const page = Math.min(pageCount, Math.floor(offset / view.limit) + 1)
  const selectionOverflow = selectedIds.length > SELECTION_ROW_CAP
  const shownRows = view.selectionOnly ? selectionRows.rows : (data?.rows ?? [])
  const error = view.selectionOnly ? selectionRows.error : rowsQuery.error
  const issue = error ? describeApiError(error) : null
  const retry = view.selectionOnly ? selectionRows.refetch : rowsQuery.refetch
  const loading = view.selectionOnly
    ? selectionRows.loading
    : rowsQuery.loading || (rowsQuery.fetching && !data)

  return (
    <section
      className="relative flex shrink-0 flex-col overflow-hidden border-t border-border bg-card"
      style={{ height }}
      aria-label="属性表"
      data-testid="attribute-table-dock"
    >
      {/* tab 条 29px：一图层一 tab，关闭 tab 不动图层 */}
      <div className="flex h-[29px] shrink-0 items-stretch overflow-hidden border-b border-border bg-background">
        {tabs.map((layer) => {
          const on = layer.path === active?.path
          return (
            <div
              key={layer.path}
              className={cn(
                'relative flex max-w-[230px] items-center gap-1.5 border-r border-border px-2.5 text-[11.5px]',
                on ? 'bg-card font-semibold text-foreground' : 'text-muted-foreground'
              )}
            >
              <button
                type="button"
                onClick={() => setTableTab(layer.path)}
                className="max-w-[180px] truncate"
                title={layer.path}
              >
                {layer.name}
              </button>
              <button
                type="button"
                aria-label={`关闭 ${layer.name} 的属性表`}
                title="关闭这个 tab（图层仍在地图上）"
                onClick={() => closeTable(layer.path)}
                className="grid size-3.5 place-items-center rounded-[2px] text-muted-foreground hover:bg-accent hover:text-foreground"
              >
                <X className="size-2.5" />
              </button>
              {on ? (
                <span aria-hidden="true" className="absolute inset-x-0 top-0 h-0.5 bg-primary" />
              ) : null}
            </div>
          )
        })}
        <div className="ml-auto flex items-center gap-1 px-1.5">
          <button
            type="button"
            onClick={openNextTable}
            disabled={layers.every(
              (layer) => !TABLE_KINDS.has(layer.kind) || tableTabs.includes(layer.path)
            )}
            title="打开下一个图层的属性表"
            className="inline-flex h-5 items-center gap-0.5 rounded-[2px] px-1 text-[10.5px] text-muted-foreground hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-45"
          >
            <Plus className="size-3" /> 表
          </button>
          <span className="text-[10px] text-muted-foreground">属性表</span>
          <button
            type="button"
            title="收起属性表（Ctrl+T）"
            aria-label="收起属性表"
            onClick={onCollapse}
            className="grid size-5 place-items-center rounded-[2px] text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            <ChevronDown className="size-3.5" />
          </button>
        </div>
      </div>

      {!active ? (
        <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-1 px-4 text-center text-[11.5px] text-muted-foreground">
          <Table2 className="size-5 opacity-50" />
          <span className="font-medium text-foreground/70">属性表未打开</span>
          <span>
            在图层行点表格按钮、在目录里双击表格，或按{' '}
            <kbd className="rounded border border-border px-1">Ctrl+T</kbd> 打开当前图层。
          </span>
        </div>
      ) : (
        <>
          {/* 工具栏 30px：筛选（服务端）/ 选择集 / 字段可见性占位 */}
          <div className="flex h-[30px] shrink-0 items-center gap-1.5 border-b border-border bg-background px-2">
            <FilterBar
              key={active.path}
              fields={fields}
              where={view.where}
              disabled={!tabular || view.selectionOnly}
              onApply={(where) => patchView({ where, offset: 0 })}
              onClear={() => patchView({ where: '', offset: 0 })}
            />
            <span className="h-[16px] w-px shrink-0 bg-border" aria-hidden="true" />
            <button
              type="button"
              aria-pressed={view.selectionOnly}
              disabled={!tabular}
              onClick={() => patchView({ selectionOnly: !view.selectionOnly, offset: 0 })}
              title={
                !idField
                  ? '该表没有 OBJECTID/FID 之类的要素标识列，选择集联动不可用'
                  : selectedIds.length === 0
                    ? '地图上还没有选中要素：切到「选择」或「框选」工具点一个'
                    : `只显示地图选择集里的 ${selectedIds.length} 个要素`
              }
              className={cn(
                'inline-flex h-[22px] shrink-0 items-center gap-1 rounded-[3px] border px-1.5 text-[10.5px]',
                view.selectionOnly
                  ? 'border-primary/45 bg-primary/12 font-semibold text-primary'
                  : 'border-border bg-card text-foreground/80 hover:bg-accent',
                !tabular && 'cursor-not-allowed opacity-45'
              )}
            >
              <SquareDashedMousePointer className="size-3" />
              仅显示选中要素
              {selectedIds.length > 0 ? (
                <span className="tabular-nums">（{formatNumber(selectedIds.length)}）</span>
              ) : null}
            </button>
            <button
              type="button"
              disabled
              title="字段可见性本期留空（不影响分页与排序）"
              className="hidden h-[22px] shrink-0 cursor-not-allowed items-center rounded-[3px] border border-border px-1.5 text-[10.5px] text-muted-foreground opacity-55 xl:inline-flex"
            >
              字段（{fields.length}）
            </button>
            <span className="min-w-0 flex-1" />
            <span className="hidden truncate text-[10.5px] text-muted-foreground lg:inline">
              点表头排序 · 双击单元格编辑（本期只读）
            </span>
            <button
              type="button"
              onClick={retry}
              title="重新读取这一页（服务端）"
              className="grid size-5 shrink-0 place-items-center rounded-[2px] text-muted-foreground hover:bg-accent hover:text-foreground"
            >
              <RefreshCw
                className={cn(
                  'size-3',
                  (view.selectionOnly ? selectionRows.loading : rowsQuery.fetching) && 'animate-spin'
                )}
              />
            </button>
          </div>

          {/* 表格主体 */}
          <div className="relative min-h-0 flex-1 overflow-auto">
            {!tabular ? (
              <div className="grid h-full place-items-center px-4 text-center text-[11.5px] leading-relaxed text-muted-foreground">
                {active.kind === 'raster'
                  ? '栅格图层没有属性表：像元 / 波段信息在右侧「元数据」里。'
                  : '该类型没有属性表（表格/工程等非空间数据）。'}
              </div>
            ) : issue ? (
              <div className="flex h-full flex-col items-center justify-center gap-1.5 px-4 text-center text-[11.5px]">
                <AlertTriangle className="size-4 text-destructive" />
                <b className="text-destructive">{issue.title}</b>
                <span className="max-w-[560px] text-muted-foreground">{issue.reason}</span>
                <button
                  type="button"
                  onClick={retry}
                  className="mt-1 inline-flex items-center gap-1 rounded-[var(--gis-radius)] border border-border bg-card px-2 py-0.5 text-[10.5px] hover:bg-accent"
                >
                  <RefreshCw className="size-3" /> 重试
                </button>
              </div>
            ) : loading ? (
              <div className="flex h-full flex-col gap-1.5 p-2">
                <div className="flex items-center gap-1.5 text-[10.5px] text-muted-foreground">
                  <Loader2 className="size-3 animate-spin" />
                  正在从服务端读取属性表（首次读表会建行缓存，之后翻页只走 SQLite）…
                </div>
                {Array.from({ length: 6 }).map((_, index) => (
                  <div
                    key={index}
                    className="h-[18px] w-full animate-pulse rounded-[2px] bg-muted"
                    style={{ width: `${96 - index * 7}%` }}
                  />
                ))}
              </div>
            ) : view.selectionOnly && selectedIds.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center gap-1 px-4 text-center text-[11.5px] text-muted-foreground">
                <SquareDashedMousePointer className="size-5 opacity-50" />
                <span className="font-medium text-foreground/70">地图上还没有选中要素</span>
                <span>
                  在命令栏点「选择」后单击要素，或点「框选」拖一个矩形；
                  <kbd className="mx-1 rounded border border-border px-1">Shift</kbd>加选、
                  <kbd className="mx-1 rounded border border-border px-1">Esc</kbd>清除。
                </span>
              </div>
            ) : shownRows.length === 0 ? (
              view.selectionOnly ? (
                <div className="grid h-full place-items-center px-4 text-center text-[11.5px] text-muted-foreground">
                  选中的要素不在行缓存里（表大于 5 万行时只缓存前 5 万行）。
                </div>
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-1.5 px-4 text-center text-[11.5px] text-muted-foreground">
                  <span className="font-medium text-foreground/70">
                    {view.where ? '没有符合筛选条件的行' : '这张表没有数据行'}
                  </span>
                  {view.where ? (
                    <button
                      type="button"
                      onClick={() => patchView({ where: '', offset: 0 })}
                      className="text-primary hover:underline"
                    >
                      清除筛选「{view.where}」
                    </button>
                  ) : null}
                </div>
              )
            ) : (
              <TableGrid
                offset={view.selectionOnly ? 0 : offset}
                fields={fields}
                rows={shownRows}
                idField={idField}
                selectedIds={selectedIds}
                order={view.order}
                dir={view.dir}
                widths={widths[path] ?? {}}
                onWidth={setWidth}
                onResetWidth={resetWidth}
                onSort={sortBy}
                onToggleRow={(id, additive) => {
                  if (!path) return
                  toggleFeature(path, id, additive)
                }}
                onReadonlyCell={() =>
                  flash('单元格编辑本期只读：写入接口未开放，双击不会修改数据。')
                }
              />
            )}

            {hint ? (
              <div
                role="status"
                className="pointer-events-none sticky bottom-1 left-1/2 z-10 mx-auto w-fit rounded-[var(--gis-radius)] border border-warning/45 bg-warning/15 px-2 py-1 text-[10.5px] text-foreground shadow-[var(--gis-shadow-float)]"
              >
                {hint}
              </div>
            ) : null}
          </div>

          {/* footer 24px：数字全部来自服务端，口径写在括号里 */}
          <div className="flex h-6 shrink-0 items-center gap-2 border-t border-border bg-background px-2 text-[10.5px] text-muted-foreground">
            {view.selectionOnly ? (
              <span data-testid="table-footer">
                仅显示选中 {formatNumber(selectedIds.length)} 个要素（选中集视图，非服务端分页）
                {selectionOverflow
                  ? ` · 已展开前 ${SELECTION_ROW_CAP} 个`
                  : ''}
              </span>
            ) : data ? (
              <span data-testid="table-footer">
                {returned > 0
                  ? `显示 ${formatNumber(offset + 1)}–${formatNumber(offset + returned)}`
                  : '显示 0'}{' '}
                / {data?.cache.truncated ? `已缓存 ${formatNumber(total)}` : formatNumber(total)}
                {view.where ? '（筛选后）' : ''}（服务端分页）
              </span>
            ) : (
              <span data-testid="table-footer">—</span>
            )}
            {rowsQuery.fetching && data && !loading ? (
              <span className="inline-flex items-center gap-0.5 text-primary">
                <Loader2 className="size-2.5 animate-spin" /> 更新中
              </span>
            ) : null}
            {!view.selectionOnly && data?.cache.truncated ? (
              <span
                className="rounded-[2px] border border-warning/45 bg-warning/15 px-1 text-[9.5px] font-semibold text-warning"
                title={data.cache.notes.join('；') || '表大于行缓存上限'}
              >
                行缓存仅前 {formatNumber(data.cache.rows)} 行
              </span>
            ) : null}
            {!view.selectionOnly && data?.cache.hit === false && data.cache.build_ms > 0 ? (
              <span title="首次读表建行缓存（一次性），之后翻页只查 SQLite">
                首次建缓存 {Math.round(data.cache.build_ms)} ms
              </span>
            ) : null}
            <span className="min-w-0 flex-1" />
            <span className="hidden shrink-0 tabular-nums md:inline">
              空间参考{' '}
              {active.item.wkid
                ? `EPSG:${active.item.wkid}`
                : active.item.crs_name || '—'}
              {' · '}
              {kindLabel(active.kind)}
            </span>
            {tabular ? (
              <>
                <span className="hidden shrink-0 opacity-60 md:inline">·</span>
                <span className="shrink-0">只读</span>
                {!view.selectionOnly && data ? (
                  <div className="ml-1 flex shrink-0 items-center gap-0.5">
                    <button
                      type="button"
                      title="第一页"
                      aria-label="第一页"
                      disabled={page <= 1}
                      onClick={() => patchView({ offset: 0 })}
                      className="grid size-[18px] place-items-center rounded-[2px] hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-35"
                    >
                      <ChevronsLeft className="size-3" />
                    </button>
                    <button
                      type="button"
                      title="上一页"
                      aria-label="上一页"
                      disabled={page <= 1}
                      onClick={() => patchView({ offset: Math.max(0, offset - view.limit) })}
                      className="grid size-[18px] place-items-center rounded-[2px] hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-35"
                    >
                      <ChevronLeft className="size-3" />
                    </button>
                    <span className="px-0.5 tabular-nums">
                      第 {formatNumber(page)} / {formatNumber(pageCount)} 页
                    </span>
                    <button
                      type="button"
                      title="下一页"
                      aria-label="下一页"
                      disabled={page >= pageCount}
                      onClick={() =>
                        patchView({ offset: Math.min((pageCount - 1) * view.limit, offset + view.limit) })
                      }
                      className="grid size-[18px] place-items-center rounded-[2px] hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-35"
                    >
                      <ChevronRight className="size-3" />
                    </button>
                    <button
                      type="button"
                      title="最后一页"
                      aria-label="最后一页"
                      disabled={page >= pageCount}
                      onClick={() => patchView({ offset: (pageCount - 1) * view.limit })}
                      className="grid size-[18px] place-items-center rounded-[2px] hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-35"
                    >
                      <ChevronsRight className="size-3" />
                    </button>
                    <Select
                      value={String(view.limit)}
                      onValueChange={(next) =>
                        patchView({ limit: Number(next), offset: 0 })
                      }
                      options={PAGE_SIZES}
                      aria-label="每页行数"
                      className="h-[18px] rounded-[2px] border border-border bg-card px-1 text-[10px] text-muted-foreground"
                    />
                  </div>
                ) : null}
              </>
            ) : null}
          </div>
        </>
      )}
    </section>
  )
}

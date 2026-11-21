import { keepPreviousData, useQueries, useQuery } from '@tanstack/react-query'

import {
  getCatalogRows,
  type CatalogField,
  type CatalogRowsResult,
} from '@/lib/api'

/**
 * 属性表每个 tab 的**视图状态**（排序 / 筛选 / 页码）。
 *
 * 存在 dock 组件里按图层路径分键，所以切 tab 再切回来还是原来那一页；
 * 数据本身由 React Query 按 queryKey 缓存，切回来在 `staleTime` 内不发请求。
 */
export interface TextViewState {
  offset: number
  limit: number
  /** 排序字段名；空 = 按服务端行号。 */
  order: string
  dir: 'asc' | 'desc'
  /** 已应用的筛选表达式（`字段 op 值`），空 = 不筛选。 */
  where: string
  /** 只显示地图选择集里的要素（选择集来自点选/框选，不是服务端分页）。 */
  selectionOnly: boolean
}

export const DEFAULT_VIEW: TextViewState = {
  offset: 0,
  limit: 50,
  order: '',
  dir: 'asc',
  where: '',
  selectionOnly: false,
}

/** 选择集视图最多展开多少行（每行一次按 id 的取数，别让它变成 400 个并发）。 */
export const SELECTION_ROW_CAP = 50

/** 字段类型是否数值：`where` 里数值不加引号，字符串必须加引号。 */
export function isNumericField(field: CatalogField | undefined): boolean {
  const type = String(field?.type ?? '').toLowerCase()
  return ['oid', 'integer', 'double', 'single', 'real', 'number', 'bigint'].some((token) =>
    type.includes(token)
  )
}

/**
 * 要素标识字段：arcpy 的 OID 类型优先，其次认常见列名。
 *
 * 地图点选/框选靠它把 `queryRenderedFeatures` 查到的要素对回表格行；
 * 认不出来就如实禁用选择联动（地图上仍可点选，只是表格无法高亮对应行）。
 */
export function idFieldOf(fields: CatalogField[]): string | null {
  const oid = fields.find((field) => String(field.type ?? '').toUpperCase() === 'OID')
  if (oid) return oid.name
  const named = fields.find((field) => /^(objectid|fid|oid|gid|id)$/i.test(field.name))
  return named?.name ?? null
}

/** 值 → `where` 字面量：数值直写，字符串加单引号并转义内部的引号。 */
export function whereValue(field: CatalogField | undefined, raw: string): string {
  const value = raw.trim()
  if (!value) return ''
  if (isNumericField(field) && Number.isFinite(Number(value))) return value
  return `'${value.replace(/'/g, "''")}'`
}

export function buildWhere(
  field: CatalogField | undefined,
  op: string,
  raw: string
): string {
  const value = whereValue(field, raw)
  if (!field || !op || !value) return ''
  return `${field.name} ${op} ${value}`
}

export interface TableRowsState {
  data?: CatalogRowsResult
  error?: Error
  /** 首次加载（没有可展示的上一页数据）。 */
  loading: boolean
  /** 翻页/改排序时仍在取数：表格保留上一页，footer 标「更新中」。 */
  fetching: boolean
  refetch: () => void
}

/**
 * 服务端分页：`/api/catalog/rows`。排序 / 筛选全部交给服务端，
 * **不许**把已加载的 50 行在本地再排再筛（那会让 total 与首行都失真）。
 */
export function useTableRows(
  path: string | null,
  view: TextViewState,
  enabled: boolean
): TableRowsState {
  const query = useQuery<CatalogRowsResult>({
    queryKey: [
      'catalog-rows',
      path,
      view.offset,
      view.limit,
      view.order,
      view.dir,
      view.where,
    ],
    queryFn: () =>
      getCatalogRows(path as string, {
        offset: view.offset,
        limit: view.limit,
        order: view.order || undefined,
        dir: view.order ? view.dir : undefined,
        where: view.where || undefined,
      }),
    enabled: Boolean(path) && enabled,
    placeholderData: keepPreviousData,
    staleTime: 30_000,
    retry: false,
  })

  return {
    data: query.data,
    error: (query.error as Error | null) ?? undefined,
    loading: query.isLoading && Boolean(path) && enabled,
    fetching: query.isFetching,
    refetch: () => void query.refetch(),
  }
}

export interface SelectionRowsState {
  rows: Record<string, unknown>[]
  /** 这次最多展开多少行（超过选择集条数时由调用方写进 footer）。 */
  shown: number
  loading: boolean
  error?: Error
  refetch: () => void
}

/**
 * 选择集视图：按 id 逐行取数（`where = <idField> = <id>`，上限 {@link SELECTION_ROW_CAP}）。
 *
 * 为什么不是服务端一次取完：`/api/catalog/rows` 的 `where` 只支持**单条件**，
 * 没有 id 集合筛选。逐行取的是服务端的真实行，不拿预览的 400 条属性冒充全表；
 * footer 会明写「选择集视图，不是服务端分页」。
 */
export function useSelectionRows(
  path: string | null,
  idField: string | null,
  numericId: boolean,
  ids: string[]
): SelectionRowsState {
  const chosen = ids.slice(0, SELECTION_ROW_CAP)
  const queries = useQueries({
    queries: chosen.map((id) => ({
      queryKey: ['catalog-row-one', path, idField, id],
      queryFn: () =>
        getCatalogRows(path as string, {
          where: `${idField} = ${numericId ? id : `'${id.replace(/'/g, "''")}'`}`,
          limit: 1,
        }),
      enabled: Boolean(path && idField),
      staleTime: 30_000,
      retry: false,
    })),
  })

  const rows = chosen.flatMap((_, index) => queries[index]?.data?.rows ?? [])
  const error = queries.find((query) => query.error)?.error as Error | undefined
  return {
    rows,
    shown: chosen.length,
    loading: chosen.length > 0 && queries.some((query) => query.isLoading),
    error,
    refetch: () => {
      for (const query of queries) void query.refetch()
    },
  }
}

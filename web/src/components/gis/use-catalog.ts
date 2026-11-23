import { useEffect, useRef } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'

import {
  listDatasets,
  triggerCatalogScan,
  type CatalogScanState,
  type CatalogScanTrigger,
  type DatasetItem,
} from '@/lib/api'

/** 目录树要完整：`/api/catalog/datasets` 单页上限 200，真实工作区有 379 条。 */
const PAGE_SIZE = 200
const MAX_PAGES = 10

/** 空闲时的轮询间隔（同时是「放入新文件 → 自动出现」的最坏延迟）。 */
const IDLE_POLL_MS = 10_000
/** 扫描进行中的轮询间隔：只为了让进度条和完成后的目录树快点出现。 */
const BUSY_POLL_MS = 1_500

export interface CatalogSnapshot {
  items: DatasetItem[]
  /** 超过 MAX_PAGES 时置 true（2000 条以上），UI 明说只显示前 N 条。 */
  truncated: boolean
}

async function listAllDatasets(): Promise<CatalogSnapshot> {
  const items: DatasetItem[] = []
  let offset = 0
  for (let page = 0; page < MAX_PAGES; page += 1) {
    const data = await listDatasets({ limit: PAGE_SIZE, offset })
    items.push(...data.items)
    if (!data.has_more || data.items.length === 0) {
      return { items, truncated: false }
    }
    offset += data.items.length
  }
  return { items, truncated: true }
}

/**
 * catalog 全量快照。分页拉取（而不是只取前 200）是因为目录树要能展开到
 * 任意一个 .gdb / 要素类；`queryKey` 与旧的 `['datasets']` 区分开，避免
 * 复用只取 200 条的那份缓存。
 *
 * 顺带管扫描（DAIR-346）：每轮轮询打一次 `POST /api/catalog/scan`——
 * 服务端先比文件系统指纹，没变化就直接回（一次目录遍历，几十毫秒），
 * 变了才真的开扫，所以「把数据放进 input/ → 目录树自动出现」不需要点刷新、
 * 也不需要先跑任务。扫完一轮（`finished_at` 变了）再重拉一次目录。
 */
export function useCatalog() {
  const queryClient = useQueryClient()

  const scan = useQuery<CatalogScanTrigger>({
    queryKey: ['catalog', 'scan'],
    // 包一层：TanStack 会把 query context 当第一个实参传进去，直接传函数会变成 force=true。
    queryFn: () => triggerCatalogScan(),
    refetchInterval: (query) =>
      query.state.data?.scan.running ? BUSY_POLL_MS : IDLE_POLL_MS,
    // 后台标签页里的定时器会被浏览器降频，回到页面时立刻补一次。
    refetchOnWindowFocus: true,
  })

  const datasets = useQuery({
    queryKey: ['datasets', 'all'],
    queryFn: listAllDatasets,
    staleTime: 30_000,
  })

  const scanState: CatalogScanState | null = scan.data?.scan ?? null
  const finishedAt = scanState?.finished_at ?? null
  const seenFinishedAt = useRef<string | null>(null)

  useEffect(() => {
    if (!finishedAt || finishedAt === seenFinishedAt.current) return
    seenFinishedAt.current = finishedAt
    void queryClient.invalidateQueries({ queryKey: ['datasets', 'all'] })
  }, [finishedAt, queryClient])

  return {
    ...datasets,
    scan: scanState,
    /** 后端正在扫：UI 显示进度，而不是「input 是空的」。 */
    isScanning: scanState?.running ?? false,
    /** 整轮扫描失败的原文（此时目录树仍是旧数据）。 */
    scanError: scanState?.error ?? '',
    /**
     * 扫描失败、但目录里还挂着上一次的结果（DAIR-369）：界面必须把它标成
     * 「已过期」，不能「报错但下面还是上次的结果」当成最新目录用。
     */
    stale: Boolean(scanState?.error) && (datasets.data?.items.length ?? 0) > 0,
    /** 刷新按钮：先让后端扫（没变化会立刻返回；上次失败也强制重扫一次），再把目录重拉一遍。 */
    refresh: async () => {
      try {
        await triggerCatalogScan(true)
      } finally {
        await datasets.refetch()
      }
    },
  }
}

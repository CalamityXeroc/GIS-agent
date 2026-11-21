import { useQuery } from '@tanstack/react-query'

import { listDatasets, type DatasetItem } from '@/lib/api'

/** 目录树要完整：`/api/catalog/datasets` 单页上限 200，真实工作区有 379 条。 */
const PAGE_SIZE = 200
const MAX_PAGES = 10

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
 */
export function useCatalog() {
  return useQuery({
    queryKey: ['datasets', 'all'],
    queryFn: listAllDatasets,
    staleTime: 30_000,
  })
}

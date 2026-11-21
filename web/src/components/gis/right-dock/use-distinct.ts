import { useQuery } from '@tanstack/react-query'

import { getDistinct, type DistinctResult } from '@/lib/api'

export interface DistinctOptions {
  /** 传了就按分位数算断点（分级色彩）；不传只取唯一值清单。 */
  classes?: number
  limit?: number
}

/**
 * 字段唯一值（`/api/catalog/distinct`）——唯一值/分级符号化的**唯一**数据源。
 *
 * 不许用 `profile.fields[].samples`（只有 3 条样本）去猜类目：那正是设计稿里
 * 标为「待后端补」的第 3 条缺口，DAIR-328 已补上真实接口。
 */
export function useDistinct(
  path: string | null,
  field: string | null,
  options: DistinctOptions = {}
) {
  const classes = options.classes
  const quantile = classes !== undefined
  return useQuery<DistinctResult>({
    queryKey: ['distinct', path, field, quantile ? `q${classes}` : 'plain'],
    queryFn: () =>
      getDistinct(path as string, field as string, {
        limit: options.limit ?? 200,
        ...(quantile ? { breaks: 'quantile' as const, classes } : {}),
      }),
    enabled: Boolean(path && field),
    staleTime: 5 * 60_000,
    retry: false,
  })
}

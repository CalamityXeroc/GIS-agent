import { useQuery } from '@tanstack/react-query'

import { getDatasetProfile, type DatasetProfile } from '@/lib/api'

/**
 * 数据集画像（`/api/catalog/datasets/{path}/profile`）。
 *
 * 右 dock 的「字段」「元数据」两个 tab 共用同一份：字段清单、栅格逐波段、
 * 范围/要素数都来自它。catalog 只读、不打开数据文件，所以「元数据」在
 * ArcGIS 内核不可用时照样能看。
 */
export function useDatasetProfile(path: string | null) {
  return useQuery<DatasetProfile>({
    queryKey: ['dataset-profile', path],
    queryFn: () => getDatasetProfile(path as string),
    enabled: Boolean(path),
    staleTime: 5 * 60_000,
    retry: false,
  })
}

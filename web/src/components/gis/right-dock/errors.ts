import { ApiError } from '@/lib/api'

/** 接口失败 → 一句人话（标题 + 原因），右 dock 的「待补」态直接用。 */
export interface ApiIssue {
  title: string
  reason: string
}

const CODE_TITLE: Record<string, string> = {
  kernel_unavailable: '需要 ArcGIS 内核',
  preview_timeout: '接口超时',
  dataset_unreadable: '数据集读不出来',
  bad_field: '字段不在白名单',
  not_raster: '不是栅格数据集',
  unsupported: '该格式不支持',
  bad_where: '筛选条件不合法',
}

export function describeApiError(error: Error): ApiIssue {
  if (error instanceof ApiError) {
    const detail = error.detail
    if (detail && typeof detail === 'object') {
      const record = detail as Record<string, unknown>
      const code = typeof record.code === 'string' ? record.code : ''
      const reason =
        (typeof record.reason === 'string' ? record.reason : '') || error.message
      const suggestion = typeof record.suggestion === 'string' ? record.suggestion : ''
      return {
        title: CODE_TITLE[code] ?? `接口返回 ${error.status}`,
        reason: suggestion ? `${reason}（${suggestion}）` : reason,
      }
    }
    if (error.status === 503) return { title: '需要 ArcGIS 内核', reason: error.message }
    if (error.status === 422) return { title: '接口拒绝了这次请求', reason: error.message }
    if (error.status === 404) return { title: 'catalog 里没有这条记录', reason: error.message }
    return { title: `接口返回 ${error.status}`, reason: error.message }
  }
  return { title: '接口不可用', reason: error.message }
}

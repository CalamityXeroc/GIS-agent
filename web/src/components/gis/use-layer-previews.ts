import { useQueries } from '@tanstack/react-query'
import { useMemo } from 'react'

import { getLayerPreview, type LayerPreview } from '@/lib/api'

import { isPreviewable, styleFields, useGisStore, type GisLayer } from './store'

export interface LayerPreviewState {
  preview?: LayerPreview
  error?: Error
  loading: boolean
  refetch: () => void
}

/**
 * 每个已上图层一份预览（≤400 要素）。地图与属性表共用同一个 queryKey，
 * React Query 只发一次请求；这里只做「path → 状态」的投影。
 *
 * 栅格/表格没有几何预览（后端 422「仅支持矢量」），不请求、不伪造，
 * 由调用方按 kind 显示「待补」占位。
 *
 * 符号化选了字段时（唯一值/分级）把字段名一并传给预览接口——否则
 * features.properties 是空的，地图上的 match/step 表达式只能全走兜底色
 * （DAIR-330）。字段进 queryKey：换字段就是另一份带属性的预览。
 *
 * 属性表学到要素标识字段（OID）后也加进请求：地图点选/框选要靠它把要素
 * 对回表格行，否则 `queryRenderedFeatures` 拿回来的 properties 是空的。
 */
export function useLayerPreviews(layers: GisLayer[]): Record<string, LayerPreviewState> {
  const idFields = useGisStore((state) => state.idFields)
  const queries = useQueries({
    queries: layers.map((layer) => {
      const fields = [idFields[layer.path], ...styleFields(layer.style)]
        .filter((name): name is string => Boolean(name))
        .filter((name, index, all) => all.indexOf(name) === index)
        .slice(0, 3)
      return {
        queryKey: ['layer-preview', layer.path, fields.join(',')],
        queryFn: () => getLayerPreview(layer.path, 400, fields),
        enabled: isPreviewable(layer.item),
        staleTime: 5 * 60_000,
        retry: false,
      }
    }),
  })

  // useQueries 每帧都返回新数组，直接拿 queries 当依赖等于每次渲染都重建返回值：
  // 拖 dock 分隔条时 setDock 每个 pointermove 触发一次渲染，地图的图层同步 effect
  // 就会跟着全量重跑（chartPalette + setPaintProperty + moveLayer）。这里只按
  // 「会改变返回值内容的量」算签名（图层身份 + 查询状态），签名没变就复用同一对象。
  const revision = layers
    .map((layer, index) => {
      const query = queries[index]
      return [
        layer.path,
        layer.kind,
        query?.status,
        query?.fetchStatus,
        query?.dataUpdatedAt,
        query?.errorUpdatedAt,
      ].join('/')
    })
    .join('|')

  return useMemo(() => {
    const result: Record<string, LayerPreviewState> = {}
    layers.forEach((layer, index) => {
      const query = queries[index]
      if (!query) return
      result[layer.path] = {
        preview: query.data,
        error: (query.error as Error | null) ?? undefined,
        loading: isPreviewable(layer.item) && query.isLoading,
        refetch: () => void query.refetch(),
      }
    })
    return result
  }, [revision])
}

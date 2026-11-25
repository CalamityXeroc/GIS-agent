import { useQueries } from '@tanstack/react-query'
import { useMemo, useRef } from 'react'

import { getLayerPreview, type LayerPreview } from '@/lib/api'

import { isPreviewable, styleFields, useGisStore, type GisLayer } from './store'
import { bboxKey } from './viewport'

export interface LayerPreviewState {
  preview?: LayerPreview
  error?: Error
  /** 还没有任何几何可画（首次加载）。地图此时才显示「正在读取」占位。 */
  loading: boolean
  /** 已经有几何在图上，正在换新视口的数据：保持画面不动，只标一下「更新中」。 */
  refreshing: boolean
  refetch: () => void
}

/**
 * 每个已上图层一份预览，**按地图视口取**（DAIR-386）。
 *
 * 以前是 `getLayerPreview(path, 400)`：不管地图看哪儿都只读前 400 条，13 万条的
 * 道路图层因此只画得出 0.3%，看上去像「线断了」。现在带上当前视口（store 里
 * 由地图 `moveend` 写入、按 zoom 量化过），服务端只读视口内的要素，并在响应里
 * 说明「视口内多少 / 全量多少 / 覆盖率」。
 *
 * 视口还没就绪（`bbox: null`）时不发请求：退回全量前 N 条同样是大图层只画几根
 * 碎线的路子，不如等地图 `load`。
 *
 * **换视口时不空屏**（DAIR-393）：视口一换，React Query 的新 key 还没有数据，
 * `query.data` 会短暂变 `undefined`。以前地图就据此把图层摘掉，每次缩放/平移
 * 都空白 2~3 s，看起来像「每次都要重新加载」。现在按图层记住最后一份成功预览
 * （`lastPreviews`），新数据回来之前继续画它，`loading` 只在「真没东西可画」时为
 * true，另给一个 `refreshing` 让图例标「更新中」。
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
  //: 每个图层最后一份成功预览：换视口时新 key 没数据，地图继续画这一份。
  const lastPreviews = useRef<Record<string, LayerPreview>>({})
  const viewport = useGisStore((state) => state.viewport)
  const viewKey = bboxKey(viewport.bbox)
  const queries = useQueries({
    queries: layers.map((layer) => {
      const fields = [idFields[layer.path], ...styleFields(layer.style)]
        .filter((name): name is string => Boolean(name))
        .filter((name, index, all) => all.indexOf(name) === index)
        .slice(0, 3)
      return {
        queryKey: ['layer-preview', layer.path, fields.join(','), viewKey, viewport.zoom],
        queryFn: () =>
          getLayerPreview(layer.path, {
            fields,
            bbox: viewport.bbox,
            zoom: viewport.zoom,
          }),
        enabled: isPreviewable(layer.item) && viewport.bbox !== null,
        // 视口 key 换成新的那一瞬间不能没有数据（地图会空一下）：下面用
        // `lastPreviews` 兜底比 `keepPreviousData` 更稳——它对 useQueries 的
        // 每个查询都生效，也不会因为 key 变化把旧值丢掉。
        staleTime: 5 * 60_000,
        // 回看上一个视口（来回缩放/平移）不该再打服务端：一个大 `.gdb` 层一次
        // 视口 2~3 s，只有缓存住才「退回去就是瞬间」（DAIR-393）。
        gcTime: 30 * 60_000,
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
      const fresh = query.data
      if (fresh) lastPreviews.current[layer.path] = fresh
      const preview = fresh ?? lastPreviews.current[layer.path]
      result[layer.path] = {
        preview,
        error: (query.error as Error | null) ?? undefined,
        // 只有「一份几何都没有」才算加载中：否则缩放着色会闪成占位。
        loading:
          isPreviewable(layer.item) && !preview && (query.isPending || query.isFetching),
        refreshing: Boolean(preview) && query.isFetching,
        refetch: () => void query.refetch(),
      }
    })
    return result
  }, [revision])
}

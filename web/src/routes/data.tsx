import { GisWorkbench } from '@/components/gis/workbench'

/**
 * 「数据资产」= GIS 工作台（DAIR-325 设计稿 v1）。
 *
 * 旧的卡片网格 + 右侧 Sheet 已整体替换：目录树/图层栈/地图/属性表/状态栏
 * 都在 `components/gis/`，这里只留路由出口，避免页面级状态再长回来。
 */
export function DataAssetsPage() {
  return <GisWorkbench />
}

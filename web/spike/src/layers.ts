/** spike 用的三份真实样例：直接取自本项目 workspace 的图层预览缓存（见 README 数据来源）。 */
export interface SpikeLayer {
  id: string
  name: string
  file: string
  /** 源数据 wkid；null = 无 .prj（按 4326 理解）。 */
  wkid: number | null
  crsLabel: string
  /** 后端 catalog 记的真实要素数（可能上百万）。 */
  featureCount: number | null
  /** 预览接口实际返回的要素数（受 400 / 5000 上限与抽稀影响）。 */
  returnedFeatures: number
  /** 首屏默认底图：本地快照 / 在线 Carto / 无。 */
  defaultBasemap: BasemapMode
  defaultZoom: number
  story: string
}

export type BasemapMode = 'offline' | 'online' | 'none'

export const LAYERS: SpikeLayer[] = [
  {
    id: 'medical',
    name: '医疗服务圈（北京）',
    file: 'data/medical_buffers_4548.geojson',
    wkid: 4548,
    crsLabel: 'CGCS2000_3_Degree_GK_CM_117E（EPSG:4548，单位米）',
    featureCount: 301,
    returnedFeatures: 301,
    defaultBasemap: 'offline',
    defaultZoom: 13.2,
    story:
      '现状 canvas 把 447114 / 4425294 这种「米」当经纬度画，只能看出节点之间的相对形状；spike 用 proj4 转 4326 后落在北京（116.38E, 39.91N），叠底图、比例尺、可缩放。',
  },
  {
    id: 'population',
    name: '人口网格 1000m（110 万点）',
    file: 'data/population_points_4508.geojson',
    wkid: 4508,
    crsLabel: 'CGCS2000_GK_CM_111E（EPSG:4508，单位米）',
    featureCount: 1103285,
    returnedFeatures: 5000,
    defaultBasemap: 'online',
    defaultZoom: 5.2,
    story:
      '110 万个点被后端抽稀到 5000 个（0.45%），且 bbox 跨 838×1314 km —— 缩放/分块/瓦片是这类数据的硬需求，canvas 单遍重绘扛不住。',
  },
  {
    id: 'china',
    name: '中国_省（42 要素，WGS84）',
    file: 'data/china_provinces_4326.geojson',
    wkid: null,
    crsLabel: '无 .prj，按 WGS84 经纬度理解',
    featureCount: null,
    returnedFeatures: 42,
    defaultBasemap: 'online',
    defaultZoom: 3.4,
    story:
      '经纬度直接当平面画 = 等经纬投影：中纬度东西向被拉伸 1/cosφ（40°N 约 1.31 倍，53°N 约 1.68 倍）。MapLibre 的 Web Mercator 保形，形状比例与 GIS 软件一致。',
  },
]
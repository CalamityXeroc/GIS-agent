# Web 端 GIS 级空间数据可视化：选型与落地（DAIR-298）

- 分支：`dair-298-webgis-spike`（spike 与文档在此分支，主干源码未改）
- 验证代码：`web/spike/gis-map/`
- 数据：北京 17 条绿道 290 段、587.8 km（本机 `greenway-vue/public/数据`，WGS84/CRS84）
- 截图：`web/spike/gis-map/screenshots/01–04`

---

## 0. 结论（先看这里）

1. **问题不是「少一个 npm 包」**。现有 `map-canvas.tsx` 把经纬度当平面坐标做等比缩放，等价于 40°N 上的 Plate Carrée：**东西向被拉长约 1.31 倍**；再叠加无底图/无缩放/仅 256 px/最多 400 要素/5 MB 上限，所以只能看出「基本空间关系」。
2. **你已经有两套成型项目可以直接抄架构，不必从零选型**：
   - `GeoViz / MapBox-Visualization`：Mapbox GL + Draw + Cesium，交互和图层样式面板的成熟形态；
   - `Greenway lite`（Vue 3 + OpenLayers 8.2）：纯开源、可离线部署，`MapManager` / `LayerManager` 分层清晰；
   - `Greenway full`（Vue 3 + GeoScene 4.33）：图层注册表、瓦片代理、2D/3D、Sketch/测量、夜间主题，功能最全但绑商业许可证。
3. **同时发现一个真实精度 bug**：`北京市.json`（AMap/DataV，GCJ-02）与 WGS84 边界相比，**边界中位偏移 459 m、重心 755 m**；`Greenway lite` 用高德栅格底图叠 WGS84 绿道数据，同样存在这个漂移。Web 端「图不准」不只是投影，基准（datum）也在出错。
4. **推荐方案：MapLibre GL JS + 自托管 PMTiles 矢量底图**（BSD-3、无 key、WebGL、真实 Web Mercator、矢量细节、可离线）。OpenLayers 作为 OGC/WMS/打印/编辑场景的备选或共存方案；deck.gl 只做大数据叠加层；Cesium 只在确有 3D 需求时启用；Kepler.gl 只做分析工具。
5. Spike 已跑通四件事并留图：**正确投影 + 真实比例 + 可缩放/平移 + 本地底图**，以及非 4326 重投影、GCJ-02 纠偏。

---

## 1. 已成型项目审计（回应「不只看包，也看成熟项目」）

### 1.1 GeoViz — `MapBox-Visualization`（Vue 3 + Mapbox GL 3.18 + Mapbox Draw + Cesium）

| 可复用点 | 位置 | 解决当前哪个缺陷 |
| --- | --- | --- |
| Map 初始化、中心/缩放/pitch、`preserveDrawingBuffer` 导出 | `src/views/Maps.vue:925` 起 | 现状没有地图实例，无法缩放/导出 |
| 底图切换：`map.setStyle(VECTOR/SATELLITE)` | `Maps.vue` 的 `switchBaseMap` | 现状无底图 |
| GeoJSON 导入后 `draw.add()` + `map.fitBounds(bbox,{padding:50})` | `Maps.vue` 文件导入段 | 现状只按 bbox 拉伸，无真实范围语义 |
| 点/线/面三套 source + filter + paint（绘制结果渲染） | `Maps.vue:821–880` | 现状只画一种蓝色线，无语义分层 |
| 样式面板（颜色/半径/线宽实时 `setPaintProperty`） | `Maps.vue` 工具箱 | 未来图层符号化 |
| Cesium 3D 沙盘路由 | `src/views/CesiumSandbox.vue` | 未来 3D |

**坑**：`mapbox-gl` 3.x 需要 Mapbox access token，且是 Mapbox 商业条款（不是 OSS）；底图样式依赖 Mapbox 在线服务，内网/离线不可用。**MapLibre 是 API 兼容的开源分支，可直接把 `mapboxgl.Map` 换成 `maplibregl.Map`，复用这套图层/交互代码。**

### 1.2 Greenway lite — `_lite_repo/greenway-vue`（Vue 3.4 + OpenLayers 8.2，纯 OSS）

| 可复用点 | 位置 | 说明 |
| --- | --- | --- |
| `MapManager`：单例、`View`（EPSG:3857）、`fromLonLat`、`fitExtent`、图层注册表、销毁 | `src/core/MapManager.js` | 最贴近「把 MapCanvas 换成真地图」的架构模板 |
| `LayerManager`：XYZ 瓦片、`GeoJSON({featureProjection:'EPSG:3857'})`、`VectorImage` 性能优化、样式工厂 | `src/core/LayerManager.js` | 图层种类/性能策略可直接映射到 MapLibre 的 source/layer |
| 图层增删/可见性/`fitExtent` 联动 | `src/components/MapViewer.vue` | 现有图层页的交互骨架 |
| 绘制/测量/导出 PNG/自定义 GeoJSON 上传 | `src/components/MapToolbar.vue` | 后续编辑能力 |

**坑（实测）**：

- 底图用高德栅格 XYZ（GCJ-02），业务 GeoJSON 是 WGS84，代码里没有做基准转换 → 在北京约 **460–750 m** 偏移；这不是「投影没开」，而是 datum 混用。
- 栅格底图强依赖外网（高德瓦片）；内网/离线直接空白，也没有矢量细节。
- 单文件瓦片/PMTiles/矢量瓦片均未使用，大图层仍是整份 GeoJSON 塞给浏览器。

### 1.3 Greenway full — `greenproject/greenway-vue`（Vue 3.4 + GeoScene 4.33 + OpenLayers 8）

| 可复用点 | 位置 | 说明 |
| --- | --- | --- |
| `LayerManager` 的 `baseLayers` / `overlayLayers` 注册表、`WebTileLayer` 代理 URL、`GeoJSONLayer`/`FeatureLayer`/`MapImageLayer` | `src/core/LayerManager.js` | 「底图/业务图层」分离 + 内网瓦片代理的正确做法 |
| 几何类型自动识别 + renderer 映射（点/线/面样式配置） | `LayerManager._detectGeometryType` / `_createSimpleRenderer` | 图层样式配置 → MapLibre paint/layout 的映射表 |
| `MapManager` 2D `MapView` / 3D `SceneView`、`goTo`/`fitExtent`、相机动画 | `src/core/MapManager.js` | 未来 2D/3D 统一入口 |
| Sketch/测量、夜间模式（CSS filter）| `MapToolbarGeoScene.vue`、`MapManager.setBaseTheme` | 编辑与主题 |
| GCJ-02 ↔ WGS84 工具 | `src/utils/coordinateTransform.js` | 只用于 GPS 点，尚未覆盖底图/业务图层 |

**坑**：`@geoscene/core` 是商业许可证（易智瑞），不能作为开源分发依赖；瓦片代理仍指向高德在线源；2D/3D 绑定单一厂商。

### 1.4 GreenProject（`.aprx` / `.gdb`）+ `greenway-backend`（PostGIS）

- `GreenProject.gdb` 是真实数据源，现有 GIS_CLI 已用 arcpy 读 gdb；`greenway-backend` 证明 PostGIS `ST_DWithin` / GeoJSON 存储 / 空间查询的可行路径。
- `greenway-backend/src/routes/gaode.js` 代理高德 API（GCJ-02）——**又一个基准混用入口**。
- 复用方式：arcpy 只做「数据 → 4326 GeoJSON / 矢量瓦片」的离线预处理，不进入 Web 进程。

### 1.5 成熟项目审计结论

| 项目 | 最佳复用 | 不能照搬 |
| --- | --- | --- |
| GeoViz | 图层/source/filter/样式面板、导入 + fitBounds、Cesium 3D | Mapbox token/许可、在线底图 |
| Greenway lite | MapManager/LayerManager 分层、纯 OSS 栈、绘制/测量/导出 | 高德栅格底图 + 未纠偏的 datum |
| Greenway full | 图层注册表、瓦片代理、2D/3D、Sketch、夜间主题 | GeoScene 商业依赖、在线瓦片 |
| GreenProject | 真实数据、arcpy/PostGIS 数据管线 | 把 arcpy 塞进 Web 请求 |

---

## 2. GitHub 上值得读的成熟开源项目（2026-11-17 元数据）

### 2.1 全栈 GIS 平台（学架构与契约，不整体引入）

| 项目 | 栈 | 许可 | ★ | 可复用 / 价值 | 不适合 |
| --- | --- | --- | --- | --- | --- |
| **GeoNode** | Django + GeoServer + OpenLayers/MapStore | GPL-3.0 | 1,737 | 图层目录、元数据、权限、WMS/WMTS/矢量瓦片发布契约 | 体量大、GPL、需要 GeoServer |
| **MapStore2** | React + Redux + OpenLayers/Leaflet/Cesium | BSD-3 | 671 | 插件化地图工作台、图层目录、打印、仪表盘/故事地图 | Java 后端 GeoStore、学习曲线陡 |
| **QWC2** | React + OpenLayers | BSD-2 | 385 | 由 QGIS 工程生成配置、图例/图层树、QGIS Server 打印 | 依赖 QGIS Server 体系 |
| **Lizmap** | QGIS Server + OpenLayers | MPL-2.0 | 338 | QGIS 工程直接发布为 Web GIS 的完整流程 | 同样绑定 QGIS Server |

### 2.2 应用/编辑器（学交互与信息架构）

| 项目 | 栈 | 许可 | ★ | 可复用 |
| --- | --- | --- | --- | --- |
| **Kepler.gl** | React + deck.gl + MapLibre | MIT | 12,096 | 图层/样式面板、颜色比例尺、过滤器、tooltip、GPU 大数据 |
| **Placemark** | MapLibre + React | MIT | 1,477 | GeoJSON/KML/CSV 导入导出、编辑体验、属性表 |
| **TerriaJS** | React + Cesium/Leaflet | Apache-2.0 | 1,381 | 数据目录（catalog）、时间轴、2D/3D 切换 |
| **NASA Worldview** | React + OpenLayers | NASA 许可 | 1,971 | 栅格时间序列、图层/时间轴 UI（未来遥感图层） |
| **GeoViz（本项目）** | Vue + Mapbox | 自有 | 2 | 已上线形态，见 1.1 |

### 2.3 渲染底座与数据工具（直接采用）

| 项目 | 作用 | 许可 | ★ | 在本项目中的位置 |
| --- | --- | --- | --- | --- |
| **MapLibre GL JS** | WebGL 矢量地图、Web Mercator、离线 style | BSD-3 | 12,105 | **主渲染库**，替换 MapCanvas |
| **OpenLayers** | OGC/WMS/WMTS/打印/编辑、任意投影 | BSD-2 | — | 备选/共存；lite 版已验证 |
| **deck.gl** | GPU 大数据图层（点云/轨迹/热力） | MIT | 14,764 | 叠加层，不替代底图 |
| **CesiumJS** | 3D 地球/地形/3D Tiles | Apache-2.0 | 15,959 | 仅 3D 需求（已有 Cesium 沙盘） |
| **PMTiles** | 单文件瓦片归档 + HTTP Range | 见仓库 | 3,220 | 离线底图/业务瓦片分发 |
| **tippecanoe** | GeoJSON → 矢量瓦片 | BSD-2 | 1,637 | 离线切片管线 |
| **Maputnik** | MapLibre style 可视化编辑器 | MIT | 2,673 | 底图样式制作 |
| **mapbox-gl-draw** | 绘制/编辑插件 | ISC | 1,094 | 编辑工具（MapLibre 兼容） |
| **maplibre-geoman** | MapLibre 绘制/编辑 | MIT | 129 | 更现代的编辑替代 |
| **Mapbox GL JS** | 商业矢量地图 | 商业条款 | — | **不选**（token/许可/离线差） |

---

## 3. 渲染库选型对比

| 维度 | MapLibre GL JS | OpenLayers | Leaflet | deck.gl | CesiumJS | Kepler.gl |
| --- | --- | --- | --- | --- | --- | --- |
| 真实投影/比例 | Web Mercator 原生，按纬度校正 | 任意投影（proj4），3857 默认 | Web Mercator（插件支持 proj4） | 跟随底图 | 3D 地球/地形 | 跟随底图 |
| 矢量底图/样式 | **强**（style spec，矢量瓦片） | 中（VectorTile 样式 API 较底层） | 弱（需插件） | 不提供底图 | 3D Tiles | 中 |
| 离线/内网底图 | **强**（PMTiles/自托管 style） | 强（XYZ/WMS/WMTS） | 强（XYZ/WMS） | 依赖底图 | 强（本地 3D Tiles） | 依赖底图 |
| 大要素性能 | GeoJSON 万级；矢量瓦片百万级 | Canvas/VectorImage，万级；瓦片百万级 | 万级 | **百万+（GPU）** | 3D 场景 | 百万级（deck.gl） |
| React 19 集成 | 直接用 `useEffect`（同 MapCanvas 风格）或 react-map-gl/maplibre-react-components | 直接用 `useEffect` | 直接 | React 原生 | resium/直接 | React 原生 |
| 编辑/测量 | maplibre-geoman/mapbox-gl-draw | **内置 Draw/Modify/Measure** | 插件 | 无 | 有 | 无 |
| 打印/OGC | 需插件；WMS 可作 raster source | **最强**（WMS/WMTS/WFS/打印） | WMS 插件 | 无 | 无 | 无 |
| 3D | 无（pitch/terrain 有限） | 无 | 无 | 无 | **强** | 无 |
| 许可 | BSD-3 | BSD-2 | BSD-2 | MIT | Apache-2.0 | MIT |
| 结论 | **主选** | **备选/共存** | 不选（能力不足） | 叠加层 | 仅 3D | 仅分析 |

---

## 4. 推荐方案与理由

**主方案：MapLibre GL JS + 自托管 PMTiles（底图与业务瓦片）+ GeoJSON（小图层/预览）**

理由（按本项目约束排序）：

1. **真实地理形态**：GeoJSON 顶点保持 4326，MapLibre 按 Web Mercator 校正，形状/距离/面积比例正确；自带 zoom/pan/rotate、比例尺、pitch。
2. **离线/内网**：style 与瓦片都可自托管，PMTiles 是单文件 + HTTP Range，不需要瓦片服务器；无 key、无外网。
3. **细节**：矢量瓦片可带道路/水系/建筑/POI；GeoJSON 只是「预览/小图层」通道。
4. **React 19 成本低**：不用引入大框架；在 `useEffect` 里初始化 map，props 变化时 `source.setData()` / `fitBounds`，与现有 `MapCanvas` 的 imperative 风格一致；也可用 `react-map-gl`（maplibre 版）。
5. **许可干净**：BSD-3；Mapbox GL 商业条款不适合作为内部分发依赖。
6. **与已有资产配合**：GeoViz 的图层/交互代码可平移（Mapbox → MapLibre API 兼容）；Greenway 的 `LayerManager` 思路可直接映射到 MapLibre source/layer 注册表。

**OpenLayers 作为备选**的条件：必须重度使用 OGC（WMS/WMTS/WFS）、服务端打印、或团队要保持与 lite 版同一套代码；此时用 OL 8 + `ol/layer/WebGLTile` + `VectorTile`，并复用 `MapManager`/`LayerManager`。

**其它**：deck.gl 仅用于 >10 万的轨迹/点位叠加；Cesium 仅当明确要 3D 地形/飞行；Kepler.gl 作为分析/验收工具，不做产品壳。

---

## 5. 本项目落地路径（React 19 + Vite + Tailwind 4 + zustand + TanStack + Python/arcpy）

### 5.1 前端替换点

当前调用（`web/src/routes/data.tsx:455`）：

```tsx
<MapCanvas
  features={preview.data.geojson.features}
  bbox={preview.data.bbox}
  className="h-64 w-full"
/>
```

目标组件（保持 props 兼容，先在预览页替换）：

```tsx
<MapView
  features={preview.data.geojson.features}
  bbox={preview.data.bbox}          // 仅作为初始相机
  crs={preview.data.crs}            // 新增：EPSG:4326
  className="h-64 w-full"
/>
```

内部：

- `new maplibregl.Map({ style: localStyle, center, zoom })`；
- `addSource('preview', { type:'geojson', data: fc })`；
- 三类 layer（fill/line/circle）按 geometry 过滤，颜色读 CSS 变量（保持深浅主题）；
- `map.fitBounds(bbox, { padding: 24 })`；`NavigationControl` + `ScaleControl`；
- `ResizeObserver` → `map.resize()`；
- 图层 > 5 万要素或 > 5 MB：不塞 GeoJSON，改请求 `/api/layers/{id}/tiles/{z}/{x}/{y}.mvt`；
- 大图层预览失败时保留 `MapCanvas` 作为 fallback（报告缩略图/无 WebGL 环境）。

保留 `MapCanvas` 用于：报表 HTML 缩略图、无 WebGL 的老内网浏览器、`oversized` 图层的范围框。

### 5.2 后端与数据

现状（`src/gis_cli/web/preview_api.py`）：

- `KNOWN_CRS = {4326, 3857}`，其它 CRS **照原样返回**，只给 `crs_warning = "未做投影变换"`；
- 默认 400 要素、硬上限 5 MB、`MAX_MAX_FEATURES = 5000`；
- `.geojson/.json` 直读，`.gdb/.shp` 走 arcpy `SearchCursor` + 抽稀。

改造（按优先级）：

1. **CRS 归一**：预览/瓦片接口统一输出 EPSG:4326。优先 `pyproj`（纯 Python，快）；gdb 走 `arcpy.management.Project` 或读取时带 `SpatialReference`。响应新增 `crs: "EPSG:4326"`；`crs_warning` 仅用于「坐标系未知」。
2. **预览契约**：`bbox` + `wkid` + `returned_features` + `truncated` 保留；新增 `geometry_source`（`arcpy`/`geojson`/`postgis`）便于排查。
3. **大图层**：> 5 万要素或 > 5 MB 时走矢量瓦片（PostGIS `ST_AsMVT`，或离线 tippecanoe 生成的 PMTiles）；服务端按 bbox/zoom 查询、`ST_SimplifyPreserveTopology` 简化；点图层可用 MapLibre cluster 或服务端网格聚合。
4. **数据管线**：`arcpy → 4326 GeoJSON → tippecanoe → PMTiles` 作为离线作业（GIS_CLI 里跑），Web 进程只读 PMTiles；避免 `import arcpy` 进 FastAPI。
5. **底图**：见 5.3。

### 5.3 离线底图三方案

| 方案 | 做法 | 体积/成本 | 适用 |
| --- | --- | --- | --- |
| **A. 自托管 PMTiles（推荐）** | OSM/天地图/自有数据 → tippecanoe → `.pmtiles`，MapLibre `pmtiles` 协议 + style | 北京 z0–14 约 0.2–0.5 GB；全国数 GB | 内网、无 key、可版本化 |
| B. 现有 GIS 服务 | GeoServer/ArcGIS Server 发 WMS/WMTS，MapLibre raster source 或 OL | 无新增数据，依赖已有服务 | 已有 OGC 基础设施 |
| C. 无底图降级 | 本地矢量面 + 经纬网 + 比例尺（spike 当前形态） | 几 KB | 断网兜底、报表缩略图 |

注意：底图若用高德/腾讯瓦片（GCJ-02），业务数据必须先转 GCJ-02；推荐直接用 WGS84 底图 + WGS84 数据，避免基准混用。

### 5.4 字体/图标/主题

- MapLibre 的 symbol 文本需要 glyph PBF；内网需自托管 `glyphs/`（fontnik 生成），或只用 `localIdeographFontFamily` 渲染中文、避免英文标签；
- 图标用本地 sprite（Maputnik 可导出）；
- 主题：维持 CSS 变量，初始化时读 `getComputedStyle` 生成/切换两套 style JSON（对应现有 `.dark`）。

### 5.5 arcpy 边界

- 不在 Web 进程 import arcpy；预览读取 gdb 已是「内核子进程 + 超时」模式（`preview_api` 的 10 s 预算），保持不变；
- 重投影/切片作为 GIS_CLI 离线任务，产物为 GeoJSON/PMTiles；
- 单实例约束下不要并发跑 arcpy（本轮也未派子代理跑 ArcGIS）。

---

## 6. Spike 实现与自检

### 6.1 内容

`web/spike/gis-map/`（独立 Vite 静态页，不改主干）：

- `index.html`：左=现状算法复刻；右=MapLibre + 本地矢量底图 + 经纬网 + 比例尺 + hover 真实长度；
- `crs.html`：EPSG:4548 原样（诊断卡）vs proj4 重投影（地图）；
- `datum.html`：GCJ-02 未纠偏 vs 纠偏后叠加；
- `scripts/prepare_data.py`：真实数据合并 + 球面长度 + pyproj 派生 4548；
- `scripts/shoot.mjs`：puppeteer-core + 系统 Chrome 截图。

### 6.2 实测数字

| 项 | 结果 |
| --- | --- |
| 数据 | 290 段 / 17 条绿道 / 587.8 km（最长中心城滨水 171.15 km） |
| 现状横向拉伸 | 40°N：1° 经度 ≈ 85.3 km，1° 纬度 ≈ 111.1 km → **×1.30** |
| GCJ-02 漂移 | 边界顶点中位 **459 m**；重心 **755 m**；单点基准偏移 555 m |
| 4548 原样坐标 | X 364,417–542,802 m；Y 4,367,896–4,547,151 m → 合法经纬度外，**0 要素可渲染** |
| 重投影后 bbox | 115.414–117.506°E / 39.443–41.059°N（正确落回北京） |
| 底图 | 本地 GeoJSON（北京面/边界），**0 个外部瓦片请求 / 0 key** |
| 交互 | 滚轮缩放、拖拽平移、比例尺随纬度变化、悬停显示真实 km |

### 6.3 截图

- `screenshots/01-overview.png`：同一 bbox，左（蓝、横向拉长）/右（多色、真实比例、比例尺 10 km、经纬网）。
- `screenshots/02-focus.png`：右侧缩放到 z11.25（2 km 比例尺），290 段细节可分辨；左侧仍是挤在一起的全局图。
- `screenshots/03-crs.png`：4548 原样 0 要素 vs 重投影后正确边界。
- `screenshots/04-datum.png`：未纠偏红线与蓝色 WGS84 边界明显错位；纠偏后重合。

### 6.4 已验证 / 未验证

已验证：Web Mercator 比例、缩放/平移、本地底图、比例尺、CRS 重投影、GCJ 纠偏、真实球面长度。

未验证（生产项）：PMTiles 切片与 Range 服务、矢量瓦片性能、字体 glyph、编辑/打印、移动端手势、老浏览器 WebGL 降级。spike 的底图只有市域面，不含路网/水系。

---

## 7. 风险与工作量

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| 底图体积/许可 | 全国 PMTiles 数 GB；OSM 衍生素材需署名/合规 | 只切项目范围、按需 z 级；法务确认；或接现有 GIS 服务 |
| 字体 glyph | 中文标签缺字/方块 | 自托管 glyphs + `localIdeographFontFamily` |
| WebGL 老内网机 | 地图不显示 | 保留 canvas fallback；检测 `webgl2` |
| PMTiles Range/代理 | 反向代理不支持 Range 会整包下载 | 配置 `Accept-Ranges`；或 Martin/pg_tileserv |
| arcpy 阻塞 | Web 请求超时 | 离线切片；预览保持子进程 + 10 s 超时 |
| 基准混用 | 再次出现 500 m 漂移 | 统一 4326；接口带 CRS；CI 加 datum 断言 |
| 交互回归 | 替换后图层页打不开 | 先并存（feature flag），保留 MapCanvas |

工作量（1 人，熟悉前端 + GIS）：

| 阶段 | 内容 | 人日 |
| --- | --- | --- |
| P0 | MapView 组件（MapLibre/本地 style/缩放/比例尺/主题）+ 预览页接入 | 2–3 |
| P1 | 后端 CRS 归一 + 预览契约 + 4548/4527 测试 | 1–2 |
| P2 | PMTiles 底图与业务瓦片管线（tippecanoe + 静态 Range） | 2–4 |
| P3 | 图层树/图例/弹窗/属性、大图层降级 | 2–3 |
| P4 | 绘制/测量/导出（maplibre-geoman） | 2–3 |
| P5 | 性能/兼容/回归测试 | 2–3 |
| 合计 | MVP（P0–P3） | **8–12 人日** |
| | 含编辑/打印/3D 的完整版 | 15–20 人日 |

---

## 8. 下一步（实现单范围建议）

**建议另开实现单：`实现：MapLibre 替换 MapCanvas（只读预览 MVP）`**

范围（严格限定）：

1. 新增 `web/src/components/layers/map-view.tsx`，props 兼容 `features/bbox/className`，新增 `crs`；
2. `data.tsx` 预览区切到 `MapView`，保留 `MapCanvas` 作为 `oversized`/无 WebGL fallback；
3. `preview_api` 统一输出 4326 + `crs` 字段；加 4548 测试（不引入 arcpy 依赖）；
4. style 用本地 JSON（先不含 glyphs 文本），底图用本地矢量面 + 经纬网；不接外网；
5. 验收：同一 bbox 截图对比；z10–14 缩放；比例尺存在；0 外网请求；`npm run build` + 现有 web 测试通过；不修改主干其它页面。

P2 的 PMTiles/路网底图另开单（依赖底图数据源与合规确认）。

### 待确认（影响选型落地，需用户回答）

1. **部署环境**：纯内网/离线，还是可访问内网 GIS 服务（GeoServer/ArcGIS Server）？底图数据许可允许自托管 OSM 衍生瓦片吗？
2. **能力优先级**：只读预览优先，还是必须同时有绘制/测量/打印？编辑是否要落到 PostGIS？
3. **数据规模与 CRS**：catalog 里最大图层多少要素？CAD/规划数据是否大量 4548/4527/地方坐标系？
4. **3D**：现有 Cesium 沙盘是否要并入同一页面，还是保持独立？

---

## 9. 本轮自检

- 未使用 `sleep` 轮询（截图用单次阻塞调用；本地服务仅用于截图，结束即杀）。
- `repair` 次数：2（MapLibre `line-width` 表达式非法；EPSG:4548 中央经线写错，已按 pyproj 校正）。
- 最长命令：`npm install` 约 9 s；截图 4 张约 40 s；无超时中断。
- 未跑全盘搜索；所有路径限定在 `/e/BST_lab` 与仓库内。
- 未并发跑 arcpy，未改主干源码。
# 制图规则与自动版式（cartography）

本项目的制图不再是"写死坐标 + 默认样式"，而是分三层：

```
facts（采集数据事实） → design（规则引擎出 LayoutSpec） → render（按 spec 落地 ArcGIS）
                            ↑                                    ↓
                            └──── qc（版面体检）←── 不合格则微调重渲 ───┘
```

- `src/gis_cli/cartography/facts.py`：采集范围长宽比、要素数、字段分布、**九宫格占用度**
- `src/gis_cli/cartography/design.py`：**纯 Python 规则引擎**（可用假 facts 单测）
- `src/gis_cli/cartography/apply.py`：把 spec 渲染成地图 + 可编辑 `.aprx`
- `src/gis_cli/cartography/qc.py`：出图后量测版面（图名居中、图例不压数据、比例尺整数刻度…）
- `src/gis_cli/cartography/profiles.py` + `config/cartography_profiles.json`：**风格档位**（用户可改）
- `src/gis_cli/cartography/styles.py` + `config/cartography_styles.json`：ArcGIS 样式项目录

对外入口：`gis_cli.cartography.design_only()`（只出设计说明）与 `create_map()`（一条龙出图）。

## 一、规则表（引擎自动决定，附理由）

| 项目 | 规则 | 依据 |
|---|---|---|
| 纸张/朝向 | 数据近方形→A4 竖版；宽扁→横版；面积/匹配度接近时优先更小的纸；屏幕档→16:9 | 图例与图名需要纵向空间；参考图一律 A4 竖版 |
| 图名 | `{区域}{尺度}{主题}{图种}` 自动拼装并去重；字号按"一行放得下"反算并夹在 12–20pt；>22 字折两行；**顶部居中** | 参考图：图名是最醒目的元素 |
| 图例 | **单一符号不放图例**；多图层符号清单**不加标题**；分类用类别名、分级默认用**简洁语义词**（少/较少/中/较多/多）；可选 `range`（自动取整）/`both` | 参考图五种图例形态 |
| 图例位置 | 按九宫格占用度放**最空的角**；四角都有墨且与中心同量级（铺满整幅）→ 移到**图框外**保证地图主体可见 | 用户明确"能看见图即可" |
| 比例尺 | 按地图比例尺反算，取 m×10^k（m∈1,2,4,5）中最接近图框宽 20–28% 的值；刻度必须整数；单位自动（<10 km 用米、否则千米） | 参考图：0 / 1,000 / 2,000 / 4,000 米 |
| 指北针 | 区域全图用**八芒罗盘玫瑰**（`ArcGIS 指北针 13`）；投影坐标系才有意义；尺寸为页宽 4–6% | 参考图 17 张全部为罗盘玫瑰 |
| 配色 | 按主题关键词选连续色带（人口/热力→YlOrRd、绿地→Greens、密度→Purples…）；分类用定性色板，重点要素高饱和 | 参考图：普通=灰、优化=浅蓝、标杆=绿 |
| 标注 | 要素数 > 阈值 或密度过高 → 关闭字段标注 | 参考图 2341 个社区不标注 |
| 地理坐标系 | 不画米制比例尺（无意义），并在决策里说明 | 制图惯例 |

每个决定都写进 `spec["decisions"]`（`key/value/why/source`），并随 `design_note` 记入 trace 与项目日志。

## 二、风格档位

| 档位 | 场景 | 要点 |
|---|---|---|
| `competition_standard`（默认） | 打印/竞赛 | A4 竖版、宋体 20pt 图名、图例右下无边框、比例尺左下黑白交替米制、罗盘玫瑰 |
| `screen_report` | 屏幕汇报 | 16:9 横版、字号放大、图例右上 |
| `dense_raster` | 满覆盖栅格 | 图例强制外置、数据边距收紧 |

改默认值：编辑 `config/cartography_profiles.json`（**只写要改的字段**，其余继承内置默认）。

## 三、覆盖方式（Agent / 用户都能干预）

出图配方（`graduated_colors_map` / `category_map`）支持：

```jsonc
{
  "input_path": "…/community_service", "field_name": "ELDER_COMM", "output_path": "…/map.jpg",
  "theme": "老年人口", "region": "郑州市", "scale": "社区尺度",   // 图名素材
  "purpose": "分布", "medium": "print_A4", "orientation": "auto",
  "legend_labels": "semantic",                                     // semantic / range / both
  "category_colors": "1:#1F77B4;2:#2CA02C",                        // 值→颜色（分类图）
  "category_names": "1:水域;2:林地;5:耕地;7:建筑区;8:裸地;11:牧场", // 值→显示名（图例用名称而非编码）
  "style_profile": "competition_standard",
  "class_bounds": "0,50,100,200,400",                              // 显式分级边界（多图统一图例）
  "overrides": {"legend.outside": true, "title.height_pt": 18}      // 点路径强制
}
```

> **统一图例（多期图可对比）**：`class_bounds` 传的是**分级边界**（n 个边界 → n−1 级），
> 例如 `"0,50,100,200,400"` 得到 4 级：0–50 / 50–100 / 100–200 / 200–400。
> 传了它就用 Manual 分级，`class_count` 被忽略；每张图的实际分级会写入 spec 的
> `renderer.applied_bounds`，用断言 `shared_legend_bounds` 可校验多张图是否真的共用同一套图例。
> **栅格与矢量两条路径都支持**（实测：4 期核密度这类栅格成果图必须走栅格分类渲染）。

> `category_names` 是给**编码存储**的类别用的：竞赛与行业规范都要求图例显示规范名称
> （如土地覆盖 1=水域 2=林地 5=耕地 7=建筑区 8=裸地 9=雪/冰 11=牧场），而不是 1/2/5/7。
> 它同时作用于工程内图例（CIM label）与导出图上的叠加图例。

Agent 侧工具：`map_design`（只出设计说明与 spec，先"读设计"再出图）、`check_map_project`（核验工程渲染器/配色/四要素）。

## 三点五、迁移图：箭头与年份标注（叠加层）

多期分布中心的迁移图需要"有方向性 + 标年份"，走叠加层（PIL）而不是 CIM 线符号箭头
（后者在 CIM 里极难调），接口：

```python
intent["overlay"] = {
    "arrows": [{"from": [x, y], "to": [x, y], "label": "1992"}],   # 数据坐标
    "labels": [{"at": [x, y], "text": "1982"}],                    # 数据坐标
    "arrow_color": "#B2182B", "label_size_pt": 9,
}
```

要点：

- 数据坐标 → 像素的换算用**渲染后实测**的地图框范围（`spec["render_frame_extent"]`，
  由 `map_frame.camera.getExtent()` 读回），不能用设计值——相机会按图框纵横比微调，
  用设计值会让箭头整体偏移。
- 换算纯函数 `image_overlay.map_to_px()` 可离线单测（含 y 翻转与 mm→px）。
- 年份标注**二选一**：给箭头带 `label` 或给中心点带 `labels`，两个都传会出现重复年份。
- 迁移图的图例应说明箭头含义（否则读者不知道红线代表什么——识图质检实测会指出这一点）。

## 四、ArcGIS 制图 API 踩坑（实测，重要）

0. **分级（classBreaks）只能走 CIM，且栅格/矢量挂的位置不同**：
   - 矢量：CIM 在 ``layer.getDefinition("V3").renderer.breaks``（**不是** `classBreaks`）；
     `classificationMethod` 也必须在 CIM 上设（元素 API 上赋值会被忽略，仍是 StandardDeviation）。
   - 栅格：栅格 symbology **没有** `updateRenderer`，必须整只替换 `definition.colorizer`
     为 `cim.CIMRasterClassifyColorizer`（其 `classBreaks[i].upperBound` 才是分级），
     顺带可给每个 break 设 `color`（CIMRGBColor）实现确定配色。
   - 读回验证要按同样路径读（栅格读 `colorizer.classBreaks`、矢量读 `renderer.breaks`），
     否则会误判成“没生效”。
   - `_hex_to_rgb()` 返回 `[r,g,b,alpha]` 四元组，别 unpack 成三个。
   - 栅格分类渲染**不要**给单波段栅格设 `colorizer.field`（没有 `Value` 字段，设了图面空白）；
     需要指定波段时用 `renderer.raster_field` 显式表示。
   - 分类 colorizer 必须逐级给颜色（没色带名时也会兜底 YlOrRd），否则出图看不到栅格。
   - `MapFrame` **没有** `getExtent()`：范围在 `camera` 上（`map_frame.camera.getExtent()`）。

1. **图名/文本字号**：`layout` 没有 `createTextElement`，只能用 CIM；字号属性是 `CIMTextSymbol.height`（`fontSize` 会被静默忽略）。
   段落文本要用 `CIMParagraphTextGraphic` + `CIMParagraphTextSymbol`；**配 `CIMTextSymbol` 会被忽略**——所以本实现用单行 `CIMTextGraphic` + `CIMTextSymbol`，多行拆成多个元素。
2. **添加顺序**：CIM 文本元素要在地图框**之后**添加（先加会被 `createMapFrame` 重置成默认样式）。
3. **图廓线**：用 `CIMPolygonGraphic`（描边）绘制；用 `CIMLineGraphic` 传多边形 shape 会得到 **NaN 几何**（画不出来）。
4. **图例**：`showHeading` / `showLayerName` / `labelSymbol` 都要经 `legend.getDefinition("V3")` 的 CIM 定义设置；直接对 `LegendItem` 赋值会静默无效。
5. **比例尺**：`units` 是**单位对象**（`CIMLinearUnit`，uwkid 9001=米 / 9036=千米），传字符串会被忽略；`division` 的数值单位随显示单位走。元素宽度决定条长，因此要按"计划条长"反算元素宽度。
6. **栅格图层**：`layer.symbology` **没有** `updateRenderer`（与矢量不同），只能"能设色带就设，否则沿用默认"，不能因此让整张图失败。
7. **样式名随安装语言变化**（本机中文：`公制黑白相间比例尺 1`、`ArcGIS 指北针 13`）→ 必须运行时枚举 + 关键字匹配 + 缺省回退，**不要静默取列表第一项**。
8. **布局边框不可持久（本环境实测）**：只用**导出时描边**这一条可靠路径。
   - `cim.CIMPolygonGraphic` / `CIMLineGraphic` 加进布局后几何恒为 **NaN**（元素报 x=nan），画不出来；
   - 文本元素（TEXT_ELEMENT）几何正确且能持久，但其 `CIMGraphicElement` **没有 frame/graphicFrame**，无法加边框；
   - 地图框自带边框（`CIMMapFrame.graphicFrame.borderSymbol`）改了在**保存/重开工程后会被元素样式覆盖**；
   - 所以：**地图框占满图廓**（框即外框），导出后用 `image_overlay.stroke_rect` 按规格描一层 0.35mm 细线；
     验收用**看图判据**（外层有线 + 内侧无第二条线）。若要在 Pro 工程里也有边框，可在布局里手动加一个矩形。
9. **图框边框**：地图框自带边框的符号在 `map_frame.getDefinition("V3").graphicFrame.borderSymbol`（不在 arcpy.mp 属性上）。
   - 置 `None` **只在当次会话有效**：保存/重开工程时 ArcGIS 会重新套用地图框样式，黑边又回来；
   - 改成**白色 CIMSymbolReference** 才能持久（重开后导出图框边缘暗像素为 0）→ 本实现用白色边框，只保留外层图廓线；
   - 验收要用**看图的判据**（`qc` 按图框四条边是否有长直线 + 外层图廓是否存在判断），不能只看 CIM 字段。
9. **图例绘制位置不可控**：ArcGIS 图例的锚点导致实际绘制位置不受 `elementPositionX/Y` 可靠控制
   （会跑到页面底部留残留）→ 本实现**隐藏工程内图例**，导出图图例由叠加层绘制；工程里图例元素仍在（可一键显示）。
10. **别在 ArcGIS Pro 运行时出图**：Pro 与我们的 GP 调用会互相干扰（Pro 弹严重错误甚至崩溃）。`render()` 会检测并提醒。

## 五、版面体检（可当断言用）

- 脚本：`python scripts/inspect_map_layout.py <工程.aprx> [图.jpg]`
- 断言：`{"type": "map_layout_ok", "path": "…aprx", "image": "…jpg", "min_font_pt": 7}`
- 检查项：图名字号/是否居中/是否溢出、图例标签是否残留多位小数、图例是否压住数据、比例尺刻度是否整数、
  四要素是否互相压盖/越出图廓、图例是否完全在图框内（留边距）、**是否只有一层边框**、地图框是否真的画上了数据。

## 六、相关脚本

| 脚本 | 用途 |
|---|---|
| `scripts/probe_map_api.py` / `probe_map_api2.py` | 探明 Layout/Legend/ScaleBar/NorthArrow 可用属性 |
| `scripts/probe_style_sheet.py` | 生成指北针/比例尺样式样张（多模态挑选用） |
| `scripts/inspect_aprx_layouts.py` | 检查 aprx 里的布局与元素几何/字号（排查版式问题） |
| `scripts/verify_map_aprx.py` | 核验渲染器/类别配色/布局四要素/数据源 |
| `scripts/inspect_map_images.py` | 图片像素级质量检查 |
| `scripts/make_cartography_demo.py` | 5 种图种端到端示例（自动版式 + 体检） |

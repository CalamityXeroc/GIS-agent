# 民生模板库（P1-1 骨架）

一个模板 = **数据清单 + 配方链 + 断言集 + 后端要求**。模板不改配方、不写代码，
只负责把已有配方按顺序串起来，并把上一步的产物传给下一步。

- 目录：`templates/<id>/manifest.yaml`（仓库根下，一个模板一个目录）
- 加载器：`src/gis_cli/templates/`（`schema.py` 定义 manifest 与绑定，`store.py` 负责预检与执行）
- 工具：`list_templates`（看模板需要什么数据、支持哪些后端）、`run_template`（跑整条链，逐步留结果）

仓库里现在有八个模板：两个**玩具**模板（只验证机制，不含业务口径）+ 六个**业务**模板
（W4 第 2 / 3 批，口径与边界见下节）：

| 模板 | 链 | 为何这样定 |
|---|---|---|
| `demo_project_then_zonal` | `project_to_crs` → `zonal_statistics` | `backend: [arcpy]`：表格版分区统计两端输出格式不通用（arcpy 写 `.dbf`/`.gdb`，GDAL 只写 `.csv`/`.gpkg`），所以这条链只能声明 arcpy；GDAL-only 机器上预检就报「需 ArcGIS」 |
| `demo_project_then_buffer` | `project_to_crs` → `buffer_dissolve` | `backend: [arcpy, gdal]`：两步产物都是 `.shp`，两种执行器都能跑，用来看双后端链路 |
| `elderly_care_coverage` | 投影社区 → 投影设施 → `buffer_dissolve`（服务半径）→ `clip_layer`（裁到社区）→ `field_calculate`（覆盖面积）→ `summarize_stats`（按片段自带 `COMM_ID` 分组汇总）→ `attribute_join_table`（挂回社区）→ `zonal_statistics`（老年人口）→ `attribute_join_table`（挂接）→ `field_calculate`（覆盖率） | `backend: [arcpy]`：第 8 步是表格版分区统计（`.dbf` 不通用）；覆盖率用**面积比**而不是人口比，是因为覆盖面积只靠矢量裁切就能算，不必再跑一次栅格统计、也不必让两张表对齐键；覆盖片段**不再做空间再归属**（`clip_layer` 产物已带 `COMM_ID`，按属性聚合才是恒等的，见下节 DAIR-171） |
| `green_space_equity` | 投影社区 → 投影绿地 → `field_calculate`（绿地面积）→ `aggregate_to_zones`（聚合回社区）→ `zonal_statistics`（人口）→ `attribute_join_table`（挂接）→ `field_calculate`（人均绿地）→ `field_calculate`（缺口标记） | `backend: [arcpy]`：第 5 步同样是表格版分区统计；绿地面积走 `aggregate_to_zones`（矢量精确求和）而不是面转栅格，避免重采样把面积改掉；守恒断言用 `SRC_COUNT`（归属斑块数）而不是面积，因为面积会随投影与容差浮动 |
| `public_service_value_rank` | 建结果库 → 投影社区 → `feature_to_centroid` → `point_buffer_raster_sum`（500 米）→ `attribute_join_table`（挂回社区）→ `field_calculate`（服务价值）→ `rank_top_bottom_flag`（前后 N 名）→ 两次 `select_by_attribute`（两个名单） | `backend: [arcpy]`：转点/排名/属性筛选三个配方只有 arcpy；缓冲区用点缓冲**解析求和**而不是“缓冲区面 → 面转栅格”，几千个重叠缓冲区在面转栅格下会丢数（源任务里明确禁止） |
| `map_quality_eval` | 复制评价图层 → `field_calculate`（类别）→ `category_map`（唯一值 + JPG + .aprx） | `backend: [arcpy]`：唯一值渲染器、版式编排、`.aprx` 都要 arcpy.mp；入参必须是 `.gdb`/`.gpkg`（`BOTTOM_FLAG` 11 字符，`.shp` 会截成 `BOTTOM_FLA`） |
| `city_hotspot_aggregate` | `csv_points_spatial_join`（人口 CSV → 区县）→ `project_to_crs`（地市）→ `aggregate_to_zones`（汇总到地市）→ 三次 `hotspot_analysis`（2011/2012/2013） | `backend: [arcpy]`：XY 转点 + SpatialJoin 与 Getis-Ord Gi* 都只有 arcpy 实现 |
| `census_indicator_similarity` | `csv_points_spatial_join` → `project_to_crs` → `census_city_indicators`（人口加权）→ `similarity_ranking`（JSON）→ 三次 `graduated_colors_map`（三张专题图 + .aprx） | `backend: [arcpy]`：普查加权汇总、相似度排序、分级设色与 `.aprx` 四个配方都只有 arcpy |

## 首批业务模板（W4 第 2 步）

两个模板都只**串已有配方**（`src/gis_cli/recipes/builtin/`），没有为模板新增一行配方代码。
共同约定：

- 输入先统一投影到 `target_srid`（默认 4527，米制），距离与面积才有物理意义；
- **模板不重投影栅格**：人口/老年人口栅格必须已经是 `target_srid` 的坐标系与格网。
  重采样会改人口总量，这个决定不替用户做；坐标系不一致时 `zonal_statistics` 会明确报错；
- 社区面需带整型唯一 ID 字段（默认 `COMM_ID`）：老年版按它把覆盖片段聚回社区、又用它做分区统计与挂接；
  绿地版还要一个**唯一字符串字段**（`aggregate_to_zones` 按 `name`，没有就取第一个字符串字段，重名会合并），
  因为绿地块本身不带社区归属，只能靠空间归属。
- **产物字段名一律 ≤10 字符**：`.shp` 的属性表是 dBASE，字段名物理上限 10 字符，
  超了 arcpy 只会静默截断（或直接报错），而后续步骤与断言引用的都是 manifest 里那个
  **没截断**的长名字 → 链一定断在中间。所以给 `SUM_` 前缀留出位置（`COVA` → `SUM_COVA`）；
  `tests/test_templates.py` 有一条静态守卫用例扫 `templates/*/manifest.yaml`，超了就红。

### `elderly_care_coverage` —— 社区适老化评估

**口径**：社区边界 + 老年人口栅格 + 养老设施点位 → 每个社区的老年人口合计、
被设施服务半径覆盖的面积、覆盖率（覆盖面积 / 社区面积）。回答「这个社区的老人
有多少还够不着服务」。服务半径默认 800 米（步行约 10 分钟），可用 `service_radius` 改。

**产物字段**（`community_metrics.shp`）：`SUM_COVA`（覆盖面积合计，m²）、`COUNT_COVA`
（归属裁切片段数）、`SUM`（老年人口合计）、`COVER_RATE`（覆盖率 = `SUM_COVA` / 社区面积）。
`COVA` 是「覆盖面积」的缩写：字段名受 `.shp` 的 10 字符上限约束，写全 `SUM_COVER_AREA`
（14 字符）会被截断，第 10 步的 `CalculateField` 就找不到字段；`COUNT_COVA` 恰好占满 10 个字符。

**为何不把覆盖片段交回空间归属**（DAIR-171 的教训）：`covered_parts` 是社区面 `clip` 出来的，
每一块都带着自己的 `COMM_ID` 与 `NAME`，按属性分组求和是恒等的；而 `aggregate_to_zones`
（`match_option=HAVE_THEIR_CENTER_IN`）会把片段**再按中心点判一次**，真实社区面形状不规则时
整块片段会被算给邻居社区 —— 实测 34/2326 片段错配、26/2341 社区 `COVER_RATE>1`、最大 10.13
（含 1 个错配面积 25,829.71 m² 的整块），`max(COVER_RATE) ≤ 1.05` 这条断言 10/10 次全红。
换成属性聚合后实测 `max(COVER_RATE)=1.000002`、`sum(SUM_COVA)` 与片段面积合计逐位相等。

**断言**（6 条）：两条 `file_exists`（最终指标表 + 分区统计表）、
`COVER_RATE` 的 `max` ≤ 1.05 与 `min` ≥ 0（覆盖率是比值，理论上界就是 1；
5% 容差留给裁切边界与浮点）、**守恒**：各社区 `COUNT_COVA`（归属片段数）之和
必须**恰好等于** `covered_parts` 的要素数（漏算/重复计入都会红）、
分区统计表 `SUM` 合计落在 `[1, 1e8]`（空表/全 0 与量级错都拦）。

**不是做什么**：不评估设施质量、床位数、无障碍改造程度；不做 60+/65+ 的年龄结构细分
（老年人口栅格需按所选口径事先备好）；不产出地图与报告；不重投影栅格。

### `green_space_equity` —— 绿地公平性

**口径**：社区边界 + 绿地斑块 + 人口栅格 → 每个社区的绿地面积、人均绿地（m²/人）、
以及缺口标记（有人口但人均绿地低于 **9 m²/人** 的社区）。回答「哪些社区的绿地不够分」。
9 m²/人 是本模板选定的口径常数（不是强制国标值）：要换阈值就改 manifest 里
`green_gap` 那一步的 `expression`（判定的是 `GPC`），并同步改这一节 —— 模板层不支持
把参数插进表达式。

**产物字段**（`green_equity.shp`）：`SUM_GRA`（绿地面积合计，m²）、`SUM`（人口合计）、
`GPC`（人均绿地，m²/人）、`GREEN_GAP`（0/1 缺口标记）。`GRA` / `GPC` 同样是缩写：
`.shp` 字段名上限 10 字符，`SUM_GREEN_AREA`（14 字符）与 `GREEN_M2_PER_CAPITA`（19 字符）
都会被截断，第 7/8 步的 `CalculateField` 与人均绿地的断言就都读不到字段。

**断言**（6 条）：两条 `file_exists`；**守恒**：各社区 `SRC_COUNT`（归属绿地斑块数）
之和必须 ≤ 第 2 步的 `feature_count`（输入斑块总数）—— 每个斑块按中心点至多归属
一个社区，`SpatialJoin` 扇出/重复计数会超上界；人口 `SUM` 合计落在 `[1, 1e8]`；
人均绿地均值落在 `[0, 1000]`（量级护栏）；缺口标记均值落在 `[0, 1]`（0/1 标记的均值）。

**已知损失**（DAIR-171 实测，同一套真实数据）：34002 个斑块里有 353 个没落到任何社区 ——
其中 **329 个整块在社区镶嵌之外**（与任何社区都不相交，天然无法归属），**24 个与社区相交但
中心点落在社区外**，这 24 块合计 16,331 m²（占绿地总面积 0.18%）。这是「整块按中心点归属」
这个口径的固有损失；要按面积切分就先 `clip` 到社区面（那样 `SUM_GRA` 会变成切分后面积，口径变了）。

**不是做什么**：不做绿地可达性/服务半径覆盖（那是 `elderly_care_coverage` 那种缓冲区链）；
不给绿地分质量等级（公园/防护/生产绿地一视同仁）；不算基尼系数/洛伦兹曲线
（公平性只用「人均绿地下限标记」这一朴素口径）；不重投影栅格。

### 跑这两个模板前要准备什么

```python
run_template(
    template_id="green_space_equity",
    data={
        "communities": "input/communities.shp",   # 面 + COMM_ID（整型）+ 唯一字符串字段
        "green_spaces": "input/green.shp",        # 绿地斑块面
        "population_raster": "input/pop.tif",     # 已在 4527 格网上
    },
)
```

产物全部落在 `output/<模板id>/` 下（`run_template` 的 `output_dir` 可改），
每步的参数、产物路径与断言明细都在返回值的 `data.steps` 里。

## 第二批业务模板（W4 第 3 批）

四个模板都从**已有任务链反推**，步骤顺序照 `run_recipe` 的建议顺序，
不额外发明配方。它们全部是 `backend: [arcpy]` 且**如实声明**：不是懒省事，而是链上确实有
arcpy-only 的配方（转点、排名筛选、Gi*、`.aprx` 工程、普查加权、相似度排序）。
把 `backend` 写成 `[arcpy, gdal]` 会让 GDAL-only 机器跑到一半才炸——这正是本批要避免的。

| 模板 | 业务问题 |
|---|---|
| `public_service_value_rank` | 社区的绿地服务价值（服务人口 / 绿地面积）谁高谁低，前后 N 名是谁 |
| `map_quality_eval` | 评价结果直接能交付成图吗（分类设色 JPG + 可编辑工程） |
| `city_hotspot_aggregate` | 区县人口汇总到地市后，逐年人口热点/冷点在哪 |
| `census_indicator_similarity` | 普查三项指标的空间格局，哪个地市与参照城市最像 |

### `public_service_value_rank` —— 服务价值排序

**口径**：社区面（带 `GREEN_AREA` 绿地面积与 `ELDER_COMM` 老年人口）+ 老年人口栅格 →
社区中心点 `service_radius`（默认 500 米）内的老年人口 `SUM_VALUE`（圆形缓冲区**逐像元解析求和**，
缓冲区之间可以任意重叠）→ `SERVICE_VALUE = SUM_VALUE / GREEN_AREA`（无绿地留**空值**，不填 0）
→ 按服务价值排前后 `top_n`/`bottom_n`（默认各 500）→ 两个名单要素类。

- **为什么不用缓冲区面**：源任务明说不要「缓冲区 → 面转栅格」——一个像元只能归属一个分区，
  几千个互相重叠的 500 米缓冲区会成片丢数。`point_buffer_raster_sum` 是解析判定（像元中心到点的距离），
  支持重叠且是秒级。
- **为什么要有 `create_file_gdb` 这一步**：该配方的 arcpy 端只往**已存在**的文件地理数据库里建表
  （它拒绝 `.dbf` 与内存工作空间），所以第 1 步显式建库，而不是等它报错。
- **空值口径**：`SERVICE_VALUE` 为空 = 无绿地 = 最该整改。排名时 `bottom_include_null=true` 让空值优先占后 N 名，
  空值内部按 `ELDER_COMM` 降序（老年人口越多越迫切）——与源任务的 `BOT_SVC` 一致。
- **只做一维**：源任务还要按人均绿地指数（`PER_CAPITA`）排一次、取两个维度的交集；本模板只落服务价值这一维，
  两个名单是「服务价值前 N / 后 N」，不是竞赛口径的「同时满足两个维度」。要那半条就再加一条并列的排名链。
- **断言**（8 条）：缓冲区表行数落在 `[1, 社区数]`、服务人口合计 > 0、最终图层的 `SUM_VALUE`/`SERVICE_VALUE`/
  `TOP_FLAG`/`BOTTOM_FLAG` 都在、排名产物要素数与投影后社区数一致（`CopyFeatures` 守恒）、
  `TOP_FLAG`/`BOTTOM_FLAG` 的 0/1 之和落在 `[1, N]`（排序或空值策略写歪会红）、两个名单要素数落在 `[1, N]`。

### `map_quality_eval` —— 社区尺度评价图

**口径**：带 `TOP_FLAG`/`BOTTOM_FLAG` 的评价图层 → 合并成一个 `CATEGORY` 字段（标杆社区 / 需整改社区 / 其他，
标杆优先）→ 唯一值设色出图（标杆 `#1F77B4`、需整改 `#2CA02C`、其他 `#E8E8E8`）→ JPG + 同名可编辑 `.aprx`。

- **先复制再改**：`field_calculate` 是原地加字段，直接改调用方传进来的图层会在人家的数据上留 `CATEGORY`，
  重跑还会因为「字段已存在」直接失败。所以第 1 步用 `select_by_attribute`（`1 = 1`）复制一份。
- **入参容器有要求**：`BOTTOM_FLAG` 是 11 个字符，`.shp` 的 dBASE 字段名上限 10，会被截成 `BOTTOM_FLA`——
  入参和产物都必须放在 `.gdb` / `.gpkg` 里（`public_service_value_rank` 的产物本来就在 `.gdb` 内）。
- **断言**（6 条）：`CATEGORY` 字段存在、类别 distinct > 2（三类都在）、JPG 存在且非空白、`.aprx` 存在、
  `aprx_map_check`（唯一值渲染器 + 两类配色 + 图名/图例/比例尺/指北针）。
- **已知阻塞**：HEAD `b726cc1f` 上 `category_map` 配方的 `code_template` 缺一行 `raise`
  （commit `6c1e6157` 删掉两条「输入/字段不存在」守卫，`if not arcpy.Exists(input_path):` 变成空 if），
  渲染出来的代码必然 `IndentationError`，**任何**调用都跑不起来。配方不在本单的文件范围内，
  所以模板先按“修好这一行即可用”交付；验收时在内存里补回这一行做的对照实验 3/3 全绿（见下节）。

### `city_hotspot_aggregate` —— 人口汇总与热点探测

**口径**：只有区县中心点经纬度的逐年人口 CSV + 区县面 + 地市面 → CSV 转点并 `INTERSECT` 挂接到区县
（得到 `Y2011..Y2020`）→ 地市面投影到 `target_srid`（默认 4508）→ 按区县**中心点**归属（`HAVE_THEIR_CENTER_IN`）
把 `Y2011/Y2012/Y2013/Y2020` 求和回地市（`SUM_Y*` + `SRC_COUNT`）→ 2011/2012/2013 三年各做一次 Getis-Ord Gi*。

- 年份与字段名**写死在 manifest 里**（`stat_fields: "Y2011,Y2012,Y2013,Y2020"` + 三个热点步骤各绑一个 `SUM_Y*`）：
  模板层不支持「一步展开成三步」，而热点分析要逐字段调用；换年份要改 manifest，不是改参数。
- 区县面是地理坐标系也能直接用：`csv_points_spatial_join` 先把点投影到面图层的坐标系做连接，
  再把结果整体投影到 `target_srid`。
- **断言**（9 条）：`SUM_Y2011..SUM_Y2020` 与 `SRC_COUNT` 都在、坐标系等于 `$data.target_srid`、
  地市要素数不超过投影后的地市面数、`SRC_COUNT`/`SUM_Y2011` 合计 > 0、热点产物含 `GiZScore`/`GiPValue`/`Gi_Bin`、
  2011 与 2013 的 `Gi_Bin` distinct > 1（只有一个取值等于没识别出冷热点）、2012 热点要素数不超过地市数。

### `census_indicator_similarity` —— 普查指标汇总与相似城市

**口径**：区县逐年人口 CSV + 普查三项指标 CSV（老年占比 / 育龄女性占比 / 性别比）+ 区县面 + 地市面 →
先挂接人口 → `census_city_indicators` 按**人口加权**（区县指标 × 人口换人数，汇总后还原市级指标）→
`OLD_PCT`/`FERT_PCT`/`SEX_RATIO` + `SRC_COUNT` → `similarity_ranking`（三指标 z-score 后欧氏距离）
写 `similarity.json` → 三个指标各出一张分级设色图（JPG + 同名 `.aprx`）。

- 加权用的是 `Y2020`（写死在 manifest 与断言里）；换权重年份要同时改 `pop_field` 与 `SUM_Y2020` 那条断言。
- 参照城市默认 `郑州市`，可用 `reference_name` 改；相似度结果文件名固定 `similarity.json`（不跟参照城市名走）。
- **断言**（17 条）：四项指标字段都在、坐标系等于 `$data.target_srid`、`SRC_COUNT`/`SUM_Y2020` 合计 > 0、
  `OLD_PCT`/`FERT_PCT` 均值落在 `[0, 100]`（漏了 `/100` 或 `×100` 会红，实测 200% 会红）、
  `similarity.json` 存在且含 `most_similar`、三张图各自存在且非空白、三个 `.aprx` 存在。

### 四个模板的验收记录（DAIR-219）

真数据：`public_service_value_rank` 用 13 届真题 PM 那套（2341 个社区 / 34002 个绿地块 / `pop_density.tif`
→ 5 m、EPSG:4548 的老年人口栅格 `elder_fill`；绿地面积是 `green_land.shp` 与社区面求交后按 `COMM_ID` 求和，
老年人口是 pm_task1 链分摊出来的 `ELDER_COMM`）；另外三个用 13 届真题 B 上午那套
（385 个区县 / 49 个地市 / `population.CSV` / `population_census.csv`）。
每次冷启动 = 新内核（含 `import arcpy`）+ 新空输出目录，入口是 `CodeRunner → RecipeLibrary → TemplateLibrary.run`
（与工具 `run_template` 同一套对象，没有手写 arcpy 复刻链路）。

| 模板 | 跑通 | 断言 | 单次墙钟（含内核启动） |
|---|---|---|---|
| `public_service_value_rank` | 3/3 | 8/8 × 3 | 53.5 / 43.4 / 42.4 s |
| `map_quality_eval` | 3/3（注） | 6/6 × 3 | 24.7 / 27.0 / 24.6 s |
| `city_hotspot_aggregate` | 3/3 | 9/9 × 3 | 58.2 / 71.5 / 75.6 s |
| `census_indicator_similarity` | 3/3 | 17/17 × 3 | 53.5 / 48.2 / 54.3 s |

后端声明不是保守写法而是唯一可行写法：逐步骤对照仓库里真实的 47 个配方，这 4 条链分别有
6 / 3 / 4 / 6 个**只有 arcpy 实现**的步骤（转点、排名、属性筛选、唯一值制图、XY 转点+SpatialJoin、
Getis-Ord Gi*、普查加权、相似度排序、`.aprx`），声明 `[arcpy, gdal]` 会让 GDAL-only 机器跑到中途才炸。

（注）`map_quality_eval` 的 3 次冷启动是在**配方库副本**里补回 `category_map` 缺失的那一行 `raise` 之后跑的；
仓库里的配方文件没有改，补丁前后唯一差别就是那一行守卫。同一份输入用 HEAD 上的配方直接跑是 0/1：
`第 3 步 (map/category_map) 失败: IndentationError: expected an indented block after 'if' statement`
—— 这一行不补，模板在任何机器上都跑不完第 3 步（配方不在本单文件范围内，见「已知边界」）。

GDAL-only 环境（执行器声明 `backends={"gdal"}`、配方库是仓库里真实的 47 个配方）下 4 个模板**全部**在预检阶段被拦下：
`BackendUnavailable` + 「需 ArcGIS（arcpy）」+ 0 次配方调用 + 0 个文件（4/4）。

`census_indicator_similarity` 的相似度结果与源任务一致：与郑州市最相似的是**长沙市**（距离 1.17，
其后是武汉市 1.77、洛阳市 2.25）。

## manifest 字段

```yaml
id: demo_project_then_zonal        # 必填，小写蛇形，与目录名一致
name: 演示：先投影，再分区统计       # 必填
description: 这个模板解决什么问题   # 建议写清「口径」与「不是做什么」
version: "1.0"
backend: [arcpy, gdal]             # 模板可用的执行后端，缺省 [arcpy]

data:                              # 数据清单：run_template 的 data 参数按这些名字传
  zones:                           # 输入名，小写蛇形（$data.zones）
    type: vector                   # vector/raster/table/file/string/integer/number/boolean
    required: true
    purpose: 分区面（行政区/社区边界）
  population_raster:
    type: raster
    required: true
    purpose: 数值栅格（如人口栅格）
  target_srid:
    type: integer
    required: false
    default: 4527                  # 非必需输入可以给默认值

pipeline:                          # 配方链，按顺序执行
  - id: project                    # 步骤 id：下游用 $steps.project... 引用
    recipe_id: project_to_crs      # 已有配方（见 list_recipes）
    params:
      input_path: $data.zones
      output_path: $out/zones_projected.shp
      target_srid: $data.target_srid
  - id: zonal
    recipe_id: zonal_statistics
    params:
      zone_path: $steps.project.output   # ← 上游产物
      raster_path: $data.population_raster
      output_table: $out/zonal.dbf
      statistics: SUM

assertions:                        # 模板级断言：全部步骤跑完后统一判定
  - type: file_exists
    path: $out/zones_projected.shp
  - type: file_exists
    path: $steps.zonal.output
```

`type` 只是给人/模型看的期望类型（`.gdb` 要素类没有扩展名，按扩展名强校验会误杀），
执行时**不做格式强校验**；`required: true` 且没有 `default` 的输入必须由调用方传入。

`assertions` 里除了 `file_exists`，**至少再写一条数值/守恒断言**
（`field_stats_between` / `field_sum_gt` / `feature_count_between` 等）—— 只写
`file_exists` 等于没验产物。断言的 `path` / `value` 同样走绑定，上下界可以直接绑
上游结果键，例如 `value: [1, $steps.project_green.feature_count]` 就是
「各社区归属斑块数之和不得超过输入斑块总数」这条守恒。

## 参数绑定语法（核心契约）

参数值是一个**整串字符串**时，按下面三种写法解析：

| 写法 | 含义 | 例 |
|---|---|---|
| `$data.<输入名>` | 调用方传入的输入（或 manifest 里的 `default`） | `$data.zones` |
| `$steps.<步骤id>.<结果键>` | **前序**步骤的结果；`output` 是最常用的产物路径 | `$steps.project.output` |
| `$out` / `$out/<相对路径>` | 本次运行的输出目录 | `$out/zonal.dbf` |

规则：

- 只认整串引用，不做字符串内插（`"$out/${x}.shp"` 不解析，原样传下去）。
- 参数里的嵌套 dict/list 会逐项递归解析；绑定也可以用在断言的 `path` / `value` 上。
- 以 `$` 开头的字符串一律当绑定解析。写错前缀（`$dat.zones`）会**当场报错**并列出
  可用写法，不会当作普通路径传给配方。
- `$steps.*` 只能引用**前面**的步骤；引用不存在的步骤、或引用后面的步骤都会报人话错误。
- 步骤结果里没有要的键时，报错会列出该步骤实际可用的键
  （例如 `output`、`feature_count`），不会静默返回 `None`。
- `$out` 下只允许相对路径，`..` 或绝对路径会被拒绝。

`run_template` 的 `output_dir` 缺省是 `workspace/output/<模板id>`；`$out/...` 全部落在它下面。

## 执行与失败行为

`run_template` 先做**预检**，预检不过时一行配方代码都不执行：

1. 模板后端：`backend` 与执行器声明的能力没有交集 → 报
   `配方声明后端 [...]，本机可用后端：...。需 ArcGIS（arcpy）。`（`error.type = BackendUnavailable`）。
2. 每一步的配方存在，且配方自己的 `backend` 在本机能跑（含“声明了 gdal 但没有 GDAL 版代码”）。
   **后一步缺后端也会在第一步之前被拦住** —— 不允许跑到一半才炸。
3. 数据清单：缺必需输入 / 传了不认识的输入名，都直接报出来该传什么。
4. 绑定：`$data.*` 有对应输入、`$steps.*` 指向前序步骤、必填配方参数都绑了、
   没有给配方传它不认识的参数名（防止 `statistic` 这种笔误被静默忽略）。

预检通过后逐步执行并记录每步：

```jsonc
{
  "steps": [
    {
      "index": 1, "step": "project", "recipe_id": "project_to_crs",
      "params": {"input_path": "...", "output_path": "...", "target_srid": 4527},
      "ok": true, "backend": "arcpy",
      "output": ".../output/demo/zones_projected.shp",
      "artifacts": ["..."],
      "assertions": [ /* 配方自带断言的结果 */ ]
    }
  ],
  "assertions": [ /* 模板级断言的结果 */ ],
  "backend": "arcpy",
  "output_dir": ".../output/demo"
}
```

- 某一步失败 → 停链，`data.steps` 里保留已执行/失败步骤的记录；不继续跑后续步骤。
- 模板级断言失败 → 整次 `run_template` 失败，但步骤记录仍完整（用来区分“链没跑通”
  和“链跑通了但产物不合规”）。
- 模板断言使用模板自己选定的后端实现（与配方断言“谁执行谁判定”的规则一致）。

## 已知边界（本骨架不做）

- **六个业务模板全部是 `backend: [arcpy]`**：原因逐条写在上面每条模板的「为何这样定」里——
  表格版分区统计两端格式不通用（`.dbf` vs `.csv`）、转点/排名/筛选/Gi*/普查加权/相似度排序/`.aprx`
  这些配方只有 arcpy 实现。不是漏声明：GDAL 端的补期待 P0-1b 配方预算里再排。
- **`map_quality_eval` 依赖的 `category_map` 在 HEAD 上编译不过**（`if not arcpy.Exists(input_path):`
  缺 body），要先补回一行 `raise` 才能真跑；模板本身已通过 3 次冷启动（内存补丁对照实验）。
- 不含报告模板与出图规范；报告/合规闸门在后续单独立项。
- 重活步骤的单独超时还没做逐步骤配置：`run_template` 的 `timeout_seconds` 对每一步生效。
- **一条链只能有一个 `backend` 声明**：终端产物格式不通用时（如 `.dbf` 表）只能声明
  arcpy 或换格式，暂不支持「同一参数按后端给不同值」；真实模板里若确实需要，
  再把 `params_by_backend` 加进 schema。
- 打包安装后仓库根的 `templates/` 不会随包分发（当前按源码目录运行）；分发方案仍未定，
  首批真实模板已落地，但分发要等 P0-1b 的 GDAL 配方齐了一起做。

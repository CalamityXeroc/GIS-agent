# P0 上手与交付：让没用过 GIS 的人也能跑通一条链

这篇讲四件事，都是「第一次用」时会卡住的地方：

1. **`doctor`** —— 环境不对时 10 秒说清缺什么
2. **数据接入** —— 手上只有 Excel / CSV / KML，怎么变成能分析的图层
3. **分析计划卡** —— 动手前先让你看懂要做什么，确认了才跑
4. **成果包** —— 跑完给一份能直接发出去的单文件报告（HTML / PDF）

---

## 1. `gis-agent doctor`

```bash
gis-agent doctor          # 常规 7 项，约 10~20 秒
gis-agent doctor --deep   # 额外启动一次持久内核，验证「真能跑一行代码」
```

七项检查：

| 检查 | 判据 | 失败时给什么 |
|---|---|---|
| `pro_installed` | 找到 ArcGIS Pro 的 python（环境变量 → 注册表 → 默认路径三路） | 列出已试过的路径，提示用 `ARCGIS_PRO_PYTHON` 指定 |
| `arcpy_importable` | 用那个 python 真的 `import arcpy` 并读到版本 | stderr 前 3 行 |
| `pro_version` | ≥ 3.2 | 建议升级 |
| `deps` | 9 个核心包都能导入 | 拼一条可直接复制的 `pip install …` |
| `path_spaces` | 工作目录与 Pro 环境路径无空格 | 指出具体是哪条路径 |
| `path_nonascii` | 路径纯 ASCII | 同上 |
| `kernel_start` | 持久内核能起来（仅 `--deep`） | 内核 stderr |

**刻意不做的事**：

- ❌ 不检查 ArcGIS Pro 许可证 —— 检测既无用又冒犯，而且用户完全可能用的是非正版
- ❌ 不引导下载、不教安装、不给购买链接
- ❌ 不自动改环境（只给命令，不动手）

换一台机器/换一个 Pro 安装时，先跑它，比对着报错猜快得多。

---

## 2. 数据接入

手上通常不是 `.gdb`，而是 Excel、CSV、KML。接入层负责把它们变成**带坐标系、能进 catalog、能直接分析**的图层。

### 2.1 CSV / Excel → 点图层

```python
from gis_cli.ingest import convert_csv_points

result = convert_csv_points("input/poi.xlsx")     # 也可以给 .csv
print(result.geojson_path)   # input/poi_points.geojson
```

会自动做：

| 处理 | 说明 |
|---|---|
| 认经纬度列 | 支持 `经度/纬度`、`lon/lat`、`x/y`、`longitude/latitude` 等常见写法，也可 `lon_field=` / `lat_field=` 显式指定 |
| 跳过坏坐标 | 坐标为空、非数字、越界的行跳过，**跳了几行记在 `skipped_rows` 里**，不静默 |
| 编码保留 | `007`、`012` 这种 0 开头纯数字串按**文本**处理（行政区代码、邮编不能被改成 7 / 12） |
| 数值列 | `人口`、`面积` 这类落成 `Double`，GeoJSON 里写 JSON 数字，`JSONToFeatures` 建出来就是数值字段 |
| 空值 | 数字列的空单元格写 `null`（不是空串，否则要素类类型推断会崩） |

Excel 多工作表用 `sheet=` 指定。

### 2.2 KML / GPKG

```python
from gis_cli.ingest import describe_kml, kml_to_geojson
print(describe_kml("input/track.kml"))        # 先看有什么，再转
kml_to_geojson("input/track.kml", "input/track.geojson")
```

GPKG 由数据目录扫描器直接识别，不需要转换。

> ⚠️ **已知限制**：KML 的 `ExtendedData`（自定义属性）目前只保留 `Name` / `Description`，其余字段会被丢弃且不报警告。带业务字段的 KML 建议先用 QGIS 转 GPKG。

### 2.3 行政区边界与底图

```python
from gis_cli.ingest import fetch_boundary, build_basemap
fetch_boundary("广州市", level="city")        # 边界缓存到 workspace/.gis_cache
build_basemap(bounds=(113.1, 23.0, 113.5, 23.3), zoom=13)
```

- 边界走多源回退（geojson.cn → DataV → longwosion），**记录实际命中的来源与许可**（`source_url` / `attribution`），不冒充
- 坐标系不确定时 `crs` 留空并写 `crs_note`（例如「可能为 GCJ-02」），不硬编 4326
- 底图按瓦片拼接并写 `.pgw`；**低 zoom 大范围会有偏移**（墨卡托纬度不是线性的，z4 跨 25° 约 5%），城市级（z13+）可忽略

### 2.4 登记进数据目录

```python
from gis_cli.datacatalog.store import Catalog
from gis_cli.ingest import register_dataset

catalog = Catalog("workspace/.gis_agent/catalog.db")
register_dataset(catalog, result.to_record(), source="csv_upload", license="用户自有")
```

登记时写入的来源、许可、原始文件名会**一直保留**：后续任何一次重扫（`refresh_catalog` / prewarm）都不会把它们清空——扫描器不认识的字段按「旧的先留、新的覆盖、瞬时键（`error`）清掉」合并。

---

## 3. 分析计划卡

动手前先给你看 5 行，确认了才执行：

```
┌ 计划确认 ────────────────────────────────────────────────
│ 做什么：用 poi 点位按行政区统计人口并出专题图
│ 用什么：input/poi_points.geojson（已登记 1 个数据集）
│ 输出：  output/（执行时按需求命名）
│ 怎么验：各区人口合计 = 总人口 / 含图名/图例/比例尺/指北针
│ 合规：  未送审，仅内部使用（不做合规判定）
└ [s] 开始  [e] 改一句  [q] 取消
```

- **「用什么」是真的**：从数据目录里按任务关键词匹配，匹配不到就老实写「执行时先确认」，不编
- **TTY**：按 `s` 开始、`e` 改一句（改完重新渲染这张卡）、`q` 取消
- **非 TTY / `--yes`**：自动通过，但**计划卡照样打印**——CI 和脚本不会被卡住，日志里也能看到当时的判断
- 合规那行是固定声明（「未送审，仅内部使用」），**不做合规判定**，免得给你虚假的安全感

---

## 4. 成果包

验收通过后自动生成 `output/成果包.html`：

| 内容 | 说明 |
|---|---|
| 结论 / 成果图 / 图层 / 方法 / 数据来源 / 验证 | 六个段落，验证段写的就是验收结论 |
| 全部内联 | 图片走 `data:`，图层走内联 GeoJSON —— **断网可看**，单文件可直接发微信 |
| 体积 | 默认上限 20MB，超了自动把图压到 1600px 宽；仍超则丢图层并在 note 里说明 |

### 4.1 出 PDF

```bash
python workspace/skills/report-pdf/scripts/html_to_pdf.py \
    --html output/成果包.html --pdf output/成果包.pdf
```

用系统自带的 Edge / Chrome 无头打印：**零新依赖、离线可用、中文天然正常**（系统字体，不像 matplotlib 要手配字体否则方框）。

> 为什么不在引擎里直接出 PDF：宿主 venv 里没有 matplotlib（它只装在 ArcGIS Pro 的 python 里），
> 所以 `build_deliverables` 的 `want_pdf` 在普通运行时恒为 `None`。
> **PDF 走 `report-pdf` 技能**，不要指望它顺手出。

---

## 5. 当前边界（别当成已支持）

| 边界 | 影响 |
|---|---|
| 接入产物**没有接进 agent 工具表** | 「在 chat 里说一句话直接出图」目前不成立，接入要写 Python 调用 |
| KML 的 ExtendedData 丢弃 | 带业务字段的 KML 请先转 GPKG |
| 底图低 zoom 纬度偏移 | z6 跨 25° 约 20px/1536，z4 约 43px/768 |
| xlsx 单元格缺 `r` 属性 | 第三方生成的 xlsx 可能整行塌到第 0 列 |
| 整列为 null 时字段类型可能声明与实际不符 | 元数据偶尔不准，分析不受影响 |
| 版面体检不通过会硬失败 | 轻微项（标题偏 2mm 等）也会拦，模型会多试几轮 |

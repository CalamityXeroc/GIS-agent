"""GIS Domain Knowledge Compendium for Expert Mode.

This module contains structured GIS expertise that gets injected into
LLM prompts when expert_mode is enabled. It teaches the LLM to make
correct GIS decisions about projections, color schemes, spatial analysis,
and map layout.

Each section is designed to be independently injectable so we can
select only relevant sections based on the task description.
"""

from __future__ import annotations

import re
from typing import Set


class GISDomainPrompts:
    """GIS domain knowledge compendium for expert-mode planning."""

    # ── Spatial Analysis Decision Guide ──────────────────────────────

    SPATIAL_ANALYSIS_GUIDE = """## 空间分析操作选择指南

根据用户需求选择正确的空间分析操作：

### 缓冲区分析 (Buffer)
- **适用场景**: "范围内"、"周边"、"附近 X 米/公里"、"影响区"
- **示例需求**: "道路周边500米范围"、"工厂周围1公里缓冲区"
- **实现方式**: execute_code + arcpy.analysis.Buffer()
- **关键参数**: buffer_distance（根据问题规模选单位）, dissolve_option="ALL"或"NONE"
- **代码模板**:
  ```python
  import arcpy
  arcpy.env.workspace = r"{workspace}"
  input_layer = r"{input_path}"
  output = r"{output_path}/buffer.shp"

  if not arcpy.Exists(input_layer):
      raise ValueError(f"输入数据不存在: {input_layer}")
  desc = arcpy.Describe(input_layer)
  if desc.shapeType not in ("Point", "Multipoint", "Polyline", "Polygon"):
      raise ValueError(f"不支持几何类型: {desc.shapeType}")
  arcpy.analysis.Buffer(input_layer, output, "{distance} {unit}", dissolve_option="ALL")
  ```

### 裁剪分析 (Clip)
- **适用场景**: "裁剪"、"提取某区域内的数据"、"按边界切分"
- **实现方式**: execute_code + arcpy.analysis.Clip()
- **代码模板**:
  ```python
  arcpy.analysis.Clip(in_features, clip_features, output)
  ```

### 叠加分析 (Intersect/Union)
- **适用场景**: "共同区域"、"叠加"、"同时满足"、"交集"
- **Intersect**: 只保留重叠部分
- **Union**: 保留所有区域,属性合并
- **实现方式**: execute_code + arcpy.analysis.Intersect() / arcpy.analysis.Union()

### 空间连接 (Spatial Join)
- **适用场景**: "统计每个省/区的数量"、"计算各区域内的点数"、"按区域汇总"
- **示例需求**: "统计各省保护区数量"、"计算每个行政区内的POI数量"
- **实现方式**: execute_code + arcpy.analysis.SpatialJoin()
- **关键参数**: join_type="JOIN_ONE_TO_ONE", match_option="INTERSECT"或"WITHIN"
- **代码模板**:
  ```python
  # 统计每个省份内的保护区数量
  import arcpy
  provinces = r"{省界图层路径}"
  protected_areas = r"{保护区图层路径}"
  output = r"{output_path}/province_stats.shp"
  # 先做空间连接,统计每个省内的保护区
  arcpy.analysis.SpatialJoin(
      target_features=provinces,
      join_features=protected_areas,
      out_feature_class=output,
      join_type="JOIN_ONE_TO_ONE",
      match_option="INTERSECT"
  )
  # 使用频数统计或汇总统计数据
  arcpy.analysis.Frequency(
      in_table=output,
      out_table=r"{output_path}/province_counts.dbf",
      frequency_fields=["Province_Name"],  # 省名称字段
      summary_fields=["Join_Count"]  # 计数会自动汇总
  )
  ```

### 核密度分析 (Kernel Density)
- **适用场景**: "热力图"、"密度分析"、"聚集区域"、"热点"
- **实现方式**: execute_code + arcpy.sa.KernelDensity()
- 需 Spatial Analyst 扩展模块

### 融合/溶解 (Dissolve)
- **适用场景**: "合并同类"、"按字段融合"、"dissolve"
- **实现方式**: execute_code + arcpy.management.Dissolve()
"""

    # ── Thematic Mapping Guide ─────────────────────────────────────

    THEMATIC_MAPPING_GUIDE = """## 专题制图指南

根据数据类型和用途选择正确的配色方案和分类方法：

### 颜色方案选择

#### 1. 顺序色系 (Sequential) — 适用于有序/数值数据
- **适用场景**: 人口密度、GDP、海拔、温度、比例、数量统计
- **推荐配色**:
  - 人口/数量: YlOrRd（黄-橙-红）, OrRd（橙-红）, Purples（紫）
  - 百分比/比例: Blues（蓝）, GnBu（绿-蓝）, YlGn（黄-绿）
  - 高程/地形: 地形专用色带（绿-棕-白）
- **示例**: "各省保护区数量" → YlOrRd 或 OrRd 顺序色系,数值越大颜色越深

#### 2. 发散色系 (Diverging) — 适用于有中点的数据
- **适用场景**: 增长率(正/负)、变化量、偏差、差值
- **推荐配色**: RdYlBu（红-黄-蓝）, PiYG（粉-绿）, Spectral（光谱）

#### 3. 定性色系 (Qualitative) — 适用于分类数据
- **适用场景**: 土地利用类型、土壤类型、行政区划、分类数据
- **推荐配色**: Set1, Pastel1, Accent, Dark2, Paired

### 数据分类方法

- **Natural Breaks (Jenks)**: 适合大多数数据,自动寻找自然分组 — 推荐默认使用
- **Equal Interval**: 适合均匀分布的数据（如温度等距分段）
- **Quantile (分位数)**: 适合偏态分布,每类包含相同数量要素
- **Manual Breaks (手动)**: 已知特定阈值时使用

### 分层设色 (Graduated Colors) 完整流程（已验证通过）

⚠️ 关键提醒（写代码前先看，避免报错）：
- `layer.symbology` 是**属性**（getter+setter），不是 `getSymbology()`/`setSymbology()`
- 布局要用 `layout.createMapFrame(polygon, map)` 创建地图框，不能用 `listElements('MAPFRAME_ELEMENT')` —— 新布局没有地图框元素
- `aprx.createLayout(w, h, unit)` 三个参数都必须传
- 中文版 ArcGIS Pro 没有英文色带名，用 `aprx.listColorRamps()[0]` 作为兜底

```python
from arcpy import mp
import arcpy

# 1. 打开项目和数据
aprx = mp.ArcGISProject(r"{project_path}")
m = aprx.listMaps()[0]
layer = m.addDataFromPath(r"{input_path}")

# 2. 设置分级设色渲染（.symbology 是属性，不是 getSymbology()）
sym = layer.symbology
sym.updateRenderer('GraduatedColorsRenderer')
sym.renderer.classificationField = "FIELD_NAME"       # 数值字段名
sym.renderer.breakCount = 5                            # 分级数
sym.renderer.classificationMethod = "NaturalBreaks"    # 分类方法
# 色带：用第一个可用色带（中文版ArcPro没有英文色带名）
ramps = aprx.listColorRamps()
if ramps:
    sym.renderer.colorRamp = ramps[0]
layer.symbology = sym   # 写回（不是 setSymbology()）

# 3. 创建布局 + 地图框
layout = aprx.createLayout(297, 420, "MILLIMETER")  # A3横版尺寸
poly = arcpy.Polygon(arcpy.Array([
    arcpy.Point(10, 10), arcpy.Point(287, 10),
    arcpy.Point(287, 300), arcpy.Point(10, 300),
    arcpy.Point(10, 10),
]))
layout.createMapFrame(poly, m, "Main Map")  # 不是 listElements()

# 4. 导出PDF
layout.exportToPDF(r"{output_path}")
```

### 仅渲染不导出（若任务只需要设置符号）
```python
sym = layer.symbology
sym.updateRenderer('GraduatedColorsRenderer')
sym.renderer.classificationField = "FIELD_NAME"
sym.renderer.breakCount = 5
ramps = aprx.listColorRamps()
if ramps:
    sym.renderer.colorRamp = ramps[0]
layer.symbology = sym
```

### 图层顺序
- 面图层(基底) → 面边界线 → 线图层 → 点图层 → 标注
- 面图层使用适当透明度(alpha=30-50%)以便看到底图
"""

    # ── Projection Selection Guide ─────────────────────────────────

    PROJECTION_SELECTION_GUIDE = """## 投影坐标系选择指南

选择正确的投影是GIS分析的基础。以下是根据区域和使用场景的投影选择标准：

### 中国区域

#### 1. 全国范围地图
- **Albers 等积圆锥投影 (Albers Equal Area Conic)**
  - 中央经线: 105°E
  - 双标准纬线: 25°N 和 47°N
  - 投影原点纬度: 0°
  - 适用: 全国专题图、面积统计、密度分析
  - WKT参考:
    ```
    PROJCS["Albers_China",
      GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],
      PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],
      PROJECTION["Albers"],
      PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],
      PARAMETER["Central_Meridian",105],
      PARAMETER["Standard_Parallel_1",25],
      PARAMETER["Standard_Parallel_2",47],
      PARAMETER["Latitude_Of_Origin",0],UNIT["Meter",1]]
    ```

#### 2. 省级/局部地区
- **Gauss-Kruger (Transverse Mercator) 3度带**
  - 各省份对应的中央经线:
    - 广西: 108°E 或 111°E (广西跨两个3度带)
    - 广东/海南: 111°E 或 114°E
    - 北京/河北: 114°E 或 117°E
    - 上海/江苏/浙江: 120°E 或 123°E
    - 四川/重庆: 102°E 或 105°E
    - 云南: 99°E 或 102°E
    - 西藏: 84°E, 87°E, 90°E, 93°E, 96°E (范围广)
    - 新疆: 75°E, 78°E, 81°E, 84°E, 87°E, 90°E (范围广)
    - 黑龙江/吉林/辽宁: 126°E 或 129°E (东北)
    - 山东: 117°E 或 120°E
    - 湖北/湖南: 111°E 或 114°E

#### 3. 各省选择原则
- 当省份东西跨度 < 3度: 选择该省中心的3度带中央经线
- 当省份东西跨度 > 3度(如西藏/新疆/内蒙古): 使用 Albers 投影或选最合适的带
- 省级专题图优先选 Gauss-Kruger, 全国图选 Albers

### 全球区域
- **世界地图**: Robinson 或 Winkel Tripel (视觉效果好)
- **数据处理**: Equirectangular (简单,适合计算)
- **Web地图**: Web Mercator (EPSG:3857) — 但面积变形大,不适合面积统计

### 重要原则
- **WGS84 (EPSG:4326) 是地理坐标系,不是投影坐标系**
- **任何涉及距离/面积的计算都必须在投影坐标系下进行**
- 中国地区优先使用 CGCS2000 或 WGS84 为基础的投影坐标系
- 数据已有投影时,检查投影是否合适;不合适则需要重投影
- 多源数据必须统一到同一投影后才能进行分析
"""

    # ── Map Layout Guide ──────────────────────────────────────────

    MAP_LAYOUT_GUIDE = """## 地图整饰指南

一张完整的地图必须包含以下要素：

### 必要地图要素
1. **图名 (Title)** — 位于地图上方中央或左上角,简明扼要
2. **图例 (Legend)** — 说明所有符号含义,位置在右下或左下
3. **比例尺 (Scale Bar)** — 根据地图范围选择合适单位(km/m)
4. **指北针 (North Arrow)** — 位于右上角
5. **图廓 (Neatline)** — 地图边框
6. **数据来源** — 左下角小字
7. **投影/坐标系统说明** — 右下角小字

### ArcPy 布局代码模板

⚠️ 创建新布局时地图框不存在，必须用 `createMapFrame()` 创建，不能用 `listElements()`
⚠️ 不要用 `createTextElement`/`createLegendElement` —— 这些方法在 ArcPro 3.6 中不存在
⚠️ 最小化导出流程：创建布局 → 创建地图框 → 导出（不要添加其他地图元素）

```python
from arcpy import mp
import arcpy

aprx = mp.ArcGISProject(r"{project_path}")
m = aprx.listMaps()[0]

# 创建布局（A4: 210x297mm, A3: 297x420mm）
layout = aprx.createLayout(297, 420, "MILLIMETER")

# 创建地图框（用 arcpy.Polygon 定义位置，不是 listElements 获取）
poly = arcpy.Polygon(arcpy.Array([
    arcpy.Point(10, 10), arcpy.Point(287, 10),
    arcpy.Point(287, 300), arcpy.Point(10, 300),
    arcpy.Point(10, 10),
]))
layout.createMapFrame(poly, m, "Main Map")

# 导出PDF（不要添加图例/指北针/标题，这些API不稳定）
layout.exportToPDF(r"{output_path}")
```
```

### 纸张尺寸选择
- **A4 (210×297mm)**: 省级/小区域地图,竖版为主
- **A3 (297×420mm)**: 省级/区域地图,横版或竖版
- **A2 (420×594mm)**: 全国/大区域地图,横版为主
- **A1 (594×841mm)**: 全国/大区域详细地图
- 中国全国图推荐使用 A3 或 A2 横版( landscape)

### 布局原则
- 边距至少 10mm
- 图名 18-24pt,加粗
- 图例标题 12-14pt,内容 8-10pt
- 数据来源 6-8pt
- 指北针大小适中,不喧宾夺主
- 比例尺通常放在图例下方或地图下方
"""

    # ── Cartographic Style Guide ──────────────────────────────────

    CARTOGRAPHIC_STYLE_GUIDE = """## 制图风格指南

### 字体选择
- **中文地图**: 标题用宋体/黑体,说明用宋体
- **英文地图**: 标题用 Arial/Calibri Bold,说明用 Arial
- **字号**: 标题 18-24pt, 副标题 14-16pt, 图例标签 8-10pt, 来源文字 6-8pt

### 线宽规范
- 行政边界: 省界 0.6pt, 县界 0.3pt, 国界 1.0pt(加特殊符号)
- 道路: 高速 1.5pt, 主干道 1.0pt, 次干道 0.5pt
- 河流: 主要河流 0.8pt, 次要河流 0.3pt

### 颜色使用原则
- 使用 ColorBrewer 配色方案,色盲友好
- 同一张地图不超过 7 个分类颜色
- 背景色用浅色(米白/浅灰),内容用鲜艳色
- 避免红绿搭配(色盲人群)
- 水体用蓝色系,植被用绿色系

### 标注规范
- 省会城市名: 10-12pt,加粗
- 一般城市名: 8-10pt
- 标注避免重叠,使用晕圈(Halo)提高可读性
"""

    # ── Data Preprocessing Guide ──────────────────────────────────

    DATA_PREPROCESSING_GUIDE = """## 数据预处理最佳实践

1. 总是先执行 scan_layers 扫描输入数据,了解可用的图层
2. 检查数据坐标系(CRS),确保所有数据在同一参考系下
3. 处理空几何: 使用 arcpy.management.RepairGeometry
4. 处理重复要素: 使用 arcpy.management.DeleteIdentical
5. 字段计算: 使用 arcpy.management.CalculateField
6. 始终输出到新位置,不修改原始数据
7. 复杂工作流使用 File Geodatabase(.gdb) 存储中间结果
8. 简单输出使用 Shapefile(.shp)
9. 最终出图使用 export_map 工具或 execute_code 生成布局后导出

### 安全代码模板

#### 安全投影变换 (Project)
```python
import arcpy
input_fc = r"{input_path}"
output_fc = r"{output_path}"
target_sr = arcpy.SpatialReference({target_wkid})

# 1. 检查数据是否存在
if not arcpy.Exists(input_fc):
    raise ValueError(f"输入数据不存在: {input_fc}")

# 2. 检查 CRS,无定义则先定义
desc = arcpy.Describe(input_fc)
if desc.spatialReference is None or desc.spatialReference.name == "Unknown":
    print(f"警告: {input_fc} 没有定义坐标系,使用 WGS84")
    arcpy.management.DefineProjection(input_fc, arcpy.SpatialReference(4326))

# 3. 执行投影
arcpy.management.Project(input_fc, output_fc, target_sr)
```

#### 安全删除重复要素 (DeleteIdentical)
```python
import arcpy
input_fc = r"{input_path}"

# 只选择 DeleteIdentical 允许的字段类型
allowed_types = {"Integer", "SmallInteger", "Single", "Double", "String", "Date"}
fields = [f for f in arcpy.ListFields(input_fc) if f.type in allowed_types]
field_names = [f.name for f in fields if f.name.upper() not in ("FID", "OBJECTID", "SHAPE")]

if not field_names:
    print("警告: 没有找到可用于 DeleteIdentical 的字段")
else:
    arcpy.management.DeleteIdentical(input_fc, field_names)
    print(f"已删除重复要素(基于字段: {field_names})")
```
"""

    # ── Section metadata for keyword matching ─────────────────────

    _SECTION_KEYWORDS = {
        "spatial_analysis": ["缓冲区", "缓冲", "buffer", "裁剪", "clip", "叠加",
                            "intersect", "union", "空间连接", "spatial join",
                            "核密度", "热力", "密度分析", "kde", "擦除", "erase",
                            "溶解", "融合", "dissolve", "范围内", "周边", "附近"],
        "thematic_mapping": ["专题图", "制图", "地图", "配色", "颜色", "符号",
                            "人口", "密度", "GDP", "统计", "分布", "分级",
                            "choropleth", "thematic", "graduated", "行政区划",
                            "land use", "土地利用"],
        "projection": ["投影", "坐标系", "coordinate", "crs", "epsg", "wgs",
                      "albers", "gauss", "utm", "中央经线", "标准纬线",
                      "reproject", "projection", "变换", "转换坐标"],
        "map_layout": ["布局", "图例", "比例尺", "指北针", "出图", "导出",
                      "print", "layout", "legend", "scale bar", "north arrow",
                      "标题", "图名", "A4", "A3", "纸张", "导出地图"],
        "cartographic_style": ["字体", "风格", "样式", "审美", "标注", "颜色",
                              "font", "style", "美观", "好看"],
        "data_preprocessing": ["数据质量", "清洗", "预处理", "修复", "检查",
                              "quality", "repair", "clean", "valid"]
    }

    # ── Public API ────────────────────────────────────────────────

    _SECTION_FUNCTIONS = {
        "spatial_analysis": lambda self: self.SPATIAL_ANALYSIS_GUIDE,
        "thematic_mapping": lambda self: self.THEMATIC_MAPPING_GUIDE,
        "projection": lambda self: self.PROJECTION_SELECTION_GUIDE,
        "map_layout": lambda self: self.MAP_LAYOUT_GUIDE,
        "cartographic_style": lambda self: self.CARTOGRAPHIC_STYLE_GUIDE,
        "data_preprocessing": lambda self: self.DATA_PREPROCESSING_GUIDE,
    }

    @classmethod
    def detect_relevant_sections(cls, task_description: str) -> Set[str]:
        """Detect which GIS domain sections are relevant to the task."""
        if not task_description:
            return set()

        text = task_description.lower()
        relevant = set()

        for section_name, keywords in cls._SECTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    relevant.add(section_name)
                    break

        return relevant

    @classmethod
    def get_relevant_sections(cls, task_description: str) -> str:
        """Return only the sections relevant to the task description.

        This keeps the prompt length manageable by only injecting
        the GIS knowledge that's actually needed.
        """
        relevant = cls.detect_relevant_sections(task_description)

        # Always include basic preprocessing
        relevant.add("data_preprocessing")

        sections = []
        for section_name in sorted(relevant):
            func = cls._SECTION_FUNCTIONS.get(section_name)
            if func:
                sections.append(func(cls))

        if not sections:
            return cls.get_all_sections()

        return "\n\n===\n\n".join(sections)

    @classmethod
    def get_all_sections(cls) -> str:
        """Return all GIS domain knowledge sections."""
        sections = [
            cls.SPATIAL_ANALYSIS_GUIDE,
            cls.THEMATIC_MAPPING_GUIDE,
            cls.PROJECTION_SELECTION_GUIDE,
            cls.MAP_LAYOUT_GUIDE,
            cls.CARTOGRAPHIC_STYLE_GUIDE,
            cls.DATA_PREPROCESSING_GUIDE,
        ]
        return "\n\n===\n\n".join(sections)

---
name: buffer_analysis
description: 对矢量数据进行缓冲区分析，生成指定距离的缓冲区多边形
tags: [analysis, buffer, spatial, 缓冲区, 周边]
category: spatial_analysis
version: "1.0"
author: GIS Agent
triggers:
  - 缓冲区
  - buffer
  - 周边范围
  - 影响范围
  - 服务范围
required_inputs:
  - input_layer
  - buffer_distance
optional_inputs:
  output_path: "./output/buffer_result.shp"
  dissolve_option: "NONE"
---

# 缓冲区分析技能

## 功能说明
根据输入的矢量图层，在指定距离内生成缓冲区多边形。常用于分析道路影响范围、设施服务区等。

## 触发条件
当用户提到以下关键词时自动触发：
- "缓冲区"、"buffer"
- "周边范围"、"影响范围"
- "服务范围"、"辐射范围"

## 执行步骤
1. 识别输入图层路径
2. 确定缓冲距离（支持米、千米等单位）
3. 设置溶解选项（是否合并相邻缓冲区）
4. 执行 arcpy.Buffer_analysis
5. 输出结果到指定位置

## 参数说明
- `input_layer`: 输入矢量图层路径（必需）
- `buffer_distance`: 缓冲距离，如 "500 Meters"、"1 Kilometers"（必需）
- `output_path`: 输出路径（可选，默认 ./output/buffer_result.shp）
- `dissolve_option`: 溶解选项 NONE/ALL/LIST（可选，默认 NONE）

## ArcPy 代码模板
```python
import arcpy
arcpy.env.overwriteOutput = True

# 输入参数
in_features = r"${input_layer}"
out_features = r"${output_path}"
buffer_distance = "${buffer_distance}"
dissolve_option = "${dissolve_option}"

# 执行缓冲区分析
arcpy.Buffer_analysis(
    in_features=in_features,
    out_feature_class=out_features,
    buffer_distance_or_field=buffer_distance,
    dissolve_option=dissolve_option
)

# 统计结果
count = arcpy.GetCount_management(out_features)[0]
print(f"缓冲区分析完成，生成 {count} 个要素")

set_result({
    "output": out_features,
    "feature_count": int(count),
    "buffer_distance": buffer_distance
})
```

## 使用示例
- "对道路做500米缓冲区"
- "分析学校周边1公里范围"
- "生成河流200米影响区"

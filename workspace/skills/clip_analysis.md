---
name: clip_analysis
description: 使用裁剪范围裁剪输入图层，提取指定区域内的要素
tags: [analysis, clip, extract, 裁剪, 提取]
category: spatial_analysis
version: "1.0"
triggers:
  - 裁剪
  - clip
  - 提取
  - 切割
  - 范围内
required_inputs:
  - input_layer
  - clip_layer
optional_inputs:
  output_path: "./output/clip_result.shp"
---

# 裁剪分析技能

## 功能说明
使用裁剪范围（如行政区边界）裁剪输入图层，只保留裁剪范围内的要素。

## 触发条件
- "裁剪"、"clip"
- "提取xxx范围内的数据"
- "按边界切割"

## 执行步骤
1. 识别输入图层和裁剪范围图层
2. 验证两个图层的几何类型
3. 执行 arcpy.Clip_analysis
4. 输出裁剪结果

## ArcPy 代码模板
```python
import arcpy
arcpy.env.overwriteOutput = True

# 输入参数
in_features = r"${input_layer}"
clip_features = r"${clip_layer}"
out_features = r"${output_path}"

# 执行裁剪
arcpy.Clip_analysis(
    in_features=in_features,
    clip_features=clip_features,
    out_feature_class=out_features
)

# 统计结果
count = arcpy.GetCount_management(out_features)[0]
print(f"裁剪完成，保留 {count} 个要素")

set_result({
    "output": out_features,
    "feature_count": int(count)
})
```

## 使用示例
- "用广西边界裁剪道路数据"
- "提取北京市范围内的POI"
- "按研究区范围裁剪土地利用数据"

---
name: intersect_analysis
description: 执行叠加相交分析，计算多个图层的交集区域
tags: [analysis, overlay, intersect, 叠加, 相交]
category: spatial_analysis
version: "1.0"
triggers:
  - 相交
  - 叠加
  - intersect
  - overlay
  - 交集
required_inputs:
  - input_layers
optional_inputs:
  output_path: "./output/intersect_result.shp"
  join_attributes: "ALL"
---

# 叠加相交分析技能

## 功能说明
计算多个输入图层的几何交集，生成包含所有输入图层属性的新图层。

## 触发条件
- "相交分析"、"叠加分析"
- "intersect"、"overlay"
- "计算两个图层的交集"

## 执行步骤
1. 识别所有输入图层
2. 验证图层几何类型兼容性
3. 执行 arcpy.Intersect_analysis
4. 输出结果

## ArcPy 代码模板
```python
import arcpy
arcpy.env.overwriteOutput = True

# 输入参数 - 多个图层用分号分隔
input_layers_str = r"${input_layers}"
input_layers = [s.strip() for s in input_layers_str.split(";")]
out_features = r"${output_path}"
join_attributes = "${join_attributes}"

# 执行相交分析
arcpy.Intersect_analysis(
    in_features=input_layers,
    out_feature_class=out_features,
    join_attributes=join_attributes
)

# 统计结果
count = arcpy.GetCount_management(out_features)[0]
print(f"叠加分析完成，生成 {count} 个要素")

set_result({
    "output": out_features,
    "feature_count": int(count),
    "input_count": len(input_layers)
})
```

## 使用示例
- "计算土地利用和行政区的交集"
- "叠加分析道路和规划区"
- "找出森林和保护区的重叠区域"

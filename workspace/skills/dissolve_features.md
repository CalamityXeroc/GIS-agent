---
name: dissolve_features
description: 根据指定字段合并相邻或相同属性的要素
tags: [dissolve, merge, aggregate, 融合, 合并]
category: data_management
version: "1.0"
triggers:
  - 融合
  - dissolve
  - 按字段合并
  - 聚合
  - 归并
required_inputs:
  - input_layer
optional_inputs:
  dissolve_field: ""
  output_path: "./output/dissolve_result.shp"
  statistics_fields: ""
---

# 要素融合技能

## 功能说明
根据指定字段值合并相邻要素，将具有相同属性值的要素融合为一个要素。

## 触发条件
- "融合"、"dissolve"
- "按xxx字段合并要素"
- "聚合多边形"

## 执行步骤
1. 确认输入图层和融合字段
2. 设置统计字段（可选）
3. 执行 arcpy.Dissolve_management
4. 输出融合结果

## ArcPy 代码模板
```python
import arcpy
arcpy.env.overwriteOutput = True

# 输入参数
in_features = r"${input_layer}"
out_features = r"${output_path}"
dissolve_field = "${dissolve_field}" or None
statistics_fields = "${statistics_fields}" or None

# 解析统计字段
stat_fields = None
if statistics_fields:
    # 格式: "AREA SUM;COUNT COUNT"
    stat_fields = [[s.split()[0], s.split()[1]] for s in statistics_fields.split(";") if s.strip()]

# 执行融合
if dissolve_field:
    dissolve_field = [f.strip() for f in dissolve_field.split(";")]

arcpy.Dissolve_management(
    in_features=in_features,
    out_feature_class=out_features,
    dissolve_field=dissolve_field,
    statistics_fields=stat_fields
)

# 统计结果
in_count = arcpy.GetCount_management(in_features)[0]
out_count = arcpy.GetCount_management(out_features)[0]
print(f"融合完成: {in_count} -> {out_count} 个要素")

set_result({
    "output": out_features,
    "input_count": int(in_count),
    "output_count": int(out_count),
    "dissolve_field": dissolve_field
})
```

## 使用示例
- "按省份字段融合县级边界"
- "合并相邻的同类型用地图斑"
- "融合所有多边形为一个"

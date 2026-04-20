---
name: field_calculator
description: 批量计算或更新字段值，支持 Python 和 VBScript 表达式
tags: [field, calculate, attribute, 字段计算, 属性]
category: data_management
version: "1.0"
triggers:
  - 字段计算
  - 计算字段
  - 更新属性
  - calculate field
  - 赋值
required_inputs:
  - input_layer
  - field_name
  - expression
optional_inputs:
  expression_type: "PYTHON3"
  code_block: ""
---

# 字段计算技能

## 功能说明
使用表达式批量计算或更新要素图层的字段值。支持简单赋值和复杂的 Python 表达式。

## 触发条件
- "字段计算"、"计算字段值"
- "更新xxx字段"
- "批量赋值"

## 执行步骤
1. 确认输入图层和目标字段
2. 检查字段是否存在，不存在则创建
3. 编写计算表达式
4. 执行 arcpy.CalculateField_management

## ArcPy 代码模板
```python
import arcpy
arcpy.env.overwriteOutput = True

# 输入参数
in_table = r"${input_layer}"
field_name = "${field_name}"
expression = "${expression}"
expression_type = "${expression_type}"
code_block = """${code_block}"""

# 检查字段是否存在
fields = [f.name for f in arcpy.ListFields(in_table)]
if field_name not in fields:
    # 添加字段（默认 TEXT 类型）
    arcpy.AddField_management(in_table, field_name, "TEXT", field_length=255)
    print(f"创建新字段: {field_name}")

# 执行字段计算
if code_block.strip():
    arcpy.CalculateField_management(
        in_table=in_table,
        field=field_name,
        expression=expression,
        expression_type=expression_type,
        code_block=code_block
    )
else:
    arcpy.CalculateField_management(
        in_table=in_table,
        field=field_name,
        expression=expression,
        expression_type=expression_type
    )

count = arcpy.GetCount_management(in_table)[0]
print(f"字段计算完成，更新 {count} 条记录")

set_result({
    "table": in_table,
    "field": field_name,
    "records_updated": int(count)
})
```

## 使用示例
- "计算面积字段"
- "将名称字段全部转大写"
- "根据类型代码更新类型名称"

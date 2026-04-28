# GIS Agent 工作区

## 文件夹说明

- **input/** - 存放输入数据（用户提供的 GIS 数据文件）
- **output/** - 存放最终输出结果（导出的地图、报告等）
- **temp/** - 存放中间处理文件

## 使用说明

1. 将您的 GIS 数据文件（shapefiles, geodatabase 等）放入 `input/` 文件夹
2. 启动 Agent 后，它会自动识别 input 文件夹中的数据
3. Agent 的所有操作都会在此工作区内进行
4. 最终结果会保存在 `output/` 文件夹中

## 注意事项

- Agent 会自动清理 `temp/` 文件夹中的旧文件
- 请不要手动修改 temp 文件夹中的内容
- 备份重要数据

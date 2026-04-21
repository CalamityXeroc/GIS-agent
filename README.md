# GIS agent

GIS Agent 是一个基于 LLM 的智能代理，专为 ArcGIS Pro 3.6用户设计（后续可能会适配多种GIS软件），旨在通过自然语言交互简化 GIS 任务的执行。它能够理解用户的意图，规划任务步骤，并自动调用 ArcPy 或其他工具来完成复杂的 GIS 工作流程。

## 前言

对于非GIS专业的同学来说，利用GIS软件进行制图或分析可能会有一定的难度，因此我就想能不能开发GIS agent来解决这一问题。这个项目来自于一个大学生在AI辅助下的尝试，虽然不够专业，但我希望它能为GIS的学习者提供一些帮助！


还在早期开发阶段，很有可能会出现问题，欢迎大家提出宝贵的意见和建议！如果你也对GIS agent感兴趣，或者有相关的经验和技能，欢迎加入我们，一起完善这个项目！



## 快速开始

### 1. 安装
```bash
pip install -e .
```

使用 ArcGIS Pro Python：

```bash
"E:\ArcGISPro3.6\bin\Python\envs\arcgispro-py3\python.exe" -m pip install -e .
```

### 2. 启动

推荐双击：

- 一键启动.bat

或命令行启动：

```bash
gis-agent chat --workspace ./workspace
```

启动脚本会自动完成：

1. 检测 Python 运行时（优先 ArcGIS Pro Python，失败回退系统 Python）
2. 同步本地 editable 安装
3. 初始化工作区目录
4. 以 `--workspace .\\workspace` 启动 Agent

### 3. 输入输出目录

- 输入数据：`workspace/input/`
- 输出结果：`workspace/output/`
- 临时文件：`workspace/temp/`
- 自定义技能：`workspace/skills/`

## 项目结构

```text
.
 config/
    execution_adapter_config.json
    intents.json
    llm_config.example.json
    llm_config.json.example
    planner_adapter_config.json
    recovery_strategies.json
 src/
    baml_client/
    gis_cli/
 workspace/
    input/
    output/
    skills/
    temp/
 INSTALL.md
 pyproject.toml
 README.md
 一键启动.bat
```

## 核心能力

- 自然语言驱动的 GIS 任务执行
- 任务规划、步骤编排与失败恢复
- ArcPy 优先执行，ArcPy 不可用时自动降级
- 图层扫描、合并、投影转换、质量检查、地图导出
- BAML 能力映射与启动前预检

## 内置技能库

### 空间分析技能
- **缓冲区分析** (`buffer_analysis`)：对矢量数据进行缓冲区分析，生成指定距离的缓冲区多边形
- **裁剪分析** (`clip_analysis`)：使用裁剪范围裁剪输入图层，提取指定区域内的要素
- **相交分析** (`intersect_analysis`)：执行叠加相交分析，计算多个图层的交集区域

### 数据管理技能
- **要素融合** (`dissolve_features`)：根据指定字段合并相邻或相同属性的要素
- **字段计算** (`field_calculator`)：批量计算或更新字段值，支持 Python 表达式

### 其他技能
还加入了一些通用技能，比如阅读pdf，word撰写等常用的技能~

所有技能均位于 `workspace/skills/` 目录，支持自定义扩展。

## 常用命令

### Agent 交互

```bash
gis-agent chat --workspace ./workspace
```

### 单次任务执行

```bash
gis-agent run "扫描并合并图层" --workspace ./workspace
gis-agent run "制作专题图" --workspace ./workspace --execute
```

### 诊断与状态

```bash
gis-agent status
gis-agent tools
gis-agent skills
gis-agent baml-check
gis-agent baml-check --strict --require "intent,task_refine,planning,recovery"
```

## LLM 配置

1. 复制模板：

```bash
cp config/llm_config.example.json config/llm_config.json
```

2. 编辑 `config/llm_config.json`，填写 `api_key/provider/model/api_base`。

注意：`config/llm_config.json` 已在 `.gitignore` 中，默认不会提交。

## BAML 预检（可选）

```bash
gis-agent chat --baml-precheck
gis-agent chat --baml-precheck --baml-strict
gis-agent chat --baml-precheck --baml-strict --baml-require "intent,planning"
```

建议发布前配置：

```json
{
  "enable_baml_preflight": true,
  "baml_preflight_strict": true,
  "enable_baml_builtin_fallback": false
}
```

## 常见问题

1. 启动后识别不到数据
- 确认数据已放入 `workspace/input/`
- 确认启动参数包含 `--workspace ./workspace`

2. 无 ArcGIS Pro 环境
- 可运行对话与规划能力
- 依赖 ArcPy 的步骤会自动降级，不会导致整机不可用

3. 模型调用失败
- 检查 `config/llm_config.json` 配置项是否完整
- 检查网络与 API Key 是否有效

## 说明文档

- 安装与故障排查：`INSTALL.md`
- 多模型适配开发文档：`docs/适配多种模型开发文档.md`

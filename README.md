# GIS Task CLI（MVP）

面向 ArcGIS Pro 工作流的通用 GIS 任务求解平台脚手架。

## 架构概览

单一引擎：**有界观察-决策-执行循环** + 原生 tool calling + 代码自修复 + 强制验收。
（早期那套 plan-then-execute 引擎已从本包移除。）

```
src/gis_cli/
├── engine/              # 新引擎：循环 / 工具 / 验收 / 自修复
│   ├── loop.py          # 有界循环：观察 → 决策 → 执行，含预算与守卫
│   ├── tools.py         # 16 个工具（含 web_search / create_map / run_recipe）
│   ├── llm_client.py    # 模型客户端：流式 / 熔断 / 降级 / 用量统计
│   ├── tool_codec.py    # 工具协议自适应（原生 tool calling ↔ JSON）
│   ├── repair.py        # 代码自修复（traceback 回喂）
│   ├── verifier.py      # 验收：结构断言 + 语义核对 + 图面识图
│   ├── project_log.py   # 项目日志（相关性 + 最近 混合检索）
│   └── runner.py        # 装配入口 build_engine()
├── runtime/             # ArcPy 运行时
│   ├── kernel.py        # 持久 Jupyter 内核（一次 import，跨步状态）
│   ├── code_runner.py   # 内核优先 + 子进程兜底 + 覆盖前自动备份
│   ├── error_memory.py  # 常见故障 → 修复建议
│   └── hygiene.py       # 交付卫生（过程文件 / 重做残留）
├── cartography/         # 制图：facts → design → render → qc
├── datacatalog/         # SQLite 数据目录（字段名 / 坐标系 / 范围）
├── recipes/             # 48 个 GIS 方法论配方（自带验证断言）
├── ui/                  # 终端界面（转录 / 状态行 / 输入框 / 命令）
├── trace/               # 每次 run 的 JSONL + Markdown 报告
├── requirements/        # 需求抽取（任务书 → 可验收清单）
├── skills/              # SKILL.md 技能（渐进披露，run_skill 读全文）
└── agent/               # CLI 入口（chat / loop）
```

### 一次任务的执行链路

```
用户任务
   ↓
需求抽取（任务书 → 可验收清单）
   ↓
┌─ 有界循环（max_turns / 整跑截止）──────────────┐
│  模型决策 → 工具调用 → 观察结果 → 再决策        │
│    ↑                      ↓                      │
│    └── 代码失败自动修复（traceback 回喂）        │
└─────────────────────────────────────────────────┘
   ↓
强制验收（结构断言 + 语义核对 + 图面识图）
   ↓ 不过 → 带缺口反馈继续修（有次数上限）
交付 + trace + 项目日志
```

### 设计要点

- **不依赖 ArcGIS 也能启动/规划/验收**：只有真正跑 GP 时才需要 ArcPy
- **有界**：轮次、诊断次数、修复次数、整跑截止时间都有上限，不会无限空转
- **可验证**：验收不通过就不算完成（`completed` ≠ 过验收）
- **可归因**：每次 run 都有 trace，`scripts/latency_report.py --cost` 看耗时与用量
- **可恢复**：`ask_user` 暂停后 `resume()` 续跑（含 trace 续写）

## 当前已实现能力

**执行与可靠性**
- 有界观察-决策-执行循环，原生 tool calling（不支持的模型自动降级 JSON 协议）
- 代码失败自动修复：traceback + 数据目录 + 配方提示一起回喂
- 持久 Jupyter 内核：一次 import arcpy，跨步保留变量（每步 ~60ms）
- ZMQ 执行串行化（修过并发写坏 wire 协议的崩溃）
- 覆盖写入前自动备份（GDB 成员走 `CopyFeatures` 导出，普通文件走 `shutil`）
- 模型熔断：连续失败进冷却、冷却时长指数升级；`fallback_models` 自动切换

**验收**
- 结构断言：存在性 / 坐标系 WKID / 字段 / 要素数 / 栅格非常量 / 图片非空白
- 语义核对：对照需求清单逐条判定（带证据，可 PARTIAL）
- 图面识图：批量调用识图模型看地图（中文是否方框、图例是否被裁、图面是否空白）
- 交付卫生：过程文件（`.pkl/.tmp/.bak`）、重做残留（同前缀重复件）、体积比

**制图**
- rules-based 设计引擎：采集数据事实 → 出 LayoutSpec → 渲染 → 版面体检
- 自动决定纸张/朝向/图名/图例位置/比例尺刻度/指北针/配色
- 统一图例分级、迁移图叠加层（方向箭头 + 年份标注）
- 版面体检不合格会带原因重渲；必须交付可编辑 `.aprx`

**数据与方法论**
- SQLite 数据目录：真实字段名 / 坐标系 / 范围 / 要素数（懒扫描，启动不阻塞）
- 48 个配方：投影 / 缓冲 / 空间连接 / 核密度 / 局部莫兰 / 可达性 / 栅格运算…
  每个配方自带 validation 断言
- SKILL.md 技能：启动只注入名字+描述，`run_skill` 读全文（渐进披露）

**可观测**
- 流式输出：首字 2~6s 到屏，状态行显示实时字数
- `[timing]` 分段日志 + `latency_report.py`：模型等待 / 工具执行 / 压缩占比
- token 用量与**前缀缓存命中率**写入 trace，`--cost` 可汇总
- 项目日志：相关性 + 最近混合检索，把"上次同类任务踩过的坑"注入下次

**界面与工程**
- 交互式终端：转录 / 状态行 / 输入框 / `@` 文件补全 / `/` 命令 / Esc 中断
- 运行中再输入 = 排队，本轮结束自动合并继续
- 会话可续（`--continue` / `--resume`）、导出 Markdown
- CI：只跑不依赖 ArcGIS 的部分（`pytest -m "not arcpy"`，约 35s）

## 快速开始

### ⚠️ 重要提示

**Agent 默认使用中文交互**：
- 所有响应默认为中文，适合中国用户
- 支持自然语言输入，不需要记忆 GIS 专业术语
- 如需英文，可以说"请用英文回答"或"switch to English"

**工作区说明**：
- Agent 使用 `workspace/` 文件夹作为数据工作区
- 将您的 GIS 数据放入 `workspace/input/` 文件夹
- Agent 会自动识别文件夹中的数据
- 所有处理过程和结果都保存在工作区内
- 文件夹结构：
  - `input/` - 输入数据（用户提供的 GIS 文件）
  - `output/` - 输出结果（最终成果）
  - `temp/` - 临时文件（中间处理文件）

**常见问题**：
- ✅ **"你好，你可以做什么"** → 得到友好的中文功能介绍
- ❌ **返回英文工具列表** → 说明意图识别有误，已在最新版修复
- 💡 **如遇问题**：查看 [CHINESE_FIX.md](CHINESE_FIX.md) 了解修复细节

### 方式一：使用 AI Agent（推荐）

#### 快速测试

在开始之前，建议先运行自动化测试验证系统状态：

```powershell
# 运行完整测试套件
python auto_test.py

# 或使用批处理脚本
.\运行自动测试.bat

# 诊断 LLM 配置（如果使用 LLM 增强）
.\诊断LLM.bat
```

测试内容包括：
- ✅ 模块导入验证
- ✅ 参数规范化测试（LLM 参数映射）
- ✅ 执行器多步骤测试
- ✅ Agent 聊天流程测试
- ✅ 完整用户会话模拟

所有测试应该显示 `Passed: 5/5`。

#### 启动 Agent

启动交互式 GIS Agent 会话：

```powershell
# 安装
pip install -e .

# 启动 Agent 交互式会话（新引擎：过程可见、运行中可纠偏、Esc 中断）
gis-agent chat --workspace .\workspace

# 接续上一次会话
gis-agent chat --continue

# 或使用快捷脚本
.\start_agent.bat

# 或使用 ArcGIS Pro Python 环境
.\一键启动.bat

# 示例对话：
# > 你好，你可以做什么？
# Agent 将介绍功能并显示当前工作区状态
#
# > 扫描我的数据
# Agent 将自动扫描 workspace/input/ 文件夹
#
# > 制作专题图
# Agent 将引导你完成专题图制作流程
```

> CLI 细节（快捷键、斜杠命令、配色、会话续聊）见 [docs/cli.md](docs/cli.md)。
> 启动为何很快、流式输出、以及各段耗时的实测数字见 [docs/performance.md](docs/performance.md)。

#### 使用工作区

1. **准备数据**：
   ```powershell
   # 将 GIS 数据文件（.shp, .gdb, .tif 等）放入 workspace/input/
   cp your_data.shp workspace/input/
   ```

2. **启动 Agent**：
   ```powershell
   gis-agent chat
   ```

3. **Agent 会自动识别数据**：
   ```
   📁 工作区状态
   工作区路径: E:\BST_lab\GIS CLI\workspace
   
   输入数据 (3 个文件):
   - province.shp (shapefile, 2.5 MB)
   - city.shp (shapefile, 1.2 MB)
   - dem.tif (raster, 15.8 MB)
   ```

4. **开始工作**：
   ```
   You: 合并这些图层并制作地图
   Agent: 明白了！我将执行以下步骤...
   ```

Agent 单次任务执行：

```powershell
# 执行单个任务（预览模式）
gis-agent run "整合数据并导出专题图" --dry-run

# 执行单个任务（实际执行）
gis-agent run "扫描并合并所有图层" --workspace ./data
```

查看可用工具和技能：

```powershell
gis-agent tools    # 列出所有工具
gis-agent skills   # 列出所有技能
gis-agent status   # 查看 Agent 状态
gis-agent eval-loop -n 30   # 运行评测闭环并生成改进建议
gis-agent benchmark-models --models "gpt-4o-mini,claude-3-5-sonnet-20241022,Qwen/Qwen3.5-32B" --repeats 2
# 运行跨模型基准评测，输出成功率差、解析错误率、平均时延
```

**启用 LLM 增强（可选但推荐）：**

Agent 支持通过 LLM API 理解任意自然语言，不局限于 GIS 关键词。

1. 复制配置示例并填入你的 API Key：

```bash
# 复制配置示例
cp config/llm_config.example.json config/llm_config.json

# 编辑配置，填入真实 API Key
```

配置文件示例（支持多种国产模型）：

```json
{
  "model": "Pro/zai-org/GLM-5",
  "api_key": "sk-你的API密钥",
  "api_base": "https://api.siliconflow.cn/v1"
}
```

⚠️ **注意**：`config/llm_config.json` 已被 `.gitignore` 排除，不会上传到 Git，保护你的 API 密钥安全。

### 思考档位（reasoning_effort）

部分推理模型默认不返回思考内容，需要在配置里显式指定档位，否则界面的 `◆ 思考` 块
只会显示"模型自述"占位，看不到真实推理：

```json
{
  "engine": {
    "reasoning_effort": "medium",
    "reasoning_effort_by_task": { "agent": "low" }
  }
}
```

- 可用值：`minimal` / `low` / `medium` / `high` / `xhigh` / `max`（具体支持范围看网关）。
- **留空则不传该字段**。换到不认这个参数的网关时务必留空，否则会因多传参数报错。
- 档位越高思考越充分但越慢，按任务长度取舍。
- `reasoning_effort_by_task` 可按任务类型覆盖（不配则全部走全局值）。
  实测交互轮用 `low`：首字 5.11s→2.37s、总耗时 8.72s→5.80s，输出质量相当；
  规划/验收类任务刻意保留 `medium`。

## 直接出图 / 自动化 ArcGIS Pro（推荐）

如果你希望输出真实 GIS 成果（如 `.gdb`、地图导出 JPG/PNG），请使用 ArcGIS Pro Python 环境。

1. 先检查 ArcPy 环境：

```powershell
./check_arcpy_env.bat
```

2. 一键自动执行（默认会 `--exec`）：

```powershell
./run_arcgispro_exec.bat 制图并导出广西专题图
```

3. 输出位置：

- `outputs/<task_id>/workspace/中间数据库.gdb`
- `outputs/<task_id>/map_exports/`（JPG/PNG）
- `outputs/<task_id>/quality_reports/`（质量报告）

说明：

- `--dry-run` 只做预演，主要输出 JSON/TXT；
- `--exec` + ArcPy 可用，才会产出真实 GIS 数据。

## 运行产物

每次 run 的产物与记录：

- `workspace/output/`：交付物（GDB / 图片 / 表格 / `.aprx`）
- `workspace/.gis_agent/traces/<run_id>.jsonl`：完整事件流（可回放）
- `workspace/.gis_agent/traces/<run_id>.md`：人读的 Markdown 报告
- `workspace/.gis_agent/runs/<run_id>.json`：任务状态快照
- `workspace/.gis_agent/backups/<时间戳>/`：覆盖写入前的自动备份
- `workspace/.gis_agent/project_log.md`：跨任务的项目日志（注入下次提示词）

## 测试覆盖

约 **460 个测试**（全量约 60s；CI 只跑不依赖 ArcGIS 的部分，约 35s）。

```powershell
# 全量
pytest

# 只跑不需要 ArcGIS 的（CI 用的就是这条）
pytest -m "not arcpy"

# 新引擎相关
pytest tests/test_engine_*.py
```

主要覆盖：
- 引擎循环 / 工具协议自适应 / 代码自修复 / 验收器
- 内核往返、ZMQ 串行化、覆盖前备份、GDB 存在性判断
- 制图设计引擎（可用假 facts 跑，不需要 ArcGIS）
- 数据目录增量扫描 / 项目日志相关性检索
- 终端界面（含无 ANSI 终端的回归）
- 依赖边界（新引擎不得依赖 langchain/litellm/chromadb/pydantic）

## 项目路线图

### 已完成 ✅
- 有界观察-决策-执行循环（原生 tool calling，不支持时自动降级 JSON 协议）
- 代码失败自动修复（traceback + 数据目录 + 配方提示一起回喂）
- 持久 Jupyter 内核（一次 import arcpy，跨步保留变量）
- 强制验收：结构断言 + 语义核对 + 图面识图 + 交付卫生
- 制图设计引擎：facts → design → render → qc（含统一图例、迁移图叠加层）
- SQLite 数据目录 + 48 个方法论配方（自带验证断言）
- SKILL.md 技能渐进披露
- 流式输出、模型熔断、前缀缓存命中率统计
- 内核往返合并、只读工具并发、覆盖写入自动备份
- 交互式终端（排队合并、会话续聊、导出）
- CI（只跑无 ArcGIS 部分）

### 剩余候选 📋
- 从历史成功 run 中自动沉淀配方（经验固化闭环）
- 工具输出按类型瘦身（现在统一按字符数截断）
- 需求抽取合流进首轮对话（省一次往返；以稳定性为前提）
- 跨 run 的失败归因统计（哪类失败最常见）
- 断点续跑（从 `runs/*.json` 恢复，而不只是 `ask_user` 暂停）
- 更多配方覆盖（空间统计、网络分析）

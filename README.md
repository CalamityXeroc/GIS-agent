# GIS Task CLI（MVP）

面向 ArcGIS Pro 工作流的通用 GIS 任务求解平台脚手架。

已适配 **ArcGIS Pro 3.7**（Pro 3.7.0.1901 / Python 3.13，真机实测通过），并保留 **3.6** 兼容；解释器与制图样式目录都按实际安装自动发现，不写死安装路径。

## 架构概览

单一引擎：**有界观察-决策-执行循环** + 原生 tool calling + 代码自修复 + 强制验收。
（早期那套 plan-then-execute 引擎已下线，代码在 `legacy/`，不参与打包。）

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
├── web/                 # 本地 Web 服务（FastAPI）：datasets / profile / rows / distinct / raster bands / preview
├── recipes/             # 48 个 GIS 方法论配方（自带验证断言）
├── ui/                  # 终端界面（转录 / 状态行 / 输入框 / 命令）
├── trace/               # 每次 run 的 JSONL + Markdown 报告
├── requirements/        # 需求抽取（任务书 → 可验收清单）
├── skills/              # SKILL.md 技能（渐进披露，run_skill 读全文）
├── legacy/              # 旧引擎（已下线，不参与打包）
└── agent/               # CLI 入口（chat / loop）
```

Web 控制台前端是独立的 `web/` 工程（Vite + React + MapLibre）：

```
web/
└── src/components/gis/    # 「数据资产」GIS 工作台前端（目录树 / 图层栈 / 地图 / 右 dock / 属性表 / 状态栏）
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
- 配方沉淀：`giscli distill` 把一次成功运行泛化成候选配方（静态门禁 → 回放验证 → 采纳入库），
  采纳后下次启动自动加载并回流进 prompt；UI 里 `/distill` 同样可用
- SKILL.md 技能：启动只注入名字+描述，`run_skill` 读全文（渐进披露）

**可观测**
- 流式输出：首字 2~6s 到屏，状态行显示实时字数
- `[timing]` 分段日志 + `latency_report.py`：模型等待 / 工具执行 / 压缩占比
- token 用量与**前缀缓存命中率**写入 trace，`--cost` 可汇总
- 项目日志：相关性 + 最近混合检索，把"上次同类任务踩过的坑"注入下次

**数据资产（Web GIS 工作台）**
- 布局对齐真实 GIS 软件：左 dock 目录树 + 图层栈、中间地图、右 dock 四 tab、底部属性表、状态栏；dock 可拖宽、可折叠，属性表高度可拖
- 目录树按 文件夹 → `.gdb` → 要素类 逐级展开，行尾带 EPSG 徽标与要素数，双击上图；`.gdb` 内的非空间表标「仅属性表」，双击只开属性表、不进图层栈
- 图层栈：显隐、拖拽排序、逐层缩放到范围、移除
- 地图：仓库内离线底图（零外部请求）、CRS 警告条、图例、比例尺、指北；非 4326 图层经 pyproj 重投影到 4326 再上图
- 右 dock 四 tab：常规（名称 / 类型 / 几何类型 / 要素数 / 可见性 / 源路径）、符号化（单色、唯一值逐类改色、描边 / 线宽 / 透明度；栅格 R/G/B 合成 + 拉伸 + 色带）、字段、元数据（空间参考 WKID·投影·基准面·单位·轴序、范围、要素、数据属性）
- 底部属性表：一图层一 tab，分页 / 排序 / 筛选全走服务端，footer 只写服务端给的数字（行缓存截断时明写「仅前 N 行」）
- 状态栏：光标坐标 / 比例尺 / 显示坐标系 / 可见图层数 / 缩放

**界面与工程**
- 交互式终端：转录 / 状态行 / 输入框 / `@` 文件补全 / `/` 命令 / Esc 中断
- 运行中再输入 = 排队，本轮结束自动合并继续
- 会话可续（`--continue` / `--resume`）、导出 Markdown

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
- 💡 **如遇问题**：先跑 `giscli doctor` 看自检输出，命令行选项见 `giscli --help`

### 方式一：装一次，之后在任何目录都能用 `giscli`（推荐）

```powershell
# 一次性安装：写 shim、只追加用户级 PATH（不需要管理员权限）
python install.py

# 新开一个终端窗口（PATH 才会生效）
giscli --help
giscli doctor
giscli chat --workspace .\workspace
```

- `install.py` 会顺手做一次环境自检并打印结果；重复跑只更新 shim 内容，不会重复追加 PATH。
- cmd / PowerShell / **Git Bash** 里都能敲 `giscli`：三份 shim 由同一个模板生成，统一转给 `gis_cli.boot`（找 Pro 的 python + `PYTHONPATH=src`）。
- 撤销 PATH：`python scripts/install_shim.py --remove --scripts-dir "<目录>"`，目录抄 `install.py`
  结尾打印的那行字面路径（cmd / PowerShell / Git Bash 通用）。cmd 里也可写
  `"%LOCALAPPDATA%\Programs\giscli"`，PowerShell 写 `"$env:LOCALAPPDATA\Programs\giscli"`——
  `%LOCALAPPDATA%` 只有 cmd 会展开。
  （`--scripts-dir` 不能省：缺省值是 `scripts\` 自己，删的就不是这次装进去的那一条）。

#### 环境自检

```powershell
# Pro python / arcpy / 核心依赖 / pyproj / web 依赖 / src / 工作空间 / token 逐项给状态与命令
giscli doctor

# 加 --deep 会多跑一次持久内核启动检查（约 20 秒）
giscli doctor --deep
```

全过时最后一行是 `环境正常`；「注意」不影响退出码，只有「失败」才是 1。

#### 启动 Agent

启动交互式 GIS Agent 会话：

```powershell
# 启动 Agent 交互式会话（新引擎：过程可见、运行中可纠偏、Esc 中断）
giscli chat --workspace .\workspace

# 接续上一次会话
giscli chat --continue

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

> 启动时会在后台自动拉起 Web 控制台（`pythonw` 无窗口、单实例、日志在 `workspace/.gis_agent/web.log`），地址见启动输出里的 `控制台: http://127.0.0.1:8760/?token=…` 一行；`GIS_NO_WEB=1` 或 `--no-web` 可关掉，`giscli web --stop` 停止后台面板（自己手动 `giscli web` 起的实例同样会落 `workspace/.gis_agent/web.pid`，`--stop` 一样能停）。
>
> CLI 细节（快捷键、斜杠命令、配色、会话续聊）在会话里用 `/help` 查看。

#### 使用工作区

1. **准备数据**：
   ```powershell
   # 将 GIS 数据文件（.shp, .gdb, .tif 等）放入 workspace/input/
   cp your_data.shp workspace/input/
   ```

2. **启动 Agent**：
   ```powershell
   giscli chat
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
# 执行单个任务（不进交互，跑完即退出）
giscli loop "扫描并合并所有图层" --workspace ./data
```

查看可用工具和技能：在交互会话里用斜杠命令

```powershell
/tools      # 列出所有工具
/skills     # 列出所有技能
/status     # 查看 Agent 状态
```

跨模型基准（跑真实引擎循环，测任务是否真的做完）：

```powershell
giscli benchmark-models --models "gpt-4o-mini,claude-3-5-sonnet-20241022,Qwen/Qwen3.5-32B" --repeats 2
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

> **版本说明（3.7 / 3.6）**
>
> - Pro **3.7** 已真机实测通过：`giscli doctor --deep` 7 项全过，`CalculateField` / 持久内核 / 制图导出均正常（Pro 3.7 自带 Python 3.13）。
> - 解释器发现是多级回退：注册表四值 → 已知安装目录一层扫描 → `ARCGIS_PRO_PYTHON` 覆盖；`ARCGIS_PRO_PYTHON` 指到 `propy.bat` 时也会先探活再采用，不会把 `.bat` 直接丢给子进程。多版本并存时按版本降序取最高版。
> - 3.7 把图例样式改了名（3.6 的 `图例 1..6` → 3.7 的 `中心标题图例` / `手动列图例` / `透明背景图例` 等），`config/cartography_styles.json` 已按 3.7 重生成，同时**保留 3.6 关键字**，两个版本都能出图。
> - 换机器或重装 Pro 后若遇到 `ERROR 000574: 未安装 Python`，那是 Pro 自身的 Python 运行时注册没跟上（`HKLM\SOFTWARE\ESRI\ArcGISPro` 的 `PythonCondaRoot` 仍指向已删除的旧目录），不是本项目的代码问题：改注册表这两项，或用 `ARCGIS_PRO_PYTHON` 指向 `...\ArcGISPro3.7\bin\Python\envs\arcgispro-py3\python.exe`。

1. 先检查环境（含 ArcPy 与 Pro 的 Python）：

```powershell
giscli doctor --deep
```

2. 跑一个真实任务（跑完即退出，不经交互）：

```powershell
giscli loop "制图并导出广西专题图" --workspace .\workspace
```

3. 输出位置：交付物在 `workspace/output/`，每次 run 的 trace 在 `workspace/.gis_agent/`（见下一节）。

## 运行产物

每次 run 的产物与记录：

- `workspace/output/`：交付物（GDB / 图片 / 表格 / `.aprx`）
- `workspace/.gis_agent/traces/<run_id>.jsonl`：完整事件流（可回放）
- `workspace/.gis_agent/traces/<run_id>.md`：人读的 Markdown 报告
- `workspace/.gis_agent/runs/<run_id>.json`：任务状态快照
- `workspace/.gis_agent/backups/<时间戳>/`：覆盖写入前的自动备份
- `workspace/.gis_agent/project_log.md`：跨任务的项目日志（注入下次提示词）

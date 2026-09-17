# GIS Agent 安装与使用（纯净发布版）

本仓库是可发布的纯净版：默认使用 `workspace/` 作为工作区，不含任何用户数据与密钥。

## 1. 环境要求

- Windows（推荐 Win10/11）
- **ArcGIS Pro 3.6**（含 `arcgispro-py3` Python 环境；ArcPy 为必需能力）
- 一个 OpenAI 兼容的大模型接口（硅基流动 / DeepSeek / 智谱 GLM / OpenAI …）

没有 ArcGIS Pro 时也能安装运行，但所有 ArcPy 相关步骤会失败并降级（只能做规划与代码生成）。

## 2. 安装

在仓库根目录执行（推荐使用 ArcGIS Pro 自带的 Python，避免与系统 Python 混装）：

```bash
"E:\ArcGISPro3.6\bin\Python\envs\arcgispro-py3\python.exe" -m pip install -e .
```

或系统 Python：

```bash
pip install -e .
```

关键依赖说明：

| 依赖 | 作用 | 缺失时表现 |
|---|---|---|
| `openai` | 调用大模型接口 | 无法运行 |
| `PyYAML` | 解析 GIS 配方（recipes） | 无法运行 |
| `jupyter-client` | ArcPy 持久内核（一次导入、跨步保持状态） | 自动降级为一次性子进程执行（较慢） |
| `Pillow` | 图片类验收断言（判断出图是否空白） | 图片断言跳过 |
| `python-docx` / `pypdf` | 读取任务书 docx/pdf | 文档解析类任务不可用 |

## 3. 初始化配置

```bash
cp config/llm_config.example.json config/llm_config.json
```

编辑 `config/llm_config.json`：

- 必填：`model`、`api_key`、`api_base`
- 建议：`fallback_models`（主模型失败时自动切换）、`routing_rules`（按任务类型分配快/强模型）
- 可选：`engine` 段调整引擎行为（见 README 参数表）

也可以不建文件，直接用环境变量提供密钥（部分客户端支持）：`GIS_LLM_API_KEY` / `OPENAI_API_KEY`。

`config/llm_config.json` 已在 `.gitignore` 中，不会被提交。

## 4. 启动（推荐）

双击仓库根目录的：

- `一键启动.bat`

脚本会自动：

1. 检测 Python 运行时（优先 ArcGIS Pro Python，失败则回退系统 Python）
2. 同步本地 editable 安装
3. 初始化工作区目录：`workspace/input`、`workspace/output`、`workspace/temp`、`workspace/skills`
4. 若缺少 `config/llm_config.json`，提示从示例复制
5. 进入对话模式：`gis-agent chat --workspace .\workspace`

引擎模式（推荐用于完成具体任务）：

```bash
一键启动.bat loop "把 county.shp 投影到 CGCS2000 111E 并统计每个地市的人口"
```

## 5. 数据放哪里

- 输入数据：`workspace/input/`
- 输出结果：`workspace/output/`
- 临时文件：`workspace/temp/`
- 自定义配方：`workspace/recipes/`（YAML，格式同内置配方）
- 自定义技能：`workspace/skills/`

运行时状态（自动创建，无需手工维护）：

```
workspace/.gis_agent/
├── catalog.db      数据目录（SQLite，自动增量扫描）
├── traces/         每次运行的 JSONL + Markdown 报告
├── backups/        破坏性操作前的自动备份
└── kernels/        ArcPy 持久内核运行时文件
```

## 6. 命令行用法

```bash
# 引擎模式（推荐）：观察-决策-执行循环 + 代码自修复 + 交付验收
gis-agent loop "任务描述" --workspace .\workspace

# 常用参数
gis-agent loop "任务描述" -w .\workspace -c .\config\llm_config.json --max-turns 40
gis-agent loop "任务描述" --no-refresh-catalog    # 跳过执行前的数据目录刷新

# 不走 PATH 时
python -m gis_cli.agent.cli loop "任务描述" --workspace .\workspace

# 对话模式 / 工具与技能查看
gis-agent chat --workspace .\workspace
gis-agent tools
gis-agent skills
```

## 7. 常见问题

**1. 启动后没看到数据**
确认数据已放到 `workspace/input/`；engine 会在运行时自动扫描并建立数据目录。

**2. 没有 ArcGIS Pro / ArcPy 不可用**
规划与代码生成仍可运行，涉及 ArcPy 的步骤会失败并给出提示。产出类任务请在装有 ArcGIS Pro 3.6 的机器执行。

**3. 模型调用失败（429 / Connection error）**
检查 `api_key`/`api_base`；网关限流时引擎会指数退避并尝试 `fallback_models`，单次调用有总时限（`engine.request_total_timeout_seconds`）。

**4. 代码执行卡住不动**
超过 `engine.kernel_exec_timeout_seconds` 会自动中断；若内核被 ArcPy 原生调用卡死，看门狗会硬杀进程，下一步自动重启新内核（日志出现 `kernel process destroyed` 属正常自愈）。

**5. 坐标系报错**
长度/面积/邻接类分析必须使用投影坐标系（如 `EPSG:4508` CGCS2000 高斯克吕格）。数据缺坐标系时用配方 `define_then_project`（先定义源坐标系，再真实投影）。

**6. 想回看某次运行过程**
打开 `workspace/.gis_agent/traces/` 下的 Markdown 报告（人类可读）或 JSONL（机器可读）。

**7. 想扩展能力**
在 `workspace/recipes/` 新增 YAML 配方（参考 `src/gis_cli/recipes/builtin/` 中的写法：`params` 声明参数、`code_template` 写代码、`validation` 写断言），启动时自动加载合并。
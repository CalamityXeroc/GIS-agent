# GIS Agent

用自然语言完成 ArcGIS Pro 的 GIS 分析与制图任务。

GIS Agent 是一个面向 **ArcGIS Pro 3.6** 的智能代理：你用中文描述需求（"把区县人口汇总到地市并做热点分析"），它自己查数据、定方法、写 ArcPy 代码并执行，出错自动修复，交付前自我验收。项目源自一名大学生在 AI 辅助下的实践，希望让非 GIS 专业的同学也能完成专业级地理分析。

> 仍在快速迭代中，欢迎提 issue 与建议；如果你对 GIS 或 Agent 感兴趣，欢迎一起完善这个项目。

---

## 一、能做什么

- **自然语言驱动**：描述目标即可，不需要记 ArcPy 函数名与参数
- **有界智能循环**：观察（看真实数据）→ 决策（选方法/配方）→ 执行（写代码跑 ArcPy）→ 校验，循环直到达标
- **代码自修复**：执行报错时把 traceback 回喂模型自动改代码重试（有次数上限）
- **方法论配方库**：内置 17 个经过验证的 GIS 配方（投影、拓扑修复、空间连接、邻接统计、汇总到分区、热点分析、普查指标、相似城市、分级设色出图…），直接调用即可，避免重复踩坑
- **数据目录感知**：自动扫描工作区数据（路径、几何类型、坐标系、要素数、字段名与样本），模型基于真实 schema 写代码而不是猜字段
- **交付前验收**：结构化断言 + 语义检查双层验收，不通过不算完成
- **安全护栏**：破坏性操作（删除/覆盖）执行前自动备份到 `.gis_agent/backups/`
- **全程可追溯**：每次运行写 JSONL + Markdown 报告到 `.gis_agent/traces/`，能看到每一步的决策与产出
- **稳定兜底**：ArcPy 持久内核执行（快）＋ 子进程兜底（稳）；内核卡死自动中断→硬杀→重启；网关限流自动退避

## 二、环境要求

- Windows + **ArcGIS Pro 3.6**（含 `arcgispro-py3` 环境；ArcPy 为必需能力）
- Python 3.10+（推荐直接用 ArcGIS Pro 自带解释器）
- 一个 OpenAI 兼容的大模型接口（硅基流动 / DeepSeek / 智谱 / OpenAI 等均可）

## 三、安装

在仓库根目录执行（推荐用 ArcGIS Pro 的 Python）：

```bash
"E:\ArcGISPro3.6\bin\Python\envs\arcgispro-py3\python.exe" -m pip install -e .
```

没有 ArcGIS Pro 时也可用系统 Python 安装，但 ArcPy 相关步骤会降级（只做规划与代码生成，不落地产出）。依赖说明：`jupyter-client` 用于 ArcPy 持久内核（缺失时自动改用一次性子进程执行）、`PyYAML` 用于配方解析、`Pillow` 用于图片类验收断言（可选）。

## 四、配置模型

```bash
cp config/llm_config.example.json config/llm_config.json
```

编辑 `config/llm_config.json`，至少填写 `model` / `api_key` / `api_base`：

```json
{
  "model": "Pro/zai-org/GLM-5",
  "api_key": "sk-你的密钥",
  "api_base": "https://api.siliconflow.cn/v1",
  "fallback_models": ["deepseek-chat"]
}
```

要点：

- `fallback_models`：主模型失败（限流/断连）时自动切换的备用模型
- `routing_rules`：按任务类型选模型（如 `agent` 用快模型、`code_repair` 用强模型），可显著提速
- `engine`：引擎行为参数，常用项如下

| 参数 | 默认 | 说明 |
|---|---|---|
| `max_turns` | 60 | 单任务最大循环轮次 |
| `max_repairs` | 4 | 单步代码自动修复次数上限 |
| `exploration_turn_budget` | 3 | 连续"只侦查不干活"的容忍轮数 |
| `escalate_after_failures` | 2 | 连续失败多少次后升级到强模型 |
| `kernel_exec_timeout_seconds` | 600 | 单次代码执行超时（超时自动中断/重启内核） |
| `auto_backup_destructive` | true | 破坏性操作前自动备份 |
| `backup_dir` | workspace/.gis_agent/backups | 备份目录 |
| `request_retry` / `request_backoff_seconds` | 6 / 8 | 网关重试次数与退避基数 |
| `request_total_timeout_seconds` | 420 | 单次模型调用（含重试）的总时限 |

> `config/llm_config.json` 已在 `.gitignore` 中，不会上传，密钥安全。

## 五、使用

### 1. 准备数据

把数据放进 `workspace/input/`（shapefile / 文件地理数据库 / 栅格 / CSV 均可），结果会写到 `workspace/output/`。

### 2. 引擎模式（推荐）

```bash
gis-agent loop "把 county.shp 定义坐标系并投影到 CGCS2000 111E，再按地市汇总人口"
```

或用 ArcGIS Pro Python 直接调用模块：

```bash
"E:\ArcGISPro3.6\bin\Python\envs\arcgispro-py3\python.exe" -m gis_cli.agent.cli loop "制图：按人口做分级设色并导出 JPG" --workspace .\workspace
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--workspace/-w` | 工作区目录（默认当前目录） |
| `--config/-c` | 指定 `llm_config.json` 路径 |
| `--max-turns` | 覆盖最大轮次 |
| `--refresh-catalog/--no-refresh-catalog` | 执行前是否增量刷新数据目录（默认刷新） |

运行中会实时打印每一步：调用的工具/配方、执行结果、自动修复、最终验收；结束时输出任务清单、产出文件、方法论与追踪文件位置。

### 3. 一键启动（对话模式）

双击 `一键启动.bat`（自动检测 Python、同步安装、初始化 `workspace/`，然后进入对话模式）：

```bash
gis-agent chat --workspace .\workspace
```

也支持直接给参数走引擎模式：

```bash
一键启动.bat loop "统计每个地市的历年总人口并做热点分析"
```

### 4. 其他命令

```bash
gis-agent tools        # 列出可用工具
gis-agent skills       # 列出可用技能
gis-agent status       # 查看 Agent 状态
gis-agent --help       # 全部命令
```

## 六、工作原理

```
用户任务
   │
   ├─ 需求抽取（无需求文档时自动从任务描述抽取验收标准）
   ├─ 数据目录扫描（真实字段名/坐标系/要素数，注入上下文）
   │
   └─ 有界循环（最多 max_turns 轮）
        观察 → 决策 → 行动（run_recipe 用配方 / execute_code 写代码）
                 ↑                    │
                 └── 失败则自动修复 ←──┘
        交付 → 验收（结构断言 + 语义检查）→ 通过则结束
```

关键设计：

- **执行用 ArcPy 持久内核**：arcpy 只导入一次，跨步骤保持变量状态，执行快；超时自动中断，卡死则硬杀重启
- **配方优先**：内置配方已把常见 GIS 方法固化成可复用单元，自带参数校验与断言，避免模型重复生成易错代码
- **自修复有界**：修复次数上限 + 修复失败升级强模型，不会无限循环
- **备份可回溯**：破坏性操作前自动复制到 `.gis_agent/backups/<时间戳>/`

## 七、内置配方（17 个）

用 `gis-agent loop` 时模型会自动检索调用；也可以直接指定"用 xx 配方"。

| 配方 ID | 用途 |
|---|---|
| `define_then_project` | 坐标系丢失时先定义再真实投影 |
| `project_to_crs` | 投影变换到目标坐标系 |
| `topology_repair_keep_attrs` | 找出重叠面并去重（保留属性） |
| `csv_points_spatial_join` | CSV 经纬度转点、投影、与面空间连接 |
| `attribute_join_table` | 按公共字段把表挂接到图层 |
| `field_calculate` | 新增/计算字段（含表达式） |
| `spatial_join_summary` | 空间连接并统计数量 |
| `polygon_neighbor_stats` | 邻接关系求取 + 邻域均值 + 高于均值标记 |
| `aggregate_to_zones` | 按空间归属把源要素汇总到分区面（SUM/MEAN 等） |
| `summarize_stats` | 按分组字段汇总统计表 |
| `hotspot_analysis` | Getis-Ord Gi* 热点/冷点分析 |
| `local_morans_i` | Anselin Local Moran's I 局部自相关 |
| `census_city_indicators` | 占比类指标按人口加权汇总到上级区划 |
| `similarity_ranking` | 多指标标准化求最相似单元（写 JSON 结果） |
| `zonal_statistics` | 分区统计栅格（均值/总和等） |
| `buffer_dissolve` | 缓冲区与融合 |
| `graduated_colors_map` | 从零建工程→分级设色→四要素→导出 JPG/PDF |

**自定义配方**：把 YAML 放进 `workspace/recipes/`，格式与内置配方一致（`id/name/params/code_template/validation`），启动时自动合并加载。配方模板中 `{{param}}` 会替换为 Python 字面量、`@@param@@` 替换为原文。

## 八、目录与产物

```
workspace/
├── input/            输入数据（自备）
├── output/           全部产出（GDB、JPG、JSON…）
├── recipes/          自定义配方（可选）
├── skills/           自定义技能（可选）
└── .gis_agent/       引擎运行状态
    ├── catalog.db    数据目录（SQLite）
    ├── traces/       每次运行的 JSONL + Markdown 报告
    ├── backups/      破坏性操作前的自动备份
    └── kernels/      ArcPy 持久内核运行时
```

## 九、常见问题

**1. 找不到数据 / 字段名报错**
确认数据在 `workspace/input/`；引擎会自动扫描数据目录，也可以在任务里写明"先用 catalog 查一下字段"。

**2. 没有 ArcGIS Pro**
仍可运行对话与规划，但 ArcPy 步骤会失败并降级；产出类任务必须在有 ArcGIS Pro 的机器上跑。

**3. 模型报错：429 / Connection error**
网关限流或断连，引擎会指数退避并在必要时切换 `fallback_models`；全球调用总时限 `request_total_timeout_seconds` 兜底，不会无限等待。建议配置一个稳定的备用模型。

**4. 执行很久没反应**
单步执行超过 `kernel_exec_timeout_seconds` 会强制中断；若内核被 native 调用卡死，看门狗会硬杀进程并在下一步自动重启（日志里会有 `kernel process destroyed` 提示）。

**5. 坐标系相关报错**
分析前必须投影到投影坐标系（如 CGCS2000 高斯克吕格 `EPSG:4508`）。丢坐标系的数据用 `define_then_project`；不要用"定义坐标系"代替"投影"。

**6. 想复现某次运行**
`.gis_agent/traces/` 下有每次运行的 Markdown 报告（决策链、配方调用、断言结果）与 JSONL 原始事件流。

## 十、旧版命令（保留）

仓库同时保留了早期版本的 `gis-cli` 命令族与 BAML 多模型适配能力（任务创建/规划/重跑建议/统计导出等），仍可正常使用：

```bash
pip install -e .
gis-cli --help        # 任务管理相关命令（task-create / task-plan / task-run ...）
gis-agent chat        # 对话模式
gis-agent loop "..."  # 引擎模式（推荐）
```

新任务建议直接用 `gis-agent loop`（配方库 + 自修复 + 验收），旧命令主要面向历史脚本兼容。

## 十一、说明文档

- 安装、依赖与故障排查：`INSTALL.md`
- 模型配置示例：`config/llm_config.example.json`

## 许可与致谢

本项目为学习与实践性质的开源项目，使用请遵守 ArcGIS Pro 及相关服务的许可条款。部分 `workspace/skills/` 下的技能文档来自公开的技能库，版权归原作者所有。
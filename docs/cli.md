# 命令行界面（CLI 使用与设计）

面向日常使用的交互说明。核心目标是让"跑一个 GIS 任务"这件事**看得见过程、随时能纠偏、结果一眼可查**。

## 启动

```bash
# 交互对话（推荐）
gis-agent chat --workspace .\workspace

# 接续上一次会话
gis-agent chat --continue

# 单次任务（不进交互）
gis-agent loop "把 county.shp 定义坐标系并投影到 CGCS2000 111E"

# 启动时就把数据目录扫全（默认不扫，见下）
gis-agent chat --refresh-catalog

# 环境自检：arcpy 能不能 import、能不能真跑一行代码（--deep 连内核一起验）
gis-agent doctor

# 一键启动（Windows，自动装依赖并进入对话）
一键启动.bat
```

### 跑之前先体检：`doctor`

```bash
gis-agent doctor          # 7 项，约 10~20 秒
gis-agent doctor --deep   # 额外启动一次持久内核
```

| 检查 | 判据 |
|---|---|
| `pro_installed` | 找到 ArcGIS Pro 的 python（环境变量 → 注册表 → 默认路径） |
| `arcpy_importable` | 用那个 python 真的 `import arcpy` |
| `pro_version` | ≥ 3.2 |
| `deps` | 9 个核心包都能导入（失败时拼一条可直接复制的 `pip install`） |
| `path_spaces` / `path_nonascii` | 工作目录与 Pro 环境路径无空格、纯 ASCII |
| `kernel_start` | 持久内核能起来（仅 `--deep`） |

**刻意不检查许可证**（既无用又冒犯），也不引导安装、不自动改环境。详见 [`p0-features.md`](p0-features.md)。

### 计划卡：执行前先让你看懂

首轮执行前会打印一张 5 行计划卡（做什么 / 用什么 / 输出 / 怎么验 / 合规）：

- **TTY**：`s` 开始、`e` 改一句（改完重新渲染）、`q` 取消
- **非 TTY / 脚本 / CI**：自动通过，但卡**照样打印**（日志里能看到当时判断）
- 「用什么」从数据目录匹配而来，匹配不到就写「执行时先确认」，不编


### 启动为什么很快，以及数据目录何时扫

**启动只加载配方与配置**（实测 0.2~0.3s），**不扫数据目录**。你发出第一条
任务后，下面两件事在后台并行做，与首轮模型调用重叠：

1. **扫描数据目录**（下次启动就命中缓存，通常 < 0.1s）；
2. **启动 ArcPy 内核**（大头是 `import arcpy`，实测 11~17s）。

这期间界面已经可用。模型需要真实字段名时会调用 `catalog_query`，它会
等扫描完成再答（对模型透明，不会拿到“空目录”然后乱猜字段）。

想恢复“启动就扫全”的旧行为：加 `--refresh-catalog`。

### 流式输出

模型边生成边显示：**思考与回复实时追加**，状态行同时显示字数
（`思考中 · 340 字`），不必盯着空屏等整段生成完。

实测（deepseek-v4.1-flash-intl）：首字 2~6s 到屏，此前要等整段（10~90s）。
上游不支持流式时会自动退回非流式（功能不变，只是拿不到逐字输出）。

## 界面长什么样

```
┌─ GIS Agent v0.7.1 ───────────────────────────────────────────────┐
│     模型  glm-5.3-flash · tokenrhythm.studio                      │
│   工作区  ./workspace                                             │
│ 数据目录  vector 2、table 1                                        │
│     配方  47 个                                                   │
│     会话  20261003-1045                                           │
└──────────────────────────────────────────────────────────────────┘

任务  把动物点位做核密度，出四张统一图例的图

⏺ catalog_query  keyword=animal_points  0s
└ ✓ 数据目录查询：2 条匹配
⏺ run_recipe  recipe_id=kernel_density periods=1982-1986,…  12s
└ ✓ 4 期核密度 → output/result_data.gdb/kde_*
! 自动修复第 1 次  KeyError '_oh' → 改用显式变量
验收 通过 ✓  7/7 断言

┌─ 结果 ───────────────────────────────────────────────────────────┐
│ 产出                                                              │
│   • output/成果图/密度_1982_1986.jpg                              │
│ 方法论  核密度（邻域 100km² → 半径 √(A/π)≈5642m）                  │
│ 状态 completed · 轮次 24                                          │
└──────────────────────────────────────────────────────────────────┘
glm-5.3-flash · 已完成 · 第 24 轮 · 6m12s · ↑31.8k ↓6.4k · 工具 11
```

三段结构：**滚动转录**（提问/决策/工具动作/结果/验收/错误）+ **底部输入框** + **状态行**。

## 输入框

真终端下是一个**四边完整**的输入框，右上角带会话状态，底边带快捷键提示：

```text
╭─ 输入 ──────────────────── deepseek-flash · 空闲 · 第 0 轮 ─╮
│  › /mo
│       /model     查看或切换模型 <值>
╰─ Enter 发送 · Shift+Enter 换行 · / 命令 · Ctrl+O 明细 ──────╯
```

> 早期版本只有上边框（用户反馈“输入框只有上半部分”）。原因：下边框当时用 `bottom_toolbar`
> 画，而 prompt_toolkit 对它加了 `renderer_height_is_known` 门禁，需要终端回应
> CPR（光标位置查询）；**终端不回 CPR 时下边框永远不渲染**。现在改成自建布局
> （`ui/tui.py`），下边框是普通 Window，不依赖那个门禁；左右竖线用固定宽度栏，
> 所以**补全菜单那几行也有边界**。

输入框本身由 `prompt_toolkit` 提供（**已列为必装依赖**）；
如果环境里实在没有它，启动时会打一行警告并降级为纯文本输入，而不是默默只显示一个 `›`。

| 按键 | 作用 |
|---|---|
| `Enter` | 发送（**补全菜单开着时，先应用当前候选**，再按一次才发送） |
| `Shift+Enter` / `Alt+Enter` / `Ctrl+J` | 换行（写多行任务） |
| `↑` `↓` | 补全菜单中移动选择；没有菜单时翻历史 |
| `Tab` | 补全命令 / **命令参数** / 路径 |
| `Esc` | 关掉补全菜单；菜单已关时**中断当前任务**（空闲时无动作） |
| `@` | 引用工作区文件（自动列出 `workspace/input` 里的数据） |
| `Ctrl+O` | 展开/收起最近一次工具的长输出 |
| `Ctrl+R` | 反向增量搜索历史 |
| `Ctrl+L` | 清屏 |
| `Ctrl+C` | 清空当前输入；空行时退出 |
| `Ctrl+D` | 退出 |

### 斜杠命令选择器

输入 `/` 时，**光标下方会就地列出全部命令**（命令名 + 灰色说明两列），边打边过滤：

```text
│  › /
│       /help      /theme     /workspace /tools     /recipes
│       /model     /color     /status    /skills    /trace     >
```

`↑` `↓` 移动高亮，`Enter` 应用，`Esc` 取消。

**参数也有菜单**（与 pi / Claude Code 一致）：输入命令加空格后列出该参数的候选——

| 输入 | 候选 |
|---|---|
| `/theme ` | `system` / `dark` / `light` |
| `/color ` | `on` / `off` |
| `/model ` | 可用模型名（配置里的 + 上次 `/model` 查到的在线清单） |
| `/resume ` | 最近会话 id |

命令的**名字 / 说明 / 参数规格**只在 `ui/commands.py:COMMANDS` 写一次，
菜单、`/help`、分发都从它生成——不会再出现“菜单里能选、实际不认”的漂移。

运行中**仍然可以继续输入**：回车会把内容作为"纠偏/补充"注入当前任务（引擎在下一轮读到）；
如果想排队到这条任务之后再执行，等它结束再输入即可（排队消息会在下一轮开始时带上）。

## 斜杠命令

> 输入 `/` 即可从菜单里选，下面这张表是全部可选项。

| 命令 | 说明 |
|---|---|
| `/help` | 命令与快捷键 |
| `/status` | 会话、模型、数据目录、日志路径 |
| `/model` | **列出并切换模型**（无参数则打印清单；切换即时生效并写回 `config/llm_config.json`） |
| `/theme` | 切换配色：`system` / `dark` / `light` |
| `/color` | 开关颜色：`on` / `off` |
| `/workspace` | 当前工作区 |
| `/tools` | 工具清单（名字 + 描述 + 参数） |
| `/skills` | 技能清单（名字 + 描述；扫描工作区与全局 skills 目录） |
| `/recipes` | 内置配方 |
| `/skill:名字 [要求]` | **手动加载某个技能**（把 SKILL.md 全文交给 agent 执行） |
| `/session` | 本会话最近记录 |
| `/resume` | 列出并切换会话（续聊） |
| `/trace` | 最近一次运行的追踪文件（`/trace full` 看末尾 80 行） |
| `/export` | 导出本次转录为 Markdown |
| `/clear` | 清屏 |
| `/exit` | 退出（别名 `/quit`、`/q`） |

## 技能（SKILL.md）

技能是“一套专项做法的说明书”（怎么写 Word、怎么读 PDF、怎么做某类 GIS 分析），
遵循 **Agent Skills 标准**：一个目录里放 `SKILL.md`，frontmatter 写 `name` 与 `description`，
正文写做法与脚本位置。

```
workspace/skills/docx/
├── SKILL.md          # frontmatter: name / description；正文：怎么做
└── scripts/          # 技能自带的脚本（相对技能目录引用）
```

### 两种触发方式

1. **自然语言自动触发**（主要方式）：
   技能的名字 + 描述常驻 system prompt，模型看到你的需求与某条描述吻合时，
   会自己调 `run_skill` 读全文再照做。例：

   ```text
   你：帮我写一个 Word 文档，介绍什么是缓冲区分析
   ◆ 思考  现在加载 docx 技能
   ⏺ run_skill  name=docx  • 已加载技能 docx（17113 字符…）
   ⏺ execute_code … → output/GIS简介.docx
   ```

2. **手动指定**：`/skill:docx 帮我写一份简介`（把技能全文直接交给 agent）。

> 描述决定“什么时候会被想起”，所以写具体（“创建/读取/编辑 Word 文档（.docx）”、
> “处理 PDF 表单”），而不是“帮助处理文档”。

### 技能放哪里

按下列顺序扫描（重名保留先找到的）：

| 位置 | 用途 |
|---|---|
| `<工作区>/skills/` | 当前工作区自己的技能 |
| `<工作区>/../skills/` | 上级目录（仓库级） |
| `~/.agents/skills/`、`~/.claude/skills/`、`~/.codex/skills/`、`~/.pi/agent/skills/` | 复用其他 harness 的技能 |

两种写法都认：目录形式 `docx/SKILL.md`，或技能根下的裸 .md（如
`buffer_analysis.md`）。**把自己从别处拿到的技能目录拷进来就能用，不需要改代码。**

### 相关工具

| 工具 | 作用 |
|---|---|
| `run_skill` | 把一个技能的 SKILL.md 全文读进对话 |
| `read_file` | 读任意文本文件（技能正文、技能自带脚本、配置） |
| `execute_code` | 跑技能里的脚本（路径相对技能目录） |

> 为什么需要这三个：技能正文和脚本都是普通文本，而原来的 `read_document`
> 只认 docx/pdf/txt，**连 SKILL.md 都读不了**——所以光“告诉模型有技能”没用。

## 会话与续聊

- 转录自动落盘：`workspace/.gis_agent/sessions/<会话id>.jsonl`
- 引擎日志（模型重试、修复细节）：`workspace/.gis_agent/logs/engine.log`
- 追踪（每轮决策与产物）：`workspace/.gis_agent/traces/`
- `--continue` 或 `/resume` 续聊：上一次会话会被压成一段"上下文前言"拼在新任务前面
  （不做对话反序列化，避免把历史错误状态带进新任务）
- `/export` 导出 Markdown，便于贴给同学或存档

## 配色

- 默认 **`system`**：跟随终端明暗（读 `COLORFGBG`），取不到时按深色处理；
- `--theme dark|light` 或 `/theme` 可显式指定，也可用环境变量 `GIS_AGENT_THEME=light`；
- `--no-color` 或 `NO_COLOR=1` 关闭颜色（老终端、重定向到文件时保留结构、去掉颜色）；
- 颜色按**角色**分配（提示/助手/工具/成功/警告/错误/次要/强调/路径/数值），想整体换风格只改一处：
  `src/gis_cli/ui/theme.py` 里的 `_DARK` / `_LIGHT`。

## 相关代码

| 文件 | 职责 |
|---|---|
| `src/gis_cli/ui/theme.py` | 主题、Console 工厂、明暗判定 |
| `src/gis_cli/ui/events.py` | 引擎事件 → 渲染单元（纯逻辑）+ 状态行累积 |
| `src/gis_cli/ui/render.py` | 横幅 / 结果卡片 / 状态行 / 折叠输出 |
| `src/gis_cli/ui/prompt.py` | 输入框（多行、历史、`/` 与 `@` 补全、Esc 中断） |
| `src/gis_cli/ui/session_runner.py` | 工作线程 + 纠偏注入 + 中断 |
| `src/gis_cli/ui/session_store.py` | 会话落盘、导出、续聊前言 |
| `src/gis_cli/agent/interactive.py` | 交互主循环与斜杠命令 |
| `src/gis_cli/datacatalog/prewarm.py` | 数据目录后台扫描（启动不阻塞） |
| `src/gis_cli/runtime/timing.py` | 分段计时（`[timing]` 日志，查“慢在哪”用） |
| `scripts/preview_ui.py` | 纯渲染预览（改样式时先跑它看效果） |
| `scripts/latency_report.py` | 从 trace 算每段耗时占比（模型等待/工具/压缩） |

## 自测

```bash
# 渲染层纯逻辑（无需 TTY）
PYTHONPATH=src python -m pytest tests/test_ui_render.py tests/test_ui_session.py tests/test_ui_session_store.py -q

# 输入框与交互全流程（需要 prompt_toolkit，用 ArcGIS 解释器）
PYTHONPATH=src "E:\ArcGISPro3.6\bin\Python\envs\arcgispro-py3\python.exe" -m pytest tests/test_ui_prompt.py tests/test_ui_chat_flow.py -q

# 只看样式效果
PYTHONPATH=src python scripts/preview_ui.py --no-color --width 96
```

## 控制台兼容（乱码排查）

**事故复盘**：用户控制台上出现满屏 `?[38;5;246m` 之类怪字符（"246m"就是 ANSI 色号 246），
中文本身正常。原因是 **rich 判定终端支持 ANSI，但该终端并不解析转义序列**。
第一版修复用"能否开启 VT"来判断 —— 在那个环境里判断为"能"，于是问题依旧。

**现在的策略：默认纯文本**（不靠探测，靠"只认明确认得的高级终端"）：

| 终端 | 行为 |
|---|---|
| Windows Terminal / VS Code / ConEmu / mintty / WezTerm 等 | 自动上色 |
| 普通 cmd / PowerShell / 认不出的终端 | **纯文本**（一个转义符都不输出） |
| 输出被重定向/管道 | 纯文本（即使是 `--color` 也不打颜色码） |

开关（任选）：

| 开关 | 作用 |
|---|---|
| `--color` / `GIS_AGENT_FORCE_COLOR=1` | 强制上色（明知终端支持 ANSI 时用） |
| `--no-color` / `NO_COLOR=1` / `GIS_AGENT_PLAIN=1` | 强制纯文本 |
| `GIS_AGENT_ASCII=1` | 符号也降级为 ASCII（`⏺ ─ ✓` → `* - v`） |
| `/color on\|off` | 会话内随时切换 |

另外两条同样重要的规则：

1. **编码跟随控制台**：能切 UTF-8 就切（`SetConsoleOutputCP(65001)`），切不动就按控制台
   自己的编码输出并 `errors="replace"`。⚠️ 早期无条件把 stdout 重配置成 UTF-8，
   在 GBK 控制台上会把 UTF-8 字节按 GBK 解码 → 整屏乱码。
2. **符号自动降级**：老终端/字体缺字形时自动换 ASCII。

`/status` 会打印一行诊断，排障时把它贴出来即可：

```
控制台 windows-console · 代码页 65001 · 输出编码 utf-8 · 颜色 off · 符号 unicode
```

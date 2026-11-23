# P2 Web 可视化控制台 —— 可行性结论 + 实施计划（v2）

> 修订依据 DAIR-233 用户回复（2026-11-10）：① 本机自用、只要可视化管理台；② Node 可用；
> ③ 地图先离线；④ **UI/UX 严格对齐 `QuantumNous/new-api`，可直接照抄**。
> 参照仓已 clone 到本机 `E:\BST_lab\_refs\new-api`（`--depth 1`，41 MB，可直接打开看）。
> 代码盘点基准：`e344d341`。**本文件不含实现代码。**

## 0. 这一版改了什么（相对 v1）

| 项 | v1 | v2 | 原因 |
|---|---|---|---|
| 前端框架 | Vue 3 | **React 19 + Tailwind 4 + shadcn/ui** | 照抄要求 → 必须同栈。new-api 就是这套（`components.json` 里 `style: base-nova`、`iconLibrary: hugeicons`），换 Vue 等于"重新翻译"，样式必飘 |
| 前端产物 | 产物进仓 / 独立仓 二选一 | **产物必须进仓**，Node 只做构建期 | 你要"本机就能用"：`pip install` 完直接 `gis-agent web`，不该为了界面去装 Node |
| 地图 | 离线默认 + 在线可选 | **只做离线** | 你的决定；顺带避开底图合规 |
| 鉴权 | 单机/局域网 二选一 | **只做 `127.0.0.1` + 本机 token** | 你的决定；不做用户体系 |
| 图表库 | 手写 SVG | **VChart**（与 new-api 同款） | 照抄堆叠柱/面积图配置才有一致观感 |
| 工程量 | 20 人日 / 5 波 | **24 人日 / 5 波** | 多出的 4 人日全部是设计系统 + 视觉对齐 |

**结论没变：可行，且比 v1 更有把握**——因为现在不只是"数据都有"，连"长什么样"都有现成参照可逐项核对。

## 1. 后端可行性（v1 实测结论，压缩保留）

| 控制台要展示的 | 落点 | 状态 |
|---|---|---|
| 实时运行态 | `engine/loop.py:1707 _emit()` 发 **22 种事件**（含 `llm_delta`、`tool_call/tool_result`、`verification`、`plan_card`），接入点 `build_engine(on_event=...)` | ✅ |
| 停止 / 插话 / 续跑 | `inject_user_message()` `loop.py:1410`、`request_stop()` `:1429`、`resume_from_checkpoint()` `:398` | ✅ |
| 运行日志 | `<ws>/.gis_agent/traces/<run_id>.jsonl|.md`，workspace 现有 **26 个**；payload 带 `usage/model/repairs/verification.problems/duration_seconds` | ✅ |
| token / 成本 | `llm_client.usage_totals`（`llm_client.py:318`）+ 每条 `llm_result.usage` | ✅ |
| 数据资产 | catalog SQLite：`datasets(wkid/geom_type/feature_count/extent)`、`fields(samples/null_ratio)`、`rasters(min/max/mean/cell_size)` | ✅ |
| 配置模型 | `config/llm_config.json`：9 个角色的 `routing_rules` + `engine` 预算段 | ✅ 缺写回 |
| 地图预览（离线） | `report/html_pack.py` 零外链 canvas 画内联 GeoJSON，配 `thin_features()` 抽稀、`bbox_of()` | ✅ |
| HTTP/SSE 服务层 | 旧 `legacy/core_tools/web/api.py` 已随旧引擎下线且被 `pyproject` 排除 | ❌ 新写 |
| 单 worker 队列 | 无 | ❌ 新写 |

两个定架构的事实（v1 已论证，不变）：
1. `ui/events.py` 是"纯逻辑、不碰 TTY"，`StatusTracker` 连线程归属都写好了 → 服务端直接复用，不碰 rich/终端。
2. `ArcpyKernel` 单例 + `_exec_lock` + `llm_max_concurrency=1` + ArcGIS 单实例 → **只能单 worker + 可见队列**。

## 2. 照抄规格（从参照仓抄出来的硬数值，实现前先对齐这一节）

### 2.1 布局骨架

| 部件 | 规格 | 出处（参照仓） |
|---|---|---|
| 顶部 header | 高 `3rem`（48px），sticky，透明底，内含 logo + 版本 badge + 横排导航 + 搜索框（带 `⌘K`）+ 图标组 + 头像 | `web/src/styles/theme.css:99`、`layout/components/header.tsx:28` |
| 左侧栏 | 宽 `13rem`（208px），移动端 `17rem`，折叠图标轨 `2.75rem`；固定在 header 之下 | `web/src/components/ui/sidebar.tsx:50-52,253` |
| 侧栏分组 | 小标题（如「常规」「个人」「管理员」）12px 灰字；项 = 16px 图标 + 文字；选中态 = 淡蓝底圆角块 | 见截图 `assets/screenshots/dashboard.zh-CN.jpg` |
| 内容区 | 页面左上大标题（~20px semibold）+ 右侧操作区；卡片纵向堆叠，间距 ~16px | 同上 |

### 2.2 颜色 token（OKLCH 原值，照抄）

| token | 浅色 | 深色 |
|---|---|---|
| `--primary` | `oklch(0.692 0.141 243.716)` | `oklch(0.54 0.142 248.516)` |
| `--accent`（选中态/悬浮底） | `color-mix(in oklch, var(--primary) 12%, var(--background))` | primary 20% |
| `--background` / `--card` | `oklch(1 0 0)` / `oklch(1 0 0)` | `oklch(0.235 0 0)` / `oklch(0.285 0 0)` |
| `--muted-foreground` | `oklch(0.49 0 0)` | `oklch(0.78 0 0)` |
| `--border` | `oklch(0.93 0 0)` | `oklch(1 0 0 / 10%)` |
| `--success` / `--warning` / `--info` | `0.596 0.145 163.225` / `0.681 0.162 75.834` / `0.588 0.158 241.966` | 同 |
| `--chart-1…5` | `0.72 0.18 250` / `0.65 0.15 200` / `0.7 0.12 280` / `0.68 0.19 325` / `0.68 0.16 155` | 同 |
| `--overview-accent-1…3` | 同 chart-1/2/3 | 同 |

深色模式靠 `.dark` 类切换（参照仓用 `next-themes`）；浅色主题下 `--background` 与 `--card` 都接近纯白，
**层次靠 1px `--border` 描边**而不是阴影——这是它整体观感"干净"的关键，别加投影。

### 2.3 圆角 / 字体 / 图标

- `--radius: 1rem`，派生 `0.6/0.8/1/1.4/1.8/2.2/2.6 ×`（卡片/输入框用 `lg`，按钮用 `md`，chip 用 `sm`）。
- 字体：正文 **Public Sans Variable**；`serif` 轴是 Lora（带完整 CJK 回退链 Noto Serif SC → 宋体）；
  **所有字体走 `@fontsource-variable` npm 包内联进产物，不引 CDN**（离线要求）。
- 图标：`lucide-react`，行内 16px、卡片标题旁 18px。

### 2.4 要照抄的组件（12 个，正好覆盖我们全部页面）

| 组件 | 作用 | 参照 |
|---|---|---|
| `AppSidebar` + `NavGroup` + `NavLinkItem` | 分组侧栏 + 选中态 | `components/layout/components/app-sidebar.tsx` |
| `Header` + `HeaderLogo` | 顶部条 | `layout/components/header.tsx` |
| `StatCard`（含 sparkline/icon badge） | KPI 卡 | `features/dashboard/components/ui/stat-card.tsx` |
| `TitledCard` | 带"图标 + 标题 + 右侧总计"的卡片壳 | `components/ui/titled-card.tsx` |
| `IconBadge` | 卡片标题前的小色块图标 | `components/ui/icon-badge.tsx` |
| `DataTable`（TanStack Table 封装：工具栏/列显隐/分页/空态） | 运行历史、数据资产 | `components/data-table/*` |
| `StatChips`（健康条里那排内联指标） | 内核/环境健康条 | 截图第 2 行 |
| `PageHeader`（标题 + 右侧动作/segmented） | 每页页头 | `layout/components/section-page-layout.tsx` |
| `Skeleton` / `Empty` | 加载态与空态 | `components/ui/skeleton.tsx`、`empty.tsx` |
| `Sonner` toast | 操作反馈 | `components/ui/sonner.tsx` |
| `Tabs` / `ToggleGroup` | 图表切换（柱状图/面积图） | `components/ui/{tabs,toggle-group}.tsx` |
| `Dialog` / `Sheet` / `Drawer` | 计划卡、trace 详情抽屉 | `components/ui/*` |

### 2.5 从截图里要复刻的 6 条"观感规则"

1. KPI 行是**一张卡片切 5 格**（格间 1px 竖线），不是 5 张卡；数字 ~30px，副标 12px 灰字。
2. 健康条是**一张卡片一行**：绿点 + 名称 + 一排内联指标 chip + 模型色点列表。
3. 图表卡右上角永远是 segmented control（柱状图 / 面积图），图例居中贴底。
4. 表格**没有竖线**，行间细线；单元格两行（主值 + 次级灰字/彩色小字），数字右对齐且 `tabular-nums`。
5. 表格底部固定「总计 N 条 / 每页行数 / 分页器」三件套。
6. 筛选区是一张独立卡片：第一行控件、第二行 chips + 右侧「重置 / 搜索（主色 pill）/ 查看」。

## 3. 页面映射（只借设计语言，页面按本项目改造）

参照仓里的**模型广场 / 渠道 / 兑换码 / 钱包 / 订阅 / 排行榜**这些业务页 **一律不做**（本项目没有对应业务）。
只借它们的**版式与组件模式**，落到本项目真实存在的五块内容上：

| 参照页（只借版式） | 我们的页 | 复用哪些部件 | 数据来自 |
|---|---|---|---|
| 数据看板（模型调用分析） | **① 总览** | StatCard×5 + 健康条 + 按天堆叠柱 + 趋势图 | trace 聚合 + `usage_totals` + catalog counts |
| 使用日志（筛选卡 + 两行单元格表格） | **③ 运行历史** | 筛选卡 + DataTable + 行展开 + 原始 JSONL 下载 | `traces/*.jsonl|.md` |
| 游乐场（流式输出面板） | **② 运行详情** | 时间线（滚动容器 + 粘底）+ 状态条 + 计划卡 Dialog + 离线地图区 | **SSE 实时事件** + `html_pack` canvas |
| 模型广场的卡片网格（只借网格形态） | **④ 数据资产** | 卡片/表格双视图 + 详情抽屉 + 图层预览 | catalog SQLite |
| 系统设置（二级导航 + 表单） | **⑤ 配置模型** | 表单 + diff 预览 + 保存前确认 | `config/llm_config.json` |

导航分组（本项目的三段式，参照仓的分组名一律不用）：

```
运行      概览 · 运行历史 · 新建运行
数据      数据资产 · 成果包
系统      配置模型 · 环境自检
```

## 4. 照抄的边界（许可证，必须先说清）

参照仓是 **AGPL-3.0**（`LICENSE`，661 行，含商业授权另议条款）。含义：

- **设计语言（配色/尺寸/圆角/布局/交互模式）不是"可版权的表达"**，照着做自己的实现没问题；
- **源码文件是**。把 `web/src/**` 的 `.tsx` 复制进我们仓库，就继承了 AGPL——将来一旦对外分发二进制或提供网络服务，整个项目要按 AGPL 开源。
- 本项目现在是**本机自用、不分发**，AGPL 的触发条件在实践中不成立；但代码一旦进 git，将来想闭源发布就要回头重写。

**决定（2026-11-10）：走 A（clean-room 复刻）**。用户要的是"按我们项目情况改造"，不是照搬它的业务页；
所以 §2 的数值规格照抄，组件与页面代码全部自己写，**不复制 `web/src/**` 任何文件**。
好处：零许可证牵连，页面也不会被参照仓的业务概念带偏。

## 5. 技术栈

| 层 | 选型 | 理由 / 替代 |
|---|---|---|
| 前端 | **React 19 + TypeScript + Vite + Tailwind 4 + shadcn/ui(base-nova) + lucide-react** | 与参照仓同栈；Vite 而非参照仓的 rsbuild：Windows 上更稳、不依赖 bun（参照仓是 `bun.lock`） |
| 路由/数据 | **TanStack Router + TanStack Query** | 同参照仓；Query 天然适配"轮询聚合指标 + SSE 增量" |
| 状态 | **zustand**（只放本地 UI 状态） | 同参照仓；不在前端存运行态（服务端才是真相） |
| 表格 | **TanStack Table**（照抄其 data-table 封装思路） | 排序/筛选/列显隐/分页一次到位 |
| 图表 | **@visactor/react-vchart** | 同参照仓；堆叠柱/面积图配置可直接对照 |
| 后端 | **Python FastAPI + Starlette `StreamingResponse`（SSE）** | 见下 |
| 构建 | `npm` + Vite，产物提交到 `src/gis_cli/web/static/` | 运行期零 Node |

**关于"Node 效果会不会更好"——两种读法都答一下：**

- **Node 做前端工具链 → 采纳**。Vite/Tailwind/shadcn 全是 Node 生态，这一层已经用上了。
- **Node 做后端 → 不建议**，代价：
  1. 引擎是 Python，Node 后端只能 `subprocess` 起 CLI，`on_event` 的进程内回调就变成"解析子进程 stdout/JSONL"——多一层协议要自己维护；
  2. 停止/插话/续跑三个钩子是**进程内方法**（`request_stop` / `inject_user_message` / `resume_from_checkpoint`），跨进程要自己发明 IPC；
  3. 单内核 + 单 worker 的约束不会因为换语言而消失，Node 换不来并发。
  → Node 后端能带来的只有"一套语言"，代价是 3 处新增故障面 + 至少 3 人日。**建议：Node 只做前端。**

## 6. 后端架构（相对 v1 无变化，此处只列结论）

```
浏览器（React SPA 静态产物）
   │  EventSource /api/runs/{id}/events?since=<seq>
   ▼
FastAPI（127.0.0.1 + 本机 token）
   ├─ JobManager：单 worker 线程 + deque 队列 + jobs.json（重启可查历史）
   ├─ EventBus：per-run 环形缓冲（2000 条 + seq），llm_delta 100ms 聚合
   └─ API（除配置写入外只读）：/api/traces /api/catalog /api/config /api/config/providers
                                  /api/config/test /api/metrics/summary
   ▼ 进程内
build_engine(on_event=..., plan_gate=threading.Event 版) → ArcpyKernel（串行）
```

- 服务端**只 import `ui/events.py`**（纯逻辑），禁止 import `render/tui/wiring`（rich 会抢 stdout）。
- 配置 API（唯一会写盘的）：`GET/PUT /api/config`、`GET /api/config/backups`、
  `POST /api/config/revert/{backup}`，外加 `GET /api/config/providers`（静态供应商预设，
  含 openai/deepseek/siliconflow/dashscope/moonshot/zhipu/ollama/custom）与
  `POST /api/config/test`（现场探 `GET {api_base}/models`，失败也 200 + `ok:false`，
  错误串先抹掉 key 与 URL 查询串）。写白名单含 `provider/model/api_base/timeout`；
  `provider` 只是元信息（引擎读 `api_base`+`model`+`api_key`）。空配置（文件不存在）时
  `GET` 返回 `{}` + `sha256:""`，首次 `PUT` 直接建文件（无备份）。
- 安全：绑 `127.0.0.1`；token 落 `.gis_agent/web_token`；路径接口 `resolve()` 后必须在 workspace 内；`api_key` 读时脱敏、写时空值即不改；**不提供任何执行任意代码/shell 的端点**。
- 地图：`/api/catalog/preview` → 读几何 → GeoJSON → `thin_features()` → 缓存（键 = path+mtime，上限 5 MB），前端用 canvas 画（照抄 `html_pack` 的绘制思路，**不引 Leaflet/瓦片**）。

## 7. 分波计划（24 人日）

| 波 | 内容 | 人日 | 依赖 | 交付物 |
|---|---|---|---|---|
| **W0** | 前端脚手架 + 设计系统（§2 全部 token/组件）+ 只读 API + `gis-agent web` | **5** | 无 | `web/`（Vite 工程）、`src/gis_cli/web/`、`[web]` extra |
| **W1** | 实时运行态：JobManager + EventBus + SSE + ②运行详情页 | **6** | W0 | `web/{jobs,event_bus,sse}.py`、前端 `/runs/:id` |
| **W2** | 看板：指标聚合 + ①总览 + ③运行历史 | **5** | W1 | `web/metrics.py`、前端 `/` 与 `/runs` |
| **W3** | 配置模型：读写 + 脱敏 + 备份 + 运行中锁 + ⑤配置页 | **3** | W0 | `web/config_api.py`、前端 `/config` |
| **W4** | 数据资产页 + 离线图层预览 + 成果包内嵌预览 | **5** | W0 | `web/preview_api.py`、前端 `/data` |

顺序理由不变：先只读（零风险、立刻能看现有 26 个 run）→ 再实时（价值最高、风险最高）→ 聚合 → 写配置（唯一会写盘的）→ 数据/地图（复用成熟渲染层）。

### W0 拆解（最容易被低估的一波）

1. Vite + React19 + TS + Tailwind4 + shadcn(base-nova) 初始化，`components.json` 与参照仓一致；
2. 把 §2.2 的 token 原值写进 `styles/theme.css`（`:root` + `.dark` 两套）；参照仓的「主题预设 / 字体轴」机制**先不做**，只留深浅两套 + 默认 sans 字体轴；
3. 装 §2.4 的 12 个组件（shadcn 生成 + 按 §2.5 规则微调）；
4. `AppSidebar` 三段分组 + `Header`（logo/版本/搜索/主题切换/头像位）；
5. 只读 API（traces/catalog/config 读）+ token 中间件 + 路径白名单；
6. `gis-agent web [--port 8760] [--no-browser]`，启动打印带 token 的 URL 并开浏览器。

**W0 验收**：`git clone` 后 `npm ci && npm run build` 一条命令出产物；`pip install -e .[web]` 后 `gis-agent web` 直接打开一个有真数据（26 个 trace）的只读控制台；1440×900 截图与参照仓同结构（侧栏/顶栏/KPI 卡齐全）。

## 8. 怎么"严格把控"样式（这节是给验收用的）

**手段**：把观感变成可勾选的条目，每波交付必须过这一遍，并附 `localhost` 截图（1440×900，浅色 + 深色各一张）到 issue。

### 视觉验收清单（12 条硬指标）

| # | 检查项 | 合格判据 |
|---|---|---|
| 1 | 顶部 header | 高度 48px，sticky，滚动时不遮侧栏 |
| 2 | 侧栏 | 宽 208px，三段分组小标题，选中项淡蓝底 + 圆角 |
| 3 | 卡片 | 圆角 16px，1px `--border` 描边，**无 box-shadow** |
| 4 | KPI 行 | 一张卡切 5 格，格间 1px 竖线，数字 ≥28px 且 `tabular-nums` |
| 5 | 健康条 | 单卡片一行，绿点 + 内联 chips |
| 6 | 图表 | 5 色板取 `--chart-1…5`，图例居中贴底，右上 segmented control |
| 7 | 表格 | 无竖线、两行单元格、数字右对齐、底部「总计/每页/分页器」 |
| 8 | 筛选卡 | 独立卡片，两行布局，主按钮为主色 pill |
| 9 | 深色模式 | `.dark` 全量生效，无一抹白底漏出 |
| 10 | 中文排版 | Public Sans + 显式 CJK 回退；无方框、无字重跳变 |
| 11 | 离线 | 产物零外链（`grep -c "http" dist/assets/*.js` ≈ 0，字体/图标已内联） |
| 12 | 空态/加载态 | 每页都有 Skeleton 与 Empty，不出现"白屏 3 秒" |

### 机器可验的补充

- `tests/web/test_static_bundle.py`：`static/index.html` 存在、不含 `http(s)://` 外链、体积上限断言。
- `pytest -m "not arcpy"` 基线 **1251 passed / 4 skipped 不掉**。
- 每波 API 的契约测试（坏行 trace 不 500、越界路径 403、SSE 断线补发、配置写入原子性、抽稀上限）。

## 9. 风险

| 风险 | 对策 |
|---|---|
| 照抄走成"逐像素抄代码" | §4 的 clean-room 边界 + §8 的清单（核对**观感条目**，不比对源码） |
| 前端工作量被低估（5 个页面 + 设计系统） | W0 单独成波并先交付可看的只读控制台；后续每波只加一页 |
| VChart 体积（~500KB gzip） | 只按需引入柱/面积/折线三种图；超标就退 recharts（配置得重写，先按 VChart 做） |
| `llm_delta` 洪泛 | 服务端 100ms 聚合 + 前端只渲染最新一屏、历史折叠成"共 N 段增量" |
| 配置写回打断在跑的 run | 有 run 时 409；写完只对下一个 run 生效 |
| 几何不投影就画（看着"正常"其实错位） | 非 4326/3857 时页面显式告警"未做投影变换"，不隐式变换 |
| 深色模式下 canvas 图未跟色 | 地图 canvas 读 CSS 变量重绘，纳入清单第 9 条 |
| 范围蔓延到"又一个在线 GIS 平台" | §10 不做清单；新增需求另开单 |

## 10. 明确不做

多用户/权限、公网部署/HTTPS/Docker、在线底图与瓦片、任意代码执行端点、Streamlit、复用 `legacy/core_tools/web/`、
新建指标数据库、WebSocket（SSE 够用）、Electron。

## 11. 实施：已拆成 9 张子单（2026-11-10）

全文实现按"单写手串行"拆成 9 段，每段独立可验收；一次只有一张单在 `todo`，上一张 `in_review` 后放下一张。

| # | 单 | 承接 |
|---|---|---|
| S1 | 后端服务骨架 + 只读 API（traces/catalog/config）+ `gis-agent web` | GIS工程师 |
| S2 | 前端工程 + 设计系统（token/布局/12 组件/零外链产物） | GIS工程师 |
| R1 | 审查 S1+S2（commit 区间 + `pytest -m "not arcpy"`） | 代码审查工程师 |
| S3 | 实时运行态（JobManager/EventBus/SSE/运行详情页） | GIS工程师 |
| S4 | 总览看板 + 运行历史 | GIS工程师 |
| S5 | 配置模型页（读写/脱敏/备份/锁） | GIS工程师 |
| S6 | 数据资产 + 离线图层预览 + 成果包预览 | GIS工程师 |
| R2 | 全量审查 + §8 十二条视觉清单核对 | 代码审查工程师 |
| B1 | 真实运行 E2E + 深浅截图交付 | 基准测试工程师 |

子单全部挂在 DAIR-233 下，用 `multica issue children 01a1c379-e7f1-7967-b54b-31e435175510` 查看进度。

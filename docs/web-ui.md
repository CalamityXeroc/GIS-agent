# Web 控制台（`gis-agent web`）

本机自用的可视化控制台：和终端 UI 共用同一个引擎、同一份 workspace、同一批 trace，
只是把「看过程 / 管运行 / 查数据 / 改配置」搬到了浏览器里。

三条边界先讲清楚，免得按错预期：

- **单机单用户**：只绑 `127.0.0.1`，靠 workspace 里的一个 token 放行，不做账号体系。
- **可选依赖**：不装 `[web]` 时，`gis-agent chat` / `run` / `tools` 等命令照常工作，
  只有 `gis-agent web` 会提示缺依赖。
- **只读优先**：控制台能看的东西都来自磁盘事实（trace / catalog / config），
  不提供任何执行任意代码或 shell 的端点。

## 1. 安装与启动

```powershell
# 控制台依赖（fastapi / uvicorn / starlette）
pip install -e ".[web]"

# 启动：默认 127.0.0.1:8760，自动打开浏览器
gis-agent web
```

启动会打印带 token 的地址，浏览器直接打开即可：

```
工作空间: E:\BST_lab\GIS-agent-repo\workspace
控制台: http://127.0.0.1:8760/?token=…
```

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host` | `127.0.0.1` | 监听地址。改到 `0.0.0.0` 等于把控制台暴露给局域网，风险自负 |
| `--port` | `8760` | 监听端口 |
| `--no-browser` | 关 | 不自动打开浏览器 |
| `--stop` | — | 停掉后台运行的控制台后退出（不需要 uvicorn） |
| `--token` | 读/生成 `.gis_agent/web_token` | 指定访问 token |
| `--workspace` / `-w` | 见下 | 工作空间目录 |
| `--config` / `-c` | `<workspace>/config/llm_config.json` | 配置文件路径 |

**工作空间是怎么选的**：没给 `--workspace` 时，先看 cwd 和 `cwd/workspace` 谁有
`.gis_agent/web.pid`（优先，保证 `--stop` 能找到正在跑的面板），再看谁有
`.gis_agent/traces/`，都没有才退回 cwd。所以在本仓根目录直接 `gis-agent web`
也能打开一个有数据的控制台，而不是一片空白。

### 1.1 从 chat 自动拉起

`gis-agent chat` 启动时会在后台把面板拉起来，不需要另开一个终端：

- **无窗口**：Windows 用 `pythonw.exe`（`DETACHED_PROCESS`），POSIX 用 `start_new_session`；
  chat 退出、终端关闭都不会带走它，历史随时能回看。
- **单实例**：靠 pid 文件 + TCP 探活判断是否已在跑，已在跑就直接复用。
- **端口顺延**：默认端口被占时自动往后试 `8761`–`8770`，实际端口写进 `web.port`。
- **失败不影响对话**：拉不起来只打印原因，不抛异常、不拖死 chat——控制台是附加能力。

状态文件都在 `<workspace>/.gis_agent/` 下：

| 文件 | 内容 |
|---|---|
| `web.pid` | 面板进程 pid |
| `web.port` | 面板**实际**监听端口 |
| `web.log` | 面板 stdout / stderr（追加） |
| `web_token` | 访问 token（32 字节随机数的十六进制，权限 0600） |

关闭方式：

```powershell
$env:GIS_NO_WEB = "1"        # 或 gis-agent chat --no-web：本次不拉起
gis-agent web --stop          # 停掉后台面板（手动起的实例同样会被停）
```

## 2. 页面

侧栏分三组，路径与终端命令一一对得上：

| 分组 | 路径 | 页面 | 内容 |
|---|---|---|---|
| 运行 | `/` | 概览 | 成功率、耗时、token 与验收缺口的聚合视图，口径全部来自 trace 事实 |
| 运行 | `/runs` | 运行历史 | 所有 trace 的筛选与回溯；每行可展开时间线摘要，下载原始 `.jsonl` / `.md` |
| 运行 | `/runs/new` | 新建运行 | 提交一个任务，进入单 worker 队列排队执行 |
| 运行 | `/runs/<run_id>` | 运行详情 | 状态条 + 实时时间线（模型增量 / 工具调用与结果 / 验收结论 / 计划卡），可停止、插话、看计划、续跑 |
| 数据 | `/data` | 数据资产 | catalog 里登记的矢量/栅格资产：坐标系、范围、要素数、字段画像与离线预览 |
| 数据 | `/packages` | 成果包 | `output/` 下每次运行的成果包与溯源卡：沙箱内嵌预览，可下载原件（`/reports` 是同一页的别名） |
| 系统 | `/config` | 配置模型 | 10 个角色的模型路由、引擎预算与回退策略；写入前自动备份、可回滚；密钥只显示脱敏值 |
| 系统 | `/env` | 环境自检 | **占位页，尚未实现**——ArcPy 桥接 / 依赖 / 模板环境检查目前仍在终端里做 |

`/env` 之外都是可用的。概览、成果包这类页面在没有 run 记录时是空态，
先跑一次 `gis-agent run "…"` 或 `gis-agent chat` 再刷新就有数据了。

## 3. 实时性怎么来的

- 运行详情订阅 `GET /api/runs/{run_id}/events`（SSE，15s 心跳）。
- 事件源头是引擎的 `build_engine(on_event=...)`：`llm_delta`、`tool_call` / `tool_result`、
  `verification`、`plan_card` 等都在同一条流上，和终端 UI 看到的是同一批事件。
- **刷新不丢**：初次连接带 `since=0`，服务端从环形缓冲回放；断线由浏览器自动重连并带
  `Last-Event-ID`，只补缺失段；前端按 `seq` 去重排序，重连叠出来的重复帧不会重复显示。
- 服务重启过、内存里没有 job 时，磁盘上的 trace 就是全部事实，照样能看。

## 4. 为什么是单 worker

`JobManager` 是**单 worker 线程 + 队列**，不是"懒得写并发"，是硬约束：

`ArcpyKernel` 是单例、`_exec_lock` 串行化、`llm_max_concurrency=1`、ArcGIS 本身也是单实例。
同时跑两个 run 不是更快，而是互相踩——内核状态会串、GDB 会被两个进程同时写。

所以队列是**可见**的：新建的运行排队等，而不是静默阻塞或直接失败。

## 5. 安全约定

| 项 | 做法 |
|---|---|
| 监听地址 | 默认 `127.0.0.1`，只本机可访问 |
| 鉴权 | 启动时把 token 落到 `<workspace>/.gis_agent/web_token`，URL 带 `?token=`；重启后同一个 token 仍然有效 |
| 路径接口 | 所有带路径的接口先 `resolve()`，结果必须落在 workspace 内，否则 403 |
| 密钥 | `api_key` 出站前脱敏成 `sk-…后4位`；写入时留空表示"不修改" |
| 代码执行 | **不提供**任何执行任意代码 / shell 的端点 |

想通过反向代理或局域网访问，需要自己改 `--host` 并承担后果——本项目的安全模型
建立在"只有本机能连"这个前提上。

## 6. API 一览

控制台用的就是这些接口，脚本里也能直接用（记得带 token）：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/metrics/summary` | 聚合指标（成功率 / 耗时 / token / 验收缺口） |
| GET | `/api/metrics/health` | 健康检查 |
| GET | `/api/traces` | trace 列表 |
| GET | `/api/traces/{run_id}` | 单个 trace 的完整事件流 |
| GET | `/api/traces/{run_id}/download` | 下载原始 trace |
| POST | `/api/runs` | 新建运行（202，入队） |
| GET | `/api/runs` | 运行列表 |
| GET | `/api/runs/{run_id}` | 运行状态 |
| POST | `/api/runs/{run_id}/stop` | 停止（等引擎在安全点退出，不硬杀） |
| POST | `/api/runs/{run_id}/inject` | 插话：把一句话排进本轮 |
| POST | `/api/runs/{run_id}/plan` | 回计划卡（放行 / 改口径） |
| POST | `/api/runs/{run_id}/resume` | 从断点续跑（202） |
| GET | `/api/runs/{run_id}/events` | SSE 事件流（`since` / `Last-Event-ID`） |
| GET | `/api/catalog/datasets` | catalog 资产列表 |
| GET | `/api/catalog/datasets/profile` | 资产画像 |
| GET | `/api/catalog/datasets/{path}/profile` | 指定资产的字段/范围画像 |
| GET | `/api/catalog/preview` | 离线预览（内联 GeoJSON，零外链） |
| GET | `/api/reports`、`/api/reports/{name}` | 成果包列表 / 单包内容 |
| GET | `/api/config` | 读配置（密钥脱敏） |
| PUT | `/api/config` | 写配置（写前自动备份） |
| GET | `/api/config/providers` | 可用 provider / 模型 |
| POST | `/api/config/test` | 测试模型连通性 |
| GET | `/api/config/backups` | 备份列表 |
| POST | `/api/config/revert/{backup_name}` | 回滚到某个备份 |

## 7. 前端工程

- 栈：React 19 + Vite + Tailwind 4 + shadcn/ui + TanStack Router。
- **构建产物进仓**：`web/` 的 Vite 输出直接落 `src/gis_cli/web/static/`，随 wheel 一起发布。
  所以 `pip install` 完就能 `gis-agent web`，**用界面不需要装 Node**。
- 字体走 `@fontsource-variable` 内联进产物，不引 CDN（离线可用）。
- 改前端才需要 Node：

```powershell
cd web
npm ci
npm run build      # 产物落 src/gis_cli/web/static/
```

服务端不 import 引擎渲染层（`ui/render` / `ui/tui` / `ui/wiring`），避免 rich 在无 TTY 的
服务进程里抢 stdout；引擎只在真正跑任务的 worker 线程里按需 import。

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `缺少 Web 依赖：pip install -e .[web]` | 没装 `[web]` extra |
| 端口被占 | 自动顺延 `8761`–`8770`；实际端口看启动输出或 `.gis_agent/web.port` |
| 打开页面 401 / 一直跳回 | URL 里少了 `?token=`；或删掉 `.gis_agent/web_token` 后重启换一个 |
| 子页面刷新 404 | 不该发生——SPA 有 `index.html` 回落；`/api/` 前缀刻意不参与回落，拼错的接口会老老实实 404 |
| 面板起不来 | 看 `<workspace>/.gis_agent/web.log`；`gis-agent web --stop` 清掉残留 pid 后重试 |
| 概览是空的 | 该 workspace 还没有 trace，先跑一次任务 |
| chat 里看不到控制台地址 | 面板启动失败（原因会在 chat 输出里），或设了 `GIS_NO_WEB=1` / `--no-web` |

## 9. 测试

控制台相关测试在 `tests/web/`，覆盖 token / 路径白名单 / 脱敏（`test_security.py`）、
任务队列与停止/插话/续跑（`test_jobs.py`）、SSE 回放与心跳（`test_sse.py`）、
配置读写与备份（`test_config_api.py`）、catalog 与预览（`test_catalog_api.py`、
`test_preview_api.py`）、前端产物完整性（`test_static_bundle.py`）、
以及"服务端不得 import 终端渲染层"的红线（`test_import_guard.py`）。

```powershell
pytest tests/web
```

这些用例都不依赖 ArcGIS，CI 里照常跑。

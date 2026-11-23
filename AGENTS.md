<!-- CODEGRAPH_START -->
## CodeGraph

This project has a CodeGraph MCP server (`codegraph_*` tools) configured. CodeGraph is a tree-sitter-parsed knowledge graph of every symbol, edge, and file. Reads are sub-millisecond and return structural information grep cannot.

### When to prefer codegraph over native search

Use codegraph for **structural** questions — what calls what, what would break, where is X defined, what is X's signature. Use native grep/read only for **literal text** queries (string contents, comments, log messages) or after you already have a specific file open.

| Question | Tool |
|---|---|
| "Where is X defined?" / "Find symbol named X" | `codegraph_search` |
| "What calls function Y?" | `codegraph_callers` |
| "What does Y call?" | `codegraph_callees` |
| "How does X reach/become Y? / trace the flow from X to Y" | `codegraph_trace` (one call = the whole path, incl. callback/React/JSX dynamic hops) |
| "What would break if I changed Z?" | `codegraph_impact` |
| "Show me Y's signature / source / docstring" | `codegraph_node` |
| "Give me focused context for a task/area" | `codegraph_context` |
| "See several related symbols' source at once" | `codegraph_explore` |
| "What files exist under path/" | `codegraph_files` |
| "Is the index healthy?" | `codegraph_status` |

### Rules of thumb

- **Answer directly — don't delegate exploration.** For "how does X work" / architecture questions, answer with 2-3 codegraph calls: `codegraph_context` first, then ONE `codegraph_explore` for the source of the symbols it surfaces. For a specific **flow** ("how does X reach Y") start with `codegraph_trace` from→to — one call returns the whole path with dynamic hops bridged — then ONE `codegraph_explore` for the bodies; don't rebuild the path with `codegraph_search` + `codegraph_callers`. Codegraph IS the pre-built index, so spawning a separate file-reading sub-task/agent — or running a grep + read loop — repeats work codegraph already did and costs more for the same answer.
- **Trust codegraph results.** They come from a full AST parse. Do NOT re-verify them with grep — that's slower, less accurate, and wastes context.
- **Don't grep first** when looking up a symbol by name. `codegraph_search` is faster and returns kind + location + signature in one call.
- **Don't chain `codegraph_search` + `codegraph_node`** when you just want context — `codegraph_context` is one call.
- **Don't loop `codegraph_node` over many symbols** — one `codegraph_explore` call returns several symbols' source grouped in a single capped call, while each separate node/Read call re-reads the whole context and costs far more.
- **Index lag — check the staleness banner, don't guess a wait.** When a codegraph response starts with "⚠️ Some files referenced below were edited since the last index sync…", the listed files are pending re-index — Read those specific files for accurate content. Files NOT in that banner are fresh and codegraph is authoritative for them. `codegraph_status` also lists pending files under "Pending sync".

### If `.codegraph/` doesn't exist

The MCP server returns "not initialized." Ask the user: *"I notice this project doesn't have CodeGraph initialized. Want me to run `codegraph init -i` to build the index?"*
<!-- CODEGRAPH_END -->

## Git 推送规则（用户明确要求，2026-11-22）

- **默认只提交本地 git，不 push GitHub。** 交付汇报写「本地 commit <sha>，未推送」。
- 只有用户在对话里**当场明确要求**「推到 github / push / 上传仓库」时才 `git push`；上一轮的授权不延续到下一轮。
- 禁止「先推上去再撤回」：push 之前必须能指出用户哪一句话要求了推送，指不出来就不推。
- 例外：新建 remote 之前先问，不自己建。


---

## 本项目架构要点（P0 重构后）

### 单一引擎（旧引擎已下线）
- `giscli chat` / `giscli loop "任务"` —— `src/gis_cli/engine/` 有界观察-决策-执行循环，
  原生 tool calling（上游不支持时自动降级 JSON 协议），代码失败自动修复（traceback 回喂），
  交付前强制验收（Verifier）。入口 `engine/runner.py:build_engine()`。
- **旧引擎已迁入 `src/gis_cli/legacy/`**（plan-then-execute 那两套 + core/tools/web），
  **不参与打包**（`[tool.setuptools.packages.find] exclude`）。别再往里加功能；
  新能力一律加到 `engine/` / `runtime/` / `ui/`。
- 新引擎**不依赖** pydantic / langchain / litellm / chromadb（有测试守着：
  `tests/test_engine_dependency_boundary.py`）。加依赖前先跑那条测试。

### 关键模块
- `runtime/kernel.py` ArcpyKernel：持久 Jupyter 内核跑 ArcPy（一次导入，跨步状态，超时中断）。
  代码里用 `set_result({...})` 返回结果；每次执行前自动注入 workspace。
- `runtime/code_runner.py` CodeRunner：内核优先、一次性子进程兜底。
  **覆盖前自动备份**：`backup_targets()` 识别"写工具的输出参数已存在"（内核里
  `overwriteOutput=True` 全局开着，正常写产出也会静默覆盖）；GDB 成员走内核
  `CopyFeatures` 导出（`Path.exists()` 对 GDB 内部要素类返回 False）。
- `engine/web_search.py` 联网搜索：实测国内只有 Bing 可达，而它对自动化请求会返回
  **地域劫持的无关结果**（查 arcpy 返回 NFL 官网），所以有 `_filter_relevant()` 相关性
  校验；配 `TAVILY_API_KEY` 才稳定。
- `datacatalog/`：SQLite 增量数据目录，`build_digest()` 注入 prompt，
  `catalog_query` 工具供模型查询真实字段名/坐标系。
  启动不扫（界面立刻可用），首个任务时后台扫描与首轮模型调用并行。
- `recipes/` GIS 方法论配方：`{{param}}` 替换为 Python 字面量，`@@param@@` 替换为原文；
  自带 validation 断言（crs_wkid/feature_count_gt/fields_present/image_not_blank 等）。
- `engine/subagent.py` 子代理：`delegate(task, ...)` 把自包含子任务放进**独立对话**，
  只回结构化结论（共享 ArcPy 内核、串行、深度 1；子运行不落 `runs/`）。
  默认关，`engine.subagent_enabled=true` 打开；四个 hook 与踩过的坑见 `docs/subagent.md`。
- `engine/subagent.py:run_review` + `review/` 分析正确性双复核（P1-5 / W5）：
  复核者只看「数据 + 结论」、不看主轨迹，配守恒/拓扑/量纲断言库；
  默认关（`engine.review_enabled`）；交付级 benchmark 任务在 YAML 顶层或 `requirements` 里写 `review: true` 打开。见 `docs/review.md`。
- `trace/` 每次 run 写 JSONL + Markdown 报告到 `workspace/.gis_agent/traces/`。

### 模型配置
`config/llm_config.json`（已 gitignore）：deepseek 官方 API 为主，`fallback_models` 兜底。
客户端已实现 429 指数退避、模型熔断（连续失败进冷却，冷却时长指数升级）；
reasoning 模型注意 max_tokens 要给足。

### 测试与基准
- 无 ArcGIS 单测：`PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_engine_*.py`
- 带 arcpy 的旧测试很慢（每个 ~12s 子进程），跑全量需 15 分钟以上，不是挂死。
- 基准：`python benchmark/runner.py --all`；标准答案判定策略见 `benchmark/README.md`；
  基准工作区只含输入数据，绝不包含 `结果/`、`result_data.gdb` 等参考答案。

### 已知坑
- ArcGIS Pro 3.6：`Map` 无 `.camera`，用 `layout.listElements("MAPFRAME_ELEMENT")`；
  导出后缀可能被追加；GDB 路径不是文件系统路径（判断存在用 `arcpy.Exists`）。
- Windows 控制台 GBK：cli.py 已把 stdout reconfigure 为 UTF-8；新代码不要用 ✓/↻ 等非 GBK 字符。
- 内核 stdout 会有 ArcGIS 代码页 warning，已在 kernel 层过滤。
- **持久内核里有两类 API 不能用**（实测）：
  - `MakeFeatureLayer` + `SelectLayerByAttribute`：会挂死且 interrupt 无效（只能看门狗硬杀），
    偶发 `Fatal Python error: InitDatetime: Could not import datetime C API`。等价替代：
    `FeatureClassToFeatureClass(输入, 库, 名, where_clause)`（最快）、`analysis.Select`、
    `da.SearchCursor(路径, 字段, where_clause)`。
  - `ZonalStatisticsAsTable` 在几千个分区（社区缓冲区等）上要 10 分钟以上，同样可能触发致命崩溃。
    改用 `zone_raster_sum` 配方（面转栅格 + numpy bincount），秒级完成、语义一致。
- 跑基准前先关掉 ArcGIS Pro：Pro 与内核同时对同一安装做 GP 调用时会互相干扰（Pro 会弹严重错误窗口）。

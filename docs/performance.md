# 性能：慢在哪、改了什么、实测多少

面向"agent 用起来很慢"这件事的归因与优化记录。每个数字都是在**本机实测**
（ArcGIS Pro 3.6 + deepseek-v4.1-flash-intl @ workbuddy 反代），不是估算。

看当前的耗时构成：

```bash
python scripts/latency_report.py            # 最近一次 run
python scripts/latency_report.py --all      # 最近几次汇总
grep "\[timing\]" workspace/.gis_agent/logs/engine.log   # 启动/内核/LLM 分段
```

---

## 一、改前实测的四类慢

| # | 问题 | 实测数字 | 根因 |
|---|---|---|---|
| 1 | **LLM 不流式** | 单轮 20~90s 全屏无输出 | 等整段 completion（含全部思维链）生成完才返回 |
| 2 | **启动同步扫数据目录** | 158.5s | 且每次启动都全量重扫（见下） |
| 3 | **GDB 缓存自失效** | 133 个要素类重扫 | 读 GDB 会建 `.sr.lock` → 目录 mtime 变 → 缓存永久失效 |
| 4 | **上下文压缩抖动** | 一个 32 轮 run 压 16 次 | 图片 base64 被当文本估算 token，一附图就必然超限 |
| 5 | **识图逐张串行** | 收尾 75.5s + 129.0s | 5 张图逐张调用，且一个 run 验两次 |

其中 #2 和 #3 是同一个根因的两面：**GDB 目录 mtime 会被读取动作本身刷新**，
而它正是"要不要重扫"的判据，于是形成死循环：

    扫描 → 建 .sr.lock → 目录 mtime 变 → 缓存失效 → 下次又全量重扫

实测证据：12 个 GDB 容器共享同一个"刚刚"的 mtime（0.05 小时前），
而容器内的数据文件是一年前的（8447 小时前）—— 数据没动，只有锁文件在变。

---

## 二、改了什么

### 1. 流式输出（感受最强）

- `EngineLLMClient` 支持 `stream=True`，增量组装成**与非流式逐字段等价**的
  `LLMResponse`（`content` / `reasoning` / `tool_calls` 分片 / `usage`），
  调用方零改动；
- 新增 `on_delta(kind, text)` 回调，界面用 rich `Live` **原地重画**思考/回复块，
  结束才定型打印一次（不逐词刷屏）；
- 上游不支持流式时三条路径都能回退：忽略 `stream` 参数（返回完整响应）、
  返回空 chunk 流、直接 400/422 拒绝。

**实测**：首字 6.3s 到屏（此前需等 11.8s 整段）；491 个增量块。

### 2. GDB 新鲜度改用"内容时间"

`gdb_content_mtime()` 只取容器内**非锁、非临时文件**的最新 mtime。
读操作不会再推新缓存；真改数据仍能被发现。

**实测**：目录扫描 156.84s → 0.05s（`scanned=0 skipped=91`）。

### 3. 启动不扫 + 后台并行预热

- `build_engine(refresh_catalog=False)`：启动只装配配方/配置；
- 首个任务开始时后台跑：**数据目录扫描**与**内核预热**同时进行，
  与首轮模型调用重叠；
- `catalog_query` 首次用到目录时 `wait_ready()`，对模型透明。

**实测**：启动 158.68s → 0.30s；内核 12.4s 与首轮 LLM 7s 并行。

### 4. 压缩不抖动

- 图片块按固定成本估算（`image_token_cost`，默认 1500），不看 base64 长度；
  修复前一张 400KB 图被估成 **26.7 万 token**（软上限才 12 万）；
- 加滞回（`compact_hysteresis`，默认 1.2）与最少条数门槛
  （`compact_min_messages`，默认 8）。

**实测**：带图长对话压缩次数归零（修复前 16 次/run）。

### 5. 识图质检批量 + 缓存

- `vision_check_images()`：一次请求带多张图，按 `index` 返回逐张结论；
- 结论按 `(路径, mtime)` 缓存：一个 run 内 `verify_outputs` 与 `finish`
  两次验收，第二遍直接命中；图片重出自动失效。

**实测**：5 张图的收尾验收从 75.5s + 129.0s 降为一次调用 + 一次缓存命中。

### 6. 思考档位按任务分档

`reasoning_effort_by_task`：交互轮用 `low`，规划/验收保持 `medium`。

**实测**（同一提示词）：

| 档位 | 首字 | 总耗时 | 思考字数 |
|---|---|---|---|
| medium | 5.11s | 8.72s | 1274 |
| low | **2.37s** | **5.80s** | 757 |

输出质量相当（两者都给出完整检查项）。

### 7. 入口惰性导入

旧引擎栈（GISAgent/pydantic/langchain/工具注册表）改函数内 import。

**实测**：`import gis_cli.agent.cli` 1.115s → 0.920s（旧栈占 0.745s，67%）。

---

## 三、顺带修的真 bug

优化过程里撞出来的、与性能同源的问题：

| bug | 现象 | 修法 |
|---|---|---|
| **数据目录误删记录** | 默认只扫 `input/` 后，`output/` 里已登记的 19 条被当"文件已消失"删掉（212 → 193） | `remove_missing(within=...)` 把清理限定在本次扫描的根目录内 |
| **验收阶段崩溃** | `.spec.json` 内容超长时被截断 → `json.loads` 抛 `JSONDecodeError` → 跑了数分钟的任务在 finish 直接挂 | `_shrink_json()` 递归缩水，保证结果仍是合法 JSON |
| **内核预热竞态** | 预热未完成就退出时抢共享内存，zmq 报 `Bad address` | `shutdown()` 先 `wait_prewarm()` 收尾再关 |
| **ZMQ socket 跨线程** | 预热在后台线程建 socket、主线程执行，交叉写坏 wire 协议（内核侧 `Invalid Signature: b'<IDS|MSG>'` / `DELIM not in msg_list`，客户端 zmq `Bad address`） | `_exec_lock` 把“发消息→收 iopub→取结果”整段串行化 |

---

## 四、还没做的（下一步候选）

实测数据支持、但本轮未动的：

| 项 | 依据 |
|---|---|
| **工具并行执行** | 当前 `for action in actions` 串行；多个只读 `catalog_query` 可以并发（内核执行已串行化，安全） |
| **需求抽取合流** | 每任务开头一次独立 LLM 往返，可并入首轮对话 |
| **内核往返合并** | 每次 `execute_code` 要 2 次往返（执行 + `_fetch_result`） |
| **温度/提示词缓存统计** | 已能从 usage 读到 `prompt_cache_hit_tokens`（实测命中 32512/35805），可做命中率趋势 |
| **工具输出瘦身** | `verify_outputs` 等长结果进上下文后推高 prefill 成本 |

---

## 五、聊天界面的三个实测 bug（定位与修法）

这三个问题都只在真实终端里可见，普通单测（用 StringIO 当输出）全会漏，
所以新增了无 ANSI Console 的复现测试（``tests/test_ui_terminal_regressions.py``）。

| 问题 | 现象 | 根因 | 修法 |
|---|---|---|---|
| **多个输入框** | 启动后出现两个框，新框状态行写“运行中 · 0s” | ``Application(full_screen=False)`` 未设 ``erase_when_done``：提交后旧框留在屏幕上，主循环又画一个 | ``erase_when_done=True`` + 提交后总是回显用户输入 |
| **流式乱码** | 满屏 ``◆ 思考 · 0s`` 和 ``?[2K?[1A?[2K`` | rich ``Live`` 每帧发光标控制序列（ESC[2K/ESC[1A），普通控制台不解析 ANSI 就变成字面量；而降级判据用的是“能不能构造 Live”（构造总是成功，永远不触发） | 判据改用项目自己的 ``ansi_ok()``；交互模式完全不用 Live，改为输入框上方的实时预览窗（由 prompt_toolkit 重绘，走 Win32 API） |
| **要按两次回车** | 输入一条没反应，再输一条才开始 | ① 提交后旧框不消失且无可见反馈，用户以为没提交；② 兕底补全链挂了 ``PathCompleter``，把任意文本当路径匹配，命中时第一次回车只是“接受候选” | ① 回显 + “正在分析需求…”提示；② 移除 PathCompleter（两条构造路径各一处）；③ Enter 只在用户按过 ↑↓ 导航时才应用候选 |

### 运行中再输入的语义

- **排队**（不再静默注入当前任务）：立刻显示 ``已排队 N 条``，本轮结束后
  **合并成同一个任务**自动继续 —— 用户补几句话通常是为了把同一件事做完；
  要换任务请直接新开对话。
- ``Esc`` 仍是中断。
- 状态行空闲时不再显示 ``未开始 · 0s ↑0 ↓0``（那会让人以为在跑）。

### 启动提速：不再每次 pip install -e .

editable 安装的全部作用 = 往 site-packages 放一个 ``.pth``，内容就是本仓库
的 ``src`` 路径（已验证），**源码改动自动生效**。而每次重装要把
langchain/chromadb/litellm 的依赖树整个解一遍，实测卡几十秒到几分钟。

现在由 ``scripts/check_install.py`` 判定，只在以下情况才装：

1. ``.pth`` 不存在（首次）；
2. ``.pth`` 指向的不是本仓库（换过目录）；
3. ``pyproject.toml`` 比 ``.pth`` 新（依赖可能变了）；
4. 显式 ``--reinstall``（或设 ``REINSTALL=1``）。

关键依赖改用 ``import`` 快速检查（<1s），缺了才装。实测安装检查 0.16s。

> 注意：``.bat`` 必须是 **CRLF** 行尾（LF 会让 cmd 解析错乱，实测），
> 且内容保持 ASCII（中文注释在 GBK 代码页下会破坏解析）。

## 六、边界与注意

- **流式依赖上游**：workbuddy 反代（CodeBuddy 上游）实测支持
  `delta.content` / `delta.reasoning_content` / `delta.tool_calls` 分片与末尾 `usage`；
  不支持的网关自动回退，功能不受影响。
- **GDB 新鲜度的取舍**：只看非锁文件的 mtime，因此"只改了锁文件"这类
  元数据变化不会触发重扫（这是刻意的）。真正重写要素类仍会被发现。
- **默认不扫 `output/`**：它是产物目录，不是数据源。需要让它参与数据目录时，
  用 `--refresh-catalog` 或显式传入根目录。
- **`reasoning_effort` 低档的代价**：实测本例输出质量相当，但复杂规划任务
  是否受影响未做大样本验证——规划/验收类任务刻意保留在 `medium`。

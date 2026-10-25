# 子代理（delegate）：上下文隔离

长任务的瓶颈不是"会不会做"，而是**单条会话越跑越贵**：每轮都要把整段历史重发一遍，
成本随轮次平方增长，最后撞 `max_turns`。14 届四a 实测：120 轮 / 1046 万 token /
中途压缩 3 次（内部估算 14.7~15 万 token 才触发）；同样的活儿拆成 4a1（17 轮 / 50.6 万）
+ 4a2（32 轮 / 216 万），成本降一个量级。

子代理就是把这件"拆"变成引擎能力：一段自包含的子任务放进**独立对话**里跑完，
只把结论带回主会话。

```
主代理 (AgentLoop)                子代理 (另一个 AgentLoop)
├─ _messages   ← 只收一条结论      ├─ _messages  ← 自己的全部过程（不回传）
├─ TaskState   父任务清单/产物     ├─ TaskState  子任务清单/产物
├─ trace       run_xxx.jsonl       └─ trace      run_xxx__sa1.jsonl
└─ 共享：workspace / ArcPy 内核 / 数据目录 / 配方库 / LLM 客户端
```

## 为什么只隔离对话

ArcPy 内核是**单例 + 串行执行锁**（`runtime/kernel.py` 的 `_exec_lock`），
另开一个内核要各自 `import arcpy`（~15s）并吃一份内存，还会和 ArcGIS Pro 抢 GP。
LLM 客户端默认 `llm_max_concurrency=1`，并行子代理只会在信号量上排队。
所以 v1 就是「父等子，一次一个」，深度 1（子代理的工具表里没有 `delegate`）。

## 工具契约

```
delegate(task, acceptance?, max_turns?, tools?)
```

- `task` 必填且**自包含**：子代理看不到父的对话，数据路径 / 字段名 / 坐标系 /
  输出位置 / 要交什么都要写进去。
- `acceptance`：这段的验收标准，进子代理的需求清单。
- `max_turns`：默认 30，夹在 5~60。
- `tools`：可选收窄（例如纯调研只给 `catalog_query` / `read_file`）。`finish` 永远保留。

返回给父模型的观察（硬上限 `subagent_result_max_chars`，默认 4000 字符）：

```json
{"subtask_id": "sa1", "status": "completed",
 "summary": "四期最大密度 9.8，统一分级 0-2-4-6-8-10",
 "methodology": "kernel_density, 100km², 100m, UTM10N",
 "artifacts": ["output/result_data.gdb/kde_1982_1986", "output/成果图/密度_1982_1986.jpg"],
 "turns": 17, "elapsed_s": 223, "tokens": 506000,
 "trace": ".gis_agent/traces/run_xxx__sa1.jsonl", "error": ""}
```

`status` ∈ `completed / max_turns / deadline_exceeded / llm_unavailable / failed /
stopped / refused / error`。**非 completed 也返回已产出的部分**，父可以接着做完或改小重派；
子代理崩了也不抛异常，父拿到 `ok=False` 继续跑。产物会并进父的 `state.artifacts`，
父的 `verify_outputs` / `finish` 照旧验收。

## 四个关键 hook（`engine/loop.py`）

| hook | 不做的后果 |
|---|---|
| `run(reset_runner=False)` | 父 `run()` 一进来就 `code_runner.reset()`，那是 `del globals()`——父在内核里攒的 df / 投影对象全被清掉 |
| `run(persist_state=False)` | 子运行会写 `.gis_agent/runs/`，`distill` 与断点续跑会把它当成「最近一次成功运行」挑走 |
| `should_stop=...` | 用户 Ctrl-C 后子代理继续跑，占着内核和 LLM 配额，而父已经在等下一次输入 |
| `LoopConfig.role_preamble` | 子代理不知道自己是子代理，会去问用户、评价整体方案、复述过程 |

另外两条：

- **子代理 `verifier=None`**：子层的 `finish` 若跑父的 Verifier，会拿**父任务的整任务要求**
  判一段子任务（还要对整个 `output/` 做结构体检），必然被拒 3 次后记 failed。
  验收统一由父做。
- **`context` 必须 `replace()` 复制**：`run()` 会写 `context.state`，共享的话子跑完
  父的 `execute_code` 会把产物记到子任务上。

## 配置（`config/llm_config.json` → `engine`）

```json
"subagent_enabled": true,
"subagent_max_per_run": 3,
"subagent_max_turns": 30,
"subagent_timeout_seconds": 900,
"subagent_exec_timeout_seconds": 300,
"subagent_result_max_chars": 4000
```

`subagent_exec_timeout_seconds` 比父的 `kernel_exec_timeout_seconds`（默认 600）短一档：
子代理的 GP 卡死要尽快失败，600s 那一档实测一次卡死吃掉 10 分钟。注意它**保护不了父的
内核变量**——看门狗硬杀内核后 CodeRunner 会重建，父的变量一样没了；所以子代理结果里带了
`kernel_restarted` 标记，父收到会被告知「你在内核里的变量已丢」。

子代理的 system prompt 也会瘦身：`skills_max` 从 40 降到 12（111 个技能全列是 13k 字符，
子任务用不到）。实测 prompt 从 15,991 字符降到 8,658。

默认 **关**：每个子代理多一份 system prompt（瘦身后 8.7k 字符、约 4k token/轮；
瘦身前是 16k 字符），拆 3 步的小活儿反而更贵。关掉时 `delegate` 从工具表里剔除，
而不是留着报错——留着模型会白花一轮。

## 什么时候值得拆

- 值得：预计 **>10 轮**、有独立产物、能独立验收的整块工作（"出 4 张共用图例的密度图"）。
- 不值得：3~5 步的小活儿、需要父边看结果边决策的连续调试、共享同一批中间变量的步骤。

## 实测：组合任务 A/B（2026-10-22）

任务：一个双交付的组合任务（4 期核密度 + 迁移图 ｜ 土地覆盖云区修补 + 面积统计，
共 20 条断言，两个交付互不依赖）。同一模型、同一份输入，各跑一次。

| 指标 | 关（off） | 开（on） |
|---|---|---|
| 断言 | 20/20 | 20/20 |
| 父会话轮次 | 28 | **21** |
| 父会话 prompt 峰值 | 76,449 | **50,078** |
| 总 token（报告口径） | 1,330,895 | 1,366,544 |
| 子代理 | 0 | 2（sa1 17 轮 / sa2 4 轮） |
| 墙钟 | 2274s | 2774s |

两条结论，都要记住：

1. **它买的是上下文和轮次预算，不是 token**。父会话峰值降 34%、轮次降 25%，但
   总 token 反而 +2.7%：两个子代理各自要重发自己的 system prompt（约 1.5 万 token/轮），
   21 轮就是 30 万+，正好抵掉父会话省下的部分。任务再长（100+ 轮）或者把子代理的
   system prompt 裁薄，天平才会翻过来。
2. **共享内核会被子代理拖死**。sa2 的云修补把内核卡进 600s 执行超时，看门狗硬杀内核，
   父会话随后丢了全部内核变量（那 8 次报错就是它在重建状态）。共享内核是刻意的取舍，
   但要让子代理的 `execute_code` 超时比父更短（`kernel_exec_timeout_seconds` 是全局的，
   后续可给子代理单独设一档），否则一次卡死会同时惩罚父子两边。

另外：**不给提示，模型不会主动拆**。一个 25 轮的纯 GDB 交付任务与这次组合任务，只在工具描述里写
「何时该拆」时，两次都是 0 次 delegate；在 system prompt 里加一条带门槛的指引
（`loop.py::_delegate_hint`，仅在 `delegate` 注册时出现）之后，第一个回合就派了 2 个。

## 瘦身 + 收窄超时之后（2026-10-22 晚复跑）

同一任务、同一模型再跑一次：18 轮 / 737,801 token / 1250.8s /
父 prompt 峰值 63,289 / **0 个子代理** / 19-20 断言。

**这次没有派子代理，所以它不是瘦身的 A/B。** trace 里模型明确权衡过一次：
「And delegate? Tasks are independent but manageable directly.」——指引只让它**考虑**，
不是让它**必拆**（两跑 1/2 派）。加上 28 / 21 / 18 轮的跑间波动，单跑 A/B 的
token 差（−46%）不能归因于瘦身；能确定的只有确定性部分：技能段 13,190 → 6,100 字符，
每个子代理每轮省 7,090 字符（约 3k token），21 轮量级 ≈ 6 万 token，占总 token 5% 左右
——**单靠瘦身翻不了天平**。

这轮真正的收获是两个 bug（都已修）：

1. **看门狗从来没杀掉过内核**。`jupyter_client` 的 `KernelManager` 没有 `kill()`，
   两处 `km.kill()` 每次都抛 AttributeError 被吞掉；卡死的 ArcGIS 进程一直活着攥着
   `.tif` 句柄，下一次执行直接吃 PermissionError。现在走 `shutdown_kernel(now=True)`
   （退回 `_kill_kernel()` / `provisioner.kill()`），并清掉 `_km`/`_kc`——否则下一次
   `start()` 会 new 一个 manager，旧进程变僵尸。
2. **`kernel_density` 的断言指向一个不存在的文件**（`{{out_dir}}/kde_all.tif`），
   多期跑法下每次都是「执行成功，断言 0/1 通过」，把模型引去修一个本来没坏的产物。
   断言现在支持 `{{output}}`（取返回值里的第一个产物）。实测该配方 1/1 通过，
   四期 max 分别 6.01 / 4.84 / 2.19 / 2.24。

## 已知限制

- **内核命名空间共享**：子代理的变量会留在内核里。prompt 要求加 `sa_` 前缀，
  但没有硬隔离（真隔离要 `exec` 到独立 dict，traceback 行号会失真，暂不做）。
- **串行**：子代理跑的时候父空转。真要并行得改内核池，成本和 GP 争用另算。
- **深度 1**：子代理不能再派子代理。
- **子代理没有 `ask_user`**：缺信息只能 `finish` 说明，由父决定要不要问用户。

## 怎么验收一次改动

```bash
# 跑任务（A/B 用 engine.subagent_enabled 切换）
gis-agent loop "<你的任务>"

# 看指标：轮次 / 总 token / 单次 prompt 峰值 / 压缩次数 / 派了几个子代理
# （trace 在 workspace/.gis_agent/traces/<run_id>.jsonl）
python - <<'PY'
import json, io, glob, sys
p = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("workspace/.gis_agent/traces/*.jsonl"))[-1]
calls, peak, comps, delegates = 0, 0, 0, 0
for line in io.open(p, encoding="utf-8"):
    d = json.loads(line)
    if d.get("event") == "llm_result":
        calls += 1
        peak = max(peak, int((d["payload"].get("usage") or {}).get("prompt_tokens") or 0))
    elif d.get("event") == "context_compacted":
        comps += 1
    elif d.get("event") == "tool_call" and d["payload"].get("tool") == "delegate":
        delegates += 1
print("calls", calls, "peak_prompt", peak, "compactions", comps, "delegates", delegates)
PY
```

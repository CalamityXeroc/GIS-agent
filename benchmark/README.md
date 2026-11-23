# Benchmark 说明与"标准答案"判定策略

本目录是 GIS Agent 的回归基准。目标不是追求与人工答案像素级一致，而是可复现地衡量
"自然语言 → GIS 方法论 → 正确产物"的端到端能力。

## 数据与答案来源

- 原始输入：`workspace/input/13第十三届全国大学生GIS应用技能大赛（兰州交通大学）/数据/{B上午,B下午}/...`
- 参考成果：
  - AM：`AM/AM/1-Process/result_data.gdb`（含 county_1、city_pop、city_pop_20xx_hl 等）
  - PM：`PM/PM/1-Process/result_data.gdb`、`PM/PM/3-成果图/*.jpg`
- 任务文本：`benchmark/specs/*.txt`（由 docx/pdf 解析）

## 标准答案判定优先级（本项目自定）

1. **结构断言优先**：产物存在、坐标系、要素数>0、关键字段存在、栅格非常量、图片非空白。
2. **参考 GDB 次之**：若参考成果含同名图层，比较字段名与要素数量级（允许 ±5% 或拓扑修复导致的
   少量差异）。
3. **成果图仅作视觉抽检**：地图类任务用结构断言（布局含图名/图例/比例尺/指北针）+ 非空白 +
   人工抽检，不做像素级比对。
4. **解题文档仅作方法学参考**：用于撰写 `requirements` 与 `pitfalls`，不作为逐字验收标准。

冲突时以上述顺序为准：结构断言 > 参考 GDB > 成果图 > 解题文档。

## 公平性约束

- 基准工作区**只复制任务所需的输入数据**，绝不包含 `结果/`、`result_data.gdb` 等参考答案。
- Agent 的输出写入临时工作区的 `output/`，不污染原始数据。
- 每个任务独立工作区，避免跨任务状态串味。

## 目录结构

```
benchmark/
├── README.md            # 本文件
├── runner.py            # 基准执行器
├── src_guard.py         # src/ 越界闸门（跑前指纹 → 跑后比对 → 越界作废 + 证据归档）
├── guard_selftest.py    # 闸门自测（假 agent 写 src/arcpy/，验证发现+作废+归档）
├── fake_escape_agent.py # 自测用假 agent（只写 src/arcpy/，不碰别的）
├── specs/               # 从 docx/pdf 抽取的任务文本
├── tasks/*.yaml         # 任务定义（prompt + inputs + assertions）
└── reports/             # 运行报告（json + md）
```

## 运行

```powershell
python benchmark/runner.py --task am_task1_quality
python benchmark/runner.py --all
```

## 工作区逃逸闸门（默认开启）

被测 agent 的本职是"让任务通过"，它会为了绕过环境报错去改宿主仓库。DAIR-57 实测：
跑 `probe_quality_score_map` 时 agent 在 run 窗口内往 `src/` 下新建了 `arcpy/` 桥接包，
同一轮里既测引擎又改引擎，结果不可信；影子包还会静默劫持后续所有 run 的 import。

`runner.py` 每次 `run_task` 都自动做三件事（`benchmark/src_guard.py`）：

1. **跑前预检**：`src/` 顶层只允许 `gis_cli`、`baml_client`（pip 生成物除外），且
   `git status --porcelain -- src` 必须为空——上一轮若改的是已跟踪文件，顶层目录一模一样，
   只有 git 看得出来。命中任一 → 直接中止，不开测（基线已被污染，先人工清理）。
2. **跑前指纹**：`src/` 全量文件 sha256 + `git status --porcelain -- src`；agent 结束后重拍。
   噪声已排除：`__pycache__/`、`*.pyc/*.pyo`、`*.egg-info/`（import 与 `pip install -e` 的正常副产物）。
3. **跑后判定**：任何新增/修改/删除 → 该轮 `invalidated=true`、`success=false`、
   `assertions_validated=false`，证据自动归档到
   `benchmark/reports/src_escape_<run_id>/<task_id>/`（`findings.json`、`git_status_src_after.txt`、
   `git_diff_src.patch`、`added/`、`modified/` 现场副本、`README.md`）。**不自动清理现场**。

报告里 `src_guard` 字段记录完整判定（跑前/跑后指纹、变更清单、git 新增条目）。
崩溃/中断那轮写 `<task>_<ts>_crash.json`，字段同样齐备（`src_guard` / `invalidated` /
`invalidated_reason`）：判定不能只在正常返回路径上落，否则下一轮会把污染当新基线。
退出码：`0` 全部成功，`1` 有断言不过或任务崩溃/预检中止（预检中止=没开测），
**`2` 有轮次被判定为越界作废**（结果不可信，与跑失败区分）——`KeyboardInterrupt` /
`SystemExit` / 普通异常同样是先落判定再定退出码，越界时退出码还是 `2`。

```powershell
# 闸门自测（约 10 s，无 LLM、无 ArcGIS）：假 agent 写 src/arcpy/ → 必须发现+作废+归档，场景结束自动清理
# 另含 DAIR-78 回归：KeyboardInterrupt / SystemExit / 异常崩溃带着越界退出 → 判定不能丢、退出码 2
python benchmark/runner.py --guard-selftest
# 零误伤探针（真任务，约 30 s）：全流程 src/ 指纹不变
python benchmark/runner.py --task probe_src_guard_clean --no-publish
```

`--no-guard` 只用于闸门自身排障；正常跑测一律开着。

注意：闸门只能**发现并作废**，不能阻止 agent 写盘。"只读挂载/仓库副本运行"是更硬的方案，
目前未实现——需要时另开 issue。

## 第 14 届（2025）上午 · 实测记录

引擎修复后的一轮完整跑（deepseek-flash 直连，2026-10-20/21）：

| 任务 | 断言 | 轮次 | 耗时 | status |
|---|---|---|---|---|
| task1 动物点位清洗/投影/研究区 | 13/13 | 36 | 1447s | failed（交付后继续打磨至时限） |
| task2a 气象站清洗 + 温降插值 | 16/16 | 51 | 1013s | completed |
| task2b DEM 拼接裁切 + 土地覆盖裁切 | 8/8 | 54 | 1166s | completed |
| task3a 土地覆盖云修补 | 11/11 | 39 | 461s | completed |
| task3b Landsat 影像更新土地覆盖 | 10/10 | 27 | 559s | completed |
| task4a1 4 期核密度图 | 7/7 | 17 | 223s | completed |
| task4a2 分布中心 + 迁移 | 8/8 | 32 | 686s | completed |
| task4b 羚羊与土地覆盖关系 | 8/8 | 24 | 263s | completed |
| task4a 完整（4 图 + 迁移图） | 12/12 | 120 | 1290s | max_turns（交付后继续打磨） |

合计 **95/95 断言通过**。

注：`status` 与 `success` 不是一回事——`success` 以断言全过为准。模型常在
交付完成后继续打磨（补报告、重出图），直到轮次/时限用尽，此时 status 是
`max_turns`/`deadline_exceeded` 但产物已经合格。

这一轮暴露并修掉的问题（都已带回归测试）：

1. **上游流式卡死无界**：代理持续发 SSE 心跳时 httpx 的 read timeout 永不
   触发，SDK 又丢弃注释行，`for chunk in stream` 阻塞 → 整跑静默 2700s+。
   改为看门狗到点强制关流。
2. **验收把已删中间件当缺失**：验收器跑在无 arcpy 的 venv 里判不了 GDB 路径，
   一律"假设存在"，内核侧又如实报不存在，两边打架 → 验收永不过、空转到轮次上限。
   改为以内核结论为准静默剔除，缺的交付物交给语义验收点名。
3. **统一图例断言扫错范围**：把迁移图（流向图，本无分级）也算进"4 张密度图
   共用图例"里。限定 pattern。
4. **分类面积断言写死分辨率**：`scale` 按 15 m 像元算，模型输出 10 m 掩膜时
   面积被放大 2.25 倍。改为优先读栅格实测像元面积。
5. **create_map 不落 .spec.json**：只有制图配方写边车文件，模型改用 create_map
   出图时一个 spec 都不留 → 验收判"未统一图例"。两条路径统一。
6. **修复调用吃掉整跑预算**：一次修复 420s 预算 + 8 次退避能烧 950s。
   给修复调用单独封顶 150s。
7. **等内核执行锁无上限**：上一次执行卡在无法中断的 GP 调用里时，后续 execute
   永久阻塞。改为 acquire(timeout=执行预算+宽限)。

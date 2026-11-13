# 分析正确性双复核（P1-5 / W5）

"交付的图很漂亮，但数字是错的"——这是 GIS 成果最贵的一种失败。双复核解决的是
**结论与数据是否一致**：把主代理的结论交给一个**看不到主轨迹**的复核子代理，让它用
守恒/拓扑/量纲断言和独立复算重新验一遍。

## 为什么必须"盲"

复核者只要能看到主代理的步骤、日志或中间推理，就会顺着主代理的口径复述一遍
（"它先 buffer 再 intersect，所以结果应该对"）。W5 的接口因此把输入收窄成三块：

```
主代理（AgentLoop）                    复核子代理（另一个 AgentLoop）
├─ 自己的消息/任务清单/trace      ✗     ├─ 只看到：任务目标
├─ methodology（怎么做）          ✗     │        + 主代理结论 claim（数字/产物）
└─ review_result(claim) ──────────────► │        + 系统量测 facts
                                        │        + 断言库
                                        └─ finish 回严格 JSON 判定
```

`ReviewClaim`（`src/gis_cli/review/claims.py`）**没有 methodology 字段**——它是"怎么做"的
叙述，属于主轨迹。claim 只允许：任务目标、结论 summary、自报数字 metrics、产物路径、
单位、可解释的变化（`expected.count_change` 等）。

复核子代理由 `engine/subagent.py:SubAgentRunner.run_review()` 启动，复用现有子代理设施
（独立对话、独立 trace、共享内核），但工具表只剩 5 个只读工具：
`assertion_check / catalog_query / read_file / execute_code / finish`。
子代理配置里显式 `review_enabled=False`，防止复核者的 finish 再触发一次复核。

## 断言库（独立于配方，也独立于 LLM）

`src/gis_cli/review/assertions.py` 里 18 条声明式断言，只吃 `claim` + `facts`：

| 类别 | type | 抓什么 |
|---|---|---|
| 守恒 | `conservation.sum_equals` | 分项合计 ≠ 总量 |
| 守恒 | `conservation.matches_source` | 声明总量 ≠ 源数据量测总量 |
| 守恒 | `conservation.partition_matches_source` | **把 A 区人口搬到 B 区**（合计不变也能抓） |
| 守恒 | `conservation.area_partition` | 分区面积合计 ≠ 量测总面积（重叠/遗漏/口径）；有逐分区面积量测时同时**逐区核对**（合计相等但分区之间互换也能抓），只有总量时给 `conservation.partition_unverified`（warning），不再假装全过 |
| 守恒 | `conservation.count_matches` | 声明要素数 ≠ 数据要素数 |
| 拓扑 | `topology.geometry_valid` | 自相交/非法几何 |
| 拓扑 | `topology.no_overlap` | 结果面重叠（同一块面积被算两次），且结论自报的重叠率必须与量测一致 |
| 拓扑 | `topology.coverage` | 覆盖缺口 |
| 拓扑 | `topology.within_extent` | 结果超出研究区 |
| 拓扑 | `topology.count_change` | 要素数变化没有解释，或解释里没给出与变化对得上的数量（“已按字段去重”这种空口解释不算） |
| 拓扑 | `topology.relation_count` | 声明“有多少要素与其他要素相邻/包含/被包含”≠ 量测数量（12 个地块只 3 个相邻却写“全部相邻”）；量测只有反方向（量测 `within`、结论写 `contains`）也报 `relation_direction_mismatch` |
| 量纲 | `dimension.crs_units` | 投影坐标系却声明"度"，或地理坐标系直接按米算 |
| 量纲 | `dimension.area_unit` | 面积差 10/1000 倍（m²/km²、公顷/km² 混用） |
| 量纲 | `dimension.distance_unit` | 距离差 10/1000 倍（声明 500 米、实测 500 公里） |
| 量纲 | `dimension.density` | 密度 ≠ 计数 ÷ 面积、分母单位不一致；**分子/分母还要回到源数据勾稽**（人口按“十人”为单位、面积被截断都抓得到） |
| 量纲 | `dimension.precision` | 声明值写了几位小数就只能要求精确到那一位（`152.34` 写成 `152.0` 不再因为 2% 容差静默通过） |
| 量纲 | `dimension.value_range` | 声明的最小/最大值超出量测值域 |
| 量纲 | `dimension.raster_cell_area` | 像元数 × 像元尺寸² ≠ 声明面积 |

调用方式（纯函数，不读盘、不联网、不碰 arcpy）：

```python
from gis_cli.review import ReviewClaim, collect_facts, run_assertions

facts = collect_facts(["output/zones.shp"], workspace=ws, runner=code_runner,
                      group_fields={"POP": "ZONE"})
claim = ReviewClaim(goal="按城市汇总人口", metrics={"total_pop": 8420000,
                     "zone_pop": {"Z01": 2100000, "Z02": 6320000}},
                     units={"area": "km2"})
report = run_assertions(claim.assertions, facts, claim)
report.ok, report.to_dict()["findings"]
```

约束：

- **缺事实** → 该条断言进 `skipped`（"没检查" ≠ "检查通过"）；
- **断言写错/参数类型错** → `errors`，且 `report.ok` 必为 `False`、`summary` 以「断言库未跑成」开头
  （以前只记 errors 不改 ok，报告会写「断言全部通过」）；
- **只校验了一部分** → 用 `severity="warning"` 的 finding（不影响 `ok`，但会出现在 summary/findings 里），
  例：`conservation.partition_unverified`；
- 每条断言只判一个口径，失败时带 `claimed` / `measured` 两个数。

## 量测事实

`src/gis_cli/review/facts.py:collect_facts()` 有两条通道：

- **ArcPy 内核**（生产路径）：要素数、面积、CRS、非法几何（CheckGeometry）、重叠率
  （≤150 个面的两两求交）、相邻/包含/被包含关系（≤150 个面，数据集内部：共享边界/角接/重叠的
  要素数与对数；`within`/`contains` 两个方向，计数为 0 时不写，避免跨数据集关系被误判）、
  字段统计、分组汇总（声明了 `group_fields` 时还会多给一份**按分组字段汇总的面积**
  `group_sums.AREA_M2` / `AREA_DEG2`，供 `area_partition` 逐分区核对）；栅格给像元数/像元尺寸/min/max；
- **纯 Python 回退**（无 ArcGIS 时）：CSV / JSON / GeoJSON 的计数、Shoelace 面积、
  环自相交、字段统计、分组汇总。shp/.gdb/.tif 没有内核时明确标
  `collector=unavailable`，断言据此 skipped，绝不猜。

## 触发与开关

计划原文是「只在交付级任务开启（token 翻倍），开关进 `requirements`，默认关」。
这里的 `requirements` 指**任务定义里的验收要求**（benchmark YAML 的 `requirements:` 列表、
`TaskState.requirements`），不是某个叫 `requirements` 的配置段。所以开关落在两处，都默认关：

1. **任务侧（交付级）**：benchmark YAML 顶层 `review: true`，或 `requirements` 里任一条
   带 `review: true`（只对某条交付要求开）。`benchmark/runner.py:review_switch()` 在
   `build_engine()` **之前**把它定下来，因此关着的时候复核工具压根不进工具表。
2. **引擎侧**：`config/llm_config.json` 的 `engine` 段（与 `subagent_enabled` 同层），
   适合生产环境统一打开：

```json
"review_enabled": false,
"review_max_per_run": 1,
"review_max_turns": 20,
"review_timeout_seconds": 600,
"review_exec_timeout_seconds": 300,
"review_block_on_fail": false,
"review_require_before_finish": true
```

不做运行时（`loop.run(requirements=...)`）切换：工具表在 `build_engine` 时就定好了，
中途开启只会让模型看到一个不在表里的工具，或白花一轮去调一个不可用的工具。

- 关掉时 `assertion_check` / `review_result` 都不进工具表（与 `delegate` 同样处理），
  模型不会白花一轮去调一个不可用的工具。
- 开启后 `finish` 前必须调用一次 `review_result`（只拦一次；第二次放行并在 summary 里
  写"未执行双复核"，避免无限循环）。产物变更后 `signature` 不一致，要重做复核。
- `review_block_on_fail` 默认 **false**：复核 fail 只提示、不阻断。打开后 fail 会拒绝
  finish，最多拒 `max_finish_rejections` 次，之后放行并要求人工复核 findings。

复核子代理的 finish 摘要约定（严格 JSON）：

```json
{"verdict": "pass|fail",
 "findings": [{"code": "conservation.partition_mismatch", "severity": "error",
               "detail": "Z02 声明 632 万，量测 230 万",
               "claimed": 6320000, "measured": 2300000}],
 "checked": ["conservation.sum_equals"], "notes": ""}
```

解析失败 = `unparsed`（**不**当通过，但默认不阻断）；没跑完 = `max_turns/error/...`，
同样不能算通过。

## 验收口径（W5）

`python -m gis_cli.review.eval`（纯 Python，毫秒级）：

```
任务 32 个，注入 26 个错误
抓出 26/26 = 100.0%（验收 ≥50%）
误杀 0/32 = 0.0%（验收 ≤10%）
仅 warning（未判错，不算抓出/误杀） 3 条
  conservation: 10/10   topology: 7/7   dimension: 9/9
```

- **32 个任务**取 `benchmark/tasks/` 的既有形态（投影/拓扑/连接/汇总/密度/插值/栅格/
  适老化/绿地/服务区…），每个都有一份干净样本；
- **26 个注入错误**：前 20 个是断言清单与注入同源设计的老样本；后 5 个（`w5x_*`）是审查侧
  另造的**非同源**形态，专测“没见过的错误”（见下节「同源偏倚」）；另 1 个是整数人口声明
  少写 8000（0.095%，DAIR-199 项 4，配合下面「整数声明不放地板」的 ±0.5 末位判据）；
- **抓出** = 注入样本命中至少一条预期断言；**误杀** = 干净样本出现任何 error 级发现。
  误杀率的分母是 32 个干净任务（不是只跑注入样本）。
- **0 误杀必须是「真查过」**：每条干净样本至少要有 1 条断言进入 `checked`（不是全 `skipped`），
  否则这个 0 没有意义。`run_eval()` 会把这种用例列进 `unchecked` 并让入口返回非 0；
  `tests/test_review_assertions.py` 里也把它当断言。
- **warning 单列计数**：`warning_hits` / `warning_hit_cases` 报干净样本上只出 warning、没出 error
  的用例数。它既不算抓出也不算误杀，**不进** `ok` 判据（避免基准口径漂移）——但「一条 error 都没出、
  全靠 warning」的样本对账强度是打折的，看报告时要连这条一起看（DAIR-195）。

**口径边界（必须说清）**：上面量的是**断言层**的抓出/误杀，不是 LLM 复核者的端到端
表现。复核 subagent 调用的就是这套断言，但 LLM 是否按提示调用、是否被措辞带偏，
需要真实模型预算才能量；本轮没有跑端到端 LLM 基准（W5 的 token 会翻倍）。
`tests/test_review_subagent.py` 用 stub LLM 覆盖了协议层：上下文隔离、工具收窄、
JSON 判定解析、限额、finish 闸门。

**同源偏倚（100% 只对已知形态成立）**：20 个注入错误与断言清单是同一批人设计的，
“抓出 20/20” 只说明清单覆盖了**注入形态**，不代表断言库能抓任意错误。W5 双复核另造了
5 个避开原有形态的新错误，走同一条 `run_assertions` 路径，**当时抓出 0/5**；DAIR-185 把这
5 类固化进评测集（`w5x_*` 5 条 + 对应断言），现在抓出 5/5：

| 新错误 | 当时的漏因 / 现在靠什么抓 |
|---|---|
| 分区人口 A/B 整段互换（合计守恒） | 清单只给了 `sum_equals` → 补 `conservation.partition_matches_source`（清单缺项，不是实现写错） |
| 12 个地块只 3 个相邻，结论写“全部相邻” | 库无相邻/包含方向断言 → 新增 `topology.relation_count`（ArcPy 路径量相邻要素数） |
| 重叠率实测 0.42%，结论写 0.0% | `no_overlap` 不比对 claim 自报比率 → 补上比对；claim 没写比率仍 `skipped` |
| 要素数 118→113，用“已按字段去重”空口解释 | `count_change` 只要求 reason 非空 → 要求解释里给出与变化对得上的数量 |
| 缓冲距离声明 500 米、实测 500 公里 | 无距离类量纲断言（`crs_units` 只查米/度）→ 新增 `dimension.distance_unit` |

第 1 例只要清单补上 `conservation.partition_matches_source` 就立刻抓到——漏的原因是
**清单覆盖了注入形态**，不是断言实现写错。所以 100% 只代表对已知形态有效，
**不是对外承诺**；换一批没见过的错误形态，先补断言清单再谈抓出率。

## 静默判通过（DAIR-191）

DAIR-182 独立造 20 个错误时抓出 17/20，漏掉的 3 条里有 2 条是**静默判通过**：
`checked` 非空、summary 写“断言全部通过”，复核者会当 pass 上报。比漏抓更危险。
现在这 5 类都有回归用例（`tests/test_review_silent_pass.py`）：

| 缺陷 | 改前 | 改后 |
|---|---|---|
| 密度分子无勾稽（人口按“十人”为单位，32000 写成 32 万） | 三个数自洽 → 通过 | `dimension.density` 把分子拉回源数据同名字段/分组汇总对账 → `density_mismatch` |
| 精度截断（152.34 写成 152.0，0.22%） | 落在 `rel_tol=0.02` 里 → 通过 | `dimension.density` 的分母按声明位数对账；另新增 `dimension.precision` |
| 断言内部抛异常（`matches_source` 被喂分组字典） | 只进 `errors`，`ok=True` | `ok` 加 `and not errors`；summary 先报“断言库未跑成”；错误文案可读 |
| 关系方向反（量测 `within`、结论写 `contains`） | 缺 `contains` 事实 → skipped | 断言能拿反方向事实判定 `relation_direction_mismatch`；量测侧同时产出 `within`/`contains` |
| 分区面积交换（合计不变，只给总面积） | `area_partition` 只看合计 → 通过 | 量测默认多给分区面积汇总 → 逐区核对；只有总量时给 `partition_unverified`（warning） |

DAIR-182 的 `run_errors.py` 重跑（脚本一字未改）：自然清单 **17/20 → 20/20**，全量清单 17/20 → 19/20。

同一路径上 DAIR-195 又收了两处“看着像对过账、实际没对”：多分组数据集的面积键是
`AREA_M2_BY_<cat>`，以前 `_AREA_KEY_UNITS` 不认 → `area_partition` 永远只给 warning；声明域内
0 候选、域外有同名量测时 `candidates=[]` → 静默通过（现在给 `dimension.density_unverified` warning，`ok` 不变）。

## 已知限制

- **复核者可能误杀**：断言是确定性的，但 LLM 会补充判断；`review_block_on_fail=false`
  是默认值就是为了不因一次误判毁掉交付。
- **量测覆盖有限**：重叠率只对小数据集两两求交（≤150 面）；非法几何在 shp 上依赖
  `CheckGeometry`；度² 与米制面积不可比时断言直接 skipped，不做猜测。
- **相邻/包含/距离断言要有量测事实**：相邻/包含关系只在 ArcPy 路径、面数 ≤150 且几何全量收集时量测
  （纯 Python 回退不猜，直接 skipped），且只量**数据集内部**的关系（结果面 vs 研究区这种跨数据集的
  不量测）；`collect_facts` 不产出“缓冲距离”这类距离量测，
  `dimension.distance_unit` 只有在 facts 里显式给出距离时才会检查，否则 skipped。
- **精度容差有 0.1% 地板（整数声明除外）**：`dimension.precision` 与 `dimension.density` 的分子/分母对账用
  “声明位数（半个末位）”，小数声明不比 0.1% 更严——同一份数据用不同算法/投影算出来的差异是正常噪声。
  所以 0.2% 以上的截断能抓（`152.34 → 152.0` 是 0.22%），0.05% 这种量级不报。
  整数声明（`decimals == 0`）不放地板，按 ±0.5 的最小单位判：`pop=8_412_000` vs 量测 8_420_000
  （差 0.095%）会报——写成整数就是声称精确到个位；量测本身有取整误差时用 `cross_rel_tol` / `min_rel` 显式放宽。
- **逐分区面积核对依赖分组字段**：`group_sums` 的面积键在 `group_fields` 声明了一个分组字段时是
  `AREA_M2`、声明了多个时是 `AREA_M2_BY_<cat>`（`_AREA_KEY_UNITS` 两种都认）；没声明就只能校验合计，
  `area_partition` 会给 `partition_unverified`（warning）而不是报错。多个分组键都能对上 claim 的分区标签
  （两个分组字段都用 `A/B/C`、都用行政区代码这类撞名）时先用 claim 的分区值判定：有一个键能逐区对上就是它
  （DAIR-197 的「拿第一个键误判 error」），全都对不上才报 `partition_mismatch` 并列出全部候选键
  （DAIR-201：以前只要标签有一个交集就降级 warning，标签偶然同名的无关分组也能把逐区核对整条打掉，
  claim 分区全错甚至整段对调也会是 `ok=True`）。
  撞名时「按值挑键」对**分区互换**无能为力（claim 恰好与另一个分组键的数值一致就会挑错），
  要精确指定就在 spec 里写 `group_field: "KIND"`（DAIR-199）：先按声明的分组字段过滤候选，
  只比 `AREA_M2_BY_KIND`；单分组数据集的键是 `AREA_M2`/`AREA_DEG2`（`facts.py:544`），没有后缀可比对，
  声明 `group_field` 时直接按这唯一的键逐区比；若该键的分区标签与 claim 无交集，文案会明确说「没有交集」，
  不再回落到「只有总面积…让 collect_facts 声明 group_fields」（DAIR-214）；声明了但量测里没这个键时**不拿别的键凑数**，给一条说得清的
  `partition_unverified`（warning）。**`partition_unverified` 不是「逐区核对通过」的等价替换**：
  它只是没核对，claim 的分区可能全错，报告不能拿它当分区校验过了；走到这条回落只剩三种情况——
  `parts` 写成数组（claim 没给分区标签，此时合计对得上就一律 `partition_unverified`，有多个分组键时
  文案会列出全部候选键；合计也对不上时只报 `area_mismatch`，不再叠一条 `partition_unverified`）、
  量测里没有面积分组汇总（没声明 `group_fields` / 标签全对不上）、或 spec 声明的 `group_field`
  没有对应键（文案会点名声明的分组字段与实测到的键；声明的键存在、只是分区标签与 claim 无交集时，
  文案会点出该键在、是标签对不上——DAIR-209）。
  0 面积的分区也会写进 facts（不按真假值过滤），否则 claim 写 0 的分区会被反报「多出分区」。
- **`dimension.density` 的自动勾稽是启发式**：先看 `claim.data_paths` 声明的数据集——没声明、或声明的路径
  一条都对不上 facts 的键时退回扫全部。分子按 `count` 键名找同名数值字段/分组汇总（`total_pop` → `POP`），
  分母收集所有 `area_total` 候选、任一在容差内即通过（全不符时报最接近的那个）。对不上时可显式传
  `count_source_path` / `area_source_path`（或 `cross_check="strict"`）按 error 报，`cross_check=false` 关掉。
  严重级别按口径明确程度分三档：声明了多个数据集时自动命中的不一致降为 warning（分不清「写错了」和
  「同名不同量」）；只声明一个数据集**且路径对得上**时按 error；只声明一个但路径**漂移**（一条都对不上）
  同样退回扫全部并降为 warning。声明域内 0 候选、域外有同名量测时不静默通过：给
  `dimension.density_unverified`（warning，「另有 N 处未声明数据集的同名量测未参与对账」），`ok` 不变。
  分子（计数）和分母（面积）两侧都适用：域内一个 `area_total` 都没有、只有域外有时同样给这条 warning（DAIR-199）。
- **只比对自报数字，不读叙述**：claim 里没写相邻要素数/重叠率/距离时，对应断言 skipped；
  纯文字结论（写“全部相邻”却不给数）仍抓不到——断言层只做确定性的数字对账。
- **GeoJSON 的自相交只认「真交叉」**：连续重复顶点（导出工具常见）与「捏点」（子环在一个
  顶点相接）不算非法几何——它们不改面积口径，判成非法只会误杀。实测 `中国_省.geojson`
  42 个要素、海南省 328 个子环，旧实现报 5 个非法几何（全是连续重复顶点），修复后 0 个；
  真自相交（bowtie）仍能抓出（`test_geojson_duplicate_vertices_and_pinch_are_not_self_intersection`）。
- **GeoJSON 不推断 CRS**：没有显式 `crs` 成员时 `is_projected=null`、`area_unit=null`，
  量纲断言 skipped。这是故意的：产物存成经纬度、面积却是先在投影里算好的，是很常见的
  合法做法，按“地理坐标系就一定是度²”去判会制造误杀。
- **facts 会截断**：复核 prompt 上限 22k 字符；超过时保留计数/面积/CRS/分组汇总，
  其余让复核者用 `assertion_check(paths=...)` 重新量测。
- **共享内核**：复核与主代理共享 ArcPy 内核，变量加 `rv_` 前缀；复核者的
  `execute_code` 超时 300s，卡死会重启内核（父变量丢失，`kernel_restarted` 会回传）。
- **不做图面质检**：W5 只判数字/几何/单位；图面由 Verifier 的 vision/hygiene 负责。

## 验收命令

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m gis_cli.review.eval
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_review_*.py -q
```

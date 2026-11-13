# DAIR-289 修复：`arcpy.env` 在执行层按次隔离 —— 配方第 1 步不再被跑前 env 带偏

- 被验版本：HEAD（本单改动）；对照版本：`d643274a`（DAIR-138 收尾，本单缺口）
- 缺陷来源：DAIR-288 审查结论里"要单开"的那一条（DAIR-135 / DAIR-138 只修了"跑完不许留下"）
- 环境：Windows + ArcGIS Pro 3.7（arcpy / py3.13）+ 引擎 `.venv`（py3.12）
- 真数据：`benchmark/reports/p0-1c_dual_backend_20261102/work/{county4508.shp,rasterA_4508.tif}`
  （都是 EPSG:4508，385 个分区，栅格 cell=1000）

## 1. 根因

`arcpy.env` 是持久内核里的**进程级全局态**。11 个内置配方的 `CopyFeatures`（各配方的第 1 步）
都发生在各自 `try` 之前（`zone_raster_sum.yaml:78` vs `:104`），配方级的 `finally` 只能守住
自己 `try` 之后的代码。于是"跑前 env 里有别的任务留下的 `extent` / `outputCoordinateSystem`"
会让第 1 步就把分区**静默重投影 + 按范围裁掉**，后续 `ids` 为空，错误最终落在
`np.bincount(... int(ids.max()) + 1)` 上，报一个和真因毫无关系的错。

逐个配方补 `finally` 补不到"最早那一步"，所以隔离放在唯一不遗漏的位置：内核执行层。

## 2. 修法（方案 A，一次性收口）

`src/gis_cli/runtime/env_guard.py`（新）+ `ArcpyKernel.execute()`：

```
快照 6 项（extent / snapRaster / outputCoordinateSystem / cellSize / mask / overwriteOutput）
  → 盖成干净基线（前 5 项 = None，overwriteOutput = True）
  → 用户代码
  → 还原快照
```

- 快照在 `execute()` 前缀**改写之前**拍，记的才是"这次执行之前"的外部态；
- 成功路径：还原语句接在取结果尾巴之后，**不多一次 ZMQ 往返**（`test_...` 里锁死
  `len(calls) == 1`）；
- 失败路径：用户代码抛异常会让 cell 中断、尾巴不执行，`_repair_after_failure()` 补一趟
  （顺带把 DAIR-13 的 cwd 归位并进同一趟，`len(calls) == 2`）；
- 还原本身吞异常（快照里可能攥着已被删的数据源），失败只往 stderr 留一行诊断 ——
  一次成功的执行不能被"还原"翻成失败。

**语义变化（有意为之）**：`arcpy.env` 从此是**按次作用域**，上一步设的 extent/mask 不会
自动带进下一步。这正是本单要的效果（"跑前不许被已有 env 带偏"）；要跨步传的值放变量里。
`arcpy.env.workspace` 不在隔离范围 —— 它由执行前缀每次按任务工作区重新钉死。

配方级的 `finally`（DAIR-135/138）保留：没有内核的子进程回退路径（`CodeRunner._run_subprocess`）
靠它，而且它是内核层之外的冗余保险。

**同一次执行里设的 env 照常生效**（真 arcpy 实测）：

```python
runner.run('arcpy.env.extent = arcpy.Extent(0,0,10,10)\n'
           'arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(4326)\n'
           'set_result({"inside": str(arcpy.env.extent)})')
# → inside = '0 0 10 10 NaN NaN NaN NaN'（期间生效）
# 裸读（_execute_raw）→ EXT=None CS=None（跑完不跨步）
```

## 3. 修复前后对照（`probe_env_guard.py`，同一内核、同一套参数）

预置方式：走 `kernel._execute_raw()`（**绕过**本单新加的按次隔离）往内核里塞
`extent = (0,0,1000,1000)` + `outputCoordinateSystem = EPSG:4326` —— 与 county4508 的
EPSG:4508 范围完全不交，模拟"别的任务留下的状态"。

| 观察项 | `d643274a`（修复前） | HEAD（修复后） |
|---|---|---|
| 预置污染态跑 `zone_raster_sum` | ❌ `ValueError: zero-size array to reduction operation maximum which has no identity`（= DAIR-288 原文） | ✅ 成功，断言 3/3 |
| `zone_count` / `total_value` | 无结果 | **382 / 29,601,147.237510424** |
| 干净 env 下同一配方 | 382 / 29,601,147.237510424 | 382 / 29,601,147.237510424（**逐值相同**） |
| 跑后 env | 回到预置值 | 回到预置值（**逐字段**：extent `(0,0,1000,1000)`、CS `4326`） |
| 干净 env 跑完 | `overwriteOutput` 留成 `True`（历史泄漏） | 6 项全是 None/False，**逐字段不变** |
| 同内核紧接 `zonal_statistics` | 成功，`record_count=385` | 成功，`record_count=385` |

证据 JSON：`evidence_env_guard.json`（HEAD，`ok=true`、`errors=[]`）；对照树同一脚本跑出
`errors` 含 zero-size array（`ok=false`，`d643274a` 树上生成，不入本目录）。

### 预置不同时的症状（Pro 3.7 实测，同一根因）

| 预置 | 修复前表现 |
|---|---|
| `extent=(0,0,1000,1000)` + CS `4326` | `ValueError: zero-size array ...`（本单复现的那条，静默裁空） |
| 只 `extent=(0,0,1000,1000)` | 同上（zero-size array） |
| CS `32610`（+任意 extent） | `ERROR 000210: 无法创建输出 ... / ERROR 080467: Failed to project input data`（第 1 步直接挂，而不是静默裁空） |

三类都是"第 1 步 `CopyFeatures` 在别人的 env 下跑"，修好后三类都成功且结果与干净 env 一致。

## 4. 回归测试（新增 `tests/test_kernel_env_guard.py`）

用例**真跑** `execute()` 拼出来的那段代码（`_execute_raw` 换成 compile+exec 到同一个持久
命名空间，配假 arcpy，其 `CopyFeatures` 按真 arcpy 的脾气办事：`env.extent` 非 None 就把
范围外要素裁掉 → 0 行）：

| 用例 | 守什么 |
|---|---|
| `test_guard_scope_is_the_six_env_keys` | 隔离范围 = 本单点名的 6 项，不多不少 |
| `test_first_call_is_not_poisoned_by_preset_env` | 预置污染后第 1 步拿到 1 行（不是 0 行）；期间 5 项为 None、`overwriteOutput` 为 True；成功路径只 1 趟 |
| `test_env_restored_after_success` | 跑完 6 项逐字段回到预置值（含 `overwriteOutput`，证明快照早于前缀改写） |
| `test_env_restored_after_failure` | 用户代码抛异常后错误路径补一趟并还原 |
| `test_removing_restore_goes_red` | **负向对照**：把还原挪掉后残留可见（上面的断言不是空转） |

**"把守卫挪掉就红"实测**（临时改 `kernel.py` 后跑该文件，改完已还原）：

- 挪掉"快照+基线"：`test_first_call_...` / `test_env_restored_after_success` /
  `test_env_restored_after_failure` / `test_removing_restore_goes_red` 4 个红；
- 只挪掉"还原"（成功尾巴 + 失败补丁趟）：两个 `test_env_restored_*` 红。

**防假阴性**：`probe_env_guard.py` 打印并断言 `gis_cli.__file__` 落在被验树里
（HEAD 为 `E:\BST_lab\GIS_CLI\src\...`，对照树为 `...\Temp\dair289_prefix\src\...`）。
DAIR-288 踩过的坑在这里显式兜住 —— 临时树必须用**绝对** `PYTHONPATH`，否则会 import 到
仓库 editable 安装的那份。

## 5. 未改动的部分（回归边界）

- **配方一个没动**：47 个 arcpy 配方的渲染快照逐字不变
  （`tests/test_recipe_backend_templates.py::test_arcpy_render_is_byte_identical_to_snapshot` 通过）；
- **子进程回退**：一次性进程本来就没有"上一次的 env"，不需要守卫；
- **`arcpy.env.workspace`**：仍由执行前缀按任务工作区钉死（未纳入隔离）。

全量回归（本机真 arcpy + 真 GDAL，`-m "not arcpy" -k "not arcpy"`）：
**1486 passed, 4 skipped, 189 deselected**（2348s）。deselect 掉的是本机能真跑的
`[arcpy]` 参数化用例（每例一个内核往返，本机全套 15min+ 就跑不完）；它们覆盖的"同内核
紧接 zonal_statistics = 385" 已由 `probe_env_guard.py` 在真内核上验过。

## 6. 对既有探针的影响（需要知道的一条）

`arcpy.env` 变成按次作用域后，**“先在一个 cell 里把 env 设脏、下一个 cell 跑配方”这种探针
写法不再能造出脏环境** —— 设脏的那一步跑完就被还原了。受影响的是历史证据脚本里的
负向对照，例如：

- `benchmark/reports/dair137_p01c_closeout_20261102/probe_r1_chain.py` 的 `LEAK_CODE` 分支
  （“手动把 extent 设成 DEM 范围后 zonal 必挂”）；
- `benchmark/reports/dair135_extent_leak_20261102/probe_after_fix.py` 的旧模板对照；
- `benchmark/reports/p0-1c_dual_backend_20261102/probe_extent_leak.py`。

它们是一次性证据（不在 CI 里），要重放负向对照得把预置改成裸通道
`kernel._execute_raw(...)`（本单的 `probe_env_guard.py` 就是这么做的）—— 语义上也更诚实：
预置的本来就是“绕过了按次隔离的外部状态”。

## 7. 复现命令

```
# 单测（不依赖 arcpy）
cd E:/BST_lab/GIS_CLI
PYTHONIOENCODING=utf-8 PYTHONPATH=src timeout 900 .venv/Scripts/python.exe -m pytest \
  tests/test_kernel_env_guard.py tests/test_kernel_roundtrip.py tests/test_kernel_hard_kill.py -q

# 真 arcpy 红→绿（约 1 分钟：内核冷启动 ~20s）
PYTHONIOENCODING=utf-8 PYTHONPATH=src timeout 900 .venv/Scripts/python.exe \
  benchmark/reports/dair289_env_guard_20261113/probe_env_guard.py

# 修复前对照（在 d643274a 的 git archive 树上跑同一脚本）
git archive d643274a | tar -x -C <临时目录>
# 那棵树里 shp/tif 没入版本库，数据要指回工作树：
DAIR289_DATA=E:/BST_lab/GIS_CLI/benchmark/reports/p0-1c_dual_backend_20261102/work \
DAIR289_EVIDENCE=<临时目录>/.probe_prefix.json \
PYTHONPATH=<临时目录>/src <venv>/python.exe <临时目录>/benchmark/reports/dair289_env_guard_20261113/probe_env_guard.py
```

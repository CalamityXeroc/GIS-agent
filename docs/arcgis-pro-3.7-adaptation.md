# ArcGIS Pro 3.7 适配方案

调研单：DAIR-281。**本文只出方案，不改源码**（本轮 `src/`、`tests/`、`config/` 一行未动）。
实测环境：本机 `E:\ArcGISPro3.7`（Pro 3.7.0.1901，Python 3.13.13），项目 venv 3.12.6，日期 2026-11-13。
标注约定：**[实测]** = 本机跑过，命令与输出见 §5；**[推断]** = 有证据但未做最终验证。

---

## 1. 结论

**可适配。代码侧改动小（≈2–3 小时），但本机是「环境坏了 + 发现链只信注册表」的双重故障，两件事必须分开做。**

### 结论 A（代码）：3.7 本身能跑，坏的是环境发现

- **[实测]** 子进程契约、持久内核、制图链路在 3.7 / Python 3.13.13 上全部正常：
  - `run_arcpy_code(python_executable=<3.7 python>)` 成功，中文 stdout / `set_result` 数据**逐字一致**（`'中文输出测试 ok'`、`'中文数据'`），`TimeoutExpired` 在 8.0 s 准确触发（E5）。
  - `ArcpyKernel.start()` 27.0 s，`execute()` 返回 `{'v': '3.7', 'cn': '中文结果'}`（E6）。
  - `updateRenderer("UniqueValueRenderer")` → 改色/标签 → `saveACopy` → `createLayout` → `createMapFrame` → `exportToJPEG`（25 860 B）→ 回读 renderer/items 一致（E10）。
- **[实测] 唯一的真缺陷是发现链**：`_build_python_candidates()` 给出的 6 个候选**全部不存在** —— 注册表 `InstallDir` 指向已删的 `E:\ArcGISPro3.6\`，硬编码兜底 `E:\ArcGISPro3.6`、`C:\Program Files\ArcGIS\Pro`、`D:\ArcGIS\Pro` 也都不存在 → `discover_arcgis_pro_python()` 抛 `ArcGISDiscoveryError`，连带 **bridge / 持久内核 / `gis-agent doctor` / Web 健康度** 四条路一起红（E4）。
- **[实测]** Python 3.11→3.13 对**子进程契约没有影响**：runner 只用 `json/io/traceback/contextlib` + `from __future__ import annotations`，结果用 `write_text(encoding="utf-8")`，父进程 `encoding="utf-8", errors="replace"`；`PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` / 丢弃 `PYTHONPATH`、`VIRTUAL_ENV` 在 3.13 上仍是正确隔离方式。全库 grep 无 `cgi`/`distutils`/`imp`/`ast.Num`/`np.float` 这类 3.12/3.13/NumPy 2.0 已移除项（E3）。

### 结论 B（环境，硬阻塞）：Pro 自己的 Python 运行时没注册

- **[实测]** `arcpy.management.CalculateField(fc, "POP", "!POP! * 2", "PYTHON3")` → `ERROR 000574: 未安装 Python。`；默认 `expression_type` 同样是 PYTHON3，一样失败；`CalculateFields(..., "PYTHON3", ...)` 也一样；**同层 ARCADE 版本正常**（E9）。
- **[推断]** 根因是 `HKLM\SOFTWARE\ESRI\ArcGISPro\PythonCondaRoot = E:\ArcGISPro3.6\bin\Python`（该目录已不存在）——Pro 的 GP 工具按 `PythonCondaRoot\envs\PythonCondaEnv` 定位 python.exe（`PythonCondaEnv=arcgispro-py3`）。一步可验：把注册表这两项改成 3.7 后重跑同一条命令（§3 的 V1）。
- **影响面（不是代码 bug，但会表现为"配方失败"）**：
  - `src/gis_cli/recipes/builtin/aggregate_to_zones.yaml:98`、`:122`
  - `src/gis_cli/recipes/builtin/census_city_indicators.yaml:132`、`:162`
  - `src/gis_cli/recipes/builtin/field_calculate.yaml:47`（`:22` 的默认值就是 `PYTHON3`）
  - 以及模型用 `execute_code` 自己写 `CalculateField` 的任何代码。
  - **改 gisagent 代码修不了这条**：要么修环境，要么把这些调用改成 ARCADE / `arcpy.da.UpdateCursor`。

### 结论 C（数据）：3.7 把图例样式重命名了

- **[实测]** `styles.refresh_catalog()` 在 3.7 上枚举出 `NORTH_ARROW 39 / SCALE_BAR 34 / LEGEND 12`；前两类与仓库缓存**完全一致**，`LEGEND` 从 6 项变成 12 项且名称全换（3.6 的 `图例 1..6` → 3.7 的 `中心标题图例`/`手动列图例`/`透明背景图例`…）。当前 `DEFAULT_KEYWORDS["LEGEND"]` 全部落空 → `apply.py:133` 传 `fallback_index=None` → **静默回退 ArcGIS 默认图例**（有 notes 记录，不崩），属出图观感退化（E12）。

### 推荐顺序

1. 修环境（30 分钟，可逆，见 §3 V1）—— 不修则 CalculateField 类配方永远红，且容易被误判成代码回归。
2. 改发现链（2 小时，见 §2 A）—— 修完 doctor / 内核 / Web 一起恢复。
3. 重生成样式目录（10 分钟，见 §2 C）—— 恢复 3.6 图例观感。

---

## 2. 改动清单（`文件:行` + 为什么）

### A. 环境发现（核心，必须改）

| 位置 | 为什么 / 怎么改 |
|---|---|
| `src/gis_cli/arcpy_bridge.py:129-158` `_iter_registry_install_dirs` | 现在**只读 `InstallDir`**（`:149`）。3.7 的 `InstallDir` 是坏的而 `Version=3.7.0.1901` 是好的 —— 说明注册表不能当唯一真相。改为读同一键下的 `InstallDir` + `PythonCondaRoot` + `PythonCondaEnv` + `Version`，返回 `(source, install_dir, conda_env)`；`PythonCondaRoot` 去掉尾部 `\bin\Python` 即安装目录。`PythonCondaEnv` 目前**完全没被读**，而用户克隆环境（`arcgispro-py3-clone`）是 Pro 的常规操作，`:34-37` 里写死的 `arcgispro-py3` 会直接找不到。 |
| `src/gis_cli/arcpy_bridge.py:181-186` | 硬编码 `Path(r"E:\ArcGISPro3.6")`，注释还写着"用户的安装路径"——升级即失效。改为「已知根目录 + 浅层 glob」：`E:\`、`D:\`、`C:\Program Files\ArcGIS\`、`C:\Program Files (x86)\ArcGIS\` 下 `ArcGISPro*` / `Pro*` 各取一层（**不要**递归、不要全盘扫，见 §4 R2）。 |
| `src/gis_cli/arcpy_bridge.py:160-195` | 增加「候选是不是真的 Pro」的校验：`<install>\Resources\ArcPy\arcpy\__init__.py` 存在（**[实测]** 3.7 该文件存在，E1）。**不启动子进程**即可排除 propy 型伪候选，也避免 `Path.exists()` 这种弱校验（见下一条）。 |
| `src/gis_cli/arcpy_bridge.py:34-37` | `ARCGIS_PYTHON_RELATIVE_PATHS` 把 `propy.bat` 与 `python.exe` 并列当解释器，`:175-179`/`:188-193` 会直接把 `.bat` 丢给 `subprocess`。**[实测]** 3.7 的 `propy.bat` 因注册表指向 3.6 直接失败，然后**落到系统 `C:\Program Files\python\python.exe`**（无 arcpy）→ 桥里表现为 `ModuleNotFoundError: No module named 'arcpy'`（E7）；若 `require_arcpy=False`，则**在错误解释器里静默执行成功**。建议：解释器候选只收 `python.exe`，`propy.bat` 单独作为最后兜底并强制 `import arcpy` 探活。 |
| `src/gis_cli/arcpy_bridge.py:198-212` | `discover` 只做 `Path.exists()`（`:202`）。改为「存在 + 属于真实 Pro（上条）+ 版本降序」；多版本并存时选最高版本（本机 3.6 已删，其他机器不会）。异常信息里补「试过哪几条路」，与 doctor 共用（见下条）。`lru_cache(maxsize=1)`（`:198`）保留，`clear_discovery_cache()`（`:215`）已够用。 |
| `src/gis_cli/envcheck/doctor.py:274-281` | `_tried_routes()` 硬编码 `E:\ArcGISPro3.6`，与真实候选链脱节。建议把「试过哪几条路」下沉成 `arcpy_bridge` 的公共函数，doctor 只负责排版，避免两边再次漂移。 |
| `src/gis_cli/runtime/kernel.py:224-232` | 内核侧的发现异常只有英文一句 + 只提环境变量。建议复用同一份诊断（同上）。`kernel.json` 的 env 设置（`kernel.py:234-255`）在 3.13 上实测可用，不用改。 |

### B. 子进程契约（结论：不需要改，只建议补提示）

| 位置 | 为什么 |
|---|---|
| `src/gis_cli/arcpy_bridge.py:222-311` runner | **[实测]** 3.13 下 `exec/compile` + 重定向 + `write_text(utf-8)` 全正常，中文逐字一致。**不动**。 |
| `src/gis_cli/arcpy_bridge.py:315-333` `_build_subprocess_env` | `PYTHONUTF8`/`PYTHONIOENCODING`/丢弃 `PYTHONPATH`、`VIRTUAL_ENV` 在 3.13 仍正确（E5）。**不动**。 |
| `src/gis_cli/arcpy_bridge.py:336-355` `_build_execution_hint` | 建议加 `000574` / `未安装 Python` / `CalculateField` 的映射：「Pro 自带的 Python 运行时没注册（注册表 `PythonCondaRoot` 指向旧版本），需修复安装，或改用 ARCADE」。这条直接决定模型能不能自我纠正。另：现有 `module not found` + `arcpy` → 「请确认使用的是 ArcGIS Pro 自带 Python 环境」（`:349-350`），在 E7 那种"落到系统 Python"场景下是正确提示，保留。 |
| `src/gis_cli/arcpy_bridge.py:359-400` | 发现失败分支的建议语只提环境变量；多级发现落地后同步改文案。 |
| `src/gis_cli/arcpy_bridge.py:1108`、`:1180-1215` | 导出走 `aprx.createLayout(...)` + `layout.exportToJPEG`，**[实测]** 可用（E10）。注意 3.7 的 `Blank.aprx` **0 个 layout**（E13），任何"取 `listLayouts()[0]`"的新代码都会 IndexError —— 现有 `runner/arcpy_runner.py:379-409` 已有 map 兜底，别再引入无兜底的写法。 |

### C. 版本相关数据与文案

| 位置 | 为什么 / 怎么改 |
|---|---|
| `config/cartography_styles.json`（LEGEND 段） | **[实测]** 3.7 名称全变（E12）。用 `cartography.styles.refresh_catalog(template_aprx=<3.7 出的工程>, path=...)` 重新生成；`NORTH_ARROW`/`SCALE_BAR` 两段 3.7 与 3.6 逐字相同，可不动。 |
| `src/gis_cli/cartography/styles.py:31-36` | `DEFAULT_KEYWORDS["LEGEND"] = ["图例 1", "legend 1", "图例 2", "legend"]` 在 3.7 全落空。改成同时覆盖新名（如 `标题和`、`手动列`、`透明背景`），并**保留旧关键字**让 3.6 机器继续可用。 |
| `src/gis_cli/cartography/styles.py:65-70` | `pick_style_name` 的默认参数仍是 `fallback_index=0`（"静默取第一项"），正是该模块 docstring 说要消灭的病。现有调用方 `apply.py:133` 传了 `None`，但默认值该改成 `None`，防止下一个调用方踩。 |
| `install.py:18-27` | `find_arcgis_python()` 自己抄了一份候选表，第一条也是 `E:\ArcGISPro3.6`。改成 `from gis_cli.arcpy_bridge import discover_arcgis_pro_python` 同源。 |
| `scripts/diag/loop_import.ps1:11`、`measure_import.ps1:13`、`wct_arcpy_hang.ps1:12` | 默认参数 `$Py = 'E:\ArcGISPro3.6\...\python.exe'`，现在一跑就报文件不存在。改 3.7 或改为「不传就调 doctor 取」。 |
| `src/gis_cli/envcheck/doctor.py:279`、`src/gis_cli/engine/tools.py:333` | 文案里的 `E:\ArcGISPro3.6`：前者是给用户的"试过哪些路径"，后者是给模型的"外部工程示例"。都不影响逻辑，但 3.7 会让这两处误导人/模型。 |
| `tests/test_envcheck_doctor.py:119-121` | 断言了 `filesystem:fallback` 这个来源标签；改来源命名/新增来源时要同步。其它测试里的 `E:\ArcGISPro3.6\...`（`test_engine_guardrails.py:254`、`test_engine_memory_catalog_log.py:305`）只是 fixture 字符串，**不构成版本假设**，顺手改 3.7 即可。 |
| `src/gis_cli/legacy/**` | 还有若干 `Pro 3.6` 的 API 注释（`legacy/core_tools/tools/execute_code.py:367`、`legacy/agent_legacy/prompts.py:1068`）。该目录已从打包排除，**不动**。 |

### D. 不受影响（已核）

- `src/gis_cli/engine/tools.py` 的"外部产物"拦截按 **workspace 边界**判断，没有写死 3.6 路径（只有 `:333` 的注释举例），3.7 的 `Blank.aprx` 照样会被拒，不需要改。
- `src/gis_cli/recipes/store.py:1756` 遍历 `listLayouts()`，空列表只是不产出元素检查，不会崩。
- `src/gis_cli/web/metrics.py:272-284` 只在发现失败时降级成 warn，发现链修好即自动恢复。
- `pyproject.toml` 的 `requires-python = ">=3.10"` 与依赖版本下界对 3.13 均无冲突（3.7 环境里 typer 0.24 / openai 2.30 / httpx 0.28 / PIL 12.2 都在，E3）。

---

## 3. 验证与回滚

### 验证（按顺序，前两条是环境侧，其余是代码侧）

- **V1 环境（人工操作，Agent 不要直接改 HKLM）**：先 `reg export "HKLM\SOFTWARE\ESRI\ArcGISPro" pro37.reg` 备份，再把 `InstallDir` 改 `E:\ArcGISPro3.7\`、`PythonCondaRoot` 改 `E:\ArcGISPro3.7\bin\Python`，然后重跑
  `arcpy.management.CalculateField(fc, "POP", "!POP! * 2", "PYTHON3")` —— 应成功。这是 §1 结论 B 那条推断的**一步验证**。不生效就用 Pro 安装包做一次修复安装。
- **V2 发现链**：**不设** `ARCGIS_PRO_PYTHON` 时
  `python -c "from gis_cli.arcpy_bridge import discover_arcgis_pro_python as d; print(d())"`
  应返回 `E:\ArcGISPro3.7\bin\Python\envs\arcgispro-py3\python.exe`，且 `source` 能说清来自哪一级（注册表 / 目录探测）。
- **V3 体检**：`gis-agent doctor`（含 `--deep`）全绿，`pro_installed` / `arcpy_importable` / `kernel_start` 三项 PASS。
- **V4 契约回归**：中文 stdout + `set_result` 断言（`stdout.strip() == "中文输出测试 ok"`）、`timeout_seconds=8` 的 `TimeoutExpired`、`ArcpyKernel` 冷启动 ≤ 45 s（本机 3.7 实测 27.0 s，arcpy 冷导入 11.7–11.9 s）。
- **V5 单测**：`pytest -m "not arcpy"` 不低于上轮基线（DAIR-228 记录 1115 passed / 268 skipped，本轮需重测取现值）。
  **注意**：发现链修好后，一批原本 `skip` 的**跨后端一致性**用例（GDAL vs arcpy，如 `tests/test_buffer_dissolve_gdal.py:149`）会**真的跑起来**，passed/skipped 数会变、耗时会长 —— 这是预期变化，不是回归。
- **V6 真配方**：`aggregate_to_zones`（含两处 CalculateField，验证 V1 是否真的通了）+ 一个纯制图配方（验证 E10 链路）。
- **V7 样式目录**：`refresh_catalog` 后 `LEGEND` 命中新名，渲染 notes 里出现 `LEGEND: 使用样式「…」` 而不是「未匹配到样式名」。

### 回滚

- **代码**：改动建议压成 1–2 个 commit（发现链一个、样式/文案一个）。`git revert <sha>` 即回滚。发现链的改造原则是**只增候选、不改旧候选的相对顺序**，且 `ARCGIS_PRO_PYTHON` 的硬覆盖始终优先 —— 回滚后最差退化成今天的 `ArcGISDiscoveryError`，**不会**退化到"错误解释器里静默执行"。
- **数据**：`config/cartography_styles.json` 单独 commit，revert 即恢复 3.6 名字；改 `DEFAULT_KEYWORDS` 时保留旧关键字，3.6 机器不受影响。
- **环境（唯一不可 revert）**：注册表改动先 `reg export` 备份；`ARCGIS_PRO_PYTHON` 环境变量是随时可撤的临时兜底（代码侧永远优先读它）。

---

## 4. 风险与工作量

| 风险 | 影响 | 处置 |
|---|---|---|
| R1 只改 `InstallDir`/`PythonCondaRoot` 不足以恢复 `000574` | CalculateField 类配方继续红 | 用 V1 立刻验证；不行就修复安装 Pro（**推断**，未验证） |
| R2 目录探测写成全盘扫描 | 一条 `find`/`glob` 能吃满整轮 run（本仓库有前科） | 只扫已知根目录 + 一层 glob，禁递归；单条命令带 timeout |
| R3 多版本并存时选到旧版 | 在新机器上跑进 3.6 的残留环境 | 候选按版本降序；返回的 `source` 里带版本号，doctor 展示 |
| R4 样式目录改动让 3.6 出图变样 | 回归/观感争议 | 保留 3.6 关键字做兼容，别按版本拆目录（多一份文件更难维护） |
| R5 `require_arcpy=False` + propy 伪候选 | 在系统 Python 里"成功"执行、结果不可信 | 解释器候选只收 `python.exe`；或强制探活 |
| R6 3.13 生态尾部风险 | 3.15 才会变的行为（实测 `jupyter_client` 触发「无年份 `strptime`」DeprecationWarning） | 现在不用管，记一笔 |
| R7 CalculateField 类配方在环境修好前"假失败" | 被当成代码回归、白查一轮 | 在单测/验收口径里区分「环境未修」状态；V1 先做 |

**工作量估计**：发现链重构 + 测试 ≈ 2 h；文案/脚本清理 ≈ 0.5 h；样式目录重生成 + 关键字 ≈ 0.5 h；审阅 ≈ 2 h；环境修复（注册表/修复安装）≈ 0.5–2 h 人工。**合计 ≈ 1 人日**（不含真机跑全量配方的时间）。

**风险等级**：低（发现链改动可回滚、不碰数据格式、不碰子进程契约）。

---

## 5. 实测记录（2026-11-13，本机）

| # | 项目 | 结果 |
|---|---|---|
| E1 | Pro 3.7 身份 | `arcpy.__file__ = E:\ArcGISPro3.7\Resources\ArcPy\arcpy\__init__.py`；`GetInstallInfo()`：`Version=3.7`、`InstallDir=e:\arcgispro3.7\`、**无 `PythonDir` 键** |
| E2 | 注册表 | `HKLM\SOFTWARE\ESRI\ArcGISPro`：`InstallDir=E:\ArcGISPro3.6\`、`PythonCondaRoot=E:\ArcGISPro3.6\bin\Python`、`PythonCondaEnv=arcgispro-py3`、`Version=3.7.0.1901`、`RealVersion=3.7.0`、`InstallDate=2026/11/13`；`E:\ArcGISPro3.6` **不存在** |
| E3 | 3.7 环境 | Python 3.13.13 (Anaconda)、numpy 2.3.5、matplotlib 3.10.8、ipykernel 6.29.5、jupyter_client 8.6.0、zmq 27.1.0、typer 0.24.0、openai 2.30.0、httpx 0.28.1、PIL 12.2.0；项目 venv 为 3.12.6 |
| E4 | 发现链 | 6 个候选全部 `exists=False`（注册表 2 个 + fallback 4 个）→ `ArcGISDiscoveryError` |
| E5 | 桥接（显式解释器） | 成功；`stdout.strip()=="中文输出测试 ok"`、`data["cn"]=="中文数据"`、`data["mixed"]=="路径 D:\\数据\\test.gdb"` 逐字成立；`timeout_seconds=8` → `TimeoutExpired` @ 8.0 s |
| E6 | 持久内核 | `ArcpyKernel.start()` 27.0 s；`execute()` → `{'v': '3.7', 'cn': '中文结果'}`，`ok=True` |
| E7 | propy 回退 | `propy.bat` → `No environment named "E:\ArcGISPro3.6\bin\Python\envs\arcgispro-py3" exists`，落到 `C:\Program Files\python\python.exe` → 桥报 `ModuleNotFoundError: No module named 'arcpy'`（11.2 s，exit 1） |
| E8 | 核心 GP | `CreateFileGDB` / `CreateFeatureclass` / `InsertCursor` / `GetCount` / `Describe`(sr/wkid) / `Buffer` 全通过 |
| E9 | CalculateField | `PYTHON3`（显式与默认）→ `ERROR 000574: 未安装 Python。`；`ARCADE`（含 `CalculateFields`）→ OK |
| E10 | 制图链路 | `updateRenderer("UniqueValueRenderer")`→`fields`→3 类改色/标签→`saveACopy`(14 711 B)→`createLayout`→`createMapFrame`→`exportToJPEG`(25 860 B)→回读 `UniqueValueRenderer` + 3 items |
| E11 | matplotlib 中文 | `mpl_setup.configure()` → `available=True`、注册 `C:\Windows\Fonts\msyh.ttc`、Agg、中文 PNG 10 331 B（mpl 3.10.8） |
| E12 | 样式目录 | 3.7 枚举：`NORTH_ARROW 39`（与缓存逐字相同）、`SCALE_BAR 34`（相同）、`LEGEND 12`（缓存 6，名称全变：`图例 1..6` → `中心标题图例`/`图层名称和大号文本图例`/`手动列图例`/`透明背景图例`…） |
| E13 | Blank.aprx | 3.7 模板：`maps=1`、`layouts=0`（代码用 `createLayout`，不依赖模板布局） |
| E14 | 冷启动耗时 | arcpy 冷导入 11.91 s / 11.71 s（两次）；`ARCPY_PROBE_TIMEOUT=120`、`DEFAULT_TIMEOUT_SECONDS=300` 有足够余量 |

**推断（未做最终验证）**

- I1 `000574` 的根因是 `PythonCondaRoot` 指向已删的 3.6 → 修注册表即恢复（V1 一步可验）。
- I2 修注册表后 `propy.bat` 也会恢复（它走 `activate.bat` + `PythonCondaEnv`），但仍不建议把它当解释器（E7 的失败模式太安静）。
- I3 47 个配方在 3.7 上没有其他 arcpy API 删除项（依据是 E8/E10 两条抽样，未逐条跑）。
- I4 非默认安装位置（如 `E:\ArcGIS\Pro3.7`）仅靠 `ArcGISPro*` 通配覆盖不到，需要 `Pro*` 通配 + 安装目录校验兜底。

**复现命令（关键几条）**

```bash
# E2
reg query 'HKLM\SOFTWARE\ESRI\ArcGISPro' /v InstallDir
reg query 'HKLM\SOFTWARE\ESRI\ArcGISPro' /v PythonCondaRoot
# E3/E14
"E:/ArcGISPro3.7/bin/Python/envs/arcgispro-py3/python.exe" -c "import sys,numpy,matplotlib;print(sys.version,numpy.__version__,matplotlib.__version__)"
# E4
PYTHONPATH=src .venv/Scripts/python.exe -c "from gis_cli.arcpy_bridge import discover_arcgis_pro_python as d; print(d())"
# E5（把 PY37 换成 3.7 的解释器）
.venv/Scripts/python.exe - <<'PY'
import sys; sys.path.insert(0, "src")
from gis_cli import arcpy_bridge as b
PY37 = r"E:\ArcGISPro3.7\bin\Python\envs\arcgispro-py3\python.exe"
r = b.run_arcpy_code('print("中文输出测试 ok")\nset_result({"cn":"中文数据"})', python_executable=PY37, timeout_seconds=180)
print(r.status, r.stdout.strip() == "中文输出测试 ok", r.data)
PY
```

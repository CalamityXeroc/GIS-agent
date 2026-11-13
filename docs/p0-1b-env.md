# P0-1b 前置：GDAL 运行环境（`.venv-gdal`）

> 依据 DAIR-56。目的：给 P0-1b（把模板所需的 8–10 个配方移植成 GDAL 版）一个**能跑、能验证**的解释器 ——
> 没有它，移植出来的配方既不能执行也不能验证，等于盲写。
>
> 结论先说：GDAL 走 **wheel 自带的二进制**（pyogrio / rasterio / pyproj），
> 不需要 OSGeo4W，也不需要联网运行。冒烟脚本一次跑通，5 秒内出三项结果。

## 1. 环境位置与版本

| 项 | 值 |
|---|---|
| 解释器 | `E:\BST_lab\GIS_CLI\.venv-gdal\Scripts\python.exe` |
| Python | 3.12.6（与开发 venv `.venv` 同一版本，来自本机 `py -3.12`） |
| 创建命令 | `py -3.12 -m venv .venv-gdal` |
| 是否入库 | 否，`.gitignore` 已加 `.venv-gdal/`（`git check-ignore -v` 已验证生效） |
| 磁盘占用 | 421 MB（`du -sh .venv-gdal`） |

开发 venv `.venv` 里一个包都没动：`osgeo` / `geopandas` / `rasterio` / `pyogrio` / `shapely` /
`pyproj` / `matplotlib` / `pandas` 仍然全部 MISSING，已装包仍是 63 个 —— 新栈没有变成硬依赖。

## 2. 安装命令

```bat
py -3.12 -m venv .venv-gdal
.venv-gdal\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ^
    geopandas pyogrio shapely pyproj rasterio matplotlib
```

- **必须给镜像**：官方 PyPI 实测 23.8 MB 的 pyogrio 90 秒没下完（<0.3 MB/s），
  清华源 4–5 MB/s，整套约 2 分钟装完。外网是通的，只是慢，不算阻塞。
- 装完 `pip check` → `No broken requirements found.`，安装日志里没有降级/冲突告警。
- 等价写法：`pip install "gdal"` 那套不成立（见 §5）；`[project.optional-dependencies] gdal`
  已对齐到上面这组轮子，`pip install "gis-task-cli[gdal]"` 装出来的就是它。

## 3. 装上的包（含各自打包的 GDAL / PROJ）

| 包 | 版本 | 说明 |
|---|---|---|
| geopandas | 1.2.0 | 矢量入口，读写走 pyogrio 引擎 |
| pyogrio | 0.13.0 | 自带 GDAL **3.12.4**，GPKG / shapefile 读写 |
| rasterio | 1.5.2 | 自带 GDAL **3.12.2**，栅格读写 + 掩膜统计 |
| pyproj | 3.8.0 | 自带 PROJ **9.8.1**（EPSG 库） |
| shapely | 2.2.0 | 自带 GEOS |
| matplotlib | 3.11.2 | 出图；中文字体走系统「微软雅黑 / SimHei」 |
| pandas | 3.1.0 | geopandas 传递依赖 |
| numpy | 2.5.4 | 传递依赖 |

两份 GDAL 小版本不同（3.12.2 / 3.12.4）但互不干扰：各自把 libgdal 打进 `rasterio.libs/`
与 `pyogrio.libs/`，只在各自进程内加载。磁盘大头就是这两份：
pandas 75M、rasterio.libs 57M、pyogrio.libs 48M、rasterio 39M、numpy 34M、matplotlib 33M。

## 4. 冒烟：`scripts/gdal_smoke.py`

```bat
> .venv-gdal\Scripts\python.exe scripts\gdal_smoke.py
（已忽略外部 PROJ/GDAL 环境变量：PROJ_LIB、GDAL_DATA）
[1/3] 点落在面内计数：东区=16，中区=16，西区=16
[2/3] 按矢量分区栅格均值：东区=4.500，中区=14.500，西区=24.500
[3/3] PNG：E:\BST_lab\GIS_CLI\output\gdal_smoke\gdal_smoke.png（字体 Microsoft YaHei）
OK 耗时 1.3s
```

| # | 项 | 用的 API | 期望值 |
|---|---|---|---|
| 1 | 点落在面内计数 | `gpd.read_file(engine="pyogrio")` + `gpd.sjoin(predicate="within")` | 3 个面各 16 个（另 3 个区外点必须不算） |
| 2 | 按矢量分区栅格均值 | `rasterio.open` + `rasterio.mask.mask` + masked array | 4.500 / 14.500 / 24.500 |
| 3 | 带中文标题的 PNG | matplotlib（`Agg`）+ 系统字体回退 | 文件非空 |

- **确定性**：输入由脚本现场生成到 `output/gdal_smoke/`（`zones.gpkg` / `points.gpkg` / `dem.tif`），
  几何与栅格值都取整，期望值可手算，脚本用 `assert` 锁死 —— 只打印不断言的"冒烟"在栈坏掉时照样通过。
- **离线**：不读外部数据、不联网。**幂等**：重跑覆盖同名文件，连跑两次都是 `OK`。
- **耗时**：内部 1.3–1.8 s，连解释器启动 ~3–5 s，远低于 30 s 预算。
- 图面人工看过：中文标题/坐标轴/标注都正常，没有落到方框（字体 Microsoft YaHei）。

### 4.1 踩到的坑：全局 `PROJ_LIB` 指向 PostgreSQL 的 proj.db

本机装了 PostgreSQL/PostGIS，系统环境变量里留着：

```
PROJ_LIB=E:\PostgreSQL\share\contrib\postgis-3.6\proj
GDAL_DATA=E:\PostgreSQL\gdal-data
```

那套 `proj.db` 的 `DATABASE.LAYOUT.VERSION.MINOR = 2`，PROJ 9.8 要求 ≥ 6，于是任何碰坐标系的调用直接抛：

```
CRSError: The EPSG code is unknown. PROJ: proj_create_from_database:
  ...\postgis-3.6\proj\proj.db contains DATABASE.LAYOUT.VERSION.MINOR = 2
  whereas a number >= 6 is expected. It comes from another PROJ installation.
```

冒烟脚本在 import 前把**指向 venv 之外**的 `PROJ_LIB` / `PROJ_DATA` / `GDAL_DATA` 丢掉
（并在第一行说明丢了哪几个），PROJ 回落到 wheel 自带数据。以后任何新入口（尤其起子进程跑配方的
GDAL 执行器）都要做同样处理，否则会以"EPSG 4326 unknown"这种看不懂的形式炸出来。

## 5. 没装成 / 主动没装的

| 包 | 状态 | 原因 / 影响 |
|---|---|---|
| `GDAL`（osgeo 绑定） | **装不上** | PyPI 上只有源码包、没有 Windows wheel（`pip download --only-binary :all: GDAL` → `from versions: none`），要它得先本机有 libgdal / OSGeo4W —— 按需求明确不走这条路。**影响：`import osgeo` 永远失败。** |
| fiona | 未装 | geopandas 1.x 默认 pyogrio 引擎，用不到。真要装 `pip install fiona`（有 wheel，自带 GDAL）。 |
| rasterstats | 未装 | 分区统计用 `rasterio.mask` + masked array 自己算（冒烟里就是），少一个依赖。 |
| GDAL 命令行工具（`gdalinfo` / `gdal_translate` / `ogr2ogr`） | 没有 | wheel 只带库不带 exe。需要命令行工具就得上 OSGeo4W / conda，本次不做。 |
| 降级 | 无 | 没有包被降级或跳过，`pip check` 干净。 |

## 6. 对 P0-1b 的具体约束

1. 配方里**不要** `from osgeo import gdal / ogr`：本机没有该模块。矢量用 `pyogrio` / `geopandas`，
   栅格用 `rasterio`，几何用 `shapely`，坐标转换用 `pyproj`。
2. 后端探测别用 `find_spec("osgeo")`（永远是 `None`），按 `find_spec("pyogrio")` /
   `find_spec("rasterio")` 判 GDAL 栈是否可用。`src/gis_cli/recipes/backends.py` 与
   `recipes/store.py` 注释里那句"报 `No module named 'osgeo'`"是历史说法，实现时别当真。
3. `.venv-gdal` 里**没装项目本身**（保持干净的最小包集）。跑引擎侧代码用
   `PYTHONPATH=src .venv-gdal\Scripts\python.exe ...`，或 `pip install -e . --no-deps`。
4. 起子进程跑 GDAL 配方前先清 `PROJ_LIB` / `GDAL_DATA`（见 §4.1）。
5. 中文出图：`font.sans-serif` 里放「微软雅黑 / SimHei」+ `axes.unicode_minus=False`
   （`scripts/gdal_smoke.py` 里有现成写法）。

## 7. 复现验收

```bat
:: 1) 冒烟（离线，3–5 秒，三项结果 + PNG）
.venv-gdal\Scripts\python.exe scripts\gdal_smoke.py

:: 2) 开发 venv 不受影响
set PYTHONPATH=src
.venv\Scripts\python.exe -m pytest tests -q -m "not arcpy"
```

实测（2026-10-28，含本次改动）：**803 passed, 4 skipped, 30 deselected in 87.7s**，0 失败。
（DAIR-56 里写的 748 passed 是旧数，之后又落了 48/50/54 等一批测试；口径不变：全绿。）

`PYTHONPATH=src` 不能省：开发 venv 里 `gis-task-cli` 的 editable 安装指向改名前的目录
`E:\BST_lab\GIS CLI\src`（带空格），直接 `pytest` 会 `ModuleNotFoundError: No module named 'gis_cli'`。
这是既有环境问题，不在本次范围，未改动。

## 8. 本次改动清单

| 文件 | 改动 |
|---|---|
| `scripts/gdal_smoke.py` | 新增：离线冒烟（点面计数 / 分区栅格均值 / 中文 PNG），带断言 |
| `docs/p0-1b-env.md` | 新增：本文件 |
| `.gitignore` | 加 `.venv-gdal/` |
| `pyproject.toml` | `[gdal]` 从 `GDAL>=3.6`（Windows 装不上）换成实际能装的 6 个 wheel |
| `tests/test_recipe_backend_routing.py` | `test_gdal_dependency_is_optional` 判定口径改为"必须列自带 libgdal 的 wheel（pyogrio+rasterio）、不许列 GDAL 源码包" |

## 9. GDAL 代码执行器（GdalCodeRunner，DAIR-58）

`[gdal]` 配方要的不是"另一个配方模板"，而是**另一个执行器**：`runtime/code_runner.py` 的内核与
子进程都在 ArcGIS 的 Python 里，跑不了这个 venv 的 GDAL 栈。执行器在
`src/gis_cli/runner/gdal_runner.py`（类 `GdalCodeRunner`），接口与 `CodeRunner` 对齐：

```python
runner = GdalCodeRunner(workspace="workspace/任务名")   # 或传 python_executable=
result = runner.execute(code, timeout=600, workspace=None)  # -> ExecResult
result = runner.run(code, timeout=600)                      # 配方层调的是这个，同义
runner.backends  # {"gdal"} —— 配方层按这个声明路由
runner.backend   # 上一次实际路径：subprocess / unavailable / none
```

### 9.1 解释器怎么找（`GIS_GDAL_PYTHON` 优先）

| 顺序 | 位置 | 说明 |
|---|---|---|
| 1 | 环境变量 `GIS_GDAL_PYTHON` | 最高优先；指向任意 GDAL 解释器（conda / 别的 venv 都行） |
| 2 | `<仓库根>\.venv-gdal\Scripts\python.exe` | 缺省，DAIR-56 建的那个 |
| 3 | `<当前目录>\.venv-gdal\Scripts\python.exe` | 从别的目录起进程时的兜底 |
| — | 都没有 | `ExecResult(ok=False)`，`error.type = GdalBackendUnavailable`，消息是<br>`需 GDAL 后端：未找到 .venv-gdal，见 docs/p0-1b-env.md（GIS_GDAL_PYTHON=<配置值>）。…` |

**不会抛 `FileNotFoundError`**：缺环境是可预期的降级，配方层只看到一条 `ExecResult`，
照常走"需 GDAL 后端"的提示，而不是让整个 run 崩在 `subprocess.run` 里。

### 9.2 协议与 ArcPy 子进程路径是同一份

driver 脚本直接复用 `arcpy_bridge._build_runner_script()`（`payload` 里 `require_arcpy=False`），
所以 `set_result(value)` 的返回体、stdout/stderr 捕获、异常回传
（`{"type","message","traceback"}`）、结果文件缺失时的 `RunnerError`，与 `CodeRunner`
的子进程路径逐项一致 —— 没有第二套协议要维护。

与 `CodeRunner` 有意的三点差别：

1. **cwd 就是 workspace**：GDAL 侧没有 `arcpy.env.workspace`，代码里的
   `"input/x.shp"` 只能靠进程工作目录落到工作区（测试 `test_workspace_is_process_cwd` 锁死）。
2. **清 `PROJ_LIB` / `PROJ_DATA` / `GDAL_DATA`**：指向 venv 之外的（本机 PostgreSQL 那套
   proj.db layout=2）一律丢掉，理由见 §4.1；venv 自己的路径保留。
3. **超时杀进程树**：`taskkill /PID <pid> /T /F`（POSIX 用 `killpg`），再 `wait()` 确认主进程
   退出。只杀 direct child 会留下攥着 `.tif` / `.gpkg` 句柄的孙进程（GDAL 代码常用
   `multiprocessing` / 命令行子进程），下一次写入就是 `PermissionError`。
   测试真起一个孙进程验证整棵树都死了。

运行时**不 import** 任何第三方 GIS 包、也不 import arcpy：GDAL 栈仍是可选依赖，
开发 venv 与 47 个 arcpy 配方不受影响。

### 9.3 定位与验证

```bat
:: 协议五条路径 + 路由 + 真 GDAL 集成（.venv-gdal 不在则 skip）
set PYTHONPATH=src
.venv\Scripts\python.exe -m pytest tests\test_gdal_runner.py -q
:: 基线不许掉
.venv\Scripts\python.exe -m pytest tests -q -m "not arcpy"
```

实测（2026-10-28）：`tests/test_gdal_runner.py` **14 passed**（含一条真 `.venv-gdal` 的
`import geopandas` 集成，跑通 1.2.0）；全量 `-m "not arcpy"` **817 passed, 4 skipped, 30 deselected**
（DAIR-56 时的 803 + 本次 14，基线未掉）。

**还没接进引擎**：引擎仍只建 `CodeRunner`（`engine/runner.py:251`）。本单只交付执行器本身，
等 P0-1b 移植配方那批再把 `GdalCodeRunner` 挂进 `RecipeLibrary` 的后端路由。

## 10. 本次改动清单（DAIR-58）

| 文件 | 改动 |
|---|---|
| `src/gis_cli/runner/gdal_runner.py` | 新增：`GdalCodeRunner`（子进程跑 `.venv-gdal`，协议同 CodeRunner，超时杀进程树） |
| `tests/test_gdal_runner.py` | 新增：不依赖真 GDAL 的 13 条协议/环境/路由用例 + 1 条真 GDAL 集成（缺环境 skip） |
| `docs/p0-1b-env.md` | 本文件 §9/§10：解释器路径（`GIS_GDAL_PYTHON` 优先）、协议、超时与环境清理 |

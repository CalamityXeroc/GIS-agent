# GIS Agent 快速安装指南

## 前置要求

- **ArcGIS Pro 3.6 / 3.7**（可选）：跑真实 GIS 处理、出图（`.gdb` / `.aprx` / JPG）才需要；
  只做数据浏览、对话、规划不需要。
- **Python ≥ 3.10**：`install.py` 用它写 shim（不需要装进 Pro 环境，也不需要管理员权限）。

---

## 🚀 一键安装（推荐）

### Windows / Git Bash / PowerShell

```bash
python install.py
```

1. 脚本会：
   - ✅ 用 `arcpy_bridge` 的发现链找 ArcGIS Pro 的 Python
   - ✅ 在临时 venv 里验一次 `pip install -e .` 是否可行（装不上不影响）
   - ✅ 把 `giscli.cmd` / `giscli.ps1` / `giscli` 写进 `%LOCALAPPDATA%\Programs\giscli`
   - ✅ **只**向用户级 PATH（`HKCU\Environment`）追加这一个目录，幂等
   - ✅ 跑一次 `giscli doctor` 并打印结果
2. **后续使用**：任何目录敲 `giscli`（不需要管理员权限，不装进 Pro 环境）

---

## 📋 手动安装（备选方案）

如果一键安装失败，请按以下步骤手动安装。

### 步骤 1：确认解释器

```bash
python -V          # 需要 ≥ 3.10
```

跑真实 GIS 处理需要一个能 `import arcpy` 的 Python（ArcGIS Pro 自带的那份，通常位于
`C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe`）。
`giscli` 默认会自己找它；装在别处就设 `ARCGIS_PRO_PYTHON` 指过去。

### 步骤 2：安装依赖

在仓库目录里执行（系统 Python 即可，不用装进 Pro 环境）：

```bash
python -m pip install -e .
```

`prompt-toolkit` 已包含在依赖里（输入框 / 斜杠命令菜单 / 历史搜索靠它）：
缺了它 CLI 不会崩，但输入框会降级成只有一个 `›` 的纯文本输入，并在启动时打一行警告。
若你的环境装不上它（离线等），可以显式设置 `GIS_AGENT_ALLOW_PLAIN_INPUT=1` 静默该项警告。

`pyproj` 也是硬依赖：Web 控制台的数据预览靠它把非 EPSG:4326 的矢量重投影。
缺了它界面照开、4326 数据照看，但 EPSG:32610 / 4548 这类图层只会显示一条
「坐标系未归一，画不上几何」的告警条。`scripts/check_install.py` 会在启动时把缺包与修复命令打出来。

### 步骤 3：启动

```bash
# 交互式 Agent（在哪个目录跑，哪个目录就是工作区根）
python -m gis_cli.agent.cli chat

# 或 Web 控制台
python -m gis_cli.agent.cli web --port 8760
```

---

## 🧭 用 `giscli` 在任何目录启动（推荐）

一条命令装好，之后任何目录、cmd / PowerShell / **Git Bash** 里都能敲 `giscli`：

```bash
# 一次性安装：写 shim → 只追加用户级 PATH → 跑一次自检
python install.py

# 之后（新开一个终端）在任何目录都能用
giscli --help
giscli doctor                        # Pro python / arcpy / 依赖 / pyproj / web / src / 工作空间
giscli init D:\giswork\demo          # 建好 demo\workspace\{input,output,temp}
cd /d D:\giswork\demo && giscli chat  # 数据丢进 workspace\input

giscli --new-window chat              # 新开一个终端窗口跑同一件事

# 撤销 PATH 改动（只删这一条，别的条目原样保留）
# --scripts-dir 不能省：缺省值是 scripts\ 自己，删的就不是这次装进去的那一条
# 目录用 install.py 结尾打印的字面路径（cmd / PowerShell / Git Bash 通用）；
# 下面这条是 cmd 写法，PowerShell 换成 "$env:LOCALAPPDATA\Programs\giscli"
python scripts/install_shim.py --remove --scripts-dir "%LOCALAPPDATA%\Programs\giscli"
```

装的是什么：`%LOCALAPPDATA%\Programs\giscli\` 下的三份 shim（`giscli.cmd` /
`giscli.ps1` / `giscli`），它们只做一件事——用 `arcpy_bridge` 那**同一套**发现链
找 Pro 的 python，把仓库 `src` 放进 `PYTHONPATH` 后跑 `python -m gis_cli.agent.cli`。
**不装进 Pro 环境、不生成 exe、不动系统 PATH。** 仓库放在中文目录也不影响。

> 旧机器上如果 PATH 里还留着仓库 `scripts\` 目录（老 shim 的装法）且排在前面，
> `giscli` 会先解析到那一份——三份 shim 内容等价，行为一样；不想要旧条目就单独删掉它。

> **已知限制（不修）**：Pro 环境里 pip 生成的
> `...\arcgispro-py3\Scripts\gis-agent.exe` 是坏的（任何参数 rc=1、零输出），
> Pro 环境的 pip 本身也会报 `BuildDependencyInstallError`。两者都不动——`arcpy`
> 与 Pro 安装/许可是绑死的，硬修只会把环境搞得更乱；`giscli` 绕开它们。
> `gis-agent` 这个名字保留为兼容别名。
> **PATH 只改用户级**（`HKCU\Environment`），系统级 PATH 一律不碰；**不需要管理员权限**。

---

## 🔧 常见问题

### Q1: 写用户级 PATH 失败

**不需要管理员权限**：`install.py` 只写 `HKCU\Environment`，不碰系统级 PATH。
如果这一步报错（组策略锁了注册表等），shim 其实已经写好，手动把这一个目录加进用户 PATH 即可：

```
%LOCALAPPDATA%\Programs\giscli
```

### Q2: 找不到 ArcGIS Pro Python

**解决**：指定解释器后重跑

```bash
# cmd
set ARCGIS_PRO_PYTHON=C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe
# PowerShell
$env:ARCGIS_PRO_PYTHON="C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe"
# Git Bash
export ARCGIS_PRO_PYTHON="/c/Program Files/ArcGIS/Pro/bin/Python/envs/arcgispro-py3/python.exe"

python install.py
```

不知道路径就在 ArcGIS Pro 的 Python 窗口里跑 `import sys; print(sys.executable)`。

### Q3: pip install 超时

**解决**：使用国内镜像

```bash
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q4: 没有 ArcGIS Pro，能用吗？

**可以部分使用**：
- ✅ 数据浏览 / 属性表 / Web GIS 工作台（矢量、栅格都能看）
- ✅ 对话、规划、扫描数据
- ❌ 真正的 GIS 处理与出图（需要 arcpy）

用系统 Python 安装即可：

```bash
pip install -e .
```

---

## 📂 目录结构

```
<仓库>/
├── install.py                 # 一键安装：写 shim + 追加用户级 PATH + 自检
├── src/gis_cli/               # 本体（引擎 / 运行时 / 制图 / 数据目录 / Web 服务 / CLI）
├── web/                       # Web 控制台前端源码（Vite + React + MapLibre）
├── config/                    # 配置样例；复制 *.example.json 成实体文件后填 key
├── templates/                 # 运行时模板库
└── scripts/                   # giscli 三份 shim / install_shim.py / check_install.py
```

运行时在 `<工作区>/.gis_agent/` 下写 trace、日志与 token；`workspace/{input,output,temp}`
骨架随仓库自带（说明见 `workspace/README.md`），自己另建项目目录用 `giscli init <目录>`。

---

## 🎯 安装验证

```bash
# 环境自检：Pro python / arcpy / 依赖 / pyproj / web / src / 工作空间 / token
# 全过时最后一行是「环境正常」，退出码 0
giscli doctor

# Web 控制台（端口可自选；启动输出里的地址带 token）
giscli web --port 8760
```

看到下面这段说明 CLI 通了：

```
────────────────────── Welcome ─────────────────────────
│ GIS Agent - AI-powered GIS workflow assistant        │
│ Type your GIS task or question. Type 'exit' to quit. │
╰──────────────────────────────────────────────────────╯

You:
```

---

## 💡 使用建议

1. **首次使用**：
   - 运行 `python install.py`
   - 新开一个终端，跑 `giscli doctor` 确认环境
   - 把 GIS 数据放进工作区的 `workspace/input/`

2. **日常使用**：
   - `giscli chat`：交互式 Agent
   - `giscli web`：Web 控制台（数据资产 GIS 工作台 + trace 查看）
   - `giscli loop "任务描述"`：跑单个任务，跑完即退出

3. **处理数据**：
   - 将 GIS 数据放入 `workspace/input/`
   - 启动 Agent，说「扫描我的数据」

---

## 🆘 获取帮助

如果遇到问题：

1. 先跑 `giscli doctor`，按它给的「建议」列处理
2. 参考本指南的「常见问题」
3. 查看项目 README.md
4. 提交 Issue

---

## 🔄 更新

```bash
git pull
python install.py      # 幂等：只更新 shim 内容，不会重复追加 PATH
```

---

**祝使用愉快！** 🎉

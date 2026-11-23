# GIS Agent 快速安装指南

## 🚀 一键安装（推荐）

### Windows 用户

1. **双击运行**：`安装.bat`
2. 脚本会自动：
   - ✅ 查找 ArcGIS Pro Python 环境
   - ✅ 安装所有依赖
   - ✅ 配置启动脚本
   - ✅ 询问是否立即启动

3. **后续使用**：双击 `快速启动.bat`

### 所有平台

运行 Python 安装脚本：
```bash
python install.py
```

---

## 📋 手动安装（备选方案）

如果一键安装失败，请按以下步骤手动安装：

### 步骤 1：找到 ArcGIS Pro Python

ArcGIS Pro 的 Python 通常位于：
```
C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe
```

或使用 ArcGIS Pro 自带的 Python 命令提示符。

### 步骤 2：安装依赖

打开命令行，运行：
```bash
# 切换到项目目录
cd "E:\BST_lab\GIS CLI"

# 使用 ArcGIS Pro Python 安装
"C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe" -m pip install -e .
```

`prompt-toolkit` 已包含在依赖里（输入框 / 斜杠命令菜单 / 历史搜索靠它）：
缺了它 CLI 不会崩，但输入框会降级成只有一个 `›` 的纯文本输入，并在启动时打一行警告。
若你的环境装不上它（离线等），可以显式设置 `GIS_AGENT_ALLOW_PLAIN_INPUT=1` 静默该项警告。

`pyproj` 也是硬依赖：Web 控制台的数据预览靠它把非 EPSG:4326 的矢量重投影。
缺了它界面照开、4326 数据照看，但 EPSG:32610 / 4548 这类图层只会显示一条
「坐标系未归一，画不上几何」的告警条。`一键启动.bat` 会自动补装，
`scripts/check_install.py` 也会在启动时把缺包与修复命令打出来。

### 步骤 3：启动

```bash
# 启动 Agent
"C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe" -m gis_cli.agent.cli chat
```

---

## 🧭 用 `giscli` 在任何目录启动（推荐）

仓库自带入口 `scripts\giscli.cmd`：它用 `scripts\find_pro_python.cmd` 找到 Pro 自带的
Python（`ARCGIS_PRO_PYTHON` → `E:\ArcGISPro*` → `D:\ArcGISPro*` →
`C:\Program Files\ArcGIS\Pro*`），把仓库 `src` 放进 `PYTHONPATH` 后执行
`python -m gis_cli.agent.cli`。**不需要 pip install，也不生成 exe**。

```bash
# 一次性：把 scripts 目录加入用户级 PATH（幂等，重复跑不会重复追加）
python scripts\install_shim.py

# 之后（新开一个终端）在任何目录都能用
giscli --help
giscli init D:\giswork\demo          # 建好 demo\workspace\{input,output,temp}
cd /d D:\giswork\demo && giscli chat  # 数据丢进 workspace\input

giscli --new-window chat              # 新开一个终端窗口跑同一件事

# 撤销 PATH 改动（只删这一条，别的条目原样保留）
python scripts\install_shim.py --remove
```

> **已知限制（不修）**：Pro 环境里 pip 生成的
> `...\arcgispro-py3\Scripts\gis-agent.exe` 是坏的（任何参数 rc=1、零输出），
> Pro 环境的 pip 本身也会报 `BuildDependencyInstallError`。两者都不动——`arcpy`
> 与 Pro 安装/许可是绑死的，硬修只会把环境搞得更乱；`giscli` 绕开它们。
> `gis-agent` 这个名字保持原有行为不变。
> **PATH 只改用户级**（`HKCU\Environment`），系统级 PATH 一律不碰。

---

## 🔧 常见问题

### Q1: 安装失败，提示权限不足

**解决**：以管理员身份运行 `安装.bat`
- 右键 → "以管理员身份运行"

### Q2: 找不到 ArcGIS Pro Python

**解决**：手动指定路径
1. 打开 ArcGIS Pro
2. 在 Python 窗口中运行：`import sys; print(sys.executable)`
3. 记录输出的路径
4. 运行 `install.py` 时选择"手动指定路径"

### Q3: pip install 超时

**解决**：使用国内镜像
```bash
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q4: 没有 ArcGIS Pro，能用吗？

**可以部分使用**：
- ✅ 对话、规划、扫描数据
- ❌ 真正的 GIS 处理（需要 arcpy）

使用系统 Python 安装：
```bash
pip install -e .
```

---

## 📂 安装后的文件结构

```
GIS CLI/
├── 安装.bat              # 一键安装脚本
├── 快速启动.bat          # 启动脚本（安装后生成）
├── install.py           # Python 安装脚本
├── .python_path         # Python 路径配置（安装后生成）
├── workspace/           # 工作区
│   ├── input/          # 放置输入数据
│   ├── output/         # 输出结果
│   └── temp/           # 临时文件
└── src/gis_cli/        # 源代码
```

---

## 🎯 安装验证

安装完成后，测试是否正常：

```bash
# 使用快速启动
快速启动.bat

# 或直接命令行
python -m gis_cli.agent.cli chat
```

看到以下界面说明安装成功：
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
   - 运行 `安装.bat`
   - 等待安装完成
   - 选择立即启动

2. **日常使用**：
   - 双击 `快速启动.bat`
   - 或运行 `python -m gis_cli.agent.cli chat`

3. **处理数据**：
   - 将 GIS 数据放入 `workspace/input/`
   - 启动 Agent
   - 说"扫描我的数据"

---

## 🆘 获取帮助

如果遇到问题：

1. 查看错误信息
2. 参考本指南的"常见问题"
3. 查看项目 README.md
4. 提交 Issue

---

## 🔄 更新

更新到新版本：
```bash
git pull
python install.py
```

或重新运行 `安装.bat`

---

**祝使用愉快！** 🎉

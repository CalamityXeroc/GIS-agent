# GIS Agent 安装与使用（纯净发布版）

这个仓库是可发布的纯净版，默认已采用 `workspace/` 作为工作区。

## 1. 安装

在仓库根目录执行：

```bash
pip install -e .
```

如果你用 ArcGIS Pro Python，请改用对应解释器：

```bash
"E:\ArcGISPro3.6\bin\Python\envs\arcgispro-py3\python.exe" -m pip install -e .
```

## 2. 初始化配置

首次运行前，准备 LLM 配置：

```bash
cp config/llm_config.example.json config/llm_config.json
```

然后编辑 `config/llm_config.json` 填写真实 API Key。

## 3. 启动（推荐）

双击：

- `一键启动.bat`

该脚本会自动：

1. 检测 Python 运行时（优先 ArcGIS Pro Python，失败则回退系统 Python）
2. 同步本地 editable 安装
3. 自动创建工作区目录：
   - `workspace/input`
   - `workspace/output`
   - `workspace/temp`
   - `workspace/skills`
4. 以 `--workspace .\workspace` 启动 Agent

## 4. 输入输出数据放哪里

- 输入数据：放到 `workspace/input/`
- 输出结果：写到 `workspace/output/`
- 临时文件：`workspace/temp/`

## 5. 命令行启动（可选）

```bash
gis-agent chat --workspace ./workspace
```

或（不走 PATH）：

```bash
python -m gis_cli.agent.cli chat --workspace ./workspace
```

## 6. 常见问题

1. 启动后没看到数据：确认数据已放到 `workspace/input/`
2. 没有 ArcGIS Pro：仍可运行对话和部分流程，ArcPy 相关步骤会降级
3. 模型不可用：检查 `config/llm_config.json` 中 `api_key/provider/model/api_base`

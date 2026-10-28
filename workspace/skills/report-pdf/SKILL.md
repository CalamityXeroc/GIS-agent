---
name: report-pdf
description: 把 GIS 成果包（单文件 HTML 报告）或任意本地 HTML 转成 PDF。用于「出一份 PDF 报告 / 把成果包转成 PDF / 交付 PDF 版成果」这类需求。不需要装任何 Python PDF 库——用系统自带的 Edge/Chrome 无头打印，中文与地图都能正确渲染。
---

# 成果包 / HTML → PDF

## 什么时候用

- 用户要「PDF 版报告」「把成果包发成 PDF」「成果包.html 转 PDF」
- 成果已经跑完、`output/成果包.html` 已生成（`build_deliverables` 在验收通过后自动写它）
- 需要一份**能直接发微信/邮件、断网也能看**的静态文件

## 为什么不用 Python 库

`matplotlib` / `reportlab` / `pypdf` 在宿主 venv 里**都没有**（matplotlib 只装在 ArcGIS Pro 的 python 里）。
而系统自带的 Edge（Windows）或 Chrome 用无头模式 `--print-to-pdf`：

- 零新依赖、离线可用
- **中文天然正常**（用系统字体，不像 matplotlib 要手配字体，否则方框）
- 与 HTML 版**视觉一致**（同一份 CSS、同一批内联图）

## 用法

```bash
python workspace/skills/report-pdf/scripts/html_to_pdf.py \
    --html output/成果包.html \
    --pdf  output/成果包.pdf
```

输出示例：

```
[report-pdf] 浏览器: E:\...\msedge.exe
[report-pdf] 输出: output/成果包.pdf  (1.21 MB, 14 页)
```

脚本会自己找 Edge / Chrome，校验产出的 PDF 头与页数，失败时给出明确原因。

## 关键坑（都实测过）

1. **必须用绝对路径**：`--print-to-pdf` 与 `file:///` URL 都不吃相对路径，脚本内部会 `Path.resolve()`。中文路径要转成 `file:///` 形式（脚本已处理）。
2. **必须 `--headless=new`**：老 `--headless` 在部分 Edge 版本上不触发打印。
3. **等渲染完再退出**：内联图片多时浏览器需要时间解码，脚本给 `--virtual-time-budget=15000` 兜底；太小会打出白页。
4. **不要加 `--no-pdf-header-footer` 之外的分页参数**：默认分页已经把长图切成多页；硬塞 `--print-to-pdf-page-size` 会把地图压变形。
5. **PDF 里没有目录/书签**：需要书签就用 `pypdf`（本机没装），或者让 HTML 自带目录页。
6. **不要用它转交互式页面**：`position: fixed`、懒加载图片、需要 JS 请求数据的页面会打不全。成果包是纯静态内联的，正好没问题。

## 验收

```bash
# 1. 头必须是 %PDF-1.x
head -c 8 output/成果包.pdf

# 2. 页数与体积合理（成果包通常 6~20 页、1~5 MB）
python -c "import re;raw=open(r'output/成果包.pdf','rb').read();print(len(raw)/1048576,'MB', re.findall(rb'/Count\s+(\d+)',raw)[:3])"

# 3. 中文不是方框：PDF 里应能看到嵌入字体
python -c "raw=open(r'output/成果包.pdf','rb').read();print('fonts embedded:', b'FontFile' in raw)"
```

## 兜底方案（Edge/Chrome 都找不到时）

用 ArcGIS Pro 自带的 python 走 matplotlib 拼页（`src/gis_cli/report/pdf_pack.py` 已有实现）：

```bash
"E:\ArcGISPro3.6\bin\Python\envs\arcgispro-py3\python.exe" -c "
import sys; sys.path.insert(0, r'<仓库根>\src')
from pathlib import Path
from gis_cli.report.pdf_pack import write_pdf
# content 需要先由 narrative.build_content 构造，见 report/__init__.py
"
```

注意：这条路**中文字体要先配**（`matplotlib.rcParams['font.sans-serif']`），否则图例中文变方框；而且只能拼图，不能还原 HTML 的排版。**优先用浏览器那条**。

## 与成果包的关系

`build_deliverables` 只在 HTML 那步是可靠的；它内部的 `want_pdf` 在宿主 venv 里恒为 `None`（matplotlib 缺失，静默降级）。
所以：**PDF 走这个 skill，不要指望 `build_deliverables` 顺手出 PDF**。需要 PDF 时在任务里明确说「出 PDF 版」，或跑完补一条 `html_to_pdf.py`。

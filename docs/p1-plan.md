# P1 实施方案（成果"可信、可发布"）

> 依据 `GIS-Agent-总开发路线-2026Q4.md` §6。P0 四项已收口（doctor / ingest / 计划卡 / 成果包），
> 本文件把 P1 五项拆成**可独立交付、可独立验收**的波次，并给出每波的文件级改动与验收命令。
> 一次只开一波：开发 → 审查 → 基准，通过才进下一波。

## 0. 与总计划的两处偏差（先看）

| 偏差 | 原因 | 处理 |
|---|---|---|
| 总计划 P1-1 模板库依赖 **P0-1b**（6 个模板所需 8–10 个配方的 GDAL 版），P0-1b 尚未开工 | 模板若只有 arcpy 版，"没装 ArcGIS 的人"这条 P0 目标在 P1 里断链 | P1-1 排到最后；第一版模板 manifest 允许 `backend: [arcpy]`，GDAL 执行随 P0-1b 补齐，manifest 不改 |
| 总计划两条验收需要外部输入：P1-3「3 名资深制图员人工评分 ρ≥0.7」、P1-2「20 个已知问题样本」 | 人工评分与真实图件样本只能由用户提供 | 开发阶段先做**可机器验证的替代口径**（各项验收里写明），人工校准/真实样本回放等用户给料后补，不阻塞开发 |

## 1. 波次与依赖

| 波次 | 内容 | 人日 | 依赖 | 主交付物 |
|---|---|---|---|---|
| **W1** | P1-4 成果溯源卡 | 6 | 无（纯报告层，不碰 arcpy） | `report/provenance.py` + 成果包新增溯源卡节 + `output/溯源卡.html` |
| **W2** | P1-3 成果质量分 | 15 | W1（验证段落复用）+ 现有 `cartography/qc.py` | `cartography/score.py` + `check_map_project` 返回分数 |
| **W3** | P1-2 地图合规闸门 | 12 | W1（边界数据来源登记落点）+ W2（分级输出） | `gis_cli/compliance/` + `create_map`/`finish` 闸门 |
| **W4** | P1-1 民生模板库（首批 2 个） | 30 | **P0-1b** | `templates/*/manifest.yaml` + `list_templates`/`run_template` |
| **W5** | P1-5 分析正确性双复核 | 12 | 现有 `engine/subagent.py` | 复核 subagent + 守恒/拓扑/量纲断言库 |

**顺序理由**：W1 最小、零 arcpy 依赖、且正好是 W3「边界数据来源登记」的落地位置（先有卡片，闸门才有地方写）；
W2 复用已成熟的 `qc.py` 判定，是"敢不敢发"的闸门本体；W3 依赖前两者的输出格式；W4 被 P0-1b 阻塞；
W5 token 成本翻倍，最后做。

### 1.1 插入 P0-1（2026-10-28 决定）

P1-1 依赖 P0-1b，而 P0-1b 依赖 P0-1a 的契约，所以把 P0-1 拆成三段插进流水线（由用户拍板，优先于 W3/W4）：

| 段 | 内容 | 状态 |
|---|---|---|
| **P0-1a** | 配方 `backend` 声明 + `recipes/store.py:run()` 按后端路由 + 缺后端明确报错 | ✅ 已交付 `860f14d4`（审查 DAIR-51 通过） |
| **P0-1b** | 只移植模板所需的 8–10 个配方到 GDAL，**按配方分批，一批一单** | 待开 |
| **P0-1c** | 双后端一致性回归（同一输入逐指标比对，差异 <1%） | 待开（依赖 1b 的前两批） |

插在 W2 之前还是并行，取决于写手是否空闲——**同一时刻只有一个写手**这条压过排期。

### 1.2 实际进展（滚动更新）

| 波次 | 提交 | 审查 | 基准 |
|---|---|---|---|
| W1 溯源卡 | `6d483dec` | DAIR-45 通过 | DAIR-46 真跑 1172.9s，四段真数据、0 外链、卡片 4.5KB |
| W1 收口 | `9e18132a` | DAIR-53 通过 | 不单独跑（无行为变更） |
| W1 缺陷修复 | `449b0a39` | DAIR-53 通过（新用例在 `860f14d4` 上 3 failed 真红） | 仅退化场景，无需真跑 |
| P0-1a | `860f14d4` | DAIR-51 通过 | 契约层，无数据跑 |
| W2 质量分 | `e5e73b72` | DAIR-55 通过（权重改 0 变异验证真红） | DAIR-57 真跑 673.6s：真图 100/100，造缺陷对照确实扣 8 分；**同时抽出 B1–B6 六条缺陷**（见下） |
| P0-1b 环境 | `e170adb1` | DAIR-59 通过 | `.venv-gdal` Python 3.12.6，geopandas 1.2.0 / rasterio 1.5.2 |
| P0-1b 执行器 | `7e75d632` | DAIR-59 通过 | 接口对齐 CodeRunner，超时杀整棵进程树 |
| W3 合规闸门 | `7cdaff77` | 待审（审查单见父单 DAIR-43） | 20 个合成样本 0 漏报 / 误报 0% / 单图 <1s；非 arcpy 全量 1251 passed / 4 skipped |

测试基线：`6d483dec` 时 725 passed / 4 skipped → `449b0a39` 时 748 → `e5e73b72` 后 817 → `7cdaff77` 后 1251 passed / 4 skipped。

### 1.3 DAIR-57 基准抽出的缺陷（2026-10-28）

| 编号 | 严重度 | 内容 | 处置 |
|---|---|---|---|
| B1 | 高 | `recipes/store.py:1413` `_aprx_map_check` 没有 `code_runner` 回退，CLI 无 arcpy 时**合格工程也拿 `total=0.0`**（`map_layout_ok` 早在 `store.py:730` 修过同一个坑） | 立即修 |
| B2 | 高 | 被测 agent 在 `src/` 下自建 `arcpy/` 桥接包，**越出 benchmark 工作区写宿主源码** | 隔离与防护另开单 |
| B3 | 中 | `qc.py:226` 读不到图名字号 → `title_too_small`（8 分）恒真 | 收口 |
| B4 | 中 | `quality` 不进 trace，基准只能靠模型拄数字 | 随 B1 一起修 |
| B5 | 低 | `design.py:926` 未知 override 点路径静默接受，造不出对照组 | 收口 |
| B6 | 低 | `qc.py:354` `legend_covers_data` 通过时 detail 文案与判定方向相反 | 收口 |

## 2. W1：成果溯源卡（P1-4）

**目标**：数据 → 步骤 → 参数 → 验证结果，一页可审计，随成果包一起交付。

### 改动面

- 新模块 `src/gis_cli/report/provenance.py`
  - `@dataclass ProvenanceCard`：`run_id / generated_at / goal / data[] / steps[] / params[] / verifications[] / artifacts[] / missing[]`
  - `build_card(*, state, verdict, catalog, trace_path, recipes, workspace) -> ProvenanceCard`
    纯函数：只读入参，不碰 arcpy、不联网、不写盘。
  - `render_card_html(card) -> str`：返回**内联样式、无外链**的 `<section>`，既能嵌进成果包，也能独立成页。
- 数据一律复用现有采集，不新造：
  - 数据 → `narrative.source_rows()`（catalog 的来源/许可/采集时间）
  - 步骤 + 参数 → `narrative.recipes_from_trace()`（trace JSONL 里的配方调用与入参）
  - 验证 → `verdict`（断言明细）
  - 产物 → `state.artifacts`
- 集成 `src/gis_cli/report/__init__.py:build_deliverables`
  - 写 `output/溯源卡.html`
  - 主成果包内插一节 `<h2>溯源卡</h2>`
  - 任何一步失败只进 `notes`，**不阻断交付**（沿用现有约定）

### 验收（机器可验）

1. `tests/test_report_provenance.py`：合成 trace JSONL + 假 catalog + verdict → 四段都非空，参数摘要含实际调用参数值
2. 退化输入（无 trace / trace 半截 / catalog 抛错）→ 卡片仍生成，缺的段落写"未记录"，不抛异常
3. 卡片 HTML 不含 `http://` / `https://` 外链；单卡 < 200KB
4. `pytest tests -q -m "not arcpy"` 全绿，基线 718 passed / 4 skipped 不掉
5. 基准阶段真跑一次：`output/成果包.html` 含溯源卡节、`output/溯源卡.html` 存在且可离线打开

### 不做

数字签名 / 哈希链（P2 再说）；不改 narrative 现有段落；不做 UI 页面。

## 3. W2：成果质量分（P1-3）

- 复用 `cartography/qc.py` 的判定结果，产出 **0–100 分 + 三维度**：规范性（图名/图例/比例尺/指北针/字体/分级边界）、可读性（图面占用/标注密度/对比度）、正确性（断言覆盖度/守恒/单位）。
- 新模块 `cartography/score.py`：`score_layout(qc, *, assertions=None) -> QualityScore`，权重表独立成常量（可读、可测、可改）。
- 接口：`check_map_project` / `create_map` 返回值增加 `quality: {total, normativity, readability, correctness, deductions[]}`；`finish` 前置沿用现有 `map_layout` 闸门，不新增硬失败。
- 验收：每条扣分项各有合成用例（≥8 条）；同一缺陷重复出现分数单调不增；`pytest -m "not arcpy"` 全绿；人工评分相关性（ρ≥0.7）等用户给评分表后由基准侧跑。

## 4. W3：地图合规闸门（P1-2）

- 新包 `src/gis_cli/compliance/`：YAML 规则库 + 检测器 + 分级输出（阻断 / 警告 / 提示）。
- 四条规则：① 国界与敏感要素；② 边界数据来源未登记；③ 未送审水印缺失；④ 审图号缺失提示。
- 落点：`create_map` 出图后自检、`finish` 前汇总进 `verify_outputs`。
- **只做提示 + 阻断，不做合规判定**；责任边界写进卡片与 UI 文案（与溯源卡同一节呈现）。
- 验收：20 个**合成**样本（构造已知问题，不依赖真实图件）0 漏报、误报 ≤10%、单图 ≤30 s；真实图件回放等用户给样本。

## 5. W4：民生模板库（P1-1）

- 先做模板骨架：`templates/<id>/manifest.yaml`（数据清单 + 配方链 + 断言集 + 报告模板）+ 工具 `list_templates` / `run_template`。schema 与参数绑定语法见 `docs/templates.md`。
- 首批 2 个（已有 PM 链实证）：社区适老化评估、绿地公平性；其余 4 个在 P0-1b 之后排。
- 验收：模板首次跑通率 ≥90%（10 次冷启动），断言集全绿；无 GDAL 后端时明确报"需 ArcGIS"，不静默失败。

## 6. W5：分析正确性双复核（P1-5）

- 复用 `engine/subagent.py`：复核 subagent **只看数据 + 结论**，不看主轨迹；配守恒/拓扑/量纲断言库。
- 只在"交付级"任务开启（token 翻倍），开关进 `requirements`。
- 验收：25 个既有任务人为注入 20 个错误，抓出 ≥50%，误杀 ≤10%。

## 7. 每波的闸门（不跳步）

1. **开发**：一个 agent 改源码，产出 commit + 自测结论。
2. **代码审查**：审 commit 区间（只读）+ `pytest tests -q -m "not arcpy"`，通过/打回。
3. **基准**：真实数据/长任务回归，出数字。
4. 通过后父 issue 才推进下一波；打回回开发侧，不进基准。

**并发约束**：同一时刻只允许一个 agent 改 `E:\BST_lab\GIS_CLI` 源码。

## 8. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| P1-1 被 P0-1b 阻塞 | 模板只能承诺 arcpy 机器可用 | manifest 预留 `backend` 字段；先交付骨架与断言集，GDAL 执行后补 |
| 质量分权重拍脑袋 | 分数不可解释、校准不过 | 扣分表先做到可解释、可单测；校准留给人评分 |
| 合规闸门误报率高 | 用户直接关掉闸门 | 默认"提示"级，仅"国界/敏感要素缺失"为阻断 |
| 溯源卡变成流水账 | 一页放不下、没人读 | 固定四段、每段 ≤8 行、参数只留关键值 |

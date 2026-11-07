# 地图合规闸门（P1-2 / W3）

**只做提示 + 阻断，不做合规判定。** 这个闸门提醒你"图上/数据里可能少东西了"：

- 图面有没有审图号（形如 `GS(2023)1234号`）、有没有"未送审"水印；
- 国界/敏感要素图层在不在、边界数据的来源登记了没有。

**是不是合法、能不能对外发布，以自然资源主管部门的地图审核结论为准。** 这句话同时写在
`compliance/rules.yaml`、溯源卡、TUI 验收事件与 `verify_outputs` 的结论里（有测试守着不漂移）。

---

## 1. 四条规则

规则库是 `src/gis_cli/compliance/rules.yaml`（YAML，可整份替换）：

| # | 规则 id | 级别 | 什么时候报 |
|---|---|---|---|
| ① | `national_boundary_unapproved` | **阻断** | 图面有国界/敏感要素图层（国界线、边境线、南海诸岛、九段线、军事设施…）**或**图名/注记里直接写了这些词，既没有有效审图号，也没有「未送审」水印，且按对外用途出图（`audience=public`，默认） |
| ② | `boundary_source_unregistered` | 警告 | 有边界类图层（国界/省界/县界/海岸线…），其中至少一个在数据目录里没有来源登记（无记录，或只登记成 `local`/`unknown`） |
| ③ | `watermark_missing` | 警告 | 显式按内部/送审前用途出图（`audience=internal`），但图面没有「未送审/内部使用」水印 |
| ④ | `approval_missing` | 提示 | 没有有效审图号、也没标「未送审」。**只是提一句**，不拦交付；已标未送审的图不再重复提示 |

分级语义：`block` 阻断交付、`warn` 照常交付但必须让人看见、`info` 只提示。
总体级别取命中里最重的一级。

几条刻意为之的口径（都是防误报的）：

- **图层名和图面文字都看**：真实图件的图层名常是 `boundary.shp` 这类泛名，国界线索全在图名上；
  只认图层名会漏报。
- **只认国界级/军事级词**：`敏感` 这种泛词不收——"生态敏感区"是常用图层名，收进来会把
  普通生态图判成阻断，误报一次用户就会关掉闸门。
- **"南海"不收，只收"南海诸岛"**：`南海子公园`（北京）是真实图层名。
- **拿不到数据目录时不下"来源未登记"的结论**：状态记 `unknown`，规则②不命中。
  "没核验" ≠ "没登记"。
- **已标「未送审」的图不再提示缺审图号**：未送审的图本来就不该有审图号。

## 2. 事实 → 规则：怎么扩一条规则

判定链是 **上下文 → 事实 → 规则**，规则里不写代码：

```
MapContext（图层/图面文字/声明）
      │  compliance/facts.py::collect_facts
      ▼
facts: layer_count, sensitive_count, boundary_count, unregistered_count,
       approval(valid|malformed|absent), watermark(present|absent),
       boundary_source(registered|partial|unregistered|unknown),
       catalog(available|unavailable), audience(public|internal), + 人读文案
      │  compliance/rules.py::evaluate
      ▼
rules.yaml 里的 when 条件（all / any / not / 叶子 {fact, op, value}）
```

加一条规则只改 YAML（事实够用时）：

```yaml
  - id: too_many_layers
    title: 图层过多
    level: info
    when:
      all:
        - fact: layer_count
          op: gt
          value: 8
    message: "本图叠了 {layer_count} 个图层，图例可能放不下"
    advice: "合并同类图层或拆图。"
```

算子：`eq / ne / in / not_in / gt / gte / lt / lte / truthy / falsy`。
`message`/`advice` 里的 `{xxx}` 只能引用事实名。**写错在装载时就报 `RuleError`**
（未知 fact / 未知算子 / 未知级别 / 重复 id / 文案引用了不存在的事实）——
拼错一个 fact 会让规则恒假，静默通过比误报更糟。

单条开关：YAML 里 `enabled: false`，或运行时
`check_map(ctx, enabled={"national_boundary_unapproved": False})`。
关掉的规则结论是 `off` 而不是 `pass`：**"没检查"和"检查了没问题"必须分得开**。

## 3. 两处落点

### 3.1 `create_map` 出图后自检

`cartography.create_map()` 在最后（图已落盘、版面体检之后）跑
`_compliance_selfcheck()`：

- 结论进 `result["compliance"]`（`level / blocked / findings / rules / facts / summary / notice`）；
- 上下文 + 结论写成边车 `<图名>.compliance.json`（与 `.spec.json` 同思路），并且
  **不改变出图成败** —— 是否阻断由调用方决定。

`create_map` 工具层（`engine/tools.py`）读这个级别：

- `blocked` → `ok=false`，`error.type=ComplianceBlocked`，提示里给出处理办法
  （补 `approval_number` / 加 `watermark` / 换已送审底图），**产物照常登记**，避免模型重复出图；
- 只有警告/提示 → `ok=true`，摘要写「合规提醒 N 项」。

三个声明参数都在 `create_map` 工具面上：

| 参数 | 含义 |
|---|---|
| `approval_number` | 图面已标注的审图号（`GS(2023)1234号`）；也可以用 `overlay.labels` 把文字真画到图上，闸门同样能读到 |
| `watermark` | 图面上的未送审水印文字（建议同时用 `overlay.labels` 画上去） |
| `audience` | `public`（对外，默认）/ `internal`（内部使用、送审前） |

⚠️ 闸门读的是**图面文字 + 调用方声明**，不读像素：声明了审图号但没真画上去，
闸门看不出来 —— 那是制图质量问题，归 `qc` 与人工。

### 3.2 `finish` 前汇总进 `verify_outputs`

`Verifier.verify()` 每次都汇总产物的边车（`compliance_check` → `aggregate`），
结果挂在 `verdict["compliance"]`，`verify_outputs` 工具直接返回它：

- `blocked` → `verdict["pass"]=false`、`verdict["hard_block"]=true`，
  摘要里排在最前，`repair_hint` 直接用阻断规则的 `advice`；
- 警告/提示 → 通过，但摘要里带上（"验收通过（合规闸门：警告 1 项…）"）。

`hard_block` 还挡掉了 loop 的"同一理由连拒 N 次就降级放行"（`engine/loop.py`）：
版面/语义问题可以商量，未审图的国界图不行。

边车里存的是**上下文**，验收时会按**当前**规则库重算：规则改了、某条规则被关了，
交付闸门立刻按新口径判，不用重新出图；上下文坏了才退回存档结论。

## 4. 卡片与 UI 文案

- 溯源卡新增「合规闸门」一节（紧跟"验证"，`report/provenance.py`），逐条列分级结论，
  并在下面固定一句责任边界；没有自检记录时写"未记录（不是 create_map 出的图，或旧版本出图）"。
- TUI 的验收事件（`ui/events.py`）在验收结果下面列出最多 3 条合规结论 + 同一句责任边界。

## 5. 验收与测试

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_compliance_gate.py tests/test_compliance_wiring.py -q
```

- `tests/test_compliance_gate.py`：**20 个合成样本**（构造已知问题，不依赖真实图件），
  每条样本写明"该报哪几条"，断言 **0 漏报、误报 0%**（验收上限 10%）、单图 ≤30 s；
  样本覆盖每条规则的正例、反例与边界情形（审图号格式不对、国界图层来源已登记、
  "生态敏感区"/"南海子公园"这类不该误报的名字、"已送审"不是水印、拿不到目录不指责……）。
  另有规则库校验、单条开关、YAML 扩规则、纯 Python（不 import arcpy）等用例。
- `tests/test_compliance_wiring.py`：自检→边车→验收→卡片/UI 全链路（不需要 ArcGIS）。

验收口径里"真实图件回放"仍等用户给样本（与 P1 其它人工验收项同一处理）。

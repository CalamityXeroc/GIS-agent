# -*- coding: utf-8 -*-
"""错误记忆：ArcPy 常见故障模式 → 可直接照做的修复建议。

只收录（a）跨任务反复出现、（b）修复方式确定 的条目。命中后追加到失败观察的
``hint`` 里，并随观察一起进入自修复提示词，避免模型每次重新踩同一个坑。

条目来自真实失败记录（基准实测 / 配方开发），不是凭空猜的。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# arcpy 工具所在模块的候选（用于"这个工具到底在哪个包"的实时探测）
_MODULE_CANDIDATES = (
    "management",
    "analysis",
    "sa",
    "da",
    "cartography",
    "stats",
    "ia",
    "ddc",
    "na",
    "3d",
    "conversion",
    "edit",
    "mp",
    "lyrx",
)


@dataclass
class ErrorHint:
    """一条错误记忆命中结果。"""

    key: str
    title: str
    fix: str

    def render(self) -> str:
        return f"[错误记忆/{self.key}] {self.title} → {self.fix}"


# key, 正则列表, 标题, 修复建议
_RULES: list[tuple[str, list[str], str, str]] = [
    (
        "sa-module",
        [r"module 'arcpy\.sa' has no attribute '([\w]+)'", r"module 'arcpy' has no attribute '([\w]+)'"],
        "工具不在这个模块里",
        "arcpy.sa 只有地图代数（Con/IsNull/SetNull/Reclassify/…）与少数栅格分析工具；"
        "GetRasterProperties、RasterToPolygon、ProjectRaster、ZonalStatisticsAsTable 等管理/分析类工具在 "
        "arcpy.management / arcpy.analysis。先探测：`import arcpy; print(hasattr(arcpy.management, 'GetRasterProperties'))`，"
        "不要凭记忆写模块前缀。",
    ),
    (
        "field-539",
        [r"ERROR 000539", r"ERROR 000339"],
        "字段名或表达式不合法",
        "字段名必须与数据里的真实名字完全一致（中文名、DBF 截断后的名字都要先查）。"
        "先 `[f.name for f in arcpy.ListFields(路径)]` 或调用 catalog_query 列出真实字段名再写表达式；"
        "字符串转数值要显式写，如 `float(!老年人![:-1])/100.0*float(!常住人!)`。",
    ),
    (
        "name-354",
        [r"ERROR 000354"],
        "输出名/路径不合法",
        "GDB 内的要素类不能带扩展名（写 `out_gdb\\layer`，不要 `out_gdb\\layer.shp`）；"
        "名字不能以数字开头、不含空格与 `-`；目录必须靠 `arcpy.management.CreateFileGDB` 创建，"
        "不要用 `os.makedirs` 造假目录（arcpy 不认）。",
    ),
    (
        "param-622",
        [r"ERROR 000622"],
        "参数取值不在允许集合内",
        "查该参数的合法取值再传：如 PolygonToRaster 的 build_ratings 只接受 BUILD / DO_NOT_BUILD；"
        "ZonalStatistics 的 statistics 只接受 SUM/MEAN/MIN/MAX/…（或 ALL）。",
    ),
    (
        "exists-725",
        [r"ERROR 000725", r"already exists", r"已存在"],
        "输出已存在或输入被占用",
        "内核每次执行已自动 `arcpy.env.overwriteOutput = True`；若仍报错，说明目标被锁（ArcGIS Pro 打开中）"
        "或正在被上次执行占用，改名输出或先 `arcpy.management.Delete(目标)`。",
    ),
    (
        "err-999999",
        [r"ERROR 999999", r"999999"],
        "通用失败（先重试，再定位）",
        "999999 多为瞬时问题（内存/锁/驱动）。原样重试一次；仍失败则缩小范围（先跑最小样例），"
        "检查输入/输出是否同一工作空间、坐标系是否一致、是否有 0 长度的几何。",
    ),
    (
        "gdb-exists",
        [r"CreateFileGDB[\s\S]{0,80}(无法创建|failed|已存在)", r"cannot create[\s\S]{0,40}\.gdb"],
        "建库失败",
        "先用 `arcpy.Exists(gdb路径)` 判断是否已存在，存在就直接复用，不要重复建库。",
    ),
    (
        "os-exists-gdb",
        [r"os\.path\.exists", r"Path\(.*\)\.exists\(\)[\s\S]{0,40}gdb"],
        "GDB 路径不是文件系统路径",
        "`.gdb` 及其内部数据集要用 `arcpy.Exists()` 判断；`os.path.exists` 对库内要素类永远为 False。",
    ),
    (
        "path-732",
        [r"ERROR 000732", r"输入不存在", r"does not exist or is not supported"],
        "输入路径不存在",
        "多为路径笔误或反斜杠转义（如 `\\B`、`\\t`）。统一用原始字符串 `r\"...\"` 或正斜杠；"
        "写路径前用 `arcpy.Exists(路径)` 或先列目录 `os.listdir(上级)` 核对。",
    ),
    (
        "crs-missing",
        [r"ERROR 000048", r"unknown spatial reference", r"Unknown coordinate system", r"坐标系未知"],
        "坐标系缺失或未知",
        "数据无 .prj 时先 `arcpy.management.DefineProjection` 定义再投影；"
        "投影工具要求输入已有坐标系，否则报错。",
    ),
    (
        "crs-mismatch",
        [r"ERROR 000117", r"does not match the spatial reference", r"坐标系不一致"],
        "输入之间坐标系不一致",
        "先用 `arcpy.Describe(路径).spatialReference.factoryCode` 核对各输入，统一投影到同一坐标系（含像元大小）再做分析。",
    ),
    (
        "py-import",
        [r"KeyError: '__import__'", r"NameError: name '__import__'"],
        "表达式求值缺少内置函数",
        "受限 eval 必须显式传入完整 builtins：`eval(expr, {'__builtins__': __builtins__, **ns})`；"
        "arcpy.sa 内部会调用 `__import__`。",
    ),
    (
        "gbk-encode",
        [r"UnicodeEncodeError.*gbk", r"codec can't encode"],
        "Windows 控制台编码",
        "把 stdout 重配置为 UTF-8（`sys.stdout.reconfigure(encoding='utf-8')`），"
        "或用 `PYTHONUTF8=1` 运行；写文件显式 `encoding='utf-8'`。",
    ),
    (
        "zone-field",
        [r"ZonalStatistics[\s\S]{0,120}zone[\s\S]{0,60}(invalid|错误)", r"ERROR 000865"],
        "分区字段类型不合适",
        "分区字段用整数型唯一 ID（如 OBJECTID 先拷成整型字段）；文本/浮点分区字段会报错或合并错误。",
    ),
    (
        "raster-nodata",
        [r"NoData[\s\S]{0,60}(空|为 0|zero)", r"ERROR 010240"],
        "NoData 被当成数值参与运算",
        "地图代数里 NoData 会传染：先用 `Con(IsNull(r), 0, r)` 补缺，再运算；"
        "除零用 `Con(分母 == 0, 0, 分子/分母)` 兜底。",
    ),
    (
        "wrong-type-864",
        [r"ERROR 000864", r"属性类型.*不在定义域", r"The value is not within the domain"],
        "属性类型/值不在定义域内",
        "参数类型不匹配：该参数要图层或表视图时不能传要素类路径；区县等分区字段要用整型唯一 ID"
        "（文本/浮点作 zone_field 会报错），必要时先用 field_calculate 新建长整型 ID 字段，"
        "再建图层：`arcpy.management.MakeFeatureLayer(路径, 'lyr')`。",
    ),
    (
        "need-layer-1628",
        [r"ERROR 001628", r"并非图层或具有连接的表", r"is not a layer or table view"],
        "参数需要图层/表视图而非路径",
        "先 `arcpy.management.MakeFeatureLayer(fc, 'lyr')`（表则 `MakeTableView`），把返回的图层名传给该参数；"
        "SelectLayerByAttribute/AddJoin/CalculateField 等对“图层”参数都要这样处理。",
    ),
    (
        "mpl-cjk",
        [r"missing from font", r"findfont: Font family .* not found", r"Glyph \d+ .* missing"],
        "matplotlib 缺中文字形（图里的中文会变成方框）",
        "内核启动时已自动配置中文字体（微软雅黑/黑体），若仍报缺字形，说明代码里显式指定了不带中文字形的字体"
        "（如 `fontfamily='Arial'`、`plt.rcParams['font.family']='serif'`）或覆盖了 rcParams。"
        "正确做法：`plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','SimSun']` 且 "
        "`plt.rcParams['axes.unicode_minus']=False`；**不要用 simsunb.ttf**（SimSun-ExtB 只有生僻字，会全变方框）。"
        "画完必须 `savefig` 到 output/ 再交付，交付前看一眼图上中文是否可读。",
    ),
    (
        "mpl-show",
        [r"FigureCanvasAgg.*non-interactive", r"plt\.show\(\).*(Agg|non-interactive)", r"show\(\) is deprecated"],
        "后台无窗口，plt.show() 不会显示图",
        "内核使用无界面后端（Agg），图必须 `fig.savefig(输出路径, dpi=150, bbox_inches='tight')` 落盘后才能作为成果交付，"
        "不要依赖 `plt.show()`。",
    ),
    (
        "layer-select-hang",
        [
            r"SelectLayerByAttribute",
            r"MakeFeatureLayer",
        ],
        "图层选择类 API 在持久内核里会挂死",
        "实测 SelectLayerByAttribute（配合 MakeFeatureLayer）在持久内核里会卡死且无法中断，"
        "甚至让内核进程致命退出。改用不需要图层对象的等价做法："
        "`arcpy.conversion.FeatureClassToFeatureClass(输入, 输出库, '名称', where_clause)`（最快）"
        "或 `arcpy.analysis.Select(输入, 输出, where_clause)`；"
        "只读筛选也可用 `arcpy.da.SearchCursor(路径, 字段, where_clause)`。",
    ),
    (
        "aprx-open-hang",
        [
            r"ArcGISProject",
            r"aprx[\s\S]{0,40}(挂起|hang|超时)",
            r"Execution exceeded.*did not stop",
        ],
        "在持久内核里打开 .aprx 会挂死",
        "实测在持久内核里用 `mp.ArcGISProject` 打开工程（尤其第二次）会挂住且中断无效。"
        "核验工程请改用工程工具 `check_map_project(aprx_path, renderer=..., colors=..., elements=...)`，"
        "或在独立进程跑 `scripts/verify_map_aprx.py`；agent 交付前只需确认图片与工程文件已生成。",
    ),
    (
        "kernel-timeout",
        [r"KernelTimeout", r"执行超时", r"timeout after"],
        "内核执行超时",
        "拆分步骤：把大范围运算按街区/分块拆成多次执行；或减少迭代（用游标批量代替逐要素循环）；"
        "确需长任务时在 execute_code 提高 timeout_seconds。",
    ),
]


def _probe_module(attribute: str) -> str:
    """实时探测某个遗漏属性实际属于哪个 arcpy 模块（探测失败则返回空串）。"""
    if not attribute or not re.match(r"^[A-Za-z_]\w*$", attribute):
        return ""
    try:
        import arcpy  # type: ignore

        for name in _MODULE_CANDIDATES:
            module = getattr(arcpy, name, None)
            if module is not None and hasattr(module, attribute):
                return f"实测：{attribute} 属于 arcpy.{name}"
    except Exception as exc:  # pragma: no cover - 只在无 arcpy 环境发生
        logger.debug("probe module failed: %s", exc)
    return ""


def match(text: str, limit: int = 3) -> list[ErrorHint]:
    """在错误文本里匹配已知故障模式，返回不超过 limit 条建议。"""
    if not text:
        return []
    haystack = str(text)
    hits: dict[str, ErrorHint] = {}
    for key, patterns, title, fix in _RULES:
        if key in hits:
            continue
        for pattern in patterns:
            found = re.search(pattern, haystack, re.IGNORECASE)
            if not found:
                continue
            extra = ""
            if key in {"sa-module"}:
                groups = [g for g in found.groups() if g]
                if groups:
                    probe = _probe_module(groups[0])
                    extra = f"（{probe}）" if probe else ""
            hits[key] = ErrorHint(key=key, title=title, fix=fix + extra)
            break
        if len(hits) >= limit:
            break
    return list(hits.values())


def format_hints(hints: list[Any]) -> str:
    """把命中结果拼成一段可直接拼进 hint 的文本。"""
    lines = []
    for item in hints:
        if isinstance(item, ErrorHint):
            lines.append(item.render())
        elif isinstance(item, str):
            lines.append(item)
    return "\n".join(lines)


def enrich_hint(hint: str, text: str, limit: int = 3) -> str:
    """把错误记忆追加到已有 hint 后面（去重、保持顺序）。"""
    hints = [item for item in match(text, limit=limit) if item.render() not in (hint or "")]
    if not hints:
        return hint
    block = format_hints(hints)
    if hint:
        return f"{hint}\n{block}"
    return block


__all__ = ["ErrorHint", "match", "format_hints", "enrich_hint"]
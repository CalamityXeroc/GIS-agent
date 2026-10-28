# -*- coding: utf-8 -*-
"""分析计划卡：执行前用 5 行人话讲清"做什么/用什么/出什么/怎么验/合规"。

设计边界（见 P0-3 详细计划 §3）：

- **不新增理解能力**：内容全部来自已有的 ``requirements``（需求清单）、
  ``datacatalog``（数据目录）与一个合规标记，纯确定性渲染；
- **不做合规判定**：只把"未送审，仅内部使用"这类标记如实写出来（判定在 P1）；
- 纯 stdlib，不碰 TTY、不调模型 —— 交互（问一句/读一行）放在 :class:`PlanGate`
  （loop）与 :class:`ChatPlanGate`（chat，只投递请求）。

引擎侧只负责“建卡 → 发事件 → 问闸门 → 改了重建”，渲染与提问都在这层。

两个闸门实现（动作词由 :func:`parse_decision` 单点定义）：

- :class:`PlanGate` —— ``gis-agent loop``：主线程直接读 ``input()``；
- :class:`ChatPlanGate` —— ``gis-agent chat``：引擎跑在工作线程，跨线程读 ``input()``
  会抢 prompt_toolkit 的按键，所以它只投递 :class:`PlanRequest`，由 UI 线程收键。
"""

from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .console_env import interactive_terminal as _console_interactive_terminal
from .console_env import stdin_is_tty, stdout_is_tty  # 单一实现，原样转出去（见 __all__）

#: 默认合规提示（明确写"未判定"，避免让人误以为已经过审）。
COMPLIANCE_DEFAULT = "未送审，仅内部使用（不做合规判定）"

#: 5 行的固定标签。顺序就是给人读的顺序：干什么 → 用什么 → 出什么 → 怎么验 → 合规。
_ROW_LABELS = ("做什么", "用什么", "输出", "怎么验", "合规")

#: 会被当成"产出"的路径后缀：任务描述里写了这些名字，就说明要落盘。
_OUTPUT_EXTS = (
    ".gdb",
    ".shp",
    ".gpkg",
    ".kml",
    ".csv",
    ".xlsx",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".pdf",
    ".html",
    ".aprx",
    ".mxd",
)

#: 值列的折行宽度（显示宽度，CJK 按 2 算）。
_VALUE_WIDTH = 56

#: 每行最多列几条（多了一屏放不下，也没人看）。
_MAX_ITEMS = 4

#: 路径分词的分隔符：空白与中英文标点（“output/a.gdb，output/b.gdb” 要拆开）
_TOKEN_SPLIT = re.compile(r"[\s，,;；、。]+")


# ---------------------------------------------------------------- 小工具


def display_width(text: str) -> int:
    """显示宽度：CJK 全角按 2 列算（对齐框线用，等价于 ``ui.render.disp_width``）。"""
    total = 0
    for ch in str(text):
        total += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return total


def _cut(text: str, limit: int) -> str:
    """按显示宽度截断，超出加省略号。"""
    text = str(text)
    if display_width(text) <= limit:
        return text
    out = ""
    for ch in text:
        if display_width(out + ch) > limit - 1:
            break
        out += ch
    return out + "…"


def _wrap(text: str, limit: int) -> list[str]:
    """按显示宽度折行（不切词，够用即可）。"""
    lines: list[str] = []
    current = ""
    for ch in str(text):
        if display_width(current + ch) > limit:
            lines.append(current)
            current = ch
        else:
            current += ch
    if current or not lines:
        lines.append(current)
    return lines


def _flat(text: Any) -> str:
    """压成一行（任务目标常带多行/首尾空白）。"""
    return " ".join(str(text or "").split())


def _relative(path: str, workspace: str | Path | None) -> str:
    """绝对路径压成工作区相对路径（人看的是 ``input/街道.shp`` 而不是整条盘符路径）。"""
    raw = str(path or "").replace("\\", "/")
    if workspace:
        try:
            return str(Path(path).relative_to(Path(workspace))).replace("\\", "/")
        except (ValueError, OSError):
            pass
    return raw


# ---------------------------------------------------------------- 计划卡


@dataclass
class PlanCard:
    """一张计划卡（5 行固定结构）。"""

    goal: str = ""
    data: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    compliance: str = COMPLIANCE_DEFAULT
    #: 数据目录里已登记的数据集总数（“已登记 N 个” 用）
    registered: int = 0
    #: 数据是不是“猜的”（任务没点名，从 input/ 回退）；猜的要写成候选
    data_guess: bool = False
    #: 用户在执行前提出的修改（一句话一条），会显示在"做什么"后面
    revisions: list[str] = field(default_factory=list)
    #: 数据行拿不到数据时的说明（没对上任务描述时提示"执行时再确认"）
    data_note: str = ""

    # -------------------------------------------------------------- 渲染
    def rows(self) -> list[tuple[str, str]]:
        """5 行 (标签, 值)。值已压成一行、可直接放进框里。"""
        doing = _flat(self.goal)
        if self.revisions:
            doing += "；修改：" + "；".join(_cut(_flat(r), 40) for r in self.revisions)

        if self.data:
            used = " + ".join(self.data[: _MAX_ITEMS])
            note = f"已登记 {self.registered} 个数据集" if self.registered else "取自数据目录"
            if self.data_guess:
                used = "候选（任务未点名）：" + used
            data = f"{used}（{note}）"
        else:
            data = self.data_note or "数据目录里还没对上任务要的数据，执行时先确认"

        outputs = " + ".join(self.outputs[: _MAX_ITEMS]) if self.outputs else "output/（执行时按需求命名）"
        if len(self.outputs) > _MAX_ITEMS:
            outputs += " …"

        checks = " / ".join(self.checks[:3]) if self.checks else "按需求清单逐条核对"

        return [
            ("做什么", doing),
            ("用什么", data),
            ("输出", outputs),
            ("怎么验", checks),
            ("合规", self.compliance or COMPLIANCE_DEFAULT),
        ]

    def render(self, *, interactive: bool = False) -> str:
        """画成带框的文本块（直接 print 就是计划卡）。"""
        body: list[str] = []
        label_width = max(display_width(f"{label}：") for label in _ROW_LABELS)
        for label, value in self.rows():
            head = f"{label}：" + " " * (label_width - display_width(f"{label}："))
            for i, chunk in enumerate(_wrap(_flat(value), _VALUE_WIDTH)):
                body.append(("│ " + (head if i == 0 else " " * label_width) + chunk).rstrip())

        width = max(display_width(line) for line in body) if body else 0
        title = "┌ 计划确认 "
        top = title + "─" * max(0, width - display_width(title))
        bottom = ("└ " + INTERACTIVE_HINT) if interactive else ("└ " + AUTO_START_NOTE)
        return "\n".join([top, *body, bottom])

    def to_dict(self) -> dict[str, Any]:
        """给 trace / 事件用的结构化版本（字段与 5 行一一对应）。"""
        return {
            "rows": [{"label": label, "value": value} for label, value in self.rows()],
            "goal": self.goal,
            "data": list(self.data),
            "outputs": list(self.outputs),
            "checks": list(self.checks),
            "compliance": self.compliance,
            "registered": self.registered,
            "data_guess": self.data_guess,
            "revisions": list(self.revisions),
            "text": self.render(),
        }


# ---------------------------------------------------------------- 建卡


def _keywords_text(goal: str, requirements: Sequence[Any]) -> str:
    """把任务目标与需求清单拼成一段可做子串匹配的文本。"""
    parts = [str(goal or "")]
    for req in requirements or []:
        parts.append(getattr(req, "text", "") or "")
        parts.append(getattr(req, "acceptance", "") or "")
    return "\n".join(parts)


def _exists_near(workspace: str | Path, rel: str) -> bool:
    """路径是否已经存在于工作区（根目录或 ``input/`` 下）。

    任务描述里的 ``用地.tif`` 可能只写文件名，而它真在 ``input/`` 里。
    """
    base = Path(workspace)
    for candidate in (base / rel, base / "input" / Path(rel).name):
        try:
            if candidate.exists():
                return True
        except OSError:  # pragma: no cover - 非法路径名
            continue
    return False


def _pick_inputs(
    haystack: str,
    records: Sequence[Any],
    *,
    workspace: str | Path | None = None,
    limit: int = _MAX_ITEMS,
) -> tuple[list[str], bool]:
    """从数据目录里挑“这次要用的数据”。

    先按名字在任务描述里出现来匹配（能追到 catalog 记录）；一个没对上时，
    退回 ``input/`` 下已登记的数据（外行不会写全文件名，这是常见情况）。
    第二个返回值 = 是否为“猜的”（回退结果）——猜的要在卡片上标明候选，
    不能冒充“任务点名要的数据”。
    """
    hits: list[Any] = []
    for rec in records:
        path = str(getattr(rec, "path", "") or "")
        if not path:
            continue
        names = {Path(path).name, Path(path).stem, str(getattr(rec, "name", "") or "")}
        if any(len(n) >= 2 and n in haystack for n in names):
            hits.append(rec)
    guessed = not hits
    if guessed:
        hits = [
            rec
            for rec in records
            if _relative(str(getattr(rec, "path", "") or ""), workspace).startswith(("input/", "workspace/input/"))
        ]
    out: list[str] = []
    for rec in hits:
        rel = _relative(str(getattr(rec, "path", "") or ""), workspace)
        if rel and rel not in out:
            out.append(rel)
        if len(out) >= limit:
            break
    return out, guessed and bool(out)


def _pick_outputs(
    haystack: str,
    records: Sequence[Any],
    *,
    workspace: str | Path | None = None,
    limit: int = _MAX_ITEMS,
) -> list[str]:
    """从任务描述里挑“这次要产出的东西”。

    判据：任务描述里出现、带产出后缀、且不是输入数据。宁可少列
    （留空让执行时命名），也不要凭空编一个路径。

    “是不是输入”不能只信 catalog：首次任务 catalog 可能还没扫完，那时
    ``known`` 是空的，把已有数据当产出写进卡片就是在编造。所以补两条
    磁盘侧判据：``input/`` 前缀、以及路径真的已存在（但 ``output/`` 下
    已存在的视为“这次要重写的旧产物”，不该排除）。
    """
    known = {
        str(Path(str(getattr(rec, "path", "") or "")).name)
        for rec in records
        if getattr(rec, "path", "")
    }
    out: list[str] = []
    for token in _TOKEN_SPLIT.split(str(haystack).replace("\\", "/")):
        raw = token.strip("，,。;；、()（）[]【】\"'`")
        rel = _relative(raw, workspace)
        if not rel.lower().endswith(_OUTPUT_EXTS):
            continue
        # 只有后缀没有文件名（“导出为 .gdb”）：pathlib 把 ``.gdb`` 当整名，suffix 为空
        if not Path(rel).suffix:
            continue
        # 光一个文件名说不清是输入还是产物（“把用地.tif 重分类”）——只认带目录的路径
        if "/" not in rel:
            continue
        if Path(rel).name in known or rel.startswith(("input/", "workspace/input/")):
            continue
        if workspace and not rel.startswith("output/") and _exists_near(workspace, rel):
            continue
        if rel not in out:
            out.append(rel)
        if len(out) >= limit:
            break
    return out


def _pick_checks(requirements: Sequence[Any]) -> list[str]:
    """"怎么验"取需求清单里的验收标准（没有就退回需求原文）。"""
    checks = [_flat(getattr(r, "acceptance", "")) for r in requirements or []]
    checks = [c for c in checks if c]
    if not checks:
        checks = [_flat(getattr(r, "text", "")) for r in requirements or []]
        checks = [c for c in checks if c]
    return [_cut(c, 34) for c in checks[:3]]


def build_plan_card(
    goal: str,
    requirements: Sequence[Any] = (),
    records: Iterable[Any] = (),
    *,
    workspace: str | Path | None = None,
    revisions: Sequence[str] = (),
    compliance: str = COMPLIANCE_DEFAULT,
    registered: int | None = None,
) -> PlanCard:
    """由任务目标 + 需求清单 + 数据目录记录生成计划卡（纯确定性）。

    ``registered``：数据目录里总共登记了多少个（拿 ``catalog.counts()`` 的合计，
    比“本次列出来的条数”准——列表是有 limit 的，写错就是在骗人）。
    """
    rows = list(records or [])
    haystack = _keywords_text(goal, requirements)
    data, guessed = _pick_inputs(haystack, rows, workspace=workspace)
    card = PlanCard(
        # “做什么”是重点行：允许两行，别截成一个孤零零的“…”
        goal=_cut(_flat(goal), _VALUE_WIDTH * 2),
        data=data,
        outputs=_pick_outputs(haystack, rows, workspace=workspace),
        checks=_pick_checks(requirements),
        compliance=compliance or COMPLIANCE_DEFAULT,
        registered=len(rows) if registered is None else max(int(registered), len(rows)),
        data_guess=guessed,
        revisions=[_flat(r) for r in revisions if _flat(r)],
    )
    if not data and rows:
        card.data_note = f"数据目录已登记 {len(rows)} 个数据集，但没对上任务描述，执行时先确认"
    return card


# ---------------------------------------------------------------- 确认闸门


@dataclass
class PlanDecision:
    """闸门的回答：``start`` / ``revise``（带一句话改动）/ ``cancel``。"""

    action: str = "start"
    text: str = ""


#: 非交互（``--yes`` / 非 TTY / 没注入闸门）时的底栏。
#: **不能写“已自动确认”**：卡片照打但根本没人按过键，写了就是在骗人
#: —— chat 入口以前就是这样（DAIR-30 修的就是这个）。
AUTO_START_NOTE = "未确认，直接开始（要改就在下一句里直接说）"

#: 真等人确认时的按键提示（loop 的主线程 input 与 chat 的输入框共用一套动作词）。
INTERACTIVE_HINT = "[s]/Enter 开始  [e] 修改  [q] 取消"

_START_WORDS = ("s", "start", "enter", "y", "yes", "开始", "确认", "ok")
_CANCEL_WORDS = ("q", "quit", "cancel", "取消", "退出")
_REVISE_WORDS = ("e", "edit", "修改", "改")


def parse_decision(text: str) -> PlanDecision | None:
    """把用户的一句话译成闸门动作；返回 ``None`` = 还要再问一句“怎么改”。

    动作词在这里单点定义：``PlanGate``（loop 在主线程读 ``input()``）与
    ``ChatPlanGate``（chat 在 prompt_toolkit 输入框里收键）必须认同一套，
    否则同一个按键在两个入口做不同的事。
    """
    answer = str(text or "").strip()
    if not answer:
        return PlanDecision("start")  # 卡片上写着 Enter 开始
    low = answer.lower()
    if low in _CANCEL_WORDS:
        return PlanDecision("cancel")
    if low in _REVISE_WORDS:
        return None
    if low in _START_WORDS:
        return PlanDecision("start")
    # 直接说出改动 = 修改（比逼用户先敲 e 再说话自然）
    return PlanDecision("revise", answer)


@dataclass
class PlanRequest:
    """一次待答复的确认请求：引擎线程投递，UI 线程答复。

    引擎在闸门前**不读 stdin**：请求挂到 :attr:`ChatPlanGate.pending`，引擎线程
    停在 :attr:`event` 上；主线程照常在 prompt_toolkit 的会话里读键，按 s/e/q
    调 :meth:`resolve`。跨线程 ``input()`` 会抢按键、把输入框冲乱——本单修的就是它。
    """

    card: PlanCard
    round: int = 1
    event: threading.Event = field(default_factory=threading.Event)
    decision: PlanDecision | None = None

    @property
    def answered(self) -> bool:
        return self.decision is not None

    def resolve(self, action: str, text: str = "") -> bool:
        """答复。幂等：已答过返回 ``False``（重复按键不能把答案覆盖掉）。"""
        if self.decision is not None:
            return False
        self.decision = PlanDecision(str(action or "start"), str(text or ""))
        self.event.set()
        return True

    def wait(self, timeout: float | None = None) -> PlanDecision:
        """等答复；没等到就按“开始”放行（闸门不能把任务拖死）。"""
        self.event.wait(timeout)
        return self.decision or PlanDecision("start")


class ChatPlanGate:
    """chat 用的闸门：把确认请求投递给 UI 线程，自己不碰终端。

    ``interactive=False``（非 TTY 的降级路径）：不暂停、直接 ``start``，行为与
    ``plan_gate=None`` 一致（卡片照样打印，底栏写“未确认，直接开始”）。
    """

    @property
    def pauses(self) -> bool:
        """这次会不会真的停下来（loop 据此决定要不要发 plan_pending 暂停态）。

        非交互降级（管道/重定向）不暂停：不能凭空发一个没人会答的暂停态。
        """
        return self.interactive

    def __init__(self, *, interactive: bool = True) -> None:
        self.interactive = bool(interactive)
        self._lock = threading.Lock()
        #: 当前等待答复的请求（主线程据此判断“下一行输入是计划确认”）。
        self.pending: PlanRequest | None = None
        self._round = 0

    def ask(self, card: PlanCard) -> PlanDecision:
        if not self.interactive:
            return PlanDecision("start")
        self._round += 1
        request = PlanRequest(card=card, round=self._round)
        with self._lock:
            self.pending = request
        try:
            return request.wait()
        finally:
            with self._lock:
                if self.pending is request:
                    self.pending = None

    def cancel_pending(self) -> bool:
        """把等待中的请求判成 cancel（Esc / 退出会话用；别把引擎线程挂死）。"""
        with self._lock:
            request = self.pending
        return bool(request is not None and request.resolve("cancel"))


def can_ask() -> bool:
    """能不能真的停下来问人：stdin **和** stdout 都得是终端。

    只看 stdin 会漏掉 ``gis-agent loop "…" | tee run.log`` 这类用法：提示进了
    管道，屏幕上什么都没有，人不知道要按 Enter，进程就默默卡在那里。

    判定实现在 :mod:`gis_cli.ui.console_env`（解 rich ``FileProxy`` 也在那里）；
    与 ``prompt.interactive_terminal()`` 只差 ``prompt_toolkit`` 可用性这一条
    —— 这里读 ``input()``，不需要它（DAIR-38）。
    """
    return _console_interactive_terminal()


class PlanGate:
    """执行前的确认闸门。

    - TTY（stdin 与 stdout 都是）：等人按一下（``Enter`` 开始 / ``e`` 改一句 / ``q`` 取消）；
    - 非 TTY 或 ``assume_yes``：不问，直接放行（CI 与脚本不阻塞）。
      调用方仍会打印计划卡——"看不看得见"和"要不要等"是两件事。
    """

    def __init__(
        self,
        *,
        interactive: bool | None = None,
        input_fn: Callable[[str], str] | None = None,
        echo: Callable[[str], None] | None = None,
    ) -> None:
        self.interactive = can_ask() if interactive is None else bool(interactive)
        self._input = input_fn
        self._echo = echo

    def ask(self, card: PlanCard) -> PlanDecision:
        """问一次。非交互直接 ``start``（不读 stdin，绝不阻塞）。"""
        if not self.interactive:
            return PlanDecision("start")
        read = self._input or (lambda prompt: input(prompt))
        say = self._echo or (lambda text: print(text, flush=True))
        while True:
            try:
                raw = read("> ")
            except EOFError:  # 输入被关掉：当"开始"，别把任务卡死
                return PlanDecision("start")
            decision = parse_decision(raw)
            if decision is None:  # e：再问一句改动
                try:
                    extra = (read("怎么改（一句话）：") or "").strip()
                except EOFError:
                    return PlanDecision("start")
                if not extra:
                    continue
                return PlanDecision("revise", extra)
            if decision.action == "revise":
                say("（按修改处理，改完再确认一次）")
            return decision


__all__ = [
    "AUTO_START_NOTE",
    "COMPLIANCE_DEFAULT",
    "INTERACTIVE_HINT",
    "ChatPlanGate",
    "PlanCard",
    "PlanDecision",
    "PlanGate",
    "PlanRequest",
    "build_plan_card",
    "can_ask",
    "display_width",
    "parse_decision",
    "stdin_is_tty",
    "stdout_is_tty",
]

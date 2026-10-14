# -*- coding: utf-8 -*-
"""技能发现与读取（Agent Skills 标准，``SKILL.md``）。

## 为什么要新写一套

仓库里已有 ``skills/loader.py``（旧引擎用），但它面向"参数模板 + 可执行代码"
的自研格式（``code_template``/``is_executable``），与主流 skills 不兼容。
主流格式（Claude Code / pi / agentskills.io）只要两样东西：

```
---
name: docx
description: 什么情况下该用它……
---

正文：怎么用、脚本在哪、注意事项
```

**只要 frontmatter 里有 name 与 description 就能用**——这样用户从别处
（如 Anthropic 官方 skills、``~/.claude/skills``）拿来的技能可以直接放进来。

## 工作方式：progressive disclosure

1. 启动时只把 **名字 + 描述** 放进 system prompt（省 context）；
2. 用户说"帮我写个 Word 文档"时，模型看到 docx 的描述匹配，调用 ``run_skill``
   读全文，再照做（脚本用 ``execute_code`` 跑）。

描述写得越具体，模型越可能在对的时候想起它——所以描述必须原样带上。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: ``SKILL.md`` 里 frontmatter 的最大长度（正常几 KB；超过说明不是 frontmatter）
_FRONTMATTER_MAX_CHARS = 16_000


@dataclass
class SkillInfo:
    """一个被发现到的技能。"""

    name: str
    description: str
    #: ``SKILL.md`` 所在目录（skill 的根；脚本相对它引用）
    root: Path
    #: skill 文件路径
    skill_file: Path
    #: 来源描述（哪个搜索路径找到的，便于排查"为什么没发现我的技能"）
    source: str = ""
    #: 附加元数据（license/compatibility 等，原样保留）
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def script_dir(self) -> Path:
        """相对脚本的基准目录（skill 自己的目录）。"""
        return self.root


def _split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """剥离并解析 ``---`` 包裹的 YAML frontmatter。

    只做"够用"的解析：顶层的 ``key: value``、``key: [a, b]``、以及折叠的
    ``>-``/``|`` 块。不引入 pyyaml 依赖，也不追求完整的 YAML 语义——
    frontmatter 里出现复杂嵌套时，``name`` 与 ``description`` 通常仍是简单标量。
    """
    if not text.startswith("---"):
        return {}, text
    head, _, rest = text[3:].partition("\n---")
    if not _:
        return {}, text
    if len(head) > _FRONTMATTER_MAX_CHARS:
        return {}, text

    data: dict[str, object] = {}
    key: str | None = None
    buffer: list[str] = []
    block_indent: int | None = None

    def _flush() -> None:
        nonlocal key, buffer
        if key:
            value = "\n".join(buffer).strip()
            # 去掉折叠块常见的引号
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            data[key] = value
        key, buffer = None, []

    for raw_line in head.splitlines():
        if not raw_line.strip():
            if key:
                buffer.append("")
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if key and block_indent is not None and indent >= block_indent:
            buffer.append(raw_line.strip())
            continue
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            continue
        matched = re.match(r"^([A-Za-z0-9_\-]+)\s*:\s*(.*)$", stripped)
        if not matched:
            if key:
                buffer.append(stripped)
            continue
        _flush()
        name, value = matched.group(1), matched.group(2).strip()
        if value in (">", ">-", "|", "|-"):
            key, buffer, block_indent = name, [], indent + 2
            continue
        if value.startswith("[") and value.endswith("]"):
            items = [part.strip().strip("\"'") for part in value[1:-1].split(",")]
            data[name] = [item for item in items if item]
            block_indent = None
            continue
        data[name] = value.strip("\"'")
        block_indent = None
    _flush()
    return data, rest.lstrip("\n")


def parse_skill_file(path: Path, *, source: str = "") -> SkillInfo | None:
    """解析一个 ``SKILL.md``；缺 name/description 的按规范丢弃（返回 None）。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("读取技能文件失败 %s: %s", path, exc)
        return None
    meta, _body = _split_frontmatter(text)
    name = str(meta.get("name") or "").strip()
    description = str(meta.get("description") or "").strip()
    if not name:
        # 容许没有 frontmatter 但文件名像技能的情况：用父目录名兜底
        if path.name.upper() == "SKILL.MD" and path.parent.name:
            name = path.parent.name
        else:
            return None
    if not description:
        # 规范要求 description 必填；没有就无法让模型判断何时使用
        logger.debug("技能 %s 缺 description，按规范跳过", path)
        return None
    extra = {k: v for k, v in meta.items() if k not in ("name", "description")}
    return SkillInfo(
        name=name,
        description=description,
        root=path.parent,
        skill_file=path,
        source=source,
        metadata=extra,
    )


#: 固定的全局技能目录（相对用户主目录）；``include_global=False`` 时剔掉
_GLOBAL_SKILL_DIRS = (
    ".agents/skills",
    ".claude/skills",
    ".codex/skills",
    ".pi/agent/skills",
)


def default_skill_roots(workspace: str | Path) -> list[tuple[Path, str]]:
    """默认搜索路径（``(目录, 来源说明)``）。

    对齐主流做法（见 pi ``docs/skills.md`` 的 Locations）：项目内优先，
    再把常见 harness 的全局目录挂上——**用户从别处拿来的技能能直接生效**。
    """
    ws = Path(workspace)
    roots: list[tuple[Path, str]] = [
        (ws / "skills", "工作区 skills/"),
        (ws.parent / "skills", "上级目录 skills/"),
        (Path.home() / ".agents" / "skills", "全局 ~/.agents/skills"),
        (Path.home() / ".claude" / "skills", "Claude Code ~/.claude/skills"),
        (Path.home() / ".codex" / "skills", "Codex ~/.codex/skills"),
        (Path.home() / ".pi" / "agent" / "skills", "pi ~/.pi/agent/skills"),
    ]
    # 仓库自带的 workspace/skills：**只在工作区确实位于仓库内时**才认，
    # 否则在别处建工作区也会把仓库的技能扫进来（测试里实测过这种污染）。
    repo_skills = _repo_skills_dir()
    if repo_skills is not None and _looks_like_inside(ws, repo_skills.parent.parent):
        roots.append((repo_skills, "仓库 workspace/skills"))
    return roots


def _repo_skills_dir() -> Path | None:
    """本仓库自带的 ``workspace/skills`` 目录（不在源码树里跑时为 None）。"""
    try:
        candidate = Path(__file__).resolve().parents[3] / "workspace" / "skills"
    except Exception:  # pragma: no cover - 打包环境
        return None
    return candidate if candidate.is_dir() else None


def _looks_like_inside(path: Path, parent: Path) -> bool:
    """path 是否在 parent 之下（路径不合法时按不在处理）。"""
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except Exception:
        return False


def discover_skills(
    workspace: str | Path,
    *,
    extra_roots: list[tuple[Path, str]] | None = None,
    include_global: bool = True,
    limit: int = 200,
) -> list[SkillInfo]:
    """扫描所有技能目录，返回去重后的技能列表（重名保留先找到的）。

    发现规则：
    - **子目录**里的 ``SKILL.md`` → 一个技能（主流形态：``docx/SKILL.md``）
    - 根下的 **裸 .md** → 也是一个技能（本项目 GIS 技能原来就是这种，如
      ``buffer_analysis.md``）；但如果同名目录里已有 ``SKILL.md``，以目录里的为准。

    ``include_global=False`` 只扫项目内目录（测试要可重现：不能因开发者本机
    装了多少全局技能而变）。
    """
    roots = _resolve_roots(workspace, extra_roots, include_global=include_global)
    found: dict[str, SkillInfo] = {}

    def _remember(info: SkillInfo | None) -> None:
        if info is None or info.name in found:
            return
        found[info.name] = info

    for root, source in roots:
        try:
            if not root.is_dir():
                continue
        except OSError:
            continue
        # 1) 子目录里的 SKILL.md（含更深层，兼容 scripts/ 之类无关目录）
        try:
            for path in sorted(root.glob("*/SKILL.md")):
                _remember(parse_skill_file(path, source=source))
        except OSError as exc:
            logger.debug("扫描 %s 失败: %s", root, exc)
        # 2) 根下的裸 .md
        try:
            for path in sorted(root.glob("*.md")):
                _remember(parse_skill_file(path, source=source))
        except OSError as exc:
            logger.debug("扫描 %s 失败: %s", root, exc)
        if len(found) >= limit:
            break

    return sorted(found.values(), key=lambda item: item.name)


def _resolve_roots(
    workspace: str | Path,
    extra_roots: list[tuple[Path, str]] | None,
    *,
    include_global: bool,
) -> list[tuple[Path, str]]:
    """定下要扫哪些目录。

    ``extra_roots`` 给定时只用它（测试用，可重现）；否则用默认列表，
    并且 ``include_global=False`` 时剔掉用户主目录下的那些（否则测试结果
    会因开发者本机装了多少全局技能而变）。
    """
    if extra_roots is not None:
        return list(extra_roots)
    roots = default_skill_roots(workspace)
    if include_global:
        return roots
    # 剔掉固定的那几个“全局”目录。不能用“在 home 之下”判断：
    # 测试的 tmp_path 也在 home 下（AppData/Local/Temp），会被误剔。
    global_dirs = {(Path.home() / part).resolve() for part in _GLOBAL_SKILL_DIRS}
    kept: list[tuple[Path, str]] = []
    for path, source in roots:
        try:
            if path.resolve() in global_dirs:
                continue
        except Exception:  # pragma: no cover - 路径不可解析
            pass
        kept.append((path, source))
    return kept


def read_skill_text(info: SkillInfo, *, max_chars: int = 60_000) -> str:
    """读技能全文（给 ``run_skill`` 用）。"""
    try:
        text = info.skill_file.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"（读取技能失败：{exc}）"
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n…（已截断，原文 {len(text)} 字符）"
    return text


def skills_prompt_section(skills: list[SkillInfo], *, max_skills: int = 40) -> str:
    """生成 system prompt 里的技能段（只放名字 + 描述，省 context）。

    技能可能有上百个（扫了 ``~/.agents/skills`` / ``~/.claude/skills``）；
    全部展开会在 system prompt 里吃掉两万多字符（实测）。所以：

    - 先排**优先类**（文档/办公/GIS 相关，即“写 word、看 pdf”这类真实需求）；
    - 剩下的按名字排在后面，总共最多 ``max_skills`` 条；
    - 取不完的只报数量——反正 ``/skills`` 能看全，而模型需要时也知道
      还有别的（并可用 ``run_skill`` 试探）。
    """
    if not skills:
        return ""
    ordered = _prioritized(skills)
    shown = ordered[:max_skills]
    hidden = len(ordered) - len(shown)

    lines = [
        "## 可用技能（SKILL.md）",
        "下面是已安装的技能。**当任务与某个技能描述吻合时（例如用户要写 Word 文档、",
        "读 PDF、做幻灯片），先调用 run_skill 读取该技能全文，再严格按其中的步骤执行**；",
        "技能自带的脚本用 execute_code 运行（路径相对技能目录）。",
        "",
    ]
    for item in shown:
        description = " ".join(item.description.split())
        lines.append(f"- **{item.name}**：{description}")
        lines.append(f"  （目录：{item.root}）")
    if hidden > 0:
        lines.append(f"（另有 {hidden} 个技能未列出，可用 /skills 查看全部。）")
    lines.append("")
    lines.append("以上之外的常规 GIS 操作仍按配方/自写代码处理，不必强行套用技能。")
    return "\n".join(lines)


#: 优先展示的技能关键词（命中越多越靠前）——真实高频需求所在
_PRIORITY_HINTS = (
    "docx", "word", "pdf", "pptx", "xlsx", "excel", "office", "md-to-office",
    "doc-coauthoring", "gis", "geospatial", "map", "chart", "cartograph",
    "report", "paper", "notes", "skill", "pdf", "document",
)


def _prioritized(skills: list[SkillInfo]) -> list[SkillInfo]:
    """优先类排前面（文档/办公/GIS），其余保持名字序。

    目的：用户说“写个 word 文档”时，docx 的描述必须在 system prompt 里——
    如果它被 100 个无关技能排到后面截掉了，自然语言触发就失效了。
    """

    def score(item: SkillInfo) -> int:
        haystack = f"{item.name} {item.description}".lower()
        return sum(1 for hint in _PRIORITY_HINTS if hint in haystack)

    return sorted(skills, key=lambda item: (-score(item), item.name))


__all__ = [
    "SkillInfo",
    "default_skill_roots",
    "discover_skills",
    "parse_skill_file",
    "read_skill_text",
    "skills_prompt_section",
]

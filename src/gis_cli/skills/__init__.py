"""Skills 包。

## 两种技能体系并存（历史原因）

- **``catalog``**：主流 Agent Skills 格式（``SKILL.md`` + frontmatter）。
  **新引擎用这套**（``engine/tools.py`` 的 ``run_skill`` 与
  ``engine/loop.py`` 的技能列表）。它只依赖标准库，零重包。
- **``base`` / ``bundled`` / ``loader``**：旧引擎的自研格式
  （``code_template`` + ``ToolRegistry``），依赖 pydantic。

## 为什么要惰性导出

以前 ``__init__.py`` 立即 import ``base``/``bundled``/``loader``，导致

    import gis_cli.skills.catalog     # 新引擎只需要这个
      → 触发 skills.__init__
      → 触发 base → pydantic

实测：在没有 pydantic 的环境里 `import gis_cli.skills.catalog` 直接
ImportError。现在改成 PEP 562 惰性导出，新引擎不再被旧技能体系拖累。
"""

from __future__ import annotations

from typing import Any

from .catalog import (  # noqa: F401  (新引擎直接用这些)
    SkillInfo,
    discover_skills,
    parse_skill_file,
    read_skill_text,
    skills_prompt_section,
)

#: 旧技能体系（惰性；随旧引擎下线一起删除）。
_LEGACY_EXPORTS: dict[str, str] = {
    "Skill": "..legacy.core_tools.skills_legacy.base",
    "SkillContext": "..legacy.core_tools.skills_legacy.base",
    "SkillResult": "..legacy.core_tools.skills_legacy.base",
    "SkillRegistry": "..legacy.core_tools.skills_legacy.base",
    "ThematicMapSkill": "..legacy.core_tools.skills_legacy.bundled",
    "DataIntegrationSkill": "..legacy.core_tools.skills_legacy.bundled",
    "QualityAssuranceSkill": "..legacy.core_tools.skills_legacy.bundled",
    "SkillDefinition": "..legacy.core_tools.skills_legacy.loader",
    "SkillMatch": "..legacy.core_tools.skills_legacy.loader",
    "SkillLoader": "..legacy.core_tools.skills_legacy.loader",
    "parse_skill_md": "..legacy.core_tools.skills_legacy.loader",
    "get_skill_loader": "..legacy.core_tools.skills_legacy.loader",
    "reload_skills": "..legacy.core_tools.skills_legacy.loader",
}

__all__ = [
    # 新引擎（SKILL.md 格式）
    "SkillInfo",
    "discover_skills",
    "parse_skill_file",
    "read_skill_text",
    "skills_prompt_section",
    # 旧引擎（惰性）
    *_LEGACY_EXPORTS.keys(),
]


def __getattr__(name: str) -> Any:
    """PEP 562：访问旧名字时才 import（见模块 docstring）。"""
    module_name = _LEGACY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)

"""Skills module - Bundled and custom skills for GIS CLI.

Skills are high-level capabilities that combine multiple tools to achieve
complex workflows. Similar to Claude Code's skills system.

支持两种技能定义方式：
1. Python 类（内置技能）- 在 bundled.py 中定义
2. SKILL.md 文件（自定义技能）- 在 workspace/skills/ 目录中定义
"""

from .base import Skill, SkillContext, SkillResult, SkillRegistry
from .bundled import (
    ThematicMapSkill,
    DataIntegrationSkill,
    QualityAssuranceSkill,
)
from .loader import (
    SkillDefinition,
    SkillMatch,
    SkillLoader,
    parse_skill_md,
    get_skill_loader,
    reload_skills,
)

__all__ = [
    # 基础类
    "Skill",
    "SkillContext",
    "SkillResult",
    "SkillRegistry",
    # 内置技能
    "ThematicMapSkill",
    "DataIntegrationSkill",
    "QualityAssuranceSkill",
    # SKILL.md 加载器
    "SkillDefinition",
    "SkillMatch",
    "SkillLoader",
    "parse_skill_md",
    "get_skill_loader",
    "reload_skills",
]

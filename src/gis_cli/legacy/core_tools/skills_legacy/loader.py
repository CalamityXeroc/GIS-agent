# -*- coding: utf-8 -*-
"""SKILL.md 文件加载器

实现类似 Claude Code 的 SKILL.md 技能定义格式。
用户可以在 workspace/skills/ 目录下创建 .md 文件来定义自定义技能。

SKILL.md 格式示例：
```markdown
---
name: buffer_analysis
description: 对矢量数据进行缓冲区分析
tags: [analysis, buffer, spatial]
---

# 缓冲区分析技能

## 触发条件
当用户提到"缓冲区"、"buffer"、"周边范围"等关键词时触发。

## 执行步骤
1. 识别输入图层
2. 确定缓冲距离
3. 执行缓冲区分析
4. 输出结果

## ArcPy 代码模板
```python
import arcpy
arcpy.Buffer_analysis(
    in_features="${input_layer}",
    out_feature_class="${output_path}",
    buffer_distance_or_field="${buffer_distance}"
)
```
```
"""

from __future__ import annotations

import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SkillDefinition:
    """从 SKILL.md 文件解析出的技能定义"""
    
    # 基本信息（来自 YAML frontmatter）
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    
    # 可选字段
    category: str = "custom"
    version: str = "1.0"
    author: str = ""
    
    # 触发条件
    triggers: list[str] = field(default_factory=list)  # 关键词列表
    
    # 执行步骤
    steps: list[str] = field(default_factory=list)
    
    # 代码模板
    code_template: str = ""

    # 是否可直接用于 execute_code 执行
    is_executable: bool = False
    
    # 参数定义
    required_inputs: list[str] = field(default_factory=list)
    optional_inputs: dict[str, Any] = field(default_factory=dict)
    
    # 原始内容
    raw_content: str = ""
    source_file: Path | None = None
    
    def matches(self, query: str) -> bool:
        """检查查询是否匹配此技能"""
        query_lower = query.lower()
        
        # 检查名称
        if self.name.lower() in query_lower:
            return True
        
        # 检查触发词
        for trigger in self.triggers:
            if trigger.lower() in query_lower:
                return True
        
        # 检查标签
        for tag in self.tags:
            if tag.lower() in query_lower:
                return True
        
        # 检查描述中的关键词
        desc_words = self.description.lower().split()
        for word in desc_words:
            if len(word) >= 3 and word in query_lower:
                return True
        
        return False
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "category": self.category,
            "triggers": self.triggers,
            "steps": self.steps,
            "required_inputs": self.required_inputs,
            "optional_inputs": self.optional_inputs,
            "code_template": self.code_template,
            "is_executable": self.is_executable,
        }


@dataclass
class SkillMatch:
    """技能匹配结果（带评分与命中证据）。"""
    skill: SkillDefinition
    score: int
    name_hit: bool = False
    trigger_hits: list[str] = field(default_factory=list)
    tag_hits: list[str] = field(default_factory=list)


def parse_skill_md(content: str, source_file: Path | None = None) -> SkillDefinition | None:
    """解析 SKILL.md 文件内容
    
    Args:
        content: Markdown 文件内容
        source_file: 源文件路径（可选）
    
    Returns:
        SkillDefinition 或 None（解析失败时）
    """
    # 提取 YAML frontmatter（允许无 frontmatter 的弱格式技能）
    frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
    match = re.match(frontmatter_pattern, content, re.DOTALL)

    frontmatter: dict[str, Any] = {}
    body = content
    if match:
        try:
            parsed_frontmatter = yaml.safe_load(match.group(1))
            if isinstance(parsed_frontmatter, dict):
                frontmatter = parsed_frontmatter
            else:
                return None
        except yaml.YAMLError:
            return None
        body = content[match.end():]

    adapted = _adapt_frontmatter(frontmatter, body, source_file)

    # 必需字段
    name = adapted.get("name")
    description = adapted.get("description")

    if not name or not description:
        return None
    
    # 解析触发条件
    triggers = adapted.get("triggers", [])
    if not triggers:
        # 从正文中提取
        triggers = _extract_triggers(body)
    
    # 解析步骤
    steps = adapted.get("steps", [])
    if not steps:
        # 从正文中提取
        steps = _extract_steps(body)
    
    # 解析代码模板
    code_template = adapted.get("code_template", "")
    if not code_template:
        # 从正文中提取
        code_template = _extract_code_template(body)
    
    # 解析参数
    required_inputs = adapted.get("required_inputs", [])
    optional_inputs = adapted.get("optional_inputs", {})

    # 解析执行模式（可通过 frontmatter 显式指定）
    explicit_executable = frontmatter.get("executable") if isinstance(frontmatter, dict) else None
    execution_mode = str(frontmatter.get("execution_mode") or frontmatter.get("mode") or "").strip().lower() if isinstance(frontmatter, dict) else ""
    
    # 如果没有指定参数，从代码模板中提取
    if not required_inputs and code_template:
        required_inputs = _extract_params_from_code(code_template)

    if isinstance(explicit_executable, bool):
        is_executable = explicit_executable
    elif execution_mode in {"guide", "reference", "doc", "documentation"}:
        is_executable = False
    else:
        is_executable = _looks_executable_code_template(code_template)
    
    return SkillDefinition(
        name=name,
        description=description,
        tags=adapted.get("tags", []),
        category=adapted.get("category", "custom"),
        version=adapted.get("version", "1.0"),
        author=adapted.get("author", ""),
        triggers=triggers,
        steps=steps,
        code_template=code_template,
        is_executable=is_executable,
        required_inputs=required_inputs,
        optional_inputs=optional_inputs,
        raw_content=content,
        source_file=source_file
    )


def _adapt_frontmatter(
    frontmatter: dict[str, Any],
    body: str,
    source_file: Path | None,
) -> dict[str, Any]:
    """将外部通用 skill 元数据映射到内部统一字段。"""

    def _first_nonempty(*values: Any) -> str:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _to_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str) and value.strip():
            parts = re.split(r"[,，;；\n]", value)
            return [p.strip() for p in parts if p.strip()]
        return []

    name = _first_nonempty(
        frontmatter.get("name"),
        frontmatter.get("skill_name"),
        frontmatter.get("id"),
    )
    if not name and source_file is not None:
        name = source_file.stem.strip()

    description = _first_nonempty(
        frontmatter.get("description"),
        frontmatter.get("summary"),
        frontmatter.get("purpose"),
    )
    if not description:
        description = _extract_description(body)

    tags = _to_list(frontmatter.get("tags") or frontmatter.get("keywords") or frontmatter.get("labels"))
    triggers = _to_list(frontmatter.get("triggers") or frontmatter.get("keywords") or frontmatter.get("activation"))

    required_inputs = frontmatter.get("required_inputs")
    if not isinstance(required_inputs, list):
        required_inputs = frontmatter.get("required")
    required_inputs = [str(x).strip() for x in (required_inputs or []) if str(x).strip()]

    optional_inputs = frontmatter.get("optional_inputs")
    if not isinstance(optional_inputs, dict):
        optional_inputs = frontmatter.get("defaults")
    if not isinstance(optional_inputs, dict):
        optional_inputs = {}

    code_template = _first_nonempty(
        frontmatter.get("code_template"),
        frontmatter.get("template"),
        frontmatter.get("code"),
    )

    return {
        "name": name,
        "description": description,
        "tags": tags,
        "category": _first_nonempty(frontmatter.get("category")) or "custom",
        "version": _first_nonempty(frontmatter.get("version")) or "1.0",
        "author": _first_nonempty(frontmatter.get("author")),
        "triggers": triggers,
        "steps": frontmatter.get("steps") or [],
        "required_inputs": required_inputs,
        "optional_inputs": optional_inputs,
        "code_template": code_template,
    }


def _extract_description(body: str) -> str:
    """从 Markdown 正文提取首个可用描述。"""
    for line in body.splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("#"):
            continue
        return text[:200]
    return ""


def _extract_triggers(body: str) -> list[str]:
    """从正文中提取触发条件"""
    triggers = []
    
    # 查找"触发条件"或"Triggers"部分
    patterns = [
        r'##\s*触发条件\s*\n(.*?)(?=\n##|\Z)',
        r'##\s*Triggers?\s*\n(.*?)(?=\n##|\Z)',
        r'##\s*关键词\s*\n(.*?)(?=\n##|\Z)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, body, re.DOTALL | re.IGNORECASE)
        if match:
            section = match.group(1)
            # 提取引号中的词或列表项
            triggers.extend(re.findall(r'"([^"]+)"', section))
            triggers.extend(re.findall(r"'([^']+)'", section))
            # 提取列表项
            triggers.extend(re.findall(r'-\s*(.+)', section))
            break
    
    return [t.strip() for t in triggers if t.strip()]


def _extract_steps(body: str) -> list[str]:
    """从正文中提取执行步骤"""
    steps = []
    
    # 查找"执行步骤"或"Steps"部分
    patterns = [
        r'##\s*执行步骤\s*\n(.*?)(?=\n##|\Z)',
        r'##\s*Steps?\s*\n(.*?)(?=\n##|\Z)',
        r'##\s*工作流程?\s*\n(.*?)(?=\n##|\Z)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, body, re.DOTALL | re.IGNORECASE)
        if match:
            section = match.group(1)
            # 提取编号列表
            numbered = re.findall(r'\d+\.\s*(.+)', section)
            if numbered:
                steps.extend(numbered)
            else:
                # 提取无序列表
                steps.extend(re.findall(r'-\s*(.+)', section))
            break
    
    return [s.strip() for s in steps if s.strip()]


def _extract_code_template(body: str) -> str:
    """从正文中提取代码模板"""
    # 查找 Python 代码块
    patterns = [
        r'```python\s*\n(.*?)\n```',
        r'```py\s*\n(.*?)\n```',
        r'```arcpy\s*\n(.*?)\n```',
        r'```\s*\n(.*?)\n```',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, body, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return ""


def _extract_params_from_code(code: str) -> list[str]:
    """从代码模板中提取参数（${param_name} 格式）"""
    params = re.findall(r'\$\{(\w+)\}', code)
    return list(set(params))  # 去重


def _looks_executable_code_template(code: str) -> bool:
    """判断代码模板是否可用于执行。

    规则：
    - 空模板不可执行
    - 包含 set_result / ArcPy / 参数占位符（${...}）之一，视为可执行
    - 仅文档示例代码（如普通 Python 片段）默认视为不可执行，避免误触发
    """
    if not isinstance(code, str):
        return False
    text = code.strip()
    if not text:
        return False

    lowered = text.lower()
    has_set_result = "set_result(" in lowered
    has_arcpy = "import arcpy" in lowered or "arcpy." in lowered
    has_placeholders = bool(re.search(r'\$\{\w+\}', text))

    return has_set_result or has_arcpy or has_placeholders


class SkillLoader:
    """技能加载器 - 从文件系统加载 SKILL.md 文件"""
    
    def __init__(self, skills_dir: Path | str | None = None):
        """
        Args:
            skills_dir: 技能目录路径，默认为 workspace/skills/
        """
        if skills_dir is None:
            # 默认路径
            skills_dir = Path.cwd() / "workspace" / "skills"
        
        self.skills_dir = Path(skills_dir)
        self._skills: dict[str, SkillDefinition] = {}
        self._loaded = False
    
    def load(self) -> int:
        """加载所有技能文件
        
        Returns:
            加载的技能数量
        """
        self._skills.clear()
        
        if not self.skills_dir.exists():
            self._loaded = True
            return 0
        
        count = 0

        # 仅加载候选技能文件：
        # - skills 根目录下的 *.md（本地轻量技能）
        # - 任意子目录中的 SKILL.md（主流目录化技能）
        for md_file in self._iter_candidate_skill_files():
            if any(part.startswith(".") for part in md_file.relative_to(self.skills_dir).parts):
                continue
            try:
                content = md_file.read_text(encoding="utf-8")
                skill = parse_skill_md(content, md_file)
                
                if skill:
                    key = skill.name
                    if key in self._skills:
                        suffix_base = md_file.parent.name if md_file.name.lower() == "skill.md" else md_file.stem
                        key = f"{skill.name}__{suffix_base}"
                        dedup_index = 2
                        while key in self._skills:
                            key = f"{skill.name}__{suffix_base}_{dedup_index}"
                            dedup_index += 1
                        skill.name = key
                    self._skills[key] = skill
                    count += 1
            except Exception as e:
                print(f"警告: 加载 {md_file} 失败: {e}")
        
        self._loaded = True
        return count
    
    def reload(self) -> int:
        """重新加载所有技能"""
        return self.load()
    
    def get(self, name: str) -> SkillDefinition | None:
        """获取指定技能"""
        if not self._loaded:
            self.load()
        return self._skills.get(name)
    
    def list_skills(self) -> list[SkillDefinition]:
        """获取所有技能"""
        if not self._loaded:
            self.load()
        return list(self._skills.values())
    
    def search(self, query: str) -> list[SkillDefinition]:
        """搜索匹配的技能
        
        Args:
            query: 用户查询文本
        
        Returns:
            匹配的技能列表
        """
        if not self._loaded:
            self.load()
        
        matches = []
        for skill in self._skills.values():
            if skill.matches(query):
                matches.append(skill)
        
        return matches
    
    def find_best_match(self, query: str) -> SkillDefinition | None:
        """找到最匹配的技能
        
        Args:
            query: 用户查询文本
        
        Returns:
            最匹配的技能，或 None
        """
        match = self.find_best_match_with_score(query, min_score=1)
        return match.skill if match else None

    def find_best_match_with_score(
        self,
        query: str,
        min_score: int = 6,
        only_executable: bool = False,
    ) -> SkillMatch | None:
        """找到最匹配技能并返回评分详情。"""
        matches = self.search(query)
        if only_executable:
            matches = [s for s in matches if s.is_executable]
        if not matches:
            return None

        query_lower = query.lower()
        scored: list[SkillMatch] = []

        for skill in matches:
            name_hit = skill.name.lower() in query_lower
            trigger_hits = [t for t in skill.triggers if t.lower() in query_lower]
            tag_hits = [t for t in skill.tags if t.lower() in query_lower]

            score = 0
            if name_hit:
                score += 12
            score += len(trigger_hits) * 5
            score += len(tag_hits) * 3

            # 轻量描述匹配，避免漏召回
            desc_words = [w for w in skill.description.lower().split() if len(w) >= 3]
            if any(w in query_lower for w in desc_words):
                score += 1

            scored.append(SkillMatch(
                skill=skill,
                score=score,
                name_hit=name_hit,
                trigger_hits=trigger_hits,
                tag_hits=tag_hits
            ))

        best = max(scored, key=lambda m: m.score)
        if best.score < min_score:
            return None
        return best

    def _iter_candidate_skill_files(self) -> list[Path]:
        """枚举技能候选文件，避免把 references/docs 都当技能加载。"""
        candidates: list[Path] = []
        for md_file in sorted(self.skills_dir.rglob("*.md")):
            if not md_file.is_file():
                continue
            rel = md_file.relative_to(self.skills_dir)
            is_top_level_md = len(rel.parts) == 1
            is_skill_md = md_file.name.lower() == "skill.md"
            if is_top_level_md or is_skill_md:
                candidates.append(md_file)
        return candidates
    
    def generate_code(
        self,
        skill: SkillDefinition,
        params: dict[str, Any]
    ) -> str:
        """根据技能模板生成代码
        
        Args:
            skill: 技能定义
            params: 参数值
        
        Returns:
            生成的代码
        """
        if not skill.code_template:
            return ""
        
        code = skill.code_template
        
        # 替换参数
        for key, value in params.items():
            placeholder = f"${{{key}}}"
            code = code.replace(placeholder, str(value))
        
        return code


# === 全局单例 ===

_loader: SkillLoader | None = None


def get_skill_loader(skills_dir: Path | str | None = None) -> SkillLoader:
    """获取全局技能加载器"""
    global _loader

    if _loader is None:
        _loader = SkillLoader(skills_dir)
        return _loader

    if skills_dir is not None:
        requested = Path(skills_dir).resolve()
        current = _loader.skills_dir.resolve()
        if requested != current:
            _loader = SkillLoader(requested)
    
    return _loader


def reload_skills() -> int:
    """重新加载所有技能"""
    return get_skill_loader().reload()

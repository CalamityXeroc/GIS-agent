from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gis_cli.skills.loader import SkillLoader, parse_skill_md


def test_parse_skill_md_supports_external_alias_frontmatter(tmp_path):
    skill_file = tmp_path / "word_reader.md"
    content = """---
id: read_word_doc
summary: 读取 Word 文档并提取任务要求
keywords: [word, docx, 文档]
required: [file_path]
defaults:
  max_chars: 8000
template: |
  print(\"hello\")
---

# 读取文档技能
"""

    skill = parse_skill_md(content, source_file=skill_file)
    assert skill is not None
    assert skill.name == "read_word_doc"
    assert "读取 Word 文档" in skill.description
    assert "docx" in skill.tags
    assert "file_path" in skill.required_inputs
    assert "print(\"hello\")" in skill.code_template


def test_parse_skill_md_without_frontmatter_uses_filename_and_body(tmp_path):
    skill_file = tmp_path / "buffer_generic.md"
    content = """
# 通用缓冲

这是一个通用缓冲技能。

```python
import arcpy
print('buffer')
```
"""
    skill = parse_skill_md(content, source_file=skill_file)
    assert skill is not None
    assert skill.name == "buffer_generic"
    assert "通用缓冲技能" in skill.description
    assert "print('buffer')" in skill.code_template


def test_skill_loader_loads_markdown_recursively(tmp_path):
    skills_dir = tmp_path / "workspace" / "skills"
    nested_dir = skills_dir / "external" / "office"
    nested_dir.mkdir(parents=True, exist_ok=True)

    (nested_dir / "SKILL.md").write_text(
        """---
name: read_word_doc
description: 读取word
triggers: [word]
code_template: |
  set_result({"status": "ok"})
---
""",
        encoding="utf-8",
    )

    # 非候选文件：应被忽略
    (nested_dir / "README.md").write_text("# doc", encoding="utf-8")

    loader = SkillLoader(skills_dir=skills_dir)
    loaded = loader.load()
    assert loaded == 1
    assert loader.get("read_word_doc") is not None


def test_skill_loader_loads_top_level_md_and_ignores_nested_non_skill_md(tmp_path):
    skills_dir = tmp_path / "workspace" / "skills"
    nested = skills_dir / "pkg" / "references"
    nested.mkdir(parents=True, exist_ok=True)

    (skills_dir / "dissolve.md").write_text(
        """---
name: dissolve
description: dissolve features
triggers: [dissolve]
code_template: |
  import arcpy
  set_result({"ok": True})
---
""",
        encoding="utf-8",
    )
    (nested / "guide.md").write_text("# should be ignored", encoding="utf-8")

    loader = SkillLoader(skills_dir=skills_dir)
    loaded = loader.load()
    assert loaded == 1
    assert loader.get("dissolve") is not None


def test_parse_skill_md_marks_non_executable_for_reference_style_skill(tmp_path):
    skill_file = tmp_path / "pdf" / "SKILL.md"
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    content = """---
name: pdf
description: PDF guide
triggers: [pdf]
---

# PDF guide

```python
from pypdf import PdfReader
reader = PdfReader("doc.pdf")
print(len(reader.pages))
```
"""

    skill = parse_skill_md(content, source_file=skill_file)
    assert skill is not None
    assert skill.is_executable is False


def test_parse_skill_md_marks_executable_for_arcpy_or_set_result(tmp_path):
    skill_file = tmp_path / "dissolve.md"
    content = """---
name: dissolve
description: dissolve features
---

```python
import arcpy
set_result({"output": "x.shp"})
```
"""

    skill = parse_skill_md(content, source_file=skill_file)
    assert skill is not None
    assert skill.is_executable is True

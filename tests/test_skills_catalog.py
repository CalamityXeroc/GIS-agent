# -*- coding: utf-8 -*-
"""技能发现与注入的测试（Agent Skills 标准）。

覆盖两个真实需求：
1. 用户自己往 ``skills/`` 里加技能 → 自动被扫到、自动进 system prompt；
2. 说"写个 word 文档"时模型能知道有 docx 技能可用（描述必须常驻 prompt）。
"""

from __future__ import annotations

from pathlib import Path

from gis_cli.skills.catalog import (
    discover_skills,
    parse_skill_file,
    read_skill_text,
    skills_prompt_section,
)


def _write_skill(root: Path, name: str, description: str, body: str = "# 用我\n") -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "SKILL.md"
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{body}", encoding="utf-8")
    return path


def test_discovers_skill_md_in_subdirectory(tmp_path):
    _write_skill(tmp_path / "skills", "docx", "创建和编辑 Word 文档")
    found = discover_skills(tmp_path, include_global=False)
    names = [item.name for item in found]
    assert "docx" in names
    docx = next(item for item in found if item.name == "docx")
    assert docx.description == "创建和编辑 Word 文档"
    assert docx.root == tmp_path / "skills" / "docx"


def test_discovers_bare_markdown_in_skills_root(tmp_path):
    """裸 .md 也要认——本项目原有 5 个 GIS 技能就是这种写法。"""
    root = tmp_path / "skills"
    root.mkdir(parents=True)
    (root / "buffer_analysis.md").write_text(
        "---\nname: buffer_analysis\ndescription: 缓冲区分析\n---\n\n正文\n", encoding="utf-8"
    )
    found = discover_skills(tmp_path, include_global=False)
    assert [item.name for item in found] == ["buffer_analysis"]


def test_skill_without_description_is_skipped(tmp_path):
    """规范要求 description 必填：没有它模型无法判断何时使用。"""
    root = tmp_path / "skills"
    root.mkdir(parents=True)
    path = root / "bad" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text("---\nname: bad\n---\n\n正文\n", encoding="utf-8")
    assert discover_skills(tmp_path, include_global=False) == []


def test_name_falls_back_to_directory(tmp_path):
    """允许缺 name（用目录名兜底），但 description 必须在。"""
    path = tmp_path / "skills" / "my-skill" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text("---\ndescription: 干点活\n---\n\n正文\n", encoding="utf-8")
    info = parse_skill_file(path)
    assert info is not None and info.name == "my-skill"


def test_folded_and_quoted_description(tmp_path):
    """frontmatter 里 description 写成折叠块或多行引号都要能读。"""
    root = tmp_path / "skills"
    root.mkdir(parents=True)
    path = root / "folded" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\nname: folded\ndescription: >-\n  第一行\n  第二行\n---\n\n正文\n",
        encoding="utf-8",
    )
    info = parse_skill_file(path)
    assert info is not None
    assert "第一行" in info.description and "第二行" in info.description


def test_extra_roots_are_scanned(tmp_path):
    """支持挂载别的 harness 目录（用户的技能可能在 ~/.claude/skills）。"""
    external = tmp_path / "external"
    _write_skill(external, "from-claude", "来自 Claude 的技能")
    found = discover_skills(tmp_path, extra_roots=[(external, "外部")], include_global=False)
    assert [item.name for item in found] == ["from-claude"]
    assert found[0].source == "外部"


def test_duplicate_names_keep_first(tmp_path):
    primary = tmp_path / "skills"
    external = tmp_path / "external"
    _write_skill(primary, "dup", "先找到的")
    _write_skill(external, "dup", "后找到的")
    found = discover_skills(tmp_path, extra_roots=[(primary, "主"), (external, "外")], include_global=False)
    assert len([item for item in found if item.name == "dup"]) == 1
    assert found[0].description == "先找到的"


def test_prompt_section_includes_name_and_description(tmp_path):
    """关键需求：描述必须进 system prompt，模型才知道"写 word"该用 docx。"""
    _write_skill(tmp_path / "skills", "docx", "创建、读取、编辑 Word 文档（.docx）")
    found = discover_skills(tmp_path, include_global=False)
    section = skills_prompt_section(found)
    assert "docx" in section
    assert "Word 文档" in section
    assert "run_skill" in section  # 告诉模型怎么读全文


def test_prompt_section_prioritizes_document_skills(tmp_path):
    """文档类技能必须排在前面：否则被上百个无关技能挤出列表就触发不了。"""
    root = tmp_path / "skills"
    for index in range(30):
        _write_skill(root, f"aaa-{index}", "无关技能")
    _write_skill(root, "docx", "处理 Word 文档")
    _write_skill(root, "pdf", "处理 PDF 文档")
    section = skills_prompt_section(discover_skills(tmp_path, include_global=False), max_skills=5)
    head = section.split("## 可用技能")[1][:600]
    assert "docx" in head and "pdf" in head
    assert "未列出" in section  # 超出的要有折叠提示


def test_prompt_section_empty_when_no_skills(tmp_path):
    assert skills_prompt_section([]) == ""
    assert discover_skills(tmp_path, include_global=False) == []


def test_read_skill_text_truncates(tmp_path):
    _write_skill(tmp_path / "skills", "big", "大技能", body="x" * 5000)
    info = discover_skills(tmp_path, include_global=False)[0]
    text = read_skill_text(info, max_chars=200)
    assert "已截断" in text and len(text) < 500


def test_engine_prompt_includes_skills(tmp_path):
    """引擎的 system prompt 要真的带上技能段（不是只有工具里有）。"""
    _write_skill(tmp_path / "skills", "docx", "创建 Word 文档")
    from gis_cli.engine.loop import AgentLoop

    holder = type("H", (), {"workspace": tmp_path, "_skills_section": AgentLoop._skills_section})()
    section = holder._skills_section()
    assert "docx" in section and "创建 Word 文档" in section


def test_engine_prompt_survives_broken_skill_dir(tmp_path, monkeypatch):
    """技能扫描出错不能拖垮 system prompt（只是少一段）。"""
    from gis_cli.engine.loop import AgentLoop
    import gis_cli.skills.catalog as catalog

    def _boom(*_a, **_k):
        raise RuntimeError("扫崩了")

    monkeypatch.setattr(catalog, "discover_skills", _boom)
    holder = type("H", (), {"workspace": tmp_path, "_skills_section": AgentLoop._skills_section})()
    assert holder._skills_section() == ""

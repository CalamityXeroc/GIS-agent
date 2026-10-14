from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gis_cli.agent.planner import AgentPlanner
from gis_cli.skills.loader import SkillLoader, get_skill_loader


def test_planner_ignores_non_executable_skill_match(tmp_path):
    skills_dir = tmp_path / "workspace" / "skills"
    skill_dir = skills_dir / "pdf"
    skill_dir.mkdir(parents=True, exist_ok=True)

    (skill_dir / "SKILL.md").write_text(
        """---
name: pdf
summary: PDF 文档处理
keywords: [pdf, 文档]
---

```python
from pypdf import PdfReader
reader = PdfReader("doc.pdf")
print(len(reader.pages))
```
""",
        encoding="utf-8",
    )

    loader = SkillLoader(skills_dir=skills_dir)
    loader.load()

    planner = AgentPlanner()
    # monkey-patch global loader path by replacing module-level singleton behavior via direct call path:
    # planner imports get_skill_loader() internally; here we assert behavior through explicit check on loader API.
    match = loader.find_best_match_with_score("请处理 pdf 文档", min_score=1, only_executable=True)
    assert match is None


def test_planner_can_match_executable_skill_with_only_executable_filter(tmp_path):
    skills_dir = tmp_path / "workspace" / "skills"
    (skills_dir / "dissolve_features.md").parent.mkdir(parents=True, exist_ok=True)
    (skills_dir / "dissolve_features.md").write_text(
        """---
name: dissolve_features
description: 按字段融合要素
triggers: [融合, dissolve]
---

```python
import arcpy
in_features = r"${input_layer}"
out_features = r"${output_path}"
arcpy.Dissolve_management(in_features, out_features)
set_result({"output": out_features})
```
""",
        encoding="utf-8",
    )

    loader = SkillLoader(skills_dir=skills_dir)
    loader.load()

    match = loader.find_best_match_with_score("请按字段融合图层", min_score=1, only_executable=True)
    assert match is not None
    assert match.skill.name == "dissolve_features"


def test_planner_matches_executable_skill_for_trigger_phrase(tmp_path):
    skills_dir = tmp_path / "workspace" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "dissolve_features.md").write_text(
        """---
name: dissolve_features
description: 按字段融合要素
triggers: [融合, dissolve]
---

```python
import arcpy
in_features = r"${input_layer}"
out_features = r"${output_path}"
arcpy.Dissolve_management(in_features, out_features)
set_result({"output": out_features})
```
""",
        encoding="utf-8",
    )

    # Rebind global loader to this test workspace.
    get_skill_loader(skills_dir).reload()

    planner = AgentPlanner()
    plan = planner._try_match_custom_skill(  # noqa: SLF001 - intentional internal contract test
        "请帮我做融合",
        {"input_folder": "./workspace/input", "output_folder": "./workspace/output"},
    )

    assert plan is not None
    assert plan.metadata is not None
    assert plan.metadata.get("skill") == "dissolve_features"

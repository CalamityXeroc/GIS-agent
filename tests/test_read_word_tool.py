from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gis_cli.core import ToolContext
from gis_cli.tools.read_word import ReadWordTool


def test_read_word_tool_dry_run_success(tmp_path):
    doc_path = tmp_path / "task.docx"
    doc_path.touch()

    tool = ReadWordTool()
    context = ToolContext(dry_run=True, working_directory=str(tmp_path), arcpy_available=False)
    result = tool.execute({"file_path": str(doc_path), "max_chars": 1000}, context)

    assert result.success is True
    assert result.data is not None
    assert "DRY RUN" in result.data.text


def test_read_word_tool_rejects_non_docx(tmp_path):
    txt_path = tmp_path / "task.txt"
    txt_path.write_text("hello", encoding="utf-8")

    tool = ReadWordTool()
    context = ToolContext(dry_run=True, working_directory=str(tmp_path), arcpy_available=False)
    result = tool.execute({"file_path": str(txt_path), "max_chars": 1000}, context)

    assert result.success is False
    assert "仅支持 .docx" in (result.error or "")

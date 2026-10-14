from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gis_cli.core import ToolContext
from gis_cli.tools.read_pdf import ReadPdfTool


def test_read_pdf_tool_dry_run_success(tmp_path):
    pdf_path = tmp_path / "task.pdf"
    pdf_path.touch()

    tool = ReadPdfTool()
    context = ToolContext(dry_run=True, working_directory=str(tmp_path), arcpy_available=False)
    result = tool.execute({"file_path": str(pdf_path), "max_chars": 1000}, context)

    assert result.success is True
    assert result.data is not None
    assert "DRY RUN" in result.data.text


def test_read_pdf_tool_rejects_non_pdf(tmp_path):
    txt_path = tmp_path / "task.txt"
    txt_path.write_text("hello", encoding="utf-8")

    tool = ReadPdfTool()
    context = ToolContext(dry_run=True, working_directory=str(tmp_path), arcpy_available=False)
    result = tool.execute({"file_path": str(txt_path), "max_chars": 1000}, context)

    assert result.success is False
    assert "仅支持 .pdf" in (result.error or "")

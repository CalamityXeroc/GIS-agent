"""ReadPdfTool - Read content from PDF documents."""

from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field

from ..core import (
    PermissionResult,
    Tool,
    ToolCategory,
    ToolContext,
    ToolResult,
    ValidationResult,
    register_tool,
)


class ReadPdfInput(BaseModel):
    """Input schema for ReadPdfTool."""

    file_path: str = Field(..., description="PDF 文件路径（.pdf）")
    max_chars: int = Field(default=12000, ge=200, le=200000, description="最多提取字符数")


class ReadPdfOutput(BaseModel):
    """Output schema for ReadPdfTool."""

    source_path: str
    text: str
    page_count: int
    char_count: int
    truncated: bool = False


@register_tool
class ReadPdfTool(Tool[ReadPdfInput, ReadPdfOutput]):
    """Read and extract text from PDF documents."""

    name = "read_pdf"
    description = "读取 PDF 文档（.pdf）并提取文本内容"
    category = ToolCategory.FILE_OPERATION
    search_hint = "read pdf document parse extract text"
    input_model = ReadPdfInput

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def validate_input(self, input_data: ReadPdfInput) -> ValidationResult:
        path = Path(input_data.file_path)
        if not path.exists():
            return ValidationResult.failure(f"文件不存在: {input_data.file_path}", error_code=1)
        if not path.is_file():
            return ValidationResult.failure(f"不是有效文件: {input_data.file_path}", error_code=2)
        if path.suffix.lower() != ".pdf":
            return ValidationResult.failure("目前仅支持 .pdf 文件", error_code=3)
        return ValidationResult.success()

    def check_permissions(self, input_data: ReadPdfInput, context: ToolContext) -> PermissionResult:
        return PermissionResult.allow()

    def call(self, input_data: ReadPdfInput, context: ToolContext) -> ToolResult[ReadPdfOutput]:
        path = Path(input_data.file_path).resolve()

        if context.dry_run:
            preview = ReadPdfOutput(
                source_path=str(path),
                text=f"[DRY RUN] 将读取文档: {path}",
                page_count=0,
                char_count=0,
                truncated=False,
            )
            return ToolResult.ok(preview, outputs=[])

        try:
            from pypdf import PdfReader
        except Exception:
            return ToolResult.fail(
                "缺少依赖 pypdf，请先安装：pip install pypdf",
                "missing_dependency",
            )

        try:
            reader = PdfReader(str(path))
            pages_text: list[str] = []
            for page in reader.pages:
                extracted = page.extract_text() or ""
                extracted = extracted.strip()
                if extracted:
                    pages_text.append(extracted)
        except Exception as e:
            return ToolResult.fail(f"读取 PDF 失败: {e}", "read_error")

        text = "\n\n".join(pages_text)
        truncated = False
        if len(text) > input_data.max_chars:
            text = text[: input_data.max_chars]
            truncated = True

        output = ReadPdfOutput(
            source_path=str(path),
            text=text,
            page_count=len(reader.pages),
            char_count=len(text),
            truncated=truncated,
        )
        return ToolResult.ok(output, outputs=[])

"""ReadWordTool - Read content from Word (.docx) documents."""

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


class ReadWordInput(BaseModel):
    """Input schema for ReadWordTool."""

    file_path: str = Field(..., description="Word 文件路径（.docx）")
    max_chars: int = Field(default=12000, ge=200, le=200000, description="最多提取字符数")


class ReadWordOutput(BaseModel):
    """Output schema for ReadWordTool."""

    source_path: str
    text: str
    paragraph_count: int
    char_count: int
    truncated: bool = False


@register_tool
class ReadWordTool(Tool[ReadWordInput, ReadWordOutput]):
    """Read and extract text from Word documents."""

    name = "read_word"
    description = "读取 Word 文档（.docx）并提取文本内容"
    category = ToolCategory.FILE_OPERATION
    search_hint = "read word docx document parse extract text"
    input_model = ReadWordInput

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def validate_input(self, input_data: ReadWordInput) -> ValidationResult:
        path = Path(input_data.file_path)
        if not path.exists():
            return ValidationResult.failure(f"文件不存在: {input_data.file_path}", error_code=1)
        if not path.is_file():
            return ValidationResult.failure(f"不是有效文件: {input_data.file_path}", error_code=2)
        if path.suffix.lower() != ".docx":
            return ValidationResult.failure("目前仅支持 .docx 文件", error_code=3)
        return ValidationResult.success()

    def check_permissions(self, input_data: ReadWordInput, context: ToolContext) -> PermissionResult:
        return PermissionResult.allow()

    def call(self, input_data: ReadWordInput, context: ToolContext) -> ToolResult[ReadWordOutput]:
        path = Path(input_data.file_path).resolve()

        if context.dry_run:
            preview = ReadWordOutput(
                source_path=str(path),
                text=f"[DRY RUN] 将读取文档: {path}",
                paragraph_count=0,
                char_count=0,
                truncated=False,
            )
            return ToolResult.ok(preview, outputs=[])

        try:
            from docx import Document
        except Exception:
            return ToolResult.fail(
                "缺少依赖 python-docx，请先安装：pip install python-docx",
                "missing_dependency",
            )

        try:
            doc = Document(str(path))
        except Exception as e:
            return ToolResult.fail(f"读取 Word 失败: {e}", "read_error")

        paragraphs = [p.text.strip() for p in doc.paragraphs if isinstance(p.text, str) and p.text.strip()]
        text = "\n".join(paragraphs)
        truncated = False
        if len(text) > input_data.max_chars:
            text = text[: input_data.max_chars]
            truncated = True

        output = ReadWordOutput(
            source_path=str(path),
            text=text,
            paragraph_count=len(paragraphs),
            char_count=len(text),
            truncated=truncated,
        )
        return ToolResult.ok(output, outputs=[])

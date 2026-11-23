from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocumentParseResult:
    title: str
    normalized_prompt: str
    constraints: list[str] = field(default_factory=list)
    scoring_items: list[str] = field(default_factory=list)
    extracted_actions: list[str] = field(default_factory=list)


class DocumentTaskParser:
    def parse_text(self, text: str) -> DocumentParseResult:
        lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()]
        if not lines:
            raise ValueError("Document text is empty.")

        title = lines[0][:120]
        constraints = [
            ln for ln in lines if any(k in ln for k in ("投影", "坐标", "比例尺", "命名", "导出"))
        ]
        scoring_items = [ln for ln in lines if ("分" in ln and ("(" in ln or "）" in ln or ")" in ln))]

        action_lines = [
            ln
            for ln in lines
            if any(
                key in ln
                for key in (
                    "创建",
                    "整合",
                    "合并",
                    "转换",
                    "提取",
                    "标注",
                    "检查",
                    "修复",
                    "导出",
                    "制图",
                    "分析",
                )
            )
        ]
        extracted_actions = action_lines[:20]

        prompt_parts: list[str] = []
        if extracted_actions:
            prompt_parts.extend(extracted_actions[:8])
        if constraints:
            prompt_parts.extend(constraints[:4])

        # Preserve order while removing duplicated lines from overlapping extraction rules.
        deduped_parts: list[str] = []
        seen: set[str] = set()
        for part in prompt_parts:
            if part not in seen:
                seen.add(part)
                deduped_parts.append(part)

        normalized_prompt = "；".join(deduped_parts)[:800]
        if not normalized_prompt:
            normalized_prompt = "；".join(lines[:8])[:800]

        return DocumentParseResult(
            title=title,
            normalized_prompt=normalized_prompt,
            constraints=constraints[:30],
            scoring_items=scoring_items[:30],
            extracted_actions=extracted_actions,
        )

    def parse_file(self, file_path: str) -> DocumentParseResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        text = self._read_text(path)
        return self.parse_text(text)

    def _read_text(self, path: Path) -> str:
        for encoding in ("utf-8", "gbk", "utf-16"):
            try:
                content = path.read_text(encoding=encoding)
                if content:
                    return content
            except UnicodeDecodeError:
                continue

        # Fallback to binary decode replacement for mixed encodings.
        return path.read_bytes().decode("utf-8", errors="replace")

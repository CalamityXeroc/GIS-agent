# -*- coding: utf-8 -*-
"""Extract text from competition task documents for benchmark authoring."""
import sys
from pathlib import Path

base = Path(r"E:\BST_lab\GIS_CLI\workspace\input\13第十三届全国大学生GIS应用技能大赛（兰州交通大学）")
targets = [
    base / "AM" / "AM" / "上午解题文档.docx",
    base / "AM" / "AM" / "B上午_比赛要求.pdf",
    base / "PM" / "PM" / "下午解题文档-小火苗GIS.pdf",
]
out_dir = Path(r"E:\BST_lab\GIS_CLI\benchmark\specs")
out_dir.mkdir(parents=True, exist_ok=True)

for target in targets:
    if not target.exists():
        print("MISSING:", target)
        continue
    print("=" * 80)
    print("FILE:", target.name)
    text = ""
    if target.suffix.lower() == ".docx":
        import docx
        doc = docx.Document(str(target))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        text = "\n".join(parts)
    elif target.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(target))
        pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            pages.append(f"\n--- page {i+1} ---\n{page_text}")
        text = "\n".join(pages)
    out = out_dir / (target.stem + ".txt")
    out.write_text(text, encoding="utf-8")
    print(f"chars={len(text)} -> {out}")
    print(text[:3000])

# -*- coding: utf-8 -*-
"""让 map_layout_ok 断言自动加载同目录的 .spec.json。

实测（14 届四a-2 迁移图）：设计明确决定"图例放图框外"（spec.legend.outside=True，
版面体检对此有豁免），但断言没把 spec 传进去 → 体检按"图例必须在图框内"判定，
误报"图例压出图框：右越界 33.0mm"。
"""
from pathlib import Path
import ast

p = Path("src/gis_cli/recipes/store.py")
s = p.read_text(encoding="utf-8")

old = '''        if kind == "map_layout_ok":
            # 版面体检：图名居中/图例不压数据/比例尺整数刻度/元素不越界（制图交付的硬指标）
            try:
                from ..cartography import qc as _qc

                verdict = _qc.check_layout(
                    target,
                    spec=raw.get("spec") or None,
                    image_path=str(raw.get("image") or ""),
                    min_font_pt=float(raw.get("min_font_pt") or 7.0),
                )'''
new = '''        if kind == "map_layout_ok":
            # 版面体检：图名居中/图例不压数据/比例尺整数刻度/元素不越界（制图交付的硬指标）
            try:
                from ..cartography import qc as _qc

                verdict = _qc.check_layout(
                    target,
                    spec=raw.get("spec") or self._load_sibling_spec(target, raw.get("image")),
                    image_path=str(raw.get("image") or ""),
                    min_font_pt=float(raw.get("min_font_pt") or 7.0),
                )'''
assert s.count(old) == 1, f"匹配 {s.count(old)} 处"
s = s.replace(old, new, 1)

helper = '''    @staticmethod
    def _load_sibling_spec(target: str, image: Any = "") -> dict[str, Any] | None:
        """加载与工程/成图同名的 ``.spec.json``（设计规格）。

        版面体检里有些判据依赖设计决策（例如"图例刻意放在图框外"是允许的），
        不传 spec 会把刻意设计误报成越界（14 届四a-2 实测）。
        """
        import json as _json

        candidates: list[Path] = []
        for raw_path in (target, image):
            if not raw_path:
                continue
            path = Path(str(raw_path))
            candidates.append(path.with_suffix(".spec.json"))
            candidates.append(path.with_suffix(path.suffix + ".spec.json"))
        for candidate in candidates:
            try:
                if candidate.exists():
                    return _json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
        return None

'''
anchor = "    def _field_counts_all(self, path: str, field: str) -> tuple[dict[str, int] | None, str]:"
assert s.count(anchor) == 1
s = s.replace(anchor, helper + anchor, 1)

p.write_text(s, encoding="utf-8")
ast.parse(s)
print("store.py 已更新：map_layout_ok 自动加载同名 .spec.json")

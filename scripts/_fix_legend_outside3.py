# -*- coding: utf-8 -*-
"""统一 legend_outside_frame 与 legend_position_sane 的口径。

原判据要求"图例必须在地图框内"，与 legend_position_sane 的"满覆盖数据时图例放框外
是预期行为"直接冲突，导致刻意设计被误报（14 届四a-2 迁移图：右越界 33.0mm）。

新口径（只针对"图例压线"这一真实缺陷）：
- 完全在图框内且留够边距 → 通过
- 完全在图框外（无论左右上下，属设计选择）→ 通过
- 与图框边界**相交/压线**（半进半出、压住框线）→ 失败
"""
from pathlib import Path
import ast

p = Path("src/gis_cli/cartography/qc.py")
s = p.read_text(encoding="utf-8")

start = s.index("        allowed_outside_but_inside_neatline = False")
end = s.index("    # 3.8) 单层边框", start)
old_block = s[start:end]

new_block = '''        # 完全在图框内（留边距）
        fully_inside = (
            lx >= fx + margin - 0.6 and ly >= fy + margin - 0.6
            and lx + lw <= fx + fw - margin + 0.6 and ly + lh <= fy + fh - margin + 0.6
        )
        # 与图框边界相交（压线）——这才是真正要报的版面缺陷
        crosses_frame_edge = not (
            lx + lw <= fx + 0.6 or lx >= fx + fw - 0.6
            or ly + lh <= fy + 0.6 or ly >= fy + fh - 0.6
        )
        if fully_inside:
            checks.append({
                "name": "legend_outside_frame", "ok": True,
                "detail": "图例完全在图框内（留 %.1fmm 边距）" % margin,
            })
        elif not crosses_frame_edge:
            checks.append({
                "name": "legend_outside_frame", "ok": True,
                "detail": "图例整体在地图框外（设计选择：图面较密时避免压数据）；是否越出版面由 element_outside_neatline 判定",
            })
        else:
            checks.append({
                "name": "legend_outside_frame", "ok": False,
                "detail": "图例与图框压线（半进半出）：" + "、".join(problems) if problems else "图例与图框压线",
            })

'''
s = s[:start] + new_block + s[end:]
p.write_text(s, encoding="utf-8")
ast.parse(s)
print("legend_outside_frame 口径已统一")

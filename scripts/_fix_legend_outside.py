# -*- coding: utf-8 -*-
"""qc.legend_outside_frame：图例在图框外但仍在图廓（版面边框）内时，判为允许。

实测（14 届四a-2 迁移图）：设计刻意把图例放到图框右侧（图面很密时避免压数据），
原判据只按"必须在地图框内"比较，误报"图例压出图框：右越界 33.0mm"。
真正该失败的是"图例越出版面边框"，那已由 element_outside_neatline 覆盖。
"""
from pathlib import Path
import ast

p = Path("src/gis_cli/cartography/qc.py")
s = p.read_text(encoding="utf-8")

old = '''        checks.append({
            "name": "legend_outside_frame", "ok": not problems,
            "detail": ("图例完全在图框内（留 %.1fmm 边距）" % margin) if not problems
                      else "图例压出图框：" + "、".join(problems),
        })'''
new = '''        if problems and neatline:
            # 图框外但在图廓内：属于"图面很密时把图例放到框外"的设计选择，判为允许；
            # 真越出版面边框的情况由 element_outside_neatline 负责报错。
            nx, ny, nw, nh = neatline
            outside_neatline = (
                lx < nx - 0.6 or ly < ny - 0.6
                or lx + lw > nx + nw + 0.6 or ly + lh > ny + nh + 0.6
            )
            if not outside_neatline:
                checks.append({
                    "name": "legend_outside_frame", "ok": True,
                    "detail": "图例在地图框外但仍在图廓内（设计选择：图面较密时避免压数据）",
                })
                problems = []
        checks.append({
            "name": "legend_outside_frame", "ok": not problems,
            "detail": ("图例完全在图框内（留 %.1fmm 边距）" % margin) if not problems
                      else "图例压出图框：" + "、".join(problems),
        })'''
assert s.count(old) == 1, f"匹配 {s.count(old)} 处"
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
ast.parse(s)
print("qc.py legend_outside_frame 已放行'图框外但图廓内'")

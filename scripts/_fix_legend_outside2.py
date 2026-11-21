# -*- coding: utf-8 -*-
"""修正上一处补丁：放行分支不应再追加第二条 legend_outside_frame 记录。"""
from pathlib import Path
import ast

p = Path("src/gis_cli/cartography/qc.py")
s = p.read_text(encoding="utf-8")

old = '''        if problems and neatline:
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

new = '''        allowed_outside_but_inside_neatline = False
        if problems and neatline:
            # 图框外但在图廓内：属于"图面很密时把图例放到框外"的设计选择，判为允许；
            # 真越出版面边框的情况由 element_outside_neatline 负责报错。
            nx, ny, nw, nh = neatline
            allowed_outside_but_inside_neatline = not (
                lx < nx - 0.6 or ly < ny - 0.6
                or lx + lw > nx + nw + 0.6 or ly + lh > ny + nh + 0.6
            )
        if allowed_outside_but_inside_neatline:
            checks.append({
                "name": "legend_outside_frame", "ok": True,
                "detail": "图例在地图框外但仍在图廓内（设计选择：图面较密时避免压数据）",
            })
        else:
            checks.append({
                "name": "legend_outside_frame", "ok": not problems,
                "detail": ("图例完全在图框内（留 %.1fmm 边距）" % margin) if not problems
                          else "图例压出图框：" + "、".join(problems),
            })'''

assert s.count(old) == 1, f"匹配 {s.count(old)} 处"
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
ast.parse(s)
print("已修正：放行分支只记一条")

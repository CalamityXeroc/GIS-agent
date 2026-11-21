# -*- coding: utf-8 -*-
"""探查 ArcGIS Pro 唯一值渲染器（UniqueValueRenderer）的真实属性名。"""
import sys, io, os, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")
from pathlib import Path
from gis_cli.runtime.code_runner import CodeRunner

REPO = Path(r"E:\BST_lab\GIS_CLI")
ws = Path(sorted(glob.glob(str(REPO / "benchmark/runs/*/pm_task2b_service_value_rank/workspace")))[-1])
exp = Path(r"C:\Users\cheng\AppData\Local\Temp\sym_probe")
exp.mkdir(parents=True, exist_ok=True)
r = CodeRunner(workspace=ws, state_dir=ws / ".gis_agent" / "k_probe")

CODE = """
import arcpy, os, glob, shutil
from arcpy import mp
gdb = r'__GDB__'
layer_path = os.path.join(gdb, 'community_eval')
target = r'__APRX__'
if os.path.exists(target):
    os.remove(target)
install = arcpy.GetInstallInfo().get('InstallDir', '')
cand = os.path.join(install, 'Resources', 'ArcToolBox', 'Services', 'routingservices', 'data', 'Blank.aprx')
if not os.path.exists(cand):
    found = glob.glob(os.path.join(install, 'Resources', '**', 'Blank.aprx'), recursive=True)
    cand = found[0]
shutil.copy2(cand, target)
aprx = mp.ArcGISProject(target)
m = aprx.listMaps()[0]
lyr = m.addDataFromPath(layer_path)
sym = lyr.symbology
sym.updateRenderer('UniqueValueRenderer')
sym.renderer.fields = ['类别']
grp = sym.renderer.groups[0]
print('renderer 属性:', [a for a in dir(sym.renderer) if not a.startswith('_')])
print('group 属性:', [a for a in dir(grp) if not a.startswith('_')])
items = getattr(grp, 'items', None)
print('group.items 类型:', type(items).__name__ if items is not None else None,
      '| 数量:', len(items) if items is not None else None)
if items:
    it = items[0]
    print('item 属性:', [a for a in dir(it) if not a.startswith('_')])
    print('item.values:', list(it.values), '| label:', getattr(it, 'label', None))
    try:
        it.symbol.color = {'RGB': [31, 119, 180, 100]}
        print('设置颜色成功; RGB =', it.symbol.color.RGB)
    except Exception as exc:
        print('设置颜色失败:', str(exc)[:120])
    try:
        print('改 label:', end=' ')
        it.label = str(list(it.values)[0])
        print('OK ->', it.label)
    except Exception as exc:
        print('失败:', str(exc)[:100])
sym.renderer.fields = ['类别']
lyr.symbology = sym
try:
    lyr.symbology = sym
    print('回写 symbology OK')
except Exception as exc:
    print('回写失败:', str(exc)[:120])
print('类别渲染后 group 数:', len(lyr.symbology.renderer.groups),
      '| items 数:', len(lyr.symbology.renderer.groups[0].items))
labels = [str(list(i.values)[0]) for i in lyr.symbology.renderer.groups[0].items]
print('类别标签:', labels)
aprx.save()
print('aprx 已保存')
set_result({'ok': True, 'labels': labels})
"""
res = r.run(CODE.replace("__GDB__", str(ws / "output" / "result_data.gdb")).replace("__APRX__", str(exp / "probe.aprx")),
            timeout=600, workspace=str(ws))
print("ok:", res.ok, "| err:", str((res.error or {}).get("message", ""))[:200])
print((res.stdout or "")[-2000:])
r.shutdown()
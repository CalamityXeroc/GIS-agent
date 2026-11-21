# -*- coding: utf-8 -*-
"""Manual smoke test for kernel + catalog against competition data."""
import sys, time, json
from pathlib import Path

sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")
from gis_cli.runtime.kernel import ArcpyKernel
from gis_cli.datacatalog.store import Catalog
from gis_cli.datacatalog.scanner import discover, refresh_catalog
from gis_cli.datacatalog.summary import build_digest

base = Path(r"E:\BST_lab\GIS_CLI\workspace\input\13第十三届全国大学生GIS应用技能大赛（兰州交通大学）\AM\AM\1-Process")
items = discover([base])
print("discovered:", [(d.kind, Path(d.path).name) for d in items])

k = ArcpyKernel(state_dir=r"E:\BST_lab\GIS_CLI\workspace\.gis_agent\kernel_test")
k.start()
cat = Catalog(r"E:\BST_lab\GIS_CLI\workspace\.gis_agent\catalog_test.db")
t0 = time.time()
stats = refresh_catalog(cat, [base], k)
print("refresh:", stats, f"{time.time()-t0:.1f}s")
print("---- DIGEST ----")
print(build_digest(cat, max_datasets=20, with_samples=False)[:4000])
k.shutdown()
cat.close()

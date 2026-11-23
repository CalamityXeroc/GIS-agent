# -*- coding: utf-8 -*-
"""Verify catalog refresh finds data for a given workspace root."""
import sys, time
from pathlib import Path

sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")
from gis_cli.runtime.code_runner import CodeRunner
from gis_cli.datacatalog.store import Catalog
from gis_cli.datacatalog.scanner import refresh_catalog
from gis_cli.datacatalog.summary import build_digest
from gis_cli.engine.runner import _candidate_data_roots

ws = Path(sys.argv[1] if len(sys.argv) > 1 else r"E:\BST_lab\GIS_CLI")
roots = _candidate_data_roots(ws)
print("roots:", roots)
runner = CodeRunner(workspace=ws, state_dir=ws / ".gis_agent" / "kernel")
cat = Catalog(ws / ".gis_agent" / "catalog.db")
t0 = time.time()
stats = refresh_catalog(cat, roots, runner, force=False)
print("stats:", stats, f"{time.time()-t0:.1f}s")
print("counts:", cat.counts())
print(build_digest(cat, max_datasets=6, max_fields=8)[:2500])
runner.shutdown()
cat.close()

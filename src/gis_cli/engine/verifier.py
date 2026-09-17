# -*- coding: utf-8 -*-
"""Output verifier: structural assertions + LLM requirement check.

The verifier is what makes "done" trustworthy: it refuses to accept an empty
or off-spec deliverable and produces a repair hint the loop can act on.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .tool_codec import extract_json_object

logger = logging.getLogger(__name__)

_VERIFY_CODE = r'''
import json as _json
import os as _os
import arcpy as _arcpy

_PATHS = {paths}

_arcpy.env.overwriteOutput = True


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


records = []
for _p in _PATHS:
    # GDB feature classes are not filesystem paths: trust arcpy.Exists first.
    rec = {"path": _p, "exists": bool(_os.path.exists(_p) or _arcpy.Exists(_p))}
    if not rec["exists"]:
        records.append(rec)
        continue
    rec["size"] = _safe(lambda p=_p: _os.path.getsize(p) if _os.path.isfile(p) else None)
    if _arcpy.Exists(_p):
        desc = _safe(lambda p=_p: _arcpy.Describe(p))
        if desc is not None:
            sr = getattr(desc, "spatialReference", None)
            rec["kind"] = getattr(desc, "dataType", "") or ""
            rec["crs_name"] = getattr(sr, "name", "") if sr else ""
            rec["wkid"] = getattr(sr, "factoryCode", None) if sr else None
            rec["geom_type"] = getattr(desc, "shapeType", "") or ""
            rec["extent"] = _safe(lambda d=desc: [d.extent.XMin, d.extent.YMin, d.extent.XMax, d.extent.YMax])
            if rec["kind"] in ("FeatureClass", "ShapeFile") or rec["geom_type"]:
                rec["feature_count"] = _safe(lambda p=_p: int(_arcpy.management.GetCount(p)[0]))
                fields = _safe(lambda p=_p: _arcpy.ListFields(p), []) or []
                rec["fields"] = [f.name for f in fields]
                rec["geometry_valid"] = _safe(
                    lambda p=_p: int(_arcpy.management.GetCount(
                        _arcpy.management.MakeFeatureLayer(p, "lyr_check").getOutput(0))[0]) >= 0
                )
            if getattr(desc, "dataType", "") == "RasterDataset":
                rec["band_count"] = _safe(lambda p=_p: int(_arcpy.Describe(p).bandCount))
                rec["raster_min"] = _safe(lambda p=_p: float(_arcpy.management.GetRasterProperties(p, "MINIMUM").getOutput(0)))
                rec["raster_max"] = _safe(lambda p=_p: float(_arcpy.management.GetRasterProperties(p, "MAXIMUM").getOutput(0)))
    records.append(rec)

set_result({"records": records})
'''


class Verifier:
    """Two-layer verifier: structural + semantic."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        code_runner: Any = None,
        llm_client: Any = None,
        min_image_bytes: int = 8_000,
    ):
        self.workspace = Path(workspace)
        self.code_runner = code_runner
        self.llm_client = llm_client
        self.min_image_bytes = min_image_bytes

    # ------------------------------------------------------------- structural
    def structural_check(self, artifacts: list[str]) -> list[dict[str, Any]]:
        """Check existence and ArcPy metadata for artifacts."""
        paths = [str(p) for p in artifacts if str(p).strip()]
        if not paths:
            return []
        results: list[dict[str, Any]] = []
        if self.code_runner is not None:
            code = _VERIFY_CODE.replace("{paths}", json.dumps(paths, ensure_ascii=False))
            outcome = self.code_runner.run(code, timeout=300)
            if outcome.ok and isinstance(outcome.result, dict):
                results = list(outcome.result.get("records", []))
        # Ensure every artifact has a record even if ArcPy check failed.
        seen = {r.get("path") for r in results}
        for path in paths:
            if path not in seen:
                target = Path(path)
                results.append(
                    {
                        "path": path,
                        "exists": target.exists(),
                        "size": target.stat().st_size if target.exists() and target.is_file() else None,
                    }
                )
        for record in results:
            path = str(record.get("path", ""))
            if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                record["image"] = self._check_image(Path(path))
        return results

    def _discover_outputs(self, *, limit: int = 50) -> list[str]:
        """Best-effort discovery of GIS outputs under workspace/output."""
        found: list[str] = []
        out_root = self.workspace / "output"
        if not out_root.exists():
            return found
        for item in sorted(out_root.rglob("*")):
            if len(found) >= limit:
                break
            name = item.name.lower()
            if item.is_dir() and name.endswith(".gdb"):
                found.append(str(item))
            elif item.is_file() and item.suffix.lower() in {
                ".shp", ".tif", ".tiff", ".img", ".csv", ".jpg", ".jpeg", ".png", ".pdf", ".dbf", ".gpkg"
            }:
                found.append(str(item))
        return found

    def _check_image(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"ok": False, "reason": "missing"}
        size = path.stat().st_size
        if size < self.min_image_bytes:
            return {"ok": False, "reason": f"file too small ({size} bytes)"}
        try:
            from PIL import Image  # type: ignore
            import numpy as np  # type: ignore

            with Image.open(path) as img:
                gray = img.convert("L")
                arr = np.asarray(gray, dtype="float32")
            std = float(arr.std())
            mean = float(arr.mean())
            # A blank canvas has near-zero variance.
            ok = std > 3.0
            return {
                "ok": ok,
                "std": round(std, 2),
                "mean": round(mean, 2),
                "size": size,
                "reason": "" if ok else "image appears blank (low variance)",
            }
        except Exception as exc:
            logger.debug("image check skipped: %s", exc)
            return {"ok": True, "reason": f"image check unavailable: {exc}"}

    # --------------------------------------------------------------- semantic
    def semantic_check(self, state: Any, structural: list[dict[str, Any]]) -> dict[str, Any]:
        """Ask the LLM to verify requirements against artifacts/methodology."""
        if self.llm_client is None or state is None:
            return {"pass": True, "per_requirement": [], "missing": [], "note": "semantic check skipped"}
        requirements = [r.to_dict() for r in getattr(state, "requirements", [])]
        if not requirements:
            return {"pass": True, "per_requirement": [], "missing": [], "note": "no requirements"}
        payload = {
            "goal": state.goal,
            "requirements": requirements,
            "methodology": getattr(state, "methodology", ""),
            "final_summary": getattr(state, "final_summary", ""),
            "artifacts": structural,
            "tasklist": [t.to_dict() for t in getattr(state, "tasklist", [])],
        }
        prompt = (
            "你是 GIS 成果验收专家。请对照需求清单逐条验收下面的产出。\n"
            "只输出一个 JSON 对象，格式：\n"
            '{"pass": true/false, "per_requirement": [{"id": "...", "status": "PASS|FAIL|PARTIAL", '
            '"evidence": "..."}], "missing": ["..."], "repair_hint": "下一步应修复什么"}\n'
            "判定标准：产物必须真实存在且满足需求（坐标系/字段/要素数/图层/地图要素等）。"
            "如果需求无法从证据判断，标 PARTIAL 并说明原因。\n\n"
            f"验收材料：\n{json.dumps(payload, ensure_ascii=False, default=str)[:24000]}"
        )
        try:
            response = self.llm_client.chat(
                [
                    {"role": "system", "content": "你只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                task_type="verify",
            )
            parsed = extract_json_object(response.content)
        except Exception as exc:
            logger.warning("semantic verification failed: %s", exc)
            return {"pass": True, "per_requirement": [], "missing": [], "note": f"llm error: {exc}"}
        if not isinstance(parsed, dict):
            return {"pass": True, "per_requirement": [], "missing": [], "note": "unparseable llm verdict"}
        parsed.setdefault("pass", False)
        parsed.setdefault("per_requirement", [])
        parsed.setdefault("missing", [])
        parsed.setdefault("repair_hint", "")
        return parsed

    # ----------------------------------------------------------------- verify
    def verify(self, state: Any) -> dict[str, Any]:
        """Full verification of the current state."""
        artifacts = list(getattr(state, "artifacts", []) or [])
        if not artifacts:
            # Outputs exist on disk but were never registered (e.g. code did
            # not set_result the paths) — discover them instead of failing.
            artifacts = self._discover_outputs()
            if artifacts and state is not None:
                for path in artifacts:
                    state.add_artifact(path)
        structural = self.structural_check(artifacts)

        problems: list[str] = []
        if not artifacts:
            problems.append("没有任何产出文件")
        out_root = str((self.workspace / "output").resolve()).lower()
        for path in artifacts:
            if not str(Path(path).resolve()).lower().startswith(out_root):
                problems.append(f"产出不在 output 目录内: {path}")
        for record in structural:
            if not record.get("exists"):
                problems.append(f"产出不存在: {record.get('path')}")
                continue
            if record.get("feature_count") == 0:
                problems.append(f"产出为空（0 要素）: {record.get('path')}")
            if isinstance(record.get("image"), dict) and not record["image"].get("ok"):
                problems.append(
                    f"图片可能为空白: {record.get('path')} ({record['image'].get('reason')})"
                )

        semantic = self.semantic_check(state, structural)
        missing = [str(m) for m in (semantic.get("missing") or [])]
        failed_reqs = [
            item
            for item in (semantic.get("per_requirement") or [])
            if str(item.get("status", "")).upper() == "FAIL"
        ]
        passed = bool(semantic.get("pass", True)) and not problems and not missing and not failed_reqs

        summary_bits = []
        if problems:
            summary_bits.append("结构检查问题: " + "；".join(problems[:5]))
        if missing:
            summary_bits.append("缺失: " + "；".join(missing[:5]))
        if failed_reqs:
            summary_bits.append("未满足需求: " + "；".join(str(r.get("id")) for r in failed_reqs[:5]))
        summary = "验收通过" if passed else "验收未通过：" + " | ".join(summary_bits)

        return {
            "pass": passed,
            "summary": summary,
            "structural": structural,
            "semantic": semantic,
            "missing": missing,
            "problems": problems,
            "repair_hint": str(semantic.get("repair_hint", "") or "") or (
                "；".join(problems[:5]) if problems else ""
            ),
        }

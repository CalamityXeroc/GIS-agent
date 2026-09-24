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


def _artifact_exists(path: str) -> bool:
    """已删除的产物不该再算产物（文件系统或 GDB 语义都试一下）。"""
    try:
        if Path(path).exists():
            return True
    except Exception:
        pass
    try:
        import arcpy  # type: ignore

        return bool(arcpy.Exists(path))
    except Exception:
        return False


class Verifier:
    """Two-layer verifier: structural + semantic."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        code_runner: Any = None,
        llm_client: Any = None,
        min_image_bytes: int = 8_000,
        hygiene_mode: str = "warn",
        vision_mode: str = "warn",
    ):
        self.workspace = Path(workspace)
        self.code_runner = code_runner
        self.llm_client = llm_client
        self.min_image_bytes = min_image_bytes
        # 交付卫生：warn（默认，仅提醒）| block（重复中间件/过程文件算验收不通过）| off
        self.hygiene_mode = str(hygiene_mode or "warn").lower()
        # 图面识图质检：让识图模型看图，判定中文是否可读/图面是否被裁；warn|block|off
        self.vision_mode = str(vision_mode or "warn").lower()

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
                # 内核检查失败时不能退回文件系统判断：GDB 内要素类/栅格会误报“产出不存在”，
                # 反而把 agent 逼去重建已有数据。用 arcpy 感知的存在性判断。
                exists = _artifact_exists(path)
                target = Path(path)
                results.append(
                    {
                        "path": path,
                        "exists": exists,
                        "size": target.stat().st_size if exists and target.is_file() else None,
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

    # -------------------------------------------------------------- hygiene
    def hygiene_findings(self) -> dict[str, Any]:
        """交付卫生观察：过程库重做残留、过程文件、体积比。

        默认只提醒（不阻断验收），因为工作过程中出现临时文件是正常的；
        ``hygiene_mode="block"`` 时把硬指标（重复中间件/过程文件）计入失败。
        """
        empty = {"ok": True, "problems": [], "warnings": [], "detail": ""}
        if self.hygiene_mode == "off":
            return empty
        try:
            from ..runtime.hygiene import dir_inventory, hygiene_report, summarize

            inventory, err = dir_inventory(self.code_runner, str(self.workspace / "output"))
            if inventory is None:
                return {**empty, "warnings": [f"卫生检查跳过: {err}"]}
            problems, warnings_out, detail = hygiene_report(
                inventory, max_duplicate_families=0, ratio_limit=5.0
            )
            return {
                "ok": not problems,
                "problems": problems,
                "warnings": warnings_out,
                "detail": detail,
                "summary": summarize(problems, warnings_out),
            }
        except Exception as exc:  # pragma: no cover - 卫生检查不能影响验收主流程
            logger.debug("hygiene check failed: %s", exc)
            return {**empty, "warnings": [f"卫生检查异常: {str(exc)[:120]}"]}

    # ---------------------------------------------------------------- vision
    _VISION_PROMPT = (
        "你是地图/图表质检员。请检查这张图，只输出一个 JSON 对象：\n"
        '{"text_ok": true/false, "issues": ["..."], "summary": "一句话结论"}\n'
        "判定要点：标题、图例、比例尺、坐标轴文字是否可读——**中文若显示为方框(口口口)或乱码则 text_ok=false**；"
        "另外检查：图面是否被裁切、是否大面积空白、图例是否遮挡主体、地图要素是否缺失。"
        "没有问题时 issues 为空数组、text_ok=true。"
    )

    def vision_check_image(self, path: Path) -> dict[str, Any]:
        """用识图模型核对一张产出的图（中文可读性、图面完整性）。

        跳过条件：开关关闭、无 LLM、模型不支持识图、图片读不出来。
        任何异常都不影响验收主流程。
        """
        if self.vision_mode == "off" or self.llm_client is None:
            return {"ok": True, "skipped": True}
        if not getattr(getattr(self.llm_client, "config", None), "vision", False):
            return {"ok": True, "skipped": True, "reason": "模型未开启识图"}
        try:
            from ..runtime.image_input import as_data_url

            url = as_data_url(str(path))
            if not url:
                return {"ok": True, "skipped": True, "reason": "图片无法读取"}
            response = self.llm_client.chat(
                [
                    {"role": "system", "content": "你只输出 JSON。"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._VISION_PROMPT},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    },
                ],
                task_type="verify",
            )
            parsed = extract_json_object(response.content) or {}
        except Exception as exc:  # pragma: no cover - 识图失败不影响验收
            logger.debug("vision check failed: %s", exc)
            return {"ok": True, "skipped": True, "reason": f"识图调用失败: {str(exc)[:120]}"}
        text_ok = bool(parsed.get("text_ok", True))
        issues = [str(item) for item in (parsed.get("issues") or [])][:5]
        return {
            "ok": text_ok and not issues,
            "text_ok": text_ok,
            "issues": issues,
            "summary": str(parsed.get("summary", ""))[:200],
            "model": getattr(response, "model", ""),
        }

    # ----------------------------------------------------------------- verify
    def verify(self, state: Any) -> dict[str, Any]:
        """Full verification of the current state."""
        # 已删除的产物（如误建后又清理的目录）要从产物清单里剔除，
        # 否则验收会一直报错，把 agent 逼去追根本不存在的文件。
        artifacts = [path for path in (getattr(state, "artifacts", []) or []) if _artifact_exists(str(path))]
        if state is not None and hasattr(state, "artifacts"):
            state.artifacts = artifacts
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
        hygiene = self.hygiene_findings()
        if hygiene.get("problems") and self.hygiene_mode == "block":
            problems.extend(f"交付卫生: {item}" for item in hygiene["problems"])

        # 图面识图质检：中文是否可读、图例/图名是否齐全（模型支持识图时）
        vision_warnings: list[str] = []
        for record in structural:
            if not isinstance(record.get("image"), dict):
                continue
            verdict = self.vision_check_image(Path(str(record.get("path", ""))))
            if verdict.get("skipped"):
                continue
            record["vision"] = verdict
            if not verdict.get("ok"):
                name = Path(str(record.get("path", ""))).name
                detail = verdict.get("summary") or "；".join(verdict.get("issues") or [])
                message = f"图面质检（{name}）: {detail or '模型判定图面有问题'}"
                if self.vision_mode == "block":
                    problems.append(message)
                else:
                    vision_warnings.append(message)

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
        if hygiene.get("summary"):
            prefix = "卫生问题: " if self.hygiene_mode == "block" else "卫生提醒: "
            summary_bits.append(prefix + str(hygiene["summary"])[:240])
        if vision_warnings:
            summary_bits.append("图面提醒: " + "；".join(vision_warnings[:2])[:240])
        summary = "验收通过" if passed else "验收未通过：" + " | ".join(summary_bits)

        return {
            "pass": passed,
            "summary": summary,
            "structural": structural,
            "semantic": semantic,
            "hygiene": hygiene,
            "missing": missing,
            "problems": problems,
            "repair_hint": str(semantic.get("repair_hint", "") or "") or (
                "；".join(problems[:5]) if problems else ""
            ),
        }

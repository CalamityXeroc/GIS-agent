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


def _mtime_key(path: Path) -> float:
    """文件 mtime（取不到时返回 0）——识图缓存的失效依据。"""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _shrink_json(value: Any, budget: int, depth: int = 0) -> Any:
    """递归缩水 JSON（保持结构合法），给“内容太长”的验收材料用。

    存在的理由：直接把 JSON 字符串切到 max_chars 会截在字符串中间，
    ``json.loads`` 随即抛 JSONDecodeError，把整个任务拖死（实测踩到）。
    这里只截长字符串、只留前几个列表项，得到的仍是合法 JSON。
    """
    if depth > 6:
        return "…"
    if isinstance(value, str):
        room = max(16, budget // 4)
        return value if len(value) <= room else value[: room - 1] + "…"
    if isinstance(value, list):
        kept = [_shrink_json(item, max(32, budget // 2), depth + 1) for item in value[:3]]
        if len(value) > 3:
            kept.append(f"…（共 {len(value)} 项）")
        return kept
    if isinstance(value, dict):
        share = max(32, budget // max(1, len(value)))
        return {str(k): _shrink_json(v, share, depth + 1) for k, v in list(value.items())[:12]}
    return value


def _artifact_exists(path: str) -> bool:
    """已删除的产物不该再算产物（文件系统或 GDB 语义都试一下）。

    注意：本函数可能跑在**没有 arcpy 的解释器**里（benchmark/CLI 的 venv）。
    GDB 内的要素类/栅格在文件系统上根本不存在，只有 arcpy 能判——此时
    “判不了”不能当成“不存在”，否则会把合法产物剔除（14 届 4a-1 实测：
    finish 里列的 4 个 kde_* 栅格全被剔除，验收只说“缺失栅格元数据”，
    agent 空转到轮次上限）。所以无法用 arcpy 时，把 GDB 内路径当作存在，
    交给后面跑在内核里的 structural_check 去实测。
    """
    try:
        if Path(path).exists():
            return True
    except Exception:
        pass
    normalized = str(path).lower().replace("\\", "/")
    in_gdb = ".gdb/" in normalized
    try:
        import arcpy  # type: ignore

        return bool(arcpy.Exists(path))
    except Exception:
        # 没有 arcpy：GDB 内要素判不了，不要当不存在（见 docstring）
        return in_gdb


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
        hygiene_max_duplicate_families: int = 0,
        hygiene_ratio_limit: float = 5.0,
        hygiene_forbid_suffixes: list[str] | None = None,
    ):
        self.workspace = Path(workspace)
        self.code_runner = code_runner
        self.llm_client = llm_client
        self.min_image_bytes = min_image_bytes
        # 交付卫生：warn（默认，仅提醒）| block（重复中间件/过程文件算验收不通过）| off
        self.hygiene_mode = str(hygiene_mode or "warn").lower()
        # 卫生判据参数（与配方侧的 deliverable_hygiene_ok 用同一套键，
        # 以前这里硬编码 0 / 5.0，导致同一份交付物“配方判不过、验收判过”）。
        self.hygiene_max_duplicate_families = max(0, int(hygiene_max_duplicate_families or 0))
        self.hygiene_ratio_limit = max(0.0, float(hygiene_ratio_limit or 0.0))
        self.hygiene_forbid_suffixes = (
            [str(item).lower() for item in hygiene_forbid_suffixes]
            if hygiene_forbid_suffixes
            else None
        )
        # 图面识图质检：让识图模型看图，判定中文是否可读/图面是否被裁；warn|block|off
        self.vision_mode = str(vision_mode or "warn").lower()
        #: 识图结论缓存：``(路径, mtime) -> 结论``。
        #: 同一批产物在一个 run 里会被验两次（verify_outputs + finish），
        #: 没有缓存就要把每张图看两遍（实测验图开销高达 200s+）。
        self._vision_cache: dict[tuple[str, float], dict[str, Any]] = {}

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
        """Best-effort discovery of produced files under workspace/output.

        扩展名清单与 tools 共用一份（以前两边各写一份，都漏了 Office 格式，
        导致 Word/PPT 产出发现不了）。
        """
        from .tools import GIS_OUTPUT_EXT, _NON_OUTPUT_EXT

        found: list[str] = []
        out_root = self.workspace / "output"
        if not out_root.exists():
            return found
        for item in sorted(out_root.rglob("*")):
            if len(found) >= limit:
                break
            name = item.name.lower()
            suffix = item.suffix.lower()
            if suffix in _NON_OUTPUT_EXT or name.startswith("~$"):
                continue  # 中间产物与 Office 临时文件
            if item.is_dir() and name.endswith(".gdb"):
                found.append(str(item))
            elif item.is_file() and suffix in GIS_OUTPUT_EXT:
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
            # 设计规格（.spec.json）的**实际内容**。
            # 实测坑：以前只把 structural（路径+存在性）给模型，模型看不到
            # class_bounds 之类的实际值，于是无论 agent 做什么都回“无法确认四图
            # 分级是否一致”/“缺失 .spec.json”，把 agent 逼到反复重做直至轮次上限
            # （14 届 4a-1 实测：验收 problems 明明是空的）。
            "spec_contents": self._read_spec_contents(),
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

    def _read_spec_contents(self, *, limit: int = 6, max_chars: int = 1200) -> list[dict[str, Any]]:
        """读出产出目录下 ``*.spec.json`` 的关键内容，供语义验收核对。

        只取**判据相关**的字段（分级边界/渲染器决策），不把整份设计文档塞进 prompt：
        ``.spec.json`` 可达 8 KB，整份塞进去会挤掉其它证据。
        """
        out: list[dict[str, Any]] = []
        out_root = self.workspace / "output"
        if not out_root.exists():
            return out
        for path in sorted(out_root.rglob("*.spec.json")):
            if len(out) >= limit:
                break
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            keys = (
                "class_bounds", "bounds", "break_values", "class_breaks",
                "legend_class_bounds", "explicit_bounds", "renderer", "decisions",
                "title", "theme", "region", "scale", "purpose",
            )
            picked = {k: raw[k] for k in keys if k in raw}
            if not picked:
                picked = {"keys": sorted(raw.keys())[:20]}
            text = json.dumps(picked, ensure_ascii=False, default=str)
            if len(text) > max_chars:
                # 超长时**不能直接截断再 json.loads**：截在字符串中间会得到
                # 非法 JSON 并抛 JSONDecodeError，把整个任务拖死（实测：有
                # 孤立的 .spec.json 内容不完整 → 验收阶段崩 → run 直接挂）。
                # 这里改成保留结构地递归缩水，实在不行就退成纯文本。
                picked = _shrink_json(picked, max_chars)
                text = json.dumps(picked, ensure_ascii=False, default=str)
                if len(text) > max_chars:
                    picked = {"note": "内容过长，已省略", "raw_prefix": text[: max(0, max_chars - 40)]}
            out.append({
                "path": str(path.relative_to(self.workspace)).replace("\\", "/"),
                "content": picked,
            })
        return out

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
                inventory,
                max_duplicate_families=self.hygiene_max_duplicate_families,
                forbid_suffixes=self.hygiene_forbid_suffixes,
                ratio_limit=self.hygiene_ratio_limit,
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

    #: 一次调用同时检查多张图时的提示词（省去每张一次网络往返）
    _VISION_BATCH_PROMPT = (
        "你是地图/图表质检员。下面依次给出 {count} 张图（顺序与编号一致）。\n"
        "只输出一个 JSON 对象：\n"
        '{{"images": [{{"index": 1, "text_ok": true, "issues": [], "summary": "一句话结论"}}]}}\n'
        "判定要点：标题、图例、比例尺、坐标轴文字是否可读——**中文若显示为方框(口口口)或乱码则 text_ok=false**；"
        "另外检查：图面是否被裁切、是否大面积空白、图例是否遮挡主体、地图要素是否缺失。\n"
        "images 数组必须包含全部 {count} 项，index 从 1 开始。"
    )

    def vision_check_images(self, paths: list[Path]) -> dict[int, dict[str, Any]]:
        """一次调用质检多张图，返回 ``{下标: 结论}``。

        为什么要批量：实测收尾验收对 5 张图**逐张串行**调用，一次 run 光验图
        就花了 200s+（占该 run 总耗时 82%）。一次请求带多张图，往返从 N 次降为 1 次。
        任何失败都退回“逐张检查”的结果（``{}`` = 让调用方自己回退）。
        """
        if self.vision_mode == "off" or self.llm_client is None or not paths:
            return {}
        if not getattr(getattr(self.llm_client, "config", None), "vision", False):
            return {}
        if getattr(self.llm_client, "vision_rejected", False):
            return {}
        try:
            from ..runtime.image_input import as_data_url

            parts: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": self._VISION_BATCH_PROMPT.format(count=len(paths)),
                }
            ]
            used = 0
            for path in paths:
                url = as_data_url(str(path))
                if not url:
                    continue
                used += 1
                parts.append({"type": "text", "text": f"第 {used} 张：{path.name}"})
                parts.append({"type": "image_url", "image_url": {"url": url}})
            if used == 0:
                return {}
            response = self.llm_client.chat(
                [
                    {"role": "system", "content": "你只输出 JSON。"},
                    {"role": "user", "content": parts},
                ],
                task_type="verify",
            )
            parsed = extract_json_object(response.content) or {}
            items = parsed.get("images") if isinstance(parsed, dict) else None
            if not isinstance(items, list):
                return {}
            out: dict[int, dict[str, Any]] = {}
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    index = int(item.get("index", 0)) - 1
                except Exception:
                    continue
                if 0 <= index < len(paths):
                    out[index] = self._normalize_vision(item, model=getattr(response, "model", ""))
            return out
        except Exception as exc:  # pragma: no cover - 识图失败退回逐张路径
            logger.debug("batch vision check failed: %s", exc)
            return {}

    @staticmethod
    def _normalize_vision(parsed: dict[str, Any], *, model: str = "") -> dict[str, Any]:
        """把识图结论归一成统一结构。"""
        text_ok = bool(parsed.get("text_ok", True))
        issues = [str(item) for item in (parsed.get("issues") or [])][:5]
        return {
            "ok": text_ok and not issues,
            "text_ok": text_ok,
            "issues": issues,
            "summary": str(parsed.get("summary", ""))[:200],
            "model": model,
        }

    def _vision_verdicts(self, records: list[dict[str, Any]]) -> None:
        """给每条图片产物填上 ``record["vision"]``（批量 + 缓存）。

        - **批量**：多张图一次调用看完（实测逐张串行时 5 张图要 200s+）；
        - **缓存**：以 ``(路径, mtime)`` 为键。同一批产物在一个 run 里会被验两次
          （``verify_outputs`` 与 ``finish``），第二次直接命中缓存；图片改了
          mtime 就自动失效。

        """
        if not records:
            return
        keys = [(str(r.get("path", "")), _mtime_key(Path(str(r.get("path", ""))))) for r in records]
        missing = [i for i, key in enumerate(keys) if key not in self._vision_cache]
        if missing:
            paths = [Path(keys[i][0]) for i in missing]
            verdicts = self.vision_check_images(paths) if len(paths) > 1 else {}
            if not verdicts and len(paths) == 1:
                verdicts = {0: self.vision_check_image(paths[0])}
            if not verdicts:
                # 批量不支持（或全失败）→ 退回逐张（至少部分图能验到）
                for index in missing:
                    verdicts[index] = self.vision_check_image(Path(keys[index][0]))
            for index, verdict in verdicts.items():
                if 0 <= index < len(missing):
                    self._vision_cache[keys[missing[index]]] = verdict
        for record, key in zip(records, keys):
            verdict = self._vision_cache.get(key)
            if verdict is not None:
                record["vision"] = verdict

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
        return self._normalize_vision(parsed, model=getattr(response, "model", ""))

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
        problems: list[str] = []
        out_root = str((self.workspace / "output").resolve()).lower()
        ws_root = str(self.workspace.resolve()).lower()
        # workspace 之外的路径根本不是本次产出（ArcGIS 自带模板 aprx、用户 Documents 下的
        # 工程），是模型把工具回参当产物登记进来的。这种直接剔除，不要当成 blocked 问题——
        # 否则验收每次都不通过，agent 会空转到轮次上限（14 届 4a-1 实测：断言 7/7 全过
        # 却 status=max_turns）。真正该报的只是“在 workspace 内、但不在 output/ 下”。
        inside: list[str] = []
        for path in artifacts:
            resolved = str(Path(path).resolve()).lower()
            if not resolved.startswith(ws_root):
                continue
            inside.append(path)
            if not resolved.startswith(out_root):
                problems.append(f"产出不在 output 目录内: {path}")
        if state is not None and hasattr(state, "artifacts"):
            state.artifacts = inside
        artifacts = inside
        if not artifacts:
            problems.append("没有任何产出文件")
        # 结构体检要在剔除外部路径**之后**做，否则会去读 Blank.aprx 之类的无关文件。
        structural = self.structural_check(artifacts)
        # 内核（有 arcpy）才判得准 GDB 内的要素类/栅格是否存在。模型中途建的
        # 中间产物（tmp_train / idw_p1 之类）删掉后仍留在 artifacts 里，若在此
        # 报“产出不存在”，验收就永远不过 —— 14 届任务二a 实测：模型被逼着重建
        # 已删的中间件，空转到 60 轮上限后改去建 result_data_v2.gdb。
        # 正确做法：以内核结论为准静默剔除，缺的交付物交给语义验收去点名。
        alive: list[str] = []
        for record in structural:
            if record.get("exists"):
                alive.append(str(record.get("path", "")))
        if state is not None and hasattr(state, "artifacts"):
            state.artifacts = alive
        if artifacts and not alive:
            # 登记过的产出全都不在了：这不是“中间件被清理”，是真的什么都没交。
            problems.append("没有任何产出文件（登记的产出均已不存在）")
        for record in structural:
            if not record.get("exists"):
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
        image_records = [r for r in structural if isinstance(r.get("image"), dict)]
        self._vision_verdicts(image_records)
        for record in image_records:
            verdict = record.get("vision")
            if not isinstance(verdict, dict) or verdict.get("skipped"):
                continue
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

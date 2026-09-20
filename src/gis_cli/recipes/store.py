# -*- coding: utf-8 -*-
"""Recipe library: load, search, execute, and validate GIS methodology units."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import Recipe

logger = logging.getLogger(__name__)

_SLOT = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


@dataclass
class ValidationResult:
    """Outcome of one recipe assertion."""

    kind: str
    target: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "target": self.target, "ok": self.ok, "detail": self.detail}


class RecipeLibrary:
    """Loads built-in and user recipes, and runs them safely."""

    def __init__(
        self,
        *,
        builtin_dir: str | Path | None = None,
        user_dir: str | Path | None = None,
        code_runner: Any = None,
        catalog: Any = None,
    ):
        self.builtin_dir = Path(builtin_dir) if builtin_dir else Path(__file__).parent / "builtin"
        self.user_dir = Path(user_dir) if user_dir else None
        self.code_runner = code_runner
        self.catalog = catalog
        self._recipes: dict[str, Recipe] = {}

    # ------------------------------------------------------------------- load
    def load(self) -> int:
        """(Re)load all recipe YAML files."""
        self._recipes = {}
        for directory in [self.builtin_dir, self.user_dir]:
            if directory is None or not directory.exists():
                continue
            for path in sorted(directory.glob("*.y*ml")):
                try:
                    recipe = Recipe.from_yaml(path)
                    if recipe.id:
                        self._recipes[recipe.id] = recipe
                except Exception as exc:
                    logger.warning("failed to load recipe %s: %s", path, exc)
        return len(self._recipes)

    def ensure_loaded(self) -> None:
        if not self._recipes:
            self.load()

    # ------------------------------------------------------------------ query
    def get(self, recipe_id: str) -> Recipe | None:
        self.ensure_loaded()
        return self._recipes.get(recipe_id)

    def all(self) -> list[Recipe]:
        self.ensure_loaded()
        return list(self._recipes.values())

    def search(self, query: str = "", limit: int = 8) -> list[dict[str, Any]]:
        self.ensure_loaded()
        scored = [(recipe.match_score(query), recipe) for recipe in self._recipes.values()]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [recipe.to_dict() for _, recipe in scored[:limit]]

    def catalog_digest(self, *, max_chars: int = 5200) -> str:
        """紧凑操作目录：按类别分组列出全部配方（id / 用途 / 必填参数）。

        目的是让模型不必反复调用 list_recipes 就能看到全部可选操作。
        """
        self.ensure_loaded()
        groups: dict[str, list[Recipe]] = {}
        for recipe in self._recipes.values():
            groups.setdefault(recipe.category or recipe.domain or "general", []).append(recipe)
        order = sorted(groups, key=lambda key: (-len(groups[key]), key))
        label = {
            "raster": "栅格",
            "overlay": "叠加分析",
            "geometry": "几何处理",
            "analysis": "矢量分析",
            "spatial_analysis": "空间分析",
            "spatial_statistics": "空间统计",
            "table": "属性表",
            "data_management": "数据管理",
            "projection": "投影与坐标系",
            "data_quality": "数据质量",
            "cartography": "制图",
            "mapping": "制图",
            "infra": "基础设施",
        }
        lines: list[str] = []
        for key in order:
            items = sorted(groups[key], key=lambda r: r.id)
            title = label.get(key, key)
            lines.append(f"### {title}（{len(items)}）")
            for recipe in items:
                required = [
                    name
                    for name, spec in recipe.params.items()
                    if spec.required
                ]
                params = ", ".join(required) if required else "—"
                brief = (recipe.name or recipe.description).strip()
                brief = brief.split("（")[0].split("(")[0].strip()
                lines.append(f"- {recipe.id}: {brief} ｜必填: {params}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max_chars - 40] + "\n…（更多配方用 list_recipes 关键词检索）"
        return text

    # -------------------------------------------------------------------- run
    def run(self, recipe_id: str, params: dict[str, Any], ctx: Any, *, timeout: float | None = None) -> Any:
        """Execute a recipe and its validation assertions.

        ``timeout`` 可覆盖默认执行超时（缓冲区分区统计这类重活常需要 20 分钟以上）。
        """
        from ..engine.state import Observation

        recipe = self.get(recipe_id)
        if recipe is None:
            return Observation(
                ok=False,
                summary=f"配方不存在: {recipe_id}",
                error={"type": "RecipeNotFound", "message": f"可用配方: {', '.join(self._recipes)}"},
            )
        if self.code_runner is None and getattr(ctx, "code_runner", None) is None:
            return Observation(ok=False, summary="没有可用的代码执行器", error={"type": "NoRunner"})
        runner = getattr(ctx, "code_runner", None) or self.code_runner
        try:
            code = recipe.render(params or {})
        except Exception as exc:
            return Observation(
                ok=False,
                summary=f"配方参数错误: {exc}",
                error={"type": "RecipeParamError", "message": str(exc)},
            )

        default_timeout = float(getattr(ctx, "exec_timeout", 0) or 0)
        result = runner.run(
            code,
            timeout=float(timeout or default_timeout or 600.0),
            workspace=str(getattr(ctx, "workspace", ".")),
        )
        if not result.ok:
            return Observation(
                ok=False,
                summary=f"配方 {recipe.id} 执行失败: [{(result.error or {}).get('type')}] {(result.error or {}).get('message')}",
                data={"stdout": (result.stdout or "")[-3000:], "code": code},
                error=result.error,
                hint="可改用 execute_code 手动实现，或修正参数后重试。",
            )

        resolved = recipe.resolve_params(params or {})
        assertions = self._run_assertions(recipe, resolved, result)
        failed = [a for a in assertions if not a.ok]
        from ..engine.tools import extract_outputs

        artifacts = extract_outputs(result.result, str(getattr(ctx, "workspace", ".")))
        summary = f"配方 {recipe.id} 执行成功"
        if assertions:
            summary += f"，断言 {len(assertions) - len(failed)}/{len(assertions)} 通过"
        if failed:
            summary += "；未通过: " + "；".join(f"{a.kind}({a.target}) {a.detail}" for a in failed[:3])
        return Observation(
            ok=not failed,
            summary=summary,
            data={
                "stdout": (result.stdout or "")[-3000:],
                "result": result.result,
                "assertions": [a.to_dict() for a in assertions],
                "recipe": recipe.to_dict(),
            },
            artifacts=artifacts,
            images=result.display_images[:3],
            hint="" if not failed else "请检查配方断言失败项，必要时用 execute_code 修正。",
        )

    # ------------------------------------------------------------- assertions
    def run_assertions(self, assertions: list[dict[str, Any]], *, result: Any = None) -> list[ValidationResult]:
        """Public entry for evaluating raw assertion dicts (used by benchmarks)."""
        out: list[ValidationResult] = []
        for raw in assertions or []:
            kind = str(raw.get("type", "") or "")
            target = str(raw.get("path", "") or raw.get("target", ""))
            expected = raw.get("value")
            try:
                out.append(self._check(kind, target, expected, raw, result))
            except Exception as exc:
                out.append(ValidationResult(kind, target, False, f"断言执行异常: {exc}"))
        return out

    def _run_assertions(self, recipe: Recipe, params: dict[str, Any], result: Any) -> list[ValidationResult]:
        out: list[ValidationResult] = []
        for raw in recipe.validation:
            # 解析断言里所有字符串槽位（path/target/value 以及 field/reference 等自定义键）
            raw = {
                k: (_resolve(v, params) if isinstance(v, str) else v)
                for k, v in raw.items()
            }
            kind = str(raw.get("type", "") or "")
            target = str(raw.get("path", "") or raw.get("target", ""))
            expected = raw.get("value")
            if isinstance(expected, list):
                expected = [_resolve(str(v), params) for v in expected]
            try:
                out.append(self._check(kind, target, expected, raw, result))
            except Exception as exc:
                out.append(ValidationResult(kind, target, False, f"断言执行异常: {exc}"))
        return out

    def _check(
        self,
        kind: str,
        target: str,
        expected: Any,
        raw: dict[str, Any],
        result: Any,
    ) -> ValidationResult:
        path = Path(target)
        if kind == "file_exists":
            return ValidationResult(kind, target, path.exists(), "" if path.exists() else "文件不存在")
        if kind == "file_contains":
            if not path.exists():
                return ValidationResult(kind, target, False, "文件不存在")
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                return ValidationResult(kind, target, False, f"读取失败: {exc}")
            needle = str(expected or "")
            ok = bool(needle) and needle in text
            return ValidationResult(kind, target, ok, f"contains({needle})={ok}")
        if kind in {"vector_exists", "raster_exists"}:
            # GDB 要素类/栅格不是文件系统路径：用 arcpy.Exists 判断。
            # arcpy 对新写入的数据集有可见性缓存，刚 save 完可能瞬时为 False，
            # 因此这里做一次短重试，避开“刚产出却判为不存在”的偶发误报。
            if path.exists():
                return ValidationResult(kind, target, True, "")
            record = self._describe(target)
            ok = bool(record and record.get("exists"))
            if not ok:
                time.sleep(0.8)
                record = self._describe(target)
                ok = bool(record and record.get("exists"))
            return ValidationResult(kind, target, ok, "" if ok else "要素类/栅格不存在（arcpy.Exists 也为 False）")
        if kind == "feature_count_gt":
            count = self._feature_count(target)
            ok = count is not None and count > int(expected or 0)
            return ValidationResult(kind, target, ok, f"count={count}")
        if kind == "feature_count_equal":
            count = self._feature_count(target)
            ok = count is not None and int(expected) == count
            return ValidationResult(kind, target, ok, f"count={count} expected={expected}")
        if kind == "feature_count_lt":
            count = self._feature_count(target)
            ok = count is not None and count < int(expected)
            return ValidationResult(kind, target, ok, f"count={count} expected<{expected}")
        if kind == "feature_count_between":
            count = self._feature_count(target)
            try:
                lo, hi = int(expected[0]), int(expected[1])
            except Exception:
                return ValidationResult(kind, target, False, f"非法范围: {expected}")
            ok = count is not None and lo <= count <= hi
            return ValidationResult(kind, target, ok, f"count={count} expected in [{lo},{hi}]")
        if kind == "field_value_counts":
            expect = raw.get("value") or {}
            if not isinstance(expect, dict) or not expect:
                return ValidationResult(kind, target, False, "value 需要写成 {类别: 数量} 映射")
            counts, bad = self._field_value_counts(
                target, str(raw.get("field", "") or ""), {str(k): int(v) for k, v in expect.items()}
            )
            ok = counts is not None and not bad
            detail = f"counts={counts}" if ok else f"不符: {bad}；实际={counts}"
            return ValidationResult(kind, target, ok, detail)
        if kind == "field_values_subset":
            allowed = expected if isinstance(expected, list) else [expected]
            allowed_set = {str(a) for a in allowed}
            values, bad = self._field_values(target, str(raw.get("field", "") or ""), allowed_set=allowed_set)
            if values is None:
                return ValidationResult(kind, target, False, f"字段读取失败: {bad}")
            ok = not bad
            return ValidationResult(kind, target, ok, f"values={sorted(values)[:8]} unexpected={sorted(bad)[:8]}")
        if kind == "overlap_pairs_max":
            max_allowed = int(expected if expected is not None else 0)
            count, truncated = self._overlap_pairs(target, max_allowed)
            ok = count is not None and count <= max_allowed
            detail = f"overlaps={count}" + (" (truncated)" if truncated else "") + f" max={max_allowed}"
            return ValidationResult(kind, target, ok, detail)
        if kind == "extent_matches_reference":
            return self._extent_check(target, str(raw.get("reference", "") or ""), raw, result)
        if kind == "crs_wkid":
            wkid = self._crs_wkid(target)
            ok = wkid is not None and int(wkid) == int(expected)
            return ValidationResult(kind, target, ok, f"wkid={wkid} expected={expected}")
        if kind == "fields_present":
            fields = self._fields(target)
            required = expected if isinstance(expected, list) else [expected]
            missing = [f for f in required if f not in fields]
            return ValidationResult(kind, target, not missing, f"missing={missing}")
        if kind == "field_sum_gt":
            field = str(raw.get("field", "") or "")
            total = self._field_stat(target, field, "sum")
            ok = total is not None and total > float(expected or 0)
            return ValidationResult(kind, target, ok, f"sum({field})={total} expected>{expected}")
        if kind == "field_distinct_gt":
            field = str(raw.get("field", "") or "")
            distinct = self._field_stat(target, field, "distinct")
            ok = distinct is not None and distinct > int(expected or 0)
            return ValidationResult(kind, target, ok, f"distinct({field})={distinct} expected>{expected}")
        if kind == "field_stats_between":
            field = str(raw.get("field", "") or "")
            stat = str(raw.get("stat", "mean") or "mean").lower()
            value = self._field_stat(target, field, stat)
            try:
                lo, hi = float(expected[0]), float(expected[1])
            except Exception:
                return ValidationResult(kind, target, False, f"非法区间: {expected}")
            ok = value is not None and lo <= value <= hi
            return ValidationResult(kind, target, ok, f"{stat}({field})={value} expected in [{lo},{hi}]")
        if kind == "join_null_rate_below":
            field = str(raw.get("field", "") or "")
            rate = self._field_stat(target, field, "null_rate")
            limit = float(expected if expected is not None else 0.05)
            ok = rate is not None and rate <= limit
            return ValidationResult(kind, target, ok, f"null_rate({field})={rate} expected<={limit}")
        if kind == "raster_stat_between":
            stat = str(raw.get("stat", "sum") or "sum").lower()
            info = self._raster_stats(target)
            try:
                lo, hi = float(expected[0]), float(expected[1])
            except Exception:
                return ValidationResult(kind, target, False, f"非法区间: {expected}")
            value = info.get(stat) if info else None
            ok = value is not None and lo <= float(value) <= hi
            detail = f"{stat}={value} expected in [{lo},{hi}]" + (f" stats={info}" if info and not ok else "")
            return ValidationResult(kind, target, ok, detail)
        if kind == "extent_within_reference":
            reference = str(raw.get("reference", "") or "")
            ok, detail = self._extent_within(target, reference)
            return ValidationResult(kind, target, ok, detail)
        if kind == "raster_not_constant":
            info = self._raster_range(target)
            ok = info is not None and info[0] is not None and info[1] is not None and info[0] != info[1]
            return ValidationResult(kind, target, ok, f"range={info}")
        if kind == "image_not_blank":
            return self._image_check(target)
        if kind == "aprx_map_check":
            return self._aprx_map_check(target, raw)
        if kind == "map_layout_ok":
            # 版面体检：图名居中/图例不压数据/比例尺整数刻度/元素不越界（制图交付的硬指标）
            try:
                from ..cartography import qc as _qc

                verdict = _qc.check_layout(
                    target,
                    spec=raw.get("spec") or None,
                    image_path=str(raw.get("image") or ""),
                    min_font_pt=float(raw.get("min_font_pt") or 7.0),
                )
                return ValidationResult(kind, target, bool(verdict.get("ok")), _qc.summarize(verdict))
            except Exception as exc:  # pragma: no cover - 依赖 arcpy
                return ValidationResult(kind, target, False, f"版面体检失败: {str(exc)[:120]}")
        if kind == "code_ok":
            ok = bool(getattr(result, "ok", False))
            return ValidationResult(kind, target, ok, "")
        if kind == "result_key_truthy":
            key = str(raw.get("key", "") or "")
            payload = getattr(result, "result", None)
            value = payload.get(key) if isinstance(payload, dict) else None
            return ValidationResult(kind, target, bool(value), f"{key}={value!r}")
        if kind == "result_file_exists":
            key = str(raw.get("key", "") or "output")
            payload = getattr(result, "result", None)
            raw_path = payload.get(key) if isinstance(payload, dict) else None
            exists = bool(raw_path) and Path(str(raw_path)).exists()
            return ValidationResult(kind, target, exists, f"{key}={raw_path}")
        return ValidationResult(kind, target, True, "未知断言类型，已跳过")

    def _field_value_counts(
        self, path: str, field: str, expected: dict[str, int]
    ) -> tuple[dict[str, int] | None, dict[str, tuple[int, int]]]:
        """统计字段各取值个数，返回 (实际计数, {取值: (实际, 期望)}) 不符项。"""
        if not field:
            return None, {}
        try:
            import arcpy  # type: ignore

            if not arcpy.Exists(path):
                return None, {}
            fields = {f.name for f in arcpy.ListFields(path)}
            if field not in fields:
                return None, {}
            counts: dict[str, int] = {}
            with arcpy.da.SearchCursor(path, [field]) as cur:
                for (value,) in cur:
                    key = "" if value is None else str(value)
                    counts[key] = counts.get(key, 0) + 1
            bad = {
                key: (counts.get(key, 0), want)
                for key, want in expected.items()
                if counts.get(key, 0) != want
            }
            return counts, bad
        except Exception:
            return None, {}

    def _field_values(self, path: str, field: str, *, allowed_set: set[str]) -> tuple[set | None, set]:
        """Read all values of a field via the kernel; return (values, unexpected)."""
        if self.code_runner is None or not field:
            return None, {"字段为空或无执行器"}
        code = (
            "import arcpy\n"
            f"_p = {path!r}\n_f = {field!r}\n"
            "_vals = set()\n"
            "with arcpy.da.SearchCursor(_p, [_f]) as cur:\n"
            "    for row in cur:\n"
            "        if row[0] is not None:\n"
            "            _vals.add(str(row[0]))\n"
            "set_result({'values': sorted(_vals)})\n"
        )
        outcome = self.code_runner.run(code, timeout=180)
        if outcome.ok and isinstance(outcome.result, dict):
            vals = {str(v) for v in outcome.result.get("values", [])}
            unexpected = {v for v in vals if v not in allowed_set}
            return vals, unexpected
        return None, {str(outcome.error)[:80] if outcome.error else "读取失败"}

    def _field_sum(self, path: str, field: str) -> float | None:
        """Sum a numeric field via the kernel."""
        return self._field_stat(path, field, "sum")

    def _field_stat(self, path: str, field: str, kind: str) -> float | None:
        """Compute a statistic of a field via the kernel.

        kind: sum | min | max | mean | count | distinct | null_rate
        """
        if self.code_runner is None or not field:
            return None
        code = (
            "import arcpy, statistics as _st\n"
            f"_p = {path!r}\n_f = {field!r}\n"
            "_vals = []\n_n = 0\n_null = 0\n_distinct = set()\n"
            "with arcpy.da.SearchCursor(_p, [_f]) as cur:\n"
            "    for row in cur:\n"
            "        _n += 1\n"
            "        _v = row[0]\n"
            "        if _v is None:\n"
            "            _null += 1\n"
            "            continue\n"
            "        _distinct.add(str(_v))\n"
            "        try:\n"
            "            _vals.append(float(_v))\n"
            "        except (TypeError, ValueError):\n"
            "            pass\n"
            "set_result({'sum': sum(_vals), 'min': min(_vals) if _vals else None,\n"
            "            'max': max(_vals) if _vals else None,\n"
            "            'mean': (_st.fmean(_vals) if _vals else None),\n"
            "            'count': _n, 'null': _null, 'distinct': len(_distinct),\n"
            "            'null_rate': (_null / _n) if _n else None})\n"
        )
        outcome = self.code_runner.run(code, timeout=180)
        if outcome.ok and isinstance(outcome.result, dict):
            value = outcome.result.get(kind)
            return float(value) if value is not None else None
        return None

    def _raster_stats(self, path: str) -> dict[str, float] | None:
        """Exact raster statistics (NoData-aware) via the kernel."""
        if self.code_runner is None:
            return None
        code = (
            "import arcpy, numpy as np\n"
            f"_r = arcpy.Raster({path!r})\n"
            "_a = arcpy.RasterToNumPyArray(_r, nodata_to_value=np.nan).astype('float64')\n"
            "_valid = int(np.count_nonzero(~np.isnan(_a)))\n"
            "set_result({'sum': float(np.nansum(_a)),\n"
            "            'min': float(np.nanmin(_a)) if _valid else None,\n"
            "            'max': float(np.nanmax(_a)) if _valid else None,\n"
            "            'mean': float(np.nanmean(_a)) if _valid else None,\n"
            "            'count': _valid, 'nodata': int(_a.size - _valid)})\n"
        )
        outcome = self.code_runner.run(code, timeout=300)
        if outcome.ok and isinstance(outcome.result, dict):
            return {k: v for k, v in outcome.result.items() if v is not None}
        return None

    def _extent_within(self, target: str, reference: str) -> tuple[bool, str]:
        """True when target's extent is inside reference's extent (small tolerance)."""
        if self.code_runner is None or not reference:
            return False, "缺少 reference 参数"
        code = (
            "import arcpy\n"
            f"_t = {target!r}\n_r = {reference!r}\n"
            "if not arcpy.Exists(_t) or not arcpy.Exists(_r):\n"
            "    raise ValueError('目标或参考数据不存在')\n"
            "_te = arcpy.Describe(_t).extent\n_re = arcpy.Describe(_r).extent\n"
            "_tol = max(_re.width, _re.height) * 1e-6\n"
            "_ok = (_te.XMin >= _re.XMin - _tol and _te.YMin >= _re.YMin - _tol\n"
            "       and _te.XMax <= _re.XMax + _tol and _te.YMax <= _re.YMax + _tol)\n"
            "set_result({'ok': bool(_ok), 'target': [_te.XMin,_te.YMin,_te.XMax,_te.YMax],\n"
            "            'reference': [_re.XMin,_re.YMin,_re.XMax,_re.YMax]})\n"
        )
        outcome = self.code_runner.run(code, timeout=180)
        if outcome.ok and isinstance(outcome.result, dict):
            payload = outcome.result
            return bool(payload.get("ok")), f"target={payload.get('target')} reference={payload.get('reference')}"
        return False, f"范围比较失败: {(outcome.error or {}).get('message', '')[:80]}"

    def _overlap_pairs(self, path: str, max_allowed: int) -> tuple[int | None, bool]:
        """Count polygon pairs that overlap/duplicate (early-exit at max_allowed)."""
        if self.code_runner is None:
            return None, False
        code = (
            "import arcpy\n"
            f"_p = {path!r}\n"
            f"_limit = {int(max_allowed)}\n"
            "_oids = [(o, s) for o, s in arcpy.da.SearchCursor(_p, ['OID@', 'SHAPE@'])]\n"
            "_count = 0\n"
            "_truncated = False\n"
            "for _i in range(len(_oids)):\n"
            "    _gi = _oids[_i][1]\n"
            "    for _j in range(_i + 1, len(_oids)):\n"
            "        _gj = _oids[_j][1]\n"
            "        if _gi.equals(_gj) or _gi.overlaps(_gj) or _gi.contains(_gj) or _gi.within(_gj):\n"
            "            _count += 1\n"
            "            if _count > _limit:\n"
            "                _truncated = True\n"
            "                break\n"
            "    if _truncated:\n"
            "        break\n"
            "set_result({'overlaps': _count, 'truncated': _truncated})\n"
        )
        outcome = self.code_runner.run(code, timeout=300)
        if outcome.ok and isinstance(outcome.result, dict):
            return int(outcome.result.get("overlaps", 0)), bool(outcome.result.get("truncated"))
        return None, False

    def _extent_check(self, path: str, reference: str, raw: dict[str, Any], result: Any) -> ValidationResult:
        """Compare extents against a reference layer.

        This catches the classic 'DefineProjection instead of Project' failure:
        relabeled data keeps its original (degree-scale) extent while a truly
        projected layer lands in meter-scale GK coordinates.
        """
        if self.code_runner is None:
            return ValidationResult("extent_matches_reference", path, False, "无内核")
        tolerance = float(raw.get("tolerance", 1.0) or 1.0)
        code = (
            "import arcpy\n"
            f"_out = {path!r}\n_ref = {reference!r}\n"
            "def _ext(p):\n"
            "    d = arcpy.Describe(p)\n"
            "    e = d.extent\n"
            "    return [e.XMin, e.YMin, e.XMax, e.YMax]\n"
            "set_result({'out': _ext(_out), 'ref': _ext(_ref)})\n"
        )
        outcome = self.code_runner.run(code, timeout=180)
        payload = outcome.result if outcome.ok and isinstance(outcome.result, dict) else None
        if not payload:
            return ValidationResult("extent_matches_reference", path, False, f"范围读取失败: {outcome.error}")
        diffs = [abs(a - b) for a, b in zip(payload["out"], payload["ref"])]
        worst = max(diffs)
        ok = worst <= tolerance
        detail = f"max_diff={worst:.3f} (tol={tolerance}) out={payload['out']} ref={payload['ref']}"
        return ValidationResult("extent_matches_reference", path, ok, detail)

    def _feature_count(self, path: str) -> int | None:
        record = self._describe(path)
        return record.get("feature_count") if record else None

    def _crs_wkid(self, path: str) -> int | None:
        record = self._describe(path)
        return record.get("wkid") if record else None

    def _fields(self, path: str) -> list[str]:
        record = self._describe(path)
        return list(record.get("fields") or []) if record else []

    def _raster_range(self, path: str) -> tuple[float | None, float | None] | None:
        record = self._describe(path)
        if not record:
            return None
        return record.get("raster_min"), record.get("raster_max")

    def _describe(self, path: str, *, attempts: int = 2) -> dict[str, Any] | None:
        """Describe one path via the kernel (best effort)，失败再退到进程内 arcpy。

        Uses ``arcpy.Exists`` rather than ``Path.exists`` so GDB feature
        classes/tables (which are not filesystem paths) work too.

        刚被看门狗杀掉内核时，GDB 可能仍被残留进程占着，``arcpy.Exists`` 会瞬时为 False，
        因此这里重试一次再下结论，避免把已有数据判成"不存在"。
        """
        record: dict[str, Any] | None = None
        for index in range(max(1, attempts)):
            record = self._describe_once(path)
            if record and record.get("exists"):
                return record
            if index + 1 < max(1, attempts):
                time.sleep(1.0)
        return record

    def _describe_once(self, path: str) -> dict[str, Any] | None:
        if self.code_runner is None:
            return self._describe_in_process(path)
        code = (
            "import arcpy, json\n"
            f"_p = {path!r}\n"
            "_rec = {'exists': arcpy.Exists(_p)}\n"
            "if arcpy.Exists(_p):\n"
            "    _d = arcpy.Describe(_p)\n"
            "    _sr = getattr(_d, 'spatialReference', None)\n"
            "    _rec['wkid'] = getattr(_sr, 'factoryCode', None) if _sr else None\n"
            "    _rec['crs_name'] = getattr(_sr, 'name', '') if _sr else ''\n"
            "    _rec['geom_type'] = getattr(_d, 'shapeType', '') or ''\n"
            "    try:\n"
            "        _rec['feature_count'] = int(arcpy.management.GetCount(_p)[0])\n"
            "    except Exception:\n"
            "        _rec['feature_count'] = None\n"
            "    try:\n"
            "        _rec['fields'] = [f.name for f in arcpy.ListFields(_p)]\n"
            "    except Exception:\n"
            "        _rec['fields'] = []\n"
            "    try:\n"
            "        _rec['raster_min'] = float(arcpy.management.GetRasterProperties(_p, 'MINIMUM').getOutput(0))\n"
            "        _rec['raster_max'] = float(arcpy.management.GetRasterProperties(_p, 'MAXIMUM').getOutput(0))\n"
            "    except Exception:\n"
            "        _rec['raster_min'] = None\n"
            "        _rec['raster_max'] = None\n"
            "set_result(_rec)\n"
        )
        outcome = self.code_runner.run(code, timeout=180)
        if outcome.ok and isinstance(outcome.result, dict):
            return outcome.result
        # 内核忙/卡死时不能就此判定“不存在”（那样会误报产出缺失，把 agent 逼去重建数据）。
        # 退回本进程内直接调 arcpy（断言评估通常在带 arcpy 的主进程里跑）。
        return self._describe_in_process(path)

    @staticmethod
    def _describe_in_process(path: str) -> dict[str, Any] | None:
        """不经内核、在当前进程内用 arcpy 描述数据（内核不可用时的兼容退路）。"""
        try:
            import arcpy  # type: ignore
        except Exception:
            return None
        try:
            record: dict[str, Any] = {"exists": bool(arcpy.Exists(path))}
            if not record["exists"]:
                return record
            desc = arcpy.Describe(path)
            sr = getattr(desc, "spatialReference", None)
            record["wkid"] = getattr(sr, "factoryCode", None) if sr else None
            record["crs_name"] = getattr(sr, "name", "") if sr else ""
            record["geom_type"] = getattr(desc, "shapeType", "") or ""
            try:
                record["feature_count"] = int(arcpy.management.GetCount(path)[0])
            except Exception:
                record["feature_count"] = None
            try:
                record["fields"] = [f.name for f in arcpy.ListFields(path)]
            except Exception:
                record["fields"] = []
            try:
                record["raster_min"] = float(arcpy.management.GetRasterProperties(path, "MINIMUM").getOutput(0))
                record["raster_max"] = float(arcpy.management.GetRasterProperties(path, "MAXIMUM").getOutput(0))
            except Exception:
                record["raster_min"] = None
                record["raster_max"] = None
            return record
        except Exception:
            return None

    def _aprx_map_check(self, path: str, raw: dict[str, Any]) -> ValidationResult:
        """核验 .aprx 工程：渲染器类型、各类别配色、布局四要素是否都落盘。

        断言参数：
          renderer: 期望渲染器类型（UniqueValueRenderer / GraduatedColorsRenderer）
          colors: 期望配色，形如 "标杆社区=#1F77B4;需整改社区=#2CA02C"
          elements: 必须存在的布局要素（子集：图名/图例/比例尺/指北针）
        """
        target = Path(path)
        if not target.exists():
            return ValidationResult("aprx_map_check", path, False, "工程文件不存在")
        try:
            from arcpy import mp  # type: ignore

            project = mp.ArcGISProject(str(target))
        except Exception as exc:
            return ValidationResult("aprx_map_check", path, False, f"无法打开工程: {str(exc)[:120]}")

        problems: list[str] = []
        want_renderer = str(raw.get("renderer", "") or "")
        want_colors = _parse_color_expectations(str(raw.get("colors", "") or ""))
        want_elements = [str(e) for e in (raw.get("elements") or [])]

        found_colors: dict[str, str] = {}
        renderer_types: list[str] = []
        for map_obj in project.listMaps():
            for layer in map_obj.listLayers():
                if layer.isGroupLayer:
                    continue
                try:
                    renderer = layer.symbology.renderer
                except Exception:
                    continue
                renderer_types.append(type(renderer).__name__)
                if type(renderer).__name__ == "UniqueValueRenderer":
                    for group in renderer.groups:
                        for item in group.items:
                            vals = list(item.values) if item.values else []
                            key = ""
                            if vals:
                                v0 = vals[0]
                                key = str(v0[0]) if isinstance(v0, (list, tuple)) else str(v0)
                            found_colors[key] = _symbol_hex(item.symbol)

        if want_renderer and want_renderer not in renderer_types:
            problems.append(f"渲染器为 {renderer_types or ['(无图层)']}，期望 {want_renderer}")
        for key, expected_hex in want_colors.items():
            actual = found_colors.get(key, "")
            if actual.upper() != expected_hex.upper():
                problems.append(f"{key} 配色={actual or '(未找到)'} 期望 {expected_hex}")

        if want_elements:
            element_names: list[str] = []
            element_types: list[str] = []
            for layout in project.listLayouts():
                for element in layout.listElements():
                    element_names.append(str(element.name))
                    element_types.append(str(element.type).upper())
            joined = " ".join(element_names).lower()
            surrounds = sum(1 for t in element_types if "MAPSURROUND" in t)
            checks = {
                "图名": any("TEXT" in t for t in element_types),
                "图例": any("LEGEND" in t for t in element_types),
                "比例尺": ("scale" in joined or "比例尺" in " ".join(element_names)) or surrounds >= 2,
                "指北针": ("north" in joined or "指北针" in " ".join(element_names)) or surrounds >= 2,
                "地图框": any("MAPFRAME" in t for t in element_types),
            }
            for name in want_elements:
                if not checks.get(name, False):
                    problems.append(f"布局缺少「{name}」")

        detail = "；".join(problems[:4])
        if not problems:
            detail = f"渲染器={renderer_types[:1]} 配色={found_colors}"
        return ValidationResult("aprx_map_check", path, not problems, detail)

    def _image_check(self, path: str) -> ValidationResult:
        target = Path(path)
        if not target.exists():
            return ValidationResult("image_not_blank", path, False, "文件不存在")
        try:
            from PIL import Image  # type: ignore
            import numpy as np  # type: ignore

            with Image.open(target) as img:
                arr = np.asarray(img.convert("L"), dtype="float32")
            std = float(arr.std())
            return ValidationResult("image_not_blank", path, std > 3.0, f"std={std:.2f}")
        except Exception as exc:
            return ValidationResult("image_not_blank", path, True, f"跳过: {exc}")

    # ------------------------------------------------------------------- save
    def save(self, recipe: Recipe, *, user: bool = True) -> str:
        """Persist a recipe (distillation target)."""
        directory = self.user_dir if user and self.user_dir else self.builtin_dir
        if directory is None:
            raise ValueError("no recipe directory configured")
        path = directory / f"{recipe.id}.yaml"
        recipe.to_yaml(path)
        self._recipes[recipe.id] = recipe
        return str(path)


def _symbol_hex(symbol: Any) -> str:
    """取符号颜色的 #RRGGBB（ArcGIS 返回 Color 对象或 dict 两种形态）。"""
    color = getattr(symbol, "color", None)
    if color is None:
        return ""
    rgb = getattr(color, "RGB", None)
    if rgb is None and isinstance(color, dict):
        rgb = color.get("RGB")
    if not rgb:
        return ""
    try:
        return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    except Exception:
        return ""


def _parse_color_expectations(text: str) -> dict[str, str]:
    """解析 "值=#RRGGBB;值2=#RRGGBB" 形式的期望配色。"""
    out: dict[str, str] = {}
    for chunk in str(text or "").split(";"):
        if "=" in chunk:
            key, value = chunk.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def _resolve(text: str, params: dict[str, Any]) -> str:
    """Resolve {param} slots inside assertion values."""
    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        value = params.get(key)
        return "" if value is None else str(value)

    return _SLOT.sub(_sub, text)

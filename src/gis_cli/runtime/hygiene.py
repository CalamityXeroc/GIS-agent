# -*- coding: utf-8 -*-
"""交付卫生检查：识别过程数据里的"重做残留"与不该交付的中间文件。

背景（第 14 届基准实测）
------------------------
任务二b/三a 的 ``temp_data.gdb`` 里堆了 **4 份重复的 DEM 拼接中间件**
（``dem_mosaic_wgs84 / dem_mosaic_utm30 / dem_mosaic / dem_mosaic_utm``），
过程数据 384 MB vs 结果数据 17 MB —— **占交付量的 96%**，而考卷明确要求
"提交最终成果时，请从数据库中删除其他非必要的数据"。

模块分两层，便于单测与复用：
- ``dir_inventory()``：IO 层，列交付目录下的 GDB（条目+体积）与散文件（内核优先，进程内兜底）
- ``hygiene_report()``：纯逻辑层，输入 inventory 给出 (problems, warnings, detail)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

#: 交付目录里不该出现的过程文件后缀
DEFAULT_FORBID_SUFFIXES: tuple[str, ...] = (".pkl", ".tmp", ".bak")


def dir_inventory(code_runner: Any, target: str) -> tuple[dict | None, str]:
    """列出交付目录下的 GDB（含条目与体积）与散文件。

    GDB 条目需要 arcpy（要素类/栅格/表）；内核优先，失败退进程内 arcpy。
    返回 ``(inventory, error)``；``inventory`` 为 None 表示失败。
    """
    if code_runner is not None:
        lf = chr(10)
        code = lf.join(
            [
                "import arcpy, os",
                f"_t = {target!r}",
                "_out = {}",
                "_files = []",
                "if os.path.isdir(_t):",
                "    for _e in sorted(os.listdir(_t)):",
                "        _p = os.path.join(_t, _e)",
                "        if _e.lower().endswith('.gdb') and os.path.isdir(_p):",
                "            arcpy.env.workspace = _p",
                "            _items = (arcpy.ListFeatureClasses() or []) + (arcpy.ListRasters() or []) + (arcpy.ListTables() or [])",
                "            _size = 0",
                "            for _root, _dirs, _fs in os.walk(_p):",
                "                for _f in _fs:",
                "                    try:",
                "                        _size += os.path.getsize(os.path.join(_root, _f))",
                "                    except OSError:",
                "                        pass",
                "            _out[_e] = {'items': list(_items), 'bytes': _size}",
                "        elif os.path.isfile(_p):",
                "            _files.append(_e)",
                "set_result({'gdb': _out, 'files': _files})",
            ]
        )
        try:
            outcome = code_runner.run(code, timeout=300)
        except Exception as exc:  # pragma: no cover - 运行器异常
            outcome = None
            logger.debug("inventory via runner failed: %s", exc)
        payload = None
        if outcome is not None and getattr(outcome, "ok", False) and isinstance(outcome.result, dict):
            payload = outcome.result
        if isinstance(payload, dict) and "gdb" in payload:
            return payload, ""
        if outcome is not None and not getattr(outcome, "ok", True):
            return None, f"内核执行失败: {str(getattr(outcome, 'error', ''))[:160]}"

    try:
        gdb: dict[str, dict] = {}
        files: list[str] = []
        arcpy = None
        try:
            import arcpy  # type: ignore
        except Exception:
            arcpy = None
        if not os.path.isdir(target):
            return {"gdb": {}, "files": []}, ""
        for entry in sorted(os.listdir(target)):
            path = os.path.join(target, entry)
            if entry.lower().endswith(".gdb") and os.path.isdir(path):
                items: list[str] = []
                if arcpy is not None:
                    arcpy.env.workspace = path
                    items = (
                        (arcpy.ListFeatureClasses() or [])
                        + (arcpy.ListRasters() or [])
                        + (arcpy.ListTables() or [])
                    )
                size = 0
                for root, _dirs, fs in os.walk(path):
                    for f in fs:
                        try:
                            size += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            pass
                gdb[entry] = {"items": list(items), "bytes": size}
            elif os.path.isfile(path):
                files.append(entry)
        return {"gdb": gdb, "files": files}, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:200]}"


def hygiene_report(
    inventory: dict,
    *,
    max_duplicate_families: int = 0,
    forbid_suffixes: list[str] | None = None,
    ratio_limit: float = 0.0,
) -> tuple[list[str], list[str], str]:
    """纯逻辑：从 inventory 得出 (problems, warnings, detail)。

    - **重做残留**：``temp*.gdb`` 里同一基础名派生出多个变体
      （如 ``dem_mosaic`` + ``dem_mosaic_wgs84`` + ``dem_mosaic_utm`` + ``dem_mosaic_utm30``）。
      只查 temp 库——结果库里 ``city_pop_2010 / city_pop_2020`` 这类同前缀是合理的多期成果。
    - **禁止的过程文件**：交付目录下 ``.pkl/.tmp/.bak`` 之类中间文件。
    - **体积比**：temp 总量远大于 result 总量时只给警告（DEM 拼接这类天然偏大）。
    """
    problems: list[str] = []
    warnings_out: list[str] = []
    gdbs: dict[str, dict] = inventory.get("gdb") or {}
    files: list[str] = [str(f) for f in (inventory.get("files") or [])]

    for suffix in forbid_suffixes if forbid_suffixes is not None else list(DEFAULT_FORBID_SUFFIXES):
        hit = [f for f in files if f.lower().endswith(str(suffix).lower())]
        if hit:
            problems.append(f"交付目录存在过程文件 {'、'.join(hit[:4])}（建议删除或移入过程库）")

    dup_families = 0
    dup_messages: list[str] = []
    for name, info in gdbs.items():
        if not name.lower().startswith("temp"):
            continue
        for base, variants in _redo_residue(info.get("items") or []).items():
            dup_families += 1
            dup_messages.append(
                f"{name} 存在重做残留（{base} 派生出 {'、'.join(variants[:6])}）"
            )
    if dup_families > max_duplicate_families:
        problems.extend(dup_messages)

    if ratio_limit and ratio_limit > 0:
        temp_bytes = sum(v.get("bytes") or 0 for k, v in gdbs.items() if k.lower().startswith("temp"))
        result_bytes = sum(v.get("bytes") or 0 for k, v in gdbs.items() if k.lower().startswith("result"))
        if temp_bytes and result_bytes and temp_bytes / max(1.0, result_bytes) > ratio_limit:
            warnings_out.append(
                f"过程库体积（{temp_bytes / 1e6:.0f} MB）是结果库（{result_bytes / 1e6:.0f} MB）的 "
                f"{temp_bytes / result_bytes:.1f} 倍，交付前建议精简非必要中间件"
            )

    parts = [f"GDB={len(gdbs)}"]
    for name, info in list(gdbs.items())[:4]:
        parts.append(f"{name}:{(info.get('bytes') or 0) / 1e6:.0f}MB/{len(info.get('items') or [])}项")
    if files:
        parts.append(f"散文件={len(files)}")
    if not gdbs and not files:
        parts.append("目录为空")
    return problems, warnings_out, " ".join(parts)


def _redo_residue(items: list, *, min_family_size: int = 3) -> dict[str, list[str]]:
    """找出“重做残留”：一条名称是另一条的前缀（下划线分段），且族内变体足够多。

    判据演进（第 14 届基准两次误报后定稿）：
    - ``dem_mosaic`` / ``dem_mosaic_wgs84`` / ``dem_mosaic_utm`` / ``dem_mosaic_utm30``
      —— 同一东西被重做多次，基础名是其余名字的前缀且变体 ≥2 个 → 真残留 ✓
    - ``tmp_idw_train`` / ``tmp_idw_mask`` —— 同前缀但用途不同，互不为前缀 → 不报 ✓
    - ``tmp_idw2`` / ``tmp_idw2_mask`` —— 前缀关系成立但只有 1 个变体（“结果栅格 + 掩膜”
      在工作流里很常见）→ 族内成员不足，不报 ✓

    返回 ``{基础名: [派生名, ...]}``（仅保留达到 ``min_family_size`` 的族）。
    """
    tokens = [(str(item), str(item).lower().split("_")) for item in items if str(item).strip()]
    residue: dict[str, list[str]] = {}
    for index, (name_a, tokens_a) in enumerate(tokens):
        for name_b, tokens_b in tokens[index + 1 :]:
            if len(tokens_a) < len(tokens_b) and tokens_b[: len(tokens_a)] == tokens_a:
                residue.setdefault(name_a, []).append(name_b)
            elif len(tokens_b) < len(tokens_a) and tokens_a[: len(tokens_b)] == tokens_b:
                residue.setdefault(name_b, []).append(name_a)
    return {
        base: variants
        for base, variants in residue.items()
        if len(variants) >= max(2, int(min_family_size) - 1)
    }


def summarize(problems: list[str], warnings_out: list[str]) -> str:
    """把结论拼成一句人可读文本（空则返回空串）。"""
    bits = list(problems) + [f"警告: {w}" for w in warnings_out]
    return "；".join(bits)


__all__ = [
    "DEFAULT_FORBID_SUFFIXES",
    "dir_inventory",
    "hygiene_report",
    "summarize",
]
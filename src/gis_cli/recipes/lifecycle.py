# -*- coding: utf-8 -*-
"""候选配方的生命周期：门禁 → 候选目录 → 显式采纳。

为什么需要这一层：蒸馏出来的配方是模型对一次成功运行的**泛化猜测**，
不是验证过的方法论。最常见的两种坏配方是

1. 把任务特定的绝对路径写死（``E:\\...\\benchmark\\runs\\20261021_xxx\\output\\a.gdb``）
   —— 换台机器、换个任务立刻失效；
2. 参数槽位写错（代码里用 ``{{input}}`` 但 params 里没有，或反过来声明了
   一堆没用到的参数）—— 调用时才炸，而且报错指向配方内部，很难查。

两者都能用**纯静态检查**挡住，不需要跑 GIS。所以候选先进
``.gis_agent/recipes/candidates/``，采纳（进 ``workspace/recipes/``，
下次 ``build_engine`` 自动加载）是显式动作。
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import WORKSPACE_SENTINEL, _RAW_SLOT, _SLOT, Recipe

logger = logging.getLogger(__name__)

#: 渲染后代码里不该出现的路径特征。命中即判失败——写死路径的配方不可能复用。
_HARDCODED_PATH_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[A-Za-z]:\\\\"), "Windows 绝对路径"),
    (re.compile(r"[A-Za-z]:/"), "Windows 绝对路径（正斜杠）"),
    (re.compile(r"(?<![\w.])/(?:home|mnt|Users|opt|srv|var)/"), "Unix 绝对路径"),
    (re.compile(r"benchmark/runs/"), "基准运行目录"),
    (re.compile(r"benchmark\\\\runs\\\\"), "基准运行目录"),
)

#: 配方 id 的合法形态（小写蛇形）。带空格/中文/大写的 id 会让 run_recipe 很难用。
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass
class GateCheck:
    """一条门禁检查的结论。"""

    name: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def _same_content(a: Recipe, b: Recipe) -> bool:
    """配方内容是否一致，**忽略采纳时才改的元数据**。

    采纳会把 ``version`` 从 ``0.1-draft`` 升到 ``1.0``、给 ``provenance``
    追加 ``| adopted``。直接比 ``to_dict()`` 会让“已采纳过的那份”与候选不等，
    于是重复采纳时又落一个 ``-2`` 副本（实测）。
    """
    da, db = a.to_dict(with_code=True), b.to_dict(with_code=True)
    for key in ("version", "provenance"):
        da.pop(key, None)
        db.pop(key, None)
    return da == db


@dataclass
class GateResult:
    """一次门禁评估的整体结论。"""
    ok: bool
    checks: list[GateCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [c.to_dict() for c in self.checks],
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        bad = [c for c in self.checks if not c.ok]
        if self.ok:
            text = f"门禁通过（{len(self.checks)} 项）"
        else:
            text = "门禁未通过：" + "；".join(f"{c.name}({c.detail})" for c in bad[:3])
        if self.warnings:
            text += "；警告: " + "、".join(self.warnings[:3])
        return text


def _slots_in(text: str) -> set[str]:
    """提取 ``{{x}}`` 与 ``@@x@@`` 两种槽位的名字。"""
    return set(_SLOT.findall(text or "")) | set(_RAW_SLOT.findall(text or ""))


def _path_needles(path: str | Path) -> list[str]:
    """一条路径在源码里可能出现的几种写法（反斜杠可能被转义一层）。

    ``repr(r"C:\\a\\b")`` 得到 ``'C:\\\\a\\\\b'``，所以只比原串会漏掉
    “用 repr 拼进代码”这种情况。
    """
    raw = str(path)
    return [
        raw,
        raw.replace("\\", "\\\\"),  # 反斜杠被转义一层
        raw.replace("\\", "/"),
        raw.replace("\\\\", "/"),
    ]


def _assertion_slots(assertions: list[dict[str, Any]]) -> set[str]:
    """断言里所有字符串值引用的槽位（path/target/value 以及自定义键）。"""
    found: set[str] = set()

    def _walk(value: Any) -> None:
        if isinstance(value, str):
            found.update(_slots_in(value))
        elif isinstance(value, dict):
            for item in value.values():
                _walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _walk(item)

    _walk(assertions or [])
    return found


def _field_text(recipe: Recipe, name: str) -> str:
    """把 steps/pitfalls/examples 这类列表字段拼成一段文本。"""
    value = getattr(recipe, name, None)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return ""


def sanitize_distilled(recipe: Recipe, *, source_workspace: str | Path = "") -> list[str]:
    """把蒸馏结果里常见的“不可移植”写法改成可移植的，返回改动说明。

    实测（14 届 task2a 蒸馏）两个必顶：

    1. **双重引号**：模型写 ``ws = r"'{{workspace}}'"`` —— 而 ``{{param}}``
       渲染时已经是带引号的 Python 字面量，再包一层就变成 ``r"'E:\\...'"``，
       渲染结果直接语法错。
    2. **默认值是源工作区的绝对路径**：参数本身合理（workspace 确实该是个
       参数），但 default 不能是本机路径，否则换台机器回放/调用就废。
    """
    notes: list[str] = []

    # 1) 槽位外层多余的引号
    quoted_slot = re.compile(r"(['\"])[ \t]*(\{\{\s*([A-Za-z_]\w*)\s*\}\}|@@\s*([A-Za-z_]\w*)\s*@@)[ \t]*\1")
    new_template, count = quoted_slot.subn(lambda m: m.group(2), recipe.code_template or "")
    if count:
        recipe.code_template = new_template
        notes.append(f"去掉 {count} 处槽位外层的多余引号（{{{{param}}}} 自带引号）")

    # 2) 参数默认值里的源工作区路径 → 换成工作区哨兵（运行时自动填真实路径）
    needles = _path_needles(source_workspace) if source_workspace else []
    for name, spec in (recipe.params or {}).items():
        default = spec.default
        if not isinstance(default, str) or not default:
            continue
        replaced = default
        for needle in needles:
            if needle and needle in replaced:
                replaced = replaced.replace(needle, "").lstrip("\\/")
                break
        if replaced != default:
            # 原来是“整个值就是源工作区” → 哨兵；原来是子路径 → 相对形式
            spec.default = WORKSPACE_SENTINEL if not replaced else replaced
            notes.append(f"参数 {name} 的默认值去掉源工作区前缀 → {spec.default!r}")
    return notes


class RecipeLifecycle:
    """候选配方的存/查/采纳，以及静态门禁。"""

    def __init__(self, library: Any, workspace: str | Path, *, code_runner: Any = None):
        self.library = library
        self.workspace = Path(workspace)
        self.code_runner = code_runner

    # ------------------------------------------------------------------ paths
    @property
    def candidate_dir(self) -> Path:
        return self.workspace / ".gis_agent" / "recipes" / "candidates"

    def _candidate_yaml(self, recipe_id: str) -> Path:
        return self.candidate_dir / f"{recipe_id}.yaml"

    def _candidate_meta(self, recipe_id: str) -> Path:
        return self.candidate_dir / f"{recipe_id}.meta.json"

    # ------------------------------------------------------------------ gate
    def static_gate(self, recipe: Recipe, *, source_workspace: str = "") -> GateResult:
        """不跑 GIS 的静态门禁：字段、渲染、语法、硬编码路径、槽位一致。

        这一步的目标不是"证明配方正确"，而是"证明它至少不是一个自相矛盾、
        换台机器就废的空壳"。真正的正确性由 replay（回放验证）负责。
        """
        checks: list[GateCheck] = []
        warnings: list[str] = []

        # 1) 必填字段
        missing = [name for name in ("id", "name", "code_template") if not str(getattr(recipe, name, "") or "").strip()]
        checks.append(
            GateCheck(
                "required_fields",
                not missing,
                "缺少 " + "、".join(missing) if missing else "id/name/code_template 齐全",
            )
        )
        if not _ID_RE.match(recipe.id or ""):
            checks.append(
                GateCheck(
                    "id_format",
                    False,
                    f"id={recipe.id!r} 不是小写蛇形（run_recipe 调用会很难用）",
                )
            )
        else:
            checks.append(GateCheck("id_format", True, recipe.id))

        # 2) 必填参数都要有 default（否则既无法回放，也无法被 agent 直接调用）
        no_default = [n for n, p in (recipe.params or {}).items() if p.required and p.default is None]
        checks.append(
            GateCheck(
                "params_have_defaults",
                not no_default,
                "必填参数缺默认值: " + "、".join(no_default[:5]) if no_default else "必填参数均有默认值",
            )
        )

        defaults = {name: spec.default for name, spec in (recipe.params or {}).items() if spec.default is not None}

        # 3) 渲染
        rendered = ""
        try:
            rendered = recipe.render(defaults)
            checks.append(GateCheck("renders", True, f"{len(rendered)} 字符"))
        except Exception as exc:
            checks.append(GateCheck("renders", False, f"{type(exc).__name__}: {exc}"))

        # 4) 渲染结果是合法 Python
        if rendered:
            try:
                ast.parse(rendered)
                checks.append(GateCheck("valid_python", True, "ast.parse 通过"))
            except SyntaxError as exc:
                checks.append(GateCheck("valid_python", False, f"第 {exc.lineno} 行: {exc.msg}"))
        else:
            checks.append(GateCheck("valid_python", False, "未渲染出代码"))

        # 5) 硬编码路径（蒸馏配方最常见的死法）
        hits: list[str] = []
        for pattern, label in _HARDCODED_PATH_PATTERNS:
            for match in pattern.finditer(rendered):
                start = max(0, match.start() - 20)
                hits.append(f"{label}: …{rendered[start:match.end() + 40].splitlines()[0][:80]}")
        if source_workspace:
            # 源运行目录字符串本身也禁止。注意两边的反斜杠可能被转义一层：
            # repr() 写进代码时 ``C:\\a\\b`` 会变成 ``C:\\\\a\\\\b``，直接子串匹配会漏。
            needles = _path_needles(source_workspace)
            for needle in needles:
                if needle and needle in rendered:
                    hits.append(f"残留源工作区路径: {needle[:60]}")
                    break
            # 非代码字段也会进 prompt（steps/pitfalls/examples/description），
            # 里面写了本机路径一样是泄漏。
            for field_name in ("description", "steps", "pitfalls", "examples", "preconditions"):
                text = _field_text(recipe, field_name)
                if not text:
                    continue
                for needle in needles:
                    if needle and needle in text:
                        hits.append(f"{field_name} 里残留源工作区路径")
                        break
                for pattern, label in _HARDCODED_PATH_PATTERNS:
                    if pattern.search(text):
                        hits.append(f"{field_name} 里有{label}")
                        break
        checks.append(
            GateCheck(
                "no_hardcoded_paths",
                not hits,
                "；".join(hits[:3]) if hits else "未发现写死的绝对路径",
            )
        )

        # 6) 槽位一致性：代码里用到的必须声明
        declared = set((recipe.params or {}).keys())
        used = _slots_in(recipe.code_template)
        undeclared = sorted(used - declared)
        checks.append(
            GateCheck(
                "slots_declared",
                not undeclared,
                "代码里用了未声明的槽位: " + "、".join(undeclared[:5]) if undeclared else "槽位均已声明",
            )
        )

        # 7) 反向：声明了但没用到 → 警告（多半是模型臆造）
        unused = sorted(declared - used - _assertion_slots(recipe.validation))
        if unused:
            warnings.append("声明但未使用的参数: " + "、".join(unused[:5]))

        # 7b) 参数过多 → 警告。实测蒸馏会把“任务特有常量”也做成参数
        # （某次异常块的经纬度范围、清理开关……），最后 30 个必填参数，
        # 调用方根本填不过来，等于没泛化。
        if len(declared) > 12:
            warnings.append(f"参数过多（{len(declared)} 个），建议把任务特有常量写回代码")

        # 8) 断言存在且引用已声明参数
        if not recipe.validation:
            checks.append(GateCheck("assertions_present", False, "没有任何 validation 断言，无法自证"))
        else:
            bad = sorted(_assertion_slots(recipe.validation) - declared)
            checks.append(
                GateCheck(
                    "assertions_use_declared_params",
                    not bad,
                    "断言引用了未声明的槽位: " + "、".join(bad[:5]) if bad else f"{len(recipe.validation)} 条断言",
                )
            )

        # 9) id 是否与已加载配方冲突（冲突不致命，采纳时会加后缀）
        existing = None
        try:
            existing = self.library.get(recipe.id) if self.library is not None else None
        except Exception:
            existing = None
        if existing is not None:
            warnings.append(f"id 与已有配方冲突（{existing.source_path or existing.id}），采纳时会自动加后缀")

        ok = all(c.ok for c in checks)
        return GateResult(ok=ok, checks=checks, warnings=warnings)

    # -------------------------------------------------------------- candidates
    def save_candidate(
        self,
        recipe: Recipe,
        *,
        provenance: str = "",
        gate: GateResult | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """把配方写进候选目录（同名覆盖），并落一份门禁元数据。"""
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        recipe.provenance = provenance or recipe.provenance or "distilled"
        if not recipe.version or recipe.version == "1.0":
            recipe.version = "0.1-draft"
        yaml_path = self._candidate_yaml(recipe.id)
        recipe.to_yaml(yaml_path)
        meta = {
            "id": recipe.id,
            "name": recipe.name,
            "provenance": recipe.provenance,
            "gate": (gate or self.static_gate(recipe)).to_dict(),
            "yaml": str(yaml_path),
        }
        if extra:
            meta.update(extra)
        self._candidate_meta(recipe.id).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return yaml_path

    def list_candidates(self) -> list[dict[str, Any]]:
        """列出候选（按 meta 里的 gate 结论排序：通过的在前）。"""
        if not self.candidate_dir.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for meta_path in sorted(self.candidate_dir.glob("*.meta.json")):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            data["gate_ok"] = bool((data.get("gate") or {}).get("ok"))
            out.append(data)
        out.sort(key=lambda item: (not item.get("gate_ok"), item.get("id", "")))
        return out

    def load_candidate(self, recipe_id: str) -> Recipe | None:
        path = self._candidate_yaml(recipe_id)
        if not path.exists():
            return None
        try:
            return Recipe.from_yaml(path)
        except Exception as exc:
            logger.warning("candidate %s unreadable: %s", recipe_id, exc)
            return None

    def reject(self, recipe_id: str) -> bool:
        """删掉候选（不影响已采纳的配方）。"""
        removed = False
        for path in (self._candidate_yaml(recipe_id), self._candidate_meta(recipe_id)):
            try:
                path.unlink()
                removed = True
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("reject %s failed: %s", path, exc)
        return removed

    # ------------------------------------------------------------------ replay
    @property
    def replay_dir(self) -> Path:
        return self.workspace / ".gis_agent" / "recipes" / "replay"

    @staticmethod
    def _looks_like_output_param(name: str, value: Any) -> bool:
        """猜一个参数是不是**输出路径**。

        回放要保证不覆盖已交付产物，所以必须把输出路径改到隔离目录。
        但“名字带 output”不等于“值是路径”：实测把 ``output_fc_name``
        （值是 ``animal_points``）也重定向成路径，结果
        ``CreateFeatureclass(gdb, '<路径>')`` 直接报 ERROR 000840。
        所以先要求**值长得像路径**，再看名字。
        """
        if not isinstance(value, str) or not value:
            return False
        norm = value.replace("\\", "/")
        has_sep = "/" in norm
        suffix = Path(norm).suffix.lower()
        # 裸目录名也是路径：蒸馏出来的配方常把输出目录写成 ``output`` 这种
        # 相对目录名——既没有分隔符也没有后缀，按“值长得像路径”会漏判，
        # 于是回放直接放弃（2026-10-22 端到端验证实测）。
        bare_dir = norm.strip("/").lower() in {"output", "outputs", "out", "results", "result"}
        path_like = has_sep or bare_dir or suffix in {
            ".gdb", ".shp", ".tif", ".tiff", ".img", ".md", ".csv",
            ".json", ".aprx", ".jpg", ".jpeg", ".png", ".pdf", ".txt",
        }
        if not path_like:
            return False
        lowered = name.lower()
        return (
            "/output/" in norm.lower()
            or norm.lower().startswith("output/")
            # 不要用裸的 "path"/"gdb" 做关键词：``target_path='data/x.txt'`` 是输入，
            # 误当成输出会把输入改到隔离目录，配方直接跑不起来。
            or any(token in lowered for token in ("output", "out_", "_out", "report"))
        )

    def replay(
        self,
        recipe_id: str,
        *,
        timeout: float = 600.0,
        keep: bool = False,
    ) -> GateResult:
        """门禁 B：用默认参数**在隔离目录**重跑一遍，看自带断言是否全过。

        与静态门禁的分工：静态只能证明“配方自相矛盾/不可移植”，回放才能
        证明“它真的能跑出符合自己断言的结果”。

        不覆盖产物的做法：把输出类参数（默认值里出现 ``output/`` 或参数名
        带 out/report/gdb）重定向到 ``.gis_agent/recipes/replay/<id>/``。
        因为 ``_run_assertions`` 对 path/target/value 也做槽位替换，代码与
        断言会**一起**指向隔离目录，原产物不动。
        """
        checks: list[GateCheck] = []
        warnings: list[str] = []
        recipe = self.load_candidate(recipe_id)
        if recipe is None:
            return GateResult(False, [GateCheck("candidate", False, f"候选不存在: {recipe_id}")], [])

        gate = self.static_gate(recipe, source_workspace=str(self.workspace))
        if not gate.ok:
            return GateResult(False, [GateCheck("static_gate", False, gate.summary())], gate.warnings)
        checks.append(GateCheck("static_gate", True, "静态门禁已通过"))

        runner = self.code_runner or getattr(self.library, "code_runner", None)
        if runner is None:
            return GateResult(False, checks + [GateCheck("runner", False, "没有可用的代码执行器")], warnings)

        # 1) 准备参数：默认值 + 输出类参数重定向到隔离目录
        scratch = self.replay_dir / recipe.id
        if scratch.exists():
            import shutil

            shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True, exist_ok=True)

        params: dict[str, Any] = {}
        redirected: list[str] = []
        for name, spec in (recipe.params or {}).items():
            value = spec.default
            if value is None:
                continue
            if isinstance(value, str) and value == WORKSPACE_SENTINEL:
                params[name] = str(self.workspace)
                continue
            if self._looks_like_output_param(name, value):
                # 保留相对层级（result_data.gdb/xxx → scratch/result_data.gdb/xxx）
                rel = str(value).replace("\\", "/")
                if rel.strip("/").lower() in {"output", "outputs", "out", "results", "result"}:
                    # 裸目录名：直接落到隔离目录根，不要再套一层 output/
                    rel = ""
                else:
                    for prefix in ("output/", "/output/"):
                        if rel.startswith(prefix):
                            rel = rel[len(prefix):]
                            break
                        if prefix in rel:
                            rel = rel.split(prefix, 1)[1]
                            break
                params[name] = str(scratch / rel.lstrip("/")) if rel else str(scratch)
                redirected.append(name)
            else:
                params[name] = value
        if not redirected:
            warnings.append("没识别出输出类参数，回放可能写回原产物（已中止）")
            return GateResult(
                False,
                checks + [GateCheck("output_params", False, "无法识别输出参数，为避免覆盖产物已中止")],
                warnings,
            )
        checks.append(GateCheck("output_params", True, "已重定向: " + "、".join(redirected[:5])))

        # 2) 渲染并执行
        try:
            code = recipe.render(params)
            ast.parse(code)
        except Exception as exc:
            return GateResult(False, checks + [GateCheck("render", False, f"{type(exc).__name__}: {exc}")], warnings)
        outcome = runner.run(code, timeout=timeout, workspace=str(self.workspace))
        if not getattr(outcome, "ok", False):
            err = getattr(outcome, "error", None) or {}
            detail = f"[{err.get('type')}] {str(err.get('message'))[:160]}" if isinstance(err, dict) else str(err)[:160]
            return GateResult(False, checks + [GateCheck("executes", False, detail)], warnings)
        checks.append(GateCheck("executes", True, "代码执行成功"))

        # 3) 跑自带断言（路径也已重定向到隔离目录）
        resolved = recipe.resolve_params(params)
        results = self.library._run_assertions(recipe, resolved, outcome, workspace=str(self.workspace))
        failed = [r for r in results if not r.ok]
        for r in results:
            checks.append(
                GateCheck(
                    f"assert:{r.kind}",
                    r.ok,
                    f"{r.target} {r.detail}"[:200],
                )
            )
        if not results:
            warnings.append("配方没有任何断言，回放无法判定对错")

        if not keep:
            try:
                import shutil

                shutil.rmtree(scratch, ignore_errors=True)
            except Exception:
                pass

        ok = not failed and bool(results)
        return GateResult(ok, checks, warnings)

    # ------------------------------------------------------------------ adopt
    def adopt(self, recipe_id: str, *, force: bool = False) -> tuple[Path | None, str]:
        """把候选提升进 ``workspace/recipes/``（下次 build_engine 自动加载）。

        返回 ``(落盘路径, 说明)``。采纳前重跑一次静态门禁——候选文件在磁盘上
        可能被手改过，不能只信当初的结论。
        """
        recipe = self.load_candidate(recipe_id)
        if recipe is None:
            return None, f"候选不存在: {recipe_id}"
        gate = self.static_gate(recipe, source_workspace=str(self.workspace))
        if not gate.ok:
            return None, f"门禁未通过，拒绝采纳：{gate.summary()}"

        target_dir = getattr(self.library, "user_dir", None) or (self.workspace / "recipes")
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # ① 先处理 **同 id** 冲突：内容一致 → 无需重复采纳；内容不同 → 加后缀。
        # 这一步要在相似度查重之前——同一个配方被采纳两次是正常事。
        final_id = recipe.id
        target = target_dir / f"{final_id}.yaml"
        suffix = 2
        while target.exists() and not force:
            existing = None
            try:
                existing = Recipe.from_yaml(target)
            except Exception:
                existing = None
            if existing is not None and _same_content(existing, recipe):
                return target, "已存在相同配方，无需重复采纳"
            final_id = f"{recipe.id}-{suffix}"
            target = target_dir / f"{final_id}.yaml"
            suffix += 1

        # ② 再查**不同 id** 的近义配方：同一条方法论被反复蒸馏会产出一堆
        #    csv_group_sum_* 变体（实测两个并存）。
        #    只在同 domain 内查——矢量裁切和栅格提取都会叫“按掩膜裁切”，
        #    光看文本相似会把它们误杀（实测 clip_by_mask 被
        #    raster_extract_by_mask 挡下）。
        #    相似度用序列比对而不是关键词：中文没有分词，match_score 对
        #    “按掩膜裁切要素 / 按掩膜裁切入库”这种几乎相同的名字给不了分。
        from difflib import SequenceMatcher

        def _sim(a: str, b: str) -> float:
            return SequenceMatcher(None, a or "", b or "").ratio()

        for existing in self.library.all() or []:
            if (existing.domain or "general") != (recipe.domain or "general"):
                continue
            # 要采纳的配方自己已在库里（同 id 或其碰撞变体）不算“别人”
            if existing.id == recipe.id or existing.id == final_id:
                continue
            name_sim = _sim(recipe.name, existing.name)
            desc_sim = _sim(recipe.description, existing.description)
            if max(name_sim, desc_sim) >= 0.6:
                return None, (
                    f"已有相似配方「{existing.id} — {existing.name}」"
                    f"（名称相似度 {name_sim:.0%}），不再重复入库；确需覆盖请改 id 后再试"
                )

        if final_id != recipe.id:
            recipe.id = final_id
        recipe.version = "1.0" if recipe.version in ("", "0.1-draft") else recipe.version
        recipe.provenance = f"{recipe.provenance} | adopted" if recipe.provenance else "adopted"
        recipe.to_yaml(target)

        try:
            if self.library is not None:
                self.library.load()
        except Exception as exc:  # 加载失败不影响落盘结论
            logger.warning("library reload after adopt failed: %s", exc)
        return target, f"已采纳为 {final_id}"

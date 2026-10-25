# -*- coding: utf-8 -*-
"""Engine assembly: build a ready-to-run AgentLoop from workspace + config."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .llm_client import EngineLLMClient, EngineLLMConfig, default_config_path
from .loop import AgentLoop, LoopConfig
from .tools import EngineContext, build_default_registry
from .verifier import Verifier

logger = logging.getLogger(__name__)


@dataclass
class EngineBundle:
    """All components of a running engine."""

    loop: AgentLoop
    llm: EngineLLMClient
    catalog: Any
    code_runner: Any
    recipes: Any
    verifier: Verifier
    context: EngineContext
    workspace: Path
    #: 数据目录后台预热器（``None`` = 启动时已同步扫完）
    prewarmer: Any = None

    def close(self) -> None:
        try:
            if self.code_runner is not None:
                self.code_runner.shutdown()
        except Exception:
            pass
        try:
            if self.catalog is not None:
                self.catalog.close()
        except Exception:
            pass

    # ------------------------------------------------------------ 配方沉淀
    def _runs_dir(self) -> Path:
        return self.workspace / ".gis_agent" / "runs"

    def trace_path_for(self, run_id: str) -> str | None:
        """按 run_id 找 trace 文件（没有就返回 None）。"""
        if not run_id:
            return None
        path = self.workspace / ".gis_agent" / "traces" / f"{run_id}.jsonl"
        return str(path) if path.exists() else None

    def latest_completed_state(self, run_id: str | None = None) -> tuple[Any, str | None]:
        """找一次 **completed** 的运行（默认取最新），返回 ``(state, trace_path)``。

        为什么不用 ``loop.state``：CLI 里引擎是新建的，还没跑过任务；
        用户想要的是“把上一次成功的活儿沉淀成配方”。
        """
        import json as _json

        from .state import TaskState

        runs = self._runs_dir()
        if not runs.is_dir():
            return None, None
        candidates: list[tuple[float, Path, str]] = []
        for path in runs.glob("*.json"):
            if path.name.endswith(".messages.json"):
                continue
            try:
                data = _json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            status = str(data.get("status") or "")
            rid = str(data.get("run_id") or path.stem)
            if run_id is not None and rid != run_id:
                continue
            if status != "completed":
                continue
            try:
                candidates.append((path.stat().st_mtime, path, rid))
            except OSError:
                continue
        if not candidates:
            return None, None
        _mtime, path, rid = max(candidates, key=lambda item: item[0])
        try:
            return TaskState.load(str(path)), self.trace_path_for(rid)
        except Exception:
            return None, None

    def distill(
        self,
        state: Any = None,
        *,
        adopt: bool = False,
        replay: bool = False,
        extra_snippets: list[str] | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """把一次**成功**运行蒸馏成候选配方。

        链路：收 execute_code 片段 → 模型泛化 → 净化 → 静态门禁 → 写候选目录
        （→ 可选回放验证 → 可选采纳）。

        ``state`` 不传时从 ``.gis_agent/runs/`` 里找**最近一次 completed** 的运行
        （``run_id`` 可指定）。返回字典（不抛异常）：
        ``{ok, reason?, recipe, gate, candidate, replay?, adopted}``。
        """
        from ..recipes.distiller import RecipeDistiller
        from ..recipes.lifecycle import RecipeLifecycle
        from ..recipes.lifecycle import sanitize_distilled

        trace: str | None = None
        if state is None:
            state, trace = self.latest_completed_state(run_id)
        if state is None:
            return {
                "ok": False,
                "reason": "没有可蒸馏的运行（没有 completed 状态的历史运行）"
                if run_id is None
                else f"找不到运行 {run_id} 的 completed 状态",
            }
        if str(getattr(state, "status", "")) != "completed":
            return {
                "ok": False,
                "reason": f"运行状态是 {getattr(state, 'status', '?')}，只有 completed 才值得沉淀",
            }

        snippets = list(extra_snippets or [])
        if not snippets:
            if trace is None:
                trace = self.trace_path_for(getattr(state, "run_id", ""))
            if trace:
                snippets = RecipeDistiller.collect_successful_code(trace)
        if not snippets:
            return {
                "ok": False,
                "reason": "本次运行没有可泛化的 execute_code 片段（全部走配方/工具，没什么可沉淀的）",
            }

        distiller = RecipeDistiller(self.llm, self.recipes)
        recipe = distiller.distill(
            goal=str(getattr(state, "goal", "") or ""),
            methodology=str(getattr(state, "methodology", "") or ""),
            final_summary=str(getattr(state, "final_summary", "") or ""),
            code_snippets=snippets,
            artifacts=list(getattr(state, "artifacts", []) or []),
        )
        if recipe is None:
            return {"ok": False, "reason": "模型没有产出可解析的配方（见日志）"}

        lifecycle = RecipeLifecycle(self.recipes, self.workspace, code_runner=self.code_runner)
        # 蒸馏结果先净化（槽位多余引号 / 默认值里的源绝对路径），再过门禁。
        # 净化不掉的（真正的硬编码）就交给门禁拦下。
        notes = sanitize_distilled(recipe, source_workspace=str(self.workspace))
        gate = lifecycle.static_gate(recipe, source_workspace=str(self.workspace))
        candidate = lifecycle.save_candidate(
            recipe,
            provenance=f"run:{getattr(state, 'run_id', '')}",
            gate=gate,
            extra={
                "goal": str(getattr(state, "goal", "") or "")[:500],
                "artifacts": list(getattr(state, "artifacts", []) or [])[:10],
            },
        )
        result: dict[str, Any] = {
            "ok": gate.ok,
            "recipe": recipe.to_dict(),
            "gate": gate.to_dict(),
            "candidate": str(candidate),
            "adopted": None,
            "sanitized": notes,
        }

        if replay and gate.ok:
            replay_result = lifecycle.replay(recipe.id).to_dict()
            result["replay"] = replay_result
            # 回放不过就不算成功——否则调用方会以为“已通过验证”。
            if not replay_result.get("ok"):
                result["ok"] = False
                result["reason"] = "回放验证未通过：" + str(
                    next((c["detail"] for c in replay_result.get("checks", []) if not c.get("ok")), "")
                )[:200]

        if adopt and gate.ok and (not replay or (result.get("replay") or {}).get("ok")):
            target, note = lifecycle.adopt(recipe.id)
            result["adopted"] = str(target) if target else None
            result["adopt_note"] = note
            if target is None:
                result["ok"] = False
        return result


def load_engine_config(workspace: str | Path, config_path: str | Path | None = None) -> tuple[EngineLLMConfig, dict[str, Any]]:
    """Load LLM + engine config, returning (llm_config, engine_dict)."""
    path = Path(config_path) if config_path else default_config_path(workspace)
    llm_config = EngineLLMConfig.from_file(path)
    engine_dict: dict[str, Any] = {}
    if path.exists():
        import json

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data.get("engine"), dict):
                engine_dict = data["engine"]
        except Exception as exc:
            logger.warning("failed to read engine config: %s", exc)
    return llm_config, engine_dict


def build_engine(
    *,
    workspace: str | Path,
    config_path: str | Path | None = None,
    python_executable: str | None = None,
    refresh_catalog: bool = True,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
    verbose: bool = False,
    engine_overrides: dict[str, Any] | None = None,
) -> EngineBundle:
    """Assemble the engine and optionally warm the catalog.

    ``engine_overrides`` 允许调用方（如跨模型基准）按任务覆盖 engine 段配置，
    例如用任务 YAML 的 ``timeout_seconds`` 覆盖 ``run_deadline_seconds``。
    """
    from ..datacatalog.scanner import refresh_catalog as do_refresh
    from ..datacatalog.store import Catalog
    from ..recipes.store import RecipeLibrary
    from ..runtime.code_runner import CodeRunner
    from ..runtime.timing import StageTimer

    _total = StageTimer("build_engine.total")
    ws = Path(workspace)
    state_dir = ws / ".gis_agent"
    state_dir.mkdir(parents=True, exist_ok=True)

    llm_config, engine_cfg = load_engine_config(ws, config_path)
    if engine_overrides:
        engine_cfg = {**engine_cfg, **{k: v for k, v in engine_overrides.items() if v is not None}}
    llm = EngineLLMClient(llm_config)

    catalog = Catalog(state_dir / "catalog.db")
    code_runner = CodeRunner(
        workspace=ws,
        python_executable=python_executable,
        state_dir=state_dir / "kernel",
        exec_timeout=float(engine_cfg.get("kernel_exec_timeout_seconds", 600)),
    )
    _recipes_timer = StageTimer("recipes.load")
    recipes = RecipeLibrary(
        user_dir=ws / "recipes",
        code_runner=code_runner,
        catalog=catalog,
    )
    recipes.load()
    _recipes_timer.done(count=len(recipes.all()))
    verifier = Verifier(
        workspace=ws,
        code_runner=code_runner,
        llm_client=llm,
        hygiene_mode=str(engine_cfg.get("hygiene_check", "warn")),
        vision_mode=str(engine_cfg.get("vision_check", "warn")),
        # 卫生判据参数：与配方侧 deliverable_hygiene_ok 用同一套键，
        # 避免“同一份交付物配方判不过、验收判过”的两套口径。
        hygiene_max_duplicate_families=int(
            engine_cfg.get("hygiene_max_duplicate_families", 0) or 0
        ),
        hygiene_ratio_limit=float(engine_cfg.get("hygiene_ratio_limit", 5.0) or 0.0),
        hygiene_forbid_suffixes=[
            str(item)
            for item in (engine_cfg.get("hygiene_forbid_suffixes") or [])
            if str(item).strip()
        ]
        or None,
    )
    registry = build_default_registry()
    # 子代理默认关：它是真金白银的 token，不该偷偷开。关掉时把工具从表里
    # **剔除**而不是留着报错——留着模型会白花一轮去调一个不可用的工具。
    subagent_enabled = bool(engine_cfg.get("subagent_enabled", False))
    if not subagent_enabled:
        registry = registry.subset([name for name in registry.names() if name != "delegate"])

    context = EngineContext(
        workspace=ws,
        catalog=catalog,
        code_runner=code_runner,
        recipes=recipes,
        verifier=verifier,
        output_dir=ws / "output",
        backup_dir=Path(engine_cfg.get("backup_dir", ws / ".gis_agent" / "backups")),
        auto_backup=bool(engine_cfg.get("auto_backup_destructive", True)),
        exec_timeout=float(engine_cfg.get("kernel_exec_timeout_seconds", 600)),
    )

    loop_config = LoopConfig(
        max_turns=int(engine_cfg.get("max_turns", 60)),
        max_repairs=int(engine_cfg.get("max_repairs", 4)),
        max_repair_seconds=float(engine_cfg.get("max_repair_seconds", 600.0)),
        distill_after_success=str(engine_cfg.get("distill_after_success", "suggest") or "suggest"),
        exec_timeout=float(engine_cfg.get("kernel_exec_timeout_seconds", 600)),
        tool_protocol=str(engine_cfg.get("tool_protocol", "auto")),
        context_soft_limit_tokens=int(engine_cfg.get("context_soft_limit_tokens", 120_000)),
        image_token_cost=int(engine_cfg.get("image_token_cost", 1_500)),
        compact_hysteresis=float(engine_cfg.get("compact_hysteresis", 1.2)),
        compact_min_messages=int(engine_cfg.get("compact_min_messages", 8)),
        max_diagnosis_turns=int(engine_cfg.get("exploration_turn_budget", 3)),
        escalate_after_failures=int(engine_cfg.get("escalate_after_failures", 2)),
        run_deadline_seconds=float(engine_cfg.get("run_deadline_seconds", 7200)),
        verbose=verbose,
    )

    # 数据目录后台预热器：``refresh_catalog=False``（启动不阻塞）时用它，
    # 与首轮模型调用重叠；``catalog_query`` 首次用到目录时会等它完成。
    from ..datacatalog.prewarm import CatalogPrewarmer

    prewarmer: Any = None
    if not refresh_catalog:
        prewarmer = CatalogPrewarmer(catalog=catalog, code_runner=code_runner, workspace=ws)
        context.extra["prewarmer"] = prewarmer

    loop = AgentLoop(
        workspace=ws,
        llm_client=llm,
        registry=registry,
        context=context,
        code_runner=code_runner,
        catalog=catalog,
        verifier=verifier,
        recipes=recipes,
        config=loop_config,
        on_event=on_event,
        auto_extract_requirements=bool(engine_cfg.get("auto_extract_requirements", True)),
        catalog_prewarmer=prewarmer,
    )
    context.state = loop.state  # updated per run in loop.run

    if subagent_enabled:
        from .subagent import SubAgentRunner

        subagent_runner = SubAgentRunner(
            loop,
            max_per_run=int(engine_cfg.get("subagent_max_per_run", 3)),
            max_turns=int(engine_cfg.get("subagent_max_turns", 30)),
            timeout_seconds=float(engine_cfg.get("subagent_timeout_seconds", 900)),
            exec_timeout_seconds=float(engine_cfg.get("subagent_exec_timeout_seconds", 300)),
            result_max_chars=int(engine_cfg.get("subagent_result_max_chars", 4000)),
        )
        # 工具 handler 在调用时才取工厂，所以可以在 loop 建好后再挂
        context.extra["subagent_factory"] = subagent_runner.run
        context.extra["subagent_runner"] = subagent_runner
        logger.info("subagent enabled (max_per_run=%s, max_turns=%s)", subagent_runner.max_per_run, subagent_runner.max_turns)

    bundle = EngineBundle(
        loop=loop,
        llm=llm,
        catalog=catalog,
        code_runner=code_runner,
        recipes=recipes,
        verifier=verifier,
        context=context,
        workspace=ws,
        prewarmer=prewarmer,
    )

    if refresh_catalog:
        timer = StageTimer("catalog.refresh")
        try:
            existing_roots = _candidate_data_roots(ws)
            if existing_roots:
                stats = do_refresh(catalog, existing_roots, code_runner, force=False)
                logger.info("catalog refreshed: %s", stats)
                timer.done(
                    discovered=stats.discovered,
                    scanned=stats.scanned,
                    skipped=stats.skipped,
                )
            else:
                timer.done(roots=0)
        except Exception as exc:
            timer.done(error=str(exc)[:60])
            logger.warning("catalog refresh failed (continuing): %s", exc)
        _total.done(refresh=True)
        return bundle

    _total.done(refresh=False, prewarm="deferred")
    return bundle


def _candidate_data_roots(workspace: Path) -> list[str]:
    """Find likely data roots, supporting both repo-root and workspace layouts."""
    candidates = [
        workspace / "input",
        workspace / "workspace" / "input",
        workspace / "data",
        workspace / "output",
        workspace / "workspace" / "output",
    ]
    roots = [str(p) for p in candidates if p.exists()]
    if roots:
        return roots
    # Fall back to the workspace itself so ad-hoc layouts still work.
    return [str(workspace)]

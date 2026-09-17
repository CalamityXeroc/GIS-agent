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
) -> EngineBundle:
    """Assemble the engine and optionally warm the catalog."""
    from ..datacatalog.scanner import refresh_catalog as do_refresh
    from ..datacatalog.store import Catalog
    from ..recipes.store import RecipeLibrary
    from ..runtime.code_runner import CodeRunner

    ws = Path(workspace)
    state_dir = ws / ".gis_agent"
    state_dir.mkdir(parents=True, exist_ok=True)

    llm_config, engine_cfg = load_engine_config(ws, config_path)
    llm = EngineLLMClient(llm_config)

    catalog = Catalog(state_dir / "catalog.db")
    code_runner = CodeRunner(
        workspace=ws,
        python_executable=python_executable,
        state_dir=state_dir / "kernel",
        exec_timeout=float(engine_cfg.get("kernel_exec_timeout_seconds", 600)),
    )
    recipes = RecipeLibrary(
        user_dir=ws / "recipes",
        code_runner=code_runner,
        catalog=catalog,
    )
    recipes.load()
    verifier = Verifier(workspace=ws, code_runner=code_runner, llm_client=llm)
    registry = build_default_registry()

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
        exec_timeout=float(engine_cfg.get("kernel_exec_timeout_seconds", 600)),
        tool_protocol=str(engine_cfg.get("tool_protocol", "auto")),
        context_soft_limit_tokens=int(engine_cfg.get("context_soft_limit_tokens", 120_000)),
        max_diagnosis_turns=int(engine_cfg.get("exploration_turn_budget", 3)),
        escalate_after_failures=int(engine_cfg.get("escalate_after_failures", 2)),
        verbose=verbose,
    )

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
    )
    context.state = loop.state  # updated per run in loop.run

    bundle = EngineBundle(
        loop=loop,
        llm=llm,
        catalog=catalog,
        code_runner=code_runner,
        recipes=recipes,
        verifier=verifier,
        context=context,
        workspace=ws,
    )

    if refresh_catalog:
        try:
            existing_roots = _candidate_data_roots(ws)
            if existing_roots:
                stats = do_refresh(catalog, existing_roots, code_runner, force=False)
                logger.info("catalog refreshed: %s", stats)
        except Exception as exc:
            logger.warning("catalog refresh failed (continuing): %s", exc)
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

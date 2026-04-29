"""Unified context discovery hub for planning and execution.

This module centralizes workspace/resource discovery so planners and executors
can share the same source of truth for:
- input roots
- ArcGIS Pro projects (.aprx)
- data candidates (shp / gdb)
- output locations
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ContextSnapshot:
    workspace_path: str
    input_roots: list[str] = field(default_factory=list)
    output_root: str = ""
    aprx_files: list[str] = field(default_factory=list)
    shapefiles: list[str] = field(default_factory=list)
    geodatabases: list[str] = field(default_factory=list)
    output_data: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_path": self.workspace_path,
            "input_roots": self.input_roots,
            "output_root": self.output_root,
            "aprx_files": self.aprx_files,
            "shapefiles": self.shapefiles,
            "geodatabases": self.geodatabases,
            "output_data": self.output_data,
            "unresolved": self.unresolved,
        }


class ContextHub:
    """Context discovery middleware used by agent/planner/executor."""

    def __init__(self, workspace: Path | str, memory: Any = None):
        self.workspace = Path(workspace)
        self.memory = memory
        self._cache_snapshot: ContextSnapshot | None = None
        self._cache_signature: tuple[str, str, str, int] | None = None
        self._cache_created_at: float = 0.0
        self._cache_ttl_seconds: float = 3.0

    def discover(self, max_items: int = 200, force_refresh: bool = False) -> ContextSnapshot:
        signature = self._build_discovery_signature(max_items)
        now = time.time()
        if (
            (not force_refresh)
            and self._cache_snapshot is not None
            and self._cache_signature == signature
            and (now - self._cache_created_at) <= self._cache_ttl_seconds
        ):
            return self._clone_snapshot(self._cache_snapshot)

        roots = self._candidate_input_roots()
        aprx_files: list[str] = []
        shapefiles: list[str] = []
        geodatabases: list[str] = []

        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            for p in root.rglob("*.aprx"):
                aprx_files.append(str(p))
                if len(aprx_files) >= max_items:
                    break
            for p in root.rglob("*.shp"):
                shapefiles.append(str(p))
                if len(shapefiles) >= max_items:
                    break
            for p in root.rglob("*.gdb"):
                geodatabases.append(str(p))
                if len(geodatabases) >= max_items:
                    break

        # Also scan output folder for data produced by previous runs
        output_root = str(self.workspace / "workspace" / "output")
        output_data: list[str] = []
        output_dir = Path(output_root)
        if output_dir.exists() and output_dir.is_dir():
            for p in output_dir.rglob("*.shp"):
                output_data.append(str(p))
            for p in output_dir.rglob("*.gdb"):
                output_data.append(str(p))

        unresolved: list[str] = []
        if not roots:
            unresolved.append("未发现可用输入目录")
        if not aprx_files:
            unresolved.append("未发现 .aprx 项目文件")
        if not shapefiles and not geodatabases and not output_data:
            unresolved.append("未发现输入数据（.shp/.gdb）")

        snapshot = ContextSnapshot(
            workspace_path=str(self.workspace),
            input_roots=[str(r) for r in roots],
            output_root=output_root,
            aprx_files=sorted(dict.fromkeys(aprx_files)),
            shapefiles=sorted(dict.fromkeys(shapefiles)),
            geodatabases=sorted(dict.fromkeys(geodatabases)),
            output_data=sorted(dict.fromkeys(output_data)),
            unresolved=unresolved,
        )
        self._cache_snapshot = self._clone_snapshot(snapshot)
        self._cache_signature = signature
        self._cache_created_at = now
        return snapshot

    def invalidate_cache(self) -> None:
        self._cache_snapshot = None
        self._cache_signature = None
        self._cache_created_at = 0.0

    def best_project_path(self, snapshot: ContextSnapshot) -> str | None:
        if snapshot.aprx_files:
            return snapshot.aprx_files[0]
        return None

    def best_input_root(self, snapshot: ContextSnapshot) -> str:
        if snapshot.input_roots:
            return snapshot.input_roots[0]
        return str(self.workspace / "input")

    def build_planner_payload(self, snapshot: ContextSnapshot) -> dict[str, Any]:
        hints: list[str] = []
        if snapshot.aprx_files:
            hints.append(f"检测到 {len(snapshot.aprx_files)} 个 ArcGIS Pro 项目")
        if snapshot.shapefiles or snapshot.geodatabases:
            hints.append(
                f"检测到输入数据: shp={len(snapshot.shapefiles)}, gdb={len(snapshot.geodatabases)}"
            )
        if snapshot.output_data:
            hints.append(f"检测到输出目录历史数据: {len(snapshot.output_data)} 个文件")
        if snapshot.unresolved:
            hints.extend([f"待澄清: {item}" for item in snapshot.unresolved])

        return {
            "context_snapshot": snapshot.to_dict(),
            "smart_hints": hints,
            "detected_project_path": self.best_project_path(snapshot),
            "detected_input_root": self.best_input_root(snapshot),
            "data_schema": self._build_data_schema(snapshot),
        }

    def _build_data_schema(self, snapshot: ContextSnapshot) -> str:
        """Build a formatted data schema string from discovered layers.

        This produces a compact summary that gets injected into the LLM
        planning prompt so generated ArcPy code knows actual field names,
        types, and CRS info.
        """
        parts: list[str] = []

        # Basic schema from file-based discovery
        for shp in snapshot.shapefiles[:20]:
            p = Path(shp)
            parts.append(f"  [input] {p.name}: path={shp}, type=Shapefile")

        for gdb in snapshot.geodatabases[:10]:
            p = Path(gdb)
            parts.append(f"  [input] {p.name}: path={gdb}, type=FileGeodatabase")

        for out in snapshot.output_data[:10]:
            p = Path(out)
            parts.append(f"  [output] {p.name}: path={out}, type=Shapefile (先前产出数据)")

        # Try enriching with ArcPy if available (fields, wkid, geometry)
        if snapshot.shapefiles or snapshot.geodatabases or snapshot.output_data:
            try:
                from ..arcpy_bridge import scan_workspace_layers
                # Enrich input data
                for root in snapshot.input_roots:
                    result = scan_workspace_layers(root)
                    if result.status == "success" and result.data:
                        for layer in result.data.get("layers", []):
                            name = layer.get("name", "")
                            geom = layer.get("type", "Unknown")
                            fc = layer.get("feature_count", "")
                            sr = layer.get("spatial_reference", "Unknown")
                            wkid = layer.get("wkid")
                            fields = layer.get("fields", [])
                            field_strs = []
                            for f in fields[:20]:
                                s = f"{f['name']}({f['type']})"
                                samples = f.get("samples")
                                if samples:
                                    s += f": {json.dumps(samples, ensure_ascii=False)}"
                                field_strs.append(s)
                            field_str = ", ".join(field_strs)
                            if len(fields) > 20:
                                field_str += f" (+{len(fields)-20} more)"
                            wkid_str = f" (WKID {wkid})" if wkid else ""
                            fc_str = f", features={fc}" if fc else ""
                            # Remove the basic [input] line for this layer if exists
                            parts = [p for p in parts if not p.endswith(f": type=Shapefile") or name not in p]
                            parts.append(
                                f"  [input] {name}: geom={geom}, sr={sr}{wkid_str}{fc_str}, "
                                f"fields={{{field_str}}}"
                            )
                        break

                # Enrich output data
                out_dir = Path(snapshot.output_root)
                if out_dir.exists() and out_dir.is_dir():
                    result = scan_workspace_layers(str(out_dir))
                    if result.status == "success" and result.data:
                        for layer in result.data.get("layers", []):
                            name = layer.get("name", "")
                            geom = layer.get("type", "Unknown")
                            fc = layer.get("feature_count", "")
                            sr = layer.get("spatial_reference", "Unknown")
                            fields = layer.get("fields", [])
                            field_strs = []
                            for f in fields[:20]:
                                s = f"{f['name']}({f['type']})"
                                samples = f.get("samples")
                                if samples:
                                    s += f": {json.dumps(samples, ensure_ascii=False)}"
                                field_strs.append(s)
                            field_str = ", ".join(field_strs)
                            if len(fields) > 20:
                                field_str += f" (+{len(fields)-20} more)"
                            fc_str = f", features={fc}" if fc else ""
                            # Remove basic [output] line if exists
                            parts = [p for p in parts if not (p.startswith(f"  [output] {name}") or p.endswith(f": {name}: type=Shapefile"))]
                            parts.append(
                                f"  [output] {name}: geom={geom}, sr={sr}{fc_str}, "
                                f"fields={{{field_str}}}"
                            )
            except ImportError:
                pass
            except Exception:
                pass

        if not parts:
            return ""

        lines = ["\n## 数据 Schema"] + parts
        return "\n".join(lines)

    def _candidate_input_roots(self) -> list[Path]:
        roots: list[Path] = []

        if self.memory is not None:
            memory_input = self.memory.get_context("input_folder")
            if isinstance(memory_input, str) and memory_input.strip():
                roots.append(Path(memory_input.strip()))
            memory_workspace = self.memory.get_context("workspace")
            if isinstance(memory_workspace, str) and memory_workspace.strip():
                ws = Path(memory_workspace.strip())
                roots.append(ws / "input")
                roots.append(ws / "workspace" / "input")
                roots.append(ws)

        roots.extend(
            [
                self.workspace / "input",
                self.workspace / "workspace" / "input",
                self.workspace,
                Path.cwd() / "workspace" / "input",
                Path.cwd() / "input",
            ]
        )

        deduped: list[Path] = []
        seen: set[str] = set()
        for p in roots:
            key = str(p.resolve()) if p.exists() else str(p)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)

        return [p for p in deduped if p.exists() and p.is_dir()]

    def _build_discovery_signature(self, max_items: int) -> tuple[str, str, str, int]:
        workspace_sig = str(self.workspace.resolve()) if self.workspace.exists() else str(self.workspace)
        memory_input_sig = ""
        memory_workspace_sig = ""
        if self.memory is not None:
            memory_input = self.memory.get_context("input_folder")
            memory_workspace = self.memory.get_context("workspace")
            if isinstance(memory_input, str):
                memory_input_sig = memory_input.strip().lower()
            if isinstance(memory_workspace, str):
                memory_workspace_sig = memory_workspace.strip().lower()
        return (workspace_sig, memory_input_sig, memory_workspace_sig, int(max_items))

    def _clone_snapshot(self, snapshot: ContextSnapshot) -> ContextSnapshot:
        return ContextSnapshot(
            workspace_path=snapshot.workspace_path,
            input_roots=list(snapshot.input_roots),
            output_root=snapshot.output_root,
            aprx_files=list(snapshot.aprx_files),
            shapefiles=list(snapshot.shapefiles),
            geodatabases=list(snapshot.geodatabases),
            output_data=list(snapshot.output_data),
            unresolved=list(snapshot.unresolved),
        )

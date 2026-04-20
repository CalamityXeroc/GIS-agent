"""Context module - System and user context injection.

Similar to Claude Code's context injection patterns, this module
provides context about the system, user, and workspace to tools and skills.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SystemContext:
    """System-level context information."""
    
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""
    arcgis_version: str | None = None
    arcpy_available: bool = False
    arcgis_python: str | None = None
    
    # Paths
    home_dir: str = ""
    temp_dir: str = ""
    current_dir: str = ""
    
    @classmethod
    def detect(cls) -> "SystemContext":
        """Detect system context."""
        import sys
        
        ctx = cls()
        ctx.os_name = platform.system()
        ctx.os_version = platform.version()
        ctx.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ctx.home_dir = str(Path.home())
        ctx.temp_dir = os.environ.get("TEMP", os.environ.get("TMP", "/tmp"))
        ctx.current_dir = os.getcwd()
        
        # Check ArcPy using ArcPy Bridge (doesn't require arcpy in current Python)
        ctx._detect_arcpy()
        
        return ctx
    
    def _detect_arcpy(self) -> None:
        """Detect ArcPy availability via ArcPy Bridge."""
        try:
            # First try direct import (if running in ArcGIS Pro Python)
            import arcpy
            self.arcpy_available = True
            self.arcgis_version = arcpy.GetInstallInfo().get("Version", "Unknown")
            return
        except ImportError:
            pass
        
        # Try ArcPy Bridge (runs in separate process)
        try:
            from ..arcpy_bridge import discover_arcgis_pro_python, check_arcpy_available
            
            # Check if ArcGIS Pro Python is available
            info = discover_arcgis_pro_python()
            self.arcgis_python = info.python_executable
            
            # Verify ArcPy actually works
            result = check_arcpy_available()
            if result.status == "success" and result.data:
                self.arcpy_available = True
                self.arcgis_version = result.data.get("version", "Unknown")
            else:
                self.arcpy_available = False
        except Exception:
            self.arcpy_available = False
            self.arcgis_version = None


@dataclass
class UserContext:
    """User-level context and preferences."""
    
    username: str = ""
    language: str = "zh-CN"
    
    # Preferences
    default_crs: str = "Asia_North_Albers_Equal_Area_Conic"
    default_output_format: str = "PDF"
    default_resolution: int = 300
    
    # Custom settings
    settings: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def load(cls, config_path: Path | None = None) -> "UserContext":
        """Load user context from config file or environment."""
        ctx = cls()
        ctx.username = os.environ.get("USERNAME", os.environ.get("USER", "unknown"))
        ctx.language = os.environ.get("LANG", "zh-CN").split(".")[0]
        
        # Load from config file if exists
        if config_path and config_path.exists():
            try:
                import json
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    ctx.default_crs = data.get("default_crs", ctx.default_crs)
                    ctx.default_output_format = data.get("default_output_format", ctx.default_output_format)
                    ctx.default_resolution = data.get("default_resolution", ctx.default_resolution)
                    ctx.settings = data.get("settings", {})
            except Exception:
                pass
        
        return ctx
    
    def save(self, config_path: Path) -> None:
        """Save user context to config file."""
        import json
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "default_crs": self.default_crs,
            "default_output_format": self.default_output_format,
            "default_resolution": self.default_resolution,
            "settings": self.settings
        }
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


@dataclass
class WorkspaceContext:
    """Workspace-level context."""
    
    workspace_path: str = ""
    output_dir: str = ""
    
    # Project info
    project_name: str = ""
    project_path: str | None = None
    
    # Data inventory
    shapefiles: list[str] = field(default_factory=list)
    geodatabases: list[str] = field(default_factory=list)
    rasters: list[str] = field(default_factory=list)
    
    @classmethod
    def scan(cls, workspace: Path) -> "WorkspaceContext":
        """Scan workspace to build context."""
        ctx = cls()
        ctx.workspace_path = str(workspace)
        ctx.output_dir = str(workspace / "outputs")
        
        if workspace.exists():
            # Scan for common GIS files
            for p in workspace.rglob("*.shp"):
                ctx.shapefiles.append(str(p))
            
            for p in workspace.rglob("*.gdb"):
                ctx.geodatabases.append(str(p))
            
            for ext in ["*.tif", "*.tiff", "*.img"]:
                for p in workspace.rglob(ext):
                    ctx.rasters.append(str(p))
            
            # Find project files
            aprx_files = list(workspace.rglob("*.aprx"))
            if aprx_files:
                ctx.project_path = str(aprx_files[0])
                ctx.project_name = aprx_files[0].stem
        
        return ctx


@dataclass
class ExecutionContext:
    """Combined execution context for tools and skills."""
    
    system: SystemContext = field(default_factory=SystemContext)
    user: UserContext = field(default_factory=UserContext)
    workspace: WorkspaceContext = field(default_factory=WorkspaceContext)
    
    # Runtime flags
    dry_run: bool = False
    verbose: bool = False
    debug: bool = False
    
    @classmethod
    def build(
        cls,
        workspace: Path | None = None,
        config_path: Path | None = None,
        dry_run: bool = False,
        verbose: bool = False,
        debug: bool = False
    ) -> "ExecutionContext":
        """Build complete execution context."""
        ctx = cls()
        ctx.system = SystemContext.detect()
        ctx.user = UserContext.load(config_path)
        
        if workspace:
            ctx.workspace = WorkspaceContext.scan(workspace)
        
        ctx.dry_run = dry_run
        ctx.verbose = verbose
        ctx.debug = debug
        
        return ctx
    
    @property
    def arcpy_available(self) -> bool:
        """Shortcut to check ArcPy availability."""
        return self.system.arcpy_available
    
    def summary(self) -> str:
        """Generate context summary."""
        lines = [
            f"System: {self.system.os_name} ({self.system.python_version})",
            f"ArcPy: {'Available' if self.arcpy_available else 'Not available'}",
        ]
        
        if self.system.arcgis_version:
            lines.append(f"ArcGIS: {self.system.arcgis_version}")
        
        if self.workspace.workspace_path:
            lines.append(f"Workspace: {self.workspace.workspace_path}")
            lines.append(f"  - Shapefiles: {len(self.workspace.shapefiles)}")
            lines.append(f"  - Geodatabases: {len(self.workspace.geodatabases)}")
            lines.append(f"  - Rasters: {len(self.workspace.rasters)}")
        
        if self.workspace.project_name:
            lines.append(f"Project: {self.workspace.project_name}")
        
        return "\n".join(lines)

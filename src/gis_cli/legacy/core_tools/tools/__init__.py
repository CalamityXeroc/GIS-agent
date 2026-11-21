"""Tools __init__.py - Import all tool modules and register them."""

from gis_cli.legacy.core_tools.core.registry import ToolRegistry
from gis_cli.legacy.core_tools.tools import ScanLayersTool
from gis_cli.legacy.core_tools.tools import MergeLayersTool
from gis_cli.legacy.core_tools.tools import ProjectLayersTool
from gis_cli.legacy.core_tools.tools import ExportMapTool
from gis_cli.legacy.core_tools.tools import QualityCheckTool
from gis_cli.legacy.core_tools.tools import ExecuteCodeTool
from gis_cli.legacy.core_tools.tools import ReadWordTool
from gis_cli.legacy.core_tools.tools import ReadPdfTool
from gis_cli.legacy.core_tools.tools import WebSearchTool
from gis_cli.legacy.core_tools.tools import ThematicMapTool

__all__ = [
    "ScanLayersTool",
    "MergeLayersTool",
    "ProjectLayersTool",
    "ExportMapTool",
    "QualityCheckTool",
    "ExecuteCodeTool",
    "ReadWordTool",
    "ReadPdfTool",
    "WebSearchTool",
    "ThematicMapTool",
]


def register_all_tools() -> None:
    """Register all tools with the global registry."""
    registry = ToolRegistry.get_instance()
    
    tools = [
        ScanLayersTool(),
        MergeLayersTool(),
        ProjectLayersTool(),
        ExportMapTool(),
        QualityCheckTool(),
        ExecuteCodeTool(),
        ReadWordTool(),
        ReadPdfTool(),
        WebSearchTool(),
        ThematicMapTool(),
    ]
    
    for tool in tools:
        try:
            registry.register(tool)
        except ValueError:
            # Already registered
            pass


# Auto-register on import
register_all_tools()

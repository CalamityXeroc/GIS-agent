"""Tools __init__.py - Import all tool modules and register them."""

from ..core.registry import ToolRegistry
from .scan_layers import ScanLayersTool
from .merge_layers import MergeLayersTool
from .project_layers import ProjectLayersTool
from .export_map import ExportMapTool
from .quality_check import QualityCheckTool
from .execute_code import ExecuteCodeTool
from .read_word import ReadWordTool
from .read_pdf import ReadPdfTool
from .web_search import WebSearchTool

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
    ]
    
    for tool in tools:
        try:
            registry.register(tool)
        except ValueError:
            # Already registered
            pass


# Auto-register on import
register_all_tools()

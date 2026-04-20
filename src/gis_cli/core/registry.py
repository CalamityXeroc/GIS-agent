"""Tool registry and discovery for GIS CLI.

Inspired by Claude Code's tools.ts, this provides:
- Tool registration and discovery
- Tool pool assembly based on capabilities
- Tool matching by name
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .tool import Tool, ToolCategory

if TYPE_CHECKING:
    from .task import TaskContext


class ToolRegistry:
    """Registry for all available tools."""
    
    _instance: "ToolRegistry | None" = None
    
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._categories: dict[ToolCategory, list[Tool]] = {}
    
    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    # Alias for consistency with SkillRegistry
    instance = get_instance
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} already registered")
        
        self._tools[tool.name] = tool
        
        # Index by category
        if tool.category not in self._categories:
            self._categories[tool.category] = []
        self._categories[tool.category].append(tool)
    
    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_all(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    # Alias for list_tools
    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return self.get_all()
    
    def get_by_category(self, category: ToolCategory) -> list[Tool]:
        """Get tools by category."""
        return self._categories.get(category, [])
    
    def search(self, query: str) -> list[Tool]:
        """Search tools by name, description, or search hint."""
        query_lower = query.lower()
        results = []
        
        for tool in self._tools.values():
            if (
                query_lower in tool.name.lower()
                or query_lower in tool.description.lower()
                or query_lower in tool.search_hint.lower()
            ):
                results.append(tool)
        
        return results
    
    def matches_name(self, tool: Tool, name: str) -> bool:
        """Check if tool matches a given name (case-insensitive)."""
        return (
            tool.name.lower() == name.lower()
            or tool.user_facing_name.lower() == name.lower()
        )
    
    def find_by_name(self, name: str) -> Tool | None:
        """Find a tool by name (case-insensitive)."""
        for tool in self._tools.values():
            if self.matches_name(tool, name):
                return tool
        return None


def assemble_tool_pool(
    categories: list[ToolCategory] | None = None,
    include_read_only: bool = True,
    include_write: bool = True,
) -> list[Tool]:
    """Assemble a pool of tools based on criteria.
    
    Args:
        categories: Filter by categories (None = all)
        include_read_only: Include read-only tools
        include_write: Include write tools
    
    Returns:
        List of matching tools
    """
    registry = ToolRegistry.get_instance()
    tools = registry.get_all()
    
    # Filter by category
    if categories:
        tools = [t for t in tools if t.category in categories]
    
    # Filter by read/write
    if not include_read_only:
        tools = [t for t in tools if not t.is_read_only()]
    if not include_write:
        tools = [t for t in tools if t.is_read_only()]
    
    return tools


def get_tools_json_schema() -> list[dict[str, Any]]:
    """Get JSON schema for all tools (for LLM tool use)."""
    registry = ToolRegistry.get_instance()
    return [tool.to_json_schema() for tool in registry.get_all()]


# === Registration Decorators ===

def register_tool(tool_class: type[Tool]) -> type[Tool]:
    """Decorator to register a tool class."""
    instance = tool_class()
    ToolRegistry.get_instance().register(instance)
    return tool_class


def tool(name: str, description: str, category: ToolCategory = ToolCategory.FILE_OPERATION):
    """Decorator to create and register a tool from a function.
    
    Usage:
        @tool("scan_layers", "Scan directory for GIS layers", ToolCategory.FILE_OPERATION)
        def scan_layers(input_data: ScanInput, context: ToolContext) -> ToolResult:
            ...
    """
    from .tool import ToolBuilder, ToolResult
    
    def decorator(func):
        builder = ToolBuilder(name)
        builder.with_description(description)
        builder.with_category(category)
        builder.with_call(func)
        
        tool_class = builder.build()
        instance = tool_class()
        ToolRegistry.get_instance().register(instance)
        
        return instance
    
    return decorator


# === Built-in Tool Registration ===

def _register_builtin_tools():
    """Register all built-in tools."""
    # Import tool modules to trigger registration
    # This is called during module initialization
    pass


# Register builtins on import
_register_builtin_tools()

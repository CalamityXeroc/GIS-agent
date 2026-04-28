"""WebSearchTool - Search the web for GIS information.

This tool allows the agent to search the web when it needs information
beyond its built-in knowledge (e.g. latest ArcPy API changes, specific
coordinate system details, or new ESRI features).

Supports multiple backends:
- tavily: Requires TAVILY_API_KEY, best quality for AI agents (recommended)
- duckduckgo: Free, no API key needed, rate-limited
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any

from pydantic import BaseModel, Field

from ..core import (
    Tool,
    ToolCategory,
    ToolContext,
    ToolResult,
    ValidationResult,
    PermissionResult,
    register_tool,
)

TOOL_NAME = "web_search"

DESCRIPTION = """Search the web for GIS-related information.

Use this tool when you need:
- Latest ArcPy or ArcGIS Pro version features
- Specific coordinate system WKIDs or parameters
- ESRI documentation that may have changed
- Troubleshooting obscure error messages
- Any GIS information not covered by your built-in knowledge

Results are returned as structured snippets with titles and URLs.
"""

SEARCH_HINT = "web search internet online lookup find google bing"


class WebSearchInput(BaseModel):
    """Input schema for WebSearchTool."""
    query: str = Field(
        min_length=2,
        max_length=500,
        description="Search query (GIS-related)"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of search results to return"
    )


class WebSearchResult(BaseModel):
    """A single search result."""
    title: str
    snippet: str
    url: str


class WebSearchOutput(BaseModel):
    """Output schema for WebSearchTool."""
    query: str
    results: list[WebSearchResult]
    total_found: int
    backend: str
    duration_ms: int = 0


@register_tool
class WebSearchTool(Tool[WebSearchInput, WebSearchOutput]):
    """Tool to search the web for GIS information."""

    name = TOOL_NAME
    description = DESCRIPTION
    category = ToolCategory.FILE_OPERATION
    search_hint = SEARCH_HINT
    input_model = WebSearchInput

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def get_activity_description(self, input_data: WebSearchInput) -> str:
        return f"Searching web for: {input_data.query}"

    def get_tool_use_summary(self, input_data: WebSearchInput) -> str:
        return f"web_search({input_data.query})"

    def validate_input(self, input_data: WebSearchInput) -> ValidationResult:
        if not input_data.query.strip():
            return ValidationResult.failure("Search query cannot be empty", error_code=1)
        return ValidationResult.success()

    def check_permissions(
        self,
        input_data: WebSearchInput,
        context: ToolContext,
    ) -> PermissionResult:
        return PermissionResult.allow()

    def call(
        self,
        input_data: WebSearchInput,
        context: ToolContext,
    ) -> ToolResult[WebSearchOutput]:
        """Execute web search."""
        start_time = time.time()

        query = input_data.query.strip()
        max_results = input_data.max_results

        # Try Tavily first (requires API key)
        tavily_key = self._get_config("TAVILY_API_KEY") or os.environ.get("TAVILY_API_KEY")
        if tavily_key:
            results, backend = self._search_tavily(query, max_results, tavily_key)
        else:
            results, backend = self._search_duckduckgo(query, max_results)

        duration_ms = int((time.time() - start_time) * 1000)

        output = WebSearchOutput(
            query=query,
            results=results,
            total_found=len(results),
            backend=backend,
            duration_ms=duration_ms,
        )

        if not results:
            return ToolResult.fail(
                f"No results found for: {query}",
                error_code=404,
                data=output,
            )

        return ToolResult.ok(
            data=output,
            outputs=[r.url for r in results[:3]],
            duration_ms=duration_ms,
        )

    def _search_tavily(
        self, query: str, max_results: int, api_key: str
    ) -> tuple[list[WebSearchResult], str]:
        """Search using Tavily API."""
        try:
            data = json.dumps({
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.tavily.com/search",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            results = []
            for item in (result.get("results") or [])[:max_results]:
                results.append(WebSearchResult(
                    title=item.get("title", ""),
                    snippet=item.get("content", ""),
                    url=item.get("url", ""),
                ))
            return results, "tavily"
        except Exception:
            return [], "tavily_failed"

    def _search_duckduckgo(
        self, query: str, max_results: int
    ) -> tuple[list[WebSearchResult], str]:
        """Search using DuckDuckGo's HTML API (free, no key needed)."""
        try:
            encoded = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded}"

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            results = self._parse_duckduckgo_html(html, max_results)
            return results, "duckduckgo"
        except Exception:
            return [], "duckduckgo_failed"

    def _parse_duckduckgo_html(
        self, html: str, max_results: int
    ) -> list[WebSearchResult]:
        """Parse DuckDuckGo HTML search results."""
        results = []
        # Look for result links in the HTML
        # DuckDuckGo HTML results use <a rel="nofollow" class="result__a"> tags
        import re

        # Find result blocks
        blocks = re.split(r'<div class="result__body', html)[1:] if html.count('<div class="result__body') > 1 else []

        for block in blocks[:max_results]:
            try:
                # Extract title
                title_match = re.search(r'<a[^>]*class="result__a"[^>]*>(.*?)</a>', block, re.DOTALL)
                title = ""
                if title_match:
                    title = re.sub(r'<[^>]+>', "", title_match.group(1)).strip()

                # Extract snippet
                snippet_match = re.search(
                    r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', block, re.DOTALL
                )
                snippet = ""
                if snippet_match:
                    snippet = re.sub(r'<[^>]+>', "", snippet_match.group(1)).strip()

                # Extract URL
                url_match = re.search(r'href="(https?://[^"]+)"', block)
                url = ""
                if url_match:
                    url = url_match.group(1)
                    # DuckDuckGo wraps URLs in redirect
                    if "uddg=" in url:
                        from urllib.parse import parse_qs, urlparse
                        parsed = urlparse(url)
                        qs = parse_qs(parsed.query)
                        url = qs.get("uddg", [url])[0]

                if title and url:
                    results.append(WebSearchResult(
                        title=title,
                        snippet=snippet or title,
                        url=url,
                    ))
            except Exception:
                continue

        return results

    def _get_config(self, key: str) -> str | None:
        """Read configuration from llm_config.json."""
        try:
            config_path = None
            # Try common config paths
            for candidate in [
                os.path.join(os.getcwd(), "config", "llm_config.json"),
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "llm_config.json"),
            ]:
                if os.path.exists(candidate):
                    config_path = candidate
                    break
            if config_path:
                with open(config_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                return cfg.get("web_search", {}).get(key)
        except Exception:
            pass
        return None

    def render_tool_use_message(self, input_data: WebSearchInput) -> str:
        return f"Searching web for: {input_data.query}"

    def render_tool_result_message(self, result: ToolResult[WebSearchOutput]) -> str:
        if not result.success:
            return f"Web search failed: {result.error}"

        data = result.data
        if data is None or not data.results:
            return "No results found"

        lines = [f"Found {data.total_found} result(s)", ""]
        for r in data.results:
            lines.append(f"- **{r.title}**")
            lines.append(f"  {r.snippet}")
            lines.append(f"  {r.url}")
            lines.append("")

        return "\n".join(lines)

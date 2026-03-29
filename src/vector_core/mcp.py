"""MCP server utilities for vector-core.

Provides common functionality for MCP servers built on vector-core.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


class ToolRegistrationError(RuntimeError):
    """Raised when expected tools are not registered."""

    def __init__(self, server_name: str, missing_tools: set[str]):
        self.server_name = server_name
        self.missing_tools = missing_tools
        super().__init__(
            f"{server_name} is missing expected tools: {sorted(missing_tools)}"
        )


def verify_tools_registered(
    mcp: "FastMCP",
    expected_tools: list[str],
    server_name: str,
) -> None:
    """
    Verify that expected tools are registered with FastMCP instance.

    This catches silent failures where tool modules fail to import (e.g., due to
    import errors) and the server starts with missing functionality.

    Call this AFTER importing all tool modules:

        from myserver import tools  # Triggers tool registration
        verify_tools_registered(mcp, EXPECTED_TOOLS, "myserver")

    Args:
        mcp: FastMCP instance to check
        expected_tools: List of tool names that should be registered
        server_name: Server name for error messages

    Raises:
        ToolRegistrationError: If any expected tools are missing
    """
    # FastMCP stores tools in _tool_manager.tools dict
    # Access via the public list_tools() method if available, or internal structure
    try:
        # Try the internal structure (FastMCP implementation detail)
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "tools"):
            registered = set(mcp._tool_manager.tools.keys())
        elif hasattr(mcp, "_tools"):
            # Alternative internal structure
            registered = set(mcp._tools.keys())
        else:
            # Fallback: try to get tool names from list_tools
            logger.warning(
                f"Cannot verify tools for {server_name}: "
                "FastMCP internal structure not accessible. "
                "Tool registration verification skipped."
            )
            return
    except Exception as e:
        logger.warning(
            f"Cannot verify tools for {server_name}: {e}. "
            "Tool registration verification skipped."
        )
        return

    expected = set(expected_tools)
    missing = expected - registered

    if missing:
        raise ToolRegistrationError(server_name, missing)

    # Also log any unexpected tools (informational only)
    extra = registered - expected
    if extra:
        logger.info(
            f"{server_name} has additional tools not in expected list: {sorted(extra)}"
        )

    logger.info(
        f"{server_name} tool verification passed: {len(registered)} tools registered"
    )


def log_registered_tools(mcp: "FastMCP", server_name: str) -> list[str]:
    """
    Log all registered tools for debugging.

    Args:
        mcp: FastMCP instance
        server_name: Server name for log messages

    Returns:
        List of registered tool names
    """
    try:
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "tools"):
            tools = list(mcp._tool_manager.tools.keys())
        elif hasattr(mcp, "_tools"):
            tools = list(mcp._tools.keys())
        else:
            logger.warning(f"{server_name}: Cannot access tool list")
            return []
    except Exception as e:
        logger.warning(f"{server_name}: Error accessing tools: {e}")
        return []

    logger.info(f"{server_name} registered tools ({len(tools)}): {sorted(tools)}")
    return tools

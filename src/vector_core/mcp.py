"""MCP server utilities for vector-core.

Provides common functionality for MCP servers built on vector-core.
"""

import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from mcp.server import MCPServer

logger = logging.getLogger(__name__)


class ToolRegistrationError(RuntimeError):
    """Raised when expected tools are not registered."""

    def __init__(self, server_name: str, missing_tools: set[str]):
        self.server_name = server_name
        self.missing_tools = missing_tools
        super().__init__(f"{server_name} is missing expected tools: {sorted(missing_tools)}")


def _get_registered_tool_names(mcp: object) -> set[str] | None:
    """
    Collect the names of the tools registered on an MCPServer instance.

    MCPServer exposes no supported registry accessor, so the manager is discovered
    defensively: the tool manager's ``list_tools()`` first, then the dict
    layouts used by older releases.

    Returns:
        The registered tool names, or None if the registry could not be read at
        all. An empty set means the registry was read and holds no tools, which
        is a different condition from None and must stay distinguishable.
    """
    try:
        tool_manager = cast(Any, mcp)._tool_manager
    except Exception:
        tool_manager = None

    if tool_manager is not None:
        try:
            list_tools = tool_manager.list_tools
            if callable(list_tools):
                names: set[str] = set()
                for tool in list_tools():
                    name = tool.name
                    if not isinstance(name, str):
                        raise TypeError("Tool name is not a string")
                    names.add(name)
                return names
        except Exception:
            pass

    for owner, attribute in (
        (tool_manager, "tools"),
        (tool_manager, "_tools"),
        (mcp, "_tools"),
    ):
        if owner is None:
            continue
        try:
            tool_store = getattr(owner, attribute)
            return {str(name) for name in tool_store.keys()}
        except Exception:
            continue

    return None


def verify_tools_registered(
    mcp: "MCPServer",
    expected_tools: list[str],
    server_name: str,
) -> None:
    """
    Verify that expected tools are registered with an MCPServer instance.

    This catches silent failures where tool modules fail to import (e.g., due to
    import errors) and the server starts with missing functionality.

    Call this AFTER importing all tool modules:

        from myserver import tools  # Triggers tool registration
        verify_tools_registered(mcp, EXPECTED_TOOLS, "myserver")

    Args:
        mcp: MCPServer instance to check
        expected_tools: List of tool names that should be registered
        server_name: Server name for error messages

    Raises:
        ToolRegistrationError: If any expected tools are missing
    """
    registered = _get_registered_tool_names(mcp)
    if registered is None:
        logger.warning(
            f"Cannot verify tools for {server_name}: "
            "MCPServer tool registry not accessible. "
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
        logger.info(f"{server_name} has additional tools not in expected list: {sorted(extra)}")

    logger.info(f"{server_name} tool verification passed: {len(registered)} tools registered")


def log_registered_tools(mcp: "MCPServer", server_name: str) -> list[str]:
    """
    Log all registered tools for debugging.

    Args:
        mcp: MCPServer instance
        server_name: Server name for log messages

    Returns:
        List of registered tool names
    """
    registered = _get_registered_tool_names(mcp)
    if registered is None:
        logger.warning(f"{server_name}: Cannot access tool list")
        return []

    tools = list(registered)
    logger.info(f"{server_name} registered tools ({len(tools)}): {sorted(tools)}")
    return tools

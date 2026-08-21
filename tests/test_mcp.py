import logging

import pytest
from mcp.server import MCPServer

from vector_core.mcp import (
    ToolRegistrationError,
    _get_registered_tool_names,
    log_registered_tools,
    verify_tools_registered,
)


def test_log_registered_tools_discovers_tools_from_real_mcp_server() -> None:
    mcp = MCPServer("test-server")

    @mcp.tool()
    def first_tool() -> str:
        return "first"

    @mcp.tool()
    def second_tool() -> str:
        return "second"

    expected = {
        "first_tool",
        "second_tool",
    }
    assert _get_registered_tool_names(mcp) == expected
    assert set(log_registered_tools(mcp, "test-server")) == expected


def test_get_registered_tool_names_distinguishes_empty_registry() -> None:
    assert _get_registered_tool_names(MCPServer("empty-server")) == set()


def test_verify_tools_registered_raises_for_missing_tool() -> None:
    mcp = MCPServer("test-server")

    @mcp.tool()
    def registered_tool() -> str:
        return "registered"

    with pytest.raises(ToolRegistrationError) as exc_info:
        verify_tools_registered(
            mcp,
            ["registered_tool", "missing_tool"],
            "test-server",
        )

    assert exc_info.value.missing_tools == {"missing_tool"}


def test_verify_tools_registered_warns_for_undiscoverable_object(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)

    verify_tools_registered(object(), ["missing_tool"], "unknown-server")  # type: ignore[arg-type]

    assert _get_registered_tool_names(object()) is None
    assert "Tool registration verification skipped" in caplog.text

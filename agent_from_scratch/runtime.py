"""
Runtime module for AI agent from scratch.
This provides a runtime similar to LangChain's runtime for dependency injection and context sharing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable
import uuid


@dataclass
class ExecutionInfo:
    """Identity and retry information for the current execution."""

    thread_id: str
    run_id: str
    attempt_number: int = 1


@dataclass
class ServerInfo:
    """Server-specific metadata when running on a server."""

    assistant_id: Optional[str] = None
    graph_id: Optional[str] = None
    user_identity: Optional[str] = None


@dataclass
class Runtime:
    """
    Runtime object that provides context, store, stream writer, execution info, and server info.
    This is a simplified version of LangGraph's Runtime.
    """

    context: Any = None
    store: Dict[str, Any] = field(default_factory=dict)
    stream_writer: Optional[Callable[[str], None]] = None
    execution_info: ExecutionInfo = field(
        default_factory=lambda: ExecutionInfo(
            thread_id=str(uuid.uuid4()), run_id=str(uuid.uuid4())
        )
    )
    server_info: Optional[ServerInfo] = None


@dataclass
class ToolRuntime:
    """Wrapper for tools to access runtime information."""

    runtime: Runtime

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying runtime."""
        return getattr(self.runtime, name)

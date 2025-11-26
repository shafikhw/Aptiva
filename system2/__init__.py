"""
System 2 (Lebanon) package.

Currently a stub with placeholder modules to keep a parallel structure with
System 1 while we build Lebanon-specific logic.
"""

from . import real_estate_agent
from .session import System2AgentSession


def run() -> None:
    """Start the System 2 placeholder flow."""
    real_estate_agent.main()


__all__ = ["run", "real_estate_agent", "System2AgentSession"]

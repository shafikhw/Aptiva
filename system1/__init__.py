"""
System 1 (US) package.

Exposes a simple `run()` entrypoint so callers (CLI now, web later)
can invoke the existing real estate agent workflow.
"""

from . import real_estate_agent


def run() -> None:
    """Start the System 1 (US) workflow."""
    real_estate_agent.main()


__all__ = ["run", "real_estate_agent"]

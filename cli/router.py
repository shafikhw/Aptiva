"""
Routing helpers for selecting which system to run based on location.

Keep this logic isolated so we can reuse it from a future HTTP endpoint or
frontend without depending on CLI-only code.
"""

from __future__ import annotations

from typing import Callable

SystemHandler = Callable[[], None]

def _system1_runner() -> None:
    from system1 import run as run_system1_flow

    run_system1_flow()


def _system2_runner() -> None:
    from system2 import run as run_system2_flow

    run_system2_flow()


SYSTEM_CHOICES = {
    "1": _system1_runner,
    "us": _system1_runner,
    "usa": _system1_runner,
    "united states": _system1_runner,
    "2": _system2_runner,
    "lb": _system2_runner,
    "lebanon": _system2_runner,
}


def select_system_by_location(location_code: str) -> SystemHandler:
    """
    Return the system runner for a given location code.

    This will later be called from a web route or controller instead of CLI.
    """
    normalized = (location_code or "").strip().lower()
    handler = SYSTEM_CHOICES.get(normalized)
    if not handler:
        raise ValueError("Unsupported location selection")
    return handler


def run_system1() -> None:
    """Convenience wrapper for dispatching to System 1."""
    _system1_runner()


def run_system2() -> None:
    """Convenience wrapper for dispatching to System 2."""
    _system2_runner()

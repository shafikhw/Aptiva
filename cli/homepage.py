"""
CLI homepage simulator for routing between System 1 and System 2.

This will later be replaced by a web homepage; keep the selection logic thin
so it can be reused by HTTP handlers.
"""

from __future__ import annotations

from cli.router import select_system_by_location


def _render_welcome() -> None:
    print("Welcome to Aptiva")
    print("Find your next home faster with our location-specific assistants.")
    print("Choose your location to continue:")
    print("  1) United States")
    print("  2) Lebanon")


def run_homepage_cli() -> None:
    """Entry point for the CLI menu that mimics the future web homepage."""
    _render_welcome()
    while True:
        choice = input("Enter 1 or 2: ").strip()
        try:
            handler = select_system_by_location(choice)
        except ValueError:
            print("Invalid choice. Please type 1 for US or 2 for Lebanon.")
            continue

        # Dispatch to the selected system. In a web setting, this would map to
        # a controller or handler instead of a CLI call.
        handler()
        break

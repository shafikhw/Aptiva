"""
CLI homepage simulator for routing between System 1 and System 2.

Supports interactive prompts as well as non-interactive usage via command-line
arguments so you can launch the desired system straight from the terminal:

    python -m cli.homepage --location us
"""

from __future__ import annotations

import argparse
from typing import Optional

from cli.router import select_system_by_location


def _render_welcome() -> None:
    print("Welcome to Aptiva")
    print("Find your next home faster with our location-specific assistants.")
    print("Choose your location to continue:")
    print("  1) United States")
    print("  2) Lebanon")
    print("  (You can also rerun this command with --location us or --location lb to skip this menu.)")


def run_homepage_cli(initial_choice: Optional[str] = None) -> None:
    """
    Entry point for the CLI menu that mimics the future web homepage.

    Args:
        initial_choice: optional location code (e.g., "1", "us", "lb"). When
            provided, the CLI will route directly to that system; otherwise it
            prompts interactively.
    """
    choice = (initial_choice or "").strip()
    while True:
        if not choice:
            _render_welcome()
            choice = input("Enter 1 or 2: ").strip()
        try:
            handler = select_system_by_location(choice)
        except ValueError:
            print("Invalid choice. Please type 1 for US or 2 for Lebanon (or rerun with --location).")
            choice = ""
            continue

        handler()
        break


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aptiva CLI launcher")
    parser.add_argument(
        "--location",
        "-l",
        metavar="CODE",
        help="Optional location shortcut (1/us = United States, 2/lb = Lebanon).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_homepage_cli(args.location)


if __name__ == "__main__":
    main()

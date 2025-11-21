"""Command line entry for selecting System 1 (US) or System 2 (Lebanon)."""

from cli.homepage import run_homepage_cli


def main() -> None:
    """Run the CLI homepage flow."""
    run_homepage_cli()


if __name__ == "__main__":
    main()

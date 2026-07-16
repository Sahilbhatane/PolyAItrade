"""Entry point: ``python -m ai_trader.tui [--url http://host:port]``."""

from __future__ import annotations

import argparse

from ai_trader.tui.app import run


def main() -> None:
    parser = argparse.ArgumentParser(description="PolyVITrade operator TUI")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the running PolyVITrade FastAPI server",
    )
    args = parser.parse_args()
    run(base_url=args.url)


if __name__ == "__main__":
    main()

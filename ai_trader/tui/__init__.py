"""PolyVITrade operator TUI — the primary keyboard-driven trading interface.

The TUI is a *pure consumer*: it reads a read-model and an event stream over
HTTP from the running FastAPI server and submits *intents*. It holds no
business logic and never touches a broker directly. If this process crashes,
the execution engine keeps running untouched.
"""

__all__ = ["run"]


def run(*args, **kwargs):  # pragma: no cover - thin entry indirection
    from ai_trader.tui.app import run as _run

    return _run(*args, **kwargs)

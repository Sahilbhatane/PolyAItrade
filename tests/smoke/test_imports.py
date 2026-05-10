"""Smoke: critical imports (full tree import can pull optional heavy deps)."""


def test_import_core_modules():
    import ai_trader.agents.orchestrator  # noqa: F401
    import ai_trader.app  # noqa: F401
    import ai_trader.rl.env  # noqa: F401
    import ai_trader.routes.rl  # noqa: F401
    import ai_trader.strategies.registry  # noqa: F401

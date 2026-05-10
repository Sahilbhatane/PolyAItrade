#!/usr/bin/env python3
"""Startup validation checklist (config, imports, optional broker health)."""

from __future__ import annotations

import sys

from ai_trader.config import get_config


def main() -> int:
    ok = True
    try:
        cfg = get_config()
        print(f"[OK] config loaded env={cfg.environment} broker={cfg.broker.name}")
    except Exception as e:
        print(f"[FAIL] config: {e}")
        ok = False

    try:
        import ai_trader.agents.orchestrator  # noqa: F401
        import ai_trader.rl.env  # noqa: F401

        print("[OK] critical imports")
    except Exception as e:
        print(f"[FAIL] imports: {e}")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

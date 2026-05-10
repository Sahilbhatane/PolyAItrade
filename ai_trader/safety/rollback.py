"""CLI helpers to inspect parameter history."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="PolyVITrade parameter rollback helper")
    parser.add_argument("--db", default="ai_trader_parameter_history.db")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--to", dest="hash_prefix", default=None)
    args = parser.parse_args()

    path = Path(args.db)
    if not path.exists():
        print("no_history_db")
        return

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    if args.list:
        for row in cur.execute("SELECT id, ts, hash, payload FROM params ORDER BY id DESC LIMIT 20"):
            print(row[0], row[1], row[2], row[3][:120])
        conn.close()
        return

    if args.hash_prefix:
        row = cur.execute(
            "SELECT payload FROM params WHERE hash LIKE ? ORDER BY id DESC LIMIT 1",
            (args.hash_prefix + "%",),
        ).fetchone()
        conn.close()
        if row:
            print(json.dumps(json.loads(row[0]), indent=2))
        else:
            print("not_found")
        return

    parser.print_help()


if __name__ == "__main__":
    main()

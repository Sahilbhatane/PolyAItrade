"""Tests for the lazy, seekable log reader (virtualization backend)."""

import json

from ai_trader.service.log_reader import parse_line, read_logs, tail_logs


def _write_log(path, n, level_cycle=("INFO", "WARNING", "ERROR")):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            rec = {
                "level": level_cycle[i % len(level_cycle)],
                "event": f"event_{i}",
                "timestamp": f"2024-01-01T00:00:{i % 60:02d}Z",
                "logger": "ai_trader.test" if i % 2 == 0 else "ai_trader.other",
            }
            f.write(json.dumps(rec) + "\n")


def test_parse_line_valid_and_invalid():
    rec = parse_line('{"level": "info", "event": "hi", "logger": "x"}')
    assert rec.level == "INFO"
    assert rec.event == "hi"

    bad = parse_line("not json at all")
    assert bad.level == "UNKNOWN"
    assert bad.raw == "not json at all"


def test_missing_file_returns_empty(tmp_path):
    out = read_logs(str(tmp_path / "nope.log"))
    assert out["records"] == []
    assert out["bof"] is True


def test_tail_returns_newest_last(tmp_path):
    path = tmp_path / "app.log"
    _write_log(path, 10)
    records = tail_logs(str(path), limit=3)
    assert len(records) == 3
    # oldest-first ordering within the page; newest overall is last.
    assert records[-1]["event"] == "event_9"
    assert records[0]["event"] == "event_7"


def test_pagination_walks_backwards(tmp_path):
    path = tmp_path / "app.log"
    _write_log(path, 20)

    page1 = read_logs(str(path), limit=5)
    assert [r["event"] for r in page1["records"]] == [
        "event_15",
        "event_16",
        "event_17",
        "event_18",
        "event_19",
    ]
    assert page1["bof"] is False
    assert page1["next_cursor"] is not None

    page2 = read_logs(str(path), limit=5, cursor=page1["next_cursor"])
    assert [r["event"] for r in page2["records"]] == [
        "event_10",
        "event_11",
        "event_12",
        "event_13",
        "event_14",
    ]


def test_pagination_reaches_bof(tmp_path):
    path = tmp_path / "app.log"
    _write_log(path, 6)
    page = read_logs(str(path), limit=100)
    assert len(page["records"]) == 6
    assert page["bof"] is True
    assert page["next_cursor"] is None


def test_level_filter(tmp_path):
    path = tmp_path / "app.log"
    _write_log(path, 30)
    out = read_logs(str(path), limit=100, level="ERROR")
    assert all(r["level"] == "ERROR" for r in out["records"])
    assert len(out["records"]) == 10


def test_component_and_search_filter(tmp_path):
    path = tmp_path / "app.log"
    _write_log(path, 20)
    out = read_logs(str(path), limit=100, component="other")
    assert all("other" in r["logger"] for r in out["records"])

    out2 = read_logs(str(path), limit=100, search="event_1")
    events = [r["event"] for r in out2["records"]]
    assert "event_1" in events
    assert "event_19" in events  # substring match (event_1, event_10..19)


def test_large_file_bounded_read(tmp_path):
    """100k lines: a small page returns quickly without loading everything."""
    path = tmp_path / "big.log"
    _write_log(path, 100_000)
    out = read_logs(str(path), limit=50)
    assert len(out["records"]) == 50
    assert out["records"][-1]["event"] == "event_99999"
    assert out["bof"] is False

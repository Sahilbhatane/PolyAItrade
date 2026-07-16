"""Lazy, memory-bounded reader for the structured JSON log file.

The Logs screen must remain usable with log files containing millions of
records. We therefore never read the whole file: we scan *backwards* from a
byte cursor in fixed-size blocks, yielding complete lines with their true byte
offsets, apply filters, and stop as soon as ``limit`` matches are collected.
Pagination is expressed as a byte ``cursor`` (the start offset of the oldest
line returned) so the client can request progressively older pages without
re-scanning newer ones.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Iterator

_BLOCK = 64 * 1024


@dataclass
class LogRecord:
    raw: str
    level: str = "UNKNOWN"
    event: str = ""
    timestamp: str = ""
    logger: str = ""
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "event": self.event,
            "timestamp": self.timestamp,
            "logger": self.logger,
            "fields": self.fields,
            "raw": self.raw,
        }


def parse_line(line: str) -> LogRecord:
    """Parse one JSON log line, tolerating malformed / non-JSON lines."""
    line = line.rstrip("\n")
    try:
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError
    except (ValueError, TypeError):
        return LogRecord(raw=line)

    known = {"level", "event", "timestamp", "logger", "logger_name"}
    return LogRecord(
        raw=line,
        level=str(data.get("level", "UNKNOWN")).upper(),
        event=str(data.get("event", "")),
        timestamp=str(data.get("timestamp", "")),
        logger=str(data.get("logger", data.get("logger_name", ""))),
        fields={k: v for k, v in data.items() if k not in known},
    )


def _iter_lines_reverse(f: BinaryIO, end: int) -> Iterator[tuple[int, bytes]]:
    """Yield ``(start_offset, line_bytes)`` from ``end`` backward to file start.

    ``line_bytes`` excludes the trailing newline. Offsets are absolute file
    positions of the first byte of each line.
    """
    pos = end
    buffer = b""
    while pos > 0:
        read_size = min(_BLOCK, pos)
        pos -= read_size
        f.seek(pos)
        buffer = f.read(read_size) + buffer
        while True:
            idx = buffer.rfind(b"\n")
            if idx == -1:
                break
            yield pos + idx + 1, buffer[idx + 1:]
            buffer = buffer[:idx]
    if buffer:
        yield 0, buffer


def _matches(
    record: LogRecord,
    level: str | None,
    component: str | None,
    search: str | None,
) -> bool:
    if level and record.level != level.upper():
        return False
    if component and component.lower() not in record.logger.lower():
        return False
    if search and search.lower() not in record.raw.lower():
        return False
    return True


def read_logs(
    path: str,
    limit: int = 200,
    cursor: int | None = None,
    level: str | None = None,
    component: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """Return up to ``limit`` matching log records ending at ``cursor``.

    Returns dict with ``records`` (oldest-first for natural top-to-bottom
    display), ``next_cursor`` (byte offset to pass for the next older page, or
    ``None`` when the beginning of file is reached), and ``bof``.
    """
    limit = max(1, min(int(limit), 5000))

    if not os.path.exists(path):
        return {"records": [], "next_cursor": None, "bof": True}

    file_size = os.path.getsize(path)
    end = file_size if cursor is None else max(0, min(int(cursor), file_size))

    collected: list[LogRecord] = []
    oldest_offset: int | None = None
    reached_start = True

    with open(path, "rb") as f:
        for offset, raw in _iter_lines_reverse(f, end):
            if not raw.strip():
                continue
            record = parse_line(raw.decode("utf-8", errors="replace"))
            if not _matches(record, level, component, search):
                continue
            collected.append(record)
            oldest_offset = offset
            if len(collected) >= limit:
                reached_start = False
                break

    bof = reached_start
    next_cursor = None if bof or oldest_offset is None else oldest_offset
    records = [r.to_dict() for r in reversed(collected)]
    return {"records": records, "next_cursor": next_cursor, "bof": bof}


def tail_logs(path: str, limit: int = 200, **filters: Any) -> list[dict[str, Any]]:
    """Convenience: newest ``limit`` records (oldest-first) for initial load."""
    return read_logs(path, limit=limit, **filters)["records"]

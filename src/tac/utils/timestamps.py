"""ISO 8601 helpers for Conversation Orchestrator payloads."""

from __future__ import annotations

from datetime import datetime, timezone


def parse_iso8601(value: str | None) -> datetime | None:
    """Parse an ISO 8601 timestamp into a timezone-aware datetime.

    Accepts a trailing ``Z``, which ``datetime.fromisoformat`` rejects before
    Python 3.11. A naive value is taken as UTC. Returns ``None`` for a missing
    or unparseable value.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def elapsed_ms(start: str | None, end: str | None) -> int | None:
    """Whole milliseconds from ``start`` to ``end``.

    ``None`` if either is missing or unparseable, or ``end`` precedes
    ``start`` — a duration is left out rather than guessed.
    """
    start_at = parse_iso8601(start)
    end_at = parse_iso8601(end)
    if start_at is None or end_at is None or end_at < start_at:
        return None
    return int((end_at - start_at).total_seconds() * 1000)

"""Tests for ISO 8601 helpers used on Conversation Orchestrator payloads."""

from datetime import datetime, timezone

import pytest

from tac.utils.timestamps import elapsed_ms, parse_iso8601


class TestParseIso8601:
    def test_trailing_z(self) -> None:
        assert parse_iso8601("2026-09-30T10:00:00Z") == datetime(
            2026, 9, 30, 10, 0, tzinfo=timezone.utc
        )

    def test_offset(self) -> None:
        assert parse_iso8601("2026-09-30T12:00:00+02:00") == datetime(
            2026, 9, 30, 10, 0, tzinfo=timezone.utc
        )

    def test_naive_is_taken_as_utc(self) -> None:
        parsed = parse_iso8601("2026-09-30T10:00:00")
        assert parsed == datetime(2026, 9, 30, 10, 0, tzinfo=timezone.utc)

    def test_fractional_seconds(self) -> None:
        parsed = parse_iso8601("2026-09-30T10:00:00.500Z")
        assert parsed is not None
        assert parsed.microsecond == 500000

    @pytest.mark.parametrize("value", [None, "", "not a timestamp"])
    def test_none_for_missing_or_unparseable(self, value: str | None) -> None:
        assert parse_iso8601(value) is None


class TestElapsedMs:
    def test_whole_milliseconds(self) -> None:
        assert elapsed_ms("2026-09-30T10:00:00Z", "2026-09-30T10:01:30.250Z") == 90250

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            (None, "2026-09-30T10:00:00Z"),
            ("2026-09-30T10:00:00Z", None),
            ("garbage", "2026-09-30T10:00:00Z"),
        ],
    )
    def test_none_when_either_side_is_missing(self, start: str | None, end: str | None) -> None:
        assert elapsed_ms(start, end) is None

    def test_none_when_end_precedes_start(self) -> None:
        assert elapsed_ms("2026-09-30T10:05:00Z", "2026-09-30T10:00:00Z") is None

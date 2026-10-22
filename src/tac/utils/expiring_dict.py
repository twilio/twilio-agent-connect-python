"""A small bounded, time-expiring mapping for short-lived cross-request state."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Generic, TypeVar

_V = TypeVar("_V")

#: Default entry lifetime. Sized for the gap between one Twilio request
#: minting a value and the call's WebSocket connecting.
DEFAULT_TTL_SECONDS = 900.0

#: Default cap on live entries.
DEFAULT_MAX_ENTRIES = 1000


class ExpiringDict(Generic[_V]):
    """A string-keyed mapping whose entries expire and whose size is bounded.

    For state one Twilio request creates and a later one for the same call
    consumes. Normally popped within seconds — but a call that is never
    answered, or whose second request lands on another instance, leaves an
    entry nobody will collect. The TTL and cap make that leak impossible
    rather than unlikely.

    Not thread-safe; single-event-loop use.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        # Insertion-ordered, so the oldest entries are also the first to expire.
        self._entries: OrderedDict[str, tuple[float, _V]] = OrderedDict()

    def _now(self) -> float:
        return time.monotonic()

    def _purge(self) -> None:
        """Drop expired entries, then the oldest survivors if still over capacity."""
        now = self._now()
        while self._entries:
            _key, (expires_at, _value) = next(iter(self._entries.items()))
            if expires_at > now:
                break
            self._entries.popitem(last=False)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def __setitem__(self, key: str, value: _V) -> None:
        self._entries.pop(key, None)
        self._entries[key] = (self._now() + self._ttl, value)
        self._purge()

    def __getitem__(self, key: str) -> _V:
        entry = self._entries.get(key)
        if entry is None or entry[0] <= self._now():
            raise KeyError(key)
        return entry[1]

    def pop(self, key: str, default: _V | None = None) -> _V | None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return default
        expires_at, value = entry
        if expires_at <= self._now():
            return default
        return value

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        entry = self._entries.get(key)
        return entry is not None and entry[0] > self._now()

    def __len__(self) -> int:
        self._purge()
        return len(self._entries)

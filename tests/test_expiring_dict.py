"""Tests for ExpiringDict — the bound that keeps cross-request state from leaking."""

import pytest

from tac.utils.expiring_dict import ExpiringDict


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_dict(clock: FakeClock, **kwargs: float | int) -> ExpiringDict[str]:
    d: ExpiringDict[str] = ExpiringDict(**kwargs)  # type: ignore[arg-type]
    d._now = lambda: clock.now  # type: ignore[method-assign]
    return d


class TestBasics:
    def test_set_get_pop(self) -> None:
        d: ExpiringDict[str] = ExpiringDict()
        d["a"] = "one"

        assert "a" in d
        assert d["a"] == "one"
        assert d.pop("a") == "one"
        assert "a" not in d
        assert d.pop("a") is None
        assert d.pop("a", "fallback") == "fallback"

    def test_getitem_raises_for_missing_key(self) -> None:
        d: ExpiringDict[str] = ExpiringDict()
        with pytest.raises(KeyError):
            d["nope"]

    def test_reassigning_a_key_keeps_one_entry(self) -> None:
        d: ExpiringDict[str] = ExpiringDict()
        d["a"] = "one"
        d["a"] = "two"

        assert len(d) == 1
        assert d.pop("a") == "two"

    def test_rejects_nonsense_construction(self) -> None:
        with pytest.raises(ValueError):
            ExpiringDict(ttl_seconds=0)
        with pytest.raises(ValueError):
            ExpiringDict(max_entries=0)

    def test_non_string_key_is_never_contained(self) -> None:
        d: ExpiringDict[str] = ExpiringDict()
        d["a"] = "one"

        assert 1 not in d


class TestExpiry:
    def test_entry_expires_after_its_ttl(self) -> None:
        clock = FakeClock()
        d = make_dict(clock, ttl_seconds=60)
        d["a"] = "one"

        clock.advance(59)
        assert d.pop("a") == "one"

        d["b"] = "two"
        clock.advance(61)
        assert d.pop("b") is None
        assert "b" not in d
        assert len(d) == 0

    def test_expired_entries_are_purged_on_write(self) -> None:
        """A call that is never answered must not keep its entry forever."""
        clock = FakeClock()
        d = make_dict(clock, ttl_seconds=60)
        for i in range(5):
            d[f"abandoned_{i}"] = "config"

        clock.advance(61)
        d["fresh"] = "config"

        assert len(d) == 1
        assert d.pop("fresh") == "config"


class TestCapacity:
    def test_oldest_entries_are_evicted_over_capacity(self) -> None:
        d: ExpiringDict[str] = ExpiringDict(max_entries=3)
        for i in range(5):
            d[f"k{i}"] = str(i)

        assert len(d) == 3
        assert "k0" not in d
        assert "k1" not in d
        assert d.pop("k4") == "4"

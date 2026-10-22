"""Tests for fitting app metadata into Conversation Orchestrator's limits."""

from tac.utils.conversation_metadata import fit_conversation_metadata


class TestFitConversationMetadata:
    def test_keeps_valid_entries(self) -> None:
        persisted, skipped = fit_conversation_metadata(
            {"appointment_id": "apt_42", "campaign.v2": "spring-1"}
        )
        assert persisted == {"appointment_id": "apt_42", "campaign.v2": "spring-1"}
        assert skipped == []

    def test_none_gives_nothing(self) -> None:
        assert fit_conversation_metadata(None) == ({}, [])

    def test_direction_is_an_ordinary_app_key(self) -> None:
        assert fit_conversation_metadata({"direction": "inbound"}) == (
            {"direction": "inbound"},
            [],
        )

    def test_skips_entries_co_would_reject(self) -> None:
        persisted, skipped = fit_conversation_metadata(
            {
                "count": 3,
                "has space": "x",
                "k" * 129: "x",
                "long_value": "v" * 513,
                "ok": "fine",
            }
        )
        assert persisted == {"ok": "fine"}
        assert sorted(skipped) == sorted(["count", "has space", "k" * 129, "long_value"])

    def test_key_with_a_trailing_newline_is_rejected(self) -> None:
        _persisted, skipped = fit_conversation_metadata({"key\n": "x"})
        assert skipped == ["key\n"]

    def test_caps_at_eight_keys(self) -> None:
        app = {f"k{i}": "v" for i in range(9)}
        persisted, skipped = fit_conversation_metadata(app)
        assert list(persisted) == [f"k{i}" for i in range(8)]
        assert skipped == ["k8"]

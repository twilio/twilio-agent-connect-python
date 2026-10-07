"""Tests for fitting app metadata into Conversation Orchestrator's limits."""

from tac.utils.conversation_metadata import fit_conversation_metadata

RESERVED = {"direction": "outbound"}


class TestFitConversationMetadata:
    def test_keeps_reserved_and_valid_entries(self) -> None:
        persisted, skipped = fit_conversation_metadata(
            {"appointment_id": "apt_42", "campaign.v2": "spring-1"}, reserved=RESERVED
        )
        assert persisted == {
            "direction": "outbound",
            "appointment_id": "apt_42",
            "campaign.v2": "spring-1",
        }
        assert skipped == []

    def test_none_gives_only_reserved(self) -> None:
        assert fit_conversation_metadata(None, reserved=RESERVED) == (RESERVED, [])

    def test_reserved_wins_over_an_app_key_of_the_same_name(self) -> None:
        persisted, skipped = fit_conversation_metadata({"direction": "inbound"}, reserved=RESERVED)
        assert persisted == RESERVED
        assert skipped == []

    def test_skips_entries_co_would_reject(self) -> None:
        persisted, skipped = fit_conversation_metadata(
            {
                "count": 3,
                "has space": "x",
                "k" * 129: "x",
                "long_value": "v" * 513,
                "ok": "fine",
            },
            reserved=RESERVED,
        )
        assert persisted == {"direction": "outbound", "ok": "fine"}
        assert sorted(skipped) == sorted(["count", "has space", "k" * 129, "long_value"])

    def test_key_with_a_trailing_newline_is_rejected(self) -> None:
        _persisted, skipped = fit_conversation_metadata({"key\n": "x"}, reserved=RESERVED)
        assert skipped == ["key\n"]

    def test_caps_at_eight_keys_including_reserved(self) -> None:
        app = {f"k{i}": "v" for i in range(9)}
        persisted, skipped = fit_conversation_metadata(app, reserved=RESERVED)
        assert len(persisted) == 8
        assert list(persisted)[0] == "direction"
        assert skipped == ["k7", "k8"]

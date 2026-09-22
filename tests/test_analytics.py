"""Tests for product usage telemetry.

The tracking plan these events are validated against rejects an entire event
when it carries an undeclared property or one of the wrong type, server-side
and without any signal back to the SDK. The property-set and integer-type
assertions here are what catch that before it ships.
"""

import os
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tac import TAC, TACConfig
from tac.channels.chat import ChatChannel
from tac.channels.rcs import RCSChannel
from tac.channels.sms import SMSChannel
from tac.channels.voice import VoiceChannel
from tac.channels.whatsapp import WhatsAppChannel
from tac.core import analytics
from tac.core.analytics import _reset_analytics, shutdown_analytics, track_event

# Every property the tracking plan declares per event. A key emitted outside
# its event's set is silently dropped along with the whole event, so these sets
# are the contract, not a convenience.
_COMMON = {"account_sid", "channel", "sdk_version", "sdk_package"}
_VOICE_ONLY = {"provider", "orchestrator_enabled"}
EXPECTED_PROPERTIES: dict[str, set[str]] = {
    "Conversation Started": _COMMON | {"conversation_id", "has_profile_id"},
    "Conversation Ended": _COMMON | {"conversation_id", "duration_ms"},
    "Message Received": _COMMON | {"conversation_id"},
    "Response Sent": _COMMON | {"conversation_id", "response_type"} | _VOICE_ONLY,
    "Conversation Initialized": _COMMON | {"conversation_id"} | _VOICE_ONLY,
    "Voice Interrupt": (_COMMON | {"conversation_id", "duration_until_interrupt_ms"} | _VOICE_ONLY),
    "Websocket Connected": _COMMON | _VOICE_ONLY,
    "Websocket Disconnected": _COMMON | {"conversation_id"} | _VOICE_ONLY,
}

# Integer in the tracking plan, and a float is a violation that drops the
# event. Python's duration arithmetic produces floats by default, so this is
# the easiest property type to get wrong.
INTEGER_PROPERTIES = {"duration_ms", "duration_until_interrupt_ms"}


def get_test_config() -> TACConfig:
    return TACConfig(
        account_sid="ACtest123",
        auth_token="test_token",
        api_key="SKtest123",
        api_secret="test_secret",
        phone_number="+15551234567",
        conversation_configuration_id="conv_configuration_test123",
        rcs_sender_id="rcs_test_sender",
        whatsapp_number="+15559876543",
    )


@pytest.fixture
def mock_client() -> Iterator[MagicMock]:
    """Enable analytics with a mocked Segment client."""
    _reset_analytics()
    os.environ.pop("TAC_ANALYTICS_DISABLED", None)
    client = MagicMock()
    with patch.object(analytics, "Client", return_value=client):
        yield client
    _reset_analytics()
    os.environ["TAC_ANALYTICS_DISABLED"] = "true"


def tracked(client: MagicMock) -> list[dict[str, Any]]:
    """The kwargs of every track() call made on the mocked client."""
    return [call.kwargs for call in client.track.call_args_list]


def only(client: MagicMock, event: str) -> dict[str, Any]:
    """The single call for ``event``, failing if there isn't exactly one."""
    matches = [c for c in tracked(client) if c["event"] == event]
    assert len(matches) == 1, f"expected exactly one {event!r}, got {len(matches)}"
    return matches[0]


class TestTrackEvent:
    def test_event_shape(self, mock_client: MagicMock) -> None:
        track_event(
            "Websocket Connected",
            "AC123",
            channel="voice",
            provider="conversation_relay",
            orchestrator_enabled=True,
        )

        call = only(mock_client, "Websocket Connected")
        assert call["anonymous_id"] == "AC123"
        assert call["properties"]["account_sid"] == "AC123"
        assert call["properties"]["channel"] == "voice"
        assert call["properties"]["sdk_package"] == "twilio-agent-connect-python"
        assert isinstance(call["properties"]["sdk_version"], str)

    def test_omits_properties_not_supplied(self, mock_client: MagicMock) -> None:
        track_event("Message Received", "AC123", channel="sms", conversation_id="conv-1")

        properties = only(mock_client, "Message Received")["properties"]
        assert set(properties) == {
            "account_sid",
            "channel",
            "conversation_id",
            "sdk_version",
            "sdk_package",
        }

    def test_account_sid_is_anonymous_id(self, mock_client: MagicMock) -> None:
        track_event("Conversation Started", "AC456", channel="sms", conversation_id="c")

        assert only(mock_client, "Conversation Started")["anonymous_id"] == "AC456"

    def test_sdk_identity_cannot_be_overridden(self, mock_client: MagicMock) -> None:
        """A caller-supplied value must not displace the SDK's own identity."""
        track_event("Message Received", "AC123", sdk_package="spoofed", sdk_version="9.9.9")

        properties = only(mock_client, "Message Received")["properties"]
        assert properties["sdk_package"] == "twilio-agent-connect-python"
        assert properties["sdk_version"] != "9.9.9"

    def test_disabled_by_env_var(self, mock_client: MagicMock) -> None:
        os.environ["TAC_ANALYTICS_DISABLED"] = "true"
        _reset_analytics()

        track_event("Websocket Connected", "AC123", channel="voice")

        mock_client.track.assert_not_called()

    def test_never_raises_when_track_fails(self, mock_client: MagicMock) -> None:
        mock_client.track.side_effect = RuntimeError("network error")

        track_event("Websocket Connected", "AC123", channel="voice")

    def test_never_raises_when_client_construction_fails(self) -> None:
        _reset_analytics()
        os.environ.pop("TAC_ANALYTICS_DISABLED", None)
        try:
            with patch.object(analytics, "Client", side_effect=RuntimeError("bad key")):
                track_event("Websocket Connected", "AC123", channel="voice")
        finally:
            _reset_analytics()
            os.environ["TAC_ANALYTICS_DISABLED"] = "true"


class TestShutdown:
    def test_flushes_the_client(self, mock_client: MagicMock) -> None:
        track_event("Websocket Connected", "AC123", channel="voice")

        shutdown_analytics()

        mock_client.shutdown.assert_called_once()

    def test_no_op_without_a_client(self, mock_client: MagicMock) -> None:
        shutdown_analytics()

        mock_client.shutdown.assert_not_called()

    def test_survives_a_blocking_client(self, mock_client: MagicMock) -> None:
        """A client that never returns must not hang the caller."""
        mock_client.shutdown.side_effect = lambda: __import__("time").sleep(30)
        track_event("Websocket Connected", "AC123", channel="voice")

        with patch.object(analytics, "_FLUSH_TIMEOUT_SECONDS", 0.05):
            shutdown_analytics()

    def test_is_idempotent(self, mock_client: MagicMock) -> None:
        track_event("Websocket Connected", "AC123", channel="voice")

        shutdown_analytics()
        shutdown_analytics()

        mock_client.shutdown.assert_called_once()


class TestPropertyContract:
    """Every emitted property must be declared for its event, at the right type."""

    def test_integer_durations(self, mock_client: MagicMock) -> None:
        track_event(
            "Conversation Ended",
            "AC123",
            channel="sms",
            conversation_id="c",
            duration_ms=int(timedelta(seconds=1.5).total_seconds() * 1000),
        )

        duration = only(mock_client, "Conversation Ended")["properties"]["duration_ms"]
        assert isinstance(duration, int) and not isinstance(duration, bool)


def assert_contract(client: MagicMock) -> None:
    """Assert every tracked call matches the tracking plan's declared schema."""
    calls = tracked(client)
    assert calls, "expected at least one tracked event"
    for call in calls:
        event = call["event"]
        assert event in EXPECTED_PROPERTIES, f"undeclared event {event!r}"
        properties = call["properties"]
        undeclared = set(properties) - EXPECTED_PROPERTIES[event]
        assert not undeclared, f"{event!r} emitted undeclared properties: {undeclared}"
        for key in INTEGER_PROPERTIES & set(properties):
            value = properties[key]
            assert isinstance(value, int) and not isinstance(value, bool), (
                f"{event!r} property {key!r} must be an int, got {type(value).__name__}"
            )
        assert properties["account_sid"] == call["anonymous_id"]


class TestMessagingCallSites:
    @pytest.fixture
    def tac(self) -> TAC:
        return TAC(get_test_config())

    @pytest.mark.parametrize(
        ("channel_cls", "expected"),
        [
            (SMSChannel, "sms"),
            (RCSChannel, "rcs"),
            (WhatsAppChannel, "whatsapp"),
            (ChatChannel, "chat"),
        ],
    )
    def test_channel_label_is_lowercase(
        self, mock_client: MagicMock, tac: TAC, channel_cls: Any, expected: str
    ) -> None:
        channel = channel_cls(tac)
        channel._start_conversation("conv-1")

        assert only(mock_client, "Conversation Started")["properties"]["channel"] == expected
        assert_contract(mock_client)

    def test_conversation_started_reports_profile_presence(
        self, mock_client: MagicMock, tac: TAC
    ) -> None:
        channel = SMSChannel(tac)
        channel._start_conversation("conv-1", profile_id="PRtest123")

        properties = only(mock_client, "Conversation Started")["properties"]
        assert properties["has_profile_id"] is True
        assert properties["conversation_id"] == "conv-1"

    def test_conversation_started_emits_once_per_session(
        self, mock_client: MagicMock, tac: TAC
    ) -> None:
        channel = SMSChannel(tac)
        channel._start_conversation("conv-1")
        channel._start_conversation("conv-1")

        assert len([c for c in tracked(mock_client) if c["event"] == "Conversation Started"]) == 1

    @pytest.mark.asyncio
    async def test_conversation_ended_duration_is_an_int(
        self, mock_client: MagicMock, tac: TAC
    ) -> None:
        channel = SMSChannel(tac)
        session = channel._start_conversation("conv-1")
        session.started_at = datetime.now() - timedelta(milliseconds=1500)

        await channel._end_conversation("conv-1")

        properties = only(mock_client, "Conversation Ended")["properties"]
        assert isinstance(properties["duration_ms"], int)
        assert properties["duration_ms"] >= 1500
        assert_contract(mock_client)

    @pytest.mark.asyncio
    async def test_conversation_ended_silent_without_a_session(
        self, mock_client: MagicMock, tac: TAC
    ) -> None:
        channel = SMSChannel(tac)

        await channel._end_conversation("conv-unknown")

        assert not tracked(mock_client)

    @pytest.mark.asyncio
    async def test_response_sent_on_success(self, mock_client: MagicMock, tac: TAC) -> None:
        channel = SMSChannel(tac)
        session = channel._start_conversation("conv-1")
        session.author_info = MagicMock(participant_id="p-customer", address="+15551112222")
        session.ai_agent_info = MagicMock(participant_id="p-agent")
        channel.conversation_orchestrator_client.create_action = AsyncMock()

        await channel.send_response("conv-1", "hello")

        properties = only(mock_client, "Response Sent")["properties"]
        assert properties["response_type"] == "full"
        assert properties["channel"] == "sms"
        # Messaging carries neither of the voice-only properties.
        assert "provider" not in properties
        assert "orchestrator_enabled" not in properties
        assert_contract(mock_client)

    @pytest.mark.asyncio
    async def test_response_sent_suppressed_when_send_fails(
        self, mock_client: MagicMock, tac: TAC
    ) -> None:
        """send_response swallows the error, so tracking must sit inside the try."""
        channel = SMSChannel(tac)
        session = channel._start_conversation("conv-1")
        session.author_info = MagicMock(participant_id="p-customer", address="+15551112222")
        session.ai_agent_info = MagicMock(participant_id="p-agent")
        channel.conversation_orchestrator_client.create_action = AsyncMock(
            side_effect=RuntimeError("Orchestrator down")
        )

        await channel.send_response("conv-1", "hello")

        assert not [c for c in tracked(mock_client) if c["event"] == "Response Sent"]


class TestVoiceCallSites:
    @pytest.fixture
    def tac(self) -> TAC:
        return TAC(get_test_config())

    @pytest.mark.asyncio
    async def test_websocket_connected_has_no_conversation_id(
        self, mock_client: MagicMock, tac: TAC
    ) -> None:
        channel = VoiceChannel(tac)
        channel._provider.handle_websocket = AsyncMock()

        await channel.handle_websocket(MagicMock())

        properties = only(mock_client, "Websocket Connected")["properties"]
        assert "conversation_id" not in properties
        assert properties["channel"] == "voice"
        assert properties["provider"] == "conversation_relay"
        assert properties["orchestrator_enabled"] is True
        assert_contract(mock_client)

    @pytest.mark.asyncio
    async def test_voice_response_sent_reports_provider(
        self, mock_client: MagicMock, tac: TAC
    ) -> None:
        channel = VoiceChannel(tac)
        channel._start_conversation("conv-1")
        channel._provider._websocket_manager.add_websocket("conv-1", AsyncMock())

        await channel.send_response("conv-1", "hello")

        properties = only(mock_client, "Response Sent")["properties"]
        assert properties["response_type"] == "full"
        assert properties["channel"] == "voice"
        assert properties["provider"] == "conversation_relay"
        assert_contract(mock_client)

    @pytest.mark.asyncio
    async def test_voice_response_type_streaming(self, mock_client: MagicMock, tac: TAC) -> None:
        async def stream() -> Any:
            yield "hello"

        channel = VoiceChannel(tac)
        channel._start_conversation("conv-1")
        channel._provider._websocket_manager.add_websocket("conv-1", AsyncMock())

        await channel.send_response("conv-1", stream())

        assert only(mock_client, "Response Sent")["properties"]["response_type"] == "streaming"

    @pytest.mark.asyncio
    async def test_auto_sent_reply_is_reported(self, mock_client: MagicMock, tac: TAC) -> None:
        """A reply from the message-ready callback bypasses the channel wrapper.

        The provider auto-sends it directly, so tracking on
        ``VoiceChannel.send_response`` would never see it.
        """
        channel = VoiceChannel(tac)
        channel._start_conversation("conv-1")
        channel._provider._websocket_manager.add_websocket("conv-1", AsyncMock())

        await channel._provider.send_response("conv-1", "hello", role="assistant")

        assert only(mock_client, "Response Sent")["properties"]["channel"] == "voice"
        assert_contract(mock_client)

    @pytest.mark.asyncio
    async def test_response_sent_suppressed_without_a_websocket(
        self, mock_client: MagicMock, tac: TAC
    ) -> None:
        channel = VoiceChannel(tac)
        channel._start_conversation("conv-1")

        await channel.send_response("conv-1", "hello")

        assert not [c for c in tracked(mock_client) if c["event"] == "Response Sent"]

    @pytest.mark.asyncio
    async def test_response_sent_suppressed_on_interrupt(
        self, mock_client: MagicMock, tac: TAC
    ) -> None:
        """An interrupted reply never fully reached the caller."""
        import asyncio

        websocket = AsyncMock()
        websocket.send_text.side_effect = asyncio.CancelledError()
        channel = VoiceChannel(tac)
        channel._start_conversation("conv-1")
        channel._provider._websocket_manager.add_websocket("conv-1", websocket)

        with pytest.raises(asyncio.CancelledError):
            await channel.send_response("conv-1", "hello")

        assert not [c for c in tracked(mock_client) if c["event"] == "Response Sent"]

    def test_relay_only_reports_orchestrator_disabled(self, mock_client: MagicMock) -> None:
        config = get_test_config()
        config.conversation_configuration_id = None
        channel = VoiceChannel(TAC(config))

        channel._provider._track_conversation_initialized("conv-1")

        properties = only(mock_client, "Conversation Initialized")["properties"]
        assert properties["orchestrator_enabled"] is False
        assert properties["provider"] == "conversation_relay"
        assert_contract(mock_client)

    def test_custom_provider_id_default(self) -> None:
        """A provider defined outside the SDK still reports a usable value."""
        from tac.channels.voice.provider import VoiceProvider

        assert VoiceProvider(MagicMock()).provider_id == "custom"


class TestVoiceInterrupt:
    @pytest.fixture
    def channel(self) -> VoiceChannel:
        return VoiceChannel(TAC(get_test_config()))

    def test_reports_duration_as_an_int(
        self, mock_client: MagicMock, channel: VoiceChannel
    ) -> None:
        from tac.models.voice import InterruptMessage

        channel._start_conversation("conv-1")
        channel.tac.trigger_interrupt = MagicMock()

        channel._provider._handle_interrupt(
            "conv-1",
            InterruptMessage(type="interrupt", durationUntilInterruptMs=1234),
        )

        properties = only(mock_client, "Voice Interrupt")["properties"]
        assert properties["duration_until_interrupt_ms"] == 1234
        assert isinstance(properties["duration_until_interrupt_ms"], int)
        assert_contract(mock_client)

    def test_omits_duration_when_not_reported(
        self, mock_client: MagicMock, channel: VoiceChannel
    ) -> None:
        """Omitted, not null — a null would violate the declared integer type."""
        from tac.models.voice import InterruptMessage

        channel._start_conversation("conv-1")
        channel.tac.trigger_interrupt = MagicMock()

        channel._provider._handle_interrupt("conv-1", InterruptMessage(type="interrupt"))

        properties = only(mock_client, "Voice Interrupt")["properties"]
        assert "duration_until_interrupt_ms" not in properties
        assert_contract(mock_client)

    def test_silent_for_unknown_conversation(
        self, mock_client: MagicMock, channel: VoiceChannel
    ) -> None:
        from tac.models.voice import InterruptMessage

        channel._provider._handle_interrupt("conv-unknown", InterruptMessage(type="interrupt"))

        assert not [c for c in tracked(mock_client) if c["event"] == "Voice Interrupt"]

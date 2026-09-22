"""Tests for product usage telemetry.

Events are validated against a fixed schema after they leave the SDK, and one
carrying an unexpected property or the wrong type for a known one is discarded
whole, with no signal back to the caller. The property-set and integer-type
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
from tac.channels.websocket_protocol import WebSocketDisconnectError
from tac.channels.whatsapp import WhatsAppChannel
from tac.core import analytics
from tac.core.analytics import _reset_analytics, shutdown_analytics, track_event
from tac.models.handoff import PendingHandoffData

# Every property each event is allowed to carry. A key emitted outside its
# event's set is silently dropped along with the whole event, so these sets are
# the contract, not a convenience.
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

# Typed as integers, and a float is a violation that drops the event. Python's
# duration arithmetic produces floats by default, so this is the easiest
# property type to get wrong.
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

    def test_none_valued_properties_are_dropped(self, mock_client: MagicMock) -> None:
        """Omitting a property is always accepted; a null against a typed
        property is a violation that discards the whole event."""
        track_event(
            "Voice Interrupt",
            "AC123",
            channel="voice",
            conversation_id="conv-1",
            duration_until_interrupt_ms=None,
        )

        properties = only(mock_client, "Voice Interrupt")["properties"]
        assert "duration_until_interrupt_ms" not in properties

    @pytest.mark.parametrize(
        ("event", "key", "value"),
        [
            ("Conversation Started", "has_profile_id", False),
            ("Conversation Ended", "duration_ms", 0),
            ("Websocket Connected", "orchestrator_enabled", False),
        ],
    )
    def test_falsy_values_are_kept(
        self, mock_client: MagicMock, event: str, key: str, value: Any
    ) -> None:
        """Dropping must key off None, not falsiness — these are real values."""
        track_event(event, "AC123", channel="sms", **{key: value})

        assert only(mock_client, event)["properties"][key] == value
        assert_contract(mock_client)

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
    """Assert every tracked call matches the schema declared above."""
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


def media_streams_channel(kind: str) -> VoiceChannel:
    """A VoiceChannel backed by one of the Media Streams providers."""
    if kind == "gpt_live":
        from tac.channels.voice.media_streams.gpt_live import GPTLiveProviderConfig

        config: Any = GPTLiveProviderConfig(
            openai_api_key="sk-test",
            default_session_config={"model": "gpt-live-1"},
        )
    else:
        from tac.channels.voice.media_streams.openai_realtime import (
            OpenAIRealtimeProviderConfig,
        )

        config = OpenAIRealtimeProviderConfig(openai_api_key="sk-test")
    return VoiceChannel(TAC(get_test_config()), config=config)


@pytest.mark.parametrize(
    ("kind", "expected_provider"),
    [("gpt_live", "gpt_live"), ("openai_realtime", "openai_realtime")],
)
class TestMediaStreamsCallSites:
    """The Media Streams providers emit their own lifecycle events.

    These paths run under the other provider suites, which proves they don't
    raise, but nothing there asserts what they report — so the event names and
    properties are checked here.
    """

    def test_provider_id(self, kind: str, expected_provider: str) -> None:
        assert media_streams_channel(kind)._provider.provider_id == expected_provider

    def test_telemetry_channel_is_voice_not_the_transport(
        self, kind: str, expected_provider: str
    ) -> None:
        """`get_channel_name()` returns the transport here, which is why the
        telemetry label can't be derived from it."""
        channel = media_streams_channel(kind)

        assert channel.get_channel_name().startswith("VOICE_MEDIA_STREAM")
        assert channel._telemetry_channel == "voice"

    def test_conversation_initialized_on_stream_start(
        self, mock_client: MagicMock, kind: str, expected_provider: str
    ) -> None:
        channel = media_streams_channel(kind)

        channel._provider._register_call({"streamSid": "MZ123", "callSid": "CA123"}, MagicMock())

        properties = only(mock_client, "Conversation Initialized")["properties"]
        assert properties["conversation_id"] == "CA123"
        assert properties["channel"] == "voice"
        assert properties["provider"] == expected_provider
        assert properties["orchestrator_enabled"] is True
        assert_contract(mock_client)

    @pytest.mark.asyncio
    async def test_websocket_disconnected_precedes_conversation_ended(
        self, mock_client: MagicMock, kind: str, expected_provider: str
    ) -> None:
        channel = media_streams_channel(kind)
        channel._provider._register_call({"streamSid": "MZ123", "callSid": "CA123"}, MagicMock())

        await channel._provider._cleanup_call("CA123")

        events = [c["event"] for c in tracked(mock_client)]
        assert events.index("Websocket Disconnected") < events.index("Conversation Ended")
        properties = only(mock_client, "Websocket Disconnected")["properties"]
        assert properties["conversation_id"] == "CA123"
        assert properties["provider"] == expected_provider
        assert_contract(mock_client)


class TestOpenAIRealtimeBargeIn:
    """Realtime reports its own barge-in; GPT-Live exposes no interrupt signal."""

    @pytest.mark.asyncio
    async def test_voice_interrupt_reports_audio_heard(self, mock_client: MagicMock) -> None:
        from tac.channels.voice.media_streams.openai_realtime.models import _CallState

        channel = media_streams_channel("openai_realtime")
        provider = channel._provider
        provider._model_send = AsyncMock()
        provider._twilio_send = AsyncMock()
        session = channel._start_conversation("CA123")
        session.metadata["stream_sid"] = "MZ123"
        call = _CallState(twilio_ws=MagicMock())
        call.barge_in.last_assistant_item = "item-1"
        call.barge_in.current_item_audio_ms = 1234
        provider._calls["CA123"] = call

        await provider._handle_barge_in("CA123", session, call)

        properties = only(mock_client, "Voice Interrupt")["properties"]
        assert properties["duration_until_interrupt_ms"] == 1234
        assert isinstance(properties["duration_until_interrupt_ms"], int)
        assert properties["provider"] == "openai_realtime"
        assert_contract(mock_client)

    @pytest.mark.asyncio
    async def test_silent_when_no_reply_was_playing(self, mock_client: MagicMock) -> None:
        """The early return means a bare VAD trigger is not an interrupt."""
        from tac.channels.voice.media_streams.openai_realtime.models import _CallState

        channel = media_streams_channel("openai_realtime")
        provider = channel._provider
        provider._model_send = AsyncMock()
        provider._twilio_send = AsyncMock()
        session = channel._start_conversation("CA123")
        call = _CallState(twilio_ws=MagicMock())
        assert call.barge_in.last_assistant_item is None
        provider._calls["CA123"] = call

        await provider._handle_barge_in("CA123", session, call)

        assert not [c for c in tracked(mock_client) if c["event"] == "Voice Interrupt"]

    def test_gpt_live_has_no_interrupt_site(self) -> None:
        """Documents the accepted asymmetry, so removing it is a visible change."""
        import inspect

        from tac.channels.voice.media_streams.gpt_live import provider as gpt_live

        assert "Voice Interrupt" not in inspect.getsource(gpt_live)


class TestConversationInitializedIsEmittedOnce:
    """The two initialization paths are mutually exclusive.

    ``_track_conversation_initialized`` is called from both
    ``_initialize_conversation`` (orchestrated) and the relay-only branch of
    ``handle_websocket``, because Python has no single point where the two
    converge. Which one runs is decided by whether an init task was created at
    ``setup``, so exactly one may fire per connection.
    """

    @staticmethod
    def relay_only_tac() -> TAC:
        config = get_test_config()
        config.conversation_configuration_id = None
        return TAC(config)

    @staticmethod
    def socket(*messages: dict[str, Any]) -> AsyncMock:
        websocket = AsyncMock()
        websocket.receive_json.side_effect = [*messages, Exception("stop-iteration")]
        return websocket

    @pytest.mark.asyncio
    async def test_relay_only_emits_exactly_once(self, mock_client: MagicMock) -> None:
        channel = VoiceChannel(self.relay_only_tac())
        initialize = AsyncMock()
        channel._provider._initialize_conversation = initialize  # type: ignore[method-assign]

        await channel.handle_websocket(
            self.socket(
                {"type": "setup", "callSid": "CA_relay", "from": "+15551230000"},
                {"type": "prompt", "voicePrompt": "hello", "final": True},
                {"type": "prompt", "voicePrompt": "again", "final": True},
            )
        )

        initialized = [c for c in tracked(mock_client) if c["event"] == "Conversation Initialized"]
        assert len(initialized) == 1
        assert initialized[0]["properties"]["conversation_id"] == "CA_relay"
        assert initialized[0]["properties"]["orchestrator_enabled"] is False
        # The orchestrated coroutine is the other emitter; it must not run.
        initialize.assert_not_called()
        assert_contract(mock_client)

    @pytest.mark.asyncio
    async def test_orchestrated_does_not_take_the_relay_only_branch(
        self, mock_client: MagicMock
    ) -> None:
        """With the orchestrated coroutine stubbed out, nothing should emit.

        Any `Conversation Initialized` here would mean the relay-only branch
        ran in orchestrated mode — i.e. a second, duplicate emitter.
        """
        channel = VoiceChannel(TAC(get_test_config()))
        channel._provider._initialize_conversation = AsyncMock(  # type: ignore[method-assign]
            return_value=("conv_orchestrated", None)
        )

        await channel.handle_websocket(
            self.socket(
                {"type": "setup", "callSid": "CA_orch", "from": "+15551230000"},
                {"type": "prompt", "voicePrompt": "hello", "final": True},
            )
        )

        assert not [c for c in tracked(mock_client) if c["event"] == "Conversation Initialized"]


class TestVoiceResponseDelivery:
    """`Response Sent` must mean a complete response reached the caller.

    ``ConversationRelayProvider.send_response`` swallows websocket failures
    locally rather than letting them reach its outer handler, so every way a
    send can fall short needs its own case — the outer ``try`` is not the
    boundary it looks like.
    """

    @pytest.fixture
    def tac(self) -> TAC:
        return TAC(get_test_config())

    def prepare(self, tac: TAC, websocket: Any) -> VoiceChannel:
        channel = VoiceChannel(tac)
        channel._start_conversation("conv-1")
        channel._provider._websocket_manager.add_websocket("conv-1", websocket)
        return channel

    def sent(self, client: MagicMock) -> list[dict[str, Any]]:
        return [c for c in tracked(client) if c["event"] == "Response Sent"]

    @pytest.mark.asyncio
    async def test_reports_a_fully_streamed_response(
        self, mock_client: MagicMock, tac: TAC
    ) -> None:
        async def stream() -> Any:
            yield "hel"
            yield "lo"

        channel = self.prepare(tac, AsyncMock())

        await channel.send_response("conv-1", stream())

        assert len(self.sent(mock_client)) == 1
        assert_contract(mock_client)

    @pytest.mark.asyncio
    async def test_suppressed_when_socket_dies_mid_stream(
        self, mock_client: MagicMock, tac: TAC
    ) -> None:
        """The token send fails, is swallowed locally, and the marker is skipped."""

        async def stream() -> Any:
            yield "first"
            yield "second"

        websocket = AsyncMock()
        websocket.send_text.side_effect = [None, WebSocketDisconnectError()]
        channel = self.prepare(tac, websocket)

        await channel.send_response("conv-1", stream())

        assert not self.sent(mock_client)

    @pytest.mark.asyncio
    async def test_suppressed_when_final_marker_fails(
        self, mock_client: MagicMock, tac: TAC
    ) -> None:
        """Without the `last` marker the turn never completes for the caller."""

        async def stream() -> Any:
            yield "hello"

        websocket = AsyncMock()
        websocket.send_text.side_effect = [None, WebSocketDisconnectError()]
        channel = self.prepare(tac, websocket)

        await channel.send_response("conv-1", stream())

        assert not self.sent(mock_client)

    @pytest.mark.asyncio
    async def test_suppressed_for_an_empty_stream(self, mock_client: MagicMock, tac: TAC) -> None:
        """Nothing reached the caller, so there is no response to report."""

        async def stream() -> Any:
            return
            yield  # pragma: no cover - makes this an async generator

        channel = self.prepare(tac, AsyncMock())

        await channel.send_response("conv-1", stream())

        assert not self.sent(mock_client)

    @pytest.mark.asyncio
    async def test_suppressed_for_an_empty_string(self, mock_client: MagicMock, tac: TAC) -> None:
        channel = self.prepare(tac, AsyncMock())

        await channel.send_response("conv-1", "")

        assert not self.sent(mock_client)

    @pytest.mark.asyncio
    async def test_reported_when_only_the_handoff_send_fails(
        self, mock_client: MagicMock, tac: TAC
    ) -> None:
        """The response itself was delivered; only the transfer was lost."""
        websocket = AsyncMock()
        websocket.send_text.side_effect = [None, WebSocketDisconnectError()]
        channel = self.prepare(tac, websocket)
        channel._conversations["conv-1"].pending_handoff_data = PendingHandoffData(handoffData="{}")

        await channel.send_response("conv-1", "hello")

        assert len(self.sent(mock_client)) == 1
        assert_contract(mock_client)


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

    def test_reported_even_when_the_callback_raises(
        self, mock_client: MagicMock, channel: VoiceChannel
    ) -> None:
        """`trigger_interrupt` does not guard a synchronous callback.

        The interrupt happened regardless of what the application does with
        it, so a raising callback must not erase the record.
        """
        from tac.models.voice import InterruptMessage

        def explode(session: Any, interrupt_data: Any) -> None:
            raise RuntimeError("application callback failed")

        channel._start_conversation("conv-1")
        channel.tac.on_interrupt(explode)

        with pytest.raises(RuntimeError, match="application callback failed"):
            channel._provider._handle_interrupt(
                "conv-1", InterruptMessage(type="interrupt", durationUntilInterruptMs=900)
            )

        properties = only(mock_client, "Voice Interrupt")["properties"]
        assert properties["duration_until_interrupt_ms"] == 900
        assert_contract(mock_client)

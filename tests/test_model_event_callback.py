"""Tests for the on_model_event callback and its wiring into Media Streams providers.

Covers:
- TAC.on_model_event / TAC.trigger_model_event registration and sync/async invocation.
- MediaStreamsOpenAIProvider forwarding every raw model event to trigger_model_event
  before it is interpreted by _dispatch_model_event.
"""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tac import TAC
from tac.channels.voice import VoiceChannel
from tac.channels.voice.media_streams.openai_realtime import OpenAIRealtimeProviderConfig
from tac.channels.voice.media_streams.openai_realtime.provider import (
    TWILIO_MEDIA_STREAM_AUDIO_FORMAT,
)


def get_test_config() -> dict[str, Any]:
    return {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": "conv_configuration_test123",
        "phone_number": "+15551234567",
        "voice_public_domain": "example.com",
    }


class TestModelEventCallbackRegistration:
    """TAC.on_model_event registration and trigger_model_event dispatch."""

    @pytest.mark.asyncio
    async def test_sync_model_event_callback(self) -> None:
        tac = TAC(get_test_config())
        received: list[tuple[str, dict[str, Any]]] = []

        def handler(conversation_id: str, event: dict[str, Any]) -> None:
            received.append((conversation_id, event))

        tac.on_model_event(handler)

        await tac.trigger_model_event("CA1", {"type": "turn.done"})

        assert len(received) == 1
        assert received[0] == ("CA1", {"type": "turn.done"})

    @pytest.mark.asyncio
    async def test_async_model_event_callback(self) -> None:
        tac = TAC(get_test_config())
        received: list[tuple[str, dict[str, Any]]] = []

        async def handler(conversation_id: str, event: dict[str, Any]) -> None:
            received.append((conversation_id, event))

        tac.on_model_event(handler)

        await tac.trigger_model_event("CA2", {"type": "response.audio_transcript.delta"})

        assert len(received) == 1
        assert received[0] == ("CA2", {"type": "response.audio_transcript.delta"})

    @pytest.mark.asyncio
    async def test_no_handler_is_noop(self) -> None:
        """No handler registered → trigger_model_event is a no-op."""
        tac = TAC(get_test_config())
        # Must not raise.
        await tac.trigger_model_event("CA1", {"type": "turn.done"})

    @pytest.mark.asyncio
    async def test_handler_exception_is_swallowed_and_logged(self) -> None:
        """A handler that raises must not propagate; the failure is logged instead."""
        tac = TAC(get_test_config())

        def bad_handler(conversation_id: str, event: dict[str, Any]) -> None:
            raise ValueError("handler blew up")

        tac.on_model_event(bad_handler)

        with patch.object(tac.logger, "error") as mock_error:
            # Must not raise despite the handler raising.
            await tac.trigger_model_event("CA1", {"type": "turn.done"})

        mock_error.assert_called_once()
        assert "on_model_event callback raised an exception" in mock_error.call_args.args[0]


class FakeModelWebSocket:
    """Fake OpenAI model WebSocket. Async-iterates over queued raw events, then
    ends the stream (a closed connection)."""

    def __init__(self, events: list[dict]) -> None:
        self._events = list(events)
        self.sent: list[dict] = []
        self.closed = False

    def __aiter__(self) -> "FakeModelWebSocket":
        return self

    async def __anext__(self) -> str:
        if self._events:
            return json.dumps(self._events.pop(0))
        raise StopAsyncIteration

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self) -> None:
        self.closed = True


class FakeTwilioWebSocket:
    """Fake Twilio-facing WebSocket. Yields queued events, then hangs forever
    until the awaiting task is cancelled."""

    def __init__(self, events: list[dict]) -> None:
        self._events = list(events)
        self.accepted = False
        self.sent: list[dict] = []
        self.closed = False

    async def accept(self) -> None:
        self.accepted = True

    async def receive_json(self) -> dict:
        if self._events:
            return self._events.pop(0)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self) -> None:
        self.closed = True


def make_channel() -> VoiceChannel:
    tac = TAC(get_test_config())
    config = OpenAIRealtimeProviderConfig(
        openai_api_key="sk-test",
        default_session_config={
            "model": "gpt-realtime-test",
            "audio": {
                "input": {"format": TWILIO_MEDIA_STREAM_AUDIO_FORMAT},
                "output": {"format": TWILIO_MEDIA_STREAM_AUDIO_FORMAT},
            },
        },
    )
    return VoiceChannel(tac, config=config)


class TestModelEventWiring:
    """Every raw event read off the model WebSocket must reach on_model_event,
    regardless of whether _dispatch_model_event also interprets it."""

    @pytest.mark.asyncio
    async def test_model_events_forwarded_to_on_model_event(self) -> None:
        channel = make_channel()
        provider = channel._provider

        received: list[tuple[str, dict[str, Any]]] = []
        channel.tac.on_model_event(lambda conv_id, event: received.append((conv_id, event)))

        twilio_ws = FakeTwilioWebSocket(
            events=[
                {"event": "start", "start": {"callSid": "CA999", "streamSid": "MZ999"}},
                {"event": "stop"},
            ]
        )
        model_ws = FakeModelWebSocket(
            events=[
                {"type": "session.created"},
                {"type": "response.audio_transcript.delta", "delta": "hi"},
            ]
        )

        with patch(
            "tac.channels.voice.media_streams.openai_realtime.provider.websockets.connect",
            new=AsyncMock(return_value=model_ws),
        ):
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        assert received == [
            ("CA999", {"type": "session.created"}),
            ("CA999", {"type": "response.audio_transcript.delta", "delta": "hi"}),
        ]

    @pytest.mark.asyncio
    async def test_callback_mutation_does_not_affect_dispatch(self) -> None:
        """A callback that mutates its event (including nested payloads) must not
        alter what _dispatch_model_event subsequently sees for the same event."""
        channel = make_channel()
        provider = channel._provider

        def mutating_handler(conv_id: str, event: dict[str, Any]) -> None:
            if event.get("type") == "response.done":
                event["response"]["output"] = []

        channel.tac.on_model_event(mutating_handler)

        twilio_ws = FakeTwilioWebSocket(
            events=[
                {"event": "start", "start": {"callSid": "CA777", "streamSid": "MZ777"}},
                {"event": "stop"},
            ]
        )
        model_ws = FakeModelWebSocket(
            events=[
                {
                    "type": "response.done",
                    "response": {
                        "output": [
                            {
                                "role": "assistant",
                                "content": [{"transcript": "hello there"}],
                            }
                        ]
                    },
                },
            ]
        )

        captured_transcripts: list[list[dict[str, Any]]] = []
        original_dispatch = provider._dispatch_model_event

        async def spying_dispatch(conv_id: str, session: Any, event: dict[str, Any]) -> None:
            await original_dispatch(conv_id, session, event)
            captured_transcripts.append(list(session.metadata.get("transcript", [])))

        with (
            patch(
                "tac.channels.voice.media_streams.openai_realtime.provider.websockets.connect",
                new=AsyncMock(return_value=model_ws),
            ),
            patch.object(provider, "_dispatch_model_event", side_effect=spying_dispatch),
        ):
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        assert captured_transcripts == [[{"role": "assistant", "text": "hello there"}]]

    @pytest.mark.asyncio
    async def test_no_handler_registered_does_not_break_dispatch(self) -> None:
        """Without on_model_event registered, model events are still dispatched
        normally (trigger_model_event is a no-op, not an error)."""
        channel = make_channel()
        provider = channel._provider

        twilio_ws = FakeTwilioWebSocket(
            events=[
                {"event": "start", "start": {"callSid": "CA111", "streamSid": "MZ111"}},
                {"event": "stop"},
            ]
        )
        model_ws = FakeModelWebSocket(events=[{"type": "session.created"}])

        with patch(
            "tac.channels.voice.media_streams.openai_realtime.provider.websockets.connect",
            new=AsyncMock(return_value=model_ws),
        ):
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        assert provider._calls == {}

"""Tests for RealtimeVoiceChannel (Media Streams + OpenAI Realtime bridge)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from tac import TAC
from tac.channels.realtime import (
    RealtimeVoiceChannel,
    RealtimeVoiceChannelConfig,
    generate_stream_twiml,
)
from tac.channels.websocket_protocol import WebSocketDisconnectError
from tac.tools import function_tool


def get_test_config() -> dict:
    """Valid TAC config for tests."""
    return {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": "conv_configuration_test123",
        "phone_number": "+15551234567",
        "voice_public_domain": "example.com",
    }


def realtime_config() -> RealtimeVoiceChannelConfig:
    return RealtimeVoiceChannelConfig(openai_api_key="sk-test-123")


class FakeModelWS:
    """Stand-in for the OpenAI Realtime ClientConnection.

    Records what was sent, and yields a scripted list of inbound events when
    iterated (async), then stops — mimicking the server closing the socket.
    """

    def __init__(self, inbound: list[dict] | None = None) -> None:
        self.sent: list[dict] = []
        self._inbound = inbound or []
        self.closed = False

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> FakeModelWS:
        self._iter = iter(self._inbound)
        return self

    async def __anext__(self) -> str:
        try:
            return json.dumps(next(self._iter))
        except StopIteration:
            raise StopAsyncIteration from None


# --- TwiML -----------------------------------------------------------------


class TestStreamTwiML:
    def test_emits_connect_stream(self) -> None:
        xml = generate_stream_twiml("wss://example.com/voice-realtime")
        assert "<Connect>" in xml
        assert "<Stream" in xml
        assert 'url="wss://example.com/voice-realtime"' in xml

    def test_custom_parameters(self) -> None:
        xml = generate_stream_twiml(
            "wss://example.com/voice-realtime",
            custom_parameters={"profileId": "p_123"},
        )
        assert 'name="profileId"' in xml
        assert 'value="p_123"' in xml

    def test_empty_url_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty websocket_url"):
            generate_stream_twiml("   ")


# --- Channel basics --------------------------------------------------------


class TestRealtimeVoiceChannelBasics:
    def test_channel_name_is_voice(self) -> None:
        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        assert channel.get_channel_name() == "voice"

    def test_accepts_dict_config(self) -> None:
        channel = RealtimeVoiceChannel(
            TAC(get_test_config()), {"openai_api_key": "sk-x", "voice": "verse"}
        )
        assert channel.config.voice == "verse"

    def test_build_stream_twiml(self) -> None:
        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        xml = channel.build_stream_twiml("wss://example.com/voice-realtime")
        assert "<Stream" in xml
        assert 'url="wss://example.com/voice-realtime"' in xml

    @pytest.mark.asyncio
    async def test_send_response_not_supported(self) -> None:
        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        with pytest.raises(NotImplementedError):
            await channel.send_response("conv", "hi")

    @pytest.mark.asyncio
    async def test_process_webhook_is_noop(self) -> None:
        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        assert await channel.process_webhook({"eventType": "X"}) is None


# --- WebSocket bridge ------------------------------------------------------


class TestRealtimeBridge:
    @pytest.mark.asyncio
    async def test_start_connects_model_and_configures_session(self) -> None:
        """On 'start', the channel connects to OpenAI and sends session.update
        with the telephony audio format, and registers the conversation."""
        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        fake_model = FakeModelWS()

        twilio_ws = AsyncMock()
        twilio_ws.receive_json = AsyncMock(
            side_effect=[
                {"event": "connected"},
                {"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA_call_1"}},
                {"event": "stop"},
            ]
        )

        with patch(
            "tac.channels.realtime.channel._ws_connect",
            AsyncMock(return_value=fake_model),
        ) as mock_connect:
            await channel.handle_websocket(twilio_ws)

        # Connected to the Realtime endpoint with auth headers.
        assert mock_connect.await_count == 1
        url_arg = mock_connect.await_args[0][0]
        assert url_arg.startswith("wss://api.openai.com/v1/realtime?model=")
        headers = mock_connect.await_args[1]["additional_headers"]
        assert headers["Authorization"] == "Bearer sk-test-123"

        # First message to the model is a GA-shape session.update: nested audio
        # config with u-law (audio/pcmu) format and output_modalities.
        assert fake_model.sent, "expected a session.update to be sent"
        session_update = fake_model.sent[0]
        assert session_update["type"] == "session.update"
        session_obj = session_update["session"]
        assert session_obj["type"] == "realtime"
        assert session_obj["output_modalities"] == ["audio"]
        assert session_obj["audio"]["input"]["format"] == {"type": "audio/pcmu"}
        assert session_obj["audio"]["output"]["format"] == {"type": "audio/pcmu"}
        assert session_obj["audio"]["input"]["turn_detection"] == {
            "type": "semantic_vad",
            "eagerness": "low",
        }
        # No beta header — that would trigger beta_api_shape_disabled on gpt-realtime.
        assert "OpenAI-Beta" not in headers

        # Conversation was started and then cleaned up on stop/disconnect.
        assert "CA_call_1" not in channel._conversations

    @pytest.mark.asyncio
    async def test_welcome_greeting_triggers_response_create(self) -> None:
        """With a welcome greeting set (the default), a response.create follows
        session.update so the model greets the caller first."""
        channel = RealtimeVoiceChannel(
            TAC(get_test_config()),
            RealtimeVoiceChannelConfig(openai_api_key="sk-test-123", welcome_greeting="Hi there!"),
        )
        fake_model = FakeModelWS()
        twilio_ws = AsyncMock()
        twilio_ws.receive_json = AsyncMock(
            side_effect=[
                {"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA_g"}},
                {"event": "stop"},
            ]
        )

        with patch(
            "tac.channels.realtime.channel._ws_connect",
            AsyncMock(return_value=fake_model),
        ):
            await channel.handle_websocket(twilio_ws)

        types = [m["type"] for m in fake_model.sent]
        assert types[0] == "session.update"
        assert "response.create" in types
        greeting = next(m for m in fake_model.sent if m["type"] == "response.create")
        assert "Hi there!" in greeting["response"]["instructions"]

    @pytest.mark.asyncio
    async def test_no_greeting_when_disabled(self) -> None:
        """With welcome_greeting=None the model waits for the caller (no response.create)."""
        channel = RealtimeVoiceChannel(
            TAC(get_test_config()),
            RealtimeVoiceChannelConfig(openai_api_key="sk-test-123", welcome_greeting=None),
        )
        fake_model = FakeModelWS()
        twilio_ws = AsyncMock()
        twilio_ws.receive_json = AsyncMock(
            side_effect=[
                {"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA_ng"}},
                {"event": "stop"},
            ]
        )

        with patch(
            "tac.channels.realtime.channel._ws_connect",
            AsyncMock(return_value=fake_model),
        ):
            await channel.handle_websocket(twilio_ws)

        assert [m["type"] for m in fake_model.sent] == ["session.update"]

    @pytest.mark.asyncio
    async def test_media_forwarded_to_model_as_audio_append(self) -> None:
        """Twilio 'media' frames are forwarded to the model as input_audio_buffer.append."""
        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        fake_model = FakeModelWS()

        twilio_ws = AsyncMock()
        twilio_ws.receive_json = AsyncMock(
            side_effect=[
                {"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA_call_2"}},
                {"event": "media", "media": {"timestamp": 20, "payload": "BASE64AUDIO"}},
                WebSocketDisconnectError(),
            ]
        )

        with patch(
            "tac.channels.realtime.channel._ws_connect",
            AsyncMock(return_value=fake_model),
        ):
            await channel.handle_websocket(twilio_ws)

        appends = [m for m in fake_model.sent if m.get("type") == "input_audio_buffer.append"]
        assert len(appends) == 1
        assert appends[0]["audio"] == "BASE64AUDIO"

    @pytest.mark.asyncio
    async def test_model_audio_delta_forwarded_to_twilio(self) -> None:
        """OpenAI 'response.output_audio.delta' events are forwarded to Twilio as media.

        Driven through the real reader loop with a scripted model socket.
        """
        from tac.channels.realtime.channel import _RealtimeCallSession

        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        twilio_ws = AsyncMock()
        session = _RealtimeCallSession(twilio_ws)
        session.stream_sid = "MZ9"
        session.model_ws = FakeModelWS(
            inbound=[
                {"type": "response.output_audio.delta", "delta": "AUDIO_OUT", "item_id": "item_1"},
            ]
        )  # type: ignore[assignment]

        await channel._pump_model_to_twilio(session)

        sent_to_twilio = [json.loads(c[0][0]) for c in twilio_ws.send_text.call_args_list]
        media_events = [e for e in sent_to_twilio if e.get("event") == "media"]
        assert media_events
        assert media_events[0]["streamSid"] == "MZ9"
        assert media_events[0]["media"]["payload"] == "AUDIO_OUT"
        # last_assistant_item is tracked for barge-in truncation.
        assert session.last_assistant_item == "item_1"

    @pytest.mark.asyncio
    async def test_barge_in_truncates_and_clears(self) -> None:
        """speech_started while an assistant reply is playing truncates the model
        item and clears Twilio's buffer."""
        from tac.channels.realtime.channel import _RealtimeCallSession

        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        twilio_ws = AsyncMock()
        session = _RealtimeCallSession(twilio_ws)
        session.stream_sid = "MZ9"
        # Simulate a reply already mid-playback: item known, started at 500ms,
        # caller now at 800ms.
        session.last_assistant_item = "item_1"
        session.response_start_timestamp = 500
        session.response_active = True
        session.latest_media_timestamp = 800
        session.model_ws = FakeModelWS(inbound=[{"type": "input_audio_buffer.speech_started"}])  # type: ignore[assignment]

        await channel._pump_model_to_twilio(session)

        sent = session.model_ws.sent
        truncates = [m for m in sent if m.get("type") == "conversation.item.truncate"]
        assert truncates and truncates[0]["item_id"] == "item_1"
        assert truncates[0]["audio_end_ms"] == 300
        clears = [
            json.loads(c[0][0])
            for c in twilio_ws.send_text.call_args_list
            if json.loads(c[0][0]).get("event") == "clear"
        ]
        assert clears

    @pytest.mark.asyncio
    async def test_response_done_does_not_reset_barge_in_tracking(self) -> None:
        """response.done means the model finished *generating*, not that Twilio
        finished *playing* the audio already sent to it. If a barge-in landed in
        that gap and this state had been reset, the caller's interrupt would
        find "nothing to truncate/clear" and the stale audio already queued on
        Twilio's side would keep playing — so this must NOT reset here."""
        from tac.channels.realtime.channel import _RealtimeCallSession

        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        twilio_ws = AsyncMock()
        session = _RealtimeCallSession(twilio_ws)
        session.stream_sid = "MZ9"
        session.last_assistant_item = "item_1"
        session.response_start_timestamp = 500
        session.model_ws = FakeModelWS(inbound=[{"type": "response.done"}])  # type: ignore[assignment]

        await channel._pump_model_to_twilio(session)

        assert session.last_assistant_item == "item_1"
        assert session.response_start_timestamp == 500

    @pytest.mark.asyncio
    async def test_barge_in_during_playback_gap_after_response_done(self) -> None:
        """A barge-in that lands after response.done (while Twilio is still
        draining previously-sent audio) must still clear Twilio's leftover
        buffer — this is the scenario the old reset-on-response.done behavior
        silently dropped. But it must NOT send conversation.item.truncate: the
        model already finished generating this item, so there's nothing left
        to cut short, and OpenAI rejects a truncate past the item's actual
        audio length ("Audio content of Xms is already shorter than Yms")."""
        from tac.channels.realtime.channel import _RealtimeCallSession

        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        twilio_ws = AsyncMock()
        session = _RealtimeCallSession(twilio_ws)
        session.stream_sid = "MZ9"
        session.model_ws = FakeModelWS(
            inbound=[
                {"type": "response.done"},
                {"type": "input_audio_buffer.speech_started"},
            ]
        )  # type: ignore[assignment]
        # Simulate a response that already finished generating (item_1, started
        # at 500ms) with the caller's timeline having since moved to 900ms —
        # i.e. Twilio is still ~400ms into playing it back.
        session.last_assistant_item = "item_1"
        session.response_start_timestamp = 500
        session.response_active = True
        session.latest_media_timestamp = 900

        await channel._pump_model_to_twilio(session)

        # response.done (processed first) flips response_active off before the
        # interrupt arrives, so no truncate should be sent for this item.
        truncates = [
            m for m in session.model_ws.sent if m.get("type") == "conversation.item.truncate"
        ]
        assert not truncates
        clears = [
            json.loads(c[0][0])
            for c in twilio_ws.send_text.call_args_list
            if json.loads(c[0][0]).get("event") == "clear"
        ]
        assert clears

    @pytest.mark.asyncio
    async def test_new_item_reanchors_barge_in_timing(self) -> None:
        """The first delta of a genuinely new item resets the barge-in clock to
        its own start, so a later interrupt measures playback from *this*
        item's start rather than an unrelated earlier one."""
        from tac.channels.realtime.channel import _RealtimeCallSession

        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        twilio_ws = AsyncMock()
        session = _RealtimeCallSession(twilio_ws)
        session.stream_sid = "MZ9"
        session.last_assistant_item = "item_old"
        session.response_start_timestamp = 100
        session.latest_media_timestamp = 5000
        session.model_ws = FakeModelWS(
            inbound=[
                {"type": "response.output_audio.delta", "delta": "A", "item_id": "item_new"},
            ]
        )  # type: ignore[assignment]

        await channel._pump_model_to_twilio(session)

        assert session.last_assistant_item == "item_new"
        assert session.response_start_timestamp == 5000
        assert session.response_active is True

    @pytest.mark.asyncio
    async def test_stale_audio_after_barge_in_is_dropped(self) -> None:
        """After a barge-in truncates item_1, the model may still have more of
        item_1's (cancelled) audio queued up server-side. Those late deltas must
        not be forwarded to Twilio — otherwise the assistant keeps audibly
        talking after the caller already interrupted it. A delta for a genuinely
        new item (item_2) should still play normally."""
        from tac.channels.realtime.channel import _RealtimeCallSession

        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        twilio_ws = AsyncMock()
        session = _RealtimeCallSession(twilio_ws)
        session.stream_sid = "MZ9"
        session.last_assistant_item = "item_1"
        session.response_start_timestamp = 500
        session.response_active = True
        session.latest_media_timestamp = 800
        session.model_ws = FakeModelWS(
            inbound=[
                {"type": "input_audio_buffer.speech_started"},
                # Stale tail of the just-truncated response — must be dropped.
                {"type": "response.output_audio.delta", "delta": "STALE", "item_id": "item_1"},
                # A genuinely new response — must still play.
                {"type": "response.output_audio.delta", "delta": "FRESH", "item_id": "item_2"},
            ]
        )  # type: ignore[assignment]

        await channel._pump_model_to_twilio(session)

        media_payloads = [
            json.loads(c[0][0])["media"]["payload"]
            for c in twilio_ws.send_text.call_args_list
            if json.loads(c[0][0]).get("event") == "media"
        ]
        assert media_payloads == ["FRESH"]
        assert session.last_assistant_item == "item_2"

    @pytest.mark.asyncio
    async def test_user_transcript_captured(self) -> None:
        """conversation.item.input_audio_transcription.completed appends the
        caller's transcribed text to the session transcript."""
        from tac.channels.realtime.channel import _RealtimeCallSession

        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        twilio_ws = AsyncMock()
        session = _RealtimeCallSession(twilio_ws)
        session.model_ws = FakeModelWS(
            inbound=[
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "What's my account balance?",
                }
            ]
        )  # type: ignore[assignment]

        await channel._pump_model_to_twilio(session)

        assert session.transcript == [{"role": "user", "text": "What's my account balance?"}]

    @pytest.mark.asyncio
    async def test_assistant_transcript_captured_from_response_done(self) -> None:
        """response.done already carries the full per-turn assistant text, so
        it's captured without needing response.output_audio_transcript.delta."""
        from tac.channels.realtime.channel import _RealtimeCallSession

        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        twilio_ws = AsyncMock()
        session = _RealtimeCallSession(twilio_ws)
        session.model_ws = FakeModelWS(
            inbound=[
                {
                    "type": "response.done",
                    "response": {
                        "output": [
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_audio",
                                        "transcript": "Your balance is $42.",
                                    }
                                ],
                            }
                        ]
                    },
                }
            ]
        )  # type: ignore[assignment]

        await channel._pump_model_to_twilio(session)

        assert session.transcript == [{"role": "assistant", "text": "Your balance is $42."}]

    @pytest.mark.asyncio
    async def test_transcript_saved_to_conversation_metadata_on_end(self) -> None:
        """The collected transcript is stashed on the ConversationSession's
        metadata before the conversation ends, for on_conversation_ended to use."""
        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        fake_model = FakeModelWS(
            inbound=[
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "Hi there",
                },
                {
                    "type": "response.done",
                    "response": {
                        "output": [
                            {
                                "role": "assistant",
                                "content": [{"type": "output_audio", "transcript": "Hello!"}],
                            }
                        ]
                    },
                },
            ]
        )

        captured: dict = {}

        async def fake_end_conversation(conv_id: str) -> None:
            captured["metadata"] = channel._conversations[conv_id].metadata

        channel._end_conversation = fake_end_conversation  # type: ignore[method-assign]

        events = [
            {"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA_transcript"}},
            {"event": "stop"},
        ]
        call_count = 0

        async def fake_receive_json() -> dict:
            nonlocal call_count
            event = events[call_count]
            call_count += 1
            if event["event"] == "stop":
                # Give the model_reader background task a real event-loop
                # checkpoint to drain the scripted transcript events before
                # handle_websocket cancels it.
                await asyncio.sleep(0.01)
            return event

        twilio_ws = AsyncMock()
        twilio_ws.receive_json = fake_receive_json

        with patch(
            "tac.channels.realtime.channel._ws_connect",
            AsyncMock(return_value=fake_model),
        ):
            await channel.handle_websocket(twilio_ws)

        assert captured["metadata"]["transcript"] == [
            {"role": "user", "text": "Hi there"},
            {"role": "assistant", "text": "Hello!"},
        ]

    @pytest.mark.asyncio
    async def test_response_latency_span_opened_on_speech_stopped_closed_by_first_delta(
        self,
    ) -> None:
        """The span meant to capture "caller stopped talking -> model's first
        audio byte" opens on speech_stopped and is closed by the following
        turn's first output_audio delta."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from tac.channels.realtime.channel import _RealtimeCallSession
        from tac.core.tracing import setup_tracing

        setup_tracing()
        exporter = InMemorySpanExporter()
        trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))  # type: ignore[attr-defined]

        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        twilio_ws = AsyncMock()
        session = _RealtimeCallSession(twilio_ws)
        session.stream_sid = "MZ9"
        session.conv_id = "CA_latency"
        session.model_ws = FakeModelWS(
            inbound=[
                {"type": "input_audio_buffer.speech_stopped"},
                {"type": "response.output_audio.delta", "delta": "A", "item_id": "item_1"},
            ]
        )  # type: ignore[assignment]

        await channel._pump_model_to_twilio(session)

        assert session.pending_response_span is None
        spans = [
            s for s in exporter.get_finished_spans() if s.name == "openai_realtime.response_latency"
        ]
        assert len(spans) == 1
        assert spans[0].attributes is not None
        assert spans[0].attributes["conversation_id"] == "CA_latency"
        assert spans[0].end_time is not None and spans[0].start_time is not None
        assert spans[0].end_time >= spans[0].start_time

    @pytest.mark.asyncio
    async def test_orphaned_response_latency_span_closed_by_next_speech_started(self) -> None:
        """If the caller speaks again before the previous turn ever got a
        response (e.g. they gave up waiting), the stale span must be closed
        out rather than left open forever."""
        from tac.channels.realtime.channel import _RealtimeCallSession

        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        twilio_ws = AsyncMock()
        session = _RealtimeCallSession(twilio_ws)
        session.stream_sid = "MZ9"
        session.model_ws = FakeModelWS(
            inbound=[
                {"type": "input_audio_buffer.speech_stopped"},
                {"type": "input_audio_buffer.speech_started"},
            ]
        )  # type: ignore[assignment]

        await channel._pump_model_to_twilio(session)

        assert session.pending_response_span is None


# --- Tool calling ------------------------------------------------------------


@function_tool()
def get_current_time() -> str:
    """Get the current date and time."""
    return "Tuesday, July 14, 2026 at 08:00 PM"


@function_tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class TestRealtimeToolCalling:
    def test_session_update_includes_tools_when_configured(self) -> None:
        """session.update carries the flat Realtime tool shape when tools are set."""
        channel = RealtimeVoiceChannel(
            TAC(get_test_config()),
            RealtimeVoiceChannelConfig(openai_api_key="sk-test-123", tools=[get_current_time]),
        )
        assert channel._tools_by_name == {"get_current_time": get_current_time}

    @pytest.mark.asyncio
    async def test_start_sends_tools_in_session_update(self) -> None:
        channel = RealtimeVoiceChannel(
            TAC(get_test_config()),
            RealtimeVoiceChannelConfig(openai_api_key="sk-test-123", tools=[get_current_time]),
        )
        fake_model = FakeModelWS()
        twilio_ws = AsyncMock()
        twilio_ws.receive_json = AsyncMock(
            side_effect=[
                {"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA_tools"}},
                {"event": "stop"},
            ]
        )

        with patch(
            "tac.channels.realtime.channel._ws_connect",
            AsyncMock(return_value=fake_model),
        ):
            await channel.handle_websocket(twilio_ws)

        session_obj = fake_model.sent[0]["session"]
        assert session_obj["tools"] == [
            {
                "type": "function",
                "name": "get_current_time",
                "description": "Get the current date and time.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        ]
        assert session_obj["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_no_tools_key_when_none_configured(self) -> None:
        """Without tools configured, session.update omits tools/tool_choice entirely."""
        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        fake_model = FakeModelWS()
        twilio_ws = AsyncMock()
        twilio_ws.receive_json = AsyncMock(
            side_effect=[
                {"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA_no_tools"}},
                {"event": "stop"},
            ]
        )

        with patch(
            "tac.channels.realtime.channel._ws_connect",
            AsyncMock(return_value=fake_model),
        ):
            await channel.handle_websocket(twilio_ws)

        session_obj = fake_model.sent[0]["session"]
        assert "tools" not in session_obj
        assert "tool_choice" not in session_obj

    @pytest.mark.asyncio
    async def test_function_call_executes_tool_and_replies(self) -> None:
        """A function_call item in response.done runs the matching tool and
        sends its result back as function_call_output, then triggers response.create."""
        from tac.channels.realtime.channel import _RealtimeCallSession

        channel = RealtimeVoiceChannel(
            TAC(get_test_config()),
            RealtimeVoiceChannelConfig(openai_api_key="sk-test-123", tools=[add]),
        )
        twilio_ws = AsyncMock()
        session = _RealtimeCallSession(twilio_ws)
        session.model_ws = FakeModelWS(
            inbound=[
                {
                    "type": "response.done",
                    "response": {
                        "output": [
                            {
                                "type": "function_call",
                                "name": "add",
                                "call_id": "call_123",
                                "arguments": json.dumps({"a": 2, "b": 3}),
                            }
                        ]
                    },
                }
            ]
        )  # type: ignore[assignment]

        await channel._pump_model_to_twilio(session)

        sent = session.model_ws.sent
        outputs = [m for m in sent if m.get("type") == "conversation.item.create"]
        assert len(outputs) == 1
        assert outputs[0]["item"]["type"] == "function_call_output"
        assert outputs[0]["item"]["call_id"] == "call_123"
        assert json.loads(outputs[0]["item"]["output"]) == 5
        assert {"type": "response.create"} in sent

    @pytest.mark.asyncio
    async def test_unknown_tool_reports_error_without_crashing(self) -> None:
        from tac.channels.realtime.channel import _RealtimeCallSession

        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        twilio_ws = AsyncMock()
        session = _RealtimeCallSession(twilio_ws)
        session.model_ws = FakeModelWS(
            inbound=[
                {
                    "type": "response.done",
                    "response": {
                        "output": [
                            {
                                "type": "function_call",
                                "name": "does_not_exist",
                                "call_id": "call_456",
                                "arguments": "{}",
                            }
                        ]
                    },
                }
            ]
        )  # type: ignore[assignment]

        await channel._pump_model_to_twilio(session)

        outputs = [m for m in session.model_ws.sent if m.get("type") == "conversation.item.create"]
        assert len(outputs) == 1
        output = json.loads(outputs[0]["item"]["output"])
        assert "error" in output


# --- Tracing -----------------------------------------------------------------


class TestRealtimeTracing:
    """Every span from one call should nest under a single root span/trace —
    otherwise Langfuse (or any other OTel backend) shows them as unrelated
    traces instead of one call with connect/tool_call/response_latency as
    child steps."""

    @pytest.mark.asyncio
    async def test_full_call_spans_all_share_one_trace(self) -> None:
        from opentelemetry import trace
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from tac.core.tracing import setup_tracing

        setup_tracing()
        exporter = InMemorySpanExporter()
        trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))  # type: ignore[attr-defined]

        channel = RealtimeVoiceChannel(
            TAC(get_test_config()),
            RealtimeVoiceChannelConfig(openai_api_key="sk-test-123", tools=[add]),
        )
        fake_model = FakeModelWS(
            inbound=[
                {"type": "input_audio_buffer.speech_stopped"},
                {
                    "type": "response.done",
                    "response": {
                        "output": [
                            {
                                "type": "function_call",
                                "name": "add",
                                "call_id": "call_trace",
                                "arguments": json.dumps({"a": 1, "b": 2}),
                            }
                        ]
                    },
                },
                {"type": "response.output_audio.delta", "delta": "A", "item_id": "item_1"},
            ]
        )

        events = [
            {"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA_trace"}},
            {"event": "stop"},
        ]
        call_count = 0

        async def fake_receive_json() -> dict:
            nonlocal call_count
            event = events[call_count]
            call_count += 1
            if event["event"] == "stop":
                await asyncio.sleep(0.01)
            return event

        twilio_ws = AsyncMock()
        twilio_ws.receive_json = fake_receive_json

        with patch(
            "tac.channels.realtime.channel._ws_connect",
            AsyncMock(return_value=fake_model),
        ):
            await channel.handle_websocket(twilio_ws)

        spans = exporter.get_finished_spans()
        by_name = {s.name: s for s in spans}
        assert "twilio_media_stream.call" in by_name
        assert "openai_realtime.connect" in by_name
        assert "openai_realtime.tool_call" in by_name
        assert "openai_realtime.response_latency" in by_name

        root_trace_id = by_name["twilio_media_stream.call"].context.trace_id
        for name, span in by_name.items():
            assert span.context.trace_id == root_trace_id, f"{name} is not part of the call's trace"

        tool_span = by_name["openai_realtime.tool_call"]
        assert tool_span.attributes is not None
        assert tool_span.attributes["tool_name"] == "add"
        assert tool_span.parent is not None
        assert tool_span.parent.span_id == by_name["twilio_media_stream.call"].context.span_id

    @pytest.mark.asyncio
    async def test_barge_in_span_records_truncation(self) -> None:
        """A barge-in that actually interrupts a streaming reply produces a
        span recording that it truncated the reply and how much had played."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from tac.channels.realtime.channel import _RealtimeCallSession
        from tac.core.tracing import setup_tracing

        setup_tracing()
        exporter = InMemorySpanExporter()
        trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))  # type: ignore[attr-defined]

        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        twilio_ws = AsyncMock()
        session = _RealtimeCallSession(twilio_ws)
        session.stream_sid = "MZ9"
        session.conv_id = "CA_bargein"
        # Reply started at 500ms, caller interrupts at 800ms -> 300ms played.
        session.last_assistant_item = "item_1"
        session.response_start_timestamp = 500
        session.response_active = True
        session.latest_media_timestamp = 800
        session.model_ws = FakeModelWS(inbound=[{"type": "input_audio_buffer.speech_started"}])  # type: ignore[assignment]

        await channel._pump_model_to_twilio(session)

        spans = [s for s in exporter.get_finished_spans() if s.name == "openai_realtime.barge_in"]
        assert len(spans) == 1
        span = spans[0]
        assert span.attributes is not None
        assert span.attributes["conversation_id"] == "CA_bargein"
        assert span.attributes["truncated"] is True
        assert span.attributes["played_ms"] == 300

    @pytest.mark.asyncio
    async def test_no_barge_in_span_when_nothing_to_interrupt(self) -> None:
        """speech_started with no assistant item playing is normal turn-taking,
        not a barge-in — it shouldn't produce a barge_in span."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from tac.channels.realtime.channel import _RealtimeCallSession
        from tac.core.tracing import setup_tracing

        setup_tracing()
        exporter = InMemorySpanExporter()
        trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))  # type: ignore[attr-defined]

        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        twilio_ws = AsyncMock()
        session = _RealtimeCallSession(twilio_ws)
        session.stream_sid = "MZ9"
        session.model_ws = FakeModelWS(inbound=[{"type": "input_audio_buffer.speech_started"}])  # type: ignore[assignment]

        await channel._pump_model_to_twilio(session)

        spans = [s for s in exporter.get_finished_spans() if s.name == "openai_realtime.barge_in"]
        assert spans == []


class TestRealtimeVoiceServer:
    def test_registers_twiml_and_ws_routes(self) -> None:
        from tac.server import RealtimeVoiceServer

        tac = TAC(get_test_config())
        channel = RealtimeVoiceChannel(tac, realtime_config())
        server = RealtimeVoiceServer(
            tac, channel, twiml_path="/twiml-realtime", websocket_path="/voice-realtime"
        )

        paths = {getattr(r, "path", None) for r in server.app.routes}
        assert "/twiml-realtime" in paths
        assert "/voice-realtime" in paths

    def test_websocket_url_from_public_domain(self) -> None:
        from tac.server import RealtimeVoiceServer

        tac = TAC(get_test_config())
        channel = RealtimeVoiceChannel(tac, realtime_config())
        server = RealtimeVoiceServer(tac, channel, websocket_path="/voice-realtime")
        assert server.websocket_url == "wss://example.com/voice-realtime"

    def test_requires_public_domain(self) -> None:
        from tac.server import RealtimeVoiceServer

        config = get_test_config()
        del config["voice_public_domain"]
        tac = TAC(config)
        channel = RealtimeVoiceChannel(tac, realtime_config())
        with pytest.raises(ValueError, match="voice_public_domain"):
            RealtimeVoiceServer(tac, channel)

    def test_twiml_endpoint_returns_stream_xml(self) -> None:
        from fastapi.testclient import TestClient

        from tac.server import RealtimeVoiceServer

        tac = TAC(get_test_config())
        channel = RealtimeVoiceChannel(tac, realtime_config())
        # Disable signature validation so the test can POST without a Twilio signature.
        server = RealtimeVoiceServer(tac, channel, validate_twiml_signature=False)

        client = TestClient(server.app)
        resp = client.post("/twiml-realtime")
        assert resp.status_code == 200
        assert "<Stream" in resp.text
        assert 'url="wss://example.com/voice-realtime"' in resp.text

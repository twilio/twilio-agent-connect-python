"""Tests for RealtimeVoiceChannel (Media Streams + OpenAI Realtime bridge)."""

from __future__ import annotations

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
        assert session_obj["audio"]["input"]["turn_detection"] == {"type": "server_vad"}
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
    async def test_response_done_resets_barge_in_tracking(self) -> None:
        """A reply finishing normally clears barge-in state so the next turn
        doesn't measure playback from a stale start (avoids truncate overruns)."""
        from tac.channels.realtime.channel import _RealtimeCallSession

        channel = RealtimeVoiceChannel(TAC(get_test_config()), realtime_config())
        twilio_ws = AsyncMock()
        session = _RealtimeCallSession(twilio_ws)
        session.stream_sid = "MZ9"
        session.last_assistant_item = "item_1"
        session.response_start_timestamp = 500
        session.model_ws = FakeModelWS(inbound=[{"type": "response.done"}])  # type: ignore[assignment]

        await channel._pump_model_to_twilio(session)

        assert session.last_assistant_item is None
        assert session.response_start_timestamp is None


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

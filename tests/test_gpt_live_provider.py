"""Tests for GPTLiveProvider: connection lifecycle, tool calls, transcript."""

import asyncio
import json
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tac import TAC
from tac.channels.voice import VoiceChannel
from tac.channels.voice.media_streams.gpt_live import (
    GPT_LIVE_SESSION_ID_METADATA_KEY,
    TWILIO_AUDIO_FORMAT_FOR_GPT_LIVE,
    GPTLiveProviderConfig,
)
from tac.channels.voice.media_streams.gpt_live.models import _CallState
from tac.channels.voice.media_streams.gpt_live.provider import _SESSION_CONFIG_TOKEN_PARAM
from tac.models.outbound import (
    InitiateVoiceConversationOptions,
    InitiateVoiceConversationOptionsGPTLive,
)
from tac.models.voice import TwiMLRequest
from tac.tools import function_tool

_VALID_SESSION_CONFIG = {
    "model": "gpt-live-1",
    "audio": {"format": TWILIO_AUDIO_FORMAT_FOR_GPT_LIVE},
}


def _extract_custom_parameter(twiml: str, name: str) -> str:
    """Pull a <Parameter name="..." value="..."> value out of generated TwiML."""
    match = re.search(rf'<Parameter name="{re.escape(name)}" value="([^"]*)"', twiml)
    assert match, f"parameter {name!r} not found in TwiML: {twiml}"
    return match.group(1)


def get_test_tac_config() -> dict:
    return {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": "conv_configuration_test123",
        "phone_number": "+15551234567",
        "voice_public_domain": "example.com",
    }


def make_channel(**config_kwargs: object) -> VoiceChannel:
    tac = TAC(get_test_tac_config())
    config = GPTLiveProviderConfig(
        openai_api_key="sk-test",
        default_session_config=dict(_VALID_SESSION_CONFIG),
        **config_kwargs,
    )
    return VoiceChannel(tac, config=config)


class FakeTwilioWebSocket:
    """Fake Twilio-facing WebSocket. Yields queued events, then hangs
    forever (like a real, still-open connection with nothing new to say)
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


class FakeModelWebSocket:
    """Fake GPT-Live WebSocket. Async-iterates over queued raw events.

    When ``events`` runs out: if ``stay_open`` is True, hangs forever (a
    live connection with nothing new to say). Otherwise ends the stream
    immediately (a closed connection).
    """

    def __init__(self, events: list[dict] | None = None, stay_open: bool = False) -> None:
        self._events = list(events or [])
        self._stay_open = stay_open
        self._ended = False
        self.sent: list[dict] = []
        self.closed = False

    def __aiter__(self) -> "FakeModelWebSocket":
        return self

    async def __anext__(self) -> str:
        if self._events:
            return json.dumps(self._events.pop(0))
        if self._stay_open:
            await asyncio.Event().wait()
        self._ended = True
        raise StopAsyncIteration

    async def send(self, data: str) -> None:
        if self._ended:
            raise RuntimeError("connection closed")
        self.sent.append(json.loads(data))

    async def close(self) -> None:
        self.closed = True


class TestConfigValidation:
    def test_requires_openai_api_key(self) -> None:
        with pytest.raises(ValueError, match="openai_api_key is required"):
            GPTLiveProviderConfig(
                openai_api_key=None, default_session_config=dict(_VALID_SESSION_CONFIG)
            )

    def test_neither_source_alone_is_also_valid(self) -> None:
        """Legitimate for an outbound-only provider supplying session_config
        per-call via InitiateVoiceConversationOptionsGPTLive."""
        GPTLiveProviderConfig(openai_api_key="sk-test")

    def test_on_inbound_call_session_config_alone_is_valid(self) -> None:
        async def customizer(req: TwiMLRequest) -> dict | None:
            return dict(_VALID_SESSION_CONFIG)

        GPTLiveProviderConfig(openai_api_key="sk-test", on_inbound_call_session_config=customizer)

    def test_default_session_config_carries_model(self) -> None:
        config = GPTLiveProviderConfig(
            openai_api_key="sk-test", default_session_config=dict(_VALID_SESSION_CONFIG)
        )
        assert config.default_session_config["model"] == "gpt-live-1"


class TestOutboundCallSessionConfig:
    """Per-outbound-call session_config via InitiateVoiceConversationOptionsGPTLive."""

    @pytest.mark.asyncio
    async def test_session_config_stashed_under_a_token_before_call_is_placed(self) -> None:
        """Stashed under a token embedded in the TwiML, not call.sid — Twilio
        connecting the stream doesn't happen-after calls.create() returning
        call.sid, so keying by call.sid can race _register_call."""
        channel = make_channel()
        provider = channel._provider

        mock_call = MagicMock(sid="CA_OUT")
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call

        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            result = await provider.initiate_outbound_conversation(
                InitiateVoiceConversationOptionsGPTLive(
                    to="+15551234567",
                    session_config=dict(_VALID_SESSION_CONFIG),
                )
            )

        assert result.call_sid == "CA_OUT"
        assert "CA_OUT" not in provider._call_session_configs
        twiml = mock_client.calls.create.call_args.kwargs["twiml"]
        token = _extract_custom_parameter(twiml, _SESSION_CONFIG_TOKEN_PARAM)
        assert provider._call_session_configs[token] == _VALID_SESSION_CONFIG

    @pytest.mark.asyncio
    async def test_session_config_token_rekeyed_to_call_sid_on_connect(self) -> None:
        channel = make_channel()
        provider = channel._provider

        mock_call = MagicMock(sid="CA_OUT4")
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call

        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            await provider.initiate_outbound_conversation(
                InitiateVoiceConversationOptionsGPTLive(
                    to="+15551234567",
                    session_config={
                        "model": "gpt-live-1",
                        "audio": {"format": TWILIO_AUDIO_FORMAT_FOR_GPT_LIVE},
                        "instructions": "outbound override",
                    },
                )
            )

        twiml = mock_client.calls.create.call_args.kwargs["twiml"]
        token = _extract_custom_parameter(twiml, _SESSION_CONFIG_TOKEN_PARAM)

        twilio_ws = FakeTwilioWebSocket(
            events=[
                {
                    "event": "start",
                    "start": {
                        "callSid": "CA_OUT4",
                        "streamSid": "MZ_OUT4",
                        "customParameters": {_SESSION_CONFIG_TOKEN_PARAM: token},
                    },
                }
            ]
        )
        model_ws = FakeModelWebSocket(events=[], stay_open=False)

        with patch(
            "tac.channels.voice.media_streams.gpt_live.provider.websockets.connect",
            new=AsyncMock(return_value=model_ws),
        ):
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        sent_session = next(m for m in model_ws.sent if m["type"] == "session.start")
        assert sent_session["session"]["instructions"] == "outbound override"
        assert provider._call_session_configs == {}

    @pytest.mark.asyncio
    async def test_session_config_token_cleaned_up_if_call_creation_fails(self) -> None:
        channel = make_channel()
        provider = channel._provider

        mock_client = MagicMock()
        mock_client.calls.create.side_effect = RuntimeError("boom")

        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            with pytest.raises(RuntimeError):
                await provider.initiate_outbound_conversation(
                    InitiateVoiceConversationOptionsGPTLive(
                        to="+15551234567",
                        session_config=dict(_VALID_SESSION_CONFIG),
                    )
                )

        assert provider._call_session_configs == {}

    @pytest.mark.asyncio
    async def test_plain_options_type_ignored_falls_back_to_default(self) -> None:
        channel = make_channel()
        provider = channel._provider

        mock_call = MagicMock(sid="CA_OUT2")
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call

        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            await provider.initiate_outbound_conversation(
                InitiateVoiceConversationOptions(to="+15551234567")
            )

        assert provider._call_session_configs == {}

    @pytest.mark.asyncio
    async def test_outbound_only_config_with_no_default_connects_via_per_call_override(
        self,
    ) -> None:
        tac = TAC(get_test_tac_config())
        config = GPTLiveProviderConfig(openai_api_key="sk-test")
        channel = VoiceChannel(tac, config=config)
        provider = channel._provider

        mock_call = MagicMock(sid="CA_OUT3")
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call

        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            await provider.initiate_outbound_conversation(
                InitiateVoiceConversationOptionsGPTLive(
                    to="+15551234567",
                    session_config=dict(_VALID_SESSION_CONFIG),
                )
            )

        twiml = mock_client.calls.create.call_args.kwargs["twiml"]
        token = _extract_custom_parameter(twiml, _SESSION_CONFIG_TOKEN_PARAM)

        twilio_ws = FakeTwilioWebSocket(
            events=[
                {
                    "event": "start",
                    "start": {
                        "callSid": "CA_OUT3",
                        "streamSid": "MZ_OUT3",
                        "customParameters": {_SESSION_CONFIG_TOKEN_PARAM: token},
                    },
                }
            ]
        )
        model_ws = FakeModelWebSocket(events=[], stay_open=False)

        with patch(
            "tac.channels.voice.media_streams.gpt_live.provider.websockets.connect",
            new=AsyncMock(return_value=model_ws),
        ) as mock_connect:
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        assert mock_connect.call_args.args[0] == "wss://api.openai.com/v1/live/sessions"

    @pytest.mark.asyncio
    async def test_session_config_token_not_leaked_if_twiml_build_fails(self) -> None:
        """A failure building the TwiML itself (e.g. no resolvable WebSocket
        URL) happens before calls.create() is ever attempted — the token
        must not be stashed until that succeeds, or it's never cleaned up."""
        tac = TAC(get_test_tac_config())
        config = GPTLiveProviderConfig(
            openai_api_key="sk-test", default_session_config=dict(_VALID_SESSION_CONFIG)
        )
        channel = VoiceChannel(tac, config=config)
        provider = channel._provider
        channel.tac.config.voice_public_domain = None  # no fallback WebSocket URL

        with pytest.raises(ValueError, match="needs a WebSocket URL"):
            await provider.initiate_outbound_conversation(
                InitiateVoiceConversationOptionsGPTLive(
                    to="+15551234567",
                    session_config=dict(_VALID_SESSION_CONFIG),
                )
            )

        assert provider._call_session_configs == {}


class TestHandleWebSocketLifecycle:
    @pytest.mark.asyncio
    async def test_model_disconnect_ends_call_without_hanging(self) -> None:
        channel = make_channel()
        provider = channel._provider

        twilio_ws = FakeTwilioWebSocket(
            events=[{"event": "start", "start": {"callSid": "CA123", "streamSid": "MZ123"}}]
        )
        model_ws = FakeModelWebSocket(events=[])  # ends immediately

        with patch(
            "tac.channels.voice.media_streams.gpt_live.provider.websockets.connect",
            new=AsyncMock(return_value=model_ws),
        ):
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        assert provider._calls == {}
        assert model_ws.closed is True
        assert "CA123" not in channel._conversations

    @pytest.mark.asyncio
    async def test_input_audio_forwarded_with_gpt_live_event_shape(self) -> None:
        channel = make_channel()
        provider = channel._provider

        twilio_ws = FakeTwilioWebSocket(
            events=[
                {"event": "start", "start": {"callSid": "CA789", "streamSid": "MZ789"}},
                {"event": "media", "media": {"payload": "abcd"}},
                {"event": "stop"},
            ]
        )
        model_ws = FakeModelWebSocket(events=[{"type": "session.closed"}], stay_open=True)

        with patch(
            "tac.channels.voice.media_streams.gpt_live.provider.websockets.connect",
            new=AsyncMock(return_value=model_ws),
        ):
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        assert {"type": "session.input_audio.append", "audio": "abcd"} in model_ws.sent

    @pytest.mark.asyncio
    async def test_session_close_sent_on_teardown(self) -> None:
        channel = make_channel()
        provider = channel._provider

        twilio_ws = FakeTwilioWebSocket(
            events=[
                {"event": "start", "start": {"callSid": "CA_CLOSE", "streamSid": "MZ_CLOSE"}},
                {"event": "stop"},
            ]
        )
        model_ws = FakeModelWebSocket(events=[{"type": "session.closed"}], stay_open=True)

        with patch(
            "tac.channels.voice.media_streams.gpt_live.provider.websockets.connect",
            new=AsyncMock(return_value=model_ws),
        ):
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        assert {"type": "session.close"} in model_ws.sent


class TestConnectModelSessionConfig:
    @pytest.mark.asyncio
    async def test_missing_audio_format_raises_at_connect_time(self) -> None:
        channel = make_channel()
        provider = channel._provider
        provider._call_session_configs["CA_BAD"] = {"instructions": "no audio.format here"}

        with pytest.raises(ValueError, match="audio.format"):
            await provider._connect_model("CA_BAD")

    @pytest.mark.asyncio
    async def test_missing_model_raises_at_connect_time(self) -> None:
        channel = make_channel()
        provider = channel._provider
        provider._call_session_configs["CA_MODEL"] = {
            "audio": {"format": TWILIO_AUDIO_FORMAT_FOR_GPT_LIVE},
        }

        with pytest.raises(ValueError, match="must include 'model'"):
            await provider._connect_model("CA_MODEL")

    @pytest.mark.asyncio
    async def test_wss_connect_uses_configured_model(self) -> None:
        channel = make_channel()
        provider = channel._provider

        twilio_ws = FakeTwilioWebSocket(
            events=[{"event": "start", "start": {"callSid": "CA_URL", "streamSid": "MZ_URL"}}]
        )
        model_ws = FakeModelWebSocket(events=[], stay_open=False)

        with patch(
            "tac.channels.voice.media_streams.gpt_live.provider.websockets.connect",
            new=AsyncMock(return_value=model_ws),
        ) as mock_connect:
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        assert mock_connect.call_args.args[0] == "wss://api.openai.com/v1/live/sessions"
        headers = mock_connect.call_args.kwargs["additional_headers"]
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["User-Agent"].startswith("twilio-agent-connect/Python ")

        sent_session = next(m for m in model_ws.sent if m["type"] == "session.start")
        assert sent_session["session"]["model"] == "gpt-live-1"


class TestInboundCallSessionConfig:
    @pytest.mark.asyncio
    async def test_customizer_result_used_verbatim_for_that_call(self) -> None:
        async def customizer(req: TwiMLRequest) -> dict | None:
            if req.caller_country == "MX":
                return {
                    "model": "gpt-live-1",
                    "instructions": "Habla en español.",
                    "audio": {"format": TWILIO_AUDIO_FORMAT_FOR_GPT_LIVE},
                }
            return None

        channel = make_channel(on_inbound_call_session_config=customizer)
        provider = channel._provider

        await provider.handle_incoming_call(
            twiml_request=TwiMLRequest(call_sid="CA_MX", caller_country="MX")
        )
        assert provider._call_session_configs["CA_MX"]["instructions"] == "Habla en español."

        twilio_ws = FakeTwilioWebSocket(
            events=[{"event": "start", "start": {"callSid": "CA_MX", "streamSid": "MZ_MX"}}]
        )
        model_ws = FakeModelWebSocket(events=[], stay_open=False)

        with patch(
            "tac.channels.voice.media_streams.gpt_live.provider.websockets.connect",
            new=AsyncMock(return_value=model_ws),
        ):
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        sent_session = next(m for m in model_ws.sent if m["type"] == "session.start")
        assert sent_session["session"]["instructions"] == "Habla en español."
        assert "CA_MX" not in provider._call_session_configs


class TestTranscriptDeltas:
    @pytest.mark.asyncio
    async def test_input_transcript_delta_appends_user_turn(self) -> None:
        channel = make_channel()
        provider = channel._provider

        provider._calls["CA1"] = _CallState()
        session = channel._start_conversation("CA1", profile_id=None)

        await provider._dispatch_model_event(
            "CA1",
            session,
            {"type": "session.input_transcript.delta", "delta": "hi"},
        )

        assert session.metadata["transcript"] == [{"role": "user", "text": "hi"}]

    @pytest.mark.asyncio
    async def test_consecutive_same_role_deltas_merge_into_one_entry(self) -> None:
        channel = make_channel()
        provider = channel._provider

        provider._calls["CA1b"] = _CallState()
        session = channel._start_conversation("CA1b", profile_id=None)

        for chunk in ("he", "llo"):
            await provider._dispatch_model_event(
                "CA1b",
                session,
                {"type": "session.output_transcript.delta", "delta": chunk},
            )

        assert session.metadata["transcript"] == [{"role": "assistant", "text": "hello"}]


class TestGPTLiveSessionId:
    @pytest.mark.asyncio
    async def test_session_started_records_session_id_on_the_session(self) -> None:
        channel = make_channel()
        provider = channel._provider

        provider._calls["CA1c"] = _CallState()
        session = channel._start_conversation("CA1c", profile_id=None)

        await provider._dispatch_model_event(
            "CA1c",
            session,
            {"type": "session.started", "session": {"id": "rtc_123", "status": "active"}},
        )

        assert session.metadata[GPT_LIVE_SESSION_ID_METADATA_KEY] == "rtc_123"

    @pytest.mark.asyncio
    async def test_session_closed_records_session_id_when_started_was_missed(self) -> None:
        channel = make_channel()
        provider = channel._provider

        provider._calls["CA1d"] = _CallState()
        session = channel._start_conversation("CA1d", profile_id=None)

        await provider._dispatch_model_event(
            "CA1d",
            session,
            {"type": "session.closed", "reason": "client_request", "session": {"id": "live_456"}},
        )

        assert session.metadata[GPT_LIVE_SESSION_ID_METADATA_KEY] == "live_456"
        assert provider._calls["CA1d"].closed_event.is_set()

    @pytest.mark.asyncio
    async def test_snapshot_without_an_id_leaves_metadata_untouched(self) -> None:
        channel = make_channel()
        provider = channel._provider

        provider._calls["CA1e"] = _CallState()
        session = channel._start_conversation("CA1e", profile_id=None)

        await provider._dispatch_model_event("CA1e", session, {"type": "session.started"})

        assert GPT_LIVE_SESSION_ID_METADATA_KEY not in session.metadata


class TestOutputAudioDelta:
    @pytest.mark.asyncio
    async def test_output_audio_delta_relayed_to_twilio(self) -> None:
        channel = make_channel()
        provider = channel._provider

        twilio_ws = FakeTwilioWebSocket(events=[])
        provider._calls["CA2"] = _CallState(twilio_ws=twilio_ws)
        session = channel._start_conversation("CA2", profile_id=None)
        session.metadata["stream_sid"] = "MZ2"

        await provider._dispatch_model_event(
            "CA2", session, {"type": "session.output_audio.delta", "delta": "xyz"}
        )

        assert twilio_ws.sent == [
            {"event": "media", "streamSid": "MZ2", "media": {"payload": "xyz"}}
        ]


class TestToolCalls:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_without_raising(self) -> None:
        channel = make_channel()
        provider = channel._provider

        result = await provider._run_tool_call("CA6", "does_not_exist", "{}")

        assert result == {"error": "Unknown tool 'does_not_exist'"}

    @pytest.mark.asyncio
    async def test_function_call_sends_output_then_continues_response(self) -> None:
        @function_tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        channel = make_channel(tools=[add])
        provider = channel._provider

        model_ws = FakeModelWebSocket()
        provider._calls["CA3"] = _CallState(model_ws=model_ws)

        await provider._handle_function_call(
            "CA3",
            {
                "call_id": "call_1",
                "name": "add",
                "arguments": json.dumps({"a": 2, "b": 3}),
            },
        )

        assert len(model_ws.sent) == 2
        output_sent, continue_sent = model_ws.sent
        assert output_sent["type"] == "response.item.create"
        assert output_sent["item"]["call_id"] == "call_1"
        assert json.loads(output_sent["item"]["output"]) == 5
        assert continue_sent == {"type": "response.create"}

    @pytest.mark.asyncio
    async def test_non_serializable_output_still_sends_function_call_output(self) -> None:
        @function_tool()
        def broken_output() -> object:
            """Return a value json.dumps can't serialize."""
            return object()

        channel = make_channel(tools=[broken_output])
        provider = channel._provider

        model_ws = FakeModelWebSocket()
        provider._calls["CA4"] = _CallState(model_ws=model_ws)

        await provider._handle_function_call(
            "CA4", {"call_id": "call_1", "name": "broken_output", "arguments": "{}"}
        )

        assert len(model_ws.sent) == 2
        payload = json.loads(model_ws.sent[0]["item"]["output"])
        assert "error" in payload
        assert model_ws.sent[1] == {"type": "response.create"}

    @pytest.mark.asyncio
    async def test_missing_call_id_is_dropped_without_sending_anything(self) -> None:
        """No call_id means we can't tell GPT-Live which delegation this
        answers — sending a function_call_output with call_id=None would
        likely be rejected and leaves the real pending call still hanging."""

        @function_tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        channel = make_channel(tools=[add])
        provider = channel._provider

        model_ws = FakeModelWebSocket()
        provider._calls["CA5"] = _CallState(model_ws=model_ws)

        await provider._handle_function_call(
            "CA5", {"name": "add", "arguments": json.dumps({"a": 2, "b": 3})}
        )

        assert model_ws.sent == []

    @pytest.mark.asyncio
    async def test_missing_name_sends_structured_error_without_running_a_tool(self) -> None:
        ran = False

        @function_tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            nonlocal ran
            ran = True
            return a + b

        channel = make_channel(tools=[add])
        provider = channel._provider

        model_ws = FakeModelWebSocket()
        provider._calls["CA6"] = _CallState(model_ws=model_ws)

        await provider._handle_function_call(
            "CA6", {"call_id": "call_1", "arguments": json.dumps({"a": 2, "b": 3})}
        )

        assert ran is False
        assert len(model_ws.sent) == 2
        assert model_ws.sent[1] == {"type": "response.create"}
        sent = model_ws.sent[0]
        assert sent["item"]["call_id"] == "call_1"
        payload = json.loads(sent["item"]["output"])
        assert "error" in payload

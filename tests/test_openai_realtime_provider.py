"""Tests for OpenAIRealtimeProvider: connection lifecycle, barge-in, tool calls."""

import asyncio
import json
import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tac import TAC
from tac.channels.voice import VoiceChannel
from tac.channels.voice.media_streams.openai_realtime import OpenAIRealtimeProviderConfig
from tac.channels.voice.media_streams.openai_realtime.models import _BargeInState, _CallState
from tac.channels.voice.media_streams.openai_realtime.provider import (
    _SESSION_CONFIG_TOKEN_PARAM,
    TWILIO_MEDIA_STREAM_AUDIO_FORMAT,
)
from tac.models.outbound import (
    InitiateVoiceConversationOptions,
    InitiateVoiceConversationOptionsOpenAIRealtime,
)
from tac.models.voice import TwiMLRequest
from tac.tools import function_tool

_VALID_AUDIO = {
    "input": {"format": TWILIO_MEDIA_STREAM_AUDIO_FORMAT},
    "output": {"format": TWILIO_MEDIA_STREAM_AUDIO_FORMAT},
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
    config = OpenAIRealtimeProviderConfig(
        openai_api_key="sk-test",
        default_session_config={"model": "gpt-realtime-test", "audio": _VALID_AUDIO},
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
    """Fake OpenAI Realtime WebSocket. Async-iterates over queued raw
    events.

    When ``events`` runs out: if ``stay_open`` is True, hangs forever (a
    live connection with nothing new to say — required whenever a test also
    exercises the Twilio leg, since ending this instantly races the
    model_reader task against the Twilio recv task in handle_websocket's
    asyncio.wait, non-deterministically dropping already-queued Twilio
    events). Otherwise ends the stream immediately (a closed connection).
    """

    def __init__(self, events: list[dict] | None = None, stay_open: bool = False) -> None:
        self._events = list(events or [])
        self._stay_open = stay_open
        self.sent: list[dict] = []
        self.closed = False

    def __aiter__(self) -> "FakeModelWebSocket":
        return self

    async def __anext__(self) -> str:
        if self._events:
            return json.dumps(self._events.pop(0))
        if self._stay_open:
            await asyncio.Event().wait()
        raise StopAsyncIteration

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self) -> None:
        self.closed = True


class TestHandleWebSocketLifecycle:
    """The Twilio-read loop and the OpenAI model-event reader run as two
    tasks raced against each other — whichever leg disconnects first must
    tear the whole call down instead of leaving the other leg running."""

    @pytest.mark.asyncio
    async def test_model_disconnect_ends_call_without_hanging(self) -> None:
        """If the OpenAI leg closes first, the Twilio read loop must stop
        (rather than keep pumping caller audio into a dead model socket)
        and the call must be cleaned up."""
        channel = make_channel()
        provider = channel._provider

        twilio_ws = FakeTwilioWebSocket(
            events=[{"event": "start", "start": {"callSid": "CA123", "streamSid": "MZ123"}}]
        )
        model_ws = FakeModelWebSocket(events=[])  # ends immediately

        with patch(
            "tac.channels.voice.media_streams.openai_realtime.provider.websockets.connect",
            new=AsyncMock(return_value=model_ws),
        ):
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        # Call was cleaned up: no leftover state, model socket closed,
        # conversation ended.
        assert provider._calls == {}
        assert model_ws.closed is True
        assert "CA123" not in channel._conversations

    @pytest.mark.asyncio
    async def test_twilio_stop_event_ends_call(self) -> None:
        """The Twilio leg ending normally (a 'stop' event) must also clean
        up the still-running model reader, not just break out silently."""
        channel = make_channel()
        provider = channel._provider

        twilio_ws = FakeTwilioWebSocket(
            events=[
                {"event": "start", "start": {"callSid": "CA456", "streamSid": "MZ456"}},
                {"event": "stop"},
            ]
        )
        # Stays open so this test exercises "Twilio said stop" cleanup —
        # not "model disconnected first" (covered separately above).
        model_ws = FakeModelWebSocket(events=[], stay_open=True)

        with patch(
            "tac.channels.voice.media_streams.openai_realtime.provider.websockets.connect",
            new=AsyncMock(return_value=model_ws),
        ):
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        assert provider._calls == {}
        assert model_ws.closed is True
        assert "CA456" not in channel._conversations

    @pytest.mark.asyncio
    async def test_media_event_forwarded_to_model(self) -> None:
        """Caller audio arriving before the model disconnects is forwarded
        as an input_audio_buffer.append event."""
        channel = make_channel()
        provider = channel._provider

        twilio_ws = FakeTwilioWebSocket(
            events=[
                {"event": "start", "start": {"callSid": "CA789", "streamSid": "MZ789"}},
                {"event": "media", "media": {"payload": "abcd"}},
                {"event": "stop"},
            ]
        )
        # Stays open — must not race the "media" event's processing.
        model_ws = FakeModelWebSocket(events=[], stay_open=True)

        with patch(
            "tac.channels.voice.media_streams.openai_realtime.provider.websockets.connect",
            new=AsyncMock(return_value=model_ws),
        ):
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        forwarded = [m for m in model_ws.sent if m.get("type") == "input_audio_buffer.append"]
        assert forwarded == [{"type": "input_audio_buffer.append", "audio": "abcd"}]

    @pytest.mark.asyncio
    async def test_malformed_start_event_rejected_not_collapsed_to_shared_key(self) -> None:
        """A 'start' event missing callSid/streamSid must fail validation
        instead of falling back to a shared "unknown-call" id that a second
        malformed connection could collide with and clobber."""
        channel = make_channel()
        provider = channel._provider

        twilio_ws = FakeTwilioWebSocket(events=[{"event": "start", "start": {}}])
        model_ws = FakeModelWebSocket(events=[], stay_open=True)

        with patch(
            "tac.channels.voice.media_streams.openai_realtime.provider.websockets.connect",
            new=AsyncMock(return_value=model_ws),
        ):
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        # Rejected before a call was ever registered under any key.
        assert provider._calls == {}
        assert channel._conversations == {}


class TestBargeIn:
    """Barge-in truncates the model's memory of the interrupted reply and
    clears Twilio's playback buffer."""

    @pytest.mark.asyncio
    async def test_noop_when_no_assistant_audio_sent_yet(self) -> None:
        """If nothing has been sent to Twilio since the last barge-in,
        there's nothing to truncate or clear."""
        channel = make_channel()
        provider = channel._provider

        twilio_ws = FakeTwilioWebSocket(events=[])
        model_ws = FakeModelWebSocket()
        provider._calls["CA1"] = _CallState(twilio_ws=twilio_ws, model_ws=model_ws)
        session = channel._start_conversation("CA1", profile_id=None)

        call = provider._calls["CA1"]
        assert call.barge_in.last_assistant_item is None

        await provider._handle_barge_in("CA1", session, call)

        assert model_ws.sent == []
        assert twilio_ws.sent == []

    @pytest.mark.asyncio
    async def test_truncates_and_clears_when_interrupted_mid_reply(self) -> None:
        """A barge-in mid-reply sends response.cancel (if a response is
        still generating), truncates at the exact sent-audio duration, and
        clears Twilio's buffer."""
        channel = make_channel()
        provider = channel._provider

        twilio_ws = FakeTwilioWebSocket(events=[])
        model_ws = FakeModelWebSocket()
        provider._calls["CA2"] = _CallState(
            twilio_ws=twilio_ws,
            model_ws=model_ws,
            barge_in=_BargeInState(
                last_assistant_item="item_1", current_item_audio_ms=250, response_active=True
            ),
        )
        session = channel._start_conversation("CA2", profile_id=None)
        session.metadata["stream_sid"] = "MZ123"

        call = provider._calls["CA2"]
        await provider._handle_barge_in("CA2", session, call)

        assert {"type": "response.cancel"} in model_ws.sent
        truncate = next(m for m in model_ws.sent if m["type"] == "conversation.item.truncate")
        assert truncate["item_id"] == "item_1"
        assert truncate["audio_end_ms"] == 250
        assert {"event": "clear", "streamSid": "MZ123"} in twilio_ws.sent

        # State reset for the next reply.
        assert call.barge_in.last_assistant_item is None
        assert call.barge_in.current_item_audio_ms == 0
        assert call.barge_in.muted_item_id == "item_1"
        assert call.barge_in.response_active is False

    @pytest.mark.asyncio
    async def test_skips_response_cancel_when_no_response_in_flight(self) -> None:
        """response.cancel is only sent while a response is actually
        generating — sending it otherwise is itself an OpenAI error event."""
        channel = make_channel()
        provider = channel._provider

        twilio_ws = FakeTwilioWebSocket(events=[])
        model_ws = FakeModelWebSocket()
        provider._calls["CA3"] = _CallState(
            twilio_ws=twilio_ws,
            model_ws=model_ws,
            barge_in=_BargeInState(
                last_assistant_item="item_2", current_item_audio_ms=100, response_active=False
            ),
        )
        session = channel._start_conversation("CA3", profile_id=None)

        call = provider._calls["CA3"]
        await provider._handle_barge_in("CA3", session, call)

        assert {"type": "response.cancel"} not in model_ws.sent


class TestErrorEventDowngrade:
    """A response.cancel that raced a response.done is benign, not a
    genuine error worth alarming on."""

    @pytest.mark.asyncio
    async def test_response_cancel_not_active_is_not_logged_as_error(self) -> None:
        channel = make_channel()
        provider = channel._provider

        provider._calls["CA4"] = _CallState()
        session = channel._start_conversation("CA4", profile_id=None)

        with (
            patch.object(provider.logger, "error") as mock_error,
            patch.object(provider.logger, "debug") as mock_debug,
        ):
            await provider._dispatch_model_event(
                "CA4",
                session,
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "response_cancel_not_active",
                        "message": "Cancellation failed: no active response found",
                    },
                },
            )

        mock_error.assert_not_called()
        assert any(
            "response.cancel raced response.done" in c.args[0] for c in mock_debug.call_args_list
        )

    @pytest.mark.asyncio
    async def test_other_error_codes_still_logged_as_error(self) -> None:
        channel = make_channel()
        provider = channel._provider

        provider._calls["CA5"] = _CallState()
        session = channel._start_conversation("CA5", profile_id=None)

        with patch.object(provider.logger, "error") as mock_error:
            await provider._dispatch_model_event(
                "CA5",
                session,
                {"type": "error", "error": {"code": "some_other_error", "message": "boom"}},
            )

        mock_error.assert_called_once()


class TestToolCalls:
    """Tool execution errors must be reported back to the model, not raised
    and left to crash the call."""

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_without_raising(self) -> None:
        channel = make_channel()
        provider = channel._provider

        result = await provider._run_tool_call("CA6", "does_not_exist", "{}")

        assert result == {"error": "Unknown tool 'does_not_exist'"}

    @pytest.mark.asyncio
    async def test_tool_exception_returns_error_without_raising(self) -> None:
        @function_tool()
        def broken_tool() -> str:
            """A tool that always fails."""
            raise RuntimeError("boom")

        channel = make_channel(tools=[broken_tool])
        provider = channel._provider

        result = await provider._run_tool_call("CA7", "broken_tool", "{}")

        assert result == {"error": "Tool 'broken_tool' failed to execute."}

    @pytest.mark.asyncio
    async def test_successful_tool_call_returns_output(self) -> None:
        @function_tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        channel = make_channel(tools=[add])
        provider = channel._provider

        result = await provider._run_tool_call("CA8", "add", json.dumps({"a": 2, "b": 3}))

        assert result == 5

    @pytest.mark.asyncio
    async def test_non_serializable_output_still_sends_function_call_output(self) -> None:
        """A tool that runs successfully but returns something json.dumps
        can't handle must still produce a function_call_output — otherwise
        the model is left waiting on a call_id it never gets a result for."""

        @function_tool()
        def broken_output() -> object:
            """Return a value json.dumps can't serialize."""
            return object()

        channel = make_channel(tools=[broken_output])
        provider = channel._provider

        model_ws = FakeModelWebSocket()
        provider._calls["CA9"] = _CallState(model_ws=model_ws)

        await provider._handle_function_call(
            "CA9", {"call_id": "call_1", "name": "broken_output", "arguments": "{}"}
        )

        outputs = [m for m in model_ws.sent if m["type"] == "conversation.item.create"]
        assert len(outputs) == 1
        payload = json.loads(outputs[0]["item"]["output"])
        assert "error" in payload
        assert {"type": "response.create"} in model_ws.sent


class TestConfigValidation:
    """OpenAIRealtimeProviderConfig fails fast on unusable combinations."""

    def test_requires_openai_api_key(self) -> None:
        with pytest.raises(ValueError, match="openai_api_key is required"):
            OpenAIRealtimeProviderConfig(openai_api_key=None, default_session_config={"model": "x"})

    def test_on_inbound_call_session_config_alone_is_valid(self) -> None:
        async def customizer(req: TwiMLRequest) -> dict | None:
            return {"model": "gpt-realtime-test"}

        # Must not raise — a customizer alone is a legitimate session_config source.
        OpenAIRealtimeProviderConfig(
            openai_api_key="sk-test", on_inbound_call_session_config=customizer
        )

    def test_neither_source_alone_is_also_valid(self) -> None:
        """Legitimate for an outbound-only provider supplying session_config
        per-call via InitiateVoiceConversationOptionsOpenAIRealtime."""
        OpenAIRealtimeProviderConfig(openai_api_key="sk-test")


class TestInboundCallSessionConfig:
    """Per-inbound-call session_config via on_inbound_call_session_config."""

    @pytest.mark.asyncio
    async def test_customizer_result_used_verbatim_for_that_call(self) -> None:

        async def customizer(req: TwiMLRequest) -> dict | None:
            if req.caller_country == "MX":
                return {
                    "model": "gpt-realtime-mx",
                    "instructions": "Habla en español.",
                    "audio": _VALID_AUDIO,
                }
            return None

        channel = make_channel(on_inbound_call_session_config=customizer)
        provider = channel._provider

        twiml = await provider.handle_incoming_call(
            twiml_request=TwiMLRequest(call_sid="CA_MX", caller_country="MX")
        )
        # The config rides out on the TwiML under a token, so it reaches
        # whichever replica Twilio opens the stream to.
        token = _extract_custom_parameter(twiml, _SESSION_CONFIG_TOKEN_PARAM)
        assert provider._call_session_configs[token] == {
            "model": "gpt-realtime-mx",
            "instructions": "Habla en español.",
            "audio": _VALID_AUDIO,
        }

        twilio_ws = FakeTwilioWebSocket(
            events=[
                {
                    "event": "start",
                    "start": {
                        "callSid": "CA_MX",
                        "streamSid": "MZ_MX",
                        "customParameters": {_SESSION_CONFIG_TOKEN_PARAM: token},
                    },
                }
            ]
        )
        model_ws = FakeModelWebSocket(events=[], stay_open=False)

        with patch(
            "tac.channels.voice.media_streams.openai_realtime.provider.websockets.connect",
            new=AsyncMock(return_value=model_ws),
        ) as mock_connect:
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        assert "model=gpt-realtime-mx" in mock_connect.call_args.args[0]
        sent_session = next(m for m in model_ws.sent if m["type"] == "session.update")
        assert sent_session["session"] == {
            "model": "gpt-realtime-mx",
            "instructions": "Habla en español.",
            "audio": _VALID_AUDIO,
        }
        assert token not in provider._call_session_configs
        assert "CA_MX" not in provider._call_session_configs

    @pytest.mark.asyncio
    async def test_customizer_returning_none_falls_back_to_default(self) -> None:
        async def customizer(req: TwiMLRequest) -> dict | None:
            return None

        channel = make_channel(on_inbound_call_session_config=customizer)
        provider = channel._provider

        twiml = await provider.handle_incoming_call(
            twiml_request=TwiMLRequest(call_sid="CA_US", caller_country="US")
        )
        assert _SESSION_CONFIG_TOKEN_PARAM not in twiml
        assert len(provider._call_session_configs) == 0

        twilio_ws = FakeTwilioWebSocket(
            events=[{"event": "start", "start": {"callSid": "CA_US", "streamSid": "MZ_US"}}]
        )
        model_ws = FakeModelWebSocket(events=[], stay_open=False)

        with patch(
            "tac.channels.voice.media_streams.openai_realtime.provider.websockets.connect",
            new=AsyncMock(return_value=model_ws),
        ):
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        sent_session = next(m for m in model_ws.sent if m["type"] == "session.update")
        assert sent_session["session"] == {"model": "gpt-realtime-test", "audio": _VALID_AUDIO}

    @pytest.mark.asyncio
    async def test_customizer_runs_without_a_call_sid(self) -> None:
        """Correlation is by token now, so a CallSid isn't a prerequisite."""
        customizer = AsyncMock(return_value={"model": "gpt-realtime-mx", "audio": _VALID_AUDIO})

        channel = make_channel(on_inbound_call_session_config=customizer)
        provider = channel._provider

        twiml = await provider.handle_incoming_call(twiml_request=TwiMLRequest(caller_country="MX"))

        customizer.assert_awaited_once()
        token = _extract_custom_parameter(twiml, _SESSION_CONFIG_TOKEN_PARAM)
        assert provider._call_session_configs[token]["model"] == "gpt-realtime-mx"

    @pytest.mark.asyncio
    async def test_no_customizer_registered_adds_no_token(self) -> None:
        channel = make_channel()
        provider = channel._provider

        twiml = await provider.handle_incoming_call(
            twiml_request=TwiMLRequest(call_sid="CA_US", caller_country="US")
        )

        assert _SESSION_CONFIG_TOKEN_PARAM not in twiml
        assert len(provider._call_session_configs) == 0

    @pytest.mark.asyncio
    async def test_missing_model_field_raises_at_connect_time(self) -> None:
        """Calls _connect_model directly — handle_websocket swallows
        exceptions into a log line."""
        channel = make_channel()
        provider = channel._provider
        provider._call_session_configs["CA_BAD"] = {"instructions": "no model field here"}

        with pytest.raises(ValueError, match="must include a 'model' field"):
            await provider._connect_model("CA_BAD")

    @pytest.mark.asyncio
    async def test_missing_session_config_entirely_raises_at_connect_time(self) -> None:
        async def customizer(req: TwiMLRequest) -> dict | None:
            return None

        tac = TAC(get_test_tac_config())
        config = OpenAIRealtimeProviderConfig(
            openai_api_key="sk-test", on_inbound_call_session_config=customizer
        )
        provider = VoiceChannel(tac, config=config)._provider

        with pytest.raises(ValueError, match="No session_config available"):
            await provider._connect_model("CA_NONE")


class TestOutboundCallSessionConfig:
    """Per-outbound-call session_config via InitiateVoiceConversationOptionsOpenAIRealtime."""

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
                InitiateVoiceConversationOptionsOpenAIRealtime(
                    to="+15551234567",
                    session_config={"model": "gpt-realtime-outbound"},
                )
            )

        assert result.call_sid == "CA_OUT"
        assert "CA_OUT" not in provider._call_session_configs
        twiml = mock_client.calls.create.call_args.kwargs["twiml"]
        token = _extract_custom_parameter(twiml, _SESSION_CONFIG_TOKEN_PARAM)
        assert provider._call_session_configs[token] == {"model": "gpt-realtime-outbound"}

    @pytest.mark.asyncio
    async def test_session_config_token_rekeyed_to_call_sid_on_connect(self) -> None:
        """Once the WebSocket start event arrives, the token entry is
        rekeyed to call_sid and consumed by _connect_model."""
        channel = make_channel()
        provider = channel._provider

        mock_call = MagicMock(sid="CA_OUT4")
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call

        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            await provider.initiate_outbound_conversation(
                InitiateVoiceConversationOptionsOpenAIRealtime(
                    to="+15551234567",
                    session_config={"model": "gpt-realtime-outbound4", "audio": _VALID_AUDIO},
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
            "tac.channels.voice.media_streams.openai_realtime.provider.websockets.connect",
            new=AsyncMock(return_value=model_ws),
        ) as mock_connect:
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        assert "model=gpt-realtime-outbound4" in mock_connect.call_args.args[0]
        sent_session = next(m for m in model_ws.sent if m["type"] == "session.update")
        assert sent_session["session"] == {
            "model": "gpt-realtime-outbound4",
            "audio": _VALID_AUDIO,
        }
        assert len(provider._call_session_configs) == 0

    @pytest.mark.asyncio
    async def test_session_config_token_cleaned_up_if_call_creation_fails(self) -> None:
        channel = make_channel()
        provider = channel._provider

        mock_client = MagicMock()
        mock_client.calls.create.side_effect = RuntimeError("boom")

        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            with pytest.raises(RuntimeError):
                await provider.initiate_outbound_conversation(
                    InitiateVoiceConversationOptionsOpenAIRealtime(
                        to="+15551234567",
                        session_config={"model": "gpt-realtime-outbound"},
                    )
                )

        assert len(provider._call_session_configs) == 0

    @pytest.mark.asyncio
    async def test_session_config_token_not_leaked_if_twiml_build_fails(self) -> None:
        """A failure building the TwiML itself (e.g. no resolvable WebSocket
        URL) happens before calls.create() is ever attempted — the token
        must not be stashed until that succeeds, or it's never cleaned up."""
        tac = TAC(get_test_tac_config())
        config = OpenAIRealtimeProviderConfig(
            openai_api_key="sk-test", default_session_config={"model": "x", "audio": _VALID_AUDIO}
        )
        channel = VoiceChannel(tac, config=config)
        provider = channel._provider
        channel.tac.config.voice_public_domain = None  # no fallback WebSocket URL

        with pytest.raises(ValueError, match="needs a WebSocket URL"):
            await provider.initiate_outbound_conversation(
                InitiateVoiceConversationOptionsOpenAIRealtime(
                    to="+15551234567",
                    session_config={"model": "gpt-realtime-outbound"},
                )
            )

        assert len(provider._call_session_configs) == 0

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

        assert len(provider._call_session_configs) == 0

    @pytest.mark.asyncio
    async def test_outbound_only_config_with_no_default_connects_via_per_call_override(
        self,
    ) -> None:
        tac = TAC(get_test_tac_config())
        config = OpenAIRealtimeProviderConfig(openai_api_key="sk-test")
        channel = VoiceChannel(tac, config=config)
        provider = channel._provider

        mock_call = MagicMock(sid="CA_OUT3")
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call

        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            await provider.initiate_outbound_conversation(
                InitiateVoiceConversationOptionsOpenAIRealtime(
                    to="+15551234567",
                    session_config={
                        "model": "gpt-realtime-outbound-only",
                        "audio": _VALID_AUDIO,
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
                        "callSid": "CA_OUT3",
                        "streamSid": "MZ_OUT3",
                        "customParameters": {_SESSION_CONFIG_TOKEN_PARAM: token},
                    },
                }
            ]
        )
        model_ws = FakeModelWebSocket(events=[], stay_open=False)

        with patch(
            "tac.channels.voice.media_streams.openai_realtime.provider.websockets.connect",
            new=AsyncMock(return_value=model_ws),
        ) as mock_connect:
            await asyncio.wait_for(provider.handle_websocket(twilio_ws), timeout=5)

        assert "model=gpt-realtime-outbound-only" in mock_connect.call_args.args[0]


class TestCallEventCallbackWiring:
    """``OpenAIRealtimeProvider._build_call_kwargs`` shares its call-event-URL
    wiring with ``ConversationRelayProvider`` via
    ``VoiceProvider._apply_call_event_callbacks`` — covers the same behavior
    on this provider's outbound path (see ``TestNoAutoWiring`` in
    test_outbound.py for the ConversationRelay side).
    """

    @staticmethod
    def _noop_handler() -> Any:
        async def handler(event: Any) -> None:
            return None

        return handler

    async def _place_call(
        self, channel: VoiceChannel, *, websocket_url: str | None = None
    ) -> dict[str, Any]:
        mock_call = MagicMock(sid="CA_WIRE")
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call
        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            await channel.initiate_outbound_conversation(
                InitiateVoiceConversationOptions(to="+15559876543", websocket_url=websocket_url)
            )
        return mock_client.calls.create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_no_auto_wiring_without_handlers(self) -> None:
        channel = make_channel()
        kwargs = await self._place_call(channel)
        assert "status_callback" not in kwargs
        assert "async_amd_status_callback" not in kwargs
        assert "recording_status_callback" not in kwargs

    @pytest.mark.asyncio
    async def test_each_callback_wired_only_for_its_handler(self) -> None:
        channel = make_channel()
        channel.on_amd(self._noop_handler())
        kwargs = await self._place_call(channel)
        assert kwargs["async_amd_status_callback"] == "https://example.com/twilio/call-events/amd"
        assert "status_callback" not in kwargs
        assert "recording_status_callback" not in kwargs

    @pytest.mark.asyncio
    async def test_all_callbacks_wired_when_all_handlers_registered(self) -> None:
        channel = make_channel()
        channel.on_call_status(self._noop_handler())
        channel.on_amd(self._noop_handler())
        channel.on_recording(self._noop_handler())
        kwargs = await self._place_call(channel)
        assert kwargs["status_callback"] == "https://example.com/twilio/call-events/status"
        assert kwargs["async_amd_status_callback"] == "https://example.com/twilio/call-events/amd"
        assert (
            kwargs["recording_status_callback"]
            == "https://example.com/twilio/call-events/recording"
        )

    @pytest.mark.asyncio
    async def test_no_auto_wiring_without_domain(self) -> None:
        tac = TAC({**get_test_tac_config(), "voice_public_domain": None})
        channel = VoiceChannel(
            tac,
            config=OpenAIRealtimeProviderConfig(
                openai_api_key="sk-test",
                default_session_config={"model": "gpt-realtime-test", "audio": _VALID_AUDIO},
            ),
        )
        channel.on_call_status(self._noop_handler())
        kwargs = await self._place_call(channel, websocket_url="wss://example.com/ws")
        assert "status_callback" not in kwargs

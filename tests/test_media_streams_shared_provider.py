"""Tests for ``MediaStreamsOpenAIProvider``, the base shared by
``OpenAIRealtimeProvider`` and ``GPTLiveProvider``.

Exercised through ``OpenAIRealtimeProvider`` since the methods under test
(``get_transcript``, ``get_websocket``, ``handle_incoming_call``'s type
checks) are verified identical across both providers — see
test_openai_realtime_provider.py's ``TestCallEventCallbackWiring`` docstring
for the same reasoning applied to call-event wiring.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tac import TAC
from tac.channels.voice import VoiceChannel
from tac.channels.voice.media_streams.gpt_live import GPTLiveProviderConfig
from tac.channels.voice.media_streams.openai_realtime import OpenAIRealtimeProviderConfig
from tac.channels.voice.media_streams.openai_realtime.provider import (
    TWILIO_AUDIO_FORMAT_FOR_REALTIME,
)
from tac.channels.voice.media_streams.shared.openai_provider import SESSION_CONFIG_TOKEN_PARAM
from tac.models.voice import (
    TwiMLRequest,
    VoiceTwiMLOptionsConversationRelay,
    VoiceTwiMLOptionsMediaStreams,
)

_VALID_AUDIO = {
    "input": {"format": TWILIO_AUDIO_FORMAT_FOR_REALTIME},
    "output": {"format": TWILIO_AUDIO_FORMAT_FOR_REALTIME},
}


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


class TestGetTranscript:
    def test_returns_transcript_from_session_metadata(self) -> None:
        channel = make_channel()
        provider = channel._provider
        session = channel._start_conversation("CA1", profile_id=None)
        session.metadata["transcript"] = [{"role": "user", "text": "hi"}]

        assert provider.get_transcript("CA1") == [{"role": "user", "text": "hi"}]

    def test_returns_empty_list_for_unknown_conversation(self) -> None:
        channel = make_channel()
        provider = channel._provider

        assert provider.get_transcript("no-such-call") == []


class TestGetWebsocket:
    def test_returns_none_for_unknown_conversation(self) -> None:
        channel = make_channel()
        provider = channel._provider

        assert provider.get_websocket("no-such-call") is None


class TestHandleIncomingCallTypeChecks:
    @pytest.mark.asyncio
    async def test_wrong_host_twiml_options_type_raises(self) -> None:
        channel = make_channel()
        provider = channel._provider

        with pytest.raises(TypeError, match="requires host_twiml_options"):
            await provider.handle_incoming_call(
                host_twiml_options=VoiceTwiMLOptionsConversationRelay()
            )

    @pytest.mark.asyncio
    async def test_customizer_returning_valid_type_is_used(self) -> None:
        async def customizer(req: TwiMLRequest) -> VoiceTwiMLOptionsMediaStreams:
            return VoiceTwiMLOptionsMediaStreams(name="custom-stream")

        channel = make_channel()
        channel.on_inbound_call_twiml(customizer)
        provider = channel._provider

        twiml = await provider.handle_incoming_call(twiml_request=TwiMLRequest(call_sid="CA1"))

        assert 'name="custom-stream"' in twiml

    @pytest.mark.asyncio
    async def test_customizer_returning_wrong_type_raises(self) -> None:
        async def bad_customizer(req: TwiMLRequest) -> VoiceTwiMLOptionsConversationRelay:
            return VoiceTwiMLOptionsConversationRelay()

        channel = make_channel()
        channel.on_inbound_call_twiml(bad_customizer)
        provider = channel._provider

        with pytest.raises(TypeError, match="on_inbound_call_twiml customizer"):
            await provider.handle_incoming_call(twiml_request=TwiMLRequest(call_sid="CA1"))


def make_media_streams_provider(kind: str) -> Any:
    tac = TAC(get_test_tac_config())
    config: Any
    if kind == "gpt_live":
        config = GPTLiveProviderConfig(
            openai_api_key="sk-test", default_session_config={"model": "gpt-live-1"}
        )
    else:
        config = OpenAIRealtimeProviderConfig(openai_api_key="sk-test")
    return VoiceChannel(tac, config=config)._provider


def stream_start(token: str | None) -> dict[str, Any]:
    params = {} if token is None else {SESSION_CONFIG_TOKEN_PARAM: token}
    return {"callSid": "CA1", "streamSid": "MZ1", "customParameters": params}


@pytest.mark.parametrize("kind", ["openai_realtime", "gpt_live"])
class TestSessionConfigClaim:
    """A per-call session config is stashed on the instance that served the
    TwiML webhook or placed the call; the stream must claim it there."""

    def test_claims_a_config_stashed_on_this_instance(self, kind: str) -> None:
        provider = make_media_streams_provider(kind)
        provider._call_session_configs["t1"] = {"model": "per-call"}

        with patch.object(provider.logger, "warning") as warning:
            provider._register_call(stream_start("t1"), MagicMock())

        assert provider._call_session_configs.pop("CA1") == {"model": "per-call"}
        assert "t1" not in provider._call_session_configs
        warning.assert_not_called()

    def test_warns_when_the_config_was_stashed_elsewhere(self, kind: str) -> None:
        provider = make_media_streams_provider(kind)

        with patch.object(provider.logger, "warning") as warning:
            provider._register_call(stream_start("t_other_instance"), MagicMock())

        warning.assert_called_once()
        assert "instance_public_domain" in warning.call_args.args[0]
        assert "CA1" not in provider._call_session_configs

    def test_a_call_without_a_per_call_config_does_not_warn(self, kind: str) -> None:
        provider = make_media_streams_provider(kind)

        with patch.object(provider.logger, "warning") as warning:
            provider._register_call(stream_start(None), MagicMock())

        warning.assert_not_called()

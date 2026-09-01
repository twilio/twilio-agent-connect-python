"""Tests for ``MediaStreamsOpenAIProvider``, the base shared by
``OpenAIRealtimeProvider`` and ``GPTLiveProvider``.

Exercised through ``OpenAIRealtimeProvider`` since the methods under test
(``get_transcript``, ``get_websocket``, ``handle_incoming_call``'s type
checks) are verified identical across both providers — see
test_openai_realtime_provider.py's ``TestCallEventCallbackWiring`` docstring
for the same reasoning applied to call-event wiring.
"""

import pytest

from tac import TAC
from tac.channels.voice import VoiceChannel
from tac.channels.voice.media_streams.openai_realtime import OpenAIRealtimeProviderConfig
from tac.channels.voice.media_streams.openai_realtime.provider import (
    TWILIO_MEDIA_STREAM_AUDIO_FORMAT,
)
from tac.models.voice import (
    TwiMLRequest,
    VoiceTwiMLOptionsConversationRelay,
    VoiceTwiMLOptionsMediaStreams,
)

_VALID_AUDIO = {
    "input": {"format": TWILIO_MEDIA_STREAM_AUDIO_FORMAT},
    "output": {"format": TWILIO_MEDIA_STREAM_AUDIO_FORMAT},
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

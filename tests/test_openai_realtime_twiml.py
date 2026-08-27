"""Tests for OpenAIRealtimeProvider's TwiML generation and layering.

Media Streams' ``generate_twiml``/``TwiMLBuilderMediaStreams`` are a
separate implementation from ConversationRelay's (see test_voice_channel.py
for that one) — same shape (customizer > default > host > TAC-default
layering), different verb (``<Connect><Stream>`` vs ``<ConversationRelay>``).
"""

import pytest

from tac.channels.voice.media_streams.openai_realtime import OpenAIRealtimeProviderConfig
from tac.channels.voice.media_streams.openai_realtime.twiml import (
    TwiMLBuilderMediaStreams,
    generate_twiml,
)
from tac.core.config import TACConfig
from tac.models.voice import VoiceTwiMLOptionsMediaStreams


def get_test_tac_config(**overrides: object) -> TACConfig:
    defaults: dict[str, object] = {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "phone_number": "+15551234567",
        "voice_public_domain": "example.com",
    }
    defaults.update(overrides)
    return TACConfig(**defaults)  # type: ignore[arg-type]


def get_test_channel_config(
    default_twiml_options: VoiceTwiMLOptionsMediaStreams | None = None,
) -> OpenAIRealtimeProviderConfig:
    return OpenAIRealtimeProviderConfig(
        openai_api_key="sk-test",
        default_session_config={"model": "gpt-realtime-test"},
        default_twiml_options=default_twiml_options,
    )


class TestGenerateTwiml:
    def test_minimal(self) -> None:
        twiml = generate_twiml("wss://example.com/ws")

        assert '<?xml version="1.0" encoding="UTF-8"?>' in twiml
        assert '<Stream url="wss://example.com/ws" />' in twiml
        assert "<Connect>" in twiml and "</Connect>" in twiml
        assert "<Parameter" not in twiml

    def test_url_from_options_only(self) -> None:
        twiml = generate_twiml(
            options=VoiceTwiMLOptionsMediaStreams(websocket_url="wss://o.com/ws")
        )

        assert 'url="wss://o.com/ws"' in twiml

    def test_positional_url_wins_over_options(self) -> None:
        twiml = generate_twiml(
            "wss://positional.com/ws",
            VoiceTwiMLOptionsMediaStreams(websocket_url="wss://options.com/ws"),
        )

        assert 'url="wss://positional.com/ws"' in twiml
        assert "options.com" not in twiml

    def test_requires_a_url(self) -> None:
        with pytest.raises(ValueError, match="requires a WebSocket URL"):
            generate_twiml()

    def test_whitespace_only_url_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires a WebSocket URL"):
            generate_twiml("   ")

    def test_dict_options_coerced(self) -> None:
        twiml = generate_twiml("wss://example.com/ws", {"name": "my-stream"})

        assert 'name="my-stream"' in twiml

    def test_name_status_callback_rendered(self) -> None:
        twiml = generate_twiml(
            "wss://example.com/ws",
            VoiceTwiMLOptionsMediaStreams(
                name="my-stream",
                status_callback="https://example.com/status",
                status_callback_method="GET",
            ),
        )

        assert 'name="my-stream"' in twiml
        assert 'statusCallback="https://example.com/status"' in twiml
        assert 'statusCallbackMethod="GET"' in twiml

    def test_action_url_and_method_on_connect(self) -> None:
        twiml = generate_twiml(
            "wss://example.com/ws",
            VoiceTwiMLOptionsMediaStreams(
                action_url="https://example.com/action", action_method="GET"
            ),
        )

        assert '<Connect action="https://example.com/action" method="GET">' in twiml

    def test_custom_parameters_rendered_as_parameter_children(self) -> None:
        twiml = generate_twiml(
            "wss://example.com/ws",
            VoiceTwiMLOptionsMediaStreams(
                custom_parameters={"profile_id": "prof_1", "count": 3, "skip_me": None}
            ),
        )

        assert '<Parameter name="profile_id" value="prof_1" />' in twiml
        assert '<Parameter name="count" value="3" />' in twiml
        assert "skip_me" not in twiml

    def test_no_custom_parameters_by_default(self) -> None:
        twiml = generate_twiml("wss://example.com/ws")

        assert "<Parameter" not in twiml


class TestTwiMLBuilderMediaStreamsPrecedence:
    """per_call > default_twiml_options > host > TAC defaults, merged per-field."""

    def test_tac_default_websocket_url_used_when_nothing_else_set(self) -> None:
        builder = TwiMLBuilderMediaStreams(
            get_test_tac_config(voice_public_domain="example.com"),
            get_test_channel_config(),
        )

        twiml = builder.build("test")

        assert 'url="wss://example.com/ws"' in twiml

    def test_no_url_anywhere_raises_with_caller_name(self) -> None:
        builder = TwiMLBuilderMediaStreams(
            get_test_tac_config(voice_public_domain=None), get_test_channel_config()
        )

        with pytest.raises(ValueError, match="my_caller needs a WebSocket URL"):
            builder.build("my_caller")

    def test_explicit_websocket_url_arg_wins_over_everything(self) -> None:
        builder = TwiMLBuilderMediaStreams(
            get_test_tac_config(voice_public_domain="fallback.com"),
            get_test_channel_config(
                default_twiml_options=VoiceTwiMLOptionsMediaStreams(
                    websocket_url="wss://default.com/ws"
                )
            ),
        )

        twiml = builder.build(
            "test",
            per_call=VoiceTwiMLOptionsMediaStreams(websocket_url="wss://per-call.com/ws"),
            websocket_url="wss://explicit.com/ws",
        )

        assert 'url="wss://explicit.com/ws"' in twiml

    def test_default_twiml_options_used_when_no_per_call_or_host(self) -> None:
        builder = TwiMLBuilderMediaStreams(
            get_test_tac_config(),
            get_test_channel_config(
                default_twiml_options=VoiceTwiMLOptionsMediaStreams(name="default-stream")
            ),
        )

        twiml = builder.build("test")

        assert 'name="default-stream"' in twiml

    def test_per_call_overrides_default_twiml_options(self) -> None:
        builder = TwiMLBuilderMediaStreams(
            get_test_tac_config(),
            get_test_channel_config(
                default_twiml_options=VoiceTwiMLOptionsMediaStreams(name="default-stream")
            ),
        )

        twiml = builder.build(
            "test", per_call=VoiceTwiMLOptionsMediaStreams(name="per-call-stream")
        )

        assert 'name="per-call-stream"' in twiml
        assert "default-stream" not in twiml

    def test_per_call_only_overrides_the_fields_it_sets(self) -> None:
        """Fields not set on per_call fall through to default_twiml_options —
        overriding one field doesn't wipe out the others."""
        builder = TwiMLBuilderMediaStreams(
            get_test_tac_config(),
            get_test_channel_config(
                default_twiml_options=VoiceTwiMLOptionsMediaStreams(
                    name="default-stream", status_callback="https://example.com/status"
                )
            ),
        )

        twiml = builder.build(
            "test", per_call=VoiceTwiMLOptionsMediaStreams(name="per-call-stream")
        )

        assert 'name="per-call-stream"' in twiml
        assert 'statusCallback="https://example.com/status"' in twiml

    def test_host_options_fall_through_when_no_default_or_per_call(self) -> None:
        builder = TwiMLBuilderMediaStreams(get_test_tac_config(), get_test_channel_config())

        twiml = builder.build(
            "test", host=VoiceTwiMLOptionsMediaStreams(websocket_url="wss://host.com/ws")
        )

        assert 'url="wss://host.com/ws"' in twiml

    def test_default_twiml_options_overrides_host(self) -> None:
        builder = TwiMLBuilderMediaStreams(
            get_test_tac_config(),
            get_test_channel_config(
                default_twiml_options=VoiceTwiMLOptionsMediaStreams(name="default-stream")
            ),
        )

        twiml = builder.build("test", host=VoiceTwiMLOptionsMediaStreams(name="host-stream"))

        assert 'name="default-stream"' in twiml
        assert "host-stream" not in twiml

    def test_custom_parameters_replace_wholesale_not_merged(self) -> None:
        """A higher-priority layer's custom_parameters replaces the lower
        layer's entirely — no per-key merging."""
        builder = TwiMLBuilderMediaStreams(
            get_test_tac_config(),
            get_test_channel_config(
                default_twiml_options=VoiceTwiMLOptionsMediaStreams(
                    custom_parameters={"a": "1", "b": "2"}
                )
            ),
        )

        twiml = builder.build(
            "test",
            per_call=VoiceTwiMLOptionsMediaStreams(custom_parameters={"c": "3"}),
        )

        assert '<Parameter name="c" value="3" />' in twiml
        assert 'name="a"' not in twiml
        assert 'name="b"' not in twiml

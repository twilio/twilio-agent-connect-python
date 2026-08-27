"""``OpenAIRealtimeProvider`` configuration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pydantic import Field

from tac.channels.voice.media_streams.openai_realtime.provider import OpenAIRealtimeProvider
from tac.channels.voice.provider import VoiceProvider, VoiceProviderConfig
from tac.core.config import TACConfig
from tac.models.voice import VoiceTwiMLOptionsMediaStreams
from tac.tools import TACTool

if TYPE_CHECKING:
    from tac.channels.voice.channel import VoiceChannel

#: Twilio Media Streams always sends 8kHz G.711 u-law audio — this is a fact
#: about Twilio's stream, not a provider choice, so ``session_config`` should
#: use this same audio format for both input and output. No ``rate`` field —
#: OpenAI's Realtime ``session.audio.*.format`` schema rejects it as an
#: unknown parameter (g711 is inherently fixed-rate, so it isn't settable).
TWILIO_MEDIA_STREAM_AUDIO_FORMAT: dict[str, Any] = {"type": "audio/pcmu"}


class OpenAIRealtimeProviderConfig(VoiceProviderConfig):
    """Configuration for ``OpenAIRealtimeProvider``."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    openai_api_key: str | None = Field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY"),
        description="OpenAI API key. Defaults to the OPENAI_API_KEY environment variable.",
    )
    tools: list[TACTool] = Field(
        default_factory=list,
        description=(
            "Executable TACTool implementations, looked up by name to run mid-call tool "
            "requests. This alone does not tell the model these tools exist — also add each "
            "tool's `to_realtime_format()` schema to `session_config['tools']`."
        ),
    )
    welcome_greeting_response: dict[str, Any] | None = Field(
        default=None,
        description="If set, sent verbatim as response.create's 'response' payload "
        "when the call connects — e.g. {'instructions': 'Hi there!'}. No SDK-added "
        "wrapping text or language assumption.",
    )
    session_config: dict[str, Any] = Field(
        ...,
        description="The session.update payload's 'session' body, sent once the "
        "model connects. See https://developers.openai.com/api/reference/resources/"
        "realtime/client-events#session.update for the schema. If using `tools`, its "
        "'tools' entry must separately list each tool's `to_realtime_format()` schema — "
        "this config is passed to OpenAI as-is, with no tool schemas merged in.",
    )
    default_twiml_options: VoiceTwiMLOptionsMediaStreams | None = Field(
        default=None,
        description="Static VoiceTwiMLOptionsMediaStreams applied to every inbound call. "
        "Per-call customization is registered via VoiceChannel.on_inbound_call_twiml(...), "
        "which takes precedence over this.",
    )

    def create_provider(self, channel: VoiceChannel, tac_config: TACConfig) -> VoiceProvider:
        return OpenAIRealtimeProvider(channel, tac_config, self)

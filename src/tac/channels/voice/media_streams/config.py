"""Shared configuration for ``VoiceProvider``s built on Twilio Media Streams."""

from __future__ import annotations

from pydantic import Field

from tac.channels.voice.provider import VoiceProviderConfig
from tac.models.voice import VoiceTwiMLOptionsMediaStreams


class MediaStreamsProviderConfig(VoiceProviderConfig):
    """Base configuration for a Media Streams (``<Connect><Stream>``) provider.

    ``OpenAIRealtimeProviderConfig`` is the only subclass today. Lives here
    (not in ``openai_realtime/config.py``) so a future Media Streams provider's
    config only needs to import this module, not anything OpenAI-specific.
    """

    default_twiml_options: VoiceTwiMLOptionsMediaStreams | None = Field(
        default=None,
        description="Static VoiceTwiMLOptionsMediaStreams applied to every inbound call. "
        "Per-call customization is registered via VoiceChannel.on_inbound_call_twiml(...), "
        "which takes precedence over this.",
    )

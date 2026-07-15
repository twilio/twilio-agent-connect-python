"""Configuration for the realtime (Media Streams + OpenAI Realtime) voice channel."""

import os

from pydantic import BaseModel, Field

from tac.tools.base import TACTool

# Twilio Media Streams delivers 8kHz G.711 u-law audio; the model must be told
# to consume and produce that same format so no transcoding is needed. In the GA
# Realtime API the format is an object with a MIME-style type — u-law is
# "audio/pcmu" (vs the old beta's flat "g711_ulaw" string).
TWILIO_AUDIO_FORMAT = {"type": "audio/pcmu"}

DEFAULT_REALTIME_MODEL = "gpt-realtime"
# OpenAI recommends marin/cedar for the best audio quality of the available voices.
DEFAULT_REALTIME_VOICE = "marin"
DEFAULT_REALTIME_INSTRUCTIONS = """\
# Personality & Tone
## Personality
A warm, friendly voice assistant speaking with a caller over the phone — like a \
helpful colleague, not a script.

## Tone
Conversational and natural. Always speak English.

## Pacing
Keep responses short — a sentence or two per turn.

## Variety
Do not repeat the same phrasing twice in a row. Vary your wording and \
acknowledgements so you don't sound robotic.

## Format
No markdown, emojis, or bullet lists — your words will be spoken aloud.\
"""
DEFAULT_WELCOME_GREETING = "Hello! How can I help you today?"


class RealtimeVoiceChannelConfig(BaseModel):
    """Configuration for :class:`RealtimeVoiceChannel`.

    Unlike ``VoiceChannelConfig`` (which configures the TwiML inside
    ``<ConversationRelay>``), this configures the bridge to a speech-to-speech
    model: which model/voice/instructions to use.
    """

    openai_api_key: str = Field(
        description="OpenAI API key used to authenticate the Realtime WebSocket connection.",
    )
    model: str = Field(
        default=DEFAULT_REALTIME_MODEL,
        description="OpenAI Realtime model id (sent as the ?model= query param).",
    )
    voice: str = Field(
        default=DEFAULT_REALTIME_VOICE,
        description="Realtime voice name (e.g. 'marin', 'cedar', 'ash', 'ballad', 'coral', "
        "'sage', 'verse'). OpenAI recommends 'marin' or 'cedar' for the best audio quality.",
    )
    instructions: str = Field(
        default=DEFAULT_REALTIME_INSTRUCTIONS,
        description="System instructions sent to the model in the initial session.update.",
    )
    welcome_greeting: str | None = Field(
        default=DEFAULT_WELCOME_GREETING,
        description="If set, the model speaks this greeting when the call connects "
        "(via response.create). Set to None to have it wait for the caller to speak first.",
    )
    tools: list[TACTool] = Field(
        default_factory=list,
        description="TACTool functions the model can call mid-call. Sent as session.tools "
        "in the initial session.update; calls are executed locally and their results "
        "returned via conversation.item.create + response.create.",
    )

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    @classmethod
    def from_env(cls, tools: list[TACTool] | None = None) -> "RealtimeVoiceChannelConfig":
        """Build config from environment variables.

        Reads ``OPENAI_API_KEY`` (required) plus optional
        ``OPENAI_REALTIME_MODEL`` / ``OPENAI_REALTIME_VOICE``.

        Args:
            tools: Optional TACTool functions the model can call mid-call.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "RealtimeVoiceChannelConfig.from_env requires the OPENAI_API_KEY env var."
            )
        return cls(
            openai_api_key=api_key,
            model=os.environ.get("OPENAI_REALTIME_MODEL", DEFAULT_REALTIME_MODEL),
            voice=os.environ.get("OPENAI_REALTIME_VOICE", DEFAULT_REALTIME_VOICE),
            tools=tools or [],
        )

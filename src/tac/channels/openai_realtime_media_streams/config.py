"""Configuration for the OpenAI Realtime (Media Streams) voice channel."""

import os

from pydantic import BaseModel, Field

from tac.tools import TACTool

# Twilio Media Streams delivers 8kHz G.711 u-law audio; the model must be told
# to consume and produce that same format so no transcoding is needed. GA
# Realtime API audio formats are objects with a MIME-style type — u-law is
# "audio/pcmu" (not the old beta's flat "g711_ulaw" string).
TWILIO_AUDIO_FORMAT = {"type": "audio/pcmu"}

DEFAULT_REALTIME_MODEL = "gpt-realtime-2.1"


class OpenAIRealtimeMediaStreamsChannelConfig(BaseModel):
    """Configuration for ``OpenAIRealtimeMediaStreamsChannel``.

    Unlike ``OpenAIRealtimeSipChannelConfig`` (an accept-payload builder for a
    call OpenAI answers directly), this configures a channel that itself
    bridges raw audio between a Twilio Media Stream and the OpenAI Realtime
    WebSocket — so there's no accept step; connecting the WebSocket *is* the
    acceptance.

    What the session actually looks like (model, voice, turn detection,
    instructions) is deliberately NOT here — it's built per call via TAC's
    ``on_message_ready`` callback (reused from text channels: called with an
    empty ``user_message`` and ``memory_response=None``, and expected to
    return a JSON-encoded session dict instead of reply text). This config
    only holds what's needed to open the connection and dispatch tool calls
    locally.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    openai_api_key: str | None = Field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY"),
        description="OpenAI API key. Defaults to the OPENAI_API_KEY environment variable.",
    )
    model: str = Field(
        default=DEFAULT_REALTIME_MODEL,
        description="OpenAI Realtime model id (sent as the ?model= query param when "
        "opening the WebSocket connection). The session's own 'model' field, sent in "
        "session.update, comes from the on_message_ready callback and may differ if desired.",
    )
    tools: list[TACTool] = Field(
        default_factory=list,
        description="TACTool functions the model can call mid-call. Calls are executed "
        "locally and their results returned via conversation.item.create + "
        "response.create. Include the matching tool.to_realtime_format() entries in the "
        "session dict returned by on_message_ready for the model to see them.",
    )
    welcome_greeting: str | None = Field(
        default=None,
        description="If set, the model speaks this greeting when the call connects "
        "(via response.create). Set to None (the default) to have it wait for the "
        "caller to speak first. A fixed string rather than a callback since — unlike "
        "the session config itself — the greeting rarely needs to vary per call.",
    )

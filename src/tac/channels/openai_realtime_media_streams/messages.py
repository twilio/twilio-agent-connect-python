"""Pydantic models for Twilio Media Streams WebSocket messages, as bridged by
``OpenAIRealtimeMediaStreamsChannel``. Twilio's raw event shape is documented at
https://www.twilio.com/docs/voice/media-streams/websocket-messages — this
only models the ``start`` payload for now (the rest of ``handle_websocket``
still reads ``media``/``stop`` fields directly; promote them here too if
they grow beyond a couple of ``.get()`` calls).
"""

from typing import Any

from pydantic import BaseModel, Field


class StreamStartMessage(BaseModel):
    """The ``start`` field of Twilio's Media Stream ``start`` event.

    ``customParameters`` carries whatever ``<Parameter>`` tags were put in
    the ``<Stream>`` TwiML — see ``OpenAIRealtimeMediaStreamsChannel.build_stream_twiml``.
    """

    stream_sid: str | None = Field(None, alias="streamSid")
    call_sid: str | None = Field(None, alias="callSid")
    media_format: dict[str, Any] | None = Field(None, alias="mediaFormat")
    custom_parameters: dict[str, str] = Field(default_factory=dict, alias="customParameters")

    model_config = {"populate_by_name": True}

    @property
    def conversation_id(self) -> str:
        """The id this channel tracks the call under — Twilio's callSid,
        falling back to the stream's own sid, then a fixed placeholder if
        Twilio ever omits both (shouldn't happen in practice).
        """
        return self.call_sid or self.stream_sid or "unknown-call"

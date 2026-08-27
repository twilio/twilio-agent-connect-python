"""Pydantic models for Twilio Media Streams (``<Connect><Stream>``) WebSocket messages."""

from typing import Any

from pydantic import BaseModel, Field


class StreamStartMessage(BaseModel):
    """The ``start`` field of Twilio's Media Stream ``start`` event."""

    stream_sid: str | None = Field(None, alias="streamSid")
    call_sid: str | None = Field(None, alias="callSid")
    media_format: dict[str, Any] | None = Field(None, alias="mediaFormat")
    custom_parameters: dict[str, str] = Field(default_factory=dict, alias="customParameters")

    model_config = {"populate_by_name": True}

    @property
    def conversation_id(self) -> str:
        return self.call_sid or self.stream_sid or "unknown-call"

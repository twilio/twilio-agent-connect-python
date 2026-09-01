"""``MediaStreamsOpenAIProviderConfig``: config fields shared by every ``VoiceProvider``
bridging Twilio Media Streams to an OpenAI real-time voice API.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import Field, model_validator

from tac.channels.voice.media_streams.config import MediaStreamsProviderConfig
from tac.models.voice import TwiMLRequest
from tac.tools import TACTool


class MediaStreamsOpenAIProviderConfig(MediaStreamsProviderConfig):
    """Shared configuration for a Media Streams provider bridging to an OpenAI
    real-time voice API.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    openai_api_key: str | None = Field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY"),
        description="OpenAI API key. Defaults to the OPENAI_API_KEY environment variable.",
    )
    tools: list[TACTool] = Field(
        default_factory=list,
        description="Executable TACTool implementations, looked up by name to run "
        "mid-call tool requests.",
    )
    default_session_config: dict[str, Any] | None = Field(
        default=None,
        description="The session.update payload's 'session' body, sent once the model "
        "connects — used for any call that doesn't supply its own via "
        "`on_inbound_call_session_config`.",
    )
    on_inbound_call_session_config: (
        Callable[[TwiMLRequest], Awaitable[dict[str, Any] | None]] | None
    ) = Field(
        default=None,
        description="Per-inbound-call override for `default_session_config`, called with the "
        "TwiMLRequest. Its return value is used verbatim (not merged with "
        "`default_session_config`); return None to fall back to it.",
    )

    @model_validator(mode="after")
    def _validate_openai_api_key(self) -> MediaStreamsOpenAIProviderConfig:
        if not self.openai_api_key:
            raise ValueError(
                f"openai_api_key is required. Set the OPENAI_API_KEY environment "
                f"variable or provide openai_api_key in {type(self).__name__}."
            )
        return self

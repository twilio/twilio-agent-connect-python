"""OpenAI Realtime (SIP) channel configuration."""

import os
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from tac.channels.openai_realtime_sip.models import (
    OpenAIRealtimeSipCallIncoming,
    OpenAIRealtimeSipSessionConfig,
)
from tac.models.memory import MemoryMode

CallIncomingHandler = Callable[
    [OpenAIRealtimeSipCallIncoming], Awaitable[OpenAIRealtimeSipSessionConfig]
]


class OpenAIRealtimeSipChannelConfig(BaseModel):
    """Configuration for the OpenAI Realtime (SIP) channel.

    This channel bridges phone calls to OpenAI's Realtime API over SIP:
    Twilio forwards the call at the SIP level directly to OpenAI, so audio
    never passes through TAC. TAC only handles the ``realtime.call.incoming``
    webhook (deciding whether/how to accept the call) and, optionally, a
    JSON-only control WebSocket for tool-calling and event observation.

    This is independent of Conversation Orchestrator/Memory — no
    ``conversation_configuration_id`` is required.

    Attributes:
        openai_api_key: OpenAI API key used for the accept/reject/hangup/refer
            REST calls and the control WebSocket. Defaults to the
            ``OPENAI_API_KEY`` environment variable.
        openai_webhook_secret: Secret used to verify the ``realtime.call.incoming``
            webhook signature. Defaults to the ``OPENAI_WEBHOOK_SECRET``
            environment variable.
        memory_mode: Memory retrieval mode for this channel. Default is "never".
    """

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    openai_api_key: str | None = Field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY"),
        description="OpenAI API key. Defaults to the OPENAI_API_KEY environment variable.",
    )
    openai_webhook_secret: str | None = Field(
        default_factory=lambda: os.environ.get("OPENAI_WEBHOOK_SECRET"),
        description="Secret used to verify the realtime.call.incoming webhook signature. "
        "Defaults to the OPENAI_WEBHOOK_SECRET environment variable.",
    )
    memory_mode: MemoryMode = Field(
        default="never",
        description="Memory retrieval mode for this channel",
    )

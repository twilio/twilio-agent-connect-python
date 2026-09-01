"""``OpenAIRealtimeProvider`` configuration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pydantic import Field

from tac.channels.voice.media_streams.openai_realtime.provider import OpenAIRealtimeProvider
from tac.channels.voice.media_streams.shared.config import MediaStreamsOpenAIProviderConfig
from tac.channels.voice.provider import VoiceProvider
from tac.core.config import TACConfig
from tac.models.voice import TwiMLRequest
from tac.tools import TACTool

if TYPE_CHECKING:
    from tac.channels.voice.channel import VoiceChannel


class OpenAIRealtimeProviderConfig(MediaStreamsOpenAIProviderConfig):
    """Configuration for ``OpenAIRealtimeProvider``."""

    tools: list[TACTool] = Field(
        default_factory=list,
        description=(
            "Executable TACTool implementations, looked up by name to run mid-call tool "
            "requests. This alone does not tell the model these tools exist — also add each "
            "tool's `to_realtime_format()` schema to `default_session_config['tools']`."
        ),
    )
    welcome_greeting_response: dict[str, Any] | None = Field(
        default=None,
        description="If set, sent verbatim as response.create's 'response' payload "
        "when the call connects — e.g. {'instructions': 'Hi there!'}. No SDK-added "
        "wrapping text or language assumption.",
    )
    default_session_config: dict[str, Any] | None = Field(
        default=None,
        description="The session.update payload's 'session' body, sent once the model "
        "connects — used for any call that doesn't supply its own via "
        "`on_inbound_call_session_config` or `InitiateVoiceConversationOptionsOpenAIRealtime`. "
        "See https://developers.openai.com/api/reference/resources/realtime/"
        "client-events#session.update for the schema. If using `tools`, its 'tools' entry "
        "must separately list each tool's `to_realtime_format()` schema — this config is "
        "passed to OpenAI as-is, with no tool schemas merged in.",
    )
    on_inbound_call_session_config: (
        Callable[[TwiMLRequest], Awaitable[dict[str, Any] | None]] | None
    ) = Field(
        default=None,
        description="Per-inbound-call override for `default_session_config`, called with the "
        "TwiMLRequest. Its return value is used verbatim (not merged with "
        "`default_session_config`); return None to fall back to it. Outbound calls don't use "
        "this — see `InitiateVoiceConversationOptionsOpenAIRealtime`.",
    )

    def create_provider(self, channel: VoiceChannel, tac_config: TACConfig) -> VoiceProvider:
        return OpenAIRealtimeProvider(channel, tac_config, self)

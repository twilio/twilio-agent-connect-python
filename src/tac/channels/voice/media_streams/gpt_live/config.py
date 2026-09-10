"""``GPTLiveProvider`` configuration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pydantic import Field

from tac.channels.voice.media_streams.gpt_live.provider import GPTLiveProvider
from tac.channels.voice.media_streams.shared.config import MediaStreamsOpenAIProviderConfig
from tac.channels.voice.provider import VoiceProvider
from tac.core.config import TACConfig
from tac.models.voice import TwiMLRequest
from tac.tools import TACTool

if TYPE_CHECKING:
    from tac.channels.voice.channel import VoiceChannel


class GPTLiveProviderConfig(MediaStreamsOpenAIProviderConfig):
    """Configuration for ``GPTLiveProvider``."""

    tools: list[TACTool] = Field(
        default_factory=list,
        description="Executable TACTool implementations, looked up by name to run "
        "Responses-delegated tool calls. This alone does not tell the model these tools "
        "exist — also add each tool's `to_realtime_format()` schema to "
        "`default_session_config['delegation']['responses']['tools']`.",
    )
    welcome_instruction: str | None = Field(
        default=None,
        description="If set, sent verbatim as a `session.commentary.append` "
        "once `session.started` arrives. Word it as an "
        "instruction, not just a greeting, e.g. 'Greet the caller immediately "
        "using: Hi, how can I help you today?' — a bare greeting won't make the "
        "model speak first.",
    )
    default_session_config: dict[str, Any] | None = Field(
        default=None,
        description="The session.start payload's 'session' body, sent once the model "
        "connects — used for any call that doesn't supply its own via "
        "`on_inbound_call_session_config` or `InitiateVoiceConversationOptionsGPTLive`. "
        "Must include `model` (e.g. 'gpt-live-1'). Set `audio.format` to "
        "`TWILIO_AUDIO_FORMAT_FOR_GPT_LIVE` and, for tool calling, "
        "`delegation = {'type': 'responses', 'responses': {'model': ..., 'tools': [...]}}`.",
    )
    on_inbound_call_session_config: (
        Callable[[TwiMLRequest], Awaitable[dict[str, Any] | None]] | None
    ) = Field(
        default=None,
        description="Per-inbound-call override for `default_session_config`, called with "
        "the TwiMLRequest. Its return value is used verbatim (not merged with "
        "`default_session_config`); return None to fall back to it.",
    )

    def create_provider(self, channel: VoiceChannel, tac_config: TACConfig) -> VoiceProvider:
        return GPTLiveProvider(channel, tac_config, self)

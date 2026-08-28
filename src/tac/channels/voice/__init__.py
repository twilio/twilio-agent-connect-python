"""Voice channel for handling voice-based conversations.

``OpenAIRealtimeProvider`` needs the optional ``websockets`` dependency, so
import it from ``tac.channels.voice.media_streams.openai_realtime`` instead
of from here.
"""

from typing import TYPE_CHECKING, Any

from tac._deprecation import resolve_deprecated_alias
from tac.channels.voice.channel import (
    AmdHandler,
    CallStatusHandler,
    InboundCallTwiMLHandler,
    RecordingHandler,
    VoiceChannel,
)
from tac.channels.voice.conversation_relay import (
    ConversationRelayProvider,
    ConversationRelayProviderConfig,
    generate_twiml,
)
from tac.channels.voice.provider import VoiceProvider, VoiceProviderConfig
from tac.models.outbound import CallOptions
from tac.models.voice import (
    AmdEvent,
    CallStatusEvent,
    InterruptMode,
    LanguageConfig,
    RecordingEvent,
    TwiMLRequest,
    VoiceTwiMLOptions,
    VoiceTwiMLOptionsConversationRelay,
)

__all__ = [
    "AmdEvent",
    "AmdHandler",
    "CallOptions",
    "CallStatusEvent",
    "CallStatusHandler",
    "ConversationRelayProvider",
    "ConversationRelayProviderConfig",
    "InboundCallTwiMLHandler",
    "InterruptMode",
    "LanguageConfig",
    "RecordingEvent",
    "RecordingHandler",
    "TwiMLRequest",
    "VoiceChannel",
    "VoiceProvider",
    "VoiceProviderConfig",
    "VoiceTwiMLOptions",
    "VoiceTwiMLOptionsConversationRelay",
    "generate_twiml",
]

if TYPE_CHECKING:  # static type only, see tac._deprecation
    TwiMLOptions = VoiceTwiMLOptionsConversationRelay
    VoiceChannelConfig = ConversationRelayProviderConfig


def __getattr__(name: str) -> Any:
    # See tac._deprecation. TODO(3.0): remove.
    if name == "TwiMLOptions":
        return resolve_deprecated_alias("TwiMLOptions", VoiceTwiMLOptionsConversationRelay)
    if name == "VoiceChannelConfig":
        return resolve_deprecated_alias("VoiceChannelConfig", ConversationRelayProviderConfig)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

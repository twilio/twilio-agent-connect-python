"""Voice channel for handling voice-based conversations.

``OpenAIRealtimeProvider`` needs the optional ``websockets`` dependency, so
import it from ``tac.channels.voice.media_streams.openai_realtime`` instead
of from here.
"""

from typing import Any

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
    "TwiMLOptions",
    "TwiMLRequest",
    "VoiceChannel",
    "VoiceChannelConfig",
    "VoiceProvider",
    "VoiceProviderConfig",
    "VoiceTwiMLOptions",
    "VoiceTwiMLOptionsConversationRelay",
    "generate_twiml",
]


def __getattr__(name: str) -> Any:
    # Resolved directly (not forwarded to another module's __getattr__) so the
    # warning attributes correctly to this access. See
    # tac._deprecation.resolve_deprecated_alias. TODO(3.0): remove.
    if name == "TwiMLOptions":
        return resolve_deprecated_alias("TwiMLOptions", VoiceTwiMLOptionsConversationRelay)
    if name == "VoiceChannelConfig":
        return resolve_deprecated_alias("VoiceChannelConfig", ConversationRelayProviderConfig)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

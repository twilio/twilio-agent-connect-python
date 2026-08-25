"""Voice channel for handling voice-based conversations.

``VoiceChannel`` is the single public entry point for voice — it owns the
Twilio Calls API lifecycle. The media transport itself is a
``VoiceProvider``; ``ConversationRelayProvider`` (Twilio-managed ASR/TTS)
is the only implementation today.
"""

from tac.channels.voice.channel import VoiceChannel
from tac.channels.voice.config import (
    AmdHandler,
    CallStatusHandler,
    ConversationRelayProviderConfig,
    InboundCallTwiMLHandler,
    RecordingHandler,
    VoiceChannelConfig,
)
from tac.channels.voice.conversation_relay import ConversationRelayProvider
from tac.channels.voice.provider import VoiceProvider
from tac.channels.voice.twiml import generate_twiml
from tac.models.outbound import CallOptions
from tac.models.voice import (
    AmdEvent,
    CallStatusEvent,
    InterruptMode,
    LanguageConfig,
    RecordingEvent,
    TwiMLOptions,
    TwiMLRequest,
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
    "generate_twiml",
]

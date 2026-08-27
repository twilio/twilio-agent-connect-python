"""Voice channel for handling voice-based conversations."""

from tac.channels.voice.channel import (
    AmdHandler,
    CallStatusHandler,
    InboundCallTwiMLHandler,
    RecordingHandler,
    VoiceChannel,
)
from tac.channels.voice.config import ConversationRelayProviderConfig, VoiceChannelConfig
from tac.channels.voice.conversation_relay import ConversationRelayProvider
from tac.channels.voice.media_streams.openai_realtime import (
    TWILIO_MEDIA_STREAM_AUDIO_FORMAT,
    OpenAIRealtimeProvider,
    OpenAIRealtimeProviderConfig,
    VoiceTwiMLOptionsMediaStreams,
)
from tac.channels.voice.provider import VoiceProvider, VoiceProviderConfig
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
    "OpenAIRealtimeProvider",
    "OpenAIRealtimeProviderConfig",
    "RecordingEvent",
    "RecordingHandler",
    "TWILIO_MEDIA_STREAM_AUDIO_FORMAT",
    "TwiMLOptions",
    "TwiMLRequest",
    "VoiceChannel",
    "VoiceChannelConfig",
    "VoiceProvider",
    "VoiceProviderConfig",
    "VoiceTwiMLOptions",
    "VoiceTwiMLOptionsConversationRelay",
    "VoiceTwiMLOptionsMediaStreams",
    "generate_twiml",
]

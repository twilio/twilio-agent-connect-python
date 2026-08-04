"""Voice channel for handling voice-based conversations."""

from tac.channels.voice.channel import VoiceChannel
from tac.channels.voice.config import (
    AmdHandler,
    CallStatusHandler,
    InboundCallTwiMLHandler,
    RecordingHandler,
    VoiceChannelConfig,
)
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
    "InboundCallTwiMLHandler",
    "InterruptMode",
    "LanguageConfig",
    "RecordingEvent",
    "RecordingHandler",
    "TwiMLOptions",
    "TwiMLRequest",
    "VoiceChannel",
    "VoiceChannelConfig",
    "generate_twiml",
]

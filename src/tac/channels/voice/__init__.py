"""Voice channel for handling voice-based conversations."""

from tac.channels.voice.channel import VoiceChannel
from tac.channels.voice.config import (
    CallEventHandler,
    InboundCallTwiMLHandler,
    VoiceChannelConfig,
)
from tac.channels.voice.twiml import generate_twiml
from tac.models.voice import (
    CallEvent,
    InterruptMode,
    LanguageConfig,
    TwiMLOptions,
    TwiMLRequest,
)

__all__ = [
    "CallEvent",
    "CallEventHandler",
    "InboundCallTwiMLHandler",
    "InterruptMode",
    "LanguageConfig",
    "TwiMLOptions",
    "TwiMLRequest",
    "VoiceChannel",
    "VoiceChannelConfig",
    "generate_twiml",
]

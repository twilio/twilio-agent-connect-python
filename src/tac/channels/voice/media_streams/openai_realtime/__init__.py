"""``OpenAIRealtimeProvider``: bridges Twilio Media Streams to OpenAI's Realtime API."""

from tac.channels.voice.media_streams.openai_realtime.config import OpenAIRealtimeProviderConfig
from tac.channels.voice.media_streams.openai_realtime.provider import (
    TWILIO_AUDIO_FORMAT_FOR_REALTIME,
    OpenAIRealtimeProvider,
)
from tac.channels.voice.media_streams.twiml import generate_twiml
from tac.models.outbound import InitiateVoiceConversationOptionsOpenAIRealtime
from tac.models.stream import StreamStartMessage
from tac.models.voice import VoiceTwiMLOptionsMediaStreams

__all__ = [
    "TWILIO_AUDIO_FORMAT_FOR_REALTIME",
    "InitiateVoiceConversationOptionsOpenAIRealtime",
    "OpenAIRealtimeProvider",
    "OpenAIRealtimeProviderConfig",
    "StreamStartMessage",
    "VoiceTwiMLOptionsMediaStreams",
    "generate_twiml",
]

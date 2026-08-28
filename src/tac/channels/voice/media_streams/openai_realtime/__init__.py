"""``OpenAIRealtimeProvider``: bridges Twilio Media Streams to OpenAI's Realtime API."""

from tac.channels.voice.media_streams.openai_realtime.config import OpenAIRealtimeProviderConfig
from tac.channels.voice.media_streams.openai_realtime.provider import (
    TWILIO_MEDIA_STREAM_AUDIO_FORMAT,
    OpenAIRealtimeProvider,
)
from tac.channels.voice.media_streams.openai_realtime.twiml import (
    VoiceTwiMLOptionsMediaStreams,
    generate_twiml,
)
from tac.models.outbound import InitiateVoiceConversationOptionsOpenAIRealtime
from tac.models.stream import StreamStartMessage

__all__ = [
    "TWILIO_MEDIA_STREAM_AUDIO_FORMAT",
    "InitiateVoiceConversationOptionsOpenAIRealtime",
    "OpenAIRealtimeProvider",
    "OpenAIRealtimeProviderConfig",
    "StreamStartMessage",
    "VoiceTwiMLOptionsMediaStreams",
    "generate_twiml",
]

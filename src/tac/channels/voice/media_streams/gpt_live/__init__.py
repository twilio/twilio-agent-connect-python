"""``GPTLiveProvider``: bridges Twilio Media Streams to OpenAI's GPT-Live alpha.

GPT-Live is an unreleased OpenAI alpha API.
"""

from tac.channels.voice.media_streams.gpt_live.config import GPTLiveProviderConfig
from tac.channels.voice.media_streams.gpt_live.provider import (
    TWILIO_MEDIA_STREAM_AUDIO_FORMAT,
    GPTLiveProvider,
)
from tac.channels.voice.media_streams.twiml import generate_twiml
from tac.models.outbound import InitiateVoiceConversationOptionsGPTLive
from tac.models.stream import StreamStartMessage
from tac.models.voice import VoiceTwiMLOptionsMediaStreams

__all__ = [
    "GPTLiveProvider",
    "GPTLiveProviderConfig",
    "InitiateVoiceConversationOptionsGPTLive",
    "StreamStartMessage",
    "TWILIO_MEDIA_STREAM_AUDIO_FORMAT",
    "VoiceTwiMLOptionsMediaStreams",
    "generate_twiml",
]

"""OpenAI Realtime (Media Streams) voice channel for TAC.

Bridges phone calls to OpenAI's Realtime API by relaying raw audio through
our own server — Twilio's Media Streams WebSocket on one side, OpenAI's
Realtime WebSocket on the other. Unlike
``tac.channels.openai_realtime_sip``, audio does pass through TAC here.
"""

from tac.channels.openai_realtime_media_streams.channel import (
    OpenAIRealtimeMediaStreamsChannel,
)
from tac.channels.openai_realtime_media_streams.config import (
    OpenAIRealtimeMediaStreamsChannelConfig,
)
from tac.channels.openai_realtime_media_streams.twiml import generate_stream_twiml

__all__ = [
    "OpenAIRealtimeMediaStreamsChannel",
    "OpenAIRealtimeMediaStreamsChannelConfig",
    "generate_stream_twiml",
]

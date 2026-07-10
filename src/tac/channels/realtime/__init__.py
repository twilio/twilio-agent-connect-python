"""Realtime voice channel: Twilio Media Streams bridged to a speech-to-speech
model (OpenAI Realtime API).

This is an alternative to the ConversationRelay-based :class:`~tac.channels.voice.VoiceChannel`.
Where ConversationRelay speaks a *text* protocol (Twilio does STT/TTS and TAC's
``on_message_ready`` callback returns text), this channel speaks an *audio*
protocol: Twilio streams raw call audio over ``<Connect><Stream>`` and TAC
forwards it to a speech-to-speech model that does its own STT/TTS and returns
audio.

The two live side by side — pick ConversationRelay for a text LLM you already
have, or this for the lowest-latency, most natural speech-to-speech experience.
"""

from tac.channels.realtime.channel import RealtimeVoiceChannel
from tac.channels.realtime.config import RealtimeVoiceChannelConfig
from tac.channels.realtime.twiml import generate_stream_twiml

__all__ = [
    "RealtimeVoiceChannel",
    "RealtimeVoiceChannelConfig",
    "generate_stream_twiml",
]

"""OpenAI Realtime (SIP) voice channel for TAC.

Bridges phone calls to OpenAI's Realtime API over SIP — Twilio forwards the
call at the SIP level directly to OpenAI, so audio never passes through TAC.
"""

from tac.channels.openai_realtime_sip.channel import OpenAIRealtimeSipChannel
from tac.channels.openai_realtime_sip.client import OpenAIRealtimeSipClient
from tac.channels.openai_realtime_sip.config import (
    CallIncomingHandler,
    OpenAIRealtimeSipChannelConfig,
)
from tac.channels.openai_realtime_sip.models import (
    OpenAIRealtimeSipCallIncoming,
    OpenAIRealtimeSipSessionConfig,
)

__all__ = [
    "CallIncomingHandler",
    "OpenAIRealtimeSipCallIncoming",
    "OpenAIRealtimeSipChannel",
    "OpenAIRealtimeSipChannelConfig",
    "OpenAIRealtimeSipClient",
    "OpenAIRealtimeSipSessionConfig",
]

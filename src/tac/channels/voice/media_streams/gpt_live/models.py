"""Per-call state for ``GPTLiveProvider``, not part of the public API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from tac.channels.voice.media_streams.shared.models import MediaStreamsOpenAICallState


@dataclass
class _CallState(MediaStreamsOpenAICallState):
    """Per-call bookkeeping this provider needs beyond ``ConversationSession``."""

    #: Set once ``session.closed`` arrives, so ``_cleanup_call`` can wait for
    #: graceful finalization before tearing down the socket.
    closed_event: asyncio.Event = field(default_factory=asyncio.Event)

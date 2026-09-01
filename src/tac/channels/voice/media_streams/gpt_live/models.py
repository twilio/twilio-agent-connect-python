"""Per-call state for ``GPTLiveProvider``, not part of the public API."""

from __future__ import annotations

from dataclasses import dataclass

from tac.channels.voice.media_streams.shared.models import MediaStreamsOpenAICallState


@dataclass
class _CallState(MediaStreamsOpenAICallState):
    """Per-call bookkeeping this provider needs beyond ``ConversationSession``.

    No barge-in state, unlike ``OpenAIRealtimeProvider``'s ``_CallState`` —
    GPT-Live is full-duplex and handles interruption itself; there's no
    client-driven truncate/cancel to track state for.
    """

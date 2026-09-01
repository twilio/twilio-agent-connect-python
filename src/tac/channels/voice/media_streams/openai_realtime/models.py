"""Per-call state for ``OpenAIRealtimeProvider``, not part of the public API."""

from __future__ import annotations

from dataclasses import dataclass, field

from tac.channels.voice.media_streams.shared.models import MediaStreamsOpenAICallState


@dataclass
class _BargeInState:
    """Per-call barge-in bookkeeping.

    ``current_item_audio_ms`` is the exact duration of audio actually sent
    to Twilio for ``last_assistant_item``, computed from delta byte counts —
    not a wall-clock estimate. ``conversation.item.truncate`` rejects an
    ``audio_end_ms`` beyond the item's real content, so this must never
    overstate it.

    ``response_active`` tracks whether a response is still being generated
    (set on ``response.created``, cleared on ``response.done`` or once
    barge-in cancels it) — ``response.cancel`` with nothing in flight is
    itself an error event, so this gates whether to send it.
    """

    last_assistant_item: str | None = None
    current_item_audio_ms: int = 0
    muted_item_id: str | None = None
    response_active: bool = False


@dataclass
class _CallState(MediaStreamsOpenAICallState):
    """Per-call bookkeeping this provider needs beyond ``ConversationSession``."""

    barge_in: _BargeInState = field(default_factory=_BargeInState)

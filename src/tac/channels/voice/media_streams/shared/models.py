"""``MediaStreamsOpenAICallState``: per-call state shared by every ``VoiceProvider``
bridging Twilio Media Streams to an OpenAI real-time voice API,
not part of the public API.
"""

from __future__ import annotations

from dataclasses import dataclass

from websockets.asyncio.client import ClientConnection

from tac.channels.websocket_protocol import WebSocketProtocol


@dataclass
class MediaStreamsOpenAICallState:
    """Both legs of one call's audio bridge — the Twilio-facing socket and the
    model-facing socket — live here together, rather than in two parallel
    dicts keyed by conversation id that could drift out of sync.

    ``stream_sid`` and ``transcript`` live on ``ConversationSession.metadata``
    instead, not here — this dict is popped before ``on_conversation_ended``
    fires, so anything a handler needs to read after the call ends must
    survive on the session, not in here.
    """

    twilio_ws: WebSocketProtocol | None = None
    model_ws: ClientConnection | None = None

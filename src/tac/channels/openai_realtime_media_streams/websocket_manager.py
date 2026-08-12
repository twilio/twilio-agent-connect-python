"""WebSocket connection management for the OpenAI Realtime (Media Streams) channel.

Unlike ``VoiceChannel`` (one WebSocket per conversation, tracked by the shared
``tac.channels.websocket_manager.WebSocketManager``), this channel bridges TWO
independent sockets for the same conversation id: Twilio's Media Stream
connection and the outbound connection to OpenAI's Realtime API. The two
never exist for unrelated conversations — they're always a pair scoped to one
call — so they're stored together as one entry per conversation id (a socket
pair, one or both legs possibly still unset during connection setup/teardown)
rather than as two parallel dicts that could drift out of sync.
"""

from dataclasses import dataclass
from typing import Any

from tac.channels.websocket_protocol import WebSocketProtocol


@dataclass
class _CallSockets:
    """Both legs of one call's audio bridge. Either may briefly be None —
    the Twilio socket exists before the OpenAI connection is established,
    and both are cleared (independently) during teardown.
    """

    twilio: WebSocketProtocol | None = None
    model: Any = None


class RealtimeWebSocketManager:
    """Tracks, per conversation, the Twilio-facing socket and the OpenAI
    Realtime model socket as one paired entry — pure connection routing, no
    call-specific data (that lives on the conversation's
    ``ConversationSession.metadata`` instead).
    """

    def __init__(self) -> None:
        self._calls: dict[str, _CallSockets] = {}

    def _get_or_create(self, conversation_id: str) -> _CallSockets:
        return self._calls.setdefault(conversation_id, _CallSockets())

    # -- Twilio-facing socket -------------------------------------------

    def add_twilio_socket(self, conversation_id: str, websocket: WebSocketProtocol) -> None:
        self._get_or_create(conversation_id).twilio = websocket

    def get_twilio_socket(self, conversation_id: str) -> WebSocketProtocol | None:
        sockets = self._calls.get(conversation_id)
        return sockets.twilio if sockets else None

    # -- OpenAI Realtime model socket -------------------------------------

    def add_model_socket(self, conversation_id: str, model_ws: Any) -> None:
        self._get_or_create(conversation_id).model = model_ws

    def get_model_socket(self, conversation_id: str) -> Any:
        sockets = self._calls.get(conversation_id)
        return sockets.model if sockets else None

    # -- Teardown ----------------------------------------------------------

    async def pop_sockets(self, conversation_id: str) -> None:
        """Remove both legs for one call and close the model socket (the
        Twilio socket is owned by the ASGI layer, never closed here).
        """
        sockets = self._calls.pop(conversation_id, None)
        if sockets is not None and sockets.model is not None:
            await sockets.model.close()

    def __len__(self) -> int:
        """Number of calls with at least one active socket."""
        return len(self._calls)

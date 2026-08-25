"""``VoiceProvider``: the Provider interface that lets ``VoiceChannel`` host
more than one kind of real-time media transport.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from tac.channels.websocket_protocol import WebSocketProtocol
from tac.models.memory import MemoryMode
from tac.models.outbound import InitiateVoiceConversationOptions, InitiateVoiceConversationResult
from tac.models.voice import TwiMLOptions, TwiMLRequest

if TYPE_CHECKING:
    from tac.channels.voice.channel import VoiceChannel


class VoiceProvider(ABC):
    """Provider interface for a ``VoiceChannel``'s real-time media transport.

    ``ConversationRelayProvider`` is the only implementation today.
    """

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Channel name identifier, e.g. ``"VOICE"`` or ``"VOICE_GPT_LIVE"``.

        Returned by ``VoiceChannel.get_channel_name()`` and stamped onto every
        ``ConversationSession`` this provider creates.
        """

    @property
    def memory_mode(self) -> MemoryMode:
        """Memory retrieval mode for this provider. Default: ``"never"``.

        ``ConversationRelayProvider`` overrides this to read from its config.
        """
        return "never"

    @abstractmethod
    def build_twiml(self, websocket_url: str, options: TwiMLOptions) -> str:
        """Build the ``<Connect>`` TwiML that points Twilio at this provider's
        WebSocket endpoint.
        """

    async def handle_incoming_call(
        self,
        channel: VoiceChannel,
        twiml_request: TwiMLRequest | None,
        host_twiml_options: TwiMLOptions | None,
    ) -> str:
        """Build the TwiML string for an inbound call.

        Default: minimal — no customizer, no per-channel defaults, just
        ``host_twiml_options`` (if any) over a channel-derived WebSocket URL.
        ``ConversationRelayProvider`` overrides this with a fuller merge
        (customizer, ``default_twiml_options``, TAC defaults).
        """
        websocket_url = (
            host_twiml_options.websocket_url
            if host_twiml_options is not None and host_twiml_options.websocket_url is not None
            else channel._resolve_websocket_url("handle_incoming_call")
        )
        return self.build_twiml(websocket_url, host_twiml_options or TwiMLOptions())

    async def initiate_outbound_conversation(
        self,
        channel: VoiceChannel,
        options: InitiateVoiceConversationOptions,
    ) -> InitiateVoiceConversationResult:
        """Place an outbound call. Default: not supported."""
        raise NotImplementedError(f"{type(self).__name__} does not support outbound calls.")

    def on_inbound_call_twiml(
        self, callback: Callable[[TwiMLRequest], Awaitable[TwiMLOptions]]
    ) -> None:
        """Register a per-call inbound TwiML customizer.

        Default no-op — only ``ConversationRelayProvider`` supports this.
        """
        return None

    @abstractmethod
    async def handle_websocket(self, channel: VoiceChannel, websocket: WebSocketProtocol) -> None:
        """Drive one WebSocket connection from accept to disconnect."""

    @abstractmethod
    async def send_response(
        self,
        channel: VoiceChannel,
        conversation_id: str,
        response: str | AsyncGenerator[str | dict[str, Any], None],
        role: str | None = None,
    ) -> None:
        """Send a text response back through this provider's transport, if supported."""

    async def handle_conversation_relay_callback(
        self,
        channel: VoiceChannel,
        payload_dict: dict[str, str],
    ) -> None:
        """Handle the ConversationRelay callback webhook, if this provider uses one.

        Default no-op — only ``ConversationRelayProvider`` receives this webhook.
        """
        return None

    def get_websocket(self, conversation_id: str) -> WebSocketProtocol | None:
        """Return the Twilio-facing WebSocket for a conversation, if tracked."""
        return None

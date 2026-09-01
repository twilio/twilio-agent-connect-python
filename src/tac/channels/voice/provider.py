"""``VoiceProvider``: the interface that lets ``VoiceChannel`` host more than
one kind of real-time media provider.

``ConversationRelayProvider`` is the only implementation today.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from tac.channels.websocket_protocol import WebSocketProtocol
from tac.core.config import CallEventKind, TACConfig
from tac.core.logging import get_logger
from tac.models.memory import MemoryMode
from tac.models.outbound import InitiateVoiceConversationOptions, InitiateVoiceConversationResult
from tac.models.voice import TwiMLRequest, VoiceTwiMLOptions

if TYPE_CHECKING:
    from tac.channels.voice.channel import VoiceChannel


class VoiceProvider:
    """Base class for a ``VoiceChannel``'s real-time media provider.

    Holds the owning ``channel`` (Calls API lifecycle, conversation
    bookkeeping, ``TAC``).
    """

    def __init__(self, channel: VoiceChannel) -> None:
        self.channel = channel
        self.logger = get_logger(self.__class__.__module__)

    @property
    def channel_name(self) -> str:
        """Channel name identifier, e.g. ``"VOICE"``.

        Returned by ``VoiceChannel.get_channel_name()`` and stamped onto every
        ``ConversationSession`` this provider creates.
        """
        return "VOICE"

    @property
    def _conversation_closed_by_orchestrator(self) -> bool:
        """Whether a Conversation Orchestrator ``CLOSED`` webhook will end this
        provider's conversations.

        Decides who owns ``on_conversation_ended``. ``False`` (the default):
        no CO conversation exists, so ``VoiceChannel`` fires it itself at
        teardown. ``True``: only ``on_call_ended`` fires at teardown and the
        CLOSED webhook — which may reach another instance — does the rest.
        """
        return False

    async def handle_incoming_call(
        self,
        twiml_request: TwiMLRequest | None = None,
        *,
        host_twiml_options: VoiceTwiMLOptions | None = None,
    ) -> str:
        """Build the response for an inbound call. Default: not supported."""
        raise NotImplementedError(f"{type(self).__name__} does not support inbound calls.")

    async def handle_twilio_provider_callback(
        self,
        payload_dict: dict[str, str],
    ) -> None:
        """Handle this provider's own out-of-band lifecycle webhook, if it has one.

        Not every provider has an equivalent — Twilio's ConversationRelay posts to
        ``<Connect action=...>`` when the session ends (``ConversationRelayProvider``
        uses this as a WebSocket-disconnect backup); Media Streams instead has its
        own independent ``statusCallback`` (``stream-started``/``stream-stopped``/
        ``stream-error``), which is purely informational and doesn't gate call flow.
        Default no-op for providers with nothing to do here.
        """
        return None

    async def handle_websocket(self, websocket: WebSocketProtocol) -> None:
        """Drive one WebSocket connection from accept to disconnect."""
        raise NotImplementedError(f"{type(self).__name__} does not support WebSocket connections.")

    async def initiate_outbound_conversation(
        self,
        options: InitiateVoiceConversationOptions,
    ) -> InitiateVoiceConversationResult:
        """Place an outbound call. Default: not supported."""
        raise NotImplementedError(f"{type(self).__name__} does not support outbound calls.")

    async def send_response(
        self,
        conversation_id: str,
        response: str | AsyncGenerator[str | dict[str, Any], None],
        role: str | None = None,
    ) -> None:
        """Send a text response back through this provider's transport, if supported."""
        raise NotImplementedError(f"{type(self).__name__} does not support send_response.")

    def get_websocket(self, conversation_id: str) -> WebSocketProtocol | None:
        """Return the Twilio-facing WebSocket for a conversation, if tracked."""
        return None

    def _apply_call_event_callbacks(self, call_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Set callback URLs on ``call_kwargs`` for every registered call-event handler.

        A URL is derived only when its handler is registered — an unwanted
        call-event URL would otherwise surface as silent 11200 alerts for a
        feature nobody asked for. If TAC isn't serving these routes, set the
        URLs explicitly via ``CallOptions`` (or
        ``ConversationRelayProviderConfig.default_call_options``, where available).
        """
        wiring: list[tuple[CallEventKind, str, Callable[..., Any] | None]] = [
            ("status", "status_callback", self.channel._on_call_status),
            ("amd", "async_amd_status_callback", self.channel._on_amd),
            ("recording", "recording_status_callback", self.channel._on_recording),
        ]
        for kind, param, handler in wiring:
            if handler is None:
                continue
            url = self.channel.tac.config.call_event_url(kind)
            if url is not None:
                call_kwargs.setdefault(param, url)
        return call_kwargs


class VoiceProviderConfig(BaseModel):
    """Base configuration for a ``VoiceChannel``'s real-time media provider.

    ``ConversationRelayProviderConfig`` is the only subclass today. Lives
    here (not in ``config.py``) so a future provider's config only needs to
    import this module, not anything ConversationRelay-specific.
    """

    memory_mode: MemoryMode = Field(
        default="never", description="Memory retrieval mode for this channel"
    )

    def create_provider(self, channel: VoiceChannel, tac_config: TACConfig) -> VoiceProvider:
        """Build the ``VoiceProvider`` this config configures.

        Args:
            channel: The owning ``VoiceChannel``.
            tac_config: ``TACConfig`` — providers that talk TwiML need it to
                derive default URLs (``voice_public_domain`` etc.).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement create_provider() to be usable "
            "as a VoiceChannel config."
        )

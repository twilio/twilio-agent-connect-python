"""``VoiceProvider``: the interface that lets ``VoiceChannel`` host more than
one kind of real-time media provider.

Empty for now — a placeholder for the follow-up PR that moves
``VoiceChannel``'s ConversationRelay-specific logic (TwiML, WebSocket
protocol handling, outbound calls) behind this interface.
``ConversationRelayProvider`` is the only implementation today.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from tac.core.config import TACConfig
from tac.core.logging import get_logger

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


class VoiceProviderConfig(BaseModel):
    """Base configuration for a ``VoiceChannel``'s real-time media provider.

    ``ConversationRelayProviderConfig`` is the only subclass today. Lives
    here (not in ``config.py``) so a future provider's config only needs to
    import this module, not anything ConversationRelay-specific.
    """

    def create_provider(self, channel: VoiceChannel, tac_config: TACConfig) -> VoiceProvider:
        """Build the ``VoiceProvider`` this config configures.

        Args:
            channel: The owning ``VoiceChannel`` — stored on the provider so
                it can reach the generic functionality the channel still owns.
            tac_config: ``TACConfig`` — providers that talk TwiML need it to
                derive default URLs (``voice_public_domain`` etc.).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement create_provider() to be usable "
            "as a VoiceChannel config."
        )

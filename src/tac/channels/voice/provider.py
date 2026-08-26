"""``VoiceProvider``: the interface that lets ``VoiceChannel`` host more than
one kind of real-time media provider.

Empty for now — a placeholder for the follow-up PR that moves
``VoiceChannel``'s ConversationRelay-specific logic (TwiML, WebSocket
protocol handling, outbound calls) behind this interface.
``ConversationRelayProvider`` is the only implementation today.
"""

from pydantic import BaseModel


class VoiceProvider:
    """Base class for a ``VoiceChannel``'s real-time media provider."""


class VoiceProviderConfig(BaseModel):
    """Base configuration for a ``VoiceChannel``'s real-time media provider.

    ``ConversationRelayProviderConfig`` is the only subclass today. Lives
    here (not in ``config.py``) so a future provider's config only needs to
    import this module, not anything ConversationRelay-specific.
    """

    def create_provider(self) -> VoiceProvider:
        """Build the ``VoiceProvider`` this config configures."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement create_provider() to be usable "
            "as a VoiceChannel config."
        )

"""``ConversationRelayProvider``: the default ``VoiceProvider`` — Twilio
ConversationRelay's managed setup/prompt/interrupt loop over one WebSocket.
"""

from typing import TYPE_CHECKING, Any

from tac._deprecation import resolve_deprecated_alias
from tac.channels.voice.conversation_relay.config import ConversationRelayProviderConfig
from tac.channels.voice.conversation_relay.provider import ConversationRelayProvider
from tac.channels.voice.conversation_relay.twiml import generate_twiml

__all__ = [
    "ConversationRelayProvider",
    "ConversationRelayProviderConfig",
    "generate_twiml",
]

if TYPE_CHECKING:  # static type only, see tac._deprecation
    VoiceChannelConfig = ConversationRelayProviderConfig


def __getattr__(name: str) -> Any:
    # See tac._deprecation. TODO(3.0): remove.
    if name == "VoiceChannelConfig":
        return resolve_deprecated_alias("VoiceChannelConfig", ConversationRelayProviderConfig)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

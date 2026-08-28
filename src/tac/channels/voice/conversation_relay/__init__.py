"""``ConversationRelayProvider``: the default ``VoiceProvider`` — Twilio
ConversationRelay's managed setup/prompt/interrupt loop over one WebSocket.
"""

from typing import Any

from tac._deprecation import resolve_deprecated_alias
from tac.channels.voice.conversation_relay.config import ConversationRelayProviderConfig
from tac.channels.voice.conversation_relay.provider import ConversationRelayProvider
from tac.channels.voice.conversation_relay.twiml import generate_twiml

__all__ = [
    "ConversationRelayProvider",
    "ConversationRelayProviderConfig",
    "VoiceChannelConfig",
    "generate_twiml",
]


def __getattr__(name: str) -> Any:
    # Resolved directly (not forwarded to config.py's own __getattr__) so the
    # warning attributes correctly to this access, not to config.py's caller.
    # See tac._deprecation.resolve_deprecated_alias. TODO(3.0): remove.
    if name == "VoiceChannelConfig":
        return resolve_deprecated_alias("VoiceChannelConfig", ConversationRelayProviderConfig)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

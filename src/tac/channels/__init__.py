"""Communication channels for the Twilio Agent Connect."""

from typing import Any

from tac._deprecation import resolve_deprecated_alias
from tac.channels.base import BaseChannel
from tac.channels.chat import ChatChannel, ChatChannelConfig
from tac.channels.messaging import MessagingChannel, MessagingChannelConfig
from tac.channels.rcs import RCSChannel, RCSChannelConfig
from tac.channels.sms import SMSChannel, SMSChannelConfig
from tac.channels.voice import VoiceChannel
from tac.channels.voice.conversation_relay.config import ConversationRelayProviderConfig
from tac.channels.whatsapp import WhatsAppChannel, WhatsAppChannelConfig

__all__ = [
    "BaseChannel",
    "ChatChannel",
    "ChatChannelConfig",
    "RCSChannel",
    "RCSChannelConfig",
    "SMSChannel",
    "SMSChannelConfig",
    "MessagingChannel",
    "MessagingChannelConfig",
    "VoiceChannel",
    "VoiceChannelConfig",
    "WhatsAppChannel",
    "WhatsAppChannelConfig",
]


def __getattr__(name: str) -> Any:
    # Resolved directly (not forwarded to another module's __getattr__) so the
    # warning attributes correctly to this access. See
    # tac._deprecation.resolve_deprecated_alias. TODO(3.0): remove.
    if name == "VoiceChannelConfig":
        return resolve_deprecated_alias("VoiceChannelConfig", ConversationRelayProviderConfig)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

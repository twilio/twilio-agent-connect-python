"""Communication channels for the Twilio Agent Connect."""

from typing import TYPE_CHECKING, Any

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
    "WhatsAppChannel",
    "WhatsAppChannelConfig",
]

if TYPE_CHECKING:  # static type only, see tac._deprecation
    VoiceChannelConfig = ConversationRelayProviderConfig


def __getattr__(name: str) -> Any:
    # See tac._deprecation. TODO(3.0): remove.
    if name == "VoiceChannelConfig":
        return resolve_deprecated_alias("VoiceChannelConfig", ConversationRelayProviderConfig)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

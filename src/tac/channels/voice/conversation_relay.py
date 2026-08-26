"""``ConversationRelayProvider``: the default ``VoiceProvider``.

Empty for now — a placeholder. ``VoiceChannel`` still owns all
ConversationRelay-specific logic directly; a follow-up PR moves it here
(TwiML building via ``twiml.TwiMLBuilderConversationRelay``, WebSocket
protocol handling, outbound calls).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tac.channels.voice.provider import VoiceProvider
from tac.core.config import TACConfig

if TYPE_CHECKING:
    from tac.channels.voice.config import ConversationRelayProviderConfig


class ConversationRelayProvider(VoiceProvider):
    """Twilio ConversationRelay: Twilio handles ASR/TTS and exchanges JSON
    ``setup``/``prompt``/``interrupt`` messages over one WebSocket.

    This is the default provider ``VoiceChannel`` builds when none is passed
    explicitly.
    """

    def __init__(self, tac_config: TACConfig, config: ConversationRelayProviderConfig) -> None:
        self.config = config

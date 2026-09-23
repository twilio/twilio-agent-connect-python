"""WhatsApp Channel implementation for TAC."""

from typing import Any

from pydantic import Field

from tac import TAC
from tac.channels.messaging import MessagingChannel, MessagingChannelConfig
from tac.models.conversation import ParticipantAddress
from tac.models.outbound import (
    InitiateConversationResult,
    InitiateMessagingConversationOptions,
)


class WhatsAppChannelConfig(MessagingChannelConfig):
    """Configuration for WhatsApp channel.

    Inherits dedup_capacity and memory_mode from MessagingChannelConfig.
    """

    dedup_capacity: int = Field(
        default=10000,
        gt=0,
        description="Maximum number of idempotency tokens to track for deduplication",
    )


class WhatsAppChannel(MessagingChannel):
    """WhatsApp Channel for handling WhatsApp-based conversations.

    Inherits shared messaging channel webhook processing from MessagingChannel
    and provides WhatsApp-specific message sending and filtering.

    WhatsApp uses WhatsApp sender phone numbers configured in TACConfig
    (via TWILIO_WHATSAPP_NUMBER). Address format: whatsapp:+1234567890
    """

    def __init__(
        self,
        tac: TAC,
        config: WhatsAppChannelConfig | dict[str, Any] | None = None,
    ):
        if isinstance(config, dict):
            config = WhatsAppChannelConfig(**config)
        elif config is None:
            config = WhatsAppChannelConfig()

        super().__init__(
            tac,
            dedup_capacity=config.dedup_capacity,
            memory_mode=config.memory_mode,
        )

        if not tac.config.whatsapp_numbers:
            raise ValueError(
                "whatsapp_number(s) is required for WhatsApp channel. "
                "Set TWILIO_WHATSAPP_NUMBER / TWILIO_WHATSAPP_NUMBERS or "
                "provide whatsapp_number / whatsapp_numbers in TACConfig."
            )

    def get_channel_name(self) -> str:
        return "WHATSAPP"

    def is_default_agent_address(self, author_address: str) -> bool:
        """Check if the author address is one of the configured WhatsApp numbers."""
        return author_address in self.tac.config.whatsapp_numbers

    def get_agent_address(self, conversation_id: str) -> ParticipantAddress:
        """Get the agent's default participant address for this conversation."""
        if self.tac.config.whatsapp_number is None:
            raise RuntimeError("whatsapp_number is required for WhatsApp channel.")
        return ParticipantAddress(channel="WHATSAPP", address=self.tac.config.whatsapp_number)

    async def initiate_outbound_conversation(
        self,
        options: InitiateMessagingConversationOptions,
    ) -> InitiateConversationResult:
        """Initiate an outbound WhatsApp conversation.

        Creates a conversation via Conversation Orchestrator with inline
        participants, then sends the initial message via the Actions API.
        Uses `options.from_` when provided (must be one of the configured
        WhatsApp numbers), otherwise falls back to the default `whatsapp_number`
        from TACConfig. If an active conversation with the same addresses
        already exists (group-by dedup), CO returns 409 and the existing
        conversation is reused.

        Args:
            options: Conversation initiation options (to address and message)

        Returns:
            InitiateConversationResult with conversation_id and session

        Raises:
            ValueError: If `options.from_` is set but is not one of the configured
                WhatsApp numbers.
        """
        return await self._initiate_messaging_conversation(
            options=options,
            from_address=self._resolve_outbound_from(
                options.from_,
                allowlist=self.tac.config.whatsapp_numbers,
                default=self.tac.config.whatsapp_number,
            ),
            customer_address_kwargs={},
            agent_address_kwargs={},
        )

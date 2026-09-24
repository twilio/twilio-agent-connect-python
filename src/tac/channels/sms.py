"""SMS Channel implementation for TAC."""

from typing import Any

from pydantic import Field

from tac import TAC
from tac.channels.messaging import MessagingChannel, MessagingChannelConfig
from tac.models.conversation import ParticipantAddress
from tac.models.outbound import (
    InitiateConversationResult,
    InitiateMessagingConversationOptions,
)


class SMSChannelConfig(MessagingChannelConfig):
    """Configuration for SMS channel.

    Inherits dedup_capacity and memory_mode from MessagingChannelConfig.
    """

    dedup_capacity: int = Field(
        default=10000,
        gt=0,
        description="Maximum number of idempotency tokens to track for deduplication",
    )


class SMSChannel(MessagingChannel):
    """SMS Channel for handling SMS-based conversations.

    Inherits shared messaging channel webhook processing from MessagingChannel
    and provides SMS-specific message sending and filtering.
    """

    def __init__(
        self,
        tac: TAC,
        config: SMSChannelConfig | dict[str, Any] | None = None,
    ):
        if isinstance(config, dict):
            config = SMSChannelConfig(**config)
        elif config is None:
            config = SMSChannelConfig()

        super().__init__(
            tac,
            dedup_capacity=config.dedup_capacity,
            memory_mode=config.memory_mode,
        )

        if not tac.config.phone_numbers:
            raise ValueError(
                "phone_number(s) is required for SMS channel. "
                "Set TWILIO_PHONE_NUMBER / TWILIO_PHONE_NUMBERS or "
                "provide phone_number / phone_numbers in TACConfig."
            )

    def get_channel_name(self) -> str:
        return "SMS"

    def is_default_agent_address(self, author_address: str) -> bool:
        return author_address in self.tac.config.phone_numbers

    def get_agent_address(self, conversation_id: str) -> ParticipantAddress:
        if self.tac.config.phone_number is None:
            raise RuntimeError("phone_number is required for SMS channel.")
        return ParticipantAddress(channel="SMS", address=self.tac.config.phone_number)

    async def initiate_outbound_conversation(
        self,
        options: InitiateMessagingConversationOptions,
    ) -> InitiateConversationResult:
        """Initiate an outbound SMS conversation.

        Creates a conversation via Conversation Orchestrator with inline
        participants, then sends the initial message via the Actions API.
        If an active conversation with the same addresses already exists
        (group-by dedup), CO returns 409 and the existing conversation is reused.
        """
        return await self._initiate_messaging_conversation(
            options=options,
            from_address=self._resolve_outbound_from(
                options.from_,
                allowlist=self.tac.config.phone_numbers,
                default=self.tac.config.phone_number,
            ),
            customer_address_kwargs={},
            agent_address_kwargs={},
        )

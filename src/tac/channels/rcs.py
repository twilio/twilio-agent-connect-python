"""RCS Channel implementation for TAC."""

from typing import Any

from pydantic import Field

from tac import TAC
from tac.channels.messaging import MessagingChannel, MessagingChannelConfig
from tac.models.conversation import ParticipantAddress
from tac.models.outbound import (
    InitiateConversationResult,
    InitiateMessagingConversationOptions,
)


class RCSChannelConfig(MessagingChannelConfig):
    """Configuration for RCS channel.

    Inherits dedup_capacity and memory_mode from MessagingChannelConfig.
    """

    dedup_capacity: int = Field(
        default=10000,
        gt=0,
        description="Maximum number of idempotency tokens to track for deduplication",
    )


class RCSChannel(MessagingChannel):
    """RCS Channel for handling RCS-based conversations.

    Inherits shared messaging channel webhook processing from MessagingChannel
    and provides RCS-specific message sending and filtering.

    RCS uses RCS Sender IDs configured in TACConfig (via TWILIO_RCS_SENDER_ID).
    """

    def __init__(
        self,
        tac: TAC,
        config: RCSChannelConfig | dict[str, Any] | None = None,
    ):
        if isinstance(config, dict):
            config = RCSChannelConfig(**config)
        elif config is None:
            config = RCSChannelConfig()

        super().__init__(
            tac,
            dedup_capacity=config.dedup_capacity,
            memory_mode=config.memory_mode,
        )

        if not tac.config.rcs_sender_id:
            raise ValueError(
                "rcs_sender_id is required for RCS channel. "
                "Please set TWILIO_RCS_SENDER_ID environment variable or "
                "provide rcs_sender_id in TACConfig."
            )

    def get_channel_name(self) -> str:
        return "RCS"

    def is_default_agent_address(self, author_address: str) -> bool:
        """Check if the author address matches the configured RCS sender ID."""
        if not self.tac.config.rcs_sender_id:
            raise RuntimeError("rcs_sender_id is required for RCS channel.")
        return author_address == self.tac.config.rcs_sender_id

    def get_agent_address(self, conversation_id: str) -> ParticipantAddress:
        """Get the agent's participant address for this conversation."""
        if not self.tac.config.rcs_sender_id:
            raise RuntimeError("rcs_sender_id is required for RCS channel.")
        return ParticipantAddress(channel="RCS", address=self.tac.config.rcs_sender_id)

    async def initiate_outbound_conversation(
        self,
        options: InitiateMessagingConversationOptions,
    ) -> InitiateConversationResult:
        """Initiate an outbound RCS conversation.

        Creates a conversation via Conversation Orchestrator with inline
        participants, then sends the initial message via the Actions API.
        Uses the RCS sender ID from TACConfig as the from address.
        If an active conversation with the same addresses already exists
        (group-by dedup), CO returns 409 and the existing conversation is reused.

        Args:
            options: Conversation initiation options (to address and message)

        Returns:
            InitiateConversationResult with conversation_id and session

        Raises:
            RuntimeError: If rcs_sender_id is not configured
        """
        if not self.tac.config.rcs_sender_id:
            raise RuntimeError("rcs_sender_id is required for RCS channel.")

        return await self._initiate_messaging_conversation(
            options=options,
            from_address=self.tac.config.rcs_sender_id,
            customer_address_kwargs={},
            agent_address_kwargs={},
        )

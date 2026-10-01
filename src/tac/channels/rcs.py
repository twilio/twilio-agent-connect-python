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

        if not tac.config.rcs_sender_ids:
            raise ValueError(
                "rcs_sender_id(s) is required for RCS channel. "
                "Set TWILIO_RCS_SENDER_ID / TWILIO_RCS_SENDER_IDS or "
                "provide rcs_sender_id / rcs_sender_ids in TACConfig."
            )

    def get_channel_name(self) -> str:
        return "RCS"

    def is_default_agent_address(self, author_address: str) -> bool:
        """Check if the author address is one of the configured RCS senders."""
        return author_address in self.tac.config.rcs_sender_ids

    def get_agent_address(self, conversation_id: str) -> ParticipantAddress:
        """Get the agent's default participant address for this conversation."""
        if self.tac.config.rcs_sender_id is None:
            raise RuntimeError("rcs_sender_id is required for RCS channel.")
        return ParticipantAddress(channel="RCS", address=self.tac.config.rcs_sender_id)

    async def initiate_outbound_conversation(
        self,
        options: InitiateMessagingConversationOptions,
    ) -> InitiateConversationResult:
        """Initiate an outbound RCS conversation.

        Creates a conversation via Conversation Orchestrator with inline
        participants, then sends the initial message via the Actions API.
        Uses `options.from_` when provided (must be one of the configured RCS
        senders), otherwise falls back to the default `rcs_sender_id` from
        TACConfig. If an active conversation with the same addresses already
        exists (group-by dedup), CO returns 409 and the existing conversation
        is reused.

        Args:
            options: Conversation initiation options (to address and message)

        Returns:
            InitiateConversationResult with conversation_id and session

        Raises:
            ValueError: If `options.from_` is set but is not one of the configured
                RCS senders.
        """
        return await self._initiate_messaging_conversation(
            options=options,
            from_address=self._resolve_outbound_from(
                options.from_,
                allowlist=self.tac.config.rcs_sender_ids,
                default=self.tac.config.rcs_sender_id,
            ),
            customer_address_kwargs={},
            agent_address_kwargs={},
        )

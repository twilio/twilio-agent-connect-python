"""RCS Channel implementation for TAC."""

from collections.abc import AsyncGenerator
from typing import Any

from pydantic import Field

from tac import TAC
from tac.channels.messaging import MessagingChannel, MessagingChannelConfig
from tac.models.conversation import (
    ActionChannelSettings,
    ActionParticipantRef,
    ActionTextContent,
    ParticipantAddress,
    SendMessageActionPayload,
    SendMessageActionRequest,
)
from tac.models.outbound import (
    InitiateConversationResult,
    InitiateMessagingConversationOptions,
)
from tac.utils.redaction import mask_address


class RCSChannelConfig(MessagingChannelConfig):
    """Configuration for RCS channel.

    Inherits dedup_capacity and auto_retrieve_memory from MessagingChannelConfig.
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
            auto_retrieve_memory=config.auto_retrieve_memory,
        )

        if not tac.config.rcs_sender_id:
            raise ValueError(
                "rcs_sender_id is required for RCS channel. "
                "Please set TWILIO_RCS_SENDER_ID environment variable or "
                "provide rcs_sender_id in TACConfig."
            )

    def get_channel_name(self) -> str:
        return "rcs"

    def get_channel_type_upper(self) -> str:
        return "RCS"

    def is_default_agent_address(self, author_address: str) -> bool:
        return author_address == self.tac.config.rcs_sender_id

    def get_agent_address(self, conversation_id: str) -> ParticipantAddress:
        return ParticipantAddress(channel="RCS", address=self.tac.config.rcs_sender_id)

    async def send_response(
        self,
        conversation_id: str,
        response: str | AsyncGenerator[str | dict[str, Any], None],
        role: str | None = None,
    ) -> None:
        """Send RCS response using the Conversation Orchestrator Send API.

        Args:
            conversation_id: Conversation ID to send response to
            response: Message content (must be string for RCS)
            role: Optional message role (not used in RCS channel)

        Raises:
            TypeError: If response is not a string
        """
        if not isinstance(response, str):
            raise TypeError("RCS channel only supports string responses")

        try:
            participants = await self.tac.conversation_orchestrator_client.list_participants(
                conversation_id
            )
        except Exception as e:
            self.logger.error(
                "Failed to list participants",
                conversation_id=conversation_id,
                error=str(e),
            )
            return

        # Use from_address from session metadata (set during outbound initiation),
        # falling back to the configured rcs_sender_id for inbound conversations
        session = self._conversations.get(conversation_id)
        agent_address = self.tac.config.rcs_sender_id
        if session:
            from_addr = session.metadata.get("from_address")
            if isinstance(from_addr, str):
                agent_address = from_addr

        # Find the CUSTOMER participant by address on the RCS channel
        customer_participant = None
        customer_address = None
        for participant in participants:
            if participant.type == "CUSTOMER":
                for address in participant.addresses:
                    if address.channel == "RCS":
                        customer_participant = participant
                        customer_address = address.address
                        break
                if customer_participant:
                    break

        agent_participant = await self._ensure_agent_participant(
            conversation_id,
            existing_participants=participants,
            agent_address=ParticipantAddress(channel="RCS", address=agent_address),
        )
        if not agent_participant:
            raise RuntimeError(
                f"Failed to resolve AI_AGENT participant for conversation {conversation_id}"
            )

        if not customer_participant or not customer_address:
            raise RuntimeError(
                "Customer participant with RCS address not found for conversation "
                f"{conversation_id}"
            )

        channel_id = session.metadata.get("channel_id") if session else None
        channel_settings = (
            ActionChannelSettings(channel_id=channel_id)
            if isinstance(channel_id, str) and channel_id
            else None
        )

        try:
            action_request = SendMessageActionRequest(
                payload=SendMessageActionPayload(
                    from_=ActionParticipantRef(
                        channel="RCS",
                        participant_id=agent_participant.id,
                    ),
                    to=[
                        ActionParticipantRef(
                            channel="RCS",
                            participant_id=customer_participant.id,
                        )
                    ],
                    content=ActionTextContent(text=response),
                    channel_settings=channel_settings,
                ),
            )

            await self.tac.conversation_orchestrator_client.create_action(
                conversation_id, action_request
            )

            self.logger.info(
                "Sent RCS response via Actions API",
                conversation_id=conversation_id,
                to_address=mask_address(customer_address),
            )
        except Exception as e:
            self.logger.error(
                "Failed to create action",
                conversation_id=conversation_id,
                error=str(e),
                exc_info=True,
            )

    async def initiate_outbound_conversation(
        self,
        options: InitiateMessagingConversationOptions,
    ) -> InitiateConversationResult:
        """Initiate an outbound RCS conversation.

        Creates a conversation via Conversation Orchestrator with inline
        participants, then sends the initial message via the Actions API.
        If an active conversation with the same addresses already exists
        (group-by dedup), CO returns 409 and the existing conversation is reused.
        """
        from_address = options.from_ or self.tac.config.rcs_sender_id
        return await self._initiate_messaging_conversation(
            options=options,
            from_address=from_address,
            customer_address_kwargs={},
            agent_address_kwargs={},
        )

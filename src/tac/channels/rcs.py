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
        """Check if the author address matches the configured RCS sender ID."""
        if not self.tac.config.rcs_sender_id:
            raise RuntimeError("rcs_sender_id is required for RCS channel.")
        return author_address == self.tac.config.rcs_sender_id

    def get_agent_address(self, conversation_id: str) -> ParticipantAddress:
        """Get the agent's participant address for this conversation."""
        if not self.tac.config.rcs_sender_id:
            raise RuntimeError("rcs_sender_id is required for RCS channel.")
        return ParticipantAddress(channel="RCS", address=self.tac.config.rcs_sender_id)

    async def send_response(
        self,
        conversation_id: str,
        response: str | AsyncGenerator[str | dict[str, Any], None],
        role: str | None = None,
    ) -> None:
        """Send RCS response using the Conversation Orchestrator Send API.

        Reads the agent and customer participant ids stashed on the session
        by inbound reconciliation or outbound initiation. Missing ids are a
        misuse — send_response is only expected to be called after an inbound
        webhook (COMMUNICATION_CREATED → reconcile) or after
        `initiate_outbound_conversation`, both of which populate the session.

        Args:
            conversation_id: Conversation ID to send response to
            response: Message content (must be string for RCS)
            role: Optional message role (not used in RCS channel)

        Raises:
            TypeError: If response is not a string
            RuntimeError: If the session or participant ids are missing
        """
        if not isinstance(response, str):
            raise TypeError("RCS channel only supports string responses")

        session = self._conversations.get(conversation_id)
        if session is None or not session.author_info or not session.ai_agent_info:
            raise RuntimeError(
                f"Unable to send RCS: send_response called without a reconciled "
                f"session for conversation {conversation_id}. Wait for an inbound "
                "webhook or call initiate_outbound_conversation first."
            )

        customer_participant_id = session.author_info.participant_id
        agent_participant_id = session.ai_agent_info.participant_id
        if not customer_participant_id or not agent_participant_id:
            raise RuntimeError(
                f"Unable to send RCS: session for conversation {conversation_id} is "
                "missing participant ids."
            )

        channel_id = session.metadata.get("channel_id")
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
                        participant_id=agent_participant_id,
                    ),
                    to=[
                        ActionParticipantRef(
                            channel="RCS",
                            participant_id=customer_participant_id,
                        )
                    ],
                    content=ActionTextContent(text=response),
                    channel_settings=channel_settings,
                ),
            )

            await self.conversation_orchestrator_client.create_action(
                conversation_id, action_request
            )

            self.logger.info(
                "Sent RCS response via Actions API",
                conversation_id=conversation_id,
                to_address=mask_address(session.author_info.address),
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

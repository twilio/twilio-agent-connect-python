"""SMS Channel implementation for TAC."""

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
from tac.utils.redaction import mask_phone


class SMSChannelConfig(MessagingChannelConfig):
    """Configuration for SMS channel.

    Inherits dedup_capacity and auto_retrieve_memory from MessagingChannelConfig.
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
            auto_retrieve_memory=config.auto_retrieve_memory,
        )

        if not tac.config.phone_number:
            raise ValueError(
                "phone_number is required for SMS channel. "
                "Please set TWILIO_PHONE_NUMBER environment variable or "
                "provide phone_number in TACConfig."
            )

    def get_channel_name(self) -> str:
        return "sms"

    def get_channel_type_upper(self) -> str:
        return "SMS"

    def is_default_agent_address(self, author_address: str) -> bool:
        return author_address == self.tac.config.phone_number

    def get_agent_address(self, conversation_id: str) -> ParticipantAddress:
        return ParticipantAddress(channel="SMS", address=self.tac.config.phone_number)

    async def send_response(
        self,
        conversation_id: str,
        response: str | AsyncGenerator[str | dict[str, Any], None],
        role: str | None = None,
    ) -> None:
        """Send SMS response using the Conversation Orchestrator Send API.

        Reads the agent and customer participant ids stashed on the session
        by inbound reconciliation or outbound initiation. Missing ids are a
        misuse — send_response is only expected to be called after an inbound
        webhook (COMMUNICATION_CREATED → reconcile) or after
        `initiate_outbound_conversation`, both of which populate the session.

        Args:
            conversation_id: Conversation ID to send response to
            response: Message content (must be string for SMS)
            role: Optional message role (not used in SMS channel)

        Raises:
            TypeError: If response is not a string
            RuntimeError: If the session or participant ids are missing
        """
        if not isinstance(response, str):
            raise TypeError("SMS channel only supports string responses")

        session = self._conversations.get(conversation_id)
        if session is None or not session.author_info or not session.ai_agent_info:
            raise RuntimeError(
                f"Unable to send SMS: send_response called without a reconciled "
                f"session for conversation {conversation_id}. Wait for an inbound "
                "webhook or call initiate_outbound_conversation first."
            )

        customer_participant_id = session.author_info.participant_id
        agent_participant_id = session.ai_agent_info.participant_id
        if not customer_participant_id or not agent_participant_id:
            raise RuntimeError(
                f"Unable to send SMS: session for conversation {conversation_id} is "
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
                        channel="SMS",
                        participant_id=agent_participant_id,
                    ),
                    to=[
                        ActionParticipantRef(
                            channel="SMS",
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
                "Sent SMS response via Actions API",
                conversation_id=conversation_id,
                to_address=mask_phone(session.author_info.address),
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
        """Initiate an outbound SMS conversation.

        Creates a conversation via Conversation Orchestrator with inline
        participants, then sends the initial message via the Actions API.
        If an active conversation with the same addresses already exists
        (group-by dedup), CO returns 409 and the existing conversation is reused.
        """
        return await self._initiate_messaging_conversation(
            options=options,
            from_address=self.tac.config.phone_number,
            customer_address_kwargs={},
            agent_address_kwargs={},
        )

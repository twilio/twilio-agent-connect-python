"""SMS Channel implementation for TAC."""

from collections.abc import AsyncGenerator
from typing import Any

from pydantic import Field

from tac import TAC
from tac.channels.messaging import (
    SESSION_META_AGENT_PARTICIPANT_ID,
    SESSION_META_CUSTOMER_PARTICIPANT_ID,
    MessagingChannel,
    MessagingChannelConfig,
)
from tac.models.conversation import (
    ActionChannelSettings,
    ActionParticipantRef,
    ActionTextContent,
    ParticipantAddress,
    SendMessageActionPayload,
    SendMessageActionRequest,
)


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

    def is_own_message(self, author_address: str) -> bool:
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

        Reads participant ids resolved by `_reconcile_participants` from the
        session; requires an inbound webhook to have populated them.

        Args:
            conversation_id: Conversation ID to send response to
            response: Message content (must be string for SMS)
            role: Optional message role (not used in SMS channel)

        Raises:
            TypeError: If response is not a string
            RuntimeError: If participant reconciliation has not run for this conversation
        """
        if not isinstance(response, str):
            raise TypeError("SMS channel only supports string responses")

        session = self._conversations.get(conversation_id)
        if not session:
            raise RuntimeError(
                f"No active session for conversation {conversation_id}; "
                "send_response requires a prior inbound webhook."
            )

        agent_id = session.metadata.get(SESSION_META_AGENT_PARTICIPANT_ID)
        customer_id = session.metadata.get(SESSION_META_CUSTOMER_PARTICIPANT_ID)
        if not isinstance(agent_id, str) or not isinstance(customer_id, str):
            raise RuntimeError(
                f"Participant ids not resolved for conversation {conversation_id}; "
                "reconciliation must run via an inbound webhook before send_response."
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
                    from_=ActionParticipantRef(channel="SMS", participant_id=agent_id),
                    to=[ActionParticipantRef(channel="SMS", participant_id=customer_id)],
                    content=ActionTextContent(text=response),
                    channel_settings=channel_settings,
                ),
            )

            await self.tac.conversation_orchestrator_client.create_action(
                conversation_id, action_request
            )

            self.logger.info(
                "Sent SMS response via Actions API",
                conversation_id=conversation_id,
            )
        except Exception as e:
            self.logger.error(
                "Failed to create action",
                conversation_id=conversation_id,
                error=str(e),
                exc_info=True,
            )

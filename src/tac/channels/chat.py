"""Chat Channel implementation for TAC."""

from collections.abc import AsyncGenerator
from typing import Any

from pydantic import Field

from tac import TAC
from tac.channels.messaging import (
    SESSION_META_AGENT_PARTICIPANT_ID,
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


class ChatChannelConfig(MessagingChannelConfig):
    """Configuration for Chat channel.

    Attributes:
        agent_address: Chat agent identity string used to identify the bot's messages.
    """

    agent_address: str = Field(
        default="ai-assistant",
        description="Chat agent identity string for bot message filtering",
    )


class ChatChannel(MessagingChannel):
    """Chat Channel for handling web chat conversations.

    Uses identity-based addressing instead of phone numbers. Customer-side
    resolution is author-driven (from the inbound communication), so
    reconciliation only promotes the agent participant.
    """

    reconcile_customer_type = False

    def __init__(
        self,
        tac: TAC,
        config: ChatChannelConfig | dict[str, Any] | None = None,
    ):
        if isinstance(config, dict):
            config = ChatChannelConfig(**config)
        elif config is None:
            config = ChatChannelConfig()

        super().__init__(
            tac,
            dedup_capacity=config.dedup_capacity,
            auto_retrieve_memory=config.auto_retrieve_memory,
        )
        self.agent_address = config.agent_address

    def get_channel_name(self) -> str:
        return "chat"

    def get_channel_type_upper(self) -> str:
        return "CHAT"

    def is_own_message(self, author_address: str) -> bool:
        return author_address == self.agent_address

    def get_agent_address(self, conversation_id: str) -> ParticipantAddress:
        session = self._conversations.get(conversation_id)
        channel_id = session.metadata.get("channel_id") if session else None
        return ParticipantAddress(
            channel="CHAT",
            address=self.agent_address,
            channel_id=channel_id if isinstance(channel_id, str) else None,
        )

    async def send_response(
        self,
        conversation_id: str,
        response: str | AsyncGenerator[str | dict[str, Any], None],
        role: str | None = None,
    ) -> None:
        """Send chat response using the Conversation Orchestrator Send API.

        Reads the agent participant id resolved by `_reconcile_participants`
        from the session; the customer (recipient) remains author-driven from
        the inbound communication's `author_info`.

        Args:
            conversation_id: Conversation ID to send response to
            response: Message content (must be string for Chat)
            role: Optional message role (not used in Chat channel)

        Raises:
            TypeError: If response is not a string
            RuntimeError: If no inbound webhook has populated the session state
        """
        if not isinstance(response, str):
            raise TypeError("Chat channel only supports string responses")

        session = self._conversations.get(conversation_id)
        if not session:
            self.logger.error(
                "No active session found",
                conversation_id=conversation_id,
            )
            return

        if not session.author_info:
            self.logger.error(
                "No author info found - no inbound message received yet",
                conversation_id=conversation_id,
            )
            return

        # channelId (Chat Channel SID) is required for CHAT delivery — the V1
        # Chat backend uses it to pick the destination thread. Inbound webhooks
        # always populate it, so a missing value here is a misuse.
        chat_channel_sid = session.metadata.get("channel_id")
        if not chat_channel_sid or not isinstance(chat_channel_sid, str):
            raise RuntimeError(
                "Missing required session.metadata['channel_id'] for chat send_response; "
                "this is normally populated by an inbound webhook. Ensure an inbound "
                "message has been processed before calling send_response, or set "
                "session.metadata['channel_id'] explicitly in advanced usage."
            )

        agent_id = session.metadata.get(SESSION_META_AGENT_PARTICIPANT_ID)
        if not isinstance(agent_id, str):
            raise RuntimeError(
                f"Agent participant id not resolved for conversation {conversation_id}; "
                "reconciliation must run via an inbound webhook before send_response."
            )

        # TODO(maestro): Drop `chat_service` here once the Actions API resolves the
        # V1 Chat service SID server-side. Maestro team confirmed this should not be
        # required client-side; keep the workaround until the server-side fix ships.
        # `channel_id` stays — it's a permanent per-conversation requirement.
        chat_service_sid = self.tac.conversations_v1_service_sid
        channel_settings = ActionChannelSettings(
            channel_id=chat_channel_sid,
            chat_service=chat_service_sid,
        )

        try:
            action_request = SendMessageActionRequest(
                payload=SendMessageActionPayload(
                    from_=ActionParticipantRef(channel="CHAT", participant_id=agent_id),
                    to=[
                        ActionParticipantRef(
                            channel="CHAT",
                            participant_id=session.author_info.participant_id,
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
                "Sent chat response via Actions API",
                conversation_id=conversation_id,
                channel_id=chat_channel_sid,
            )
        except Exception as e:
            self.logger.error(
                "Failed to create action",
                conversation_id=conversation_id,
                error=str(e),
                exc_info=True,
            )

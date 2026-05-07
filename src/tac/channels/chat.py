"""Chat Channel implementation for TAC."""

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
    InitiateChatConversationOptions,
    InitiateConversationResult,
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

    Uses identity-based addressing instead of phone numbers.
    Automatically creates AI_AGENT participant if needed (lazy creation)
    and manages conversation lifecycle through Conversation Orchestrator webhooks.
    """

    # Chat identifies the customer author-driven from the webhook's
    # `author.participant_id`; promoting some other channel-matching UNKNOWN
    # CHAT participant could pick the wrong recipient.
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
            memory_mode=config.memory_mode,
        )
        self.agent_address = config.agent_address

    def get_channel_name(self) -> str:
        return "chat"

    def get_channel_type_upper(self) -> str:
        return "CHAT"

    def is_default_agent_address(self, author_address: str) -> bool:
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

        Reads the agent and customer participant ids stashed on the session
        by inbound reconciliation or outbound initiation. Missing ids are a
        misuse — send_response is only expected to be called after an inbound
        webhook (COMMUNICATION_CREATED → reconcile) or after
        `initiate_outbound_conversation`, both of which populate the session.

        Args:
            conversation_id: Conversation ID to send response to
            response: Message content (must be string for Chat)
            role: Optional message role (not used in Chat channel)

        Raises:
            TypeError: If response is not a string
            RuntimeError: If the session, channel_id, or participant ids are missing
        """
        if not isinstance(response, str):
            raise TypeError("Chat channel only supports string responses")

        session = self._conversations.get(conversation_id)
        if session is None or not session.author_info or not session.ai_agent_info:
            raise RuntimeError(
                f"Unable to send chat message: send_response called without a "
                f"reconciled session for conversation {conversation_id}. Wait for "
                "an inbound webhook or call initiate_outbound_conversation first."
            )

        customer_participant_id = session.author_info.participant_id
        agent_participant_id = session.ai_agent_info.participant_id
        if not customer_participant_id or not agent_participant_id:
            raise RuntimeError(
                f"Unable to send chat message: session for conversation "
                f"{conversation_id} is missing participant ids."
            )

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

        channel_settings = ActionChannelSettings(channel_id=chat_channel_sid)

        try:
            action_request = SendMessageActionRequest(
                payload=SendMessageActionPayload(
                    from_=ActionParticipantRef(
                        channel="CHAT",
                        participant_id=agent_participant_id,
                    ),
                    to=[
                        ActionParticipantRef(
                            channel="CHAT",
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

    async def initiate_outbound_conversation(
        self,
        options: InitiateChatConversationOptions,
    ) -> InitiateConversationResult:
        """Initiate an outbound Chat conversation.

        Creates a conversation via Conversation Orchestrator with inline
        participants, then sends the initial message via the Actions API.
        If an active conversation with the same addresses already exists
        (group-by dedup), CO returns 409 and the existing conversation is reused.
        """
        return await self._initiate_messaging_conversation(
            options=options,
            from_address=self.agent_address,
            customer_address_kwargs={"channel_id": options.channel_id},
            agent_address_kwargs={"channel_id": options.channel_id},
            extra_metadata={"channel_id": options.channel_id},
            channel_settings=ActionChannelSettings(channel_id=options.channel_id),
        )

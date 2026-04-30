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

    def is_default_agent_address(self, author_address: str) -> bool:
        return author_address == self.agent_address

    async def send_response(
        self,
        conversation_id: str,
        response: str | AsyncGenerator[str | dict[str, Any], None],
        role: str | None = None,
    ) -> None:
        """Send chat response using the Conversation Orchestrator Send API.

        Lazily creates an AI_AGENT participant if one doesn't exist yet.

        Args:
            conversation_id: Conversation ID to send response to
            response: Message content (must be string for Chat)
            role: Optional message role (not used in Chat channel)

        Raises:
            TypeError: If response is not a string
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

        try:
            participants = await self.conversation_orchestrator_client.list_participants(
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
        # falling back to the configured agent_address for inbound conversations
        from_addr = session.metadata.get("from_address")
        agent_addr = from_addr if isinstance(from_addr, str) else self.agent_address

        agent_participant = await self._ensure_agent_participant(
            conversation_id,
            existing_participants=participants,
            agent_address=ParticipantAddress(
                channel="CHAT",
                address=agent_addr,
                channel_id=chat_channel_sid,
            ),
        )
        if not agent_participant:
            raise RuntimeError(
                f"Failed to resolve AI_AGENT participant for conversation {conversation_id}"
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
                    from_=ActionParticipantRef(
                        channel="CHAT",
                        participant_id=agent_participant.id,
                    ),
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
        from_address = options.from_ or self.agent_address
        chat_service_sid = self.tac.conversations_v1_service_sid
        if not chat_service_sid:
            raise RuntimeError(
                "conversations_v1_service_sid is not set — the Conversation Orchestrator "
                "configuration has no Conversations V1 bridge. Chat outbound requires it."
            )

        return await self._initiate_messaging_conversation(
            options=options,
            from_address=from_address,
            customer_address_kwargs={"channel_id": options.channel_id},
            agent_address_kwargs={"channel_id": options.channel_id},
            extra_metadata={"channel_id": options.channel_id},
            channel_settings=ActionChannelSettings(
                channel_id=options.channel_id,
                chat_service=chat_service_sid,
            ),
        )

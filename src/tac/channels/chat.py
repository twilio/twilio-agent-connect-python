"""Chat Channel implementation for TAC."""

from typing import Any

from pydantic import Field

from tac import TAC
from tac.channels.messaging import MessagingChannel, MessagingChannelConfig
from tac.models.conversation import ActionChannelSettings, ParticipantAddress
from tac.models.outbound import (
    InitiateChatConversationOptions,
    InitiateConversationResult,
)
from tac.models.session import ConversationSession


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

    def _build_channel_settings(
        self, conversation_id: str, session: ConversationSession
    ) -> ActionChannelSettings:
        # channelId (Chat Channel SID) is required for CHAT delivery — the V1
        # Chat backend uses it to pick the destination thread. Inbound webhooks
        # always populate it, so a missing value here is a misuse.
        channel_id = session.metadata.get("channel_id")
        if not channel_id or not isinstance(channel_id, str):
            raise RuntimeError(
                "Missing required session.metadata['channel_id'] for chat send_response; "
                "this is normally populated by an inbound webhook. Ensure an inbound "
                "message has been processed before calling send_response, or set "
                "session.metadata['channel_id'] explicitly in advanced usage."
            )
        return ActionChannelSettings(channel_id=channel_id)

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

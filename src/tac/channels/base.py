"""Base channel interface for TAC channels."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import AsyncGenerator
from typing import Any

from tac import TAC
from tac.core.logging import get_logger
from tac.models.memory import MemoryMode
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse


class BaseChannel(ABC):
    """
    Abstract base class for TAC channels.

    Channels handle protocol-specific webhook processing and response delivery
    for different communication channels (SMS, Voice, etc.).

    This class provides common conversation lifecycle management that is shared
    across all channel types.
    """

    def __init__(
        self,
        tac: TAC,
        memory_mode: MemoryMode = "never",
        dedup_capacity: int = 10000,
    ):
        """
        Initialize base channel.

        Args:
            tac: TAC instance for memory/context operations
            memory_mode: Memory retrieval mode. Default is "never".
                Set to "always" to retrieve memory for every message.
            dedup_capacity: Maximum number of idempotency tokens to track for
                webhook deduplication. Default 10000. Must be positive.
        """
        if dedup_capacity <= 0:
            raise ValueError(f"dedup_capacity must be positive, got {dedup_capacity}")

        self.tac = tac
        self.logger = get_logger(self.__class__.__module__)
        self.memory_mode = memory_mode

        # Track active conversations (shared across all channel types)
        self._conversations: dict[str, ConversationSession] = {}

        # Webhook deduplication
        self._processed_tokens: OrderedDict[str, bool] = OrderedDict()
        self._max_tracked_tokens = dedup_capacity

    @abstractmethod
    async def process_webhook(
        self, webhook_data: dict[str, Any], idempotency_token: str | None = None
    ) -> None:
        """
        Process incoming webhook event from Twilio.

        This method should:
        1. Parse and validate webhook data
        2. Handle conversation lifecycle (start, message, end)
        3. Trigger memory retrieval via TAC
        4. Invoke registered callbacks

        Args:
            webhook_data: Raw webhook event data from Twilio
            idempotency_token: Optional Twilio idempotency token from request headers
        """
        pass

    @abstractmethod
    async def send_response(
        self,
        conversation_id: str,
        response: str | AsyncGenerator[str | dict[str, Any], None],
        role: str | None = None,
    ) -> None:
        """
        Send response back through the channel.

        Supports both simple string responses and streaming via async generators.

        Args:
            conversation_id: Conversation ID to send response to
            response: Message content (string) or async generator for streaming
            role: Optional message role (e.g., 'assistant', 'user', 'system')
        """
        pass

    @abstractmethod
    def get_channel_name(self) -> str:
        """
        Get the channel name identifier.

        Returns:
            Channel name (e.g., 'sms', 'voice')
        """
        # TODO: Parse Channel Type based on webhook data
        pass

    def get_channel_type_upper(self) -> str:
        """
        Get uppercase channel type for webhook filtering.

        Returns:
            Uppercase channel type (e.g., 'SMS', 'VOICE')
        """
        return self.get_channel_name().upper()

    def _is_duplicate_webhook(self, idempotency_token: str) -> bool:
        """Check if a webhook has already been processed using Twilio's idempotency token.

        Uses a sliding window approach with fixed capacity to track tokens.

        Args:
            idempotency_token: Twilio's i-twilio-idempotency-token header value

        Returns:
            True if the webhook has already been processed
        """
        if idempotency_token in self._processed_tokens:
            return True

        if len(self._processed_tokens) >= self._max_tracked_tokens:
            self._processed_tokens.popitem(last=False)

        self._processed_tokens[idempotency_token] = True
        return False

    def _is_event_for_this_channel(self, webhook_data: dict[str, Any]) -> bool:
        """Self-filtering: check if webhook event belongs to this channel.

        For COMMUNICATION_CREATED: require author.channel matches this channel type.
        For CONVERSATION_UPDATED: only process if conversation is tracked locally.
        Other events pass through.
        """
        event_type = webhook_data.get("eventType")
        event_data = webhook_data.get("data")

        if event_type == "COMMUNICATION_CREATED":
            if not isinstance(event_data, dict):
                return False
            author = event_data.get("author")
            if not isinstance(author, dict):
                return False
            author_channel = author.get("channel")
            if not author_channel:
                return False
            return bool(author_channel == self.get_channel_type_upper())

        if event_type == "CONVERSATION_UPDATED":
            if not isinstance(event_data, dict):
                return False
            conv_id = event_data.get("id")
            if conv_id and conv_id not in self._conversations:
                return False

        return True

    def _start_conversation(
        self,
        conv_id: str,
        profile_id: str | None = None,
    ) -> ConversationSession:
        """
        Initialize new conversation session with optional profile_id.

        Profile data is fetched lazily during retrieve_memory() when needed.

        Args:
            conv_id: Conversation ID
            profile_id: Profile ID for the conversation (optional)

        Returns:
            The new or existing ConversationSession.
        """
        if conv_id in self._conversations:
            self.logger.debug(
                "Conversation already exists, skipping initialization",
                conversation_id=conv_id,
                channel=self.get_channel_name(),
            )
            return self._conversations[conv_id]

        # Store conversation session
        self._conversations[conv_id] = ConversationSession(
            conversation_id=conv_id,
            profile_id=profile_id,
            channel=self.get_channel_name(),
        )

        self.logger.info(
            f"CONVERSATION | Started {self.get_channel_name().upper()} conversation",
            conversation_id=conv_id,
            profile_id=profile_id,
        )
        return self._conversations[conv_id]

    async def _end_conversation(self, conv_id: str) -> None:
        """
        Clean up conversation session.

        Pops the session from the conversation dict, then triggers the
        on_conversation_ended callback with the removed session data.

        Args:
            conv_id: Conversation ID
        """
        session = self._conversations.pop(conv_id, None)
        if session is not None:
            try:
                await self.tac.trigger_conversation_ended(session)
            except Exception as e:
                self.logger.error(
                    "Error in conversation ended callback",
                    conversation_id=conv_id,
                    error=str(e),
                    exc_info=True,
                )

            self.logger.debug(
                "Ended conversation",
                conversation_id=conv_id,
                channel=self.get_channel_name(),
            )

    async def _retrieve_memory_if_enabled(
        self, session: ConversationSession, query: str | None, conv_id: str
    ) -> TACMemoryResponse | None:
        """
        Retrieve memory only when ``self.memory_mode == "always"``.

        This method handles the common logic for memory retrieval across all channels,
        including error handling and debug logging. If ``memory_mode`` is any value
        other than ``"always"``, automatic memory retrieval is skipped.

        Args:
            session: Conversation session containing profile_id and context
            query: Optional query string for memory retrieval
            conv_id: Conversation ID for logging

        Returns:
            TACMemoryResponse wrapper if memory was retrieved, None otherwise
        """
        memory_response = None
        if self.memory_mode == "always":
            try:
                memory_response = await self.tac.retrieve_memory(session, query=query)
                self.logger.debug(
                    "Memory retrieved",
                    conversation_id=conv_id,
                )
            except Exception as e:
                self.logger.error(
                    "Failed to retrieve memory",
                    conversation_id=conv_id,
                    error=str(e),
                    exc_info=True,
                )
                # Continue without memory rather than failing the entire message processing
        else:
            self.logger.debug(
                "Memory mode not set to 'always', skipping memory retrieval",
                conversation_id=conv_id,
                memory_mode=self.memory_mode,
            )
        return memory_response

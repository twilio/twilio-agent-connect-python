"""Base channel interface for TAC channels."""

import asyncio
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import AsyncGenerator
from typing import Any

from tac import TAC
from tac.core.logging import get_logger
from tac.models.conversation import ParticipantResponse
from tac.models.memory import MemoryMode
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse

# Participant types that represent TAC itself at TAC's (channel, address).
# `AI_AGENT` is the canonical type; `AGENT` is the legacy Conversation
# Orchestrator form. A participant typed either way at TAC's address is
# recognized as TAC and not overwritten; anything else (HUMAN_AGENT,
# CUSTOMER, …) is someone else's assignment.
AGENT_TYPES: frozenset[str] = frozenset({"AGENT", "AI_AGENT"})


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
                - "always": Retrieve memory for every message with the query string
                - "once": Retrieve memory once at conversation start with empty query and cache it.
                         Cache is invalidated when conversation becomes INACTIVE.
                - "never": Skip memory retrieval
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
            Channel name (e.g., 'SMS', 'VOICE')
        """
        # TODO: Parse Channel Type based on webhook data
        pass

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
            return bool(author_channel == self.get_channel_name())

        if event_type == "CONVERSATION_UPDATED":
            if not isinstance(event_data, dict):
                return False
            conv_id = event_data.get("id")
            if conv_id and conv_id not in self._conversations:
                return False

        return True

    @staticmethod
    def _owns_address(
        participant: ParticipantResponse,
        channel: str,
        address: str,
        extra: dict[str, str | None] | None = None,
    ) -> bool:
        """Whether a participant holds the given (channel, address).

        This is the address-only predicate — it does NOT consider participant
        type. Use it directly when the type is decided separately (e.g.
        reconciliation, which must detect an UNKNOWN at TAC's address before
        promoting it); use `_find_agent_participant` when you want the agent.

        `extra` matches additional ParticipantAddress fields (e.g. messaging
        sub-addressing); only truthy values are compared.
        """
        return any(
            a.channel == channel
            and a.address == address
            and (not extra or all(getattr(a, k) == v for k, v in extra.items() if v))
            for a in participant.addresses
        )

    @classmethod
    def _find_agent_participant(
        cls,
        participants: list[ParticipantResponse],
        channel: str,
        address: str,
        extra: dict[str, str | None] | None = None,
    ) -> ParticipantResponse | None:
        """Find the participant representing TAC's agent in a conversation.

        The agent is the participant that owns TAC's (channel, address) AND has
        an agent type (`AGENT` or `AI_AGENT`). A HUMAN_AGENT or other type at
        that address is someone else and is NOT returned.
        """
        return next(
            (
                p
                for p in participants
                if p.type in AGENT_TYPES and cls._owns_address(p, channel, address, extra)
            ),
            None,
        )

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
            f"CONVERSATION | Started {self.get_channel_name()} conversation",
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
        Retrieve memory based on memory_mode setting.

        Memory modes:
        - "always": Fetch with query on every message
        - "once": Fetch once with empty query, cache it. Cache invalidated on INACTIVE.
                  Uses session.cache_lock for task-safe concurrency within the event loop.
        - "never": Skip retrieval

        Args:
            session: Conversation session containing profile_id and context
            query: Optional query string for memory retrieval (ignored in "once" mode)
            conv_id: Conversation ID for logging

        Returns:
            TACMemoryResponse wrapper if memory was retrieved, None otherwise
        """
        memory_response = None

        if self.memory_mode == "always":
            try:
                memory_response = await self.tac.retrieve_memory(
                    session, query=query, conversation_id=session.conversation_id
                )
                self.logger.debug(
                    "Memory retrieved",
                    conversation_id=conv_id,
                )
            except asyncio.CancelledError:
                # Re-raise to allow proper cancellation (e.g., Voice channel interrupts)
                raise
            except Exception as e:
                self.logger.error(
                    "Failed to retrieve memory",
                    conversation_id=conv_id,
                    error=str(e),
                    exc_info=True,
                )
                # Continue without memory rather than failing the entire message processing
        elif self.memory_mode == "once":
            # Use lock to prevent race conditions between cache read/write and webhook invalidation
            async with session.cache_lock:
                # Check if memory is already cached
                if session.cached_memory is not None:
                    self.logger.debug(
                        "Using cached memory",
                        conversation_id=conv_id,
                    )
                    memory_response = session.cached_memory
                else:
                    # First retrieval - use empty query and cache result. No
                    # per-turn topic here, so leave conversation_id unset too
                    # (avoids an expensive server-side query-expansion step).
                    try:
                        memory_response = await self.tac.retrieve_memory(session, query=None)
                        session.cached_memory = memory_response
                        self.logger.debug(
                            "Memory retrieved and cached",
                            conversation_id=conv_id,
                        )
                    except asyncio.CancelledError:
                        # Re-raise to allow proper cancellation (e.g., Voice channel interrupts)
                        raise
                    except Exception as e:
                        self.logger.error(
                            "Failed to retrieve memory",
                            conversation_id=conv_id,
                            error=str(e),
                            exc_info=True,
                        )
                        # Continue without memory rather than failing the entire message processing
        else:
            # Handles "never" mode and any unexpected values
            self.logger.debug(
                "Memory retrieval disabled",
                conversation_id=conv_id,
                memory_mode=self.memory_mode,
            )
        return memory_response

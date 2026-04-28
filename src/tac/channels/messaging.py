"""MessagingChannel base class for messaging channels (SMS, Chat)."""

from abc import abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field

from tac import TAC
from tac.channels.base import BaseChannel
from tac.models.conversation import (
    ActionChannelSettings,
    ActionParticipantRef,
    ActionTextContent,
    Communication,
    ConversationResponse,
    ParticipantAddress,
    ParticipantRequest,
    ParticipantResponse,
    SendMessageActionPayload,
    SendMessageActionRequest,
)
from tac.models.outbound import InitiateConversationResult, InitiateMessagingConversationOptions
from tac.models.session import AuthorInfo
from tac.utils.redaction import mask_address


class MessagingChannelConfig(BaseModel):
    """Base configuration for messaging channels (SMS, Chat).

    Attributes:
        dedup_capacity: Maximum number of idempotency tokens to track.
            Default 10000 is suitable for most applications.
            Uses Twilio's i-twilio-idempotency-token header for deduplication.
        auto_retrieve_memory: If True, automatically retrieve memory
            before invoking the on_message_ready callback.
    """

    dedup_capacity: int = Field(
        default=10000,
        gt=0,
        description="Maximum number of idempotency tokens to track for deduplication",
    )
    auto_retrieve_memory: bool = Field(
        default=False,
        description="Automatically retrieve memory before on_message_ready callback",
    )


class MessagingChannel(BaseChannel):
    """Abstract base class for messaging channels (SMS, Chat).

    Provides shared webhook processing logic for channels that use
    Conversation Orchestrator webhooks with PARTICIPANT_ADDED,
    COMMUNICATION_CREATED, and CONVERSATION_UPDATED event types.

    Subclasses must implement:
    - is_default_agent_address(): Fast-path check for the channel's default agent address
    - get_channel_type_upper(): Return uppercase channel type ("SMS", "CHAT")
    - send_response(): Send messages back through the channel
    - get_channel_name(): Return lowercase channel name ("sms", "chat")
    """

    def __init__(
        self,
        tac: TAC,
        dedup_capacity: int = 10000,
        auto_retrieve_memory: bool = False,
    ):
        super().__init__(
            tac, auto_retrieve_memory=auto_retrieve_memory, dedup_capacity=dedup_capacity
        )

    @abstractmethod
    def is_default_agent_address(self, author_address: str) -> bool:
        """Fast-path check: is the author address this channel's default agent address?

        For example, config.phone_number for SMS, agent_address for Chat.

        Args:
            author_address: The address of the message author

        Returns:
            True if the address matches the channel's default agent address
        """
        pass

    async def _is_own_message(
        self,
        author_address: str,
        conversation_id: str,
        author_participant_id: str | None,
    ) -> bool:
        """Check if a message is from the bot itself (3-tier).

        1. Default agent address (stateless, no API call)
        2. Session metadata from_address (same-process, for custom from)
        3. API fallback via listParticipants (cross-process / multi-worker)
        """
        if self.is_default_agent_address(author_address):
            return True

        session = self._conversations.get(conversation_id)
        from_address = session.metadata.get("from_address") if session else None
        if from_address == author_address:
            return True

        # If this process knows the outbound sender (from_address is set) and it
        # didn't match, this is a customer message — skip the API call.
        if session and from_address:
            return False

        if author_participant_id:
            try:
                participants = await self.tac.conversation_orchestrator_client.list_participants(
                    conversation_id
                )
                author_p = next((p for p in participants if p.id == author_participant_id), None)
                if author_p:
                    if author_p.type is None:
                        self.logger.warning(
                            "Participant type is undefined",
                            conversation_id=conversation_id,
                            participant_id=author_participant_id,
                        )
                    if author_p.type in ("AI_AGENT", "HUMAN_AGENT", "AGENT"):
                        return True
            except Exception as e:
                self.logger.warning(
                    "Failed to look up participant type for self-message check",
                    conversation_id=conversation_id,
                    participant_id=author_participant_id,
                    error=str(e),
                )

        return False

    @abstractmethod
    def get_channel_type_upper(self) -> str:
        """Return the uppercase channel type for webhook filtering.

        Returns:
            Channel type string (e.g., "SMS", "CHAT")
        """
        pass

    @abstractmethod
    async def send_response(
        self,
        conversation_id: str,
        response: str | AsyncGenerator[str | dict[str, Any], None],
        role: str | None = None,
    ) -> None:
        pass

    async def process_webhook(
        self, webhook_data: dict[str, Any], idempotency_token: str | None = None
    ) -> None:
        """Process messaging channel webhook event and manage conversation lifecycle.

        Handles:
        - PARTICIPANT_ADDED: Initialize conversation and track profile_id
        - COMMUNICATION_CREATED: Process incoming messages from customers
        - CONVERSATION_UPDATED: Clean up when conversation is closed

        Note: Conversation tracking uses instance-local memory. In multi-instance
        deployments, webhooks may route to a different instance, preventing cleanup.
        See CLAUDE.md for horizontal scaling considerations.

        Args:
            webhook_data: Raw webhook event data from Twilio
            idempotency_token: Optional Twilio idempotency token from request headers
        """
        if not self._is_event_for_this_channel(webhook_data):
            return

        if idempotency_token:
            if self._is_duplicate_webhook(idempotency_token):
                return

        event_type = webhook_data.get("eventType")
        event_data = webhook_data.get("data")

        if not isinstance(event_data, dict):
            self.logger.warning(
                "Webhook missing or malformed data field, skipping",
                event_type=event_type,
            )
            return

        if event_type == "PARTICIPANT_ADDED":
            self._handle_participant_added(event_data)
        elif event_type == "COMMUNICATION_CREATED":
            await self._handle_communication_created(event_data)
        elif event_type == "CONVERSATION_UPDATED":
            await self._handle_conversation_updated(event_data)

    def _handle_participant_added(self, event_data: Any) -> None:
        """Handle PARTICIPANT_ADDED event.

        Only processes CUSTOMER participants with addresses matching this channel type.
        """
        participant_data = ParticipantResponse.model_validate(event_data)
        conv_id = participant_data.conversation_id
        profile_id = participant_data.profile_id
        participant_type = participant_data.type

        if participant_type != "CUSTOMER":
            return

        has_matching_address = any(
            address.channel == self.get_channel_type_upper()
            for address in participant_data.addresses
        )

        if not has_matching_address:
            return

        if conv_id not in self._conversations:
            self._start_conversation(conv_id, profile_id)

        if profile_id:
            session = self._conversations[conv_id]
            session.profile_id = profile_id

        self.logger.debug(
            "Customer participant added",
            conversation_id=conv_id,
            profile_id=profile_id,
        )

    async def _handle_communication_created(self, event_data: Any) -> None:
        """Handle COMMUNICATION_CREATED event (incoming message)."""
        communication_data = Communication.model_validate(event_data)
        conv_id = communication_data.conversation_id
        message_text = communication_data.content.text

        if not message_text or not message_text.strip():
            return

        if await self._is_own_message(
            communication_data.author.address,
            conv_id,
            communication_data.author.participant_id,
        ):
            return

        if conv_id not in self._conversations:
            self._start_conversation(conv_id, profile_id=None)

        session = self._conversations[conv_id]

        session.author_info = AuthorInfo(
            address=communication_data.author.address,
            participant_id=communication_data.author.participant_id,
        )

        # Store channelId in session metadata for outbound reply channelSettings
        if communication_data.channel_id:
            session.metadata["channel_id"] = communication_data.channel_id

        memory_response = await self._retrieve_memory_if_enabled(session, message_text, conv_id)

        try:
            response = await self.tac.trigger_message_ready(message_text, session, memory_response)
            # Auto-send if callback returned a string (None = manual send_response flow)
            if response is not None:
                await self.send_response(conv_id, response, role="assistant")
        except Exception as e:
            self.logger.error(
                "Error in message ready callback",
                conversation_id=conv_id,
                error=str(e),
                exc_info=True,
            )

    async def _handle_conversation_updated(self, event_data: Any) -> None:
        """Handle CONVERSATION_UPDATED event.

        Only processes CLOSED status for conversations tracked by this channel.
        """
        conversation_data = ConversationResponse.model_validate(event_data)
        conv_id = conversation_data.id
        status = conversation_data.status

        if (
            conversation_data.configuration_id == self.tac.config.conversation_configuration_id
            and status == "CLOSED"
        ):
            session = self._conversations.get(conv_id)
            if session and session.channel == self.get_channel_name():
                await self._end_conversation(conv_id)

    async def _initiate_messaging_conversation(
        self,
        options: InitiateMessagingConversationOptions,
        from_address: str,
        customer_address_kwargs: dict[str, str | None],
        agent_address_kwargs: dict[str, str | None],
        extra_metadata: dict[str, str] | None = None,
        channel_settings: ActionChannelSettings | None = None,
    ) -> InitiateConversationResult:
        """Shared outbound initiation logic for messaging channels (SMS, Chat).

        Subclasses call this with channel-specific address kwargs and settings.
        """
        channel_type = self.get_channel_type_upper()
        conversation_id: str | None = None
        reused = False

        try:
            (
                conversation_id,
                reused,
            ) = await self.tac.conversation_orchestrator_client.create_or_reuse_conversation(
                participants=[
                    ParticipantRequest(
                        type="CUSTOMER",
                        addresses=[
                            ParticipantAddress(
                                channel=channel_type,
                                address=options.to,
                                **customer_address_kwargs,
                            )
                        ],
                    ),
                    ParticipantRequest(
                        type="AI_AGENT",
                        addresses=[
                            ParticipantAddress(
                                channel=channel_type,
                                address=from_address,
                                **agent_address_kwargs,
                            )
                        ],
                    ),
                ]
            )

            participants = await self.tac.conversation_orchestrator_client.list_participants(
                conversation_id
            )

            def _match_address(
                p_addresses: list[ParticipantAddress],
                addr: str,
                extra_kwargs: dict[str, str | None],
            ) -> bool:
                return any(
                    a.channel == channel_type
                    and a.address == addr
                    and all(getattr(a, k) == v for k, v in extra_kwargs.items() if v)
                    for a in p_addresses
                )

            customer = next(
                (
                    p
                    for p in participants
                    if p.type == "CUSTOMER"
                    and _match_address(p.addresses, options.to, customer_address_kwargs)
                ),
                None,
            )
            if not customer:
                raise RuntimeError("Customer participant not found after conversation creation")

            agent = next(
                (
                    p
                    for p in participants
                    if p.type in ("AI_AGENT", "HUMAN_AGENT", "AGENT")
                    and _match_address(p.addresses, from_address, agent_address_kwargs)
                ),
                None,
            )
            if not agent:
                raise RuntimeError("Agent participant not found after conversation creation")

            session = self._start_conversation(conversation_id)
            session.author_info = AuthorInfo(address=options.to, participant_id=customer.id)
            session.metadata.update(
                {
                    **(options.metadata or {}),
                    **(extra_metadata or {}),
                    "direction": "outbound",
                    "from_address": from_address,
                }
            )

            action_request = SendMessageActionRequest(
                payload=SendMessageActionPayload(
                    from_=ActionParticipantRef(channel=channel_type, participant_id=agent.id),
                    to=[ActionParticipantRef(channel=channel_type, participant_id=customer.id)],
                    content=ActionTextContent(text=options.message),
                    channel_settings=channel_settings,
                ),
            )
            await self.tac.conversation_orchestrator_client.create_action(
                conversation_id, action_request
            )

            self.logger.info(
                f"Outbound {self.get_channel_name()} conversation initiated",
                conversation_id=conversation_id,
                to=mask_address(options.to),
            )
            return InitiateConversationResult(conversation_id=conversation_id, session=session)

        except Exception:
            if conversation_id:
                self._conversations.pop(conversation_id, None)
            if conversation_id and not reused:
                try:
                    await self.tac.conversation_orchestrator_client.update_conversation(
                        conversation_id, "CLOSED"
                    )
                except Exception as close_err:
                    self.logger.warning(
                        "Failed to close orphaned conversation after initiation error",
                        conversation_id=conversation_id,
                        error=str(close_err),
                    )
            raise

    async def _ensure_agent_participant(
        self,
        conversation_id: str,
        existing_participants: list[ParticipantResponse],
        agent_address: ParticipantAddress,
    ) -> ParticipantResponse | None:
        """Return the conversation's AI_AGENT participant, creating one if absent.

        Returns the first participant in `existing_participants` whose type is
        AI_AGENT / AGENT / HUMAN_AGENT and owns `agent_address`. If none match,
        creates an AI_AGENT with that address. On a 409 from another worker
        creating it concurrently, re-lists and re-matches.

        Returns None if match-then-create-then-retry all fail. The caller should
        log and bail on None.
        """

        def _matches(p: ParticipantResponse) -> bool:
            return p.type in ("AI_AGENT", "HUMAN_AGENT", "AGENT") and any(
                a.channel == agent_address.channel and a.address == agent_address.address
                for a in p.addresses
            )

        agent = next((p for p in existing_participants if _matches(p)), None)
        if agent:
            return agent

        self.logger.debug(
            "No agent participant found, creating AI_AGENT",
            conversation_id=conversation_id,
            channel=agent_address.channel,
            address=mask_address(agent_address.address),
        )
        try:
            agent = await self.tac.conversation_orchestrator_client.add_participant(
                conversation_id,
                addresses=[agent_address],
                participant_type="AI_AGENT",
            )
            self.logger.debug(
                "Created AI_AGENT participant",
                conversation_id=conversation_id,
                participant_id=agent.id,
            )
            return agent
        except Exception as e:
            # Most likely a 409 race (another worker just created the agent), but
            # we catch broadly here — log the original error so a real 5xx isn't
            # hidden by the generic "failed to create or find" log below.
            self.logger.warning(
                "Failed to create AI_AGENT, retrying participant list",
                conversation_id=conversation_id,
                error=str(e),
            )

        try:
            retried = await self.tac.conversation_orchestrator_client.list_participants(
                conversation_id
            )
        except Exception as e:
            self.logger.error(
                "Failed to retry listing participants",
                conversation_id=conversation_id,
                error=str(e),
            )
            return None

        agent = next((p for p in retried if _matches(p)), None)
        if not agent:
            self.logger.error(
                "Failed to create or find AI_AGENT participant",
                conversation_id=conversation_id,
            )
        return agent

"""MessagingChannel base class for messaging channels (SMS, RCS, WhatsApp, Chat)."""

import time
from abc import abstractmethod
from collections.abc import AsyncGenerator
from contextlib import nullcontext
from typing import Any

import httpx
from pydantic import BaseModel, Field

from tac import TAC
from tac.channels.base import BaseChannel
from tac.context.conversation import ConversationClient
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
from tac.models.memory import MemoryMode
from tac.models.outbound import InitiateConversationResult, InitiateMessagingConversationOptions
from tac.models.session import AuthorInfo
from tac.telemetry.tracer import TracerClient
from tac.utils.redaction import mask_address

# Participant types that represent TAC itself at TAC's (channel, address).
# `AI_AGENT` is the canonical type; `AGENT` is the legacy Conversation
# Orchestrator form. A participant typed either way at TAC's address is
# recognized as TAC and not overwritten; anything else (HUMAN_AGENT,
# CUSTOMER, …) is someone else's assignment.
AGENT_TYPES: frozenset[str] = frozenset({"AGENT", "AI_AGENT"})


class MessagingChannelConfig(BaseModel):
    """Base configuration for messaging channels (SMS, RCS, WhatsApp, Chat).

    Attributes:
        dedup_capacity: Maximum number of idempotency tokens to track.
            Default 10000 is suitable for most applications.
            Uses Twilio's i-twilio-idempotency-token header for deduplication.
        memory_mode: Memory retrieval mode. Default is "never".
            - "always": Retrieve memory for every message with the query string
            - "once": Retrieve memory once at conversation start with empty query and cache it.
                     Cache is invalidated when conversation becomes INACTIVE and is fetched
                     again the next time a message triggers memory retrieval after the
                     conversation becomes ACTIVE.
            - "never": Skip memory retrieval
    """

    dedup_capacity: int = Field(
        default=10000,
        gt=0,
        description="Maximum number of idempotency tokens to track for deduplication",
    )
    memory_mode: MemoryMode = Field(
        default="never",
        description="Memory retrieval mode for this channel",
    )


class MessagingChannel(BaseChannel):
    """Abstract base class for messaging channels (SMS, RCS, WhatsApp, Chat).

    Provides shared webhook processing logic for channels that use
    Conversation Orchestrator webhooks with COMMUNICATION_CREATED
    and CONVERSATION_UPDATED event types.

    Subclasses must implement:
    - is_default_agent_address(): Fast-path check for the channel's default agent address
    - get_channel_type_upper(): Return uppercase channel type ("SMS", "RCS", "WHATSAPP", "CHAT")
    - get_agent_address(conversation_id): Return the agent's ParticipantAddress for a conversation
    - send_response(): Send messages back through the channel
    - get_channel_name(): Return lowercase channel name ("sms", "rcs", "whatsapp", "chat")

    Subclass class attributes:
    - reconcile_customer_type: If True, reconciliation will also promote a
      channel-matching UNKNOWN participant (not owning the agent address) to
      CUSTOMER. Set False for channels where the customer is identified
      author-driven (e.g. chat).
    """

    reconcile_customer_type: bool = True

    def __init__(
        self,
        tac: TAC,
        dedup_capacity: int = 10000,
        memory_mode: MemoryMode = "never",
    ):
        if tac.conversation_orchestrator_client is None:
            raise ValueError(
                f"{type(self).__name__} requires Conversation Orchestrator to be configured. "
                "Set `conversation_configuration_id` on TACConfig to enable messaging channels."
            )
        self.conversation_orchestrator_client: ConversationClient = (
            tac.conversation_orchestrator_client
        )
        super().__init__(tac, memory_mode=memory_mode, dedup_capacity=dedup_capacity)

        # Initialize telemetry clients
        try:
            from tac.telemetry.metrics import MetricsClient

            self._metrics = MetricsClient()
        except ImportError:
            self._metrics = None

        try:
            self._tracer = TracerClient()
        except ImportError:
            self._tracer = None

    @abstractmethod
    def is_default_agent_address(self, author_address: str) -> bool:
        """Fast-path check: is the author address this channel's default agent address?

        For example, config.phone_number for SMS, config.rcs_sender_id for RCS,
        config.whatsapp_number for WhatsApp, agent_address for Chat.

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
        """Check if a message is from the bot itself (2-tier).

        1. Default agent address (stateless, no API call)
        2. API fallback via listParticipants (cross-process / multi-worker)
        """
        if self.is_default_agent_address(author_address):
            return True

        if author_participant_id:
            try:
                participants = await self.conversation_orchestrator_client.list_participants(
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
                    if author_p.type in AGENT_TYPES:
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
    def get_agent_address(self, conversation_id: str) -> ParticipantAddress:
        """Return the agent-side ParticipantAddress for this conversation.

        Used by `_reconcile_participants` to identify which participant (by
        channel + address) represents the agent. May read from session state
        (e.g. chat's per-conversation channelId) to build the address.
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
        - COMMUNICATION_CREATED: Process incoming messages from customers
        - CONVERSATION_UPDATED: Clean up when conversation is closed

        Note: Conversation tracking uses instance-local memory. In multi-instance
        deployments, webhooks may route to a different instance, preventing cleanup.
        See CLAUDE.md for horizontal scaling considerations.

        Args:
            webhook_data: Raw webhook event data from Twilio
            idempotency_token: Optional Twilio idempotency token from request headers
        """
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

        if not self._is_event_for_this_channel(webhook_data):
            return

        if event_type == "COMMUNICATION_CREATED":
            await self._handle_communication_created(event_data)
        elif event_type == "CONVERSATION_UPDATED":
            await self._handle_conversation_updated(event_data)

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

        channel_name = self.get_channel_name()
        span_attributes = {
            "channel": channel_name,
            "conversation_id": conv_id,
        }

        # Only include message content if explicitly enabled (privacy-safe by default)
        if self._tracer and self._tracer.include_message_content:
            span_attributes["input"] = message_text

        # Start root span for entire message processing
        # We need to manually start the span to be able to add attributes later
        if self._tracer:
            from opentelemetry import trace

            message_span = self._tracer.tracer.start_span(
                "message.processing", attributes=span_attributes
            )
            span_context = trace.use_span(message_span, end_on_exit=True)
        else:
            message_span = None
            span_context = nullcontext()

        with span_context:
            try:
                # 📊 Metric: Message received
                if self._metrics:
                    self._metrics.message_received_count.add(
                        1, attributes={"channel": channel_name}
                    )

                if conv_id not in self._conversations:
                    # 📊 Metric: Conversation start
                    start_conv_time = time.time()
                    self._start_conversation(conv_id, profile_id=None)
                    if self._metrics:
                        self._metrics.conversation_start_duration.record(
                            time.time() - start_conv_time, attributes={"channel": channel_name}
                        )
                        self._metrics.conversation_active_count.add(1)

                session = self._conversations[conv_id]

                session.author_info = AuthorInfo(
                    address=communication_data.author.address,
                    participant_id=communication_data.author.participant_id,
                )

                # Store channelId in session metadata for outbound reply channelSettings
                if communication_data.channel_id:
                    session.metadata["channel_id"] = communication_data.channel_id

                # Reconcile participant types pre-LLM so v1-bridge's UNKNOWN gets
                # promoted to CUSTOMER (with a Conversation Memory profile attached
                # when possible) and to stash both participant ids on the session
                # for send_response. If reconciliation can't identify both sides,
                # any eventual reply would fail too — skip the callback so the LLM
                # doesn't waste a turn on an un-replyable conversation.
                #
                # Skip reconcile entirely when both sides are already stashed from
                # a prior turn — Conversation Orchestrator's state was written by
                # us and doesn't drift.
                if session.ai_agent_info is None or session.author_info is None:
                    resolved = await self._reconcile_participants(conv_id)
                    if resolved is None:
                        self.logger.warning(
                            "Reconciliation failed; skipping callback for this inbound",
                            conversation_id=conv_id,
                        )
                        return

                    agent_participant, customer_participant = resolved
                    session.ai_agent_info = AuthorInfo(
                        address=self.get_agent_address(conv_id).address,
                        participant_id=agent_participant.id,
                    )
                    # When reconcile resolved a customer (SMS path — chat disables
                    # customer reconciliation and uses the author_info captured from
                    # the webhook above), use its authoritative participant id and
                    # lift any resolved profile.
                    if customer_participant is not None and session.author_info is not None:
                        session.author_info.participant_id = customer_participant.id
                        if customer_participant.profile_id and not session.profile_id:
                            session.profile_id = customer_participant.profile_id

                # Add user ID to span if profile_id is available
                if message_span and session.profile_id:
                    message_span.set_attribute("enduser.id", session.profile_id)

                # 🔍 Span: Memory retrieval
                with (
                    self._tracer.start_span("memory.retrieve", attributes=span_attributes)
                    if self._tracer
                    else nullcontext()
                ):
                    memory_response = await self._retrieve_memory_if_enabled(
                        session, message_text, conv_id
                    )

                # 🔍 Span: User callback execution
                with (
                    self._tracer.start_span("conversation.ready", attributes=span_attributes)
                    if self._tracer
                    else nullcontext()
                ):
                    callback_start = time.time()
                    response = await self.tac.trigger_message_ready(
                        message_text, session, memory_response
                    )
                    callback_duration = time.time() - callback_start

                    if self._metrics:
                        self._metrics.conversation_ready_duration.record(
                            callback_duration, attributes={"channel": channel_name}
                        )

                # Auto-send if callback returned a string (None = manual send_response flow)
                if response is not None:
                    # Add output to the message.processing span (only if message content is enabled)
                    if message_span and self._tracer.include_message_content:
                        message_span.set_attribute("output", response)

                    # 🔍 Span: Send response
                    with (
                        self._tracer.start_span("message.send", attributes=span_attributes)
                        if self._tracer
                        else nullcontext()
                    ):
                        await self.send_response(conv_id, response, role="assistant")

                        # 📊 Metric: Message sent
                        if self._metrics:
                            self._metrics.message_sent_count.add(
                                1, attributes={"channel": channel_name}
                            )

            except Exception as e:
                # 📊 Metric: Message error
                if self._metrics:
                    self._metrics.message_error_count.add(
                        1,
                        attributes={
                            "channel": channel_name,
                            "error_type": type(e).__name__,
                        },
                    )

                self.logger.error(
                    "Error in message ready callback",
                    conversation_id=conv_id,
                    error=str(e),
                    exc_info=True,
                )
                raise

    async def _handle_conversation_updated(self, event_data: Any) -> None:
        """Handle CONVERSATION_UPDATED event.

        - CLOSED: Remove session (clears cache)
        - INACTIVE: Invalidate cached memory (Orchestrator updates memory on INACTIVE)
        """
        conversation_data = ConversationResponse.model_validate(event_data)
        conv_id = conversation_data.id
        status = conversation_data.status

        if conversation_data.configuration_id != self.tac.config.conversation_configuration_id:
            return

        session = self._conversations.get(conv_id)
        if not session or session.channel != self.get_channel_name():
            return

        if status == "CLOSED":
            # 📊 Metric: Conversation end
            channel_name = self.get_channel_name()
            end_start_time = time.time()

            if self._metrics:
                self._metrics.conversation_active_count.add(-1)

            await self._end_conversation(conv_id)

            if self._metrics:
                self._metrics.conversation_end_duration.record(
                    time.time() - end_start_time,
                    attributes={"channel": channel_name, "reason": "completed"},
                )
        elif status == "INACTIVE" and self.memory_mode == "once":
            # Invalidate cached memory when conversation becomes inactive
            # Memory is updated by Conversation Orchestrator on INACTIVE transition
            async with session.cache_lock:
                if session.cached_memory is not None:
                    session.cached_memory = None
                    self.logger.debug(
                        "Invalidated cached memory on INACTIVE status",
                        conversation_id=conv_id,
                    )

    async def _initiate_messaging_conversation(
        self,
        options: InitiateMessagingConversationOptions,
        from_address: str,
        customer_address_kwargs: dict[str, str | None],
        agent_address_kwargs: dict[str, str | None],
        extra_metadata: dict[str, str] | None = None,
        channel_settings: ActionChannelSettings | None = None,
    ) -> InitiateConversationResult:
        """Shared outbound initiation logic for messaging channels (SMS, RCS, WhatsApp, Chat).

        Subclasses call this with channel-specific address kwargs and settings.
        """
        channel_type = self.get_channel_type_upper()
        conversation_id: str | None = None
        reused = False

        try:
            (
                conversation_id,
                reused,
            ) = await self.conversation_orchestrator_client.create_or_reuse_conversation(
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

            participants = await self.conversation_orchestrator_client.list_participants(
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
                    if p.type in AGENT_TYPES
                    and _match_address(p.addresses, from_address, agent_address_kwargs)
                ),
                None,
            )
            if not agent:
                raise RuntimeError("Agent participant not found after conversation creation")

            session = self._start_conversation(conversation_id)
            session.author_info = AuthorInfo(address=options.to, participant_id=customer.id)
            session.ai_agent_info = AuthorInfo(address=from_address, participant_id=agent.id)
            session.metadata.update(
                {
                    **(options.metadata or {}),
                    **(extra_metadata or {}),
                    "direction": "outbound",
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
            await self.conversation_orchestrator_client.create_action(
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
                    await self.conversation_orchestrator_client.update_conversation(
                        conversation_id, "CLOSED"
                    )
                except Exception as close_err:
                    self.logger.warning(
                        "Failed to close orphaned conversation after initiation error",
                        conversation_id=conversation_id,
                        error=str(close_err),
                    )
            raise

    async def _reconcile_participants(
        self,
        conversation_id: str,
    ) -> tuple[ParticipantResponse, ParticipantResponse | None] | None:
        """Reconcile Conversation Orchestrator's participants to the types TAC needs for sending.

        v1-bridge capture can leave TAC's agent participant as `UNKNOWN` (wrong
        type at our address), or omit it entirely (customer-only conversation).
        This pass fixes those cases; it refuses to rewrite anything else at our
        address. Decision matrix:

            | Agent side           | Customer side       | Action                        |
            |----------------------|---------------------|-------------------------------|
            | AGENT / AI_AGENT     | CUSTOMER            | Use as-is (no profile work).  |
            | AGENT / AI_AGENT     | UNKNOWN, no CUST    | Resolve profile, PUT → CUST.  |
            | UNKNOWN at our addr  | CUSTOMER            | PUT agent → AI_AGENT.         |
            | UNKNOWN at our addr  | UNKNOWN, no CUST    | PUT agent; resolve, PUT CUST. |
            | other at our addr    | any                 | Return None (log ERROR).      |
            | none at our addr     | CUSTOMER or UNKNOWN | POST AI_AGENT, then proceed.  |
            | any                  | no resolvable cust  | Return None (caller WARNs).   |

        TAC recognizes both `AGENT` and `AI_AGENT` at its address as itself.
        `HUMAN_AGENT` is NOT treated as TAC (a real human is a separate
        participant — TAC must not speak on their behalf); it falls into the
        "other at our addr" row and causes the reconcile to bail.

        Customer-side reconciliation is gated by `reconcile_customer_type`.
        Chat sets it to `False` because chat identifies the customer
        author-driven (via `session.author_info.participant_id`), so promoting
        some other `UNKNOWN` CHAT participant could pick the wrong recipient.

        Returns:
            `(agent, customer_or_none)` on success. `customer` is `None` when
            `reconcile_customer_type` is `False`. `None` overall when either
            the agent or the customer cannot be resolved — the caller
            (`_handle_communication_created`) treats `None` as a hard stop
            and skips the message-ready callback, since any eventual reply
            would fail too.
        """
        agent_address = self.get_agent_address(conversation_id)

        try:
            participants = await self.conversation_orchestrator_client.list_participants(
                conversation_id
            )
        except Exception as e:
            self.logger.error(
                "Failed to list participants for reconciliation",
                conversation_id=conversation_id,
                error=str(e),
            )
            return None

        channel = agent_address.channel

        def _owns_agent_address(p: ParticipantResponse) -> bool:
            return any(
                a.channel == channel and a.address == agent_address.address for a in p.addresses
            )

        def _matches_channel(p: ParticipantResponse) -> bool:
            return any(a.channel == channel for a in p.addresses)

        agent_candidate = next((p for p in participants if _owns_agent_address(p)), None)
        if agent_candidate is None:
            agent_candidate = await self._add_agent_participant(
                conversation_id=conversation_id,
                agent_address=agent_address,
            )
            if agent_candidate is None:
                return None
        elif agent_candidate.type == "UNKNOWN":
            # Only promote UNKNOWN — an already-typed participant at TAC's
            # address that isn't AGENT/AI_AGENT (e.g., CUSTOMER, HUMAN_AGENT)
            # is someone else's assignment and must not be overwritten.
            agent_candidate = await self._promote_participant(
                conversation_id=conversation_id,
                participant=agent_candidate,
                new_type="AI_AGENT",
            )
            if agent_candidate is None:
                return None
        elif agent_candidate.type not in AGENT_TYPES:
            self.logger.error(
                "Participant at TAC's address has a conflicting type; refusing to "
                "overwrite. Check Conversation Orchestrator participant state — a non-agent "
                "participant is holding TAC's (channel, address).",
                conversation_id=conversation_id,
                participant_id=agent_candidate.id,
                participant_type=agent_candidate.type,
            )
            return None

        if not self.reconcile_customer_type:
            return agent_candidate, None

        customer = next(
            (
                p
                for p in participants
                if p.type == "CUSTOMER" and _matches_channel(p) and not _owns_agent_address(p)
            ),
            None,
        )
        if customer is not None:
            return agent_candidate, customer

        customer_unknown = next(
            (
                p
                for p in participants
                if p.type == "UNKNOWN" and _matches_channel(p) and not _owns_agent_address(p)
            ),
            None,
        )
        if customer_unknown is not None:
            profile_id = await self._resolve_customer_profile(customer_unknown, channel)
            promoted_customer = await self._promote_participant(
                conversation_id=conversation_id,
                participant=customer_unknown,
                new_type="CUSTOMER",
                profile_id=profile_id,
            )
            if promoted_customer is not None:
                return agent_candidate, promoted_customer

        self.logger.warning(
            "No customer participant resolvable; skipping webhook",
            conversation_id=conversation_id,
            channel=channel,
        )
        return None

    async def _resolve_customer_profile(
        self,
        customer: ParticipantResponse,
        channel: str,
    ) -> str | None:
        """Find or mint a Conversation Memory profile for a customer being promoted from UNKNOWN.

        Only resolves for phone-based channels (SMS, VOICE). Looks up by phone
        identifier first; on miss, creates a new profile using the configured
        phone trait group/field. Returns None on any failure — the caller still
        promotes the participant, just without a `profile_id` attached.
        """
        if channel not in ("SMS", "VOICE"):
            return None

        memory_client = self.tac.conversation_memory_client
        if memory_client is None:
            return None

        phone_address = next(
            (a.address for a in customer.addresses if a.channel == channel and a.address),
            None,
        )
        if not phone_address:
            return None

        try:
            lookup = await memory_client.lookup_profile(
                id_type="phone",
                value=phone_address,
            )
            if lookup.profiles:
                return lookup.profiles[0]
        except Exception as e:
            self.logger.warning(
                "Profile lookup failed during reconciliation; falling back to create",
                conversation_id=customer.conversation_id,
                error=str(e),
            )

        memory_config = self.tac.config.memory_config
        trait_group = memory_config.phone_trait_group
        trait_field = memory_config.phone_trait_field

        try:
            return await memory_client.create_profile(
                traits={trait_group: {trait_field: phone_address}},
            )
        except Exception as e:
            self.logger.warning(
                "Profile creation failed during reconciliation; promoting without profile",
                conversation_id=customer.conversation_id,
                error=str(e),
            )
            return None

    async def _promote_participant(
        self,
        conversation_id: str,
        participant: ParticipantResponse,
        new_type: str,
        profile_id: str | None = None,
    ) -> ParticipantResponse | None:
        """PUT a participant to `new_type`.

        Conversation Orchestrator's PUT is a full-resource replacement, so we
        pass the existing `name` and `addresses` back unchanged to avoid wiping
        them. `profile_id` defaults to the participant's current value; pass a
        non-None override to attach a newly resolved profile during CUSTOMER
        reconciliation.

        Returns None on any error (including 409). A 409 from Conversation
        Orchestrator here means the promotion is structurally blocked — stop
        and surface it; don't retry.
        """
        effective_profile_id = profile_id if profile_id is not None else participant.profile_id
        try:
            updated = await self.conversation_orchestrator_client.update_participant(
                conversation_id=conversation_id,
                participant_id=participant.id,
                participant_type=new_type,  # type: ignore[arg-type]
                addresses=participant.addresses,
                name=participant.name,
                profile_id=effective_profile_id,
            )
            self.logger.debug(
                "Promoted participant",
                conversation_id=conversation_id,
                participant_id=participant.id,
                from_type=participant.type,
                to_type=new_type,
            )
            return updated
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 409:
                self.logger.warning(
                    "Conversation Orchestrator returned 409 on participant promotion; "
                    "skipping — likely a conflicting conversation or grouping constraint. "
                    "Check Conversation Orchestrator for duplicate active conversations.",
                    conversation_id=conversation_id,
                    participant_id=participant.id,
                    target_type=new_type,
                    conflicting_resource_id=e.response.headers.get("X-Conflicting-Resource-Id"),
                )
                return None
            self.logger.error(
                "Failed to promote participant",
                conversation_id=conversation_id,
                participant_id=participant.id,
                target_type=new_type,
                error=str(e),
            )
            return None
        except Exception as e:
            self.logger.error(
                "Failed to promote participant",
                conversation_id=conversation_id,
                participant_id=participant.id,
                target_type=new_type,
                error=str(e),
            )
            return None

    async def _add_agent_participant(
        self,
        conversation_id: str,
        agent_address: ParticipantAddress,
    ) -> ParticipantResponse | None:
        """POST an `AI_AGENT` participant owning `agent_address`.

        Returns None on any error (including 409). A 409 here means the
        address is already owned or the conversation's participant set
        can't accept a new AI_AGENT — stop and surface it; don't retry.
        """
        try:
            created = await self.conversation_orchestrator_client.add_participant(
                conversation_id=conversation_id,
                addresses=[agent_address],
                participant_type="AI_AGENT",
            )
            self.logger.debug(
                "Added AI_AGENT participant",
                conversation_id=conversation_id,
                participant_id=created.id,
            )
            return created
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 409:
                self.logger.warning(
                    "Conversation Orchestrator returned 409 on AI_AGENT participant add; "
                    "skipping — address is already owned or the conversation can't accept "
                    "a new AI_AGENT. Check Conversation Orchestrator participant state.",
                    conversation_id=conversation_id,
                    conflicting_resource_id=e.response.headers.get("X-Conflicting-Resource-Id"),
                )
                return None
            self.logger.error(
                "Failed to add AI_AGENT participant",
                conversation_id=conversation_id,
                error=str(e),
            )
            return None
        except Exception as e:
            self.logger.error(
                "Failed to add AI_AGENT participant",
                conversation_id=conversation_id,
                error=str(e),
            )
            return None

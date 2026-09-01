"""MessagingChannel base class for messaging channels (SMS, RCS, WhatsApp, Chat)."""

from abc import abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from tac import TAC
from tac.channels.base import AGENT_TYPES, BaseChannel
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
from tac.models.session import AuthorInfo, ConversationSession
from tac.utils.redaction import mask_address

MessagingMemoryMode = Literal["never", "always"]
"""Memory modes a messaging channel supports.

Narrower than :data:`~tac.models.memory.MemoryMode`: ``"once"`` caches a recall
on a long-lived session, which only voice has.
"""


class MessagingChannelConfig(BaseModel):
    """Base configuration for messaging channels (SMS, RCS, WhatsApp, Chat).

    Attributes:
        dedup_capacity: Maximum number of idempotency tokens to track.
            Default 10000 is suitable for most applications.
            Uses Twilio's i-twilio-idempotency-token header for deduplication.
        memory_mode: Memory retrieval mode. Default is `"never"`.

            - `"always"`: Retrieve memory for every message with the query string
            - `"never"`: Skip memory retrieval

            `"once"` is **not** available on messaging channels — see
            `MessagingMemoryMode`.
    """

    dedup_capacity: int = Field(
        default=10000,
        gt=0,
        description="Maximum number of idempotency tokens to track for deduplication",
    )
    memory_mode: MessagingMemoryMode = Field(
        default="never",
        description="Memory retrieval mode for this channel",
    )


class MessagingChannel(BaseChannel):
    """Abstract base class for messaging channels (SMS, RCS, WhatsApp, Chat).

    Provides shared webhook processing logic for channels that use
    Conversation Orchestrator webhooks with COMMUNICATION_CREATED
    and CONVERSATION_UPDATED event types.

    **Stateless by construction.** No conversation state is kept between
    webhooks: each request derives its own `ConversationSession` from the
    payload, config, and one `list_participants` call it needs anyway. So any
    replica can serve any webhook — no sticky sessions, no shared datastore.

    Subclasses must implement:

    - `is_default_agent_address()`: Fast-path check for the channel's default agent address
    - `get_agent_address(session)`: Return the agent's `ParticipantAddress` for a session
    - `get_channel_name()`: Return channel name (`"SMS"`, `"RCS"`, `"WHATSAPP"`, `"CHAT"`)

    `send_response()` is provided here as a shared implementation. Subclasses may
    override `_build_channel_settings()` to customize how `ActionChannelSettings`
    is built for the outbound send (e.g. chat requires `channel_id`).

    Subclass class attributes:

    - `reconcile_customer_type`: If True, reconciliation will also promote a
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
        if memory_mode == "once":
            raise ValueError(
                f'{type(self).__name__} does not support memory_mode="once" — it caches a '
                "recall on a long-lived session, and messaging channels hold none between "
                'webhooks. Use "always". ("once" is still available on the Voice channel.)'
            )
        self.conversation_orchestrator_client: ConversationClient = (
            tac.conversation_orchestrator_client
        )
        super().__init__(tac, memory_mode=memory_mode, dedup_capacity=dedup_capacity)

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

    def _is_own_message(
        self,
        author_participant_id: str | None,
        participants: list[ParticipantResponse],
        conversation_id: str,
    ) -> bool:
        """Whether the author is TAC, judged from an already-fetched list.

        The caller does the free check (author address == configured agent
        address) first; this catches TAC speaking from some other address.
        """
        if not author_participant_id:
            return False

        author = next((p for p in participants if p.id == author_participant_id), None)
        if author is None:
            return False
        if author.type is None:
            self.logger.warning(
                "Participant type is undefined",
                conversation_id=conversation_id,
                participant_id=author_participant_id,
            )
        return author.type in AGENT_TYPES

    async def _list_participants(self, conversation_id: str) -> list[ParticipantResponse] | None:
        """Fetch a conversation's participants, returning None on failure.

        One call per inbound message, shared by the self-message check,
        reconciliation, and profile resolution. None of them can proceed
        without it, so a failure is a hard stop for the caller.
        """
        try:
            return await self.conversation_orchestrator_client.list_participants(conversation_id)
        except Exception as e:
            self.logger.error(
                "Failed to list participants",
                conversation_id=conversation_id,
                error=str(e),
            )
            return None

    @abstractmethod
    def get_agent_address(self, session: ConversationSession) -> ParticipantAddress:
        """Return the agent-side ParticipantAddress for this session.

        Identifies which participant represents the agent, and supplies the
        `from` address for outbound sends. Reads the session where the address
        isn't pure config — chat's per-conversation `channelId`, for instance.

        !!! warning "Breaking change"
            Was `get_agent_address(conversation_id: str)`. It takes the session
            now because the channel no longer holds one to look up. Subclasses
            outside TAC must update their override.
        """
        pass

    def _build_channel_settings(
        self, conversation_id: str, session: ConversationSession
    ) -> ActionChannelSettings | None:
        """Build the ActionChannelSettings for an outbound send, if any.

        Default behavior (SMS, RCS, WhatsApp): channel_id is optional and,
        when present in session metadata, is passed through as-is. Chat
        overrides this since channel_id is required for delivery.
        """
        channel_id = session.metadata.get("channel_id")
        return (
            ActionChannelSettings(channel_id=channel_id)
            if isinstance(channel_id, str) and channel_id
            else None
        )

    def _participant_ref(
        self, address: str | None, participant_id: str | None
    ) -> ActionParticipantRef:
        """Build an Actions API participant reference.

        Conversation Orchestrator resolves by participant id or by explicit
        `(channel, address)`. Prefer the id — reconciliation has already
        established it — and fall back to the address when there isn't one.
        """
        channel_name = self.get_channel_name()
        if participant_id:
            return ActionParticipantRef(channel=channel_name, participant_id=participant_id)
        return ActionParticipantRef(channel=channel_name, address=address)

    def _session_from_participants(
        self, conversation_id: str, participants: list[ParticipantResponse]
    ) -> ConversationSession | None:
        """Rebuild a session for a conversation this process never handled.

        Returns ``None`` when no participant is on this channel, which is how
        another channel's conversation is filtered out.
        """
        channel_name = self.get_channel_name()
        session = ConversationSession(conversation_id=conversation_id, channel=channel_name)
        agent_address = self.get_agent_address(session)

        def _matches_channel(p: ParticipantResponse) -> bool:
            return any(a.channel == channel_name for a in p.addresses)

        if not any(_matches_channel(p) for p in participants):
            return None

        agent = self._find_agent_participant(participants, channel_name, agent_address.address)
        session.ai_agent_info = AuthorInfo(
            address=agent_address.address,
            participant_id=agent.id if agent else None,
        )

        customer = next(
            (
                p
                for p in participants
                if p.type == "CUSTOMER"
                and _matches_channel(p)
                and not self._owns_address(p, channel_name, agent_address.address)
            ),
            None,
        )
        if customer is not None:
            session.profile_id = customer.profile_id
            customer_address = next(
                (a.address for a in customer.addresses if a.channel == channel_name),
                None,
            )
            if customer_address:
                session.author_info = AuthorInfo(
                    address=customer_address, participant_id=customer.id
                )
        return session

    async def send_response(
        self,
        conversation: str | ConversationSession,
        response: str | AsyncGenerator[str | dict[str, Any], None],
        role: str | None = None,
    ) -> None:
        """Send a text response using the Conversation Orchestrator Send API.

        **Pass the session, not the id.** Hand back the one `on_message_ready`
        gave you (or that `initiate_outbound_conversation` returned) and the
        send costs no extra API call — it already carries both participants.
        A bare id still works, but the channel keeps no session to look up, so
        it spends a `list_participants` call rebuilding one.

        Args:
            conversation: The `ConversationSession` to reply within
                (preferred), or the conversation id.
            response: Message content. Must be ``str`` — messaging channels send a
                single complete message via the Conversation Orchestrator Send API
                and do not support streaming (unlike the Voice channel).
            role: Optional message role (unused by messaging channels)

        !!! warning "Breaking change"
            The first parameter was named `conversation_id`. Positional calls
            are unaffected; a call passing `conversation_id=` needs updating.

        Raises:
            TypeError: If response is not a string (e.g. an async generator is
                passed, since messaging channels don't support streaming)
            RuntimeError: If no recipient can be resolved for the conversation.
        """
        channel_name = self.get_channel_name()
        if not isinstance(response, str):
            raise TypeError(f"{channel_name} channel only supports string responses")

        if isinstance(conversation, ConversationSession):
            session: ConversationSession | None = conversation
            conversation_id = conversation.conversation_id
        else:
            conversation_id = conversation
            participants = await self._list_participants(conversation_id)
            if participants is None:
                raise RuntimeError(
                    f"Unable to send {channel_name} message: could not read participants for "
                    f"conversation {conversation_id}. Pass the ConversationSession from "
                    "on_message_ready or initiate_outbound_conversation to avoid this lookup."
                )
            session = self._session_from_participants(conversation_id, participants)

        if session is None or session.author_info is None:
            raise RuntimeError(
                f"Unable to send {channel_name} message: no recipient resolved for "
                f"conversation {conversation_id}. Pass the ConversationSession you were "
                "handed by on_message_ready or initiate_outbound_conversation."
            )

        agent_address = (
            session.ai_agent_info.address
            if session.ai_agent_info
            else self.get_agent_address(session).address
        )
        channel_settings = self._build_channel_settings(conversation_id, session)

        try:
            action_request = SendMessageActionRequest(
                payload=SendMessageActionPayload(
                    from_=self._participant_ref(
                        agent_address,
                        session.ai_agent_info.participant_id if session.ai_agent_info else None,
                    ),
                    to=[
                        self._participant_ref(
                            session.author_info.address,
                            session.author_info.participant_id,
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
                f"Sent {channel_name} response via Actions API",
                conversation_id=conversation_id,
                to_address=mask_address(session.author_info.address),
                channel_id=channel_settings.channel_id if channel_settings else None,
            )
        except Exception as e:
            self.logger.error(
                "Failed to create action",
                conversation_id=conversation_id,
                error=str(e),
                exc_info=True,
            )

    async def process_webhook(
        self, webhook_data: dict[str, Any], idempotency_token: str | None = None
    ) -> None:
        """Process messaging channel webhook event and manage conversation lifecycle.

        Handles:

        - COMMUNICATION_CREATED: Process incoming messages from customers
        - CONVERSATION_UPDATED: Fire `on_conversation_ended` when the
          conversation closes

        Any replica can handle any webhook. The idempotency cache is
        per-process, though, so a Twilio retry landing on a different replica
        is processed again — `on_message_ready` handlers must tolerate being
        called twice for the same message.

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

    def _resolve_session(self, conv_id: str, communication: Communication) -> ConversationSession:
        """Build this request's session from the webhook payload and config.

        Network-free — the author, the agent address and `channel_id` are all
        in hand already. Reconciliation layers the participant ids on after.
        """
        session = ConversationSession(
            conversation_id=conv_id,
            channel=self.get_channel_name(),
        )
        session.author_info = AuthorInfo(
            address=communication.author.address,
            participant_id=communication.author.participant_id,
        )
        # Set before get_agent_address: chat's agent address carries channelId.
        if communication.channel_id:
            session.metadata["channel_id"] = communication.channel_id
        session.ai_agent_info = AuthorInfo(address=self.get_agent_address(session).address)
        return session

    async def _handle_communication_created(self, event_data: Any) -> None:
        """Handle COMMUNICATION_CREATED event (incoming message)."""
        communication_data = Communication.model_validate(event_data)
        conv_id = communication_data.conversation_id
        message_text = communication_data.content.text

        if not message_text or not message_text.strip():
            return

        # TAC's own echo costs zero API calls — the address is right there.
        if self.is_default_agent_address(communication_data.author.address):
            return

        session = self._resolve_session(conv_id, communication_data)
        channel = self.get_channel_name()

        # One participant fetch per message, shared by the self-message check,
        # reconciliation, and profile resolution. All three need it, and none
        # can proceed without it.
        participants = await self._list_participants(conv_id)
        if participants is None:
            await self._drop_inbound(
                conv_id, channel, "Could not read participants; inbound message dropped"
            )
            return

        if self._is_own_message(communication_data.author.participant_id, participants, conv_id):
            return

        # Reconcile participant types pre-LLM so v1-bridge's UNKNOWN gets
        # promoted to CUSTOMER (with a Conversation Memory profile attached when
        # possible) and to resolve both participant ids for the reply. If it
        # can't identify both sides, any eventual reply would fail too — skip
        # the callback rather than waste an LLM turn on an un-replyable
        # conversation. Running it every message is free: it reuses the list
        # above, and the happy-path row issues no writes.
        resolved = await self._reconcile_participants(session, participants)
        if resolved is None:
            await self._drop_inbound(
                conv_id, channel, "Participant reconciliation failed; inbound message dropped"
            )
            return

        agent_participant, customer_participant = resolved
        assert session.ai_agent_info is not None  # set by _resolve_session
        session.ai_agent_info.participant_id = agent_participant.id
        # When reconcile resolved a customer (chat disables customer
        # reconciliation and keeps the webhook author), its participant id is
        # the authoritative reply recipient — the author of an inbound message
        # is not necessarily the customer.
        if customer_participant is not None and session.author_info is not None:
            session.author_info.participant_id = customer_participant.id
            if customer_participant.profile_id:
                session.profile_id = customer_participant.profile_id

        if session.profile_id is None and self.memory_mode != "never":
            # Free, and more reliable than Memory's address-based lookup, which
            # infers the identifier type and misses on CHAT and RCS.
            session.profile_id = self._profile_id_from_participants(session, participants)

        memory_response = await self._retrieve_memory_if_enabled(session, message_text, conv_id)

        try:
            response = await self.tac.trigger_message_ready(message_text, session, memory_response)
            # Auto-send if callback returned a string (None = manual send_response flow)
            if response is not None:
                await self.send_response(session, response, role="assistant")
        except Exception as e:
            self.logger.error(
                "Error in message ready callback",
                conversation_id=conv_id,
                error=str(e),
                exc_info=True,
            )

    def _profile_id_from_participants(
        self, session: ConversationSession, participants: list[ParticipantResponse]
    ) -> str | None:
        """Read the customer's profile id off the participant list already fetched."""
        author_participant_id = session.author_info.participant_id if session.author_info else None
        if not author_participant_id:
            return None
        author = next((p for p in participants if p.id == author_participant_id), None)
        return author.profile_id if author else None

    async def _drop_inbound(self, conv_id: str, channel: str, reason: str) -> None:
        """Log a dropped inbound message and surface it via ``on_error``."""
        self.logger.error(reason, conversation_id=conv_id, channel=channel)
        await self.tac.trigger_error(
            RuntimeError(reason),
            {"conversation_id": conv_id, "channel": channel, "dropped_inbound": True},
        )

    async def _handle_conversation_updated(self, event_data: Any) -> None:
        """Handle CONVERSATION_UPDATED event.

        Only CLOSED is acted on, and only when an `on_conversation_ended`
        handler is registered, so the common case costs nothing. The rebuild
        also confirms the conversation belongs to this channel (a CHAT close
        must not fire on SMS), and is what lets the hook fire on whichever
        replica Twilio picked.
        """
        conversation_data = ConversationResponse.model_validate(event_data)
        conv_id = conversation_data.id

        if conversation_data.configuration_id != self.tac.config.conversation_configuration_id:
            return
        if conversation_data.status != "CLOSED":
            return
        if not self.tac._has_conversation_ended_callback():
            return

        participants = await self._list_participants(conv_id)
        if participants is None:
            return
        session = self._session_from_participants(conv_id, participants)
        if session is None:
            return
        await self._trigger_conversation_ended(session)

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
        channel_type = self.get_channel_name()
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

            customer = next(
                (
                    p
                    for p in participants
                    if p.type == "CUSTOMER"
                    and self._owns_address(p, channel_type, options.to, customer_address_kwargs)
                ),
                None,
            )
            if not customer:
                raise RuntimeError("Customer participant not found after conversation creation")

            agent = self._find_agent_participant(
                participants, channel_type, from_address, agent_address_kwargs
            )
            if not agent:
                raise RuntimeError("Agent participant not found after conversation creation")

            session = ConversationSession(
                conversation_id=conversation_id,
                channel=channel_type,
            )
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
        session: ConversationSession,
        participants: list[ParticipantResponse],
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

        Args:
            session: Supplies the conversation id and agent address to match on.
            participants: Already fetched by the caller — reusing that response
                is what makes running this on every message free.
        """
        conversation_id = session.conversation_id
        agent_address = self.get_agent_address(session)
        channel = agent_address.channel

        def _owns_agent_address(p: ParticipantResponse) -> bool:
            return self._owns_address(p, channel, agent_address.address)

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

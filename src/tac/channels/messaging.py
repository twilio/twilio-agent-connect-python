"""MessagingChannel base class for messaging channels (SMS, Chat)."""

from abc import abstractmethod
from collections import OrderedDict
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from pydantic import BaseModel, Field

from tac import TAC
from tac.channels.base import BaseChannel
from tac.models.conversation import (
    Communication,
    ConversationResponse,
    ParticipantAddress,
    ParticipantResponse,
)
from tac.models.session import AuthorInfo

# Session metadata keys populated by participant reconciliation.
SESSION_META_AGENT_PARTICIPANT_ID = "agent_participant_id"
SESSION_META_CUSTOMER_PARTICIPANT_ID = "customer_participant_id"


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
    Conversation Orchestrator webhooks with COMMUNICATION_CREATED
    and CONVERSATION_UPDATED event types.

    Subclasses must implement:
    - is_own_message(): Check if a message is from the bot itself
    - get_channel_type_upper(): Return uppercase channel type ("SMS", "CHAT")
    - get_agent_address(conversation_id): Return the agent's ParticipantAddress for a conversation
    - send_response(): Send messages back through the channel
    - get_channel_name(): Return lowercase channel name ("sms", "chat")

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
        auto_retrieve_memory: bool = False,
    ):
        super().__init__(tac, auto_retrieve_memory=auto_retrieve_memory)
        self._processed_tokens: OrderedDict[str, bool] = OrderedDict()
        self._max_tracked_tokens = dedup_capacity

    @abstractmethod
    def is_own_message(self, author_address: str) -> bool:
        """Check if a message is from the bot itself.

        Args:
            author_address: The address of the message author

        Returns:
            True if the message is from the bot
        """
        pass

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

        Callers must ensure `webhook_data["data"]` is a dict before invoking.
        """
        event_type = webhook_data.get("eventType")
        event_data = webhook_data["data"]

        if event_type == "COMMUNICATION_CREATED":
            author = event_data.get("author")
            author_channel = author.get("channel") if isinstance(author, dict) else None
            if not author_channel:
                return False
            return bool(author_channel == self.get_channel_type_upper())

        if event_type == "CONVERSATION_UPDATED":
            conv_id = event_data.get("id")
            if conv_id and conv_id not in self._conversations:
                return False

        return True

    async def process_webhook(
        self, webhook_data: dict[str, Any], idempotency_token: str | None = None
    ) -> None:
        """Process messaging channel webhook event and manage conversation lifecycle.

        Handles:
        - COMMUNICATION_CREATED: Process incoming messages from customers
        - CONVERSATION_UPDATED: Clean up when conversation is closed

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

        if (
            self.is_own_message(communication_data.author.address)
            or not message_text
            or not message_text.strip()
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

        # Reconcile participant types so `send_response` can attribute outbound
        # Actions to the correct AI_AGENT / CUSTOMER ids. Runs pre-LLM so a
        # broken conversation fails fast without wasting an agent turn.
        resolved = await self._reconcile_participants(conv_id)
        if resolved is None:
            return
        agent_participant, customer_participant = resolved
        session.metadata[SESSION_META_AGENT_PARTICIPANT_ID] = agent_participant.id
        if customer_participant is not None:
            session.metadata[SESSION_META_CUSTOMER_PARTICIPANT_ID] = customer_participant.id

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

        if (
            conversation_data.configuration_id == self.tac.config.conversation_configuration_id
            and conversation_data.status == "CLOSED"
            and conv_id in self._conversations
            and self._conversations[conv_id].channel == self.get_channel_name()
        ):
            await self._end_conversation(conv_id)

    async def _reconcile_participants(
        self,
        conversation_id: str,
    ) -> tuple[ParticipantResponse, ParticipantResponse | None] | None:
        """Reconcile Maestro's participants to the types TAC needs for sending.

        TAC treats itself strictly as `AI_AGENT`: any participant that owns
        TAC's (channel, address) but has a different type — including `AGENT`
        and `HUMAN_AGENT` — is promoted to `AI_AGENT` via `PUT /Participants`.
        This is deliberate: `HUMAN_AGENT` in particular must never be reused as
        TAC's `from`, because a real human participant (e.g. a Studio handoff)
        joins with a *different* address/participant — collisions at our
        address mean the type is simply wrong, not that a human is speaking.

        v1-bridge capture emits two broken shapes this pass handles:
        (1) the agent side is created as `UNKNOWN` (so it owns our address but
        has the wrong type), and (2) the agent side is not created at all — on
        inbound SMS the bridge often produces only the customer participant,
        and TAC must add itself before replying. Decision matrix (coordinated
        with the Maestro team):

            | Agent side                  | Customer side           | Action                      |
            |-----------------------------|-------------------------|-----------------------------|
            | AI_AGENT                    | CUSTOMER                | Use as-is.                  |
            | AI_AGENT                    | UNKNOWN (no CUSTOMER)   | PUT customer → CUSTOMER.    |
            | other, owns our address     | CUSTOMER                | PUT agent → AI_AGENT.       |
            | other, owns our address     | UNKNOWN (no CUSTOMER)   | Two PUTs.                   |
            | nobody owns our address     | CUSTOMER or UNKNOWN     | POST AI_AGENT, then proceed.|
            | any                         | no resolvable customer  | Skip webhook (WARN).        |

        Customer-side reconciliation is gated by `reconcile_customer_type`.
        Chat sets it to `False` because chat identifies the customer
        author-driven (via `session.author_info.participant_id`), so promoting
        some other `UNKNOWN` CHAT participant could pick the wrong recipient.

        PUT 409 is treated as concurrent-update success: re-list and use the
        current server view of that participant id (see `_promote_participant`).
        POST 409 (another worker added the agent first) is likewise handled by
        re-listing and picking up the existing AI_AGENT owner.

        Returns:
            `(agent, customer_or_none)` when the agent side is resolvable.
            `customer` is `None` when `reconcile_customer_type` is `False`.
            `None` overall means the agent cannot be resolved — the caller
            should skip the webhook without invoking the LLM.
        """
        agent_address = self.get_agent_address(conversation_id)

        try:
            participants = await self.tac.conversation_orchestrator_client.list_participants(
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
        elif agent_candidate.type != "AI_AGENT":
            agent_candidate = await self._promote_participant(
                conversation_id=conversation_id,
                participant=agent_candidate,
                new_type="AI_AGENT",
            )
            if agent_candidate is None:
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
            promoted_customer = await self._promote_participant(
                conversation_id=conversation_id,
                participant=customer_unknown,
                new_type="CUSTOMER",
            )
            if promoted_customer is not None:
                return agent_candidate, promoted_customer

        self.logger.warning(
            "No customer participant resolvable; skipping webhook",
            conversation_id=conversation_id,
            channel=channel,
        )
        return None

    async def _promote_participant(
        self,
        conversation_id: str,
        participant: ParticipantResponse,
        new_type: str,
    ) -> ParticipantResponse | None:
        """PUT a participant to `new_type`.

        Treats 409 as a concurrent-update success: re-lists participants and
        returns the current server view of this participant id. Other errors
        return None.
        """
        try:
            updated = await self.tac.conversation_orchestrator_client.update_participant(
                conversation_id=conversation_id,
                participant_id=participant.id,
                participant_type=new_type,  # type: ignore[arg-type]
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
                # Another worker already updated this participant. Re-fetch to
                # get the current type; return it only if it matches our target.
                self.logger.debug(
                    "Participant update 409; re-fetching after concurrent update",
                    conversation_id=conversation_id,
                    participant_id=participant.id,
                )
                try:
                    refreshed = await self.tac.conversation_orchestrator_client.list_participants(
                        conversation_id
                    )
                except Exception as list_err:
                    self.logger.error(
                        "Failed to re-list participants after 409",
                        conversation_id=conversation_id,
                        error=str(list_err),
                    )
                    return None
                current = next((p for p in refreshed if p.id == participant.id), None)
                if current is not None and current.type == new_type:
                    return current
                self.logger.error(
                    "Participant not at target type after concurrent update",
                    conversation_id=conversation_id,
                    participant_id=participant.id,
                    target_type=new_type,
                    current_type=current.type if current else None,
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

        On 409 (another worker added it first), re-lists and returns the
        existing owner of `agent_address` if one now exists. Other errors
        return None.
        """
        try:
            created = await self.tac.conversation_orchestrator_client.add_participant(
                conversation_id=conversation_id,
                addresses=[agent_address],
                participant_type="AI_AGENT",
            )
            self.logger.debug(
                "Added AI_AGENT participant",
                conversation_id=conversation_id,
                participant_id=created.id,
                channel=agent_address.channel,
                address=agent_address.address,
            )
            return created
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 409:
                self.logger.debug(
                    "Add agent participant 409; re-listing after concurrent add",
                    conversation_id=conversation_id,
                )
                try:
                    refreshed = await self.tac.conversation_orchestrator_client.list_participants(
                        conversation_id
                    )
                except Exception as list_err:
                    self.logger.error(
                        "Failed to re-list participants after 409",
                        conversation_id=conversation_id,
                        error=str(list_err),
                    )
                    return None
                existing = next(
                    (
                        p
                        for p in refreshed
                        if p.type == "AI_AGENT"
                        and any(
                            a.channel == agent_address.channel
                            and a.address == agent_address.address
                            for a in p.addresses
                        )
                    ),
                    None,
                )
                if existing is not None:
                    return existing
                self.logger.error(
                    "No AI_AGENT owner found after add 409",
                    conversation_id=conversation_id,
                    channel=agent_address.channel,
                    address=agent_address.address,
                )
                return None
            self.logger.error(
                "Failed to add AI_AGENT participant",
                conversation_id=conversation_id,
                channel=agent_address.channel,
                address=agent_address.address,
                error=str(e),
            )
            return None
        except Exception as e:
            self.logger.error(
                "Failed to add AI_AGENT participant",
                conversation_id=conversation_id,
                channel=agent_address.channel,
                address=agent_address.address,
                error=str(e),
            )
            return None

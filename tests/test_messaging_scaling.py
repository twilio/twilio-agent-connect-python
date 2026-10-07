"""Messaging behaviour that horizontal scaling depends on.

Messaging is request/response: any replica can serve any webhook, because a
channel derives its session per request and stores nothing. These tests pin
the three things that makes true — no residual state, cross-instance
`on_conversation_ended`, and an API-call budget that doesn't quietly grow to
compensate.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tac import TAC
from tac.channels.chat import ChatChannel
from tac.channels.rcs import RCSChannel
from tac.channels.sms import SMSChannel
from tac.channels.whatsapp import WhatsAppChannel
from tac.context.memory import MemoryClient
from tac.models.conversation import ConversationResponse, ParticipantAddress, ParticipantResponse
from tac.models.memory import (
    MemoryRetrievalMeta,
    MemoryRetrievalResponse,
    ProfileLookupResponse,
    ProfileResponse,
)
from tac.models.outbound import InitiateMessagingConversationOptions
from tac.models.session import ConversationSession

CONFIGURATION_ID = "conv_configuration_test123"
AGENT_NUMBER = "+15551234567"
CUSTOMER_NUMBER = "+12345678901"


def get_test_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": CONFIGURATION_ID,
        "phone_number": AGENT_NUMBER,
    }
    config.update(overrides)
    return config


def participant(
    pid: str,
    ptype: str,
    address: str,
    *,
    conv_id: str = "CH123",
    channel: str = "SMS",
    profile_id: str | None = None,
) -> ParticipantResponse:
    return ParticipantResponse(
        id=pid,
        conversation_id=conv_id,
        account_id="ACtest123",
        name=address,
        type=ptype,  # type: ignore[arg-type]
        profile_id=profile_id,
        addresses=[ParticipantAddress(channel=channel, address=address)],  # type: ignore[arg-type]
    )


def healthy_participants(
    conv_id: str = "CH123", profile_id: str | None = None
) -> list[ParticipantResponse]:
    """Both sides correctly typed — reconciliation's happy-path row."""
    return [
        participant("PA_AGENT", "AI_AGENT", AGENT_NUMBER, conv_id=conv_id),
        participant(
            "PA_CUSTOMER", "CUSTOMER", CUSTOMER_NUMBER, conv_id=conv_id, profile_id=profile_id
        ),
    ]


def inbound(
    conv_id: str = "CH123",
    *,
    text: str = "hello",
    author_address: str = CUSTOMER_NUMBER,
    author_participant_id: str = "PA_CUSTOMER",
    comm_id: str = "comms_communication_01",
) -> dict[str, Any]:
    return {
        "eventType": "COMMUNICATION_CREATED",
        "data": {
            "id": comm_id,
            "conversationId": conv_id,
            "accountId": "ACtest123",
            "author": {
                "address": author_address,
                "channel": "SMS",
                "participantId": author_participant_id,
            },
            "content": {"type": "TEXT", "text": text},
            "recipients": [],
            "createdAt": "2026-04-27T00:00:00Z",
        },
    }


def closed(conv_id: str = "CH123", metadata: dict[str, str] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": conv_id,
        "accountId": "ACtest123",
        "configurationId": CONFIGURATION_ID,
        "status": "CLOSED",
    }
    if metadata is not None:
        data["metadata"] = metadata
    return {"eventType": "CONVERSATION_UPDATED", "data": data}


async def send_outbound(channel: SMSChannel, metadata: dict[str, Any] | None = None) -> Any:
    return await channel.initiate_outbound_conversation(
        InitiateMessagingConversationOptions(
            to=CUSTOMER_NUMBER, message="Reminder", metadata=metadata
        )
    )


class CallCounter:
    """Counts every Conversation Orchestrator / Memory call a webhook makes."""

    def __init__(self, tac: TAC, profile_id: str | None = None) -> None:
        self.calls: list[str] = []
        co = tac.conversation_orchestrator_client
        assert co is not None
        self._install(
            co, "list_participants", lambda conv_id: healthy_participants(conv_id, profile_id)
        )
        self._install(co, "create_action", lambda *a, **k: None)
        self._install(co, "update_participant", lambda *a, **k: None)
        self._install(co, "add_participant", lambda *a, **k: None)
        self._install(co, "list_communications", lambda *a, **k: [])
        self.co_metadata: dict[str, str] = {}
        self._install(co, "create_or_reuse_conversation", lambda *a, **k: ("CH123", False))
        self._install(
            co,
            "get_conversation",
            lambda conv_id: ConversationResponse(
                id=conv_id, account_id="ACtest123", metadata=dict(self.co_metadata)
            ),
        )
        self._install(
            co,
            "patch_conversation_metadata",
            lambda conv_id, metadata: ConversationResponse(
                id=conv_id,
                account_id="ACtest123",
                metadata={**self.co_metadata, **metadata},
            ),
        )

        memory = tac.conversation_memory_client
        if memory is not None:
            self._install(
                memory,
                "retrieve_memory",
                lambda *a, **k: MemoryRetrievalResponse(
                    observations=[], summaries=[], meta=MemoryRetrievalMeta(queryTime=0)
                ),
            )
            self._install(
                memory,
                "get_profile",
                lambda *a, **k: ProfileResponse(id="profile_1", createdAt="2026-01-01T00:00:00Z"),
            )
            self._install(
                memory,
                "lookup_profile",
                lambda *a, **k: ProfileLookupResponse(
                    normalizedValue=CUSTOMER_NUMBER, profiles=["profile_1"]
                ),
            )

    def _install(self, client: Any, name: str, result: Any) -> None:
        async def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return result(*args, **kwargs)

        setattr(client, name, call)

    def count(self, name: str) -> int:
        return self.calls.count(name)


def make_sms_channel(
    mode: str = "never", profile_id: str | None = None, **memory_overrides: Any
) -> tuple[TAC, SMSChannel, CallCounter]:
    from tac.core.config import TwilioMemoryConfig

    tac = TAC(get_test_config(memory_config=TwilioMemoryConfig(**memory_overrides)))
    tac.conversation_memory_client = MemoryClient(
        store_id="MGtest123", api_key=tac.config.api_key, api_secret=tac.config.api_secret
    )
    channel = SMSChannel(tac, config={"memory_mode": mode})
    return tac, channel, CallCounter(tac, profile_id)


class TestNothingIsStored:
    @pytest.mark.asyncio
    async def test_channel_has_no_session_store_at_all(self) -> None:
        """Messaging keeps no session store. The regression guard: if this
        attribute comes back, so does the leak. (The bounded metadata cache is
        best-effort and nothing relies on it for correctness.)"""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        assert not hasattr(channel, "_conversations")

    @pytest.mark.asyncio
    async def test_inbound_leaves_no_residue(self) -> None:
        """Inbound leaves no session store behind; only the best-effort cache."""
        tac, channel, counter = make_sms_channel()
        sessions: list[ConversationSession] = []
        tac.on_message_ready(lambda msg, ctx, mem: sessions.append(ctx))

        await channel.process_webhook(inbound())

        assert len(sessions) == 1
        assert not hasattr(channel, "_conversations")


class TestTwoInstances:
    """A and B share nothing but configuration — which is the point."""

    @pytest.mark.asyncio
    async def test_closed_fires_on_the_instance_that_never_saw_the_message(self) -> None:
        tac_a = TAC(get_test_config())
        tac_b = TAC(get_test_config())
        instance_a = SMSChannel(tac_a)
        instance_b = SMSChannel(tac_b)

        ended: list[ConversationSession] = []
        tac_b.on_conversation_ended(lambda ctx: ended.append(ctx))

        # The message is handled by A.
        tac_a.on_message_ready(lambda msg, ctx, mem: None)
        with patch.object(
            tac_a.conversation_orchestrator_client,
            "list_participants",
            new=AsyncMock(return_value=healthy_participants()),
        ):
            await instance_a.process_webhook(inbound())

        # CLOSED is delivered to B.
        with patch.object(
            tac_b.conversation_orchestrator_client,
            "list_participants",
            new=AsyncMock(return_value=healthy_participants()),
        ):
            await instance_b.process_webhook(closed())

        assert len(ended) == 1
        assert ended[0].conversation_id == "CH123"
        assert ended[0].channel == "SMS"
        assert ended[0].author_info is not None
        assert ended[0].author_info.address == CUSTOMER_NUMBER
        assert ended[0].ai_agent_info is not None
        assert ended[0].ai_agent_info.participant_id == "PA_AGENT"

    @pytest.mark.asyncio
    async def test_a_chat_close_does_not_fire_on_the_sms_channel(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        ended: list[ConversationSession] = []
        tac.on_conversation_ended(lambda ctx: ended.append(ctx))

        with patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            new=AsyncMock(
                return_value=[
                    participant("PA_U", "CUSTOMER", "user-1", channel="CHAT"),
                ]
            ),
        ):
            await channel.process_webhook(closed())

        assert ended == []

    @pytest.mark.asyncio
    async def test_reply_works_from_a_session_alone(self) -> None:
        """The session `on_message_ready` receives is enough to reply with —
        no channel state, so a reply from any replica behaves identically."""
        tac, channel, counter = make_sms_channel()
        sessions: list[ConversationSession] = []
        tac.on_message_ready(lambda msg, ctx, mem: sessions.append(ctx))

        await channel.process_webhook(inbound())
        counter.calls.clear()

        await channel.send_response(sessions[0], "hi back")

        assert counter.calls == ["create_action"]

    @pytest.mark.asyncio
    async def test_send_response_keeps_its_conversation_id_keyword(self) -> None:
        """`main` named the parameter `conversation_id`; keyword calls still
        work, with an id or with the session."""
        tac, channel, counter = make_sms_channel()
        sessions: list[ConversationSession] = []
        tac.on_message_ready(lambda msg, ctx, mem: sessions.append(ctx))
        await channel.process_webhook(inbound())
        counter.calls.clear()

        await channel.send_response(conversation_id="CH123", response="by id")
        await channel.send_response(conversation_id=sessions[0], response="by session")

        assert counter.calls == ["list_participants", "create_action", "create_action"]


class RecordingSMSChannel(SMSChannel):
    """A custom channel overriding `get_agent_address` the way `main` defined it."""

    def __init__(self, tac: TAC) -> None:
        super().__init__(tac)
        self.requested: list[object] = []

    def get_agent_address(self, conversation_id: str) -> ParticipantAddress:
        self.requested.append(conversation_id)
        return ParticipantAddress(channel="SMS", address=AGENT_NUMBER)


class TestGetAgentAddressCompatibility:
    """`get_agent_address(conversation_id)` keeps `main`'s signature, for both
    callers and subclasses that override it."""

    @pytest.mark.asyncio
    async def test_an_override_of_the_old_signature_gets_conversation_ids(self) -> None:
        tac = TAC(get_test_config())
        channel = RecordingSMSChannel(tac)
        tac.on_message_ready(lambda msg, ctx, mem: "reply")
        co = tac.conversation_orchestrator_client
        assert co is not None

        # Customer-only: reconciliation needs the agent address to add TAC.
        with (
            patch.object(
                co,
                "list_participants",
                new=AsyncMock(
                    return_value=[participant("PA_CUSTOMER", "CUSTOMER", CUSTOMER_NUMBER)]
                ),
            ),
            patch.object(
                co,
                "add_participant",
                new=AsyncMock(return_value=participant("PA_AGENT", "AI_AGENT", AGENT_NUMBER)),
            ),
            patch.object(co, "create_action", new=AsyncMock()) as create_action,
        ):
            await channel.process_webhook(inbound())

        assert channel.requested
        assert all(r == "CH123" for r in channel.requested)
        create_action.assert_awaited_once()

    @pytest.mark.parametrize(
        ("channel_type", "expected"),
        [
            (SMSChannel, ParticipantAddress(channel="SMS", address=AGENT_NUMBER)),
            (RCSChannel, ParticipantAddress(channel="RCS", address="rcs_sender")),
            (WhatsAppChannel, ParticipantAddress(channel="WHATSAPP", address="+15559876543")),
            (ChatChannel, ParticipantAddress(channel="CHAT", address="ai-assistant")),
        ],
    )
    def test_built_in_channels_answer_by_conversation_id(
        self, channel_type: type[Any], expected: ParticipantAddress
    ) -> None:
        tac = TAC(get_test_config(rcs_sender_id="rcs_sender", whatsapp_number="+15559876543"))

        assert channel_type(tac).get_agent_address("CH123") == expected

    @pytest.mark.asyncio
    async def test_chat_by_id_carries_the_channel_id_this_instance_saw(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        tac.on_message_ready(lambda msg, ctx, mem: None)
        event = inbound(author_address="user@example.com", author_participant_id="PA_USER")
        event["data"]["author"]["channel"] = "CHAT"
        event["data"]["channelId"] = "CH_CHAT_SID"

        with patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            new=AsyncMock(
                return_value=[
                    participant("PA_AGENT", "AI_AGENT", "ai-assistant", channel="CHAT"),
                    participant("PA_USER", "CUSTOMER", "user@example.com", channel="CHAT"),
                ]
            ),
        ):
            await channel.process_webhook(event)

        assert channel.get_agent_address("CH123").channel_id == "CH_CHAT_SID"
        assert ChatChannel(tac).get_agent_address("CH123").channel_id is None


def recipient(address: str) -> dict[str, Any]:
    return {
        "address": address,
        "channel": "SMS",
        "participantId": "PA_AGENT",
        "deliveryStatus": "DELIVERED",
    }


class TestParticipantLookupFailure:
    """A failed participant lookup on a later turn is answered from what this
    instance last reconciled, as `main` did from its in-memory session."""

    @staticmethod
    def failing_lookup(counter: CallCounter, tac: TAC) -> None:
        async def fail(conv_id: str) -> Any:
            counter.calls.append("list_participants")
            raise RuntimeError("CO unavailable")

        co = tac.conversation_orchestrator_client
        assert co is not None
        co.list_participants = fail  # type: ignore[method-assign]

    @pytest.mark.asyncio
    async def test_a_later_turn_on_the_same_instance_is_answered(self) -> None:
        tac, channel, counter = make_sms_channel(profile_id="profile_1")
        sessions: list[ConversationSession] = []
        errors: list[dict[str, Any]] = []
        tac.on_message_ready(lambda msg, ctx, mem: sessions.append(ctx) or "reply")
        tac.on_error(lambda e, ctx: errors.append(ctx))
        await channel.process_webhook(inbound(comm_id="comms_1"))

        self.failing_lookup(counter, tac)
        counter.calls.clear()
        with patch.object(channel.logger, "warning") as warning:
            await channel.process_webhook(inbound(comm_id="comms_2"))

        assert len(sessions) == 2
        second = sessions[1]
        assert second.author_info is not None
        assert second.author_info.participant_id == "PA_CUSTOMER"
        assert second.ai_agent_info is not None
        assert second.ai_agent_info.participant_id == "PA_AGENT"
        assert second.profile_id == "profile_1"
        assert counter.calls == ["list_participants", "create_action"]
        assert errors == []
        assert "last-known participants" in warning.call_args.args[0]

    @pytest.mark.asyncio
    async def test_a_turn_this_instance_never_reconciled_is_dropped(self) -> None:
        tac, channel, counter = make_sms_channel()
        handled: list[str] = []
        errors: list[dict[str, Any]] = []
        tac.on_message_ready(lambda msg, ctx, mem: handled.append(msg))
        tac.on_error(lambda e, ctx: errors.append(ctx))
        self.failing_lookup(counter, tac)

        await channel.process_webhook(inbound())

        assert handled == []
        assert len(errors) == 1
        assert errors[0]["dropped_inbound"] is True

    @pytest.mark.asyncio
    async def test_a_message_to_a_different_sender_is_not_answered_from_cache(self) -> None:
        tac = TAC(get_test_config(phone_numbers=[AGENT_NUMBER, "+15550009999"]))
        channel = SMSChannel(tac)
        counter = CallCounter(tac)
        handled: list[str] = []
        tac.on_message_ready(lambda msg, ctx, mem: handled.append(msg))
        tac.on_error(lambda e, ctx: None)
        first = inbound(comm_id="comms_1")
        first["data"]["recipients"] = [recipient(AGENT_NUMBER)]
        await channel.process_webhook(first)

        self.failing_lookup(counter, tac)
        second = inbound(comm_id="comms_2")
        second["data"]["recipients"] = [recipient("+15550009999")]
        await channel.process_webhook(second)

        assert handled == ["hello"]


class TestReplyRecipient:
    @pytest.mark.asyncio
    async def test_reply_goes_to_the_reconciled_customer_not_the_author(self) -> None:
        """A non-customer can author an inbound message. The reply still goes
        to the customer — the author's participant id is not the recipient."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        tac.on_message_ready(lambda msg, ctx, mem: "reply")

        participants = [
            participant("PA_AGENT", "AI_AGENT", AGENT_NUMBER),
            participant("PA_CUSTOMER", "CUSTOMER", CUSTOMER_NUMBER),
            participant("PA_HUMAN", "HUMAN_AGENT", "+15557778888"),
        ]

        with (
            patch.object(
                tac.conversation_orchestrator_client,
                "list_participants",
                new=AsyncMock(return_value=participants),
            ),
            patch.object(
                tac.conversation_orchestrator_client, "create_action", new=AsyncMock()
            ) as create_action,
        ):
            await channel.process_webhook(
                inbound(author_address="+15557778888", author_participant_id="PA_HUMAN")
            )

        request = create_action.await_args.args[1]
        assert request.payload.to[0].participant_id == "PA_CUSTOMER"

    @pytest.mark.asyncio
    async def test_chat_replies_to_the_author(self) -> None:
        """Chat disables customer reconciliation deliberately — its identities
        are opaque, so promoting some other UNKNOWN could pick the wrong thread."""
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        tac.on_message_ready(lambda msg, ctx, mem: "reply")

        event = inbound(author_address="user@example.com", author_participant_id="PA_USER")
        event["data"]["author"]["channel"] = "CHAT"
        event["data"]["channelId"] = "CH_CHAT_SID"

        participants = [
            participant("PA_AGENT", "AI_AGENT", "ai-assistant", channel="CHAT"),
            participant("PA_USER", "CUSTOMER", "user@example.com", channel="CHAT"),
            participant("PA_OTHER", "UNKNOWN", "someone-else", channel="CHAT"),
        ]

        with (
            patch.object(
                tac.conversation_orchestrator_client,
                "list_participants",
                new=AsyncMock(return_value=participants),
            ),
            patch.object(
                tac.conversation_orchestrator_client, "create_action", new=AsyncMock()
            ) as create_action,
        ):
            await channel.process_webhook(event)

        request = create_action.await_args.args[1]
        assert request.payload.to[0].participant_id == "PA_USER"
        assert request.payload.channel_settings.channel_id == "CH_CHAT_SID"


class TestApiCallBudget:
    """Statelessness must not be paid for with extra API calls.

    These counts are the contract. If one moves, either the budget genuinely
    changed and this test should be updated deliberately, or something started
    re-fetching what it already had.
    """

    @pytest.mark.asyncio
    async def test_memory_never_costs_two_calls(self) -> None:
        tac, channel, counter = make_sms_channel()
        tac.on_message_ready(lambda msg, ctx, mem: "reply")

        await channel.process_webhook(inbound())

        assert counter.calls == ["list_participants", "create_action"]

    @pytest.mark.asyncio
    async def test_own_echo_costs_nothing(self) -> None:
        tac, channel, counter = make_sms_channel()
        tac.on_message_ready(lambda msg, ctx, mem: "reply")

        await channel.process_webhook(inbound(author_address=AGENT_NUMBER))

        assert counter.calls == []

    @pytest.mark.asyncio
    async def test_reconcile_writes_nothing_when_both_sides_are_typed(self) -> None:
        """Reconciliation runs on every message; on the happy path it writes
        nothing, which is what makes running it every time free."""
        tac, channel, counter = make_sms_channel()
        tac.on_message_ready(lambda msg, ctx, mem: "reply")

        for i in range(3):
            await channel.process_webhook(inbound(comm_id=f"comm_{i}"))

        assert counter.count("update_participant") == 0
        assert counter.count("add_participant") == 0
        assert counter.count("list_participants") == 3

    @pytest.mark.asyncio
    async def test_memory_always_costs_four_calls(self) -> None:
        tac, channel, counter = make_sms_channel(mode="always", profile_id="profile_1")
        tac.on_message_ready(lambda msg, ctx, mem: "reply")

        await channel.process_webhook(inbound())

        assert counter.calls == [
            "list_participants",
            "get_profile",
            "retrieve_memory",
            "create_action",
        ]

    @pytest.mark.asyncio
    async def test_profile_id_comes_off_the_participant_list(self) -> None:
        """No `lookup_profile` — the id is already in the response we fetched."""
        tac, channel, counter = make_sms_channel(mode="always", profile_id="profile_1")
        tac.on_message_ready(lambda msg, ctx, mem: None)

        await channel.process_webhook(inbound())

        assert counter.count("lookup_profile") == 0

    @pytest.mark.asyncio
    async def test_disabling_trait_fetch_returns_to_three_calls(self) -> None:
        tac, channel, counter = make_sms_channel(
            mode="always", profile_id="profile_1", fetch_profile_traits=False
        )
        tac.on_message_ready(lambda msg, ctx, mem: "reply")

        await channel.process_webhook(inbound())

        assert counter.calls == ["list_participants", "retrieve_memory", "create_action"]

    @pytest.mark.asyncio
    async def test_closed_with_no_handler_costs_nothing(self) -> None:
        tac, channel, counter = make_sms_channel()

        await channel.process_webhook(closed())

        assert counter.calls == []

    @pytest.mark.asyncio
    async def test_closed_with_a_handler_costs_one_call(self) -> None:
        tac, channel, counter = make_sms_channel()
        tac.on_conversation_ended(lambda ctx: None)

        await channel.process_webhook(closed())

        assert counter.calls == ["list_participants"]

    @pytest.mark.asyncio
    async def test_outbound_create_costs_the_creation_calls_only(self) -> None:
        _tac, channel, counter = make_sms_channel()

        await send_outbound(channel, {"appointment_id": "apt_42"})

        assert counter.calls == [
            "create_or_reuse_conversation",
            "list_participants",
            "create_action",
        ]

    @pytest.mark.asyncio
    async def test_outbound_reuse_adds_one_metadata_patch(self) -> None:
        tac, channel, counter = make_sms_channel()
        counter._install(
            tac.conversation_orchestrator_client,
            "create_or_reuse_conversation",
            lambda *a, **k: ("CH123", True),
        )

        await send_outbound(channel, {"appointment_id": "apt_42"})

        assert counter.calls == [
            "create_or_reuse_conversation",
            "patch_conversation_metadata",
            "list_participants",
            "create_action",
        ]

    @pytest.mark.asyncio
    async def test_reading_conversation_metadata_on_another_instance_adds_one_fetch(
        self,
    ) -> None:
        tac, channel, counter = make_sms_channel()

        async def on_message(msg: str, session: ConversationSession, mem: Any) -> str:
            await session.conversation_metadata()
            return "reply"

        tac.on_message_ready(on_message)

        await channel.process_webhook(inbound())

        assert counter.calls == ["list_participants", "get_conversation", "create_action"]


class TestConversationMetadata:
    """Outbound metadata survives to later turns: in session.metadata on the
    instance that saw the conversation before (main's behaviour), and through
    session.conversation_metadata() on any instance."""

    @pytest.mark.asyncio
    async def test_outbound_writes_fitting_metadata_to_co_at_creation(self) -> None:
        tac, channel, counter = make_sms_channel()
        create = AsyncMock(return_value=("CH123", False))
        tac.conversation_orchestrator_client.create_or_reuse_conversation = create

        await send_outbound(channel, {"appointment_id": "apt_42"})

        assert create.await_args.kwargs["metadata"] == {
            "direction": "outbound",
            "appointment_id": "apt_42",
        }
        assert counter.count("patch_conversation_metadata") == 0

    @pytest.mark.asyncio
    async def test_unfit_entries_are_skipped_with_a_warning_but_kept_on_the_session(
        self,
    ) -> None:
        tac, channel, _counter = make_sms_channel()
        create = AsyncMock(return_value=("CH123", False))
        tac.conversation_orchestrator_client.create_or_reuse_conversation = create

        with patch.object(channel.logger, "warning") as warning:
            result = await send_outbound(channel, {"count": 3, "ok": "yes"})

        assert create.await_args.kwargs["metadata"] == {"direction": "outbound", "ok": "yes"}
        assert result.session.metadata["count"] == 3
        warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_reused_conversation_gets_the_metadata_patched_in(self) -> None:
        tac, channel, counter = make_sms_channel()
        counter.co_metadata = {"earlier": "kept"}
        tac.conversation_orchestrator_client.create_or_reuse_conversation = AsyncMock(
            return_value=("CH123", True)
        )

        result = await send_outbound(channel, {"appointment_id": "apt_42"})

        assert counter.count("patch_conversation_metadata") == 1
        assert await result.session.conversation_metadata() == {
            "earlier": "kept",
            "direction": "outbound",
            "appointment_id": "apt_42",
        }
        assert counter.count("get_conversation") == 0

    @pytest.mark.asyncio
    async def test_a_failed_patch_on_reuse_does_not_fail_the_send(self) -> None:
        tac, channel, _counter = make_sms_channel()
        co = tac.conversation_orchestrator_client
        co.create_or_reuse_conversation = AsyncMock(return_value=("CH123", True))
        co.patch_conversation_metadata = AsyncMock(side_effect=RuntimeError("CO down"))

        result = await send_outbound(channel, {"appointment_id": "apt_42"})

        assert result.conversation_id == "CH123"
        assert result.session.metadata["appointment_id"] == "apt_42"
        assert co.patch_conversation_metadata.await_count == 1
        # The failed PATCH left the CO metadata unknown, so the accessor fetches.
        co.get_conversation = AsyncMock(
            return_value=ConversationResponse(
                id="CH123", account_id="ACtest123", metadata={"earlier": "kept"}
            )
        )
        assert await result.session.conversation_metadata() == {"earlier": "kept"}
        assert co.get_conversation.await_count == 1

    @pytest.mark.asyncio
    async def test_a_reply_on_the_same_instance_sees_outbound_metadata(self) -> None:
        tac, channel, counter = make_sms_channel()
        seen: list[ConversationSession] = []
        tac.on_message_ready(lambda msg, session, mem: seen.append(session))

        await send_outbound(channel, {"appointment_id": "apt_42"})
        await channel.process_webhook(inbound(text="R"))

        assert seen[0].metadata["appointment_id"] == "apt_42"
        assert seen[0].metadata["direction"] == "outbound"
        assert await seen[0].conversation_metadata() == {
            "direction": "outbound",
            "appointment_id": "apt_42",
        }
        assert counter.count("get_conversation") == 0

    @pytest.mark.asyncio
    async def test_writes_in_one_turn_are_seen_in_the_next_on_the_same_instance(self) -> None:
        tac, channel, _counter = make_sms_channel()
        seen: list[dict[str, Any]] = []

        def on_message(msg: str, session: ConversationSession, mem: Any) -> None:
            seen.append(dict(session.metadata))
            session.metadata["step"] = str(len(seen))

        tac.on_message_ready(on_message)

        await channel.process_webhook(inbound(text="one", comm_id="c1"))
        await channel.process_webhook(inbound(text="two", comm_id="c2"))

        assert "step" not in seen[0]
        assert seen[1]["step"] == "1"

    @pytest.mark.asyncio
    async def test_another_instance_reads_metadata_through_the_accessor(self) -> None:
        tac_b, instance_b, counter_b = make_sms_channel()
        counter_b.co_metadata = {"direction": "outbound", "appointment_id": "apt_42"}
        seen: list[ConversationSession] = []
        tac_b.on_message_ready(lambda msg, session, mem: seen.append(session))

        await instance_b.process_webhook(inbound(text="R"))

        assert "appointment_id" not in seen[0].metadata
        metadata = await seen[0].conversation_metadata()
        assert metadata["appointment_id"] == "apt_42"
        await seen[0].conversation_metadata()
        assert counter_b.count("get_conversation") == 1

    @pytest.mark.asyncio
    async def test_a_failed_fetch_returns_empty_without_failing_the_turn(self) -> None:
        tac, channel, _counter = make_sms_channel()
        get = AsyncMock(side_effect=RuntimeError("CO down"))
        tac.conversation_orchestrator_client.get_conversation = get
        results: list[dict[str, str]] = []
        counts: list[int] = []

        async def on_message(msg: str, session: ConversationSession, mem: Any) -> None:
            results.append(await session.conversation_metadata())
            counts.append(get.await_count)
            results.append(await session.conversation_metadata())
            counts.append(get.await_count)

        tac.on_message_ready(on_message)

        await channel.process_webhook(inbound())

        assert results == [{}, {}]
        # A failed lookup is retried on the next call.
        assert counts == [1, 2]

    @pytest.mark.asyncio
    async def test_a_fetched_result_is_kept_for_the_turn_only(self) -> None:
        tac, channel, counter = make_sms_channel()
        counter.co_metadata = {"appointment_id": "apt_42"}
        results: list[dict[str, str]] = []

        async def on_message(msg: str, session: ConversationSession, mem: Any) -> None:
            results.append(await session.conversation_metadata())

        tac.on_message_ready(on_message)

        await channel.process_webhook(inbound(text="one", comm_id="c1"))
        counter.co_metadata = {"appointment_id": "apt_43"}
        await channel.process_webhook(inbound(text="two", comm_id="c2"))

        assert counter.count("get_conversation") == 2
        assert results == [{"appointment_id": "apt_42"}, {"appointment_id": "apt_43"}]

    @pytest.mark.asyncio
    async def test_closed_without_payload_metadata_fetches_it(self) -> None:
        tac, channel, counter = make_sms_channel()
        counter.co_metadata = {"appointment_id": "apt_42"}
        results: list[dict[str, str]] = []

        async def on_ended(session: ConversationSession) -> None:
            results.append(await session.conversation_metadata())

        tac.on_conversation_ended(on_ended)

        await channel.process_webhook(closed())

        assert results == [{"appointment_id": "apt_42"}]
        assert counter.count("get_conversation") == 1

    @pytest.mark.asyncio
    async def test_closed_on_another_instance_carries_the_payload_metadata(self) -> None:
        tac, channel, counter = make_sms_channel()
        ended: list[ConversationSession] = []
        tac.on_conversation_ended(lambda s: ended.append(s))

        await channel.process_webhook(closed(metadata={"appointment_id": "apt_42"}))

        assert ended[0].metadata["appointment_id"] == "apt_42"
        assert await ended[0].conversation_metadata() == {"appointment_id": "apt_42"}
        assert counter.count("get_conversation") == 0

    @pytest.mark.asyncio
    async def test_closed_on_the_same_instance_carries_local_metadata_and_evicts(self) -> None:
        tac, channel, _counter = make_sms_channel()
        ended: list[ConversationSession] = []
        tac.on_conversation_ended(lambda s: ended.append(s))

        def on_message(msg: str, session: ConversationSession, mem: Any) -> None:
            session.metadata["note"] = "local only"

        tac.on_message_ready(on_message)
        await channel.process_webhook(inbound())

        await channel.process_webhook(closed())

        assert ended[0].metadata["note"] == "local only"
        assert "CH123" not in channel._metadata_cache

    @pytest.mark.asyncio
    async def test_closed_without_handler_or_analytics_still_evicts_and_costs_nothing(
        self,
    ) -> None:
        tac, channel, counter = make_sms_channel()
        await channel.process_webhook(inbound())
        calls_before = len(counter.calls)

        await channel.process_webhook(closed())

        assert "CH123" not in channel._metadata_cache
        assert len(counter.calls) == calls_before

    def test_the_local_cache_is_bounded(self) -> None:
        _tac, channel, _counter = make_sms_channel()
        assert channel._metadata_cache._ttl == 24 * 60 * 60
        assert channel._metadata_cache._max_entries == 10_000

    @pytest.mark.asyncio
    async def test_closed_with_non_string_metadata_values_still_ends_the_conversation(
        self,
    ) -> None:
        tac, channel, _counter = make_sms_channel()
        results: list[dict[str, str]] = []

        async def on_ended(session: ConversationSession) -> None:
            results.append(await session.conversation_metadata())

        tac.on_conversation_ended(on_ended)

        await channel.process_webhook(closed(metadata={"n": 1, "x": None, "ok": "yes"}))

        assert results == [{"ok": "yes"}]

    @pytest.mark.asyncio
    async def test_closed_without_payload_metadata_uses_what_this_instance_wrote(self) -> None:
        tac, channel, counter = make_sms_channel()
        tac.conversation_orchestrator_client.create_or_reuse_conversation = AsyncMock(
            return_value=("CH123", False)
        )
        results: list[dict[str, str]] = []

        async def on_ended(session: ConversationSession) -> None:
            results.append(await session.conversation_metadata())

        tac.on_conversation_ended(on_ended)

        await send_outbound(channel, {"appointment_id": "apt_42"})
        await channel.process_webhook(closed())

        assert results == [{"direction": "outbound", "appointment_id": "apt_42"}]
        assert counter.count("get_conversation") == 0

    @pytest.mark.asyncio
    async def test_closed_session_metadata_is_the_payload_plus_local_entries(self) -> None:
        tac, channel, _counter = make_sms_channel()
        tac.conversation_orchestrator_client.create_or_reuse_conversation = AsyncMock(
            return_value=("CH123", False)
        )
        seen: list[dict[str, Any]] = []
        tac.on_conversation_ended(lambda session: seen.append(dict(session.metadata)))

        await send_outbound(channel, {"appointment_id": "apt_42"})
        await channel.process_webhook(closed(metadata={"from_payload": "yes"}))

        assert seen[0]["from_payload"] == "yes"
        assert seen[0]["appointment_id"] == "apt_42"
        assert seen[0]["direction"] == "outbound"

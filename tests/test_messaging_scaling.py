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
from tac.channels.sms import SMSChannel
from tac.context.memory import MemoryClient
from tac.models.conversation import ParticipantAddress, ParticipantResponse
from tac.models.memory import (
    MemoryRetrievalMeta,
    MemoryRetrievalResponse,
    ProfileLookupResponse,
    ProfileResponse,
)
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


def closed(conv_id: str = "CH123") -> dict[str, Any]:
    return {
        "eventType": "CONVERSATION_UPDATED",
        "data": {
            "id": conv_id,
            "accountId": "ACtest123",
            "configurationId": CONFIGURATION_ID,
            "status": "CLOSED",
        },
    }


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
        """The regression guard: if this attribute comes back, so does the leak."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        assert not hasattr(channel, "_conversations")

    @pytest.mark.asyncio
    async def test_inbound_leaves_no_residue(self) -> None:
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

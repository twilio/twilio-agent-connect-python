"""Voice behaviour that horizontal scaling depends on.

Three things have to hold for N replicas behind a load balancer:

1. A call's WebSocket closing frees **everything** local, on every path.
2. ``on_conversation_ended`` fires on whichever instance Conversation
   Orchestrator's CLOSED webhook happens to reach, even one that never saw
   the call.
3. Shutdown drains live calls instead of dropping them silently.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tac import TAC
from tac.channels.voice import VoiceChannel
from tac.channels.voice.conversation_relay import ConversationRelayProviderConfig
from tac.channels.voice.media_streams.gpt_live import GPTLiveProviderConfig
from tac.channels.voice.media_streams.gpt_live.models import _CallState as GPTLiveCallState
from tac.channels.voice.media_streams.openai_realtime import OpenAIRealtimeProviderConfig
from tac.channels.voice.media_streams.openai_realtime.models import (
    _CallState as RealtimeCallState,
)
from tac.channels.voice.media_streams.openai_realtime.provider import (
    TWILIO_AUDIO_FORMAT_FOR_REALTIME,
)
from tac.channels.websocket_protocol import WebSocketDisconnectError
from tac.models.conversation import (
    ConversationResponse,
    ParticipantAddress,
    ParticipantResponse,
)
from tac.models.handoff import PendingHandoffData
from tac.models.session import AuthorInfo, ConversationSession
from tac.session import ThreadSafeSessionManager
from tests.voice_invariants import assert_no_residual_state

CONFIGURATION_ID = "conv_configuration_test123"


def get_test_config(**overrides: Any) -> dict:
    config = {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": CONFIGURATION_ID,
        "phone_number": "+15551234567",
        "voice_public_domain": "example.com",
    }
    config.update(overrides)
    return config


def closed_webhook(conv_id: str) -> dict:
    """A CONVERSATION_UPDATED/CLOSED payload, as CO documents it: the
    Conversation resource carries no channel or channelId."""
    return {
        "eventType": "CONVERSATION_UPDATED",
        "data": {"id": conv_id, "status": "CLOSED", "configurationId": CONFIGURATION_ID},
    }


def collect(sink: list[ConversationSession]) -> Any:
    """An async handler that records the session it's given."""

    async def handler(session: ConversationSession) -> None:
        sink.append(session)

    return handler


def voice_participants(conv_id: str, call_sid: str | None = None) -> list[ParticipantResponse]:
    """A realistic post-call participant set: customer + TAC's AI_AGENT.

    ``call_sid`` lands on the VOICE addresses' ``channelId``, which is where
    CO records the call a voice participant is on.
    """
    return [
        ParticipantResponse(
            id="PA_customer",
            conversation_id=conv_id,
            account_id="ACtest123",
            name="Caller",
            type="CUSTOMER",
            profile_id="profile_caller",
            addresses=[
                ParticipantAddress(channel="VOICE", address="+15559998888", channel_id=call_sid)
            ],
        ),
        ParticipantResponse(
            id="PA_agent",
            conversation_id=conv_id,
            account_id="ACtest123",
            name="TAC Agent",
            type="AI_AGENT",
            addresses=[
                ParticipantAddress(channel="VOICE", address="+15551234567", channel_id=call_sid)
            ],
        ),
    ]


def mixed_channel_participants(conv_id: str) -> list[ParticipantResponse]:
    """A conversation grouped across channels: an SMS-only CUSTOMER listed
    before the caller, and the agent's leg on its own CallSid."""
    sms_customer = ParticipantResponse(
        id="PA_sms_customer",
        conversation_id=conv_id,
        account_id="ACtest123",
        name="Texter",
        type="CUSTOMER",
        profile_id="profile_sms",
        addresses=[ParticipantAddress(channel="SMS", address="+15550001111")],
    )
    customer, agent = voice_participants(conv_id, call_sid="CA_customer")
    agent.addresses[0].channel_id = "CA_agent_leg"
    return [sms_customer, customer, agent]


def conversation(conv_id: str, status: str, created_at: str) -> ConversationResponse:
    return ConversationResponse(
        id=conv_id,
        account_id="ACtest123",
        status=status,
        configuration_id=CONFIGURATION_ID,
        created_at=created_at,
    )


class TestHookSplit:
    """``on_call_ended`` is the call; ``on_conversation_ended`` is the conversation."""

    @pytest.mark.asyncio
    async def test_orchestrated_teardown_fires_call_ended_only(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        call_ended: list[ConversationSession] = []
        conversation_ended: list[ConversationSession] = []

        channel.on_call_ended(collect(call_ended))
        tac.on_conversation_ended(lambda s: conversation_ended.append(s))

        channel._start_conversation("conv_1", "profile_caller")
        await channel._provider._cleanup_connection("conv_1")

        assert [s.conversation_id for s in call_ended] == ["conv_1"]
        assert conversation_ended == []
        assert_no_residual_state(channel, "conv_1")

    @pytest.mark.asyncio
    async def test_relay_only_teardown_fires_both(self) -> None:
        """No Conversation Orchestrator conversation exists, so nothing will
        close it later — the channel must fire both hooks itself."""
        tac = TAC(get_test_config(conversation_configuration_id=None))
        channel = VoiceChannel(tac)
        call_ended: list[ConversationSession] = []
        conversation_ended: list[ConversationSession] = []

        channel.on_call_ended(collect(call_ended))
        tac.on_conversation_ended(lambda s: conversation_ended.append(s))

        channel._start_conversation("CA_relay", None)
        await channel._provider._cleanup_connection("CA_relay")

        assert [s.conversation_id for s in call_ended] == ["CA_relay"]
        assert [s.conversation_id for s in conversation_ended] == ["CA_relay"]
        assert_no_residual_state(channel, "CA_relay")

    @pytest.mark.asyncio
    async def test_call_ended_carries_state_the_rebuild_cannot(self) -> None:
        """The transcript only exists in memory — this hook is its last chance."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        captured: list[ConversationSession] = []

        channel.on_call_ended(collect(captured))

        session = channel._start_conversation("conv_2", None)
        session.call_sid = "CA_2"
        session.metadata["transcript"] = [{"role": "user", "text": "hello"}]

        await channel._provider._cleanup_connection("conv_2")

        assert captured[0].metadata["transcript"] == [{"role": "user", "text": "hello"}]
        assert captured[0].call_sid == "CA_2"


class TestStatelessConversationClosed:
    """CLOSED must work on an instance that never held the call (§7.3)."""

    @pytest.mark.asyncio
    async def test_fires_on_an_instance_that_never_saw_the_call(self) -> None:
        tac_a = TAC(get_test_config())
        tac_b = TAC(get_test_config())
        instance_a = VoiceChannel(tac_a)
        instance_b = VoiceChannel(tac_b)

        ended: list[ConversationSession] = []
        tac_b.on_conversation_ended(lambda s: ended.append(s))

        # Instance A takes the call and tears it down when the caller hangs up.
        instance_a._start_conversation("conv_shared", "profile_caller")
        await instance_a._provider._cleanup_connection("conv_shared")

        # CO's CLOSED webhook lands on B, which has never seen this call.
        tac_b.conversation_orchestrator_client.list_participants = AsyncMock(
            return_value=voice_participants("conv_shared", call_sid="CA_shared")
        )
        await instance_b.process_webhook(closed_webhook("conv_shared"))

        assert len(ended) == 1
        rebuilt = ended[0]
        assert rebuilt.conversation_id == "conv_shared"
        assert rebuilt.call_sid == "CA_shared"
        assert rebuilt.profile_id == "profile_caller"
        assert rebuilt.author_info is not None
        assert rebuilt.author_info.address == "+15559998888"
        assert rebuilt.ai_agent_info is not None
        assert rebuilt.ai_agent_info.participant_id == "PA_agent"

    @pytest.mark.asyncio
    async def test_rebuild_takes_the_voice_customer_not_the_first_customer(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        ended: list[ConversationSession] = []
        tac.on_conversation_ended(lambda s: ended.append(s))
        tac.conversation_orchestrator_client.list_participants = AsyncMock(
            return_value=mixed_channel_participants("conv_mixed")
        )

        await channel.process_webhook(closed_webhook("conv_mixed"))

        assert len(ended) == 1
        rebuilt = ended[0]
        assert rebuilt.profile_id == "profile_caller"
        assert rebuilt.call_sid == "CA_customer"
        assert rebuilt.author_info is not None
        assert rebuilt.author_info.address == "+15559998888"
        assert rebuilt.author_info.participant_id == "PA_customer"

    @pytest.mark.asyncio
    async def test_fires_once_after_local_teardown_on_the_same_instance(self) -> None:
        """The common single-instance path: teardown, then CLOSED arrives here."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        ended: list[ConversationSession] = []
        tac.on_conversation_ended(lambda s: ended.append(s))

        channel._start_conversation("conv_3", None)
        await channel._provider._cleanup_connection("conv_3")
        assert ended == []

        tac.conversation_orchestrator_client.list_participants = AsyncMock(
            return_value=voice_participants("conv_3")
        )
        await channel.process_webhook(closed_webhook("conv_3"))

        assert len(ended) == 1

    @pytest.mark.asyncio
    async def test_no_callback_registered_costs_no_api_call(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        list_participants = AsyncMock(return_value=voice_participants("conv_4"))
        tac.conversation_orchestrator_client.list_participants = list_participants

        await channel.process_webhook(closed_webhook("conv_4"))

        list_participants.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ignores_a_conversation_with_no_voice_participant(self) -> None:
        """A CHAT conversation closing must not fire the voice channel's hook."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        ended: list[ConversationSession] = []
        tac.on_conversation_ended(lambda s: ended.append(s))

        tac.conversation_orchestrator_client.list_participants = AsyncMock(
            return_value=[
                ParticipantResponse(
                    id="PA_chat",
                    conversation_id="conv_chat",
                    account_id="ACtest123",
                    name="Chat User",
                    type="CUSTOMER",
                    addresses=[ParticipantAddress(channel="CHAT", address="user-1")],
                )
            ]
        )
        await channel.process_webhook(closed_webhook("conv_chat"))

        assert ended == []

    @pytest.mark.asyncio
    async def test_ignores_another_configurations_conversation(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        ended: list[ConversationSession] = []
        tac.on_conversation_ended(lambda s: ended.append(s))
        list_participants = AsyncMock(return_value=voice_participants("conv_other"))
        tac.conversation_orchestrator_client.list_participants = list_participants

        webhook = closed_webhook("conv_other")
        webhook["data"]["configurationId"] = "conv_configuration_someone_else"
        await channel.process_webhook(webhook)

        assert ended == []
        list_participants.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_participant_lookup_failure_is_survivable(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        ended: list[ConversationSession] = []
        tac.on_conversation_ended(lambda s: ended.append(s))
        tac.conversation_orchestrator_client.list_participants = AsyncMock(
            side_effect=RuntimeError("CO unreachable")
        )

        await channel.process_webhook(closed_webhook("conv_5"))

        assert ended == []

    @pytest.mark.asyncio
    async def test_rebuilt_call_sid_is_none_without_a_voice_channel_id(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        ended: list[ConversationSession] = []
        tac.on_conversation_ended(lambda s: ended.append(s))

        tac.conversation_orchestrator_client.list_participants = AsyncMock(
            return_value=voice_participants("conv_no_sid")
        )
        await channel.process_webhook(closed_webhook("conv_no_sid"))

        assert len(ended) == 1
        assert ended[0].call_sid is None

    @pytest.mark.asyncio
    async def test_prefers_the_customers_call_sid_over_the_agents(self) -> None:
        """The customer's VOICE channelId wins even if the agent is listed first."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        participants = [
            ParticipantResponse(
                id="PA_agent",
                conversation_id="conv_both",
                account_id="ACtest123",
                name="TAC Agent",
                type="AI_AGENT",
                addresses=[
                    ParticipantAddress(
                        channel="VOICE", address="+15551234567", channel_id="CA_agent"
                    )
                ],
            ),
            ParticipantResponse(
                id="PA_customer",
                conversation_id="conv_both",
                account_id="ACtest123",
                name="Caller",
                type="CUSTOMER",
                profile_id="profile_caller",
                addresses=[
                    ParticipantAddress(
                        channel="VOICE", address="+15559998888", channel_id="CA_customer"
                    )
                ],
            ),
        ]
        tac.conversation_orchestrator_client.list_participants = AsyncMock(
            return_value=participants
        )

        session = await channel._rebuild_session("conv_both")

        assert session is not None
        assert session.call_sid == "CA_customer"

    @pytest.mark.asyncio
    async def test_falls_back_to_the_agents_call_sid_when_the_customer_has_none(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        participants = [
            ParticipantResponse(
                id="PA_customer",
                conversation_id="conv_agent_only",
                account_id="ACtest123",
                name="Caller",
                type="CUSTOMER",
                profile_id="profile_caller",
                addresses=[ParticipantAddress(channel="VOICE", address="+15559998888")],
            ),
            ParticipantResponse(
                id="PA_agent",
                conversation_id="conv_agent_only",
                account_id="ACtest123",
                name="TAC Agent",
                type="AI_AGENT",
                addresses=[
                    ParticipantAddress(
                        channel="VOICE", address="+15551234567", channel_id="CA_agent"
                    )
                ],
            ),
        ]
        tac.conversation_orchestrator_client.list_participants = AsyncMock(
            return_value=participants
        )

        session = await channel._rebuild_session("conv_agent_only")

        assert session is not None
        assert session.call_sid == "CA_agent"

    @pytest.mark.asyncio
    async def test_explicit_call_sid_overrides_the_derived_one(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        tac.conversation_orchestrator_client.list_participants = AsyncMock(
            return_value=voice_participants("conv_x", call_sid="CA_derived")
        )

        session = await channel._rebuild_session("conv_x", "CA_explicit")

        assert session is not None
        assert session.call_sid == "CA_explicit"

    @pytest.mark.asyncio
    async def test_empty_string_channel_id_yields_no_call_sid(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        participants = [
            ParticipantResponse(
                id="PA_customer",
                conversation_id="conv_empty",
                account_id="ACtest123",
                name="Caller",
                type="CUSTOMER",
                profile_id="profile_caller",
                addresses=[
                    ParticipantAddress(channel="VOICE", address="+15559998888", channel_id="")
                ],
            ),
            ParticipantResponse(
                id="PA_agent",
                conversation_id="conv_empty",
                account_id="ACtest123",
                name="TAC Agent",
                type="AI_AGENT",
                addresses=[
                    ParticipantAddress(channel="VOICE", address="+15551234567", channel_id="")
                ],
            ),
        ]
        tac.conversation_orchestrator_client.list_participants = AsyncMock(
            return_value=participants
        )

        session = await channel._rebuild_session("conv_empty")

        assert session is not None
        assert session.call_sid is None

    @pytest.mark.asyncio
    async def test_relay_only_closed_webhook_does_not_double_fire(self) -> None:
        """Relay-only already fired at teardown; a CLOSED must not fire again."""
        tac = TAC(get_test_config(conversation_configuration_id=None))
        channel = VoiceChannel(tac)
        ended: list[ConversationSession] = []
        tac.on_conversation_ended(lambda s: ended.append(s))

        channel._start_conversation("CA_relay2", None)
        await channel._provider._cleanup_connection("CA_relay2")
        assert len(ended) == 1

        await channel.process_webhook(closed_webhook("CA_relay2"))

        assert len(ended) == 1


class TestClosedDuringLiveCall:
    """CO can close a conversation while its call is still live (a closed
    timeout during a long hold, or the app closing it). The call keeps
    running, so the session must survive until teardown."""

    @pytest.mark.asyncio
    async def test_fires_conversation_ended_now_and_keeps_the_session(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        call_ended: list[ConversationSession] = []
        conversation_ended: list[ConversationSession] = []
        channel.on_call_ended(collect(call_ended))
        tac.on_conversation_ended(lambda s: conversation_ended.append(s))
        tac.conversation_orchestrator_client.list_participants = AsyncMock()

        live = channel._start_conversation("conv_live", "profile_caller")
        live.call_sid = "CA_live"
        await channel.process_webhook(closed_webhook("conv_live"))

        assert [s.conversation_id for s in conversation_ended] == ["conv_live"]
        assert conversation_ended[0].call_sid == "CA_live"
        assert channel._conversations["conv_live"] is live
        assert call_ended == []
        # The live session is used as-is; no rebuild from CO.
        tac.conversation_orchestrator_client.list_participants.assert_not_awaited()

        await channel._provider._cleanup_connection("conv_live")

        assert [s.conversation_id for s in call_ended] == ["conv_live"]
        assert len(conversation_ended) == 1
        assert_no_residual_state(channel, "conv_live")

    @pytest.mark.asyncio
    async def test_callback_gets_a_snapshot_of_the_live_session(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        conversation_ended: list[ConversationSession] = []
        tac.on_conversation_ended(lambda s: conversation_ended.append(s))

        live = channel._start_conversation("conv_snap", None)
        live.metadata["transcript"] = ["hello"]
        await channel.process_webhook(closed_webhook("conv_snap"))
        live.metadata["late_key"] = "written after CLOSED"

        snapshot = conversation_ended[0]
        assert snapshot is not live
        assert snapshot.metadata == {"transcript": ["hello"]}

    @pytest.mark.asyncio
    async def test_callback_mutating_nested_data_leaves_the_live_session_unchanged(
        self,
    ) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        live = channel._start_conversation("conv_nested", "profile_caller")
        live.metadata["transcript"] = [{"role": "user", "text": "hello"}]
        live.author_info = AuthorInfo(address="+15559998888", participant_id="PA_customer")
        live.pending_handoff_data = PendingHandoffData(handoffData='{"reason": "agent"}')
        held: list[ConversationSession] = []

        def on_ended(snapshot: ConversationSession) -> None:
            held.append(snapshot)
            snapshot.metadata["transcript"].clear()
            assert snapshot.author_info is not None
            snapshot.author_info.address = "changed"
            assert snapshot.pending_handoff_data is not None
            snapshot.pending_handoff_data.handoff_data = "changed"

        tac.on_conversation_ended(on_ended)
        await channel.process_webhook(closed_webhook("conv_nested"))

        # The callback's edits don't reach the live call...
        assert live.metadata["transcript"] == [{"role": "user", "text": "hello"}]
        assert live.author_info.address == "+15559998888"
        assert live.pending_handoff_data.handoff_data == '{"reason": "agent"}'
        # ...and the live call's later writes don't reach the snapshot it kept.
        live.metadata["transcript"].append({"role": "assistant", "text": "still here"})
        assert held[0].metadata["transcript"] == []
        # The snapshot doesn't hold the live call's lock.
        assert held[0].cache_lock is not live.cache_lock

    @pytest.mark.asyncio
    async def test_uncopyable_metadata_still_fires_the_hook(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        live = channel._start_conversation("conv_uncopyable", None)
        app_handle = threading.Lock()  # deepcopy raises TypeError on this
        live.metadata["handle"] = app_handle
        live.metadata["transcript"] = ["hello"]
        ended: list[ConversationSession] = []
        tac.on_conversation_ended(lambda s: ended.append(s))

        await channel.process_webhook(closed_webhook("conv_uncopyable"))

        assert len(ended) == 1
        assert ended[0].metadata["handle"] is app_handle
        assert ended[0].metadata["transcript"] == ["hello"]
        assert ended[0].metadata["transcript"] is not live.metadata["transcript"]

    @pytest.mark.asyncio
    async def test_end_call_after_the_live_close_still_fires_call_ended_once(self) -> None:
        """Hanging up after a mid-call CLOSED still releases the session
        normally: on_call_ended fires once, and on_conversation_ended isn't
        fired again (it already fired from the live CLOSED path)."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        call_ended: list[ConversationSession] = []
        conversation_ended: list[ConversationSession] = []
        channel.on_call_ended(collect(call_ended))
        tac.on_conversation_ended(lambda s: conversation_ended.append(s))

        live = channel._start_conversation("conv_end_live", "profile_caller")
        live.call_sid = "CA_end_live"
        await channel.process_webhook(closed_webhook("conv_end_live"))
        assert len(conversation_ended) == 1

        with patch.object(channel, "_get_twilio_client", return_value=MagicMock()):
            await channel.end_call("CA_end_live")

        assert [s.conversation_id for s in call_ended] == ["conv_end_live"]
        assert len(conversation_ended) == 1
        assert_no_residual_state(channel, "conv_end_live")

    @pytest.mark.asyncio
    async def test_a_raising_callback_leaves_the_session_for_teardown(self) -> None:
        """A developer's on_conversation_ended raising on the live path must
        not prevent the session from staying for the call's own teardown."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        call_ended: list[ConversationSession] = []
        channel.on_call_ended(collect(call_ended))

        async def boom(session: ConversationSession) -> None:
            raise RuntimeError("handler exploded")

        tac.on_conversation_ended(boom)

        live = channel._start_conversation("conv_raise", None)
        live.call_sid = "CA_raise"
        await channel.process_webhook(closed_webhook("conv_raise"))

        assert channel._conversations["conv_raise"] is live

        await channel._provider._cleanup_connection("conv_raise")

        assert [s.conversation_id for s in call_ended] == ["conv_raise"]
        assert_no_residual_state(channel, "conv_raise")


class TestTeardownInvariant:
    """No failure mode may leave residue behind (§3.2)."""

    @staticmethod
    def _channel() -> VoiceChannel:
        return VoiceChannel(
            TAC(get_test_config(conversation_configuration_id=None)),
            config=ConversationRelayProviderConfig(session_manager=ThreadSafeSessionManager()),
        )

    @staticmethod
    def _websocket(messages: list[dict], *, error: Exception | None = None) -> Any:
        """A websocket yielding ``messages`` then raising ``error`` (disconnect by default)."""
        queue = list(messages)
        ws = AsyncMock()

        async def receive_json() -> dict:
            if queue:
                return queue.pop(0)
            raise error or WebSocketDisconnectError()

        ws.receive_json = receive_json
        return ws

    @pytest.mark.asyncio
    async def test_normal_hangup(self) -> None:
        channel = self._channel()
        await channel.handle_websocket(
            self._websocket(
                [
                    {"type": "setup", "callSid": "CA_a", "from": "+15559998888"},
                    {"type": "prompt", "voicePrompt": "hi", "final": True},
                ]
            )
        )
        assert_no_residual_state(channel, "CA_a")

    @pytest.mark.asyncio
    async def test_disconnect_before_any_prompt(self) -> None:
        channel = self._channel()
        await channel.handle_websocket(
            self._websocket([{"type": "setup", "callSid": "CA_b", "from": "+15559998888"}])
        )
        assert_no_residual_state(channel, "CA_b")

    @pytest.mark.asyncio
    async def test_abrupt_error_mid_stream(self) -> None:
        channel = self._channel()
        await channel.handle_websocket(
            self._websocket(
                [
                    {"type": "setup", "callSid": "CA_c", "from": "+15559998888"},
                    {"type": "prompt", "voicePrompt": "hi", "final": True},
                ],
                error=RuntimeError("socket blew up"),
            )
        )
        assert_no_residual_state(channel, "CA_c")

    @pytest.mark.asyncio
    async def test_message_ready_callback_raising(self) -> None:
        channel = self._channel()

        async def boom(message: str, session: ConversationSession, memory: Any) -> str:
            raise RuntimeError("LLM exploded")

        channel.tac.on_message_ready(boom)
        await channel.handle_websocket(
            self._websocket(
                [
                    {"type": "setup", "callSid": "CA_d", "from": "+15559998888"},
                    {"type": "prompt", "voicePrompt": "hi", "final": True},
                ]
            )
        )
        assert_no_residual_state(channel, "CA_d")

    @pytest.mark.asyncio
    async def test_initialize_conversation_raising(self) -> None:
        """Orchestrated mode where the CO lookup fails outright."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        tac.conversation_orchestrator_client.list_conversations = AsyncMock(
            side_effect=RuntimeError("CO unreachable")
        )

        await channel.handle_websocket(
            self._websocket(
                [
                    {"type": "setup", "callSid": "CA_e", "from": "+15559998888"},
                    {"type": "prompt", "voicePrompt": "hi", "final": True},
                ]
            )
        )
        assert channel._conversations == {}

    @pytest.mark.asyncio
    async def test_leak_loop(self) -> None:
        """Many connect/disconnect cycles leave nothing behind in aggregate."""
        channel = self._channel()
        for i in range(25):
            await channel.handle_websocket(
                self._websocket(
                    [
                        {"type": "setup", "callSid": f"CA_loop_{i}", "from": "+15559998888"},
                        {"type": "prompt", "voicePrompt": "hi", "final": True},
                    ]
                )
            )
            assert_no_residual_state(channel, f"CA_loop_{i}")

        assert channel._conversations == {}
        assert len(channel._provider._websocket_manager) == 0


class TestDrain:
    @pytest.mark.asyncio
    async def test_releases_sessions_still_live_at_shutdown(self) -> None:
        tac = TAC(get_test_config(conversation_configuration_id=None))
        channel = VoiceChannel(tac)
        call_ended: list[ConversationSession] = []
        conversation_ended: list[ConversationSession] = []
        channel.on_call_ended(collect(call_ended))
        tac.on_conversation_ended(lambda s: conversation_ended.append(s))

        channel._start_conversation("CA_drain_1", None)
        channel._start_conversation("CA_drain_2", None)

        await channel.aclose(grace_period=0)

        assert {s.conversation_id for s in call_ended} == {"CA_drain_1", "CA_drain_2"}
        assert len(conversation_ended) == 2
        assert channel._conversations == {}

    @pytest.mark.asyncio
    async def test_waits_for_a_call_that_ends_on_its_own(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        channel._start_conversation("CA_drain_3", None)

        async def hang_up_shortly() -> None:
            await asyncio.sleep(0.05)
            await channel._release_session("CA_drain_3")

        asyncio.create_task(hang_up_shortly())
        await channel.aclose(grace_period=5)

        assert channel._conversations == {}

    @pytest.mark.asyncio
    async def test_waits_for_a_call_still_initializing(self) -> None:
        """An admitted WebSocket still in setup (e.g. the background CO lookup)
        has no session yet. The drain must wait for it rather than return while
        it can still create one."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        call_ended: list[ConversationSession] = []
        channel.on_call_ended(collect(call_ended))
        setup_done = asyncio.Event()

        async def slow_handler(websocket: Any) -> None:
            await setup_done.wait()  # still initializing: no session yet
            channel._start_conversation("conv_late", None)
            await channel._release_session("conv_late")

        channel._provider.handle_websocket = slow_handler  # type: ignore[method-assign]
        handler = asyncio.create_task(channel.handle_websocket(MagicMock()))
        await asyncio.sleep(0)  # admitted, waiting in setup
        assert channel._conversations == {}

        async def finish_setup_shortly() -> None:
            await asyncio.sleep(0.05)
            setup_done.set()

        asyncio.create_task(finish_setup_shortly())
        await channel.aclose(grace_period=5)

        assert handler.done()
        assert [s.conversation_id for s in call_ended] == ["conv_late"]
        assert channel._conversations == {}

    @pytest.mark.asyncio
    async def test_cancels_a_call_still_initializing_after_the_grace_period(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        never = asyncio.Event()

        async def stuck_handler(websocket: Any) -> None:
            await never.wait()

        channel._provider.handle_websocket = stuck_handler  # type: ignore[method-assign]
        handler = asyncio.create_task(channel.handle_websocket(MagicMock()))
        await asyncio.sleep(0)

        await channel.aclose(grace_period=0.1)

        assert handler.done()
        assert handler.cancelled()
        assert channel._conversations == {}

    @pytest.mark.asyncio
    async def test_waits_for_a_blocked_call_ended_callback(self) -> None:
        """`_release_session` pops the session before awaiting `on_call_ended`,
        so an empty session store doesn't mean teardown has finished."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        archived: list[str] = []
        release_callback = asyncio.Event()

        async def archive(session: ConversationSession) -> None:
            await release_callback.wait()
            archived.append(session.conversation_id)

        channel.on_call_ended(archive)
        channel._start_conversation("conv_archive", None)
        teardown = asyncio.create_task(channel._release_session("conv_archive"))
        await asyncio.sleep(0)  # popped, now blocked in on_call_ended
        assert "conv_archive" not in channel._conversations

        async def finish_archiving_shortly() -> None:
            await asyncio.sleep(0.05)
            release_callback.set()

        asyncio.create_task(finish_archiving_shortly())
        await channel.aclose(grace_period=5)

        assert archived == ["conv_archive"]
        assert teardown.done()

    @pytest.mark.asyncio
    async def test_force_close_tears_down_a_relay_call(self) -> None:
        """A call still held after the grace period is torn down at the
        transport too: socket closed and unregistered, stream task cancelled,
        session-manager state removed — not just the channel's session."""
        tac = TAC(get_test_config())
        session_manager = ThreadSafeSessionManager()
        channel = VoiceChannel(
            tac, config=ConversationRelayProviderConfig(session_manager=session_manager)
        )
        call_ended: list[ConversationSession] = []
        channel.on_call_ended(collect(call_ended))
        websocket = AsyncMock()
        channel._provider._websocket_manager.add_websocket("conv_force", websocket)
        state = session_manager.get_or_create_session("conv_force")
        state.stream_task = asyncio.create_task(asyncio.sleep(3600))
        channel._start_conversation("conv_force", None)

        await channel.aclose(grace_period=0)

        websocket.close.assert_awaited_once()
        assert state.stream_task.cancelled()
        assert [s.conversation_id for s in call_ended] == ["conv_force"]
        assert_no_residual_state(channel, "conv_force")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["openai_realtime", "gpt_live"])
    async def test_force_close_tears_down_a_media_streams_call(self, kind: str) -> None:
        tac = TAC(get_test_config())
        config: Any
        call_state: Any
        if kind == "gpt_live":
            config = GPTLiveProviderConfig(
                openai_api_key="sk-test", default_session_config={"model": "gpt-live-1"}
            )
            call_state = GPTLiveCallState
        else:
            config = OpenAIRealtimeProviderConfig(openai_api_key="sk-test")
            call_state = RealtimeCallState
        channel = VoiceChannel(tac, config=config)
        provider: Any = channel._provider
        twilio_ws = AsyncMock()
        model_ws = AsyncMock()
        call = call_state(twilio_ws=twilio_ws, model_ws=model_ws)
        if kind == "gpt_live":
            call.closed_event.set()  # the model already acknowledged session.close
        provider._calls["CA_force"] = call
        provider._call_session_configs["CA_force"] = {"model": "per-call"}
        channel._start_conversation("CA_force", None)

        await channel.aclose(grace_period=0)

        twilio_ws.close.assert_awaited_once()
        model_ws.close.assert_awaited_once()
        assert_no_residual_state(channel, "CA_force")

    @pytest.mark.asyncio
    async def test_teardown_after_a_force_close_is_a_no_op(self) -> None:
        """A stuck handler that finishes after the force-close runs its own
        cleanup again; that must not fire the end-of-call hooks twice."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        call_ended: list[ConversationSession] = []
        channel.on_call_ended(collect(call_ended))
        channel._provider._websocket_manager.add_websocket("conv_twice", AsyncMock())
        channel._start_conversation("conv_twice", None)

        await channel.aclose(grace_period=0)
        with patch("tac.channels.voice.conversation_relay.provider.track_event") as tracked:
            await channel._provider._cleanup_connection("conv_twice")

        assert len(call_ended) == 1
        tracked.assert_not_called()

    @pytest.mark.asyncio
    async def test_refuses_new_calls_once_draining(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        await channel.aclose(grace_period=0)

        websocket = AsyncMock()
        await channel.handle_websocket(websocket)

        websocket.close.assert_awaited_once()
        websocket.accept.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_is_idempotent(self) -> None:
        channel = VoiceChannel(TAC(get_test_config()))
        await channel.aclose(grace_period=0)
        await channel.aclose(grace_period=0)


class TestServerDrainWiring:
    def test_shutdown_handler_registered(self) -> None:
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        server = TACFastAPIServer(tac=tac, voice_channel=channel)

        assert server.aclose in server.app.router.on_shutdown

    @pytest.mark.asyncio
    async def test_aclose_drains_the_voice_channel(self) -> None:
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        server = TACFastAPIServer(tac=tac, voice_channel=channel)
        with patch.object(channel, "aclose", new=AsyncMock()) as aclose:
            await server.aclose()
        aclose.assert_awaited_once_with(grace_period=server.config.shutdown_grace_period)

    @pytest.mark.asyncio
    async def test_aclose_without_a_voice_channel_is_a_noop(self) -> None:
        from tac.channels.sms import SMSChannel
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        server = TACFastAPIServer(tac=tac, messaging_channels=[SMSChannel(tac)])
        await server.aclose()


class TestEndCallDoesNotDoubleFire:
    @pytest.mark.asyncio
    async def test_orchestrated_end_call_leaves_conversation_ended_to_the_webhook(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        conversation_ended: list[ConversationSession] = []
        tac.on_conversation_ended(lambda s: conversation_ended.append(s))

        session = channel._start_conversation("conv_end", None)
        session.call_sid = "CA_end"

        with patch.object(channel, "_get_twilio_client", return_value=MagicMock()):
            await channel.end_call("CA_end")

        assert conversation_ended == []

        tac.conversation_orchestrator_client.list_participants = AsyncMock(
            return_value=voice_participants("conv_end")
        )
        await channel.process_webhook(closed_webhook("conv_end"))

        assert len(conversation_ended) == 1


class TestInstanceAffinity:
    """Every URL TAC hands Twilio for a live call points at the process
    holding that call, when the deployment can address one."""

    @staticmethod
    def _channel(**overrides: Any) -> VoiceChannel:
        tac = TAC(get_test_config(**overrides))
        channel = VoiceChannel(tac)
        # Registering the handlers is what makes TAC pass the callback URLs.
        channel.on_call_status(AsyncMock())
        channel.on_amd(AsyncMock())
        channel.on_recording(AsyncMock())
        return channel

    @pytest.mark.asyncio
    async def test_inbound_twiml_uses_the_instance_domain(self) -> None:
        channel = self._channel(instance_public_domain="pod-7.voice.svc.cluster.local")

        twiml = await channel.handle_incoming_call()

        assert 'url="wss://pod-7.voice.svc.cluster.local/ws"' in twiml
        assert 'action="https://pod-7.voice.svc.cluster.local' in twiml
        assert "example.com" not in twiml

    def test_call_event_urls_use_the_instance_domain(self) -> None:
        channel = self._channel(instance_public_domain="pod-7.voice.svc.cluster.local")

        for kind in ("status", "amd", "recording"):
            url = channel.tac.config.call_event_url(kind)  # type: ignore[arg-type]
            assert url is not None
            assert url.startswith("https://pod-7.voice.svc.cluster.local/")

    @pytest.mark.asyncio
    async def test_outbound_call_pins_every_callback_to_this_instance(self) -> None:
        channel = self._channel(instance_public_domain="pod-7.voice.svc.cluster.local")
        twilio_client = MagicMock()
        twilio_client.calls.create.return_value = MagicMock(sid="CA_out")

        from tac.models.outbound import InitiateVoiceConversationOptions

        with patch.object(channel, "_get_twilio_client", return_value=twilio_client):
            await channel.initiate_outbound_conversation(
                InitiateVoiceConversationOptions(to="+15559998888")
            )

        kwargs = twilio_client.calls.create.call_args.kwargs
        assert "wss://pod-7.voice.svc.cluster.local/ws" in kwargs["twiml"]
        for param in ("status_callback", "async_amd_status_callback", "recording_status_callback"):
            assert kwargs[param].startswith("https://pod-7.voice.svc.cluster.local/")

    @pytest.mark.asyncio
    async def test_shared_domain_remains_the_default(self) -> None:
        """Deployments without per-pod addressing are unaffected."""
        channel = self._channel()

        twiml = await channel.handle_incoming_call()

        assert 'url="wss://example.com/ws"' in twiml
        assert channel.tac.config.call_event_url("status") == (
            "https://example.com/twilio/call-events/status"
        )

    def test_scheme_and_trailing_slash_are_stripped(self) -> None:
        channel = self._channel(instance_public_domain="https://pod-7.example.com/")

        assert channel.tac.config.instance_public_domain == "pod-7.example.com"


class TestResolveSessionByCallSid:
    """A CallSid-only event can find its session's identity on any instance."""

    @pytest.mark.asyncio
    async def test_returns_the_live_session_when_held_locally(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        tac.conversation_orchestrator_client.list_conversations = AsyncMock()
        live = channel._start_conversation("conv_local", None)
        live.call_sid = "CA_local"

        assert await channel.resolve_conversation_session_by_call_sid("CA_local") is live
        tac.conversation_orchestrator_client.list_conversations.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_the_live_session_as_is_after_a_mid_call_close(self) -> None:
        """On the instance holding the call, a mid-call CLOSED still leaves
        the live session reachable under the old conversation id; CO's new
        ACTIVE conversation for the call isn't consulted."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        live = channel._start_conversation("conv_old", None)
        live.call_sid = "CA_hold"
        await channel.process_webhook(closed_webhook("conv_old"))

        client = tac.conversation_orchestrator_client
        client.list_conversations = AsyncMock(
            return_value=[conversation("conv_new", "ACTIVE", "2026-09-30T10:20:00Z")]
        )

        session = await channel.resolve_conversation_session_by_call_sid("CA_hold")

        assert session is live
        assert session.conversation_id == "conv_old"
        client.list_conversations.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rebuilds_from_co_when_held_elsewhere(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        client = tac.conversation_orchestrator_client
        client.list_conversations = AsyncMock(
            return_value=[conversation("conv_remote", "ACTIVE", "2026-09-30T10:00:00Z")]
        )
        client.list_participants = AsyncMock(return_value=voice_participants("conv_remote"))

        session = await channel.resolve_conversation_session_by_call_sid("CA_remote")

        assert session is not None
        assert session.conversation_id == "conv_remote"
        assert session.call_sid == "CA_remote"
        assert session.profile_id == "profile_caller"
        client.list_conversations.assert_awaited_once_with(channel_id="CA_remote")
        # Identity only: nothing is tracked locally.
        assert "conv_remote" not in channel._conversations

    @pytest.mark.asyncio
    async def test_rebuild_takes_the_voice_customer_not_the_first_customer(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        client = tac.conversation_orchestrator_client
        client.list_conversations = AsyncMock(
            return_value=[conversation("conv_mixed", "ACTIVE", "2026-09-30T10:00:00Z")]
        )
        client.list_participants = AsyncMock(return_value=mixed_channel_participants("conv_mixed"))

        session = await channel.resolve_conversation_session_by_call_sid("CA_customer")

        assert session is not None
        assert session.profile_id == "profile_caller"
        assert session.author_info is not None
        assert session.author_info.participant_id == "PA_customer"

    @pytest.mark.asyncio
    async def test_participants_are_never_served_from_the_shared_cache(self) -> None:
        """Early in a call participants still change (profile_id, customer
        joining), so each lookup must see fresh data."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        client = tac.conversation_orchestrator_client
        client.list_conversations = AsyncMock(
            return_value=[conversation("conv_remote", "ACTIVE", "2026-09-30T10:00:00Z")]
        )

        def participants_with(profile_id: str) -> list[ParticipantResponse]:
            return [
                p.model_copy(update={"profile_id": profile_id}) if p.type == "CUSTOMER" else p
                for p in voice_participants("conv_remote")
            ]

        client.list_participants = AsyncMock(return_value=participants_with("profile_old"))
        first = await channel.resolve_conversation_session_by_call_sid("CA_remote")
        client.list_participants.return_value = participants_with("profile_new")
        second = await channel.resolve_conversation_session_by_call_sid("CA_remote")

        assert first is not None and first.profile_id == "profile_old"
        assert second is not None and second.profile_id == "profile_new"
        assert client.list_participants.await_count == 2

    @pytest.mark.asyncio
    async def test_prefers_the_active_conversation_over_a_closed_one(self) -> None:
        """After a mid-call CLOSED, CO starts a new conversation for the same call."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        client = tac.conversation_orchestrator_client
        client.list_conversations = AsyncMock(
            return_value=[
                conversation("conv_old", "CLOSED", "2026-09-30T10:00:00Z"),
                conversation("conv_new", "ACTIVE", "2026-09-30T10:20:00Z"),
            ]
        )
        client.list_participants = AsyncMock(
            side_effect=lambda conv_id: voice_participants(conv_id)
        )

        session = await channel.resolve_conversation_session_by_call_sid("CA_x")

        assert session is not None
        assert session.conversation_id == "conv_new"

    @pytest.mark.asyncio
    async def test_falls_back_to_the_newest_when_none_is_active(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        client = tac.conversation_orchestrator_client
        client.list_conversations = AsyncMock(
            return_value=[
                conversation("conv_old", "CLOSED", "2026-09-30T10:00:00Z"),
                conversation("conv_newer", "CLOSED", "2026-09-30T10:20:00Z"),
            ]
        )
        client.list_participants = AsyncMock(
            side_effect=lambda conv_id: voice_participants(conv_id)
        )

        session = await channel.resolve_conversation_session_by_call_sid("CA_y")

        assert session is not None
        assert session.conversation_id == "conv_newer"

    @pytest.mark.asyncio
    async def test_none_when_co_knows_no_conversation(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        tac.conversation_orchestrator_client.list_conversations = AsyncMock(return_value=[])

        assert await channel.resolve_conversation_session_by_call_sid("CA_none") is None

    @pytest.mark.asyncio
    async def test_none_when_the_lookup_fails(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        tac.conversation_orchestrator_client.list_conversations = AsyncMock(
            side_effect=RuntimeError("CO down")
        )

        assert await channel.resolve_conversation_session_by_call_sid("CA_err") is None

    @pytest.mark.asyncio
    async def test_relay_only_mode_has_no_fallback(self) -> None:
        tac = TAC(get_test_config(conversation_configuration_id=None))
        channel = VoiceChannel(tac)

        assert await channel.resolve_conversation_session_by_call_sid("CA_relay") is None

    @pytest.mark.asyncio
    async def test_picks_the_later_conversation_by_parsed_datetime_not_string(self) -> None:
        """As strings, "...10:00:00Z" sorts after "...10:00:00.500Z", but
        before it as a datetime; the fix must compare parsed datetimes."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        client = tac.conversation_orchestrator_client
        client.list_conversations = AsyncMock(
            return_value=[
                conversation("conv_a", "CLOSED", "2026-09-30T10:00:00Z"),
                conversation("conv_b", "CLOSED", "2026-09-30T10:00:00.500Z"),
            ]
        )
        client.list_participants = AsyncMock(
            side_effect=lambda conv_id: voice_participants(conv_id)
        )

        session = await channel.resolve_conversation_session_by_call_sid("CA_z")

        assert session is not None
        assert session.conversation_id == "conv_b"

    @pytest.mark.asyncio
    async def test_media_streams_provider_has_no_fallback(self) -> None:
        """Media Streams providers have no CO conversation behind the call,
        even in orchestrated mode: don't bother asking CO."""
        tac = TAC(get_test_config())
        config = OpenAIRealtimeProviderConfig(
            openai_api_key="sk-test",
            default_session_config={
                "model": "gpt-realtime-test",
                "audio": {
                    "input": {"format": TWILIO_AUDIO_FORMAT_FOR_REALTIME},
                    "output": {"format": TWILIO_AUDIO_FORMAT_FOR_REALTIME},
                },
            },
        )
        channel = VoiceChannel(tac, config=config)
        client = tac.conversation_orchestrator_client
        client.list_conversations = AsyncMock()

        assert await channel.resolve_conversation_session_by_call_sid("CA_media") is None
        client.list_conversations.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_none_when_no_participant_is_on_the_voice_channel(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        client = tac.conversation_orchestrator_client
        client.list_conversations = AsyncMock(
            return_value=[conversation("conv_other", "ACTIVE", "2026-09-30T10:00:00Z")]
        )
        client.list_participants = AsyncMock(
            return_value=[
                ParticipantResponse(
                    id="PA_customer",
                    conversation_id="conv_other",
                    account_id="ACtest123",
                    name="Caller",
                    type="CUSTOMER",
                    profile_id="profile_caller",
                    addresses=[ParticipantAddress(channel="SMS", address="+15559998888")],
                )
            ]
        )

        assert await channel.resolve_conversation_session_by_call_sid("CA_nonvoice") is None

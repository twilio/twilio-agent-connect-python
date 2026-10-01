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
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tac import TAC
from tac.channels.voice import VoiceChannel
from tac.channels.voice.conversation_relay import ConversationRelayProviderConfig
from tac.channels.websocket_protocol import WebSocketDisconnectError
from tac.models.conversation import ParticipantAddress, ParticipantResponse
from tac.models.session import ConversationSession
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


def closed_webhook(conv_id: str, call_sid: str | None = None) -> dict:
    data: dict[str, Any] = {
        "id": conv_id,
        "status": "CLOSED",
        "configurationId": CONFIGURATION_ID,
    }
    if call_sid is not None:
        data["channelId"] = call_sid
    return {"eventType": "CONVERSATION_UPDATED", "data": data}


def collect(sink: list[ConversationSession]) -> Any:
    """An async handler that records the session it's given."""

    async def handler(session: ConversationSession) -> None:
        sink.append(session)

    return handler


def voice_participants(conv_id: str) -> list[ParticipantResponse]:
    """A realistic post-call participant set: customer + TAC's AI_AGENT."""
    return [
        ParticipantResponse(
            id="PA_customer",
            conversation_id=conv_id,
            account_id="ACtest123",
            name="Caller",
            type="CUSTOMER",
            profile_id="profile_caller",
            addresses=[ParticipantAddress(channel="VOICE", address="+15559998888")],
        ),
        ParticipantResponse(
            id="PA_agent",
            conversation_id=conv_id,
            account_id="ACtest123",
            name="TAC Agent",
            type="AI_AGENT",
            addresses=[ParticipantAddress(channel="VOICE", address="+15551234567")],
        ),
    ]


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
            return_value=voice_participants("conv_shared")
        )
        await instance_b.process_webhook(closed_webhook("conv_shared", call_sid="CA_shared"))

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

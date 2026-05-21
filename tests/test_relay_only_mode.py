"""Tests for relay-only mode (TAC without Conversation Orchestrator).

Relay-only mode is active when TACConfig.conversation_configuration_id is not set.
In this mode:
- VoiceChannel works with pure ConversationRelay (no CO, no Memory).
- SMSChannel and ChatChannel raise at construction.
- TAC.retrieve_memory returns an empty TACMemoryResponse.
- The ConversationRelay callback for "completed" ends the local session without CO.
"""

from unittest.mock import AsyncMock

import pytest

from tac import TAC
from tac.channels.chat import ChatChannel
from tac.channels.sms import SMSChannel
from tac.channels.voice import VoiceChannel
from tac.models.session import ConversationSession
from tac.models.voice import ConversationRelayCallbackPayload


def relay_only_config() -> dict:
    """Config without conversation_configuration_id — triggers relay-only mode."""
    return {
        "account_sid": "AC123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_secret",
        "phone_number": "+15551234567",
        "voice_public_domain": "example.com",
    }


def orchestrated_config() -> dict:
    """Config with conversation_configuration_id — triggers orchestrated mode."""
    return {
        **relay_only_config(),
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": "conv_configuration_test123",
    }


class TestRelayOnlyMode:
    """Tests for TAC relay-only mode."""

    @pytest.mark.no_auto_mock
    def test_tac_init_without_config_succeeds(self) -> None:
        """TAC initializes without conversation_configuration_id and reports relay-only."""
        tac = TAC(relay_only_config())

        assert tac.is_orchestrator_enabled() is False
        assert tac.conversation_orchestrator_client is None
        assert tac.conversation_memory_client is None

    def test_tac_init_with_config_succeeds(self) -> None:
        """TAC with conversation_configuration_id reports orchestrator enabled.

        Uses the autouse mock_conversation_configuration fixture.
        """
        tac = TAC(orchestrated_config())

        assert tac.is_orchestrator_enabled() is True
        assert tac.conversation_orchestrator_client is not None
        assert tac.conversation_memory_client is not None

    @pytest.mark.no_auto_mock
    @pytest.mark.asyncio
    async def test_retrieve_memory_returns_empty_in_relay_only(self) -> None:
        """retrieve_memory returns empty TACMemoryResponse in relay-only mode."""
        tac = TAC(relay_only_config())

        session = ConversationSession(
            conversation_id="CA_relay_only",
            profile_id=None,
            channel="voice",
        )

        response = await tac.retrieve_memory(session)

        assert response.has_memory_features is False
        assert response.observations == []
        assert response.summaries == []
        assert response.communications == []

    @pytest.mark.no_auto_mock
    @pytest.mark.asyncio
    async def test_handle_incoming_call_twiml_omits_conversation_configuration(self) -> None:
        """TwiML does not include conversationConfiguration in relay-only mode."""
        from tac.channels.voice import VoiceChannelConfig
        from tac.models.voice import TwiMLOptions

        tac = TAC(relay_only_config())
        channel = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                default_twiml_options=TwiMLOptions(welcome_greeting="Hello"),
            ),
        )

        twiml = await channel.handle_incoming_call()

        assert "conversationConfiguration" not in twiml

    @pytest.mark.no_auto_mock
    @pytest.mark.asyncio
    async def test_handle_websocket_first_prompt_uses_call_sid(self) -> None:
        """First prompt in relay-only mode uses call_sid as conv_id without CO calls."""
        tac = TAC(relay_only_config())
        channel = VoiceChannel(tac)

        mock_ws = AsyncMock()
        mock_ws.receive_json.side_effect = [
            {"type": "setup", "callSid": "CA_relay_abc", "from": "+15551230000"},
            {
                "type": "prompt",
                "voicePrompt": "Hello there",
                "final": True,
            },
            Exception("stop-iteration"),
        ]

        callback_seen: dict = {}

        def on_message(user_message, session, memory):
            callback_seen["user_message"] = user_message
            callback_seen["session"] = session
            callback_seen["memory"] = memory
            return None

        tac.on_message_ready(on_message)

        await channel.handle_websocket(mock_ws)

        assert callback_seen["user_message"] == "Hello there"
        assert callback_seen["session"].conversation_id == "CA_relay_abc"
        assert callback_seen["session"].profile_id is None

    @pytest.mark.no_auto_mock
    @pytest.mark.asyncio
    async def test_call_completed_ends_local_session_without_co(self) -> None:
        """On call completed, relay-only mode ends local session without CO calls."""
        tac = TAC(relay_only_config())
        channel = VoiceChannel(tac)

        channel._start_conversation("CA_relay_xyz", None)
        assert "CA_relay_xyz" in channel._conversations

        ended: list[str] = []

        async def on_ended(session):
            ended.append(session.conversation_id)

        tac.on_conversation_ended(on_ended)

        payload = ConversationRelayCallbackPayload(
            CallSid="CA_relay_xyz",
            CallStatus="completed",
            AccountSid="AC123",
            From="+15551230000",
            To="+15551234567",
            Direction="inbound",
        ).model_dump(by_alias=True, exclude_none=True)
        payload = {k: str(v) for k, v in payload.items()}

        await channel.handle_conversation_relay_callback(payload)

        assert "CA_relay_xyz" not in channel._conversations
        assert ended == ["CA_relay_xyz"]

    @pytest.mark.no_auto_mock
    @pytest.mark.asyncio
    async def test_callback_ignores_mismatched_account_sid(self) -> None:
        """Callback with wrong account_sid is silently ignored."""
        tac = TAC(relay_only_config())
        channel = VoiceChannel(tac)
        channel._start_conversation("CA_relay_mismatch", None)

        payload = ConversationRelayCallbackPayload(
            CallSid="CA_relay_mismatch",
            CallStatus="completed",
            AccountSid="AC_WRONG",
            From="+15551230000",
            To="+15551234567",
            Direction="inbound",
        ).model_dump(by_alias=True, exclude_none=True)
        payload = {k: str(v) for k, v in payload.items()}

        await channel.handle_conversation_relay_callback(payload)

        assert "CA_relay_mismatch" in channel._conversations

    @pytest.mark.no_auto_mock
    @pytest.mark.asyncio
    async def test_interrupt_callback_fires_in_relay_only(self) -> None:
        """on_interrupt callback fires in relay-only mode with session for call_sid."""
        tac = TAC(relay_only_config())
        channel = VoiceChannel(tac)

        channel._start_conversation("CA_relay_int", None)

        seen: list = []

        def on_interrupt(session, interrupt_data):
            seen.append((session.conversation_id, session.profile_id))

        tac.on_interrupt(on_interrupt)

        from tac.models.voice import InterruptMessage

        channel._handle_interrupt(
            "CA_relay_int",
            InterruptMessage(
                type="interrupt",
                utteranceUntilInterrupt="Hello, I was saying...",
                durationUntilInterruptMs=1500,
            ),
        )

        assert seen == [("CA_relay_int", None)]

    @pytest.mark.no_auto_mock
    def test_sms_channel_raises_in_relay_only(self) -> None:
        """SMSChannel construction fails in relay-only mode."""
        tac = TAC(relay_only_config())

        with pytest.raises(ValueError, match="Conversation Orchestrator"):
            SMSChannel(tac)

    @pytest.mark.no_auto_mock
    def test_chat_channel_raises_in_relay_only(self) -> None:
        """ChatChannel construction fails in relay-only mode."""
        tac = TAC(relay_only_config())

        with pytest.raises(ValueError, match="Conversation Orchestrator"):
            ChatChannel(tac)

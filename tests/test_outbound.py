"""Tests for outbound conversation support (SMS, RCS, WhatsApp, Chat, Voice)."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tac import TAC
from tac.channels.chat import ChatChannel
from tac.channels.rcs import RCSChannel
from tac.channels.sms import SMSChannel
from tac.channels.voice import VoiceChannel
from tac.channels.whatsapp import WhatsAppChannel
from tac.core.config import TwilioMemoryConfig
from tac.models.conversation import (
    ActionResponse,
    ConversationResponse,
    ParticipantAddress,
    ParticipantResponse,
)
from tac.models.outbound import (
    InitiateChatConversationOptions,
    InitiateMessagingConversationOptions,
    InitiateVoiceConversationOptions,
)


def get_test_config() -> dict[str, Any]:
    return {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_secret",
        "conversation_configuration_id": "conv_configuration_test123",
        "phone_number": "+15551234567",
        "rcs_sender_id": "rcs:my_agent",
        "whatsapp_number": "whatsapp:+15551234567",
        "memory_config": TwilioMemoryConfig(trait_groups=["Contact"]),
    }


def make_participant(
    *,
    id: str,
    conversation_id: str,
    type: str,
    channel: str,
    address: str,
    channel_id: str | None = None,
) -> ParticipantResponse:
    return ParticipantResponse(
        id=id,
        conversation_id=conversation_id,
        accountId="ACtest123",
        name=address,
        type=type,
        addresses=[ParticipantAddress(channel=channel, address=address, channel_id=channel_id)],
    )


def make_action_response(conversation_id: str) -> ActionResponse:
    return ActionResponse(
        id="ACT_test",
        type="SEND_MESSAGE",
        status="PENDING",
        conversationId=conversation_id,
    )


# =============================================================================
# ConversationClient — create_or_reuse_conversation
# =============================================================================


class TestCreateOrReuseConversation:
    @pytest.mark.asyncio
    async def test_returns_new_conversation(self) -> None:
        tac = TAC(get_test_config())
        tac.conversation_orchestrator_client.create_conversation = AsyncMock(
            return_value=ConversationResponse(id="CHnew123", accountId="ACtest123", status="ACTIVE")
        )

        conv_id, reused = await tac.conversation_orchestrator_client.create_or_reuse_conversation(
            participants=[]
        )
        assert conv_id == "CHnew123"
        assert reused is False

    @pytest.mark.asyncio
    async def test_extracts_409_header(self) -> None:
        tac = TAC(get_test_config())

        response_409 = httpx.Response(
            409,
            headers={"x-conflicting-resource-id": "CHexisting456"},
            request=httpx.Request("POST", "https://test.com"),
        )
        tac.conversation_orchestrator_client.create_conversation = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "409", request=response_409.request, response=response_409
            )
        )

        conv_id, reused = await tac.conversation_orchestrator_client.create_or_reuse_conversation(
            participants=[]
        )
        assert conv_id == "CHexisting456"
        assert reused is True

    @pytest.mark.asyncio
    async def test_raises_on_409_without_header(self) -> None:
        tac = TAC(get_test_config())

        response_409 = httpx.Response(
            409,
            headers={},
            request=httpx.Request("POST", "https://test.com"),
        )
        tac.conversation_orchestrator_client.create_conversation = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "409", request=response_409.request, response=response_409
            )
        )

        with pytest.raises(RuntimeError, match="x-conflicting-resource-id header is missing"):
            await tac.conversation_orchestrator_client.create_or_reuse_conversation(participants=[])

    @pytest.mark.asyncio
    async def test_reraises_non_409_errors(self) -> None:
        tac = TAC(get_test_config())

        response_500 = httpx.Response(500, request=httpx.Request("POST", "https://test.com"))
        tac.conversation_orchestrator_client.create_conversation = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "500", request=response_500.request, response=response_500
            )
        )

        with pytest.raises(httpx.HTTPStatusError):
            await tac.conversation_orchestrator_client.create_or_reuse_conversation(participants=[])


# =============================================================================
# SMS outbound
# =============================================================================


def _mock_sms_outbound(
    tac: TAC,
    conv_id: str = "CHsms_out",
    *,
    to: str = "+15559876543",
    from_addr: str = "+15551234567",
    reused: bool = False,
) -> None:
    co = tac.conversation_orchestrator_client
    co.create_or_reuse_conversation = AsyncMock(return_value=(conv_id, reused))
    co.list_participants = AsyncMock(
        return_value=[
            make_participant(
                id="PAcust", conversation_id=conv_id, type="CUSTOMER", channel="SMS", address=to
            ),
            make_participant(
                id="PAagent",
                conversation_id=conv_id,
                type="AI_AGENT",
                channel="SMS",
                address=from_addr,
            ),
        ]
    )
    co.create_action = AsyncMock(return_value=make_action_response(conv_id))


class TestSMSOutbound:
    @pytest.mark.asyncio
    async def test_creates_conversation_and_sends_message(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        _mock_sms_outbound(tac)

        result = await channel.initiate_outbound_conversation(
            InitiateMessagingConversationOptions(to="+15559876543", message="Hello!")
        )

        assert result.conversation_id == "CHsms_out"
        assert result.session.channel == "sms"
        assert result.session.metadata["direction"] == "outbound"
        assert result.session.author_info is not None
        assert result.session.author_info.address == "+15559876543"
        tac.conversation_orchestrator_client.create_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_local_session(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        _mock_sms_outbound(tac)

        result = await channel.initiate_outbound_conversation(
            InitiateMessagingConversationOptions(to="+15559876543", message="Hi")
        )

        assert "CHsms_out" in channel._conversations
        assert channel._conversations["CHsms_out"] is result.session

    @pytest.mark.asyncio
    async def test_custom_metadata(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        _mock_sms_outbound(tac)

        result = await channel.initiate_outbound_conversation(
            InitiateMessagingConversationOptions(
                to="+15559876543",
                message="Hi",
                metadata={"campaign": "welcome", "source": "crm"},
            )
        )

        assert result.session.metadata["campaign"] == "welcome"
        assert result.session.metadata["source"] == "crm"
        assert result.session.metadata["direction"] == "outbound"

    @pytest.mark.asyncio
    async def test_reuses_conversation_on_409(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        _mock_sms_outbound(tac, conv_id="CHexisting", reused=True)

        result = await channel.initiate_outbound_conversation(
            InitiateMessagingConversationOptions(to="+15559876543", message="Hello again")
        )

        assert result.conversation_id == "CHexisting"
        assert result.session.metadata["direction"] == "outbound"

    @pytest.mark.asyncio
    async def test_does_not_close_reused_conversation_on_failure(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        co = tac.conversation_orchestrator_client
        co.create_or_reuse_conversation = AsyncMock(return_value=("CHreused", True))
        co.list_participants = AsyncMock(return_value=[])
        co.update_conversation = AsyncMock()

        with pytest.raises(RuntimeError, match="Customer participant not found"):
            await channel.initiate_outbound_conversation(
                InitiateMessagingConversationOptions(to="+15559876543", message="Hello")
            )

        co.update_conversation.assert_not_called()
        assert "CHreused" not in channel._conversations

    @pytest.mark.asyncio
    async def test_closes_new_conversation_on_failure(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        co = tac.conversation_orchestrator_client
        co.create_or_reuse_conversation = AsyncMock(return_value=("CHnew", False))
        co.list_participants = AsyncMock(return_value=[])
        co.update_conversation = AsyncMock()

        with pytest.raises(RuntimeError, match="Customer participant not found"):
            await channel.initiate_outbound_conversation(
                InitiateMessagingConversationOptions(to="+15559876543", message="Hello")
            )

        co.update_conversation.assert_called_once_with("CHnew", "CLOSED")
        assert "CHnew" not in channel._conversations

    @pytest.mark.asyncio
    async def test_passes_participants_in_create(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        _mock_sms_outbound(tac)

        await channel.initiate_outbound_conversation(
            InitiateMessagingConversationOptions(to="+15559876543", message="Test")
        )

        call_args = tac.conversation_orchestrator_client.create_or_reuse_conversation.call_args
        participants = call_args.kwargs["participants"]
        assert len(participants) == 2
        assert participants[0].type == "CUSTOMER"
        assert participants[0].addresses[0].channel == "SMS"
        assert participants[0].addresses[0].address == "+15559876543"
        assert participants[1].type == "AI_AGENT"
        assert participants[1].addresses[0].address == "+15551234567"


# =============================================================================
# SMS sendResponse after outbound
# =============================================================================


# =============================================================================
# Chat outbound
# =============================================================================


def _mock_chat_outbound(
    tac: TAC,
    conv_id: str = "CHchat_out",
    *,
    to: str = "customer@example.com",
    from_addr: str = "ai-assistant",
    channel_id: str = "CHSIDabc",
    reused: bool = False,
) -> None:
    co = tac.conversation_orchestrator_client
    co.create_or_reuse_conversation = AsyncMock(return_value=(conv_id, reused))
    co.list_participants = AsyncMock(
        return_value=[
            make_participant(
                id="PAchatcust",
                conversation_id=conv_id,
                type="CUSTOMER",
                channel="CHAT",
                address=to,
                channel_id=channel_id,
            ),
            make_participant(
                id="PAchatagent",
                conversation_id=conv_id,
                type="AI_AGENT",
                channel="CHAT",
                address=from_addr,
                channel_id=channel_id,
            ),
        ]
    )
    co.create_action = AsyncMock(return_value=make_action_response(conv_id))


class TestChatOutbound:
    @pytest.mark.asyncio
    async def test_creates_conversation_and_sends_message(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        _mock_chat_outbound(tac)

        result = await channel.initiate_outbound_conversation(
            InitiateChatConversationOptions(
                to="customer@example.com",
                channel_id="CHSIDabc",
                message="Welcome!",
            )
        )

        assert result.conversation_id == "CHchat_out"
        assert result.session.channel == "chat"
        assert result.session.metadata["direction"] == "outbound"
        assert result.session.metadata["channel_id"] == "CHSIDabc"
        tac.conversation_orchestrator_client.create_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_matches_channel_id_in_participants(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        _mock_chat_outbound(tac)

        await channel.initiate_outbound_conversation(
            InitiateChatConversationOptions(
                to="customer@example.com",
                channel_id="CHSIDabc",
                message="Test",
            )
        )

        call_args = tac.conversation_orchestrator_client.create_or_reuse_conversation.call_args
        participants = call_args.kwargs["participants"]
        assert participants[0].addresses[0].channel_id == "CHSIDabc"

    @pytest.mark.asyncio
    async def test_409_reuse_for_chat(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        _mock_chat_outbound(tac, conv_id="CHchat_reuse", reused=True)

        result = await channel.initiate_outbound_conversation(
            InitiateChatConversationOptions(
                to="customer@example.com",
                channel_id="CHSIDabc",
                message="Hello again",
            )
        )

        assert result.conversation_id == "CHchat_reuse"


# =============================================================================
# Voice outbound
# =============================================================================


class TestVoiceOutbound:
    @pytest.mark.asyncio
    async def test_places_call_with_twiml(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        mock_call = MagicMock()
        mock_call.sid = "CAtestcall123"
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call

        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            result = await channel.initiate_outbound_conversation(
                InitiateVoiceConversationOptions(
                    to="+15559876543",
                    websocket_url="wss://example.com/ws",
                )
            )

        assert result.call_sid == "CAtestcall123"
        mock_client.calls.create.assert_called_once()
        call_kwargs = mock_client.calls.create.call_args.kwargs
        assert call_kwargs["to"] == "+15559876543"
        assert call_kwargs["from_"] == "+15551234567"
        assert "conversationConfiguration" in call_kwargs["twiml"]

    @pytest.mark.asyncio
    async def test_returns_call_sid(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        mock_call = MagicMock()
        mock_call.sid = "CAsid789"
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call

        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            result = await channel.initiate_outbound_conversation(
                InitiateVoiceConversationOptions(
                    to="+15559876543",
                    websocket_url="wss://example.com/ws",
                )
            )

        assert result.call_sid == "CAsid789"


# =============================================================================
# isOwnMessage 2-tier
# =============================================================================


class TestIsOwnMessage:
    @pytest.mark.asyncio
    async def test_tier1_default_agent_address(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        result = await channel._is_own_message("+15551234567", "CHtest", None)
        assert result is True

    @pytest.mark.asyncio
    async def test_tier2_api_fallback(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        tac.conversation_orchestrator_client.list_participants = AsyncMock(
            return_value=[
                make_participant(
                    id="PAagent_custom",
                    conversation_id="CHtest",
                    type="AI_AGENT",
                    channel="SMS",
                    address="+15550009999",
                ),
            ]
        )

        # No local session — triggers API fallback
        result = await channel._is_own_message("+15550009999", "CHtest", "PAagent_custom")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_for_customer(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        channel._start_conversation("CHtest")

        result = await channel._is_own_message("+15559876543", "CHtest", None)
        assert result is False

    @pytest.mark.asyncio
    async def test_tier2_fires_when_session_exists(self) -> None:
        """API fallback fires when author is not the default agent address."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        tac.conversation_orchestrator_client.list_participants = AsyncMock(
            return_value=[
                make_participant(
                    id="PAsomeid",
                    conversation_id="CHtest",
                    type="AI_AGENT",
                    channel="SMS",
                    address="+15559999999",
                ),
            ]
        )

        channel._start_conversation("CHtest")

        result = await channel._is_own_message("+15559999999", "CHtest", "PAsomeid")
        assert result is True
        tac.conversation_orchestrator_client.list_participants.assert_called_once()

    @pytest.mark.asyncio
    async def test_tier2_handles_api_error_gracefully(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        tac.conversation_orchestrator_client.list_participants = AsyncMock(
            side_effect=Exception("API error")
        )

        result = await channel._is_own_message("+15559999999", "CHtest", "PAsomeid")
        assert result is False


# =============================================================================
# Chat sendResponse after outbound
# =============================================================================


class TestChatSendResponseAfterOutbound:
    @pytest.mark.asyncio
    async def test_send_response_includes_channel_settings(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        _mock_chat_outbound(tac)

        await channel.initiate_outbound_conversation(
            InitiateChatConversationOptions(
                to="customer@example.com",
                channel_id="CHSIDabc",
                message="First",
            )
        )

        tac.conversation_orchestrator_client.create_action = AsyncMock(
            return_value=make_action_response("CHchat_out")
        )

        await channel.send_response("CHchat_out", "Follow-up")

        call_args = tac.conversation_orchestrator_client.create_action.call_args
        action_request = call_args.args[1]
        assert action_request.payload.channel_settings is not None
        assert action_request.payload.channel_settings.channel_id == "CHSIDabc"


# =============================================================================
# create_action failure during initiate_outbound_conversation
# =============================================================================


class TestInitiateConversationActionFailure:
    @pytest.mark.asyncio
    async def test_closes_new_conversation_on_action_failure(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        co = tac.conversation_orchestrator_client
        co.create_or_reuse_conversation = AsyncMock(return_value=("CHnew_action", False))
        co.list_participants = AsyncMock(
            return_value=[
                make_participant(
                    id="PAcust",
                    conversation_id="CHnew_action",
                    type="CUSTOMER",
                    channel="SMS",
                    address="+15559876543",
                ),
                make_participant(
                    id="PAagent",
                    conversation_id="CHnew_action",
                    type="AI_AGENT",
                    channel="SMS",
                    address="+15551234567",
                ),
            ]
        )
        co.create_action = AsyncMock(side_effect=Exception("Action API error"))
        co.update_conversation = AsyncMock()

        with pytest.raises(Exception, match="Action API error"):
            await channel.initiate_outbound_conversation(
                InitiateMessagingConversationOptions(to="+15559876543", message="Hello")
            )

        co.update_conversation.assert_called_once_with("CHnew_action", "CLOSED")
        assert "CHnew_action" not in channel._conversations

    @pytest.mark.asyncio
    async def test_does_not_close_reused_conversation_on_action_failure(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        co = tac.conversation_orchestrator_client
        co.create_or_reuse_conversation = AsyncMock(return_value=("CHreused_action", True))
        co.list_participants = AsyncMock(
            return_value=[
                make_participant(
                    id="PAcust",
                    conversation_id="CHreused_action",
                    type="CUSTOMER",
                    channel="SMS",
                    address="+15559876543",
                ),
                make_participant(
                    id="PAagent",
                    conversation_id="CHreused_action",
                    type="AI_AGENT",
                    channel="SMS",
                    address="+15551234567",
                ),
            ]
        )
        co.create_action = AsyncMock(side_effect=Exception("Action API error"))
        co.update_conversation = AsyncMock()

        with pytest.raises(Exception, match="Action API error"):
            await channel.initiate_outbound_conversation(
                InitiateMessagingConversationOptions(to="+15559876543", message="Hello")
            )

        co.update_conversation.assert_not_called()
        assert "CHreused_action" not in channel._conversations


# =============================================================================
# Voice outbound error paths
# =============================================================================


class TestVoiceOutboundErrors:
    @pytest.mark.asyncio
    async def test_reraises_twilio_rest_error(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        mock_client = MagicMock()
        mock_client.calls.create.side_effect = Exception("Twilio REST error: invalid number")

        with (
            patch.object(channel, "_get_twilio_client", return_value=mock_client),
            pytest.raises(Exception, match="Twilio REST error"),
        ):
            await channel.initiate_outbound_conversation(
                InitiateVoiceConversationOptions(
                    to="+15559876543",
                    websocket_url="wss://example.com/ws",
                )
            )

    @pytest.mark.asyncio
    async def test_custom_parameters_in_twiml(self) -> None:
        from tac.models.voice import TwiMLOptions

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        mock_call = MagicMock()
        mock_call.sid = "CAcustom"
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call

        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            await channel.initiate_outbound_conversation(
                InitiateVoiceConversationOptions(
                    to="+15559876543",
                    websocket_url="wss://example.com/ws",
                    twiml_options=TwiMLOptions(custom_parameters={"foo": "bar"}),
                )
            )

        call_kwargs = mock_client.calls.create.call_args.kwargs
        assert "foo" in call_kwargs["twiml"]
        assert "bar" in call_kwargs["twiml"]

    @pytest.mark.asyncio
    async def test_welcome_greeting_in_twiml(self) -> None:
        from tac.models.voice import TwiMLOptions

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        mock_call = MagicMock()
        mock_call.sid = "CAgreet"
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call

        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            await channel.initiate_outbound_conversation(
                InitiateVoiceConversationOptions(
                    to="+15559876543",
                    websocket_url="wss://example.com/ws",
                    twiml_options=TwiMLOptions(welcome_greeting="Hi there!"),
                )
            )

        call_kwargs = mock_client.calls.create.call_args.kwargs
        assert "Hi there!" in call_kwargs["twiml"]

    @pytest.mark.asyncio
    async def test_channel_twiml_options_applied(self) -> None:
        """VoiceChannelConfig.twiml_options flows into outbound TwiML."""
        from tac.channels.voice import VoiceChannelConfig
        from tac.models.voice import TwiMLOptions

        tac = TAC(get_test_config())
        channel = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                default_twiml_options=TwiMLOptions(voice="en-US-Journey-D", interruptible="speech"),
            ),
        )

        mock_call = MagicMock()
        mock_call.sid = "CAchan"
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call

        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            await channel.initiate_outbound_conversation(
                InitiateVoiceConversationOptions(
                    to="+15559876543",
                    websocket_url="wss://example.com/ws",
                )
            )

        twiml_xml = mock_client.calls.create.call_args.kwargs["twiml"]
        assert 'voice="en-US-Journey-D"' in twiml_xml
        assert 'interruptible="speech"' in twiml_xml

    @pytest.mark.asyncio
    async def test_per_call_twiml_options_override_channel(self) -> None:
        """Per-call twiml_options win over channel-static twiml_options."""
        from tac.channels.voice import VoiceChannelConfig
        from tac.models.voice import TwiMLOptions

        tac = TAC(get_test_config())
        channel = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                default_twiml_options=TwiMLOptions(voice="en-US-Journey-D"),
            ),
        )

        mock_call = MagicMock()
        mock_call.sid = "CApercall"
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call

        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            await channel.initiate_outbound_conversation(
                InitiateVoiceConversationOptions(
                    to="+15559876543",
                    websocket_url="wss://example.com/ws",
                    twiml_options=TwiMLOptions(voice="es-MX-Neural2-A"),
                )
            )

        twiml_xml = mock_client.calls.create.call_args.kwargs["twiml"]
        assert 'voice="es-MX-Neural2-A"' in twiml_xml
        assert "en-US-Journey-D" not in twiml_xml

    @pytest.mark.asyncio
    async def test_studio_handoff_used_when_no_action_url(self) -> None:
        """Studio handoff URL drives action_url on outbound when no override."""
        flow_sid = "FW" + "a" * 32
        tac = TAC({**get_test_config(), "studio_handoff_flow_sid": flow_sid})
        channel = VoiceChannel(tac)

        mock_call = MagicMock()
        mock_call.sid = "CAstudio"
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call

        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            await channel.initiate_outbound_conversation(
                InitiateVoiceConversationOptions(
                    to="+15559876543",
                    websocket_url="wss://example.com/ws",
                )
            )

        twiml_xml = mock_client.calls.create.call_args.kwargs["twiml"]
        assert f"Flows/{flow_sid}" in twiml_xml

    @pytest.mark.asyncio
    async def test_deprecated_flat_fields_emit_warning_and_forward(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        mock_call = MagicMock()
        mock_call.sid = "CAdepr"
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call

        with (
            patch.object(channel, "_get_twilio_client", return_value=mock_client),
            pytest.warns(DeprecationWarning, match="flat fields"),
        ):
            await channel.initiate_outbound_conversation(
                InitiateVoiceConversationOptions(
                    to="+15559876543",
                    websocket_url="wss://example.com/ws",
                    welcome_greeting="Legacy greeting",
                    custom_parameters={"legacy": "true"},
                )
            )

        twiml_xml = mock_client.calls.create.call_args.kwargs["twiml"]
        assert "Legacy greeting" in twiml_xml
        assert 'name="legacy"' in twiml_xml

    def test_deprecated_flat_fields_marked_as_explicitly_set_after_forwarding(
        self,
    ) -> None:
        """Forwarded deprecated values must end up in model_fields_set on the
        twiml_options object, otherwise the merge layer in handle_incoming_call
        / initiate_outbound_conversation would treat them as 'not set' and the
        fallback default would override them."""
        import warnings

        from tac.models.voice import TwiMLOptions

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            opts = InitiateVoiceConversationOptions(
                to="+15559876543",
                websocket_url="wss://example.com/ws",
                twiml_options=TwiMLOptions(voice="en-US-Journey-D"),
                welcome_greeting="Legacy",
            )
        assert opts.twiml_options is not None
        assert opts.twiml_options.welcome_greeting == "Legacy"
        # Critical: must be in model_fields_set so the merge layer treats it
        # as an explicit override, not a fallthrough.
        assert "welcome_greeting" in opts.twiml_options.model_fields_set
        assert "voice" in opts.twiml_options.model_fields_set

    @pytest.mark.asyncio
    async def test_deprecated_flat_fields_lose_to_explicit_twiml_options(self) -> None:
        """If both flat welcome_greeting and twiml_options.welcome_greeting are
        set, the twiml_options value wins (the user's explicit modern call)."""
        from tac.models.voice import TwiMLOptions

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        mock_call = MagicMock()
        mock_call.sid = "CAboth"
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call

        with (
            patch.object(channel, "_get_twilio_client", return_value=mock_client),
            pytest.warns(DeprecationWarning),
        ):
            await channel.initiate_outbound_conversation(
                InitiateVoiceConversationOptions(
                    to="+15559876543",
                    websocket_url="wss://example.com/ws",
                    welcome_greeting="Legacy",
                    twiml_options=TwiMLOptions(welcome_greeting="Modern"),
                )
            )

        twiml_xml = mock_client.calls.create.call_args.kwargs["twiml"]
        assert "Modern" in twiml_xml
        assert "Legacy" not in twiml_xml


# =============================================================================
# RCS outbound
# =============================================================================


def _mock_rcs_outbound(
    tac: TAC,
    conv_id: str = "CHrcs_out",
    *,
    to: str = "rcs:+15559876543",
    from_addr: str = "rcs:my_agent",
    reused: bool = False,
) -> None:
    co = tac.conversation_orchestrator_client
    co.create_or_reuse_conversation = AsyncMock(return_value=(conv_id, reused))
    co.list_participants = AsyncMock(
        return_value=[
            make_participant(
                id="PArcscust", conversation_id=conv_id, type="CUSTOMER", channel="RCS", address=to
            ),
            make_participant(
                id="PArcsagent",
                conversation_id=conv_id,
                type="AI_AGENT",
                channel="RCS",
                address=from_addr,
            ),
        ]
    )
    co.create_action = AsyncMock(return_value=make_action_response(conv_id))


class TestRCSOutbound:
    @pytest.mark.asyncio
    async def test_creates_conversation_and_sends_message(self) -> None:
        tac = TAC(get_test_config())
        from tac.channels.rcs import RCSChannelConfig

        channel = RCSChannel(tac, config=RCSChannelConfig())
        _mock_rcs_outbound(tac)

        result = await channel.initiate_outbound_conversation(
            InitiateMessagingConversationOptions(to="rcs:+15559876543", message="Hello from RCS!")
        )

        assert result.conversation_id == "CHrcs_out"
        assert result.session.channel == "rcs"
        assert result.session.metadata["direction"] == "outbound"
        assert result.session.author_info is not None
        assert result.session.author_info.address == "rcs:+15559876543"
        tac.conversation_orchestrator_client.create_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_reuses_conversation_on_409(self) -> None:
        tac = TAC(get_test_config())
        from tac.channels.rcs import RCSChannelConfig

        channel = RCSChannel(tac, config=RCSChannelConfig())
        _mock_rcs_outbound(tac, conv_id="CHrcs_existing", reused=True)

        result = await channel.initiate_outbound_conversation(
            InitiateMessagingConversationOptions(to="rcs:+15559876543", message="Hello again")
        )

        assert result.conversation_id == "CHrcs_existing"
        assert result.session.metadata["direction"] == "outbound"

    @pytest.mark.asyncio
    async def test_passes_participants_in_create(self) -> None:
        tac = TAC(get_test_config())
        from tac.channels.rcs import RCSChannelConfig

        channel = RCSChannel(tac, config=RCSChannelConfig())
        _mock_rcs_outbound(tac)

        await channel.initiate_outbound_conversation(
            InitiateMessagingConversationOptions(to="rcs:+15559876543", message="Test")
        )

        call_args = tac.conversation_orchestrator_client.create_or_reuse_conversation.call_args
        participants = call_args.kwargs["participants"]
        assert len(participants) == 2
        assert participants[0].type == "CUSTOMER"
        assert participants[0].addresses[0].channel == "RCS"
        assert participants[0].addresses[0].address == "rcs:+15559876543"
        assert participants[1].type == "AI_AGENT"
        assert participants[1].addresses[0].address == "rcs:my_agent"


# =============================================================================
# RCS sendResponse after outbound
# =============================================================================

# =============================================================================
# WhatsApp outbound
# =============================================================================


def _mock_whatsapp_outbound(
    tac: TAC,
    conv_id: str = "CHwhatsapp_out",
    *,
    to: str = "whatsapp:+15559876543",
    from_addr: str = "whatsapp:+15551234567",
    reused: bool = False,
) -> None:
    co = tac.conversation_orchestrator_client
    co.create_or_reuse_conversation = AsyncMock(return_value=(conv_id, reused))
    co.list_participants = AsyncMock(
        return_value=[
            make_participant(
                id="PAwhatsappcust",
                conversation_id=conv_id,
                type="CUSTOMER",
                channel="WHATSAPP",
                address=to,
            ),
            make_participant(
                id="PAwhatsappagent",
                conversation_id=conv_id,
                type="AI_AGENT",
                channel="WHATSAPP",
                address=from_addr,
            ),
        ]
    )
    co.create_action = AsyncMock(return_value=make_action_response(conv_id))


class TestWhatsAppOutbound:
    @pytest.mark.asyncio
    async def test_creates_conversation_and_sends_message(self) -> None:
        tac = TAC(get_test_config())
        from tac.channels.whatsapp import WhatsAppChannelConfig

        channel = WhatsAppChannel(tac, config=WhatsAppChannelConfig())
        _mock_whatsapp_outbound(tac)

        result = await channel.initiate_outbound_conversation(
            InitiateMessagingConversationOptions(
                to="whatsapp:+15559876543", message="Hello from WhatsApp!"
            )
        )

        assert result.conversation_id == "CHwhatsapp_out"
        assert result.session.channel == "whatsapp"
        assert result.session.metadata["direction"] == "outbound"
        assert result.session.author_info is not None
        assert result.session.author_info.address == "whatsapp:+15559876543"
        tac.conversation_orchestrator_client.create_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_reuses_conversation_on_409(self) -> None:
        tac = TAC(get_test_config())
        from tac.channels.whatsapp import WhatsAppChannelConfig

        channel = WhatsAppChannel(tac, config=WhatsAppChannelConfig())
        _mock_whatsapp_outbound(tac, conv_id="CHwhatsapp_existing", reused=True)

        result = await channel.initiate_outbound_conversation(
            InitiateMessagingConversationOptions(to="whatsapp:+15559876543", message="Hello again")
        )

        assert result.conversation_id == "CHwhatsapp_existing"
        assert result.session.metadata["direction"] == "outbound"

    @pytest.mark.asyncio
    async def test_passes_participants_in_create(self) -> None:
        tac = TAC(get_test_config())
        from tac.channels.whatsapp import WhatsAppChannelConfig

        channel = WhatsAppChannel(tac, config=WhatsAppChannelConfig())
        _mock_whatsapp_outbound(tac)

        await channel.initiate_outbound_conversation(
            InitiateMessagingConversationOptions(to="whatsapp:+15559876543", message="Test")
        )

        call_args = tac.conversation_orchestrator_client.create_or_reuse_conversation.call_args
        participants = call_args.kwargs["participants"]
        assert len(participants) == 2
        assert participants[0].type == "CUSTOMER"
        assert participants[0].addresses[0].channel == "WHATSAPP"
        assert participants[0].addresses[0].address == "whatsapp:+15559876543"
        assert participants[1].type == "AI_AGENT"
        assert participants[1].addresses[0].address == "whatsapp:+15551234567"

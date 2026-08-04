"""Tests for outbound conversation support (SMS, RCS, WhatsApp, Chat, Voice)."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tac import TAC
from tac.channels.chat import ChatChannel
from tac.channels.rcs import RCSChannel
from tac.channels.sms import SMSChannel
from tac.channels.voice import VoiceChannel, VoiceChannelConfig
from tac.channels.whatsapp import WhatsAppChannel
from tac.core.config import TwilioMemoryConfig
from tac.models.conversation import (
    ActionResponse,
    ConversationResponse,
    ParticipantAddress,
    ParticipantResponse,
)
from tac.models.outbound import (
    CallOptions,
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
        assert result.session.channel == "SMS"
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
        assert result.session.channel == "CHAT"
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

    async def _place_call(
        self, channel: VoiceChannel, options: InitiateVoiceConversationOptions
    ) -> dict[str, Any]:
        """Place a call with a mocked Twilio client and return the calls.create kwargs."""
        mock_call = MagicMock()
        mock_call.sid = "CAxyz"
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call
        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            await channel.initiate_outbound_conversation(options)
        return mock_client.calls.create.call_args.kwargs

    @staticmethod
    def _noop_handler() -> Any:
        async def handler(event: Any) -> None:
            return None

        return handler

    @pytest.mark.asyncio
    async def test_call_options_passthrough(self) -> None:
        """Typed call_options are forwarded to calls.create."""
        tac = TAC(get_test_config())  # no voice_public_domain → no auto-wiring
        channel = VoiceChannel(tac)

        kwargs = await self._place_call(
            channel,
            InitiateVoiceConversationOptions(
                to="+15559876543",
                websocket_url="wss://example.com/ws",
                call_options={"machine_detection": "Enable", "async_amd": True, "timeout": 20},
            ),
        )
        assert kwargs["machine_detection"] == "Enable"
        assert kwargs["timeout"] == 20

    @pytest.mark.asyncio
    async def test_untyped_calls_api_params_pass_through(self) -> None:
        """Calls API params TAC doesn't type explicitly still forward."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        kwargs = await self._place_call(
            channel,
            InitiateVoiceConversationOptions(
                to="+15559876543",
                websocket_url="wss://example.com/ws",
                call_options={"sip_auth_username": "alice", "byoc": "BYxxx"},
            ),
        )
        assert kwargs["sip_auth_username"] == "alice"
        assert kwargs["byoc"] == "BYxxx"

    def test_rejects_params_the_sdk_does_not_accept(self) -> None:
        """Unknown keys are a TypeError at call time; caught here instead."""
        with pytest.raises(ValueError, match="does not accept"):
            CallOptions(machine_detecton="Enable")  # typo'd machine_detection

    def test_accepted_params_come_from_the_installed_sdk(self) -> None:
        """Read from the SDK signature, so it can't drift from the installed version."""
        from tac.models.outbound import _twilio_call_create_params

        accepted = _twilio_call_create_params()
        # Sanity-check introspection actually found the real signature.
        assert {"machine_detection", "async_amd", "record", "timeout"} <= accepted
        assert "machine_detecton" not in accepted

    def test_every_typed_field_is_a_real_sdk_param(self) -> None:
        """Guards against TAC typing a field the SDK would reject at call time."""
        from tac.models.outbound import _twilio_call_create_params

        accepted = _twilio_call_create_params()
        assert set(CallOptions.model_fields) <= accepted

    def test_goes_permissive_if_the_sdk_takes_kwargs(self) -> None:
        """A **kwargs signature accepts anything, so there's nothing to validate."""
        import inspect

        from tac.models import outbound

        class FakeCallList:
            def create(self, to=None, from_=None, twiml=None, **kwargs):  # type: ignore[no-untyped-def]
                ...

        real_signature = inspect.signature

        def fake_signature(obj: object) -> inspect.Signature:
            return real_signature(FakeCallList.create)

        outbound._twilio_call_create_params.cache_clear()
        try:
            with patch.object(inspect, "signature", fake_signature):
                assert outbound._twilio_call_create_params() == frozenset()
        finally:
            outbound._twilio_call_create_params.cache_clear()

        # Without the guard, named params would become the accepted set and a
        # real Calls-API param would be rejected as unknown.
        CallOptions(caller_id="+15551234567")

    @pytest.mark.asyncio
    async def test_async_amd_serialized_as_string(self) -> None:
        """Twilio's SDK types async_amd as a string and record as a bool."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        kwargs = await self._place_call(
            channel,
            InitiateVoiceConversationOptions(
                to="+15559876543",
                websocket_url="wss://example.com/ws",
                call_options={"machine_detection": "Enable", "async_amd": True, "record": True},
            ),
        )
        assert kwargs["async_amd"] == "true"
        assert kwargs["record"] is True

    @pytest.mark.parametrize("reserved", ["to", "from", "from_", "twiml", "url", "application_sid"])
    def test_call_options_rejects_reserved(self, reserved: str) -> None:
        """TAC owns these calls.create params; rejected at construction."""
        with pytest.raises(ValueError, match="TAC-owned"):
            CallOptions(**{reserved: "x"})

    @pytest.mark.parametrize(
        "amd_options",
        [
            {"async_amd": True},  # detection never runs
            {"machine_detection": "Enable"},  # AnsweredBy goes nowhere inline TwiML can read
        ],
    )
    def test_amd_requires_both_flags(self, amd_options: dict[str, Any]) -> None:
        """Either flag alone silently yields no AMD event."""
        with pytest.raises(ValueError, match="requires both"):
            CallOptions(**amd_options)

    @pytest.mark.asyncio
    async def test_no_auto_wiring_without_handlers(self) -> None:
        """No handler → no URL advertised, so Twilio never gets pointed at a 404."""
        config = {**get_test_config(), "voice_public_domain": "example.com"}
        channel = VoiceChannel(TAC(config))

        kwargs = await self._place_call(
            channel,
            InitiateVoiceConversationOptions(
                to="+15559876543", websocket_url="wss://example.com/ws"
            ),
        )
        assert "status_callback" not in kwargs
        assert "async_amd_status_callback" not in kwargs
        assert "recording_status_callback" not in kwargs

    @pytest.mark.asyncio
    async def test_each_callback_wired_only_for_its_handler(self) -> None:
        """Registering on_amd wires the AMD URL and nothing else."""
        config = {**get_test_config(), "voice_public_domain": "example.com"}
        channel = VoiceChannel(TAC(config))
        channel.on_amd(self._noop_handler())

        kwargs = await self._place_call(
            channel,
            InitiateVoiceConversationOptions(
                to="+15559876543",
                websocket_url="wss://example.com/ws",
                call_options={"machine_detection": "Enable", "async_amd": True},
            ),
        )
        assert kwargs["async_amd_status_callback"] == "https://example.com/twilio/call-events/amd"
        assert "status_callback" not in kwargs
        assert "recording_status_callback" not in kwargs

    @pytest.mark.asyncio
    async def test_all_callbacks_wired_when_all_handlers_registered(self) -> None:
        config = {**get_test_config(), "voice_public_domain": "example.com"}
        channel = VoiceChannel(TAC(config))
        channel.on_call_status(self._noop_handler())
        channel.on_amd(self._noop_handler())
        channel.on_recording(self._noop_handler())

        kwargs = await self._place_call(
            channel,
            InitiateVoiceConversationOptions(
                to="+15559876543",
                websocket_url="wss://example.com/ws",
                call_options={"machine_detection": "Enable", "async_amd": True, "record": True},
            ),
        )
        assert kwargs["status_callback"] == "https://example.com/twilio/call-events/status"
        assert kwargs["async_amd_status_callback"] == "https://example.com/twilio/call-events/amd"
        assert (
            kwargs["recording_status_callback"]
            == "https://example.com/twilio/call-events/recording"
        )

    @pytest.mark.asyncio
    async def test_explicit_callback_url_wins_over_auto_wiring(self) -> None:
        """setdefault: an explicit URL in call_options is never overwritten."""
        config = {**get_test_config(), "voice_public_domain": "example.com"}
        channel = VoiceChannel(TAC(config))
        channel.on_call_status(self._noop_handler())

        kwargs = await self._place_call(
            channel,
            InitiateVoiceConversationOptions(
                to="+15559876543",
                websocket_url="wss://example.com/ws",
                call_options={"status_callback": "https://other.example/cb"},
            ),
        )
        assert kwargs["status_callback"] == "https://other.example/cb"

    @pytest.mark.asyncio
    async def test_no_auto_wiring_without_domain(self) -> None:
        tac = TAC(get_test_config())  # no voice_public_domain
        channel = VoiceChannel(tac)
        channel.on_call_status(self._noop_handler())

        kwargs = await self._place_call(
            channel,
            InitiateVoiceConversationOptions(
                to="+15559876543", websocket_url="wss://example.com/ws"
            ),
        )
        assert "status_callback" not in kwargs

    @pytest.mark.asyncio
    async def test_trailing_slash_in_path_normalized(self) -> None:
        """A trailing slash on voice_call_event_path doesn't produce '//status'."""
        config = {
            **get_test_config(),
            "voice_public_domain": "example.com",
            "voice_call_event_path": "/hooks/calls/",
        }
        channel = VoiceChannel(TAC(config))
        channel.on_call_status(self._noop_handler())

        kwargs = await self._place_call(
            channel,
            InitiateVoiceConversationOptions(
                to="+15559876543", websocket_url="wss://example.com/ws"
            ),
        )
        assert kwargs["status_callback"] == "https://example.com/hooks/calls/status"


class TestDefaultCallOptions:
    """Channel-wide CallOptions layer, mirroring default_twiml_options. This is
    how a custom server or non-default routes supply their own callback URLs."""

    async def _place(
        self, channel: VoiceChannel, per_call: CallOptions | None = None
    ) -> dict[str, Any]:
        mock_client = MagicMock()
        mock_client.calls.create.return_value = MagicMock(sid="CAxyz")
        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            await channel.initiate_outbound_conversation(
                InitiateVoiceConversationOptions(
                    to="+15559876543",
                    websocket_url="wss://example.com/ws",
                    call_options=per_call,
                )
            )
        return mock_client.calls.create.call_args.kwargs

    @staticmethod
    def _channel(**default_call_options: Any) -> VoiceChannel:
        config = {**get_test_config(), "voice_public_domain": "example.com"}
        return VoiceChannel(
            TAC(config),
            config=VoiceChannelConfig(default_call_options=CallOptions(**default_call_options)),
        )

    @pytest.mark.asyncio
    async def test_applies_to_every_call(self) -> None:
        kwargs = await self._place(self._channel(timeout=45))
        assert kwargs["timeout"] == 45

    @pytest.mark.asyncio
    async def test_url_beats_the_derived_default(self) -> None:
        """The custom-server case: TAC isn't serving /twilio/call-events/amd."""
        channel = self._channel(
            machine_detection="Enable",
            async_amd=True,
            async_amd_status_callback="https://my-flask-app.com/amd-hook",
        )

        async def handler(event: Any) -> None:
            return None

        channel.on_amd(handler)

        kwargs = await self._place(channel)
        assert kwargs["async_amd_status_callback"] == "https://my-flask-app.com/amd-hook"

    @pytest.mark.asyncio
    async def test_per_call_beats_channel_wide(self) -> None:
        kwargs = await self._place(self._channel(timeout=45), CallOptions(timeout=10))
        assert kwargs["timeout"] == 10

    @pytest.mark.asyncio
    async def test_unset_per_call_fields_fall_through(self) -> None:
        """Per-field merge: setting timeout doesn't drop the channel's record flag."""
        kwargs = await self._place(self._channel(record=True, timeout=45), CallOptions(timeout=10))
        assert kwargs["timeout"] == 10
        assert kwargs["record"] is True

    @pytest.mark.asyncio
    async def test_per_call_can_disable_amd(self) -> None:
        """Clearing both flags is valid — it turns AMD off for this call."""
        kwargs = await self._place(
            self._channel(machine_detection="Enable", async_amd=True),
            CallOptions(machine_detection=None, async_amd=None),
        )
        assert "machine_detection" not in kwargs
        assert "async_amd" not in kwargs

    @pytest.mark.asyncio
    async def test_merged_result_is_revalidated(self) -> None:
        """Clearing only one flag leaves the other from the channel default — an
        invalid combination reachable only by layering."""
        with pytest.raises(ValueError, match="requires both"):
            await self._place(
                self._channel(machine_detection="Enable", async_amd=True),
                CallOptions(machine_detection=None),
            )

    @pytest.mark.asyncio
    async def test_extras_merge_too(self) -> None:
        kwargs = await self._place(
            self._channel(byoc="BYdefault"), CallOptions(sip_auth_username="alice")
        )
        assert kwargs["byoc"] == "BYdefault"
        assert kwargs["sip_auth_username"] == "alice"


class TestInitiateVoiceConversationOptionsForbidsExtra:
    """Migration safety: removed fields raise ValidationError instead of
    being silently dropped, so callers upgrading from older TAC versions
    get a clear signal that their code needs updating."""

    def test_removed_welcome_greeting_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="welcome_greeting"):
            InitiateVoiceConversationOptions(to="+15551234567", welcome_greeting="Hi!")  # type: ignore[call-arg]

    def test_removed_action_url_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="action_url"):
            InitiateVoiceConversationOptions(
                to="+15551234567", action_url="https://example.com/end"
            )  # type: ignore[call-arg]

    def test_removed_custom_parameters_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="custom_parameters"):
            InitiateVoiceConversationOptions(to="+15551234567", custom_parameters={"k": "v"})  # type: ignore[call-arg]


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
        assert result.session.channel == "RCS"
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
        assert result.session.channel == "WHATSAPP"
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

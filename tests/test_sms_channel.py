"""Tests for SMS Channel."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tac import TAC
from tac.channels.sms import SMSChannel
from tac.models.conversation import ParticipantAddress, ParticipantResponse
from tac.models.memory import MemoryRetrievalMeta, MemoryRetrievalResponse
from tac.models.session import AuthorInfo, ConversationSession
from tac.models.tac import TACMemoryResponse


def make_participant(
    pid: str,
    ptype: str,
    address: str,
    channel: str = "SMS",
    conv_id: str = "CH123456",
) -> ParticipantResponse:
    return ParticipantResponse(
        **{  # type: ignore[arg-type]
            "id": pid,
            "accountId": "ACtest123",
            "conversationId": conv_id,
            "name": address,
            "type": ptype,
            "addresses": [
                ParticipantAddress(channel=channel, address=address).model_dump(by_alias=True)  # type: ignore[arg-type]
            ],
        }
    )


def create_communication_created_webhook(
    conversation_id: str,
    participant_id: str,
    message_text: str,
    timestamp: str,
    author_address: str = "+12345678901",
    channel: str = "SMS",
) -> dict[str, Any]:
    comm_id = f"comms_communication_{timestamp.replace(':', '').replace('.', '').replace('-', '')}"
    return {
        "eventType": "COMMUNICATION_CREATED",
        "timestamp": timestamp,
        "data": {
            "id": comm_id,
            "conversationId": conversation_id,
            "accountId": "ACtest123",
            "serviceId": "IStest123",
            "author": {
                "address": author_address,
                "channel": channel,
                "participantId": participant_id,
            },
            "content": {"type": "TEXT", "text": message_text},
            "channelId": None,
            "recipients": [
                {
                    "address": "+15551234567",
                    "channel": "SMS",
                    "participantId": "comms_participant_agent",
                    "deliveryStatus": "DELIVERED",
                }
            ],
            "createdAt": timestamp,
            "updatedAt": timestamp,
        },
    }


def create_conversation_updated_webhook(
    conversation_id: str, status: str, timestamp: str
) -> dict[str, Any]:
    return {
        "eventType": "CONVERSATION_UPDATED",
        "timestamp": timestamp,
        "data": {
            "id": conversation_id,
            "accountId": "ACtest123",
            "configurationId": "conv_configuration_test123",
            "serviceId": "IStest123",
            "status": status,
            "name": "Test Conversation",
            "createdAt": "2025-11-18T00:00:00.000Z",
            "updatedAt": timestamp,
            "configuration": {"intelligenceServiceIds": []},
        },
    }


def get_test_config(with_memory: bool = True) -> dict[str, Any]:
    config: dict[str, Any] = {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": "conv_configuration_test123",
        "phone_number": "+15551234567",
    }
    if with_memory:
        from tac.core.config import TwilioMemoryConfig

        config["memory_config"] = TwilioMemoryConfig(trait_groups=["Contact"])
    return config


class TestSMSChannel:
    """Test SMS Channel functionality."""

    def test_initialization(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        assert channel.tac == tac

    def test_initialization_without_phone_number(self) -> None:
        config = get_test_config()
        del config["phone_number"]
        with pytest.raises(ValueError):
            TAC(config)

    @pytest.mark.asyncio
    async def test_inbound_message_auto_initializes_session(self) -> None:
        """COMMUNICATION_CREATED lazy-initializes a session; no profile_id seeded."""
        tac = TAC(get_test_config())

        from tac.context.memory import MemoryClient

        tac.conversation_memory_client = MemoryClient(
            store_id="MGtest123",
            api_key=tac.config.api_key,
            api_secret=tac.config.api_secret,
        )

        channel = SMSChannel(tac)
        captured_context: list[ConversationSession] = []

        def cb(msg: str, ctx: ConversationSession, mem: TACMemoryResponse | None) -> None:
            captured_context.append(ctx)

        tac.on_message_ready(cb)

        empty_response = MemoryRetrievalResponse(
            observations=[], summaries=[], meta=MemoryRetrievalMeta(queryTime=0)
        )
        tac.conversation_memory_client.retrieve_memory = AsyncMock(return_value=empty_response)

        participants = [
            make_participant("PA_AGENT", "AI_AGENT", "+15551234567"),
            make_participant("PA_CUSTOMER", "CUSTOMER", "+12345678901"),
        ]

        webhook = create_communication_created_webhook(
            "CH123456", "PA_CUSTOMER", "Hello", "2025-11-18T00:00:00.000Z"
        )

        with patch.object(
            tac.conversation_orchestrator_client, "list_participants", return_value=participants
        ):
            await channel.process_webhook(webhook)

        assert len(captured_context) == 1
        assert captured_context[0].conversation_id == "CH123456"
        assert captured_context[0].channel == "sms"

    @pytest.mark.asyncio
    async def test_process_empty_message_ignored(self) -> None:
        tac = TAC(get_test_config())
        from tac.context.memory import MemoryClient

        tac.conversation_memory_client = MemoryClient(
            store_id="MGtest123",
            api_key=tac.config.api_key,
            api_secret=tac.config.api_secret,
        )
        channel = SMSChannel(tac)
        webhook = create_communication_created_webhook(
            "CH123456", "MB123", "", "2025-11-18T00:00:00.000Z"
        )
        tac.conversation_memory_client.retrieve_memory = AsyncMock()
        await channel.process_webhook(webhook)
        tac.conversation_memory_client.retrieve_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_response_uses_resolved_participant_ids(self) -> None:
        """send_response reads agent/customer ids stashed by reconciliation."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        # Simulate the state left by _handle_communication_created after
        # reconciliation: session populated + participant ids stashed.
        channel._conversations["CH123456"] = ConversationSession(
            conversation_id="CH123456",
            channel="sms",
            author_info=AuthorInfo(address="+12345678901", participant_id="PA_CUSTOMER"),
        )
        channel._conversations["CH123456"].metadata.update(
            {
                "agent_participant_id": "PA_AGENT",
                "customer_participant_id": "PA_CUSTOMER",
            }
        )

        with patch.object(
            tac.conversation_orchestrator_client, "create_action"
        ) as mock_create_action:
            await channel.send_response("CH123456", "Test response")

        mock_create_action.assert_called_once()
        call_args = mock_create_action.call_args
        assert call_args[0][0] == "CH123456"
        request = call_args[0][1]
        assert request.type == "SEND_MESSAGE"
        assert request.payload.from_.participant_id == "PA_AGENT"
        assert request.payload.from_.channel == "SMS"
        assert request.payload.to[0].participant_id == "PA_CUSTOMER"
        assert request.payload.content.text == "Test response"
        assert request.payload.channel_settings is None

    @pytest.mark.asyncio
    async def test_send_response_forwards_channel_id_when_present(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        channel._conversations["CH_WITH_CH_ID"] = ConversationSession(
            conversation_id="CH_WITH_CH_ID",
            channel="sms",
            author_info=AuthorInfo(address="+12345678901", participant_id="PA_CUSTOMER"),
        )
        channel._conversations["CH_WITH_CH_ID"].metadata.update(
            {
                "agent_participant_id": "PA_AGENT",
                "customer_participant_id": "PA_CUSTOMER",
                "channel_id": "SMabcdef",
            }
        )

        with patch.object(
            tac.conversation_orchestrator_client, "create_action"
        ) as mock_create_action:
            await channel.send_response("CH_WITH_CH_ID", "Test response")

        request = mock_create_action.call_args[0][1]
        assert request.payload.channel_settings is not None
        assert request.payload.channel_settings.channel_id == "SMabcdef"

    @pytest.mark.asyncio
    async def test_send_response_raises_without_session(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        with pytest.raises(RuntimeError, match="No active session"):
            await channel.send_response("CH_NONE", "hi")

    @pytest.mark.asyncio
    async def test_send_response_raises_without_resolved_ids(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        channel._conversations["CH_NOIDS"] = ConversationSession(
            conversation_id="CH_NOIDS",
            channel="sms",
            author_info=AuthorInfo(address="+12345678901", participant_id="PA_CUSTOMER"),
        )
        with pytest.raises(RuntimeError, match="Participant ids not resolved"):
            await channel.send_response("CH_NOIDS", "hi")

    @pytest.mark.asyncio
    async def test_conversation_ended_callback_fires_on_close(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        captured: list[ConversationSession] = []

        def handler(ctx: ConversationSession) -> None:
            captured.append(ctx)

        tac.on_conversation_ended(handler)

        # Seed a session directly; PARTICIPANT_ADDED no longer handled.
        channel._conversations["CH_CB1"] = ConversationSession(
            conversation_id="CH_CB1",
            channel="sms",
            profile_id="prof_cb1",
        )

        await channel.process_webhook(
            create_conversation_updated_webhook("CH_CB1", "CLOSED", "2025-11-18T00:10:00.000Z")
        )

        assert len(captured) == 1
        assert captured[0].conversation_id == "CH_CB1"
        assert captured[0].profile_id == "prof_cb1"
        assert captured[0].channel == "sms"

    @pytest.mark.asyncio
    async def test_conversation_ended_async_callback(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        captured: list[ConversationSession] = []

        async def async_handler(ctx: ConversationSession) -> None:
            captured.append(ctx)

        tac.on_conversation_ended(async_handler)

        channel._conversations["CH_ASYNC1"] = ConversationSession(
            conversation_id="CH_ASYNC1", channel="sms"
        )
        await channel.process_webhook(
            create_conversation_updated_webhook("CH_ASYNC1", "CLOSED", "2025-11-18T00:10:00.000Z")
        )

        assert len(captured) == 1
        assert captured[0].conversation_id == "CH_ASYNC1"

    @pytest.mark.asyncio
    async def test_ignores_chat_messages(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        captured: list[str] = []
        tac.on_message_ready(lambda msg, ctx, mem: captured.append(msg))

        webhook = create_communication_created_webhook(
            "CH123456", "MB123", "Chat message", "2025-11-18T00:00:00.000Z", channel="CHAT"
        )
        await channel.process_webhook(webhook)
        assert captured == []

    @pytest.mark.asyncio
    async def test_ignores_messages_without_author_channel(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        captured: list[str] = []
        tac.on_message_ready(lambda msg, ctx, mem: captured.append(msg))

        webhook = create_communication_created_webhook(
            "CH123456", "MB123", "No channel", "2025-11-18T00:00:00.000Z"
        )
        del webhook["data"]["author"]["channel"]
        await channel.process_webhook(webhook)
        assert captured == []

    @pytest.mark.asyncio
    async def test_callback_auto_send_response(self) -> None:
        tac = TAC(get_test_config(with_memory=False))
        channel = SMSChannel(tac, config={"auto_retrieve_memory": False})

        async def message_callback(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None,
        ) -> str:
            return "This is my automated response"

        tac.on_message_ready(message_callback)

        participants = [
            make_participant("PA_AGENT", "AI_AGENT", "+15551234567", conv_id="CH_AUTO_SEND"),
            make_participant("PA_CUSTOMER", "CUSTOMER", "+12345678901", conv_id="CH_AUTO_SEND"),
        ]

        with (
            patch.object(
                tac.conversation_orchestrator_client,
                "list_participants",
                return_value=participants,
            ),
            patch.object(
                tac.conversation_orchestrator_client, "create_action"
            ) as mock_create_action,
        ):
            message_webhook = create_communication_created_webhook(
                "CH_AUTO_SEND", "PA_AUTO", "Test message", "2025-11-18T00:00:01.000Z"
            )
            await channel.process_webhook(message_webhook)

        mock_create_action.assert_called_once()
        call_args = mock_create_action.call_args
        assert call_args[0][0] == "CH_AUTO_SEND"
        request = call_args[0][1]
        assert request.payload.content.text == "This is my automated response"

    @pytest.mark.asyncio
    async def test_callback_no_auto_send_on_none(self) -> None:
        tac = TAC(get_test_config(with_memory=False))
        channel = SMSChannel(tac, config={"auto_retrieve_memory": False})

        async def message_callback(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None,
        ) -> None:
            return None

        tac.on_message_ready(message_callback)

        participants = [
            make_participant("PA_AGENT", "AI_AGENT", "+15551234567", conv_id="CH_NO_AUTO"),
            make_participant("PA_CUSTOMER", "CUSTOMER", "+12345678901", conv_id="CH_NO_AUTO"),
        ]

        with (
            patch.object(
                tac.conversation_orchestrator_client,
                "list_participants",
                return_value=participants,
            ),
            patch.object(
                tac.conversation_orchestrator_client, "create_action"
            ) as mock_create_action,
        ):
            message_webhook = create_communication_created_webhook(
                "CH_NO_AUTO", "PA_AUTO", "Test message", "2025-11-18T00:00:01.000Z"
            )
            await channel.process_webhook(message_webhook)

        mock_create_action.assert_not_called()

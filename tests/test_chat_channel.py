"""Tests for Chat Channel."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tac import TAC
from tac.channels.chat import ChatChannel, ChatChannelConfig
from tac.models.conversation import ParticipantAddress, ParticipantResponse
from tac.models.memory import MemoryRetrievalMeta, MemoryRetrievalResponse
from tac.models.session import AuthorInfo, ConversationSession
from tac.models.tac import TACMemoryResponse


def make_participant(
    pid: str,
    ptype: str,
    address: str,
    channel: str = "CHAT",
    channel_id: str | None = "CH_CHAT_SID_123",
    conv_id: str = "CH123",
) -> ParticipantResponse:
    addr = ParticipantAddress(
        channel=channel,
        address=address,
        channel_id=channel_id,  # type: ignore[arg-type]
    ).model_dump(by_alias=True)
    return ParticipantResponse(
        **{  # type: ignore[arg-type]
            "id": pid,
            "accountId": "ACtest123",
            "conversationId": conv_id,
            "name": address,
            "type": ptype,
            "addresses": [addr],
        }
    )


def create_communication_created_webhook(
    conversation_id: str,
    participant_id: str,
    message_text: str,
    timestamp: str,
    author_address: str = "user@example.com",
    author_channel: str = "CHAT",
    channel_id: str = "CH_CHAT_SID_123",
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
                "channel": author_channel,
                "participantId": participant_id,
            },
            "content": {"type": "TEXT", "text": message_text},
            "channelId": channel_id,
            "recipients": [
                {
                    "address": "ai-assistant",
                    "channel": "CHAT",
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
            "name": "Test Chat Conversation",
            "createdAt": "2025-11-18T00:00:00.000Z",
            "updatedAt": timestamp,
        },
    }


def get_test_config() -> dict[str, Any]:
    return {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": "conv_configuration_test123",
        "phone_number": "+15551234567",
    }


class TestChatChannel:
    """Test Chat Channel functionality."""

    def test_initialization_defaults(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        assert channel.agent_address == "ai-assistant"
        assert channel.get_channel_name() == "chat"
        assert channel.get_channel_type_upper() == "CHAT"

    def test_initialization_custom_agent_address(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac, config=ChatChannelConfig(agent_address="my-bot"))
        assert channel.agent_address == "my-bot"

    def test_initialization_from_dict(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac, config={"agent_address": "custom-bot"})
        assert channel.agent_address == "custom-bot"

    def test_is_own_message(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        assert channel.is_own_message("ai-assistant") is True
        assert channel.is_own_message("user@example.com") is False

    @pytest.mark.asyncio
    async def test_inbound_message_auto_initializes_session(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        captured: list[ConversationSession] = []

        def cb(msg: str, ctx: ConversationSession, mem: TACMemoryResponse | None) -> None:
            captured.append(ctx)

        tac.on_message_ready(cb)

        participants = [
            make_participant("PA_AGENT", "AI_AGENT", "ai-assistant", channel_id="CH_CHAT_SID_123"),
            make_participant(
                "PA_CUSTOMER", "CUSTOMER", "user@example.com", channel_id="CH_CHAT_SID_123"
            ),
        ]

        webhook = create_communication_created_webhook(
            "CH123", "PA_CUSTOMER", "Hello", "2025-11-18T00:00:01.000Z"
        )

        with patch.object(
            tac.conversation_orchestrator_client, "list_participants", return_value=participants
        ):
            await channel.process_webhook(webhook)

        assert "CH123" in channel._conversations
        assert channel._conversations["CH123"].channel == "chat"
        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_ignores_sms_messages(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        captured: list[str] = []
        tac.on_message_ready(lambda m, c, mem: captured.append(m))

        webhook = create_communication_created_webhook(
            "CH123", "MB123", "sms message", "2025-11-18T00:00:00.000Z", author_channel="SMS"
        )
        await channel.process_webhook(webhook)
        assert captured == []

    @pytest.mark.asyncio
    async def test_conversation_ended_callback(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        captured: list[ConversationSession] = []

        def handler(ctx: ConversationSession) -> None:
            captured.append(ctx)

        tac.on_conversation_ended(handler)

        channel._conversations["CH_CB1"] = ConversationSession(
            conversation_id="CH_CB1",
            channel="chat",
            profile_id="prof_cb1",
        )

        await channel.process_webhook(
            create_conversation_updated_webhook("CH_CB1", "CLOSED", "2025-11-18T00:10:00.000Z")
        )

        assert len(captured) == 1
        assert captured[0].conversation_id == "CH_CB1"
        assert captured[0].channel == "chat"

    @pytest.mark.asyncio
    async def test_send_response_uses_resolved_agent_id(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)

        channel._conversations["CH123"] = ConversationSession(
            conversation_id="CH123",
            channel="chat",
            author_info=AuthorInfo(address="user@example.com", participant_id="PA_CUSTOMER"),
        )
        channel._conversations["CH123"].metadata.update(
            {"agent_participant_id": "PA_AGENT", "channel_id": "CH_CHAT_SID_123"}
        )

        with patch.object(
            tac.conversation_orchestrator_client, "create_action"
        ) as mock_create_action:
            await channel.send_response("CH123", "Hi there")

        mock_create_action.assert_called_once()
        request = mock_create_action.call_args[0][1]
        assert request.payload.from_.participant_id == "PA_AGENT"
        assert request.payload.from_.channel == "CHAT"
        assert request.payload.to[0].participant_id == "PA_CUSTOMER"
        assert request.payload.channel_settings is not None
        assert request.payload.channel_settings.channel_id == "CH_CHAT_SID_123"

    @pytest.mark.asyncio
    async def test_send_response_forwards_chat_service_when_set(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        tac.conversations_v1_service_sid = "ISabc123"

        channel._conversations["CH123"] = ConversationSession(
            conversation_id="CH123",
            channel="chat",
            author_info=AuthorInfo(address="user@example.com", participant_id="PA_CUSTOMER"),
        )
        channel._conversations["CH123"].metadata.update(
            {"agent_participant_id": "PA_AGENT", "channel_id": "CH_CHAT_SID_123"}
        )

        with patch.object(
            tac.conversation_orchestrator_client, "create_action"
        ) as mock_create_action:
            await channel.send_response("CH123", "Hi")

        request = mock_create_action.call_args[0][1]
        assert request.payload.channel_settings.chat_service == "ISabc123"

    @pytest.mark.asyncio
    async def test_send_response_raises_without_agent_id(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        channel._conversations["CH_NO_AGENT"] = ConversationSession(
            conversation_id="CH_NO_AGENT",
            channel="chat",
            author_info=AuthorInfo(address="user@example.com", participant_id="PA_CUSTOMER"),
        )
        channel._conversations["CH_NO_AGENT"].metadata["channel_id"] = "CH_CHAT_SID_123"
        with pytest.raises(RuntimeError, match="Agent participant id not resolved"):
            await channel.send_response("CH_NO_AGENT", "Hi")

    @pytest.mark.asyncio
    async def test_send_response_raises_without_channel_id(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        channel._conversations["CH_NO_CID"] = ConversationSession(
            conversation_id="CH_NO_CID",
            channel="chat",
            author_info=AuthorInfo(address="user@example.com", participant_id="PA_CUSTOMER"),
        )
        with pytest.raises(RuntimeError, match="channel_id"):
            await channel.send_response("CH_NO_CID", "Hi")

    @pytest.mark.asyncio
    async def test_deduplication(self) -> None:
        """Same idempotency token processed once."""
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        captured: list[str] = []
        tac.on_message_ready(lambda m, c, mem: captured.append(m))

        participants = [
            make_participant("PA_AGENT", "AI_AGENT", "ai-assistant", channel_id="CH_CHAT_SID_123"),
            make_participant(
                "PA_CUSTOMER", "CUSTOMER", "user@example.com", channel_id="CH_CHAT_SID_123"
            ),
        ]

        webhook = create_communication_created_webhook(
            "CH123", "PA_CUSTOMER", "Hi", "2025-11-18T00:00:01.000Z"
        )

        with patch.object(
            tac.conversation_orchestrator_client, "list_participants", return_value=participants
        ):
            await channel.process_webhook(webhook, idempotency_token="tok123")
            await channel.process_webhook(webhook, idempotency_token="tok123")

        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_auto_retrieve_memory(self) -> None:
        tac = TAC(get_test_config())

        from tac.context.memory import MemoryClient
        from tac.models.memory import ProfileLookupResponse

        tac.conversation_memory_client = MemoryClient(
            store_id="MGtest123",
            api_key=tac.config.api_key,
            api_secret=tac.config.api_secret,
        )

        channel = ChatChannel(tac, config={"auto_retrieve_memory": True})

        empty_response = MemoryRetrievalResponse(
            observations=[], summaries=[], meta=MemoryRetrievalMeta(queryTime=0)
        )
        tac.conversation_memory_client.retrieve_memory = AsyncMock(return_value=empty_response)
        tac.conversation_memory_client.lookup_profile = AsyncMock(
            return_value=ProfileLookupResponse(
                profiles=["mem_profile_test"], normalizedValue="user@example.com"
            )
        )
        tac.conversation_memory_client.get_profile = AsyncMock(side_effect=Exception("skip"))

        participants = [
            make_participant("PA_AGENT", "AI_AGENT", "ai-assistant", channel_id="CH_CHAT_SID_123"),
            make_participant(
                "PA_CUSTOMER",
                "CUSTOMER",
                "user@example.com",
                channel_id="CH_CHAT_SID_123",
            ),
        ]

        webhook = create_communication_created_webhook(
            "CH123", "PA_CUSTOMER", "Hi", "2025-11-18T00:00:01.000Z"
        )

        with patch.object(
            tac.conversation_orchestrator_client, "list_participants", return_value=participants
        ):
            await channel.process_webhook(webhook)

        tac.conversation_memory_client.retrieve_memory.assert_called_once()

"""Tests for Chat Channel."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tac import TAC
from tac.channels.chat import ChatChannel, ChatChannelConfig
from tac.models.memory import MemoryRetrievalMeta, MemoryRetrievalResponse
from tac.models.session import AuthorInfo, ConversationSession
from tac.models.tac import TACMemoryResponse


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

    def test_is_default_agent_address(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        assert channel.is_default_agent_address("ai-assistant") is True
        assert channel.is_default_agent_address("user@example.com") is False

    @pytest.mark.asyncio
    async def test_process_message(self) -> None:
        from tac.models.conversation import ParticipantAddress, ParticipantResponse

        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        captured_messages: list[str] = []

        def message_callback(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None,
        ) -> None:
            captured_messages.append(user_message)

        tac.on_message_ready(message_callback)

        webhook = create_communication_created_webhook(
            "CH123", "PA_USER", "Hello from chat!", "2025-11-18T00:00:00.000Z"
        )

        # Mock reconcile to return an agent (customer is None for chat).
        mock_agent = ParticipantResponse(
            **{  # type: ignore[arg-type]
                "id": "PA_AGENT",
                "accountId": "ACtest123",
                "conversationId": "CH123",
                "name": "Test Agent",
                "type": "AI_AGENT",
                "addresses": [
                    ParticipantAddress(
                        channel="CHAT", address="ai-assistant", channel_id="CH_CHAT_SID_123"
                    ).model_dump(by_alias=True)
                ],
            }
        )

        with patch.object(
            channel,
            "_reconcile_participants",
            new=AsyncMock(return_value=(mock_agent, None)),
        ):
            await channel.process_webhook(webhook)

        assert len(captured_messages) == 1
        assert captured_messages[0] == "Hello from chat!"

    @pytest.mark.asyncio
    async def test_channel_id_stored_in_metadata(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)

        tac.on_message_ready(lambda msg, ctx, mem: None)

        webhook = create_communication_created_webhook(
            "CH123",
            "PA_USER",
            "Test",
            "2025-11-18T00:00:00.000Z",
            channel_id="CH_CHAT_SID_456",
        )
        await channel.process_webhook(webhook)

        session = channel._conversations["CH123"]
        assert session.metadata["channel_id"] == "CH_CHAT_SID_456"

    @pytest.mark.asyncio
    async def test_ignores_sms_messages(self) -> None:
        """COMMUNICATION_CREATED with author.channel=SMS is filtered by ChatChannel."""
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        captured: list[str] = []

        tac.on_message_ready(lambda msg, ctx, mem: captured.append(msg))

        webhook = create_communication_created_webhook(
            "CH123",
            "PA_USER",
            "SMS message",
            "2025-11-18T00:00:00.000Z",
            author_channel="SMS",
        )
        await channel.process_webhook(webhook)

        assert len(captured) == 0

    @pytest.mark.asyncio
    async def test_ignores_own_messages(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        captured: list[str] = []

        tac.on_message_ready(lambda msg, ctx, mem: captured.append(msg))

        webhook = create_communication_created_webhook(
            "CH123",
            "PA_AGENT",
            "Bot response",
            "2025-11-18T00:00:00.000Z",
            author_address="ai-assistant",
        )
        await channel.process_webhook(webhook)

        assert len(captured) == 0

    @pytest.mark.asyncio
    async def test_ignores_empty_messages(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        captured: list[str] = []

        tac.on_message_ready(lambda msg, ctx, mem: captured.append(msg))

        webhook = create_communication_created_webhook(
            "CH123", "PA_USER", "", "2025-11-18T00:00:00.000Z"
        )
        await channel.process_webhook(webhook)

        assert len(captured) == 0

    @pytest.mark.asyncio
    async def test_process_conversation_ended(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)

        channel._conversations["CH123"] = ConversationSession(
            conversation_id="CH123", channel="chat", profile_id="profile_123"
        )

        await channel.process_webhook(
            create_conversation_updated_webhook("CH123", "CLOSED", "2025-11-18T00:10:00.000Z")
        )
        assert "CH123" not in channel._conversations

    @pytest.mark.asyncio
    async def test_conversation_ended_callback(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        captured: list[ConversationSession] = []

        tac.on_conversation_ended(lambda ctx: captured.append(ctx))

        channel._conversations["CH123"] = ConversationSession(
            conversation_id="CH123", channel="chat", profile_id="profile_123"
        )
        await channel.process_webhook(
            create_conversation_updated_webhook("CH123", "CLOSED", "2025-11-18T00:10:00.000Z")
        )

        assert len(captured) == 1
        assert captured[0].conversation_id == "CH123"
        assert captured[0].channel == "chat"

    @pytest.mark.asyncio
    async def test_send_response_uses_stashed_ids(self) -> None:
        """send_response reads both participant ids from the session stash."""
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)

        # Session is pre-populated as if reconcile (or outbound initiation) ran.
        channel._conversations["CH123"] = ConversationSession(
            conversation_id="CH123",
            channel="chat",
            author_info=AuthorInfo(address="user@example.com", participant_id="PA_USER"),
            ai_agent_info=AuthorInfo(address="ai-assistant", participant_id="PA_AGENT"),
            metadata={"channel_id": "CH_CHAT_SID_123"},
        )

        with patch.object(tac.conversation_orchestrator_client, "create_action") as mock_send:
            await channel.send_response("CH123", "Hello from bot!")

            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[0][0] == "CH123"
            request = call_args[0][1]
            # from/to send participantId + channel only (no address) for Mode 1 resolution
            assert request.payload.from_.participant_id == "PA_AGENT"
            assert request.payload.from_.channel == "CHAT"
            assert request.payload.from_.address is None
            assert request.payload.content.text == "Hello from bot!"
            assert request.payload.to[0].participant_id == "PA_USER"
            assert request.payload.to[0].channel == "CHAT"
            assert request.payload.to[0].address is None
            assert request.payload.channel_settings is not None
            assert request.payload.channel_settings.channel_id == "CH_CHAT_SID_123"

    @pytest.mark.asyncio
    async def test_send_response_no_session(self) -> None:
        """Missing session → send_response raises (no conversation to reply to)."""
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        with pytest.raises(RuntimeError, match="without a reconciled session"):
            await channel.send_response("CH999", "Hello!")

    @pytest.mark.asyncio
    async def test_send_response_no_channel_id(self) -> None:
        """When session has no channel_id, send raises without calling create_action."""
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)

        channel._conversations["CH123"] = ConversationSession(
            conversation_id="CH123",
            channel="chat",
            author_info=AuthorInfo(address="user@example.com", participant_id="PA_USER"),
            ai_agent_info=AuthorInfo(address="ai-assistant", participant_id="PA_AGENT"),
            metadata={},  # No channel_id
        )

        with patch.object(tac.conversation_orchestrator_client, "create_action") as mock_send:
            with pytest.raises(RuntimeError, match=r"session\.metadata\['channel_id'\]"):
                await channel.send_response("CH123", "Hello!")
            mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_response_raises_when_missing_ai_agent_info(self) -> None:
        """If author_info is stashed but ai_agent_info is not (reconcile failed),
        send_response raises — it will not invent ids."""
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)

        channel._conversations["CH123"] = ConversationSession(
            conversation_id="CH123",
            channel="chat",
            author_info=AuthorInfo(address="user@example.com", participant_id="PA_USER"),
            # ai_agent_info is intentionally missing
            metadata={"channel_id": "CH_CHAT_SID_123"},
        )

        with patch.object(tac.conversation_orchestrator_client, "create_action") as mock_send:
            with pytest.raises(RuntimeError, match="without a reconciled session"):
                await channel.send_response("CH123", "Hello!")
            mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_response_rejects_non_string(self) -> None:
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        with pytest.raises(TypeError, match="Chat channel only supports string responses"):
            await channel.send_response("CH123", 123)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_deduplication(self) -> None:
        from tac.models.conversation import ParticipantAddress, ParticipantResponse

        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        captured: list[str] = []

        tac.on_message_ready(lambda msg, ctx, mem: captured.append(msg))

        webhook = create_communication_created_webhook(
            "CH123", "PA_USER", "Dedup test", "2025-11-18T00:00:00.000Z"
        )

        # Mock reconcile to return an agent (customer is None for chat).
        mock_agent = ParticipantResponse(
            **{  # type: ignore[arg-type]
                "id": "PA_AGENT",
                "accountId": "ACtest123",
                "conversationId": "CH123",
                "name": "Test Agent",
                "type": "AI_AGENT",
                "addresses": [
                    ParticipantAddress(
                        channel="CHAT", address="ai-assistant", channel_id="CH_CHAT_SID_123"
                    ).model_dump(by_alias=True)
                ],
            }
        )

        with patch.object(
            channel,
            "_reconcile_participants",
            new=AsyncMock(return_value=(mock_agent, None)),
        ):
            await channel.process_webhook(webhook, idempotency_token="token_1")
            await channel.process_webhook(webhook, idempotency_token="token_1")

        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_skips_webhook_with_missing_data(self) -> None:
        """Webhook with missing or null data field should be skipped gracefully."""
        tac = TAC(get_test_config())
        channel = ChatChannel(tac)
        captured: list[str] = []

        tac.on_message_ready(lambda msg, ctx, mem: captured.append(msg))

        # Missing data field entirely
        await channel.process_webhook({"eventType": "COMMUNICATION_CREATED"})
        # Null data field
        await channel.process_webhook({"eventType": "COMMUNICATION_CREATED", "data": None})

        assert len(captured) == 0
        assert len(channel._conversations) == 0

    @pytest.mark.asyncio
    async def test_memory_mode(self) -> None:
        from tac.context.memory import MemoryClient
        from tac.models.conversation import ParticipantAddress, ParticipantResponse

        tac = TAC(get_test_config())
        tac.conversation_memory_client = MemoryClient(
            store_id="MGtest123",
            api_key=tac.config.api_key,
            api_secret=tac.config.api_secret,
        )
        channel = ChatChannel(tac, config={"memory_mode": "always"})

        # Pre-seed session with profile_id so retrieve_memory skips the
        # lookup_profile fallback path.
        channel._conversations["CH123"] = ConversationSession(
            conversation_id="CH123", channel="chat", profile_id="profile_test_123"
        )
        tac.conversation_memory_client.get_profile = AsyncMock(
            side_effect=Exception("skip profile")
        )

        captured_memory: list[TACMemoryResponse | None] = []

        def callback(msg: str, ctx: ConversationSession, mem: TACMemoryResponse | None) -> None:
            captured_memory.append(mem)

        tac.on_message_ready(callback)

        empty_response = MemoryRetrievalResponse(
            observations=[], summaries=[], meta=MemoryRetrievalMeta(queryTime=0)
        )
        tac.conversation_memory_client.retrieve_memory = AsyncMock(return_value=empty_response)

        # Mock reconcile to return an agent (customer is None for chat).
        mock_agent = ParticipantResponse(
            **{  # type: ignore[arg-type]
                "id": "PA_AGENT",
                "accountId": "ACtest123",
                "conversationId": "CH123",
                "name": "Test Agent",
                "type": "AI_AGENT",
                "addresses": [
                    ParticipantAddress(
                        channel="CHAT", address="ai-assistant", channel_id="CH_CHAT_SID_123"
                    ).model_dump(by_alias=True)
                ],
            }
        )

        webhook = create_communication_created_webhook(
            "CH123", "PA_USER", "Memory test", "2025-11-18T00:00:02.000Z"
        )
        with patch.object(
            channel,
            "_reconcile_participants",
            new=AsyncMock(return_value=(mock_agent, None)),
        ):
            await channel.process_webhook(webhook)

        tac.conversation_memory_client.retrieve_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_auto_send_response(self) -> None:
        """Test callback returning string automatically sends response via create_action."""
        from tac.models.conversation import ParticipantAddress, ParticipantResponse

        tac = TAC(get_test_config())
        channel = ChatChannel(tac, config={"memory_mode": "never"})

        # Callback that returns a string (should auto-send)
        async def message_callback(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None,
        ) -> str:
            return "This is my automated response"

        tac.on_message_ready(message_callback)

        # Mock reconcile to return an agent (customer is None for chat —
        # ChatChannel has reconcile_customer_type=False and identifies the
        # customer author-driven from the webhook).
        mock_agent = ParticipantResponse(
            **{  # type: ignore[arg-type]
                "id": "PA_AGENT",
                "accountId": "ACtest123",
                "conversationId": "CH_AUTO_SEND",
                "name": "Test Agent",
                "type": "AI_AGENT",
                "addresses": [
                    ParticipantAddress(
                        channel="CHAT", address="ai-assistant", channel_id="CH_CHAT_SID_123"
                    ).model_dump(by_alias=True)
                ],
            }
        )

        with (
            patch.object(
                channel,
                "_reconcile_participants",
                new=AsyncMock(return_value=(mock_agent, None)),
            ),
            patch.object(
                tac.conversation_orchestrator_client, "create_action"
            ) as mock_create_action,
        ):
            # Process message that triggers callback
            message_webhook = create_communication_created_webhook(
                "CH_AUTO_SEND", "PA_USER", "Test message", "2025-11-18T00:00:01.000Z"
            )
            await channel.process_webhook(message_webhook)

            # Verify create_action was called once with auto-sent response
            mock_create_action.assert_called_once()
            call_args = mock_create_action.call_args
            assert call_args[0][0] == "CH_AUTO_SEND"
            request = call_args[0][1]
            assert request.payload.content.text == "This is my automated response"

    @pytest.mark.asyncio
    async def test_callback_no_auto_send_on_none(self) -> None:
        """Test that callback returning None does not auto-send (manual send_response required)."""
        tac = TAC(get_test_config())
        channel = ChatChannel(tac, config={"memory_mode": "never"})

        # Callback that returns None (manual send_response flow)
        async def message_callback(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None,
        ) -> None:
            # User will manually call channel.send_response() later
            pass

        tac.on_message_ready(message_callback)

        with patch.object(
            tac.conversation_orchestrator_client, "create_action"
        ) as mock_create_action:
            # Process message that triggers callback
            message_webhook = create_communication_created_webhook(
                "CH_NO_AUTO", "PA_NO_AUTO", "Test message", "2025-11-18T00:00:01.000Z"
            )
            await channel.process_webhook(message_webhook)

            # Verify create_action was NOT called (callback returned None)
            mock_create_action.assert_not_called()

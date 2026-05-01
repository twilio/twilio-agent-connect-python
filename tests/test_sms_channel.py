"""Tests for SMS Channel."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tac import TAC
from tac.channels.sms import SMSChannel
from tac.models.memory import MemoryRetrievalMeta, MemoryRetrievalResponse
from tac.models.session import AuthorInfo, ConversationSession
from tac.models.tac import TACMemoryResponse


def create_conversation_created_webhook(conversation_id: str, timestamp: str) -> dict[str, Any]:
    """Create a CONVERSATION_CREATED webhook event."""
    return {
        "eventType": "CONVERSATION_CREATED",
        "timestamp": timestamp,
        "data": {
            "id": conversation_id,
            "accountId": "ACtest123",
            "serviceId": "IStest123",
            "status": "ACTIVE",
            "name": "Test Conversation",
            "createdAt": timestamp,
            "updatedAt": timestamp,
            "configuration": {"intelligenceServiceIds": []},
        },
    }


def create_communication_created_webhook(
    conversation_id: str,
    participant_id: str,
    message_text: str,
    timestamp: str,
    author_address: str = "+12345678901",
) -> dict[str, Any]:
    """Create a COMMUNICATION_CREATED webhook event."""
    # Generate unique communication ID using timestamp to avoid deduplication
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
                "channel": "SMS",
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
    """Create a CONVERSATION_UPDATED webhook event."""
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
    """Get a valid test configuration."""
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
        """Test SMS channel initialization."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        assert channel.tac == tac

    def test_initialization_without_phone_number(self) -> None:
        """Test TAC config validation fails without phone_number."""
        config = get_test_config()
        del config["phone_number"]

        # phone_number is now required at TACConfig level
        with pytest.raises(ValueError):
            TAC(config)

    @pytest.mark.asyncio
    async def test_process_message_auto_initialize(self) -> None:
        """Test processing message auto-initializes conversation if not started."""
        from tac.models.conversation import ParticipantAddress, ParticipantResponse

        tac = TAC(get_test_config())

        # Manually create memory_client for this test
        from tac.context.memory import MemoryClient

        tac.conversation_memory_client = MemoryClient(
            store_id="MGtest123",
            api_key=tac.config.api_key,
            api_secret=tac.config.api_secret,
        )

        channel = SMSChannel(tac)

        # Callback to capture context
        captured_context = None
        captured_memories = None

        def message_callback(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None,
        ) -> None:
            nonlocal captured_context, captured_memories
            captured_context = context
            captured_memories = memory_response

        tac.on_message_ready(message_callback)

        webhook_data = create_communication_created_webhook(
            "CH123456", "MB123", "Hello, I need help", "2025-11-18T00:00:00.000Z"
        )

        empty_response = MemoryRetrievalResponse(
            observations=[],
            summaries=[],
            meta=MemoryRetrievalMeta(queryTime=0),
        )
        tac.conversation_memory_client.retrieve_memory = AsyncMock(return_value=empty_response)

        # Mock reconcile to return (agent, customer) so the callback fires.
        mock_agent = ParticipantResponse(
            **{  # type: ignore[arg-type]
                "id": "PA_AGENT",
                "accountId": "ACtest123",
                "conversationId": "CH123456",
                "name": "Test Agent",
                "type": "AI_AGENT",
                "addresses": [
                    ParticipantAddress(channel="SMS", address="+15551234567").model_dump(
                        by_alias=True
                    )
                ],
            }
        )
        mock_customer = ParticipantResponse(
            **{  # type: ignore[arg-type]
                "id": "PA_CUSTOMER",
                "accountId": "ACtest123",
                "conversationId": "CH123456",
                "name": "+12345678901",
                "type": "CUSTOMER",
                "addresses": [
                    ParticipantAddress(channel="SMS", address="+12345678901").model_dump(
                        by_alias=True
                    )
                ],
            }
        )

        with patch.object(
            channel,
            "_reconcile_participants",
            new=AsyncMock(return_value=(mock_agent, mock_customer)),
        ):
            await channel.process_webhook(webhook_data)

        # Verify callback was invoked
        assert captured_context is not None
        assert captured_context.conversation_id == "CH123456"
        # No profile_id since message auto-initialized without participant.added event
        assert captured_context.profile_id is None
        assert captured_context.channel == "sms"

    @pytest.mark.asyncio
    async def test_process_message_with_existing_conversation(self) -> None:
        """Test processing message with pre-existing conversation."""
        from tac.models.conversation import ParticipantAddress, ParticipantResponse

        tac = TAC(get_test_config())

        # Manually create memory_client for this test
        from tac.context.memory import MemoryClient

        tac.conversation_memory_client = MemoryClient(
            store_id="MGtest123",
            api_key=tac.config.api_key,
            api_secret=tac.config.api_secret,
        )

        channel = SMSChannel(
            tac, config={"auto_retrieve_memory": True}
        )  # Enable memory retrieval for test

        # Message webhook auto-initializes the session.
        message_webhook = create_communication_created_webhook(
            "CH123456", "MB123", "Test message", "2025-11-18T00:00:02.000Z"
        )

        empty_response = MemoryRetrievalResponse(
            observations=[],
            summaries=[],
            meta=MemoryRetrievalMeta(queryTime=0),
        )
        tac.conversation_memory_client.retrieve_memory = AsyncMock(return_value=empty_response)

        # Mock reconcile to return (agent, customer) so the callback fires.
        # Customer carries a profile_id so retrieve_memory skips the
        # lookup_profile fallback path.
        mock_agent = ParticipantResponse(
            **{  # type: ignore[arg-type]
                "id": "PA_AGENT",
                "accountId": "ACtest123",
                "conversationId": "CH123456",
                "name": "Test Agent",
                "type": "AI_AGENT",
                "addresses": [
                    ParticipantAddress(channel="SMS", address="+15551234567").model_dump(
                        by_alias=True
                    )
                ],
            }
        )
        mock_customer = ParticipantResponse(
            **{  # type: ignore[arg-type]
                "id": "PA_CUSTOMER",
                "accountId": "ACtest123",
                "conversationId": "CH123456",
                "name": "+12345678901",
                "type": "CUSTOMER",
                "profileId": "profile_test_123",
                "addresses": [
                    ParticipantAddress(channel="SMS", address="+12345678901").model_dump(
                        by_alias=True
                    )
                ],
            }
        )
        tac.conversation_memory_client.get_profile = AsyncMock(
            side_effect=Exception("skip profile")
        )

        with patch.object(
            channel,
            "_reconcile_participants",
            new=AsyncMock(return_value=(mock_agent, mock_customer)),
        ):
            await channel.process_webhook(message_webhook)

        # Verify memory retrieval was called
        tac.conversation_memory_client.retrieve_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_empty_message_ignored(self) -> None:
        """Test that empty messages are ignored."""
        tac = TAC(get_test_config())

        # Manually create memory_client for this test
        from tac.context.memory import MemoryClient

        tac.conversation_memory_client = MemoryClient(
            store_id="MGtest123",
            api_key=tac.config.api_key,
            api_secret=tac.config.api_secret,
        )

        channel = SMSChannel(tac)

        webhook_data = create_communication_created_webhook(
            "CH123456", "MB123", "", "2025-11-18T00:00:00.000Z"
        )

        tac.conversation_memory_client.retrieve_memory = AsyncMock()

        await channel.process_webhook(webhook_data)

        # Verify memory retrieval was NOT called
        tac.conversation_memory_client.retrieve_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_conversation_ended(self) -> None:
        """Test processing onConversationRemoved event."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        channel._conversations["CH123456"] = ConversationSession(
            conversation_id="CH123456",
            channel="sms",
            profile_id="profile_test_123",
        )

        # End conversation (status changed to CLOSED)
        end_webhook = create_conversation_updated_webhook(
            "CH123456", "CLOSED", "2025-11-18T00:10:00.000Z"
        )

        # Should not raise
        await channel.process_webhook(end_webhook)

    @pytest.mark.asyncio
    async def test_send_response_with_active_conversation(self) -> None:
        """Test sending response reads ids from the stashed session."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        # Session is pre-populated as if reconcile (or outbound initiation) ran.
        channel._conversations["CH123456"] = ConversationSession(
            conversation_id="CH123456",
            channel="sms",
            author_info=AuthorInfo(address="+12345678901", participant_id="PA_CUSTOMER"),
            ai_agent_info=AuthorInfo(address="+15551234567", participant_id="PA_AGENT"),
        )

        with patch.object(
            tac.conversation_orchestrator_client, "create_action"
        ) as mock_create_action:
            await channel.send_response("CH123456", "Test response")

            # Verify create_action was called
            mock_create_action.assert_called_once()
            call_args = mock_create_action.call_args
            assert call_args[0][0] == "CH123456"  # conversation_id

            # Verify request structure
            request = call_args[0][1]
            assert request.type == "SEND_MESSAGE"
            # from/to send participantId + channel only (no address) for Mode 1 resolution
            assert request.payload.from_.participant_id == "PA_AGENT"
            assert request.payload.from_.channel == "SMS"
            assert request.payload.from_.address is None
            assert request.payload.content.text == "Test response"
            assert len(request.payload.to) == 1
            assert request.payload.to[0].participant_id == "PA_CUSTOMER"
            assert request.payload.to[0].channel == "SMS"
            assert request.payload.to[0].address is None
            # No channel_id in metadata → channelSettings omitted
            assert request.payload.channel_settings is None

    @pytest.mark.asyncio
    async def test_send_response_forwards_channel_id_when_present(self) -> None:
        """When session.metadata has channel_id, it's forwarded as channelSettings.channelId."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        # Seed a session with participant ids + channel_id in metadata
        # (as inbound ingestion + reconcile would).
        channel._conversations["CH_WITH_CH_ID"] = ConversationSession(
            conversation_id="CH_WITH_CH_ID",
            channel="sms",
            author_info=AuthorInfo(address="+12345678901", participant_id="PA_CUSTOMER"),
            ai_agent_info=AuthorInfo(address="+15551234567", participant_id="PA_AGENT"),
            metadata={"channel_id": "SMabcdef"},
        )

        with patch.object(
            tac.conversation_orchestrator_client, "create_action"
        ) as mock_create_action:
            await channel.send_response("CH_WITH_CH_ID", "Test response")

            mock_create_action.assert_called_once()
            request = mock_create_action.call_args[0][1]
            assert request.payload.channel_settings is not None
            assert request.payload.channel_settings.channel_id == "SMabcdef"

    @pytest.mark.asyncio
    async def test_multiple_concurrent_conversations(self) -> None:
        """Test handling multiple concurrent conversations."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        channel._conversations["CH111"] = ConversationSession(
            conversation_id="CH111", channel="sms", profile_id="PR111"
        )
        channel._conversations["CH222"] = ConversationSession(
            conversation_id="CH222", channel="sms", profile_id="PR222"
        )

        # Verify both conversations tracked.
        assert "CH111" in channel._conversations
        assert "CH222" in channel._conversations

        # End first conversation (should not raise)
        await channel.process_webhook(
            create_conversation_updated_webhook("CH111", "CLOSED", "2025-11-18T00:10:00.000Z")
        )

        # Verify first conversation was removed
        assert "CH111" not in channel._conversations
        assert "CH222" in channel._conversations

    @pytest.mark.asyncio
    async def test_ignores_unsupported_event_types(self) -> None:
        """Test that unsupported event types are ignored."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        webhook_data = {
            "eventType": "SOME_UNSUPPORTED_EVENT",
            "timestamp": "2025-11-18T00:00:00.000Z",
            "data": {"id": "CH123456"},
        }

        # Should not raise, just log debug message
        await channel.process_webhook(webhook_data)

    @pytest.mark.asyncio
    async def test_conversation_ended_callback_fires_on_close(self) -> None:
        """Closing an SMS conversation triggers on_conversation_ended with correct data."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        captured: list[ConversationSession] = []

        def handler(ctx: ConversationSession) -> None:
            captured.append(ctx)

        tac.on_conversation_ended(handler)

        channel._conversations["CH_CB1"] = ConversationSession(
            conversation_id="CH_CB1", channel="sms", profile_id="prof_cb1"
        )

        # Close conversation
        await channel.process_webhook(
            create_conversation_updated_webhook("CH_CB1", "CLOSED", "2025-11-18T00:10:00.000Z")
        )

        assert len(captured) == 1
        assert captured[0].conversation_id == "CH_CB1"
        assert captured[0].profile_id == "prof_cb1"
        assert captured[0].channel == "sms"

    @pytest.mark.asyncio
    async def test_conversation_ended_callback_error_does_not_prevent_cleanup(self) -> None:
        """If on_conversation_ended callback raises, the session is still cleaned up."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        def bad_handler(ctx: ConversationSession) -> None:
            raise RuntimeError("boom")

        tac.on_conversation_ended(bad_handler)

        channel._conversations["CH_CB2"] = ConversationSession(
            conversation_id="CH_CB2", channel="sms", profile_id="prof_cb2"
        )
        await channel.process_webhook(
            create_conversation_updated_webhook("CH_CB2", "CLOSED", "2025-11-18T00:10:00.000Z")
        )

        # Session should still be cleaned up despite the error
        assert "CH_CB2" not in channel._conversations

    @pytest.mark.asyncio
    async def test_conversation_ended_async_callback(self) -> None:
        """Async on_conversation_ended callback is awaited correctly."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        captured: list[ConversationSession] = []

        async def async_handler(ctx: ConversationSession) -> None:
            captured.append(ctx)

        tac.on_conversation_ended(async_handler)

        channel._conversations["CH_ASYNC1"] = ConversationSession(
            conversation_id="CH_ASYNC1", channel="sms", profile_id="prof_async1"
        )
        await channel.process_webhook(
            create_conversation_updated_webhook("CH_ASYNC1", "CLOSED", "2025-11-18T00:10:00.000Z")
        )

        assert len(captured) == 1
        assert captured[0].conversation_id == "CH_ASYNC1"
        assert captured[0].channel == "sms"

    @pytest.mark.asyncio
    async def test_conversation_ended_no_callback_registered(self) -> None:
        """Closing a conversation without a registered callback cleans up silently."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        # No callback registered — should not raise
        channel._conversations["CH_NOCB"] = ConversationSession(
            conversation_id="CH_NOCB", channel="sms", profile_id="prof_nocb"
        )
        await channel.process_webhook(
            create_conversation_updated_webhook("CH_NOCB", "CLOSED", "2025-11-18T00:10:00.000Z")
        )

        assert "CH_NOCB" not in channel._conversations

    @pytest.mark.asyncio
    async def test_send_response_raises_when_no_customer_on_sms(self) -> None:
        """If the session has no author_info, send_response raises — reconcile
        (or outbound initiation) must stash both participant ids first."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        # ai_agent_info is set, but author_info is missing — misuse.
        channel._conversations["CH123456"] = ConversationSession(
            conversation_id="CH123456",
            channel="sms",
            ai_agent_info=AuthorInfo(address="+15551234567", participant_id="PA_AGENT"),
        )

        with pytest.raises(RuntimeError, match="without a reconciled session"):
            await channel.send_response("CH123456", "Reply")

    @pytest.mark.asyncio
    async def test_ignores_chat_messages(self) -> None:
        """COMMUNICATION_CREATED with author.channel=CHAT is filtered by SMSChannel."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        captured: list[str] = []

        tac.on_message_ready(lambda msg, ctx, mem: captured.append(msg))

        webhook = create_communication_created_webhook(
            "CH123456", "MB123", "Chat message", "2025-11-18T00:00:00.000Z"
        )
        # Override author channel to CHAT
        webhook["data"]["author"]["channel"] = "CHAT"

        await channel.process_webhook(webhook)
        assert len(captured) == 0

    @pytest.mark.asyncio
    async def test_ignores_messages_without_author_channel(self) -> None:
        """COMMUNICATION_CREATED without author.channel is rejected for safe fanout."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)
        captured: list[str] = []

        tac.on_message_ready(lambda msg, ctx, mem: captured.append(msg))

        webhook = create_communication_created_webhook(
            "CH123456", "MB123", "No channel", "2025-11-18T00:00:00.000Z"
        )
        # Remove author channel
        del webhook["data"]["author"]["channel"]

        await channel.process_webhook(webhook)
        assert len(captured) == 0

    @pytest.mark.asyncio
    async def test_callback_auto_send_response(self) -> None:
        """Test callback returning string automatically sends response via create_action."""
        from tac.models.conversation import ParticipantAddress, ParticipantResponse

        tac = TAC(get_test_config(with_memory=False))
        channel = SMSChannel(tac, config={"auto_retrieve_memory": False})

        # Callback that returns a string (should auto-send)
        async def message_callback(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None,
        ) -> str:
            return "This is my automated response"

        tac.on_message_ready(message_callback)

        # Mock reconcile to return (agent, customer) — this is what stashes
        # ai_agent_info / author_info on the session pre-callback.
        mock_agent = ParticipantResponse(
            **{  # type: ignore[arg-type]
                "id": "PA_AGENT",
                "accountId": "ACtest123",
                "conversationId": "CH_AUTO_SEND",
                "name": "Test Agent",
                "type": "AI_AGENT",
                "addresses": [
                    ParticipantAddress(channel="SMS", address="+15551234567").model_dump(
                        by_alias=True
                    )
                ],
            }
        )
        mock_customer = ParticipantResponse(
            **{  # type: ignore[arg-type]
                "id": "PA_CUSTOMER",
                "accountId": "ACtest123",
                "conversationId": "CH_AUTO_SEND",
                "name": "+12345678901",
                "type": "CUSTOMER",
                "addresses": [
                    ParticipantAddress(channel="SMS", address="+12345678901").model_dump(
                        by_alias=True
                    )
                ],
            }
        )

        with (
            patch.object(
                channel,
                "_reconcile_participants",
                new=AsyncMock(return_value=(mock_agent, mock_customer)),
            ),
            patch.object(
                tac.conversation_orchestrator_client, "create_action"
            ) as mock_create_action,
        ):
            # Process message that triggers callback
            message_webhook = create_communication_created_webhook(
                "CH_AUTO_SEND", "PA_AUTO", "Test message", "2025-11-18T00:00:01.000Z"
            )
            await channel.process_webhook(message_webhook)

            # Verify create_action was called once with auto-sent response
            mock_create_action.assert_called_once()
            call_args = mock_create_action.call_args
            assert call_args[0][0] == "CH_AUTO_SEND"
            request = call_args[0][1]
            assert request.payload.content.text == "This is my automated response"
            # Webhook fixture has channelId=None, so channel_settings should be omitted
            assert request.payload.channel_settings is None

    @pytest.mark.asyncio
    async def test_callback_no_auto_send_on_none(self) -> None:
        """Test that callback returning None does not auto-send (manual send_response required)."""
        tac = TAC(get_test_config(with_memory=False))
        channel = SMSChannel(tac, config={"auto_retrieve_memory": False})

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

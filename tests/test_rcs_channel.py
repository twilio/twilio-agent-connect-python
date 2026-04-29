"""Tests for RCS Channel."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tac import TAC
from tac.channels.rcs import RCSChannel, RCSChannelConfig
from tac.models.memory import MemoryRetrievalMeta, MemoryRetrievalResponse
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse


def create_participant_added_webhook(
    conversation_id: str,
    participant_id: str,
    profile_id: str,
    timestamp: str,
    address: str = "rcs:+12345678901",
) -> dict[str, Any]:
    """Create a PARTICIPANT_ADDED webhook event for RCS."""
    return {
        "eventType": "PARTICIPANT_ADDED",
        "timestamp": timestamp,
        "data": {
            "id": participant_id,
            "conversationId": conversation_id,
            "accountId": "ACtest123",
            "serviceId": "IStest123",
            "name": address,
            "type": "CUSTOMER",
            "profileId": profile_id,
            "addresses": [{"channel": "RCS", "address": address, "channelId": None}],
            "createdAt": timestamp,
            "updatedAt": timestamp,
        },
    }


def create_communication_created_webhook(
    conversation_id: str,
    participant_id: str,
    message_text: str,
    timestamp: str,
    author_address: str = "rcs:+12345678901",
) -> dict[str, Any]:
    """Create a COMMUNICATION_CREATED webhook event for RCS."""
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
                "channel": "RCS",
                "participantId": participant_id,
            },
            "content": {"type": "TEXT", "text": message_text},
            "channelId": None,
            "recipients": [
                {
                    "address": "rcs:twilio_signal_test_agent",
                    "channel": "RCS",
                    "participantId": "comms_participant_agent",
                    "deliveryStatus": "DELIVERED",
                }
            ],
            "createdAt": timestamp,
            "updatedAt": timestamp,
        },
    }


def create_conversation_updated_webhook(
    conversation_id: str, status: str, timestamp: str, configuration_id: str = "default_config"
) -> dict[str, Any]:
    """Create a CONVERSATION_UPDATED webhook event."""
    return {
        "eventType": "CONVERSATION_UPDATED",
        "timestamp": timestamp,
        "data": {
            "id": conversation_id,
            "accountId": "ACtest123",
            "serviceId": "IStest123",
            "configurationId": configuration_id,
            "status": status,
            "name": "Test Conversation",
            "createdAt": timestamp,
            "updatedAt": timestamp,
        },
    }


def get_test_config() -> dict[str, Any]:
    """Get a valid test configuration for RCS."""
    return {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": "default_config",
        "phone_number": "+15551234567",  # Required by TACConfig
    }


@pytest.fixture
def mock_tac() -> TAC:
    """Create a mock TAC instance."""
    return TAC(get_test_config())


@pytest.fixture
def rcs_channel(mock_tac: TAC) -> RCSChannel:
    """Create RCS channel instance."""
    return RCSChannel(
        mock_tac, config=RCSChannelConfig(agent_address="rcs:twilio_signal_test_agent")
    )


@pytest.mark.asyncio
async def test_rcs_channel_initialization(mock_tac: TAC) -> None:
    """Test RCS channel initialization requires config with agent_address."""
    with pytest.raises(TypeError):
        RCSChannel(mock_tac)  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_rcs_channel_initialization_with_config(mock_tac: TAC) -> None:
    """Test RCS channel initialization with explicit config."""
    config = RCSChannelConfig(agent_address="rcs:twilio_signal_test_agent")
    channel = RCSChannel(mock_tac, config=config)
    assert channel.get_channel_name() == "rcs"
    assert channel.get_channel_type_upper() == "RCS"
    assert channel.agent_address == "rcs:twilio_signal_test_agent"


@pytest.mark.asyncio
async def test_is_default_agent_address_configured(mock_tac: TAC) -> None:
    """Test agent address detection with configured agent address."""
    config = RCSChannelConfig(agent_address="rcs:twilio_signal_test_agent")
    channel = RCSChannel(mock_tac, config=config)

    assert channel.is_default_agent_address("rcs:twilio_signal_test_agent") is True
    assert channel.is_default_agent_address("rcs:+12345678901") is False


@pytest.mark.asyncio
async def test_participant_added_webhook(rcs_channel: RCSChannel) -> None:
    """Test PARTICIPANT_ADDED webhook processing."""
    webhook = create_participant_added_webhook(
        conversation_id="conv_123",
        participant_id="part_customer",
        profile_id="prof_123",
        timestamp="2025-01-15T10:15:30Z",
        address="rcs:+12345678901",
    )

    await rcs_channel.process_webhook(webhook)

    # Verify conversation was started
    assert "conv_123" in rcs_channel._conversations
    session = rcs_channel._conversations["conv_123"]
    assert session.conversation_id == "conv_123"
    assert session.profile_id == "prof_123"
    assert session.channel == "rcs"


@pytest.mark.asyncio
async def test_communication_created_webhook(rcs_channel: RCSChannel) -> None:
    """Test COMMUNICATION_CREATED webhook processing."""
    conversation_id = "conv_123"
    participant_id = "part_customer"
    message_text = "Hello from RCS!"
    timestamp = "2025-01-15T10:15:30Z"

    # Mock the callback
    callback_called = False
    callback_message = None
    callback_context = None

    async def mock_callback(
        message: str, context: ConversationSession, memory: TACMemoryResponse | None
    ) -> str:
        nonlocal callback_called, callback_message, callback_context
        callback_called = True
        callback_message = message
        callback_context = context
        return "RCS response"

    rcs_channel.tac.on_message_ready(mock_callback)

    # Mock send_response to avoid actual API calls
    rcs_channel.send_response = AsyncMock()

    webhook = create_communication_created_webhook(
        conversation_id=conversation_id,
        participant_id=participant_id,
        message_text=message_text,
        timestamp=timestamp,
        author_address="rcs:+12345678901",
    )

    await rcs_channel.process_webhook(webhook)

    # Verify callback was invoked
    assert callback_called
    assert callback_message == message_text
    assert callback_context.conversation_id == conversation_id

    # Verify send_response was called with the callback response
    rcs_channel.send_response.assert_called_once()


@pytest.mark.asyncio
async def test_communication_from_agent_ignored(rcs_channel: RCSChannel) -> None:
    """Test that messages from the agent are ignored."""
    conversation_id = "conv_123"

    # First, process PARTICIPANT_ADDED for agent to detect agent address
    agent_webhook = {
        "eventType": "PARTICIPANT_ADDED",
        "timestamp": "2025-01-15T10:15:29Z",
        "data": {
            "id": "part_agent",
            "conversationId": conversation_id,
            "accountId": "ACtest123",
            "serviceId": "IStest123",
            "name": "Agent",
            "type": "AI_AGENT",
            "profileId": None,
            "addresses": [
                {"channel": "RCS", "address": "rcs:twilio_signal_test_agent", "channelId": None}
            ],
            "createdAt": "2025-01-15T10:15:29Z",
            "updatedAt": "2025-01-15T10:15:29Z",
        },
    }
    await rcs_channel.process_webhook(agent_webhook)

    # Mock the callback
    callback_called = False

    async def mock_callback(
        message: str, context: ConversationSession, memory: TACMemoryResponse | None
    ) -> str:
        nonlocal callback_called
        callback_called = True
        return "Should not be called"

    rcs_channel.tac.on_message_ready(mock_callback)

    # Message from agent (should be detected via cached agent address)
    webhook = create_communication_created_webhook(
        conversation_id=conversation_id,
        participant_id="part_agent",
        message_text="Agent message",
        timestamp="2025-01-15T10:15:30Z",
        author_address="rcs:twilio_signal_test_agent",
    )

    await rcs_channel.process_webhook(webhook)

    # Verify callback was NOT invoked
    assert not callback_called


@pytest.mark.asyncio
async def test_conversation_updated_closed(rcs_channel: RCSChannel) -> None:
    """Test CONVERSATION_UPDATED with CLOSED status."""
    conversation_id = "conv_123"

    # Start a conversation first
    rcs_channel._start_conversation(conversation_id, profile_id="prof_123")
    assert conversation_id in rcs_channel._conversations

    # Mock the conversation ended callback
    callback_called = False
    callback_context = None

    async def mock_ended_callback(context: ConversationSession) -> None:
        nonlocal callback_called, callback_context
        callback_called = True
        callback_context = context

    rcs_channel.tac.on_conversation_ended(mock_ended_callback)

    webhook = create_conversation_updated_webhook(
        conversation_id=conversation_id,
        status="CLOSED",
        timestamp="2025-01-15T10:20:30Z",
        configuration_id="default_config",
    )

    await rcs_channel.process_webhook(webhook)

    # Verify conversation was ended
    assert conversation_id not in rcs_channel._conversations
    assert callback_called
    assert callback_context.conversation_id == conversation_id


@pytest.mark.asyncio
async def test_send_response_type_error(rcs_channel: RCSChannel) -> None:
    """Test that send_response raises TypeError for non-string responses."""
    with pytest.raises(TypeError, match="RCS channel only supports string responses"):

        async def async_gen():
            yield "chunk"

        await rcs_channel.send_response("conv_123", async_gen())


@pytest.mark.asyncio
async def test_webhook_deduplication(rcs_channel: RCSChannel) -> None:
    """Test webhook deduplication using idempotency tokens."""
    webhook = create_communication_created_webhook(
        conversation_id="conv_123",
        participant_id="part_customer",
        message_text="Test message",
        timestamp="2025-01-15T10:15:30Z",
    )

    callback_count = 0

    async def mock_callback(
        message: str, context: ConversationSession, memory: TACMemoryResponse | None
    ) -> str:
        nonlocal callback_count
        callback_count += 1
        return "Response"

    rcs_channel.tac.on_message_ready(mock_callback)
    rcs_channel.send_response = AsyncMock()

    # Process webhook with idempotency token
    idempotency_token = "test_token_123"
    await rcs_channel.process_webhook(webhook, idempotency_token=idempotency_token)
    assert callback_count == 1

    # Process same webhook again with same token - should be deduplicated
    await rcs_channel.process_webhook(webhook, idempotency_token=idempotency_token)
    assert callback_count == 1  # Should not increment


@pytest.mark.asyncio
async def test_initiate_outbound_conversation(mock_tac: TAC) -> None:
    """Test initiating an outbound RCS conversation."""
    from tac.models.outbound import InitiateMessagingConversationOptions

    config = RCSChannelConfig(agent_address="rcs:test_agent")
    channel = RCSChannel(mock_tac, config=config)

    # Mock the conversation orchestrator client methods
    with (
        patch.object(
            channel.tac.conversation_orchestrator_client,
            "create_or_reuse_conversation",
            new_callable=AsyncMock,
        ) as mock_create,
        patch.object(
            channel.tac.conversation_orchestrator_client,
            "list_participants",
            new_callable=AsyncMock,
        ) as mock_list,
        patch.object(
            channel.tac.conversation_orchestrator_client,
            "create_action",
            new_callable=AsyncMock,
        ) as mock_action,
    ):
        from tac.models.conversation import ParticipantResponse

        mock_create.return_value = ("conv_123", False)
        mock_list.return_value = [
            ParticipantResponse(
                id="part_customer",
                conversation_id="conv_123",
                account_id="ACtest123",
                name="Customer",
                type="CUSTOMER",
                addresses=[{"channel": "RCS", "address": "rcs:+16505551234"}],
            ),
            ParticipantResponse(
                id="part_agent",
                conversation_id="conv_123",
                account_id="ACtest123",
                name="Agent",
                type="AI_AGENT",
                addresses=[{"channel": "RCS", "address": "rcs:test_agent"}],
            ),
        ]

        options = InitiateMessagingConversationOptions(
            to="rcs:+16505551234",
            message="Hello from RCS!",
        )

        result = await channel.initiate_outbound_conversation(options)

        assert result.conversation_id == "conv_123"
        assert result.session.conversation_id == "conv_123"
        mock_create.assert_called_once()
        mock_action.assert_called_once()


@pytest.mark.asyncio
async def test_auto_retrieve_memory_enabled(mock_tac: TAC) -> None:
    """Test RCS channel with auto_retrieve_memory enabled."""
    config = RCSChannelConfig(agent_address="rcs:test_agent", auto_retrieve_memory=True)
    channel = RCSChannel(mock_tac, config=config)

    assert channel.auto_retrieve_memory is True

    # Mock retrieve_memory to return test data
    mock_memory = TACMemoryResponse(
        data=MemoryRetrievalResponse(
            observations=[],
            summaries=[],
            communications=[],
            meta=MemoryRetrievalMeta(total_observations=0, total_summaries=0),
        )
    )

    with patch.object(mock_tac, "retrieve_memory", new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.return_value = mock_memory

        callback_called = False
        callback_memory = None

        async def mock_callback(
            message: str, context: ConversationSession, memory: TACMemoryResponse | None
        ) -> str:
            nonlocal callback_called, callback_memory
            callback_called = True
            callback_memory = memory
            return "Response"

        channel.tac.on_message_ready(mock_callback)
        channel.send_response = AsyncMock()

        webhook = create_communication_created_webhook(
            conversation_id="conv_123",
            participant_id="part_customer",
            message_text="Test",
            timestamp="2025-01-15T10:15:30Z",
        )

        await channel.process_webhook(webhook)

        # Verify retrieve_memory was called
        assert mock_retrieve.called
        assert callback_called
        assert callback_memory is not None

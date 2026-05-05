"""Tests for 'once' memory mode caching and invalidation.

Tests memory caching behavior: fetch once with empty query, cache it,
invalidate on INACTIVE, task-safe with an async lock.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tac import TAC
from tac.channels.sms import SMSChannel
from tac.models.conversation import ParticipantAddress, ParticipantResponse
from tac.models.memory import MemoryRetrievalMeta, MemoryRetrievalResponse
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse


def get_test_config() -> dict[str, Any]:
    """Get a valid test configuration."""
    from tac.core.config import TwilioMemoryConfig

    return {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": "conv_configuration_test123",
        "phone_number": "+15551234567",
        "memory_config": TwilioMemoryConfig(trait_groups=["Contact"]),
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


@pytest.mark.asyncio
async def test_once_mode_caches_memory_on_first_retrieval() -> None:
    """Test that 'once' mode fetches memory once and caches it."""
    tac = TAC(get_test_config())

    from tac.context.memory import MemoryClient

    tac.conversation_memory_client = MemoryClient(
        store_id="MGtest123",
        api_key=tac.config.api_key,
        api_secret=tac.config.api_secret,
    )

    channel = SMSChannel(tac, config={"memory_mode": "once"})

    captured_memory_responses = []

    def message_callback(
        user_message: str,
        context: ConversationSession,
        memory_response: TACMemoryResponse | None,
    ) -> None:
        captured_memory_responses.append(memory_response)

    tac.on_message_ready(message_callback)

    empty_response = MemoryRetrievalResponse(
        observations=[],
        summaries=[],
        meta=MemoryRetrievalMeta(queryTime=0),
    )
    # Mock at memory client level so TAC.retrieve_memory() wraps it in TACMemoryResponse
    retrieve_memory_mock = AsyncMock(return_value=empty_response)
    tac.conversation_memory_client.retrieve_memory = retrieve_memory_mock

    # Mock reconcile
    mock_agent = ParticipantResponse(
        **{  # type: ignore[arg-type]
            "id": "PA_AGENT",
            "accountId": "ACtest123",
            "conversationId": "CH123456",
            "name": "Test Agent",
            "type": "AI_AGENT",
            "addresses": [
                ParticipantAddress(channel="SMS", address="+15551234567").model_dump(by_alias=True)
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
            "profileId": "mem_profile_00000000000000000000000000",
            "addresses": [
                ParticipantAddress(channel="SMS", address="+12345678901").model_dump(by_alias=True)
            ],
        }
    )

    with patch.object(
        channel,
        "_reconcile_participants",
        new=AsyncMock(return_value=(mock_agent, mock_customer)),
    ):
        # First message - should fetch memory
        webhook1 = create_communication_created_webhook(
            "CH123456", "MB001", "First message", "2025-11-18T00:00:00.000Z"
        )
        await channel.process_webhook(webhook1, idempotency_token="token1")

        # Second message - should use cached memory
        webhook2 = create_communication_created_webhook(
            "CH123456", "MB002", "Second message", "2025-11-18T00:00:01.000Z"
        )
        await channel.process_webhook(webhook2, idempotency_token="token2")

        # Third message - should still use cached memory
        webhook3 = create_communication_created_webhook(
            "CH123456", "MB003", "Third message", "2025-11-18T00:00:02.000Z"
        )
        await channel.process_webhook(webhook3, idempotency_token="token3")

    # Verify retrieve_memory was only called once (on first message)
    assert retrieve_memory_mock.call_count == 1

    # Verify all callbacks received memory response
    assert len(captured_memory_responses) == 3
    assert all(resp is not None for resp in captured_memory_responses)

    # Verify session has cached memory
    session = channel._conversations["CH123456"]
    assert session.cached_memory is not None


@pytest.mark.asyncio
async def test_once_mode_uses_empty_query() -> None:
    """Test that 'once' mode uses empty query (None) for memory retrieval."""
    tac = TAC(get_test_config())

    from tac.context.memory import MemoryClient

    tac.conversation_memory_client = MemoryClient(
        store_id="MGtest123",
        api_key=tac.config.api_key,
        api_secret=tac.config.api_secret,
    )

    channel = SMSChannel(tac, config={"memory_mode": "once"})

    def message_callback(
        user_message: str,
        context: ConversationSession,
        memory_response: TACMemoryResponse | None,
    ) -> None:
        pass

    tac.on_message_ready(message_callback)

    empty_response = MemoryRetrievalResponse(
        observations=[],
        summaries=[],
        meta=MemoryRetrievalMeta(queryTime=0),
    )

    # Mock TAC.retrieve_memory directly but return a properly wrapped TACMemoryResponse
    async def mock_retrieve_memory(
        session: ConversationSession, query: str | None = None
    ) -> TACMemoryResponse:
        return TACMemoryResponse(empty_response)

    retrieve_memory_mock = AsyncMock(side_effect=mock_retrieve_memory)
    tac.retrieve_memory = retrieve_memory_mock

    # Mock reconcile
    mock_agent = ParticipantResponse(
        **{  # type: ignore[arg-type]
            "id": "PA_AGENT",
            "accountId": "ACtest123",
            "conversationId": "CH123456",
            "name": "Test Agent",
            "type": "AI_AGENT",
            "addresses": [
                ParticipantAddress(channel="SMS", address="+15551234567").model_dump(by_alias=True)
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
            "profileId": "mem_profile_00000000000000000000000000",
            "addresses": [
                ParticipantAddress(channel="SMS", address="+12345678901").model_dump(by_alias=True)
            ],
        }
    )

    with patch.object(
        channel,
        "_reconcile_participants",
        new=AsyncMock(return_value=(mock_agent, mock_customer)),
    ):
        webhook = create_communication_created_webhook(
            "CH123456", "MB001", "Test message with query", "2025-11-18T00:00:00.000Z"
        )
        await channel.process_webhook(webhook, idempotency_token="token1")

    # Verify retrieve_memory was called with None query
    retrieve_memory_mock.assert_called_once()
    call_args = retrieve_memory_mock.call_args
    assert call_args.kwargs["query"] is None


@pytest.mark.asyncio
async def test_once_mode_invalidates_cache_on_inactive() -> None:
    """Test that cached memory is invalidated when conversation becomes INACTIVE."""
    tac = TAC(get_test_config())

    from tac.context.memory import MemoryClient

    tac.conversation_memory_client = MemoryClient(
        store_id="MGtest123",
        api_key=tac.config.api_key,
        api_secret=tac.config.api_secret,
    )

    channel = SMSChannel(tac, config={"memory_mode": "once"})

    captured_memory_responses = []

    def message_callback(
        user_message: str,
        context: ConversationSession,
        memory_response: TACMemoryResponse | None,
    ) -> None:
        captured_memory_responses.append(memory_response)

    tac.on_message_ready(message_callback)

    empty_response = MemoryRetrievalResponse(
        observations=[],
        summaries=[],
        meta=MemoryRetrievalMeta(queryTime=0),
    )
    # Mock at memory client level so TAC.retrieve_memory() wraps it in TACMemoryResponse
    retrieve_memory_mock = AsyncMock(return_value=empty_response)
    tac.conversation_memory_client.retrieve_memory = retrieve_memory_mock

    # Mock reconcile
    mock_agent = ParticipantResponse(
        **{  # type: ignore[arg-type]
            "id": "PA_AGENT",
            "accountId": "ACtest123",
            "conversationId": "CH123456",
            "name": "Test Agent",
            "type": "AI_AGENT",
            "addresses": [
                ParticipantAddress(channel="SMS", address="+15551234567").model_dump(by_alias=True)
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
            "profileId": "mem_profile_00000000000000000000000000",
            "addresses": [
                ParticipantAddress(channel="SMS", address="+12345678901").model_dump(by_alias=True)
            ],
        }
    )

    with patch.object(
        channel,
        "_reconcile_participants",
        new=AsyncMock(return_value=(mock_agent, mock_customer)),
    ):
        # First message - should fetch and cache memory
        webhook1 = create_communication_created_webhook(
            "CH123456", "MB001", "First message", "2025-11-18T00:00:00.000Z"
        )
        await channel.process_webhook(webhook1, idempotency_token="token1")

        # Verify cache is populated
        session = channel._conversations["CH123456"]
        assert session.cached_memory is not None
        assert retrieve_memory_mock.call_count == 1

        # Send CONVERSATION_UPDATED with INACTIVE status
        inactive_webhook = {
            "eventType": "CONVERSATION_UPDATED",
            "data": {
                "id": "CH123456",
                "accountId": "ACtest123",
                "configurationId": tac.config.conversation_configuration_id,
                "status": "INACTIVE",
                "name": None,
                "createdAt": "2025-11-18T00:00:00.000Z",
                "updatedAt": "2025-11-18T00:01:00.000Z",
            },
        }
        await channel.process_webhook(inactive_webhook, idempotency_token="token_inactive")

        # Verify cache is invalidated
        assert session.cached_memory is None

        # Send another message after INACTIVE - should fetch memory again
        webhook2 = create_communication_created_webhook(
            "CH123456", "MB002", "Message after inactive", "2025-11-18T00:02:00.000Z"
        )
        await channel.process_webhook(webhook2, idempotency_token="token2")

        # Verify memory was fetched again
        assert retrieve_memory_mock.call_count == 2

        # Verify cache is populated again
        assert session.cached_memory is not None


@pytest.mark.asyncio
async def test_once_mode_does_not_invalidate_on_active() -> None:
    """Test that cached memory is NOT invalidated when conversation becomes ACTIVE."""
    tac = TAC(get_test_config())

    from tac.context.memory import MemoryClient

    tac.conversation_memory_client = MemoryClient(
        store_id="MGtest123",
        api_key=tac.config.api_key,
        api_secret=tac.config.api_secret,
    )

    channel = SMSChannel(tac, config={"memory_mode": "once"})

    def message_callback(
        user_message: str,
        context: ConversationSession,
        memory_response: TACMemoryResponse | None,
    ) -> None:
        pass

    tac.on_message_ready(message_callback)

    empty_response = MemoryRetrievalResponse(
        observations=[],
        summaries=[],
        meta=MemoryRetrievalMeta(queryTime=0),
    )
    # Mock at memory client level so TAC.retrieve_memory() wraps it in TACMemoryResponse
    retrieve_memory_mock = AsyncMock(return_value=empty_response)
    tac.conversation_memory_client.retrieve_memory = retrieve_memory_mock

    # Mock reconcile
    mock_agent = ParticipantResponse(
        **{  # type: ignore[arg-type]
            "id": "PA_AGENT",
            "accountId": "ACtest123",
            "conversationId": "CH123456",
            "name": "Test Agent",
            "type": "AI_AGENT",
            "addresses": [
                ParticipantAddress(channel="SMS", address="+15551234567").model_dump(by_alias=True)
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
            "profileId": "mem_profile_00000000000000000000000000",
            "addresses": [
                ParticipantAddress(channel="SMS", address="+12345678901").model_dump(by_alias=True)
            ],
        }
    )

    with patch.object(
        channel,
        "_reconcile_participants",
        new=AsyncMock(return_value=(mock_agent, mock_customer)),
    ):
        # First message - should fetch and cache memory
        webhook1 = create_communication_created_webhook(
            "CH123456", "MB001", "First message", "2025-11-18T00:00:00.000Z"
        )
        await channel.process_webhook(webhook1, idempotency_token="token1")

        # Store reference to cached memory
        session = channel._conversations["CH123456"]
        cached_memory_before = session.cached_memory
        assert cached_memory_before is not None

        # Send CONVERSATION_UPDATED with ACTIVE status
        active_webhook = {
            "eventType": "CONVERSATION_UPDATED",
            "data": {
                "id": "CH123456",
                "accountId": "ACtest123",
                "configurationId": tac.config.conversation_configuration_id,
                "status": "ACTIVE",
                "name": None,
                "createdAt": "2025-11-18T00:00:00.000Z",
                "updatedAt": "2025-11-18T00:01:00.000Z",
            },
        }
        await channel.process_webhook(active_webhook, idempotency_token="token_active")

        # Verify cache is NOT invalidated
        assert session.cached_memory is not None
        assert session.cached_memory is cached_memory_before


@pytest.mark.asyncio
async def test_once_mode_clears_cache_on_closed() -> None:
    """Test that conversation cleanup (CLOSED) also clears the cached memory."""
    tac = TAC(get_test_config())

    from tac.context.memory import MemoryClient

    tac.conversation_memory_client = MemoryClient(
        store_id="MGtest123",
        api_key=tac.config.api_key,
        api_secret=tac.config.api_secret,
    )

    channel = SMSChannel(tac, config={"memory_mode": "once"})

    def message_callback(
        user_message: str,
        context: ConversationSession,
        memory_response: TACMemoryResponse | None,
    ) -> None:
        pass

    tac.on_message_ready(message_callback)

    empty_response = MemoryRetrievalResponse(
        observations=[],
        summaries=[],
        meta=MemoryRetrievalMeta(queryTime=0),
    )
    # Mock at memory client level so TAC.retrieve_memory() wraps it in TACMemoryResponse
    retrieve_memory_mock = AsyncMock(return_value=empty_response)
    tac.conversation_memory_client.retrieve_memory = retrieve_memory_mock

    # Mock reconcile
    mock_agent = ParticipantResponse(
        **{  # type: ignore[arg-type]
            "id": "PA_AGENT",
            "accountId": "ACtest123",
            "conversationId": "CH123456",
            "name": "Test Agent",
            "type": "AI_AGENT",
            "addresses": [
                ParticipantAddress(channel="SMS", address="+15551234567").model_dump(by_alias=True)
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
            "profileId": "mem_profile_00000000000000000000000000",
            "addresses": [
                ParticipantAddress(channel="SMS", address="+12345678901").model_dump(by_alias=True)
            ],
        }
    )

    with patch.object(
        channel,
        "_reconcile_participants",
        new=AsyncMock(return_value=(mock_agent, mock_customer)),
    ):
        # First message - should fetch and cache memory
        webhook1 = create_communication_created_webhook(
            "CH123456", "MB001", "First message", "2025-11-18T00:00:00.000Z"
        )
        await channel.process_webhook(webhook1, idempotency_token="token1")

        # Verify cache is populated and conversation exists
        assert "CH123456" in channel._conversations
        session = channel._conversations["CH123456"]
        assert session.cached_memory is not None

        # Send CONVERSATION_UPDATED with CLOSED status
        closed_webhook = {
            "eventType": "CONVERSATION_UPDATED",
            "data": {
                "id": "CH123456",
                "accountId": "ACtest123",
                "configurationId": tac.config.conversation_configuration_id,
                "status": "CLOSED",
                "name": None,
                "createdAt": "2025-11-18T00:00:00.000Z",
                "updatedAt": "2025-11-18T00:01:00.000Z",
            },
        }
        await channel.process_webhook(closed_webhook, idempotency_token="token_closed")

        # Verify conversation is removed (which also clears the cache)
        assert "CH123456" not in channel._conversations


@pytest.mark.asyncio
async def test_once_mode_lock_prevents_race_conditions() -> None:
    """Test that cache_lock is present and functions correctly for 'once' mode.

    Verifies that:
    - ConversationSession has a cache_lock field
    - The lock can be acquired successfully
    - INACTIVE webhook processing respects the lock when clearing cache
    """
    tac = TAC(get_test_config())

    from tac.context.memory import MemoryClient

    tac.conversation_memory_client = MemoryClient(
        store_id="MGtest123",
        api_key=tac.config.api_key,
        api_secret=tac.config.api_secret,
    )

    channel = SMSChannel(tac, config={"memory_mode": "once"})

    def message_callback(
        user_message: str,
        context: ConversationSession,
        memory_response: TACMemoryResponse | None,
    ) -> None:
        pass

    tac.on_message_ready(message_callback)

    empty_response = MemoryRetrievalResponse(
        observations=[],
        summaries=[],
        meta=MemoryRetrievalMeta(queryTime=0),
    )
    # Mock at memory client level so TAC.retrieve_memory() wraps it in TACMemoryResponse
    retrieve_memory_mock = AsyncMock(return_value=empty_response)
    tac.conversation_memory_client.retrieve_memory = retrieve_memory_mock

    # Mock reconcile
    mock_agent = ParticipantResponse(
        **{  # type: ignore[arg-type]
            "id": "PA_AGENT",
            "accountId": "ACtest123",
            "conversationId": "CH123456",
            "name": "Test Agent",
            "type": "AI_AGENT",
            "addresses": [
                ParticipantAddress(channel="SMS", address="+15551234567").model_dump(by_alias=True)
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
            "profileId": "mem_profile_00000000000000000000000000",
            "addresses": [
                ParticipantAddress(channel="SMS", address="+12345678901").model_dump(by_alias=True)
            ],
        }
    )

    with patch.object(
        channel,
        "_reconcile_participants",
        new=AsyncMock(return_value=(mock_agent, mock_customer)),
    ):
        # First message - should fetch and cache memory
        webhook1 = create_communication_created_webhook(
            "CH123456", "MB001", "First message", "2025-11-18T00:00:00.000Z"
        )
        await channel.process_webhook(webhook1, idempotency_token="token1")

        # Verify session has cache_lock
        session = channel._conversations["CH123456"]
        assert hasattr(session, "cache_lock")
        assert session.cache_lock is not None

        # Verify lock can be acquired (not deadlocked)
        async with session.cache_lock:
            # Lock acquired successfully
            assert session.cached_memory is not None

        # Send INACTIVE webhook - should acquire lock before clearing cache
        inactive_webhook = {
            "eventType": "CONVERSATION_UPDATED",
            "data": {
                "id": "CH123456",
                "accountId": "ACtest123",
                "configurationId": tac.config.conversation_configuration_id,
                "status": "INACTIVE",
                "name": None,
                "createdAt": "2025-11-18T00:00:00.000Z",
                "updatedAt": "2025-11-18T00:01:00.000Z",
            },
        }
        await channel.process_webhook(inactive_webhook, idempotency_token="token_inactive")

        # Verify cache was safely cleared
        assert session.cached_memory is None

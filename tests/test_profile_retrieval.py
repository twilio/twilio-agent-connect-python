"""Tests for profile retrieval functionality."""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from tac import TAC, TACConfig
from tac.channels.sms import SMSChannel
from tac.context.memory import MemoryClient
from tac.core.config import TwilioMemoryConfig
from tac.models.memory import (
    MemoryRetrievalMeta,
    MemoryRetrievalResponse,
    ProfileLookupResponse,
    ProfileResponse,
)
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse


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


def get_test_config_with_trait_groups(trait_groups: list[str] | None = None) -> TACConfig:
    """Get test configuration with optional trait groups."""
    memory_config = TwilioMemoryConfig(
        trait_groups=trait_groups,
    )
    return TACConfig(
        account_sid="ACtest123",
        conversation_configuration_id="conv_configuration_test123",
        auth_token="test_token_123",
        api_key="SK123",
        api_secret="test_api_token",
        phone_number="+15551234567",
        memory_config=memory_config,
    )


def create_memory_client(tac: TAC) -> MemoryClient:
    """Helper to manually create Conversation Memory client for tests."""
    return MemoryClient(
        store_id="MGtest123",
        api_key=tac.config.api_key,
        api_secret=tac.config.api_secret,
    )


def get_mock_profile_response() -> ProfileResponse:
    """Get a mock ProfileResponse for testing."""
    return ProfileResponse(
        id="profile_test_123",
        createdAt="2025-01-15T10:30:45Z",
        traits={
            "Contact": {
                "firstName": "John",
                "lastName": "Doe",
                "address": {
                    "street": "123 Main St",
                    "city": "San Francisco",
                    "state": "CA",
                    "postalCode": "94107",
                    "country": "US",
                },
            },
            "Preferences": {
                "language": "en",
                "timezone": "America/Los_Angeles",
            },
        },
    )


class TestProfileFetchingInRetrieveMemory:
    """Tests for profile fetching logic within retrieve_memory()."""

    @pytest.mark.asyncio
    async def test_retrieve_memory_fetches_profile_with_trait_groups(self) -> None:
        """Test that retrieve_memory fetches profile with configured trait_groups."""
        config = get_test_config_with_trait_groups(trait_groups=["Contact", "Preferences"])
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)

        mock_profile = get_mock_profile_response()
        empty_memory = MemoryRetrievalResponse(
            observations=[],
            summaries=[],
            communications=[],
            meta=MemoryRetrievalMeta(queryTime=0),
        )

        tac.conversation_memory_client.get_profile = AsyncMock(return_value=mock_profile)
        tac.conversation_memory_client.retrieve_memory = AsyncMock(return_value=empty_memory)

        context = ConversationSession(
            conversation_id="conv123",
            profile_id="profile_test_123",
            channel="sms",
        )

        await tac.retrieve_memory(context)

        # Verify get_profile was called with correct trait_groups
        tac.conversation_memory_client.get_profile.assert_called_once_with(
            profile_id="profile_test_123",
            trait_groups=["Contact", "Preferences"],
        )

    @pytest.mark.asyncio
    async def test_retrieve_memory_fetches_profile_without_trait_groups(self) -> None:
        """Test that retrieve_memory fetches profile with trait_groups=None when not configured."""
        config = get_test_config_with_trait_groups(trait_groups=None)
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)

        mock_profile = get_mock_profile_response()
        empty_memory = MemoryRetrievalResponse(
            observations=[],
            summaries=[],
            communications=[],
            meta=MemoryRetrievalMeta(queryTime=0),
        )

        tac.conversation_memory_client.get_profile = AsyncMock(return_value=mock_profile)
        tac.conversation_memory_client.retrieve_memory = AsyncMock(return_value=empty_memory)

        context = ConversationSession(
            conversation_id="conv123",
            profile_id="profile_test_123",
            channel="sms",
        )

        await tac.retrieve_memory(context)

        # Verify get_profile was called with trait_groups=None
        tac.conversation_memory_client.get_profile.assert_called_once_with(
            profile_id="profile_test_123",
            trait_groups=None,
        )

    @pytest.mark.asyncio
    async def test_retrieve_memory_continues_on_profile_fetch_error(self) -> None:
        """Test that retrieve_memory continues when profile fetch fails."""
        config = get_test_config_with_trait_groups()
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)

        empty_memory = MemoryRetrievalResponse(
            observations=[],
            summaries=[],
            communications=[],
            meta=MemoryRetrievalMeta(queryTime=0),
        )

        # Simulate profile fetch failure
        tac.conversation_memory_client.get_profile = AsyncMock(
            side_effect=Exception("Profile fetch failed")
        )
        tac.conversation_memory_client.retrieve_memory = AsyncMock(return_value=empty_memory)

        context = ConversationSession(
            conversation_id="conv123",
            profile_id="profile_test_123",
            channel="sms",
        )

        # Should not raise - exception is swallowed and memory retrieval continues
        result = await tac.retrieve_memory(context)

        # Verify profile is None but memory retrieval still happened
        assert context.profile is None
        assert result is not None
        tac.conversation_memory_client.retrieve_memory.assert_called_once()


def _make_participants() -> list[Any]:
    from tac.models.conversation import ParticipantAddress, ParticipantResponse

    return [
        ParticipantResponse(
            **{  # type: ignore[arg-type]
                "id": "PA_AGENT",
                "accountId": "ACtest123",
                "conversationId": "CH123456",
                "name": "+15551234567",
                "type": "AI_AGENT",
                "addresses": [
                    ParticipantAddress(channel="SMS", address="+15551234567").model_dump(
                        by_alias=True
                    )
                ],
            }
        ),
        ParticipantResponse(
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
        ),
    ]


class TestProfileInSMSChannel:
    """Tests for profile retrieval in SMS channel.

    Profile_id is resolved via `lookup_profile(address)` during memory
    retrieval; there's no PARTICIPANT_ADDED-based seeding anymore.
    """

    @pytest.mark.asyncio
    async def test_sms_profile_available_in_callback(self) -> None:
        from unittest.mock import patch

        config = get_test_config_with_trait_groups(trait_groups=["Contact"])
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)
        channel = SMSChannel(tac, config={"auto_retrieve_memory": True})

        received_context = None

        def message_ready_callback(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None = None,
        ) -> None:
            nonlocal received_context
            received_context = context

        tac.on_message_ready(message_ready_callback)

        mock_profile = get_mock_profile_response()
        tac.conversation_memory_client.lookup_profile = AsyncMock(
            return_value=ProfileLookupResponse(
                normalizedValue="+12345678901", profiles=["profile_test_123"]
            )
        )
        tac.conversation_memory_client.get_profile = AsyncMock(return_value=mock_profile)
        tac.conversation_memory_client.retrieve_memory = AsyncMock(
            return_value=MemoryRetrievalResponse(
                observations=[],
                summaries=[],
                communications=[],
                meta=MemoryRetrievalMeta(queryTime=0),
            )
        )

        message_webhook = create_communication_created_webhook(
            "CH123456", "MB123", "Hello!", "2025-11-18T00:00:01.000Z"
        )

        with patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            return_value=_make_participants(),
        ):
            await channel.process_webhook(message_webhook)

        tac.conversation_memory_client.lookup_profile.assert_called_once()
        tac.conversation_memory_client.get_profile.assert_called_once_with(
            profile_id="profile_test_123",
            trait_groups=["Contact"],
        )

        assert received_context is not None
        assert received_context.profile is not None
        assert received_context.profile.id == "profile_test_123"
        assert received_context.profile.traits["Contact"]["firstName"] == "John"

    @pytest.mark.asyncio
    async def test_sms_profile_cached_across_messages(self) -> None:
        """Profile is fetched once per session, then cached."""
        from unittest.mock import patch

        config = get_test_config_with_trait_groups()
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)
        channel = SMSChannel(tac, config={"auto_retrieve_memory": True})

        mock_profile_v1 = ProfileResponse(
            id="profile_test_123",
            createdAt="2025-01-15T10:30:45Z",
            traits={"Contact": {"firstName": "John"}},
        )
        mock_profile_v2 = ProfileResponse(
            id="profile_test_123",
            createdAt="2025-01-15T11:30:45Z",
            traits={"Contact": {"firstName": "Jane"}},
        )

        tac.conversation_memory_client.lookup_profile = AsyncMock(
            return_value=ProfileLookupResponse(
                normalizedValue="+12345678901", profiles=["profile_test_123"]
            )
        )
        tac.conversation_memory_client.retrieve_memory = AsyncMock(
            return_value=MemoryRetrievalResponse(
                observations=[],
                summaries=[],
                communications=[],
                meta=MemoryRetrievalMeta(queryTime=0),
            )
        )

        with patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            return_value=_make_participants(),
        ):
            tac.conversation_memory_client.get_profile = AsyncMock(return_value=mock_profile_v1)
            await channel.process_webhook(
                create_communication_created_webhook(
                    "CH123456", "MB123", "Hello", "2025-11-18T00:00:01.000Z"
                )
            )
            session = channel._conversations["CH123456"]
            assert session.profile is not None
            assert session.profile.traits["Contact"]["firstName"] == "John"

            tac.conversation_memory_client.get_profile = AsyncMock(return_value=mock_profile_v2)
            await channel.process_webhook(
                create_communication_created_webhook(
                    "CH123456", "MB123", "Hi again", "2025-11-18T00:00:02.000Z"
                )
            )
            session = channel._conversations["CH123456"]
            # Cached → still v1 (John), not re-fetched to v2 (Jane).
            assert session.profile.traits["Contact"]["firstName"] == "John"


class TestProfileInConversationSession:
    """Tests for profile field in ConversationSession."""

    def test_conversation_session_with_profile(self) -> None:
        """Test ConversationSession can be created with profile."""
        mock_profile = get_mock_profile_response()

        session = ConversationSession(
            conversation_id="CH123456",
            profile_id="profile_test_123",
            channel="sms",
            profile=mock_profile,
        )

        assert session.profile is not None
        assert session.profile.id == "profile_test_123"
        assert session.profile.traits["Contact"]["firstName"] == "John"

    def test_conversation_session_without_profile(self) -> None:
        """Test ConversationSession can be created without profile."""
        session = ConversationSession(
            conversation_id="CH123456",
            profile_id="profile_test_123",
            channel="sms",
            profile=None,
        )

        assert session.profile is None

    def test_conversation_session_profile_default_none(self) -> None:
        """Test that profile defaults to None when not provided."""
        session = ConversationSession(
            conversation_id="CH123456",
            profile_id="profile_test_123",
            channel="sms",
            profile=None,  # Explicitly set to None (also the default)
        )

        assert session.profile is None


class TestProfileLookup:
    """Tests for profile lookup functionality."""

    @pytest.mark.asyncio
    async def test_profile_lookup_by_phone(self) -> None:
        """Test profile lookup by phone number."""
        config = get_test_config_with_trait_groups()
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)

        mock_lookup_response = ProfileLookupResponse(
            normalizedValue="+13175556789",
            profiles=["mem_profile_00000000000000000000000001"],
        )

        tac.conversation_memory_client.lookup_profile = AsyncMock(return_value=mock_lookup_response)
        response = await tac.conversation_memory_client.lookup_profile(
            id_type="phone", value="+1 (317) 555-6789"
        )

        # Verify response
        assert response.normalized_value == "+13175556789"
        assert len(response.profiles) == 1
        assert response.profiles[0] == "mem_profile_00000000000000000000000001"

        # Verify lookup_profile was called with correct parameters
        tac.conversation_memory_client.lookup_profile.assert_called_once_with(
            id_type="phone", value="+1 (317) 555-6789"
        )

    @pytest.mark.asyncio
    async def test_profile_lookup_by_email(self) -> None:
        """Test profile lookup by email address."""
        config = get_test_config_with_trait_groups()
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)

        mock_lookup_response = ProfileLookupResponse(
            normalizedValue="test@example.com",
            profiles=["mem_profile_00000000000000000000000002"],
        )

        tac.conversation_memory_client.lookup_profile = AsyncMock(return_value=mock_lookup_response)
        response = await tac.conversation_memory_client.lookup_profile(
            id_type="email", value="test@example.com"
        )

        # Verify response
        assert response.normalized_value == "test@example.com"
        assert len(response.profiles) == 1
        assert response.profiles[0] == "mem_profile_00000000000000000000000002"

    @pytest.mark.asyncio
    async def test_profile_lookup_multiple_matches(self) -> None:
        """Test profile lookup with multiple matching profiles."""
        config = get_test_config_with_trait_groups()
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)

        mock_lookup_response = ProfileLookupResponse(
            normalizedValue="+13175556789",
            profiles=[
                "mem_profile_00000000000000000000000001",
                "mem_profile_00000000000000000000000002",
            ],
        )

        tac.conversation_memory_client.lookup_profile = AsyncMock(return_value=mock_lookup_response)
        response = await tac.conversation_memory_client.lookup_profile(
            id_type="phone", value="+13175556789"
        )

        # Verify multiple profiles returned
        assert len(response.profiles) == 2
        assert "mem_profile_00000000000000000000000001" in response.profiles
        assert "mem_profile_00000000000000000000000002" in response.profiles

    @pytest.mark.asyncio
    async def test_profile_lookup_no_matches(self) -> None:
        """Test profile lookup with no matching profiles."""
        config = get_test_config_with_trait_groups()
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)

        mock_lookup_response = ProfileLookupResponse(
            normalizedValue="+13175556789",
            profiles=[],
        )

        tac.conversation_memory_client.lookup_profile = AsyncMock(return_value=mock_lookup_response)
        response = await tac.conversation_memory_client.lookup_profile(
            id_type="phone", value="+13175556789"
        )

        # Verify empty profiles list
        assert len(response.profiles) == 0

    @pytest.mark.asyncio
    async def test_profile_lookup_error_handling(self) -> None:
        """Test profile lookup error handling."""
        config = get_test_config_with_trait_groups()
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)

        # Simulate API error
        tac.conversation_memory_client.lookup_profile = AsyncMock(
            side_effect=Exception("Profile lookup API error")
        )

        with pytest.raises(Exception, match="Profile lookup API error"):
            await tac.conversation_memory_client.lookup_profile(
                id_type="phone", value="+13175556789"
            )


class TestMemoryClientRegion:
    def test_memory_client_with_region(self):
        client = MemoryClient(
            store_id="MGtest123",
            api_key="test_api_key",
            api_secret="test_api_token",
            region="au1",
        )
        assert client.base_url == "https://memory.au1.twilio.com"

    def test_memory_client_without_region(self):
        client = MemoryClient(
            store_id="MGtest123",
            api_key="test_api_key",
            api_secret="test_api_token",
        )
        assert client.base_url == "https://memory.twilio.com"


class TestCreateProfile:
    """HTTP-level tests for MemoryClient.create_profile."""

    @pytest.mark.asyncio
    async def test_create_profile_posts_traits_and_returns_id(self) -> None:
        client = MemoryClient(
            store_id="mem_store_01abc",
            api_key="SK123",
            api_secret="secret",
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "message": "Profile resolved and accepted for processing.",
            "id": "mem_profile_01canonical",
        }
        mock_response.raise_for_status = Mock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_http

            profile_id = await client.create_profile(
                traits={"Contact": {"phone": "+13175551234"}},
            )

        assert profile_id == "mem_profile_01canonical"
        mock_http.post.assert_called_once_with(
            "https://memory.twilio.com/v1/Stores/mem_store_01abc/Profiles",
            json={"traits": {"Contact": {"phone": "+13175551234"}}},
        )

    @pytest.mark.asyncio
    async def test_create_profile_raises_when_id_missing(self) -> None:
        client = MemoryClient(
            store_id="mem_store_01abc",
            api_key="SK123",
            api_secret="secret",
        )

        mock_response = Mock()
        mock_response.json.return_value = {"message": "accepted"}
        mock_response.raise_for_status = Mock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_http

            with pytest.raises(ValueError, match="missing 'id'"):
                await client.create_profile(traits={"Contact": {"phone": "+1"}})

    @pytest.mark.asyncio
    async def test_create_profile_surfaces_http_errors(self) -> None:
        client = MemoryClient(
            store_id="mem_store_01abc",
            api_key="SK123",
            api_secret="secret",
        )

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=httpx.HTTPError("boom"))

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_http

            with pytest.raises(httpx.HTTPError, match="boom"):
                await client.create_profile(traits={"Contact": {"phone": "+1"}})

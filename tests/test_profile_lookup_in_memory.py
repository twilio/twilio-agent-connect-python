"""Tests for automatic profile lookup in memory retrieval."""

from unittest.mock import AsyncMock

import pytest

from tac import TAC, TACConfig
from tac.context.memory import MemoryClient
from tac.core.config import TwilioMemoryConfig
from tac.models.memory import MemoryRetrievalResponse, ProfileLookupResponse
from tac.models.session import AuthorInfo, ConversationSession


def get_test_config_with_memory() -> TACConfig:
    """Get test configuration with Twilio Memory."""
    return TACConfig(
        account_sid="ACtest123",
        conversation_configuration_id="conv_configuration_test123",
        auth_token="test_token_123",
        api_key="SK123",
        api_secret="test_api_token",
        phone_number="+15551234567",
        memory_config=TwilioMemoryConfig(),
    )


def create_memory_client(tac: TAC) -> MemoryClient:
    """Helper to manually create Conversation Memory client for tests."""
    return MemoryClient(
        store_id="MGtest123",
        api_key=tac.config.api_key,
        api_secret=tac.config.api_secret,
    )


class TestProfileLookupInMemoryRetrieval:
    """Tests for automatic profile lookup when profile_id is missing."""

    @pytest.mark.asyncio
    async def test_retrieve_memory_with_profile_id(self) -> None:
        """Test retrieve_memory works normally when profile_id is provided."""
        config = get_test_config_with_memory()
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)

        # Mock memory retrieval
        mock_memory_response = MemoryRetrievalResponse(
            observations=[],
            summaries=[],
            communications=[],
        )
        tac.conversation_memory_client.retrieve_memory = AsyncMock(
            return_value=mock_memory_response
        )

        # Create conversation session WITH profile_id
        session = ConversationSession(
            conversation_id="conv_test_123",
            profile_id="mem_profile_existing",
            channel="sms",
        )

        # Retrieve memory
        result = await tac.retrieve_memory(session, query="test query")

        # Verify memory was retrieved without lookup - result is wrapped in TACMemoryResponse
        from tac.models.tac import TACMemoryResponse

        assert isinstance(result, TACMemoryResponse)
        assert result.raw_data == mock_memory_response
        tac.conversation_memory_client.retrieve_memory.assert_called_once_with(
            profile_id="mem_profile_existing",
            conversation_id="conv_test_123",
            query="test query",
            observations_limit=20,
            summaries_limit=5,
            communications_limit=0,
            relevance_threshold=0.0,
        )

    @pytest.mark.asyncio
    async def test_retrieve_memory_with_profile_lookup(self) -> None:
        """Test retrieve_memory automatically looks up profile when missing."""
        config = get_test_config_with_memory()
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)

        # Mock profile lookup
        mock_lookup_response = ProfileLookupResponse(
            normalizedValue="+13175556789",
            profiles=["mem_profile_00000000000000000000000001"],
        )
        tac.conversation_memory_client.lookup_profile = AsyncMock(return_value=mock_lookup_response)

        # Mock memory retrieval
        mock_memory_response = MemoryRetrievalResponse(
            observations=[],
            summaries=[],
            communications=[],
        )
        tac.conversation_memory_client.retrieve_memory = AsyncMock(
            return_value=mock_memory_response
        )

        # Create conversation session WITHOUT profile_id but WITH author_info
        session = ConversationSession(
            conversation_id="conv_test_123",
            profile_id=None,  # No profile_id
            channel="sms",
            author_info=AuthorInfo(
                address="+1 (317) 555-6789",
                participant_id="participant_123",
            ),
        )

        # Retrieve memory
        result = await tac.retrieve_memory(session, query="test query")

        # Verify profile was looked up
        tac.conversation_memory_client.lookup_profile.assert_called_once_with(
            id_type="phone",
            value="+1 (317) 555-6789",
        )

        # Verify profile_id was assigned
        assert session.profile_id == "mem_profile_00000000000000000000000001"

        # Verify memory was retrieved with the looked up profile_id
        tac.conversation_memory_client.retrieve_memory.assert_called_once_with(
            profile_id="mem_profile_00000000000000000000000001",
            conversation_id="conv_test_123",
            query="test query",
            observations_limit=20,
            summaries_limit=5,
            communications_limit=0,
            relevance_threshold=0.0,
        )

        # Result is wrapped in TACMemoryResponse
        from tac.models.tac import TACMemoryResponse

        assert isinstance(result, TACMemoryResponse)
        assert result.raw_data == mock_memory_response

    @pytest.mark.asyncio
    async def test_retrieve_memory_lookup_uses_first_profile(self) -> None:
        """Test that first profile is used when multiple profiles are found."""
        config = get_test_config_with_memory()
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)

        # Mock profile lookup with multiple profiles
        mock_lookup_response = ProfileLookupResponse(
            normalizedValue="+13175556789",
            profiles=[
                "mem_profile_00000000000000000000000001",
                "mem_profile_00000000000000000000000002",
                "mem_profile_00000000000000000000000003",
            ],
        )
        tac.conversation_memory_client.lookup_profile = AsyncMock(return_value=mock_lookup_response)

        # Mock memory retrieval
        mock_memory_response = MemoryRetrievalResponse()
        tac.conversation_memory_client.retrieve_memory = AsyncMock(
            return_value=mock_memory_response
        )

        # Create conversation session WITHOUT profile_id
        session = ConversationSession(
            conversation_id="conv_test_123",
            profile_id=None,
            channel="sms",
            author_info=AuthorInfo(address="+13175556789"),
        )

        # Retrieve memory
        await tac.retrieve_memory(session)

        # Verify first profile was used
        assert session.profile_id == "mem_profile_00000000000000000000000001"
        tac.conversation_memory_client.retrieve_memory.assert_called_once_with(
            profile_id="mem_profile_00000000000000000000000001",
            conversation_id="conv_test_123",
            query=None,
            observations_limit=20,
            summaries_limit=5,
            communications_limit=0,
            relevance_threshold=0.0,
        )

    @pytest.mark.asyncio
    async def test_retrieve_memory_no_author_info_error(self) -> None:
        """Test retrieve_memory falls back to Conversation Orchestrator
        without profile_id or author_info."""
        config = get_test_config_with_memory()
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)

        # Mock Conversation Orchestrator fallback (since profile_id is unavailable)
        from tac.models.conversation import (
            Communication,
            CommunicationContent,
            CommunicationParticipant,
        )

        mock_communications = [
            Communication(
                id="CM123",
                conversationId="conv_test_123",
                accountId="AC123",
                author=CommunicationParticipant(
                    address="+1234567890",
                    channel="SMS",
                    participantId="MB123",
                ),
                content=CommunicationContent(type="TEXT", text="Hello"),
                recipients=[
                    CommunicationParticipant(
                        address="+1234567890",
                        channel="SMS",
                        participantId="MB456",
                    )
                ],
                createdAt="2025-01-01T00:00:00.000Z",
                updatedAt="2025-01-01T00:00:00.000Z",
            )
        ]
        tac.conversation_orchestrator_client.list_communications = AsyncMock(
            return_value=mock_communications
        )

        # Create conversation session WITHOUT profile_id and WITHOUT author_info
        session = ConversationSession(
            conversation_id="conv_test_123",
            profile_id=None,
            channel="sms",
            author_info=None,  # No author info
        )

        # Should complete without raising exception (falls back to Conversation Orchestrator)
        result = await tac.retrieve_memory(session)

        # Should return valid response from Conversation Orchestrator fallback
        assert result is not None
        assert len(result.communications) > 0
        assert session.profile_id is None  # No profile lookup performed

    @pytest.mark.asyncio
    async def test_retrieve_memory_no_address_error(self) -> None:
        """Test retrieve_memory falls back to Conversation Orchestrator when address is missing."""
        config = get_test_config_with_memory()
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)

        # Mock Conversation Orchestrator fallback (since profile_id is unavailable)
        from tac.models.conversation import (
            Communication,
            CommunicationContent,
            CommunicationParticipant,
        )

        mock_communications = [
            Communication(
                id="CM123",
                conversationId="conv_test_123",
                accountId="AC123",
                author=CommunicationParticipant(
                    address="+1234567890",
                    channel="SMS",
                    participantId="MB123",
                ),
                content=CommunicationContent(type="TEXT", text="Hello"),
                recipients=[
                    CommunicationParticipant(
                        address="+1234567890",
                        channel="SMS",
                        participantId="MB456",
                    )
                ],
                createdAt="2025-01-01T00:00:00.000Z",
                updatedAt="2025-01-01T00:00:00.000Z",
            )
        ]
        tac.conversation_orchestrator_client.list_communications = AsyncMock(
            return_value=mock_communications
        )

        # Create conversation session WITHOUT profile_id and WITHOUT address
        session = ConversationSession(
            conversation_id="conv_test_123",
            profile_id=None,
            channel="sms",
            author_info=AuthorInfo(
                address="",  # Empty address
                participant_id="participant_123",
            ),
        )

        # Should complete without raising exception (falls back to Conversation Orchestrator)
        result = await tac.retrieve_memory(session)

        # Should return valid response from Conversation Orchestrator fallback
        assert result is not None
        assert len(result.communications) > 0
        assert session.profile_id is None  # No profile lookup performed

    @pytest.mark.asyncio
    async def test_retrieve_memory_no_profiles_found_error(self) -> None:
        """Test retrieve_memory falls back to Conversation Orchestrator
        when profile lookup returns no profiles."""
        config = get_test_config_with_memory()
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)

        # Mock profile lookup with empty profiles list
        mock_lookup_response = ProfileLookupResponse(
            normalizedValue="+13175556789",
            profiles=[],  # No profiles found
        )
        tac.conversation_memory_client.lookup_profile = AsyncMock(return_value=mock_lookup_response)

        # Mock Conversation Orchestrator fallback (since profile lookup returned empty)
        from tac.models.conversation import (
            Communication,
            CommunicationContent,
            CommunicationParticipant,
        )

        mock_communications = [
            Communication(
                id="CM123",
                conversationId="conv_test_123",
                accountId="AC123",
                author=CommunicationParticipant(
                    address="+13175556789",
                    channel="SMS",
                    participantId="MB123",
                ),
                content=CommunicationContent(type="TEXT", text="Hello"),
                recipients=[
                    CommunicationParticipant(
                        address="+13175556789",
                        channel="SMS",
                        participantId="MB456",
                    )
                ],
                createdAt="2025-01-01T00:00:00.000Z",
                updatedAt="2025-01-01T00:00:00.000Z",
            )
        ]
        tac.conversation_orchestrator_client.list_communications = AsyncMock(
            return_value=mock_communications
        )

        # Create conversation session WITHOUT profile_id
        session = ConversationSession(
            conversation_id="conv_test_123",
            profile_id=None,
            channel="sms",
            author_info=AuthorInfo(address="+13175556789"),
        )

        # Should complete without raising exception (falls back to Conversation Orchestrator)
        result = await tac.retrieve_memory(session)

        # Should return valid response from Conversation Orchestrator fallback
        assert result is not None
        assert len(result.communications) > 0
        assert session.profile_id is None  # Profile lookup found nothing

    @pytest.mark.asyncio
    async def test_retrieve_memory_lookup_api_error(self) -> None:
        """Test that retrieve_memory falls back to Conversation
        Orchestrator when profile lookup API fails."""
        config = get_test_config_with_memory()
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)

        # Mock profile lookup to raise an exception
        tac.conversation_memory_client.lookup_profile = AsyncMock(
            side_effect=Exception("Profile lookup API error")
        )

        # Mock Conversation Orchestrator fallback (since profile lookup failed)
        from tac.models.conversation import (
            Communication,
            CommunicationContent,
            CommunicationParticipant,
        )

        mock_communications = [
            Communication(
                id="CM123",
                conversationId="conv_test_123",
                accountId="AC123",
                author=CommunicationParticipant(
                    address="+13175556789",
                    channel="SMS",
                    participantId="MB123",
                ),
                content=CommunicationContent(type="TEXT", text="Hello"),
                recipients=[
                    CommunicationParticipant(
                        address="+13175556789",
                        channel="SMS",
                        participantId="MB456",
                    )
                ],
                createdAt="2025-01-01T00:00:00.000Z",
                updatedAt="2025-01-01T00:00:00.000Z",
            )
        ]
        tac.conversation_orchestrator_client.list_communications = AsyncMock(
            return_value=mock_communications
        )

        # Create conversation session WITHOUT profile_id
        session = ConversationSession(
            conversation_id="conv_test_123",
            profile_id=None,
            channel="sms",
            author_info=AuthorInfo(address="+13175556789"),
        )

        # Should complete without raising exception (falls back to Conversation Orchestrator)
        result = await tac.retrieve_memory(session)

        # Should return valid response from Conversation Orchestrator fallback
        assert result is not None
        assert len(result.communications) > 0
        assert session.profile_id is None  # Profile lookup failed

    @pytest.mark.asyncio
    async def test_retrieve_memory_lookup_modifies_session(self) -> None:
        """Test that profile lookup modifies the original session object."""
        config = get_test_config_with_memory()
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)

        # Mock profile lookup
        mock_lookup_response = ProfileLookupResponse(
            normalizedValue="+13175556789",
            profiles=["mem_profile_looked_up"],
        )
        tac.conversation_memory_client.lookup_profile = AsyncMock(return_value=mock_lookup_response)

        # Mock memory retrieval
        tac.conversation_memory_client.retrieve_memory = AsyncMock(
            return_value=MemoryRetrievalResponse()
        )

        # Create conversation session
        session = ConversationSession(
            conversation_id="conv_test_123",
            profile_id=None,
            channel="sms",
            author_info=AuthorInfo(address="+13175556789"),
        )

        # Profile_id should be None initially
        assert session.profile_id is None

        # Retrieve memory
        await tac.retrieve_memory(session)

        # Profile_id should now be set
        assert session.profile_id == "mem_profile_looked_up"

    @pytest.mark.asyncio
    async def test_retrieve_memory_with_normalized_phone_number(self) -> None:
        """Test that phone numbers are normalized during lookup."""
        config = get_test_config_with_memory()
        tac = TAC(config)
        tac.conversation_memory_client = create_memory_client(tac)

        # Mock profile lookup (simulating normalization)
        mock_lookup_response = ProfileLookupResponse(
            normalizedValue="+13175556789",  # Normalized format
            profiles=["mem_profile_normalized"],
        )
        tac.conversation_memory_client.lookup_profile = AsyncMock(return_value=mock_lookup_response)

        # Mock memory retrieval
        tac.conversation_memory_client.retrieve_memory = AsyncMock(
            return_value=MemoryRetrievalResponse()
        )

        # Create conversation session with formatted phone number
        session = ConversationSession(
            conversation_id="conv_test_123",
            profile_id=None,
            channel="sms",
            author_info=AuthorInfo(
                address="+1 (317) 555-6789"  # Formatted input
            ),
        )

        # Retrieve memory
        await tac.retrieve_memory(session)

        # Verify lookup was called with the formatted input
        tac.conversation_memory_client.lookup_profile.assert_called_once_with(
            id_type="phone",
            value="+1 (317) 555-6789",
        )

        # Verify profile_id was assigned correctly
        assert session.profile_id == "mem_profile_normalized"

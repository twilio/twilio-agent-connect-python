"""Tests for Memory API models."""

from tac.models.memory import MemoryParticipant, ProfileLookupResponse


class TestMemoryModels:
    """Test Pydantic models for Memory API."""

    def test_memory_participant_with_customer_type(self):
        """Test MemoryParticipant with CUSTOMER type."""
        participant_data = {
            "id": "comms_participant_123",
            "name": "John Doe",
            "address": "+12025551234",
            "channel": "SMS",
            "type": "CUSTOMER",
            "profileId": "mem_profile_123",
        }

        participant = MemoryParticipant(**participant_data)

        assert participant.id == "comms_participant_123"
        assert participant.name == "John Doe"
        assert participant.type == "CUSTOMER"
        assert participant.profile_id == "mem_profile_123"

    def test_memory_participant_with_agent_type(self):
        """Test MemoryParticipant with AGENT type."""
        participant_data = {
            "id": "comms_participant_agent",
            "name": "Agent Bot",
            "address": "+18887608751",
            "channel": "SMS",
            "type": "AGENT",
        }

        participant = MemoryParticipant(**participant_data)

        assert participant.id == "comms_participant_agent"
        assert participant.type == "AGENT"
        assert participant.name == "Agent Bot"

    def test_memory_participant_with_unknown_type(self):
        """Test MemoryParticipant with UNKNOWN type."""
        participant_data = {
            "id": "comms_participant_unknown",
            "name": "Unknown Entity",
            "address": "unknown@example.com",
            "channel": "EMAIL",
            "type": "UNKNOWN",
        }

        participant = MemoryParticipant(**participant_data)

        assert participant.id == "comms_participant_unknown"
        assert participant.type == "UNKNOWN"
        assert participant.name == "Unknown Entity"

    def test_memory_participant_with_ai_agent_type(self):
        """Test MemoryParticipant with AI_AGENT type."""
        participant_data = {
            "id": "comms_participant_ai",
            "name": "AI Assistant",
            "address": "ai@chat.example.com",
            "channel": "CHAT",
            "type": "AI_AGENT",
            "profileId": "mem_profile_ai_123",
        }

        participant = MemoryParticipant(**participant_data)

        assert participant.id == "comms_participant_ai"
        assert participant.type == "AI_AGENT"
        assert participant.profile_id == "mem_profile_ai_123"

    def test_memory_participant_with_human_agent_type(self):
        """Test MemoryParticipant with HUMAN_AGENT type."""
        participant_data = {
            "id": "comms_participant_human",
            "name": "Support Agent",
            "address": "+15551234567",
            "channel": "VOICE",
            "type": "HUMAN_AGENT",
        }

        participant = MemoryParticipant(**participant_data)

        assert participant.id == "comms_participant_human"
        assert participant.type == "HUMAN_AGENT"
        assert participant.name == "Support Agent"

    def test_memory_participant_all_types(self):
        """Test MemoryParticipant accepts all valid participant types."""
        valid_types = ["HUMAN_AGENT", "CUSTOMER", "AI_AGENT", "AGENT", "UNKNOWN"]

        for participant_type in valid_types:
            participant_data = {
                "id": f"comms_participant_{participant_type.lower()}",
                "name": f"{participant_type} Participant",
                "address": "+12025551234",
                "channel": "SMS",
                "type": participant_type,
            }

            participant = MemoryParticipant(**participant_data)
            assert participant.type == participant_type

    def test_memory_participant_without_type(self):
        """Test MemoryParticipant with type=None (optional field)."""
        participant_data = {
            "id": "comms_participant_123",
            "name": "John Doe",
            "address": "+12025551234",
            "channel": "SMS",
        }

        participant = MemoryParticipant(**participant_data)

        assert participant.id == "comms_participant_123"
        assert participant.type is None

    def test_memory_participant_serialization(self):
        """Test MemoryParticipant serializes correctly with AGENT type."""
        participant = MemoryParticipant(
            id="comms_participant_agent",
            name="Agent System",
            address="+18887608751",
            channel="SMS",
            type="AGENT",
            profile_id="mem_profile_agent",
        )

        payload = participant.model_dump(by_alias=True, exclude_none=True)

        assert payload["id"] == "comms_participant_agent"
        assert payload["type"] == "AGENT"
        assert payload["profileId"] == "mem_profile_agent"
        assert "profile_id" not in payload  # Should use alias


class TestProfileLookupResponse:
    """Memora's Lookup endpoint returns `profiles` as optional per its OpenAPI
    spec, so null and omitted variants must both parse to an empty list."""

    def test_accepts_null_profiles(self):
        response = ProfileLookupResponse.model_validate(
            {"normalizedValue": "+13175556789", "profiles": None}
        )
        assert response.profiles == []

    def test_accepts_missing_profiles(self):
        response = ProfileLookupResponse.model_validate({"normalizedValue": "+13175556789"})
        assert response.profiles == []

    def test_accepts_populated_profiles(self):
        response = ProfileLookupResponse.model_validate(
            {
                "normalizedValue": "+13175556789",
                "profiles": ["mem_profile_00000000000000000000000000"],
            }
        )
        assert response.profiles == ["mem_profile_00000000000000000000000000"]

"""Tests for Conversation Intelligence event processing."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tac.intelligence.operator_result_processor import (
    OperatorResultProcessor,
    _extract_profile_ids,
    _generate_content,
    _parse_summaries_content,
)
from tac.models.intelligence import (
    ClassificationResult,
    CommunicationsRange,
    ExecutionDetails,
    ExtractionEntity,
    ExtractionResult,
    IntelligenceConfiguration,
    JSONResult,
    Operator,
    OperatorProcessingResult,
    OperatorResultEvent,
    Participant,
    TextGenerationResult,
    TriggerDetails,
)

# Test fixtures
VALID_STORE_ID = "mem_store_01234567890123456789abcdef"
VALID_PROFILE_ID = "mem_profile_01234567890123456789abcdef"
VALID_CONV_ID = "conv_conversation_01234567890123456789abcdef"
VALID_CONFIG_ID = "GA00000000000000000000000000000000"
VALID_NON_SUMMARY_OPERATOR_SID = "LY00000000000000000000000000000001"
VALID_SUMMARY_OPERATOR_SID = "LY00000000000000000000000000000002"


def make_valid_event(
    friendly_name: str = f"CONVERSATION_MEMORY_{VALID_STORE_ID}",
    operator_friendly_name: str = "Observation Extractor",
    profile_id: str = VALID_PROFILE_ID,
    conversation_id: str = VALID_CONV_ID,
    memory_store_id: str = VALID_STORE_ID,
    configuration_id: str = VALID_CONFIG_ID,
    operator_id: str = VALID_NON_SUMMARY_OPERATOR_SID,
    result: Any = None,
) -> dict[str, Any]:
    """Create a valid webhook event payload for testing.

    This returns the new structure with operatorResults array.
    """
    if result is None:
        result = {"payload": '{"observations": [{"content": "Test observation"}]}'}

    return {
        "accountId": "AC00000000000000000000000000000000",
        "conversationId": conversation_id,
        "memoryStoreId": memory_store_id,
        "intelligenceConfiguration": {
            "id": configuration_id,
            "friendlyName": friendly_name,
            "version": 1,
        },
        "operatorResults": [
            {
                "id": "intelligence_operatorresult_0123456789abcdefghijklmno",
                "operator": {
                    "id": operator_id,
                    "friendlyName": operator_friendly_name,
                    "version": 1,
                },
                "outputFormat": "JSON",
                "result": result,
                "dateCreated": "2025-01-15T10:30:45Z",
                "referenceIds": [],
                "executionDetails": {
                    "trigger": {
                        "on": "conversation_closed",
                        "timestamp": "2025-01-15T10:30:45Z",
                    },
                    "participants": [
                        {
                            "id": "comms_participant_0123456789abcdefghijklmno",
                            "profileId": profile_id,
                            "type": "CUSTOMER",
                        }
                    ],
                },
            }
        ],
    }


def make_operator_result(
    operator_friendly_name: str = "Observation Extractor",
    operator_id: str = VALID_NON_SUMMARY_OPERATOR_SID,
    profile_id: str = VALID_PROFILE_ID,
    result: Any = None,
) -> dict[str, Any]:
    """Create a valid operator result payload for testing helper functions."""
    if result is None:
        result = {"payload": '{"observations": [{"content": "Test observation"}]}'}

    return {
        "id": "intelligence_operatorresult_0123456789abcdefghijklmno",
        "operator": {
            "id": operator_id,
            "friendlyName": operator_friendly_name,
            "version": 1,
        },
        "outputFormat": "JSON",
        "result": result,
        "dateCreated": "2025-01-15T10:30:45Z",
        "referenceIds": [],
        "executionDetails": {
            "trigger": {"on": "conversation_closed", "timestamp": "2025-01-15T10:30:45Z"},
            "participants": [
                {
                    "id": "comms_participant_0123456789abcdefghijklmno",
                    "profileId": profile_id,
                    "type": "CUSTOMER",
                }
            ],
        },
    }


class TestModelParsing:
    """Test Pydantic model parsing."""

    def test_operator_result_event_parsing(self):
        """Test parsing a complete OperatorResultEvent (webhook wrapper)."""
        payload = make_valid_event()
        event = OperatorResultEvent(**payload)

        assert event.account_id == "AC00000000000000000000000000000000"
        assert event.conversation_id == VALID_CONV_ID
        assert event.memory_store_id == VALID_STORE_ID
        assert len(event.operator_results) == 1

        # Check the first operator result
        op_result = event.operator_results[0]
        assert op_result.output_format == "JSON"
        assert op_result.date_created == "2025-01-15T10:30:45Z"

    def test_operator_result_parsing(self):
        """Test parsing an individual OperatorResult."""
        from tac.models.intelligence import OperatorResult

        payload = make_operator_result()
        op_result = OperatorResult(**payload)

        assert op_result.id == "intelligence_operatorresult_0123456789abcdefghijklmno"
        assert op_result.output_format == "JSON"
        assert op_result.date_created == "2025-01-15T10:30:45Z"
        assert op_result.operator is not None
        assert op_result.operator.friendly_name == "Observation Extractor"

    def test_intelligence_configuration_parsing(self):
        """Test IntelligenceConfiguration model."""
        config = IntelligenceConfiguration(
            id="GA123",
            friendly_name="CONVERSATION_MEMORY_test",
            version=1,
            rule_id="rule_123",
        )
        assert config.id == "GA123"
        assert config.friendly_name == "CONVERSATION_MEMORY_test"
        assert config.version == 1

    def test_operator_parsing(self):
        """Test Operator model."""
        operator = Operator(
            id="LY123",
            friendly_name="Summary Extractor",
            version=2,
            parameters={"key": "value"},
        )
        assert operator.id == "LY123"
        assert operator.friendly_name == "Summary Extractor"
        assert operator.version == 2
        assert operator.parameters == {"key": "value"}

    def test_participant_parsing(self):
        """Test Participant model."""
        participant = Participant(
            id="comms_participant_123",
            profile_id="mem_profile_123",
            type="CUSTOMER",
        )
        assert participant.id == "comms_participant_123"
        assert participant.profile_id == "mem_profile_123"
        assert participant.type == "CUSTOMER"

    def test_execution_details_parsing(self):
        """Test ExecutionDetails model."""
        details = ExecutionDetails(
            trigger=TriggerDetails(on="utterance", timestamp="2025-01-15T10:30:45Z"),
            communications=CommunicationsRange(first="comm_1", last="comm_2"),
            channels=["SMS", "Voice"],
            participants=[Participant(id="p1")],
            context={"key": "value"},
        )
        assert details.trigger is not None
        assert details.trigger.on == "utterance"
        assert details.channels == ["SMS", "Voice"]

    def test_result_types(self):
        """Test result type models."""
        classification = ClassificationResult(label="positive")
        assert classification.label == "positive"

        extraction = ExtractionResult(entities=[ExtractionEntity(label="PERSON", text="John")])
        assert len(extraction.entities) == 1

        text_gen = TextGenerationResult(result="Generated text", format="text")
        assert text_gen.result == "Generated text"

        json_result = JSONResult(payload='{"key": "value"}')
        assert json_result.payload == '{"key": "value"}'


class TestProfileExtraction:
    """Test profile ID extraction from OperatorResult."""

    def test_extract_profile_ids_valid(self):
        """Test extracting valid profile IDs."""
        from tac.models.intelligence import OperatorResult

        payload = make_operator_result()
        op_result = OperatorResult(**payload)
        profile_ids = _extract_profile_ids(op_result)
        assert len(profile_ids) == 1
        assert profile_ids[0] == VALID_PROFILE_ID

    def test_extract_profile_ids_multiple_customers(self):
        """Test extracting multiple CUSTOMER profile IDs."""
        from tac.models.intelligence import OperatorResult

        second_profile_id = "mem_profile_11234567890123456789abcdef"
        payload = make_operator_result()
        payload["executionDetails"]["participants"] = [
            {
                "id": "p1",
                "profileId": VALID_PROFILE_ID,
                "type": "CUSTOMER",
            },
            {
                "id": "p2",
                "profileId": second_profile_id,
                "type": "CUSTOMER",
            },
        ]
        op_result = OperatorResult(**payload)
        profile_ids = _extract_profile_ids(op_result)
        assert len(profile_ids) == 2

    def test_extract_profile_ids_filters_non_customer(self):
        """Test that non-CUSTOMER participants are filtered out."""
        from tac.models.intelligence import OperatorResult

        payload = make_operator_result()
        payload["executionDetails"]["participants"] = [
            {
                "id": "p1",
                "profileId": VALID_PROFILE_ID,
                "type": "CUSTOMER",
            },
            {
                "id": "p2",
                "profileId": "mem_profile_11234567890123456789abcdef",
                "type": "AGENT",
            },
        ]
        op_result = OperatorResult(**payload)
        profile_ids = _extract_profile_ids(op_result)
        # Only CUSTOMER participants are included
        assert len(profile_ids) == 1
        assert profile_ids[0] == VALID_PROFILE_ID

    def test_extract_profile_ids_accepts_any_format(self):
        """Test that CUSTOMER profile IDs are accepted regardless of format."""
        from tac.models.intelligence import OperatorResult

        payload = make_operator_result()
        payload["executionDetails"]["participants"] = [
            {"id": "p1", "profileId": "any_profile_id", "type": "CUSTOMER"},
            {
                "id": "p2",
                "profileId": VALID_PROFILE_ID,
                "type": "CUSTOMER",
            },
        ]
        op_result = OperatorResult(**payload)
        profile_ids = _extract_profile_ids(op_result)
        assert len(profile_ids) == 2
        assert profile_ids[0] == "any_profile_id"
        assert profile_ids[1] == VALID_PROFILE_ID

    def test_extract_profile_ids_empty_participants(self):
        """Test extraction with no participants."""
        from tac.models.intelligence import OperatorResult

        payload = make_operator_result()
        payload["executionDetails"]["participants"] = []
        op_result = OperatorResult(**payload)
        profile_ids = _extract_profile_ids(op_result)
        assert len(profile_ids) == 0


class TestContentGeneration:
    """Test content generation from operator results."""

    def test_generate_content_json(self):
        """Test JSON content generation."""
        from tac.models.intelligence import OperatorResult

        payload = make_operator_result(
            result={"payload": '{"observations": [{"content": "test"}]}'}
        )
        op_result = OperatorResult(**payload)
        content = _generate_content(op_result)
        assert content == '{"observations": [{"content": "test"}]}'

    def test_generate_content_classification(self):
        """Test classification content generation."""
        from tac.models.intelligence import OperatorResult

        payload = make_operator_result(result={"label": "positive"})
        payload["outputFormat"] = "CLASSIFICATION"
        op_result = OperatorResult(**payload)
        content = _generate_content(op_result)
        assert content == "positive"

    def test_generate_content_text(self):
        """Test text generation content."""
        from tac.models.intelligence import OperatorResult

        payload = make_operator_result(result={"result": "Generated text content"})
        payload["outputFormat"] = "TEXT"
        op_result = OperatorResult(**payload)
        content = _generate_content(op_result)
        assert content == "Generated text content"


class TestContentParsing:
    """Test summary content parsing."""

    def test_parse_summaries_array_format(self):
        """Test parsing summaries array format."""
        json_content = '{"summaries": [{"summary": "sum1"}, {"summary": "sum2"}]}'
        contents = _parse_summaries_content(json_content)
        assert len(contents) == 2
        assert contents[0] == "sum1"
        assert contents[1] == "sum2"

    def test_parse_summaries_single_format(self):
        """Test parsing single summary format."""
        json_content = '{"summary": "Single summary content"}'
        contents = _parse_summaries_content(json_content)
        assert len(contents) == 1
        assert contents[0] == "Single summary content"

    def test_parse_summaries_fallback(self):
        """Test summaries fallback to raw content."""
        json_content = "Raw summary content"
        contents = _parse_summaries_content(json_content)
        assert len(contents) == 1
        assert contents[0] == "Raw summary content"


class TestOperatorResultProcessor:
    """Test the OperatorResultProcessor class."""

    @pytest.fixture
    def mock_memory_client(self):
        """Create a mock memory client."""
        client = MagicMock()
        client.create_observation = AsyncMock(return_value={"id": "obs_123"})
        client.create_conversation_summaries = AsyncMock(return_value={"message": "Success"})
        return client

    @pytest.fixture
    def ci_config(self):
        """Create a CI config for testing."""
        from tac.core.config import ConversationIntelligenceConfig

        return ConversationIntelligenceConfig(
            configuration_id=VALID_CONFIG_ID,
            summary_operator_sid=VALID_SUMMARY_OPERATOR_SID,
        )

    @pytest.fixture
    def processor(self, mock_memory_client, ci_config):
        """Create a processor with mock client and config."""
        return OperatorResultProcessor(mock_memory_client, ci_config)

    @pytest.mark.asyncio
    async def test_process_event_skips_when_no_profile_ids(self, processor):
        """A matching operator result without profile IDs is skipped, not failed."""
        payload = make_valid_event(
            operator_friendly_name="Summary Extractor",
            operator_id=VALID_SUMMARY_OPERATOR_SID,
            result={"payload": '{"summary": "Test summary"}'},
        )
        # executionDetails is now nested inside operatorResults
        payload["operatorResults"][0]["executionDetails"]["participants"] = []
        result = await processor.process_event(payload)

        assert result.success is True
        assert result.skipped is True
        assert "No profile IDs found" in result.skip_reason

    @pytest.mark.asyncio
    async def test_process_event_skips_when_content_empty(self, processor):
        """A matching operator result with empty content is skipped, not failed."""
        payload = make_valid_event(
            operator_friendly_name="Summary Extractor",
            operator_id=VALID_SUMMARY_OPERATOR_SID,
            result={"payload": ""},
        )
        result = await processor.process_event(payload)

        assert result.success is True
        assert result.skipped is True
        assert "empty content" in result.skip_reason

    @pytest.mark.asyncio
    async def test_non_summary_operator_does_not_fail_event(self, processor, mock_memory_client):
        """A non-summary operator result skips even when it has no profile IDs."""
        payload = make_valid_event()
        payload["operatorResults"][0]["executionDetails"]["participants"] = []
        result = await processor.process_event(payload)

        assert result.success is True
        assert result.skipped is True
        assert "Operator SID mismatch" in result.skip_reason
        mock_memory_client.create_conversation_summaries.assert_not_called()

    @pytest.mark.asyncio
    async def test_observation_operator_event_no_longer_creates_observations(
        self, processor, mock_memory_client
    ):
        """Observation auto-creation was removed; matching events are skipped."""
        payload = make_valid_event(
            result={"payload": '{"observations": [{"content": "Test observation"}]}'}
        )
        result = await processor.process_event(payload)

        assert result.success is True
        assert result.skipped is True
        mock_memory_client.create_observation.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_summary_event_success(self, processor, mock_memory_client):
        """Test successful summary event processing."""
        payload = make_valid_event(
            operator_friendly_name="Summary Extractor",
            operator_id=VALID_SUMMARY_OPERATOR_SID,
            result={"payload": '{"summary": "Test summary content"}'},
        )
        result = await processor.process_event(payload)

        assert result.success is True
        assert result.event_type == "summary"
        assert result.created_count == 1
        mock_memory_client.create_conversation_summaries.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_event_skips_mismatched_configuration_id(self, processor):
        """Test that events with mismatched configuration ID are skipped."""
        payload = make_valid_event(
            configuration_id="GA_DIFFERENT_CONFIG_ID_00000000000",
        )
        result = await processor.process_event(payload)

        assert result.success is True
        assert result.skipped is True
        assert "Configuration ID mismatch" in result.skip_reason

    @pytest.mark.asyncio
    async def test_process_event_skips_mismatched_operator_sid(self, processor):
        """Test that events with mismatched operator SID are skipped."""
        payload = make_valid_event(
            operator_id="LY_DIFFERENT_OPERATOR_SID_0000000000",
        )
        result = await processor.process_event(payload)

        assert result.success is True
        assert result.skipped is True
        assert "Operator SID mismatch" in result.skip_reason

    @pytest.mark.asyncio
    async def test_process_event_skips_when_summary_operator_not_configured(
        self, mock_memory_client
    ):
        """Test skip reason when no summary operator SID is configured."""
        from tac.core.config import ConversationIntelligenceConfig

        config = ConversationIntelligenceConfig(configuration_id=VALID_CONFIG_ID)
        processor = OperatorResultProcessor(mock_memory_client, config)
        payload = make_valid_event()
        result = await processor.process_event(payload)

        assert result.success is True
        assert result.skipped is True
        assert "Summary operator SID not configured" in result.skip_reason

    @pytest.mark.asyncio
    async def test_process_event_multiple_customer_profiles(self, processor, mock_memory_client):
        """Test processing with multiple CUSTOMER profiles (summary path)."""
        second_profile_id = "mem_profile_11234567890123456789abcdef"
        payload = make_valid_event(
            operator_friendly_name="Summary Extractor",
            operator_id=VALID_SUMMARY_OPERATOR_SID,
            result={"payload": '{"summary": "Test summary"}'},
        )
        # executionDetails is now nested inside operatorResults
        payload["operatorResults"][0]["executionDetails"]["participants"] = [
            {
                "id": "p1",
                "profileId": VALID_PROFILE_ID,
                "type": "CUSTOMER",
            },
            {
                "id": "p2",
                "profileId": second_profile_id,
                "type": "CUSTOMER",
            },
        ]
        result = await processor.process_event(payload)

        assert result.success is True
        assert result.created_count == 2  # One for each CUSTOMER profile
        assert mock_memory_client.create_conversation_summaries.call_count == 2

    @pytest.mark.asyncio
    async def test_process_event_api_error_handling(self, processor, mock_memory_client):
        """Test handling of API errors (summary path)."""
        mock_memory_client.create_conversation_summaries.side_effect = Exception("API Error")
        payload = make_valid_event(
            operator_friendly_name="Summary Extractor",
            operator_id=VALID_SUMMARY_OPERATOR_SID,
            result={"payload": '{"summary": "Test summary"}'},
        )
        result = await processor.process_event(payload)

        assert result.success is False
        assert "Failed to create summaries" in result.error

    @pytest.mark.asyncio
    async def test_process_event_invalid_payload(self, processor):
        """Test handling of invalid payload."""
        payload = {"invalid": "payload"}
        result = await processor.process_event(payload)

        assert result.success is False
        assert "Failed to parse event payload" in result.error

    @pytest.mark.asyncio
    async def test_process_event_uses_memory_store_id(self, processor, mock_memory_client):
        """Test that memory_store_id is used when available."""
        payload = make_valid_event(
            operator_friendly_name="Summary Extractor",
            operator_id=VALID_SUMMARY_OPERATOR_SID,
            memory_store_id=VALID_STORE_ID,
            result={"payload": '{"summary": "Test summary"}'},
        )
        result = await processor.process_event(payload)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_process_event_extracts_store_id_from_friendly_name(
        self, processor, mock_memory_client
    ):
        """Test fallback to extracting store ID from friendly name."""
        payload = make_valid_event(
            operator_friendly_name="Summary Extractor",
            operator_id=VALID_SUMMARY_OPERATOR_SID,
            friendly_name=f"CONVERSATION_MEMORY_{VALID_STORE_ID}",
            memory_store_id=None,
            result={"payload": '{"summary": "Test summary"}'},
        )
        # Remove memory_store_id
        del payload["memoryStoreId"]

        result = await processor.process_event(payload)

        assert result.success is True


class TestOperatorProcessingResult:
    """Test OperatorProcessingResult model."""

    def test_processing_result_success(self):
        """Test successful processing result."""
        result = OperatorProcessingResult(
            success=True,
            event_type="observation",
            created_count=5,
        )
        assert result.success is True
        assert result.event_type == "observation"
        assert result.created_count == 5
        assert result.skipped is False
        assert result.error is None

    def test_processing_result_skipped(self):
        """Test skipped processing result."""
        result = OperatorProcessingResult(
            success=True,
            skipped=True,
            skip_reason="Non-conversation-memory event",
        )
        assert result.success is True
        assert result.skipped is True
        assert result.skip_reason == "Non-conversation-memory event"

    def test_processing_result_error(self):
        """Test error processing result."""
        result = OperatorProcessingResult(
            success=False,
            error="Validation failed",
        )
        assert result.success is False
        assert result.error == "Validation failed"

"""Tests for per-item resilience when parsing the Memory /Recall response.

A single malformed item (bad field value or missing required field) must not
collapse the entire response to empty; valid items should still be returned.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from tac.context.memory import MemoryClient
from tac.models.memory import MemoryRetrievalResponse


def _make_client() -> MemoryClient:
    return MemoryClient(
        store_id="mem_store_01abc",
        api_key="SK123",
        api_secret="secret",
    )


def _valid_observation(obs_id: str) -> dict:
    return {
        "content": "Customer prefers email contact.",
        "source": "conversational-intelligence",
        "id": obs_id,
        "createdAt": "2025-01-15T10:30:45Z",
        "updatedAt": "2025-01-15T10:30:45Z",
    }


def _valid_summary(summary_id: str) -> dict:
    return {
        "content": "Customer resolved a billing issue.",
        "id": summary_id,
        "createdAt": "2025-01-15T10:30:45Z",
        "updatedAt": "2025-01-15T10:30:45Z",
    }


def _valid_communication(comm_id: str) -> dict:
    return {
        "id": comm_id,
        "author": {
            "id": "conv_participant_1",
            "name": "John Doe",
            "address": "+12025551234",
            "channel": "SMS",
        },
        "content": {"text": "Hello"},
        "recipients": [],
        "createdAt": "2025-01-15T10:15:30Z",
    }


async def _call_retrieve(recall_payload: dict) -> MemoryRetrievalResponse:
    client = _make_client()

    mock_response = Mock()
    mock_response.json.return_value = recall_payload
    mock_response.raise_for_status = Mock()

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)

    mock_client = Mock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_http)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        return await client.retrieve_memory(profile_id="mem_profile_01abc")


class TestRecallParsingResilience:
    @pytest.mark.asyncio
    async def test_invalid_observation_does_not_drop_valid_items(self) -> None:
        payload = {
            "observations": [
                _valid_observation("mem_observation_1"),
                {"content": "missing required id and timestamps"},  # invalid
                _valid_observation("mem_observation_2"),
            ],
            "summaries": [_valid_summary("mem_summary_1")],
            "communications": [_valid_communication("conv_communication_1")],
            "meta": {"queryTime": 100},
        }

        result = await _call_retrieve(payload)

        # The two valid observations survive; the invalid one is dropped.
        assert [o.id for o in result.observations] == [
            "mem_observation_1",
            "mem_observation_2",
        ]
        # Other sections are untouched.
        assert [s.id for s in result.summaries] == ["mem_summary_1"]
        assert [c.id for c in result.communications] == ["conv_communication_1"]
        assert result.meta.query_time == 100

    @pytest.mark.asyncio
    async def test_invalid_summary_does_not_drop_valid_items(self) -> None:
        payload = {
            "observations": [],
            "summaries": [
                _valid_summary("mem_summary_1"),
                {"content": ""},  # invalid: content min_length=1, missing id/timestamps
                _valid_summary("mem_summary_2"),
            ],
            "communications": [],
        }

        result = await _call_retrieve(payload)

        assert [s.id for s in result.summaries] == ["mem_summary_1", "mem_summary_2"]

    @pytest.mark.asyncio
    async def test_invalid_communication_does_not_drop_valid_items(self) -> None:
        payload = {
            "observations": [],
            "summaries": [],
            "communications": [
                _valid_communication("conv_communication_1"),
                {"id": "conv_communication_bad"},  # invalid: missing author/content/recipients
                _valid_communication("conv_communication_2"),
            ],
        }

        result = await _call_retrieve(payload)

        assert [c.id for c in result.communications] == [
            "conv_communication_1",
            "conv_communication_2",
        ]

    @pytest.mark.asyncio
    async def test_invalid_meta_falls_back_to_defaults(self) -> None:
        payload = {
            "observations": [_valid_observation("mem_observation_1")],
            "summaries": [_valid_summary("mem_summary_1")],
            "communications": [_valid_communication("conv_communication_1")],
            "meta": {"queryTime": "not-a-number"},  # invalid: queryTime must be an int
        }

        result = await _call_retrieve(payload)

        # Valid items still parse.
        assert [o.id for o in result.observations] == ["mem_observation_1"]
        assert [s.id for s in result.summaries] == ["mem_summary_1"]
        assert [c.id for c in result.communications] == ["conv_communication_1"]
        # Invalid meta is dropped and falls back to defaults.
        assert result.meta.query_time is None

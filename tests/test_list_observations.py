"""Tests for MemoryClient.list_observations and TAC.list_observations."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tac import TAC
from tac.context.memory import MemoryClient
from tac.core.config import TwilioMemoryConfig
from tac.models.memory import ObservationInfo, MemoryRetrievalResponse
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse


def get_test_config():
    return {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": "conv_configuration_test123",
        "phone_number": "+15551234567",
        "memory_config": TwilioMemoryConfig(trait_groups=["Contact"]),
    }


def make_obs_dict(n: int) -> dict:
    return {
        "id": f"mem_observation_{n:026d}",
        "content": f"Observation {n}",
        "source": "conversation-intelligence",
        "createdAt": "2025-01-15T10:30:45Z",
        "updatedAt": "2025-01-15T10:30:45Z",
    }


class TestMemoryClientListObservations:
    """Unit tests for MemoryClient.list_observations."""

    @pytest.mark.asyncio
    async def test_returns_observations_from_api(self) -> None:
        client = MemoryClient(
            store_id="MGtest123",
            api_key="SK123",
            api_secret="secret",
        )

        mock_response_data = {
            "observations": [make_obs_dict(1), make_obs_dict(2)],
        }

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = mock_response_data
        mock_http_response.raise_for_status.return_value = None

        mock_http_client = AsyncMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client.get = AsyncMock(return_value=mock_http_response)

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.list_observations(profile_id="mem_profile_test")

        assert isinstance(result, MemoryRetrievalResponse)
        assert len(result.observations) == 2
        assert result.observations[0].content == "Observation 1"
        assert result.observations[1].content == "Observation 2"

    @pytest.mark.asyncio
    async def test_passes_limit_param(self) -> None:
        client = MemoryClient(
            store_id="MGtest123",
            api_key="SK123",
            api_secret="secret",
        )

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {"observations": []}
        mock_http_response.raise_for_status.return_value = None

        mock_http_client = AsyncMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client.get = AsyncMock(return_value=mock_http_response)

        with patch.object(client, "_get_client", return_value=mock_http_client):
            await client.list_observations(profile_id="mem_profile_test", limit=100)

        call_kwargs = mock_http_client.get.call_args
        assert call_kwargs.kwargs["params"]["limit"] == 100

    @pytest.mark.asyncio
    async def test_passes_created_before_param(self) -> None:
        client = MemoryClient(
            store_id="MGtest123",
            api_key="SK123",
            api_secret="secret",
        )

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {"observations": []}
        mock_http_response.raise_for_status.return_value = None

        mock_http_client = AsyncMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client.get = AsyncMock(return_value=mock_http_response)

        with patch.object(client, "_get_client", return_value=mock_http_client):
            await client.list_observations(
                profile_id="mem_profile_test",
                created_before="2025-06-01T00:00:00Z",
            )

        call_kwargs = mock_http_client.get.call_args
        assert call_kwargs.kwargs["params"]["createdBefore"] == "2025-06-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_omits_created_before_when_none(self) -> None:
        client = MemoryClient(
            store_id="MGtest123",
            api_key="SK123",
            api_secret="secret",
        )

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {"observations": []}
        mock_http_response.raise_for_status.return_value = None

        mock_http_client = AsyncMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client.get = AsyncMock(return_value=mock_http_response)

        with patch.object(client, "_get_client", return_value=mock_http_client):
            await client.list_observations(profile_id="mem_profile_test")

        call_kwargs = mock_http_client.get.call_args
        assert "createdBefore" not in call_kwargs.kwargs["params"]

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self) -> None:
        client = MemoryClient(
            store_id="MGtest123",
            api_key="SK123",
            api_secret="secret",
        )

        mock_http_client = AsyncMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "404", request=MagicMock(), response=MagicMock()
            )
        )

        with patch.object(client, "_get_client", return_value=mock_http_client):
            with pytest.raises(httpx.HTTPStatusError):
                await client.list_observations(profile_id="mem_profile_test")

    @pytest.mark.asyncio
    async def test_empty_observations_list(self) -> None:
        client = MemoryClient(
            store_id="MGtest123",
            api_key="SK123",
            api_secret="secret",
        )

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {"observations": []}
        mock_http_response.raise_for_status.return_value = None

        mock_http_client = AsyncMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client.get = AsyncMock(return_value=mock_http_response)

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.list_observations(profile_id="mem_profile_test")

        assert result.observations == []


class TestTACListObservations:
    """Unit tests for TAC.list_observations (passes created_before through to MemoryClient)."""

    @pytest.mark.asyncio
    async def test_returns_tac_memory_response(self) -> None:
        tac = TAC(get_test_config())
        tac.conversation_memory_client = MemoryClient(
            store_id="MGtest123",
            api_key="SK123",
            api_secret="secret",
        )

        mock_response = MemoryRetrievalResponse(
            observations=[ObservationInfo(**make_obs_dict(1))],
        )
        tac.conversation_memory_client.list_observations = AsyncMock(return_value=mock_response)

        context = ConversationSession(
            conversation_id="CH_test",
            channel="voice",
            profile_id="mem_profile_test",
            metadata={"call_sid": "CA_test"},
        )
        result = await tac.list_observations(context)

        assert isinstance(result, TACMemoryResponse)
        assert len(result.observations) == 1

    @pytest.mark.asyncio
    async def test_passes_created_before_through(self) -> None:
        tac = TAC(get_test_config())
        tac.conversation_memory_client = MemoryClient(
            store_id="MGtest123",
            api_key="SK123",
            api_secret="secret",
        )

        mock_response = MemoryRetrievalResponse(observations=[])
        list_obs_mock = AsyncMock(return_value=mock_response)
        tac.conversation_memory_client.list_observations = list_obs_mock

        context = ConversationSession(
            conversation_id="CH_test",
            channel="voice",
            profile_id="mem_profile_test",
            metadata={"call_sid": "CA_test"},
        )
        await tac.list_observations(context, created_before="2025-06-01T00:00:00Z")

        list_obs_mock.assert_called_once()
        call_kwargs = list_obs_mock.call_args.kwargs
        assert call_kwargs["created_before"] == "2025-06-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_falls_back_to_retrieve_memory_without_memory_client(self) -> None:
        tac = TAC(get_test_config())
        tac.conversation_memory_client = None

        fallback_response = TACMemoryResponse(MemoryRetrievalResponse(observations=[]))
        tac.retrieve_memory = AsyncMock(return_value=fallback_response)

        context = ConversationSession(
            conversation_id="CH_test",
            channel="voice",
            profile_id="mem_profile_test",
            metadata={"call_sid": "CA_test"},
        )
        result = await tac.list_observations(context)

        tac.retrieve_memory.assert_called_once()
        assert isinstance(result, TACMemoryResponse)

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self) -> None:
        tac = TAC(get_test_config())
        tac.conversation_memory_client = MemoryClient(
            store_id="MGtest123",
            api_key="SK123",
            api_secret="secret",
        )
        tac.conversation_memory_client.list_observations = AsyncMock(
            side_effect=httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())
        )

        context = ConversationSession(
            conversation_id="CH_test",
            channel="voice",
            profile_id="mem_profile_test",
            metadata={"call_sid": "CA_test"},
        )
        result = await tac.list_observations(context)

        assert isinstance(result, TACMemoryResponse)
        assert result.observations == []

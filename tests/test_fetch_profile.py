"""Tests for TAC.fetch_profile(), the on-demand profile fetch."""

from unittest.mock import AsyncMock

import pytest

from tac import TAC, TACConfig
from tac.context.memory import MemoryClient
from tac.core.config import TwilioMemoryConfig
from tac.models.memory import ProfileResponse


def make_tac(trait_groups: list[str] | None = None, with_memory: bool = True) -> TAC:
    tac = TAC(
        TACConfig(
            account_sid="ACtest123",
            conversation_configuration_id="conv_configuration_test123",
            auth_token="test_token_123",
            api_key="SK123",
            api_secret="test_api_token",
            phone_number="+15551234567",
            memory_config=TwilioMemoryConfig(trait_groups=trait_groups),
        )
    )
    if with_memory:
        tac.conversation_memory_client = MemoryClient(
            store_id="MGtest123", api_key="SK123", api_secret="test_api_token"
        )
    else:
        tac.conversation_memory_client = None
    return tac


PROFILE = ProfileResponse(
    id="mem_profile_existing",
    created_at="2024-01-01T00:00:00Z",
    traits={"Contact": {"firstName": "Ada"}},
)


class TestFetchProfile:
    @pytest.mark.asyncio
    async def test_returns_profile_with_configured_trait_groups(self) -> None:
        tac = make_tac(trait_groups=["Contact"])
        assert tac.conversation_memory_client is not None
        tac.conversation_memory_client.get_profile = AsyncMock(return_value=PROFILE)

        result = await tac.fetch_profile("mem_profile_existing")

        assert result == PROFILE
        tac.conversation_memory_client.get_profile.assert_awaited_once_with(
            profile_id="mem_profile_existing", trait_groups=["Contact"]
        )

    @pytest.mark.asyncio
    async def test_empty_profile_id_returns_none_without_a_request(self) -> None:
        tac = make_tac()
        assert tac.conversation_memory_client is not None
        tac.conversation_memory_client.get_profile = AsyncMock(return_value=PROFILE)

        assert await tac.fetch_profile("") is None
        tac.conversation_memory_client.get_profile.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_memory_client_returns_none(self) -> None:
        tac = make_tac(with_memory=False)

        assert await tac.fetch_profile("mem_profile_existing") is None

    @pytest.mark.asyncio
    async def test_api_failure_returns_none(self) -> None:
        tac = make_tac()
        assert tac.conversation_memory_client is not None
        tac.conversation_memory_client.get_profile = AsyncMock(side_effect=RuntimeError("boom"))

        assert await tac.fetch_profile("mem_profile_existing") is None

"""Tests for ConversationSession.conversation_metadata()."""

from unittest.mock import AsyncMock

import pytest

from tac.models.session import ConversationSession


def make_session() -> ConversationSession:
    return ConversationSession(conversation_id="CH1", channel="SMS")


class TestConversationMetadataAccessor:
    @pytest.mark.asyncio
    async def test_known_metadata_needs_no_lookup(self) -> None:
        session = make_session()
        loader = AsyncMock(return_value={"x": "1"})
        session._co_metadata = {"appointment_id": "apt_42"}
        session._co_metadata_loader = loader

        assert await session.conversation_metadata() == {"appointment_id": "apt_42"}
        loader.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_loads_once_and_keeps_the_result(self) -> None:
        session = make_session()
        loader = AsyncMock(return_value={"appointment_id": "apt_42"})
        session._co_metadata_loader = loader

        assert await session.conversation_metadata() == {"appointment_id": "apt_42"}
        assert await session.conversation_metadata() == {"appointment_id": "apt_42"}
        loader.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_failed_load_returns_empty_and_retries(self) -> None:
        session = make_session()
        loader = AsyncMock(side_effect=[None, {"appointment_id": "apt_42"}])
        session._co_metadata_loader = loader

        assert await session.conversation_metadata() == {}
        assert await session.conversation_metadata() == {"appointment_id": "apt_42"}
        assert loader.await_count == 2

    @pytest.mark.asyncio
    async def test_no_loader_and_nothing_known_is_empty(self) -> None:
        assert await make_session().conversation_metadata() == {}

    @pytest.mark.asyncio
    async def test_returns_a_copy(self) -> None:
        session = make_session()
        session._co_metadata = {"appointment_id": "apt_42"}

        (await session.conversation_metadata())["appointment_id"] = "changed"

        assert await session.conversation_metadata() == {"appointment_id": "apt_42"}

    def test_private_state_is_not_dumped(self) -> None:
        session = make_session()
        session._co_metadata = {"appointment_id": "apt_42"}
        assert "_co_metadata" not in session.model_dump()
        assert "co_metadata" not in session.model_dump()

    def test_metadata_dict_keeps_identity_on_assignment(self) -> None:
        """Messaging shares one dict across a conversation's turns."""
        shared: dict[str, str] = {}
        session = make_session()
        session.metadata = shared
        session.metadata["step"] = "2"
        assert shared == {"step": "2"}

"""Tests for TAC core class."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from tac import TAC, TACConfig


def get_test_config(with_memory=False):
    """Get a valid test configuration."""
    config = {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": "conv_configuration_test123",
        "phone_number": "+15551234567",
    }
    if with_memory:
        config["memory_config"] = {
            "memory_store_id": "MGtest123",
        }
    return config


class TestTAC:
    """Test TAC core class."""

    def test_init_with_config_dict(self):
        """Test TAC initialization with configuration dictionary."""
        config_dict = get_test_config()
        tac = TAC(config_dict)

        assert isinstance(tac.config, TACConfig)
        assert tac.config.auth_token == "test_token_123"

    def test_init_with_config_object(self):
        """Test TAC initialization with TACConfig object."""
        config = TACConfig(**get_test_config())
        tac = TAC(config)

        assert isinstance(tac.config, TACConfig)
        assert tac.config.auth_token == "test_token_123"

    def test_init_with_empty_config_dict_fails(self):
        """Test TAC initialization with empty configuration dictionary fails."""
        config_dict = {}
        with pytest.raises(ValueError, match="Invalid configuration"):
            TAC(config_dict)

    def test_init_with_invalid_config_type(self):
        """Test TAC initialization with invalid configuration type."""
        with pytest.raises(ValueError, match="Config must be TACConfig instance or dictionary"):
            TAC("invalid_config")

    def test_region_propagated_to_clients(self):
        config = TACConfig(**{**get_test_config(), "region": "au1"})
        tac = TAC(config)
        assert (
            tac.conversation_orchestrator_client.base_url == "https://conversations.au1.twilio.com"
        )
        assert tac.conversation_memory_client.base_url == "https://memory.au1.twilio.com"

    def test_region_propagated_to_knowledge_client(self):
        config = TACConfig(
            **{**get_test_config(), "region": "au1", "knowledge_base_id": "know_kb_test"}
        )
        tac = TAC(config)
        assert tac.knowledge_client is not None
        assert tac.knowledge_client.base_url == "https://knowledge.au1.twilio.com"

    def test_no_region_uses_default_urls(self):
        tac = TAC(get_test_config())
        assert tac.conversation_orchestrator_client.base_url == "https://conversations.twilio.com"
        assert tac.conversation_memory_client.base_url == "https://memory.twilio.com"

    @pytest.mark.asyncio
    async def test_callback_return_type_validation_int(self):
        """Test that callback returning int raises TypeError."""
        from tac.models.session import ConversationSession

        tac = TAC(get_test_config())

        # Callback that returns an int (invalid)
        def bad_callback(user_message, context, memory_response):
            return 123

        tac.on_message_ready(bad_callback)

        session = ConversationSession(
            conversation_id="CH123",
            profile_id="prof123",
            channel="SMS",
        )

        with pytest.raises(
            TypeError,
            match="on_message_ready callback must return str or None, got int",
        ):
            await tac.trigger_message_ready("test message", session, None)

    @pytest.mark.asyncio
    async def test_callback_return_type_validation_dict(self):
        """Test that callback returning dict raises TypeError."""
        from tac.models.session import ConversationSession

        tac = TAC(get_test_config())

        # Callback that returns a dict (invalid)
        async def bad_callback(user_message, context, memory_response):
            return {"message": "test"}

        tac.on_message_ready(bad_callback)

        session = ConversationSession(
            conversation_id="CH123",
            profile_id="prof123",
            channel="SMS",
        )

        with pytest.raises(
            TypeError,
            match="on_message_ready callback must return str or None, got dict",
        ):
            await tac.trigger_message_ready("test message", session, None)

    @pytest.mark.asyncio
    async def test_callback_return_type_validation_list(self):
        """Test that callback returning list raises TypeError."""
        from tac.models.session import ConversationSession

        tac = TAC(get_test_config())

        # Callback that returns a list (invalid)
        def bad_callback(user_message, context, memory_response):
            return ["response1", "response2"]

        tac.on_message_ready(bad_callback)

        session = ConversationSession(
            conversation_id="CH123",
            profile_id="prof123",
            channel="SMS",
        )

        with pytest.raises(
            TypeError,
            match="on_message_ready callback must return str or None, got list",
        ):
            await tac.trigger_message_ready("test message", session, None)

    @pytest.mark.asyncio
    async def test_callback_return_type_validation_valid_str_sync(self):
        """Test that sync callback returning str works correctly."""
        from tac.models.session import ConversationSession

        tac = TAC(get_test_config())

        # Sync callback that returns a string (valid)
        def good_callback(user_message, context, memory_response):
            return "Valid response from sync"

        tac.on_message_ready(good_callback)

        session = ConversationSession(
            conversation_id="CH123",
            profile_id="prof123",
            channel="SMS",
        )

        result = await tac.trigger_message_ready("test message", session, None)
        assert result == "Valid response from sync"

    @pytest.mark.asyncio
    async def test_callback_return_type_validation_valid_str_async(self):
        """Test that async callback returning str works correctly."""
        from tac.models.session import ConversationSession

        tac = TAC(get_test_config())

        # Async callback that returns a string (valid)
        async def good_callback(user_message, context, memory_response):
            return "Valid response from async"

        tac.on_message_ready(good_callback)

        session = ConversationSession(
            conversation_id="CH123",
            profile_id="prof123",
            channel="SMS",
        )

        result = await tac.trigger_message_ready("test message", session, None)
        assert result == "Valid response from async"

    @pytest.mark.asyncio
    async def test_callback_return_type_validation_valid_none(self):
        """Test that callback returning None works correctly."""
        from tac.models.session import ConversationSession

        tac = TAC(get_test_config())

        # Callback that returns None (valid)
        def good_callback(user_message, context, memory_response):
            # Manual send_response flow
            pass

        tac.on_message_ready(good_callback)

        session = ConversationSession(
            conversation_id="CH123",
            profile_id="prof123",
            channel="SMS",
        )

        result = await tac.trigger_message_ready("test message", session, None)
        assert result is None


class TestSharedParticipantLookup:
    """One CO webhook fans out to every channel; their participant lookups
    for the same conversation must cost one request."""

    @staticmethod
    def make_tac() -> TAC:
        return TAC(get_test_config())

    @pytest.mark.asyncio
    async def test_concurrent_lookups_share_one_request(self) -> None:
        tac = self.make_tac()
        calls = 0
        release = asyncio.Event()

        async def list_participants(conversation_id: str) -> list[str]:
            nonlocal calls
            calls += 1
            await release.wait()
            return ["participant"]

        tac.conversation_orchestrator_client.list_participants = list_participants
        first = asyncio.create_task(tac._list_participants_shared("conv_1"))
        second = asyncio.create_task(tac._list_participants_shared("conv_1"))
        await asyncio.sleep(0)
        release.set()

        assert await first == ["participant"]
        assert await second == ["participant"]
        assert calls == 1

    @pytest.mark.asyncio
    async def test_repeat_lookup_is_served_from_the_cache(self) -> None:
        tac = self.make_tac()
        lookup = AsyncMock(return_value=["participant"])
        tac.conversation_orchestrator_client.list_participants = lookup

        await tac._list_participants_shared("conv_1")
        await tac._list_participants_shared("conv_1")

        lookup.assert_awaited_once_with("conv_1")

    @pytest.mark.asyncio
    async def test_each_conversation_is_looked_up_separately(self) -> None:
        tac = self.make_tac()
        lookup = AsyncMock(return_value=[])
        tac.conversation_orchestrator_client.list_participants = lookup

        await tac._list_participants_shared("conv_1")
        await tac._list_participants_shared("conv_2")

        assert lookup.await_count == 2

    @pytest.mark.asyncio
    async def test_a_failure_is_not_cached(self) -> None:
        tac = self.make_tac()
        lookup = AsyncMock(side_effect=[RuntimeError("CO down"), ["participant"]])
        tac.conversation_orchestrator_client.list_participants = lookup

        with pytest.raises(RuntimeError, match="CO down"):
            await tac._list_participants_shared("conv_1")
        assert await tac._list_participants_shared("conv_1") == ["participant"]
        assert lookup.await_count == 2

    @pytest.mark.asyncio
    async def test_cancelling_one_waiter_does_not_cancel_the_lookup(self) -> None:
        tac = self.make_tac()
        calls = 0
        release = asyncio.Event()

        async def list_participants(conversation_id: str) -> list[str]:
            nonlocal calls
            calls += 1
            await release.wait()
            return ["participant"]

        tac.conversation_orchestrator_client.list_participants = list_participants
        first = asyncio.create_task(tac._list_participants_shared("conv_1"))
        second = asyncio.create_task(tac._list_participants_shared("conv_1"))
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        release.set()

        assert await second == ["participant"]
        assert calls == 1

    @pytest.mark.asyncio
    async def test_failure_after_all_waiters_cancelled_is_not_cached(self) -> None:
        tac = self.make_tac()
        release = asyncio.Event()

        async def failing(conversation_id: str) -> list[str]:
            await release.wait()
            raise RuntimeError("CO down")

        tac.conversation_orchestrator_client.list_participants = failing
        waiter = asyncio.create_task(tac._list_participants_shared("conv_1"))
        await asyncio.sleep(0)
        waiter.cancel()
        release.set()
        for _ in range(5):
            await asyncio.sleep(0)

        tac.conversation_orchestrator_client.list_participants = AsyncMock(return_value=["ok"])
        assert await tac._list_participants_shared("conv_1") == ["ok"]

    @pytest.mark.asyncio
    async def test_concurrent_failure_reaches_both_waiters_then_is_evicted(self) -> None:
        tac = self.make_tac()
        release = asyncio.Event()

        async def failing(conversation_id: str) -> list[str]:
            await release.wait()
            raise RuntimeError("CO down")

        tac.conversation_orchestrator_client.list_participants = failing
        first = asyncio.create_task(tac._list_participants_shared("conv_1"))
        second = asyncio.create_task(tac._list_participants_shared("conv_1"))
        await asyncio.sleep(0)
        release.set()

        with pytest.raises(RuntimeError, match="CO down"):
            await first
        with pytest.raises(RuntimeError, match="CO down"):
            await second

        lookup = AsyncMock(return_value=["ok"])
        tac.conversation_orchestrator_client.list_participants = lookup
        assert await tac._list_participants_shared("conv_1") == ["ok"]
        lookup.assert_awaited_once_with("conv_1")

    @pytest.mark.asyncio
    async def test_raises_when_orchestrator_is_not_configured(self) -> None:
        tac = self.make_tac()
        tac.conversation_orchestrator_client = None

        with pytest.raises(RuntimeError, match="not configured"):
            await tac._list_participants_shared("conv_1")

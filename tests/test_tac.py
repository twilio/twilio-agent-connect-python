"""Tests for TAC core class."""

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

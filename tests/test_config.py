"""Tests for TAC configuration models."""

import pytest
from pydantic import ValidationError

from tac import TACConfig
from tac.core.config import TwilioMemoryConfig


class TestTACConfig:
    """Test TACConfig model."""

    def test_config_with_required_fields(self):
        """Test config with all required fields."""
        config = TACConfig(
            auth_token="test_token_123",
            api_key="SK123",
            api_secret="test_api_token",
            account_sid="ACtest123",
            conversation_configuration_id="conv_configuration_test123",
            phone_number="+15551234567",
        )
        assert config.auth_token == "test_token_123"
        assert config.api_key == "SK123"
        assert config.api_secret == "test_api_token"
        assert config.log_level == "INFO"  # Default value
        assert config.memory_config is not None  # Always initialized with defaults
        assert config.memory_config.observations_limit == 20  # Default value
        assert config.memory_config.summaries_limit == 5  # Default value
        assert config.memory_config.communications_limit == 0  # Default value
        assert config.memory_config.relevance_threshold == 0.0  # Default value
        assert config.memory_config.trait_groups is None  # Default value

    def test_config_with_custom_log_level(self):
        """Test config with custom log level."""
        config = TACConfig(
            auth_token="test_token_123",
            api_key="SK123",
            api_secret="test_api_token",
            account_sid="ACtest123",
            conversation_configuration_id="conv_configuration_test123",
            phone_number="+15551234567",
            log_level="DEBUG",
        )
        assert config.auth_token == "test_token_123"
        assert config.api_key == "SK123"
        assert config.api_secret == "test_api_token"
        assert config.log_level == "DEBUG"

    def test_config_with_memory_enabled(self):
        """Test config with Twilio Memory enabled."""
        memory_config = TwilioMemoryConfig(trait_groups=["Contact", "Preferences"])
        config = TACConfig(
            auth_token="test_token_123",
            api_key="SK123",
            api_secret="test_api_token",
            account_sid="ACtest123",
            conversation_configuration_id="conv_configuration_test123",
            phone_number="+15551234567",
            memory_config=memory_config,
        )
        assert config.memory_config is not None
        assert config.memory_config.trait_groups == ["Contact", "Preferences"]

    def test_config_dict_conversion(self):
        """Test converting config to dictionary."""
        config = TACConfig(
            auth_token="test_token_123",
            api_key="SK123",
            api_secret="test_api_token",
            account_sid="ACtest123",
            conversation_configuration_id="conv_configuration_test123",
            phone_number="+15551234567",
            memory_config=TwilioMemoryConfig(trait_groups=["Contact", "Preferences"]),
        )
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert "auth_token" in config_dict
        assert config_dict["auth_token"] == "test_token_123"
        assert "api_key" in config_dict
        assert config_dict["api_key"] == "SK123"
        assert "api_secret" in config_dict
        assert config_dict["api_secret"] == "test_api_token"
        assert "log_level" in config_dict
        assert config_dict["log_level"] == "INFO"
        assert "memory_config" in config_dict
        assert config_dict["memory_config"]["trait_groups"] == ["Contact", "Preferences"]

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_data = {
            "account_sid": "ACtest123",
            "auth_token": "test_token_123",
            "api_key": "SK123",
            "api_secret": "test_api_token",
            "conversation_configuration_id": "conv_configuration_test123",
            "phone_number": "+15551234567",
            "memory_config": {
                "trait_groups": ["Contact", "Preferences"],
            },
        }
        config = TACConfig(**config_data)
        assert config.auth_token == "test_token_123"
        assert config.api_key == "SK123"
        assert config.api_secret == "test_api_token"
        assert config.memory_config is not None
        assert config.memory_config.trait_groups == ["Contact", "Preferences"]

    def test_config_json_schema(self):
        """Test that config has valid JSON schema."""
        schema = TACConfig.model_json_schema()

        assert "properties" in schema
        assert "auth_token" in schema["properties"]
        assert "phone_number" in schema["properties"]
        assert "log_level" in schema["properties"]

        # Check required fields (conversation_configuration_id is optional
        # to support ConversationRelay-only mode).
        assert "required" in schema
        required_fields = schema["required"]
        assert "api_key" in required_fields
        assert "api_secret" in required_fields
        assert "conversation_configuration_id" not in required_fields
        assert "phone_number" in required_fields
        assert "auth_token" in required_fields
        assert "account_sid" in required_fields

    def test_config_equality(self):
        """Test config equality comparison."""
        base_config = {
            "account_sid": "ACtest123",
            "auth_token": "test_token_123",
            "api_key": "SK123",
            "api_secret": "test_api_token",
            "conversation_configuration_id": "conv_configuration_test123",
            "phone_number": "+15551234567",
        }
        config1 = TACConfig(**base_config)
        config2 = TACConfig(**base_config)

        different_config = base_config.copy()
        different_config["auth_token"] = "different_token"
        config3 = TACConfig(**different_config)

        assert config1 == config2
        assert config1 != config3

    def test_missing_required_fields(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TACConfig()

        error = exc_info.value
        assert "auth_token" in str(error)
        assert "account_sid" in str(error)
        assert "api_key" in str(error)
        assert "api_secret" in str(error)
        assert "phone_number" in str(error)

    def test_minimal_relay_only_config(self):
        """Test that relay-only mode only requires conversation_configuration_id to be omitted."""
        config = TACConfig(
            account_sid="ACtest123",
            auth_token="test_token_123",
            api_key="SK123",
            api_secret="test_api_secret",
            phone_number="+15551234567",
        )
        assert config.phone_number == "+15551234567"
        assert config.conversation_configuration_id is None

    def test_config_with_region(self):
        config = TACConfig(
            account_sid="ACtest123",
            auth_token="test_token_123",
            api_key="SK123",
            api_secret="test_api_token",
            conversation_configuration_id="conv_configuration_test123",
            phone_number="+15551234567",
            region="au1",
        )
        assert config.region == "au1"

    def test_config_without_region(self):
        config = TACConfig(
            account_sid="ACtest123",
            auth_token="test_token_123",
            api_key="SK123",
            api_secret="test_api_token",
            conversation_configuration_id="conv_configuration_test123",
            phone_number="+15551234567",
        )
        assert config.region is None

    def test_config_region_strips_whitespace(self):
        config = TACConfig(
            account_sid="ACtest123",
            auth_token="test_token_123",
            api_key="SK123",
            api_secret="test_api_token",
            conversation_configuration_id="conv_configuration_test123",
            phone_number="+15551234567",
            region="  au1  ",
        )
        assert config.region == "au1"

    def test_config_region_empty_string_becomes_none(self):
        config = TACConfig(
            account_sid="ACtest123",
            auth_token="test_token_123",
            api_key="SK123",
            api_secret="test_api_token",
            conversation_configuration_id="conv_configuration_test123",
            phone_number="+15551234567",
            region="",
        )
        assert config.region is None

    def test_config_region_rejects_invalid_values(self):
        invalid_regions = [
            "has spaces",
            "has/slash",
            "has:colon",
            "UPPERCASE",
            "-leading-dash",
            "trailing-dash-",
        ]
        for region in invalid_regions:
            with pytest.raises(ValidationError, match="Invalid Twilio region format"):
                TACConfig(
                    account_sid="ACtest123",
                    auth_token="test_token_123",
                    api_key="SK123",
                    api_secret="test_api_token",
                    conversation_configuration_id="conv_configuration_test123",
                    phone_number="+15551234567",
                    region=region,
                )

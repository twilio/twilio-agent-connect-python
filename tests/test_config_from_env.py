"""Tests for TwilioMemoryConfig.from_env() and TACConfig.from_env() methods."""

import pytest
from pydantic import ValidationError

from tac.core.config import TACConfig, TwilioMemoryConfig


class TestTwilioMemoryConfigFromEnv:
    """Test suite for TwilioMemoryConfig.from_env() factory method."""

    def test_from_env_no_vars_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env() returns config with defaults when no environment variables are set."""
        for env_var in (
            "TWILIO_MEMORY_PROFILE_TRAIT_GROUPS",
            "TWILIO_MEMORY_OBSERVATIONS_LIMIT",
            "TWILIO_MEMORY_SUMMARIES_LIMIT",
            "TWILIO_MEMORY_COMMUNICATIONS_LIMIT",
            "TWILIO_MEMORY_RELEVANCE_THRESHOLD",
        ):
            monkeypatch.delenv(env_var, raising=False)

        config = TwilioMemoryConfig.from_env()

        assert config is not None
        assert config.trait_groups is None
        assert config.observations_limit == 20
        assert config.summaries_limit == 5
        assert config.communications_limit == 0
        assert config.relevance_threshold == 0.0

    def test_from_env_with_trait_groups_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env() parses TWILIO_MEMORY_PROFILE_TRAIT_GROUPS from environment."""
        monkeypatch.setenv("TWILIO_MEMORY_PROFILE_TRAIT_GROUPS", "Contact, Preferences, Custom")

        config = TwilioMemoryConfig.from_env()

        assert config is not None
        assert config.trait_groups == ["Contact", "Preferences", "Custom"]

    def test_from_env_empty_trait_groups(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env() handles empty TWILIO_MEMORY_PROFILE_TRAIT_GROUPS environment variable."""
        monkeypatch.setenv("TWILIO_MEMORY_PROFILE_TRAIT_GROUPS", "")

        config = TwilioMemoryConfig.from_env()

        assert config is not None
        assert config.trait_groups is None

    def test_from_env_trait_groups_filters_empty_strings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test from_env() filters out empty strings from trait groups."""
        monkeypatch.setenv("TWILIO_MEMORY_PROFILE_TRAIT_GROUPS", "Contact,,Preferences,")

        config = TwilioMemoryConfig.from_env()

        assert config is not None
        assert config.trait_groups == ["Contact", "Preferences"]

    def test_from_env_default_memory_limits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env() uses default values when memory limit env vars not set."""
        monkeypatch.delenv("TWILIO_MEMORY_OBSERVATIONS_LIMIT", raising=False)
        monkeypatch.delenv("TWILIO_MEMORY_SUMMARIES_LIMIT", raising=False)
        monkeypatch.delenv("TWILIO_MEMORY_COMMUNICATIONS_LIMIT", raising=False)
        monkeypatch.delenv("TWILIO_MEMORY_RELEVANCE_THRESHOLD", raising=False)

        config = TwilioMemoryConfig.from_env()

        assert config.observations_limit == 20
        assert config.summaries_limit == 5
        assert config.communications_limit == 0
        assert config.relevance_threshold == 0.0

    def test_from_env_custom_memory_limits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env() parses custom memory limit values."""
        monkeypatch.setenv("TWILIO_MEMORY_OBSERVATIONS_LIMIT", "10")
        monkeypatch.setenv("TWILIO_MEMORY_SUMMARIES_LIMIT", "3")
        monkeypatch.setenv("TWILIO_MEMORY_COMMUNICATIONS_LIMIT", "5")
        monkeypatch.setenv("TWILIO_MEMORY_RELEVANCE_THRESHOLD", "0.7")

        config = TwilioMemoryConfig.from_env()

        assert config.observations_limit == 10
        assert config.summaries_limit == 3
        assert config.communications_limit == 5
        assert config.relevance_threshold == 0.7

    def test_from_env_memory_limit_zero_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env() accepts 0 values for memory limits."""
        monkeypatch.setenv("TWILIO_MEMORY_OBSERVATIONS_LIMIT", "0")
        monkeypatch.setenv("TWILIO_MEMORY_SUMMARIES_LIMIT", "0")
        monkeypatch.setenv("TWILIO_MEMORY_COMMUNICATIONS_LIMIT", "0")

        config = TwilioMemoryConfig.from_env()

        assert config.observations_limit == 0
        assert config.summaries_limit == 0
        assert config.communications_limit == 0

    def test_from_env_invalid_observations_limit_raises_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test from_env() raises ValueError for invalid observations_limit."""
        monkeypatch.setenv("TWILIO_MEMORY_OBSERVATIONS_LIMIT", "invalid")

        with pytest.raises(ValueError, match="Invalid memory configuration"):
            TwilioMemoryConfig.from_env()

    def test_from_env_invalid_relevance_threshold_raises_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test from_env() raises ValueError for invalid relevance_threshold."""
        monkeypatch.setenv("TWILIO_MEMORY_RELEVANCE_THRESHOLD", "not_a_float")

        with pytest.raises(ValueError, match="Invalid memory configuration"):
            TwilioMemoryConfig.from_env()

    def test_from_env_out_of_range_observations_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test from_env() raises ValidationError for out-of-range observations_limit."""
        monkeypatch.setenv("TWILIO_MEMORY_OBSERVATIONS_LIMIT", "150")

        with pytest.raises(ValidationError):
            TwilioMemoryConfig.from_env()

    def test_from_env_out_of_range_relevance_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test from_env() raises ValidationError for out-of-range relevance_threshold."""
        monkeypatch.setenv("TWILIO_MEMORY_RELEVANCE_THRESHOLD", "1.5")

        with pytest.raises(ValidationError):
            TwilioMemoryConfig.from_env()

    def test_from_env_empty_string_uses_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env() treats empty string env vars as unset and uses defaults."""
        monkeypatch.setenv("TWILIO_MEMORY_OBSERVATIONS_LIMIT", "")
        monkeypatch.setenv("TWILIO_MEMORY_SUMMARIES_LIMIT", "")
        monkeypatch.setenv("TWILIO_MEMORY_COMMUNICATIONS_LIMIT", "")
        monkeypatch.setenv("TWILIO_MEMORY_RELEVANCE_THRESHOLD", "")

        config = TwilioMemoryConfig.from_env()

        assert config.observations_limit == 20
        assert config.summaries_limit == 5
        assert config.communications_limit == 0
        assert config.relevance_threshold == 0.0

    def test_from_env_whitespace_only_uses_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env() treats whitespace-only env vars as unset and uses defaults."""
        monkeypatch.setenv("TWILIO_MEMORY_OBSERVATIONS_LIMIT", "   ")
        monkeypatch.setenv("TWILIO_MEMORY_SUMMARIES_LIMIT", "  \t  ")
        monkeypatch.setenv("TWILIO_MEMORY_COMMUNICATIONS_LIMIT", "\n")
        monkeypatch.setenv("TWILIO_MEMORY_RELEVANCE_THRESHOLD", "  ")

        config = TwilioMemoryConfig.from_env()

        assert config.observations_limit == 20
        assert config.summaries_limit == 5
        assert config.communications_limit == 0
        assert config.relevance_threshold == 0.0


class TestTACConfigFromEnv:
    """Test suite for TACConfig.from_env() factory method."""

    def _set_all_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Helper to set all TACConfig environment variables."""
        monkeypatch.setenv("TWILIO_CONVERSATION_CONFIGURATION_ID", "conv_configuration_123")
        monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test_auth_token")
        monkeypatch.setenv("TWILIO_API_KEY", "SK123")
        monkeypatch.setenv("TWILIO_API_SECRET", "test_api_token")
        monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+1234567890")

    def test_from_env_with_all_required_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env() creates config when all required env vars are set."""
        self._set_all_env_vars(monkeypatch)

        config = TACConfig.from_env()

        assert config.conversation_configuration_id == "conv_configuration_123"
        assert config.account_sid == "AC123"
        assert config.auth_token == "test_auth_token"
        assert config.api_key == "SK123"
        assert config.api_secret == "test_api_token"
        assert config.phone_number == "+1234567890"
        assert config.log_level == "INFO"  # Default
        # Always returns config (memory store ID auto-fetched)
        assert config.memory_config is not None
        assert config.memory_config.trait_groups is None  # No trait groups env var set

    def test_from_env_with_memory_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env() includes memory config when memory env vars are set."""
        self._set_all_env_vars(monkeypatch)
        monkeypatch.setenv("TWILIO_MEMORY_PROFILE_TRAIT_GROUPS", "Contact, Preferences")

        config = TACConfig.from_env()

        assert config.memory_config is not None
        assert config.memory_config.trait_groups == ["Contact", "Preferences"]

    def test_from_env_with_custom_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env() respects custom TWILIO_LOG_LEVEL."""
        self._set_all_env_vars(monkeypatch)
        monkeypatch.setenv("TWILIO_LOG_LEVEL", "DEBUG")

        config = TACConfig.from_env()

        assert config.log_level == "DEBUG"

    def test_from_env_missing_conversation_configuration_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test from_env() succeeds with conversation_configuration_id=None when
        TWILIO_CONVERSATION_CONFIGURATION_ID is missing (ConversationRelay-only mode)."""
        self._set_all_env_vars(monkeypatch)
        monkeypatch.delenv("TWILIO_CONVERSATION_CONFIGURATION_ID", raising=False)

        config = TACConfig.from_env()

        assert config.conversation_configuration_id is None

    def test_from_env_missing_account_sid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env() raises KeyError when TWILIO_ACCOUNT_SID is missing."""
        self._set_all_env_vars(monkeypatch)
        monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)

        with pytest.raises(KeyError, match="TWILIO_ACCOUNT_SID"):
            TACConfig.from_env()

    def test_from_env_missing_auth_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env() raises KeyError when TWILIO_AUTH_TOKEN is missing."""
        self._set_all_env_vars(monkeypatch)
        monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)

        with pytest.raises(KeyError, match="TWILIO_AUTH_TOKEN"):
            TACConfig.from_env()

    def test_from_env_missing_phone_number(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env() raises KeyError when TWILIO_PHONE_NUMBER is missing."""
        self._set_all_env_vars(monkeypatch)
        monkeypatch.delenv("TWILIO_PHONE_NUMBER", raising=False)

        with pytest.raises(KeyError, match="TWILIO_PHONE_NUMBER"):
            TACConfig.from_env()

    def test_from_env_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env() raises KeyError when TWILIO_API_KEY is missing."""
        self._set_all_env_vars(monkeypatch)
        monkeypatch.delenv("TWILIO_API_KEY", raising=False)

        with pytest.raises(KeyError, match="TWILIO_API_KEY"):
            TACConfig.from_env()

    def test_from_env_missing_api_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env() raises KeyError when TWILIO_API_SECRET is missing."""
        self._set_all_env_vars(monkeypatch)
        monkeypatch.delenv("TWILIO_API_SECRET", raising=False)

        with pytest.raises(KeyError, match="TWILIO_API_SECRET"):
            TACConfig.from_env()

    def test_from_env_missing_multiple_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env() raises KeyError when environment variables are missing."""
        monkeypatch.delenv("TWILIO_CONVERSATION_CONFIGURATION_ID", raising=False)
        monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("TWILIO_PHONE_NUMBER", raising=False)

        with pytest.raises(KeyError):
            TACConfig.from_env()

    def test_from_env_with_region(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_all_env_vars(monkeypatch)
        monkeypatch.setenv("TWILIO_REGION", "au1")

        config = TACConfig.from_env()

        assert config.region == "au1"

    def test_from_env_without_region(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_all_env_vars(monkeypatch)
        monkeypatch.delenv("TWILIO_REGION", raising=False)

        config = TACConfig.from_env()

        assert config.region is None

    def test_from_env_with_studio_handoff_flow_sid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_all_env_vars(monkeypatch)
        monkeypatch.setenv("TWILIO_STUDIO_HANDOFF_FLOW_SID", "FW" + "a" * 32)

        config = TACConfig.from_env()

        assert config.studio_handoff_flow_sid == "FW" + "a" * 32

    def test_from_env_without_studio_handoff_flow_sid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_all_env_vars(monkeypatch)
        monkeypatch.delenv("TWILIO_STUDIO_HANDOFF_FLOW_SID", raising=False)

        config = TACConfig.from_env()

        assert config.studio_handoff_flow_sid is None

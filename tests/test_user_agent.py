"""Tests for User-Agent header in API clients."""

import platform
import re

import pytest

from tac import TAC, TACConfig, __version__
from tac.context.base import PartnerConnector
from tac.context.conversation import ConversationClient
from tac.context.knowledge import KnowledgeClient
from tac.context.memory import MemoryClient


class TestUserAgent:
    """Test User-Agent header generation."""

    def test_base_client_user_agent_format(self) -> None:
        """Test that BaseAPIClient generates correct User-Agent format."""
        client = ConversationClient(
            api_key="test_key",
            api_secret="test_token",
            configuration_id="test_config",
        )

        user_agent = client._get_user_agent()

        # Should match pattern: twilio-agent-connect-python/{version} ({os} {arch}) Python/{py_ver}
        # Use re.escape(__version__) to support PEP 440 versioning (e.g., 0.1.0a1, .dev0, +local)
        pattern = (
            rf"twilio-agent-connect-python/{re.escape(__version__)} "
            r"\(.+ .+\) Python/[\d\.]+"
        )
        assert re.match(pattern, user_agent), f"User-Agent doesn't match pattern: {user_agent}"

        # Verify it contains expected components
        assert f"twilio-agent-connect-python/{__version__}" in user_agent
        assert platform.system() in user_agent
        assert platform.machine() in user_agent
        assert f"Python/{platform.python_version()}" in user_agent

    def test_user_agent_matches_twilio_pattern(self) -> None:
        """Test that User-Agent follows Twilio SDK pattern."""
        client = ConversationClient(
            api_key="test_key",
            api_secret="test_token",
            configuration_id="test_config",
        )

        user_agent = client._get_user_agent()
        expected_format = (
            f"twilio-agent-connect-python/{__version__} "
            f"({platform.system()} {platform.machine()}) "
            f"Python/{platform.python_version()}"
        )

        assert user_agent == expected_format

    @pytest.mark.asyncio
    async def test_conversation_client_sends_user_agent(self) -> None:
        """Test that ConversationClient includes User-Agent in requests."""
        client = ConversationClient(
            api_key="test_key",
            api_secret="test_token",
            configuration_id="test_configuration",
        )

        async with client._get_client() as http_client:
            headers = http_client.headers
            assert "User-Agent" in headers
            assert "twilio-agent-connect-python" in headers["User-Agent"]

    @pytest.mark.asyncio
    async def test_memory_client_sends_user_agent(self) -> None:
        """Test that MemoryClient includes User-Agent in requests."""
        client = MemoryClient(
            store_id="test_store",
            api_key="test_key",
            api_secret="test_token",
        )

        async with client._get_client() as http_client:
            headers = http_client.headers
            assert "User-Agent" in headers
            assert "twilio-agent-connect-python" in headers["User-Agent"]

    @pytest.mark.asyncio
    async def test_knowledge_client_sends_user_agent(self) -> None:
        """Test that KnowledgeClient includes User-Agent in requests."""
        client = KnowledgeClient(
            api_key="test_key",
            api_secret="test_token",
        )

        async with client._get_client() as http_client:
            headers = http_client.headers
            assert "User-Agent" in headers
            assert "twilio-agent-connect-python" in headers["User-Agent"]

    def test_sync_client_sends_user_agent(self) -> None:
        """Test that synchronous client includes User-Agent in requests."""
        client = ConversationClient(
            api_key="test_key",
            api_secret="test_token",
            configuration_id="test_config",
        )

        with client._get_sync_client() as http_client:
            headers = http_client.headers
            assert "User-Agent" in headers
            assert "twilio-agent-connect-python" in headers["User-Agent"]


class TestPartnerConnectorUserAgent:
    """Tests for partner connector User-Agent suffix."""

    def test_user_agent_has_no_partner_suffix_by_default(self) -> None:
        client = ConversationClient(
            api_key="test_key", api_secret="test_token", configuration_id="test_config"
        )
        assert "tac-aws" not in client._get_user_agent()
        assert "tac-azure" not in client._get_user_agent()

    def test_set_partner_connector_appends_to_user_agent(self) -> None:
        client = ConversationClient(
            api_key="test_key", api_secret="test_token", configuration_id="test_config"
        )
        client._set_partner_connector(PartnerConnector.AZURE_AGENT_FRAMEWORK, "0.1.0")

        expected_suffix = "tac-azure/0.1.0 (AgentFrameworkConnector)"
        assert client._get_user_agent().endswith(expected_suffix)

    def test_set_partner_connector_rejects_non_enum(self) -> None:
        client = ConversationClient(
            api_key="test_key", api_secret="test_token", configuration_id="test_config"
        )
        with pytest.raises(TypeError, match="PartnerConnector"):
            client._set_partner_connector("aws-bedrock", "1.0.0")  # type: ignore[arg-type]

    def test_partner_connector_values(self) -> None:
        """Lock in the enum values so partner packages can rely on them."""
        assert PartnerConnector.AWS_STRANDS.package_name == "tac-aws"
        assert PartnerConnector.AWS_STRANDS.connector_name == "StrandsConnector"
        assert PartnerConnector.AWS_BEDROCK.package_name == "tac-aws"
        assert PartnerConnector.AWS_BEDROCK.connector_name == "BedrockConnector"
        assert PartnerConnector.AWS_AGENTCORE.package_name == "tac-aws"
        assert PartnerConnector.AWS_AGENTCORE.connector_name == "BedrockAgentCoreConnector"
        assert PartnerConnector.AZURE_AGENT_FRAMEWORK.package_name == "tac-azure"
        assert PartnerConnector.AZURE_AGENT_FRAMEWORK.connector_name == "AgentFrameworkConnector"
        assert PartnerConnector.AZURE_VOICE_LIVE.package_name == "tac-azure"
        assert PartnerConnector.AZURE_VOICE_LIVE.connector_name == "VoiceLiveConnector"

    def test_full_user_agent_format_with_partner(self) -> None:
        client = ConversationClient(
            api_key="test_key", api_secret="test_token", configuration_id="test_config"
        )
        client._set_partner_connector(PartnerConnector.AWS_BEDROCK, "0.2.3")

        user_agent = client._get_user_agent()
        pattern = (
            rf"^twilio-agent-connect-python/{re.escape(__version__)} "
            r"\(.+ .+\) Python/[\d\.]+ "
            r"tac-aws/0\.2\.3 \(BedrockConnector\)$"
        )
        assert re.match(pattern, user_agent), f"User-Agent doesn't match: {user_agent}"


class TestTACRegisterPartnerConnector:
    """Tests for the ``TAC.register_partner_connector`` method."""

    def _make_tac(self, with_knowledge: bool = False) -> TAC:
        config = {
            "account_sid": "ACtest123",
            "auth_token": "test_token_123",
            "api_key": "SK123",
            "api_secret": "test_api_token",
            "conversation_configuration_id": "conv_configuration_test123",
            "phone_number": "+15551234567",
        }
        if with_knowledge:
            config["knowledge_base_id"] = "know_kb_test"
        return TAC(TACConfig(**config))

    def test_tags_every_api_client_on_tac(self) -> None:
        tac = self._make_tac(with_knowledge=True)
        tac.register_partner_connector(
            PartnerConnector.AZURE_AGENT_FRAMEWORK, package_version="0.1.0"
        )

        clients = [
            tac.conversation_orchestrator_client,
            tac.conversation_memory_client,
            tac.knowledge_client,
        ]
        for client in clients:
            assert client is not None
            assert client._partner_connector is PartnerConnector.AZURE_AGENT_FRAMEWORK
            assert client._partner_package_version == "0.1.0"
            assert "tac-azure/0.1.0 (AgentFrameworkConnector)" in client._get_user_agent()

    def test_rejects_non_enum_connector(self) -> None:
        tac = self._make_tac()
        with pytest.raises(TypeError, match="PartnerConnector"):
            tac.register_partner_connector("aws-bedrock", package_version="1.0.0")  # type: ignore[arg-type]

    def test_rejects_empty_version(self) -> None:
        tac = self._make_tac()
        with pytest.raises(ValueError, match="package_version"):
            tac.register_partner_connector(PartnerConnector.AWS_BEDROCK, package_version="")

    def test_works_without_knowledge_client(self) -> None:
        """TAC without knowledge_base_id has ``knowledge_client = None`` — must not break."""
        tac = self._make_tac(with_knowledge=False)
        tac.register_partner_connector(PartnerConnector.AWS_STRANDS, package_version="0.2.0")

        assert (
            tac.conversation_orchestrator_client._partner_connector is PartnerConnector.AWS_STRANDS
        )
        assert tac.knowledge_client is None

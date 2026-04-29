import platform
from enum import Enum

import httpx

from tac import __version__
from tac.core.logging import get_logger


class PartnerConnector(Enum):
    """Closed set of partner connectors allowed to identify themselves in the User-Agent.

    Partner packages built on top of TAC (e.g. ``tac_aws``, ``tac_azure``) select
    a value from this enum and pass it to :func:`register_partner_connector`. The
    enum is intentionally closed so that customers cannot set arbitrary
    User-Agent values. Adding a new partner connector requires a release of
    core TAC.

    Each value is a ``(package_name, connector_name)`` tuple. The package name
    becomes a User-Agent product token and the connector name becomes a
    comment, producing e.g. ``tac-azure/0.1.0 (AgentFrameworkConnector)``.
    """

    AWS_STRANDS = ("tac-aws", "StrandsConnector")
    AWS_BEDROCK = ("tac-aws", "BedrockConnector")
    AWS_AGENTCORE = ("tac-aws", "BedrockAgentCoreConnector")
    AZURE_AGENT_FRAMEWORK = ("tac-azure", "AgentFrameworkConnector")
    AZURE_VOICE_LIVE = ("tac-azure", "VoiceLiveConnector")

    @property
    def package_name(self) -> str:
        return self.value[0]

    @property
    def connector_name(self) -> str:
        return self.value[1]


class BaseAPIClient:
    """Base client for Twilio API interactions with shared HTTP client logic."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        region: str | None = None,
    ) -> None:
        """
        Initialize the base API client.

        Args:
            api_key: Twilio API Key SID for authentication
            api_secret: Twilio API Key Secret for authentication
            region: Optional Twilio region (e.g., 'au1', 'ie1')
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.region = region
        self._partner_connector: PartnerConnector | None = None
        self._partner_package_version: str | None = None
        self.logger = get_logger(self.__class__.__name__)

    @staticmethod
    def _build_base_url(product: str, region: str | None) -> str:
        if region:
            return f"https://{product}.{region}.twilio.com"
        return f"https://{product}.twilio.com"

    def _set_partner_connector(self, connector: PartnerConnector, package_version: str) -> None:
        """Tag this client with a partner connector identifier.

        Not part of the public API — partner packages should call
        :func:`register_partner_connector` on the :class:`~tac.TAC` instance.
        """
        if not isinstance(connector, PartnerConnector):
            raise TypeError(
                f"connector must be a PartnerConnector enum value, got {type(connector).__name__}"
            )
        self._partner_connector = connector
        self._partner_package_version = package_version

    def _get_user_agent(self) -> str:
        """Generate User-Agent header following Twilio SDK conventions."""
        os_name = platform.system()
        os_arch = platform.machine()
        python_version = platform.python_version()
        user_agent = (
            f"twilio-agent-connect-python/{__version__} "
            f"({os_name} {os_arch}) Python/{python_version}"
        )
        if self._partner_connector is not None and self._partner_package_version is not None:
            user_agent += (
                f" {self._partner_connector.package_name}/{self._partner_package_version} "
                f"({self._partner_connector.connector_name})"
            )
        return user_agent

    def _get_client(self) -> httpx.AsyncClient:
        """Create a new httpx.AsyncClient for each request to avoid event loop issues."""
        return httpx.AsyncClient(
            auth=(self.api_key, self.api_secret),
            headers={"User-Agent": self._get_user_agent()},
            timeout=30.0,
            follow_redirects=True,
        )

    def _get_sync_client(self) -> httpx.Client:
        """Create a new synchronous httpx.Client."""
        return httpx.Client(
            auth=(self.api_key, self.api_secret),
            headers={"User-Agent": self._get_user_agent()},
            timeout=30.0,
            follow_redirects=True,
        )

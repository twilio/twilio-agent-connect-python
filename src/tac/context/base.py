import platform
import time
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

        # Initialize metrics client
        try:
            from tac.telemetry.metrics import MetricsClient

            self._metrics = MetricsClient()
        except ImportError:
            self._metrics = None

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

    def _record_api_request(
        self,
        method: str,
        start_time: float,
        status_code: int | None = None,
        error: Exception | None = None,
    ) -> None:
        """Record API request metrics.

        Args:
            method: HTTP method (GET, POST, etc.)
            start_time: Request start timestamp
            status_code: HTTP response status code (if available)
            error: Exception if request failed
        """
        if not self._metrics:
            return

        client_type = self.__class__.__name__.replace("Client", "").lower()
        duration = time.time() - start_time

        # Base attributes for all metrics
        base_attrs = {"client_type": client_type, "method": method}

        # 📊 Metric 7: API request count
        self._metrics.api_request_count.add(1, attributes=base_attrs)

        # 📊 Metric 8: API request duration
        self._metrics.api_request_duration.record(duration, attributes=base_attrs)

        # 📊 Metric 9: API error count
        if error:
            # Categorize error types to limit cardinality
            error_category = self._categorize_error(error)
            error_attrs = {**base_attrs, "error_type": error_category}

            # Add status code if available (from HTTPError)
            if status_code:
                error_attrs["status_code"] = str(status_code)

            self._metrics.api_error_count.add(1, attributes=error_attrs)

    @staticmethod
    def _categorize_error(error: Exception) -> str:
        """Categorize errors to limit metric cardinality.

        Args:
            error: Exception to categorize

        Returns:
            Error category string
        """
        # HTTPError with status codes
        if hasattr(error, "response") and hasattr(error.response, "status_code"):
            status_code = error.response.status_code
            if 400 <= status_code < 500:
                return "http_4xx"
            elif 500 <= status_code < 600:
                return "http_5xx"

        # Network errors
        error_name = type(error).__name__
        if "Timeout" in error_name or "timeout" in str(error).lower():
            return "timeout"
        if "Connect" in error_name or "connection" in str(error).lower():
            return "connection"

        # Validation/parsing errors
        if "Validation" in error_name or "Parse" in error_name:
            return "validation"

        # Generic fallback
        return "other"

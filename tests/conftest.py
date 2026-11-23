"""Shared pytest fixtures for TAC tests."""

import os
from unittest.mock import patch

import httpx
import pytest
import requests
import requests.adapters

from tac.models.conversation import ConversationConfiguration

# Keep telemetry off for the whole suite so tests never construct a real
# client or send fixture events. Set at import, before any test runs, because
# a channel emits on the first conversation it starts. Tests that exercise the
# analytics module clear this and mock the client themselves.
os.environ["TAC_ANALYTICS_DISABLED"] = "true"


@pytest.fixture(autouse=True)
def mock_conversation_configuration(request):
    """
    Mock ConversationClient.get_configuration() for all tests by default.

    TAC initialization requires fetching the Conversation Orchestrator configuration to get
    the memory_store_id. This fixture mocks the API call to prevent actual
    network requests during tests.

    To disable this fixture for specific tests (e.g., to test get_configuration itself),
    use the pytest marker:
        @pytest.mark.no_auto_mock
        def test_something(self):
            ...
    """
    # Skip auto-mocking if test is marked with no_auto_mock
    if "no_auto_mock" in request.keywords:
        yield None
        return

    mock_config = ConversationConfiguration(
        id="conv_configuration_test123",
        memory_store_id="MGtest123",
        display_name="Test Configuration",
        description="Test configuration for unit tests",
        conversation_grouping_type="GROUP_BY_PARTICIPANT_ADDRESSES_AND_CHANNEL_TYPE",
    )
    with patch(
        "tac.context.conversation.ConversationClient.get_configuration", return_value=mock_config
    ):
        yield mock_config


@pytest.fixture(autouse=True)
def block_unmocked_http():
    """Fail any test that sends a real HTTP request.

    Raising alone isn't enough: TAC catches many request failures and falls
    back (e.g. the profile lookup), so a leaked request would still pass.
    Each request is recorded and the test fails at teardown. Only the real
    network transports are patched (httpx for TAC's clients, requests for
    twilio-python), so ``httpx.MockTransport``, FastAPI's ``TestClient`` and
    mocked client methods are unaffected.
    """
    unmocked: list[str] = []

    def _record(method: str, url: object) -> str:
        entry = f"{method} {url}"
        unmocked.append(entry)
        return f"Unmocked HTTP request in test: {entry}"

    def _block_httpx(_transport, request: httpx.Request):
        raise httpx.ConnectError(_record(request.method, request.url), request=request)

    async def _block_httpx_async(_transport, request: httpx.Request):
        raise httpx.ConnectError(_record(request.method, request.url), request=request)

    def _block_requests(_adapter, request, *args, **kwargs):
        raise requests.exceptions.ConnectionError(_record(request.method, request.url))

    with (
        patch.object(httpx.HTTPTransport, "handle_request", _block_httpx),
        patch.object(httpx.AsyncHTTPTransport, "handle_async_request", _block_httpx_async),
        patch.object(requests.adapters.HTTPAdapter, "send", _block_requests),
    ):
        yield

    if unmocked:
        pytest.fail(
            f"Test made {len(unmocked)} unmocked HTTP request(s); mock them:\n  "
            + "\n  ".join(unmocked),
            pytrace=False,
        )

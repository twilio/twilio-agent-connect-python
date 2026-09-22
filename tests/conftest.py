"""Shared pytest fixtures for TAC tests."""

import os
from unittest.mock import patch

import pytest

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

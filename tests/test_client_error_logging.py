"""Failed API requests are logged by method, URL and status only, never with customer data."""

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from tac.context.base import BaseAPIClient
from tac.context.conversation import ConversationClient
from tac.context.knowledge import KnowledgeClient
from tac.context.memory import MemoryClient
from tac.models.conversation import (
    ActionParticipantRef,
    ActionTextContent,
    CommunicationContent,
    CommunicationParticipant,
    CommunicationRequest,
    ParticipantAddress,
    ParticipantRequest,
    SendMessageActionPayload,
    SendMessageActionRequest,
)

PHONE = "+15555550100"
EMAIL = "user@example.com"
SECRET = "private-customer-detail"
PII_MARKERS = (PHONE, EMAIL, SECRET)


def _echo_400(request: httpx.Request) -> httpx.Response:
    """Reject the request with a body that echoes the request, as a validation error might."""
    return httpx.Response(
        400, text=f"bad request: {request.content.decode()} query={request.url.query.decode()}"
    )


@pytest.fixture
def mock_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = httpx.MockTransport(_echo_400)
    original_async_client = httpx.AsyncClient
    original_client = httpx.Client

    def async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_async_client(*args, **{**kwargs, "transport": transport})

    def sync_client(*args: Any, **kwargs: Any) -> httpx.Client:
        return original_client(*args, **{**kwargs, "transport": transport})

    monkeypatch.setattr(httpx, "AsyncClient", async_client)
    monkeypatch.setattr(httpx, "Client", sync_client)


def _memory() -> MemoryClient:
    return MemoryClient(store_id="mem_store_test", api_key="key", api_secret="secret")


def _conversation() -> ConversationClient:
    return ConversationClient(api_key="key", api_secret="secret", configuration_id="conf_test")


def _knowledge() -> KnowledgeClient:
    return KnowledgeClient(api_key="key", api_secret="secret")


def _address() -> ParticipantAddress:
    return ParticipantAddress(channel="SMS", address=PHONE)


def _comm_participant() -> CommunicationParticipant:
    return CommunicationParticipant(address=PHONE, channel="SMS", participant_id="p_1")


def _action_ref() -> ActionParticipantRef:
    return ActionParticipantRef(address=PHONE, channel="SMS", participant_id="p_1")


CASES: list[tuple[str, Callable[[], BaseAPIClient], Callable[[Any], Any], str]] = [
    (
        "retrieve_memory",
        _memory,
        lambda c: c.retrieve_memory(profile_id="mem_profile_1", query=f"{SECRET} {EMAIL}"),
        "POST",
    ),
    (
        "get_profile",
        _memory,
        lambda c: c.get_profile(profile_id="mem_profile_1", trait_groups=[SECRET]),
        "GET",
    ),
    ("lookup_profile", _memory, lambda c: c.lookup_profile(id_type="phone", value=PHONE), "POST"),
    (
        "create_profile",
        _memory,
        lambda c: c.create_profile(traits={"Contact": {"phone": PHONE, "email": EMAIL}}),
        "POST",
    ),
    (
        "create_observation",
        _memory,
        lambda c: c.create_observation(profile_id="mem_profile_1", content=SECRET),
        "POST",
    ),
    (
        "create_conversation_summaries",
        _memory,
        lambda c: c.create_conversation_summaries(
            profile_id="mem_profile_1", summaries=[{"content": SECRET}]
        ),
        "POST",
    ),
    (
        "list_conversations",
        _conversation,
        lambda c: c.list_conversations(channel_id=SECRET),
        "GET",
    ),
    (
        "add_participant",
        _conversation,
        lambda c: c.add_participant(conversation_id="conv_1", addresses=[_address()]),
        "POST",
    ),
    (
        "update_participant",
        _conversation,
        lambda c: c.update_participant(
            conversation_id="conv_1",
            participant_id="p_1",
            participant_type="CUSTOMER",
            addresses=[_address()],
            name=SECRET,
        ),
        "PUT",
    ),
    (
        "create_conversation",
        _conversation,
        lambda c: c.create_conversation(
            name=SECRET, participants=[ParticipantRequest(name=SECRET, addresses=[_address()])]
        ),
        "POST",
    ),
    (
        "update_conversation",
        _conversation,
        lambda c: c.update_conversation(conversation_id="conv_1", status="ACTIVE", name=SECRET),
        "PUT",
    ),
    (
        "patch_conversation_metadata",
        _conversation,
        lambda c: c.patch_conversation_metadata(conversation_id="conv_1", metadata={"k": SECRET}),
        "PATCH",
    ),
    (
        "create_communication",
        _conversation,
        lambda c: c.create_communication(
            conversation_id="conv_1",
            communication_request=CommunicationRequest(
                author=_comm_participant(),
                content=CommunicationContent(type="TEXT", text=SECRET),
                recipients=[_comm_participant()],
            ),
        ),
        "POST",
    ),
    (
        "list_communications",
        _conversation,
        lambda c: c.list_communications(conversation_id="conv_1", channel_id=SECRET),
        "GET",
    ),
    (
        "create_action",
        _conversation,
        lambda c: c.create_action(
            conversation_id="conv_1",
            request=SendMessageActionRequest(
                payload=SendMessageActionPayload(
                    from_=_action_ref(), to=[_action_ref()], content=ActionTextContent(text=SECRET)
                )
            ),
        ),
        "POST",
    ),
    (
        "search_knowledge_base",
        _knowledge,
        lambda c: c.search_knowledge_base(knowledge_base_id="know_kb_1", query=SECRET),
        "POST",
    ),
]


def _logged_text(logger: MagicMock) -> str:
    return " ".join(str(c) for c in logger.mock_calls)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_transport")
@pytest.mark.parametrize(
    ("name", "make_client", "call", "method"), CASES, ids=[c[0] for c in CASES]
)
async def test_failed_request_logs_no_customer_data(
    name: str,
    make_client: Callable[[], BaseAPIClient],
    call: Callable[[Any], Any],
    method: str,
) -> None:
    client = make_client()
    client.logger = MagicMock()

    try:
        await call(client)
    except httpx.HTTPStatusError:
        pass

    logged = _logged_text(client.logger)
    assert client.logger.mock_calls, f"{name} logged nothing on failure"
    for marker in PII_MARKERS:
        assert marker not in logged, f"{name} logged customer data: {logged}"
    assert method in logged
    assert "400" in logged
    assert client.base_url in logged


@pytest.mark.no_auto_mock
@pytest.mark.usefixtures("mock_transport")
def test_failed_get_configuration_logs_method_url_and_status() -> None:
    client = _conversation()
    client.logger = MagicMock()

    with pytest.raises(httpx.HTTPStatusError):
        client.get_configuration("conf_test")

    logged = _logged_text(client.logger)
    assert "GET" in logged
    assert "400" in logged
    assert f"{client.base_url}/v2/ControlPlane/Configurations/conf_test" in logged
    assert "bad request" not in logged

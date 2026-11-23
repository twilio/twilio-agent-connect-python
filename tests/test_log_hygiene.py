"""Failed API calls must not log request or response content."""

import json
import logging
from collections.abc import Awaitable, Callable, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from tac.channels.base import BaseChannel
from tac.context.base import BaseAPIClient, describe_error
from tac.context.conversation import ConversationClient
from tac.context.memory import MemoryClient
from tac.models.conversation import (
    ActionParticipantRef,
    ActionTextContent,
    CommunicationContent,
    CommunicationParticipant,
    CommunicationRequest,
    ParticipantAddress,
    SendMessageActionPayload,
    SendMessageActionRequest,
)

# Not phone- or email-shaped, so the log scrubber can't hide a leak.
SECRET = "end-user-secret-text"
RESPONSE_SECRET = "response-secret-text"


@pytest.fixture
def records() -> Iterator[list[logging.LogRecord]]:
    captured: list[logging.LogRecord] = []

    class ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    logger = logging.getLogger("tac")
    handler = ListHandler()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield captured
    logger.removeHandler(handler)
    logger.setLevel(previous_level)


def _logged(records: list[logging.LogRecord]) -> str:
    return "\n".join(f"{r.getMessage()} {r.__dict__}" for r in records)


def _failing_client(status: int = 400) -> Callable[[], httpx.AsyncClient]:
    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "code": 20001,
            "message": f"{RESPONSE_SECRET} {request.content.decode()}",
        }
        return httpx.Response(status, json=body)

    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _responding_client(body: object, status: int = 200) -> Callable[[], httpx.AsyncClient]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _conversation_client() -> ConversationClient:
    return ConversationClient(api_key="SK1", api_secret="secret", configuration_id="conv_cfg_1")


def _memory_client() -> MemoryClient:
    return MemoryClient(store_id="mem_store_1", api_key="SK1", api_secret="secret")


def _address() -> ParticipantAddress:
    return ParticipantAddress(channel="CHAT", address=SECRET)


def _action() -> SendMessageActionRequest:
    return SendMessageActionRequest(
        payload=SendMessageActionPayload(
            from_=ActionParticipantRef(address=SECRET, channel="CHAT"),
            to=[ActionParticipantRef(address=SECRET, channel="CHAT")],
            content=ActionTextContent(text=SECRET),
        )
    )


def _communication() -> CommunicationRequest:
    participant = CommunicationParticipant(address=SECRET, channel="CHAT", participant_id="part_1")
    return CommunicationRequest(
        author=participant,
        content=CommunicationContent(type="TEXT", text=SECRET),
        recipients=[participant],
    )


CallFactory = Callable[[Any], Awaitable[object]]

CALLS: list[tuple[str, Callable[[], BaseAPIClient], CallFactory, str, str]] = [
    (
        "create_action",
        _conversation_client,
        lambda c: c.create_action("conv_1", _action()),
        "POST",
        "/v2/Conversations/conv_1/Actions",
    ),
    (
        "create_communication",
        _conversation_client,
        lambda c: c.create_communication("conv_1", _communication()),
        "POST",
        "/v2/Conversations/conv_1/Communications",
    ),
    (
        "add_participant",
        _conversation_client,
        lambda c: c.add_participant("conv_1", addresses=[_address()]),
        "POST",
        "/v2/Conversations/conv_1/Participants",
    ),
    (
        "update_participant",
        _conversation_client,
        lambda c: c.update_participant("conv_1", "part_1", "CUSTOMER", [_address()], name=SECRET),
        "PUT",
        "/v2/Conversations/conv_1/Participants/part_1",
    ),
    (
        "create_conversation",
        _conversation_client,
        lambda c: c.create_conversation(name=SECRET, metadata={"note": SECRET}),
        "POST",
        "/v2/Conversations",
    ),
    (
        "update_conversation",
        _conversation_client,
        lambda c: c.update_conversation("conv_1", "CLOSED", name=SECRET),
        "PUT",
        "/v2/Conversations/conv_1",
    ),
    (
        "patch_conversation_metadata",
        _conversation_client,
        lambda c: c.patch_conversation_metadata("conv_1", {"note": SECRET}),
        "PATCH",
        "/v2/Conversations/conv_1",
    ),
    (
        "retrieve_memory",
        _memory_client,
        lambda c: c.retrieve_memory("mem_profile_1", query=SECRET),
        "POST",
        "/v1/Stores/mem_store_1/Profiles/mem_profile_1/Recall",
    ),
    (
        "get_profile",
        _memory_client,
        lambda c: c.get_profile("mem_profile_1", trait_groups=["Contact"]),
        "GET",
        "/v1/Stores/mem_store_1/Profiles/mem_profile_1",
    ),
    (
        "lookup_profile",
        _memory_client,
        lambda c: c.lookup_profile("phone", SECRET),
        "POST",
        "/v1/Stores/mem_store_1/Profiles/Lookup",
    ),
    (
        "create_profile",
        _memory_client,
        lambda c: c.create_profile({"Contact": {"phone": SECRET}}),
        "POST",
        "/v1/Stores/mem_store_1/Profiles",
    ),
    (
        "create_observation",
        _memory_client,
        lambda c: c.create_observation("mem_profile_1", SECRET),
        "POST",
        "/v1/Stores/mem_store_1/Profiles/mem_profile_1/Observations",
    ),
    (
        "create_conversation_summaries",
        _memory_client,
        lambda c: c.create_conversation_summaries("mem_profile_1", [{"content": SECRET}]),
        "POST",
        "/v1/Stores/mem_store_1/Profiles/mem_profile_1/ConversationSummaries",
    ),
    (
        "list_conversations",
        _conversation_client,
        lambda c: c.list_conversations(status=["ACTIVE"]),
        "GET",
        "/v2/Conversations",
    ),
    (
        "list_communications",
        _conversation_client,
        lambda c: c.list_communications("conv_1"),
        "GET",
        "/v2/Conversations/conv_1/Communications",
    ),
    (
        "list_participants",
        _conversation_client,
        lambda c: c.list_participants("conv_1"),
        "GET",
        "/v2/Conversations/conv_1/Participants",
    ),
    (
        "get_conversation",
        _conversation_client,
        lambda c: c.get_conversation("conv_1"),
        "GET",
        "/v2/Conversations/conv_1",
    ),
    (
        "clear_status_callbacks",
        _conversation_client,
        lambda c: c.clear_status_callbacks("conv_1"),
        "PATCH",
        "/v2/Conversations/conv_1",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("make_client", "call", "method", "path"),
    [c[1:] for c in CALLS],
    ids=[c[0] for c in CALLS],
)
async def test_http_error_logs_no_request_or_response_content(
    records: list[logging.LogRecord],
    make_client: Callable[[], BaseAPIClient],
    call: CallFactory,
    method: str,
    path: str,
) -> None:
    client = make_client()
    with patch.object(client, "_get_client", _failing_client()):
        try:
            await call(client)
        except httpx.HTTPStatusError:
            pass

    logged = _logged(records)
    assert f"{method} {path}: status 400, Twilio error code 20001" in logged
    assert SECRET not in logged
    assert RESPONSE_SECRET not in logged


@pytest.mark.asyncio
async def test_invalid_profile_response_logs_no_traits(
    records: list[logging.LogRecord],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"traits": SECRET}))

    client = _memory_client()
    with (
        patch.object(
            client,
            "_get_client",
            lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ),
        pytest.raises(ValueError) as excinfo,
    ):
        await client.get_profile("mem_profile_1")

    assert SECRET not in str(excinfo.value)
    assert SECRET not in _logged(records)


def test_dropped_recall_item_logs_no_content(records: list[logging.LogRecord]) -> None:
    response = _memory_client()._parse_recall_response(
        {"observations": [{"content": SECRET, "id": 1}]}
    )

    assert response.observations == []
    assert "Dropping invalid Conversation Memory ObservationInfo" in _logged(records)
    assert SECRET not in _logged(records)


@pytest.mark.parametrize("code", [SECRET, {"detail": SECRET}, True])
def test_non_integer_error_code_is_not_logged(code: object) -> None:
    request = httpx.Request("POST", "https://conversations.twilio.com/v2/Conversations")
    response = httpx.Response(400, json={"code": code}, request=request)
    error = httpx.HTTPStatusError("failed", request=request, response=response)

    assert describe_error(error) == "POST /v2/Conversations: status 400"


@pytest.mark.asyncio
async def test_conversation_dedup_409_logs_no_content(records: list[logging.LogRecord]) -> None:
    client = _conversation_client()
    with (
        patch.object(client, "_get_client", _failing_client(status=409)),
        pytest.raises(httpx.HTTPStatusError),
    ):
        await client.create_conversation(name=SECRET)

    logged = _logged(records)
    assert "POST /v2/Conversations: status 409" in logged
    assert SECRET not in logged
    assert RESPONSE_SECRET not in logged


@pytest.mark.asyncio
async def test_invalid_participants_response_logs_no_content(
    records: list[logging.LogRecord],
) -> None:
    client = _conversation_client()
    with patch.object(
        client, "_get_client", _responding_client({"participants": [{"name": SECRET}]})
    ):
        assert await client.list_participants("conv_1") == []

    assert "Invalid list participants response for conversation conv_1" in _logged(records)
    assert SECRET not in _logged(records)


@pytest.mark.asyncio
async def test_invalid_lookup_response_logs_no_content(records: list[logging.LogRecord]) -> None:
    client = _memory_client()
    with (
        patch.object(client, "_get_client", _responding_client({"profiles": SECRET})),
        pytest.raises(ValueError) as excinfo,
    ):
        await client.lookup_profile("phone", "value")

    assert SECRET not in str(excinfo.value)
    assert SECRET not in _logged(records)


@pytest.mark.asyncio
async def test_create_profile_without_id_raises_without_content() -> None:
    client = _memory_client()
    with (
        patch.object(client, "_get_client", _responding_client({"traits": SECRET})),
        pytest.raises(ValueError) as excinfo,
    ):
        await client.create_profile({"Contact": {"phone": "value"}})

    assert SECRET not in str(excinfo.value)


def test_dropped_recall_meta_logs_no_content(records: list[logging.LogRecord]) -> None:
    _memory_client()._parse_recall_response({"meta": {"queryTime": SECRET}})

    assert "Dropping invalid Conversation Memory meta" in _logged(records)
    assert SECRET not in _logged(records)


def _sync_client(body: object, status: int) -> Callable[[], httpx.Client]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return lambda: httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.no_auto_mock
@pytest.mark.parametrize(
    ("body", "status", "expected"),
    [
        (
            {"code": 20404, "message": RESPONSE_SECRET},
            404,
            "GET /v2/ControlPlane/Configurations/conv_cfg_1: status 404, Twilio error code 20404",
        ),
        ({"id": RESPONSE_SECRET}, 200, "Failed to parse configuration conv_cfg_1"),
    ],
    ids=["http_error", "invalid_response"],
)
def test_get_configuration_logs_no_response_content(
    records: list[logging.LogRecord], body: object, status: int, expected: str
) -> None:
    client = _conversation_client()
    with (
        patch.object(client, "_get_sync_client", _sync_client(body, status)),
        pytest.raises((httpx.HTTPStatusError, ValueError)),
    ):
        client.get_configuration("conv_cfg_1")

    assert expected in _logged(records)
    assert RESPONSE_SECRET not in _logged(records)


def test_unparseable_participant_added_logs_no_content() -> None:
    channel = MagicMock()

    BaseChannel._track_conversation_started(channel, {"name": SECRET})

    channel.logger.debug.assert_called_once()
    assert SECRET not in str(channel.logger.mock_calls)

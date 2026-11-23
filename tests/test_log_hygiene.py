"""Failed API calls must not log request or response content."""

import json
import logging
from collections.abc import Awaitable, Callable, Iterator
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from tac.context.base import BaseAPIClient
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

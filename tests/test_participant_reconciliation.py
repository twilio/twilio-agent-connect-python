"""Tests for `_reconcile_participants` in MessagingChannel.

Covers the matrix of participant states that v1-bridge capture can leave us
with. The resolution rules were agreed with the Maestro team.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from tac import TAC
from tac.channels.sms import SMSChannel
from tac.models.conversation import ParticipantAddress, ParticipantResponse


def _participant(
    pid: str,
    ptype: str,
    address: str,
    channel: str = "SMS",
    conv_id: str = "CH123",
) -> ParticipantResponse:
    addr = ParticipantAddress(channel=channel, address=address).model_dump(  # type: ignore[arg-type]
        by_alias=True
    )
    return ParticipantResponse(
        **{  # type: ignore[arg-type]
            "id": pid,
            "accountId": "ACtest",
            "conversationId": conv_id,
            "name": address,
            "type": ptype,
            "addresses": [addr],
        }
    )


def _tac() -> TAC:
    cfg: dict[str, Any] = {
        "account_sid": "ACtest",
        "auth_token": "t",
        "api_key": "SK",
        "api_secret": "s",
        "conversation_configuration_id": "conv_configuration_test",
        "phone_number": "+15551234567",
    }
    return TAC(cfg)


@pytest.fixture(autouse=True)
def stub_memory_profile_calls(request):
    """Stub memory-client profile calls so reconciliation tests don't hit the network.

    Profile lookup/create is invoked during UNKNOWN → CUSTOMER promotion. Tests
    that specifically exercise the profile resolution path can mark themselves
    with `no_stub_profile_calls` to manage their own mocks.
    """
    if "no_stub_profile_calls" in request.keywords:
        yield
        return
    with (
        patch(
            "tac.context.memory.MemoryClient.lookup_profile",
            new=AsyncMock(side_effect=RuntimeError("lookup_profile not stubbed")),
        ),
        patch(
            "tac.context.memory.MemoryClient.create_profile",
            new=AsyncMock(side_effect=RuntimeError("create_profile not stubbed")),
        ),
    ):
        yield


@pytest.mark.asyncio
async def test_agent_plus_customer_no_puts() -> None:
    """Happy path: both sides correctly typed → no PUTs."""
    tac = _tac()
    channel = SMSChannel(tac)

    agent = _participant("PA_A", "AI_AGENT", "+15551234567")
    customer = _participant("PA_C", "CUSTOMER", "+12345678901")

    with (
        patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            return_value=[agent, customer],
        ),
        patch.object(tac.conversation_orchestrator_client, "update_participant") as mock_update,
    ):
        result = await channel._reconcile_participants("CH123")

    assert result is not None
    assert result[0].id == "PA_A"
    assert result[1] is not None
    assert result[1].id == "PA_C"
    mock_update.assert_not_called()


@pytest.mark.asyncio
async def test_agent_plus_unknown_customer_promotes_customer() -> None:
    """Agent is good; customer is UNKNOWN → promote to CUSTOMER."""
    tac = _tac()
    channel = SMSChannel(tac)

    agent = _participant("PA_A", "AI_AGENT", "+15551234567")
    unknown_customer = _participant("PA_C", "UNKNOWN", "+12345678901")
    promoted = _participant("PA_C", "CUSTOMER", "+12345678901")

    with (
        patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            return_value=[agent, unknown_customer],
        ),
        patch.object(
            tac.conversation_orchestrator_client,
            "update_participant",
            new=AsyncMock(return_value=promoted),
        ) as mock_update,
    ):
        result = await channel._reconcile_participants("CH123")

    assert result is not None
    assert result[0].id == "PA_A"
    assert result[1] is not None
    assert result[1].id == "PA_C"
    assert result[1].type == "CUSTOMER"
    mock_update.assert_called_once()
    call = mock_update.call_args
    assert call.kwargs["participant_id"] == "PA_C"
    assert call.kwargs["participant_type"] == "CUSTOMER"


@pytest.mark.asyncio
async def test_unknown_agent_plus_customer_promotes_agent() -> None:
    """v1-bridge bug: agent side is UNKNOWN → promote to AI_AGENT."""
    tac = _tac()
    channel = SMSChannel(tac)

    unknown_agent = _participant("PA_A", "UNKNOWN", "+15551234567")
    customer = _participant("PA_C", "CUSTOMER", "+12345678901")
    promoted = _participant("PA_A", "AI_AGENT", "+15551234567")

    with (
        patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            return_value=[unknown_agent, customer],
        ),
        patch.object(
            tac.conversation_orchestrator_client,
            "update_participant",
            new=AsyncMock(return_value=promoted),
        ) as mock_update,
    ):
        result = await channel._reconcile_participants("CH123")

    assert result is not None
    assert result[0].id == "PA_A"
    assert result[0].type == "AI_AGENT"
    assert result[1] is not None
    assert result[1].id == "PA_C"
    mock_update.assert_called_once()
    call = mock_update.call_args
    assert call.kwargs["participant_id"] == "PA_A"
    assert call.kwargs["participant_type"] == "AI_AGENT"


@pytest.mark.asyncio
async def test_unknown_agent_plus_unknown_customer_promotes_both() -> None:
    """Both sides UNKNOWN → two PUTs."""
    tac = _tac()
    channel = SMSChannel(tac)

    unknown_agent = _participant("PA_A", "UNKNOWN", "+15551234567")
    unknown_customer = _participant("PA_C", "UNKNOWN", "+12345678901")
    promoted_agent = _participant("PA_A", "AI_AGENT", "+15551234567")
    promoted_customer = _participant("PA_C", "CUSTOMER", "+12345678901")

    def update_side_effect(**kwargs: Any) -> ParticipantResponse:
        return promoted_agent if kwargs["participant_id"] == "PA_A" else promoted_customer

    with (
        patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            return_value=[unknown_agent, unknown_customer],
        ),
        patch.object(
            tac.conversation_orchestrator_client,
            "update_participant",
            new=AsyncMock(side_effect=update_side_effect),
        ) as mock_update,
    ):
        result = await channel._reconcile_participants("CH123")

    assert result is not None
    assert result[0].type == "AI_AGENT"
    assert result[1] is not None
    assert result[1].type == "CUSTOMER"
    assert mock_update.call_count == 2


@pytest.mark.asyncio
async def test_solo_customer_posts_agent() -> None:
    """v1-bridge inbound: only customer participant → POST AI_AGENT, then use."""
    tac = _tac()
    channel = SMSChannel(tac)

    customer = _participant("PA_C", "CUSTOMER", "+12345678901")
    created_agent = _participant("PA_A", "AI_AGENT", "+15551234567")

    with (
        patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            return_value=[customer],
        ),
        patch.object(
            tac.conversation_orchestrator_client,
            "add_participant",
            new=AsyncMock(return_value=created_agent),
        ) as mock_add,
        patch.object(tac.conversation_orchestrator_client, "update_participant") as mock_update,
    ):
        result = await channel._reconcile_participants("CH123")

    assert result is not None
    assert result[0].id == "PA_A"
    assert result[0].type == "AI_AGENT"
    assert result[1] is not None
    assert result[1].id == "PA_C"
    mock_add.assert_called_once()
    call = mock_add.call_args
    assert call.kwargs["participant_type"] == "AI_AGENT"
    assert call.kwargs["addresses"][0].address == "+15551234567"
    mock_update.assert_not_called()


@pytest.mark.asyncio
async def test_solo_unknown_customer_posts_agent_and_promotes_customer() -> None:
    """Only an UNKNOWN customer-side participant → POST agent + PUT customer."""
    tac = _tac()
    channel = SMSChannel(tac)

    unknown_customer = _participant("PA_C", "UNKNOWN", "+12345678901")
    created_agent = _participant("PA_A", "AI_AGENT", "+15551234567")
    promoted_customer = _participant("PA_C", "CUSTOMER", "+12345678901")

    with (
        patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            return_value=[unknown_customer],
        ),
        patch.object(
            tac.conversation_orchestrator_client,
            "add_participant",
            new=AsyncMock(return_value=created_agent),
        ) as mock_add,
        patch.object(
            tac.conversation_orchestrator_client,
            "update_participant",
            new=AsyncMock(return_value=promoted_customer),
        ) as mock_update,
    ):
        result = await channel._reconcile_participants("CH123")

    assert result is not None
    assert result[0].type == "AI_AGENT"
    assert result[1] is not None
    assert result[1].type == "CUSTOMER"
    mock_add.assert_called_once()
    mock_update.assert_called_once()
    update_call = mock_update.call_args
    assert update_call.kwargs["participant_id"] == "PA_C"
    assert update_call.kwargs["participant_type"] == "CUSTOMER"


@pytest.mark.asyncio
async def test_add_agent_409_picks_up_existing_owner() -> None:
    """POST AI_AGENT returns 409 → re-list, pick up the AI_AGENT another worker added."""
    tac = _tac()
    channel = SMSChannel(tac)

    customer = _participant("PA_C", "CUSTOMER", "+12345678901")
    added_by_other = _participant("PA_A", "AI_AGENT", "+15551234567")

    mock_response = httpx.Response(
        status_code=409, request=httpx.Request("POST", "http://example.invalid")
    )
    conflict = httpx.HTTPStatusError("409", request=mock_response.request, response=mock_response)

    list_calls = [
        [customer],  # initial reconciliation list
        [customer, added_by_other],  # re-list after POST 409
    ]

    with (
        patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            new=AsyncMock(side_effect=list_calls),
        ),
        patch.object(
            tac.conversation_orchestrator_client,
            "add_participant",
            new=AsyncMock(side_effect=conflict),
        ),
    ):
        result = await channel._reconcile_participants("CH123")

    assert result is not None
    assert result[0].id == "PA_A"
    assert result[0].type == "AI_AGENT"
    assert result[1] is not None
    assert result[1].id == "PA_C"


@pytest.mark.asyncio
async def test_empty_participants_returns_none() -> None:
    """No participants at all (e.g. brand-new conversation with no customer) → skip."""
    tac = _tac()
    channel = SMSChannel(tac)

    created_agent = _participant("PA_A", "AI_AGENT", "+15551234567")

    with (
        patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            return_value=[],
        ),
        patch.object(
            tac.conversation_orchestrator_client,
            "add_participant",
            new=AsyncMock(return_value=created_agent),
        ),
    ):
        result = await channel._reconcile_participants("CH123")

    # Agent posted, but no customer resolvable → skip.
    assert result is None


@pytest.mark.asyncio
async def test_promote_409_refetches_and_proceeds() -> None:
    """PUT returning 409 is treated as concurrent update: re-list and use."""
    tac = _tac()
    channel = SMSChannel(tac)

    unknown_agent = _participant("PA_A", "UNKNOWN", "+15551234567")
    customer = _participant("PA_C", "CUSTOMER", "+12345678901")
    promoted_by_other = _participant("PA_A", "AI_AGENT", "+15551234567")

    mock_response = httpx.Response(
        status_code=409, request=httpx.Request("PUT", "http://example.invalid")
    )
    conflict = httpx.HTTPStatusError("409", request=mock_response.request, response=mock_response)

    list_calls = [
        [unknown_agent, customer],  # first list during reconciliation
        [promoted_by_other, customer],  # second list after 409
    ]

    with (
        patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            new=AsyncMock(side_effect=list_calls),
        ),
        patch.object(
            tac.conversation_orchestrator_client,
            "update_participant",
            new=AsyncMock(side_effect=conflict),
        ),
    ):
        result = await channel._reconcile_participants("CH123")

    assert result is not None
    assert result[0].type == "AI_AGENT"


@pytest.mark.asyncio
async def test_promote_409_refetch_shows_wrong_type_returns_none() -> None:
    """PUT 409 then re-list shows the participant still at the old type → skip."""
    tac = _tac()
    channel = SMSChannel(tac)

    unknown_agent = _participant("PA_A", "UNKNOWN", "+15551234567")
    customer = _participant("PA_C", "CUSTOMER", "+12345678901")

    mock_response = httpx.Response(
        status_code=409, request=httpx.Request("PUT", "http://example.invalid")
    )
    conflict = httpx.HTTPStatusError("409", request=mock_response.request, response=mock_response)

    list_calls = [
        [unknown_agent, customer],  # first list during reconciliation
        [unknown_agent, customer],  # after 409, participant still UNKNOWN
    ]

    with (
        patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            new=AsyncMock(side_effect=list_calls),
        ),
        patch.object(
            tac.conversation_orchestrator_client,
            "update_participant",
            new=AsyncMock(side_effect=conflict),
        ),
    ):
        result = await channel._reconcile_participants("CH123")

    assert result is None


@pytest.mark.asyncio
async def test_list_participants_failure_returns_none() -> None:
    """list_participants raising (e.g. Maestro down) skips the webhook."""
    tac = _tac()
    channel = SMSChannel(tac)

    with (
        patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            new=AsyncMock(side_effect=httpx.ConnectError("maestro unreachable")),
        ),
        patch.object(tac.conversation_orchestrator_client, "update_participant") as mock_update,
    ):
        result = await channel._reconcile_participants("CH123")

    assert result is None
    mock_update.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.no_stub_profile_calls
async def test_customer_promotion_reuses_existing_profile() -> None:
    """Lookup hit: promote UNKNOWN → CUSTOMER with the existing profile_id."""
    from tac.models.memory import ProfileLookupResponse

    tac = _tac()
    channel = SMSChannel(tac)

    agent = _participant("PA_A", "AI_AGENT", "+15551234567")
    unknown_customer = _participant("PA_C", "UNKNOWN", "+12345678901")
    promoted = _participant("PA_C", "CUSTOMER", "+12345678901")

    with (
        patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            return_value=[agent, unknown_customer],
        ),
        patch.object(
            tac.conversation_orchestrator_client,
            "update_participant",
            new=AsyncMock(return_value=promoted),
        ) as mock_update,
        patch.object(
            tac.conversation_memory_client,
            "lookup_profile",
            new=AsyncMock(
                return_value=ProfileLookupResponse(
                    normalizedValue="+12345678901",
                    profiles=["mem_profile_01existing"],
                ),
            ),
        ) as mock_lookup,
        patch.object(
            tac.conversation_memory_client,
            "create_profile",
            new=AsyncMock(),
        ) as mock_create,
    ):
        result = await channel._reconcile_participants("CH123")

    assert result is not None
    assert result[1] is not None and result[1].id == "PA_C"
    mock_lookup.assert_awaited_once_with(id_type="phone", value="+12345678901")
    mock_create.assert_not_called()
    mock_update.assert_awaited_once()
    kwargs = mock_update.await_args.kwargs
    assert kwargs["participant_type"] == "CUSTOMER"
    assert kwargs["profile_id"] == "mem_profile_01existing"
    assert kwargs["addresses"] == unknown_customer.addresses


@pytest.mark.asyncio
@pytest.mark.no_stub_profile_calls
async def test_customer_promotion_creates_profile_on_lookup_miss() -> None:
    """Lookup miss: create a new profile with configured trait group/field."""
    from tac.models.memory import ProfileLookupResponse

    tac = _tac()
    channel = SMSChannel(tac)

    agent = _participant("PA_A", "AI_AGENT", "+15551234567")
    unknown_customer = _participant("PA_C", "UNKNOWN", "+12345678901")
    promoted = _participant("PA_C", "CUSTOMER", "+12345678901")

    with (
        patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            return_value=[agent, unknown_customer],
        ),
        patch.object(
            tac.conversation_orchestrator_client,
            "update_participant",
            new=AsyncMock(return_value=promoted),
        ) as mock_update,
        patch.object(
            tac.conversation_memory_client,
            "lookup_profile",
            new=AsyncMock(
                return_value=ProfileLookupResponse(
                    normalizedValue="+12345678901",
                    profiles=[],
                ),
            ),
        ) as mock_lookup,
        patch.object(
            tac.conversation_memory_client,
            "create_profile",
            new=AsyncMock(return_value="mem_profile_01new"),
        ) as mock_create,
    ):
        result = await channel._reconcile_participants("CH123")

    assert result is not None
    mock_lookup.assert_awaited_once()
    mock_create.assert_awaited_once_with(
        traits={"Contact": {"phone": "+12345678901"}},
    )
    kwargs = mock_update.await_args.kwargs
    assert kwargs["profile_id"] == "mem_profile_01new"


@pytest.mark.asyncio
@pytest.mark.no_stub_profile_calls
async def test_customer_promotion_proceeds_without_profile_on_errors() -> None:
    """Both lookup and create failing still promote the customer (profile_id=None)."""
    tac = _tac()
    channel = SMSChannel(tac)

    agent = _participant("PA_A", "AI_AGENT", "+15551234567")
    unknown_customer = _participant("PA_C", "UNKNOWN", "+12345678901")
    promoted = _participant("PA_C", "CUSTOMER", "+12345678901")

    with (
        patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            return_value=[agent, unknown_customer],
        ),
        patch.object(
            tac.conversation_orchestrator_client,
            "update_participant",
            new=AsyncMock(return_value=promoted),
        ) as mock_update,
        patch.object(
            tac.conversation_memory_client,
            "lookup_profile",
            new=AsyncMock(side_effect=httpx.ConnectError("memora down")),
        ),
        patch.object(
            tac.conversation_memory_client,
            "create_profile",
            new=AsyncMock(side_effect=httpx.ConnectError("memora down")),
        ),
    ):
        result = await channel._reconcile_participants("CH123")

    assert result is not None
    kwargs = mock_update.await_args.kwargs
    assert kwargs["profile_id"] is None
    assert kwargs["participant_type"] == "CUSTOMER"

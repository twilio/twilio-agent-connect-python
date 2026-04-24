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
async def test_no_agent_address_owner_returns_none() -> None:
    """No participant owns the agent's address → skip (None)."""
    tac = _tac()
    channel = SMSChannel(tac)

    customer = _participant("PA_C", "CUSTOMER", "+12345678901")

    with (
        patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            return_value=[customer],
        ),
        patch.object(tac.conversation_orchestrator_client, "update_participant") as mock_update,
    ):
        result = await channel._reconcile_participants("CH123")

    assert result is None
    mock_update.assert_not_called()


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

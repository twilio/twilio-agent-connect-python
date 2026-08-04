"""Tests for the on_error callback and its wiring into channel processing.

Covers:
- TAC.on_error / TAC.trigger_error registration and sync/async invocation.
- MessagingChannel routing a reconcile failure (dropped inbound) through
  on_error instead of silently dropping the message.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from tac import TAC
from tac.channels.sms import SMSChannel


def get_test_config() -> dict[str, Any]:
    return {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": "conv_configuration_test123",
        "phone_number": "+15551234567",
    }


class TestErrorCallbackRegistration:
    """TAC.on_error registration and trigger_error dispatch."""

    @pytest.mark.asyncio
    async def test_sync_error_callback(self) -> None:
        tac = TAC(get_test_config())
        received: list[tuple[Exception, dict[str, Any]]] = []

        def handler(error: Exception, context: dict[str, Any]) -> None:
            received.append((error, context))

        tac.on_error(handler)

        err = RuntimeError("boom")
        await tac.trigger_error(err, {"conversation_id": "CH1"})

        assert len(received) == 1
        assert received[0][0] is err
        assert received[0][1]["conversation_id"] == "CH1"

    @pytest.mark.asyncio
    async def test_async_error_callback(self) -> None:
        tac = TAC(get_test_config())
        received: list[tuple[Exception, dict[str, Any]]] = []

        async def handler(error: Exception, context: dict[str, Any]) -> None:
            received.append((error, context))

        tac.on_error(handler)

        err = ValueError("nope")
        await tac.trigger_error(err, {"channel": "SMS"})

        assert len(received) == 1
        assert received[0][0] is err
        assert received[0][1]["channel"] == "SMS"

    @pytest.mark.asyncio
    async def test_no_handler_is_noop(self) -> None:
        """Backward compatible: no handler registered → trigger_error is a no-op."""
        tac = TAC(get_test_config())
        # Must not raise.
        await tac.trigger_error(RuntimeError("boom"), {"conversation_id": "CH1"})

    @pytest.mark.asyncio
    async def test_handler_exception_is_swallowed_and_logged(self) -> None:
        """A handler that raises must not propagate; the failure is logged instead."""
        tac = TAC(get_test_config())

        def bad_handler(error: Exception, context: dict[str, Any]) -> None:
            raise ValueError("handler blew up")

        tac.on_error(bad_handler)

        with patch.object(tac.logger, "error") as mock_error:
            # Must not raise despite the handler raising.
            await tac.trigger_error(RuntimeError("boom"), {"conversation_id": "CH1"})

        mock_error.assert_called_once()
        assert "on_error callback raised an exception" in mock_error.call_args.args[0]


class TestReconcileFailureRoutesToOnError:
    """A reconcile failure must fire on_error and NOT the message-ready callback."""

    @pytest.mark.asyncio
    async def test_dropped_inbound_invokes_on_error_not_message_ready(self) -> None:
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        message_ready_calls: list[Any] = []
        error_calls: list[tuple[Exception, dict[str, Any]]] = []

        async def on_message_ready(*args: Any) -> None:
            message_ready_calls.append(args)

        def on_error(error: Exception, context: dict[str, Any]) -> None:
            error_calls.append((error, context))

        tac.on_message_ready(on_message_ready)
        tac.on_error(on_error)

        webhook_event = {
            "id": "comms_communication_01test",
            "conversationId": "CH123",
            "accountId": "ACtest",
            "author": {
                "address": "+12345678901",
                "channel": "SMS",
                "participantId": "PA_C",
            },
            "content": {"type": "TEXT", "text": "hi"},
            "recipients": [
                {
                    "address": "+15551234567",
                    "channel": "SMS",
                    "participantId": "PA_A",
                }
            ],
            "createdAt": "2026-04-27T00:00:00Z",
        }

        # list_participants unreachable → _reconcile_participants returns None.
        with patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            new=AsyncMock(side_effect=httpx.ConnectError("conversation orchestrator unreachable")),
        ):
            await channel._handle_communication_created(webhook_event)

        assert message_ready_calls == []
        assert len(error_calls) == 1
        _error, context = error_calls[0]
        assert context["conversation_id"] == "CH123"
        assert context["channel"] == "SMS"
        assert context["dropped_inbound"] is True

    @pytest.mark.asyncio
    async def test_dropped_inbound_without_on_error_does_not_raise(self) -> None:
        """No on_error handler → still degrades gracefully (logs, no callback, no raise)."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        message_ready_calls: list[Any] = []

        async def on_message_ready(*args: Any) -> None:
            message_ready_calls.append(args)

        tac.on_message_ready(on_message_ready)

        webhook_event = {
            "id": "comms_communication_01test",
            "conversationId": "CH123",
            "accountId": "ACtest",
            "author": {
                "address": "+12345678901",
                "channel": "SMS",
                "participantId": "PA_C",
            },
            "content": {"type": "TEXT", "text": "hi"},
            "recipients": [
                {
                    "address": "+15551234567",
                    "channel": "SMS",
                    "participantId": "PA_A",
                }
            ],
            "createdAt": "2026-04-27T00:00:00Z",
        }

        with patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            new=AsyncMock(side_effect=httpx.ConnectError("conversation orchestrator unreachable")),
        ):
            await channel._handle_communication_created(webhook_event)

        assert message_ready_calls == []

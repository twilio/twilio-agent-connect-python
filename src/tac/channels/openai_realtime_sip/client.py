"""Thin async client for OpenAI's Realtime SIP call control API.

Wraps the REST endpoints used to accept/reject/hangup a SIP call
(https://developers.openai.com/api/docs/guides/realtime-sip) and the
JSON-only control WebSocket used to observe/steer an accepted call.
"""

from typing import Any

import httpx

from tac.channels.openai_realtime_sip.models import (
    OpenAIRealtimeSipCallIncoming,
    OpenAIRealtimeSipSessionConfig,
)
from tac.core.logging import get_logger

try:
    import websockets
except ImportError as e:
    raise ImportError(
        "OpenAIRealtimeSipClient requires the 'websockets' package for the control "
        "WebSocket. Install with: pip install tac[openai-realtime-sip]"
    ) from e

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "OpenAIRealtimeSipClient requires the 'openai' package to verify webhook "
        "signatures. Install with: pip install tac[openai-realtime-sip]"
    ) from e

logger = get_logger(__name__)

REALTIME_BASE_URL = "https://api.openai.com/v1/realtime"


def verify_and_parse_incoming_call(
    payload: str | bytes, headers: dict[str, str], webhook_secret: str
) -> OpenAIRealtimeSipCallIncoming | None:
    """Verify an OpenAI webhook's signature and parse it as a call-incoming event.

    Returns ``None`` if the (verified) webhook is a different event type — TAC's
    handler should treat that as "nothing to do", not an error.

    Raises:
        openai.InvalidWebhookSignatureError: if the signature doesn't verify.
    """
    client = OpenAI(api_key="unused", webhook_secret=webhook_secret)
    event = client.webhooks.unwrap(payload, headers)
    if event.type != "realtime.call.incoming":
        return None
    return OpenAIRealtimeSipCallIncoming(
        call_id=event.data.call_id,
        sip_headers=[{"name": h.name, "value": h.value} for h in event.data.sip_headers],
    )


class OpenAIRealtimeSipClient:
    """Async client for OpenAI Realtime SIP call control and the control WebSocket."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def accept_call(
        self, call_id: str, session_config: OpenAIRealtimeSipSessionConfig
    ) -> None:
        """Accept an incoming call, configuring the Realtime session that answers it."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{REALTIME_BASE_URL}/calls/{call_id}/accept",
                headers=self._headers(),
                json=session_config.to_accept_payload(),
            )
            response.raise_for_status()

    async def reject_call(self, call_id: str, status_code: int = 603) -> None:
        """Reject an incoming call with the given SIP status code."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{REALTIME_BASE_URL}/calls/{call_id}/reject",
                headers=self._headers(),
                json={"status_code": status_code},
            )
            response.raise_for_status()

    async def hangup_call(self, call_id: str) -> None:
        """End an in-progress call."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{REALTIME_BASE_URL}/calls/{call_id}/hangup",
                headers=self._headers(),
            )
            response.raise_for_status()

    async def refer_call(self, call_id: str, target_uri: str) -> None:
        """Transfer an active call via SIP REFER."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{REALTIME_BASE_URL}/calls/{call_id}/refer",
                headers=self._headers(),
                json={"target_uri": target_uri},
            )
            response.raise_for_status()

    def control_connection(self, call_id: str) -> Any:
        """Open the control WebSocket for an already-accepted call.

        Returns the ``websockets`` connect context manager (not yet entered) —
        use with ``async with``. This single connection carries JSON events
        only (session lifecycle, transcripts, function-call events, no audio)
        and is used for both receiving server events and sending client
        events back (e.g. function call results), since sending on a separate
        connection wouldn't be attributed to the same monitoring session.
        """
        url = f"wss://api.openai.com/v1/realtime?call_id={call_id}"
        return websockets.connect(url, additional_headers=self._headers())

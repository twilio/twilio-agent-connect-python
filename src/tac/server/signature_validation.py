"""Webhook signature validation for Twilio webhooks.

This module provides utilities for validating Twilio webhook signatures
in FastAPI applications. It handles proxy headers (X-Forwarded-Proto,
X-Forwarded-Host) for environments like ngrok.

Requires: pip install tac[server]
"""

import logging
from collections.abc import Awaitable, Callable, Mapping
from urllib.parse import parse_qs

from fastapi import HTTPException, Request, WebSocket, WebSocketDisconnect
from twilio.request_validator import RequestValidator

logger = logging.getLogger(__name__)


def validate_twilio_webhook(
    request: Request,
    auth_token: str,
    body: str | Mapping[str, str],
) -> bool:
    """Validate a Twilio webhook signature.

    Verifies the X-Twilio-Signature header matches the expected signature for the
    request URL and body. Handles proxy headers (X-Forwarded-Proto, X-Forwarded-Host)
    for environments like ngrok.

    Args:
        request: FastAPI Request object containing headers and URL info.
        auth_token: Twilio Auth Token used for signature validation.
        body: Request body - pass str for JSON bodies (SMS webhooks from Conversation Orchestrator,
              where signature is computed with empty POST params), or pass a mapping
              for form-encoded bodies (Voice webhooks, where params are included).
              Accepts dict, FormData, or any Mapping[str, str].

    Returns:
        True if signature is valid, False otherwise.
    """
    signature = request.headers.get("X-Twilio-Signature")
    if not signature:
        return False

    url = _build_url(request)

    validator = RequestValidator(auth_token)

    # For JSON bodies (string), Twilio signs with URL only (empty params).
    # For form-encoded bodies (mapping), params are included in signature.
    params = dict(body) if isinstance(body, Mapping) else {}
    result: bool = validator.validate(url, params, signature)
    return result


def build_http_signature_dependency(
    auth_token: str,
) -> Callable[..., Awaitable[None]]:
    """Build a FastAPI dependency that validates Twilio webhook signatures on HTTP POST routes.

    Usage:
        sig_dep = build_http_signature_dependency(auth_token)

        @app.post("/webhook", dependencies=[Depends(sig_dep)])
        async def webhook(request: Request) -> JSONResponse:
            ...
    """

    async def _validate_http_signature(request: Request) -> None:
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            form_data = await request.form()
            body: str | Mapping[str, str] = {k: str(v) for k, v in form_data.items()}
        else:
            raw = await request.body()
            try:
                body = raw.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(status_code=403, detail="Invalid Twilio signature") from None

        if not validate_twilio_webhook(request, auth_token, body):
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    return _validate_http_signature


def build_websocket_signature_dependency(
    auth_token: str,
) -> Callable[..., Awaitable[None]]:
    """Build a FastAPI dependency that validates Twilio signatures on WebSocket upgrade requests.

    Validates the signature before the WebSocket is accepted.
    Closes with code 1008 (Policy Violation) on invalid signature.

    Usage:
        ws_dep = build_websocket_signature_dependency(auth_token)

        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket, _: None = Depends(ws_dep)) -> None:
            ...
    """

    async def _validate_ws_signature(websocket: WebSocket) -> None:
        signature = websocket.headers.get("x-twilio-signature")
        if not signature:
            logger.warning(
                "WebSocket missing x-twilio-signature header. Available headers: %s",
                list(websocket.headers.keys()),
            )
            await websocket.close(code=1008, reason="Missing Twilio signature")
            raise WebSocketDisconnect(code=1008)

        url, params = _build_websocket_url_and_params(websocket)
        validator = RequestValidator(auth_token)
        if not validator.validate(url, params, signature):
            logger.warning(
                "WebSocket signature validation failed. URL used: %s",
                url,
            )
            await websocket.close(code=1008, reason="Invalid Twilio signature")
            raise WebSocketDisconnect(code=1008)

    return _validate_ws_signature


def _build_url(request: Request) -> str:
    """Build the full URL from request, handling proxy headers.

    When behind a proxy (like ngrok), the request URL may have incorrect scheme
    or host. This function checks X-Forwarded-Proto and X-Forwarded-Host headers
    to reconstruct the original URL that Twilio signed.

    Handles comma-separated values in X-Forwarded-* headers when requests
    traverse multiple proxies (e.g., CloudFlare -> ALB -> k8s ingress).
    """
    proto_header = request.headers.get("X-Forwarded-Proto") or request.url.scheme
    proto = proto_header.split(",")[0].strip()

    host_header = (
        request.headers.get("X-Forwarded-Host") or request.headers.get("Host") or request.url.netloc
    )
    host = host_header.split(",")[0].strip()

    path = request.url.path
    query = request.url.query

    url = f"{proto}://{host}{path}"
    if query:
        url = f"{url}?{query}"

    return url


def _build_websocket_url_and_params(websocket: WebSocket) -> tuple[str, dict[str, str]]:
    """Build the validation URL and params from a WebSocket request.

    Twilio signs the wss:// URL it connects to. Query parameters are passed
    separately in the params dict for signature computation (not appended to the URL).
    When behind a proxy (like ngrok), X-Forwarded-Proto reports 'https' but Twilio
    signed 'wss://', so we convert https->wss and http->ws.
    """
    proto_header = websocket.headers.get("X-Forwarded-Proto")
    if proto_header:
        raw_proto = proto_header.split(",")[0].strip()
        proto = _http_scheme_to_ws(raw_proto)
    else:
        proto = websocket.url.scheme

    host_header = (
        websocket.headers.get("X-Forwarded-Host")
        or websocket.headers.get("Host")
        or websocket.url.netloc
    )
    host = host_header.split(",")[0].strip()

    path = websocket.url.path
    url = f"{proto}://{host}{path}"

    # Query params are included as params dict, not in the URL
    params: dict[str, str] = {}
    query = websocket.url.query
    if query:
        parsed = parse_qs(query)
        params = {k: v[0] for k, v in parsed.items()}

    return url, params


def _http_scheme_to_ws(scheme: str) -> str:
    """Convert HTTP scheme to WebSocket equivalent for URL reconstruction."""
    mapping = {"https": "wss", "http": "ws"}
    return mapping.get(scheme, scheme)

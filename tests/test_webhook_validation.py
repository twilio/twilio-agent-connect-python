"""Tests for Twilio webhook signature validation."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, WebSocketDisconnect
from twilio.request_validator import RequestValidator

from tac.server.signature_validation import (
    _build_url,
    _build_websocket_url_and_params,
    _http_scheme_to_ws,
    build_http_signature_dependency,
    build_websocket_signature_dependency,
    validate_twilio_webhook,
)


class TestValidateTwilioWebhook:
    """Test validate_twilio_webhook function."""

    def test_valid_form_body_signature(self) -> None:
        """Test validation passes with correct signature for form-encoded body."""
        auth_token = "test_auth_token"
        body = {"From": "+15551234567", "To": "+15559876543", "CallSid": "CA123"}

        validator = RequestValidator(auth_token)
        url = "https://example.com/twiml"
        signature = validator.compute_signature(url, body)

        request = MagicMock()
        request.headers = {
            "X-Twilio-Signature": signature,
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "example.com",
        }
        request.url.path = "/twiml"
        request.url.query = ""

        assert validate_twilio_webhook(request, auth_token, body) is True

    def test_invalid_signature_returns_false(self) -> None:
        """Test validation fails with incorrect signature."""
        auth_token = "test_auth_token"
        body = {"From": "+15551234567", "Body": "Hello"}

        request = MagicMock()
        request.headers = {
            "X-Twilio-Signature": "invalid_signature",
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "example.com",
        }
        request.url.path = "/webhook"
        request.url.query = ""

        assert validate_twilio_webhook(request, auth_token, body) is False

    def test_missing_signature_returns_false(self) -> None:
        """Test validation fails when X-Twilio-Signature header is missing."""
        auth_token = "test_auth_token"
        body = {"From": "+15551234567", "Body": "Hello"}

        request = MagicMock()
        request.headers = {}
        request.url.path = "/webhook"
        request.url.query = ""
        request.url.scheme = "https"

        assert validate_twilio_webhook(request, auth_token, body) is False

    def test_valid_string_body_signature(self) -> None:
        """Test validation passes with correct signature for JSON/string body."""
        auth_token = "test_auth_token"
        body_str = '{"event": "onMessageAdded", "conversationSid": "CH123"}'

        validator = RequestValidator(auth_token)
        url = "https://example.com/webhook"
        signature = validator.compute_signature(url, {})

        request = MagicMock()
        request.headers = {
            "X-Twilio-Signature": signature,
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "example.com",
        }
        request.url.path = "/webhook"
        request.url.query = ""

        assert validate_twilio_webhook(request, auth_token, body_str) is True

    def test_invalid_string_body_signature(self) -> None:
        """Test validation fails with incorrect signature for JSON/string body."""
        auth_token = "test_auth_token"
        body_str = '{"event": "onMessageAdded"}'

        request = MagicMock()
        request.headers = {
            "X-Twilio-Signature": "invalid_signature",
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "example.com",
        }
        request.url.path = "/webhook"
        request.url.query = ""

        assert validate_twilio_webhook(request, auth_token, body_str) is False


class TestBuildUrl:
    """Test _build_url function."""

    def test_uses_forwarded_headers(self) -> None:
        """Test that X-Forwarded-* headers are used."""
        request = MagicMock()
        request.headers = {
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "example.com",
        }
        request.url.path = "/webhook"
        request.url.query = ""

        url = _build_url(request)
        assert url == "https://example.com/webhook"

    def test_falls_back_to_request_url_scheme(self) -> None:
        """Test fallback to request.url.scheme when X-Forwarded-Proto is missing."""
        request = MagicMock()
        request.headers = {"Host": "example.com"}
        request.url.path = "/webhook"
        request.url.query = ""
        request.url.scheme = "http"

        url = _build_url(request)
        assert url == "http://example.com/webhook"

    def test_handles_comma_separated_headers(self) -> None:
        """Test handling of comma-separated X-Forwarded-* headers from multiple proxies."""
        request = MagicMock()
        request.headers = {
            "X-Forwarded-Proto": "https, http, http",
            "X-Forwarded-Host": "public.example.com, internal-alb.local, 10.0.0.1",
        }
        request.url.path = "/webhook"
        request.url.query = ""

        url = _build_url(request)
        assert url == "https://public.example.com/webhook"

    def test_includes_query_string(self) -> None:
        """Test that query string is included in URL."""
        request = MagicMock()
        request.headers = {
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "example.com",
        }
        request.url.path = "/webhook"
        request.url.query = "foo=bar&baz=qux"

        url = _build_url(request)
        assert url == "https://example.com/webhook?foo=bar&baz=qux"

    def test_falls_back_to_request_url_netloc(self) -> None:
        """Test fallback to request.url.netloc when both Host headers are missing."""
        request = MagicMock()
        request.headers = {}
        request.url.path = "/webhook"
        request.url.query = ""
        request.url.scheme = "https"
        request.url.netloc = "localhost:8000"

        url = _build_url(request)
        assert url == "https://localhost:8000/webhook"


class TestBuildWebsocketUrlAndParams:
    """Test _build_websocket_url_and_params function."""

    def test_preserves_wss_scheme(self) -> None:
        websocket = MagicMock()
        websocket.headers = {}
        websocket.url.path = "/ws"
        websocket.url.query = ""
        websocket.url.scheme = "wss"
        websocket.url.netloc = "example.com"

        url, params = _build_websocket_url_and_params(websocket)
        assert url == "wss://example.com/ws"
        assert params == {}

    def test_preserves_ws_scheme(self) -> None:
        websocket = MagicMock()
        websocket.headers = {}
        websocket.url.path = "/ws"
        websocket.url.query = ""
        websocket.url.scheme = "ws"
        websocket.url.netloc = "localhost:8000"

        url, params = _build_websocket_url_and_params(websocket)
        assert url == "ws://localhost:8000/ws"
        assert params == {}

    def test_converts_forwarded_https_to_wss(self) -> None:
        websocket = MagicMock()
        websocket.headers = {
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "public.example.com",
        }
        websocket.url.path = "/ws"
        websocket.url.query = ""
        websocket.url.scheme = "ws"

        url, params = _build_websocket_url_and_params(websocket)
        assert url == "wss://public.example.com/ws"
        assert params == {}

    def test_query_params_returned_separately(self) -> None:
        websocket = MagicMock()
        websocket.headers = {"Host": "example.com"}
        websocket.url.path = "/ws"
        websocket.url.query = "token=abc123&foo=bar"
        websocket.url.scheme = "wss"

        url, params = _build_websocket_url_and_params(websocket)
        assert url == "wss://example.com/ws"
        assert params == {"token": "abc123", "foo": "bar"}

    def test_handles_comma_separated_forwarded_headers(self) -> None:
        websocket = MagicMock()
        websocket.headers = {
            "X-Forwarded-Proto": "https, http",
            "X-Forwarded-Host": "public.example.com, internal.local",
        }
        websocket.url.path = "/ws"
        websocket.url.query = ""

        url, params = _build_websocket_url_and_params(websocket)
        assert url == "wss://public.example.com/ws"
        assert params == {}


class TestHttpSchemeToWs:
    """Test _http_scheme_to_ws helper."""

    def test_https_to_wss(self) -> None:
        assert _http_scheme_to_ws("https") == "wss"

    def test_http_to_ws(self) -> None:
        assert _http_scheme_to_ws("http") == "ws"

    def test_passthrough_wss(self) -> None:
        assert _http_scheme_to_ws("wss") == "wss"

    def test_passthrough_ws(self) -> None:
        assert _http_scheme_to_ws("ws") == "ws"


class TestBuildHttpSignatureDependency:
    """Test build_http_signature_dependency."""

    @pytest.mark.asyncio
    async def test_valid_json_signature_passes(self) -> None:
        auth_token = "test_token"
        url = "http://testserver/webhook"
        validator = RequestValidator(auth_token)
        signature = validator.compute_signature(url, {})

        dep = build_http_signature_dependency(auth_token)

        request = AsyncMock()
        request.headers = {"X-Twilio-Signature": signature, "content-type": "application/json"}
        request.url.path = "/webhook"
        request.url.query = ""
        request.url.scheme = "http"
        request.url.netloc = "testserver"
        request.body.return_value = b'{"event": "test"}'

        await dep(request)

    @pytest.mark.asyncio
    async def test_valid_form_signature_passes(self) -> None:
        auth_token = "test_token"
        form_data = {"CallSid": "CA123", "From": "+15551234567"}
        url = "http://testserver/twiml"
        validator = RequestValidator(auth_token)
        signature = validator.compute_signature(url, form_data)

        dep = build_http_signature_dependency(auth_token)

        request = AsyncMock()
        request.headers = {
            "X-Twilio-Signature": signature,
            "content-type": "application/x-www-form-urlencoded",
        }
        request.url.path = "/twiml"
        request.url.query = ""
        request.url.scheme = "http"
        request.url.netloc = "testserver"
        request.form.return_value = form_data

        await dep(request)

    @pytest.mark.asyncio
    async def test_missing_signature_raises_403(self) -> None:
        dep = build_http_signature_dependency("test_token")

        request = AsyncMock()
        request.headers = {"content-type": "application/json"}
        request.url.path = "/webhook"
        request.url.query = ""
        request.url.scheme = "http"
        request.url.netloc = "testserver"
        request.body.return_value = b'{"event": "test"}'

        with pytest.raises(HTTPException) as exc_info:
            await dep(request)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_invalid_signature_raises_403(self) -> None:
        dep = build_http_signature_dependency("test_token")

        request = AsyncMock()
        request.headers = {
            "X-Twilio-Signature": "bad_signature",
            "content-type": "application/json",
        }
        request.url.path = "/webhook"
        request.url.query = ""
        request.url.scheme = "http"
        request.url.netloc = "testserver"
        request.body.return_value = b'{"event": "test"}'

        with pytest.raises(HTTPException) as exc_info:
            await dep(request)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_malformed_utf8_body_raises_403(self) -> None:
        dep = build_http_signature_dependency("test_token")

        request = AsyncMock()
        request.headers = {
            "X-Twilio-Signature": "some_signature",
            "content-type": "application/json",
        }
        request.url.path = "/webhook"
        request.url.query = ""
        request.url.scheme = "http"
        request.url.netloc = "testserver"
        request.body.return_value = b"\xff\xfe invalid utf8"

        with pytest.raises(HTTPException) as exc_info:
            await dep(request)
        assert exc_info.value.status_code == 403


class TestBuildWebsocketSignatureDependency:
    """Test build_websocket_signature_dependency."""

    @pytest.mark.asyncio
    async def test_valid_signature_passes(self) -> None:
        auth_token = "test_token"
        # Twilio signs the wss:// URL with query params as params dict
        url = "wss://testserver/ws"
        validator = RequestValidator(auth_token)
        signature = validator.compute_signature(url, {})

        dep = build_websocket_signature_dependency(auth_token)

        websocket = AsyncMock()
        websocket.headers = {"x-twilio-signature": signature, "Host": "testserver"}
        websocket.url.path = "/ws"
        websocket.url.query = ""
        websocket.url.scheme = "wss"
        websocket.url.netloc = "testserver"

        await dep(websocket)
        websocket.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_signature_with_query_params(self) -> None:
        auth_token = "test_token"
        url = "wss://testserver/ws"
        params = {"token": "abc123", "foo": "bar"}
        validator = RequestValidator(auth_token)
        signature = validator.compute_signature(url, params)

        dep = build_websocket_signature_dependency(auth_token)

        websocket = AsyncMock()
        websocket.headers = {"x-twilio-signature": signature, "Host": "testserver"}
        websocket.url.path = "/ws"
        websocket.url.query = "token=abc123&foo=bar"
        websocket.url.scheme = "wss"
        websocket.url.netloc = "testserver"

        await dep(websocket)
        websocket.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_signature_closes_with_1008(self) -> None:
        dep = build_websocket_signature_dependency("test_token")

        websocket = AsyncMock()
        websocket.headers = {}
        websocket.url.path = "/ws"
        websocket.url.query = ""
        websocket.url.scheme = "ws"
        websocket.url.netloc = "testserver"

        with pytest.raises(WebSocketDisconnect) as exc_info:
            await dep(websocket)
        assert exc_info.value.code == 1008
        websocket.close.assert_called_once_with(code=1008, reason="Missing Twilio signature")

    @pytest.mark.asyncio
    async def test_invalid_signature_closes_with_1008(self) -> None:
        dep = build_websocket_signature_dependency("test_token")

        websocket = AsyncMock()
        websocket.headers = {"x-twilio-signature": "bad_signature", "Host": "testserver"}
        websocket.url.path = "/ws"
        websocket.url.query = ""
        websocket.url.scheme = "wss"
        websocket.url.netloc = "testserver"

        with pytest.raises(WebSocketDisconnect) as exc_info:
            await dep(websocket)
        assert exc_info.value.code == 1008
        websocket.close.assert_called_once_with(code=1008, reason="Invalid Twilio signature")

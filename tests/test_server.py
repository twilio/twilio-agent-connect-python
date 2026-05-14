"""Tests for TACFastAPIServer module."""

import pytest
from twilio.request_validator import RequestValidator

from tac import TAC
from tac.channels.websocket_protocol import WebSocketDisconnectError, WebSocketProtocol
from tac.server.config import TACServerConfig

AUTH_TOKEN = "test_token_123"


def get_test_config() -> dict:
    """Get a valid test configuration."""
    return {
        "account_sid": "ACtest123",
        "auth_token": AUTH_TOKEN,
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": "conv_configuration_test123",
        "phone_number": "+15551234567",
    }


def compute_signature(url: str, params: dict[str, str] | None = None) -> str:
    """Compute a valid Twilio signature for test requests."""
    return RequestValidator(AUTH_TOKEN).compute_signature(url, params or {})


class TestTACServerConfig:
    """Test TACServerConfig."""

    def test_defaults(self) -> None:
        config = TACServerConfig(public_domain="example.ngrok.io")
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.public_domain == "example.ngrok.io"
        assert config.welcome_greeting is None
        assert config.conversation_webhook_path == "/webhook"
        assert config.twiml_path == "/twiml"
        assert config.websocket_path == "/ws"
        assert config.cintel_webhook_path is None

    def test_custom_paths(self) -> None:
        config = TACServerConfig(
            public_domain="my.domain.com",
            host="127.0.0.1",
            port=3000,
            conversation_webhook_path="/conversations",
            twiml_path="/voice/twiml",
            websocket_path="/voice/ws",
            cintel_webhook_path="/ci",
        )
        assert config.host == "127.0.0.1"
        assert config.port == 3000
        assert config.conversation_webhook_path == "/conversations"
        assert config.twiml_path == "/voice/twiml"
        assert config.websocket_path == "/voice/ws"
        assert config.cintel_webhook_path == "/ci"

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TWILIO_VOICE_PUBLIC_DOMAIN", "my.ngrok.io")
        monkeypatch.setenv("TWILIO_SERVER_HOST", "127.0.0.1")
        monkeypatch.setenv("TWILIO_SERVER_PORT", "3000")
        config = TACServerConfig.from_env()
        assert config.public_domain == "my.ngrok.io"
        assert config.host == "127.0.0.1"
        assert config.port == 3000

    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TWILIO_VOICE_PUBLIC_DOMAIN", raising=False)
        monkeypatch.delenv("TWILIO_SERVER_HOST", raising=False)
        monkeypatch.delenv("TWILIO_SERVER_PORT", raising=False)
        config = TACServerConfig.from_env()
        assert config.public_domain == ""
        assert config.host == "0.0.0.0"
        assert config.port == 8000


class TestTACConfigStudioHandoffFlowSid:
    """Test TACConfig studio_handoff_flow_sid field."""

    def test_studio_handoff_flow_sid_default_none(self) -> None:
        tac = TAC(get_test_config())
        assert tac.config.studio_handoff_flow_sid is None

    def test_studio_handoff_flow_sid_accepts_value(self) -> None:
        flow_sid = "FW" + "a" * 32
        tac = TAC({**get_test_config(), "studio_handoff_flow_sid": flow_sid})
        assert tac.config.studio_handoff_flow_sid == flow_sid


class TestWebSocketDisconnectError:
    """Test WebSocketDisconnectError."""

    def test_is_exception(self) -> None:
        err = WebSocketDisconnectError("test")
        assert isinstance(err, Exception)
        assert str(err) == "test"


class TestFastAPIWebSocketAdapter:
    """Test FastAPIWebSocketAdapter wraps FastAPI WebSocket correctly."""

    def test_import_and_register(self) -> None:
        from tac.server import FastAPIWebSocketAdapter

        assert FastAPIWebSocketAdapter is not None

    @pytest.mark.asyncio
    async def test_accept(self) -> None:
        from unittest.mock import AsyncMock

        from fastapi import WebSocket

        from tac.server import FastAPIWebSocketAdapter

        mock_ws = AsyncMock(spec=WebSocket)
        adapter = FastAPIWebSocketAdapter(mock_ws)
        await adapter.accept()
        mock_ws.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_json(self) -> None:
        from unittest.mock import AsyncMock

        from fastapi import WebSocket

        from tac.server import FastAPIWebSocketAdapter

        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.receive_json.return_value = {"type": "setup"}
        adapter = FastAPIWebSocketAdapter(mock_ws)
        result = await adapter.receive_json()
        assert result == {"type": "setup"}

    @pytest.mark.asyncio
    async def test_receive_json_disconnect(self) -> None:
        from unittest.mock import AsyncMock

        from fastapi import WebSocket, WebSocketDisconnect

        from tac.server import FastAPIWebSocketAdapter

        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.receive_json.side_effect = WebSocketDisconnect()
        adapter = FastAPIWebSocketAdapter(mock_ws)
        with pytest.raises(WebSocketDisconnectError):
            await adapter.receive_json()

    @pytest.mark.asyncio
    async def test_send_text(self) -> None:
        from unittest.mock import AsyncMock

        from fastapi import WebSocket

        from tac.server import FastAPIWebSocketAdapter

        mock_ws = AsyncMock(spec=WebSocket)
        adapter = FastAPIWebSocketAdapter(mock_ws)
        await adapter.send_text("hello")
        mock_ws.send_text.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_send_text_disconnect(self) -> None:
        from unittest.mock import AsyncMock

        from fastapi import WebSocket, WebSocketDisconnect

        from tac.server import FastAPIWebSocketAdapter

        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.send_text.side_effect = WebSocketDisconnect()
        adapter = FastAPIWebSocketAdapter(mock_ws)
        with pytest.raises(WebSocketDisconnectError):
            await adapter.send_text("hello")

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        from unittest.mock import AsyncMock

        from fastapi import WebSocket

        from tac.server import FastAPIWebSocketAdapter

        mock_ws = AsyncMock(spec=WebSocket)
        adapter = FastAPIWebSocketAdapter(mock_ws)
        await adapter.close()
        mock_ws.close.assert_called_once()

    def test_isinstance_check(self) -> None:
        from unittest.mock import AsyncMock

        from fastapi import WebSocket

        from tac.server import FastAPIWebSocketAdapter

        mock_ws = AsyncMock(spec=WebSocket)
        adapter = FastAPIWebSocketAdapter(mock_ws)
        assert isinstance(adapter, WebSocketProtocol)


class TestTACFastAPIServer:
    """Test TACFastAPIServer route creation."""

    @pytest.mark.asyncio
    async def test_messaging_webhook_fanout(self) -> None:
        """Messaging webhook fans out to all configured channels with idempotency token."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from httpx import ASGITransport, AsyncClient

        from tac.channels import ChatChannel, SMSChannel
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        sms = SMSChannel(tac)
        chat = ChatChannel(tac)

        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
            messaging_channels=[sms, chat],
        )
        app = server.app

        with (
            patch.object(sms, "process_webhook", new_callable=AsyncMock) as mock_sms,
            patch.object(chat, "process_webhook", new_callable=AsyncMock) as mock_chat,
        ):
            url = "http://test/webhook"
            signature = compute_signature(url)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/webhook",
                    json={"eventType": "COMMUNICATION_CREATED", "data": {}},
                    headers={
                        "i-twilio-idempotency-token": "tok-123",
                        "X-Twilio-Signature": signature,
                    },
                )

            assert resp.status_code == 200
            # Yield control so fire-and-forget tasks (mocked, instant) complete
            await asyncio.sleep(0)

            mock_sms.assert_called_once()
            mock_chat.assert_called_once()
            # Both receive the same webhook data and idempotency token
            assert mock_sms.call_args[0][0] == {"eventType": "COMMUNICATION_CREATED", "data": {}}
            assert mock_sms.call_args[0][1] == "tok-123"
            assert mock_chat.call_args[0][1] == "tok-123"

    @pytest.mark.asyncio
    async def test_conversation_webhook_handles_channel_errors(self) -> None:
        """Webhook processing errors in one channel don't affect other channels."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from httpx import ASGITransport, AsyncClient

        from tac.channels import ChatChannel, SMSChannel
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        sms = SMSChannel(tac)
        chat = ChatChannel(tac)

        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
            messaging_channels=[sms, chat],
        )
        app = server.app

        # Make SMS channel raise an error
        with (
            patch.object(
                sms,
                "process_webhook",
                new_callable=AsyncMock,
                side_effect=Exception("SMS processing failed"),
            ) as mock_sms,
            patch.object(chat, "process_webhook", new_callable=AsyncMock) as mock_chat,
        ):
            url = "http://test/webhook"
            signature = compute_signature(url)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/webhook",
                    json={"eventType": "COMMUNICATION_CREATED", "data": {}},
                    headers={
                        "i-twilio-idempotency-token": "tok-456",
                        "X-Twilio-Signature": signature,
                    },
                )

            # Webhook should still return 200 (fire-and-forget pattern)
            assert resp.status_code == 200

            # Yield control so background tasks can run
            await asyncio.sleep(0)

            # Both channels should have been called despite SMS error
            mock_sms.assert_called_once()
            mock_chat.assert_called_once()

    def test_create_app_voice_only(self) -> None:
        from tac.channels.voice import VoiceChannel
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        vc = VoiceChannel(tac)
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
            voice_channel=vc,
        )
        app = server.app

        # Check that voice routes are registered
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/twiml" in route_paths
        assert "/ws" in route_paths
        assert "/webhook" in route_paths  # conversation webhook path (all channels)

    def test_create_app_messaging_only(self) -> None:
        from tac.channels import SMSChannel
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        sms = SMSChannel(tac)
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
            messaging_channels=[sms],
        )
        app = server.app

        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/webhook" in route_paths
        # No voice routes
        assert "/twiml" not in route_paths
        assert "/ws" not in route_paths

    def test_create_app_with_cintel(self) -> None:
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(
                public_domain="test.ngrok.io", cintel_webhook_path="/ci-webhook"
            ),
        )
        app = server.app

        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/ci-webhook" in route_paths

    def test_create_app_custom_paths(self) -> None:
        from tac.channels import SMSChannel
        from tac.channels.voice import VoiceChannel
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(
                public_domain="test.ngrok.io",
                conversation_webhook_path="/conversations",
                twiml_path="/voice/twiml",
                websocket_path="/voice/ws",
            ),
            voice_channel=VoiceChannel(tac),
            messaging_channels=[SMSChannel(tac)],
        )
        app = server.app

        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/conversations" in route_paths
        assert "/voice/twiml" in route_paths
        assert "/voice/ws" in route_paths

    def test_custom_app_is_used(self) -> None:
        """User-supplied FastAPI instance is used directly and metadata preserved."""
        from fastapi import FastAPI

        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        custom_app = FastAPI(title="My Custom Service", version="9.9.9")
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
            app=custom_app,
        )
        assert server.app is custom_app
        assert server.app.title == "My Custom Service"
        assert server.app.version == "9.9.9"

    def test_custom_app_has_tac_routes(self) -> None:
        """TAC routes are registered onto a user-supplied app."""
        from fastapi import FastAPI

        from tac.channels import SMSChannel
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        custom_app = FastAPI()
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
            messaging_channels=[SMSChannel(tac)],
            app=custom_app,
        )
        route_paths = [r.path for r in server.app.routes if hasattr(r, "path")]
        assert "/webhook" in route_paths

    def test_default_app_created(self) -> None:
        """Default FastAPI app is created with TAC Server title when no app is passed."""
        from fastapi import FastAPI

        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
        )
        assert isinstance(server.app, FastAPI)
        assert server.app.title == "TAC Server"

    @pytest.mark.asyncio
    async def test_can_add_custom_route_post_construction(self) -> None:
        """Users can add routes to server.app after construction."""
        from httpx import ASGITransport, AsyncClient

        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
        )

        @server.app.get("/health")
        async def health() -> dict:
            return {"status": "ok"}

        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_can_add_middleware_post_construction(self) -> None:
        """Users can add middleware to server.app after construction."""
        from httpx import ASGITransport, AsyncClient
        from starlette.middleware.base import BaseHTTPMiddleware

        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
        )

        class HeaderMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
                resp = await call_next(request)
                resp.headers["X-Test"] = "yes"
                return resp

        server.app.add_middleware(HeaderMiddleware)

        @server.app.get("/ping")
        async def ping() -> dict:
            return {"ok": True}

        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/ping")
        assert resp.headers.get("X-Test") == "yes"

    @pytest.mark.asyncio
    async def test_can_add_exception_handler(self) -> None:
        """Users can register exception handlers on server.app."""
        from fastapi import Request
        from fastapi.responses import JSONResponse
        from httpx import ASGITransport, AsyncClient

        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
        )

        class MyError(Exception):
            pass

        @server.app.exception_handler(MyError)
        async def handler(request: Request, exc: MyError) -> JSONResponse:
            return JSONResponse({"handled": True}, status_code=418)

        @server.app.get("/boom")
        async def boom() -> None:
            raise MyError()

        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/boom")
        assert resp.status_code == 418
        assert resp.json() == {"handled": True}

    def test_routes_registered_exactly_once(self) -> None:
        """Routes are not double-registered after refactor to eager registration."""
        from tac.channels import SMSChannel
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
            messaging_channels=[SMSChannel(tac)],
        )
        route_paths = [r.path for r in server.app.routes if hasattr(r, "path")]
        assert route_paths.count("/webhook") == 1


class TestTwiMLConnectAction:
    """TwiML <Connect action=...> routes to Studio when the Flow SID is configured."""

    def _build_server(self, **tac_overrides: object) -> object:
        from fastapi.testclient import TestClient

        from tac.channels.voice import VoiceChannel
        from tac.server import TACFastAPIServer

        tac = TAC({**get_test_config(), **tac_overrides})
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
            voice_channel=VoiceChannel(tac),
        )
        return TestClient(server.app)

    def _twiml_signature(self) -> str:
        """Compute signature for form-encoded POST to /twiml with empty body."""
        return compute_signature("http://testserver/twiml")

    def test_connect_action_uses_studio_webhook_when_flow_sid_set(self) -> None:
        flow_sid = "FW" + "a" * 32
        client = self._build_server(studio_handoff_flow_sid=flow_sid)
        resp = client.post(  # type: ignore[attr-defined]
            "/twiml",
            headers={"X-Twilio-Signature": self._twiml_signature()},
        )

        assert resp.status_code == 200
        expected = (
            f'action="https://webhooks.twilio.com/v1/Accounts/ACtest123'
            f'/Flows/{flow_sid}?Trigger=incomingCall"'
        )
        assert expected in resp.text

    def test_connect_action_uses_cleanup_url_when_no_handoff_flow(self) -> None:
        """Without Studio handoff, action_url falls back to the server's
        session-cleanup URL — no-op in orchestrated mode, drives cleanup in
        relay-only mode."""
        client = self._build_server()  # no studio_handoff_flow_sid
        resp = client.post(  # type: ignore[attr-defined]
            "/twiml",
            headers={"X-Twilio-Signature": self._twiml_signature()},
        )

        assert resp.status_code == 200
        assert 'action="https://test.ngrok.io/conversation-relay-callback"' in resp.text


class TestSignatureValidation:
    """Test that webhook signature validation is enforced on all TAC routes."""

    @pytest.mark.asyncio
    async def test_webhook_rejects_missing_signature(self) -> None:
        from httpx import ASGITransport, AsyncClient

        from tac.channels import SMSChannel
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
            messaging_channels=[SMSChannel(tac)],
        )
        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/webhook", json={"eventType": "test"})
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_webhook_rejects_invalid_signature(self) -> None:
        from httpx import ASGITransport, AsyncClient

        from tac.channels import SMSChannel
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
            messaging_channels=[SMSChannel(tac)],
        )
        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/webhook",
                json={"eventType": "test"},
                headers={"X-Twilio-Signature": "invalid"},
            )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_webhook_accepts_valid_signature(self) -> None:
        from unittest.mock import AsyncMock, patch

        from httpx import ASGITransport, AsyncClient

        from tac.channels import SMSChannel
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        sms = SMSChannel(tac)
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
            messaging_channels=[sms],
        )
        with patch.object(sms, "process_webhook", new_callable=AsyncMock):
            url = "http://test/webhook"
            signature = compute_signature(url)
            transport = ASGITransport(app=server.app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/webhook",
                    json={"eventType": "test"},
                    headers={"X-Twilio-Signature": signature},
                )
        assert resp.status_code == 200

    def test_twiml_rejects_missing_signature(self) -> None:
        from fastapi.testclient import TestClient

        from tac.channels.voice import VoiceChannel
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
            voice_channel=VoiceChannel(tac),
        )
        client = TestClient(server.app)
        resp = client.post("/twiml")
        assert resp.status_code == 403

    def test_twiml_accepts_valid_form_signature(self) -> None:
        from fastapi.testclient import TestClient

        from tac.channels.voice import VoiceChannel
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
            voice_channel=VoiceChannel(tac),
        )
        client = TestClient(server.app)
        form_data = {"CallSid": "CA123", "From": "+15551234567"}
        signature = compute_signature("http://testserver/twiml", form_data)
        resp = client.post(
            "/twiml",
            data=form_data,
            headers={"X-Twilio-Signature": signature},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_cintel_webhook_rejects_missing_signature(self) -> None:
        from httpx import ASGITransport, AsyncClient

        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(
                public_domain="test.ngrok.io", cintel_webhook_path="/ci-webhook"
            ),
        )
        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/ci-webhook", json={"event": "test"})
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_custom_routes_not_affected(self) -> None:
        from httpx import ASGITransport, AsyncClient

        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
        )

        @server.app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_websocket_accepts_valid_signature(self) -> None:
        from fastapi import Depends, FastAPI, WebSocket
        from fastapi.testclient import TestClient

        from tac.server.signature_validation import build_websocket_signature_dependency

        app = FastAPI()
        ws_sig = build_websocket_signature_dependency(AUTH_TOKEN)
        connected = False

        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket, _: None = Depends(ws_sig)) -> None:
            nonlocal connected
            await websocket.accept()
            connected = True
            await websocket.close()

        signature = compute_signature("ws://testserver/ws")
        client = TestClient(app)
        with client.websocket_connect("/ws", headers={"x-twilio-signature": signature}):
            pass
        assert connected

    def test_websocket_rejects_missing_signature(self) -> None:
        from fastapi import WebSocketDisconnect
        from fastapi.testclient import TestClient

        from tac.channels.voice import VoiceChannel
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
            voice_channel=VoiceChannel(tac),
        )
        client = TestClient(server.app)
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws"):
                pass

    def test_websocket_rejects_invalid_signature(self) -> None:
        from fastapi import WebSocketDisconnect
        from fastapi.testclient import TestClient

        from tac.channels.voice import VoiceChannel
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
            voice_channel=VoiceChannel(tac),
        )
        client = TestClient(server.app)
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws", headers={"X-Twilio-Signature": "invalid"}):
                pass


class TestTwiMLCustomizerEndToEnd:
    """Smoke test: server parses Twilio form and channel customizer shapes TwiML."""

    def test_customizer_on_voice_channel_receives_parsed_context_and_overrides_twiml(
        self,
    ) -> None:
        from fastapi.testclient import TestClient

        from tac.channels.voice import VoiceChannel, VoiceChannelConfig
        from tac.models.voice import TwiMLOptions, TwiMLRequest
        from tac.server import TACFastAPIServer

        captured: dict[str, TwiMLRequest] = {}

        async def customizer(ctx: TwiMLRequest) -> TwiMLOptions:
            captured["ctx"] = ctx
            return TwiMLOptions(voice="en-US-Journey-D", language="en-US")

        tac = TAC(get_test_config())
        vc = VoiceChannel(tac, config=VoiceChannelConfig(customize_twiml_options=customizer))
        server = TACFastAPIServer(
            tac=tac,
            config=TACServerConfig(public_domain="test.ngrok.io"),
            voice_channel=vc,
        )
        client = TestClient(server.app)
        form_data = {
            "From": "+14155551234",
            "To": "+15551234567",
            "CallerCountry": "US",
            "ApiVersion": "2010-04-01",
        }
        signature = compute_signature("http://testserver/twiml", form_data)
        resp = client.post(
            "/twiml",
            data=form_data,
            headers={"X-Twilio-Signature": signature},
        )
        assert resp.status_code == 200
        ctx = captured["ctx"]
        assert ctx.from_number == "+14155551234"
        assert ctx.caller_country == "US"
        assert ctx.extra == {"ApiVersion": "2010-04-01"}
        body = resp.text
        assert 'voice="en-US-Journey-D"' in body
        assert 'language="en-US"' in body
        assert 'url="wss://test.ngrok.io/ws"' in body


class TestDeprecatedWelcomeGreetingForwarding:
    """TACServerConfig.welcome_greeting is deprecated; verify it still reaches the channel."""

    def test_deprecated_field_emits_warning(self) -> None:
        with pytest.warns(DeprecationWarning, match="welcome_greeting"):
            TACServerConfig(public_domain="test.ngrok.io", welcome_greeting="Legacy!")

    def test_forwarded_when_channel_did_not_set_greeting(self) -> None:
        from fastapi.testclient import TestClient

        from tac.channels.voice import VoiceChannel
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        vc = VoiceChannel(tac)  # no welcome_greeting on channel
        with pytest.warns(DeprecationWarning):
            server_config = TACServerConfig(
                public_domain="test.ngrok.io", welcome_greeting="Legacy!"
            )
        server = TACFastAPIServer(tac=tac, config=server_config, voice_channel=vc)
        client = TestClient(server.app)
        signature = compute_signature("http://testserver/twiml")
        resp = client.post("/twiml", headers={"X-Twilio-Signature": signature})
        assert 'welcomeGreeting="Legacy!"' in resp.text

    def test_twiml_options_greeting_wins_over_deprecated_server_field(self) -> None:
        from fastapi.testclient import TestClient

        from tac.channels.voice import VoiceChannel, VoiceChannelConfig
        from tac.models.voice import TwiMLOptions
        from tac.server import TACFastAPIServer

        tac = TAC(get_test_config())
        vc = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                twiml_options=TwiMLOptions(welcome_greeting="Channel!"),
            ),
        )
        with pytest.warns(DeprecationWarning):
            server_config = TACServerConfig(
                public_domain="test.ngrok.io", welcome_greeting="Legacy!"
            )
        server = TACFastAPIServer(tac=tac, config=server_config, voice_channel=vc)
        client = TestClient(server.app)
        signature = compute_signature("http://testserver/twiml")
        resp = client.post("/twiml", headers={"X-Twilio-Signature": signature})
        assert 'welcomeGreeting="Channel!"' in resp.text
        assert "Legacy!" not in resp.text

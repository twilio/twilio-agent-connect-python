"""TACFastAPIServer: Batteries-included FastAPI server for TAC channels.

This module provides FastAPIWebSocketAdapter (bridges FastAPI WebSocket to
WebSocketProtocol) and TACFastAPIServer (creates a FastAPI app with routes for
voice, messaging, and CI webhooks).

Requires: pip install tac[server]
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from tac.channels.base import BaseChannel
from tac.channels.websocket_protocol import WebSocketDisconnectError
from tac.core.logging import get_logger
from tac.core.tac import TAC
from tac.models.voice import TwiMLRequestContext, VoiceServerURLs
from tac.server.config import TACServerConfig

if TYPE_CHECKING:
    from tac.channels.messaging import MessagingChannel
    from tac.channels.voice import VoiceChannel

try:
    import uvicorn
    from fastapi import Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, Response

    from tac.server.signature_validation import (
        build_http_signature_dependency,
        build_websocket_signature_dependency,
    )
except ImportError as e:
    raise ImportError(
        "TACFastAPIServer requires FastAPI and uvicorn. Install with: pip install tac[server]"
    ) from e

logger = get_logger(__name__)


class FastAPIWebSocketAdapter:
    """Adapts a FastAPI WebSocket to satisfy WebSocketProtocol.

    Converts FastAPI's WebSocketDisconnect into WebSocketDisconnectError
    so that VoiceChannel's framework-agnostic exception handling works.
    """

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket

    async def accept(self) -> None:
        await self._ws.accept()

    async def receive_json(self) -> Any:
        try:
            return await self._ws.receive_json()
        except WebSocketDisconnect:
            raise WebSocketDisconnectError("WebSocket disconnected") from None

    async def send_text(self, data: str) -> None:
        try:
            await self._ws.send_text(data)
        except WebSocketDisconnect:
            raise WebSocketDisconnectError("WebSocket disconnected") from None

    async def close(self) -> None:
        await self._ws.close()


class TACFastAPIServer:
    """Batteries-included FastAPI server for TAC channels.

    Creates (or adopts) a FastAPI app and registers routes for voice, messaging,
    and CI webhooks, then starts uvicorn when start() is called.

    Customization:
        - Pass your own FastAPI instance via ``app=...`` to control
          construction-time settings (title, version, lifespan, docs_url, ...).
          TAC routes are registered onto it immediately in ``__init__``.
        - Or mutate ``server.app`` after construction: add middleware,
          exception handlers, routers, or custom routes — before calling
          ``start()``.
        - To customize TwiML attributes (voice, language, transcription provider,
          interruption behavior, ``<Language>`` children, etc.) set a
          ``customize_twiml_options`` on ``VoiceChannelConfig``. The customizer
          receives a framework-neutral ``TwiMLRequestContext`` and returns a
          ``TwiMLOptions``; any field it explicitly sets overrides TAC defaults.
          For same-on-every-call settings, set ``twiml_options`` on
          ``VoiceChannelConfig`` directly.

    Example:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        app = FastAPI(title="My Service", version="1.2.0")
        app.add_middleware(CORSMiddleware, allow_origins=["*"])

        server = TACFastAPIServer(tac=tac, voice_channel=vc, app=app)

        @server.app.get("/health")
        async def health() -> dict:
            return {"status": "ok"}

        server.start()
    """

    def __init__(
        self,
        tac: TAC,
        voice_channel: VoiceChannel | None = None,
        messaging_channels: list[MessagingChannel] | None = None,
        config: TACServerConfig | None = None,
        app: FastAPI | None = None,
    ) -> None:
        self.tac = tac
        self.config = config or TACServerConfig.from_env()
        self.voice_channel = voice_channel
        self.messaging_channels: list[MessagingChannel] = messaging_channels or []

        # Forward deprecated TACServerConfig.welcome_greeting to the voice channel
        # if the channel wasn't given its own. Drop this forwarding when the
        # field is removed from TACServerConfig.
        if (
            self.config.welcome_greeting is not None
            and self.voice_channel is not None
            and "welcome_greeting" not in self.voice_channel.config.model_fields_set
        ):
            self.voice_channel._welcome_greeting = self.config.welcome_greeting

        # Gather all channels that need webhook processing
        self.webhook_channels: list[BaseChannel] = []
        if self.voice_channel:
            self.webhook_channels.append(self.voice_channel)
        self.webhook_channels.extend(self.messaging_channels)

        self.app: FastAPI = app if app is not None else FastAPI(title="TAC Server")
        self._register_routes(self.app)

    def _register_routes(self, app: FastAPI) -> None:
        """Register TAC routes (conversation webhook, voice, CI) onto the given FastAPI app."""
        config = self.config
        http_sig = build_http_signature_dependency(self.tac.config.auth_token)
        ws_sig = build_websocket_signature_dependency(self.tac.config.auth_token)

        if self.webhook_channels:
            channels = self.webhook_channels

            async def _process_webhook_with_error_handling(
                channel: BaseChannel, webhook_data: dict[str, Any], idempotency_token: str | None
            ) -> None:
                """Wrapper to handle exceptions in background webhook processing tasks."""
                try:
                    await channel.process_webhook(webhook_data, idempotency_token)
                except Exception as e:
                    logger.error(
                        "Error processing webhook in channel",
                        channel=channel.get_channel_name(),
                        error=str(e),
                        exc_info=True,
                    )

            @app.post(config.conversation_webhook_path, dependencies=[Depends(http_sig)])
            async def conversation_webhook(request: Request) -> JSONResponse:
                """Handle incoming conversation webhooks from Twilio (all channels)."""
                try:
                    webhook_data = await request.json()
                    if not isinstance(webhook_data, dict):
                        logger.error(
                            "Conversation webhook payload must be a JSON object",
                            payload_type=type(webhook_data).__name__,
                        )
                        return JSONResponse(
                            content={
                                "status": "error",
                                "message": "Webhook payload must be a JSON object",
                            },
                            status_code=400,
                        )
                    idempotency_token = request.headers.get("i-twilio-idempotency-token")
                    for channel in channels:
                        asyncio.create_task(
                            _process_webhook_with_error_handling(
                                channel, webhook_data, idempotency_token
                            )
                        )
                    return JSONResponse(content={"status": "ok"}, status_code=200)
                except Exception as e:
                    logger.error("Conversation webhook error", error=str(e), exc_info=True)
                    return JSONResponse(
                        content={"status": "error", "message": "Failed to process webhook"},
                        status_code=400,
                    )
        else:
            logger.warning("No channels configured — conversation webhook route disabled")

        if self.voice_channel is not None:
            vc = self.voice_channel

            if not config.public_domain:
                logger.warning(
                    "public_domain is not set — voice URLs will be malformed. "
                    "Set TWILIO_VOICE_PUBLIC_DOMAIN environment variable."
                )

            @app.post(config.twiml_path, dependencies=[Depends(http_sig)])
            async def post_twiml(request: Request) -> Response:
                """Generate TwiML for incoming voice calls."""
                server_urls = VoiceServerURLs(
                    websocket_url=f"wss://{config.public_domain}{config.websocket_path}",
                    action_url=(
                        f"https://{config.public_domain}{config.conversation_relay_callback_path}"
                    ),
                )

                form = await request.form()
                form_dict = {k: v for k, v in form.items() if isinstance(v, str)}
                request_context = TwiMLRequestContext.from_form(form_dict)

                twiml = await vc.handle_incoming_call(
                    server_urls,
                    request_context=request_context,
                )
                return Response(content=twiml, media_type="application/xml")

            @app.websocket(config.websocket_path)
            async def websocket_endpoint(websocket: WebSocket, _: None = Depends(ws_sig)) -> None:
                """Handle voice WebSocket connections."""
                adapter = FastAPIWebSocketAdapter(websocket)
                await vc.handle_websocket(adapter)

            @app.post(config.conversation_relay_callback_path)
            async def conversation_relay_callback(request: Request) -> Response:
                """Handle ConversationRelay action callback (call ended)."""
                try:
                    form_data = await request.form()
                    payload_dict = {k: str(v) for k, v in form_data.items()}
                    await vc.handle_conversation_relay_callback(payload_dict)
                except Exception:
                    logger.error("Failed to process ConversationRelay callback", exc_info=True)
                    return Response(content="", media_type="text/plain", status_code=400)
                return Response(content="", media_type="text/plain", status_code=200)

        if config.cintel_webhook_path is not None:
            tac = self.tac

            @app.post(config.cintel_webhook_path, dependencies=[Depends(http_sig)])
            async def cintel_webhook(request: Request) -> JSONResponse:
                """Handle Conversation Intelligence webhook events."""
                payload = await request.json()
                result = await tac.process_cintel_event(payload)
                return JSONResponse(content=result.model_dump())

    def start(self) -> None:
        """Start uvicorn serving ``self.app``."""
        logger.info(f"Starting TAC FastAPI Server on {self.config.host}:{self.config.port}")
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
            access_log=False,  # Disable verbose HTTP request logs
        )

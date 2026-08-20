"""OpenAIRealtimeSipServer: minimal standalone FastAPI server for OpenAIRealtimeSipChannel.

Deliberately separate from TACFastAPIServer — this channel doesn't share TwiML,
ConversationRelay WebSocket, or Twilio call-event routes with the rest of TAC's
voice/messaging channels, and its webhook uses OpenAI's signature scheme, not
Twilio's. Registers exactly one route.

Requires: pip install tac[server] tac[openai-realtime-sip]
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from tac.channels.openai_realtime_sip import OpenAIRealtimeSipChannel
from tac.core.logging import get_logger

try:
    import uvicorn
    from fastapi import FastAPI, Request, Response
except ImportError as e:
    raise ImportError(
        "OpenAIRealtimeSipServer requires FastAPI and uvicorn. "
        "Install with: pip install tac[server]"
    ) from e

try:
    from openai import InvalidWebhookSignatureError
except ImportError as e:
    raise ImportError(
        "OpenAIRealtimeSipServer requires the 'openai' package. "
        "Install with: pip install tac[openai-realtime-sip]"
    ) from e

logger = get_logger(__name__)


class OpenAIRealtimeSipServerConfig(BaseModel):
    """Configuration for ``OpenAIRealtimeSipServer``."""

    host: str = Field(default="0.0.0.0", description="Host to bind the server to")
    port: int = Field(default=8000, description="Port to bind the server to")
    webhook_path: str = Field(
        default="/openai/incoming-call",
        description="Path OpenAI's realtime.call.incoming webhook is served at. "
        "Must match the URL configured for the webhook at platform.openai.com.",
    )


class OpenAIRealtimeSipServer:
    """Minimal FastAPI server exposing just the OpenAI Realtime call-incoming webhook."""

    def __init__(
        self,
        channel: OpenAIRealtimeSipChannel,
        config: OpenAIRealtimeSipServerConfig | None = None,
        app: FastAPI | None = None,
    ) -> None:
        self.channel = channel
        self.config = config or OpenAIRealtimeSipServerConfig()
        self.app: FastAPI = app if app is not None else FastAPI(title="OpenAI Realtime SIP Server")
        self._register_routes(self.app)

    def _register_routes(self, app: FastAPI) -> None:
        channel = self.channel
        path = self.config.webhook_path

        @app.post(path)
        async def openai_incoming_call(request: Request) -> Response:
            """Handle the realtime.call.incoming webhook from OpenAI."""
            body = await request.body()
            try:
                event = channel.verify_webhook(body, dict(request.headers))
            except (InvalidWebhookSignatureError, ValueError) as e:
                logger.warning("Rejected OpenAI webhook", error=str(e))
                return Response(status_code=400)

            if event is None:
                # Verified, but not a call-incoming event — nothing to do.
                return Response(status_code=200)

            await channel.process_webhook(event.model_dump())
            return Response(status_code=200)

    def start(self) -> None:
        """Start uvicorn serving ``self.app``."""
        logger.info(f"Starting OpenAI Realtime SIP Server on {self.config.host}:{self.config.port}")
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
        )

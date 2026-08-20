"""OpenAIRealtimeMediaStreamsServer: minimal standalone FastAPI server for
OpenAIRealtimeMediaStreamsChannel.

Deliberately separate from TACFastAPIServer — this channel doesn't share
ConversationRelay routes with the rest of TAC's voice/messaging channels, and
its WebSocket route serves Twilio Media Streams' audio-bridging protocol
rather than ConversationRelay's text protocol. Registers two routes: TwiML
and the Media Stream WebSocket.

Requires: pip install tac[server] tac[openai-realtime-media-streams]
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from tac.channels.openai_realtime_media_streams import OpenAIRealtimeMediaStreamsChannel
from tac.core.logging import get_logger
from tac.core.tac import TAC

try:
    import uvicorn
    from fastapi import Depends, FastAPI, Request, WebSocket
    from fastapi.responses import Response

    from tac.server.fastapi_server import FastAPIWebSocketAdapter
    from tac.server.signature_validation import build_http_signature_dependency
except ImportError as e:
    raise ImportError(
        "OpenAIRealtimeMediaStreamsServer requires FastAPI and uvicorn. "
        "Install with: pip install tac[server]"
    ) from e

logger = get_logger(__name__)


class OpenAIRealtimeMediaStreamsServerConfig(BaseModel):
    """Configuration for ``OpenAIRealtimeMediaStreamsServer``."""

    host: str = Field(default="0.0.0.0", description="Host to bind the server to")
    port: int = Field(default=8000, description="Port to bind the server to")
    twiml_path: str = Field(default="/twiml", description="Path Twilio's Voice URL points at")
    websocket_path: str = Field(
        default="/media-stream", description="Path the Media Stream WebSocket is served at"
    )
    validate_twiml_signature: bool = Field(
        default=True,
        description="Validate Twilio's signature on the TwiML request. The WebSocket "
        "upgrade itself is not signature-validated (Twilio doesn't sign it) — fine for "
        "testing, revisit before production.",
    )


class OpenAIRealtimeMediaStreamsServer:
    """Minimal FastAPI server exposing TwiML + the Media Stream WebSocket."""

    def __init__(
        self,
        tac: TAC,
        channel: OpenAIRealtimeMediaStreamsChannel,
        config: OpenAIRealtimeMediaStreamsServerConfig | None = None,
        app: FastAPI | None = None,
    ) -> None:
        self.tac = tac
        self.channel = channel
        self.config = config or OpenAIRealtimeMediaStreamsServerConfig()

        if not self.tac.config.voice_public_domain:
            raise ValueError(
                "OpenAIRealtimeMediaStreamsServer needs TACConfig.voice_public_domain to "
                "build the public wss:// URL Twilio streams audio to. Set it or the "
                "TWILIO_VOICE_PUBLIC_DOMAIN env var."
            )

        self.app: FastAPI = (
            app if app is not None else FastAPI(title="OpenAI Realtime Media Streams Server")
        )
        self._register_routes(self.app)

    @property
    def websocket_url(self) -> str:
        """Public ``wss://`` URL emitted in the ``<Stream>`` TwiML."""
        return f"wss://{self.tac.config.voice_public_domain}{self.config.websocket_path}"

    def _register_routes(self, app: FastAPI) -> None:
        channel = self.channel
        websocket_url = self.websocket_url

        twiml_dependencies = []
        if self.config.validate_twiml_signature:
            http_sig = build_http_signature_dependency(self.tac.config.auth_token)
            twiml_dependencies.append(Depends(http_sig))

        @app.post(self.config.twiml_path, dependencies=twiml_dependencies)
        async def post_twiml(request: Request) -> Response:
            """Return <Connect><Stream> TwiML pointing Twilio at the audio WebSocket."""
            twiml = channel.build_stream_twiml(websocket_url)
            return Response(content=twiml, media_type="application/xml")

        @app.websocket(self.config.websocket_path)
        async def media_stream_websocket(websocket: WebSocket) -> None:
            """Bridge the Twilio Media Stream to the OpenAI Realtime WebSocket."""
            adapter = FastAPIWebSocketAdapter(websocket)
            await channel.handle_websocket(adapter)

    def start(self) -> None:
        """Start uvicorn serving ``self.app``."""
        logger.info(
            f"Starting OpenAI Realtime Media Streams Server on "
            f"{self.config.host}:{self.config.port}"
        )
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
        )

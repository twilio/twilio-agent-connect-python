"""RealtimeVoiceServer: dedicated FastAPI server for the realtime voice path.

This is deliberately separate from :class:`TACFastAPIServer`. That server is
built around ConversationRelay and the conversation/messaging webhook model;
this one serves a single, different protocol — Twilio Media Streams bridged to
a speech-to-speech model (see :class:`~tac.channels.realtime.RealtimeVoiceChannel`).

It exposes two routes:
  - ``POST {twiml_path}``  -> returns ``<Connect><Stream>`` TwiML (Twilio-signed)
  - ``WS   {websocket_path}`` -> the bidirectional audio bridge

Note: the WebSocket upgrade is intentionally NOT signature-validated for now
(the TwiML request that hands out the wss URL is). Revisit if you confirm how
Twilio signs the Media Streams upgrade.

Requires: pip install 'twilio-agent-connect[server,realtime]'
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tac.core.logging import get_logger
from tac.core.tac import TAC

if TYPE_CHECKING:
    from tac.channels.realtime import RealtimeVoiceChannel

try:
    import uvicorn
    from fastapi import Depends, FastAPI, Request, WebSocket
    from fastapi.responses import Response

    from tac.server.fastapi_server import FastAPIWebSocketAdapter
    from tac.server.signature_validation import build_http_signature_dependency
except ImportError as e:
    raise ImportError(
        "RealtimeVoiceServer requires FastAPI and uvicorn. "
        "Install with: pip install 'twilio-agent-connect[server,realtime]'"
    ) from e

logger = get_logger(__name__)


class RealtimeVoiceServer:
    """Batteries-included FastAPI server for a :class:`RealtimeVoiceChannel`.

    Example:
        from tac import TAC, TACConfig
        from tac.channels.realtime import RealtimeVoiceChannel, RealtimeVoiceChannelConfig
        from tac.server import RealtimeVoiceServer

        tac = TAC(TACConfig.from_env())
        channel = RealtimeVoiceChannel(tac, RealtimeVoiceChannelConfig.from_env())
        RealtimeVoiceServer(tac, channel).start()

    Point your Twilio number's voice webhook at ``POST {twiml_path}``.
    """

    def __init__(
        self,
        tac: TAC,
        realtime_voice_channel: RealtimeVoiceChannel,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        twiml_path: str = "/twiml-realtime",
        websocket_path: str = "/voice-realtime",
        validate_twiml_signature: bool = True,
        app: FastAPI | None = None,
    ) -> None:
        self.tac = tac
        self.channel = realtime_voice_channel
        self.host = host
        self.port = port
        self.twiml_path = twiml_path
        self.websocket_path = websocket_path
        self.validate_twiml_signature = validate_twiml_signature

        if not self.tac.config.voice_public_domain:
            raise ValueError(
                "RealtimeVoiceServer needs TACConfig.voice_public_domain to build the public "
                "wss:// URL Twilio streams audio to. Set it or the TWILIO_VOICE_PUBLIC_DOMAIN "
                "env var."
            )

        self.app: FastAPI = app if app is not None else FastAPI(title="TAC Realtime Voice Server")
        self._register_routes(self.app)

    @property
    def websocket_url(self) -> str:
        """Public ``wss://`` URL emitted in the ``<Stream>`` TwiML."""
        return f"wss://{self.tac.config.voice_public_domain}{self.websocket_path}"

    def _register_routes(self, app: FastAPI) -> None:
        channel = self.channel
        websocket_url = self.websocket_url

        # The TwiML request comes from Twilio and is signed; validate it unless
        # explicitly disabled. The WS upgrade is left open (see module docstring).
        twiml_dependencies = []
        if self.validate_twiml_signature:
            http_sig = build_http_signature_dependency(self.tac.config.auth_token)
            twiml_dependencies.append(Depends(http_sig))

        @app.post(self.twiml_path, dependencies=twiml_dependencies)
        async def post_realtime_twiml(request: Request) -> Response:
            """Return <Connect><Stream> TwiML pointing Twilio at the audio WebSocket."""
            twiml = channel.build_stream_twiml(websocket_url)
            return Response(content=twiml, media_type="application/xml")

        @app.websocket(self.websocket_path)
        async def realtime_websocket_endpoint(websocket: WebSocket) -> None:
            """Bridge the Twilio Media Stream to the speech-to-speech model."""
            adapter = FastAPIWebSocketAdapter(websocket)
            await channel.handle_websocket(adapter)

    def start(self) -> None:
        """Start uvicorn serving ``self.app``."""
        logger.info(f"Starting TAC Realtime Voice Server on {self.host}:{self.port}")
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False,
        )

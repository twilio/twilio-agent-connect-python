"""Server module providing batteries-included FastAPI server for TAC channels.

Requires the 'server' optional dependency: pip install tac[server]
"""

from tac.server.config import TACServerConfig
from tac.server.fastapi_server import FastAPIWebSocketAdapter, TACFastAPIServer
from tac.server.signature_validation import (
    build_http_signature_dependency,
    build_websocket_signature_dependency,
    validate_twilio_webhook,
)

__all__ = [
    "TACFastAPIServer",
    "TACServerConfig",
    "FastAPIWebSocketAdapter",
    "build_http_signature_dependency",
    "build_websocket_signature_dependency",
    "validate_twilio_webhook",
]

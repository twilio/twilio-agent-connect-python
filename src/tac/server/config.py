"""Configuration for TAC server implementations."""

import os

from pydantic import BaseModel, Field


class TACServerConfig(BaseModel):
    """Configuration for TAC server implementations.

    Controls host/port binding and the server-only webhook paths. Voice paths
    (WebSocket and ConversationRelay action) live on ``TACConfig`` because
    they're consumed by the voice channel regardless of which web framework
    is used; this server reads them from there to register its routes.
    """

    host: str = Field(default="0.0.0.0", description="Host to bind the server to")
    port: int = Field(default=8000, description="Port to bind the server to")

    conversation_webhook_path: str = Field(
        default="/webhook", description="Path for conversation webhook endpoint (for all channels)"
    )
    twiml_path: str = Field(default="/twiml", description="Path for TwiML generation endpoint")
    cintel_webhook_path: str | None = Field(
        default=None,
        description="Path for Conversation Intelligence webhook endpoint. "
        "Set to enable CI webhook route (e.g., '/ci-webhook').",
    )

    @classmethod
    def from_env(cls) -> "TACServerConfig":
        """Create config from environment variables.

        Environment variables:
            TWILIO_SERVER_HOST: Host to bind to (default: 0.0.0.0)
            TWILIO_SERVER_PORT: Port to bind to (default: 8000)

        TWILIO_VOICE_PUBLIC_DOMAIN is read by ``TACConfig.from_env`` instead.
        """
        kwargs: dict[str, object] = {}

        host = os.environ.get("TWILIO_SERVER_HOST")
        if host:
            kwargs["host"] = host

        port = os.environ.get("TWILIO_SERVER_PORT")
        if port:
            kwargs["port"] = int(port)

        return cls(**kwargs)

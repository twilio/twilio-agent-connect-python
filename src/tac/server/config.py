"""Configuration for TAC server implementations."""

import os

from pydantic import BaseModel, Field


class TACServerConfig(BaseModel):
    """Configuration for TAC server implementations.

    Controls host/port binding and webhook paths registered by the server.
    Voice-specific settings — the public domain, WebSocket path, and
    ConversationRelay action path — live on ``TACConfig`` and
    ``VoiceChannelConfig``, since they're consumed by the voice channel
    regardless of which web framework is used.
    """

    host: str = Field(default="0.0.0.0", description="Host to bind the server to")
    port: int = Field(default=8000, description="Port to bind the server to")

    conversation_webhook_path: str = Field(
        default="/webhook", description="Path for conversation webhook endpoint (for all channels)"
    )
    twiml_path: str = Field(default="/twiml", description="Path for TwiML generation endpoint")
    websocket_path: str = Field(
        default="/ws",
        description="Path to register the voice WebSocket route at. "
        "VoiceChannelConfig.websocket_path builds the public URL — keep them "
        "in sync, or set VoiceChannelConfig.websocket_url directly.",
    )
    conversation_relay_callback_path: str = Field(
        default="/conversation-relay-callback",
        description="Path to register the ConversationRelay action callback route at. "
        "Same pairing rule as websocket_path with VoiceChannelConfig.action_path.",
    )
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

"""Configuration for TAC server implementations."""

import os

from pydantic import BaseModel, Field


class TACServerConfig(BaseModel):
    """Configuration for TAC server implementations.

    Controls host/port binding, public domain for WebSocket URLs,
    and customizable webhook paths.
    """

    host: str = Field(default="0.0.0.0", description="Host to bind the server to")
    port: int = Field(default=8000, description="Port to bind the server to")
    public_domain: str = Field(
        default="", description="Public domain for WebSocket URL (e.g., 'example.ngrok.io')"
    )
    welcome_greeting: str = Field(
        default="Hello! How can I assist you today?",
        description="Initial greeting message for callers",
    )

    conversation_webhook_path: str = Field(
        default="/webhook", description="Path for conversation webhook endpoint (for all channels)"
    )
    twiml_path: str = Field(default="/twiml", description="Path for TwiML generation endpoint")
    websocket_path: str = Field(default="/ws", description="Path for voice WebSocket endpoint")
    conversation_relay_callback_path: str = Field(
        default="/conversation-relay-callback",
        description="Path for ConversationRelay action callback endpoint",
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
            TWILIO_VOICE_PUBLIC_DOMAIN: Public domain for WebSocket URLs (required for voice)
            TWILIO_SERVER_HOST: Host to bind to (default: 0.0.0.0)
            TWILIO_SERVER_PORT: Port to bind to (default: 8000)
        """
        kwargs: dict[str, object] = {}

        public_domain = os.environ.get("TWILIO_VOICE_PUBLIC_DOMAIN")
        if public_domain:
            kwargs["public_domain"] = public_domain

        host = os.environ.get("TWILIO_SERVER_HOST")
        if host:
            kwargs["host"] = host

        port = os.environ.get("TWILIO_SERVER_PORT")
        if port:
            kwargs["port"] = int(port)

        return cls(**kwargs)

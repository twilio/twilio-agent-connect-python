"""Configuration for TAC server implementations."""

import os
import warnings

from pydantic import BaseModel, Field, model_validator


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

    public_domain: str = Field(
        default="",
        description="DEPRECATED: set TACConfig.voice_public_domain instead. "
        "When set here, it is forwarded; will be removed in a future release.",
    )
    welcome_greeting: str | None = Field(
        default=None,
        description="DEPRECATED: set welcome_greeting on VoiceChannelConfig instead. "
        "When set here, it is forwarded to the voice channel as a default; it will be "
        "removed in a future release.",
    )

    @model_validator(mode="after")
    def _warn_deprecated_fields(self) -> "TACServerConfig":
        if self.welcome_greeting is not None:
            warnings.warn(
                "TACServerConfig.welcome_greeting is deprecated and will be removed "
                "in a future release. Set welcome_greeting on VoiceChannelConfig instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.public_domain:
            warnings.warn(
                "TACServerConfig.public_domain is deprecated and will be removed in "
                "a future release. Set TACConfig.voice_public_domain (or "
                "TWILIO_VOICE_PUBLIC_DOMAIN env var) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self

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

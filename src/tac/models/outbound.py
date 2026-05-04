"""Models for outbound conversation initiation."""

from typing import Any

from pydantic import BaseModel, Field

from tac.models.session import ConversationSession


class InitiateMessagingConversationOptions(BaseModel):
    """Shared options for initiating an outbound messaging conversation.

    This base model is used for messaging-style outbound conversations,
    including SMS, RCS, WhatsApp, and Chat. Each channel may extend this with
    channel-specific requirements (e.g., Chat requires channel_id).

    The sender is always TAC's configured address (``config.phone_number``
    for SMS, ``config.rcs_sender_id`` for RCS, ``config.whatsapp_number``
    for WhatsApp, ``ChatChannelConfig.agent_address`` for Chat).
    Multi-sender deployments should use one TAC instance per sender so
    inbound webhook routing, memory scoping, and configuration stay in sync.
    """

    to: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = Field(default=None)

    model_config = {"populate_by_name": True}


class InitiateChatConversationOptions(InitiateMessagingConversationOptions):
    """Options for initiating an outbound Chat conversation.

    Extends InitiateMessagingConversationOptions with a required channel_id
    (Conversations v1 Channel SID) for Chat delivery.
    """

    channel_id: str = Field(..., min_length=1)


class InitiateConversationResult(BaseModel):
    """Result of initiating an outbound messaging conversation."""

    conversation_id: str
    session: ConversationSession

    model_config = {"arbitrary_types_allowed": True}


class InitiateVoiceConversationOptions(BaseModel):
    """Options for initiating an outbound voice conversation.

    The caller identity is always TAC's configured ``config.phone_number``.
    Multi-number deployments should use one TAC instance per line.
    """

    to: str = Field(..., min_length=1)
    websocket_url: str = Field(...)
    welcome_greeting: str | None = Field(default=None)
    action_url: str | None = Field(default=None)
    custom_parameters: dict[str, str | int | bool] | None = Field(default=None)

    model_config = {"populate_by_name": True}


class InitiateVoiceConversationResult(BaseModel):
    """Result of initiating an outbound voice conversation."""

    call_sid: str

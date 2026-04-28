"""Models for outbound conversation initiation."""

from typing import Any

from pydantic import BaseModel, Field

from tac.models.session import ConversationSession


class InitiateMessagingConversationOptions(BaseModel):
    """Options for initiating an outbound SMS or Chat conversation."""

    to: str = Field(..., min_length=1)
    from_: str | None = Field(default=None, alias="from")
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
    """Options for initiating an outbound voice conversation."""

    to: str = Field(..., min_length=1)
    from_: str | None = Field(default=None, alias="from")
    websocket_url: str = Field(...)
    welcome_greeting: str | None = Field(default=None)
    action_url: str | None = Field(default=None)
    custom_parameters: dict[str, str | int | bool] | None = Field(default=None)

    model_config = {"populate_by_name": True}


class InitiateVoiceConversationResult(BaseModel):
    """Result of initiating an outbound voice conversation."""

    call_sid: str

"""Models for outbound conversation initiation."""

from typing import Any

from pydantic import BaseModel, Field

from tac.models.session import ConversationSession
from tac.models.voice import TwiMLOptions


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

    TwiML for the outbound call is built by merging per-field, highest
    precedence first:
      1. This call's ``twiml_options`` (per-call overrides)
      2. ``VoiceChannelConfig.default_twiml_options`` (channel-wide defaults)
      3. TAC defaults (welcome greeting, conversation_configuration,
         action_url resolved via Studio handoff if configured)

    Fields you don't set at a layer fall through to lower layers — so
    ``twiml_options=TwiMLOptions(voice="es-MX-Neural2-A")`` on this call
    overrides only ``voice``; ``language``, ``interruptible``, etc. from the
    channel config still apply.

    Set ``voice``, ``language``, ``interruptible``, etc. on the channel's
    ``VoiceChannelConfig.default_twiml_options`` to apply them to every call
    (both inbound and outbound). Use this model's ``twiml_options`` for
    per-call overrides (e.g. campaign-specific ``custom_parameters``).
    """

    to: str = Field(..., min_length=1)
    websocket_url: str | None = Field(
        default=None,
        description="Public WebSocket URL for ConversationRelay (e.g. "
        "'wss://your-domain.ngrok.app/ws'). Optional — defaults to "
        "``VoiceChannelConfig.websocket_url`` when not provided. Pass it here "
        "only to override the channel's URL for a specific call.",
    )
    twiml_options: TwiMLOptions | None = Field(
        default=None,
        description="Per-call TwiMLOptions overrides. Merged over "
        "VoiceChannelConfig.default_twiml_options and TAC defaults.",
    )

    model_config = {"populate_by_name": True}


class InitiateVoiceConversationResult(BaseModel):
    """Result of initiating an outbound voice conversation."""

    call_sid: str

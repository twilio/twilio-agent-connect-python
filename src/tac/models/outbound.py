"""Models for outbound conversation initiation."""

import warnings
from typing import Any

from pydantic import BaseModel, Field, model_validator

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

    TwiML for the outbound call is built by layering:
      1. This call's ``twiml_options`` (highest priority)
      2. ``VoiceChannelConfig.twiml_options`` (channel-wide defaults)
      3. TAC defaults (welcome greeting, conversation_configuration,
         action_url resolved via Studio handoff if configured)

    Set ``voice``, ``language``, ``interruptible``, etc. on the channel's
    ``VoiceChannelConfig.twiml_options`` to apply them to every call (both
    inbound and outbound). Use this model's ``twiml_options`` for per-call
    overrides (e.g. campaign-specific ``custom_parameters``).
    """

    to: str = Field(..., min_length=1)
    websocket_url: str = Field(
        ...,
        description="Public WebSocket URL for ConversationRelay, e.g. "
        "'wss://your-domain.ngrok.app/ws'. Required because outbound calls "
        "have no inbound HTTP request to derive the host from.",
    )
    twiml_options: TwiMLOptions | None = Field(
        default=None,
        description="Per-call TwiMLOptions overrides. Merged over "
        "VoiceChannelConfig.twiml_options and TAC defaults.",
    )

    # Deprecated flat fields. Forwarded into twiml_options in the validator
    # below. Remove in a future release.
    welcome_greeting: str | None = Field(
        default=None,
        description="DEPRECATED: set welcome_greeting on twiml_options or "
        "VoiceChannelConfig.twiml_options instead.",
    )
    action_url: str | None = Field(
        default=None,
        description="DEPRECATED: set action_url on twiml_options or "
        "VoiceChannelConfig.twiml_options instead.",
    )
    custom_parameters: dict[str, str | int | bool] | None = Field(
        default=None,
        description="DEPRECATED: set custom_parameters on twiml_options or "
        "VoiceChannelConfig.twiml_options instead.",
    )

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _forward_deprecated_fields(self) -> "InitiateVoiceConversationOptions":
        """Forward deprecated flat fields into twiml_options with a warning.

        If both the flat field and twiml_options.<field> are set, the explicit
        twiml_options value wins.
        """
        deprecated = {
            "welcome_greeting": self.welcome_greeting,
            "action_url": self.action_url,
            "custom_parameters": self.custom_parameters,
        }
        set_fields = {k: v for k, v in deprecated.items() if v is not None}
        if not set_fields:
            return self

        warnings.warn(
            "InitiateVoiceConversationOptions flat fields "
            f"({', '.join(set_fields)}) are deprecated. Pass them on "
            "twiml_options or configure them on VoiceChannelConfig.twiml_options.",
            DeprecationWarning,
            stacklevel=2,
        )

        if self.twiml_options is None:
            self.twiml_options = TwiMLOptions(**set_fields)
        else:
            # twiml_options explicit fields win; fill in only fields the user
            # didn't set on twiml_options.
            already_set = self.twiml_options.model_fields_set
            for key, value in set_fields.items():
                if key not in already_set:
                    setattr(self.twiml_options, key, value)
        return self


class InitiateVoiceConversationResult(BaseModel):
    """Result of initiating an outbound voice conversation."""

    call_sid: str

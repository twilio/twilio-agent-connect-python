"""Voice channel configuration."""

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from tac.models.memory import MemoryMode
from tac.models.voice import TwiMLOptions, TwiMLRequest
from tac.session import SessionManager, ThreadSafeSessionManager

InboundTwiMLCustomizer = Callable[[TwiMLRequest], Awaitable[TwiMLOptions]]


class VoiceChannelConfig(BaseModel):
    """
    Configuration for Voice channel.

    TwiML configuration layers (highest precedence first):

      Inbound calls (``handle_incoming_call``):
        1. ``customize_inbound_twiml(twiml_request)`` output  [optional]
        2. ``default_twiml_options``                          [optional]
        3. TAC defaults

      Outbound calls (``initiate_outbound_conversation``):
        1. ``InitiateVoiceConversationOptions.twiml_options`` [optional]
        2. ``default_twiml_options``                          [optional]
        3. TAC defaults

    All layers merge per-field via Pydantic's ``model_fields_set`` — only
    fields a layer explicitly sets override lower layers. Lists (``languages``)
    and nested models (``custom_parameters``) replace wholesale when set.

    Attributes:
        session_manager: SessionManager for tracking and canceling in-flight tasks.
            Defaults to ThreadSafeSessionManager for automatic task cancellation on
            interrupts and new prompts. Set to None only for debugging/testing.
        memory_mode: Memory retrieval mode. Default is "never".
            - "always": Retrieve memory for every message with the query string
            - "once": Retrieve memory once at conversation start with empty query and cache it.
                     Cache is invalidated when conversation becomes INACTIVE.
            - "never": Skip memory retrieval
        websocket_url: Public WebSocket URL for ConversationRelay (e.g.
            ``wss://example.ngrok.app/ws``). Required for outbound calls and
            any call made via ``handle_incoming_call``. ``TACFastAPIServer``
            builds and sets this automatically from its ``public_domain`` +
            ``websocket_path``; custom adapters (Flask, Django, …) must set
            it themselves.
        action_url: Public HTTPS URL for the TwiML ``<Connect action=...>``,
            used as the default session-cleanup callback. ``TACFastAPIServer``
            builds and sets this from ``public_domain`` +
            ``conversation_relay_callback_path``. Higher-priority layers
            (customizer, per-call ``twiml_options.action_url``, Studio
            handoff) still override.
        default_twiml_options: Static ``TwiMLOptions`` applied to every call
            (inbound and outbound) — voice, language, transcription provider,
            welcome_greeting, ``<Language>`` children, etc. Use this when the
            same ConversationRelay configuration is correct for every call.
        customize_inbound_twiml: Optional async callable producing per-call
            ``TwiMLOptions`` overrides for inbound calls. Receives a
            framework-neutral ``TwiMLRequest`` (parsed Twilio webhook fields).
            Outbound calls don't use this — they pass per-call TwiML via
            ``InitiateVoiceConversationOptions.twiml_options`` directly,
            because outbound is initiated from user code that already has
            per-call context in scope.
    """

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    session_manager: SessionManager | None = Field(
        default_factory=ThreadSafeSessionManager,
        description=(
            "SessionManager for task cancellation. Defaults to ThreadSafeSessionManager. "
            "Set to None only for debugging/testing."
        ),
    )
    memory_mode: MemoryMode = Field(
        default="never",
        description="Memory retrieval mode for this channel",
    )
    websocket_url: str | None = Field(
        default=None,
        description="Public WebSocket URL for ConversationRelay. Set by "
        "TACFastAPIServer automatically; custom adapters must provide it.",
    )
    action_url: str | None = Field(
        default=None,
        description="Public HTTPS URL for <Connect action=...> session cleanup. "
        "Set by TACFastAPIServer automatically; overridable by customizer, "
        "per-call twiml_options.action_url, or Studio handoff.",
    )
    default_twiml_options: TwiMLOptions | None = Field(
        default=None,
        description="Static TwiMLOptions applied to every call (inbound and "
        "outbound). For per-call customization see customize_inbound_twiml "
        "or InitiateVoiceConversationOptions.twiml_options.",
    )
    customize_inbound_twiml: InboundTwiMLCustomizer | None = Field(
        default=None,
        description="Optional async callable returning per-call TwiMLOptions "
        "overrides on inbound calls. Not invoked on outbound calls.",
    )

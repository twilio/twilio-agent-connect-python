"""Voice channel configuration."""

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from tac.models.memory import MemoryMode
from tac.models.voice import TwiMLOptions, TwiMLRequest
from tac.session import SessionManager, ThreadSafeSessionManager

InboundCallTwiMLHandler = Callable[[TwiMLRequest], Awaitable[TwiMLOptions]]


class VoiceChannelConfig(BaseModel):
    """
    Configuration for Voice channel.

    TwiML configuration layers (highest precedence first):

      Inbound calls (``handle_incoming_call``):
        1. Output of the customizer registered via
           ``VoiceChannel.on_inbound_call_twiml(...)`` [optional]
        2. ``default_twiml_options``                     [optional]
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
        websocket_path: Path the voice WebSocket is served at (e.g. ``/ws``).
            Combined with ``TACConfig.voice_public_domain`` to build the
            public WebSocket URL. ``TACFastAPIServer`` registers its
            WebSocket route at this path. Override only if you mount the
            route at a non-default path.
        action_path: Path the ConversationRelay action callback is served at
            (e.g. ``/conversation-relay-callback``). Combined with
            ``TACConfig.voice_public_domain`` to build the public action URL.
            ``TACFastAPIServer`` registers its callback route at this path.
        websocket_url: Override for the public WebSocket URL. Useful for
            cross-domain or proxy setups where the URL doesn't follow the
            standard ``wss://{voice_public_domain}{websocket_path}``
            template. When set, takes precedence over the derived value.
        action_url: Override for the public action URL — same role as
            ``websocket_url`` but for the ``<Connect action=...>`` cleanup
            callback. Higher-priority layers (customizer, per-call
            ``twiml_options.action_url``, Studio handoff) still override.
        default_twiml_options: Static ``TwiMLOptions`` applied to every call
            (inbound and outbound) — voice, language, transcription provider,
            welcome_greeting, ``<Language>`` children, etc. Use this when the
            same ConversationRelay configuration is correct for every call.

    Per-call inbound customization is registered via
    ``VoiceChannel.on_inbound_call_twiml(...)`` (not on this config).
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
    websocket_path: str = Field(
        default="/ws",
        description="Path the voice WebSocket is served at. Combined with "
        "TACConfig.voice_public_domain to build the WebSocket URL.",
    )
    action_path: str = Field(
        default="/conversation-relay-callback",
        description="Path the ConversationRelay action callback is served at. "
        "Combined with TACConfig.voice_public_domain to build the action URL.",
    )
    websocket_url: str | None = Field(
        default=None,
        description="Override for the public WebSocket URL. Defaults to "
        "wss://{voice_public_domain}{websocket_path}.",
    )
    action_url: str | None = Field(
        default=None,
        description="Override for the public <Connect action=...> URL. Defaults "
        "to https://{voice_public_domain}{action_path}.",
    )
    default_twiml_options: TwiMLOptions | None = Field(
        default=None,
        description="Static TwiMLOptions applied to every call (inbound and "
        "outbound). Per-call inbound customization is registered via "
        "VoiceChannel.on_inbound_call_twiml(...).",
    )

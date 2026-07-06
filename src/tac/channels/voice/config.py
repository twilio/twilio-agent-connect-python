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
        default_twiml_options: Static ``TwiMLOptions`` applied to every call
            (inbound and outbound). Controls the TwiML inside
            ``<ConversationRelay>`` — voice, language, transcription provider,
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
    default_twiml_options: TwiMLOptions | None = Field(
        default=None,
        description="Static TwiMLOptions for the TwiML inside <ConversationRelay>, "
        "applied to every call (inbound and outbound). Per-call inbound "
        "customization is registered via VoiceChannel.on_inbound_call_twiml(...). "
        "Note: ``custom_parameters`` and ``languages`` replace wholesale when a "
        "higher-priority layer sets them — see VoiceChannel._overlay_fields.",
    )

"""Voice channel configuration."""

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from tac.models.memory import MemoryMode
from tac.models.voice import TwiMLOptions, TwiMLRequest
from tac.session import SessionManager, ThreadSafeSessionManager

TwiMLOptionsCustomizer = Callable[[TwiMLRequest], Awaitable[TwiMLOptions]]


class VoiceChannelConfig(BaseModel):
    """
    Configuration for Voice channel.

    Attributes:
        session_manager: SessionManager for tracking and canceling in-flight tasks.
            Defaults to ThreadSafeSessionManager for automatic task cancellation on
            interrupts and new prompts. Set to None only for debugging/testing.
        memory_mode: Memory retrieval mode. Default is "never".
            - "always": Retrieve memory for every message with the query string
            - "once": Retrieve memory once at conversation start with empty query and cache it.
                     Cache is invalidated when conversation becomes INACTIVE.
            - "never": Skip memory retrieval
        twiml_options: Static ``TwiMLOptions`` applied to every call (voice,
            language, transcription provider, welcome_greeting, ``<Language>``
            children, etc.). Use this when the same ConversationRelay
            configuration is correct for every call. For per-call customization
            see ``customize_twiml_options``.
        customize_twiml_options: Optional async callable producing per-call
            ``TwiMLOptions`` overrides. Receives a framework-neutral
            ``TwiMLRequest`` (parsed Twilio webhook fields). Any field the
            function explicitly sets wins over ``twiml_options`` and TAC defaults;
            unset fields fall through.
    """

    model_config = {"arbitrary_types_allowed": True}

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
    twiml_options: TwiMLOptions | None = Field(
        default=None,
        description="Static TwiMLOptions applied to every call. Use for same-on-every-call "
        "configuration; use customize_twiml_options for per-call logic.",
    )
    customize_twiml_options: TwiMLOptionsCustomizer | None = Field(
        default=None,
        description="Optional async callable returning per-call TwiMLOptions overrides. "
        "Receives a TwiMLRequest; only fields explicitly set on the returned "
        "options override lower layers.",
    )

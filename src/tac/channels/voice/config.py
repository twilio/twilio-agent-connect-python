"""Voice channel configuration."""

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from tac.models.memory import MemoryMode
from tac.models.voice import TwiMLOptions, TwiMLRequestContext
from tac.session import SessionManager, ThreadSafeSessionManager

TwiMLOptionsResolver = Callable[[TwiMLRequestContext], Awaitable[TwiMLOptions]]


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
        resolve_twiml_options: Optional async callable that customizes the
            ConversationRelay TwiML per call. Receives a framework-neutral
            ``TwiMLRequestContext`` (parsed Twilio webhook fields) and returns
            ``TwiMLOptions`` overrides. Fields the resolver explicitly sets
            override TAC defaults; unset fields keep TAC's defaults.
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
    resolve_twiml_options: TwiMLOptionsResolver | None = Field(
        default=None,
        description="Optional async callable returning TwiMLOptions overrides per call. "
        "Receives a TwiMLRequestContext and returns TwiMLOptions; only fields explicitly "
        "set on the returned options override TAC defaults.",
    )

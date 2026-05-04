"""Voice channel configuration."""

from pydantic import BaseModel, Field

from tac.models.memory import MemoryMode
from tac.session import SessionManager, ThreadSafeSessionManager


class VoiceChannelConfig(BaseModel):
    """
    Configuration for Voice channel.

    Attributes:
        session_manager: SessionManager for tracking and canceling in-flight tasks.
            Defaults to ThreadSafeSessionManager for automatic task cancellation on
            interrupts and new prompts. Set to None only for debugging/testing.
        memory_mode: Memory retrieval mode. Default is "never".
            Set to "always" to retrieve memory for every message.
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

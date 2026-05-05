from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from tac.models.handoff import PendingHandoffData
from tac.models.memory import ProfileResponse

if TYPE_CHECKING:
    from tac.models.tac import TACMemoryResponse


class AuthorInfo(BaseModel):
    """Information about the author of a communication."""

    address: str = Field(..., description="Author address (phone number or identifier)")
    participant_id: str | None = Field(
        default=None, description="Participant ID of the author in the conversation"
    )


class ConversationSession(BaseModel):
    """
    Context information for a conversation session that's passed to callbacks.

    This provides the necessary context for developers to handle memory-ready
    events and send responses back through the appropriate channel.
    """

    conversation_id: str = Field(..., description="Unique conversation identifier")
    profile_id: str | None = Field(
        None, description="Profile ID associated with conversation (optional)"
    )
    channel: str = Field(..., description="Channel type (e.g., 'sms', 'voice')")
    started_at: datetime = Field(
        default_factory=datetime.now,
        description="When the conversation session was started",
    )
    profile: ProfileResponse | None = Field(
        None, description="Profile information with traits (optional)"
    )
    author_info: AuthorInfo | None = Field(
        None, description="Author information from communication event (optional)"
    )
    ai_agent_info: AuthorInfo | None = Field(
        None, description="AI agent information from communication event (optional)"
    )
    metadata: dict = Field(
        default_factory=dict, description="Generic metadata storage for session-specific data"
    )
    pending_handoff_data: PendingHandoffData | None = Field(
        default=None,
        description="Pending handoff payload set by the handoff tool. "
        "Voice channel sends this as a WS 'end' message after the LLM's final response.",
    )
    cached_memory: TACMemoryResponse | None = Field(
        default=None,
        description="Cached memory for 'once' mode. Set on first retrieval, cleared on INACTIVE.",
        exclude=True,
    )
    cache_lock: asyncio.Lock = Field(
        default_factory=asyncio.Lock,
        description="Lock for task-safe cache operations within the event loop in 'once' mode",
        exclude=True,
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def build_profile_prompt(self, trait_groups: list[str] | None = None) -> str | None:
        """
        Build customer profile prompt section for LLM context.

        Args:
            trait_groups: Optional list of trait group names to include.
                         If None, no filtering is applied.

        Returns:
            LLM prompt section with profile data, or None if no profile data
            is available or no traits match the filter.

        Example:
            >>> section = context.build_profile_prompt(["Contact", "Preferences"])
            >>> print(section)
            ## Customer Profile
            Information about this customer:
            - Contact: {"name": "John Doe", "email": "john@example.com"}
            - Preferences: {"language": "en", "timezone": "PST"}
        """
        if not self.profile or not self.profile.traits:
            return None

        # Apply trait group filtering if specified
        if trait_groups is not None:
            filtered_traits = {
                key: value
                for key, value in self.profile.traits.items()
                if key in trait_groups and value is not None
            }
        else:
            # No filtering - include all traits
            filtered_traits = {
                key: value for key, value in self.profile.traits.items() if value is not None
            }

        if not filtered_traits:
            return None

        lines = [
            "## Customer Profile",
            "Information about this customer:",
        ]

        for key, value in filtered_traits.items():
            lines.append(f"- {key}: {value}")

        return "\n".join(lines)

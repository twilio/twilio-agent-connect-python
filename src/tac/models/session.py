from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

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
    call_sid: str | None = Field(
        None,
        description="Twilio Call SID on the Voice channel, None on messaging. The "
        "correlation key for call events (VoiceChannel.on_call_status / on_amd / "
        "on_recording) and end_call. Equals conversation_id in relay-only mode; "
        "look the session up the other way with "
        "VoiceChannel.get_conversation_session_by_call_sid, or, from another "
        "instance, VoiceChannel.resolve_conversation_session_by_call_sid.",
    )
    profile_id: str | None = Field(
        None, description="Profile ID associated with conversation (optional)"
    )
    channel: str = Field(..., description="Channel type (e.g., 'SMS', 'VOICE')")
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
        None,
        description="AI agent information from communication event (optional). "
        "Populated on messaging channels and on Conversation Orchestrator-backed "
        "voice calls. Remains None in ConversationRelay-only voice mode, where "
        "there is no Conversation Orchestrator participant to resolve.",
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

    # Conversation Orchestrator metadata for this conversation, if known, and
    # how to fetch it when it isn't. Set by the channel; read through
    # conversation_metadata(). Private, so never dumped.
    _co_metadata: dict[str, str] | None = PrivateAttr(default=None)
    _co_metadata_loader: Callable[[], Awaitable[dict[str, str] | None]] | None = PrivateAttr(
        default=None
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

    async def conversation_metadata(self) -> dict[str, str]:
        """The metadata stored on this conversation in Conversation Orchestrator.

        `metadata` is this process's own data for the conversation, which
        another replica doesn't see. This reads what's stored on the
        conversation itself — for example the `metadata` passed to
        `initiate_outbound_conversation` — so it works on whichever replica
        handles the turn. It's answered locally when this process already
        knows it; otherwise it costs one Conversation Orchestrator request,
        kept for the rest of this turn.

        Returns a copy, so editing it changes nothing. Returns `{}` when the
        conversation has no metadata or the lookup fails; a failed lookup is
        retried on the next call.

        Example:
            ```python
            async def on_message(text, session, memory):
                metadata = await session.conversation_metadata()
                appointment_id = metadata.get("appointment_id")
            ```
        """
        if self._co_metadata is None and self._co_metadata_loader is not None:
            self._co_metadata = await self._co_metadata_loader()
        return dict(self._co_metadata or {})

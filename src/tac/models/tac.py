"""TAC unified response models."""

from typing import Literal

from pydantic import BaseModel, Field

from tac.models.conversation import Communication, Transcription
from tac.models.memory import (
    MemoryCommunication,
    MemoryRetrievalResponse,
    ObservationInfo,
    SummaryInfo,
)


class TACCommunicationAuthor(BaseModel):
    """
    Unified author model with all fields from both Memory and Conversation Orchestrator APIs.
    """

    # Common fields (both APIs)
    address: str = Field(..., description="Address of the communication author")
    channel: Literal["VOICE", "SMS", "RCS", "EMAIL", "WHATSAPP", "CHAT", "API", "SYSTEM"] = Field(
        ..., description="Channel type"
    )

    # Conversation Orchestrator-only fields
    participant_id: str | None = Field(
        default=None,
        alias="participantId",
        description="Participant ID (Conversation Orchestrator only)",
    )
    delivery_status: (
        Literal["INITIATED", "IN_PROGRESS", "DELIVERED", "COMPLETED", "FAILED"] | None
    ) = Field(
        default=None,
        alias="deliveryStatus",
        description="Delivery status (Conversation Orchestrator recipients only)",
    )

    # Memory-only fields
    id: str | None = Field(default=None, description="Author ID (Memory only)")
    name: str | None = Field(default=None, description="Display name (Memory only)")
    type: Literal["HUMAN_AGENT", "CUSTOMER", "AI_AGENT", "AGENT", "UNKNOWN"] | None = Field(
        default=None, description="Author type (Memory only)"
    )
    profile_id: str | None = Field(
        default=None, alias="profileId", description="Profile ID (Memory only)"
    )

    model_config = {"populate_by_name": True}


class TACCommunicationContent(BaseModel):
    """
    Unified content model with all fields from both Memory and Conversation Orchestrator APIs.
    """

    type: Literal["TEXT", "TRANSCRIPTION"] | None = Field(
        default=None, description="Content type discriminator (Conversation Orchestrator only)"
    )
    text: str | None = Field(default=None, description="Message text content")
    transcription: Transcription | None = Field(
        default=None,
        description=(
            "Transcription metadata (Conversation Orchestrator only, when type=TRANSCRIPTION)"
        ),
    )

    model_config = {"populate_by_name": True}


class TACCommunication(BaseModel):
    """
    Unified communication model with all fields from both Memory and Conversation Orchestrator APIs.

    Provides complete access to all communication fields regardless of the source.
    Fields not available from a particular API will be None.
    """

    # Common fields (both APIs)
    id: str = Field(..., description="Communication identifier")
    author: TACCommunicationAuthor = Field(..., description="Communication author")
    content: TACCommunicationContent = Field(..., description="Communication content")
    recipients: list[TACCommunicationAuthor] = Field(
        default_factory=list, description="Communication recipients"
    )
    channel_id: str | None = Field(
        default=None,
        alias="channelId",
        description="Channel-specific reference ID, when provided by the source API",
    )
    created_at: str | None = Field(
        default=None, alias="createdAt", description="When communication was created"
    )
    updated_at: str | None = Field(
        default=None, alias="updatedAt", description="When communication was last updated"
    )

    # Conversation Orchestrator-only fields
    conversation_id: str | None = Field(
        default=None,
        alias="conversationId",
        description="Conversation ID (Conversation Orchestrator only)",
    )
    account_id: str | None = Field(
        default=None,
        alias="accountId",
        description="Account ID (Conversation Orchestrator only)",
    )

    model_config = {"populate_by_name": True}


class TACMemoryResponse:
    """
    Unified response wrapper for TAC.retrieve_memory().

    Provides a consistent interface for accessing memory data regardless of whether
    Memory API is configured or falling back to Conversation Orchestrator Communications API.

    Memory configured:
    - observations, summaries, communications all populated
    - communications include Memory-specific fields (author id, name, type, profile_id)

    Conversation Orchestrator fallback:
    - observations and summaries are empty lists
    - communications include Conversation Orchestrator-specific fields
      (conversation_id, account_id, etc.)
    """

    def __init__(self, data: MemoryRetrievalResponse | list[Communication]):
        """
        Initialize wrapper with either Memory or Conversation Orchestrator data.

        Args:
            data: Either MemoryRetrievalResponse (Memory) or
                list[Communication] (Conversation Orchestrator)
        """
        self._data = data
        self._is_memory = isinstance(data, MemoryRetrievalResponse)

        # Convert communications once during initialization for better performance
        if self._is_memory:
            memory_data: MemoryRetrievalResponse = data  # type: ignore[assignment]
            memory_comms = memory_data.communications or []
            self._communications = [self._convert_communication(comm) for comm in memory_comms]
        else:
            orchestrator_comms: list[Communication] = data  # type: ignore[assignment]
            self._communications = [
                self._convert_communication(comm) for comm in orchestrator_comms
            ]

    @property
    def observations(self) -> list[ObservationInfo]:
        """
        Get observation memories.

        Returns:
            List of observations if Memory is configured,
            empty list for Conversation Orchestrator fallback
        """
        if self._is_memory:
            return self._data.observations  # type: ignore[union-attr]
        return []

    @property
    def summaries(self) -> list[SummaryInfo]:
        """
        Get summary memories.

        Returns:
            List of summaries if Memory is configured,
            empty list for Conversation Orchestrator fallback
        """
        if self._is_memory:
            return self._data.summaries  # type: ignore[union-attr]
        return []

    @property
    def communications(self) -> list[TACCommunication]:
        """
        Get communications in unified format with all available fields.

        Communications are converted to a common format during initialization that includes
        all fields from both Memory and Conversation Orchestrator APIs.
        Fields not available from a particular
        API will be None.

        Returns:
            List of unified communications with all available fields
        """
        return self._communications

    @property
    def has_memory_features(self) -> bool:
        """
        Check if Memory API is configured and providing full features.

        Returns:
            True if Memory is configured (observations/summaries available),
            False if using Conversation Orchestrator fallback (only communications available)
        """
        return self._is_memory

    @property
    def raw_data(self) -> MemoryRetrievalResponse | list[Communication]:
        """
        Access raw underlying data for advanced use cases.

        Use this when you need access to all fields from the original API responses,
        not just the simplified common fields.

        Returns:
            Either MemoryRetrievalResponse or list[Communication] depending on configuration
        """
        return self._data

    def build_memory_prompts(self) -> list[str]:
        """
        Build all memory prompt sections (observations, summaries, communications) for LLM context.

        Returns:
            List of LLM prompt sections. Each element is a complete section
            (e.g., observations section, summaries section). Returns empty list
            if no memory data is available.

        Example:
            >>> sections = memory_response.build_memory_prompts()
            >>> for section in sections:
            ...     print(section)
            ...     print()
            ## Key Observations
            Important notes about the customer from previous interactions:
            - Customer prefers email communication
            - Previously reported billing issue (resolved)

            ## Past Conversation Summaries
            Summaries of previous conversations with this customer:
            - Discussed product features and pricing on 2024-01-15
        """
        sections = []

        # Build observations prompt section
        observations_section = self._build_observations_prompt()
        if observations_section:
            sections.append(observations_section)

        # Build summaries prompt section
        summaries_section = self._build_summaries_prompt()
        if summaries_section:
            sections.append(summaries_section)

        # Build communications prompt section
        communications_section = self._build_communications_prompt()
        if communications_section:
            sections.append(communications_section)

        return sections

    def _build_observations_prompt(self) -> str | None:
        """Build observations LLM prompt section."""
        if not self.observations:
            return None

        lines = [
            "## Key Observations",
            "Important notes about the customer from previous interactions:",
        ]

        for obs in self.observations:
            lines.append(f"- {obs.content}")

        return "\n".join(lines)

    def _build_summaries_prompt(self) -> str | None:
        """Build summaries LLM prompt section."""
        if not self.summaries:
            return None

        lines = [
            "## Past Conversation Summaries",
            "Summaries of previous conversations with this customer:",
        ]

        for summary in self.summaries:
            lines.append(f"- {summary.content}")

        return "\n".join(lines)

    def _build_communications_prompt(self) -> str | None:
        """Build communications LLM prompt section."""
        if not self.communications:
            return None

        lines = [
            "## Recent Message History",
            "Recent messages exchanged with this customer:",
        ]

        for comm in self.communications:
            # Determine role - defaults to "Assistant" for None or non-CUSTOMER authors
            role = "User" if comm.author and comm.author.type == "CUSTOMER" else "Assistant"
            content = comm.content.text or ""
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _convert_communication(self, comm: MemoryCommunication | Communication) -> TACCommunication:
        """
        Convert Memory or Conversation Orchestrator communication to unified format.

        Pydantic automatically handles missing fields by setting them to None.
        """
        data = comm.model_dump(by_alias=True)

        data["author"] = TACCommunicationAuthor(**data["author"])
        data["content"] = TACCommunicationContent(**data["content"])
        data["recipients"] = [TACCommunicationAuthor(**r) for r in data["recipients"]]

        return TACCommunication(**data)

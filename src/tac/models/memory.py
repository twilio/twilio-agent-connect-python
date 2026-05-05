from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Memory retrieval mode for channels
MemoryMode = Literal["always", "never", "once"]


class MemoryRetrievalRequest(BaseModel):
    """Request payload for retrieving conversation memories."""

    conversation_id: str | None = Field(
        default=None,
        alias="conversationId",
        description="A unique identifier for the conversation using Twilio Type ID (TTID) format",
        json_schema_extra={"example": "comms_conversation_00000000000000000000000000"},
    )
    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=1024,
        description="Semantic search query for finding relevant memories",
        json_schema_extra={"example": "customer satisfaction feedback"},
    )
    begin_date: str | None = Field(
        default=None,
        alias="beginDate",
        max_length=30,
        description="Start date for filtering memories (inclusive)",
        json_schema_extra={"example": "2025-01-01T00:00:00Z"},
    )
    end_date: str | None = Field(
        default=None,
        alias="endDate",
        max_length=30,
        description="End date for filtering memories (exclusive)",
        json_schema_extra={"example": "2025-01-31T23:59:59Z"},
    )
    observations_limit: int | None = Field(
        default=20,
        alias="observationsLimit",
        ge=0,
        le=100,
        description="Max observations to return (0-100). Set to 0 to disable.",
        json_schema_extra={"example": 20},
    )
    summaries_limit: int | None = Field(
        default=5,
        alias="summariesLimit",
        ge=0,
        le=100,
        description="Max summaries to return (0-100). Set to 0 to disable.",
        json_schema_extra={"example": 5},
    )
    communications_limit: int | None = Field(
        default=0,
        alias="communicationsLimit",
        ge=0,
        le=100,
        description="Max communications to return (0-100). Set to 0 to disable.",
        json_schema_extra={"example": 0},
    )
    relevance_threshold: float | None = Field(
        default=0.0,
        alias="relevanceThreshold",
        ge=0.0,
        le=1.0,
        description=(
            "Min relevance score (0.0-1.0). Only applies when results are ranked by relevance."
        ),
        json_schema_extra={"example": 0.5},
    )

    model_config = {"populate_by_name": True}


class MemoryParticipant(BaseModel):
    """Participant in a Memory communication (author or recipient)."""

    id: str = Field(
        ...,
        description="Participant identifier",
        json_schema_extra={"example": "conv_participant_00000000000000000000000000"},
    )
    name: str = Field(
        ...,
        max_length=256,
        description="Participant display name",
    )
    address: str = Field(
        ...,
        max_length=254,
        description="Address of the Participant (e.g., phone number, email address)",
        json_schema_extra={"example": "+12025551234"},
    )
    channel: Literal["VOICE", "SMS", "RCS", "EMAIL", "WHATSAPP", "CHAT", "API", "SYSTEM"] = Field(
        ..., description="The channel on which the message originated"
    )
    type: Literal["HUMAN_AGENT", "CUSTOMER", "AI_AGENT", "AGENT", "UNKNOWN"] | None = Field(
        default=None,
        description="Type of Participant in the Conversation",
    )
    profile_id: str | None = Field(
        default=None,
        alias="profileId",
        description="The canonical profile ID",
        json_schema_extra={"example": "mem_profile_00000000000000000000000000"},
    )
    delivery_status: (
        Literal["INITIATED", "IN_PROGRESS", "DELIVERED", "COMPLETED", "FAILED"] | None
    ) = Field(
        default=None,
        alias="deliveryStatus",
        description="Delivery status of the Communication to this recipient (only for recipients)",
    )

    model_config = {"populate_by_name": True}


class MemoryCommunicationContent(BaseModel):
    """Content of a Memory communication."""

    text: str | None = Field(
        default=None,
        max_length=8388608,
        description="Primary text content (optional)",
        json_schema_extra={"example": "Hello, I need help with my account"},
    )

    model_config = {"populate_by_name": True}


class MemoryCommunication(BaseModel):
    """A communication from Memory API (historical conversation data)."""

    id: str = Field(
        ...,
        description="Unique communication identifier",
        json_schema_extra={"example": "conv_communication_00000000000000000000000000"},
    )
    author: MemoryParticipant = Field(..., description="Author of the communication")
    content: MemoryCommunicationContent = Field(..., description="Content of the communication")
    recipients: list[MemoryParticipant] = Field(
        ..., max_length=100, description="Communication recipients"
    )
    channel_id: str | None = Field(
        default=None,
        alias="channelId",
        max_length=256,
        description="Channel-specific ID (optional)",
        json_schema_extra={"example": "SM00000000000000000000000000000000"},
    )
    created_at: str = Field(
        ...,
        alias="createdAt",
        description="When communication was created",
        json_schema_extra={"example": "2025-01-15T10:15:30Z"},
    )
    updated_at: str | None = Field(
        default=None,
        alias="updatedAt",
        description="When communication was last updated",
        json_schema_extra={"example": "2025-01-15T10:20:30Z"},
    )

    model_config = {"populate_by_name": True}


class CiOperator(BaseModel):
    """Information about the Conversational Intelligence operator."""

    ci_service_id: str = Field(
        ...,
        alias="ciServiceId",
        max_length=34,
        description="SID of the Conversational Intelligence Service",
        json_schema_extra={"example": "GA00000000000000000000000000000000"},
    )
    id: str = Field(
        ...,
        max_length=34,
        description="ID of the language operator that extracted this observation",
        json_schema_extra={"example": "LY00000000000000000000000000000000"},
    )
    version: str = Field(
        ...,
        min_length=5,
        max_length=50,
        description="Version of the language operator that extracted this observation",
        json_schema_extra={"example": "1.2.3"},
    )

    model_config = {"populate_by_name": True}


class ObservationInfo(BaseModel):
    """An observation memory from the API response."""

    content: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The main content of the observation",
        json_schema_extra={
            "example": "Customer expressed satisfaction with recent product update."
        },
    )
    source: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Source system that generated this observation",
        json_schema_extra={"example": "conversational-intelligence"},
    )
    id: str = Field(
        ...,
        description="Unique identifier for the observation using Twilio Type ID (TTID) format",
        json_schema_extra={"example": "mem_observation_00000000000000000000000000"},
    )
    created_at: str = Field(
        ...,
        alias="createdAt",
        max_length=30,
        description="Timestamp when the observation was created",
        json_schema_extra={"example": "2025-01-15T10:30:45Z"},
    )
    updated_at: str = Field(
        ...,
        alias="updatedAt",
        max_length=30,
        description="Timestamp when the observation was last updated",
        json_schema_extra={"example": "2025-01-15T10:30:45Z"},
    )
    occurred_at: str | None = Field(
        default=None,
        alias="occurredAt",
        max_length=30,
        description="Timestamp when the observation originally occurred",
        json_schema_extra={"example": "2025-01-15T10:15:30Z"},
    )
    conversation_ids: list[str] | None = Field(
        default=None,
        alias="conversationIds",
        max_length=10,
        description="Array of conversation IDs associated with this observation",
        json_schema_extra={"example": ["comms_conversation_00000000000000000000000000"]},
    )

    model_config = {"populate_by_name": True}


class SummaryInfo(BaseModel):
    """A summary memory derived from observations at the end of conversations."""

    content: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The main content of the summary",
        json_schema_extra={
            "example": "Customer discussed billing concerns and was satisfied with resolution."
        },
    )
    conversation_id: str | None = Field(
        default=None,
        alias="conversationId",
        description="Unique identifier for the conversation using Twilio Type ID (TTID) format",
        json_schema_extra={"example": "comms_conversation_00000000000000000000000000"},
    )
    id: str = Field(
        ...,
        description="Unique identifier for the summary using Twilio Type ID (TTID) format",
        json_schema_extra={"example": "mem_summary_00000000000000000000000000"},
    )
    created_at: str = Field(
        ...,
        alias="createdAt",
        max_length=30,
        description="Timestamp when the summary was created",
        json_schema_extra={"example": "2025-01-15T10:30:45Z"},
    )
    updated_at: str = Field(
        ...,
        alias="updatedAt",
        max_length=30,
        description="Timestamp when the summary was last updated",
        json_schema_extra={"example": "2025-01-15T10:30:45Z"},
    )
    source: str | None = Field(
        default=None,
        max_length=100,
        description="Source system that generated the summary",
        json_schema_extra={"example": "conversations"},
    )
    occurred_at: str | None = Field(
        default=None,
        alias="occurredAt",
        max_length=30,
        description="Timestamp when the summary was originally created",
        json_schema_extra={"example": "2025-01-15T10:15:30Z"},
    )

    model_config = {"populate_by_name": True}


class MemoryRetrievalMeta(BaseModel):
    """Metadata about the memory retrieval operation."""

    query_time: int | None = Field(
        default=None,
        alias="queryTime",
        ge=0,
        le=600000,
        description="Query execution time in milliseconds",
        json_schema_extra={"example": 156},
    )

    model_config = {"populate_by_name": True}


class MemoryRetrievalResponse(BaseModel):
    """Response from the Memory API /Recall endpoint."""

    observations: list[ObservationInfo] = Field(
        default_factory=list, max_length=100, description="Array of observation memories"
    )
    summaries: list[SummaryInfo] = Field(
        default_factory=list,
        max_length=100,
        description="Array of summary memories from end of conversations",
    )
    communications: list[MemoryCommunication] = Field(
        default_factory=list,
        max_length=100,
        description="Array of communication memories from Memory API",
    )
    meta: MemoryRetrievalMeta = Field(
        default_factory=MemoryRetrievalMeta, description="Metadata about the retrieval operation"
    )

    model_config = {"populate_by_name": True}


class ProfileResponse(BaseModel):
    """Response from the profile retrieval API."""

    id: str = Field(
        ...,
        description="Unique identifier for the profile",
        json_schema_extra={"example": "mem_profile_00000000000000000000000000"},
    )
    created_at: str = Field(
        ...,
        alias="createdAt",
        max_length=30,
        description="Timestamp when the profile was created",
        json_schema_extra={"example": "2025-01-15T10:30:45Z"},
    )
    traits: dict[str, Any] = Field(
        ...,
        description="Profile traits organized by trait groups",
        json_schema_extra={
            "example": {
                "Contact": {
                    "firstName": "Alyssa",
                    "lastName": "Mock",
                    "address": {
                        "street": "123 Main St",
                        "city": "San Francisco",
                        "state": "CA",
                        "postalCode": "94107",
                        "country": "US",
                    },
                }
            }
        },
    )

    model_config = {"populate_by_name": True}


class ProfileLookupRequest(BaseModel):
    """Request payload for looking up profiles by identifier."""

    id_type: str = Field(
        ...,
        alias="idType",
        min_length=2,
        max_length=30,
        description="Identifier type as configured in the service's Identity Resolution Settings",
        json_schema_extra={"example": "phone"},
    )
    value: str = Field(
        ...,
        max_length=255,
        description="Raw value captured for the identifier",
        json_schema_extra={"example": "+13175556789"},
    )

    model_config = {"populate_by_name": True}


class ProfileLookupResponse(BaseModel):
    """Response from the profile lookup API."""

    normalized_value: str = Field(
        ...,
        alias="normalizedValue",
        max_length=255,
        description="Identifier value after normalization that was used for the lookup",
        json_schema_extra={"example": "+13175556789"},
    )
    profiles: list[str] = Field(
        default_factory=list,
        max_length=100,
        description="Array of profile IDs matching the identifier",
        json_schema_extra={"example": ["mem_profile_00000000000000000000000000"]},
    )

    model_config = {"populate_by_name": True}

    @field_validator("profiles", mode="before")
    @classmethod
    def _coerce_null_profiles(cls, v: Any) -> Any:
        # Conversation Memory's Lookup response omits or nulls `profiles` when no match;
        # treat both as [].
        return [] if v is None else v

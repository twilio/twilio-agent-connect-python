"""Pydantic models for Twilio Conversation Orchestrator API."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from tac.models.pagination import PaginationMeta


class StatusTimeouts(BaseModel):
    """Timeout settings for channel status transitions."""

    inactive: int | None = Field(None, ge=1, description="Inactivity timeout in minutes")
    closed: int = Field(..., ge=1, description="Close timeout in minutes")

    model_config = {"populate_by_name": True}


class CaptureRule(BaseModel):
    """Capture rule with from/to addresses and optional metadata."""

    from_address: str = Field(
        ...,
        alias="from",
        description=(
            "From address (phone number, email, etc.). "
            "Use '*' for wildcard to match any from address"
        ),
    )
    to_address: str = Field(
        ...,
        alias="to",
        description=(
            "To address (phone number, email, etc.). Use '*' for wildcard to match any to address"
        ),
    )
    metadata: dict[str, str] | None = Field(
        None,
        description=(
            "Additional matching criteria for the capture rule. "
            "For voice calls, can include 'callType' (PSTN, SIP, etc.)"
        ),
    )

    model_config = {"populate_by_name": True}


class ChannelSettings(BaseModel):
    """Configuration settings for a specific channel type."""

    status_timeouts: StatusTimeouts | None = Field(
        None,
        alias="statusTimeouts",
        description="Timeout settings for channel status transitions",
    )
    capture_rules: list[CaptureRule] | None = Field(
        None,
        alias="captureRules",
        description=(
            "Array of capture rules with from/to addresses and optional metadata. "
            "Use '*' for wildcard matching in either direction"
        ),
    )

    model_config = {"populate_by_name": True}


class StatusCallback(BaseModel):
    """Webhook configuration for status callbacks."""

    url: str = Field(..., description="Destination URL for webhooks")
    method: Literal["POST", "GET", "PUT", "DELETE", "PATCH"] | None = Field(
        default="POST",
        description="HTTP method used to invoke the webhook URL",
    )

    model_config = {"populate_by_name": True}


class ParticipantAddress(BaseModel):
    """Communication address for a conversation participant."""

    channel: Literal["VOICE", "SMS", "RCS", "EMAIL", "WHATSAPP", "CHAT", "API", "SYSTEM"] = Field(
        ..., description="The channel for Communication (VOICE, SMS, EMAIL, etc.)"
    )
    address: str = Field(..., description="The address value (phone number, email, etc.)")
    channel_id: str | None = Field(
        default=None,
        alias="channelId",
        description="Channel-specific ID for correlating Communications",
    )

    model_config = {"populate_by_name": True}


class ConversationConfiguration(BaseModel):
    """Configuration settings for a conversation response."""

    id: str = Field(..., description="Configuration ID")
    display_name: str | None = Field(
        None,
        alias="displayName",
        max_length=32,
        pattern=r"^[a-zA-Z0-9-_ ]+$",
        description="A human-readable name for the configuration. Limited to 32 characters.",
    )
    description: str = Field(
        ...,
        description=(
            "Human-readable description for the configuration. "
            "Allows spaces and special characters, typically limited to a paragraph of text. "
            "This serves as a descriptive field rather than just a name."
        ),
    )
    conversation_grouping_type: Literal[
        "GROUP_BY_PROFILE",
        "GROUP_BY_PARTICIPANT_ADDRESSES",
        "GROUP_BY_PARTICIPANT_ADDRESSES_AND_CHANNEL_TYPE",
    ] = Field(
        ...,
        alias="conversationGroupingType",
        description=(
            "Type of Conversation grouping strategy:\n"
            "- GROUP_BY_PROFILE: Groups communications by participant profile. "
            "Communications with the same profile go to the same conversation, regardless of "
            "the channel or address.\n"
            "- GROUP_BY_PARTICIPANT_ADDRESSES: Groups communications by participant addresses "
            "across all channels. A customer using +15551234567 will be in the same conversation "
            "whether they contact via SMS, WhatsApp, or RCS.\n"
            "- GROUP_BY_PARTICIPANT_ADDRESSES_AND_CHANNEL_TYPE: Groups communications by both "
            "participant addresses AND channel. A customer using +15551234567 via SMS will be in "
            "a different conversation than the same customer via WhatsApp."
        ),
    )
    memory_store_id: str = Field(
        ...,
        alias="memoryStoreId",
        description="Memory Store ID for Profile Resolution using Twilio Type ID (TTID) format",
    )
    channel_settings: dict[str, ChannelSettings] | None = Field(
        None,
        alias="channelSettings",
        description=(
            "Channel-specific configuration settings including timeout settings and capture rules"
        ),
    )
    status_callbacks: list[StatusCallback] | None = Field(
        None,
        alias="statusCallbacks",
        max_length=20,
        description=(
            "List of default webhook configurations applied to "
            "conversations under this configuration"
        ),
    )
    intelligence_configuration_ids: list[str] | None = Field(
        None,
        alias="intelligenceConfigurationIds",
        max_length=5,
        description="List of Intelligence Configuration IDs for this configuration",
    )
    created_at: str | None = Field(
        None, alias="createdAt", description="Timestamp when this configuration was created"
    )
    updated_at: str | None = Field(
        None, alias="updatedAt", description="Timestamp when this configuration was last updated"
    )
    version: int | None = Field(None, description="Version number used for optimistic locking")

    model_config = {"populate_by_name": True}


class ConversationRequest(BaseModel):
    """Request payload for creating a conversation."""

    configuration_id: str = Field(
        ...,
        alias="configurationId",
        description="Configuration ID settings to use for this conversation",
    )
    name: str | None = Field(default=None, description="Conversation name")
    participants: list["ParticipantRequest"] | None = Field(
        default=None,
        description="Optional inline participants created atomically with the conversation",
    )

    model_config = {"populate_by_name": True}


class UpdateConversationRequest(BaseModel):
    """Request payload for updating a conversation."""

    status: Literal["ACTIVE", "INACTIVE", "CLOSED"] = Field(
        ..., description="Conversation state (ACTIVE/INACTIVE/CLOSED)"
    )
    name: str | None = Field(default=None, description="Conversation name")

    model_config = {"populate_by_name": True}


class ConversationResponse(BaseModel):
    """Response from creating a conversation."""

    id: str = Field(..., description="Conversation ID")
    account_id: str = Field(..., alias="accountId", description="Twilio Account SID")

    status: Literal["ACTIVE", "INACTIVE", "CLOSED"] | None = Field(
        None, description="Conversation status"
    )
    name: str | None = Field(None, description="Conversation name")
    configuration_id: str | None = Field(
        None, alias="configurationId", description="Configuration used to create this conversation"
    )
    created_at: str | None = Field(None, alias="createdAt", description="Creation timestamp")
    updated_at: str | None = Field(None, alias="updatedAt", description="Last update timestamp")

    model_config = {"populate_by_name": True}


class ParticipantRequest(BaseModel):
    """Request payload for creating a conversation participant."""

    name: str | None = Field(default=None, description="Display name for the Participant")
    type: Literal["HUMAN_AGENT", "CUSTOMER", "AI_AGENT", "AGENT", "UNKNOWN"] | None = Field(
        default=None, description="Type of Participant in the Conversation"
    )
    profile_id: str | None = Field(
        default=None, alias="profileId", description="Resolved segment profile"
    )
    addresses: list[ParticipantAddress] | None = Field(
        default_factory=list, description="List of Communication addresses for the Participant"
    )

    model_config = {"populate_by_name": True}


class ParticipantResponse(BaseModel):
    """Response from creating a participant."""

    id: str = Field(..., description="Participant ID")
    conversation_id: str = Field(..., alias="conversationId", description="Conversation ID")
    account_id: str = Field(..., alias="accountId", description="Account ID")
    name: str = Field(..., description="Participant display name")
    type: Literal["HUMAN_AGENT", "CUSTOMER", "AI_AGENT", "AGENT", "UNKNOWN"] | None = Field(
        None, description="Type of Participant in the Conversation"
    )
    profile_id: str | None = Field(None, alias="profileId", description="Segment profile ID")
    addresses: list[ParticipantAddress] = Field(
        default_factory=list, description="Communication addresses for this Participant"
    )
    created_at: str | None = Field(
        None, alias="createdAt", description="Timestamp when this Participant was created"
    )
    updated_at: str | None = Field(
        None, alias="updatedAt", description="Timestamp when this Participant was last updated"
    )

    model_config = {"populate_by_name": True}


class CommunicationParticipant(BaseModel):
    """Author or recipient in a communication."""

    address: str = Field(
        ...,
        max_length=254,
        description="Address of the participant formatted according to channel type",
        json_schema_extra={"example": "+12025551234"},
    )
    channel: Literal["VOICE", "SMS", "RCS", "EMAIL", "WHATSAPP", "CHAT", "API", "SYSTEM"] = Field(
        ..., description="Channel type for the participant address"
    )
    participant_id: str = Field(
        ...,
        alias="participantId",
        description="Participant ID associated with this address",
        json_schema_extra={"example": "comms_participant_00000000000000000000000000"},
    )
    delivery_status: (
        Literal["INITIATED", "IN_PROGRESS", "DELIVERED", "COMPLETED", "FAILED"] | None
    ) = Field(
        default=None,
        alias="deliveryStatus",
        description="Delivery status of the Communication to this recipient (recipients only)",
    )

    model_config = {"populate_by_name": True}


class TranscriptionWord(BaseModel):
    """Word-level transcription data with timing information."""

    text: str = Field(..., description="The transcribed word")
    start_time: str | None = Field(
        default=None,
        alias="startTime",
        description="Start timestamp of this word",
        json_schema_extra={"example": "2025-01-15T10:15:30.123Z"},
    )
    end_time: str | None = Field(
        default=None,
        alias="endTime",
        description="End timestamp of this word",
        json_schema_extra={"example": "2025-01-15T10:15:30.456Z"},
    )

    model_config = {"populate_by_name": True}


class Transcription(BaseModel):
    """Transcription metadata for communication content."""

    channel: int | None = Field(
        default=None,
        description="Audio channel identifier (0 for inbound, 1 for outbound)",
        json_schema_extra={"example": 0},
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall confidence score for the transcription (0.0-1.0)",
        json_schema_extra={"example": 0.95},
    )
    engine: str | None = Field(
        default=None,
        description="Transcription engine used",
        json_schema_extra={"example": "google"},
    )
    words: list[TranscriptionWord] | None = Field(
        default=None, description="Word-level transcription data with timing information"
    )

    model_config = {"populate_by_name": True}


class CommunicationContent(BaseModel):
    """Content of a communication (ContentText or ContentTranscription)."""

    type: Literal["TEXT", "TRANSCRIPTION"] = Field(..., description="Content type discriminator")
    text: str = Field(
        ...,
        max_length=8388608,
        description="Message text content",
        json_schema_extra={"example": "Hello, I need help with my account"},
    )
    transcription: Transcription | None = Field(
        default=None, description="Transcription metadata (only present when type=TRANSCRIPTION)"
    )

    model_config = {"populate_by_name": True}


class Communication(BaseModel):
    """A communication representing a message exchanged in a conversation."""

    id: str = Field(
        ...,
        description="Unique communication identifier",
        json_schema_extra={"example": "comms_communication_00000000000000000000000000"},
    )
    conversation_id: str = Field(..., alias="conversationId", description="Conversation ID")
    account_id: str = Field(..., alias="accountId", description="Account ID")
    author: CommunicationParticipant = Field(..., description="Author of the communication")
    content: CommunicationContent = Field(
        ..., description="The content of the Communication using type field for discrimination"
    )
    recipients: list[CommunicationParticipant] = Field(
        default_factory=list, description="Communication recipients"
    )
    channel_id: str | None = Field(
        default=None,
        alias="channelId",
        description="Channel-specific reference ID",
        json_schema_extra={"example": "SM00000000000000000000000000000000"},
    )
    created_at: str | None = Field(
        default=None,
        alias="createdAt",
        max_length=30,
        description="Timestamp when this Communication was created",
        json_schema_extra={"example": "2025-01-15T10:15:30Z"},
    )
    updated_at: str | None = Field(
        default=None,
        alias="updatedAt",
        max_length=30,
        description="Timestamp when this Communication was last updated",
        json_schema_extra={"example": "2025-01-15T10:20:30Z"},
    )

    model_config = {"populate_by_name": True}


class CommunicationRequest(BaseModel):
    """Request payload for adding a communication."""

    author: CommunicationParticipant = Field(..., description="Author of the communication")
    content: CommunicationContent = Field(..., description="Content of the communication")
    recipients: list[CommunicationParticipant] = Field(
        ..., description="List of recipients for the communication"
    )
    channel_id: str | None = Field(
        default=None,
        alias="channelId",
        description="Store Call ID/Record ID/etc. for channel reference",
    )

    model_config = {"populate_by_name": True}


class CommunicationsListResponse(BaseModel):
    """Response from list communications endpoint."""

    communications: list[Communication] = Field(..., description="List of communications")
    meta: PaginationMeta = Field(..., description="Pagination metadata")

    model_config = {"populate_by_name": True}


class ConversationsListResponse(BaseModel):
    """Response from list conversations endpoint."""

    conversations: list[ConversationResponse] = Field(..., description="List of conversations")
    meta: PaginationMeta = Field(..., description="Pagination metadata")

    model_config = {"populate_by_name": True}


class ActionParticipantRef(BaseModel):
    """Participant reference for the Actions API (`from`/`to` entries).

    Either `participant_id` or `address` must be supplied; `channel` is always required.
    When both are provided, Conversation Orchestrator uses `participant_id` and
    `channel` disambiguates
    which of the participant's addresses to use.
    """

    participant_id: str | None = Field(
        default=None, alias="participantId", min_length=1, description="Participant ID"
    )
    address: str | None = Field(
        default=None, min_length=1, max_length=254, description="Participant address"
    )
    channel: Literal["VOICE", "SMS", "RCS", "EMAIL", "WHATSAPP", "CHAT", "API", "SYSTEM"] = Field(
        ..., description="Channel type"
    )

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _require_participant_id_or_address(self) -> "ActionParticipantRef":
        if not self.participant_id and not self.address:
            raise ValueError("ActionParticipantRef requires at least `participant_id` or `address`")
        return self


class ActionTextContent(BaseModel):
    """Plain-text content for a SEND_MESSAGE action."""

    text: str = Field(..., max_length=8388608, description="Message text content")

    model_config = {"populate_by_name": True}


class ActionChannelSettings(BaseModel):
    """Channel-specific settings forwarded to the downstream backend.

    Open pass-through: any field not explicitly modeled here (e.g.
    `messagingServiceSid`, `statusCallback`, `Attributes`) can be set by callers and
    will be forwarded as-is.
    """

    channel_id: str | None = Field(
        default=None,
        alias="channelId",
        description="Backend-specific channel identifier (e.g. V1 Chat channel SID)",
    )

    model_config = {"populate_by_name": True, "extra": "allow"}


class SendMessageActionPayload(BaseModel):
    """Inner payload for a SEND_MESSAGE action."""

    from_: ActionParticipantRef = Field(..., alias="from", description="Sender")
    to: list[ActionParticipantRef] = Field(..., min_length=1, description="Recipients (minimum 1)")
    content: ActionTextContent = Field(..., description="Message content")
    channel_settings: ActionChannelSettings | None = Field(
        default=None,
        alias="channelSettings",
        description="Channel-specific pass-through settings",
    )

    model_config = {"populate_by_name": True}


class SendMessageActionRequest(BaseModel):
    """Request for POST /v2/Conversations/{id}/Actions with type=SEND_MESSAGE.

    Body is discriminated by `type` with the action-specific fields under `payload`.
    """

    type: Literal["SEND_MESSAGE"] = Field(default="SEND_MESSAGE", description="Action type")
    payload: SendMessageActionPayload = Field(..., description="SEND_MESSAGE payload")

    model_config = {"populate_by_name": True}


class ActionResponse(BaseModel):
    """Response from POST /v2/Conversations/{id}/Actions (202 Accepted)."""

    id: str = Field(..., description="Action ID")
    type: str = Field(
        ...,
        description=(
            "Action type. Known values: SEND_MESSAGE. Kept as str to tolerate future additions."
        ),
    )
    status: str = Field(
        ...,
        description=(
            "Current action status. Known values: PENDING, COMPLETED, FAILED. "
            "Kept as str to tolerate future additions."
        ),
    )
    conversation_id: str = Field(..., alias="conversationId", description="Conversation ID")
    created_at: str | None = Field(
        default=None, alias="createdAt", description="Action creation timestamp"
    )

    model_config = {"populate_by_name": True}

"""Configuration models for the Twilio Agent Connect."""

import os
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConversationIntelligenceConfig(BaseModel):
    """
    Configuration for Conversation Intelligence webhook filtering.

    This config specifies which CI configuration and operators to process.
    Events that don't match are filtered out.
    """

    configuration_id: str = Field(
        description="Conversation Intelligence Configuration ID",
    )
    observation_operator_sid: str | None = Field(
        default=None,
        description="Operator SID for observation extraction (e.g., LY...)",
    )
    summary_operator_sid: str | None = Field(
        default=None,
        description="Operator SID for summary extraction (e.g., LY...)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "configuration_id": "your_ci_configuration_id",
                "observation_operator_sid": "LY00000000000000000000000000000001",
                "summary_operator_sid": "LY00000000000000000000000000000002",
            }
        },
    )

    @classmethod
    def from_env(cls) -> "ConversationIntelligenceConfig | None":
        """Create ConversationIntelligenceConfig from CONVERSATION_INTELLIGENCE_* env vars."""
        configuration_id = os.environ.get("CONVERSATION_INTELLIGENCE_CONFIGURATION_ID")

        if not configuration_id:
            return None

        return cls(
            configuration_id=configuration_id,
            observation_operator_sid=os.environ.get(
                "CONVERSATION_INTELLIGENCE_OBSERVATION_OPERATOR_SID"
            ),
            summary_operator_sid=os.environ.get("CONVERSATION_INTELLIGENCE_SUMMARY_OPERATOR_SID"),
        )


class TwilioMemoryConfig(BaseModel):
    """
    Configuration for Twilio Memory Store integration.

    Controls memory retrieval limits, relevance filtering, and profile trait groups.
    Memory client is auto-initialized from Conversation Orchestrator configuration.
    """

    trait_groups: list[str] | None = Field(
        default=None,
        description=(
            "Trait groups to include when retrieving profiles. "
            "If None, all trait groups are included."
        ),
    )

    observations_limit: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Max observations to return (0-100). Set to 0 to disable.",
    )

    summaries_limit: int = Field(
        default=5,
        ge=0,
        le=100,
        description="Max summaries to return (0-100). Set to 0 to disable.",
    )

    communications_limit: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Max communications to return (0-100). Set to 0 to disable.",
    )

    relevance_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Min relevance score for observations and summaries (0.0-1.0).",
    )

    phone_trait_group: str = Field(
        default="Contact",
        description="Trait group name that holds the phone identifier on newly created profiles. "
        "Must match the promoted-to-identifier configuration of the Conversation Memory store.",
    )
    phone_trait_field: str = Field(
        default="phone",
        description="Trait field name within phone_trait_group that holds the phone identifier.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trait_groups": ["Contact", "Preferences"],
                "phone_trait_group": "Contact",
                "phone_trait_field": "phone",
                "observations_limit": 20,
                "summaries_limit": 5,
                "communications_limit": 0,
                "relevance_threshold": 0.0,
            }
        },
    )

    @classmethod
    def from_env(cls) -> "TwilioMemoryConfig":
        """Create TwilioMemoryConfig from TWILIO_MEMORY_* environment variables."""
        # Get defaults from model to avoid duplication
        defaults = cls()

        trait_groups_str = os.environ.get("TWILIO_MEMORY_PROFILE_TRAIT_GROUPS")

        # Parse trait groups from environment variable, filtering out empty strings
        trait_groups = None
        if trait_groups_str:
            parsed_groups = [g.strip() for g in trait_groups_str.split(",") if g.strip()]
            trait_groups = parsed_groups or None

        # Parse retrieval limits from environment variables
        # Treat empty strings and whitespace-only strings as unset so model defaults
        # are applied consistently.
        # Note: Pydantic will validate that limits are 0-100 and threshold is 0.0-1.0
        def _get_int_env(name: str, default: int) -> int:
            value = os.environ.get(name)
            if value is None:
                return default
            value = value.strip()
            if not value:
                return default
            return int(value)

        def _get_float_env(name: str, default: float) -> float:
            value = os.environ.get(name)
            if value is None:
                return default
            value = value.strip()
            if not value:
                return default
            return float(value)

        try:
            observations_limit = _get_int_env(
                "TWILIO_MEMORY_OBSERVATIONS_LIMIT", defaults.observations_limit
            )
            summaries_limit = _get_int_env(
                "TWILIO_MEMORY_SUMMARIES_LIMIT", defaults.summaries_limit
            )
            communications_limit = _get_int_env(
                "TWILIO_MEMORY_COMMUNICATIONS_LIMIT", defaults.communications_limit
            )
            relevance_threshold = _get_float_env(
                "TWILIO_MEMORY_RELEVANCE_THRESHOLD", defaults.relevance_threshold
            )
        except ValueError as e:
            raise ValueError(
                "Invalid memory configuration in environment variables. "
                "Ensure TWILIO_MEMORY_*_LIMIT values are integers "
                f"and TWILIO_MEMORY_RELEVANCE_THRESHOLD is a float. Error: {e}"
            ) from e

        phone_trait_group = (
            os.environ.get("TWILIO_MEMORY_PHONE_TRAIT_GROUP") or defaults.phone_trait_group
        )
        phone_trait_field = (
            os.environ.get("TWILIO_MEMORY_PHONE_TRAIT_FIELD") or defaults.phone_trait_field
        )

        return cls(
            trait_groups=trait_groups,
            observations_limit=observations_limit,
            summaries_limit=summaries_limit,
            communications_limit=communications_limit,
            relevance_threshold=relevance_threshold,
            phone_trait_group=phone_trait_group,
            phone_trait_field=phone_trait_field,
        )


class TACConfig(BaseModel):
    """Configuration model for Twilio Agent Connect settings."""

    conversation_configuration_id: str | None = Field(
        default=None,
        description=(
            "Twilio Conversation Configuration ID. When omitted, TAC runs in "
            "ConversationRelay-only mode: only the Voice channel is usable, "
            "messaging channels cannot be constructed, and "
            "TAC.retrieve_memory() returns an empty TACMemoryResponse."
        ),
    )

    memory_config: TwilioMemoryConfig = Field(
        default_factory=TwilioMemoryConfig,
        description="Twilio Memory configuration for controlling retrieval limits, relevance "
        "threshold, and trait groups. Memory client is always initialized automatically from "
        "Conversation Orchestrator configuration.",
    )

    account_sid: str = Field(description="Twilio Account SID")
    auth_token: str = Field(description="Twilio Auth Token")
    api_key: str = Field(description="Twilio API Key SID (starts with SK)")
    api_secret: str = Field(description="Twilio API Key Secret")

    region: str | None = Field(
        default=None,
        description="Optional Twilio region (e.g., 'au1', 'ie1'). "
        "When set, API base URLs become https://product.<region>.twilio.com",
    )

    @field_validator("region", mode="before")
    @classmethod
    def _normalize_and_validate_region(cls, v: object) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("region must be a string or None")
        v = v.strip()
        if not v:
            return None
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*[a-z0-9]", v):
            raise ValueError(
                f"Invalid Twilio region format: '{v}' (must be a valid DNS label, "
                "e.g., 'au1', 'ie1')"
            )
        return v

    phone_number: str = Field(
        description="Twilio Phone Number for Voice (inbound) and SMS (send/receive).",
    )

    rcs_sender_id: str | None = Field(
        default=None,
        description="Optional Twilio RCS Sender ID",
    )

    whatsapp_number: str | None = Field(
        default=None,
        description="Optional Twilio WhatsApp-enabled phone number (format: whatsapp:+1234567890)",
    )

    knowledge_base_id: str | None = Field(
        default=None,
        description="Optional Knowledge Base ID for knowledge search functionality",
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    studio_handoff_flow_sid: str | None = Field(
        default=None,
        description="Twilio Studio Flow SID (FWxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx) "
        "for handoff. TAC constructs both the digital-handoff Studio Executions "
        "URL (studio.twilio.com/v2/Flows/{SID}/Executions) and the voice "
        "<Connect action> webhook URL "
        "(webhooks.twilio.com/v1/Accounts/{AccountSid}/Flows/{SID}?Trigger=incomingCall) "
        "from this SID.",
    )

    voice_public_domain: str | None = Field(
        default=None,
        description="Public domain where voice routes are reachable (e.g. "
        "'example.ngrok.app'). Used by VoiceChannel to construct the public "
        "WebSocket URL and ConversationRelay action URL. Required when using "
        "the Voice channel. Schemes (https://, wss://) and trailing slashes "
        "are stripped automatically.",
    )

    voice_websocket_path: str = Field(
        default="/ws",
        description="Path the voice WebSocket is served at. Combined with "
        "voice_public_domain to build the public WebSocket URL the voice "
        "channel hands to Twilio in TwiML; TACFastAPIServer also registers "
        "its WebSocket route at this path. Override only if you mount the "
        "route at a non-default path.",
    )

    voice_action_path: str = Field(
        default="/conversation-relay-callback",
        description="Path the ConversationRelay action callback is served at. "
        "Same role as voice_websocket_path but for the <Connect action=...> "
        "cleanup callback.",
    )

    @field_validator("voice_public_domain", mode="before")
    @classmethod
    def _normalize_voice_public_domain(cls, v: str | None) -> str | None:
        """Strip whitespace, schemes, and trailing slashes from voice_public_domain.

        A naive copy-paste from a browser address bar produces values like
        ``https://example.ngrok.app/`` which would otherwise concatenate into
        ``wss://https://example.ngrok.app//ws`` — clean them up at parse time.
        """
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        for scheme in ("https://", "http://", "wss://", "ws://"):
            if v.lower().startswith(scheme):
                v = v[len(scheme) :]
                break
        return v.rstrip("/") or None

    conversation_intelligence_config: ConversationIntelligenceConfig | None = Field(
        default=None,
        description="Optional Conversation Intelligence configuration for filtering webhook "
        "events. When provided to OperatorResultProcessor, only matching events are processed.",
    )

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "conversation_configuration_id": "conv_configuration_xxxxxxxxxxxxxxxxxx",
                "account_sid": "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                "auth_token": "your_auth_token_here",
                "api_key": "SKxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                "api_secret": "your_api_secret_here",
                "phone_number": "your_phone_number_here",
                "memory_config": {
                    "trait_groups": ["Contact", "Preferences"],
                },
                "conversation_intelligence_config": {
                    "configuration_id": "GAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "observation_operator_sid": "LYxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "summary_operator_sid": "LYyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy",
                },
            }
        },
    )

    @classmethod
    def from_env(cls) -> "TACConfig":
        """
        Create TACConfig from environment variables.

        Required:
        - TWILIO_ACCOUNT_SID: Twilio Account SID
        - TWILIO_AUTH_TOKEN: Twilio Auth Token for API authentication
        - TWILIO_API_KEY: Twilio API Key SID (starts with SK)
        - TWILIO_API_SECRET: Twilio API Secret for API Key authentication
        - TWILIO_PHONE_NUMBER: Phone number for voice and SMS channels

        Required for Conversation Orchestrator / Memory / Knowledge:
        - TWILIO_CONVERSATION_CONFIGURATION_ID: Conversation Orchestrator configuration ID
          (when omitted, TAC runs in ConversationRelay-only mode)

        Optional:
        - TWILIO_RCS_SENDER_ID: RCS Sender ID for RCS channel
        - TWILIO_WHATSAPP_NUMBER: WhatsApp-enabled phone number
          (format: whatsapp:+1234567890)
        - TWILIO_KNOWLEDGE_BASE_ID: Knowledge Base ID for RAG search functionality
        - TWILIO_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
          Default: INFO
        - TWILIO_REGION: Twilio region for data residency (e.g., 'au1', 'ie1')
        - TWILIO_STUDIO_HANDOFF_FLOW_SID: Studio Flow SID (FWxxx...) for handoff tool
        - TWILIO_VOICE_PUBLIC_DOMAIN: Public domain for voice routes (required for voice)
        - TWILIO_VOICE_WEBSOCKET_PATH: Path for voice WebSocket (default: /ws)
        - TWILIO_VOICE_ACTION_PATH: Path for ConversationRelay action callback
          (default: /conversation-relay-callback)

        Memory Configuration:
        - TWILIO_MEMORY_PROFILE_TRAIT_GROUPS: Trait groups to include
          (comma-separated, e.g., "Contact,Preferences")
        - TWILIO_MEMORY_OBSERVATIONS_LIMIT: Max observations in memory retrieval.
          Default: 20
        - TWILIO_MEMORY_SUMMARIES_LIMIT: Max summaries in memory retrieval. Default: 5
        - TWILIO_MEMORY_COMMUNICATIONS_LIMIT: Max communications in memory retrieval.
          Default: 0
        - TWILIO_MEMORY_RELEVANCE_THRESHOLD: Min relevance score (0.0-1.0). Default: 0.0

        Conversation Intelligence:
        - CONVERSATION_INTELLIGENCE_CONFIGURATION_ID: CI Service configuration ID
          for webhook filtering
        - CONVERSATION_INTELLIGENCE_OBSERVATION_OPERATOR_SID: Operator SID for
          observation extraction
        - CONVERSATION_INTELLIGENCE_SUMMARY_OPERATOR_SID: Operator SID for summary
          extraction
        """
        # Load memory configuration from optional env vars (config object is always present)
        memory_config = TwilioMemoryConfig.from_env()

        # Load optional conversation intelligence configuration
        conversation_intelligence_config = ConversationIntelligenceConfig.from_env()

        # Path overrides: only forward to the constructor when the env var is
        # set, so the field defaults take effect otherwise.
        path_overrides: dict[str, str] = {}
        if "TWILIO_VOICE_WEBSOCKET_PATH" in os.environ:
            path_overrides["voice_websocket_path"] = os.environ["TWILIO_VOICE_WEBSOCKET_PATH"]
        if "TWILIO_VOICE_ACTION_PATH" in os.environ:
            path_overrides["voice_action_path"] = os.environ["TWILIO_VOICE_ACTION_PATH"]

        return cls(
            conversation_configuration_id=os.environ.get("TWILIO_CONVERSATION_CONFIGURATION_ID"),
            account_sid=os.environ["TWILIO_ACCOUNT_SID"],
            auth_token=os.environ["TWILIO_AUTH_TOKEN"],
            api_key=os.environ["TWILIO_API_KEY"],
            api_secret=os.environ["TWILIO_API_SECRET"],
            phone_number=os.environ["TWILIO_PHONE_NUMBER"],
            rcs_sender_id=os.environ.get("TWILIO_RCS_SENDER_ID"),
            whatsapp_number=os.environ.get("TWILIO_WHATSAPP_NUMBER"),
            knowledge_base_id=os.environ.get("TWILIO_KNOWLEDGE_BASE_ID"),
            log_level=os.environ.get("TWILIO_LOG_LEVEL", "INFO"),
            region=os.environ.get("TWILIO_REGION"),
            studio_handoff_flow_sid=os.environ.get("TWILIO_STUDIO_HANDOFF_FLOW_SID"),
            voice_public_domain=os.environ.get("TWILIO_VOICE_PUBLIC_DOMAIN"),
            memory_config=memory_config,
            conversation_intelligence_config=conversation_intelligence_config,
            **path_overrides,
        )

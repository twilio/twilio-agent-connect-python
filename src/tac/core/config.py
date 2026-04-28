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
        """
        Create ConversationIntelligenceConfig from environment variables.

        Loads configuration from the following environment variables:
        - CONVERSATION_INTELLIGENCE_CONFIGURATION_ID: CI Configuration ID (required)
        - CONVERSATION_INTELLIGENCE_OBSERVATION_OPERATOR_SID: Operator SID for
          observations (optional)
        - CONVERSATION_INTELLIGENCE_SUMMARY_OPERATOR_SID: Operator SID for summaries (optional)

        Returns:
            ConversationIntelligenceConfig instance if configuration_id is set,
            None otherwise.
        """
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

    Note: Memory client is always initialized automatically by fetching memory_store_id
    from the Conversation Orchestrator configuration. This config only controls optional memory
    settings like which trait groups to include when fetching profiles.
    """

    trait_groups: list[str] | None = Field(
        default=None,
        description="Optional list of trait group names to include when retrieving profiles",
    )

    phone_trait_group: str = Field(
        default="Contact",
        description="Trait group name that holds the phone identifier on newly created profiles. "
        "Must match the promoted-to-identifier configuration of the Memora store.",
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
            }
        },
    )

    @classmethod
    def from_env(cls) -> "TwilioMemoryConfig":
        """
        Create TwilioMemoryConfig from environment variables.

        Loads configuration from the following environment variables:
        - MEMORY_PROFILE_TRAIT_GROUPS: Comma-separated list of trait groups (optional)

        Returns:
            TwilioMemoryConfig instance with parsed trait groups from environment,
            or default values if no environment variables are set.

        Example:
            >>> # With trait groups from environment
            >>> config = TwilioMemoryConfig.from_env()

            >>> # Or manually construct
            >>> config = TwilioMemoryConfig(trait_groups=["Contact", "Preferences"])

            >>> # Without trait groups (all traits included)
            >>> config = TwilioMemoryConfig()
        """
        trait_groups_str = os.environ.get("MEMORY_PROFILE_TRAIT_GROUPS")

        # Parse trait groups from environment variable
        trait_groups = None
        if trait_groups_str:
            trait_groups = [g.strip() for g in trait_groups_str.split(",")]

        kwargs: dict[str, object] = {"trait_groups": trait_groups}
        phone_trait_group = os.environ.get("MEMORY_PHONE_TRAIT_GROUP")
        if phone_trait_group:
            kwargs["phone_trait_group"] = phone_trait_group
        phone_trait_field = os.environ.get("MEMORY_PHONE_TRAIT_FIELD")
        if phone_trait_field:
            kwargs["phone_trait_field"] = phone_trait_field
        return cls(**kwargs)


class TACConfig(BaseModel):
    """Configuration model for Twilio Agent Connect settings."""

    conversation_configuration_id: str = Field(description="Twilio Conversation Configuration ID")

    memory_config: TwilioMemoryConfig | None = Field(
        default=None,
        description="Optional Twilio Memory configuration for controlling which trait groups "
        "to include when fetching profiles. Note: Memory client is always initialized "
        "automatically from Conversation Orchestrator configuration - "
        "this only configures trait group filtering.",
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

        Loads configuration from the following environment variables:
        - TWILIO_CONVERSATION_CONFIGURATION_ID: Twilio Conversation Configuration ID
        - TWILIO_ACCOUNT_SID: Twilio Account SID
        - TWILIO_AUTH_TOKEN: Twilio Auth Token
        - TWILIO_API_KEY: Twilio API Key SID (starts with SK)
        - TWILIO_API_SECRET: Twilio API Key Secret
        - TWILIO_PHONE_NUMBER: Twilio Phone Number for Voice and SMS channels
        - TWILIO_KNOWLEDGE_BASE_ID: Knowledge Base ID (optional)
        - TWILIO_LOG_LEVEL: Logging level (optional, defaults to INFO)
        - TWILIO_REGION: Twilio region for data residency (optional, e.g., 'au1', 'ie1')
        - TWILIO_STUDIO_HANDOFF_FLOW_SID: Studio Flow SID for handoff (optional)

        Memory configuration is automatically loaded via TwilioMemoryConfig.from_env()
        from these environment variables (all optional):
        - MEMORY_PROFILE_TRAIT_GROUPS: Comma-separated list of trait groups

        Conversation Intelligence configuration is automatically loaded via
        ConversationIntelligenceConfig.from_env() from these environment variables (all optional):
        - CONVERSATION_INTELLIGENCE_CONFIGURATION_ID: CI Configuration ID (TTID format)
        - CONVERSATION_INTELLIGENCE_OBSERVATION_OPERATOR_SID: Operator SID for observations
        - CONVERSATION_INTELLIGENCE_SUMMARY_OPERATOR_SID: Operator SID for summaries

        Returns:

        Raises:
            KeyError: If required environment variables are not set.
            ValidationError: If environment variable values are invalid.

        Example:
            >>> # With all env vars set in .env file
            >>> config = TACConfig.from_env()
            >>> tac = TAC(config=config)
        """
        # Load optional memory configuration
        memory_config = TwilioMemoryConfig.from_env()

        # Load optional conversation intelligence configuration
        conversation_intelligence_config = ConversationIntelligenceConfig.from_env()

        return cls(
            conversation_configuration_id=os.environ["TWILIO_CONVERSATION_CONFIGURATION_ID"],
            account_sid=os.environ["TWILIO_ACCOUNT_SID"],
            auth_token=os.environ["TWILIO_AUTH_TOKEN"],
            api_key=os.environ["TWILIO_API_KEY"],
            api_secret=os.environ["TWILIO_API_SECRET"],
            phone_number=os.environ["TWILIO_PHONE_NUMBER"],
            knowledge_base_id=os.environ.get("TWILIO_KNOWLEDGE_BASE_ID"),
            log_level=os.environ.get("TWILIO_LOG_LEVEL", "INFO"),
            region=os.environ.get("TWILIO_REGION"),
            studio_handoff_flow_sid=os.environ.get("TWILIO_STUDIO_HANDOFF_FLOW_SID"),
            memory_config=memory_config,
            conversation_intelligence_config=conversation_intelligence_config,
        )

"""Pydantic models for Twilio ConversationRelay Voice WebSocket messages."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class CustomParameters(BaseModel):
    """
    Custom parameters for ConversationRelay TwiML.

    Supports well-known TAC parameters plus arbitrary custom fields.
    All fields are optional since ConversationRelay handles conversation creation automatically.
    """

    conversation_id: str | None = Field(None, alias="conversationId")
    profile_id: str | None = Field(None, alias="profileId")
    customer_participant_id: str | None = Field(None, alias="customerParticipantId")
    ai_agent_participant_id: str | None = Field(None, alias="aiAgentParticipantId")

    model_config = {
        "populate_by_name": True,
        "extra": "allow",  # Accept arbitrary additional fields
    }


class SetupMessage(BaseModel):
    """
    Setup message sent when WebSocket connection is established.

    Contains call metadata from Twilio.
    """

    type: Literal["setup"] = "setup"
    session_id: str | None = Field(None, alias="sessionId")
    call_sid: str | None = Field(None, alias="callSid")
    parent_call_sid: str | None = Field(None, alias="parentCallSid")
    from_number: str | None = Field(None, alias="from")
    to_number: str | None = Field(None, alias="to")
    forwarded_from: str | None = Field(None, alias="forwardedFrom")
    caller_name: str | None = Field(None, alias="callerName")
    direction: str | None = Field(None, description="Call direction (inbound/outbound)")
    call_type: str | None = Field(None, alias="callType", description="Call type (e.g., PSTN)")
    call_status: str | None = Field(None, alias="callStatus", description="Call status")
    account_sid: str | None = Field(None, alias="accountSid")
    custom_parameters: dict[str, Any] | None = Field(
        None, alias="customParameters", description="Custom parameters passed via TwiML"
    )

    model_config = {"populate_by_name": True}


class PromptMessage(BaseModel):
    """
    Prompt message containing user's voice input.

    Sent when user speaks and speech is transcribed.
    """

    type: Literal["prompt"] = "prompt"
    conversation_id: str | None = Field(None, alias="conversationId")
    voice_prompt: str | None = Field(
        None, alias="voicePrompt", description="Transcribed user speech"
    )
    lang: str | None = Field(None, description="Language code (e.g., 'en-US')")
    last: bool | None = Field(None, description="Whether this is the last chunk")

    model_config = {"populate_by_name": True}


class InterruptMessage(BaseModel):
    """
    Interrupt message sent when user interrupts the agent.

    Contains information about what was being said when interrupted.
    """

    type: Literal["interrupt"] = "interrupt"
    conversation_id: str | None = Field(None, alias="conversationId")
    utterance_until_interrupt: str | None = Field(
        None,
        alias="utteranceUntilInterrupt",
        description="Text being spoken when interrupted",
    )
    duration_until_interrupt_ms: int | None = Field(
        None,
        alias="durationUntilInterruptMs",
        description="Duration in milliseconds until interruption",
    )

    model_config = {"populate_by_name": True}


# Discriminated union of all voice message types
VoiceMessage = SetupMessage | PromptMessage | InterruptMessage


class ConversationRelayCallbackPayload(BaseModel):
    """
    Payload received from Twilio ConversationRelay callback webhook.

    Sent when a ConversationRelay session ends or transitions state.
    """

    account_sid: str = Field(..., alias="AccountSid", description="Twilio Account SID")
    call_sid: str = Field(..., alias="CallSid", description="Twilio Call SID")
    call_status: str = Field(
        ...,
        alias="CallStatus",
        description="Call status (e.g., 'in-progress', 'completed', 'busy', 'no-answer')",
    )
    from_number: str = Field(..., alias="From", description="Caller's identifier")
    to_number: str = Field(..., alias="To", description="Recipient's identifier")
    direction: str = Field(..., alias="Direction", description="Call direction (inbound/outbound)")
    application_sid: str | None = Field(
        None, alias="ApplicationSid", description="Twilio Application SID"
    )
    session_id: str | None = Field(
        None, alias="SessionId", description="ConversationRelay Session ID"
    )
    session_status: str | None = Field(
        None,
        alias="SessionStatus",
        description="ConversationRelay session status (e.g., 'ended')",
    )
    session_duration: str | None = Field(
        None, alias="SessionDuration", description="Session duration in seconds"
    )

    model_config = {"populate_by_name": True}


class LanguageConfig(BaseModel):
    """A single ``<Language>`` child for multi-language ConversationRelay setups.

    Maps to the ``<Language>`` element documented at
    https://www.twilio.com/docs/voice/twiml/connect/conversationrelay#language-element
    """

    code: str = Field(..., description="Language code, e.g. 'es-MX'")
    voice: str | None = Field(None, description="TTS voice name for this language")
    tts_provider: str | None = Field(None, description="TTS provider, e.g. 'google'")
    transcription_provider: str | None = Field(
        None, description="Transcription provider, e.g. 'deepgram'"
    )

    model_config = {"populate_by_name": True}


class TwiMLOptions(BaseModel):
    """Options for generating ConversationRelay TwiML.

    Fields map to the attributes documented at
    https://www.twilio.com/docs/voice/twiml/connect/conversationrelay .
    All fields except ``websocket_url`` are optional. ``VoiceChannel.handle_incoming_call``
    merges these values over TAC defaults using Pydantic's ``model_fields_set`` —
    only fields explicitly set by the caller override TAC's defaults.
    """

    websocket_url: str = Field(
        default="",
        description="WebSocket URL for ConversationRelay. Usually set by the server "
        "from TACServerConfig.public_domain — resolvers and other overlay callers "
        "can leave it empty and the server/channel will fill it in.",
    )
    custom_parameters: CustomParameters | dict[str, Any] | None = Field(
        None,
        description="Custom parameters to pass to ConversationRelay",
    )
    welcome_greeting: str | None = Field(
        None,
        description="Initial greeting message for caller",
    )
    action_url: str | None = Field(
        None,
        description="URL for Twilio to request when call ends",
    )
    conversation_configuration: str | None = Field(
        None,
        description="Conversation Service SID for ConversationRelay to automatically "
        "manage conversation creation and participants.",
    )
    voice: str | None = Field(None, description="Default TTS voice name")
    language: str | None = Field(None, description="Default language code, e.g. 'en-US'")
    transcription_provider: str | None = Field(
        None, description="Default transcription provider, e.g. 'deepgram'"
    )
    tts_provider: str | None = Field(None, description="Default TTS provider, e.g. 'elevenlabs'")
    interruptible: str | bool | None = Field(
        None, description="Interruption behavior (e.g. 'speech', 'dtmf', 'any', True/False)"
    )
    dtmf_detection: bool | None = Field(None, description="Enable DTMF detection")
    debug: str | None = Field(
        None, description="Debug flags, e.g. 'speaker-events' or comma-separated list"
    )
    languages: list[LanguageConfig] | None = Field(
        None, description="Additional <Language> children for multi-language support"
    )

    model_config = {"populate_by_name": True}


class TwiMLRequestContext(BaseModel):
    """Framework-neutral view of the Twilio TwiML webhook form.

    Populated by ``TACFastAPIServer`` from the incoming Twilio webhook, then
    passed to an optional ``resolve_twiml_options`` so the application can
    produce per-call ``TwiMLOptions`` overrides without depending on FastAPI
    types.
    """

    from_number: str | None = Field(None, alias="From")
    to_number: str | None = Field(None, alias="To")
    call_sid: str | None = Field(None, alias="CallSid")
    caller_country: str | None = Field(None, alias="CallerCountry")
    caller_state: str | None = Field(None, alias="CallerState")
    caller_city: str | None = Field(None, alias="CallerCity")
    direction: str | None = Field(None, alias="Direction")
    extra: dict[str, str] = Field(
        default_factory=dict,
        description="Any other fields from the Twilio webhook not captured above",
    )

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @classmethod
    def from_form(cls, form: dict[str, str]) -> "TwiMLRequestContext":
        """Build a context from a raw Twilio form dict, bucketing unknown keys into ``extra``."""
        known_aliases = {
            "From",
            "To",
            "CallSid",
            "CallerCountry",
            "CallerState",
            "CallerCity",
            "Direction",
        }
        known: dict[str, str] = {}
        extra: dict[str, str] = {}
        for key, value in form.items():
            if key in known_aliases:
                known[key] = value
            else:
                extra[key] = value
        return cls(**known, extra=extra)

"""Pydantic models for Twilio ConversationRelay Voice WebSocket messages."""

from typing import Any, Literal

from pydantic import BaseModel, Field

# Twilio uses the same four-value enum for several attributes that control
# what caller input (DTMF, speech, both, neither) triggers a given behavior.
InterruptMode = Literal["none", "dtmf", "speech", "any"]


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

    code: str = Field(
        ...,
        description="Language code, e.g. 'es-MX'. Can be 'multi' for automatic language "
        "detection (requires ElevenLabs TTS and Deepgram STT).",
    )
    voice: str | None = Field(None, description="TTS voice name for this language")
    tts_provider: str | None = Field(None, description="TTS provider, e.g. 'google'")
    transcription_provider: str | None = Field(
        None, description="Transcription provider, e.g. 'deepgram'"
    )
    speech_model: str | None = Field(
        None,
        description="Speech model for STT. Choices vary by transcription_provider; "
        "see the provider's documentation.",
    )

    model_config = {"populate_by_name": True}


class TwiMLOptions(BaseModel):
    """Options for generating ConversationRelay TwiML.

    Fields map to the attributes documented at
    https://www.twilio.com/docs/voice/twiml/connect/conversationrelay .
    All fields are optional. ``VoiceChannel.handle_incoming_call`` merges these
    values over TAC defaults using Pydantic's ``model_fields_set`` — only
    fields explicitly set by the caller override TAC's defaults.
    """

    custom_parameters: CustomParameters | dict[str, Any] | None = Field(
        None,
        description="Custom parameters to pass to ConversationRelay",
    )
    welcome_greeting: str | None = Field(
        None,
        description="Initial greeting message for caller",
    )
    welcome_greeting_interruptible: InterruptMode | None = Field(
        None,
        description="What caller input can interrupt the welcome greeting. "
        "Defaults to 'any' on Twilio.",
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

    # Language, TTS, STT
    language: str | None = Field(
        None,
        description="Language for both STT and TTS, e.g. 'en-US'. Equivalent to setting "
        "both tts_language and transcription_language.",
    )
    tts_language: str | None = Field(
        None,
        description="TTS language code; overrides `language` for TTS.",
    )
    transcription_language: str | None = Field(
        None,
        description="STT language code; overrides `language` for transcription. "
        "Can be 'multi' for automatic language detection (Deepgram only).",
    )
    voice: str | None = Field(None, description="TTS voice name (choices vary by tts_provider)")
    tts_provider: str | None = Field(
        None,
        description="TTS provider: 'Google', 'Amazon', or 'ElevenLabs'. Defaults to 'ElevenLabs'.",
    )
    transcription_provider: str | None = Field(
        None,
        description="STT provider: 'Google' or 'Deepgram'. Defaults to 'Deepgram' (or 'Google' "
        "for accounts that used ConversationRelay before 2025-09-12).",
    )
    speech_model: str | None = Field(
        None,
        description="Speech model for STT. Choices vary by transcription_provider.",
    )
    elevenlabs_text_normalization: Literal["on", "auto", "off"] | None = Field(
        None,
        description="Text normalization for ElevenLabs TTS. Defaults to 'off'. "
        "'auto' behaves like 'off' for ConversationRelay calls.",
    )

    # Turn detection / interruption
    eot_threshold: float | None = Field(
        None,
        ge=0.5,
        le=0.9,
        description="Confidence required to finish a turn (0.5 - 0.9). "
        "Only applies with Deepgram + flux speech model.",
    )
    partial_prompts: bool | None = Field(
        None,
        description="Send unfinalized prompts and eager end-of-turn events "
        "(last=False). Only applies with Deepgram + flux speech model.",
    )
    deepgram_smart_format: bool | None = Field(
        None,
        description="Use Deepgram Smart Format for transcription output. "
        "Defaults to true when transcription_provider='Deepgram'.",
    )
    speech_timeout: int | None = Field(
        None,
        ge=600,
        le=5000,
        description="Silence (ms) after speech before finalizing the prompt. "
        "Integer in [600, 5000]. Defaults to 'auto' on Twilio.",
    )
    interruptible: InterruptMode | bool | None = Field(
        None,
        description="What caller input interrupts TTS playback. Boolean accepted "
        "for backward compat: True='any', False='none'. Defaults to 'any'.",
    )
    interrupt_sensitivity: Literal["high", "medium", "low"] | None = Field(
        None,
        description="How easily caller speech triggers an interrupt. Defaults to 'high'.",
    )
    report_input_during_agent_speech: InterruptMode | None = Field(
        None,
        description="What caller input gets reported while the agent is speaking "
        "(independent of whether playback is interrupted). Defaults to 'none' since May 2025.",
    )
    ignore_backchannel: bool | None = Field(
        None,
        description="Filter short conversational feedback ('yeah', 'uh-huh', …) "
        "so it doesn't interrupt the agent. Defaults to false.",
    )
    preemptible: bool | None = Field(
        None,
        description="Allow text tokens from the next talk cycle to interrupt the current one. "
        "Defaults to false.",
    )
    dtmf_detection: bool | None = Field(
        None,
        description="Emit DTMF keypress events over the WebSocket.",
    )

    # Recognition hints / events / debug / intelligence
    hints: str | None = Field(
        None,
        description="Comma-separated words/phrases likely to appear in speech. "
        "Capitalize proper nouns.",
    )
    events: str | None = Field(
        None,
        description="Space-separated event subscriptions, e.g. 'speaker-events tokens-played'.",
    )
    debug: str | None = Field(
        None,
        description="Debug subscription, e.g. 'debugging'. Note: 'speaker-events' and "
        "'tokens-played' have moved to the `events` attribute — only use them here "
        "for backward compatibility.",
    )
    intelligence_service: str | None = Field(
        None,
        description="Conversation Intelligence (classic) Service SID or unique name for "
        "persisting transcripts and running Language Operators.",
    )

    # Nested <Language> children
    languages: list[LanguageConfig] | None = Field(
        None, description="Additional <Language> children for multi-language support"
    )

    extra: dict[str, str | bool | int] | None = Field(
        None,
        description="Escape hatch for ConversationRelay attributes not yet typed on "
        "this model. Keys are emitted as-is on <ConversationRelay>; Twilio's SDK "
        "converts snake_case to camelCase, lowercases bools to 'true'/'false', "
        "and stringifies ints. Prefer a typed field when one exists — use "
        "``extra`` only for newly-added Twilio attributes not yet in this SDK.",
    )

    model_config = {"populate_by_name": True}


class TwiMLRequest(BaseModel):
    """Framework-neutral view of the Twilio TwiML webhook form.

    Populated by ``TACFastAPIServer`` from the incoming Twilio webhook, then
    passed to an optional ``customize_inbound_twiml`` so the application can
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
        description="Any other fields from the Twilio webhook not captured above. "
        "Values are always strings here (webhook form fields are url-encoded), "
        "unlike TwiMLOptions.extra which accepts str | bool | int for emitted "
        "TwiML attributes.",
    )

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @classmethod
    def from_form(cls, form: dict[str, str]) -> "TwiMLRequest":
        """Build a context from a raw Twilio form dict, bucketing unknown keys into ``extra``."""
        known_aliases = {f.alias for f in cls.model_fields.values() if f.alias}
        known: dict[str, str] = {}
        extra: dict[str, str] = {}
        for key, value in form.items():
            if key in known_aliases:
                known[key] = value
            else:
                extra[key] = value
        return cls(**known, extra=extra)

"""Models for outbound conversation initiation."""

from functools import lru_cache
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_serializer, model_validator

from tac.models.session import ConversationSession
from tac.models.voice import TwiMLOptions


class InitiateMessagingConversationOptions(BaseModel):
    """Shared options for initiating an outbound messaging conversation.

    This base model is used for messaging-style outbound conversations,
    including SMS, RCS, WhatsApp, and Chat. Each channel may extend this with
    channel-specific requirements (e.g., Chat requires channel_id).

    The sender is always TAC's configured address (``config.phone_number``
    for SMS, ``config.rcs_sender_id`` for RCS, ``config.whatsapp_number``
    for WhatsApp, ``ChatChannelConfig.agent_address`` for Chat).
    Multi-sender deployments should use one TAC instance per sender so
    inbound webhook routing, memory scoping, and configuration stay in sync.
    """

    to: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = Field(default=None)

    model_config = {"populate_by_name": True}


class InitiateChatConversationOptions(InitiateMessagingConversationOptions):
    """Options for initiating an outbound Chat conversation.

    Extends InitiateMessagingConversationOptions with a required channel_id
    (Conversations v1 Channel SID) for Chat delivery.
    """

    channel_id: str = Field(..., min_length=1)


class InitiateConversationResult(BaseModel):
    """Result of initiating an outbound messaging conversation."""

    conversation_id: str
    session: ConversationSession

    model_config = {"arbitrary_types_allowed": True}


@lru_cache(maxsize=1)
def _twilio_call_create_params() -> frozenset[str]:
    """Param names ``calls.create()`` accepts, read from the installed SDK.

    ``CallList.create`` takes no ``**kwargs``, so an unknown key is a TypeError
    at call time, not something Twilio validates. Empty set if there's nothing
    to validate against, making validation permissive rather than breaking
    callers.
    """
    try:
        import inspect

        from twilio.rest.api.v2010.account.call import CallList

        sig = inspect.signature(CallList.create)
        # If the SDK ever switches to **kwargs, the named params stop being the
        # accepted set — validating against them would reject every real extra.
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return frozenset()
        return frozenset(sig.parameters) - {"self"}
    except Exception:  # pragma: no cover - defensive against SDK changes
        return frozenset()


class CallOptions(BaseModel):
    """Parameters for Twilio's ``client.calls.create()``.

    Typed below are the ones outbound ConversationRelay reaches for; any other
    param ``calls.create()`` accepts is forwarded too. Unknown keys are rejected
    at construction against the SDK signature, so typos fail here.

    Example:
        ```python
        CallOptions(machine_detection="Enable", async_amd=True, record=True)
        ```
    """

    # machine_detection is what enables AMD; async_amd only picks background
    # vs. blocking detection. Both are required for on_amd to fire.
    machine_detection: Literal["Enable", "DetectMessageEnd"] | None = Field(
        default=None,
        description="Enables AMD. 'Enable' reports as soon as it can tell human from "
        "machine (to hang up on voicemail); 'DetectMessageEnd' waits out the greeting "
        "(to leave a message).",
    )
    async_amd: bool | None = Field(
        default=None,
        description="Detect in the background. Required for on_amd: with it off, "
        "AnsweredBy comes back on the TwiML request, which inline TwiML can't receive.",
    )
    async_amd_status_callback: str | None = None
    async_amd_status_callback_method: str | None = None
    machine_detection_timeout: int | None = None
    machine_detection_speech_threshold: int | None = None
    machine_detection_speech_end_threshold: int | None = None
    machine_detection_silence_timeout: int | None = None

    record: bool | None = Field(default=None, description="Required for on_recording.")
    recording_status_callback: str | None = None
    recording_status_callback_event: list[str] | None = None
    recording_channels: str | None = None
    recording_track: str | None = None

    status_callback: str | None = None
    status_callback_event: list[str] | None = Field(
        default=None,
        description="Lifecycle events to report. Omitted, Twilio sends only 'completed' "
        "— which covers busy/canceled/failed/no-answer. Set it for ringing/answered.",
    )
    status_callback_method: str | None = None
    timeout: int | None = Field(
        default=None, description="Seconds to ring before giving up. Twilio defaults to 60."
    )

    model_config = {"populate_by_name": True, "extra": "allow"}

    # TAC builds the call and its TwiML, so callers may not set these.
    RESERVED: ClassVar[frozenset[str]] = frozenset(
        {"to", "from_", "from", "twiml", "url", "application_sid"}
    )

    @model_validator(mode="after")
    def _validate_params(self) -> "CallOptions":
        supplied = set(self.__pydantic_extra__ or {})

        conflict = self.RESERVED & supplied
        if conflict:
            raise ValueError(
                f"call_options may not set TAC-owned call parameters: {sorted(conflict)}. "
                "TAC builds the call and its TwiML."
            )

        accepted = _twilio_call_create_params()
        unknown = supplied - accepted if accepted else set()
        if unknown:
            raise ValueError(
                f"call_options has parameters Twilio's calls.create() does not accept: "
                f"{sorted(unknown)}. Check for a typo, or upgrade the twilio package."
            )

        # AMD needs both flags. machine_detection turns detection on; async_amd
        # delivers AnsweredBy to a callback. Without async_amd, Twilio returns it
        # on the TwiML request instead — unreachable, since TAC sends inline TwiML.
        if bool(self.machine_detection) != bool(self.async_amd):
            raise ValueError(
                "AMD requires both machine_detection and async_amd; got "
                f"machine_detection={self.machine_detection!r}, async_amd={self.async_amd!r}."
            )
        return self

    @field_serializer("async_amd")
    def _serialize_async_amd(self, value: bool | None) -> str | None:
        """Twilio's SDK types async_amd as a string, unlike record."""
        return None if value is None else str(value).lower()

    def to_call_kwargs(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)


class InitiateVoiceConversationOptions(BaseModel):
    """Options for initiating an outbound voice conversation.

    The caller identity is always TAC's configured ``config.phone_number``.
    Multi-number deployments should use one TAC instance per line.

    TwiML for the outbound call is built by merging per-field, highest
    precedence first:
      1. This call's ``twiml_options`` (per-call overrides)
      2. ``VoiceChannelConfig.default_twiml_options`` (channel-wide defaults)
      3. TAC defaults (welcome greeting, conversation_configuration,
         action_url resolved via Studio handoff if configured)

    Fields you don't set at a layer fall through to lower layers — so
    ``twiml_options=TwiMLOptions(voice="es-MX-Neural2-A")`` on this call
    overrides only ``voice``; ``language``, ``interruptible``, etc. from the
    channel config still apply.

    Set ``voice``, ``language``, ``interruptible``, etc. on the channel's
    ``VoiceChannelConfig.default_twiml_options`` to apply them to every call
    (both inbound and outbound). Use this model's ``twiml_options`` for
    per-call overrides (e.g. campaign-specific ``custom_parameters``).
    """

    to: str = Field(..., min_length=1)
    websocket_url: str | None = Field(
        default=None,
        description="Public WebSocket URL for ConversationRelay (e.g. "
        "'wss://your-domain.ngrok.app/ws'). Optional — defaults to the URL "
        "derived from ``TACConfig.voice_public_domain`` + "
        "``voice_websocket_path``. Pass it here only to override the URL "
        "for a specific call.",
    )
    twiml_options: TwiMLOptions | None = Field(
        default=None,
        description="Per-call overrides for the TwiML inside <ConversationRelay>. "
        "Merged over VoiceChannelConfig.default_twiml_options and TAC defaults.",
    )
    call_options: CallOptions | None = Field(
        default=None,
        description="Parameters for Twilio's calls.create() — AMD, recording, status "
        "callbacks, timeout (see CallOptions). Accepts a plain dict. Callback URLs "
        "auto-wire when the matching handler is registered; an explicit URL wins.",
    )

    model_config = {"populate_by_name": True, "extra": "forbid"}


class InitiateVoiceConversationResult(BaseModel):
    """Result of initiating an outbound voice conversation."""

    call_sid: str

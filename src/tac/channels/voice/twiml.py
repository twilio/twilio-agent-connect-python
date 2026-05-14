"""TwiML generation for voice channel."""

from typing import Any

from pydantic import BaseModel
from twilio.twiml.voice_response import VoiceResponse

from tac.core.logging import get_logger
from tac.models.voice import TwiMLOptions

logger = get_logger(__name__)

_OPTIONAL_RELAY_ATTRS = (
    "welcome_greeting",
    "conversation_configuration",
    "voice",
    "language",
    "transcription_provider",
    "tts_provider",
    "interruptible",
    "dtmf_detection",
    "debug",
)


def generate_twiml(
    websocket_url: str,
    options: TwiMLOptions | dict[str, Any] | None = None,
) -> str:
    """
    Generate TwiML XML for ConversationRelay.

    This is a low-level function. Most users should call
    ``VoiceChannel.handle_incoming_call`` instead — it layers in TAC defaults,
    static ``twiml_options`` from ``VoiceChannelConfig``, and any per-call
    customizer output.

    Args:
        websocket_url: Public WebSocket URL for ConversationRelay
            (e.g. ``'wss://example.ngrok.app/ws'``).
        options: Optional ``TwiMLOptions`` (or dict). See ``TwiMLOptions``
            for supported fields. Newly-added ConversationRelay attributes
            not yet typed on the model can be passed via ``extra``.

    Returns:
        TwiML XML string ready to return to Twilio.

    Example:
        >>> twiml = generate_twiml(
        ...     "wss://example.com/voice",
        ...     TwiMLOptions(
        ...         welcome_greeting="Hello!",
        ...         conversation_configuration="conv_configuration_xxxx",
        ...     ),
        ... )
    """
    if options is None:
        options = TwiMLOptions()
    elif isinstance(options, dict):
        options = TwiMLOptions(**options)

    response = VoiceResponse()

    # Create Connect verb with optional action
    connect_kwargs: dict[str, str] = {}
    if options.action_url:
        connect_kwargs["action"] = options.action_url
    connect = response.connect(**connect_kwargs)

    # Build ConversationRelay kwargs. The twilio SDK converts snake_case to
    # camelCase automatically, and serializes bool/str as TwiML attribute values.
    relay_kwargs: dict[str, Any] = {"url": websocket_url}
    for attr in _OPTIONAL_RELAY_ATTRS:
        value = getattr(options, attr)
        if value is not None:
            relay_kwargs[attr] = value

    if options.extra:
        typed_keys = set(_OPTIONAL_RELAY_ATTRS)
        for key, value in options.extra.items():
            if key in typed_keys:
                logger.warning(
                    "TwiMLOptions.extra shadows a typed field; typed field wins.",
                    shadowed_key=key,
                )
                continue
            relay_kwargs[key] = value

    relay = connect.conversation_relay(**relay_kwargs)

    # Emit <Language> children, if any
    if options.languages:
        for lang in options.languages:
            lang_kwargs: dict[str, Any] = {"code": lang.code}
            for attr in ("voice", "tts_provider", "transcription_provider"):
                value = getattr(lang, attr)
                if value is not None:
                    lang_kwargs[attr] = value
            relay.language(**lang_kwargs)

    # Add custom parameters as <Parameter> children
    if options.custom_parameters:
        params_dict: dict[str, Any] = (
            options.custom_parameters.model_dump(by_alias=True, exclude_none=True)
            if isinstance(options.custom_parameters, BaseModel)
            else options.custom_parameters
        )
        for name, value in params_dict.items():
            if value is not None:
                relay.parameter(name=name, value=str(value))

    return str(response)

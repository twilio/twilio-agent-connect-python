"""TwiML generation for voice channel."""

from typing import Any

from pydantic import BaseModel
from twilio.twiml.voice_response import VoiceResponse

from tac.models.voice import TwiMLOptions


def generate_twiml(
    options: TwiMLOptions | dict[str, Any],
) -> str:
    """
    Generate TwiML XML for ConversationRelay with custom parameters.

    This is a low-level function for generating TwiML with arbitrary custom
    parameters. For automatic conversation creation and participant management,
    use VoiceChannel.handle_incoming_call() instead.

    Args:
        options: TwiML generation options (TwiMLOptions model or dict with:
            - websocket_url (required): WebSocket URL for ConversationRelay
            - custom_parameters (optional): Dict of custom parameters
            - welcome_greeting (optional): Initial greeting message
            - action_url (optional): URL for call end webhook
            - conversation_configuration (optional): Conversation Service SID for
              automatic conversation creation

    Returns:
        TwiML XML string ready to return to Twilio

    Example:
        >>> twiml = generate_twiml(
        ...     {
        ...         "websocket_url": "wss://example.com/voice",
        ...         "custom_parameters": {
        ...             "session_id": "sess_abc123",
        ...             "user_language": "es",
        ...         },
        ...         "welcome_greeting": "Hello!",
        ...         "conversation_configuration": "conv_configuration_xxxx",
        ...     }
        ... )
    """
    # Handle dict input (convert to TwiMLOptions)
    if isinstance(options, dict):
        options = TwiMLOptions(**options)

    # Create VoiceResponse
    response = VoiceResponse()

    # Create Connect verb with optional action
    connect_kwargs: dict[str, str] = {}
    if options.action_url:
        connect_kwargs["action"] = options.action_url
    connect = response.connect(**connect_kwargs)

    # Build ConversationRelay kwargs. The twilio SDK converts snake_case to
    # camelCase automatically, and serializes bool/str as TwiML attribute values.
    relay_kwargs: dict[str, Any] = {"url": options.websocket_url}
    optional_attrs = (
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
    for attr in optional_attrs:
        value = getattr(options, attr)
        if value is not None:
            relay_kwargs[attr] = value

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

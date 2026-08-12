"""TwiML generation for the OpenAI Realtime (Media Streams) voice channel.

Emits ``<Connect><Stream>`` — the bidirectional audio stream Twilio opens to
our WebSocket — as opposed to ``<Connect><ConversationRelay>`` (the text
protocol, used by ``tac.channels.voice``) or a SIP Trunk hand-off (used by
``tac.channels.openai_realtime_sip``, which needs no TwiML at all).

See https://www.twilio.com/docs/voice/twiml/stream for the ``<Stream>`` verb.
"""

from typing import Any

from twilio.twiml.voice_response import Connect, VoiceResponse


def generate_stream_twiml(
    websocket_url: str,
    *,
    custom_parameters: dict[str, Any] | None = None,
) -> str:
    """Generate TwiML that connects the call to a bidirectional Media Stream.

    Args:
        websocket_url: Public ``wss://`` URL of the WebSocket endpoint Twilio
            should stream call audio to (the ``<Stream url=...>`` attribute).
        custom_parameters: Optional per-call values emitted as ``<Parameter>``
            children of ``<Stream>``. They arrive back in the WebSocket
            ``start`` event under ``start.customParameters``.

    Returns:
        TwiML XML string ready to return to Twilio.
    """
    if not websocket_url or not websocket_url.strip():
        raise ValueError("generate_stream_twiml requires a non-empty websocket_url.")

    response = VoiceResponse()
    connect = Connect()
    stream = connect.stream(url=websocket_url)
    if custom_parameters:
        for name, value in custom_parameters.items():
            if value is not None:
                stream.parameter(name=name, value=str(value))

    response.append(connect)
    return str(response)

"""
Feature: ConversationRelay TwiML customization

Two layers on VoiceChannelConfig (highest precedence first):

1. ``customize_inbound_twiml`` — async callable receiving a TwiMLRequest
   (parsed Twilio webhook fields: From, To, CallerCountry, …). Inbound only.
   For outbound, pass per-call TwiMLOptions on InitiateVoiceConversationOptions.
2. ``default_twiml_options`` — static TwiMLOptions applied to every call
   (inbound and outbound).

Layers merge per-field: the customizer overrides only the fields it
explicitly sets; everything else falls through to ``default_twiml_options``
and then to TAC defaults (websocket URL, action URL, conversation_configuration).

This example shows both layers together — channel-wide defaults plus a
country-based customizer that overrides language/voice/greeting for
specific callers.
"""

from dotenv import load_dotenv

from tac import TAC, TACConfig
from tac.channels.voice import VoiceChannel, VoiceChannelConfig
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.models.voice import TwiMLOptions, TwiMLRequest
from tac.server import TACFastAPIServer

load_dotenv()

tac = TAC(config=TACConfig.from_env())


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> str:
    return f"You said: {user_message}"


tac.on_message_ready(handle_message_ready)


async def customize_twiml(req: TwiMLRequest) -> TwiMLOptions:
    """Per-call overrides for inbound calls. Only the fields you set here
    override the channel default; the rest fall through."""
    if req.caller_country == "MX":
        return TwiMLOptions(
            language="es-MX",
            welcome_greeting="¡Hola! ¿En qué puedo ayudarte?",
        )
    if req.caller_country == "FR":
        return TwiMLOptions(
            language="fr-FR",
            welcome_greeting="Bonjour ! Comment puis-je vous aider ?",
        )
    return TwiMLOptions()  # fall through to default_twiml_options


voice_channel = VoiceChannel(
    tac,
    config=VoiceChannelConfig(
        # Channel-wide defaults — apply to every call (inbound + outbound).
        default_twiml_options=TwiMLOptions(
            welcome_greeting="Hello! This is a default greeting.",
            interruptible="speech",
        ),
        # Per-call inbound overrides — runs once per inbound call.
        customize_inbound_twiml=customize_twiml,
    ),
)


if __name__ == "__main__":
    server = TACFastAPIServer(tac=tac, voice_channel=voice_channel)
    server.start()

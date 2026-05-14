"""
Feature: ConversationRelay TwiML customization

TAC exposes two user-facing layers of TwiML customization on
VoiceChannelConfig (under these, TAC fills in the websocket URL, action URL,
conversation_configuration, and a default welcome greeting):

1. ``twiml_options`` — static TwiMLOptions applied to every call.
2. ``customize_twiml_options`` — async callable for per-call logic, receives
   a TwiMLRequest (parsed Twilio webhook fields: From, To, CallerCountry, …).

The customizer wins over static options, which win over TAC defaults.

Channel ``twiml_options`` applies to both inbound and outbound calls
(``initiate_outbound_conversation``). The customizer only runs for inbound
calls — outbound calls receive per-call TwiML via
``InitiateVoiceConversationOptions.twiml_options`` at each call site instead.

This example shows the static path (voice + language the same for every
call). The customizer version for per-call localization is below in a
commented block — uncomment if you need it.
"""

from dotenv import load_dotenv

from tac import TAC, TACConfig
from tac.channels.voice import VoiceChannel, VoiceChannelConfig
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.models.voice import LanguageConfig, TwiMLOptions
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


# ---- Static TwiML (same settings on every call) ------------------------------
#
# Set ``twiml_options`` on VoiceChannelConfig for attributes that don't depend
# on who's calling. TAC fills in websocket_url, action_url, and
# conversation_configuration.

voice_channel = VoiceChannel(
    tac,
    config=VoiceChannelConfig(
        twiml_options=TwiMLOptions(
            welcome_greeting="Hello! How can I help?",
            voice="en-US-Journey-D",
            language="en-US",
            tts_provider="google",
            transcription_provider="deepgram",
            interruptible="speech",
            # <Language> children let the caller switch languages mid-call.
            languages=[
                LanguageConfig(code="en-US", voice="en-US-Journey-D", tts_provider="google"),
                LanguageConfig(code="es-MX", voice="es-MX-Neural2-A", tts_provider="google"),
            ],
        ),
    ),
)


# ---- Per-call TwiML (customize_twiml_options) --------------------------------
#
# Use this when the TwiML depends on who's calling — e.g., localization by
# caller country, per-tenant voice, A/B tests. The customizer returns
# TwiMLOptions overrides; anything it doesn't set falls through to
# ``twiml_options`` (above) and then to TAC defaults.
#
# Uncomment the block below to replace the static setup with per-call logic.
#
# from tac.models.voice import TwiMLRequest
#
#
# async def customize_twiml(req: TwiMLRequest) -> TwiMLOptions:
#     if req.caller_country == "MX":
#         return TwiMLOptions(
#             language="es-MX",
#             voice="es-MX-Neural2-A",
#             welcome_greeting="¡Hola! ¿En qué puedo ayudarte?",
#         )
#     if req.caller_country == "FR":
#         return TwiMLOptions(
#             language="fr-FR",
#             voice="fr-FR-Neural2-A",
#             welcome_greeting="Bonjour ! Comment puis-je vous aider ?",
#         )
#     return TwiMLOptions()  # fall through to static twiml_options + TAC defaults
#
#
# voice_channel = VoiceChannel(
#     tac,
#     config=VoiceChannelConfig(
#         twiml_options=TwiMLOptions(welcome_greeting="Hello! How can I help?"),
#         customize_twiml_options=customize_twiml,
#     ),
# )


if __name__ == "__main__":
    server = TACFastAPIServer(tac=tac, voice_channel=voice_channel)
    server.start()

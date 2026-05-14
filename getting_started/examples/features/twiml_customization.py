"""
Feature: Per-call TwiML customization

Demonstrates using ``VoiceChannelConfig.resolve_twiml_options`` to tailor the
ConversationRelay TwiML on every incoming voice call. The resolver receives a
``TwiMLRequestContext`` parsed from the Twilio webhook (``From``, ``To``,
``CallerCountry``, etc.) and returns ``TwiMLOptions`` overrides. Any field the
resolver explicitly sets replaces TAC's default; everything else (websocket
URL, ``action_url``, ``conversation_configuration``, welcome greeting)
continues to come from TAC/server config.

This example picks voice and language based on the caller's country and adds
``<Language>`` children so the caller can switch mid-call. For static
customization (same settings on every call), return constant ``TwiMLOptions``
from the resolver — no request inspection needed.
"""

from dotenv import load_dotenv

from tac import TAC, TACConfig
from tac.channels.voice import VoiceChannel, VoiceChannelConfig
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.models.voice import LanguageConfig, TwiMLOptions, TwiMLRequestContext
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


async def resolve_twiml(ctx: TwiMLRequestContext) -> TwiMLOptions:
    """Return TwiMLOptions overrides for this incoming call.

    Only set the fields you want to override — TAC fills in the rest
    (websocket URL, action URL, conversation configuration, welcome greeting).
    """
    if ctx.caller_country == "MX":
        primary_language = "es-MX"
        primary_voice = "es-MX-Neural2-A"
        welcome = "¡Hola! ¿En qué puedo ayudarte?"
    elif ctx.caller_country == "FR":
        primary_language = "fr-FR"
        primary_voice = "fr-FR-Neural2-A"
        welcome = "Bonjour ! Comment puis-je vous aider ?"
    else:
        primary_language = "en-US"
        primary_voice = "en-US-Journey-D"
        welcome = "Hello! How can I help?"

    return TwiMLOptions(
        language=primary_language,
        voice=primary_voice,
        tts_provider="google",
        transcription_provider="deepgram",
        interruptible="speech",
        welcome_greeting=welcome,
        # <Language> children let the caller switch languages mid-call
        languages=[
            LanguageConfig(code="en-US", voice="en-US-Journey-D", tts_provider="google"),
            LanguageConfig(code="es-MX", voice="es-MX-Neural2-A", tts_provider="google"),
        ],
    )


voice_channel = VoiceChannel(tac, config=VoiceChannelConfig(resolve_twiml_options=resolve_twiml))


if __name__ == "__main__":
    server = TACFastAPIServer(tac=tac, voice_channel=voice_channel)
    server.start()

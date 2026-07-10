"""
Voice Realtime Example — OpenAI Realtime API over Twilio Media Streams.

This is the *speech-to-speech* alternative to the ConversationRelay examples
(voice_streaming.py / relay_only.py). Instead of Twilio doing STT/TTS and TAC
handling a text turn, Twilio streams raw call audio to us over
``<Connect><Stream>`` and we bridge it to the OpenAI Realtime API, which does
its own speech recognition, reasoning, and speech synthesis.

Point your Twilio number's voice webhook at ``POST /twiml-realtime``.

Env vars required:
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_API_KEY, TWILIO_API_SECRET
- TWILIO_PHONE_NUMBER
- TWILIO_VOICE_PUBLIC_DOMAIN (ngrok or similar, e.g. 'example.ngrok.app')
- OPENAI_API_KEY

Install the realtime extra:
    pip install 'twilio-agent-connect[server,realtime]'
"""

from dotenv import load_dotenv

from tac import TAC, TACConfig
from tac.channels.realtime import RealtimeVoiceChannel, RealtimeVoiceChannelConfig
from tac.server import RealtimeVoiceServer

load_dotenv()

tac = TAC(config=TACConfig.from_env())

realtime_voice_channel = RealtimeVoiceChannel(tac, config=RealtimeVoiceChannelConfig.from_env())

if __name__ == "__main__":
    server = RealtimeVoiceServer(tac=tac, realtime_voice_channel=realtime_voice_channel)
    server.start()

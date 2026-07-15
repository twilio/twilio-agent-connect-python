"""
Voice Realtime Example — OpenAI Realtime API over Twilio Media Streams.

This is the *speech-to-speech* alternative to the ConversationRelay examples
(voice_streaming.py / relay_only.py). Instead of Twilio doing STT/TTS and TAC
handling a text turn, Twilio streams raw call audio to us over
``<Connect><Stream>`` and we bridge it to the OpenAI Realtime API, which does
its own speech recognition, reasoning, and speech synthesis.

Point your Twilio number's voice webhook at ``POST /voice-realtime`` (the
same URL the WebSocket audio bridge listens on, by default).

Env vars required:
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_API_KEY, TWILIO_API_SECRET
- TWILIO_PHONE_NUMBER
- TWILIO_VOICE_PUBLIC_DOMAIN (ngrok or similar, e.g. 'example.ngrok.app')
- OPENAI_API_KEY

Install the realtime extra:
    pip install 'twilio-agent-connect[server,realtime]'
"""

from datetime import datetime

from dotenv import load_dotenv

from tac import TAC, TACConfig
from tac.channels.realtime import RealtimeVoiceChannel, RealtimeVoiceChannelConfig
from tac.server import RealtimeVoiceServer
from tac.tools import function_tool

load_dotenv()

tac = TAC(config=TACConfig.from_env())


@function_tool()
def get_current_time() -> str:
    """Get the current date and time. Use this if the caller asks what time or day it is."""
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


realtime_voice_channel = RealtimeVoiceChannel(
    tac, config=RealtimeVoiceChannelConfig.from_env(tools=[get_current_time])
)

if __name__ == "__main__":
    server = RealtimeVoiceServer(tac=tac, realtime_voice_channel=realtime_voice_channel)
    server.start()

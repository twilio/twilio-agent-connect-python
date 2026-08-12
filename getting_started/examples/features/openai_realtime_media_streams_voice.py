"""
Example (POC): OpenAI Realtime API voice calls via Twilio Media Streams.

Unlike openai_realtime_sip_voice.py (where Twilio hands the call to OpenAI at
the SIP level and audio never touches our server), this example IS the audio
path: Twilio streams call audio to our own WebSocket, and this channel relays
it to/from the OpenAI Realtime WebSocket itself. That also means transcript
capture, tool calling, and barge-in all happen inline on one connection — no
separate sideband WebSocket needed, unlike the SIP channel.

What the Realtime session looks like (model, voice, turn detection,
instructions) is NOT baked into the channel's config — it's built per call by
reusing `tac.on_message_ready` (called with an empty user_message and no
memory_response, expected to return a JSON-encoded session dict instead of
reply text). The channel only knows how to mechanically bridge audio to
whatever session config that callback hands it.

One-time account setup:

1. Twilio Console:
   - Buy (or use an existing) Voice-capable phone number.
   - Point its Voice webhook at your public tunnel + OPENAI_REALTIME_TWIML_PATH
     (default /twiml). No SIP Trunk needed — just a normal Voice URL.

2. Run this script behind a public tunnel (e.g. `ngrok http 8000`) so Twilio
   can reach both the TwiML endpoint and the Media Stream WebSocket.

Env vars required:
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_API_KEY, TWILIO_API_SECRET
- TWILIO_PHONE_NUMBER
- TWILIO_VOICE_PUBLIC_DOMAIN (your ngrok domain or similar)
- OPENAI_API_KEY

Env vars that should NOT be set:
- TWILIO_CONVERSATION_CONFIGURATION_ID (this is a relay-only-style setup —
  no Conversation Orchestrator / Memory)
"""

import json
import os

from dotenv import load_dotenv

from tac import TAC, TACConfig
from tac.channels.openai_realtime_media_streams import (
    OpenAIRealtimeMediaStreamsChannel,
    OpenAIRealtimeMediaStreamsChannelConfig,
)
from tac.channels.openai_realtime_media_streams.config import TWILIO_AUDIO_FORMAT
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server.openai_realtime_media_streams_server import (
    OpenAIRealtimeMediaStreamsServer,
    OpenAIRealtimeMediaStreamsServerConfig,
)
from tac.tools import function_tool

load_dotenv()

tac = TAC(config=TACConfig.from_env())
assert not tac.is_orchestrator_enabled(), (
    "This example expects relay-only-style mode — unset TWILIO_CONVERSATION_CONFIGURATION_ID."
)

REALTIME_MODEL = "gpt-realtime-2.1"
SYSTEM_INSTRUCTIONS = (
    "You are a warm, friendly voice assistant speaking with a caller over the phone. "
    "Keep responses short — a sentence or two per turn. No markdown, emojis, or bullet "
    "lists; your words will be spoken aloud."
)


@function_tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny and 72F in {city}."


channel = OpenAIRealtimeMediaStreamsChannel(
    tac,
    OpenAIRealtimeMediaStreamsChannelConfig(
        model=REALTIME_MODEL,
        tools=[get_weather],
        welcome_greeting="Hello! How can I help you today?",
    ),
)


@tac.on_message_ready
async def build_session_config(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> str:
    """Build the OpenAI session.update payload for this call.

    Reused from on_message_ready (called once per call, right after the
    OpenAI WebSocket connects, with an empty user_message and no
    memory_response — neither applies to a Realtime session). Must return a
    JSON-encoded session dict rather than reply text, since
    trigger_message_ready's contract is str | None.
    """
    return json.dumps(
        {
            "type": "realtime",
            "model": REALTIME_MODEL,
            "output_modalities": ["audio"],
            "instructions": SYSTEM_INSTRUCTIONS,
            "audio": {
                "input": {
                    "format": TWILIO_AUDIO_FORMAT,
                    "turn_detection": {"type": "semantic_vad", "eagerness": "low"},
                    "transcription": {"model": "gpt-live-transcribe"},
                },
                "output": {"format": TWILIO_AUDIO_FORMAT, "voice": "marin"},
            },
            "tools": [get_weather.to_realtime_format()],
            "tool_choice": "auto",
        }
    )


@tac.on_conversation_ended
async def handle_conversation_ended(context: ConversationSession) -> None:
    """Print the full transcript once the Media Stream WebSocket closes."""
    transcript = context.metadata.get("transcript", [])
    print(f"Call {context.conversation_id} ended. Transcript:")
    for turn in transcript:
        print(f"  {turn['role']}: {turn['text']}")


if __name__ == "__main__":
    server_config = OpenAIRealtimeMediaStreamsServerConfig(
        port=int(os.environ.get("PORT", "8000")),
        twiml_path=os.environ.get("OPENAI_REALTIME_TWIML_PATH", "/twiml"),
    )
    server = OpenAIRealtimeMediaStreamsServer(tac=tac, channel=channel, config=server_config)
    server.start()

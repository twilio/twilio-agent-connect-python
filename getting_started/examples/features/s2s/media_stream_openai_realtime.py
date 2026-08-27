"""
Example (POC): OpenAI Realtime API voice calls via Twilio Media Streams.

Twilio streams call audio to our own WebSocket, and this provider relays it
to/from the OpenAI Realtime WebSocket. The session config (model, voice, turn
detection, instructions) is static, set once on the provider's config.

Unlike the ConversationRelay examples, this runs in relay-only mode
regardless of TAC's Conversation Orchestrator configuration — there's no
profile lookup or CO conversation for a Media Streams call.

One-time account setup:

1. Twilio Console:
   - Buy (or use an existing) Voice-capable phone number.
   - Point its Voice webhook at your public tunnel + TACServerConfig.twiml_path
     (default /twiml). No SIP Trunk needed — just a normal Voice URL.

2. Run this script behind a public tunnel (e.g. `ngrok http 8000`) so Twilio
   can reach both the TwiML endpoint and the Media Stream WebSocket.

Env vars required:
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_API_KEY, TWILIO_API_SECRET
- TWILIO_PHONE_NUMBER
- TWILIO_VOICE_PUBLIC_DOMAIN (your ngrok domain or similar)
- OPENAI_API_KEY

Install the extra dependencies this example needs:
    pip install "tac[server,openai-realtime]"

Usage:
    python media_stream_openai_realtime.py                    # inbound only
    python media_stream_openai_realtime.py --to +16505551234  # also place an outbound call
"""

import argparse
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from tac import TAC, TACConfig
from tac.channels.voice import VoiceChannel
from tac.channels.voice.media_streams.openai_realtime import (
    TWILIO_MEDIA_STREAM_AUDIO_FORMAT,
    OpenAIRealtimeProviderConfig,
)
from tac.models.outbound import InitiateVoiceConversationOptions
from tac.models.session import ConversationSession
from tac.server import TACFastAPIServer
from tac.tools import function_tool

load_dotenv()

tac = TAC(config=TACConfig.from_env())


@function_tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny and 72F in {city}."


SESSION_CONFIG = {
    "type": "realtime",
    "model": "gpt-realtime-2.1",
    "output_modalities": ["audio"],
    "instructions": (
        "You are a warm, friendly voice assistant speaking with a caller over the phone. "
        "Keep responses short — a sentence or two per turn. No markdown, emojis, or bullet "
        "lists; your words will be spoken aloud."
    ),
    "audio": {
        "input": {
            "format": TWILIO_MEDIA_STREAM_AUDIO_FORMAT,
            "turn_detection": {"type": "semantic_vad", "eagerness": "high"},
            "transcription": {"model": "gpt-live-transcribe"},
        },
        "output": {"format": TWILIO_MEDIA_STREAM_AUDIO_FORMAT, "voice": "marin"},
    },
    "tools": [get_weather.to_realtime_format()],
    "tool_choice": "auto",
}

voice_channel = VoiceChannel(
    tac,
    config=OpenAIRealtimeProviderConfig(
        tools=[get_weather],
        welcome_greeting_response={"instructions": "Hello! How can I help you today?"},
        session_config=SESSION_CONFIG,
    ),
)


@tac.on_conversation_ended
async def handle_conversation_ended(context: ConversationSession) -> None:
    """Print the full transcript once the Media Stream WebSocket closes."""
    transcript = context.metadata.get("transcript", [])
    print(f"Call {context.conversation_id} ended. Transcript:")
    for turn in transcript:
        print(f"  {turn['role']}: {turn['text']}")


async def place_outbound_call(to: str) -> None:
    result = await voice_channel.initiate_outbound_conversation(
        InitiateVoiceConversationOptions(to=to)
    )
    print(f"Call placed to {to} (SID: {result.call_sid})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI Realtime voice example")
    parser.add_argument("--to", help="Destination phone number to call, e.g. +16505551234")
    args = parser.parse_args()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if args.to:
            asyncio.create_task(place_outbound_call(args.to))
        yield

    app = FastAPI(title="TAC OpenAI Realtime Example", lifespan=lifespan)

    server = TACFastAPIServer(tac=tac, voice_channel=voice_channel, app=app)
    server.start()

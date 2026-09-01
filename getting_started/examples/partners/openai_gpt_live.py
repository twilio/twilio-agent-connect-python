"""
Example: OpenAI GPT-Live voice calls via Twilio Media Streams.

GPT-Live is an unreleased OpenAI alpha API, approved-project access only.
This example will fail to connect unless OPENAI_API_KEY belongs to a
GPT-Live alpha-approved project.

Twilio streams call audio to our own WebSocket, and this provider relays it
to/from the GPT-Live WebSocket. Unlike the Realtime API provider:
- GPT-Live is full-duplex — there's no barge-in/truncate to configure, the
  model handles interruption itself.
- Tool calls go through Responses delegation (`session_config["delegation"]`),
  not direct function-calling — `web_search` below is a hosted tool OpenAI
  runs server-side, no code needed here for it to work.
- The greeting is sent via `welcome_instruction` (a `session.context.append`
  once `session.started` arrives), not a `response.create`-equivalent. Unlike
  Realtime's `welcome_greeting_response`, the caller supplies the full
  instruction verbatim — GPT-Live won't speak first from a bare greeting
  string, so word it as an instruction, e.g. "Greet the caller using: ...".

Unlike the ConversationRelay examples, this runs in relay-only mode
regardless of TAC's Conversation Orchestrator configuration — there's no
profile lookup or CO conversation for a Media Streams call.

One-time account setup: same as openai_realtime.py.

Env vars required:
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_API_KEY, TWILIO_API_SECRET
- TWILIO_PHONE_NUMBER
- TWILIO_VOICE_PUBLIC_DOMAIN (your ngrok domain or similar)
- OPENAI_API_KEY (must belong to a GPT-Live alpha-approved project)

Install the extra dependencies this example needs:
    pip install "tac[server,gpt-live]"

Usage:
    python openai_gpt_live.py                    # inbound only
    # also place an outbound call — make sure you have permission to call
    # this number; unsolicited calls risk the number being flagged as spam
    python openai_gpt_live.py --to +16505551234
"""

import argparse
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from tac import TAC, TACConfig
from tac.channels.voice import VoiceChannel
from tac.channels.voice.media_streams.gpt_live import (
    TWILIO_MEDIA_STREAM_AUDIO_FORMAT,
    GPTLiveProviderConfig,
)
from tac.models.outbound import InitiateVoiceConversationOptionsGPTLive
from tac.models.session import ConversationSession
from tac.server import TACFastAPIServer
from tac.tools import function_tool

load_dotenv()

tac = TAC(config=TACConfig.from_env())


@function_tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny and 72F in {city}."


DEFAULT_SESSION_CONFIG = {
    "instructions": (
        "You are a warm, friendly voice assistant speaking with a caller over the phone. "
        "Keep responses short — a sentence or two per turn. No markdown, emojis, or "
        "bullet lists; your words will be spoken aloud."
    ),
    "audio": {
        "format": TWILIO_MEDIA_STREAM_AUDIO_FORMAT,
        "output": {"voice": "marin"},
    },
    "delegation": {
        "type": "responses",
        "responses": {
            "model": "gpt-5.6-sol",
            "tools": [{"type": "web_search"}, get_weather.to_realtime_format()],
            "tool_choice": "auto",
        },
    },
}


voice_channel = VoiceChannel(
    tac,
    config=GPTLiveProviderConfig(
        tools=[get_weather],
        welcome_instruction=(
            "Greet the caller immediately using the exact text below. Do not wait "
            "for the caller to speak first. After the greeting, pause and listen."
            "\n\nHello! How can I help you today?"
        ),
        default_session_config=DEFAULT_SESSION_CONFIG,
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
        InitiateVoiceConversationOptionsGPTLive(to=to)
    )
    print(f"Call placed to {to} (SID: {result.call_sid})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-Live voice example")
    parser.add_argument("--to", help="Destination phone number to call, e.g. +16505551234")
    args = parser.parse_args()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if args.to:
            asyncio.create_task(place_outbound_call(args.to))
        yield

    app = FastAPI(title="TAC GPT-Live Example", lifespan=lifespan)

    server = TACFastAPIServer(tac=tac, voice_channel=voice_channel, app=app)
    server.start()

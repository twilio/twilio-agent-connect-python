"""
Example: Outbound voice with AMD and call events.

Places an outbound ConversationRelay call with answering machine detection
(AMD) enabled, and reacts to Twilio's out-of-band call webhooks:

- on a detected answering machine -> hang up (don't monologue at voicemail)
- on no-answer / busy / failed     -> report the call as unreached

Shows the three outbound Calls-API seams:
  1. InitiateVoiceConversationOptions.call_options -> forwarded to calls.create
  2. VoiceChannel.on_call_event                    -> one handler for status/AMD
  3. VoiceChannel.end_call(call_sid)               -> hang up + session cleanup

TACFastAPIServer auto-registers the call-event route (at voice_call_event_path)
and auto-wires the callback URLs from TWILIO_VOICE_PUBLIC_DOMAIN, so there is no
webhook wiring to do here.

Env vars required:
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_API_KEY, TWILIO_API_SECRET
- TWILIO_PHONE_NUMBER
- TWILIO_VOICE_PUBLIC_DOMAIN (ngrok or similar)

Usage:
    python outbound_amd.py --to +16505551234
"""

import argparse
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from tac import TAC, TACConfig
from tac.channels.voice import CallEvent, VoiceChannel
from tac.models.outbound import InitiateVoiceConversationOptions
from tac.server import TACFastAPIServer

load_dotenv()

tac = TAC(config=TACConfig.from_env())
voice_channel = VoiceChannel(tac)


# Seam 2: one handler for every call webhook, discriminated by event.kind.
async def handle_call_event(event: CallEvent) -> None:
    if event.kind == "amd":
        print(f"[AMD] {event.call_sid}: answered_by={event.answered_by}")
        if event.answered_by and event.answered_by.startswith("machine"):
            # Seam 3: hang up instead of talking to an answering machine.
            await voice_channel.end_call(event.call_sid)
    elif event.kind == "status":
        print(f"[STATUS] {event.call_sid}: {event.call_status}")
        if event.call_status in {"no-answer", "busy", "failed"}:
            print(f"[STATUS] {event.call_sid} unreached — queue retry")


voice_channel.on_call_event(handle_call_event)


async def place_call(to: str) -> None:
    result = await voice_channel.initiate_outbound_conversation(
        InitiateVoiceConversationOptions(
            to=to,
            # Seam 1: forwarded to calls.create(). No callback URLs needed —
            # TAC auto-wires them from TWILIO_VOICE_PUBLIC_DOMAIN.
            call_options={
                "async_amd": "true",
                "status_callback_event": ["initiated", "ringing", "answered", "completed"],
                "timeout": 30,
            },
        )
    )
    print(f"Call placed to {to} (CallSid: {result.call_sid})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outbound voice with AMD")
    parser.add_argument("--to", required=True, help="Destination number, e.g. +16505551234")
    args = parser.parse_args()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        asyncio.create_task(place_call(args.to))
        yield

    app = FastAPI(lifespan=lifespan)
    # Registers /twiml, the WebSocket, the action callback, AND the call-event
    # route that status/AMD/recording webhooks POST to.
    server = TACFastAPIServer(tac=tac, voice_channel=voice_channel, app=app)
    server.start()

"""
Example: Voice call events (status, AMD, and recording).

Places an outbound ConversationRelay call with answering machine detection and
recording enabled, then reacts to Twilio's call webhooks: hang up on voicemail,
and log which calls went unreached. Each handler is optional — register only
the events you care about.

TACFastAPIServer registers the call-event routes and auto-wires their URLs from
TWILIO_VOICE_PUBLIC_DOMAIN, so there is no webhook setup here. The same handlers
also fire for inbound calls whose status/recording callbacks point at TAC.

Env vars:
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_API_KEY, TWILIO_API_SECRET
- TWILIO_PHONE_NUMBER
- TWILIO_VOICE_PUBLIC_DOMAIN (ngrok or similar)

Usage:
    python voice_call_events.py --to +16505551234
"""

import argparse
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from tac import TAC, TACConfig
from tac.channels.voice import AmdEvent, CallStatusEvent, RecordingEvent, VoiceChannel
from tac.models.outbound import InitiateVoiceConversationOptions
from tac.server import TACFastAPIServer

load_dotenv()

tac = TAC(config=TACConfig.from_env())
voice_channel = VoiceChannel(tac)


async def on_call_status(event: CallStatusEvent) -> None:
    print(f"[STATUS] {event.call_sid}: {event.call_status}")
    if event.call_status in {"no-answer", "busy", "failed"}:
        print(f"[STATUS] {event.call_sid} unreached — queue retry")


async def on_amd(event: AmdEvent) -> None:
    print(f"[AMD] {event.call_sid}: answered_by={event.answered_by}")
    if event.answered_by and event.answered_by.startswith("machine"):
        await voice_channel.end_call(event.call_sid)  # voicemail → hang up


async def on_recording(event: RecordingEvent) -> None:
    print(f"[RECORDING] {event.call_sid}: {event.recording_status} {event.recording_url}")


voice_channel.on_call_status(on_call_status)
voice_channel.on_amd(on_amd)
voice_channel.on_recording(on_recording)


async def place_call(to: str) -> None:
    result = await voice_channel.initiate_outbound_conversation(
        InitiateVoiceConversationOptions(
            to=to,
            # Passed through to calls.create(). Note the mixed types — Twilio
            # takes async_amd as a string but record as a bool.
            call_options={
                "async_amd": "true",  # required for the AMD callback (sync AMD won't fire it)
                "record": True,
                "timeout": 30,
            },
        )
    )
    print(f"Call placed to {to} (CallSid: {result.call_sid})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice call events (status, AMD, recording)")
    parser.add_argument("--to", required=True, help="Destination number, e.g. +16505551234")
    args = parser.parse_args()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        asyncio.create_task(place_call(args.to))
        yield

    app = FastAPI(lifespan=lifespan)
    server = TACFastAPIServer(tac=tac, voice_channel=voice_channel, app=app)
    server.start()

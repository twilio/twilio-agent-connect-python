"""
Example: Voice call events (status, AMD, and recording).

Places an outbound ConversationRelay call with answering machine detection and
recording enabled, then reacts to Twilio's call webhooks: hang up on voicemail,
log which calls went unreached. Each handler is optional.

TACFastAPIServer registers the routes and auto-wires their URLs from
TWILIO_VOICE_PUBLIC_DOMAIN, so there's no webhook setup here.

Outbound only: machine_detection and record are calls.create parameters, so AMD
and recording have no inbound equivalent. on_call_status does work for inbound,
but the URL isn't auto-wired — point the number's "Call Status Changes" webhook
at <TWILIO_VOICE_PUBLIC_DOMAIN>/twilio/call-events/status yourself.

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
from tac.models.outbound import CallOptions, InitiateVoiceConversationOptions
from tac.server import TACFastAPIServer

load_dotenv()

tac = TAC(config=TACConfig.from_env())
voice_channel = VoiceChannel(tac)


async def on_call_status(event: CallStatusEvent) -> None:
    print(f"[STATUS] {event.call_sid}: {event.call_status}")
    # The conversation session — conversation_id, profile, metadata. Set on the first prompt.
    session = voice_channel.get_conversation_session_by_call_sid(event.call_sid)
    print(f"[STATUS] session lookup -> {session.conversation_id if session else None}")
    if event.is_unreached:
        print(f"[STATUS] {event.call_sid} unreached — queue retry")


async def on_amd(event: AmdEvent) -> None:
    print(f"[AMD] {event.call_sid}: answered_by={event.answered_by}")
    if event.is_machine:
        await voice_channel.end_call(event.call_sid)  # voicemail → hang up


async def on_recording(event: RecordingEvent) -> None:
    print(f"[RECORDING] {event.call_sid}: {event.recording_status} {event.recording_url}")


# Registering is also what puts each callback URL on the outbound call below.
# Skip a handler and TAC omits its URL, so Twilio never posts that event.
voice_channel.on_call_status(on_call_status)
voice_channel.on_amd(on_amd)
voice_channel.on_recording(on_recording)


async def place_call(to: str) -> None:
    result = await voice_channel.initiate_outbound_conversation(
        InitiateVoiceConversationOptions(
            to=to,
            call_options=CallOptions(
                # AMD needs both. "Enable" reports as early as possible, which is
                # what you want to hang up on voicemail.
                machine_detection="Enable",
                async_amd=True,
                record=True,
                timeout=30,
            ),
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

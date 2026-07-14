"""
Example: Voice call events (status, AMD, and recording).

Places an outbound ConversationRelay call with answering machine detection
(AMD) and recording enabled, and reacts to Twilio's out-of-band call webhooks.
Twilio posts to three independent callback URLs, and TAC serves one route per
callback, so you register a separate typed handler for each:

- on_call_status -> call progress: ringing / answered / completed, and
                    no-answer / busy / failed (report as unreached)
- on_amd       -> answering machine detection result; hang up on a machine
                  instead of monologuing at voicemail
- on_recording -> recording ready (recording_url available)

Each handler is independently optional — register only the ones you need.

Shows the three Calls-API seams:
  1. InitiateVoiceConversationOptions.call_options -> forwarded to calls.create
  2. VoiceChannel.on_call_status / on_amd / on_recording -> one handler per callback
  3. VoiceChannel.end_call(call_sid)                -> hang up + session cleanup

This example enables AMD/recording by placing an OUTBOUND call (call_options
only applies to calls TAC creates). The call-event and end_call seams also work
for INBOUND calls — if status/recording callbacks are enabled on the number's
incoming-call config and pointed at the matching route, the same handlers
receive those events and end_call() hangs up an inbound call the same way.

TACFastAPIServer auto-registers the three call-event routes (under
voice_call_event_path: /status, /amd, /recording) and auto-wires the callback
URLs from TWILIO_VOICE_PUBLIC_DOMAIN, so there is no webhook wiring to do here.

Env vars required:
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


# Seam 2: one typed handler per Twilio call callback.
async def on_call_status(event: CallStatusEvent) -> None:
    print(f"[STATUS] {event.call_sid}: {event.call_status}")
    if event.call_status in {"no-answer", "busy", "failed"}:
        print(f"[STATUS] {event.call_sid} unreached — queue retry")


async def on_amd(event: AmdEvent) -> None:
    print(f"[AMD] {event.call_sid}: answered_by={event.answered_by}")
    if event.answered_by and event.answered_by.startswith("machine"):
        # Seam 3: hang up instead of talking to an answering machine.
        await voice_channel.end_call(event.call_sid)


async def on_recording(event: RecordingEvent) -> None:
    print(f"[RECORDING] {event.call_sid}: {event.recording_status} {event.recording_url}")


voice_channel.on_call_status(on_call_status)
voice_channel.on_amd(on_amd)
voice_channel.on_recording(on_recording)


async def place_call(to: str) -> None:
    result = await voice_channel.initiate_outbound_conversation(
        InitiateVoiceConversationOptions(
            to=to,
            # Seam 1: forwarded to calls.create(). No callback URLs needed —
            # TAC auto-wires the status/AMD/recording callbacks (each to its own
            # route) from TWILIO_VOICE_PUBLIC_DOMAIN when the feature is enabled.
            call_options={
                "async_amd": "true",  # AMD (Twilio types this as a string) -> on_amd
                "record": True,  # recording (Twilio types this as a bool) -> on_recording
                "status_callback_event": ["initiated", "ringing", "answered", "completed"],
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
    # Registers /twiml, the WebSocket, the action callback, AND the three
    # call-event routes that status/AMD/recording webhooks POST to.
    server = TACFastAPIServer(tac=tac, voice_channel=voice_channel, app=app)
    server.start()

"""
Example (POC): OpenAI Realtime API voice calls via Twilio SIP Trunking.

Unlike voice_streaming.py (ConversationRelay) or a Twilio Media Streams audio
bridge, this example never touches audio. Twilio forwards the call at the SIP
level directly to OpenAI; TAC only handles the `realtime.call.incoming`
webhook and decides how to accept the call.

One-time account setup (see conversation notes / OpenAI + Twilio docs):

1. OpenAI (platform.openai.com/settings):
   - Webhooks -> Create a webhook -> event type `realtime.call.incoming`,
     URL = your public tunnel + OPENAI_REALTIME_WEBHOOK_PATH (default
     /openai/incoming-call). Copy the webhook secret.
   - Settings -> General -> copy your Project ID (proj_xxxxx).

2. Twilio Console:
   - Buy a Voice-capable phone number.
   - Elastic SIP Trunking -> create a trunk -> Origination URI:
       sip:proj_xxxxx@sip.api.openai.com;transport=tls
   - Attach the phone number to that trunk.

3. Run this script behind a public tunnel (e.g. `ngrok http 8000`) so OpenAI's
   webhook can reach OPENAI_REALTIME_WEBHOOK_PATH.

Env vars required:
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_API_KEY, TWILIO_API_SECRET
- TWILIO_PHONE_NUMBER (TAC requires it even though this channel doesn't use it)
- OPENAI_API_KEY
- OPENAI_WEBHOOK_SECRET

Env vars that should NOT be set:
- TWILIO_CONVERSATION_CONFIGURATION_ID (this is a relay-only-style setup —
  no Conversation Orchestrator / Memory)
"""

import os

from dotenv import load_dotenv

from tac import TAC, TACConfig
from tac.channels.openai_realtime_sip import (
    OpenAIRealtimeSipCallIncoming,
    OpenAIRealtimeSipChannel,
    OpenAIRealtimeSipSessionConfig,
)
from tac.models.session import ConversationSession
from tac.server.openai_realtime_sip_server import (
    OpenAIRealtimeSipServer,
    OpenAIRealtimeSipServerConfig,
)
from tac.tools import function_tool

load_dotenv()

tac = TAC(config=TACConfig.from_env())
assert not tac.is_orchestrator_enabled(), (
    "This example expects relay-only-style mode — unset TWILIO_CONVERSATION_CONFIGURATION_ID."
)

channel = OpenAIRealtimeSipChannel(tac)

SYSTEM_INSTRUCTIONS = (
    "You are a helpful voice assistant speaking with a user over the phone. "
    "Keep responses short and conversational."
)


@function_tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny and 72F in {city}."


channel.register_tool(get_weather)


@channel.on_call_incoming
async def handle_incoming(event: OpenAIRealtimeSipCallIncoming) -> OpenAIRealtimeSipSessionConfig:
    """Accept every incoming call with a static model/voice/instructions config.

    Leaving `tools` unset here means the channel auto-fills it from every tool
    registered via `channel.register_tool(...)` above (just `get_weather`).
    Setting `input_transcription_model` opens the control WebSocket so we get
    a transcript of what the caller said, not just what the model said.
    """
    print(f"Incoming call {event.call_id} from {event.sip_header('From')}")
    return OpenAIRealtimeSipSessionConfig(
        model="gpt-realtime-2.1",
        instructions=SYSTEM_INSTRUCTIONS,
        voice="alloy",
        input_transcription_model="gpt-live-transcribe",
    )


@tac.on_conversation_ended
async def handle_conversation_ended(context: ConversationSession) -> None:
    """Print the full transcript once the control WebSocket sees the call end."""
    transcript = context.metadata.get("transcript", [])
    print(f"Call {context.conversation_id} ended. Transcript:")
    for turn in transcript:
        print(f"  {turn['role']}: {turn['text']}")


if __name__ == "__main__":
    server_config = OpenAIRealtimeSipServerConfig(
        port=int(os.environ.get("PORT", "8000")),
        webhook_path=os.environ.get("OPENAI_REALTIME_WEBHOOK_PATH", "/openai/incoming-call"),
    )
    server = OpenAIRealtimeSipServer(channel=channel, config=server_config)
    server.start()

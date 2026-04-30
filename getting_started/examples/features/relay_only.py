"""
Example: Relay-only mode — pure ConversationRelay without Conversation Orchestrator.

When `conversation_configuration_id` is omitted from TACConfig, TAC runs in
ConversationRelay-only mode:
- VoiceChannel works with ConversationRelay only (no CO session, no Memory).
- Messaging channels (SMS, Chat, ...) fail at construction.
- TAC.retrieve_memory returns an empty TACMemoryResponse.

Use this when you want TAC's voice plumbing (TwiML, WebSocket, callbacks) but
are bringing your own memory/state layer.

Env vars required:
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_API_KEY, TWILIO_API_SECRET
- TWILIO_PHONE_NUMBER
- TWILIO_VOICE_PUBLIC_DOMAIN (ngrok or similar)
- OPENAI_API_KEY

Env vars that should NOT be set for relay-only mode:
- TWILIO_CONVERSATION_CONFIGURATION_ID
"""

from collections.abc import AsyncGenerator
from typing import Any

from agents import Agent, Runner, set_tracing_disabled
from dotenv import load_dotenv

from tac import TAC, TACConfig
from tac.channels.voice import VoiceChannel
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer

load_dotenv()
set_tracing_disabled(True)

# No conversation_configuration_id — TAC runs in relay-only mode.
tac = TAC(config=TACConfig.from_env())
assert not tac.is_orchestrator_enabled(), (
    "This example expects relay-only mode — unset TWILIO_CONVERSATION_CONFIGURATION_ID."
)

voice_channel = VoiceChannel(tac)

SYSTEM_INSTRUCTIONS = (
    "You are a voice assistant speaking with a user over the phone. "
    "Keep responses short and conversational — a sentence or two. "
    "Do not use markdown, asterisks, bullets, or emojis; your words will be "
    "spoken aloud."
)

agent = Agent(name="Voice Assistant", instructions=SYSTEM_INSTRUCTIONS)

conversation_history: dict[str, list[Any]] = {}


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> None:
    """Stream voice responses through the OpenAI Agents SDK.

    Returns None and manually calls send_response() with an async generator
    so tokens are sent to the caller as they arrive from the LLM.
    """
    conv_id = context.conversation_id

    history = conversation_history.get(conv_id, [])
    agent_input = history + [{"role": "user", "content": user_message}]

    async def stream_tokens() -> AsyncGenerator[str, None]:
        result = Runner.run_streamed(agent, agent_input)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, "delta"):
                yield event.data.delta
        conversation_history[conv_id] = result.to_input_list()

    await voice_channel.send_response(conv_id, stream_tokens())


tac.on_message_ready(handle_message_ready)


async def handle_conversation_ended(context: ConversationSession) -> None:
    conversation_history.pop(context.conversation_id, None)


tac.on_conversation_ended(handle_conversation_ended)


if __name__ == "__main__":
    server = TACFastAPIServer(tac=tac, voice_channel=voice_channel)
    server.start()

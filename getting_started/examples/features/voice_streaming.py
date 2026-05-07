"""
Voice Streaming Example with OpenAI Agents SDK

Streaming reduces latency by sending LLM tokens immediately to the caller.

Performance comparison (streaming vs non-streaming):
- Streaming: Caller hears first words in ~0.5-0.7s (first token latency)
- Non-streaming: Caller waits ~1.0-1.5s for full LLM response before hearing anything
- Result: ~40-50% faster time-to-first-audio with streaming
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

tac = TAC(config=TACConfig.from_env())
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
    user_message: str, context: ConversationSession, memory_response: TACMemoryResponse | None
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

if __name__ == "__main__":
    server = TACFastAPIServer(tac=tac, voice_channel=voice_channel)
    server.start()

"""
Voice Streaming Example with OpenAI

Streaming reduces latency by sending LLM tokens immediately to the caller.

Performance comparison (streaming vs non-streaming):
- Streaming: Caller hears first words in ~0.5-0.7s (first token latency)
- Non-streaming: Caller waits ~1.0-1.5s for full LLM response before hearing anything
- Result: ~40-50% faster time-to-first-audio with streaming
"""

import os
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from tac import TAC, TACConfig
from tac.channels.voice import VoiceChannel
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer

load_dotenv()

tac = TAC(config=TACConfig.from_env())
voice_channel = VoiceChannel(tac)
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

conversation_history: dict[str, list[ChatCompletionMessageParam]] = {}
SYSTEM_MESSAGE: ChatCompletionMessageParam = {
    "role": "system",
    "content": (
        "You are a voice assistant speaking with a user over the phone. "
        "Keep responses short and conversational — a sentence or two. "
        "Do not use markdown, asterisks, bullets, or emojis; your words "
        "will be spoken aloud."
    ),
}


async def handle_message_ready(
    user_message: str, context: ConversationSession, memory_response: TACMemoryResponse | None
) -> None:
    """Return None and manually call send_response() with an async generator for streaming."""
    conv_id = context.conversation_id

    if conv_id not in conversation_history:
        conversation_history[conv_id] = [SYSTEM_MESSAGE.copy()]

    conversation_history[conv_id].append({"role": "user", "content": user_message})

    # Use conversation history for streaming
    messages = conversation_history[conv_id][:]  # Copy list

    async def stream_tokens() -> AsyncGenerator[str, None]:
        response_tokens = []

        # Stream from OpenAI
        stream = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                response_tokens.append(token)
                yield token

        full_response = "".join(response_tokens)
        conversation_history[conv_id].append({"role": "assistant", "content": full_response})

    await voice_channel.send_response(conv_id, stream_tokens())


tac.on_message_ready(handle_message_ready)

if __name__ == "__main__":
    server = TACFastAPIServer(tac=tac, voice_channel=voice_channel)
    server.start()

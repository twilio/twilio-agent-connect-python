"""
Feature: Demo UI Dashboard

SMS/Voice example with an observation dashboard for monitoring active sessions,
conversation history, agent context (profile, memory), and CI events.

WARNING: This dashboard has no authentication. It is intended for local
development and demos only. Do not use in production.

Usage:
    uv run python getting_started/examples/features/dashboard/app.py

    Then open http://localhost:8000/dashboard

Requires OPENAI_API_KEY in addition to the usual TAC env vars.
For CI events, set CONVERSATION_INTELLIGENCE_CONFIGURATION_ID.
"""

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from tac import TAC, TACConfig
from tac.adapters.openai import with_tac_memory
from tac.channels.sms import SMSChannel, SMSChannelConfig
from tac.channels.voice import VoiceChannel, VoiceChannelConfig
from tac.core.logging import get_logger
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer

load_dotenv()

logger = get_logger(__name__)

tac = TAC(config=TACConfig.from_env())

voice_channel = VoiceChannel(tac, config=VoiceChannelConfig(memory_mode="always"))
sms_channel = SMSChannel(tac, config=SMSChannelConfig(memory_mode="always"))

openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

conversation_history: dict[str, list[ChatCompletionMessageParam]] = {}

SYSTEM_MESSAGE: ChatCompletionSystemMessageParam = {
    "role": "system",
    "content": (
        "You are a customer service agent speaking with a user over voice or SMS. "
        "Keep responses short and conversational — a sentence or two. "
        "Do not use markdown, asterisks, bullets, or emojis; your words will be "
        "spoken aloud or sent as plain text."
    ),
}


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> str:
    conv_id = context.conversation_id

    try:
        if conv_id not in conversation_history:
            conversation_history[conv_id] = [SYSTEM_MESSAGE]

        user_msg: ChatCompletionUserMessageParam = {"role": "user", "content": user_message}
        conversation_history[conv_id].append(user_msg)

        client = with_tac_memory(openai_client, memory_response, context)

        response = await client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=conversation_history[conv_id],
        )

        llm_response = response.choices[0].message.content or ""

        assistant_msg: ChatCompletionAssistantMessageParam = {
            "role": "assistant",
            "content": llm_response,
        }
        conversation_history[conv_id].append(assistant_msg)

        return llm_response

    except Exception as e:
        logger.error("Error processing message", conversation_id=conv_id, error=str(e))
        return "Sorry, I encountered an error processing your message."


tac.on_message_ready(handle_message_ready)

if __name__ == "__main__":
    from tac.server.config import TACServerConfig

    server_config = TACServerConfig.from_env()
    if os.environ.get("CONVERSATION_INTELLIGENCE_CONFIGURATION_ID"):
        server_config.cintel_webhook_path = "/ci-webhook"

    server = TACFastAPIServer(
        tac=tac, voice_channel=voice_channel, messaging_channels=[sms_channel], config=server_config
    )

    from dashboard import mount_dashboard  # type: ignore[import-not-found]

    mount_dashboard(
        server.app, tac, channels=[sms_channel, voice_channel], messages=conversation_history
    )

    logger.info(f"Dashboard: http://localhost:{server.config.port}/dashboard")
    server.start()

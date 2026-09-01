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
from tac.channels.voice import ConversationRelayProviderConfig, VoiceChannel
from tac.core.logging import get_logger
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer

load_dotenv()

logger = get_logger(__name__)

tac = TAC(config=TACConfig.from_env())

voice_channel = VoiceChannel(tac, config=ConversationRelayProviderConfig(memory_mode="always"))
sms_channel = SMSChannel(tac, config=SMSChannelConfig(memory_mode="always"))

openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

conversation_history: dict[str, list[ChatCompletionMessageParam]] = {}

# Messaging channels are stateless — they derive a session per webhook and keep
# nothing — so the dashboard's live messaging panel is fed by the app instead.
# Cleared in on_conversation_ended below.
messaging_sessions: dict[str, ConversationSession] = {}

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
    if context.channel != voice_channel.get_channel_name():
        messaging_sessions[conv_id] = context

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


async def handle_conversation_ended(context: ConversationSession) -> None:
    messaging_sessions.pop(context.conversation_id, None)
    conversation_history.pop(context.conversation_id, None)


tac.on_message_ready(handle_message_ready)
tac.on_conversation_ended(handle_conversation_ended)

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
        server.app,
        tac,
        channels=[sms_channel, voice_channel],
        messages=conversation_history,
        messaging_sessions=messaging_sessions,
    )

    logger.info(f"Dashboard: http://localhost:{server.config.port}/dashboard")
    server.start()

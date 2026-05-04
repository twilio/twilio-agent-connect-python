"""
Example: WhatsApp Channel with OpenAI Agents SDK

Demonstrates WhatsApp channel with TAC memory injection.
WhatsApp supports rich media and interactive messaging.

Requires ``OPENAI_API_KEY`` and ``TWILIO_WHATSAPP_NUMBER`` in addition to standard TAC env vars.

Usage:
    python whatsapp.py

Then send messages to your Twilio WhatsApp number from your phone.
"""

from typing import Any

from agents import Agent, Runner, set_tracing_disabled
from dotenv import load_dotenv

from tac import TAC, TACConfig
from tac.channels.whatsapp import WhatsAppChannel, WhatsAppChannelConfig
from tac.core.logging import get_logger
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer

load_dotenv()
set_tracing_disabled(True)

logger = get_logger(__name__)

tac = TAC(config=TACConfig.from_env())

whatsapp_channel = WhatsAppChannel(
    tac,
    config=WhatsAppChannelConfig(
        auto_retrieve_memory=True,
    ),
)

conversation_history: dict[str, list[Any]] = {}

SYSTEM_INSTRUCTIONS = (
    "You are a friendly, helpful AI customer service agent. "
    "Keep responses conversational and concise. "
    "Do not use markdown formatting."
)


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> str:
    """Handle incoming WhatsApp messages with TAC memory injection."""
    instructions = SYSTEM_INSTRUCTIONS
    if memory_response:
        memory_sections = memory_response.build_memory_prompts()
        if memory_sections:
            instructions += "\n\n" + "\n\n".join(memory_sections)

    agent = Agent(name="WhatsApp Customer Service Agent", instructions=instructions)

    history = conversation_history.get(context.conversation_id, [])
    agent_input = history + [{"role": "user", "content": user_message}]

    result = await Runner.run(agent, agent_input)

    conversation_history[context.conversation_id] = result.to_input_list()
    return result.final_output_as(str)


tac.on_message_ready(handle_message_ready)

if __name__ == "__main__":
    server = TACFastAPIServer(tac=tac, messaging_channels=[whatsapp_channel])
    server.start()

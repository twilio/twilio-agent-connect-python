"""
Example: RCS Channel with OpenAI Agents SDK

Demonstrates RCS (Rich Communication Services) channel with TAC memory injection.
RCS supports rich media like images and location sharing.

Requires ``OPENAI_API_KEY`` and ``TWILIO_RCS_SENDER_ID`` in addition to standard TAC env vars.

Usage:
    python rcs.py

Then send messages to your Twilio RCS agent from your phone.
"""

from typing import Any

from agents import Agent, Runner, set_tracing_disabled
from dotenv import load_dotenv

from tac import TAC, TACConfig
from tac.channels.rcs import RCSChannel, RCSChannelConfig
from tac.core.logging import get_logger
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer

load_dotenv()
set_tracing_disabled(True)

logger = get_logger(__name__)

tac = TAC(config=TACConfig.from_env())

# RCS Sender ID is configured via TWILIO_RCS_SENDER_ID env var
rcs_channel = RCSChannel(
    tac,
    config=RCSChannelConfig(
        memory_mode="always",
    ),
)

conversation_history: dict[str, list[Any]] = {}

SYSTEM_INSTRUCTIONS = (
    "You are a customer service agent speaking with a user over RCS. "
    "Keep responses short and conversational — a sentence or two. "
    "Do not use markdown, asterisks, bullets, or emojis; your words will be "
    "sent as plain text."
)


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> str:
    """
    Callback invoked when a message is ready to be processed.

    This example uses the OpenAI Agents SDK with manual memory injection.

    Args:
        user_message: The customer's message text
        context: Session data (conversation_id, channel, profile, etc.)
        memory_response: Optional retrieved memories (observations, summaries, communications)

    Returns:
        Response string to send to the channel
    """
    conv_id = context.conversation_id

    try:
        instructions = SYSTEM_INSTRUCTIONS
        if memory_response:
            memory_sections = memory_response.build_memory_prompts()
            if memory_sections:
                instructions += "\n\n" + "\n\n".join(memory_sections)

        agent = Agent(name="RCS Customer Service Agent", instructions=instructions)

        history = conversation_history.get(conv_id, [])
        agent_input = history + [{"role": "user", "content": user_message}]

        result = await Runner.run(agent, agent_input)

        conversation_history[conv_id] = result.to_input_list()

        return result.final_output_as(str)

    except Exception as e:
        logger.error("Error processing RCS message", conversation_id=conv_id, error=str(e))
        return "Sorry, I encountered an error processing your message."


tac.on_message_ready(handle_message_ready)

if __name__ == "__main__":
    server = TACFastAPIServer(
        tac=tac,
        messaging_channels=[rcs_channel],
    )
    server.start()

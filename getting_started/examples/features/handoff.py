"""
Feature: Handoff to Human Agent

Demonstrates TAC's handoff tool routing a conversation to a human agent via
a Twilio Studio Flow (for example, one that routes to Flex). Works on voice
and SMS.

Requires ``TWILIO_STUDIO_HANDOFF_FLOW_SID`` in addition to the usual TAC env
vars — see ``.env.example``.
"""

from typing import Any

from agents import Agent, Runner, set_tracing_disabled
from dotenv import load_dotenv

from tac import TAC, TACConfig
from tac.channels.sms import SMSChannel, SMSChannelConfig
from tac.channels.voice import ConversationRelayProviderConfig, VoiceChannel
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer
from tac.tools.handoff import create_studio_handoff_tool

load_dotenv()
set_tracing_disabled(True)

tac = TAC(config=TACConfig.from_env())

# Verify the handoff-specific env var is set.
if not tac.config.studio_handoff_flow_sid:
    raise RuntimeError(
        "TWILIO_STUDIO_HANDOFF_FLOW_SID is required to run the handoff example. "
        "Set it in your .env (see .env.example)."
    )

SYSTEM_INSTRUCTIONS = (
    "You are a customer service agent speaking with a user over voice or SMS. "
    "Keep responses short and conversational — a sentence or two. "
    "Do not use markdown, asterisks, bullets, or emojis; your words will be "
    "spoken aloud or sent as plain text. "
    "If the user asks to speak with a human, or if you cannot resolve their issue, "
    "use the handoff tool to transfer them to a human agent."
)

# Example app-defined routing metadata attached to every handoff. Keys and
# values are arbitrary — pick whatever your downstream system expects. For
# Flex, these surface as TaskRouter task attributes.
HANDOFF_ATTRIBUTES = {
    "department": "support",
    "priority": "normal",
}

conversation_history: dict[str, list[Any]] = {}


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> str:
    handoff_tool = create_studio_handoff_tool(tac, context, attributes=HANDOFF_ATTRIBUTES)

    agent = Agent(
        name="Customer Service Agent",
        instructions=SYSTEM_INSTRUCTIONS,
        tools=[handoff_tool.to_openai_agents_sdk_tool()],
    )

    history = conversation_history.get(context.conversation_id, [])
    agent_input = history + [{"role": "user", "content": user_message}]

    result = await Runner.run(agent, agent_input)

    conversation_history[context.conversation_id] = result.to_input_list()
    return result.final_output_as(str)


voice_channel = VoiceChannel(tac, config=ConversationRelayProviderConfig())
sms_channel = SMSChannel(tac, config=SMSChannelConfig())

tac.on_message_ready(handle_message_ready)


if __name__ == "__main__":
    server = TACFastAPIServer(
        tac=tac, voice_channel=voice_channel, messaging_channels=[sms_channel]
    )
    server.start()

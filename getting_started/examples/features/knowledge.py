"""
Feature: Knowledge Base Search Tool

Demonstrates TAC's knowledge tool letting an LLM agent answer questions by
searching a Twilio Knowledge Base.

Requires ``TWILIO_KNOWLEDGE_BASE_ID`` in addition to the usual TAC env vars —
see ``.env.example``. The knowledge base must be ACTIVE.
"""

from typing import Any

from agents import Agent, Runner, set_tracing_disabled
from dotenv import load_dotenv

from tac import TAC, TACConfig
from tac.channels.sms import SMSChannel, SMSChannelConfig
from tac.channels.voice import VoiceChannel, VoiceChannelConfig
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer
from tac.tools.knowledge import create_knowledge_tool

load_dotenv()
set_tracing_disabled(True)

tac = TAC(config=TACConfig.from_env())

# The knowledge client is only initialized when TWILIO_KNOWLEDGE_BASE_ID is set.
if not tac.knowledge_client or not tac.config.knowledge_base_id:
    raise RuntimeError(
        "TWILIO_KNOWLEDGE_BASE_ID is required to run the knowledge example. "
        "Set it in your .env (see .env.example)."
    )

knowledge_client = tac.knowledge_client
knowledge_base_id = tac.config.knowledge_base_id

SYSTEM_INSTRUCTIONS = (
    "You are a customer service agent speaking with a user over voice or SMS. "
    "Keep responses short and conversational — a sentence or two. "
    "Do not use markdown, asterisks, bullets, or emojis; your words will be "
    "spoken aloud or sent as plain text. "
    "When the user asks a question, use the knowledge search tool to look up an "
    "answer before responding. If the knowledge base has no relevant information, "
    "say you don't know rather than guessing."
)

conversation_history: dict[str, list[Any]] = {}

# Build the knowledge tool once at startup. Passing name and description avoids
# a metadata lookup; omit them to derive defaults from the knowledge base.
knowledge_tool = None


async def get_knowledge_tool() -> Any:
    global knowledge_tool
    if knowledge_tool is None:
        knowledge_tool = await create_knowledge_tool(
            knowledge_client=knowledge_client,
            knowledge_base_id=knowledge_base_id,
            name="search_knowledge_base",
            description="Search the company knowledge base to answer user questions.",
            top_k=3,
        )
    return knowledge_tool


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> str:
    tool = await get_knowledge_tool()

    agent = Agent(
        name="Customer Service Agent",
        instructions=SYSTEM_INSTRUCTIONS,
        tools=[tool.to_openai_agents_sdk_tool()],
    )

    history = conversation_history.get(context.conversation_id, [])
    agent_input = history + [{"role": "user", "content": user_message}]

    result = await Runner.run(agent, agent_input)

    conversation_history[context.conversation_id] = result.to_input_list()
    return result.final_output_as(str)


voice_channel = VoiceChannel(tac, config=VoiceChannelConfig())
sms_channel = SMSChannel(tac, config=SMSChannelConfig())

tac.on_message_ready(handle_message_ready)


if __name__ == "__main__":
    server = TACFastAPIServer(
        tac=tac, voice_channel=voice_channel, messaging_channels=[sms_channel]
    )
    server.start()

"""
Example: Using OpenAI Responses API with TAC Memory Injection

Demonstrates how to use with_tac_memory with the Responses API.
For the Chat Completions API, see chat_completions.py.
"""

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

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

# Initialize TAC with configuration from environment variables
# TAC handles conversation orchestration and optional memory retrieval
tac = TAC(config=TACConfig.from_env())

# Create channel handlers for Voice and SMS
# Channels process Twilio webhooks and manage conversation lifecycle
voice_channel = VoiceChannel(tac, config=VoiceChannelConfig(auto_retrieve_memory=True))
sms_channel = SMSChannel(tac, config=SMSChannelConfig(auto_retrieve_memory=True))

# Initialize your LLM client (OpenAI in this example)
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Store conversation history per conversation
# This maintains context across multiple turns in a conversation
conversation_history: dict[str, list[dict[str, str]]] = {}

SYSTEM_INSTRUCTIONS = (
    "You are a customer service agent speaking with a user over voice or SMS. "
    "Keep responses short and conversational — a sentence or two. "
    "Do not use markdown, asterisks, bullets, or emojis; your words will be "
    "spoken aloud or sent as plain text."
)


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> str:
    """
    Callback invoked when a message is ready to be processed.

    This example uses the Responses API with automatic memory injection.

    Args:
        user_message: The customer's message text
        context: Session data (conversation_id, channel, profile, etc.)
        memory_response: Optional retrieved memories (observations, summaries, communications)

    Returns:
        Response string to send to the channel
    """
    conv_id = context.conversation_id

    try:
        # Initialize conversation history for new conversations
        if conv_id not in conversation_history:
            conversation_history[conv_id] = []

        # Add user message to conversation history
        conversation_history[conv_id].append({"role": "user", "content": user_message})

        # Wrap OpenAI client with TAC adapter for automatic memory injection
        # The adapter prepends memory context to the instructions parameter
        client = with_tac_memory(openai_client, memory_response, context)

        # Call OpenAI Responses API - memory is automatically injected
        response = await client.responses.create(
            model="gpt-5.4-mini",
            instructions=SYSTEM_INSTRUCTIONS,
            input=conversation_history[conv_id],
        )

        llm_response = str(response.output_text)

        # Save assistant response to conversation history
        conversation_history[conv_id].append({"role": "assistant", "content": llm_response})

        return llm_response

    except Exception as e:
        logger.error("Error processing message", conversation_id=conv_id, error=str(e))
        return "Sorry, I encountered an error processing your message."


# Register the message handler callback
# TAC will invoke this function whenever a message needs processing
tac.on_message_ready(handle_message_ready)

if __name__ == "__main__":
    # TACFastAPIServer creates a FastAPI app with all required endpoints:
    # - /twiml: Voice call webhook (returns TwiML with ConversationRelay)
    # - /ws: WebSocket endpoint for Voice channel
    # - /webhook: Conversation webhook for all channels
    server = TACFastAPIServer(
        tac=tac, voice_channel=voice_channel, messaging_channels=[sms_channel]
    )
    server.start()

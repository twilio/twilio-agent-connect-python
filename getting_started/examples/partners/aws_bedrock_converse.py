"""
Example: Using AWS Bedrock Converse API with TAC

Demonstrates how to connect Twilio Agent Connect with the AWS Bedrock Converse API
for voice and SMS channels using boto3 directly.

This works with any Bedrock foundation model (Claude, Llama, Mistral, etc.).

Prerequisites:
    pip install boto3

Environment Variables:
    AWS_REGION - AWS Region
    AWS_BEDROCK_MODEL_ID - Bedrock model ID (e.g. us.anthropic.claude-3-5-haiku-20241022-v1:0)

For production AWS agents, consider using the Strands Agents SDK:
    https://github.com/strands-agents/sdk-python
"""

import asyncio
import os
from typing import Any

import boto3
from dotenv import load_dotenv

from tac import TAC, TACConfig
from tac.adapters import MemoryPromptBuilder
from tac.channels.sms import SMSChannel, SMSChannelConfig
from tac.channels.voice import VoiceChannel, VoiceChannelConfig
from tac.core.logging import get_logger
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer

load_dotenv()

logger = get_logger(__name__)

# Initialize TAC with configuration from environment variables
tac = TAC(config=TACConfig.from_env())

# Create channel handlers for Voice and SMS
voice_channel = VoiceChannel(tac, config=VoiceChannelConfig(memory_mode="once"))
sms_channel = SMSChannel(tac, config=SMSChannelConfig(memory_mode="always"))

# Get Bedrock configuration
region = os.environ["AWS_REGION"]
model_id = os.environ["AWS_BEDROCK_MODEL_ID"]

# Create Bedrock Runtime client
bedrock_client = boto3.client("bedrock-runtime", region_name=region)

# Store conversation history per conversation (Bedrock Converse message format)
conversation_history: dict[str, list[dict[str, Any]]] = {}

SYSTEM_PROMPT = (
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

    This example uses the Bedrock Converse API with manual memory injection
    via MemoryPromptBuilder.

    Args:
        user_message: The customer's message text
        context: Session data (conversation_id, channel, profile, etc.)
        memory_response: Optional retrieved memories (observations, summaries, communications)

    Returns:
        Response string to send to the channel
    """
    conv_id = context.conversation_id

    try:
        if conv_id not in conversation_history:
            conversation_history[conv_id] = []

        # Compose system prompt with memory context
        system_prompt = MemoryPromptBuilder.compose(
            SYSTEM_PROMPT,
            memory_response=memory_response,
            context=context,
        )

        # Add user message in Bedrock Converse format
        conversation_history[conv_id].append({"role": "user", "content": [{"text": user_message}]})

        # Call Bedrock Converse API (boto3 is synchronous)
        response = await asyncio.to_thread(
            bedrock_client.converse,
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=conversation_history[conv_id],
        )

        # Extract response text
        llm_response: str = response["output"]["message"]["content"][0]["text"]

        # Save assistant response to conversation history
        conversation_history[conv_id].append(
            {"role": "assistant", "content": [{"text": llm_response}]}
        )

        return llm_response

    except Exception as e:
        logger.error("Error processing message", conversation_id=conv_id, error=str(e))
        return "Sorry, I encountered an error processing your message."


# Register the message handler callback
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

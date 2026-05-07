"""
Example: Using AWS Bedrock Agent with TAC

Demonstrates how to connect Twilio Agent Connect with AWS Bedrock Agents
for voice and SMS channels using boto3 directly.

Prerequisites:
    pip install boto3

For an enhanced integration experience with Amazon Bedrock Agents, check out our
dedicated connector: https://github.com/twilio/twilio-agent-connect-aws

Environment Variables:
    BEDROCK_AGENT_ID - Your Bedrock Agent ID
    BEDROCK_AGENT_ALIAS_ID - Agent Alias ID
    AWS_REGION - AWS Region
"""

import os

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

# Get Bedrock Agent configuration
agent_id = os.environ["BEDROCK_AGENT_ID"]
agent_alias_id = os.environ["BEDROCK_AGENT_ALIAS_ID"]
region = os.environ["AWS_REGION"]

# Create Bedrock Agent Runtime client
bedrock_client = boto3.client("bedrock-agent-runtime", region_name=region)


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> str:
    """
    Callback invoked when a message is ready to be processed.

    This example calls AWS Bedrock Agent and returns the complete response.

    Args:
        user_message: The customer's message text
        context: Session data (conversation_id, channel, profile, etc.)
        memory_response: Optional retrieved memories (observations, summaries, communications)

    Returns:
        Response string to send to the channel
    """
    conv_id = context.conversation_id

    try:
        # Compose user message with memory context (user message first, then memory)
        # Note: Bedrock Agents don't support system prompt at invocation time.
        # The agent's instructions are configured when deploying the agent in AWS Bedrock.
        input_text = MemoryPromptBuilder.compose(
            system_prompt=user_message,
            memory_response=memory_response,
            context=context,
        )

        # Invoke Bedrock Agent
        response = bedrock_client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=conv_id,
            inputText=input_text,
            enableTrace=False,
        )

        # Collect all response chunks
        chunks = []
        event_stream = response.get("completion", [])
        for event in event_stream:
            if "chunk" in event:
                chunk_data = event["chunk"]
                if "bytes" in chunk_data:
                    chunk_text = chunk_data["bytes"].decode("utf-8")
                    chunks.append(chunk_text)

        return "".join(chunks) if chunks else "No response from agent."

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

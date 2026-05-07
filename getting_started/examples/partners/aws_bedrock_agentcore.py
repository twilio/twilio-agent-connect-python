"""
Example: Using AWS Bedrock AgentCore with TAC

Demonstrates how to connect Twilio Agent Connect with AWS Bedrock AgentCore
for voice and SMS channels.

Prerequisites:
    pip install boto3

For an enhanced integration experience with Amazon Bedrock AgentCore, check out our
dedicated connector: https://github.com/twilio/twilio-agent-connect-aws

Environment Variables:
    BEDROCK_AGENTCORE_AGENT_ARN - ARN of deployed AgentCore runtime
    AWS_REGION - AWS Region
"""

import json
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

# Get AgentCore configuration
agent_arn = os.environ["BEDROCK_AGENTCORE_AGENT_ARN"]
region = os.environ["AWS_REGION"]

# Create Bedrock AgentCore client
agentcore_client = boto3.client("bedrock-agentcore", region_name=region)


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> str:
    """
    Callback invoked when a message is ready to be processed.

    This example calls AWS Bedrock AgentCore and returns the complete response.

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
        prompt = MemoryPromptBuilder.compose(
            system_prompt=user_message,
            memory_response=memory_response,
            context=context,
        )

        # Prepare payload for AgentCore
        payload = json.dumps({"prompt": prompt}).encode("utf-8")

        # Invoke AgentCore runtime
        response = agentcore_client.invoke_agent_runtime(
            agentRuntimeArn=agent_arn,
            runtimeSessionId=conv_id,
            payload=payload,
        )

        # Process streaming response
        content = []
        if "text/event-stream" in response.get("contentType", ""):
            for line_bytes in response["response"].iter_lines():
                if line_bytes:
                    line = line_bytes.decode("utf-8")
                    if line.startswith("data: "):
                        chunk_text = line[6:]
                        if chunk_text and chunk_text != "[DONE]":
                            try:
                                data = json.loads(chunk_text)
                                if isinstance(data, dict) and data.get("type") == "text":
                                    token = data.get("token", "")
                                    if token:
                                        content.append(token)
                            except json.JSONDecodeError:
                                content.append(chunk_text)

        return "".join(content) if content else "No response from agent."

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

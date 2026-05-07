"""
Example: Using AWS Strands with TAC

Demonstrates how to connect Twilio Agent Connect with AWS Strands agents
for voice and SMS channels.

Prerequisites:
    pip install strands-agents

    Configure AWS credentials (choose one method):
    1. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN (optional)
    2. AWS credentials file: Run `aws configure` CLI command
    3. IAM roles: If running on AWS services (EC2, ECS, Lambda)
    4. Bedrock API keys: Set AWS_BEARER_TOKEN_BEDROCK environment variable

    Ensure your AWS credentials have permissions to invoke Amazon Bedrock models.
    By default, Strands uses Amazon Bedrock with Claude Sonnet 4.

    For more details: https://strandsagents.com/docs/user-guide/quickstart/python/

For an enhanced integration experience with AWS Strands, check out our dedicated connector:
    https://github.com/twilio/twilio-agent-connect-aws

Environment Variables:
    AWS_REGION - AWS Region (required)
    STRANDS_MODEL_ID - Bedrock model ID (required, e.g., us.amazon.nova-pro-v1:0)
"""

import os

from dotenv import load_dotenv
from strands import Agent
from strands.models import BedrockModel
from strands.session.file_session_manager import FileSessionManager

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

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize TAC with configuration from environment variables
tac = TAC(config=TACConfig.from_env())

# Create channel handlers for Voice and SMS
voice_channel = VoiceChannel(tac, config=VoiceChannelConfig(memory_mode="once"))
sms_channel = SMSChannel(tac, config=SMSChannelConfig(memory_mode="always"))

# Get AWS region and model ID from environment
region = os.environ["AWS_REGION"]
model_id = os.environ["STRANDS_MODEL_ID"]

# Store session managers per conversation
session_managers: dict[str, FileSessionManager] = {}


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> str:
    """
    Callback invoked when a message is ready to be processed.

    This example uses AWS Strands Agent with TAC memory context.

    Args:
        user_message: The customer's message text
        context: Session data (conversation_id, channel, profile, etc.)
        memory_response: Optional retrieved memories (observations, summaries, communications)

    Returns:
        Response string to send to the channel
    """
    conv_id = context.conversation_id

    try:
        # Get or create session manager for this conversation
        if conv_id not in session_managers:
            session_managers[conv_id] = FileSessionManager(
                session_id=conv_id,
                storage_dir=os.path.join(SCRIPT_DIR, ".strands_sessions"),
            )

        session_manager = session_managers[conv_id]

        # Compose system prompt with memory context
        system_prompt = MemoryPromptBuilder.compose(
            system_prompt=(
                "You are a customer service agent speaking with a user over voice or SMS. "
                "Keep responses short and conversational — a sentence or two. "
                "Do not use markdown, asterisks, bullets, or emojis; your words will be "
                "spoken aloud or sent as plain text."
            ),
            memory_response=memory_response,
            context=context,
        )

        # Create agent with session manager to maintain conversation history
        agent = Agent(
            model=BedrockModel(model_id=model_id, region_name=region),
            system_prompt=system_prompt,
            session_manager=session_manager,
        )

        # Call Strands agent
        result = await agent.invoke_async(user_message)

        # Extract response text from result
        response_text = ""
        if result and hasattr(result, "message") and result.message:
            if isinstance(result.message, dict):
                content = result.message.get("content", [])
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        response_text += block.get("text", "")

        return response_text if response_text else "No response from agent."

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

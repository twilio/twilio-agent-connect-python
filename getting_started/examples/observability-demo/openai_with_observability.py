"""
Example: Using OpenAI Chat Completions API with TAC Memory Injection + OpenTelemetry

This is the same as openai_chat_completions.py, but with OpenTelemetry observability added.

Setup:
1. Start observability stack: docker compose up -d
2. Get Langfuse credentials from http://localhost:3001 → Settings → API Keys
3. Create .env file with:
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   CONVERSATION_API_KEY=your-key
   CONVERSATION_API_TOKEN=your-token
   CONVERSATION_CONFIGURATION_ID=your-config-id
   OPENAI_API_KEY=sk-...

4. Run: uv run python openai_with_observability.py
5. Expose with ngrok: ngrok http 8000
6. Configure Twilio webhooks and send messages
7. View data in:
   - Grafana: http://localhost:3000
   - Langfuse: http://localhost:3001
"""

import base64
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

# ============================================================================
# OpenTelemetry Setup - ADD THIS BEFORE IMPORTING TAC
# ============================================================================
load_dotenv()

PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")

if PUBLIC_KEY and SECRET_KEY:
    # Configure OpenTelemetry endpoints
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"  # Metrics → Prometheus

    # Traces → Langfuse
    auth_string = f"{PUBLIC_KEY}:{SECRET_KEY}"
    auth_b64 = base64.b64encode(auth_string.encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = (
        "http://localhost:3001/api/public/otel/v1/traces"
    )
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth_b64}"

    # PRIVACY: Include message content (input/output) in traces?
    # WARNING: This sends user messages and LLM responses to your observability backend.
    # Only enable if you have proper data handling agreements and comply with privacy regulations.
    # Default: false (disabled for privacy)
    os.environ["TAC_TELEMETRY_INCLUDE_MESSAGE_CONTENT"] = os.getenv(
        "TAC_TELEMETRY_INCLUDE_MESSAGE_CONTENT", "false"
    )
# ============================================================================

from tac import TAC, TACConfig
from tac.adapters.openai import with_tac_memory
from tac.channels.sms import SMSChannel, SMSChannelConfig
from tac.channels.voice import VoiceChannel, VoiceChannelConfig
from tac.core.logging import get_logger
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer

# ============================================================================
# OpenTelemetry Setup - ADD THIS AFTER IMPORTING TAC
# ============================================================================
if PUBLIC_KEY and SECRET_KEY:
    from tac.telemetry import TACTelemetry

    telemetry = TACTelemetry()
    telemetry.setup_meter(enable_otlp_exporter=True)  # Metrics → Grafana
    telemetry.setup_tracer(enable_otlp_exporter=True)  # Traces → Langfuse
    print("✅ OpenTelemetry enabled - data will be sent to Grafana and Langfuse")
else:
    print("ℹ️  OpenTelemetry disabled - set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable")
# ============================================================================

logger = get_logger(__name__)

# Initialize TAC with configuration from environment variables
tac = TAC(config=TACConfig.from_env())

# Create channel handlers for Voice and SMS
voice_channel = VoiceChannel(tac, config=VoiceChannelConfig(memory_mode="once"))
sms_channel = SMSChannel(tac, config=SMSChannelConfig(memory_mode="always"))

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Store conversation history per conversation
conversation_history: dict[str, list[ChatCompletionMessageParam]] = {}

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
    """
    Callback invoked when a message is ready to be processed.

    This example uses the Chat Completions API with automatic memory injection.

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
            conversation_history[conv_id] = [SYSTEM_MESSAGE]

        # Add user message to conversation history
        user_msg: ChatCompletionUserMessageParam = {"role": "user", "content": user_message}
        conversation_history[conv_id].append(user_msg)

        # Wrap OpenAI client with TAC adapter for automatic memory injection
        # The adapter injects memory as a system message at the start of the messages array
        client = with_tac_memory(openai_client, memory_response, context)

        # Call OpenAI Chat Completions API - memory is automatically injected
        response = await client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=conversation_history[conv_id],
        )

        llm_response = response.choices[0].message.content or ""

        # Save assistant response to conversation history
        assistant_msg: ChatCompletionAssistantMessageParam = {
            "role": "assistant",
            "content": llm_response,
        }
        conversation_history[conv_id].append(assistant_msg)

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

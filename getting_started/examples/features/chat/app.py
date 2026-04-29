"""
Chat Server for Twilio Agent Connect

Example demonstrating ChatChannel with the Twilio Conversations JS SDK.
Messages flow through Twilio's infrastructure:

    Browser (Conversations JS SDK) -> Twilio Conversations ->
    Conversation Orchestrator -> webhook -> server -> AI ->
    Conversation Orchestrator Send API -> Twilio Conversations ->
    Browser (Conversations JS SDK)

See README.md in this directory for setup and usage instructions.
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from tac import TAC, TACConfig
from tac.adapters.openai import with_tac_memory
from tac.channels.chat import ChatChannel
from tac.core.logging import get_logger, setup_logging
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse

load_dotenv()

logger = get_logger(__name__)

CHAT_IDENTITY = "ai-agent"

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize TAC and chat channel
# Note: TAC.__init__ calls setup_logging with config.log_level, so set it here
# or use TWILIO_LOG_LEVEL=DEBUG in your .env
tac = TAC(config=TACConfig.from_env())

# Override logging to DEBUG after TAC init (which resets it to config.log_level)
setup_logging(log_level="DEBUG", log_format="console")
chat_channel = ChatChannel(tac, config={"agent_address": CHAT_IDENTITY})

# Store conversation history per conversation
conversation_history: dict[str, list[ChatCompletionMessageParam]] = {}

SYSTEM_MESSAGE: ChatCompletionSystemMessageParam = {
    "role": "system",
    "content": (
        "You're an assistant chatting with a user through a web chat interface. "
        "Keep responses short and conversational. Do not use markdown, asterisks, "
        "bullets, or emojis — the chat UI renders messages as plain text, so markdown "
        "syntax will appear as literal punctuation."
    ),
}


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> None:
    conv_id = context.conversation_id

    try:
        if conv_id not in conversation_history:
            conversation_history[conv_id] = [SYSTEM_MESSAGE]

        user_msg: ChatCompletionUserMessageParam = {"role": "user", "content": user_message}
        conversation_history[conv_id].append(user_msg)

        client = with_tac_memory(openai_client, memory_response, context)

        response = await client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=conversation_history[conv_id],
        )

        llm_response = response.choices[0].message.content

        if llm_response:
            assistant_msg: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": llm_response,
            }
            conversation_history[conv_id].append(assistant_msg)

            await chat_channel.send_response(conv_id, llm_response)

    except Exception as e:
        logger.error("Error processing message", conversation_id=conv_id, error=str(e))


async def handle_conversation_ended(context: ConversationSession) -> None:
    logger.info("Chat conversation ended", conversation_id=context.conversation_id)
    conversation_history.pop(context.conversation_id, None)


tac.on_message_ready(handle_message_ready)
tac.on_conversation_ended(handle_conversation_ended)


def create_app() -> Any:
    """Create the FastAPI app with chat-specific routes."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        from fastapi.staticfiles import StaticFiles
        from twilio.jwt.access_token import AccessToken
        from twilio.jwt.access_token.grants import ChatGrant
    except ImportError as e:
        raise ImportError(
            "Chat example requires: pip install fastapi uvicorn twilio python-multipart"
        ) from e

    app = FastAPI(title="TAC Chat Server")

    # Serve static files (the chat UI)
    static_dir = Path(__file__).parent / "public"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def index() -> Any:
        from fastapi.responses import FileResponse

        return FileResponse(str(static_dir / "index.html"))

    @app.post("/token")
    async def generate_token(request: Request) -> JSONResponse:
        """Generate a Conversations SDK access token."""
        body = await request.json()
        identity = body.get("identity")

        if not identity:
            return JSONResponse({"error": "Identity is required"}, status_code=400)

        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        api_key = os.environ.get("TWILIO_API_KEY")
        api_secret = os.environ.get("TWILIO_API_SECRET")
        service_sid = os.environ.get("TWILIO_CONVERSATIONS_SERVICE_SID")

        if not all([account_sid, api_key, api_secret, service_sid]):
            logger.error("Missing required credentials for token generation")
            return JSONResponse({"error": "Missing Twilio credentials"}, status_code=500)

        token = AccessToken(account_sid, api_key, api_secret, identity=identity, ttl=3600)
        chat_grant = ChatGrant(service_sid=service_sid)
        token.add_grant(chat_grant)

        jwt = token.to_jwt()
        if isinstance(jwt, bytes):
            jwt = jwt.decode("utf-8")

        return JSONResponse({"token": jwt})

    @app.post("/webhook")
    async def conversation_webhook(request: Request) -> JSONResponse:
        """Handle Conversation Orchestrator webhook events."""
        try:
            webhook_data = await request.json()
            idempotency_token = request.headers.get("i-twilio-idempotency-token")
            asyncio.create_task(chat_channel.process_webhook(webhook_data, idempotency_token))
            return JSONResponse({"status": "ok"})
        except Exception as e:
            logger.error("Webhook error", error=str(e), exc_info=True)
            return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

    return app


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    logger.info("Starting TAC Chat Server on http://0.0.0.0:8000")
    logger.info("Open http://localhost:8000 in your browser to test chat")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=False)

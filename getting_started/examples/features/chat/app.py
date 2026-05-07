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

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from twilio.jwt.access_token import AccessToken
from twilio.jwt.access_token.grants import ChatGrant

from tac import TAC, TACConfig
from tac.adapters.openai import with_tac_memory
from tac.channels.chat import ChatChannel
from tac.core.logging import get_logger
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer

load_dotenv()

logger = get_logger(__name__)

CHAT_IDENTITY = "ai-agent"

openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

tac = TAC(config=TACConfig.from_env())

# Example-level setup check (not required by the SDK): the V1 Chat backend
# behind this example needs a classic Conversations service — with Chat enabled
# on it — attached to the CO configuration.
configuration_id = os.environ["TWILIO_CONVERSATION_CONFIGURATION_ID"]
response = httpx.get(
    f"https://conversations.twilio.com/v2/ControlPlane/Configurations/{configuration_id}",
    auth=(os.environ["TWILIO_API_KEY"], os.environ["TWILIO_API_SECRET"]),
)
response.raise_for_status()
if not (response.json().get("conversationsV1Bridge") or {}).get("serviceId"):
    sys.exit(
        f"Configuration '{configuration_id}' has no classic Conversations service attached. "
        "Attach one (with Chat enabled) via Console → Conversation Orchestrator → "
        'Conversation Configuration → Channel traffic → "+ Add messaging & chat traffic".'
    )

chat_channel = ChatChannel(tac, config={"agent_address": CHAT_IDENTITY})

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


if __name__ == "__main__":
    server = TACFastAPIServer(tac=tac, messaging_channels=[chat_channel])

    # Layer the chat UI routes onto the same FastAPI app TAC provides.
    static_dir = Path(__file__).parent / "public"
    server.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @server.app.get("/")
    async def index() -> FileResponse:
        return FileResponse(str(static_dir / "index.html"))

    @server.app.post("/token")
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
            return JSONResponse({"error": "Missing Twilio credentials"}, status_code=500)

        token = AccessToken(account_sid, api_key, api_secret, identity=identity, ttl=3600)
        token.add_grant(ChatGrant(service_sid=service_sid))
        jwt = token.to_jwt()
        if isinstance(jwt, bytes):
            jwt = jwt.decode("utf-8")
        return JSONResponse({"token": jwt})

    server.start()
